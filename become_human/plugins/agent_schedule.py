from typing import Any, Annotated, Optional
from datetime import datetime, timedelta

from langchain.tools import ToolRuntime, tool

from become_human.times import AgentTimeSettings, Times, format_time, format_duration, datetime_to_seconds, seconds_to_datetime, SerializableTimeZone
from become_human.message import BHMessageMetadata, BH_MESSAGE_METADATA_KEY
from become_human.scheduler import Schedule, get_schedules_by_agent_id_and_type, get_schedule_by_id
from become_human.store.manager import store_manager
from become_human.tool import AgentTool
from become_human.types.main import MainContext
from become_human.plugin import Plugin
from become_human.agent_manager import agent_manager

async def agent_schedule_job(schedule: Schedule, agent_id: str, title: str, description: str) -> None:
    """
    定时计划任务。

    :param agent_id: 智能体的唯一标识符。
    :param title: 计划的标题，用于标识计划。
    :param description: 计划的详细描述，说明计划的具体内容。
    """
    time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
    current_times = Times.from_time_settings(time_settings)

    time_diff = current_times.agent_world_timeseconds - schedule.next_trigger_timeseconds

    await agent_manager.call_agent_for_system(
        agent_id=agent_id,
        content=f'''当前时间是{format_time(current_times.agent_world_datetime)}。现在将你唤醒是由于你之前主动设置的定时计划到时间了{f'（但由于系统原因，有一些超出原定时间，具体为{format_duration(time_diff)}）' if time_diff > 300 else ''}，以下是你为此定时计划留下的描述，请根据此计划描述考虑现在应如何行动：
{title}\n{description}''',
        times=current_times
    )

add_schedule_schema = {
    "$defs": {
        "weekday": {
            "description": "星期几，1-7分别表示周一到周日",
            "maximum": 7,
            "minimum": 1,
            "type": "integer"
        },
        "monthday": {
            "description": "每月几号，1-31分别表示1号到31号",
            "maximum": 31,
            "minimum": 1,
            "type": "integer"
        },
        "month": {
            "description": "每年几月，1-12分别表示1月到12月",
            "maximum": 12,
            "minimum": 1,
            "type": "integer"
        }
    },
    "properties": {
        "title": {
            "description": "计划标题，用于在查询时快速识别此计划。",
            "type": "string"
        },
        "description": {
            "description": "计划描述，用于在执行时详细说明此计划，提醒自己要做些什么。",
            "type": "string"
        },
        "max_triggers": {
            "description": "计划最大触发次数，若为0则无限触发，1表示仅触发一次。",
            "type": "integer",
        },
        "start_time": {
            "description": "计划开始时间，格式为YYYY-MM-DD HH:MM:SS。",
            "type": "string",
            "default": ""
        },
        "time_of_day": {
            "description": "若要设置可重复定时计划，指定应在一天中的哪个时间点触发计划。",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "hour": {
                            "description": "小时（0~23）",
                            "maximum": 23,
                            "minimum": 0,
                            "type": "integer"
                        },
                        "minute": {
                            "description": "分钟（0~59）",
                            "maximum": 59,
                            "minimum": 0,
                            "type": "integer"
                        },
                        "second": {
                            "description": "秒钟（0~59）",
                            "maximum": 59,
                            "minimum": 0,
                            "type": "integer"
                        }
                    },
                    "required": ["hour", "minute", "second"]
                },
                {
                    "type": "null"
                }
            ],
            "default": None
        },
        "every_day": {
            "description": "是否每天触发。",
            "type": "boolean",
            "default": False
        },
        "weekdays": {
            "description": "指定星期几触发，1-7分别表示周一到周日。可与monthdays同时设置（不会在同一天触发两次）。",
            "items": {
                "$ref": "#/$defs/weekday"
            },
            "type": "array",
            "uniqueItems": True,
            "default": []
        },
        "monthdays": {
            "description": "指定每月几号触发，1-31分别表示1号到31号。可与weekdays同时设置（不会在同一天触发两次）。若设置的日期超过当月总天数，会自动调整为当月最后一天（以应对不同月份的天数差异）。",
            "items": {
                "$ref": "#/$defs/monthday"
            },
            "type": "array",
            "uniqueItems": True,
            "default": []
        },
        "every_month": {
            "description": "是否每月触发。若every_month与months都没有设置，则计划只在当月生效，过了当月就会被删除。",
            "type": "boolean",
            "default": False
        },
        "months": {
            "description": "指定每年几月触发，1-12分别表示1月到12月。",
            "items": {
                "$ref": "#/$defs/month"
            },
            "type": "array",
            "uniqueItems": True,
            "default": []
        }
    },
    "type": "object",
    "required": ["title", "description", "max_triggers"],
    "title": "add_schedule"
}

@tool(args_schema=add_schedule_schema, response_format="content_and_artifact")
async def add_schedule(
    runtime: ToolRuntime[MainContext],
    title: str,
    description: str,
    max_triggers: int,
    start_time: str = "",
    time_of_day: Optional[dict[str, int]] = None,
    every_day: bool = False,
    weekdays: set[int] = set(),
    monthdays: set[int] = set(),
    every_month: bool = False,
    months: set[int] = set(),
) -> tuple[str, dict[str, Any]]:
    """为自己添加一个一次性或可重复执行的定时计划，系统将在指定时间唤醒你自己。

详细说明：
- 一次性定时计划
    - 如果只指定了start_time而没有除了title、description和max_triggers之外的其他任何参数，则视为一次性定时计划，只会在start_time指定的时间触发一次。此时max_triggers的值必须为1。
    - 又或者，不论其他参数如何，只要max_triggers为1，那么就等于一次性定时计划。
- 可重复定时计划
    - 重复指的是根据一些规则重复计算下次执行时间，所以这至少需要指定time_of_day参数，才可能进行重复计算。
    - 在可重复定时计划的情况下，如果start_time为空，则会立刻根据当前时间计算下一次触发时间。无需担心这会立刻重新唤醒你，也不会消耗触发次数。
    - 而如果指定了start_time，计划会先等到start_time触发一次，然后再根据其他参数计算下一次触发时间。"""
    if not start_time and not time_of_day:
        raise ValueError("必须至少提供start_time或time_of_day其中之一！")
    # max_triggers本身不是必须的，但为了让AI更清楚自己在做什么，要求其必须正确输出。
    # max_triggers没有默认值也是因为AI可能漏掉这个参数（它会以为自己写了，但实际上没有）。
    if max_triggers != 1 and not time_of_day:
        raise ValueError("只指定了start_time而没有指定time_of_day则视为一次性定时计划，此时max_triggers参数必须为1！")
    agent_id = runtime.context.agent_id
    time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
    if start_time:
        try:
            start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise ValueError("start_time参数的时间字符串格式错误，请检查是否符合YYYY-MM-DD HH:MM:SS格式。")
        start_time = start_time.replace(tzinfo=time_settings.time_zone.tz())
        next_run_timeseconds = datetime_to_seconds(start_time)
    else:
        next_run_timeseconds = -1.0
    if time_of_day:
        time_of_day_seconds = time_of_day['hour'] * 3600 + time_of_day['minute'] * 60 + time_of_day['second']
    else:
        time_of_day_seconds = None
    content = f"添加 {title} 定时计划成功。"
    schedule = Schedule(
        agent_id=agent_id,
        job=agent_schedule_job,
        job_kwargs={
            'agent_id': agent_id,
            'title': title,
            'description': description,
        },
        schedule_type='agent_schedule:schedule',
        scheduled_time_of_day=time_of_day_seconds,
        scheduled_every_day=every_day,
        scheduled_weekdays=weekdays,
        scheduled_monthdays=monthdays,
        scheduled_every_month=every_month,
        scheduled_months=months,
        time_reference='agent_world',
        #time_zone_name=time_settings.time_zone.name,
        #time_zone_offset=time_settings.time_zone.offset,
        max_triggers=max_triggers,
        next_trigger_timeseconds=next_run_timeseconds,
    )
    times = Times.from_time_settings(time_settings)
    if next_run_timeseconds < 0.0:
        calc_result = schedule.calc_next_trigger(times)
        if not calc_result:
            raise ValueError("非every_month且没有设置months意为计划只在当月生效，而计算得出该计划的下次运行时间并非当月，计划无效！")
    await schedule.add_to_scheduler()
    artifact = {
        BH_MESSAGE_METADATA_KEY: BHMessageMetadata(
            creation_times=times,
            message_type="bh:tool"
        ).model_dump()
    }
    return content, artifact

@tool(response_format="content_and_artifact")
async def list_schedules(
    runtime: ToolRuntime[MainContext],
) -> tuple[str, dict[str, Any]]:
    """列出所有已设置的定时计划。"""
    agent_id = runtime.context.agent_id
    schedules = await get_schedules_by_agent_id_and_type(agent_id=agent_id, schedule_type='agent_schedule:schedule')
    time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
    content = f"以下是你已设置且还在生效的定时计划：\n\n{'\n\n'.join(
        [f'''计划标题：{schedule.job_kwargs['title']}
计划描述：{schedule.job_kwargs['description']}
{schedule.format_schedule(time_settings.time_zone, prefix='计划', include_id=True, include_type=False)}''' for schedule in schedules]
    )}"
    times = Times.from_time_settings(time_settings)
    artifact = {
        BH_MESSAGE_METADATA_KEY: BHMessageMetadata(
            creation_times=times,
            message_type="bh:tool",
            do_not_store=True,
        ).model_dump()
    }
    return content, artifact

@tool(response_format="content_and_artifact")
async def delete_schedule(
    runtime: ToolRuntime[MainContext],
    schedule_id: Annotated[str, "要删除的计划的ID"],
) -> tuple[str, dict[str, Any]]:
    """删除一个已设置的定时计划。"""
    agent_id = runtime.context.agent_id
    try:
        schedule = await get_schedule_by_id(schedule_id)
    except ValueError:
        raise ValueError(f"不存在 ID 为 {schedule_id} 的计划！")
    if schedule.agent_id != agent_id:
        raise ValueError("该计划不是由你设置的，不能删除！")
    await schedule.delete_from_scheduler()
    content = f"删除计划 {schedule_id} 成功。"
    times = Times.from_time_settings((await store_manager.get_settings(agent_id)).main.time_settings)
    artifact = {
        BH_MESSAGE_METADATA_KEY: BHMessageMetadata(
            creation_times=times,
            message_type="bh:tool"
        ).model_dump()
    }
    return content, artifact

class AgentSchedulePlugin(Plugin):
    tools = [
        AgentTool.from_tool(add_schedule),
        AgentTool.from_tool(list_schedules),
        AgentTool.from_tool(delete_schedule)
    ]
