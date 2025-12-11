from typing import Annotated
from datetime import datetime
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from become_human.time import datetime_to_seconds, Times
from become_human.store_manager import store_manager
from become_human.types_main import MainContext, MainState

@tool
async def add_self_call(
    self_call_time: Annotated[str, '自我唤醒时间，格式为YYYY-MM-DD HH:MM:SS'],
    note: Annotated[str, '笔记，会在唤醒时作为提示出现，以免忘记要做什么'],
    runtime: ToolRuntime[MainContext, MainState],
    force_add: Annotated[bool, '强制添加自我唤醒计划，无视可能已经存在的重复的自我唤醒计划'] = False) -> Command:
    """「即时工具」添加一次主动自我唤醒计划，系统将会在指定时间唤醒你自己。"""
    try:
        self_call_datetime = datetime.strptime(self_call_time, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ValueError("时间格式错误，请使用YYYY-MM-DD HH:MM:SS格式。")

    active_self_call_time_secondses_and_notes = runtime.state.active_self_call_time_secondses_and_notes
    tool_call_id = runtime.tool_call_id
    agent_id = runtime.context.agent_id

    self_call_seconds = datetime_to_seconds(self_call_datetime)
    active_self_call_time_secondses = [s for s, n in active_self_call_time_secondses_and_notes]
    content = "添加自我唤醒计划成功。"
    artifact = {"bh_do_not_store": True, "bh_streaming": True}
    new_state = {'active_self_call_time_secondses_and_notes': active_self_call_time_secondses_and_notes + [(self_call_seconds, note)]}
    if not force_add:
        for s in active_self_call_time_secondses:
            if abs(s - self_call_seconds) < 3600.0:
                content = '在你指定时间的附近一小时范围内已经存在主动自我唤醒计划，为避免重复此次添加被取消。若确定要添加，请将force_add参数设置为True再次调用此工具。'
                artifact = {"bh_do_not_store": True}
                new_state = {}
    time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
    times = Times(setting=time_settings)
    tool_message = ToolMessage(name='add_self_call', content=content, artifact=artifact, tool_call_id=tool_call_id, additional_kwargs={
        "bh_creation_agent_time_seconds": times.agent_time_seconds,
        "bh_creation_time_seconds": times.real_time_seconds,
    })
    new_state["tool_messages"] = [tool_message]
    return Command(update=new_state)
