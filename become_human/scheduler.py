from uuid import uuid4
import json
import inspect
import importlib
import aiosqlite
from pydantic import BaseModel, Field, field_validator, computed_field, model_validator, ValidationInfo
from typing import Any, Union, Optional, Self, Literal, Callable
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
from loguru import logger
from tzlocal import get_localzone

from become_human.times import TimestampUs, Times, nowtz, SerializableTimeZone, format_time, seconds_to_datetime
from become_human.store.manager import store_manager


DATABASE_PATH = "./data/schedules.sqlite"
SCHEDULE_KEYS = ['agent_id','schedule_id', 'schedule_type', 'job_module', 'job_func', 'job_args', 'job_kwargs',
                'interval_fixed', 'interval_random_min', 'interval_random_max',
                'scheduled_time_of_day', 'scheduled_every_day', 'scheduled_weekdays',
                'scheduled_monthdays', 'scheduled_every_month', 'scheduled_months',
                'timeout_seconds', 'max_triggers', 'time_reference',
                'time_zone_name', 'time_zone_offset', 'trigger_time', 'trigger_count', 'repeating']
AnyScheduleKey = Literal['agent_id','schedule_id', 'schedule_type', 'job_module', 'job_func', 'job_args', 'job_kwargs',
                'interval_fixed', 'interval_random_min', 'interval_random_max',
                'scheduled_time_of_day', 'scheduled_every_day', 'scheduled_weekdays',
                'scheduled_monthdays', 'scheduled_every_month', 'scheduled_months',
                'timeout_seconds', 'max_triggers', 'time_reference',
                'time_zone_name', 'time_zone_offset', 'trigger_time', 'trigger_count', 'repeating']

class Schedule(BaseModel):
    """定时计划

    如interval和scheduled系列参数都不设置，表示这是一次性计划，将在trigger_time时触发一次后被删除（又或者max_triggers设置为1也是同样的效果）

    trigger_time的默认值是-1，如果不修改将在下次tick时直接被触发一次（没有设置timeout的话）

    如只设置interval，表示将按指定时间间隔触发。间隔时间总是在scheduled之后被加上

    如需设置scheduled系列参数，需至少设置time_of_day参数以及其他任意一个scheduled系列参数

    可以不设置every_month和months，表示只在当月触发"""
    agent_id: str = Field(description="关联的agent_id")
    schedule_id: str = Field(default_factory=lambda: str(uuid4()), description="唯一id")
    schedule_type: str = Field(default="", description="计划类型，方便查询")
    job: Callable = Field(description="计划要执行的任务，不可使用实例方法（不会验证这一点）")
    job_args: list[Any] = Field(default_factory=list, description="任务位置参数，需可被json序列化")
    job_kwargs: dict[str, Any] = Field(default_factory=dict, description="任务关键字参数，需可被json序列化")
    interval_fixed: float = Field(default=0.0, description="固定间隔时间，0为无固定间隔。若设置了fixed则会无视random")
    interval_random_min: float = Field(default=0.0, description="随机时间最小值，0为无随机时间")
    interval_random_max: float = Field(default=0.0, description="随机时间最大值，0为无随机时间")
    scheduled_time_of_day: Optional[float] = Field(default=None, ge=0.0, le=86400.0, description="指定一天中的时间，单位为秒")
    scheduled_every_day: bool = Field(default=False, description="是否每天触发")
    scheduled_weekdays: set[int] = Field(default_factory=set, description="指定星期几触发，1-7分别表示周一到周日。可与monthdays重复指定，不会重复触发")
    scheduled_monthdays: set[int] = Field(default_factory=set, description="指定每月几号触发，1-31分别表示1-31号。可与weekdays重复指定，不会重复触发")
    scheduled_every_month: bool = Field(default=False, description="是否每月触发，为False表示只在当月触发")
    scheduled_months: set[int] = Field(default_factory=set, description="指定每年几月触发，1-12分别表示1-12月")
    timeout_seconds: float = Field(default=0.0, description="超时时间，指如果当前时间超过指定时间太久则算作超时，取消job执行。单位为秒，0则为无限制。过短可能会被系统漏掉，小于一小时可能会有夏令时切换的问题")
    max_triggers: int = Field(default=0, ge=0, description="计划最大触发次数（包括因超时未成功执行job），0表示无限制")
    time_reference: Literal['real_world', 'agent_world', 'agent_subjective'] = Field(default='real_world', description="基于何种时间计算scheduled系列参数。当为agent_subjective时，不能设置任何scheduled系列参数，只能使用interval系列参数来重复触发")
    time_zone: Optional[SerializableTimeZone] = Field(default=None, description="计算scheduled系列参数时使用的时区，若没有则使用tick输入的datetime的时区或是自动获取当前时区")
    trigger_time: Union[TimestampUs, int] = Field(default=-1, description="下次触发时间的微秒数。如果设置为负数int则跳过这次触发（不消耗trigger次数，不会使一次性计划直接失效）。如果time_reference为agent_subjective，这个值则为int而非TimestampUs")
    trigger_count: int = Field(default=0, description="已触发次数（包括超时时）")
    added: bool = Field(default=False, description="计划是否已被添加")
    deleted: bool = Field(default=False, description="计划是否已被移除。不保证可靠，因为有可能从其他地方被移除")
    repeating: bool = Field(default=False, description="当前是否已处于计划重复阶段，根据下次触发时间是否被计算过来判断。主要用于当agent时间发生变化时（准确来说是倒退时），是否需要根据可能存在的scheduled系列参数重新计算下次触发时间")

    @field_validator("job", mode="after")
    @classmethod
    def validate_job(cls, v: Callable) -> Callable:
        if v.__name__ == "<lambda>":
            raise ValueError("Lambda functions are not persistable")
        if "<locals>" in v.__qualname__:
            raise ValueError("Local/nested functions are not persistable")
        return v

    @field_validator("job_args", mode="after")
    @classmethod
    def validate_job_args(cls, v: list[Any]) -> list[Any]:
        try:
            json.dumps(v)
        except TypeError:
            raise ValueError("Job args cannot be serialized")
        return v

    @field_validator("job_kwargs", mode="after")
    @classmethod
    def validate_job_kwargs(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(v)
        except TypeError:
            raise ValueError("Job kwargs cannot be serialized")
        return v

    @field_validator("trigger_time", mode="plain")
    @classmethod
    def validate_trigger_time(cls, v: Union[TimestampUs, int], info: ValidationInfo) -> Union[TimestampUs, int]:
        is_strict = bool(info.config and info.config.get('strict'))
        if v < 0:
            if is_strict and type(v) is not int:
                raise ValueError("当trigger_time为负数时，在strict模式下必须为int")
            else:
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    raise ValueError("当trigger_time为负数时，必须可转换为int")
        elif info.data['time_reference'] == 'agent_subjective':
            if is_strict:
                if type(v) is not int:
                    raise ValueError("当time_reference为agent_subjective时，trigger_time在strict模式下必须为int")
            else:
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    raise ValueError("当time_reference为agent_subjective时，trigger_time必须可转换为int")
        else:
            if is_strict:
                if not isinstance(v, TimestampUs):
                    raise ValueError("trigger_time不是一个TimestampUs实例，在strict模式下必须为TimestampUs")
            else:
                v = TimestampUs(v)
        return v

    @model_validator(mode="after")
    def validate_schedule_parameters(self) -> Self:
        # if (
        #     self.interval_fixed or
        #     (
        #         self.interval_random_min and
        #         self.interval_random_max
        #     )
        # ):
        #     has_interval = True
        # else:
        #     has_interval = False
        if (
            self.scheduled_every_day or
            self.scheduled_weekdays or
            self.scheduled_monthdays or
            self.scheduled_every_month or
            self.scheduled_months
        ):
            if not self.scheduled_time_of_day:
                raise ValueError("当scheduled_every_day、scheduled_weekdays、scheduled_monthdays、scheduled_every_month、scheduled_months中任意一个参数被指定时，scheduled_time_of_day也必须指定")
            has_scheduled = True
        elif self.scheduled_time_of_day:
            raise ValueError("当scheduled_time_of_day被指定时，至少还需设置其他任何一个scheduled系列参数")
        else:
            has_scheduled = False
        if self.time_reference == 'agent_subjective' and has_scheduled:
            raise ValueError("当time_reference为agent_subjective时，不能设置任何scheduled系列参数，因为agent_subjective_duration是时长，而不是具体时间，只能使用interval系列参数来重复触发")
        return self

    @computed_field
    @property
    def job_module(self) -> str:
        return self.job.__module__

    @computed_field
    @property
    def job_func(self) -> str:
        return self.job.__qualname__

    def tick(self, current_time: Union[Times, datetime, TimestampUs, int]) -> tuple[bool, Optional[dict[str, Any]], bool]:
        """
        计算Schedule是否应更新？是否应执行job？

        会同步更新实例属性

        Args:
            current_time: 当前时间。如果输入的是Times实例，则会自动使用合适的时间类型计算，否则需调用者自行确认时间类型，若输入的datetime没有时区信息，则使用当前时区。int类型仅用于agent_subjective时间参考，其他时间参考请使用Times或TimestampUs。

        Returns:
            输出一个tuple，按顺序包含以下内容：

            should_update: Schedule是否需更新或删除

            schedule: 若Schedule需更新，则返回一个包含新值的dict。否则返回None，表示无更新或应移除

            should_execute: 是否应执行相应任务
        """
        # 检查是否已删除，若已删除则不应出现此次调用
        if self.deleted:
            logger.warning(f"Schedule {self.schedule_id} has been deleted, shouldn't call tick.")
            return False, None, False

        # 检查是否已超过最大循环次数
        if self.max_triggers > 0 and self.trigger_count >= self.max_triggers:
            self.deleted = True
            return True, None, False

        # 如果输入是Times实例，则自动使用合适时间类型计算
        if isinstance(current_time, Times):
            if self.time_reference == 'real_world':
                current_timestampus = current_time.real_world_timestampus
            elif self.time_reference == 'agent_world':
                current_timestampus = current_time.agent_world_timestampus
            elif self.time_reference == 'agent_subjective':
                current_timestampus = current_time.agent_subjective_tick
            else:
                raise ValueError(f"Invalid time_reference: {self.time_reference}")
        elif type(current_time) is int:
            if self.time_reference != 'agent_subjective':
                raise ValueError("int类型只能用于agent_subjective时间参考！")
            else:
                current_timestampus = current_time
        else:
            current_timestampus = TimestampUs(current_time)

        # 负数表示无需触发，但需要计算下次触发时间，且不增加trigger_count
        trigger_is_negative = False
        if self.trigger_time < 0:
            trigger_is_negative = True
            not_timeout = False

        # 如果当前时间小于下次触发时间，则直接返回
        elif current_timestampus < self.trigger_time:
            return False, None, False

        # 检查是否超时
        elif (
            self.timeout_seconds > 0.0 and
            current_timestampus > (self.trigger_time + self.timeout_seconds * 1_000_000)
        ):
            not_timeout = False
        else:
            not_timeout = True

        # 如果没有计划和间隔，则等于一次性计划（除非当trigger_time为负数时）
        if (
            not trigger_is_negative and
            self.scheduled_time_of_day is None and
            (not self.interval_fixed and (not self.interval_random_min or not self.interval_random_max))
        ):
            self.deleted = True
            return True, None, not_timeout

        new_values = self.calc_trigger_time(current_time)
        # 返回None则表示schedule之前就已触发完毕。又或是参数设置错误，下次触发时间永远不会变化
        if new_values is None:
            return True, None, False

        # 是否达到最大触发次数
        if not trigger_is_negative:
            self.trigger_count += 1
            new_values["trigger_count"] = self.trigger_count
            if self.max_triggers > 0 and self.trigger_count >= self.max_triggers:
                self.deleted = True
                return True, None, not_timeout

        return True, new_values, not_timeout

    async def process(self, current_time: Union[Times, datetime, TimestampUs, int]) -> tuple[bool, Optional[dict[str, Any]], bool]:
        """若想要单独处理schedule，请使用此方法。会在方法内直接完成更新、删除、执行操作。"""
        if self.deleted:
            logger.warning(f"Schedule {self.schedule_id} has been deleted, shouldn't call process.")
            return False, None, False
        should_update, new_values, should_execute = self.tick(current_time)
        if should_update:
            if new_values:
                update_schedules([new_values])
            elif new_values is None:
                delete_schedules([self.schedule_id])
            else:
                logger.warning(f"schedule {self.schedule_id} 的tick疑似返回了空字典：{new_values}，将跳过此schedule")
        if should_execute:
            await self.call_job()
        return should_update, new_values, should_execute

    async def add_to_db(self) -> None:
        """添加schedule到数据库。"""
        self.added = True
        await add_schedules([self])

    async def update_to_db(self, new_values: Optional[dict[str, Any]] = None) -> None:
        """更新schedule到数据库。"""
        if new_values is None:
            values = self.dump_for_db()
        else:
            if new_values.get('schedule_id'):
                schedule_id = new_values['schedule_id']
                if schedule_id != self.schedule_id:
                    raise ValueError(f"new_values 中的 schedule_id {schedule_id} 与实例的 schedule_id {self.schedule_id} 不一致")
            else:
                schedule_id = self.schedule_id
            values = new_values.copy()
            values['schedule_id'] = schedule_id
        await update_schedules([values])

    async def delete_from_db(self) -> None:
        """从数据库删除schedule。"""
        self.deleted = True
        await delete_schedules([self.schedule_id])

    async def call_job(self) -> Any:
        """调用计划任务。纯粹的调用，不进行任何检查，对自身实例没有副作用"""
        sig = inspect.signature(self.job)
        params = list(sig.parameters.values())
        needs_schedule = len(params) > 0 and isinstance(params[0].annotation, type) and issubclass(params[0].annotation, Schedule)

        if inspect.iscoroutinefunction(self.job):
            if needs_schedule:
                return await self.job(self, *self.job_args, **self.job_kwargs)
            else:
                return await self.job(*self.job_args, **self.job_kwargs)
        else:
            if needs_schedule:
                return self.job(self, *self.job_args, **self.job_kwargs)
            else:
                return self.job(*self.job_args, **self.job_kwargs)

    def dump_for_db(self) -> dict[str, Any]:
        d = self.model_dump()
        del d['job']
        d['job_args'] = json.dumps(d['job_args'])
        d['job_kwargs'] = json.dumps(d['job_kwargs'])
        d['scheduled_weekdays'] = json.dumps(list(d['scheduled_weekdays']))
        d['scheduled_monthdays'] = json.dumps(list(d['scheduled_monthdays']))
        d['scheduled_months'] = json.dumps(list(d['scheduled_months']))
        del d['time_zone']
        if self.time_zone is not None:
            d['time_zone_name'] = self.time_zone.name
            d['time_zone_offset'] = self.time_zone.offset
        else:
            d['time_zone_name'] = ''
            d['time_zone_offset'] = None
        del d['added']
        del d['deleted']
        return d

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> Self:
        kwargs = {SCHEDULE_KEYS[i]: v for i, v in enumerate(row)}
        kwargs['job_args'] = json.loads(kwargs['job_args'])
        kwargs['job_kwargs'] = json.loads(kwargs['job_kwargs'])
        kwargs['scheduled_weekdays'] = set(json.loads(kwargs['scheduled_weekdays']))
        kwargs['scheduled_monthdays'] = set(json.loads(kwargs['scheduled_monthdays']))
        kwargs['scheduled_months'] = set(json.loads(kwargs['scheduled_months']))
        module = importlib.import_module(kwargs.pop('job_module'))
        kwargs['job'] = getattr(module, kwargs.pop('job_func'))
        time_zone_name = kwargs.pop('time_zone_name')
        time_zone_offset = kwargs.pop('time_zone_offset')
        if time_zone_name:
            kwargs['time_zone'] = SerializableTimeZone(name=time_zone_name, offset=time_zone_offset)
        else:
            kwargs['time_zone'] = None
        kwargs['added'] = True
        return cls.model_validate(kwargs)

    class SameTimeError(Exception):
        """当计算下次触发时间时，发现与当前的触发时间相同（没有变化）"""
        pass

    def calc_trigger_time(
        self,
        current_time: Union[Times, datetime, TimestampUs, int]
    ) -> Optional[dict[str, Any]]:
        """直接计算下次触发时间，会同时更新实例属性。返回None则表示schedule之前就已触发完毕，应被删除。

        对于current_time的输入类型：
        - Times适用于所有情况
        - TimestampUs适用于real_world和agent_subjective，对于real_world来说，TimestampUs会转换为datetime，时区UTC
        - int只适用于agent_subjective

        ### Raises:
            Schedule.SameTimeError: 当计算结果与当前触发时间相同（没有变化）时抛出
        """
        if isinstance(current_time, TimestampUs):
            if self.time_reference == 'real_world':
                current_datetime = current_time.to_datetime()
            elif self.time_reference == 'agent_subjective':
                next_trigger_time = int(current_time)
            elif self.time_reference == 'agent_world':
                raise ValueError("当输入为TimestampUs时，不能计算agent_world的下次触发时间！")
            else:
                raise ValueError(f"Invalid time_reference: {self.time_reference}")
        # 如果输入是Times实例，则自动使用合适时间类型计算
        elif isinstance(current_time, Times):
            if self.time_reference == 'real_world':
                current_datetime = current_time.real_world_datetime
            elif self.time_reference == 'agent_world':
                current_datetime = current_time.agent_world_datetime
            elif self.time_reference == 'agent_subjective':
                next_trigger_time = current_time.agent_subjective_tick
            else:
                raise ValueError(f"Invalid time_reference: {self.time_reference}")
        elif isinstance(current_time, datetime):
            if self.time_reference == 'agent_subjective':
                raise ValueError("agent_subjective只能用Times或TimestampUs来计算下次触发时间！当前输入为datetime")
            current_datetime = current_time
            #current_timeseconds = datetime_to_seconds(current_datetime)
        else:
            if self.time_reference != 'agent_subjective':
                raise ValueError("当前输入为int时，只接受agent_subjective的时间参考！")
            next_trigger_time = current_time

        if self.time_reference != 'agent_subjective':

            # 确保current_time有时区信息
            if current_datetime.tzinfo is None:
                current_datetime = current_datetime.replace(tzinfo=get_localzone())

            # 如果指定时区，则进行转换
            if self.time_zone is not None:
                current_datetime = current_datetime.astimezone(self.time_zone.tz())

            # 初始化下次触发时间为当前时间
            next_trigger_datetime = current_datetime

            # 处理指定的一天中的时间
            if self.scheduled_time_of_day is not None:
                daily_time = seconds_to_datetime(self.scheduled_time_of_day)
                # 设置当天的时间
                next_trigger_datetime = next_trigger_datetime.replace(
                    hour=daily_time.hour,
                    minute=daily_time.minute,
                    second=daily_time.second,
                    microsecond=daily_time.microsecond
                )

                if next_trigger_datetime <= current_datetime:
                    # 如果设置的时间已经过去且是每天都触发，则移到第二天
                    if self.scheduled_every_day:
                        next_trigger_datetime += timedelta(days=1)
                    elif self.scheduled_weekdays or self.scheduled_monthdays:

                        weekday_distance = 99
                        monthday_distance = 99
                        # 处理星期几的限制
                        if self.scheduled_weekdays:
                            weekday_distance = 0
                            next_trigger_weekdays = next_trigger_datetime
                            while (next_trigger_weekdays.isoweekday()) not in self.scheduled_weekdays:
                                if weekday_distance >= 7:
                                    logger.warning(f"循环超过7次，schedule {self.schedule_id} 的weekdays参数有误，请检查：{str(self.scheduled_weekdays)}")
                                    weekday_distance = 999
                                    break
                                next_trigger_weekdays += timedelta(days=1)
                                weekday_distance += 1

                        # 处理每月几号的限制
                        if self.scheduled_monthdays:
                            monthday_distance = 0
                            next_trigger_monthdays = next_trigger_datetime
                            last_day = (current_datetime + relativedelta(day=31)).day # 获取当前月份的最后一天
                            monthdays = [min(d, last_day) for d in self.scheduled_monthdays]
                            while next_trigger_monthdays.day not in monthdays:
                                if monthday_distance >= 31:
                                    logger.warning(f"循环超过31次，schedule {self.schedule_id} 的monthdays参数有误，请检查：{str(self.scheduled_monthdays)}")
                                    monthday_distance = 999
                                    break
                                next_trigger_monthdays += timedelta(days=1)
                                monthday_distance += 1

                        if weekday_distance >= 99 and monthday_distance >= 99:
                            raise ValueError(f"schedule {self.schedule_id} 的weekdays和monthdays参数都存在错误，无法计算！")
                        next_trigger_datetime = next_trigger_weekdays if weekday_distance < monthday_distance else next_trigger_monthdays

                    # 处理月份的限制
                    if not self.scheduled_every_month:
                        if self.scheduled_months:
                            month_loop_times = 0
                            while next_trigger_datetime.month not in self.scheduled_months:
                                if month_loop_times >= 12:
                                    raise ValueError(f"循环超过12次，schedule {self.schedule_id} 的months参数有误，请检查：{str(self.scheduled_months)}")
                                next_trigger_datetime += relativedelta(months=1)
                                month_loop_times += 1
                        # 非every_month且没有设置months意为计划只在当月生效，如果不是同一个月，视为计时器已触发完毕
                        elif next_trigger_datetime.month != current_datetime.month or next_trigger_datetime.year != current_datetime.year:
                            self.deleted = True
                            return None

            next_trigger_time = TimestampUs(next_trigger_datetime)

        # 添加间隔时间
        if self.interval_fixed:
            interval_seconds = self.interval_fixed
        elif self.interval_random_min and self.interval_random_max:
            interval_seconds = random.uniform(self.interval_random_min, self.interval_random_max)
        else:
            interval_seconds = None
        if interval_seconds:
            next_trigger_time += int(interval_seconds * 1_000_000)

        # 如果计算得出下次触发时间与当前没有变化，返回一个异常
        # 一般情况下比如，在tick中，出现这种情况意味着异常，可能是参数设置错误
        # 而如果调用者主动调用该方法，就是想看看时间有没有需要更新，那么需要try&except SameTimeError
        if next_trigger_time == self.trigger_time:
            raise self.SameTimeError(f"schedule {self.schedule_id} 的下次触发时间计算结果意外地与当前的触发时间相同！")

        new_values = {'schedule_id': self.schedule_id}
        self.trigger_time = next_trigger_time
        new_values['trigger_time'] = int(next_trigger_time)
        if not self.repeating:
            self.repeating = True
            new_values['repeating'] = True
        return new_values

    def format_schedule(
        self,
        fallback_time_zone: Optional[SerializableTimeZone] = None,
        prefix: str = '计划',
        include_id: bool = True,
        include_type: bool = True
    ) -> str:
        if self.time_reference == 'agent_subjective':
            formated_next_trigger_datetime = str(int(self.trigger_time))
        else:
            next_trigger_datetime = self.trigger_time.to_datetime()
            if self.time_zone is not None:
                next_trigger_datetime = next_trigger_datetime.astimezone(self.time_zone.tz())
            elif fallback_time_zone is not None:
                next_trigger_datetime = next_trigger_datetime.astimezone(fallback_time_zone.tz())
            else:
                raise ValueError("schedule自身没有指定时区的情况下，格式化时必须提供一个时区")
            formated_next_trigger_datetime = format_time(next_trigger_datetime)


        formated_scheduled = ''
        if self.scheduled_time_of_day is not None:

            if self.scheduled_every_month:
                formated_scheduled = "每月的"
            elif self.scheduled_months:
                formated_scheduled = f"每年{'、'.join([f'{month}月' for month in self.scheduled_months])}的"
            else:
                formated_scheduled = "仅限当月的"

            if self.scheduled_every_day:
                formated_scheduled += "每天的"
            elif self.scheduled_weekdays or self.scheduled_monthdays:
                if self.scheduled_weekdays:
                    formated_scheduled += f"{'、'.join([f'周{day}' for day in self.scheduled_weekdays])}"
                    if self.scheduled_monthdays:
                        formated_scheduled += "和"
                if self.scheduled_monthdays:
                    formated_scheduled += f"{'、'.join([f'{day}号' for day in self.scheduled_monthdays])}"
            else:
                formated_scheduled += f"{next_trigger_datetime.day}号"

            formated_scheduled += (datetime(1,1,1) + timedelta(seconds=self.scheduled_time_of_day)).strftime('的%H点%M分%S秒')

        if (
            self.interval_fixed or
            (self.interval_random_min and self.interval_random_max)
        ):
            if self.scheduled_time_of_day is not None:
                formated_scheduled += "，再加上"
                if self.interval_fixed:
                    if self.time_reference != 'agent_subjective':
                        formated_scheduled += f"{self.interval_fixed}秒的间隔"
                    else:
                        formated_scheduled += f"{int(self.interval_fixed * 1_000_000)}个单位的间隔"
                else:
                    if self.time_reference != 'agent_subjective':
                        formated_scheduled += f"{self.interval_random_min}秒到{self.interval_random_max}秒的随机间隔"
                    else:
                        formated_scheduled += f"{int(self.interval_random_min * 1_000_000)}个单位到{int(self.interval_random_max * 1_000_000)}个单位的随机间隔"
            else:
                if self.interval_fixed:
                    if self.time_reference != 'agent_subjective':
                        formated_scheduled = f"每间隔{self.interval_fixed}秒"
                    else:
                        formated_scheduled = f"每间隔{int(self.interval_fixed * 1_000_000)}个单位"
                else:
                    if self.time_reference != 'agent_subjective':
                        formated_scheduled = f"每随机间隔{self.interval_random_min}秒到{self.interval_random_max}秒"
                    else:
                        formated_scheduled = f"每随机间隔{int(self.interval_random_min * 1_000_000)}个单位到{int(self.interval_random_max * 1_000_000)}个单位"

        return f'''{f'{prefix}ID：{self.schedule_id}\n' if include_id else ''}{f'{prefix}类型：{self.schedule_type}\n' if include_type else ''}
{prefix}下次运行时间：{formated_next_trigger_datetime}
{prefix}重复时间：{formated_scheduled or '该计划不可重复'}
{prefix}最大触发次数：{self.max_triggers if self.max_triggers > 0 else '无限次'}
{prefix}已触发次数：{self.trigger_count}'''

    class Condition(BaseModel):
        """用于get_schedules的查询条件"""
        key: AnyScheduleKey
        op: Literal['=', '!=', '<', '<=', '>', '>='] = Field(default='=')
        value: Any


async def init_schedules_db():
    """初始化数据库和表"""
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS schedules (
                agent_id TEXT NOT NULL,
                schedule_id TEXT PRIMARY KEY,
                schedule_type TEXT NOT NULL DEFAULT '',
                job_module TEXT NOT NULL,
                job_func TEXT NOT NULL,
                job_args TEXT NOT NULL DEFAULT '[]',
                job_kwargs TEXT NOT NULL DEFAULT '{}',
                interval_fixed REAL NOT NULL DEFAULT 0.0,
                interval_random_min REAL NOT NULL DEFAULT 0.0,
                interval_random_max REAL NOT NULL DEFAULT 0.0,
                scheduled_time_of_day REAL,
                scheduled_every_day BOOLEAN NOT NULL DEFAULT 0,
                scheduled_weekdays TEXT NOT NULL DEFAULT '[]',
                scheduled_monthdays TEXT NOT NULL DEFAULT '[]',
                scheduled_every_month BOOLEAN NOT NULL DEFAULT 0,
                scheduled_months TEXT NOT NULL DEFAULT '[]',
                timeout_seconds REAL NOT NULL DEFAULT 0.0,
                max_triggers INTEGER NOT NULL DEFAULT 0,
                time_reference TEXT NOT NULL DEFAULT 'real_world' CHECK(time_reference IN ('real_world', 'agent_world', 'agent_subjective')),
                time_zone_name TEXT NOT NULL DEFAULT '',
                time_zone_offset REAL,
                trigger_time INTEGER NOT NULL DEFAULT -1,
                trigger_count INTEGER NOT NULL DEFAULT 0,
                repeating BOOLEAN NOT NULL DEFAULT 0
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_trigger_time ON schedules (trigger_time) WHERE time_reference = 'real_world'")
        #await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON schedules (agent_id)")
        #await db.execute("CREATE INDEX IF NOT EXISTS idx_schedule_type ON schedules (schedule_type)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_agent_and_type ON schedules (agent_id, schedule_type)")
        await db.commit()

async def get_schedules(
    where: Optional[list[Schedule.Condition]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    order_by: Optional[AnyScheduleKey] = None,
    order: Literal['ASC', 'DESC'] = 'ASC'
) -> list[Schedule]:
    """灵活查询schedule

    Args:
        where: 过滤条件
        limit: 限制返回数量
        offset: 跳过数量
        order_by: 排序字段
        order: 排序方向
    """
    conds = []
    params = []

    if where:
        for cond in where:
            conds.append(f"{cond.key} {cond.op} ?")
            params.append(cond.value)

    if conds:
        where_clause = f" WHERE {" AND ".join(conds)}"
    else:
        where_clause = ""

    limit_clause = f" LIMIT {limit}" if limit is not None else ""
    offset_clause = f" OFFSET {offset}" if offset is not None else ""

    if order_by:
        if order_by not in SCHEDULE_KEYS:
            raise ValueError(f"order_by 指定的字段 {order_by} 不存在")
        if order.upper() not in ['ASC', 'DESC']:
            raise ValueError(f"order 指定的方向 {order} 不存在")
        order_by = f' ORDER BY {order_by} {order.upper()}'
    else:
        order_by = ''

    sql = f"SELECT {', '.join(SCHEDULE_KEYS)} FROM schedules{where_clause}{order_by}{limit_clause}{offset_clause}"

    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
            return [Schedule.from_row(row) for row in rows]

async def add_schedules(schedules: list[Schedule]) -> None:
    if not schedules:
        return
    async with aiosqlite.connect(DATABASE_PATH) as db:
        for schedule in schedules:
            dumped = schedule.dump_for_db()
            keys = dumped.keys()
            if len(keys) != len(SCHEDULE_KEYS):
                raise ValueError(f"{schedule} 的键值对数量与预定义不一致")
            try:
                await db.execute(
                    f"INSERT INTO schedules ({', '.join(keys)}) VALUES ({', '.join(['?'] * len(keys))})",
                    [v for v in dumped.values()]
                )
                schedule.added = True
            except aiosqlite.IntegrityError as e:
                logger.error(f'schedule添加失败，大概率是id重复，将跳过这个schedule: {e}')
        await db.commit()

async def update_schedules(schedules: list[Union[Schedule, dict[str, Any]]]) -> None:
    if not schedules:
        return
    async with aiosqlite.connect(DATABASE_PATH) as db:
        for schedule in schedules:
            if isinstance(schedule, Schedule):
                dumped_schedule = schedule.dump_for_db()
            else:
                dumped_schedule = schedule.copy()
            schedule_id = dumped_schedule.pop('schedule_id')
            cursor = await db.execute(
                f"UPDATE schedules SET {', '.join([f'{k} = ?' for k in dumped_schedule.keys()])} WHERE schedule_id = ?",
                [v for v in dumped_schedule.values()] + [schedule_id]
            )
            if cursor.rowcount == 0:
                logger.error(f"schedule更新失败，可能是由于找不到id为{schedule_id}的schedule")
        await db.commit()

async def delete_schedules(schedules: list[Union[Schedule, str]]) -> None:
    if not schedules:
        return
    schedules_len = len(schedules)
    schedule_ids = set()
    for schedule in schedules:
        if isinstance(schedule, Schedule):
            schedule_ids.add(schedule.schedule_id)
            if not schedule.deleted:
                schedule.deleted = True
        else:
            schedule_ids.add(schedule)
    async with aiosqlite.connect(DATABASE_PATH) as db:
        cursor = await db.execute(
            f"DELETE FROM schedules WHERE schedule_id IN ({', '.join(['?'] * len(schedule_ids))})",
            list(schedule_ids)
        )
        if cursor.rowcount != schedules_len:
            logger.warning(f"有{schedules_len-cursor.rowcount}个schedule删除失败，可能是由于找不到指定id的schedule（已经被删除了）")
        await db.commit()


ticking = False
async def tick_schedules(real_world_time: Optional[Union[datetime, TimestampUs]] = None) -> None:
    global ticking
    if ticking:
        logger.warning("tick schedules 已在运行，将跳过")
        return
    ticking = True
    logger.debug("开始tick schedules")

    try:
        schedules_to_execute: list[Schedule] = []
        schedule_ids_to_delete = []
        schedules_to_update = []

        current_times_caches = {}

        if real_world_time is None:
            current_datetime = nowtz()
        elif isinstance(real_world_time, TimestampUs):
            current_datetime = real_world_time.to_datetime()
        else:
            if real_world_time.tzinfo is None:
                current_datetime = real_world_time.replace(tzinfo=get_localzone())
            else:
                current_datetime = real_world_time

        def tick_schedule(schedule: Schedule, time: Union[Times, datetime]) -> None:
            should_update, new_values, should_execute = schedule.tick(time)
            if should_update:
                if new_values:
                    schedules_to_update.append(new_values)
                elif new_values is None:
                    schedule_ids_to_delete.append(schedule.schedule_id)
                else:
                    logger.warning(f"schedule {schedule.schedule_id} 的tick疑似返回了空字典：{new_values}，将跳过此schedule")
            if should_execute:
                schedules_to_execute.append(schedule)

        real_world_schedules = await get_schedules(where=[
            Schedule.Condition(key='time_reference', value='real_world'),
            Schedule.Condition(key='trigger_time', op='<=', value=TimestampUs(current_datetime)),
        ])
        for schedule in real_world_schedules:
            tick_schedule(schedule, current_datetime)

        agent_schedules = await get_schedules(where=[
            Schedule.Condition(key='time_reference', op='!=', value='real_world'),
        ])
        for schedule in agent_schedules:
            if schedule.agent_id not in current_times_caches:
                time_settings = (await store_manager.get_settings(schedule.agent_id)).main.time_settings
                current_times_caches[schedule.agent_id] = Times.from_time_settings(time_settings, current_datetime)
            tick_schedule(schedule, current_times_caches[schedule.agent_id])

        logger.debug(f"有{len(schedules_to_execute)}个schedule需要执行")
        logger.debug(f"有{len(schedules_to_update)}个schedule需要更新")
        logger.debug(f"有{len(schedule_ids_to_delete)}个schedule需要删除")

        await delete_schedules(schedule_ids_to_delete)
        await update_schedules(schedules_to_update)

        schedules_to_execute.sort(key=lambda x: x.trigger_time)
        for schedule in schedules_to_execute:
            await schedule.call_job()
            logger.debug(f"schedule {schedule.schedule_id} 执行完成")

        logger.debug("所有schedule tick完成")

    finally:
        ticking = False
