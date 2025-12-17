from pydantic import BaseModel, Field
from typing import Union, Optional, Self
from datetime import datetime, timezone, time, timedelta
from dateutil.relativedelta import relativedelta
import random
from warnings import warn
from tzlocal import get_localzone

from become_human.store import StoreModel, StoreField
from become_human.time import seconds_to_datetime, datetime_to_seconds


class Schedule(BaseModel):
    """如只设置interval，表示将按指定时间间隔运行。间隔时间总是在最后被加上

    如设置了time_of_day，则也至少需要设置every_day、weekdays或monthdays其中一项。"""
    interval: Optional[Union[float, tuple[float, float]]] = Field(default=None, description="间隔时间，固定时间或random.uniform(最小，最大)，单位为秒。在scheduled之后计算")
    scheduled_time_of_day: Optional[float] = Field(default=None, ge=0.0, le=86400.0, description="指定一天中的时间，单位为秒")
    scheduled_every_day: bool = Field(default=False, description="是否每天运行")
    scheduled_weekdays: Optional[list[int]] = Field(default=None, description="指定星期几运行，1-7分别表示周一到周日。可与monthdays重复指定，不会重复运行")
    scheduled_monthdays: Optional[list[int]] = Field(default=None, description="指定每月几号运行，1-31分别表示1-31号。可与weekdays重复指定，不会重复运行")
    scheduled_every_month: bool = Field(default=False, description="是否每月运行，为False表示只在当月运行")
    scheduled_months: Optional[list[int]] = Field(default=None, description="指定每年几月运行，1-12分别表示1-12月")
    timeout_seconds: float = Field(default=0.0, description="超时时间，指如果当前时间超过指定时间太久则算作超时。单位为秒，0则为无限制")
    max_loop_times: int = Field(default=0, ge=0, description="循环次数，0表示无限循环")
    agent_time_based: bool = Field(default=False, description="是否基于agent时间计算")

    next_run_timeseconds: float = Field(default=0.0, description="经过计算得出的下次执行时间的timeseconds")
    loop_times: int = Field(default=0, description="已运行次数，只有当存在max_loop_times时才会计算")

    def tick(self, current_datetime: datetime) -> tuple[Optional[Self], bool]:
        """
        计算Schedule是否应更新？是否应运行？是否应删除？

        Args:
            current_datetime: 当前时间

        Returns:
            输出一个tuple，按顺序包含以下内容：

            schedule: 若Schedule需更新，则返回一个新的Schedule示例。否则返回None

            should_execute: 是否应执行相应任务

            should_delete: 是否应删除Schedule。如因超出最大循环次数，或是设置存在错误
        """

        # 检查是否已超过最大循环次数
        if self.max_loop_times > 0 and self.loop_times >= self.max_loop_times:
            return None, False

        # 确保current_time有时区信息
        if current_datetime.tzinfo is None:
            current_datetime = current_datetime.replace(tzinfo=get_localzone())

        # 如果当前时间小于下次执行时间，则直接返回原Schedule
        current_time_seconds = datetime_to_seconds(current_datetime)
        if current_time_seconds <= self.next_run_timeseconds:
            return self.model_copy(deep=True), False

        # 检查是否超时
        elif self.timeout_seconds > 0.0 and current_time_seconds > (self.next_run_timeseconds + self.timeout_seconds):
            not_timeout = False
        else:
            not_timeout = True

        # 初始化下次运行时间为当前时间
        next_run = current_datetime

        # 处理指定的一天中的时间
        if self.scheduled_time_of_day is not None:
            daily_time = seconds_to_datetime(self.scheduled_time_of_day)
            # 设置当天的时间
            next_run = next_run.replace(hour=daily_time.hour, minute=daily_time.minute, second=daily_time.second, microsecond=daily_time.microsecond)

            # 如果设置的时间已经过去且是每天都运行，则移到第二天
            if next_run <= current_datetime:
                if self.scheduled_every_day:
                    next_run += timedelta(days=1)
                elif self.scheduled_weekdays or self.scheduled_monthdays:

                    weekday_distance = 99
                    monthday_distance = 99
                    # 处理星期几的限制
                    if self.scheduled_weekdays:
                        weekday_distance = 0
                        next_run_weekdays = next_run
                        while (next_run_weekdays.weekday() + 1) not in self.scheduled_weekdays:
                            if weekday_distance >= 7:
                                warn(f"循环超过7次，{self.__repr_name__} 的weekdays参数有误，请检查：{str(self.scheduled_weekdays)}")
                                weekday_distance = 999
                                break
                            next_run_weekdays += timedelta(days=1)
                            weekday_distance += 1

                    # 处理每月几号的限制
                    if self.scheduled_monthdays:
                        monthday_distance = 0
                        next_run_monthdays = next_run
                        last_day = (current_datetime + relativedelta(day=31)).day # 获取当前月份的最后一天
                        monthdays = [min(d, last_day) for d in self.scheduled_monthdays]
                        while next_run_monthdays.day not in monthdays:
                            if monthday_distance >= 31:
                                warn(f"循环超过31次，{self.__repr_name__} 的monthdays参数有误，请检查：{str(self.scheduled_monthdays)}")
                                monthday_distance = 999
                                break
                            next_run_monthdays += timedelta(days=1)
                            monthday_distance += 1

                    if weekday_distance >= 99 and monthday_distance >= 99:
                        raise ValueError(f"{self.__repr_name__} 的weekdays和monthdays参数存在错误，无法计算！")
                    next_run = next_run_weekdays if weekday_distance < monthday_distance else next_run_monthdays

                # 处理月份的限制
                if not self.scheduled_every_month:
                    if self.scheduled_months:
                        month_loop_times = 0
                        while next_run.month not in self.scheduled_months:
                            if month_loop_times >= 12:
                                raise ValueError(f"循环超过12次，{self.__repr_name__} 的months参数有误，请检查：{str(self.scheduled_months)}")
                            next_run += relativedelta(months=1)
                            month_loop_times += 1
                    # 非every_month且没有设置months意为计时器只在当月生效，如果不是同一个月，视为计时器已运行完毕
                    elif next_run.month != current_datetime.month or next_run.year != current_datetime.year:
                        return None, False

        # 添加间隔时间
        if self.interval is not None:
            if isinstance(self.interval, tuple):
                interval_seconds = random.uniform(self.interval[0], self.interval[1])
            else:
                interval_seconds = self.interval
            next_run += timedelta(seconds=interval_seconds)

        if next_run == current_datetime:
            raise ValueError(f"{self.__repr_name__} 不能在没有daily的情况下将interval设置为0.0，请检查：{str(self.interval)}")

        # 是否达到最大循环次数
        new_value = {"next_time_seconds": datetime_to_seconds(next_run)}
        if self.max_loop_times > 0:
            loop_times = self.loop_times + 1
            if loop_times >= self.max_loop_times:
                return None, not_timeout
            else:
                new_value["loop_times"] = loop_times

        return self.model_copy(update=new_value, deep=True), not_timeout


class MemoryUpdateSchedule(Schedule):
    stable_time_range: list[dict[str, float]] = Field(description="指定稳定时间范围，单位为秒")

class AgentSchedules(StoreModel):
    _namespace = ("schedules",)
    memory_update_schedules: list[MemoryUpdateSchedule] = StoreField(default_factory=lambda: [
        MemoryUpdateSchedule(interval=5.0, stable_time_range=[{'$gte': 0.0}, {'$lt': 43200.0}]),
        MemoryUpdateSchedule(interval=30.0, stable_time_range=[{'$gte': 43200.0}, {'$lt': 86400.0}]),
        MemoryUpdateSchedule(interval=60.0, stable_time_range=[{'$gte': 86400.0}, {'$lt': 864000.0}]),
        MemoryUpdateSchedule(interval=500.0, stable_time_range=[{'$gte': 864000.0}, {'$lt': 8640000.0}]),
        MemoryUpdateSchedule(interval=3600.0, stable_time_range=[{'$gte': 8640000.0}])
    ])
