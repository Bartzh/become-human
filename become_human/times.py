"""核心思想是用seconds替代timestamp以解决timestamp范围过小的问题，使用seconds可表示1~9999年的所有时间。然后是agent要有自己的时间，以锚点、时间膨胀和时区实现"""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationInfo
from typing import Any, Optional, Self, Union
from tzlocal import get_localzone_name, get_localzone
import re

def nowtz() -> datetime:
    """now() but with local ZoneInfo"""
    return datetime.now(get_localzone())

def utcnow() -> datetime:
    """now() but with UTC"""
    return datetime.now(timezone.utc)

def datetime_to_seconds(dt: datetime) -> float:
    """将datetime转换为秒数，类似于timestamp但支持datetime的所有时间范围，与timestamp一样为UTC时区

    **注意**，如果提供的datetime没有时区信息，则会被当成本地时区"""
    # 如果没有时区信息，默认它为本地时间。对于now()来说会很有用
    if dt.tzinfo is None:
       dt = dt.replace(tzinfo=get_localzone())
    # 会自动转换成utc然后计算秒数，时区偏移量会影响total_seconds，所以必须使用0偏移量的utc计算
    return (dt - datetime(1, 1, 1, tzinfo=timezone.utc)).total_seconds()

def seconds_to_datetime(seconds: float) -> datetime:
    """将秒数转换为datetime，保持UTC时区"""
    return datetime(1, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)

def now_seconds() -> float:
    """获取当前时间的秒数"""
    return datetime_to_seconds(datetime.now(timezone.utc))

class SerializableTimeZone(BaseModel):
    """agent的时区模型

    不设定offset的话会被当成ZoneInfo处理

    特殊情况是当name为"UTC"，offset为None或0，则会返回单例timezone.utc"""
    name: str = Field(description="时区名称")
    offset: Optional[float] = Field(default=None, gt=-86400.0, lt=86400.0, description="时区偏移，单位为秒")

    def tz(self) -> Union[timezone, ZoneInfo]:
        if self.name == "UTC" and not self.offset:
            return timezone.utc
        elif self.offset is None:
            return ZoneInfo(self.name)
        else:
            return timezone(timedelta(seconds=self.offset), self.name)

class AgentTimeSetting(BaseModel):
    real_time_anchor: float = Field(default=0.0, description="真实时间锚点，单位为秒")
    agent_time_anchor: float = Field(default=0.0, description="agent时间锚点，agent在此时间时真实世界的时间等于real_time_anchor。单位为秒")
    agent_time_scale: float = Field(default=1.0, description="相对于真实世界的时间膨胀，控制时间流逝速度")

class AgentTimeSettings(BaseModel):
    """agent的时间设置"""
    world_time_setting: AgentTimeSetting = Field(default_factory=AgentTimeSetting, description="agent世界时间设置")
    subjective_duration_setting: AgentTimeSetting = Field(default_factory=lambda: AgentTimeSetting(real_time_anchor=now_seconds()), description="agent主观时长设置，指agent体感（大脑）觉得自己经历了多久的时间。这个时间不能倒流")
    time_zone: SerializableTimeZone = Field(default_factory=lambda: SerializableTimeZone(name=get_localzone_name()), description="agent时区")

    @field_validator('subjective_duration_setting', mode='after')
    def validate_subjective_duration_setting(cls, v: AgentTimeSetting) -> AgentTimeSetting:
        if v.agent_time_scale < 0.0:
            raise ValueError("subjective_duration_setting的time_scale必须大于等于0.0，不能倒流！")
        return v

def real_time_to_agent_time(real_time: Union[datetime, float], setting: AgentTimeSetting, time_zone: Optional[SerializableTimeZone] = None) -> datetime:
    """将真实世界时间datetime转换为agent时间datetime

    如果提供了时区，返回的datetime将使用agent自己的时区，否则返回UTC时间"""
    if isinstance(real_time, datetime):
        real_timeseconds = datetime_to_seconds(real_time)
    else:
        real_timeseconds = real_time
    seconds = (real_timeseconds - setting.real_time_anchor) * setting.agent_time_scale + setting.agent_time_anchor
    agent_time = seconds_to_datetime(seconds)
    if time_zone is not None:
        agent_time = agent_time.astimezone(time_zone.tz())
    return agent_time

def real_seconds_to_agent_seconds(real_seconds: float, setting: AgentTimeSetting) -> float:
    """将真实世界时间秒数转换为agent时间秒数"""
    return (real_seconds - setting.real_time_anchor) * setting.agent_time_scale + setting.agent_time_anchor

def agent_seconds_to_datetime(seconds: float, time_zone: SerializableTimeZone) -> datetime:
    """将agent时间秒数转换为datetime

    虽然agent_seconds已包含时间膨胀与偏移，但时区还是UTC，使用此函数将其转换为agent时区的datetime"""
    return seconds_to_datetime(seconds).astimezone(time_zone.tz())

def now_agent_time(setting: AgentTimeSetting, time_zone: Optional[SerializableTimeZone] = None) -> datetime:
    """获取当前agent时间datetime"""
    return real_time_to_agent_time(utcnow(), setting, time_zone)

def now_agent_seconds(setting: AgentTimeSetting) -> float:
    """获取当前agent时间秒数"""
    return datetime_to_seconds(now_agent_time(setting))


AnyTz = Union[timezone, ZoneInfo, float, int, timedelta, SerializableTimeZone]
def format_time(time: Optional[Union[datetime, float]], time_zone: Optional[AnyTz] = None) -> str:
    """时间点格式化函数

    若输入是秒数，则可以选择再输入一个时区，这将输出时区转换后的时间。若无则保持UTC时间。"""
    if time is None:
        return "未知时间"
    try:
        if isinstance(time, (float, int)):
            time = seconds_to_datetime(time)
            if time_zone:
                if isinstance(time_zone, (float, int)):
                    tz = timezone(timedelta(seconds=time_zone))
                elif isinstance(time_zone, timedelta):
                    tz = timezone(time_zone)
                elif isinstance(time_zone, SerializableTimeZone):
                    tz = time_zone.tz()
                else:
                    tz = time_zone
                time = time.astimezone(tz)
        # TODO: 考虑再加上时区
        return time.strftime("%Y-%m-%d %H:%M:%S %A")
    except (OverflowError, OSError, ValueError):
        return "时间信息损坏"

def format_duration(duration: Union[datetime, float, int, timedelta]) -> str:
    """时长格式化函数"""
    decrease_one = False
    negative = False
    if isinstance(duration, (float, int)):
        if duration < 0:
            negative = True
            duration = abs(duration)
        delta = timedelta(seconds=duration)
        duration = datetime(1,1,1) + delta
        decrease_one = True
    elif isinstance(duration, timedelta):
        if duration.days < 0:
            negative = True
            duration = abs(duration)
        duration = datetime(1,1,1) + duration
        decrease_one = True
    year = duration.year
    month = duration.month
    day = duration.day
    hour = duration.hour
    minute = duration.minute
    second = duration.second
    if decrease_one:
        year -= 1
        month -= 1
        day -= 1
    result = ''
    if negative:
        result += '负'
    if year > 0:
        result += f'{year}年'
    if month > 0:
        result += f'{month}个月'
    if day > 0:
        result += f'{day}天'
    if hour > 0:
        result += f'{hour}小时'
    if minute > 0:
        result += f'{minute}分'
    if second > 0:
        result += f'{second}秒'
    return result


def parse_timedelta(time_str: str) -> timedelta:
    """
    使用正则表达式解析字符串，支持如下格式：
    - "2h" 表示2小时
    - "30m" 表示30分钟
    - "1d" 表示1天
    - "2h30m" 表示2小时30分钟
    - "1w" 表示1周

    支持的单位:
    - s: 秒
    - m: 分钟
    - h: 小时
    - d: 天
    - w: 周

    如：
    - 1d2h3m4s
    - 1d22d 2h2w  3d 4m22sqwdqwedwsqwe（22s之后这些会被忽略）

    返回timedelta对象
    """

    # 定义单位映射
    units = {
        's': 'seconds',
        'm': 'minutes',
        'h': 'hours',
        'd': 'days',
        'w': 'weeks'
    }

    # 正则表达式匹配数字和单位
    pattern = re.compile(r'(\d*\.?\d+)([smhdw])')
    matches = pattern.findall(time_str.lower())

    if not matches:
        raise ValueError(f"无法解析时间字符串: {time_str}")

    # 构建参数
    delta_args = {}
    for value, unit in matches:
        if delta_args.get(units[unit]):
            delta_args[units[unit]] += float(value)
        else:
            delta_args[units[unit]] = float(value)

    return timedelta(**delta_args)


class Times(BaseModel):
    """包含在某个时间点下所有可能需要的时间相关信息的结构

    这是一个不可变的数据结构

    无论如何都会将输入的datetime的时区转换（astimezone）至设置的时区，如果输入的datetime没有时区信息，也会为其替换时区（replace）"""
    real_world_time_zone: SerializableTimeZone
    real_world_datetime: datetime
    real_world_timeseconds: float = Field(default=None, validate_default=True)
    agent_time_settings: AgentTimeSettings
    agent_world_datetime: datetime
    agent_world_timeseconds: float = Field(default=None, validate_default=True)
    agent_subjective_duration: float

    model_config = ConfigDict(frozen=True)

    @field_validator("real_world_datetime", mode="after")
    @classmethod
    def real_world_datetime_validator(cls, value: datetime, info: ValidationInfo) -> datetime:
        time_zone: SerializableTimeZone = info.data['real_world_time_zone']
        if value.tzinfo is None:
            return value.replace(tzinfo=time_zone.tz())
        return value.astimezone(time_zone.tz())

    @field_validator("real_world_timeseconds", mode="before")
    @classmethod
    def real_world_timeseconds_validator(cls, value: Any, info: ValidationInfo) -> float:
        if value is None:
            return datetime_to_seconds(info.data['real_world_datetime'])
        return value

    @field_validator("agent_world_datetime", mode="after")
    @classmethod
    def agent_world_datetime_validator(cls, value: datetime, info: ValidationInfo) -> datetime:
        time_settings: AgentTimeSettings = info.data['agent_time_settings']
        if value.tzinfo is None:
            return value.replace(tzinfo=time_settings.time_zone.tz())
        return value.astimezone(time_settings.time_zone.tz())

    @field_validator("agent_world_timeseconds", mode="before")
    @classmethod
    def agent_world_timeseconds_validator(cls, value: Any, info: ValidationInfo) -> float:
        if value is None:
            return datetime_to_seconds(info.data['agent_world_datetime'])
        return value

    @classmethod
    def from_time_settings(cls, settings: AgentTimeSettings, time: Optional[Union[datetime, float]] = None) -> Self:
        """旨在需要两个以上的时间种类时方便地完成各类型时间的转换

        通过提供现实时间datetime或timeseconds（或留空取当前时间，本地时区）快速获取其他种类时间

        如果提供的datetime没有时区，会被当作本地时区。如果提供timeseconds，时区将为UTC"""
        if time is None:
            time = nowtz()
        if isinstance(time, datetime):
            if time.tzinfo is None:
                time = time.replace(tzinfo=get_localzone())
            real_world_datetime = time
            real_world_timeseconds = datetime_to_seconds(time)
        else:
            real_world_timeseconds = time
            real_world_datetime = seconds_to_datetime(time)
        if isinstance(real_world_datetime.tzinfo, ZoneInfo):
            real_world_time_zone = SerializableTimeZone(
                name=real_world_datetime.tzinfo.key
            )
        else:
            real_world_time_zone = SerializableTimeZone(
                name=real_world_datetime.tzname(),
                offset=real_world_datetime.utcoffset().total_seconds()
            )
        agent_world_datetime = real_time_to_agent_time(real_world_timeseconds, settings.world_time_setting, settings.time_zone)
        agent_subjective_duration = real_seconds_to_agent_seconds(real_world_timeseconds, settings.subjective_duration_setting)
        return cls(
            real_world_time_zone=real_world_time_zone,
            real_world_datetime=real_world_datetime,
            real_world_timeseconds=real_world_timeseconds,
            agent_time_settings=settings,
            agent_world_datetime=agent_world_datetime,
            agent_subjective_duration=agent_subjective_duration
        )
