"""核心思想是用seconds替代timestamp以解决timestamp范围过小的问题，使用seconds可表示1~9999年的所有时间。然后是agent要有自己的时间，以锚点、时间膨胀和时区实现"""
from typing import Any, Optional, Self, Union, overload, Literal
from functools import cached_property
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationInfo, computed_field
from pydantic_core import core_schema
from tzlocal import get_localzone_name, get_localzone
import re

def nowtz() -> datetime:
    """now() but with local ZoneInfo"""
    return datetime.now(get_localzone())

def utcnow() -> datetime:
    """now() but with UTC"""
    return datetime.now(timezone.utc)

def timedelta_to_microseconds(td: timedelta) -> int:
    """将timedelta转换为微秒数"""
    return td.days * 86400_000_000 + td.seconds * 1_000_000 + td.microseconds


EPOCH = datetime(1, 1, 1, tzinfo=timezone.utc)
MAX_TIMESTAMP_US = timedelta_to_microseconds(datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc) - EPOCH)
class TimestampUs(int):
    """微秒级时间戳，为int子类"""
    def __new__(cls, value: Union[int, float, timedelta, datetime, str, Self] = 0):
        if isinstance(value, cls):
            return value
        elif isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=get_localzone())
            # 做减法时会将所有datetime自动转换成utc计算
            value = timedelta_to_microseconds(value - EPOCH)
        elif isinstance(value, timedelta):
            value = timedelta_to_microseconds(value)
        elif isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
                if value.tzinfo is None:
                    value = value.replace(tzinfo=get_localzone())
                value = timedelta_to_microseconds(value - EPOCH)
            except ValueError:
                value = int(value)
        else:
            value = int(value)
        if value < 0 or value > MAX_TIMESTAMP_US:
            raise ValueError(f"TimestampUs must be in range [0, {MAX_TIMESTAMP_US}], got {value}")
        return super().__new__(cls, value)

    @classmethod
    def now(cls) -> Self:
        return cls(utcnow())

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> Any:
        def validate(value: Any, info: ValidationInfo) -> Self:
            if info.config and info.config.get('strict'):
                if not isinstance(value, cls):
                    raise ValueError("TimestampUs must be an instance of TimestampUs in strict mode")
                return value
            return cls(value)
        return core_schema.with_info_plain_validator_function(
            function=validate,
            json_schema_input_schema=core_schema.union_schema(
                [
                    core_schema.datetime_schema(),
                    core_schema.int_schema(ge=0),
                ],
                mode='left_to_right'
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(int)
        )

    def __repr__(self) -> str:
        return f"TimestampUs({format_time(self)})"

    def to_datetime(self) -> datetime:
        return EPOCH + timedelta(microseconds=self)

    def __add__(self, other: Union[timedelta, int, float, Self]) -> Self:
        if isinstance(other, timedelta):
            other = timedelta_to_microseconds(other)
        return self.__class__(super().__add__(int(other)))

    @overload
    def __sub__(self, other: Self) -> int: ...
    @overload
    def __sub__(self, other: Union[timedelta, int, float]) -> Self: ...

    def __sub__(self, other: Union[timedelta, int, float, Self]) -> Union[Self, int]:
        if isinstance(other, self.__class__):
            return super().__sub__(int(other))
        elif isinstance(other, timedelta):
            other = timedelta_to_microseconds(other)
        return self.__class__(super().__sub__(int(other)))

    @overload
    def __mul__(self, other: Self) -> int: ...
    @overload
    def __mul__(self, other: Union[timedelta, int, float]) -> Self: ...

    def __mul__(self, other: Union[timedelta, int, float]) -> Self:
        if isinstance(other, float) and other.is_integer():
            other = int(other)
        elif isinstance(other, self.__class__):
            return super().__mul__(other)
        elif isinstance(other, timedelta):
            other = timedelta_to_microseconds(other)
        return self.__class__(int(self) * other)

    # def __truediv__(self, other: Union[timedelta, int, float]) -> Self:
    #     if isinstance(other, timedelta):
    #         other = timedelta_to_microseconds(other)
    #     return self.__class__(super().__truediv__(int(other)))

    # def __floordiv__(self, other: Union[timedelta, int, float]) -> Self:
    #     if isinstance(other, timedelta):
    #         other = timedelta_to_microseconds(other)
    #     return self.__class__(super().__floordiv__(int(other)))


def seconds_to_datetime(seconds: float) -> datetime:
    """将秒数转换为datetime，保持UTC时区

    一般来说你不再应该需要这个函数，除非是想用比较小的数字如86400以内快速获取时间信息

    等效于`TimestampUs(seconds * 1_000_000).to_datetime()`"""
    return TimestampUs(seconds * 1_000_000).to_datetime()

class SerializableTimeZone(BaseModel):
    """可序列化时区模型

    不设定offset的话会被当成ZoneInfo处理

    特殊情况是当name为"UTC"，offset为None或0，则会返回单例timezone.utc。所以不会存在ZoneInfo('UTC')"""
    name: str = Field(description="时区名称")
    offset: Optional[float] = Field(default=None, gt=-86400.0, lt=86400.0, description="时区偏移，单位为秒")

    def tz(self) -> Union[timezone, ZoneInfo]:
        if self.name == "UTC" and not self.offset:
            return timezone.utc
        elif self.offset is None:
            return ZoneInfo(self.name)
        else:
            return timezone(timedelta(seconds=self.offset), self.name)

    @classmethod
    def from_timezone(cls, tz: Union[timezone, ZoneInfo]) -> Self:
        if tz is timezone.utc:
            return cls(name="UTC")
        elif isinstance(tz, ZoneInfo):
            return cls(name=tz.key)
        else:
            return cls(name=tz.tzname(None), offset=tz.utcoffset(None).total_seconds())

    @classmethod
    def from_datetime(cls, dt: datetime) -> Self:
        return cls.from_timezone(dt.tzinfo)

    @classmethod
    def from_local(cls) -> Self:
        return cls(name=get_localzone_name())

class AgentTimeSettings(BaseModel):
    """agent的时间设置"""
    world_real_anchor: TimestampUs = Field(default=TimestampUs(0), description="agent世界时间的真实时间锚点，单位为微秒。真实时间在此时间时agent世界时间等于world_agent_anchor")
    world_agent_anchor: TimestampUs = Field(default=TimestampUs(0), description="agent世界时间的agent时间锚点，单位为微秒")
    world_scale: Union[float, int] = Field(default=1, description="相对于真实世界的agent世界时间膨胀，控制时间流逝速度")
    # 默认使用当前时间做锚点，这可能是一个问题，如果要用过去的现实时间来计算的话会报错
    subjective_real_anchor: TimestampUs = Field(default_factory=TimestampUs.now, description="agent主观tick的真实时间锚点，单位为微秒。真实时间在此时间时agent主观tick等于subjective_agent_anchor")
    subjective_agent_anchor: int = Field(default=0, description="agent主观tick的agent锚点")
    subjective_scale: Union[float, int] = Field(default=1, description="相对于真实世界的agent主观tick膨胀，设置为0以放弃时间驱动")

    time_zone: SerializableTimeZone = Field(default_factory=SerializableTimeZone.from_local, description="agent时区")

    @field_validator('world_scale', mode='after')
    def validate_world_scale(cls, v: Union[float, int]) -> Union[float, int]:
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    @field_validator('subjective_scale', mode='after')
    def validate_subjective_scale(cls, v: Union[float, int]) -> Union[float, int]:
        if v < 0:
            raise ValueError("subjective_scale必须大于等于0")
        if isinstance(v, float) and v.is_integer():
            return int(v)
        return v

    def to_world_datetime(self, real_time: Optional[Union[datetime, TimestampUs]] = None) -> datetime:
        """将真实世界时间转换为agent时间datetime

        若未指定real_time，则自动获取当前时间"""
        real_timestampus = TimestampUs(real_time or utcnow())
        agent_time = self.world_agent_anchor + (real_timestampus - self.world_real_anchor) * self.world_scale
        return agent_time.to_datetime().astimezone(self.time_zone.tz())

    def to_subjective_tick(self, real_time: Optional[Union[datetime, TimestampUs]] = None) -> int:
        """将agent时间datetime转换为agent主观int

        若未指定real_time，则自动获取当前时间"""
        real_timestampus = TimestampUs(real_time or utcnow())
        agent_time = self.subjective_agent_anchor + (real_timestampus - self.subjective_real_anchor) * self.subjective_scale
        return int(agent_time)

    def add_offset_from_now(self, delta: Union[int, timedelta], time_type: Literal['world', 'subjective']) -> Self:
        """从现在开始为agent世界时间或主观tick添加时间偏移

        在world中，delta为微秒int，或使用timedelta实例

        在subjective中，delta只能为int类型"""
        new_time_settings = self.model_copy(deep=True)
        current_times = Times.from_time_settings(self)
        if time_type == 'world':
            new_time_settings.world_agent_anchor = current_times.agent_world_timestampus + delta
            new_time_settings.world_real_anchor = current_times.real_world_timestampus
        elif time_type == 'subjective':
            if isinstance(delta, timedelta):
                raise ValueError("类型为subjective时delta只能为int类型")
            new_time_settings.subjective_agent_anchor = current_times.agent_subjective_tick + delta
            new_time_settings.subjective_real_anchor = current_times.real_world_timestampus
        else:
            raise ValueError(f'未知的时间类型{time_type}')
        return new_time_settings

    def set_scale_from_now(self, scale: Union[float, int], time_type: Literal['world', 'subjective']) -> Self:
        """从现在开始为agent世界时间或主观tick设置时间膨胀"""
        new_time_settings = self.model_copy(deep=True)
        current_times = Times.from_time_settings(self)
        if time_type == 'world':
            new_time_settings.world_scale = scale
            new_time_settings.world_agent_anchor = current_times.agent_world_timestampus
            new_time_settings.world_real_anchor = current_times.real_world_timestampus
        elif time_type == 'subjective':
            new_time_settings.subjective_scale = scale
            new_time_settings.subjective_agent_anchor = current_times.agent_subjective_tick
            new_time_settings.subjective_real_anchor = current_times.real_world_timestampus
        else:
            raise ValueError(f'未知的时间类型{time_type}')
        return new_time_settings


AnyTz = Union[timezone, ZoneInfo, timedelta, SerializableTimeZone]
def format_time(time: Optional[Union[datetime, TimestampUs]], time_zone: Optional[AnyTz] = None) -> str:
    """时间点格式化函数

    若输入是TimestampUs，则可以选择再输入一个时区，这将输出时区转换后的时间。若无则保持UTC时间。"""
    if time is None:
        return "未知时间"
    try:
        if isinstance(time, TimestampUs):
            time = time.to_datetime()
            if time_zone:
                if isinstance(time_zone, timedelta):
                    tz = timezone(time_zone)
                elif isinstance(time_zone, SerializableTimeZone):
                    tz = time_zone.tz()
                else:
                    tz = time_zone
                time = time.astimezone(tz)
        elif not isinstance(time, datetime):
            return "时间信息损坏"
        # TODO: 考虑再加上时区
        return time.strftime("%Y-%m-%d %H:%M:%S %A")
    except (OverflowError, OSError, ValueError):
        return "时间信息损坏"

def format_duration(duration: Union[datetime, float, int, timedelta, TimestampUs], is_microseconds: bool = True) -> str:
    """时长格式化函数

    若输入是int或float，则将通过is_microseconds判断是否为微秒，或秒。"""
    decrease_one = False
    negative = False
    if isinstance(duration, (float, int)):
        if duration < 0:
            negative = True
            duration = abs(duration)
        if is_microseconds or isinstance(duration, TimestampUs):
            delta = timedelta(microseconds=duration)
        else:
            delta = timedelta(seconds=duration)
        duration = EPOCH + delta
        decrease_one = True
    elif isinstance(duration, timedelta):
        if duration.days < 0:
            negative = True
            duration = abs(duration)
        duration = EPOCH + duration
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
    """包含在某个现实时间点下所有可能需要的时间相关信息的结构

    这是一个不可变的数据结构"""
    real_world_timestampus: TimestampUs
    real_world_time_zone: SerializableTimeZone
    agent_time_settings: AgentTimeSettings
    #agent_world_datetime: datetime = Field(default=None, validate_default=True)

    model_config = ConfigDict(frozen=True)

    @computed_field
    @cached_property
    def real_world_datetime(self) -> datetime:
        return self.real_world_timestampus.to_datetime()

    # @field_validator('agent_world_datetime', mode='before')
    # @classmethod
    # def validate_agent_world_datetime(cls, v: datetime, info: ValidationInfo) -> datetime:
    #     if v is None:
    #         v = real_world_to_agent_world(info.data['real_world_timestampus'], info.data['agent_time_settings'])
    #     return v

    @computed_field
    @cached_property
    def agent_world_datetime(self) -> datetime:
        return self.agent_time_settings.to_world_datetime(self.real_world_timestampus)

    @computed_field
    @cached_property
    def agent_world_timestampus(self) -> TimestampUs:
        return TimestampUs(self.agent_world_datetime)

    @computed_field
    @cached_property
    def agent_subjective_tick(self) -> int:
        return self.agent_time_settings.to_subjective_tick(self.real_world_timestampus)

    @classmethod
    def from_time_settings(cls, settings: AgentTimeSettings, real_time: Optional[Union[datetime, TimestampUs, Self]] = None) -> Self:
        """旨在需要两个以上的时间种类时方便地完成各类型时间的转换

        通过提供现实时间datetime、TimestampUs或Times（或留空取当前时间，本地时区）快速获取其他种类时间

        如果提供的datetime没有时区，会被当作本地时区。如果提供TimestampUs，时区将为UTC。如果提供另一个Times，将会复制其现实时间点和现实时区（相当于提供带时区的datetime）"""
        if real_time is None:
            real_time = nowtz()
        if isinstance(real_time, datetime):
            if real_time.tzinfo is None:
                real_time = real_time.replace(tzinfo=get_localzone())
            real_world_timestampus = TimestampUs(real_time)
            real_world_time_zone = SerializableTimeZone.from_datetime(real_time)
        elif isinstance(real_time, TimestampUs):
            real_world_timestampus = real_time
            real_world_time_zone = SerializableTimeZone(name='UTC')
        else:
            real_world_timestampus = real_time.real_world_timestampus
            real_world_time_zone = real_time.real_world_time_zone
        return cls(
            real_world_timestampus=real_world_timestampus,
            real_world_time_zone=real_world_time_zone,
            agent_time_settings=settings
        )
