"""核心思想是用seconds替代timestamp以解决timestamp范围过小的问题，使用seconds可表示1~9999年的所有时间。然后是agent要有自己的时间，以锚点、时间膨胀和时区实现"""
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pydantic import BaseModel, Field
from typing import Optional, Union
from tzlocal import get_localzone_name, get_localzone
import re

def nowtz() -> datetime:
    """now() but with local ZoneInfo"""
    return datetime.now(get_localzone())

def utcnow() -> datetime:
    """now() but with UTC"""
    return datetime.now(timezone.utc)

def datetime_to_seconds(dt: datetime) -> float:
    """将datetime转换为秒数，类似于timestamp但支持datetime的所有时间范围，与timestamp一样为UTC时区"""
    if dt.tzinfo is None:
       dt = dt.astimezone()
    # 会自动转换成utc然后计算秒数，时区偏移量会影响total_seconds，所以必须使用0偏移量的utc计算
    return (dt - datetime(1, 1, 1, tzinfo=timezone.utc)).total_seconds()

def seconds_to_datetime(seconds: float) -> datetime:
    """将秒数转换为datetime，保持UTC时区"""
    return datetime(1, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)

def now_seconds() -> float:
    """获取当前时间的秒数"""
    return datetime_to_seconds(datetime.now(timezone.utc))

class AgentTimeZone(BaseModel):
    """agent的时区模型
    
    不设定offset的话会被当成ZoneInfo处理"""
    name: str = Field(description="时区名称")
    offset: Optional[float] = Field(default=None, gt=-86400.0, lt=86400.0, description="时区偏移，单位为秒")

    def tz(self) -> Union[timezone, ZoneInfo]:
        if self.offset is None:
            return ZoneInfo(self.name)
        else:
            return timezone(timedelta(seconds=self.offset), self.name)

class AgentTimeSettings(BaseModel):
    """agent的时间设置"""
    agent_time_anchor: float = Field(default=0.0, description="agent时间锚点，agent在此时间时真实世界的时间等于real_time_anchor。单位为秒")
    real_time_anchor: float = Field(default=0.0, description="真实时间锚点，真实世界在此时间时，agent时间等于agent_time_anchor。单位为秒")
    time_scale: float = Field(default=1.0, description="相对于真实世界的时间膨胀，控制时间流逝速度")
    time_zone: AgentTimeZone = Field(default_factory=lambda: AgentTimeZone(name=get_localzone_name()), description="agent时区")

def parse_agent_time_zone(setting: Union[AgentTimeSettings, AgentTimeZone]) -> Union[timezone, ZoneInfo]:
    """将agent时区模型解析为timezone或ZoneInfo"""
    if isinstance(setting, AgentTimeSettings):
        tz = setting.time_zone
    else:
        tz = setting
    return tz.tz()

def real_time_to_agent_time(real_time: Union[datetime, float], setting: AgentTimeSettings) -> datetime:
    """将真实世界时间datetime转换为agent时间datetime

    返回的datetime将使用agent自己的时区"""
    if not setting.agent_time_anchor or not setting.real_time_anchor:
        if isinstance(real_time, (float, int)):
            real_time = seconds_to_datetime(real_time)
        agent_time = real_time.astimezone(parse_agent_time_zone(setting))
        return agent_time
    if isinstance(real_time, datetime):
        real_time = datetime_to_seconds(real_time)
    seconds = (real_time - setting.real_time_anchor) * setting.time_scale + setting.agent_time_anchor
    agent_time = seconds_to_datetime(seconds).astimezone(parse_agent_time_zone(setting))
    return agent_time

def agent_time_to_real_time(agent_time: Union[datetime, float], setting: AgentTimeSettings) -> datetime:
    """将agent时间datetime转换为真实世界时间datetime

    返回的datetime将使用UTC时区"""
    if not setting.agent_time_anchor or not setting.real_time_anchor:
        if isinstance(agent_time, float):
            agent_time = seconds_to_datetime(agent_time)
        agent_time = agent_time.astimezone(timezone.utc)
        return agent_time
    if isinstance(agent_time, datetime):
        agent_time = datetime_to_seconds(agent_time)
    seconds = (agent_time - setting.agent_time_anchor) / setting.time_scale + setting.real_time_anchor
    agent_time = seconds_to_datetime(seconds).astimezone(timezone.utc)
    return agent_time

def real_seconds_to_agent_seconds(real_seconds: float, setting: AgentTimeSettings) -> float:
    """将真实世界时间秒数转换为agent时间秒数"""
    if not setting.agent_time_anchor or not setting.real_time_anchor:
        return real_seconds
    return (real_seconds - setting.real_time_anchor) * setting.time_scale + setting.agent_time_anchor

def agent_seconds_to_real_seconds(agent_seconds: float, setting: AgentTimeSettings) -> float:
    """将agent时间秒数转换为真实世界时间秒数"""
    if not setting.agent_time_anchor or not setting.real_time_anchor:
        return agent_seconds
    return (agent_seconds - setting.agent_time_anchor) / setting.time_scale + setting.real_time_anchor

def agent_seconds_to_datetime(seconds: float, time_zone: Union[AgentTimeSettings, AgentTimeZone]) -> datetime:
    """将agent时间秒数转换为datetime

    虽然agent_seconds已包含时间膨胀与偏移，但时区还是UTC，使用此函数将其转换为agent时区的datetime"""
    return seconds_to_datetime(seconds).astimezone(parse_agent_time_zone(time_zone))

def now_agent_time(setting: AgentTimeSettings) -> datetime:
    """获取当前agent时间datetime"""
    return real_time_to_agent_time(utcnow(), setting)

def now_agent_seconds(setting: AgentTimeSettings) -> float:
    """获取当前agent时间秒数"""
    return datetime_to_seconds(now_agent_time(setting))


AnyTz = Union[timezone, ZoneInfo, float, timedelta, AgentTimeSettings, AgentTimeZone]
def format_time(time: Optional[Union[datetime, float]], time_zone: Optional[AnyTz] = None) -> str:
    """datetime格式化函数

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
                elif isinstance(time_zone, (AgentTimeSettings, AgentTimeZone)):
                    tz = parse_agent_time_zone(time_zone)
                else:
                    tz = time_zone
                time = time.astimezone(tz)
        return time.strftime("%Y-%m-%d %H:%M:%S %A")
    except (OverflowError, OSError, ValueError):
        return "时间信息损坏"

def format_seconds(seconds: Union[datetime, float, int, timedelta]) -> str:
    """时间差格式化函数"""
    decrease_one = False
    negative = False
    if isinstance(seconds, (float, int)):
        if seconds < 0:
            negative = True
            seconds = abs(seconds)
        delta = timedelta(seconds=seconds)
        seconds = datetime(1,1,1) + delta
        decrease_one = True
    elif isinstance(seconds, timedelta):
        if seconds.days < 0:
            negative = True
            seconds = abs(seconds)
        seconds = datetime(1,1,1) + seconds
        decrease_one = True
    year = seconds.year
    month = seconds.month
    day = seconds.day
    hour = seconds.hour
    minute = seconds.minute
    second = seconds.second
    if decrease_one:
        year -= 1
        month -= 1
        day -= 1
    result = f'{'负' if negative else ''}{f'{str(year)}年' if year > 0 else ''}{f'{str(month)}个月' if month > 0 else ''}{f'{str(day)}天' if day > 0 else ''}{f'{str(hour)}小时' if hour > 0 else ''}{f'{str(minute)}分' if minute > 0 else ''}{f'{str(second)}秒' if second > 0 else ''}'
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


class Times:
    """旨在需要两个以上的时间种类时方便地完成各类型时间的转换

    通过提供四种时间之一（或留空取当前时间）快速获取其他三种时间"""
    real_time: datetime
    real_time_seconds: float
    agent_time: datetime
    agent_time_seconds: float
    time_settings: AgentTimeSettings

    def __init__(self, setting: Optional[AgentTimeSettings] = None, time: Optional[Union[datetime, float]] = None, is_agent_time: bool = False):
        if is_agent_time:
            if time is None:
                raise ValueError("is_agent_time时必须提供agent时间信息")
            if setting is None:
                raise ValueError("is_agent_time时必须提供时区信息")
            self.time_settings = setting
            if isinstance(time, (float, int)):
                self.agent_time_seconds = time
                self.agent_time = agent_seconds_to_datetime(time, setting)
            else:
                self.agent_time = time
                self.agent_time_seconds = datetime_to_seconds(self.agent_time)
            self.real_time = agent_time_to_real_time(self.agent_time, setting)
            self.real_time_seconds = datetime_to_seconds(self.real_time)
        else:
            if time is None:
                time = utcnow()
            elif isinstance(time, (float, int)):
                time = seconds_to_datetime(time)
            self.real_time = time
            self.real_time_seconds = datetime_to_seconds(self.real_time)
            if setting is not None:
                self.time_settings = setting
                self.agent_time = real_time_to_agent_time(self.real_time, setting)
                self.agent_time_seconds = datetime_to_seconds(self.agent_time)
