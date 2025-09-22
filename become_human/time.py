from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field
from typing import Optional, Union

def get_system_timezone_info() -> tuple[float, str]:
    """获取当前系统时区的偏移秒数和名称"""
    tz = datetime.now().astimezone()
    tzname = tz.tzname()
    tzoffset = tz.utcoffset()
    if tzoffset:
        tzoffset_sec = tzoffset.total_seconds()
    if tzname and tzoffset_sec:
        return tzoffset_sec, tzname
    else:
        return 0.0, "UTC"

def datetime_to_seconds(dt: datetime) -> float:
    if dt.tzinfo is None:
       dt = dt.astimezone()
    # 会自动转换成utc然后计算秒数，时区偏移量会影响total_seconds，所以必须使用0偏移量的utc计算
    return (dt - datetime(1, 1, 1, tzinfo=timezone.utc)).total_seconds()

def seconds_to_datetime(seconds: float) -> datetime:
    """UTC时区"""
    return datetime(1, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)

def now_seconds() -> float:
    return datetime_to_seconds(datetime.now(timezone.utc))

class AgentTimeZone(BaseModel):
    offset: float = Field(default_factory=lambda: get_system_timezone_info()[0], gt=-86400.0, lt=86400.0, description="时区偏移，单位为秒")
    name: str = Field(default_factory=lambda: get_system_timezone_info()[1], description="时区名称")

class AgentTimeSettings(BaseModel):
    agent_time_anchor: float = Field(default=0.0, description="agent时间锚点，agent在此时间时真实世界的时间等于real_time_anchor")
    real_time_anchor: float = Field(default=0.0, description="真实时间锚点，真实世界在此时间时，agent时间等于agent_time_anchor")
    time_scale: float = Field(default=1.0, description="相对于真实世界的时间膨胀，控制时间流逝速度")
    time_zone: AgentTimeZone = Field(default_factory=lambda: AgentTimeZone(), description="agent时区")

def get_agent_time_zone(setting: Union[AgentTimeSettings, AgentTimeZone]) -> timezone:
    if isinstance(setting, AgentTimeZone):
        return timezone(timedelta(seconds=setting.offset), setting.name)
    else:
        return timezone(timedelta(seconds=setting.time_zone.offset), setting.time_zone.name)

def real_time_to_agent_time(real_time: Union[datetime, float], setting: AgentTimeSettings) -> datetime:
    """使用agent自己的时区"""
    if not setting.agent_time_anchor or not setting.real_time_anchor:
        if isinstance(real_time, (float, int)):
            real_time = seconds_to_datetime(real_time)
        agent_time = real_time.astimezone(get_agent_time_zone(setting))
        return agent_time
    if isinstance(real_time, datetime):
        real_time = datetime_to_seconds(real_time)
    seconds = (real_time - setting.real_time_anchor) * setting.time_scale + setting.agent_time_anchor
    agent_time = seconds_to_datetime(seconds).astimezone(get_agent_time_zone(setting))
    return agent_time

def agent_time_to_real_time(agent_time: Union[datetime, float], setting: AgentTimeSettings) -> datetime:
    """返回系统时区"""
    if not setting.agent_time_anchor or not setting.real_time_anchor:
        if isinstance(agent_time, float):
            agent_time = seconds_to_datetime(agent_time)
        agent_time = agent_time.astimezone()
        return agent_time
    if isinstance(agent_time, datetime):
        agent_time = datetime_to_seconds(agent_time)
    seconds = (agent_time - setting.agent_time_anchor) / setting.time_scale + setting.real_time_anchor
    agent_time = seconds_to_datetime(seconds).astimezone()
    return agent_time

def now_agent_time(setting: AgentTimeSettings) -> datetime:
    return real_time_to_agent_time(datetime.now(timezone.utc), setting)

def now_agent_seconds(setting: AgentTimeSettings) -> float:
    return datetime_to_seconds(now_agent_time(setting))


def parse_time(time: Union[dict, datetime, float], time_zone: Optional[Union[timezone, float, timedelta, AgentTimeSettings, AgentTimeZone]] = None) -> str:
    """若输入是秒数，则可选地再输入一个时区，并输出时区转换后的时间。若无则是UTC时间。"""
    if isinstance(time, dict):
        time = time.get("creation_time_seconds", None)
        if time is None:
            return "未知时间"
    try:
        if isinstance(time, (float, int)):
            time = seconds_to_datetime(time)
            if time_zone:
                if isinstance(time_zone, (float, int)):
                    tz = timezone(timedelta(hours=time_zone))
                elif isinstance(time_zone, timedelta):
                    tz = timezone(time_zone)
                elif isinstance(time_zone, (AgentTimeSettings, AgentTimeZone)):
                    tz = get_agent_time_zone(time_zone)
                else:
                    tz = time_zone
                time = time.astimezone(tz)
        return time.strftime("%Y-%m-%d %H:%M:%S %A")
    except (OverflowError, OSError, ValueError):
        return "时间信息损坏"

def parse_seconds(seconds: Union[datetime, float, int, timedelta]) -> str:
    decrease_one = False
    negative = False
    if isinstance(seconds, (float, int)):
        if seconds < 0:
            negative = True
            seconds = abs(seconds)
        delta = timedelta(seconds=seconds)
        seconds = datetime.fromordinal(1) + delta
        decrease_one = True
    elif isinstance(seconds, timedelta):
        if seconds.days < 0:
            negative = True
            seconds = abs(seconds)
        seconds = datetime.fromordinal(1) + seconds
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