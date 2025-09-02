import json
from datetime import datetime, timedelta, timezone
import calendar
from typing import Union, Optional
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, AnyMessage, BaseMessage
from langchain_core.documents import Document

from become_human.time import seconds_to_datetime, AgentTimeSettings, AgentTimeZone, get_agent_time_zone

def is_valid_json(json_string: str) -> bool:
    try:
        json.loads(json_string)
        return True
    except json.decoder.JSONDecodeError:
        return False


def parse_time(time: Union[dict, datetime, float], time_zone: Optional[Union[timezone, float, timedelta, AgentTimeSettings, AgentTimeZone]] = None) -> str:
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

def parse_human_message(message: HumanMessage) -> str:
    return "\n".join(extract_text_parts(message.content))
    #return f"{message.name}: {message.text()}"

def parse_ai_message(message: AIMessage) -> str:
    message_string = f"我的思考: {"\n".join(extract_text_parts(message.content))}\n\n"
    if message.tool_calls:
        for tool_call in message.tool_calls:
            message_string += f"我的动作: {tool_call['name']}({tool_call['args']})\n"
    return message_string.strip()

def parse_tool_message(message: ToolMessage) -> str:
    return f"动作 {message.name} 的反馈: {"\n".join(extract_text_parts(message.content))}"

def parse_message(message: AnyMessage) -> str:
    if isinstance(message, HumanMessage):
        return parse_human_message(message)
    elif isinstance(message, AIMessage):
        return parse_ai_message(message)
    elif isinstance(message, ToolMessage):
        return parse_tool_message(message)
    return ""

def parse_messages(messages: list[AnyMessage]) -> str:
    return '\n\n\n'.join([parse_message(message) for message in messages])


def parse_memory_documents(documents: list[Document], time_zone: Optional[Union[timezone, float, timedelta, AgentTimeSettings, AgentTimeZone]] = None) -> str:
    """将记忆文档列表转换为(AI)可读的字符串"""
    output = []
    # 反过来从分数最低的开始读取
    for doc in reversed(documents):
        content = doc.page_content
        memory_type = doc.metadata["type"]
        time_seconds = doc.metadata.get("creation_time_seconds")
        if isinstance(time_seconds, (float, int)):
            readable_time = parse_time(time_seconds, time_zone)
        else:
            readable_time = "未知时间"
        output.append(f"记忆类型：{memory_type}\n记忆创建时间: {readable_time}\n记忆内容: {content}")
    if not output:
        return "没有找到任何匹配的记忆。"
    return "\n\n".join(output)


def extract_text_parts(content) -> list[str]:
    contents = []
    if isinstance(content, str):
        contents.append(content)
    elif isinstance(content, list):
        for c in content:
            if isinstance(c, str):
                contents.append(c)
            elif isinstance(c, dict):
                if c.get("type") == "text" and isinstance(c.get("text"), str):
                    contents.append(c["text"])
    return contents


def make_sure_path_exists(data_path: str = "./data", config_path: str = "./config"):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(config_path):
        os.makedirs(config_path)