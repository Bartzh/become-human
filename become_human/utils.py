import json
from datetime import datetime
from typing import Union
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, AnyMessage, BaseMessage


def is_valid_json(json_string: str) -> bool:
    try:
        json.loads(json_string)
        return True
    except json.decoder.JSONDecodeError:
        return False


def parse_time(time: Union[dict, datetime, float]) -> str:
    if isinstance(time, dict):
        time = time.get("creation_timestamp", None)
        if time is None:
            return "未知时间"
    try:
        if isinstance(time, float):
            time = datetime.fromtimestamp(time)
        return time.strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return "时间信息损坏"

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
    messages_string = ""
    for message in messages:
        messages_string += parse_message(message) + "\n\n\n"
    return messages_string


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