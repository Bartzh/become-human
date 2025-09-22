import json
from typing import Union, Optional, Any, Type
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, AnyMessage, BaseMessage

def is_valid_json(json_string: str) -> bool:
    try:
        json.loads(json_string)
        return True
    except json.decoder.JSONDecodeError:
        return False


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


def is_that_type(type_hint: Any, target_class: type) -> bool:
        """
        检查类型是否为指定的类或其子类
        """
        try:
            # 直接类型检查
            if isinstance(type_hint, type) and issubclass(type_hint, target_class):
                return True
            # 处理泛型类型（如 Optional[target_class]）
            if hasattr(type_hint, '__origin__'):
                # 检查是否为 Optional 或其他泛型包装
                origin = type_hint.__origin__
                if origin is Union:
                    # 检查 Union 中的类型参数
                    for arg in type_hint.__args__:
                        if isinstance(arg, type) and issubclass(arg, target_class):
                            return True
                elif issubclass(origin, target_class):
                    return True
            return False
        except:
            return False