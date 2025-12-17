import json
from typing import Union, Optional, Any
from typing_inspect import get_args, get_origin
import os
from pydantic import BaseModel

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, AnyMessage

from become_human.time import format_time, AnyTz

def is_valid_json(json_string: str) -> bool:
    try:
        json.loads(json_string)
        return True
    except json.decoder.JSONDecodeError:
        return False


def format_human_message_for_ai(message: HumanMessage) -> str:
    return '<others>\n' + "\n".join(extract_text_parts(message.content)) + '\n</others>'

def format_ai_message_for_ai(message: AIMessage, time_zone: Optional[AnyTz] = None) -> str:
    message_string = f"<me>\n我的思考: {"\n".join(extract_text_parts(message.content))}\n\n"
    if message.tool_calls:
        for tool_call in message.tool_calls:
            message_string += f"我的动作: {tool_call['name']}({to_json_like_string(tool_call['args'])})\n"
    else:
        message_string = f'[{format_time(message.additional_kwargs.get('bh_creation_agent_time_seconds'), time_zone)}]\n' + message_string
    return message_string.strip() + '\n</me>'

def format_tool_message_for_ai(message: ToolMessage, time_zone: Optional[AnyTz] = None) -> str:
    return f"<action>\n[{format_time(message.additional_kwargs.get('bh_creation_agent_time_seconds'), time_zone)}]\n动作 {message.name} 的反馈: {"\n".join(extract_text_parts(message.content))}\n</action>"

def format_message_for_ai(message: AnyMessage, time_zone: Optional[AnyTz] = None) -> str:
    if isinstance(message, HumanMessage):
        return format_human_message_for_ai(message)
    elif isinstance(message, AIMessage):
        return format_ai_message_for_ai(message, time_zone)
    elif isinstance(message, ToolMessage):
        return format_tool_message_for_ai(message, time_zone)
    return "<unsupported_message_type />"

def format_messages_for_ai(messages: list[AnyMessage], time_zone: Optional[AnyTz] = None) -> str:
    return '\n\n\n'.join([format_message_for_ai(message, time_zone) for message in messages])


def extract_text_parts(content: Union[list, str]) -> list[str]:
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


def dump_basemodels(items: Union[list, tuple, dict, set]) -> Union[list, tuple, dict, set]:
    if isinstance(items, list):
        new_items = []
        old_items = items
        items_type = 'list'
    elif isinstance(items, tuple):
        new_items = ()
        old_items = items
        items_type = 'tuple'
    elif isinstance(items, set):
        new_items = set()
        old_items = items
        items_type = 'set'
    else:
        new_items = {}
        old_items = items.items()
        items_type = 'dict'
    for item in old_items:
        if items_type == 'dict':
            item_value = item[1]
        else:
            item_value = item
        if isinstance(item_value, BaseModel):
            new_item = item_value.model_dump(exclude_unset=True)
        elif isinstance(item_value, (list, tuple, dict, set)):
            new_item = dump_basemodels(item_value)
        else:
            new_item = item_value
        if items_type == 'dict':
            new_items[item[0]] = new_item
        elif items_type == 'tuple':
            new_items += (new_item,)
        elif items_type == 'set':
            new_items.add(new_item)
        else:
            new_items.append(new_item)
    return new_items

def parse_env_array(env_array: Optional[str]) -> list[str]:
    if env_array:
        return [item.strip() for item in env_array.split(',')]
    else:
        return []

def to_json_like_string(a: Any) -> str:
    """将任意对象转换为JSON-like字符串

    具体来说，实现了对字符串、布尔值、None、元组、列表、字典的转换"""
    if isinstance(a, str):
        return f'"{a}"'
    elif isinstance(a, bool):
        return str(a).lower()
    elif a is None:
        return 'null'
    elif isinstance(a, (tuple, list)):
        return '[' + ', '.join([to_json_like_string(i) for i in a]) + ']'
    elif isinstance(a, dict):
        return '{' + ', '.join([f'"{k}": {to_json_like_string(v)}' for k, v in a.items()]) + '}'
    else:
        return str(a)

def get_readable_type_name(tp) -> str:
    """增强的类型名称获取"""
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Union:
        arg_names = [get_readable_type_name(a) for a in args]
        return f"Union[{', '.join(arg_names)}]"

    elif origin:
        if args:
            arg_names = [get_readable_type_name(a) for a in args]
            name = getattr(origin, '__name__', str(origin))
            return f"{name}[{', '.join(arg_names)}]"
        else:
            return getattr(origin, '__name__', str(origin))

    else:
        return getattr(tp, '__name__', str(tp))
