from typing import Optional, Union
from pydantic import Field
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, AIMessage, RemoveMessage, BaseMessage

from become_human.times import format_time
from become_human.utils import to_json_like_string
from become_human.message import (
    extract_text_parts,
    SpritesMsgMeta,
    BaseMsgMeta,
)
from become_human.plugins.memory.types import PLUGIN_NAME


DO_NOT_STORE_MESSAGE = '该动作将自己的反馈标记为不必记录，故将其省略。'

class MemoryMsgMeta(BaseMsgMeta):
    """Memory message metadata."""
    KEY = PLUGIN_NAME

    do_not_store: Optional[bool] = Field(default=None)
    do_not_store_tool_message: Optional[str] = Field(default=None)

    recycled: Optional[bool] = Field(default=None)
    extracted: Optional[bool] = Field(default=None)

    retrieved_memory_ids: Optional[list[str]] = Field(default=None)

def get_all_retrieved_memory_ids(messages: list[AnyMessage]) -> list[str]:
    ids = []
    for m in messages:
        try:
            metadata = MemoryMsgMeta.parse(m)
        except KeyError:
            continue
        if metadata.retrieved_memory_ids:
            ids.extend(metadata.retrieved_memory_ids)
    return ids

def filtering_messages(
    messages: list[AnyMessage],
    exclude_do_not_store: bool = True,
    exclude_recycled: bool = False,
    exclude_extracted: bool = True
) -> list[AnyMessage]:
    result = []
    for message in messages:
        try:
            metadata = MemoryMsgMeta.parse(message)
        except KeyError:
            result.append(message)
            continue
        if exclude_do_not_store and not isinstance(message, ToolMessage):
            # 不需要过滤ToolMessage里的do_not_store
            if metadata.do_not_store:
                continue
        if exclude_recycled:
            if metadata.recycled:
                continue
        if exclude_extracted:
            if metadata.extracted:
                continue
        result.append(message)
    return result


def format_human_message_for_ai(message: HumanMessage) -> str:
    return '<others>\n' + "\n".join(extract_text_parts(message.content)) + '\n</others>'

def format_ai_message_for_ai(message: AIMessage) -> str:
    """最好是用`format_ai_messages_for_ai`函数来合并处理AIMessage和ToolMessage"""
    message_string = "<me>\n"
    if message.tool_calls:
        for tool_call in message.tool_calls:
            message_string += f'''<action name="{tool_call['name']}" datetime="{format_time(SpritesMsgMeta.parse(message).creation_times.sprite_world_datetime)}">
<args>
{to_json_like_string(tool_call['args'])}
</args>
</action>\n'''
    return message_string.strip() + '\n</me>'

def format_ai_messages_for_ai(messages: list[Union[AIMessage, ToolMessage]]) -> str:
    """不要用标签排除掉任何ToolMessage，且ToolMessage不能是第一个消息"""
    message_string = '<me>\n'
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    tool_messages_with_id = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    tool_calls = []
    for m in ai_messages:
        tool_calls.extend(m.tool_calls)
    if tool_calls:
        if len(tool_calls) != len(tool_messages_with_id):
            raise ValueError("The number of tool calls does not match the number of tool messages.")
        for tool_call in tool_calls:
            # 让它报错
            feedback_message = tool_messages_with_id[tool_call['id']]
            try:
                feedback_memory_message_metadata = MemoryMsgMeta.parse(feedback_message)
                do_not_store = feedback_memory_message_metadata.do_not_store
            except KeyError:
                do_not_store = False
            if do_not_store:
                feedback_content = feedback_memory_message_metadata.do_not_store_tool_message or DO_NOT_STORE_MESSAGE
            else:
                feedback_content = '\n'.join(extract_text_parts(feedback_message.content))
            feedback_message_metadata = SpritesMsgMeta.parse(feedback_message)
            message_string += f'''<action name="{tool_call['name']}" datetime="{format_time(feedback_message_metadata.creation_times.sprite_world_datetime)}">
<args>
{to_json_like_string(tool_call['args'])}
</args>
<feedback>
{feedback_content}
</feedback>
</action>\n\n'''
    return message_string.strip() + '\n</me>'

def format_tool_message_for_ai(message: ToolMessage) -> str:
    """最好是用`format_ai_messages_for_ai`函数来合并处理AIMessage和ToolMessage"""
    try:
        memory_metadata = MemoryMsgMeta.parse(message)
    except KeyError:
        memory_metadata = None
    if memory_metadata is not None and memory_metadata.do_not_store:
        feedback_content = memory_metadata.do_not_store_tool_message or DO_NOT_STORE_MESSAGE
    else:
        feedback_content = '\n'.join(extract_text_parts(message.content))
    metadata = SpritesMsgMeta.parse(message)
    return f'''<action name="{message.name}" datetime="{format_time(metadata.creation_times.sprite_world_datetime)}>
<feedback>
{feedback_content}
</feedback>
</action>'''

def format_message_for_ai(message: AnyMessage) -> str:
    """最好是用`format_ai_messages_for_ai`函数来合并处理AIMessage和ToolMessage"""
    if isinstance(message, HumanMessage):
        return format_human_message_for_ai(message)
    elif isinstance(message, AIMessage):
        return format_ai_message_for_ai(message)
    elif isinstance(message, ToolMessage):
        return format_tool_message_for_ai(message)
    return "<unsupported_message_type />"

def format_messages_for_ai_as_list(
    messages: list[AnyMessage]
) -> list[tuple[str, int]]:
    """不要用标签排除掉任何ToolMessage，且ToolMessage不能是第一个消息"""
    if not messages:
        return []
    parsed_messages = []
    if isinstance(messages[0], ToolMessage):
        raise ValueError("ToolMessage cannot be the first one of messages.")
    for i, message in enumerate(messages):
        if isinstance(message, AIMessage):
            parsed_messages.append({'type': 'ai', 'messages': [message], 'index': i})
        elif isinstance(message, ToolMessage):
            parsed_messages[-1]['messages'].append(message)
        elif isinstance(message, HumanMessage):
            parsed_messages.append({'type': 'human', 'message': message, 'index': i})
    results = []
    for m in parsed_messages:
        if m['type'] == 'human':
            results.append((format_human_message_for_ai(m['message']), m['index']))
        else:
            results.append((format_ai_messages_for_ai(m['messages']), m['index']))
    return results

def format_messages_for_ai(
    messages: list[AnyMessage]
) -> str:
    """不要用标签排除掉任何ToolMessage，且ToolMessage不能是第一个消息"""
    return '\n\n\n'.join([s for s, i in format_messages_for_ai_as_list(messages)])
