from secrets import token_hex
from typing import Any, Optional, Literal, Union, cast
import uuid
from pydantic import BaseModel, Field
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
    AnyMessage,
    convert_to_messages,
    message_chunk_to_message,
    RemoveMessage,
    BaseMessageChunk,
)
from langgraph.graph.message import _add_messages_wrapper, _format_messages, Messages, REMOVE_ALL_MESSAGES
from become_human.time import AnyTz, Times, format_time, now_seconds
from become_human.utils import to_json_like_string

DO_NOT_STORE_MESSAGE = '该动作将自己的反馈标记为不必记录，故将其省略。'

AnyBHMessageType = Literal['human', 'ai', 'passive_retrieve']

class BHMessageMetadata(BaseModel):
    """Metadata for a bh message."""
    creation_real_time_seconds: float = Field(default_factory=now_seconds)
    creation_agent_time_seconds: Optional[float] = Field(default=None)
    do_not_store: Optional[bool] = Field(default=None)
    streaming: Optional[bool] = Field(default=None)
    from_system: Optional[bool] = Field(default=None)
    recycled: Optional[bool] = Field(default=None)
    extracted: Optional[bool] = Field(default=None)
    type: Optional[AnyBHMessageType] = Field(default=None)

class BHToolArtifact(BaseModel):
    """Metadata for a bh tool artifact."""
    do_not_store: Optional[bool] = Field(default=None)
    streaming: Optional[bool] = Field(default=None)

class BHBaseMessageMetadata(BaseModel):
    """Metadata for a bh base message."""
    creation_real_time_seconds: float = Field(default_factory=now_seconds)
    creation_agent_time_seconds: Optional[float] = Field(default=None)
    do_not_store: Optional[bool] = Field(default=None)
    streaming: Optional[bool] = Field(default=None)
    from_system: Optional[bool] = Field(default=None)
    recycled: Optional[bool] = Field(default=None)
    extracted: Optional[bool] = Field(default=None)
    type: Optional[AnyBHMessageType] = Field(default=None)

class BHSystemMessageMetadata(BaseModel):
    """Metadata for a bh system message."""

class BHRetrievalMessageMetadata(BaseModel):
    """Metadata for a bh retrieval message."""
    retrieval_type: Optional[str] = Field(default=None)
    retrieved_memory_ids: Optional[list[str]] = Field(default=None)

def get_all_retrieved_memory_ids(messages: list[AnyMessage]) -> list[str]:
    ids = []
    for m in messages:
        if isinstance(m, HumanMessage) and m.additional_kwargs.get('bh_message_type') == 'passive_retrieval':
            ids.extend(m.additional_kwargs.get('bh_retrieved_memory_ids', []))
        elif isinstance(m, ToolMessage) and m.name == 'retrieve_memories' and isinstance(m.artifact, dict):
            ids.extend(m.artifact.get('bh_retrieved_memory_ids', []))
    return ids


@_add_messages_wrapper
def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Optional[Literal["langchain-openai"]] = None,
) -> Messages:
    """Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Args:
        left: The base list of messages.
        right: The list of messages (or single message) to merge
            into the base list.
        format: The format to return messages in. If None then messages will be
            returned as is. If 'langchain-openai' then messages will be returned as
            BaseMessage objects with their contents formatted to match OpenAI message
            format, meaning contents can be string, 'text' blocks, or 'image_url' blocks
            and tool responses are returned as their own ToolMessages.

            !!! important "Requirement"

                Must have ``langchain-core>=0.3.11`` installed to use this feature.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`.

    Example:
        ```python title="Basic usage"
        from langchain_core.messages import AIMessage, HumanMessage
        msgs1 = [HumanMessage(content="Hello", id="1")]
        msgs2 = [AIMessage(content="Hi there!", id="2")]
        add_messages(msgs1, msgs2)
        # [HumanMessage(content='Hello', id='1'), AIMessage(content='Hi there!', id='2')]
        ```

        ```python title="Overwrite existing message"
        msgs1 = [HumanMessage(content="Hello", id="1")]
        msgs2 = [HumanMessage(content="Hello again", id="1")]
        add_messages(msgs1, msgs2)
        # [HumanMessage(content='Hello again', id='1')]
        ```

        ```python title="Use in a StateGraph"
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        builder = StateGraph(State)
        builder.add_node("chatbot", lambda state: {"messages": [("assistant", "Hello")]})
        builder.set_entry_point("chatbot")
        builder.set_finish_point("chatbot")
        graph = builder.compile()
        graph.invoke({})
        # {'messages': [AIMessage(content='Hello', id=...)]}
        ```

        ```python title="Use OpenAI message format"
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, add_messages

        class State(TypedDict):
            messages: Annotated[list, add_messages(format='langchain-openai')]

        def chatbot_node(state: State) -> list:
            return {"messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here's an image:",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "1234",
                            },
                        },
                    ]
                },
            ]}

        builder = StateGraph(State)
        builder.add_node("chatbot", chatbot_node)
        builder.set_entry_point("chatbot")
        builder.set_finish_point("chatbot")
        graph = builder.compile()
        graph.invoke({"messages": []})
        # {
        #     'messages': [
        #         HumanMessage(
        #             content=[
        #                 {"type": "text", "text": "Here's an image:"},
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": "data:image/jpeg;base64,1234"},
        #                 },
        #             ],
        #         ),
        #     ]
        # }
        ```

    """
    remove_all_idx = None
    # coerce to list
    if not isinstance(left, list):
        left = [left]  # type: ignore[assignment]
    if not isinstance(right, list):
        right = [right]  # type: ignore[assignment]
    # coerce to message
    left = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(left)
    ]
    right = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(right)
    ]
    # assign missing ids
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for idx, m in enumerate(right):
        if m.id is None:
            m.id = str(uuid.uuid4())
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx

    # 修改处
    right = messages_post_processing(right)

    if remove_all_idx is not None:
        return right[remove_all_idx + 1 :]

    # merge
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )

            merged_by_id[m.id] = len(merged)
            merged.append(m)
    merged = [m for m in merged if m.id not in ids_to_remove]

    if format == "langchain-openai":
        merged = _format_messages(merged)
    elif format:
        msg = f"Unrecognized {format=}. Expected one of 'langchain-openai', None."
        raise ValueError(msg)
    else:
        pass

    return merged

def messages_post_processing(messages: list[BaseMessage]):
    current_timeseconds = now_seconds()
    for m in messages:
        if isinstance(m, (AIMessage, HumanMessage, ToolMessage)) and not m.additional_kwargs.get("bh_post_processed"):
            m.additional_kwargs["bh_post_processed"] = True
            #给每个消息添加时间戳
            if not m.additional_kwargs.get("bh_creation_real_timeseconds"):
                m.additional_kwargs["bh_creation_real_timeseconds"] = current_timeseconds
    return messages

# def messages_post_processing(messages: list[BaseMessage]):
#     for m in messages:
#         if isinstance(m, (AIMessage, HumanMessage, ToolMessage)) and not m.additional_kwargs.get("bh_metadata"):
#             m.additional_kwargs["bh_metadata"] = BHMessageMetadata()
#     return messages

def filtering_messages(
    messages: list[AnyMessage],
    exclude_do_not_store: bool = False,
    exclude_recycled: bool = False,
    exclude_extracted: bool = True,
    exclude_system: bool = True
) -> list[AnyMessage]:
    result = []
    for message in messages:
        if exclude_do_not_store:
            if isinstance(message, ToolMessage) and message.artifact and isinstance(message.artifact, dict) and message.artifact.get("bh_do_not_store"):
                continue
            elif message.additional_kwargs.get("bh_do_not_store"):
                continue
        if exclude_recycled:
            if message.additional_kwargs.get("bh_recycled"):
                continue
        if exclude_extracted:
            if message.additional_kwargs.get("bh_extracted"):
                continue
        if exclude_system:
            if message.additional_kwargs.get("bh_from_system"):
                continue
        result.append(message)
    return result

def format_human_message_for_ai(message: HumanMessage) -> str:
    return '<others>\n' + "\n".join(extract_text_parts(message.content)) + '\n</others>'

def format_ai_message_for_ai(message: AIMessage, time_zone: Optional[AnyTz] = None) -> str:
    """最好是用`format_ai_messages_for_ai`函数来合并处理AIMessage和ToolMessage"""
    message_string = "<me>\n"
    if message.tool_calls:
        for tool_call in message.tool_calls:
            message_string += f'''<action name="{tool_call['name']}" datetime="{format_time(message.additional_kwargs.get('bh_creation_agent_timeseconds'), time_zone)}">
<args>
{to_json_like_string(tool_call['args'])}
</args>
</action>\n'''
    return message_string.strip() + '\n</me>'

def format_ai_messages_for_ai(messages: list[Union[AIMessage, ToolMessage]], time_zone: Optional[AnyTz] = None) -> str:
    """不要排除掉bh_do_not_store或bh_streaming之类的消息，总之ToolMessage不能少，且ToolMessage不能是第一个消息"""
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
            if (
                feedback_message.artifact and
                isinstance(feedback_message.artifact, dict) and
                feedback_message.artifact.get("bh_do_not_store")
            ):
                feedback_content = DO_NOT_STORE_MESSAGE
            else:
                feedback_content = '\n'.join(extract_text_parts(feedback_message.content))
            message_string += f'''<action name="{tool_call['name']}" datetime="{format_time(feedback_message.additional_kwargs.get('bh_creation_agent_timeseconds'), time_zone)}">
<args>
{to_json_like_string(tool_call['args'])}
</args>
<feedback>
{feedback_content}
</feedback>
</action>\n\n'''
    return message_string.strip() + '\n</me>'

def format_tool_message_for_ai(message: ToolMessage, time_zone: Optional[AnyTz] = None) -> str:
    """最好是用`format_ai_messages_for_ai`函数来合并处理AIMessage和ToolMessage"""
    if (
        message.artifact and
        isinstance(message.artifact, dict) and
        message.artifact.get("bh_do_not_store")
    ):
        feedback_content = DO_NOT_STORE_MESSAGE
    else:
        feedback_content = '\n'.join(extract_text_parts(message.content))
    return f'''<action name="{message.name}" datetime="{format_time(message.additional_kwargs.get('bh_creation_agent_timeseconds'), time_zone)}>
<feedback>
{feedback_content}
</feedback>
</action>'''

def format_message_for_ai(message: AnyMessage, time_zone: Optional[AnyTz] = None) -> str:
    """最好是用`format_ai_messages_for_ai`函数来合并处理AIMessage和ToolMessage"""
    if isinstance(message, HumanMessage):
        return format_human_message_for_ai(message)
    elif isinstance(message, AIMessage):
        return format_ai_message_for_ai(message, time_zone)
    elif isinstance(message, ToolMessage):
        return format_tool_message_for_ai(message, time_zone)
    return "<unsupported_message_type />"

def format_messages_for_ai_as_list(
    messages: list[AnyMessage],
    time_zone: Optional[AnyTz] = None
) -> list[str]:
    """不要排除掉bh_do_not_store或bh_streaming之类的消息，总之ToolMessage不能少，且ToolMessage不能是第一个消息"""
    if not messages:
        return []
    parsed_messages = []
    if isinstance(messages[0], ToolMessage):
        raise ValueError("ToolMessage cannot be the first one of messages.")
    for message in messages:
        if isinstance(message, AIMessage):
            parsed_messages.append({'type': 'ai', 'messages': [message]})
        elif isinstance(message, ToolMessage):
            parsed_messages[-1]['messages'].append(message)
        elif message.additional_kwargs.get('bh_message_type', '') == 'react_error':
            pass
        elif isinstance(message, HumanMessage):
            parsed_messages.append({'type': 'human', 'message': message})
    results = []
    for m in parsed_messages:
        if m['type'] == 'human':
            results.append(format_human_message_for_ai(m['message']))
        else:
            results.append(format_ai_messages_for_ai(m['messages'], time_zone))
    return results

def format_messages_for_ai(
    messages: list[AnyMessage],
    time_zone: Optional[AnyTz] = None
) -> str:
    """不要排除掉bh_do_not_store或bh_streaming之类的消息，总之ToolMessage不能少，且ToolMessage不能是第一个消息"""
    return '\n\n\n'.join(format_messages_for_ai_as_list(messages, time_zone))


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


def construct_system_message(
    content: str,
    times: Times,
    extra_kwargs: Optional[dict] = None
) -> HumanMessage:
    """additional_kwargs 默认包含 bh_from_system、bh_do_not_store、bh_creation_real_timeseconds、bh_creation_agent_timeseconds"""
    additional_kwargs = {
            "bh_from_system": True,
            "bh_do_not_store": True,
            "bh_creation_real_timeseconds": times.real_timeseconds,
            "bh_creation_agent_timeseconds": times.agent_timeseconds,
        }
    if extra_kwargs:
        additional_kwargs.update(extra_kwargs)
    system_prefix = "**这条消息来自系统（system）自动发送**\n"
    return HumanMessage(
        content=content if content.startswith(system_prefix) else system_prefix + content,
        additional_kwargs=additional_kwargs,
        name="system",
        id=str(uuid.uuid4())
    )


class InitalToolCall(BaseModel):
    name: str = Field(description="工具名称")
    args: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    result_content: Union[str, dict[str, dict]] = Field(default=None, description="工具调用结果content")
    result_artifact: Optional[dict[str, Any]] = Field(default=None, description="工具调用结果artifact")

class InitalAIMessage(BaseModel):
    """至少需要其中一项"""
    content: Union[str, dict[str, dict]] = Field(default='', description="内容")
    tool_calls: list[InitalToolCall] = Field(default_factory=list, description="工具调用列表")

    def construct_messages(self, times: Times) -> list[Union[AIMessage, ToolMessage]]:
        tool_calls_with_id = [{
            'name': tool_call.name,
            'args': tool_call.args,
            'id': 'call_' + token_hex(12),
            'result_content': tool_call.result_content,
            'result_artifact': tool_call.result_artifact
        } for tool_call in self.tool_calls]
        messages = [AIMessage(
            content=self.content,
            additional_kwargs={
                'tool_calls': [{
                    'index': i,
                    'id': tool_call['id'],
                    'function': {
                        'arguments': tool_call['args'],
                        'name': tool_call['name']
                    },
                    'type': 'function'
                } for i, tool_call in enumerate(tool_calls_with_id)],
                "bh_creation_real_timeseconds": times.real_timeseconds,
                "bh_creation_agent_timeseconds": times.agent_timeseconds
            },
            response_metadata={
                'finish_reason': 'tool_calls',
                'model_name': 'qwen-plus-2025-04-28'
            },
            tool_calls=[{
                'name': tool_call['name'],
                'args': tool_call['args'],
                'id': tool_call['id'],
                'type': 'tool_call'
            } for tool_call in tool_calls_with_id],
            id=str(uuid.uuid4())
        )]
        messages.extend([ToolMessage(
            content=tool_call['result_content'],
            name=tool_call['name'],
            artifact=tool_call['result_artifact'],
            tool_call_id=tool_call['id'],
            additional_kwargs={
                "bh_creation_real_timeseconds": times.real_timeseconds,
                "bh_creation_agent_timeseconds": times.agent_timeseconds
            },
            id=str(uuid.uuid4())
        ) for tool_call in tool_calls_with_id])
        return messages
