# 显而易见的，add_messages的代码来自langgraph（MIT许可），尽管目前没有修改add_messages的需求了
# https://github.com/langchain-ai/langgraph

from secrets import token_hex
from typing import Any, Optional, Literal, Self, Union, cast
import uuid
from pydantic import BaseModel, Field, ValidationError
from loguru import logger
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
from become_human.times import Times, format_time
from become_human.utils import to_json_like_string

DO_NOT_STORE_MESSAGE = '该动作将自己的反馈标记为不必记录，故将其省略。'

BH_MESSAGE_METADATA_KEY = 'bh_message_metadata'

# 主要用于区分HumanMessage，因为AIMessage和ToolMessage是不变的
AnyBHMessageType = Literal['bh:passive_retrieval', 'bh:ai', 'bh:tool', 'bh:user', 'bh:react_error', 'bh:system']

class BHMessageMetadata(BaseModel):
    """Metadata for a BH message."""

    creation_times: Times = Field(description="The creation times of the message")

    message_type: Optional[str] = Field(default=None)

    do_not_store: Optional[bool] = Field(default=None)
    is_streaming_tool: Optional[bool] = Field(default=None)

    recycled: Optional[bool] = Field(default=None)
    extracted: Optional[bool] = Field(default=None)

    retrieved_memory_ids: Optional[list[str]] = Field(default=None)

    @classmethod
    def parse(cls, message_or_additional_kwargs: Union[AnyMessage, dict, Self]) -> Self:
        """可输入消息或消息的additional_kwargs，解析为BHMessageMetadata

        也允许输入BHMessageMetadata是因为类型可能还没有被checkpointer转换为dict，如果没有手动model_dump的话

        由于不允许有消息没有BHMessageMetadata，所以如果没有的话会报错。"""
        if isinstance(message_or_additional_kwargs, dict):
            additional_kwargs = message_or_additional_kwargs
        elif isinstance(message_or_additional_kwargs, BaseMessage):
            additional_kwargs = message_or_additional_kwargs.additional_kwargs
        elif isinstance(message_or_additional_kwargs, cls):
            return message_or_additional_kwargs.model_copy(deep=True)
        else:
            raise ValueError("message_or_additional_kwargs must be a BaseMessage or dict, or BHMessageMetadata")
        if metadata := additional_kwargs.get(BH_MESSAGE_METADATA_KEY):
            return cls.model_validate(metadata, strict=True)
        else:
            raise ValueError(f"No {BH_MESSAGE_METADATA_KEY} found in additional_kwargs")

class BHMessageMetadataWithTimesNotRequired(BHMessageMetadata):
    """主要作用是使BHMessageMetadata可以不填写时间，由之后的节点来添加时间信息。同时还有一个默认的消息类型

    最终还是会使用BHMessageMetadata，这个结构只是方便输入

    目前支持的场景有BaseTool、InitalAIMessage、InitalToolCall、construct_system_message"""
    creation_times: Optional[Times] = Field(default=None)

def get_all_retrieved_memory_ids(messages: list[AnyMessage]) -> list[str]:
    ids = []
    for m in messages:
        metadata = BHMessageMetadata.parse(m)
        if metadata.retrieved_memory_ids:
            ids.extend(metadata.retrieved_memory_ids)
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
    #right = messages_post_processing(right)

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

# def messages_post_processing(messages: list[BaseMessage]):
#     for m in messages:
#         if (
#             isinstance(m, ToolMessage) and
#             isinstance(m.artifact, dict) and
#             (metadata_in_artifact := m.artifact.get(BH_MESSAGE_METADATA_KEY))
#         ):
#             if not m.additional_kwargs.get(BH_MESSAGE_METADATA_KEY):
#                 try:
#                     validated_model = BHMessageMetadata.model_validate(metadata_in_artifact)
#                     m.additional_kwargs[BH_MESSAGE_METADATA_KEY] = validated_model.model_dump()
#                     del m.artifact[BH_MESSAGE_METADATA_KEY]
#                 except ValidationError:
#                     logger.critical(f"尝试将 {m.id} artifact中的 {BH_MESSAGE_METADATA_KEY} 加入 additional_kwargs，但验证失败")
#             else:
#                 del m.artifact[BH_MESSAGE_METADATA_KEY]
#                 logger.warning(f"{BH_MESSAGE_METADATA_KEY} 已经存在于消息 {m.id} 的 additional_kwargs 中，但 artifact 中的还没有被清理，现在将其清理")
#     return messages


def filtering_messages(
    messages: list[AnyMessage],
    exclude_do_not_store: bool = True,
    exclude_recycled: bool = False,
    exclude_extracted: bool = True
) -> list[AnyMessage]:
    result = []
    for message in messages:
        metadata = BHMessageMetadata.parse(message)
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
            message_string += f'''<action name="{tool_call['name']}" datetime="{format_time(BHMessageMetadata.parse(message).creation_times.agent_world_datetime)}">
<args>
{to_json_like_string(tool_call['args'])}
</args>
</action>\n'''
    return message_string.strip() + '\n</me>'

def format_ai_messages_for_ai(messages: list[Union[AIMessage, ToolMessage]]) -> str:
    """不要用is_streaming_tool之类的标签排除掉任何ToolMessage，且ToolMessage不能是第一个消息"""
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
            feedback_message_metadata = BHMessageMetadata.parse(feedback_message)
            if (feedback_message_metadata.do_not_store):
                feedback_content = DO_NOT_STORE_MESSAGE
            else:
                feedback_content = '\n'.join(extract_text_parts(feedback_message.content))
            message_string += f'''<action name="{tool_call['name']}" datetime="{format_time(feedback_message_metadata.creation_times.agent_world_datetime)}">
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
    metadata = BHMessageMetadata.parse(message)
    if (metadata.do_not_store):
        feedback_content = DO_NOT_STORE_MESSAGE
    else:
        feedback_content = '\n'.join(extract_text_parts(message.content))
    return f'''<action name="{message.name}" datetime="{format_time(metadata.creation_times.agent_world_datetime)}>
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
    """不要用is_streaming_tool之类的标签排除掉任何ToolMessage，且ToolMessage不能是第一个消息"""
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
    """不要用is_streaming_tool之类的标签排除掉任何ToolMessage，且ToolMessage不能是第一个消息"""
    return '\n\n\n'.join([s for s, i in format_messages_for_ai_as_list(messages)])


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
    message_type: str = "bh:system",
    extra_kwargs: Optional[dict] = None
) -> HumanMessage:
    """additional_kwargs 默认包含 bh_message_metadata 中的 do_not_store、creation_times、message_type

    extra_kwargs 可以包含bh_message_metadata，将会与默认值合并，默认包含的字段会被相应值覆盖（若有）"""
    default_metadata = BHMessageMetadata(
        message_type=message_type,
        creation_times=times,
        do_not_store=True
    ).model_dump()
    if extra_kwargs:
        additional_kwargs = extra_kwargs
        if BH_MESSAGE_METADATA_KEY in extra_kwargs.keys():
            metadata = BHMessageMetadataWithTimesNotRequired.model_validate(extra_kwargs[BH_MESSAGE_METADATA_KEY], strict=True)
            if metadata.message_type is None:
                metadata.message_type = message_type
            if metadata.creation_times is None:
                metadata.creation_times = times
            if metadata.do_not_store is None:
                metadata.do_not_store = True
            additional_kwargs[BH_MESSAGE_METADATA_KEY] = metadata.model_dump()
        else:
            additional_kwargs[BH_MESSAGE_METADATA_KEY] = default_metadata
    else:
        additional_kwargs = {
            BH_MESSAGE_METADATA_KEY: default_metadata
        }

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
    result_additional_kwargs: Optional[dict] = Field(default=None, description="工具调用结果additional_kwargs，默认已包含bh_message_metadata")

class InitalAIMessage(BaseModel):
    """至少需要其中一项"""
    content: Union[str, dict[str, dict]] = Field(default='', description="内容")
    tool_calls: list[InitalToolCall] = Field(default_factory=list, description="工具调用列表")
    additional_kwargs: Optional[dict] = Field(default=None, description="额外参数，默认已包含bh_message_metadata")

    def construct_messages(self, times: Times) -> list[Union[AIMessage, ToolMessage]]:
        tool_calls_with_id = [{
            'name': tool_call.name,
            'args': tool_call.args,
            'id': 'call_' + token_hex(12),
            'result_content': tool_call.result_content,
            'result_artifact': tool_call.result_artifact,
            'result_additional_kwargs': tool_call.result_additional_kwargs
        } for tool_call in self.tool_calls]


        additional_kwargs = {
            'tool_calls': [{
                'index': i,
                'id': tool_call['id'],
                'function': {
                    'arguments': tool_call['args'],
                    'name': tool_call['name']
                },
                'type': 'function'
            } for i, tool_call in enumerate(tool_calls_with_id)]
        }
        default_metadata = BHMessageMetadata(
            message_type="bh:ai",
            creation_times=times
        ).model_dump()
        if self.additional_kwargs:
            if 'tool_calls' in self.additional_kwargs.keys():
                del self.additional_kwargs['tool_calls']
                logger.warning("InitalAIMessage 的 additional_kwargs 中不可包含 tool_calls，将忽略")
            additional_kwargs.update(self.additional_kwargs)
            if BH_MESSAGE_METADATA_KEY in self.additional_kwargs.keys():
                new_bh_metadata = BHMessageMetadataWithTimesNotRequired.model_validate(self.additional_kwargs[BH_MESSAGE_METADATA_KEY])
                if new_bh_metadata.creation_times is None:
                    new_bh_metadata.creation_times = times
                if new_bh_metadata.message_type is None:
                    new_bh_metadata.message_type = "bh:ai"
                additional_kwargs[BH_MESSAGE_METADATA_KEY] = new_bh_metadata.model_dump()
            else:
                additional_kwargs[BH_MESSAGE_METADATA_KEY] = default_metadata
        else:
            additional_kwargs[BH_MESSAGE_METADATA_KEY] = default_metadata

        messages = [AIMessage(
            content=self.content,
            additional_kwargs=additional_kwargs,
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


        default_tool_metadata = BHMessageMetadata(
            message_type="bh:tool",
            creation_times=times
        ).model_dump()
        for tool_call in tool_calls_with_id:
            tool_additional_kwargs = {}
            if result_additional_kwargs := tool_call['result_additional_kwargs']:
                additional_kwargs.update(result_additional_kwargs)
                if BH_MESSAGE_METADATA_KEY in result_additional_kwargs.keys():
                    new_tool_bh_metadata = BHMessageMetadataWithTimesNotRequired.model_validate(result_additional_kwargs[BH_MESSAGE_METADATA_KEY])
                    if new_tool_bh_metadata.creation_times is None:
                        new_tool_bh_metadata.creation_times = times
                    if new_tool_bh_metadata.message_type is None:
                        new_tool_bh_metadata.message_type = "bh:tool"
                    tool_additional_kwargs[BH_MESSAGE_METADATA_KEY] = new_tool_bh_metadata.model_dump()
                else:
                    tool_additional_kwargs[BH_MESSAGE_METADATA_KEY] = default_tool_metadata
            else:
                tool_additional_kwargs[BH_MESSAGE_METADATA_KEY] = default_tool_metadata

            messages.append(ToolMessage(
                content=tool_call['result_content'],
                name=tool_call['name'],
                artifact=tool_call['result_artifact'],
                tool_call_id=tool_call['id'],
                additional_kwargs=tool_additional_kwargs,
                id=str(uuid.uuid4())
            ))


        return messages
