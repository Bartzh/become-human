from typing import Annotated, Optional, Literal, cast
import uuid
from dataclasses import dataclass
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
from become_human.time import now_seconds
from become_human.memory import RetrievedMemory

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

def get_retrieved_memory_ids(messages: list[AnyMessage]) -> list[str]:
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
    current_time_seconds = now_seconds()
    for m in messages:
        if isinstance(m, (AIMessage, HumanMessage, ToolMessage)) and not m.additional_kwargs.get("bh_post_processed"):
            m.additional_kwargs["bh_post_processed"] = True
            #给每个消息添加时间戳
            if not m.additional_kwargs.get("bh_creation_time_seconds"):
                m.additional_kwargs["bh_creation_time_seconds"] = current_time_seconds
    return messages

# def messages_post_processing(messages: list[BaseMessage]):
#     for m in messages:
#         if isinstance(m, (AIMessage, HumanMessage, ToolMessage)) and not m.additional_kwargs.get("bh_metadata"):
#             m.additional_kwargs["bh_metadata"] = BHMessageMetadata()
#     return messages


@dataclass
class MainContext:
    agent_id: str
    agent_run_id: str
    is_self_call: bool = False
    self_call_type: Literal['passive', 'active'] = 'passive'

class StateEntry(BaseModel):
    description: str = Field(description="状态描述")

class MainState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list, description="消息列表")
    memories: list[RetrievedMemory] = Field(default_factory=list, description="记忆列表")
    agent_state: list[StateEntry] = Field(default_factory=list, description="状态列表，暂未使用")

    input_messages: Annotated[list[HumanMessage], add_messages] = Field(default_factory=list, description="仅用于区分的用户输入消息列表，不完全可信，不影响messages")
    new_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list, description="仅用于区分的新消息列表，不完全可信，不影响messages")
    tool_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list, description="独立的工具消息列表，第一个元素为AIMessage用于检测工具调用，其余为ToolMessage，不影响messages")
    recycle_messages: list[AnyMessage] = Field(default_factory=list, description="回收为original的消息列表，仅用于节点传递")
    overflow_messages: list[AnyMessage] = Field(default_factory=list, description="溢出的消息列表，会extract为episodic，仅用于节点传递")

    last_chat_time_seconds: float = Field(default_factory=now_seconds, description="上次与用户聊天时间")
    active_time_seconds: float = Field(default=0.0, description="活跃状态终止时间")
    self_call_time_secondses: list[float] = Field(default_factory=list, description="自我调用时间表")
    wakeup_call_time_seconds: float = Field(default=0.0, description="唤醒调用时间")
    active_self_call_time_secondses_and_notes: list[tuple[float, str]] = Field(default_factory=list, description="主动自我调用时间表和备注")
    generated: bool = Field(default=False, description="是否进入了生成流程，即经过了prepare_to_generate节点")
