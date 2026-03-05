from typing import Annotated, Literal, TypedDict, Optional, Any
from pydantic import BaseModel, Field
from langchain_core.messages import (
    HumanMessage,
    AnyMessage,
    AIMessageChunk,
    ToolMessage,
    AIMessage,
    BaseMessage,
)
from become_human.times import Times
from become_human.message import add_messages



class StateEntry(BaseModel):
    description: str = Field(description="状态描述")

class MainState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list, description="消息列表")
    sprite_state: list[StateEntry] = Field(default_factory=list, description="状态列表，暂未使用")

    input_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list, description="仅用于将打断消息排在下一次新的call的消息的前面，不影响messages")
    new_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list, description="仅用于区分的新消息列表，不影响messages")
    last_new_messages: list[AnyMessage] = Field(default_factory=list, description="这是实际上的每次调用后留下的新消息列表，只在图调用结束时刷新，这么做是由于new_messages在每次调用结束后会被清空")
    tool_messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list, description="独立的工具消息列表，第一个元素为AIMessage用于检测工具调用，其余为ToolMessage，不影响messages")
    cancelled_by_plugin: Optional[str] = Field(default=None, description='在图内部被哪个插件取消了调用，用于传递给after_call_sprite钩子')

    react_retry_count: int = Field(default=0, description="在同一轮ReAct循环中因出错导致的重试次数，用于防止死循环")

class InterruptData(TypedDict):
    chunk: AIMessageChunk
    called_tool_messages: list[ToolMessage]
    last_chunk_times: Times
