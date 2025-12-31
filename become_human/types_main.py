from typing import Annotated, Optional, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langchain_core.messages import (
    HumanMessage,
    AnyMessage,
)
from become_human.time import now_seconds
from become_human.memory import RetrievedMemory
from become_human.message import add_messages


@dataclass
class MainContext:
    agent_id: str
    agent_run_id: str
    # 系统调用，目前只给react_instruction使用，与human的区别只是不会因为不活跃而被搁置
    call_type: Literal['human', 'self', 'system'] = 'human'
    self_call_type: Literal['passive', 'active', 'wakeup'] = 'passive'

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

    react_retry_count: int = Field(default=0, description="在ReAct循环中因出错导致的重试次数，用于防止死循环")

    last_chat_time_seconds: float = Field(default_factory=now_seconds, description="上次与用户聊天时间")
    active_time_seconds: float = Field(default=0.0, description="活跃状态终止时间")
    self_call_time_secondses: list[float] = Field(default_factory=list, description="自我调用时间表")
    wakeup_call_time_seconds: float = Field(default=0.0, description="唤醒调用时间")
    active_self_call_time_secondses_and_notes: list[tuple[float, str]] = Field(default_factory=list, description="主动自我调用时间表和备注")
    generated: bool = Field(default=False, description="是否进入了生成流程，即经过了prepare_to_generate节点")
