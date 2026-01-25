from typing import Sequence, Dict, Any, Union, Callable, Optional
import collections.abc
from inspect import signature

from aiosqlite import Connection
from pydantic import BaseModel

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.channels.binop import _get_overwrite, _strip_extras

from become_human.tool import AnyTool

class BaseGraph:

    graph: CompiledStateGraph
    graph_builder: StateGraph
    conn: Connection

    llm: BaseChatModel
    tools: list[AnyTool]

    def __init__(self, llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[AnyTool]] = None
    ):
        if llm is not None:
            self.llm = llm
        if not hasattr(self, 'tools'):
            self.tools = []
        if tools is not None:
            self.tools.extend(tools)

# 基本是对langgraph.graph.channels.binop的改造简化版本
MISSING = object()
class StateMerger:
    state_schema: type[BaseModel]
    reducers: dict[str, tuple[Any, Callable[[Any, Any], Any]]]
    def __init__(self, state_schema: type[BaseModel]):
        self.state_schema = state_schema
        reducers = {}
        for key, field in state_schema.model_fields.items():
            meta = field.metadata
            # 检查callable的代码来自langgraph.graph.state的_is_field_binop
            if len(meta) >= 1 and callable(meta[-1]):
                sig = signature(meta[-1])
                params = list(sig.parameters.values())
                if (
                    sum(
                        p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                        for p in params
                    )
                    == 2
                ):
                    typ = _strip_extras(field.annotation)
                    if typ in (collections.abc.Sequence, collections.abc.MutableSequence):
                        typ = list
                    elif typ in (collections.abc.Set, collections.abc.MutableSet):
                        typ = set
                    elif typ in (collections.abc.Mapping, collections.abc.MutableMapping):
                        typ = dict
                    try:
                        default = typ()
                    except Exception:
                        default = MISSING
                    reducers[key] = (default, meta[-1])
                else:
                    raise ValueError(
                        f"Invalid reducer signature. Expected (a, b) -> c. Got {sig}"
                    )
        self.reducers = reducers

    def merge(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        """模拟 StateGraph 的状态更新"""
        if not states:
            return {}
        merged = {}
        for state in states:
            for key, value in state.items():
                if key in self.reducers:
                    # 我这里不需要对有多个overwrite的情况抛出异常，原代码里指的是一个超级步骤里不能有多个overwrite
                    is_overwrite, overwrite_value = _get_overwrite(value)
                    if is_overwrite:
                        merged[key] = overwrite_value
                    else:
                        default, reducer = self.reducers[key]
                        if key in merged:
                            merged[key] = reducer(merged[key], value)
                        else:
                            if default is not MISSING:
                                merged[key] = reducer(default, value)
                            else:
                                merged[key] = value
                else:
                    merged[key] = value
            # 验证，但无法保留可能的类型转换
            # 因为一旦使用BaseModel就需要model_dump，而model_dump要么保留所有信息，要么丢失关键信息，都不是我想要的
            # 比如model_dump(exclude_unset=True)时BaseMessage会丢失role
            self.state_schema.model_validate(merged)
        return merged
