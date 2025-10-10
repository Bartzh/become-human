from langgraph.graph.state import CompiledStateGraph, StateGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from typing import Sequence, Dict, Any, Union, Callable, Optional

import aiosqlite

from become_human.memory import MemoryManager

class BaseGraph:

    graph: CompiledStateGraph
    graph_builder: StateGraph
    conn: aiosqlite.Connection

    llm: BaseChatModel
    tools: list[BaseTool]
    memory_manager: MemoryManager

    def __init__(self, llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Union[Dict[str, Any], type, Callable, BaseTool]]] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        if llm is not None:
            self.llm = llm
        if not hasattr(self, 'tools'):
            self.tools = []
        if tools is not None:
            self.tools.extend(tools)
        if memory_manager is not None:
            self.memory_manager = memory_manager
