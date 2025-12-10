from langgraph.graph.state import CompiledStateGraph, StateGraph
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from typing import Sequence, Dict, Any, Union, Callable, Optional

from aiosqlite import Connection

class BaseGraph:

    graph: CompiledStateGraph
    graph_builder: StateGraph
    conn: Connection

    llm: BaseChatModel
    tools: list[BaseTool]

    def __init__(self, llm: Optional[BaseChatModel] = None,
        tools: Optional[Sequence[Union[Dict[str, Any], type, Callable, BaseTool]]] = None
    ):
        if llm is not None:
            self.llm = llm
        if not hasattr(self, 'tools'):
            self.tools = []
        if tools is not None:
            self.tools.extend(tools)
