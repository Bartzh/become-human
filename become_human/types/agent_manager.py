from typing import Literal, Any, TypedDict

from become_human.types.main import MainContext

class CallAgentKwargs(TypedDict):
    graph_input: dict[str, Any]
    graph_context: MainContext
    double_texting_strategy: Literal['merge', 'enqueue', 'reject']
    random_wait: bool
    is_self_call: bool
