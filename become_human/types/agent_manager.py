from typing import Literal, Any
from pydantic import BaseModel, Field

from become_human.types.main import MainContext

class CallAgentKwargs(BaseModel):
    graph_input: dict[str, Any] = Field(description='The input to the graph')
    graph_context: MainContext = Field(description='The context of the graph')
    double_texting_strategy: Literal['merge', 'enqueue', 'reject'] = Field(default='merge', description='The strategy to use when double texting')
    random_wait: bool = Field(default=False, description='Whether to random wait before calling the agent')
    is_self_call: bool = Field(default=False, description='Whether the call is a self call')
