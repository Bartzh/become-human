from typing import Literal, Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from langchain_core.messages import BaseMessage

DoubleTextingStrategy = Literal['merge', 'interrupt', 'enqueue', 'reject']

class CallSpriteRequest(BaseModel):
    sprite_id: str
    input_messages: list[BaseMessage]
    #input_messages: list[HumanMessage]
    sprite_run_id: str = Field(default_factory=lambda: str(uuid4()))
    double_texting_strategy: DoubleTextingStrategy = Field(default='merge')
    random_wait: bool = Field(default=False)
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator('extra_kwargs', mode='after')
    @classmethod
    def validate_extra_kwargs(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v.keys() & {'sprite_id', 'input_messages', 'sprite_run_id', 'double_texting_strategy', 'random_wait'}:
            raise ValueError('extra_kwargs must not contain sprite_id, input_messages, sprite_run_id, double_texting_strategy, random_wait')
        return v

class SpriteOutput(BaseModel):
    sprite_id: str
    method: str = ''
    params: dict[str, Any] = Field(default_factory=dict)
    id: str = Field(default_factory=lambda: str(uuid4()))
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator('extra_kwargs', mode='after')
    @classmethod
    def validate_extra_kwargs(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v.keys() & {'sprite_id', 'method', 'params', 'id'}:
            raise ValueError('extra_kwargs must not contain sprite_id, method, params, id')
        return v
