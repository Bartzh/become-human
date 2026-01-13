from typing import Annotated

import pytest
from pydantic import ValidationError, BaseModel, Field

from langchain.messages import HumanMessage, AIMessage, RemoveMessage, AnyMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

from become_human.graphs.base import StateMerger

class State(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    recycle_messages: list[AnyMessage] = Field(default_factory=list)
    generated: bool = Field(default=False)

class TestStateMerger:
    def setup_method(self):
        self.merger = StateMerger(State)

    def test_merge_empty_states(self):
        state1 = {}
        state2 = {}
        merged_state = self.merger.merge([state1, state2])
        assert merged_state == {}
        merged_state = self.merger.merge([state1])
        assert merged_state == {}
        merged_state = self.merger.merge([])
        assert merged_state == {}

    def test_merge_states_with_different_values(self):
        state0 = {'generated': True}
        state1 = {'generated': False}
        state2 = {'messages': {'role': 'user', 'content': 'Hello!'}}
        state3 = {'messages': {'role': 'assistant', 'content': 'Hi there!'}}
        merged_state = self.merger.merge([state0, state1, state2, state3])
        assert merged_state['generated'] == False
        assert len(merged_state['messages']) == 2
        assert isinstance(merged_state['messages'][0], HumanMessage)
        assert merged_state['messages'][0].text == 'Hello!'
        assert isinstance(merged_state['messages'][1], AIMessage)
        assert merged_state['messages'][1].text == 'Hi there!'
        state4 = {'messages': RemoveMessage(id=REMOVE_ALL_MESSAGES)}
        merged_state = self.merger.merge([merged_state, state4])
        assert merged_state['messages'] == []

    def test_merge_wrong_types(self):
        state2 = {'recycle_messages': 'list'}
        with pytest.raises(ValidationError):
            self.merger.merge([state2])
