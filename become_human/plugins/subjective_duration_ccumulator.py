from loguru import logger
from typing import Literal

from become_human.plugin import Plugin
from become_human.types.agent_manager import CallAgentKwargs
from become_human.times import now_seconds
from become_human.store.base import StoreModel, StoreField
from become_human.store.manager import store_manager
from become_human.agent_manager import agent_manager

class SubjectiveDurationAccumulatorStore(StoreModel):
    increase_by: Literal['one', 'elapsed'] = StoreField(default='elapsed')
    multiplier: float = StoreField(default=1.0)

class SubjectiveDurationAccumulatorPlugin(Plugin):
    agent_start_times: dict[str, float]

    def __init__(self) -> None:
        self.agent_start_times = {}

    async def on_manager_init(self) -> None:
        await store_manager.register_model(SubjectiveDurationAccumulatorStore)

    async def before_call_agent(self, call_agent_kwargs: CallAgentKwargs, /) -> None:
        agent_id = call_agent_kwargs['graph_context'].agent_id
        if agent_id not in self.agent_start_times:
            self.agent_start_times[agent_id] = now_seconds()

    async def after_call_agent(self, call_agent_kwargs: CallAgentKwargs, interrupted: bool, rejected_or_queuing: bool, /) -> None:
        if interrupted or rejected_or_queuing:
            return
        agent_id = call_agent_kwargs['graph_context'].agent_id
        start_time = self.agent_start_times.pop(agent_id)
        accumulator_settings = await store_manager.get_model(SubjectiveDurationAccumulatorStore, agent_id)
        time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
        new_time_settings = time_settings.model_copy(deep=True)
        if accumulator_settings.increase_by == 'elapsed':
            end_time = now_seconds()
            if end_time < start_time:
                logger.error(f'时钟回拨？start_time={start_time}, end_time={end_time}')
                return
            new_time_settings.subjective_duration_setting.agent_time_anchor += (end_time - start_time) * accumulator_settings.multiplier
        else:
            new_time_settings.subjective_duration_setting.agent_time_anchor += accumulator_settings.multiplier
        await agent_manager.set_agent_time_settings(agent_id, new_time_settings)
