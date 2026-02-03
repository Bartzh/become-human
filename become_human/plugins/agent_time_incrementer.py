from typing import Literal
from loguru import logger

from become_human.plugin import Plugin
from become_human.types.agent_manager import CallAgentKwargs
from become_human.times import TimestampUs, Times
from become_human.store.base import StoreModel, StoreField
from become_human.store.manager import store_manager
from become_human.agent_manager import agent_manager

class AgentTimeIncrementerStore(StoreModel):
    increase_by: Literal['one', 'elapsed'] = StoreField(default='elapsed')
    multiplier: float = StoreField(default=1.0)

class AgentTimeIncrementerPlugin(Plugin):
    agent_start_times: dict[str, TimestampUs]

    def __init__(self) -> None:
        self.agent_start_times = {}

    async def on_manager_init(self) -> None:
        await store_manager.register_model(AgentTimeIncrementerStore)

    async def before_call_agent(self, call_agent_kwargs: CallAgentKwargs, /) -> None:
        agent_id = call_agent_kwargs['graph_context'].agent_id
        if agent_id not in self.agent_start_times:
            self.agent_start_times[agent_id] = TimestampUs.now()

    async def after_call_agent(self, call_agent_kwargs: CallAgentKwargs, interrupted: bool, rejected_or_queuing: bool, /) -> None:
        if interrupted or rejected_or_queuing:
            return
        agent_id = call_agent_kwargs['graph_context'].agent_id
        start_time = self.agent_start_times.pop(agent_id)
        accumulator_settings = await store_manager.get_model(agent_id, AgentTimeIncrementerStore)
        time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
        if accumulator_settings.increase_by == 'elapsed':
            end_time = TimestampUs.now()
            if end_time < start_time:
                logger.error(f'时钟回拨？start_time={start_time}, end_time={end_time}')
                return
            new_time_settings = time_settings.add_offset_from_now(int((end_time - start_time) * accumulator_settings.multiplier), 'subjective')
        else:
            new_time_settings = time_settings.add_offset_from_now(int(accumulator_settings.multiplier), 'subjective')
        await agent_manager.set_agent_time_settings(agent_id, new_time_settings)
