from typing import Literal, Optional
from loguru import logger

from become_human.plugin import Plugin, Cancelled
from become_human.types.agent_manager import CallAgentKwargs
from become_human.times import TimestampUs
from become_human.store.base import StoreModel, StoreField
from become_human.store.manager import store_manager
from become_human.manager import agent_manager

class AgentTimeIncrementerStore(StoreModel):
    _namespace = 'agent_time_incrementer'
    _readable_name = 'agent时间增量器'
    _description = '用于在每次call_agent之后根据配置的规则（加1或根据调用耗时，再乘上一个系数）跳过agent的时间'

    increase_by: Literal['one', 'elapsed'] = StoreField(default='one', readable_name='增量方式', description='加1或根据调用耗时')
    multiplier: float = StoreField(default=1.0, readable_name='系数', description='increase_by将要乘以的系数')

class AgentTimeIncrementerPlugin(Plugin):
    name = 'agent_time_incrementer'
    config = AgentTimeIncrementerStore

    agent_start_times: dict[str, TimestampUs]

    def __init__(self) -> None:
        self.agent_start_times = {}

    async def before_call_agent(self, call_agent_kwargs: CallAgentKwargs, cancelled: Optional[Cancelled] = None, /) -> Optional[bool]:
        if cancelled:
            return
        agent_id = call_agent_kwargs.graph_context.agent_id
        if agent_id not in self.agent_start_times:
            self.agent_start_times[agent_id] = TimestampUs.now()

    async def after_call_agent(self, call_agent_kwargs: CallAgentKwargs, cancelled: Optional[Cancelled] = None, /) -> None:
        if cancelled:
            return
        agent_id = call_agent_kwargs.graph_context.agent_id
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
