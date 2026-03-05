from typing import Literal, Optional
from loguru import logger

from become_human.plugin import *
from become_human.types.manager import CallSpriteRequest
from become_human.times import TimestampUs
from become_human.store.base import StoreModel, StoreField
from become_human.store.manager import store_manager
from become_human.manager import sprite_manager

class TimeIncrementerStore(StoreModel):
    _namespace = 'time_incrementer_config'
    _title = '时间增量器配置'
    _description = '用于在每次call_sprite之后根据配置的规则（加1或根据调用耗时，再乘上一个系数）跳过sprite的时间'

    increase_by: Literal['one', 'elapsed'] = StoreField(default='one', title='增量方式', description='加1或根据调用耗时')
    multiplier: float = StoreField(default=1.0, title='系数', description='increase_by将要乘以的系数')

class TimeIncrementerPlugin(BasePlugin):
    name = 'time_incrementer'
    config = TimeIncrementerStore

    sprite_start_times: dict[str, TimestampUs]

    def __init__(self) -> None:
        self.sprite_start_times = {}

    async def on_call_sprite(self, request: CallSpriteRequest, info: OnCallSpriteInfo, /) -> Optional[OnCallSpriteControl]:
        if info.is_update_messages_only:
            return
        sprite_id = request.sprite_id
        if sprite_id not in self.sprite_start_times:
            self.sprite_start_times[sprite_id] = TimestampUs.now()

    async def after_call_sprite(self, request: CallSpriteRequest, info: AfterCallSpriteInfo, /) -> None:
        if info.cancelled:
            return
        sprite_id = request.sprite_id
        start_time = self.sprite_start_times.pop(sprite_id)
        accumulator_settings = store_manager.get_model(sprite_id, TimeIncrementerStore)
        time_settings = store_manager.get_settings(sprite_id).time_settings
        if accumulator_settings.increase_by == 'elapsed':
            end_time = TimestampUs.now()
            if end_time < start_time:
                logger.error(f'时钟回拨？start_time={start_time}, end_time={end_time}')
                return
            new_time_settings = time_settings.add_offset_from_now(int((end_time - start_time) * accumulator_settings.multiplier), 'subjective')
        else:
            new_time_settings = time_settings.add_offset_from_now(int(accumulator_settings.multiplier), 'subjective')
        await sprite_manager.set_time_settings(sprite_id, new_time_settings)
