from loguru import logger
import random

from become_human.plugin import *
from become_human.store.base import StoreModel, StoreField
from become_human.store.manager import store_manager
from become_human.times import format_time, Times
from become_human.message import InitalAIMessage, construct_system_message
from become_human.manager import sprite_manager


class InstructionConfig(StoreModel):
    _namespace = 'instruction_config'
    _readable_name = '引导提示词配置'

    instruction_prompt: str = StoreField(default="打个招呼吧。", readable_name="引导提示词", description="作为sprite的第一条用户消息出现，对sprite进行引导。")
    initial_ai_messages: list[InitalAIMessage] = StoreField(default_factory=list, readable_name="初始AI消息（随机列表）", description="初始AI消息，作为instruction_prompt的回复，不是必须的。会在列表中随机选择一条")
    react_instruction: bool = StoreField(default=False, readable_name="反应引导", description="是否以instruction_prompt调用sprite，这会覆盖initial_ai_messages。")

class InstructionData(StoreModel):
    _namespace = 'instruction_data'
    _readable_name = '引导提示词数据'
    is_first_time: bool = StoreField(default=True, readable_name="是否首次运行", description="是否首次运行，首次运行时会添加引导消息。")

class InstructionPlugin(BasePlugin):
    name = "instruction"
    config = InstructionConfig
    data = InstructionData

    async def on_sprite_init(self, sprite_id: str, /) -> None:
        data_store = store_manager.get_model(sprite_id, InstructionData)
        # 如果是首次运行，则添加或发送引导消息
        if data_store.is_first_time:
            data_store.is_first_time = False
            if sprite_manager.is_sprite_running(sprite_id):
                logger.warning(f"Sprite {sprite_id} 已被调用，将跳过引导消息。")
            else:
                config_store = store_manager.get_model(sprite_id, InstructionConfig)
                time_settings = store_manager.get_settings(sprite_id).time_settings
                current_times = Times.from_time_settings(time_settings)
                instruction_message = construct_system_message(
                    f'''当前时间是：{format_time(current_times.sprite_world_datetime)}。
这是你被初始化以来的第一条消息。如果你看到这条消息，说明在此消息之前你还没有收到过任何来自用户的消息。
这意味着你的“记忆”暂时是空白的，如果检索记忆时提示“没有找到任何匹配的记忆。”或检索不到什么有用的信息，这是正常的。
接下来是你与用户的初次见面，请根据你所扮演的角色以及以下的提示考虑应做出什么反应：\n''' + config_store.instruction_prompt,
                    current_times
                )
                if config_store.react_instruction:
                    await sprite_manager.call_sprite_for_system(
                        sprite_id,
                        instruction_message.text,
                        times=current_times,
                        bh_memory={
                            'passive_retrieval': ''
                        }
                    )
                else:
                    instruction_messages = [instruction_message]
                    if config_store.initial_ai_messages:
                        instruction_messages.extend(
                            random.choice(config_store.initial_ai_messages)
                            .construct_messages(current_times)
                        )
                    await sprite_manager.update_messages(sprite_id, instruction_messages)
