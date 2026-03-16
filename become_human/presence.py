from typing import Union, Optional, override
from enum import StrEnum
import random
from datetime import timedelta, datetime
from loguru import logger

from langchain_core.messages import HumanMessage, BaseMessage

from sprited.plugin import *
from sprited.store.base import StoreModel, StoreField
from sprited.types.manager import CallSpriteRequest
from sprited.message import SpritedMsgMeta, DEFAULT_USER_MSG_TYPE
from sprited.times import TimestampUs, Times, format_duration, format_time
from sprited.store.manager import store_manager
from sprited.scheduler import Schedule, get_schedules, delete_schedules, add_schedules
from sprited.tools.send_message import SEND_MESSAGE
from sprited.manager import sprite_manager
from sprited.event import event_bus

NAME = 'bh_presence'

class PresenceConfig(StoreModel):
    _namespace = NAME
    _title = '在线状态配置'

    user_message_types: tuple[str, ...] = StoreField(default=(DEFAULT_USER_MSG_TYPE,), title='用户消息类型', description="哪些消息类型被认为是用户消息，从而触发相关操作")
    always_available: bool = StoreField(default=False, title="保持可用", description="是否一直处于可用状态，也即不存在sprite因不活跃而away的情况。若是，则available_duration_range将仅用作回收消息等功能，且passive_call依然有效，只有wakeup_call会失效")
    available_duration_range: tuple[float, float] = StoreField(default=(1800.0, 7200.0), title='在线时长随机范围', description="在线时长随机范围（最小值和最大值），在这之后进入休眠状态")
    temporary_available_duration_range: tuple[float, float] = StoreField(default=(180.0, 1800.0), title='临时在线时长随机范围', description="在无新消息时self_call后sprite获得的临时在线时长的随机范围（最小值和最大值），单位为秒。")
    passive_call_intervals: Union[tuple[tuple[float, float], ...], tuple[()]] = StoreField(default=(
        (1800.0, 32400.0),
        (16200.0, 97200.0),
        (97200.0, 388800.0)
    ), title='被动自我调用时间间隔随机范围', description="在离线状态时sprite被动自我调用的时间间隔的随机范围（最小值和最大值），单位为秒，睡觉期间不算时间")
    wakeup_call_interval: Union[tuple[float, float], tuple[()]] = StoreField(default=(1.0, 10800.0), title='唤醒调用时间间隔随机范围', description="在进入离线状态后，通过发送用户消息唤醒sprite需要的时间随机范围（最小值和最大值），单位为秒")
    sleep_time_range: Union[tuple[float, float], tuple[()]] = StoreField(default=(79200.0, 18000.0), title='睡眠时间段', description="sprite进入睡眠的时间段，单位为秒。目前的作用是self_call的时间生成会跳过这个时间段。")

class PresenceState(StrEnum):
    AVAILABLE = 'available'
    AWAY = 'away'
    SLEEPING = 'sleeping'
    # 暂时没有offline
    #OFFLINE = 'offline'

    def is_available(self) -> bool:
        return self == PresenceState.AVAILABLE
    def is_away(self) -> bool:
        return self == PresenceState.AWAY
    def is_sleeping(self) -> bool:
        return self == PresenceState.SLEEPING
    #def is_offline(self) -> bool:
    #    return self == PresenceState.OFFLINE

class PresenceData(StoreModel):
    _namespace = NAME
    _title = '在线状态数据'

    presence_state: PresenceState = StoreField(default=PresenceState.AVAILABLE, description="当前状态，available, away, sleeping, offline")
    set_away_schedule_id: str = StoreField(default='', description="设置为离线状态的计划ID")
    wakeup_call_schedule_id: str = StoreField(default='', description="唤醒调用计划ID")
    sleep_schedule_id: str = StoreField(default='', description="睡眠计划ID")
    last_user_input: TimestampUs = StoreField(default_factory=TimestampUs.now, description="上次用户输入时间，为sprite_world时间")
    has_new_user_call: bool = StoreField(default=True, description="是否有新的用户调用")


async def set_away_job(sprite_id: str) -> None:
    await PresencePlugin.set_presence(sprite_id, PresenceState.AWAY)

async def self_call_job(sprite_id: str, is_wakeup: bool = False) -> None:

    config_store = store_manager.get_model(sprite_id, PresenceConfig)
    data_store = store_manager.get_model(sprite_id, PresenceData)
    time_settings = store_manager.get_settings(sprite_id).time_settings
    current_times = Times.from_time_settings(time_settings)
    # 只是用来处理提示词
    is_available = data_store.presence_state.is_available()

    # 有新的消息
    if data_store.has_new_user_call:
        await generate_new_self_calls(sprite_id, current_times.sprite_world_datetime)
        await _update_away_time(sprite_id, config_store, data_store, current_times)
        input_content3 = f'''检查到当前有新的消息，请结合上下文、时间以及你的角色设定考虑要如何回复，或在某些特殊情况下保持沉默不理会用户。只需控制`{SEND_MESSAGE}`工具的使用与否即可实现。
{'注意，休眠模式只有在用户发送消息后才会被解除或重新计时。由于用户发送了新的消息，这次唤醒会使你重新回到正常的活跃状态。' if not is_available else ''}'''

    # 没有新的消息
    else:
        temporary_available_until = current_times.sprite_world_timestampus + random.uniform(config_store.temporary_available_duration_range[0], config_store.temporary_available_duration_range[1]) * 1_000_000
        if temporary_available_until > await PresencePlugin.get_away_time(sprite_id):
            await _update_away_time(sprite_id, config_store, data_store, current_times, temporary_available_until)
        input_content3 = f'''{'检查到当前没有新的消息，' if not is_available else ''}请结合上下文、时间以及你的角色设定考虑是否要尝试主动给用户发送消息，或保持沉默继续等待用户的新消息。只需控制`{SEND_MESSAGE}`工具的使用与否即可实现。
{'注意，休眠模式只有在用户发送消息后才会被解除。由于当前没有用户发送新的消息，接下来不论你是否发送消息，你都只会短暂地回到活跃状态，之后继续保持休眠状态等待下一次唤醒（如果在短暂的活跃状态期间依然没有收到新的消息）。' if not is_available else ''}'''

    past_microseconds = current_times.sprite_world_timestampus - data_store.last_user_input
    if is_available:
        input_content2 = f'距离上一次与用户交互过去了{format_duration(past_microseconds)}。虽然目前还没有收到用户的新消息，但你触发了一次随机的自我唤醒（这是为了给你主动向用户对话的可能）。{input_content3}'
    else:
        input_content2 = f'''由于自上次与用户交互以来（{format_duration(past_microseconds)}前），在一定时间内没有用户发送新的消息，你自动进入了休眠状态（在休眠状态下你会以随机的时间间隔检查是否有新的消息并短暂地回到活跃状态，而不是当有新消息时立即响应。这主要是为了模拟在停止聊天的一段时间之后，人们可能不会一直盯着最新消息而是会去做别的事，然后时不时回来检查新消息的情景）。
现在将你唤醒，检查是否有新的消息...
{input_content3}'''

    input_content = f'当前时间是 {format_time(current_times.sprite_world_datetime)}，{input_content2}'

    await sprite_manager.call_sprite_for_system(
        sprite_id=sprite_id,
        content=input_content,
        times=current_times,
        is_self_call=True,
        bh_memory={
            'passive_retrieval': ''
        }
    )

    if is_wakeup:
        data_store.wakeup_call_schedule_id = ''

async def set_sleeping_job(sprite_id: str) -> None:
    """如果当前状态是AWAY，那么就设置为SLEEPING"""
    if PresencePlugin.get_presence(sprite_id).is_away():
        await PresencePlugin.set_presence(sprite_id, PresenceState.SLEEPING)


class PresencePlugin(BasePlugin):
    """在线状态插件"""
    name = NAME
    config: type[PresenceConfig] = PresenceConfig
    data: type[PresenceData] = PresenceData
#     prompts = PluginPrompts(
#         secondary=PluginPrompt(
#             title='被动自我唤醒',
#             content='''作为一个由AI大模型驱动的agent，一般来说只有当用户主动向你发送消息时你才会被唤醒。但现在你被赋予了被动自我唤醒的能力。
# 具体来说，当在用户没有发送消息的时候，系统会在随机时间唤醒你，此时你可以依情况尝试主动与用户互动，或者什么都不做。'''
#         )
#     )

    @staticmethod
    def get_presence(sprite_id: str) -> PresenceState:
        data_store = store_manager.get_model(sprite_id, PresenceData)
        return data_store.presence_state

    @staticmethod
    async def set_presence(sprite_id: str, presence: Union[PresenceState, str]) -> None:
        config_store = store_manager.get_model(sprite_id, PresenceConfig)
        data_store = store_manager.get_model(sprite_id, PresenceData)
        await _set_presence(sprite_id, config_store, data_store, presence)

    @staticmethod
    async def update_away_time(sprite_id: str, fixed: Optional[TimestampUs] = None) -> None:
        config_store = store_manager.get_model(sprite_id, PresenceConfig)
        data_store = store_manager.get_model(sprite_id, PresenceData)
        time_settings = store_manager.get_settings(sprite_id).time_settings
        current_times = Times.from_time_settings(time_settings)
        await _update_away_time(sprite_id, config_store, data_store, current_times, fixed)

    @staticmethod
    async def get_away_time(sprite_id: str) -> Union[TimestampUs, int]:
        schedules = await get_schedules([
            Schedule.Condition(key='sprite_id', value=sprite_id),
            Schedule.Condition(key='schedule_provider', value=NAME),
            Schedule.Condition(key='schedule_type', value='set_offline'),
        ])
        if not schedules:
            return TimestampUs(0)
        return schedules[0].trigger_time

    @staticmethod
    def is_user_input(sprite_id: str, messages: Union[CallSpriteRequest, list[BaseMessage], BaseMessage]) -> bool:
        config_store = store_manager.get_model(sprite_id, PresenceConfig)
        return _is_user_input(messages, config_store)


    @override
    async def on_manager_init(self) -> None:
        event_bus.register(NAME+':on_presence_changed')

    @override
    async def on_sprite_init(self, sprite_id: str, /) -> None:
        config_store = store_manager.get_model(sprite_id, self.config)
        if config_store.sleep_time_range:
            # TODO 暂时不支持中途修改
            data_store = store_manager.get_model(sprite_id, self.data)
            if not data_store.sleep_schedule_id:
                schedule = Schedule(
                    sprite_id=sprite_id,
                    schedule_provider=NAME,
                    schedule_type='set_sleeping',
                    job=set_sleeping_job,
                    job_args=(sprite_id,),
                    scheduled_time_of_day=config_store.sleep_time_range[0],
                    scheduled_every_day=True,
                    scheduled_every_month=True,
                )
                await schedule.add_to_db()
                data_store.sleep_schedule_id = schedule.schedule_id

    @override
    async def before_call_sprite(self, request: CallSpriteRequest, info: BeforeCallSpriteInfo, /) -> Optional[BeforeCallSpriteControl]:
        if info.cancel_ctrl.current:
            return
        sprite_id = request.sprite_id
        config_store = store_manager.get_model(sprite_id, self.config)
        data_store = store_manager.get_model(sprite_id, self.data)
        # 如果当前时间超过了活跃时间，取消调用
        if _is_user_input(request, config_store):
            if not data_store.presence_state.is_available():
                return BeforeCallSpriteControl(cancel=True, keep_input_messages=True)

    async def on_call_sprite(self, request: CallSpriteRequest, info: OnCallSpriteInfo, /) -> Optional[OnCallSpriteControl]:
        sprite_id = request.sprite_id
        config_store = store_manager.get_model(sprite_id, self.config)
        if _is_user_input(request, config_store):
            data_store = store_manager.get_model(sprite_id, self.data)
            time_settings = store_manager.get_settings(sprite_id).time_settings
            current_times = Times.from_time_settings(time_settings)
            # 这是第一次生成
            if not data_store.set_away_schedule_id:
                await _update_away_time(sprite_id, config_store, data_store, current_times)
            data_store.last_user_input = current_times.sprite_world_timestampus
            data_store.has_new_user_call = True

    async def after_call_sprite(self, request: CallSpriteRequest, info: AfterCallSpriteInfo, /) -> None:
        sprite_id = request.sprite_id
        if info.cancelled:
            # 如果是由自己取消的，说明用户在不活跃的情况下调用了sprite
            if info.cancelled_reason == 'before_call_sprite' and info.cancelled_by_plugin == self.name:
                time_settings = store_manager.get_settings(sprite_id).time_settings
                current_times = Times.from_time_settings(time_settings)
                await generate_new_self_calls(sprite_id, current_times.sprite_world_datetime, is_wakeup=True)
            return
        if not self.is_user_input(sprite_id, request):
            return
        time_settings = store_manager.get_settings(sprite_id).time_settings
        current_times = Times.from_time_settings(time_settings)
        await generate_new_self_calls(sprite_id, current_times.sprite_world_datetime)



async def _set_presence(sprite_id: str, config_store: PresenceConfig, data_store: PresenceData, presence_state: Union[PresenceState, str]) -> None:
    if not isinstance(presence_state, PresenceState):
        original = PresenceState(presence_state)
    else:
        original = presence_state
    new = original
    old = data_store.presence_state
    if config_store.always_available:
        new = PresenceState.AVAILABLE
    # 目前来说，只要在睡眠时间段内设置presence为away，就会被认为是sleeping
    elif original.is_away() and config_store.sleep_time_range:
        current_times = Times.from_time_settings(store_manager.get_settings(sprite_id).time_settings)
        if is_sleep_time(current_times.sprite_world_datetime, config_store.sleep_time_range):
            new = PresenceState.SLEEPING
    data_store.presence_state = new
    await event_bus.publish(
        NAME+':on_presence_changed',
        sprite_id,
        new=new,
        old=old,
        original=original,
    )

async def _update_away_time(sprite_id: str, config_store: PresenceConfig, data_store: PresenceData, current_times: Times, fixed: Optional[TimestampUs] = None) -> None:
    await _set_presence(sprite_id, config_store, data_store, PresenceState.AVAILABLE)
    if config_store.always_available:
        return
    if fixed is not None:
        away_time = fixed
    else:
        away_time = current_times.sprite_world_timestampus + random.uniform(config_store.available_duration_range[0], config_store.available_duration_range[1]) * 1_000_000
    if data_store.set_away_schedule_id:
        schedules = await get_schedules([
            Schedule.Condition(key='schedule_id', value=data_store.set_away_schedule_id)
        ])
        if schedules:
            await delete_schedules(schedules)
    new_schedule = Schedule(
        sprite_id=sprite_id,
        schedule_provider=NAME,
        schedule_type='set_offline',
        job=set_away_job,
        job_kwargs={'sprite_id': sprite_id},
        trigger_time=away_time,
        time_reference='sprite_world'
    )
    await new_schedule.add_to_db()
    data_store.set_away_schedule_id = new_schedule.schedule_id

def _is_user_input(messages: Union[CallSpriteRequest, list[BaseMessage], BaseMessage], config_store: PresenceConfig) -> bool:
    if isinstance(messages, CallSpriteRequest):
        messages = messages.input_messages
    elif not isinstance(messages, (list, tuple)):
        messages = [messages]
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        try:
            metadata = SpritedMsgMeta.parse(message)
        except KeyError:
            pass
        if metadata.message_type in config_store.user_message_types:
            return True
    return False


async def generate_new_self_calls(sprite_id: str, current_time: datetime, is_wakeup: bool = False) -> None:

    data_store = store_manager.get_model(sprite_id, PresenceData)
    # 检查是否需要生成wakeup
    if is_wakeup:
        if data_store.wakeup_call_schedule_id:
            return
    # 如果是passive，每次生成都会把包括wakeup在内的全部删除
    else:
        data_store.has_new_user_call = False
        current_schedules = await get_schedules([
            Schedule.Condition(key='sprite_id', value=sprite_id),
            Schedule.Condition(key='schedule_provider', value=NAME)
        ])
        if current_schedules:
            await delete_schedules(current_schedules)


    config_store = store_manager.get_model(sprite_id, PresenceConfig)
    if is_wakeup:
        if config_store.always_available or not config_store.wakeup_call_interval:
            return
        else:
            time_ranges = [config_store.wakeup_call_interval]
    else:
        time_ranges = config_store.passive_call_intervals
    if not time_ranges:
        return
    if not config_store.sleep_time_range:
        sleep_time_start = 0.0
        sleep_time_end = 0.0
    else:
        sleep_time_start = config_store.sleep_time_range[0]
        sleep_time_end = config_store.sleep_time_range[1]

    if sleep_time_start > sleep_time_end:
        sleep_time_total = 86400 - sleep_time_start + sleep_time_end
    else:
        sleep_time_total = sleep_time_end - sleep_time_start

    _delta = timedelta(
        hours=current_time.hour,
        minutes=current_time.minute,
        seconds=current_time.second,
        microseconds=current_time.microsecond
    )
    current_date = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
    self_call_time_secondses = []
    # 首先看时间本身有没有在睡眠时间段，如果有则先跳过
    if is_sleep_time(_delta, config_store.sleep_time_range):
        if sleep_time_start > sleep_time_end:
            _delta += timedelta(seconds=sleep_time_end + 86400 - _delta.seconds)
        else:
            _delta += timedelta(seconds=sleep_time_end - _delta.seconds)

    for time_range in time_ranges:
        random_seconds = random.uniform(time_range[0], time_range[1])
        random_timedelta = timedelta(seconds=random_seconds)

        # new_seconds是最终要添加的秒数
        new_seconds = random_seconds
        new_seconds += int(random_timedelta.seconds / (86400 - sleep_time_total)) * sleep_time_total
        # new_timedelta只是用来检查是否在睡眠时间段
        new_timedelta = timedelta(seconds=new_seconds) + _delta
        while is_sleep_time(new_timedelta, config_store.sleep_time_range):
            new_seconds += sleep_time_total
            new_timedelta += timedelta(seconds=sleep_time_total)

        _delta += timedelta(seconds=new_seconds)
        self_call_time_secondses.append(TimestampUs(current_date + _delta))

    schedules = [Schedule(
        sprite_id=sprite_id,
        schedule_provider=NAME,
        schedule_type='wakeup_call' if is_wakeup else 'passive_call',
        job=self_call_job,
        job_kwargs={
            'sprite_id': sprite_id,
            'is_wakeup': is_wakeup,
        },
        timeout_seconds=3600.0,
        trigger_time=timestampus,
        time_reference='sprite_world',
    ) for timestampus in self_call_time_secondses]
    if schedules:
        await add_schedules(schedules)

def is_sleep_time(current: Union[timedelta, datetime], sleep_time_range: Union[tuple[float, float], tuple[()]]) -> bool:
    if sleep_time_range:
        sleep_time_start = sleep_time_range[0]
        sleep_time_end = sleep_time_range[1]
    else:
        return False
    if isinstance(current, datetime):
        seconds = current.hour * 3600 + current.minute * 60 + current.second
    else:
        seconds = current.seconds
    result = False
    if sleep_time_start > sleep_time_end:
        if seconds >= sleep_time_start or seconds < sleep_time_end:
            result = True
    else:
        if seconds >= sleep_time_start and seconds < sleep_time_end:
            result = True
    return result
