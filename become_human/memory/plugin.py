from typing import override, Optional, Literal, Union
from uuid import uuid4
import random
from loguru import logger

from langchain_core.messages import RemoveMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages

from sprited.utils import gather_safe
from sprited.types import CallSpriteRequest
from sprited.scheduler import Schedule, get_schedules, delete_schedules
from sprited.tool import SpriteTool
from sprited.store.manager import store_manager
from sprited.message import SpritedMsgMeta, extract_text_parts, construct_system_message
from sprited.times import Times
from sprited.plugin import *
from become_human.memory.types import PLUGIN_NAME, AnyMemoryType
from become_human.memory.store import MemoryConfig, MemoryData
from become_human.memory.base import memory_manager, validated_where, MEMORY_TYPES, format_retrieved_memory_groups, InitialMemory
from become_human.memory.message import MemoryMsgMeta, get_all_retrieved_memory_ids
from become_human.memory.recycling import recycle_memories, connect_last_memory
from become_human.memory.tools import RETRIEVE_MEMORIES, retrieve_memories_tool, add_memory_tool, construct_retrieve_memories_schema
from sprited.manager import sprite_manager



async def construct_default_memory_schedules(sprite_id: str) -> list[Schedule]:
    passive_retrieval_cleaner = await get_schedules([
        Schedule.Condition(key='sprite_id', value=sprite_id),
        Schedule.Condition(key='schedule_provider', value=PLUGIN_NAME),
        Schedule.Condition(key='schedule_type', value='cleaner')
    ])
    if not passive_retrieval_cleaner:
        await Schedule(
            sprite_id=sprite_id,
            schedule_provider=PLUGIN_NAME,
            schedule_type='cleaner',
            interval_fixed=5.0,
            job=clean_passive_retrieval_messages_job,
            job_kwargs={
                'sprite_id': sprite_id
            }
        ).add_to_db()
    elif len(passive_retrieval_cleaner) > 1:
        await delete_schedules([s for s in passive_retrieval_cleaner[1:]])
        logger.warning(f"Sprite {sprite_id} 意外地存在多个被动检索清理器，已删除除第一个以外的所有清理器")


    # memory_updaters = await get_schedules([
    #     Schedule.Condition(key='sprite_id', value=sprite_id),
    #     Schedule.Condition(key='schedule_provider', value=PLUGIN_NAME),
    #     Schedule.Condition(key='schedule_type', value='updater')
    # ])
    # if len(memory_updaters) != 5:
    #     await delete_schedules([s.schedule_id for s in memory_updaters])
    #     await add_schedules([
    #         Schedule(sprite_id=sprite_id, schedule_provider=PLUGIN_NAME, schedule_type='updater', interval_fixed=5.0, job=update_memories_job, job_kwargs={
    #             'sprite_id': sprite_id,
    #             'ttl_range': [{'$gte': 0}, {'$lt': 43200}]
    #         }),
    #         Schedule(sprite_id=sprite_id, schedule_provider=PLUGIN_NAME, schedule_type='updater', interval_fixed=30.0, job=update_memories_job, job_kwargs={
    #             'sprite_id': sprite_id,
    #             'ttl_range': [{'$gte': 43200}, {'$lt': 86400}]
    #         }),
    #         Schedule(sprite_id=sprite_id, schedule_provider=PLUGIN_NAME, schedule_type='updater', interval_fixed=60.0, job=update_memories_job, job_kwargs={
    #             'sprite_id': sprite_id,
    #             'ttl_range': [{'$gte': 86400}, {'$lt': 864000}]
    #         }),
    #         Schedule(sprite_id=sprite_id, schedule_provider=PLUGIN_NAME, schedule_type='updater', interval_fixed=500.0, job=update_memories_job, job_kwargs={
    #             'sprite_id': sprite_id,
    #             'ttl_range': [{'$gte': 864000}, {'$lt': 8640000}]
    #         }),
    #         Schedule(sprite_id=sprite_id, schedule_provider=PLUGIN_NAME, schedule_type='updater', interval_fixed=3600.0, job=update_memories_job, job_kwargs={
    #             'sprite_id': sprite_id,
    #             'ttl_range': [{'$gte': 8640000}]
    #         }),
    #     ])

async def clean_passive_retrieval_messages_job(sprite_id: str) -> None:
    """清除过期的被动检索消息"""
    config_store = store_manager.get_model(sprite_id, MemoryConfig)
    passive_retrieval_ttl = config_store.passive_retrieval_ttl
    if passive_retrieval_ttl > 0:
        time_settings = store_manager.get_settings(sprite_id).time_settings
        current_times = Times.from_time_settings(time_settings)
        passive_retrieval_messages_to_remove = []
        for m in await sprite_manager.get_messages(sprite_id):
            try:
                sp_message_metadata = SpritedMsgMeta.parse(m)
            except KeyError:
                continue
            if (
                sp_message_metadata.message_type == f'{PLUGIN_NAME}:passive_retrieval' and
                passive_retrieval_ttl >= abs(current_times.sprite_subjective_tick - sp_message_metadata.creation_times.sprite_subjective_tick)
            ):
                passive_retrieval_messages_to_remove.append(RemoveMessage(id=m.id))
        if passive_retrieval_messages_to_remove:
            await sprite_manager.update_messages(sprite_id, passive_retrieval_messages_to_remove)

async def update_memories_job(sprite_id: str, ttl_range: list[dict[str, int]]) -> None:
    """更新记忆job"""
    update_count = 0
    for t in store_manager.get_model(sprite_id, MemoryConfig).memory_types:
        where = validated_where({'$and': [{'ttl': item} for item in ttl_range]})
        update_count += await memory_manager.tick_memories(sprite_id, t, where=where)

    logger.debug(f'updated {update_count} memories for sprite "{sprite_id}".')



class MemoryPlugin(BasePlugin):
    name = PLUGIN_NAME
    dependencies = [PluginRelation(name='bh_presence')]
    config = MemoryConfig
    data = MemoryData
    tools = [SpriteTool(retrieve_memories_tool, hide_by_default=True), add_memory_tool]

    async def add_memory(
        self,
        sprite_id: str,
        type: AnyMemoryType,
        content: str,
        *,
        id: Optional[str] = None,
        creation_times: Optional[Times] = None,
        ttl: Optional[Union[int, Literal['max']]] = None,
        retrievability: float = 1.0,
        lambd: float = 1.0,
        previous_memory_id: Optional[str] = None,
        next_memory_id: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        **kwargs: dict
    ) -> None:
        """添加记忆"""
        if type not in MEMORY_TYPES:
            raise ValueError(f'Invalid memory type: {type}')
        if type == 'summary':
            raise ValueError('Summary memory type is not allowed to be added manually.')
        if not content.strip():
            raise ValueError('Memory content cannot be empty.')
        if not sprite_id.strip():
            raise ValueError('Sprite ID cannot be empty.')
        if id is None:
            id = str(uuid4())
        if creation_times is None:
            creation_times = Times.from_time_settings(store_manager.get_settings(sprite_id).time_settings)
        if ttl is None:
            ttl = int(random.expovariate(lambd) * store_manager.get_model(sprite_id, MemoryConfig).memory_base_ttl)
        max_ttl = store_manager.get_model(sprite_id, MemoryConfig).memory_max_ttl
        if ttl == 'max':
            ttl = max_ttl
        else:
            ttl = min(ttl, max_ttl)
        if previous_memory_id is None:
            previous_memory_id = await connect_last_memory(sprite_id, type, [id])
        await memory_manager.add_memories([
            InitialMemory(
                content=content,
                type=type,
                creation_times=creation_times,
                ttl=ttl,
                id=id,
                retrievability=retrievability,
                previous_memory_id=previous_memory_id,
                next_memory_id=next_memory_id,
                similarity_threshold=similarity_threshold,
                extra=kwargs
            )
        ], sprite_id)

    @override
    async def on_sprite_init(self, sprite_id: str, /) -> None:
        await construct_default_memory_schedules(sprite_id)

    @override
    async def before_call_model(self, request: CallSpriteRequest, info: BeforeCallModelInfo, /) -> None:
        self.prompts = PluginPrompts(
            secondary=PluginPrompt(
                title='记忆系统',
                content=f'''- **记忆存储**：
- 记忆是自动存储的（无需你主动存储），并且遵循“用进废退”原则。经常被检索的记忆会被强化，而很少被检索的记忆会被逐渐遗忘。
- **被动检索（潜意识）**：
    - 每次你被调用时，系统会自动使用用户输入的消息检索相关记忆。这条消息是自动生成的，可以参考它来提供更准确的回答。
    - 但请注意：被动检索可能不够精确，比如当用户提到模糊的时间点如“上周”时，被动检索因无法以准确时间点进行检索很有可能获取不到多少有用的信息，请注意甄别。
- **主动检索**：
    - 如果你需要更精确的记忆，请调用`{RETRIEVE_MEMORIES}`工具。该工具允许你主动检索记忆，你可以使用更合适的查询语句来获取更好的结果。
- **检索结果**
    - 检索结果会以多个记忆组（memory_group）的形式返回给你，一个记忆组中必有一个「目标记忆」以及零或若干个「相邻记忆」：
        -「目标记忆」指被检索语句检索到的记忆，这条记忆才是与检索语句相关的，在同一个记忆组内只会存在一个。
        -「相邻记忆」（若有）则是指与记忆组内唯一的「目标记忆」在创建时间上相邻的一些记忆，虽与检索语句没有关联，但与「源记忆」结合在一起可能会得到相关联的其他信息，或还原当时情景。
    - 检索中得分越高越与检索语句相关的记忆组（以「目标记忆」为准）越靠后。得分（score）是一个0~1的值。
    - 在这些记忆中还可能会出现「模糊的记忆」，其中的`*`星号意味着暂时没想起来的细节，这是由于该记忆检索时的得分过低，如因检索语句不够相关，或记忆本身不够新鲜。假如再次检索这些记忆，由于记忆的新鲜度提升了，所以`*`星号应当会减少。

请充分利用被动检索和主动检索工具提供的记忆来提供更优质的回答。

但请注意，你检索到的记忆也可能会有问题，如：不相关、过时、又或者是你真正想要的记忆没有被检索到，也有可能是因为它已经被遗忘了。

所以，一定要注意甄别记忆的内容，当意识到检索到的记忆有以上不靠谱的情况时，你可以选择再次尝试检索，又或者放弃检索，使用“我不知道”“我忘记了”“我不确定”等表达作解释，避免胡编乱造。'''
            )
        )
        self.tools[0].set_schema(request.sprite_id, construct_retrieve_memories_schema(request.sprite_id))


    @override
    async def on_call_sprite(self, request: CallSpriteRequest, info: OnCallSpriteInfo, /) -> None:
        #if info.is_update_messages_only:
        #    return
        sprite_id = request.sprite_id
        # TODO 考虑改为只有当配置了passive_retrieval时才进行被动检索
        if request.extra_kwargs.get(PLUGIN_NAME, {}).get('passive_retrieval') is not None:
            search_string = request.extra_kwargs[PLUGIN_NAME]['passive_retrieval']
            if not isinstance(search_string, str):
                search_string = ""
                logger.warning(f"[{PLUGIN_NAME}] passive_retrieval is not str, set to empty string: {search_string}")
            elif not search_string.strip():
                search_string = ""
        else:
            search_string = "\n\n".join("\n".join(extract_text_parts(m.content)) for m in info.input_messages_ctrl.current)
        if not search_string:
            return
        messages = await sprite_manager.get_messages(sprite_id)
        message_ids = [m.id for m in messages if m.id]
        retrieved_memory_ids = get_all_retrieved_memory_ids(messages)
        exclude_memory_ids = list(set(message_ids + retrieved_memory_ids))
        config_store = store_manager.get_model(sprite_id, MemoryConfig)
        passive_retrieve_groups = await memory_manager.retrieve_memories(
            sprite_id=sprite_id,
            retrieval_configs=config_store.passive_common_retrieval_configs,
            search_string=search_string,
            exclude_memory_ids=exclude_memory_ids
        )
        time_settings = store_manager.get_settings(sprite_id).time_settings
        passive_retrieve_content = format_retrieved_memory_groups(
            passive_retrieve_groups,
            time_settings.time_zone
        )
        current_times = Times.from_time_settings(time_settings)
        return OnCallSpriteControl(
            input_messages_patch=[construct_system_message(
                content=f'以下是根据用户输入自动从你的记忆（数据库）中检索到的内容，可能会出现无关信息（但视情况依然可作为谈资），如果需要进一步检索请调用工具`{RETRIEVE_MEMORIES}`：\n\n\n{passive_retrieve_content}',
                times=current_times,
                message_type=f"{PLUGIN_NAME}:passive_retrieval",
                extra_metas=[
                    MemoryMsgMeta(
                        do_not_store=True,
                        retrieved_memory_ids=[group.source_memory.doc.id for group in passive_retrieve_groups]
                    )
                ]
            )]
        )

    @override
    async def after_call_sprite(self, request: CallSpriteRequest, info: AfterCallSpriteInfo, /) -> None:
        sprite_id = request.sprite_id
        messages = info.new_messages
        if not messages:
            return

        config_store = store_manager.get_model(sprite_id, MemoryConfig)
        memory_types = config_store.memory_types
        empty_meta = MemoryMsgMeta()

        # 没回收的滚去回收
        recycle_messages = []
        if 'original' in memory_types:
            for message in messages:
                metadata = empty_meta.parse_with_default(message)
                if not metadata.recycled:
                    metadata.recycled = True
                    metadata.set_to(message)
                    recycle_messages.append(message)

        overflow_messages = None
        # 如果消息超过阈值，进行trim
        trigger_threshold = config_store.recycling_trigger_threshold
        if count_tokens_approximately(messages) >= trigger_threshold:
            max_tokens = config_store.recycling_target_size
            new_messages = trim_messages(
                messages=messages,
                max_tokens=max_tokens,
                token_counter=count_tokens_approximately,
                strategy='last',
                start_on=HumanMessage,
                #allow_partial=True,
                #text_splitter=RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
            )
            if not new_messages:
                logger.warning("Trim messages failed on overflowing.")
                new_messages = []
            excess_count = len(messages) - len(new_messages)
            old_messages = messages[:excess_count]
            # 如果第一条消息为extracted，说明在non active时已经被提取过了，尝试只从后向前删除掉extracted的消息而不删除其他消息
            if old_messages and empty_meta.parse_with_default(old_messages[0]).extracted:
                extracted_messages = []
                for m in old_messages:
                    if empty_meta.parse_with_default(m).extracted:
                        extracted_messages.append(m)
                    else:
                        break
                # 如果长度相等，说明所有消息都被trim掉了，无视
                if len(extracted_messages) < len(messages):
                    # 保证下一条消息是HumanMessage
                    idx = len(extracted_messages)
                    while idx > 0 and not isinstance(messages[idx], HumanMessage):
                        idx -= 1
                    extracted_messages = extracted_messages[:idx]
                    # 如果剩余的token少于阈值，那么就视为成功，修改变量，否则保持trim后的原始结果
                    if count_tokens_approximately(messages[len(extracted_messages):]) < trigger_threshold:
                        # 目前就这样，直接把extracted的消息送去overflow，因为在recycle episodic时会过滤掉extracted的消息
                        old_messages = extracted_messages
                        new_messages = messages[len(extracted_messages):]
            remove_messages = [RemoveMessage(id=message.id) for message in old_messages]
            if 'episodic' in memory_types:
                overflow_messages = old_messages
            await sprite_manager.update_messages(sprite_id, recycle_messages + remove_messages)
        else:
            await sprite_manager.update_messages(sprite_id, recycle_messages)


        if recycle_messages:
            recycles = [recycle_memories('original', sprite_id, recycle_messages)]
            if overflow_messages:
                recycles.append(recycle_memories('episodic', sprite_id, overflow_messages, sprite_manager.plus_model))
            await gather_safe(*recycles)

        return

    @override
    async def on_sprite_reset(self, sprite_id: str, /) -> None:
        for t in MEMORY_TYPES:
            memory_manager.delete_collection(sprite_id, t)
