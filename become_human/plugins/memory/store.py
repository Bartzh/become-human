from typing import Literal, Optional

from become_human.plugins.memory.types import PLUGIN_NAME, MemoryRetrievalConfig, AnyMemoryType
from become_human.times import Times
from become_human.store.base import StoreModel, StoreField


class MemoryConfig(StoreModel):
    _namespace = PLUGIN_NAME
    _title = "memory设置"
    memory_types: tuple[AnyMemoryType, ...] = StoreField(default=('original', 'reflective', 'summary'), title='启用的记忆类型')
    memory_base_ttl: int = StoreField(default=259200_000_000, title='记忆稳定时长基值', description="记忆初始化时ttl的初始值")
    memory_max_words: int = StoreField(default=300, title='记忆最大Tokens数', description="单条记忆最大单词数，决定记忆难度，最大难度0.8")
    recycling_trigger_threshold: int = StoreField(default=24000, title='溢出回收阈值', description="触发溢出回收的阈值，单位为Token")
    recycling_target_size: int = StoreField(default=18000, title='溢出回收目标大小', description="溢出回收后目标大小，单位为Token")
    cleanup_on_unavailable: bool = StoreField(default=False, title='不可用时回收时清理', description="是否在不可用自动回收的同时清理回收的消息")
    cleanup_target_size: int = StoreField(default=2000, title='非活跃清理目标大小', description="非活跃清理后目标大小，单位为Token")
    summary_time_granularities: tuple[Literal['year', 'season', 'month', 'week', 'day'], ...] = StoreField(default=('year', 'season', 'month', 'week', 'day'), title='总结时间粒度')
    passive_retrieval_ttl: int = StoreField(default=3600_000_000, title='被动检索存活时长', description="被动检索消息的存活时长，按sprite主观ticks计算，单位为秒，到点后会被自动清理，设为0则不清理")
    passive_common_retrieval_configs: tuple[MemoryRetrievalConfig, ...] = StoreField(default_factory=lambda: (MemoryRetrievalConfig(
        retrievals=[
            MemoryRetrievalConfig.Retrieval(
                memory_type='original',
                k=1,
                depth=1,
                retrieve_method='similarity',
                similarity_weight=0.4,
                retrievability_weight=0.6,
            ),
            MemoryRetrievalConfig.Retrieval(
                memory_type='reflective',
                k=3,
                depth=1,
                similarity_weight=0.35,
                retrievability_weight=0.45,
                diversity_weight=0.2,
            ),
        ],
        stable_k=2,
        strength=0.4
    ),), title="被动检索配置")
    active_common_retrieval_configs: tuple[MemoryRetrievalConfig, ...] = StoreField(default_factory=lambda: (MemoryRetrievalConfig(
        retrievals=[
            MemoryRetrievalConfig.Retrieval(
                memory_type='reflective',
                k=6,
                similarity_weight=0.5,
                retrievability_weight=0.25,
                diversity_weight=0.25,
            ),
            MemoryRetrievalConfig.Retrieval(
                memory_type='summary',
                k=1,
                retrieve_method='similarity',
                similarity_weight=0.5,
                retrievability_weight=0.5
            ),
        ],
        stable_k=6
    ),), title="主动检索配置")
    active_episodic_retrieval_configs: tuple[MemoryRetrievalConfig, ...] = StoreField(default_factory=lambda: (MemoryRetrievalConfig(
        retrievals=[
            MemoryRetrievalConfig.Retrieval(
                memory_type='original',
                k=4,
                depth=3,
                similarity_weight=0.5,
                retrievability_weight=0.25,
                diversity_weight=0.25,
            ),
            MemoryRetrievalConfig.Retrieval(
                memory_type='summary',
                k=1,
                retrieve_method='similarity',
                similarity_weight=0.8,
                retrievability_weight=0.2,
            ),
        ],
        stable_k=4
    ),), title="主动情景记忆检索策略配置")
    active_semantic_retrieval_configs: tuple[MemoryRetrievalConfig, ...] = StoreField(default_factory=lambda: (MemoryRetrievalConfig(
        retrievals=[
            MemoryRetrievalConfig.Retrieval(
                memory_type='reflective',
                k=8,
                similarity_weight=0.6,
                retrievability_weight=0.15,
                diversity_weight=0.25,
            ),
        ],
        stable_k=6
    ),), title="主动语义记忆检索策略配置")


class MemoryData(StoreModel):
    _namespace = PLUGIN_NAME
    _title = "memory数据"
    last_added_memory_ids: dict[str, str] = StoreField(default_factory=dict, title='上次添加的记忆id')
    last_summarized_times: Optional[Times] = StoreField(default=None, title='上次总结过的时间')
    # {'year': 2023, 'month': 12, 'week': 52, 'day': 31, 'hour': 23, 'minute': 59, 'second': 59}
