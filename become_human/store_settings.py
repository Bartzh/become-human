from become_human.time import AgentTimeSettings
from become_human.store import StoreField, StoreModel

from pydantic import BaseModel, Field
from typing import Literal, Any, Optional, Union, Type
from datetime import date
from warnings import warn


class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="姓名")
    age: Optional[int] = Field(default=None, description="年龄")
    sex: Optional[str] = Field(default=None, description="性别")
    birthday: Optional[date] = Field(default=None, description="生日")

class MainSettings(StoreModel):
    _namespace = ('main',)
    _readable_name = "主要设置"
    role_prompt: str = StoreField(default="你是一个友善且富有同理心的助手，用简洁自然的语言为用户提供帮助。", readable_name="角色提示词")
    active_time_range: tuple[float, float] = StoreField(default=(1800.0, 7200.0), readable_name='活跃时长随机范围', description="活跃时间随机范围（最小值和最大值），在这之后进入休眠状态")
    temporary_active_time_range: tuple[float, float] = StoreField(default=(30.0, 600.0), readable_name='临时活跃时长随机范围', description="在无新消息时self_call后agent获得的临时活跃时间的随机范围（最小值和最大值），单位为秒。")
    self_call_time_ranges: list[tuple[float, float]] = StoreField(default_factory=lambda: [
        (1800.0, 10800.0),
        (5400.0, 32400.0),
        (16200.0, 97200.0),
        (97200.0, 388800.0)
    ], readable_name='休眠时自我调用时间随机范围', description="在活跃状态之后的休眠状态时self_call时间随机范围（最小值和最大值），单位为秒，睡觉期间不算时间")
    wakeup_time_range: tuple[float, float] = StoreField(default=(1.0, 10800.0), readable_name='苏醒随机时间范围', description="在进入休眠状态后（也算作self_call），通过发送消息唤醒agent需要的时间随机范围（最小值和最大值），单位为秒")
    sleep_time_range: tuple[float, float] = StoreField(default=(259200.0, 18000.0), readable_name='睡眠时间段', description="agent进入睡眠的时间段，单位为秒。目前的作用是self_call的时间生成会跳过这个时间段。")
    time_settings: AgentTimeSettings = StoreField(default_factory=AgentTimeSettings, readable_name="时间设置")
    character_settings: Person = StoreField(default_factory=Person, readable_name="角色设定")

class RecycleSettings(StoreModel):
    _namespace = ('recycle',)
    _readable_name = "回收设置"
    base_stable_time: float = StoreField(default=259200.0, readable_name='记忆稳定时长基值', description="记忆初始化时stable_time的初始值，单位为秒。目前会乘以一个0~3的随机数")
    recycle_trigger_threshold: int = StoreField(default=10000, readable_name='溢出回收阈值', description="触发溢出回收的阈值，单位为Tokens")
    recycle_target_size: int = StoreField(default=6000, readable_name='溢出回收目标大小', description="溢出回收后目标大小，单位为Tokens")
    cleanup_on_non_active_recycle: bool = StoreField(default=False, readable_name='非活跃回收时清理', description="是否在非活跃自动回收的同时清理回收的消息")
    cleanup_target_size: int = StoreField(default=1000, readable_name='非活跃清理目标大小', description="非活跃清理后目标大小，单位为Tokens")

class RetrieveMemoriesConfig(BaseModel):
    k: int = Field(default=30, ge=0, description="检索返回的记忆数量")
    fetch_k: int = Field(default=90, ge=0, description="从多少个结果中筛选出最终的结果，目前仅用于mmr")
    stable_k: int = Field(default=10, ge=0, description="最终显示几个完整的记忆，剩下的记忆会被简略化")
    original_ratio: float = Field(default=2.0, description="检索结果中原始记忆出现的初始比例", ge=0.0)
    summary_ratio: float = Field(default=3.0, description="检索结果中摘要记忆出现的初始比例", ge=0.0)
    semantic_ratio: float = Field(default=6.0, description="检索结果中语义记忆出现的初始比例", ge=0.0)
    search_method: Literal['similarity', 'mmr'] = Field(default='mmr', description="检索排序算法：[similarity, mmr]")
    similarity_weight: float = Field(default=0.5, description="检索权重：相似性权重，范围[0,1]", ge=0.0, le=1.0)
    retrievability_weight: float = Field(default=0.25, description="检索权重：可访问性权重，范围[0,1]", ge=0.0, le=1.0)
    diversity_weight: float = Field(default=0.25, description="检索权重：多样性权重，范围[0,1]。只在检索方法为mmr时生效", ge=0.0, le=1.0)
    strength: float = Field(default=1.0, description="检索强度，作为倍数将乘以被检索记忆的可检索性、稳定时长与难易度的提升幅度，范围[0,1]，也可以超过1")

class RetrieveSettings(StoreModel):
    _namespace = ('retrieve',)
    _readable_name = "检索设置"
    active_retrieve_config: RetrieveMemoriesConfig = StoreField(default_factory=RetrieveMemoriesConfig, readable_name="主动检索配置")
    passive_retrieve_config: RetrieveMemoriesConfig = StoreField(default_factory=lambda: RetrieveMemoriesConfig(
        k=18,
        fetch_k=54,
        stable_k=6,
        original_ratio=3.0,
        summary_ratio=3.0,
        semantic_ratio=5.0,
        similarity_weight=0.3,
        retrievability_weight=0.5,
        diversity_weight=0.2,
        strength=0.5
    ), readable_name="被动检索配置")

class ThreadSettings(StoreModel):
    _namespace = ('settings',)
    _readable_name = "线程设置"
    main: MainSettings
    recycle: RecycleSettings
    retrieve: RetrieveSettings



def parse_character_settings(source: dict, indent: int = 4, prefix: str = '',) -> str:
    def _parse_character_setting(model: Type[BaseModel], source: dict) -> dict:
        character_settings = {}
        for key, value in source.items():
            if value is None:
                continue
            if key in model.model_fields.keys():
                if model.model_fields[key].description:
                    cs_key = model.model_fields[key].description
                else:
                    cs_key = key
                if isinstance(value, dict):
                    if issubclass(model.model_fields[key].annotation, BaseModel):
                        character_settings[cs_key] = _parse_character_setting(model.model_fields[key].annotation, value)
                    else:
                        warn(f"{key} can not be a dict.")
                elif isinstance(value, model.model_fields[key].annotation):
                    character_settings[cs_key] = value
                else:
                    warn(f"{key} can not be a {str(model.model_fields[key].annotation)}.")
            elif key not in character_settings.keys():
                character_settings[key] = value
        return character_settings
    character_settings = _parse_character_setting(Person, source)
    def _dict_to_readable_string(d: dict, plus: int = 4, prefix: str = '', indent=0):
        result = ""
        for key, value in d.items():
            if isinstance(value, dict):
                result += " " * indent + prefix + f"{key}:\n"
                result += _dict_to_readable_string(value, plus, prefix, indent + plus)
            else:
                result += " " * indent + prefix + f"{key}: {value}\n"
        return result.strip()
    return _dict_to_readable_string(character_settings, indent, prefix)