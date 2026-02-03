from pydantic import BaseModel, Field
from typing import Literal, Any, Optional, Union, Type
from datetime import datetime, date

from become_human.times import AgentTimeSettings, seconds_to_datetime
from become_human.store.base import StoreField, StoreModel
from become_human.message import InitalAIMessage


# 新想法，通过结构化然后让AI生成所有没有指定的细节
class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="姓名")
    age: Optional[int] = Field(default=None, description="年龄")
    sex: Optional[str] = Field(default=None, description="性别")
    birthday: Optional[date] = Field(default=None, description="生日")
    additional_info: Optional[dict[str, str]] = Field(default=None, description="其他信息")

class MainSettings(StoreModel):
    _namespace = ('main',)
    _readable_name = "主要设置"
    role_prompt: str = StoreField(default="你是一个对陌生人也抱有基本尊重的普通人。你与他人是通过一个无聊天记录（阅后即焚）的即时通讯软件远程交流的。", readable_name="角色提示词")
    instruction_prompt: str = StoreField(default="打个招呼吧。", readable_name="引导提示词", description="作为agent的第一条用户消息出现，对agent进行引导。")
    initial_ai_messages: list[InitalAIMessage] = StoreField(default_factory=list, readable_name="初始AI消息（列表）", description="初始AI消息，作为instruction_prompt的回复，不是必须的。会在列表中随机选择一条")
    react_instruction: bool = StoreField(default=False, readable_name="反应引导", description="是否以instruction_prompt调用agent，这会覆盖initial_ai_messages。")
    role_description: str = StoreField(default="应该是一个有用的助手吧。", readable_name="展示用角色描述", description="直接向用户显示的一段文本，描述这个角色")
    active_time_range: tuple[float, float] = StoreField(default=(1800.0, 7200.0), readable_name='活跃时长随机范围', description="活跃时间随机范围（最小值和最大值），在这之后进入休眠状态")
    always_active: bool = StoreField(default=False, readable_name="保持活跃", description="是否一直处于活跃状态，也即不存在agent因不活跃而不回复消息的情况。若是，则active_time_range将仅用作回收消息等功能，且self_call依然有效，只有wakeup_call会失效")
    temporary_active_time_range: tuple[float, float] = StoreField(default=(180.0, 1800.0), readable_name='临时活跃时长随机范围', description="在无新消息时self_call后agent获得的临时活跃时间的随机范围（最小值和最大值），单位为秒。")
    self_call_time_ranges: list[tuple[float, float]] = StoreField(default_factory=lambda: [
        (1800.0, 32400.0),
        (16200.0, 97200.0),
        (97200.0, 388800.0)
    ], readable_name='休眠时自我调用时间随机范围', description="在活跃状态之后的休眠状态时self_call时间随机范围（最小值和最大值），单位为秒，睡觉期间不算时间")
    wakeup_time_range: Union[tuple[float, float], tuple[()]] = StoreField(default=(1.0, 10800.0), readable_name='苏醒随机时间范围', description="在进入休眠状态后（也算作self_call），通过发送消息唤醒agent需要的时间随机范围（最小值和最大值），单位为秒")
    sleep_time_range: Union[tuple[float, float], tuple[()]] = StoreField(default=(79200.0, 18000.0), readable_name='睡眠时间段', description="agent进入睡眠的时间段，单位为秒。目前的作用是self_call的时间生成会跳过这个时间段。")
    time_settings: AgentTimeSettings = StoreField(default_factory=AgentTimeSettings, readable_name="时间设置")
    character_settings: Person = StoreField(default_factory=Person, readable_name="角色设定")

    def format_character_settings(self, indent: int = 4, prefix: str = '- ',) -> str:
        def _format_character_setting(model: Type[BaseModel], source: dict) -> dict:
            character_settings = {}
            for key, value in source.items():
                if value is None:
                    continue
                if key in model.model_fields.keys():
                    if model.model_fields[key].description:
                        cs_key = model.model_fields[key].description
                    else:
                        cs_key = key
                    if isinstance(value, dict) and issubclass(model.model_fields[key].annotation, BaseModel):
                        character_settings[cs_key] = _format_character_setting(model.model_fields[key].annotation, value)
                    else:
                        character_settings[cs_key] = value
                elif key not in character_settings.keys():
                    character_settings[key] = value
            return character_settings
        person = self.character_settings
        character_settings = _format_character_setting(Person, person.model_dump())
        # always_active的话就当它不会睡觉了
        if self.sleep_time_range and not self.always_active:
            character_settings["睡觉时间段"] = f"{seconds_to_datetime(self.sleep_time_range[0]).time()} ~ {seconds_to_datetime(self.sleep_time_range[1]).time()}"
        def _dict_to_readable_string(d: dict, plus: int = 4, prefix: str = '- ', indent=0):
            result = ""
            for key, value in d.items():
                if isinstance(value, dict):
                    result += " " * indent + prefix + f"{key}:\n"
                    result += _dict_to_readable_string(value, plus, prefix, indent + plus)
                else:
                    result += " " * indent + prefix + f"{key}: {value}\n"
            return result.strip()
        return _dict_to_readable_string(character_settings, indent, prefix)

class RecyclingSettings(StoreModel):
    _namespace = ('recycling',)
    _readable_name = "回收设置"
    memory_base_stable_duration_ticks: int = StoreField(default=259200_000_000, readable_name='记忆稳定时长基值', description="记忆初始化时stable_duration_ticks的初始值")
    memory_max_words: int = StoreField(default=300, readable_name='记忆最大Tokens数', description="单条记忆最大单词数，决定记忆难度，最大难度0.8")
    recycling_trigger_threshold: int = StoreField(default=24000, readable_name='溢出回收阈值', description="触发溢出回收的阈值，单位为Token")
    recycling_target_size: int = StoreField(default=18000, readable_name='溢出回收目标大小', description="溢出回收后目标大小，单位为Token")
    cleanup_on_non_active_recycling: bool = StoreField(default=False, readable_name='非活跃回收时清理', description="是否在非活跃自动回收的同时清理回收的消息")
    cleanup_target_size: int = StoreField(default=2000, readable_name='非活跃清理目标大小', description="非活跃清理后目标大小，单位为Token")

class MemoryRetrievalConfig(BaseModel):
    k: int = Field(default=16, ge=0, description="检索返回的记忆数量")
    fetch_k: int = Field(default=250, ge=0, description="从多少个结果中筛选出最终的结果")
    stable_k: int = Field(default=10, ge=0, description="最终显示几个完整的记忆，剩下的记忆会被简略化")
    depth: int = Field(default=2, ge=0, description="检索的记忆深度，会在0之间随机取值，指被检索记忆的相邻n个记忆也会被召回")
    original_ratio: float = Field(default=2.0, description="检索结果中原始记忆出现的初始比例", ge=0.0)
    episodic_ratio: float = Field(default=4.0, description="检索结果中情景记忆出现的初始比例", ge=0.0)
    reflective_ratio: float = Field(default=5.0, description="检索结果中反思记忆出现的初始比例", ge=0.0)
    search_method: Literal['similarity', 'mmr'] = Field(default='mmr', description="检索排序算法：[similarity, mmr]")
    similarity_weight: float = Field(default=0.5, description="检索权重：相似性权重，范围[0,1]，总和需为1", ge=0.0, le=1.0)
    retrievability_weight: float = Field(default=0.25, description="检索权重：可访问性权重，范围[0,1]，总和需为1", ge=0.0, le=1.0)
    diversity_weight: float = Field(default=0.25, description="检索权重：多样性权重，范围[0,1]。只在检索方法为mmr时生效，总和需为1", ge=0.0, le=1.0)
    strength: float = Field(default=1.0, description="检索强度，作为倍数将乘以被检索记忆的可检索性、稳定时长与难易度的提升幅度，范围[0,1]，作为主动检索时固定为1")

class RetrievalSettings(StoreModel):
    _namespace = ('retrieval',)
    _readable_name = "检索设置"
    active_retrieval_config: MemoryRetrievalConfig = StoreField(default_factory=MemoryRetrievalConfig, readable_name="主动检索配置")
    passive_retrieval_config: MemoryRetrievalConfig = StoreField(default_factory=lambda: MemoryRetrievalConfig(
        k=8,
        fetch_k=150,
        stable_k=5,
        depth=1,
        original_ratio=2.5,
        episodic_ratio=4.0,
        reflective_ratio=4.5,
        similarity_weight=0.35,
        retrievability_weight=0.45,
        diversity_weight=0.2,
        strength=0.4
    ), readable_name="被动检索配置")
    passive_retrieval_ttl: int = StoreField(default=3600_000_000, readable_name='被动检索存活时长', description="被动检索消息的存活时长，按agent主观ticks计算，单位为秒，到点后会被自动清理，设为0则不清理")

class BuiltinSettings(StoreModel):
    _namespace = ('settings',)
    _readable_name = "builtin设置"
    main: MainSettings
    recycling: RecyclingSettings
    retrieval: RetrievalSettings
