from tomlkit import load, loads, dump, document, table, comment, nl, TOMLDocument
from tomlkit.items import Table
from become_human.utils import make_sure_path_exists
import os
from typing import Optional, Literal, Type, Union
from pydantic import BaseModel, Field
from warnings import warn
from copy import deepcopy

class MainConfig(BaseModel):
    role_prompt: str = Field(default="你是一个友善且富有同理心的助手，用简洁自然的语言为用户提供帮助。", description="角色提示词")
    active_time_range: tuple[float, float] = Field(default=(1800.0, 7200.0), description="活跃时间随机范围（最小值和最大值），在这之后进入休眠状态")
    temporary_active_time_range: tuple[float, float] = Field(default=(30.0, 600.0), description="在无新消息时self_call后agent获得的临时活跃时间的随机范围（最小值和最大值），单位为秒。")
    self_call_time_ranges: list[tuple[float, float]] = Field(default_factory=lambda: [
        (1800.0, 10800.0),
        (5400.0, 32400.0),
        (16200.0, 97200.0),
        (97200.0, 388800.0)
    ], description="在活跃状态之后的休眠状态时self_call时间随机范围（最小值和最大值），单位为秒，睡觉期间不算时间")
    wakeup_time_range: tuple[float, float] = Field(default=(1.0, 61201.0), description="在进入休眠状态后（也算作self_call），通过发送消息唤醒agent需要的时间随机范围（最小值和最大值），单位为秒")
    sleep_time_range: tuple[float, float] = Field(default=(79200.0, 18000.0), description="agent进入睡眠的时间段，单位为秒。目前的作用是self_call的时间生成会跳过这个时间段。")

class RecycleConfig(BaseModel):
    cleanup_on_non_active_recycle: bool = Field(default=False, description="是否在非活跃自动回收的同时清理回收的消息")
    recycle_trigger_threshold: int = Field(default=5000, ge=0.0, description="触发回收的阈值，单位为Tokens")
    recycle_target_size: int = Field(default=3000, ge=0.0, description="回收后目标大小，单位为Tokens")
    base_stable_time: float = Field(default=43200.0, ge=0.0, description="记忆初始化时stable_time的初始值，单位为秒。目前会乘以一个0~3的随机数")

class RetrieveMemoriesConfig(BaseModel):
    k: int = Field(default=6, ge=0, description="检索返回的记忆数量")
    fetch_k: int = Field(default=24, ge=0, description="从多少个结果中筛选出最终的结果，目前仅用于mmr")
    search_method: Literal['similarity', 'mmr'] = Field(default='mmr', description="检索方法：[similarity, mmr]")
    similarity_weight: float = Field(default=0.4, description="检索权重：相似性权重，范围[0,1]", ge=0.0, le=1.0)
    retrievability_weight: float = Field(default=0.3, description="检索权重：可访问性权重，范围[0,1]", ge=0.0, le=1.0)
    diversity_weight: float = Field(default=0.3, description="检索权重：多样性权重，范围[0,1]。只在检索方法为mmr生效", ge=0.0, le=1.0)
    strength: float = Field(default=1.0, description="检索强度，范围[0,1]，也可以超过1")

class RetrieveConfig(BaseModel):
    active_retrieve_config: RetrieveMemoriesConfig = Field(default_factory=lambda: RetrieveMemoriesConfig(
        k=8,
        fetch_k=24,
        search_method='mmr',
        similarity_weight=0.5,
        retrievability_weight=0.25,
        diversity_weight=0.25,
        strength=1.0
    ), description="主动检索配置")
    passive_retrieve_config: RetrieveMemoriesConfig = Field(default_factory=lambda: RetrieveMemoriesConfig(
        k=6,
        fetch_k=18,
        search_method='mmr',
        similarity_weight=0.3,
        retrievability_weight=0.5,
        diversity_weight=0.2,
        strength=0.5
    ), description="被动检索配置")

class ThreadConfig(BaseModel):
    init_on_startup: bool = Field(default=False, description="是否在程序启动时自动初始化该线程")
    main: MainConfig = Field(default_factory=lambda: MainConfig(), description="main_graph配置")
    recycle: RecycleConfig = Field(default_factory=lambda: RecycleConfig(), description="recycle_graph配置")
    retrieve: RetrieveConfig = Field(default_factory=lambda: RetrieveConfig(), description="retrieve_graph配置")


thread_configs_toml = document()
thread_configs: dict[str, ThreadConfig] = {}
default_thread_config = ThreadConfig()

make_sure_path_exists(config_path="./config")

THREADS_FILE = "./config/threads.toml"


default_thread_toml = table()
for key, value in ThreadConfig().model_dump().items():
    default_thread_toml.add(key, value)


def _add_field_comments(doc: TOMLDocument, model: Type[BaseModel], prefix: str = "") -> TOMLDocument:
    """递归地将模型字段的描述添加为TOML文档的注释"""
    for field_name, field_info in model.model_fields.items():
        # 如果字段类型是BaseModel，则递归处理
        if field_info.default_factory and isinstance(field_info.default_factory(), BaseModel):
            doc.add(nl())
            doc.add(comment(f'[thread_id.{prefix}{field_name}]{f': {field_info.description}' if field_info.description else ''}'))
            doc = _add_field_comments(doc, field_info.default_factory(), prefix+field_name+'.')
        else:
            if field_info.description:
                doc.add(comment(f'{field_name}: {field_info.description}'))
    return doc

def _add_config_comments(doc: TOMLDocument):
    doc.add(comment('配置说明'))
    doc.add(comment('[thread_id]: 会使用这个key来作为thread_id，它是唯一的'))

    # 添加字段描述
    doc = _add_field_comments(doc, ThreadConfig)

    doc.add(nl())
    return doc

def create_default_thread_configs_toml() -> TOMLDocument:
    doc = document()
    doc = _add_config_comments(doc)

    doc.add('default_thread', default_thread_toml)
    return doc


def verify_toml(override: TOMLDocument) -> TOMLDocument:
    default = deepcopy(default_thread_toml)
    doc = document()
    doc = _add_config_comments(doc)
    for key, value in override.items():
        doc.add(nl())
        doc.add(key, _merge_tomls(value, deepcopy(default)))
        doc.add(nl())
        doc.add(nl())
    return doc

def _merge_tomls(override: Table, default: Table) -> Table:
    for key, value in override.items():
        if key in default.keys() and isinstance(value, Table) and isinstance(default[key], Table):
            default[key] = loads(__merge_tomls(default[key], value).as_string().strip())
        elif key in default.keys():
            default[key] = value
    return default

def __merge_tomls(override: Table, default: Table) -> Table:
    for key, value in override.items():
        if key in default.keys() and isinstance(value, Table) and isinstance(default[key], Table):
            __merge_tomls(value, default[key])
            default[key] = loads(default[key].as_string().strip())
        elif key not in default.keys():
            default[key] = value
    return default

def load_config() -> dict[str, ThreadConfig]:
    global thread_configs_toml, thread_configs
    if not os.path.exists(THREADS_FILE):
        thread_configs_toml = create_default_thread_configs_toml()
    else:
        with open(THREADS_FILE, "r", encoding='utf-8') as f:
            thread_configs_toml = load(f)
            thread_configs_toml = verify_toml(thread_configs_toml)
    thread_configs = {key: ThreadConfig.model_validate(value) for key, value in thread_configs_toml.unwrap().items()}
    with open(THREADS_FILE, 'w', encoding='utf-8') as f:
        dump(thread_configs_toml, f)
    return thread_configs

def set_config(override: TOMLDocument) -> dict[str, ThreadConfig]:
    global thread_configs_toml, thread_configs
    thread_configs_toml = verify_toml(override)
    thread_configs = {key: ThreadConfig.model_validate(value) for key, value in thread_configs_toml.unwrap().items()}
    with open(THREADS_FILE, 'w', encoding='utf-8') as f:
        dump(thread_configs_toml, f)
    return thread_configs


load_config()


def get_thread_configs_toml() -> TOMLDocument:
    return thread_configs_toml

def get_thread_configs() -> dict[str, ThreadConfig]:
    return thread_configs

def get_thread_config(thread_id: str) -> ThreadConfig:
    thread_config = thread_configs.get(thread_id)
    if not thread_config:
        warn(f'Thread "{thread_id}" not found, using default thread config.')
        thread_config = default_thread_config
    return thread_config

def get_thread_main_config(thread_id: str) -> MainConfig:
    return get_thread_config(thread_id).main

def get_thread_recycle_config(thread_id: str) -> RecycleConfig:
    return get_thread_config(thread_id).recycle

def get_thread_retrieve_config(thread_id: str) -> RetrieveConfig:
    return get_thread_config(thread_id).retrieve
