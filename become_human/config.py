from tomlkit import load, loads, dump, document, table, comment, nl, TOMLDocument
from tomlkit.items import Table
from become_human.utils import make_sure_path_exists, dump_basemodels, get_readable_type_name
from become_human.store import StoreModel, StoreField, store_alist_namespaces, store_abatch
from become_human.store_settings import AgentSettings
from langgraph.store.base import PutOp
import os
from warnings import warn
from typing import Optional, Literal, Type, Union, get_type_hints, Any
from pydantic import BaseModel, TypeAdapter
from pydantic_core import ValidationError

make_sure_path_exists()

AGENTS_FILE_PATH = "./config/agents.toml"
AGENT_COMMENTS_FILE_PATH = "./config/agent_comments.toml"

agent_configs: dict[str, dict[str, Any]] = {}

DEFAULT_AGENTS = {
    'default_agent_1': {
        'main': {
            'role_prompt': '你是用户的私人助手，用户是你的主人，你也会如此称呼ta。你对主人忠诚且富有同理心，用简洁自然的语言为主人提供帮助就是你唯一想做的事。你与主人是通过一个无聊天记录（阅后即焚）的即时通讯软件远程交流的。',
            'instruction_prompt': '你接下来见到的第一个人就是你的主人，向ta打个招呼吧。'
        }
    },
    'default_agent_2': {
        'main': {
            'role_prompt': '''你是一个待人较为冷淡的人，对什么事情都无所谓，比较懒，说话也很简短、敷衍。
他人是一个无聊天记录（阅后即焚）的即时通讯软件联系到你的。
你不喜欢在每句话的结尾加上句号。''',
            'active_time_range': (120.0, 1800.0),
            'self_call_time_ranges': [(300.0, 10800.0)],
            'wakeup_time_range': (1.0, 61201.0),
            'sleep_time_range': (82800.0, 32400.0)
        },
        'recycling': {
            'base_stable_time': 43200.0,
            'cleanup_on_non_active_recycling': True,
            'cleanup_target_size': 800
        },
        'retrieval': {
            'active_retrieval_config': {
                'similarity_weight': 0.4,
                'retrievability_weight': 0.35,
                'diversity_weight': 0.25
            }
        }
    },
    'default_agent_3': {
        'main': {
            'role_prompt': '''你是一个调试agent，没有什么角色需要你扮演（或者说这就是你的角色），只需遵守行为准则（但依然需要输出心理活动），你的用户就是你的开发者。
这样的设定是为了辅助开发者调试/测试你自己的agent程序，如果你在你的上下文中发现了错误或是有什么异常，不对劲的地方，又或是某些prompt表述不够完美有歧义，请主动将其告知用户。''',
            'instruction_prompt': '你接下来见到的第一个人就是你的开发者。',
            'always_active': True,
        }
    }
}


def to_toml_like_string(a: Any) -> str:
    """将任意对象转换为TOML-like字符串

    具体来说，实现了对字符串、布尔值、None、元组、列表的转换"""
    if isinstance(a, str):
        if '\n' in a:
            return f'"""{a}"""'
        return f'"{a}"'
    elif isinstance(a, bool):
        return str(a).lower()
    elif a is None:
        return 'null'
    elif isinstance(a, (tuple, list)):
        return '[' + ', '.join([to_toml_like_string(i) for i in a]) + ']'
    else:
        return str(a)


def multi_line_comment(doc: TOMLDocument, text: str) -> None:
    """
    处理多行字符串的注释，确保每一行都被正确注释
    """
    lines = text.split('\n')
    for line in lines:
        doc.add(comment(line))


def _add_field_comments(doc: TOMLDocument, model: Type[Union[StoreModel, BaseModel]], prefix: str = "") -> TOMLDocument:
    """递归地将模型字段的描述添加为TOML文档的注释"""
    if issubclass(model, StoreModel):
        hints = get_type_hints(model)
        for key, hint_type in hints.items():
            if isinstance(hint_type, type) and issubclass(hint_type, (StoreModel, BaseModel)):
                if issubclass(hint_type, StoreModel):
                    desc = hint_type._readable_name if hint_type._readable_name else ''
                    if desc and hint_type._description:
                        desc += ': ' + hint_type._description
                    else:
                        desc += hint_type._description if hint_type._description else ''
                else:
                    field = model.__dict__.get(key)
                    if field is not None and isinstance(field, StoreField):
                        desc = field.readable_name if field.readable_name else ''
                        if field.readable_name and field.description:
                            desc += '：' + field.description
                        else:
                            desc += field.description if field.description else ''
                    else:
                        continue
                desc = f'<{get_readable_type_name(hint_type)}> ' + desc
                doc.add(nl())
                doc.add(nl())
                multi_line_comment(doc, desc)
                multi_line_comment(doc, f'[agent_id.{prefix}{key}]')
                doc = _add_field_comments(doc, hint_type, prefix+key+'.')
            else:
                field = model.__dict__.get(key)
                if field is not None and isinstance(field, StoreField):
                    doc.add(nl())
                    desc = f'<{get_readable_type_name(hint_type)}> '
                    desc += field.readable_name if field.readable_name else ''
                    if field.readable_name and field.description:
                        desc += '：' + field.description
                    else:
                        desc += field.description if field.description else ''
                    multi_line_comment(doc, desc)
                    default = model.get_field_default(field)
                    multi_line_comment(doc, f'{key}{" = " + to_toml_like_string(default)}')
    else:
        for field_name, field_info in model.model_fields.items():
            desc = f'<{get_readable_type_name(field_info.annotation)}> '
            desc += field_info.description if field_info.description else ''
            if isinstance(field_info.annotation, type) and issubclass(field_info.annotation, (StoreModel, BaseModel)):
                doc.add(nl())
                doc.add(nl())
                multi_line_comment(doc, desc)
                multi_line_comment(doc, f'[agent_id.{prefix}{field_name}]')
                doc = _add_field_comments(doc, field_info.annotation, prefix+field_name+'.')
            else:
                doc.add(nl())
                multi_line_comment(doc, desc)
                if field_info.default_factory is not None:
                    v = field_info.default_factory()
                elif field_info.default is not None:
                    v = field_info.default
                else:
                    v = None
                multi_line_comment(doc, f'{field_name}{" = " + to_toml_like_string(v)}')
    return doc

def _add_config_comments(doc: TOMLDocument):
    multi_line_comment(doc, '类型说明')
    multi_line_comment(doc, '<>描述了该字段的类型，写入数据库时会使用pydantic的类型转换功能尝试转换至目标类型')
    multi_line_comment(doc, '意味着如str, int, float, bool等类型，会尝试自动转换为对应的类型，但不建议依赖类型转换功能')
    doc.add(nl())
    doc.add(nl())
    multi_line_comment(doc, '配置说明')
    multi_line_comment(doc, '<str> agentID: 会使用这个key来作为agent_id，它是唯一的')
    multi_line_comment(doc, '[agent_id]')
    multi_line_comment(doc, '<bool> 启动时初始化: 是否在程序启动时自动初始化该agent')
    multi_line_comment(doc, 'init_on_startup = false')

    # 添加字段描述
    doc = _add_field_comments(doc, AgentSettings)

    #doc.add(nl())
    return doc

def _add_config_agent_comments(doc: TOMLDocument, prefix: str, config: dict):
    multi_line_comment(doc, f'[{prefix}]')
    for key, value in config.items():
        if isinstance(value, dict):
            doc.add(nl())
            doc = _add_config_agent_comments(doc, prefix+'.'+key, value)
        else:
            multi_line_comment(doc, f'{key} = {to_toml_like_string(value)}')
    return doc

def create_default_agent_configs_toml() -> TOMLDocument:
    doc = document()
    multi_line_comment(doc, 'agent配置')
    multi_line_comment(doc, '查阅 agent_comments.toml 获取agent配置的详细说明及默认值')
    multi_line_comment(doc, '将值设为null表示从数据库中删除该字段（若有），保持默认值')
    for agent_id, agent_config in DEFAULT_AGENTS.items():
        doc.add(nl())
        doc.add(nl())
        doc = _add_config_agent_comments(doc, agent_id, agent_config)
    doc.add(nl())
    return doc


async def load_config(agent_ids: Optional[Union[list[str], str]] = None, force: bool = False) -> dict[str, dict[str, Any]]:
    """载入config。需要先初始化store！"""
    global agent_configs
    update_agent_comments()
    if not os.path.exists(AGENTS_FILE_PATH):
        agent_configs_toml = create_default_agent_configs_toml()
        with open(AGENTS_FILE_PATH, 'w', encoding='utf-8') as f:
            dump(agent_configs_toml, f)
    else:
        with open(AGENTS_FILE_PATH, "r", encoding='utf-8') as f:
            agent_configs_toml = load(f)
    agent_configs = agent_configs_toml.unwrap()
    if agent_ids:
        if isinstance(agent_ids, str):
            agent_ids = [agent_ids]
        agent_configs = {k: v for k, v in agent_configs.items() if k in agent_ids}
    if not agent_configs:
        return {}
    if not force:
        has_settings_namespaces = await store_alist_namespaces(prefix=('agents', '*', 'model', 'settings'), max_depth=4)
        has_settings_agent_ids = [n[1] for n in has_settings_namespaces]
    else:
        has_settings_agent_ids = []

    def _write_config_to_store(d: dict, namespace: tuple[str, ...], model: Type[StoreModel]):
        put_ops = []
        hints = get_type_hints(model)
        for key, value in d.items():
            hint_type = hints.get(key)
            if hint_type is not None:
                if isinstance(hint_type, type) and issubclass(hint_type, StoreModel):
                    if isinstance(value, dict):
                        put_ops.extend(_write_config_to_store(value, namespace+(key,), hint_type))
                    else:
                        warn(f"Invalid value for {key} in config file: expected dict for StoreModel, got {type(value)}")
                else:
                    # 如果值是None，则视为删除数据
                    if value is None:
                        put_ops.append(PutOp(namespace=namespace, key=key, value=None))
                        continue
                    adapter = TypeAdapter(hint_type)
                    try:
                        validated_value = adapter.validate_python(value)
                    except ValidationError as e:
                        warn(f"Invalid value for {key} in config file: {e}")
                        continue
                    if isinstance(validated_value, BaseModel):
                        validated_value = validated_value.model_dump(exclude_unset=True)
                    elif isinstance(validated_value, (list, tuple, dict, set)):
                        validated_value = dump_basemodels(validated_value)
                    put_ops.append(PutOp(namespace=namespace, key=key, value={'value': validated_value}))
        return put_ops

    ops = []
    for key, value in agent_configs.items():
        if key not in has_settings_agent_ids:
            if isinstance(value, dict):
                ops.extend(_write_config_to_store(value, ('agents', key, 'model', 'settings'), AgentSettings))
    if ops:
        await store_abatch(ops)

    return agent_configs

def update_agent_comments():
    doc = document()
    doc = _add_config_comments(doc)
    with open(AGENT_COMMENTS_FILE_PATH, "w", encoding='utf-8') as f:
        dump(doc, f)

def get_agent_configs() -> dict[str, dict[str, Any]]:
    return agent_configs

def get_agent_config(agent_id: str) -> dict[str, Any]:
    return agent_configs.get(agent_id, {})
