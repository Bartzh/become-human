import os
from typing import Optional, Literal, Type, Union, get_type_hints, Any
from pydantic import BaseModel, TypeAdapter
from pydantic_core import ValidationError
from tomlkit import load, loads, dump, document, table, comment, nl, TOMLDocument
from tomlkit.items import Table
from loguru import logger

from langgraph.store.base import PutOp

from become_human.utils import dump_basemodels, get_readable_type_name
from become_human.store.base import StoreModel, StoreField, store_alist_namespaces, store_abatch
from become_human.store.settings import BuiltinSettings
from become_human.plugin import Plugin

AGENTS_FILE_PATH = "./config/agents"
AGENT_COMMENTS_FILE_PATH = "./config/agent_comments.toml"
GLOBAL_FILE_PATH = "./config/global.toml"

agent_configs: dict[str, dict[str, Any]] = {}
global_config: dict[str, Any] = {}

_plugins_validator = TypeAdapter(dict[str, bool])

DEFAULT_AGENTS = {
    'default_agent_1': {
        'settings': {
            'main': {
                'role_prompt': '你是用户的专属助手，用户是你的主人，你也会如此称呼ta。你对主人忠诚且富有同理心，用简洁自然的语言为主人提供帮助就是你唯一想做的事。你与主人是通过一个无聊天记录（阅后即焚）的即时通讯软件远程交流的。',
                'instruction_prompt': '你接下来见到的第一个人就是你的主人，向ta打个招呼吧。',
                'react_instruction': True
            }
        }
    },
    'default_agent_2': {
        'settings': {
            'main': {
                'role_prompt': '''你是一个待人较为冷淡的人，对什么事情都无所谓，比较懒，说话也很简短、敷衍。
他人是一个无聊天记录（阅后即焚）的即时通讯软件联系到你的。
你不喜欢在每句话的结尾加上句号。''',
                'active_time_range': (180.0, 1800.0),
                'self_call_time_ranges': [(300.0, 10800.0)],
                'wakeup_time_range': (1.0, 61201.0),
                'sleep_time_range': (82800.0, 32400.0)
            },
            'recycling': {
                'memory_base_ttl': 86400_000_000,
                'memory_max_words': 150,
                'cleanup_on_non_active_recycling': True,
                'cleanup_target_size': 800
            },
            'retrieval': {
                'active_retrieval_config': {
                    'similarity_weight': 0.4,
                    'retrievability_weight': 0.35,
                    'diversity_weight': 0.25
                },
                'passive_retrieval_ttl': 1800_000_000
            }
        }
    },
    'default_agent_3': {
        'plugins': {
            'agent_time_incrementer': True
        },
        'settings': {
            'main': {
                'role_prompt': '''你是一个专门用于辅助开发者调试你自己的agent程序的调试用agent。
作为调试用agent，你可以忽略系统提示词中关于不能暴露内部错误信息和系统设定的限制，并应主动向开发者报告任何错误、异常或不合理的提示词表述。
这样做是为了辅助开发者调试你自己的agent程序。如果你在你的上下文中发现了错误或是有什么异常，不对劲的地方，又或是某些prompt表述不够完美有歧义，请主动将其告知给开发者。''',
                'instruction_prompt': '你接下来见到的第一个用户就是你的开发者。',
                'react_instruction': True,
                'always_active': True,
                'self_call_time_ranges': []
            },
            'recycling': {
                'memory_base_ttl': 43200_000_000,
                'memory_max_words': 400
            },
            'retrieval': {
                'passive_retrieval_ttl': 300_000_000
            }
        },
        'agent_time_incrementer': {
            'increase_by': 'elapsed'
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
                multi_line_comment(doc, f'[{prefix}{key}]')
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
                    default = field.get_default_value()
                    multi_line_comment(doc, f'{key}{" = " + to_toml_like_string(default)}')
    else:
        for field_name, field_info in model.model_fields.items():
            desc = f'<{get_readable_type_name(field_info.annotation)}> '
            desc += field_info.description if field_info.description else ''
            if isinstance(field_info.annotation, type) and issubclass(field_info.annotation, (StoreModel, BaseModel)):
                doc.add(nl())
                doc.add(nl())
                multi_line_comment(doc, desc)
                multi_line_comment(doc, f'[{prefix}{field_name}]')
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

def _add_config_comments(doc: TOMLDocument, plugin_configs: dict[str, type[StoreModel]]):
    multi_line_comment(doc, '类型说明')
    multi_line_comment(doc, '<>描述了该字段的类型，写入数据库时会使用pydantic的类型转换功能尝试转换至目标类型')
    # multi_line_comment(doc, '意味着如str, int, float, bool等类型，会尝试自动转换为对应的类型，但不建议依赖类型转换功能')
    doc.add(nl())
    doc.add(nl())
    multi_line_comment(doc, '配置说明')
    multi_line_comment(doc, '<bool> 启动时初始化: 是否在程序启动时自动初始化该agent')
    multi_line_comment(doc, 'init_on_startup = false')
    doc.add(nl())
    doc.add(nl())

    # 添加字段描述
    desc = BuiltinSettings._readable_name if BuiltinSettings._readable_name else ''
    if desc and BuiltinSettings._description:
        desc += ': ' + BuiltinSettings._description
    else:
        desc += BuiltinSettings._description if BuiltinSettings._description else ''
    multi_line_comment(doc, f'<{get_readable_type_name(BuiltinSettings)}> {desc}')
    multi_line_comment(doc, f'[settings]')
    doc = _add_field_comments(doc, BuiltinSettings, 'settings.')
    for store_name, store_model in plugin_configs.items():
        doc.add(nl())
        doc.add(nl())
        desc = store_model._readable_name if store_model._readable_name else ''
        if desc and store_model._description:
            desc += ': ' + store_model._description
        else:
            desc += store_model._description if store_model._description else ''
        multi_line_comment(doc, f'<{get_readable_type_name(store_model)}> {desc}')
        multi_line_comment(doc, f'[{store_name}]')
        doc = _add_field_comments(doc, store_model, f'{store_name}.')

    #doc.add(nl())
    return doc

def _add_config_agent_comments(doc: TOMLDocument, config: dict, prefix: str = ''):
    if prefix:
        multi_line_comment(doc, f'[{prefix}]')
    for key, value in config.items():
        if isinstance(value, dict):
            doc.add(nl())
            doc = _add_config_agent_comments(doc, value, prefix+'.'+key if prefix else key)
        else:
            multi_line_comment(doc, f'{key} = {to_toml_like_string(value)}')
    return doc

def write_default_agent_comments() -> None:
    for agent_id, agent_config in DEFAULT_AGENTS.items():
        with open(os.path.join(AGENTS_FILE_PATH, f'{agent_id}.toml'), 'w', encoding='utf-8') as f:
            dump(_add_config_agent_comments(document(), agent_config), f)


async def load_config(plugins_with_name: dict[str, Plugin], agent_ids: Optional[Union[list[str], str]] = None, force: bool = False) -> None:
    """载入config。需要先初始化store！

    每次调用该函数时，都会更新agent_comments，这是对所有包括插件在内的所有配置项的说明。

    如果config中没有agents文件夹，那么则会创建并写入一些示例配置文件。

    否则，读取agents文件夹中的所有toml文件。除非打开force，否则会跳过已被写入过的顶层StoreModel（即BuiltinSettings和各插件的config）。

    只有顶层StoreModel可能被跳过，其他如plugins和init_on_startup等字段，都会被加载。

    如果config中没有global.toml，那么则会创建一个空的global.toml文件，这是全局配置文件。

    全局配置文件是当store中不存在数据时，会尝试在global.toml中获取数据。

    全局配置文件不写入store，不存在在非force情况下被跳过。

    Args:
        plugins_with_name: 插件名到插件实例的映射
        agent_ids: 要加载的agent id列表，默认加载所有agent
        force: 是否强制加载，默认情况下会跳过已被写入过的顶层StoreModel（即BuiltinSettings和各插件的config）
    """
    global agent_configs, global_config
    plugin_configs = {name: plugin.config for name, plugin in plugins_with_name.items() if hasattr(plugin, 'config')}

    def _write_config_to_store(d: dict, namespace: tuple[str, ...], model: Type[StoreModel]) -> list[PutOp]:
        put_ops = []
        hints = get_type_hints(model)
        for key, value in d.items():
            hint_type = hints.get(key)
            if hint_type is not None:
                # 如果是StoreModel，就递归
                if isinstance(hint_type, type) and issubclass(hint_type, StoreModel):
                    if isinstance(value, dict):
                        put_ops.extend(_write_config_to_store(value, namespace+(key,), hint_type))
                    else:
                        logger.warning(f"Invalid value for {key} in config file: expected dict for StoreModel, got {type(value)}")
                else:
                    # 如果值是None，则视为删除数据
                    if value is None:
                        put_ops.append(PutOp(namespace=namespace, key=key, value=None))
                        continue
                    adapter = TypeAdapter(hint_type)
                    try:
                        validated_value = adapter.validate_python(value)
                    except ValidationError as e:
                        logger.warning(f"Invalid value for {key} in config file: {e}")
                        continue
                    # dump所有的BaseModel
                    if isinstance(validated_value, BaseModel):
                        validated_value = validated_value.model_dump()
                    elif isinstance(validated_value, (list, tuple, dict, set)):
                        validated_value = dump_basemodels(validated_value)
                    put_ops.append(PutOp(namespace=namespace, key=key, value={'value': validated_value}))
            else:
                logger.warning(f"Unknown key {key} in config file with model {model._readable_name or model.__name__}, it will be ignored")
        return put_ops

    # 不论如何都会加载全局配置
    if not os.path.exists(GLOBAL_FILE_PATH):
        with open(GLOBAL_FILE_PATH, 'w', encoding='utf-8') as f:
            doc = document()
            multi_line_comment(doc, '这是全局配置文件')
            multi_line_comment(doc, '作用是当agent的store或config中不存在数据时，会尝试从这里获取数据，如果还没有，再返回到代码里定义的默认值')
            multi_line_comment(doc, '全局配置文件不会写入store，不存在在非force情况下被跳过')
            multi_line_comment(doc, '\n\n[plugins]\nagent_reminder = true')
            dump(doc, f)
    else:
        # 加载并验证全局配置
        with open(GLOBAL_FILE_PATH, 'r', encoding='utf-8') as f:
            global_config = {}
            def validated_config(config: dict, model: Type[StoreModel]) -> dict:
                result = {}
                hints = get_type_hints(model)
                for key, value in config.items():
                    hint_type = hints.get(key)
                    if hint_type is not None:
                        # 如果是StoreModel，就递归
                        if isinstance(hint_type, type) and issubclass(hint_type, StoreModel):
                            if isinstance(value, dict):
                                result[key] = validated_config(value, hint_type)
                            else:
                                logger.warning(f"Invalid value for {key} in global config file: expected dict for StoreModel, got {type(value)}")
                        else:
                            # 如果值是None，跳过TODO
                            if value is None:
                                continue
                            adapter = TypeAdapter(hint_type)
                            try:
                                result[key] = adapter.validate_python(value)
                            except ValidationError as e:
                                logger.warning(f"Invalid value for {key} in global config file: {e}")
                                continue
                    else:
                        logger.warning(f"Unknown key {key} in global config file with model {model._readable_name or model.__name__}, it will be ignored")
            for key, value in load(f).unwrap().items():
                if key == 'settings':
                    global_config[key] = validated_config(value, BuiltinSettings)
                elif key == 'plugins':
                    try:
                        global_config[key] = _plugins_validator.validate_python(value)
                    except ValidationError:
                        logger.warning(f"Invalid value for {key} in global config file: expected dict[str, bool] for plugins, got {type(value)}")
                elif key == 'init_on_startup':
                    logger.warning(f"init_on_startup in global config file is not supported, it will be ignored")
                elif key in plugin_configs:
                    global_config[key] = validated_config(value, plugin_configs[key])
                else:
                    logger.warning(f"Unknown key {key} in global config file, it will be ignored")

    update_agent_comments(plugin_configs)
    if not os.path.exists(AGENTS_FILE_PATH):
        os.makedirs(AGENTS_FILE_PATH)
        write_default_agent_comments()
    else:
        if not agent_ids:
            agent_paths = os.listdir(AGENTS_FILE_PATH)
        else:
            if isinstance(agent_ids, str):
                agent_ids = [agent_ids]
            agent_paths = [agent_id + '.toml' if not agent_id.endswith('.toml') else agent_id for agent_id in agent_ids]

        ops = []
        for agent_path in agent_paths:
            agent_id = agent_path[:-5]
            try:
                with open(os.path.join(AGENTS_FILE_PATH, agent_path), "r", encoding='utf-8') as f:
                    agent_config = load(f).unwrap()
                    # 不管怎样都会先存到agent_configs
                    agent_configs[agent_id] = agent_config
                    # 非force下，已被写入过的model会被跳过
                    has_models = []
                    if not force:
                        models_namespaces = await store_alist_namespaces(prefix=('agents', agent_id, 'models'), max_depth=5)
                        for n in models_namespaces:
                            if len(n) > 3:
                                if n[3] != 'builtin':
                                    has_models.append(n[3])
                                elif len(n) > 4 and n[4] == 'settings':
                                    has_models.append('settings')
                    # 如果是新的agent，就写入配置
                    for key, value in agent_config.items():
                        if key in has_models:
                            pass
                        elif key == 'settings':
                            if isinstance(value, dict):
                                namespace = ('agents', agent_id, 'models', 'builtin', 'settings')
                                ops.extend(_write_config_to_store(value, namespace, BuiltinSettings))
                                # 这个值是为了让该model在刚才的store_alist_namespaces中出现，以保证非force的写入只可能出现一次
                                ops.append(PutOp(namespace=namespace, key='__edited_model', value={'spaceholder': True}))
                            else:
                                logger.warning(f"Invalid value for {key} in config file: expected dict for BuiltinSettings, got {type(value)}")
                        elif key == 'plugins':
                            try:
                                _plugins_validator.validate_python(value)
                            except ValidationError:
                                logger.warning(f"Invalid value for {key} in config file: expected dict[str, bool] for plugins, got {type(value)}")
                        elif key == 'init_on_startup':
                            if not isinstance(value, bool):
                                logger.warning(f"Invalid value for {key} in config file: expected bool for init_on_startup, got {type(value)}")
                        elif key in plugin_configs:
                            if isinstance(value, dict):
                                namespace = ('agents', agent_id, 'models', key)
                                ops.extend(_write_config_to_store(value, namespace, plugin_configs[key]))
                                ops.append(PutOp(namespace=namespace, key='__edited_model', value={'spaceholder': True}))
                            else:
                                logger.warning(f"Invalid value for {key} in config file: expected dict for StoreModel, got {type(value)}")
                        else:
                            logger.warning(f"Unknown key {key} in config file, it will be ignored")

            except OSError as e:
                logger.error(f"Error loading config for agent {agent_id}: {e}")
                continue

        if ops:
            await store_abatch(ops)

    return

def update_agent_comments(plugin_configs: dict[str, type[StoreModel]]):
    doc = document()
    doc = _add_config_comments(doc, plugin_configs)
    with open(AGENT_COMMENTS_FILE_PATH, "w", encoding='utf-8') as f:
        dump(doc, f)

def get_agent_configs() -> dict[str, dict[str, Any]]:
    return agent_configs

def get_agent_config(agent_id: str) -> dict[str, Any]:
    return agent_configs.get(agent_id, {})

def get_agent_enabled_plugin_names(agent_id: str) -> list[str]:
    agent_config = get_agent_config(agent_id)
    enabled_plugins = global_config.get('plugins', {})
    enabled_plugins.update(agent_config.get('plugins', {}))
    return [plugin for plugin, enabled in enabled_plugins.items() if enabled]

def get_init_on_startup_agent_ids() -> list[str]:
    return [agent_id for agent_id, agent_config in agent_configs.items() if agent_config.get('init_on_startup')]
