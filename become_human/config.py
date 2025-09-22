from tomlkit import load, loads, dump, document, table, comment, nl, TOMLDocument
from tomlkit.items import Table
from become_human.utils import make_sure_path_exists
from become_human.store import StoreModel, StoreField, store_alist_namespaces, store_abatch
from become_human.store_settings import ThreadSettings
from langgraph.store.base import PutOp
import os
from warnings import warn
from typing import Optional, Literal, Type, Union, get_type_hints, Any
from typing_inspect import get_args, get_origin
from pydantic import BaseModel, TypeAdapter
from pydantic_core import ValidationError

make_sure_path_exists()

THREADS_FILE = "./config/threads.toml"
THREAD_COMMENTS_FILE = "./config/thread_comments.toml"

thread_configs: dict[str, dict[str, Any]] = {}

DEFAULT_THREAD_2 = {
    'main': {
        'role_prompt': '你是一个待人较为冷淡的人，对什么事情都无所谓，比较懒，说话也很简短、敷衍。',
        'active_time_range': (60.0, 1800.0),
        'self_call_time_ranges': [(300.0, 10800.0)],
        'wakeup_time_range': (1.0, 61201.0),
        'sleep_time_range': (82800.0, 32400.0)
    },
    'recycle': {
        'base_stable_time': 43200.0,
        'cleanup_on_non_active_recycle': True,
        'cleanup_target_size': 800
    },
    'retrieve': {
        'active_retrieve_config': {
            'similarity_weight': 0.4,
            'retrievability_weight': 0.3,
            'diversity_weight': 0.3
        }
    }
}

def parse_default(d: Any) -> str:
        if isinstance(d, str):
            return f'"{d}"'
        elif isinstance(d, bool):
            return str(d).lower()
        elif d is None:
            return 'null'
        elif isinstance(d, (tuple, list)):
            return '[' + ', '.join([parse_default(i) for i in d]) + ']'
        else:
            return str(d)

def enhanced_type_name(tp):
    """增强的类型名称获取"""
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Union:
        arg_names = [enhanced_type_name(a) for a in args]
        return f"Union[{', '.join(arg_names)}]"

    elif origin:
        if args:
            arg_names = [enhanced_type_name(a) for a in args]
            name = getattr(origin, '__name__', str(origin))
            return f"{name}[{', '.join(arg_names)}]"
        else:
            return getattr(origin, '__name__', str(origin))

    else:
        return getattr(tp, '__name__', str(tp))

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
                desc = f'<{enhanced_type_name(hint_type)}> ' + desc
                doc.add(nl())
                doc.add(nl())
                doc.add(comment(desc))
                doc.add(comment(f'[thread_id.{prefix}{key}]'))
                doc = _add_field_comments(doc, hint_type, prefix+key+'.')
            else:
                field = model.__dict__.get(key)
                if field is not None and isinstance(field, StoreField):
                    doc.add(nl())
                    desc = f'<{enhanced_type_name(hint_type)}> '
                    desc += field.readable_name if field.readable_name else ''
                    if field.readable_name and field.description:
                        desc += '：' + field.description
                    else:
                        desc += field.description if field.description else ''
                    doc.add(comment(desc))
                    default = model.get_field_default(field)
                    doc.add(comment(f'{key}{" = " + parse_default(default)}'))
    else:
        for field_name, field_info in model.model_fields.items():
            desc = f'<{enhanced_type_name(field_info.annotation)}> '
            desc += field_info.description if field_info.description else ''
            if isinstance(field_info.annotation, type) and issubclass(field_info.annotation, (StoreModel, BaseModel)):
                doc.add(nl())
                doc.add(nl())
                doc.add(comment(desc))
                doc.add(comment(f'[thread_id.{prefix}{field_name}]'))
                doc = _add_field_comments(doc, field_info.annotation, prefix+field_name+'.')
            else:
                doc.add(nl())
                doc.add(comment(desc))
                if field_info.default_factory is not None:
                    v = field_info.default_factory()
                elif field_info.default is not None:
                    v = field_info.default
                else:
                    v = None
                doc.add(comment(f'{field_name}{' = ' + parse_default(v)}'))
    return doc

def _add_config_comments(doc: TOMLDocument):
    doc.add(comment('类型说明'))
    doc.add(comment('<>描述了该字段的类型，写入数据库时会使用pydantic的类型转换功能尝试转换至目标类型'))
    doc.add(comment('意味着如str, int, float, bool等类型，会尝试自动转换为对应的类型，但不建议依赖类型转换功能'))
    doc.add(nl())
    doc.add(nl())
    doc.add(comment('配置说明'))
    doc.add(comment('<str> 线程id: 会使用这个key来作为thread_id，它是唯一的'))
    doc.add(comment('[thread_id]'))
    doc.add(comment('<bool> 启动时初始化: 是否在程序启动时自动初始化该线程'))
    doc.add(comment('init_on_startup = false'))

    # 添加字段描述
    doc = _add_field_comments(doc, ThreadSettings)

    #doc.add(nl())
    return doc

def _add_config_thread_comments(doc: TOMLDocument, prefix: str, config: dict):
    doc.add(comment(f'[{prefix}]'))
    for key, value in config.items():
        if isinstance(value, dict):
            doc.add(nl())
            doc = _add_config_thread_comments(doc, prefix+'.'+key, value)
        else:
            doc.add(comment(f'{key} = {parse_default(value)}'))
    return doc

def create_default_thread_configs_toml() -> TOMLDocument:
    doc = document()
    doc.add(comment('线程配置'))
    doc.add(comment('查阅 thread_comments.toml 获取线程配置的详细说明及默认值'))
    doc.add(nl())
    doc.add(nl())
    doc.add(comment('[default_thread]'))
    doc.add(nl())
    doc.add(nl())
    doc = _add_config_thread_comments(doc, 'default_thread_2', DEFAULT_THREAD_2)
    doc.add(nl())
    return doc


async def load_config(thread_ids: Optional[Union[list[str], str]] = None) -> dict[str, dict[str, Any]]:
    global thread_configs
    update_thread_comments()
    if not os.path.exists(THREADS_FILE):
        thread_configs_toml = create_default_thread_configs_toml()
        with open(THREADS_FILE, 'w', encoding='utf-8') as f:
            dump(thread_configs_toml, f)
    else:
        with open(THREADS_FILE, "r", encoding='utf-8') as f:
            thread_configs_toml = load(f)
    thread_configs = thread_configs_toml.unwrap()
    if thread_ids:
        if isinstance(thread_ids, str):
            thread_ids = [thread_ids]
        thread_configs = {k: v for k, v in thread_configs.items() if k in thread_ids}
    if not thread_configs:
        return {}
    has_settings_namespaces = await store_alist_namespaces(prefix=('*', 'model', 'settings'), max_depth=3, limit=99999)
    has_settings_thread_ids = [n[0] for n in has_settings_namespaces]

    def _dump_models(items: Union[list, tuple, dict]) -> Union[list, tuple, dict]:
        if isinstance(items, list):
            new_items = []
            old_items = items
            items_type = 'list'
        elif isinstance(items, tuple):
            new_items = ()
            old_items = items
            items_type = 'tuple'
        else:
            new_items = {}
            old_items = items.items()
            items_type = 'dict'
        for item in old_items:
            if items_type == 'dict':
                item_value = item[1]
            else:
                item_value = item
            if isinstance(item_value, BaseModel):
                new_item = item_value.model_dump(exclude_unset=True)
            elif isinstance(item_value, (list, tuple, dict)):
                new_item = _dump_models(item_value)
            else:
                new_item = item_value
            if items_type == 'dict':
                new_items[item[0]] = new_item
            elif items_type == 'tuple':
                new_items += (new_item,)
            else:
                new_items.append(new_item)
        return new_items
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
                    adapter = TypeAdapter(hint_type)
                    try:
                        validated_value = adapter.validate_python(value)
                    except ValidationError as e:
                        warn(f"Invalid value for {key} in config file: {e}")
                        continue
                    if isinstance(validated_value, BaseModel):
                        validated_value = validated_value.model_dump(exclude_unset=True)
                    elif isinstance(validated_value, (list, tuple, dict)):
                        validated_value = _dump_models(validated_value)
                    put_ops.append(PutOp(namespace=namespace, key=key, value={'value': validated_value}))
        return put_ops

    ops = []
    for key, value in thread_configs.items():
        if key not in has_settings_thread_ids:
            if isinstance(value, dict):
                ops.extend(_write_config_to_store(value, (key, 'model', 'settings'), ThreadSettings))
    if ops:
        await store_abatch(ops)

    return thread_configs

def update_thread_comments():
    doc = document()
    doc = _add_config_comments(doc)
    with open(THREAD_COMMENTS_FILE, "w", encoding='utf-8') as f:
        dump(doc, f)

def get_thread_configs() -> dict[str, dict[str, Any]]:
        return thread_configs