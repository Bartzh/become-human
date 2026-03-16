import json
from typing import Union, Optional, Any, Callable
from collections.abc import Coroutine
import textwrap
import asyncio
from loguru import logger
from typing_inspect import get_args, get_origin
from copy import deepcopy
from inspect import signature, Parameter
from pydantic import BaseModel

def is_valid_json(json_string: str) -> bool:
    try:
        json.loads(json_string)
        return True
    except json.decoder.JSONDecodeError:
        return False


def is_that_type(type_hint: Any, target_class: type) -> bool:
    """
    检查类型是否为指定的类或其子类
    """
    try:
        # 直接类型检查
        if isinstance(type_hint, type) and issubclass(type_hint, target_class):
            return True
        # 处理泛型类型（如 Optional[target_class]）
        if hasattr(type_hint, '__origin__'):
            # 检查是否为 Optional 或其他泛型包装
            origin = type_hint.__origin__
            if origin is Union:
                # 检查 Union 中的类型参数
                for arg in type_hint.__args__:
                    if isinstance(arg, type) and issubclass(arg, target_class):
                        return True
            elif issubclass(origin, target_class):
                return True
        return False
    except:
        return False


def dump_basemodels(items: Union[list, tuple, dict, set], exclude_unset = False) -> Union[list, tuple, dict, set]:
    if isinstance(items, list):
        new_items = []
        old_items = items
        items_type = 'list'
    elif isinstance(items, tuple):
        new_items = ()
        old_items = items
        items_type = 'tuple'
    elif isinstance(items, set):
        new_items = set()
        old_items = items
        items_type = 'set'
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
            new_item = item_value.model_dump(exclude_unset=exclude_unset)
        elif isinstance(item_value, (list, tuple, dict, set)):
            new_item = dump_basemodels(item_value)
        else:
            new_item = item_value
        if items_type == 'dict':
            new_items[item[0]] = new_item
        elif items_type == 'tuple':
            new_items += (new_item,)
        elif items_type == 'set':
            new_items.add(new_item)
        else:
            new_items.append(new_item)
    return new_items

def parse_env_array(env_array: Optional[str]) -> list[str]:
    if env_array:
        return [item.strip() for item in env_array.split(',')]
    else:
        return []

def to_json_like_string(a: Any, support_multiline_str: bool = False) -> str:
    """将任意对象转换为JSON-like字符串

    具体来说，实现了对字符串、布尔值、None、元组、列表、字典的转换"""
    if isinstance(a, str):
        if support_multiline_str and '\n' in a:
            return f'"""{a}"""'
        return f'"{a}"'
    elif isinstance(a, bool):
        return str(a).lower()
    elif a is None:
        return 'null'
    elif isinstance(a, (tuple, list, set)):
        if len(a) >= 3:
            return '[\n' + textwrap.indent(
                ',\n'.join([to_json_like_string(i) for i in a]),
                '    ',
                predicate=lambda line: line.strip() != ''
            ) + '\n]'
        else:
            return '[' + ', '.join([to_json_like_string(i) for i in a]) + ']'
    elif isinstance(a, (BaseModel, dict)):
        if isinstance(a, BaseModel):
            a = a.model_dump()
        if not a:
            return '{}'
        return '{\n' + textwrap.indent(
            ',\n'.join(f'"{k}": {to_json_like_string(v)}' for k, v in a.items()),
            '    ',
            predicate=lambda line: line.strip() != ''
        ) + '\n}'
    else:
        return str(a)

def get_readable_type_name(tp) -> str:
    """增强的类型名称获取"""
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is Union:
        arg_names = [get_readable_type_name(a) for a in args]
        return f"Union[{', '.join(arg_names)}]"

    elif origin:
        if args:
            arg_names = [get_readable_type_name(a) for a in args]
            name = getattr(origin, '__name__', str(origin))
            return f"{name}[{', '.join(arg_names)}]"
        else:
            return getattr(origin, '__name__', str(origin))

    else:
        return getattr(tp, '__name__', str(tp))

def deep_dict_update(base: dict, new: dict, max_depth: int = -1) -> None:
    """
    就地深度更新字典，支持自定义递归层数
    
    Args:
        base: 目标字典（将被原地修改）
        new: 源字典（新数据）
        max_depth: 最大递归层数
                 0 = 仅顶层（等同于 dict.update）
                 -1 = 无限递归（完全深度合并）
                 N = 最多递归 N 层
        copy: 是否对base进行深拷贝（默认False）
        exclude_none: 是否排除new中的所有None值（默认False）
    
    Returns:
        更新后的 base 字典
    
    示例:
        >>> t = {"a": {"b": 1, "c": 2}}
        >>> deep_update_dict(t, {"a": {"b": 9, "d": 3}}, max_depth=-1)
        {'a': {'b': 9, 'c': 2, 'd': 3}}  # b 更新, c 保留, d 新增
    """
    def _update(t: dict, s: dict, current_depth: int):
        for key, value in s.items():
            if isinstance(value, dict):
                if isinstance(t.get(key), dict) and (max_depth <= -1 or current_depth < max_depth):
                    # 递归合并嵌套字典
                    t[key] = _update(t.get(key, {}), value, current_depth + 1)
                else:
                    # 原value不是dic或超过最大递归深度，直接赋值
                    t[key] = value.copy()
            else:
                # 直接赋值（替换或新增）
                t[key] = value
        return t
    _update(base, new, 0)
    return None

def deep_dict_merge(base: dict, new: dict, max_depth: int = -1) -> dict:
    """
    深度合并字典，支持自定义递归层数

    与`deep_dict_update`的区别就只是该函数会深拷贝base并返回其副本

    Args:
        base: 目标字典（将被深拷贝并返回合并后的副本）
        new: 源字典（新数据）
        max_depth: 最大递归层数
                 0 = 仅顶层（等同于 dict.update）
                 -1 = 无限递归（完全深度合并）
                 N = 最多递归 N 层

    Returns:
        合并后的 base 字典

    示例:
        >>> t = {"a": {"b": 1, "c": 2}}
        >>> deep_dict_merge(t, {"a": {"b": 9, "d": 3}}, max_depth=-1)
        {'a': {'b': 9, 'c': 2, 'd': 3}}  # b 合并, c 保留, d 新增
    """
    result = deepcopy(base)
    deep_dict_update(result, new, max_depth)
    return result

def exclude_none_in_dict(d: dict) -> None:
    """排除字典中所有值为None的项，就地修改"""
    for k in list(d.keys()):
        if d[k] is None:
            del d[k]

def filter_kwargs(kwargs: dict[str, Any], handler: Callable) -> dict[str, Any]:
    """过滤kwargs，只保留handler函数的参数

    返回一个新的dict"""
    # 获取函数签名，过滤参数
    sig = signature(handler)
    filtered_kwargs = {}
    for param_name, param in sig.parameters.items():
        # 如果存在**kwargs，意味着不需要过滤，直接返回
        if param.kind == Parameter.VAR_KEYWORD:
            return kwargs.copy()
        elif param_name in kwargs:
            filtered_kwargs[param_name] = kwargs[param_name]
    return filtered_kwargs

async def gather_safe(*coros_or_futures: Coroutine | asyncio.Future, return_exceptions: bool = False) -> list[Any]:
    """
    并行执行多个异步任务，捕获记录所有异常并返回结果列表。

    Args:
        *coros_or_futures: 要执行的异步任务或Future对象
        return_exceptions: 是否返回异常对象而不是None，默认False

    Returns:
        包含所有任务结果的列表，异常位置为None或异常对象（根据return_exceptions）
    """
    results = await asyncio.gather(*coros_or_futures, return_exceptions=True)
    for idx, result in enumerate(results):
        if isinstance(result, BaseException):
            logger.opt(exception=result, depth=1).error(f"任务 {idx} 执行出错")
            if not return_exceptions:
                results[idx] = None
    return results
