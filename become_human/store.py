from langgraph.store.sqlite import AsyncSqliteStore
from langgraph.store.base import Item, NotProvided, NOT_PROVIDED, Op, Result, NamespacePath, SearchItem, PutOp

from pydantic import BaseModel, Field, TypeAdapter, PydanticSchemaGenerationError
from pydantic_core import ValidationError
from typing import Literal, Any, Iterable, Optional, Union, Callable, get_type_hints
import asyncio

from become_human.utils import make_sure_path_exists

STORE_PATH = './data/store.sqlite'

make_sure_path_exists()

async def store_setup():
    async with AsyncSqliteStore.from_conn_string(STORE_PATH) as store:
        await store.setup()
    run_listener()

async def store_aget(
    namespace: tuple[str, ...],
    key: str,
    *,
    refresh_ttl: bool | None = None,
) -> Item | None:
    async with AsyncSqliteStore.from_conn_string(STORE_PATH) as store:
        return await store.aget(namespace=namespace, key=key, refresh_ttl=refresh_ttl)

async def store_aput(
    namespace: tuple[str, ...],
    key: str,
    value: dict[str, Any],
    index: Literal[False] | list[str] | None = None,
    *,
    ttl: float | None | NotProvided = NOT_PROVIDED,
) -> None:
    async with AsyncSqliteStore.from_conn_string(STORE_PATH) as store:
        await store.aput(namespace=namespace, key=key, value=value, index=index, ttl=ttl)

async def store_adelete(
    namespace: tuple[str, ...],
    key: str,
) -> None:
    async with AsyncSqliteStore.from_conn_string(STORE_PATH) as store:
        await store.adelete(namespace=namespace, key=key)

async def store_abatch(ops: Iterable[Op]) -> list[Result]:
    async with AsyncSqliteStore.from_conn_string(STORE_PATH) as store:
        return await store.abatch(ops)

async def store_alist_namespaces(
    *,
    prefix: NamespacePath | None = None,
    suffix: NamespacePath | None = None,
    max_depth: int | None = None,
    limit: int = 0,
    offset: int = 0,
    batch_size: int = 100,
) -> list[tuple[str, ...]]:
    """limit设为0则意为没有限制，将使用batch_size遍历所有结果"""
    async with AsyncSqliteStore.from_conn_string(STORE_PATH) as store:
        if limit == 0:
            results = []
            while True:
                batch = await store.alist_namespaces(
                    prefix=prefix,
                    suffix=suffix,
                    max_depth=max_depth,
                    limit=batch_size,
                    offset=offset,
                )
                if not batch:
                    break
                results.extend(batch)
                offset += batch_size
                if len(batch) < batch_size:
                    break
            return results
        else:
            return await store.alist_namespaces(prefix=prefix, suffix=suffix, max_depth=max_depth, limit=limit, offset=offset)

async def store_asearch(
    namespace_prefix: tuple[str, ...],
    /,
    *,
    query: str | None = None,
    filter: dict[str, Any] | None = None,
    limit: int = 0,
    offset: int = 0,
    refresh_ttl: bool | None = None,
    batch_size: int = 100,
) -> list[SearchItem]:
    """limit设为0则意为没有限制，将使用batch_size遍历所有结果"""
    async with AsyncSqliteStore.from_conn_string(STORE_PATH) as store:
        if limit == 0:
            results = []
            while True:
                batch = await store.asearch(
                    namespace_prefix,
                    query=query,
                    filter=filter,
                    limit=batch_size,
                    offset=offset,
                    refresh_ttl=refresh_ttl,
                )
                if not batch:
                    break
                results.extend(batch)
                offset += batch_size
                if len(batch) < batch_size:
                    break
            return results
        else:
            return await store.asearch(namespace_prefix, query=query, filter=filter, limit=limit, offset=offset, refresh_ttl=refresh_ttl)

async def store_adelete_namespace(
    namespace: tuple[str, ...],
) -> None:
    items = await store_asearch(namespace)
    ops = [PutOp(namespace=item.namespace, key=item.key, value=None) for item in items]
    await store_abatch(ops)


store_queue = asyncio.Queue()

listener_task_is_running = False
async def store_queue_listener():
    global listener_task_is_running
    while listener_task_is_running:
        item = await store_queue.get()
        if item['action'] == 'put':
            await store_aput(item['namespace'], item['key'], item['value'])
        elif item['action'] == 'delete':
            await store_adelete(item['namespace'], item['key'])
        elif item['action'] == 'stop':
            listener_task_is_running = False
            print('store listener task stopped.')
listener_task: Optional[asyncio.Task] = None

def run_listener():
    global listener_task, listener_task_is_running
    listener_task_is_running = True
    if listener_task is None or listener_task.done():
        listener_task = asyncio.create_task(store_queue_listener())

async def stop_listener():
    global listener_task
    if listener_task is not None and not listener_task.done() and listener_task_is_running:
        await store_queue.put({'action': 'stop'})
        await listener_task
    listener_task = None


class StoreField(BaseModel):
    readable_name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    default: Optional[Any] = Field(default=None)
    default_factory: Optional[Callable[[], Any]] = Field(default=None)

class StoreItem(BaseModel):
    readable_name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    value: Optional[Any] = Field(default=None)

class StoreModel:
    """为该类的子类创建新的类属性，直接赋值为StoreField即可。需要设置_namespace。注意添加StoreModel属性时不要使用泛型也不要赋值；添加BaseModel属性时不要使用泛型。"""

    _thread_id: str
    _namespace: tuple[str, ...] = ()
    _readable_name: Optional[str] = None
    _description: Optional[str] = None
    _cache: dict[str, StoreItem]

    def __init__(self, thread_id: str, search_items: list[SearchItem], namespace: Optional[tuple[str, ...]] = None):
        self._thread_id = thread_id
        if namespace:
            self._namespace = namespace + super().__getattribute__('_namespace')
        self_namespace = self._namespace
        cached = {}
        not_cached = []
        for item in search_items:
            if item.namespace == self_namespace:
                cached[item.key] = StoreItem(
                    readable_name=item.value.get('readable_name'),
                    description=item.value.get('description'),
                    value=item.value.get('value')
                )
            else:
                not_cached.append(item)
        self._cache = cached

        self_cls = self.__class__
        type_hints = get_type_hints(self_cls)
        for attr_name, attr_type in type_hints.items():
            if not hasattr(self_cls, attr_name) and isinstance(attr_type, type) and issubclass(attr_type, StoreModel):
                nested_model = attr_type(thread_id, not_cached, super().__getattribute__('_namespace'))
                super().__setattr__(attr_name, nested_model)

    def __getattribute__(self, name: str):
        if name == '_namespace':
            return (self._thread_id,) + super().__getattribute__('_namespace')
        attr = super().__getattribute__(name)
        if not isinstance(attr, StoreField):
            return attr
        else:
            value = self._cache.get(name)
            value_type = self.__class__.__annotations__.get(name)
            if value is not None:
                value = value.value
            if value is None:
                value = self.__class__.get_field_default(attr)
                if (
                    value is None and
                    value_type is not None and
                    (str(value_type) == 'typing.Any' or not isinstance(value, value_type))
                ):
                    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}', from StoreModel.")
            if isinstance(value_type, type) and issubclass(value_type, BaseModel) and isinstance(value, dict):
                value = value_type.model_validate(value)
            return value

    def __setattr__(self, name: str, value: Any):
        try:
            attr = super().__getattribute__(name)
        except AttributeError:
            attr = None
        if not isinstance(attr, StoreField):
            super().__setattr__(name, value)
        else:
            hints = get_type_hints(self.__class__)
            value_type = hints.get(name)
            if value_type is not None:
                if isinstance(value, type) and issubclass(value_type, BaseModel):
                    if not isinstance(value, value_type):
                        raise ValueError(f"Invalid value for {self.__class__.__name__}.{name}: {e}")
                    else:
                        value = value.model_dump(exclude_unset=True)
                else:
                    adapter = TypeAdapter(value_type)
                    try:
                        adapter.validate_python(value, strict=True)
                    except ValidationError as e:
                        raise ValueError(f"Invalid value for {self.__class__.__name__}.{name}: {e}")
            item = self._cache.get(name)
            if item is not None:
                item.value = value
            else:
                item = StoreItem(value=value)
                self._cache[name] = item
            store_queue.put_nowait({'action': 'put', 'namespace': self._namespace, 'key': name, 'value': item.model_dump()})

    def __delattr__(self, name: str):
        try:
            attr = super().__getattribute__(name)
        except AttributeError:
            attr = None
        if not isinstance(attr, StoreField):
            super().__delattr__(name)
        else:
            if name in self._cache.keys():
                del self._cache[name]
            store_queue.put_nowait({'action': 'delete', 'namespace': self._namespace, 'key': name})

    @classmethod
    def get_field(cls, field_name: str) -> StoreField | None:
        """获取字段的元数据"""
        meta = cls.__dict__.get(field_name)
        if isinstance(meta, StoreField):
            return meta
        else:
            return None

    def get_field_readable_name(self, field_name: str) -> str | None:
        """获取字段的可读名称字符串。注意，这是一个实例方法，当字段里无可读名称时，会尝试从数据库中获取。需要类方法请调用get_field(field_name).readable_name。"""
        meta = self.__class__.get_field(field_name)
        readable_name = meta.readable_name if meta else None
        if readable_name is None:
            item = self._cache.get(field_name)
            readable_name = item.readable_name if item else None
        return readable_name

    def get_field_description(self, field_name: str) -> str | None:
        """获取字段的描述字符串。注意，这是一个实例方法，当字段里无描述时，会尝试从数据库中获取。需要类方法请调用get_field(field_name).description。"""
        meta = self.__class__.get_field(field_name)
        desc = meta.description if meta else None
        if desc is None:
            item = self._cache.get(field_name)
            desc = item.description if item else None
        return desc

    @classmethod
    def get_field_default(cls, field: Union[str, StoreField]) -> Any | None:
        """获取字段的默认值"""
        if isinstance(field, str):
            field = cls.get_field(field)
            if field is None:
                return None
        value = None
        if field:
            if field.default_factory:
                value = field.default_factory()
            elif field.default is not None:
                value = field.default
        return value