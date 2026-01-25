from langgraph.store.sqlite import AsyncSqliteStore
from langgraph.store.base import Item, NotProvided, NOT_PROVIDED, Op, Result, NamespacePath, SearchItem, PutOp

from pydantic import BaseModel, Field, TypeAdapter, PydanticSchemaGenerationError
from pydantic_core import ValidationError, core_schema
from typing import Literal, Any, Iterable, Optional, Self, Union, Callable, get_type_hints
#from weakref import WeakKeyDictionary
import asyncio
from loguru import logger

STORE_PATH = './data/store.sqlite'

async def store_setup():
    async with AsyncSqliteStore.from_conn_string(STORE_PATH) as store:
        await store.setup()
    store_run_listener()

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
            return await store.asearch(
                namespace_prefix,
                query=query,
                filter=filter,
                limit=limit,
                offset=offset,
                refresh_ttl=refresh_ttl
            )

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
    stop_retry_count = 0
    while listener_task_is_running:
        item = await store_queue.get()
        if item['action'] == 'put':
            await store_aput(item['namespace'], item['key'], item['value'])
        elif item['action'] == 'delete':
            await store_adelete(item['namespace'], item['key'])
        elif item['action'] == 'stop':
            if store_queue.empty():
                listener_task_is_running = False
                logger.info('store listener task stopped.')
            else:
                stop_retry_count += 1
                if stop_retry_count > 10:
                    logger.error('store listener task stop retry count exceeded 10, stop task anyway.')
                    listener_task_is_running = False
                else:
                    logger.info("store listener task can't stop because there are still items in the queue, retrying...")
                    await store_queue.put(item)
listener_task: Optional[asyncio.Task] = None

def store_run_listener():
    global listener_task, listener_task_is_running
    listener_task_is_running = True
    if listener_task is None or listener_task.done():
        listener_task = asyncio.create_task(store_queue_listener())

async def store_stop_listener():
    global listener_task
    if listener_task is not None and not listener_task.done() and listener_task_is_running:
        await store_queue.put({'action': 'stop'})
        await listener_task
    listener_task = None


class UnsetType:

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "Unset"

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self) -> Self:
        return self

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        return core_schema.is_instance_schema(cls
            #serialization=core_schema.plain_serializer_function_ser_schema(lambda x: None)
        )

    def is_unset(self, value: Any) -> bool:
        return value is self

Unset = UnsetType()

class StoreField(BaseModel):
    readable_name: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    default: Union[Any, UnsetType] = Field(default=Unset)
    default_factory: Optional[Callable[[], Any]] = Field(default=None)

    def get_default_value(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        elif self.default is not Unset:
            return self.default
        else:
            raise AttributeError(f"{self.readable_name}值没有被设置，并且没有默认值可以提供。")

class StoreItem(BaseModel):
    readable_name: Optional[Union[str, UnsetType]] = Field(default=Unset, exclude_if=Unset.is_unset)
    description: Optional[Union[str, UnsetType]] = Field(default=Unset, exclude_if=Unset.is_unset)
    value: Union[Any, UnsetType] = Field(default=Unset, exclude_if=Unset.is_unset)

class StoreModel:
    """为该类的子类创建新的类属性，直接赋值为StoreField即可。需要设置_namespace。注意添加StoreModel属性时不要使用泛型也不要赋值。

    在使用时，注意只能使用 StoreModel.xxx = xxx 或 setattr 的方式改变属性值。

    当获取一个没有被设置的值时，如果这个值的默认值是由default_factory提供的（与pydantic的default_factory无关），则会同时存储到store。"""

    _agent_id: str
    _namespace: tuple[str, ...]
    _readable_name: Optional[str] = None
    _description: Optional[str] = None
    _cache: dict[str, StoreItem]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, '_namespace'):
            raise TypeError(f"子类 {cls.__name__} 必须定义类属性 '_namespace'")

    def __init__(self, agent_id: str, search_items: list[SearchItem], namespace: Optional[tuple[str, ...]] = None):
        self._agent_id = agent_id
        if namespace:
            self._namespace = namespace + super().__getattribute__('_namespace')
        self_namespace = self._namespace
        self_cls = self.__class__
        type_hints = get_store_type_hints(self_cls)
        cached = {}
        not_cached = []
        for item in search_items:
            # 没有允许意外的key存进_cache
            if item.namespace == self_namespace and item.key in type_hints.keys():
                if item.value.get('value', Unset) is not Unset:
                    try:
                        adapter = TypeAdapter(type_hints[item.key])
                        value = adapter.validate_python(item.value['value'])
                    except ValidationError as e:
                        logger.warning(f"Invalid value for {item.key}: {e}, from store.")
                        continue
                else:
                    value = Unset
                cached[item.key] = StoreItem(
                    readable_name=item.value.get('readable_name', Unset),
                    description=item.value.get('description', Unset),
                    value=value
                )
            else:
                not_cached.append(item)
        self._cache = cached

        for attr_name, attr_type in type_hints.items():
            if not hasattr(self_cls, attr_name) and isinstance(attr_type, type) and issubclass(attr_type, StoreModel):
                nested_model = attr_type(agent_id, not_cached, super().__getattribute__('_namespace'))
                super().__setattr__(attr_name, nested_model)

    def __getattribute__(self, name: str):
        if name == '_namespace':
            return ('agents', self._agent_id, 'models') + super().__getattribute__('_namespace')
        attr = super().__getattribute__(name)
        if not isinstance(attr, StoreField):
            return attr
        else:
            item = self._cache.get(name)
            if item is None:
                item = StoreItem()
            value = item.value
            if value is Unset:
                value = attr.get_default_value()
                # 如果是default_factory生成的默认值，则保存到store中
                if attr.default_factory is not None:
                    item.value = value
                    self._cache[name] = item
                    store_queue.put_nowait({'action': 'put', 'namespace': self._namespace, 'key': name, 'value': item.model_dump()})
            return value

    def __setattr__(self, name: str, value: Any):
        try:
            attr = super().__getattribute__(name)
        except AttributeError:
            attr = None
        if not isinstance(attr, StoreField):
            super().__setattr__(name, value)
        else:
            hints = get_store_type_hints(self.__class__)
            value_type = hints.get(name)
            if value_type is not None:
                adapter = TypeAdapter(value_type)
                try:
                    value = adapter.validate_python(value, strict=True)
                except ValidationError as e:
                    raise ValueError(f"Invalid value for {self.__class__.__name__}.{name}: {e}")
            else:
                raise AttributeError(f"{self.__class__.__name__}.{name} 虽然被赋值了StoreField，但似乎没有类型注解，无法验证其类型。")
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
    def get_field(cls, field_name: str) -> StoreField:
        """获取字段的StoreField，如果找不到会抛出AttributeError"""
        meta = cls.__dict__.get(field_name)
        if isinstance(meta, StoreField):
            return meta
        else:
            raise AttributeError(f"'{cls.__name__}' 不存在 '{field_name}' 或其不是 StoreField。")

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

#_TYPE_HINTS_CACHE: WeakKeyDictionary[type, Dict[str, Any]] = WeakKeyDictionary()
_STORE_TYPE_HINTS_CACHES: dict[type[StoreModel], dict[str, Any]] = {}
def get_store_type_hints(store: type[StoreModel]) -> dict[str, Any]:
    if store not in _STORE_TYPE_HINTS_CACHES:
        _STORE_TYPE_HINTS_CACHES[store] = get_type_hints(store)
    return _STORE_TYPE_HINTS_CACHES[store]
