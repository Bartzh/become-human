import asyncio
from copy import deepcopy
from dataclasses import dataclass
from loguru import logger
from inspect import iscoroutinefunction, signature
#from weakref import WeakKeyDictionary

from langgraph.store.sqlite import AsyncSqliteStore
from langgraph.store.base import Item, NotProvided, NOT_PROVIDED, Op, Result, NamespacePath, SearchItem, PutOp

from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    PydanticSchemaGenerationError,
    PrivateAttr,
    field_validator,
    ValidationInfo,
    create_model,
    ConfigDict,
)
from pydantic_core import ValidationError, core_schema
from typing import Literal, Any, Iterable, Optional, Self, Union, Callable, get_type_hints

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
    title: Optional[str] = Field(default=None)
    description: Optional[str] = Field(default=None)
    default: Union[Any, UnsetType] = Field(default=Unset)
    default_factory: Optional[Callable[[], Any]] = Field(default=None)
    reducer: Optional[Callable[[Any, Any], Any]] = Field(default=None)
    frozen: bool = Field(default=False)
    """是否冻结该字段，冻结后将不能被修改。目前只实现了在StoreModel内的冻结，不会拦截直接修改数据库或通过配置文件的修改"""
    validator: Optional[Callable[[Any], Any]] = Field(default=None) # TODO

    model_config = ConfigDict(frozen=True)

    _owner: type['StoreModel'] = PrivateAttr()
    _attribute_name: str = PrivateAttr()

    @field_validator('default_factory', mode='after')
    @classmethod
    def validate_default_factory(cls, value: Optional[Callable[[], Any]], info: ValidationInfo) -> Optional[Callable[[], Any]]:
        if value is None and info.data['default'] is Unset:
            raise ValueError("default_factory 或 default 必须设置一个。")
        if value is not None and info.data['default'] is not Unset:
            raise ValueError("default_factory 不能与 default 同时设置。")
        return value

    @field_validator('reducer', mode='after')
    @classmethod
    def validate_reducer(cls, value: Optional[Callable[[Any, Any], Any]]) -> Optional[Callable[[Any, Any], Any]]:
        if value is not None and iscoroutinefunction(value):
            raise ValueError("reducer 不能是异步函数。")
        sig = signature(value)
        if sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and k != 'sprite_id'
            for k, p in sig.parameters.items()
        ) != 2:
            raise ValueError("reducer 函数必须有两个位置参数，第一个参数为当前值，第二个参数为新值。")
        return value

    def __set_name__(self, owner: type, name: str):
        if not issubclass(owner, StoreModel):
            raise TypeError(f"{owner.__name__} 不是 StoreModel 的子类，不能设置 StoreField。")
        self._owner = owner
        self._attribute_name = name

    def get_default_value(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        elif self.default is not Unset:
            return self.default
        else:
            raise AttributeError(f"{self._owner.__name__}的{self._attribute_name}没有设置默认值。")

    def get_default_value_with_global_config(self, namespace: tuple[str, ...]) -> tuple[Any, bool]:
        """优先从全局配置中获取默认值，若不存在则返回默认值。

        Returns:
            tuple[Any, bool]: 第一个元素为默认值，第二个元素为是否是从全局配置中获取的默认值。
        """
        from become_human.config import global_config
        value = global_config.get(namespace[3])
        if isinstance(value, dict):
            for key in namespace[4:]:
                value = value.get(key)
                if not isinstance(value, dict):
                    value = Unset
                    break
            if isinstance(value, dict):
                value = value.get(self._attribute_name, Unset)
        else:
            value = Unset
        if value is not Unset:
            type_hints = self._owner.get_type_hints()
            if self._attribute_name in type_hints:
                try:
                    value = TypeAdapter(type_hints[self._attribute_name]).validate_python(value)
                    return value, True
                except ValidationError:
                    logger.warning(f"全局配置中存在 {namespace[3:]}.{self._attribute_name} 的值，但其不符合 {type_hints[self._attribute_name]} 的类型。将使用默认值。")
        return self.get_default_value(), False

class StoreItem(BaseModel):
    title: Optional[Union[str, UnsetType]] = Field(default=Unset, exclude_if=Unset.is_unset)
    description: Optional[Union[str, UnsetType]] = Field(default=Unset, exclude_if=Unset.is_unset)
    value: Union[Any, UnsetType] = Field(default=Unset, exclude_if=Unset.is_unset)

@dataclass
class SimpleItem:
    namespace: tuple[str, ...]
    key: str
    value: dict[str, Any]

class StoreModel:
    """为该类的子类创建新的类属性，直接赋值为StoreField即可。需要设置_namespace。注意添加StoreModel属性时不要使用泛型也不要赋值。

    在使用时，注意只能使用 StoreModel.xxx = xxx 或 setattr 的方式改变属性值。

    当获取一个没有被设置的值时，如果这个值的默认值是由default_factory提供的（与pydantic的default_factory无关），则会同时存储到store。

    ### frozen

    当frozen为True时：
    - 实例的属性不能被改变
    - 不会存储数据到数据库中
    - 不会从全局配置中获取默认值
    - 若没有指定sprite_id，_namespace不会有('sprites', self._sprite_id, 'datas'/'configs')的前缀"""

    _sprite_id: str
    _namespace: tuple[str, ...]
    _title: Optional[str] = None
    _description: Optional[str] = None
    _is_config: bool = False
    _cache: dict[str, StoreItem]
    _frozen: bool

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, '_namespace'):
            raise TypeError(f"StoreModel子类 {cls.__name__} 必须定义类属性 '_namespace'")
        if isinstance(cls._namespace, str):
            cls._namespace = (cls._namespace,)
        elif not isinstance(cls._namespace, tuple):
            raise TypeError(f"StoreModel子类 {cls.__name__} 的_namespace 必须是 str 或 tuple[str, ...] 类型。")

    @classmethod
    async def from_store(cls, sprite_id: str, frozen: bool = False):
        search_items = await store_asearch(
            ('sprites', sprite_id, 'configs' if cls._is_config else 'datas') + cls._namespace
        )
        return cls(items=search_items, sprite_id=sprite_id, frozen=frozen)

    def __init__(
        self,
        items: list[Union[SearchItem, SimpleItem]],
        sprite_id: Optional[str] = None,
        namespace: Optional[tuple[str, ...]] = None,
        frozen: bool = False,
        _is_config: Optional[bool] = None
    ):
        if _is_config is not None:
            self._is_config = _is_config
        self._frozen = frozen
        self_cls = self.__class__
        type_hints = self.get_type_hints()
        cached = {}
        not_cached = []
        if not frozen and sprite_id is None:
            raise ValueError("非frozen下初始化StoreModel必须指定sprite_id")
        if sprite_id is not None:
            self._sprite_id = sprite_id
        if namespace:
            self._namespace = namespace + super().__getattribute__('_namespace')
        self_namespace = self._namespace
        for item in items:
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
                    try:
                        field = self.get_field(item.key)
                    except AttributeError:
                        logger.warning(f"在store中找到了 {item.key} ，虽然model的type_hints中存在其定义，但不是StoreField，这个值将被忽略。")
                        continue
                    # 如果要由default_factory生成默认值，则直接生成并保存到store中
                    if field.default_factory is not None:
                        if not frozen:
                            default_value, is_global = field.get_default_value_with_global_config(self_namespace)
                            if not is_global:
                                value = default_value
                                new_value = item.value.copy()
                                new_value['value'] = value
                                store_queue.put_nowait({'action': 'put', 'namespace': self_namespace, 'key': item.key, 'value': new_value})
                            else:
                                value = Unset
                        else:
                            value = field.get_default_value()
                    else:
                        value = Unset
                cached[item.key] = StoreItem(
                    title=item.value.get('title', Unset),
                    description=item.value.get('description', Unset),
                    value=value
                )
            else:
                not_cached.append(item)
        self._cache = cached

        for attr_name, attr_type in type_hints.items():
            if not hasattr(self_cls, attr_name) and isinstance(attr_type, type) and issubclass(attr_type, StoreModel):
                nested_model = attr_type(
                    items=not_cached,
                    sprite_id=sprite_id,
                    namespace=super().__getattribute__('_namespace'),
                    frozen=frozen,
                    _is_config=self._is_config if _is_config is None else _is_config
                )
                super().__setattr__(attr_name, nested_model)

    def __getattribute__(self, name: str):
        if name == '_namespace':
            if (sprite_id := getattr(self, '_sprite_id', None)) is not None:
                return ('sprites', sprite_id, 'configs' if self._is_config else 'datas') + super().__getattribute__('_namespace')
            else:
                return super().__getattribute__('_namespace')
        attr = super().__getattribute__(name)
        if not isinstance(attr, StoreField):
            return attr
        else:
            item = self._cache.get(name)
            if item is None:
                item = StoreItem()
            value = item.value
            if value is Unset:
                if not self._frozen:
                    value, is_global = attr.get_default_value_with_global_config(self._namespace)
                    # 如果是default_factory生成的默认值，则保存到store中
                    if not is_global and attr.default_factory is not None:
                        item.value = value
                        self._cache[name] = item
                        store_queue.put_nowait({'action': 'put', 'namespace': self._namespace, 'key': name, 'value': item.model_dump()})
                else:
                    value = attr.get_default_value()
                    if attr.default_factory is not None:
                        item.value = value
                        self._cache[name] = item
            return value

    def __setattr__(self, name: str, value: Any):
        try:
            attr = super().__getattribute__(name)
        except AttributeError:
            attr = None
        if not isinstance(attr, StoreField):
            super().__setattr__(name, value)
        else:
            if self._frozen:
                raise AttributeError(f"这是一个冻结的 {self.__class__.__name__} 实例，不能被修改。")
            if attr.frozen:
                raise AttributeError(f"{self.__class__.__name__}.{name} 是一个冻结的字段，不能被修改。")
            hints = self.get_type_hints()
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
                if attr.reducer is not None:
                    sig = signature(attr.reducer)
                    if 'sprite_id' in sig.parameters:
                        value = attr.reducer(item.value, value, sprite_id=self._sprite_id)
                    else:
                        value = attr.reducer(item.value, value)
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
            if self._frozen:
                raise AttributeError(f"这是一个冻结的 {self.__class__.__name__} 实例，不能删除字段。")
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

    def get_field_title(self, field_name: str) -> str | None:
        """获取字段的可读名称字符串。注意，这是一个实例方法，当字段里无可读名称时，会尝试从数据库中获取。需要类方法请调用get_field(field_name).title。"""
        meta = self.__class__.get_field(field_name)
        title = meta.title if meta else None
        if title is None:
            item = self._cache.get(field_name)
            title = item.title if item else None
        return title

    def get_field_description(self, field_name: str) -> str | None:
        """获取字段的描述字符串。注意，这是一个实例方法，当字段里无描述时，会尝试从数据库中获取。需要类方法请调用get_field(field_name).description。"""
        meta = self.__class__.get_field(field_name)
        desc = meta.description if meta else None
        if desc is None:
            item = self._cache.get(field_name)
            desc = item.description if item else None
        return desc

    def to_dict(self) -> dict[str, Any]:
        """将StoreModel转换为字典（深拷贝）"""
        result = {}
        type_hints = self.get_type_hints()
        for field_name in type_hints.keys():
            if field_name in self.__dict__:
                field = super().__getattribute__(field_name)
                if isinstance(field, StoreField):
                    result[field_name] = self.__getattribute__(field_name)
                elif isinstance(field, StoreModel):
                    result[field_name] = field.to_dict()
        result['_frozen'] = self._frozen
        if (sprite_id := getattr(self, '_sprite_id', None)) is not None:
            result['_sprite_id'] = sprite_id
        result['_namespace'] = self._namespace
        result['_title'] = self._title
        result['_description'] = self._description
        return deepcopy(result)

    @classmethod
    def get_type_hints(cls) -> dict[str, type]:
        """获取 StoreModel 的类型提示

        Returns:
            包含字段名到类型的映射字典
        """
        global _store_type_hints_caches
        if cls not in _store_type_hints_caches:
            _store_type_hints_caches[cls] = get_type_hints(cls)
        return _store_type_hints_caches[cls]

    @classmethod
    def get_pydantic_model(cls) -> type[BaseModel]:
        """将 StoreModel 转换为 Pydantic BaseModel 类型

        Returns:
            动态创建的 Pydantic BaseModel 类型（frozen）
        """
        global _store_pydantic_model_caches
        if cls not in _store_pydantic_model_caches:
            model_name = cls.__name__ + 'BaseModel'

            type_hints = cls.get_type_hints()
            fields = {}

            for field_name, field_type in type_hints.items():
                field_def = cls.__dict__.get(field_name)
                if isinstance(field_def, StoreField):
                    field_kwargs = {}
                    if field_def.default is not Unset:
                        field_kwargs['default'] = field_def.default
                    if field_def.default_factory is not None:
                        field_kwargs['default_factory'] = field_def.default_factory
                    if field_def.description:
                        field_kwargs['description'] = field_def.description
                    if field_def.title:
                        field_kwargs['title'] = field_def.title
                    fields[field_name] = (field_type, Field(**field_kwargs))
                elif isinstance(field_type, type) and issubclass(field_type, StoreModel):
                    field_kwargs = {'default_factory': field_type}
                    if field_type._title:
                        field_kwargs['title'] = field_type._title
                    if field_type._description:
                        field_kwargs['description'] = field_type._description
                    fields[field_name] = (field_type.get_pydantic_model(), Field(**field_kwargs))

            _store_pydantic_model_caches[cls] = create_model(model_name, __config__=ConfigDict(frozen=True), **fields)

        return _store_pydantic_model_caches[cls]

    def to_pydantic(self) -> BaseModel:
        """将当前 StoreModel 实例转换为 Pydantic BaseModel 实例

        Returns:
            Pydantic BaseModel 实例
        """
        return self.get_pydantic_model().model_validate(self.to_dict())
        #return self.to_pydantic_model().model_construct(**self.to_dict())

    @classmethod
    def from_global_config(cls, plugin_name: str) -> Self:
        """从插件全局配置创建 StoreModel(frozen=True) 实例

        Args:
            plugin_name: 插件名称

        Returns:
            StoreModel(frozen=True) 实例
        """
        from become_human.config import global_config
        config = global_config.get(plugin_name, {})
        if not isinstance(config, dict):
            raise ValueError(f"插件 {plugin_name} 的全局配置不是字典类型")
        def _add_items(source: dict[str, Any], model: type[StoreModel], namespace: tuple[str, ...]) -> list[SimpleItem]:
            results = []
            type_hints = model.get_type_hints()
            for key, value in source.items():
                if key in type_hints.keys():
                    hint = type_hints[key]
                    if isinstance(hint, type) and issubclass(hint, StoreModel):
                        if not isinstance(value, dict):
                            raise ValueError(f"插件 {plugin_name} 的全局配置中 {key} 的值不是字典类型")
                        results.extend(_add_items(value, hint, namespace + hint._namespace))
                    else:
                        # 在config中已经验证过了，并且在StoreModel的初始化时也有验证逻辑（都是非strict）
                        results.append(SimpleItem(namespace=namespace, key=key, value={'value': value}))
                else:
                    logger.warning(f"插件 {plugin_name} 的全局配置中 {key} 不是 {model.__name__} 的字段，将被忽略")
            return results
        items = _add_items(config, cls, cls._namespace)
        return cls(items=items, frozen=True)

#_TYPE_HINTS_CACHE: WeakKeyDictionary[type, Dict[str, Any]] = WeakKeyDictionary()
_store_type_hints_caches: dict[type[StoreModel], dict[str, Any]] = {}

_store_pydantic_model_caches: dict[type[StoreModel], type[BaseModel]] = {}
