from typing import Callable, Union
from weakref import WeakMethod
from inspect import iscoroutinefunction
from become_human.utils import filter_kwargs
from become_human.names import PROJECT_NAME

ON_SPRITE_OUTPUT_EVENT = PROJECT_NAME + ':on_sprite_output'

class EventBus:
    """事件总线"""
    _event_subscribers: dict[str, set[Union[WeakMethod, Callable]]]
    _event_names: set[str]
    _sync_event_names: set[str]
    _initialized: bool

    def __init__(self) -> None:
        self._event_subscribers = {}
        self._event_names = set()
        self._event_names.add(ON_SPRITE_OUTPUT_EVENT)
        self._sync_event_names = set()
        self._initialized = False

    def register(self, event_name: str, sync_only: bool = False) -> None:
        """注册事件

        Args:
            event_name: 事件名称
            sync_only: 是否仅支持同步订阅者

        Raises:
            KeyError: 如果事件已注册
        """
        if self.is_registered(event_name):
            raise KeyError(f'事件 {event_name} 已注册')
        self._event_names.add(event_name)
        if sync_only:
            self._sync_event_names.add(event_name)

    def is_registered(self, event_name: str) -> bool:
        """检查事件是否已注册"""
        return event_name in self._event_names

    def is_sync_event(self, event_name: str) -> bool:
        """检查事件是否为同步事件

        Raises:
            KeyError: 如果事件未注册
        """
        if not self.is_registered(event_name):
            raise KeyError(f'事件 {event_name} 未注册')
        return event_name in self._sync_event_names

    def get_subscribers(self, event_name: str) -> list[Callable]:
        """获取事件的所有订阅者

        Raises:
            KeyError: 如果事件未注册
        """
        if not self.is_registered(event_name):
            raise KeyError(f'事件 {event_name} 未注册')
        return [callback() if isinstance(callback, WeakMethod) else callback for callback in self._event_subscribers.get(event_name, set())]

    def subscribe(self, event_name: str, callback: Callable) -> None:
        """订阅事件

        Args:
            event_name: 事件名称
            callback: 回调函数

        Raises:
            KeyError: 如果事件未注册
            ValueError: 如果事件为同步事件且回调函数为异步函数
            TypeError: 如果回调函数不可哈希
        """
        if self._initialized:
            if not self.is_registered(event_name):
                raise KeyError(f'事件 {event_name} 未注册')
            if self.is_sync_event(event_name) and iscoroutinefunction(callback):
                raise ValueError(f'事件 {event_name} 仅支持同步订阅者')
        try:
            hash(callback)
        except TypeError:
            raise TypeError('回调函数必须是可哈希的')
        try:
            ref = WeakMethod(callback)
        except TypeError:
            ref = callback
        self._event_subscribers.setdefault(event_name, set()).add(ref)

    def on(self, event_name: str):
        """事件订阅装饰器，效果等同于subscribe，仅适用于静态方法或模块级函数"""
        def decorator(func: Callable):
            self.subscribe(event_name, func)
            return func
        return decorator

    async def publish(self, event_name: str, *args, **kwargs) -> None:
        """发布事件

        Args:
            event_name: 事件名称
            *args: 事件参数
            **kwargs: 事件关键字参数

        Raises:
            KeyError: 如果事件未注册
        """
        if self._initialized:
            if not self.is_registered(event_name):
                raise KeyError(f'事件 {event_name} 未注册')
        for callback in self._event_subscribers.get(event_name, []):
            if isinstance(callback, WeakMethod):
                handler = callback()
            else:
                handler = callback

            filtered_kwargs = filter_kwargs(kwargs, handler)
            if iscoroutinefunction(handler):
                await handler(*args, **filtered_kwargs)
            else:
                handler(*args, **filtered_kwargs)

    def publish_sync(self, event_name: str, *args, **kwargs) -> None:
        """发布同步事件

        Args:
            event_name: 事件名称
            *args: 事件参数
            **kwargs: 事件关键字参数

        Raises:
            KeyError: 如果事件未注册
            ValueError: 如果为异步事件
        """
        if self._initialized:
            if not self.is_registered(event_name):
                raise KeyError(f'事件 {event_name} 未注册')
            if event_name not in self._sync_event_names:
                raise ValueError(f'{event_name} 不是异步事件，不能以同步的方式发布')
        for callback in self._event_subscribers.get(event_name, []):
            if isinstance(callback, WeakMethod):
                handler = callback()
            else:
                handler = callback
            filtered_kwargs = filter_kwargs(kwargs, handler)
            handler(*args, **filtered_kwargs)

    def unsubscribe(self, event_name: str, callback: Callable) -> None:
        """取消订阅事件

        Args:
            event_name: 事件名称
            callback: 回调函数

        Raises:
            KeyError: 如果事件未注册
        """
        if self._initialized and not self.is_registered(event_name):
                raise KeyError(f'事件 {event_name} 未注册')
        try:
            self._event_subscribers[event_name].remove(callback)
        except KeyError:
            try:
                self._event_subscribers[event_name].remove(WeakMethod(callback))
            except (KeyError, TypeError):
                pass

    def has_subscribers(self, event_name: str) -> bool:
        """检查事件是否有订阅者

        Args:
            event_name: 事件名称

        Returns:
            如果事件有订阅者则返回True，否则返回False
        """
        return bool(self._event_subscribers.get(event_name, []))

    def check_events(self) -> None:
        """检查是否订阅了未注册的事件，以及同步事件是否仅订阅了同步回调函数

        Raises:
            ValueError: 如果事件未注册
        """
        for event_name in self._event_names:
            if not self.is_registered(event_name):
                raise ValueError(f'事件 {event_name} 未注册')
            if self.is_sync_event(event_name):
                subscribers = self.get_subscribers(event_name)
                for callback in subscribers:
                    if iscoroutinefunction(callback):
                        raise ValueError(f'事件 {event_name} 仅支持同步订阅者，却订阅了异步回调函数 {callback}')

    def set_initialized(self) -> None:
        """设置事件总线为已初始化，并检查是否订阅了未注册的事件"""
        self._initialized = True
        self.check_events()


event_bus = EventBus()
