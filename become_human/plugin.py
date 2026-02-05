from typing import Callable, Union, Optional, Literal
from dataclasses import dataclass
from langchain_core.tools import BaseTool
from become_human.types.agent_manager import CallAgentKwargs
from become_human.tool import AgentTool
from become_human.store.base import StoreModel

@dataclass
class Cancelled:
    reason: Literal['interrupted', 'rejected', 'queuing', 'plugin']
    plugin_name: Optional[str] = None

class Plugin:
    """插件基类

    只有name属性是必须的"""
    name: str
    """插件识别名称，这是必须值，必须为类属性，必须是唯一的"""
    tools: list[Union[Callable, BaseTool, AgentTool]]
    """插件提供的工具列表"""
    config: type[StoreModel]
    """插件配置存储模型，与data的区别是会出现在配置文件中"""
    data: type[StoreModel]
    """插件数据存储模型，与config的区别是不会出现在配置文件中"""
    commands: list[str] # TODO
    """插件提供的命令列表"""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not hasattr(cls, 'name'):
            raise ValueError(f"Plugin {cls.__name__} must have a name.")
        if cls.name in ['settings', 'plugins', 'init_on_startup']:
            raise ValueError(f"Plugin name {cls.name} is reserved.")

    async def on_manager_init(self) -> None:
        """插件在agent_manager初始化时要调用的方法"""
        ...
    async def on_manager_close(self) -> None:
        """插件在agent_manager关闭时要调用的方法"""
        ...
    async def on_agent_init(self, agent_id: str, /) -> None:
        """插件在每个agent初始化时要调用的方法"""
        ...
    async def on_agent_close(self, agent_id: str, /) -> None:
        """插件在每个agent关闭时要调用的方法"""
        ...
    async def before_call_agent(
        self,
        call_agent_kwargs: CallAgentKwargs,
        cancelled: Optional[Cancelled] = None,
    /) -> Optional[bool]:
        """插件在每次call_agent前要调用的方法"""
        ...
    async def after_call_agent(
        self,
        call_agent_kwargs: CallAgentKwargs,
        cancelled: Optional[Cancelled] = None,
    /) -> None:
        """插件在每次call_agent后要调用的方法"""
        ...
