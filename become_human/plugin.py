from typing import Callable, Union, Optional, Literal, ClassVar, Any, Self
from packaging.version import Version
from packaging.specifiers import SpecifierSet
from pydantic import BaseModel, Field, ConfigDict, field_validator, ValidationInfo
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, RemoveMessage
from langchain_core.tools import BaseTool
from become_human.types.manager import CallSpriteRequest, DoubleTextingStrategy
from become_human.tool import SpriteTool
from become_human.store.base import StoreModel
from become_human.message import add_messages

__all__ = [
    'BasePlugin',
    'PluginPriority',
    'PluginDependency',
    'PluginPrompt',
    'PluginPrompts',

    'BeforeCallSpriteInfo',
    'BeforeCallSpriteControl',

    'OnCallSpriteInfo',
    'OnCallSpriteControl',

    'BeforeCallModelInfo',
    #'BeforeCallModelControl',

    'AfterCallModelInfo',
    'AfterCallModelControl',

    'AfterCallToolsInfo',
    'AfterCallToolsControl',

    'AfterCallSpriteInfo',

    'OnUpdateMessagesInfo',
    'OnUpdateMessagesControl',
]




class FieldChange[T](BaseModel):
    """字段变更"""
    value: T
    """变更后的值"""
    plugin_name: str
    """变更插件名称"""

    model_config = ConfigDict(frozen=True)

CURRENT = object()
class ChangeableField[T](BaseModel):
    """可变更字段"""
    current: T
    """当前值"""
    changes: list[FieldChange[T]] = Field(default_factory=list)
    """变更记录"""
    original: T = Field(default=CURRENT, validate_default=True)
    """原始值"""
    model_config = ConfigDict(frozen=True)

    @field_validator('original', mode='before')
    @classmethod
    def _validate_original(cls, v: T, info: ValidationInfo) -> T:
        if v is CURRENT:
            return info.data['current']
        return v

    def _change(self, value: T, plugin_name: str) -> Self:
        """变更值"""
        return self.model_copy(update={
            'current': value,
            'changes': self.changes + [FieldChange(
                value=value,
                plugin_name=plugin_name
            )],
        })


UNSET = object()



class BaseControl(BaseModel):
    pass

class BaseInfo(BaseModel):
    """插件钩子信息基类，所有此类都是不可修改的"""

    model_config = ConfigDict(frozen=True)




# def _recover_messages_from_changes(
#     messages_changes: list[FieldChange[list[BaseMessage]]],
#     index: Union[int, str],
#     original_messages: list[BaseMessage]
# ):
#     if isinstance(index, str) and index not in [change.plugin_name for change in messages_changes]:
#         raise ValueError(f"index {index} not found in input_messages_changes")
#     elif isinstance(index, int) and (index < 0 or index >= len(messages_changes)):
#         raise ValueError(f"index {index} out of range")
#     elif not isinstance(index, (str, int)):
#         raise TypeError(f"index must be int or str, but got {type(index)}")
#     new_messages = original_messages
#     for i, change in enumerate(messages_changes):
#         change_messages = [m for m in change.value]
#         new_messages = add_messages(new_messages, change_messages)
#         if isinstance(index, str) and change.plugin_name == index:
#             break
#         elif isinstance(index, int) and i == index:
#             break
#     return new_messages



class BeforeCallSpriteControl(BaseControl):
    cancel: bool = Field(default=UNSET)
    """是否取消call_sprite的执行"""
    keep_input_messages: bool = Field(default=UNSET)
    """如果取消call_sprite的执行，是否也保留input_messages，否则input_messages不会更新至sprite的messages"""
    double_texting_strategy: DoubleTextingStrategy = Field(default=UNSET)
    """双重短信（并发）策略，即是否允许打断已在运行的sprite图"""

class BeforeCallSpriteInfo(BaseInfo):
    cancel_ctrl: ChangeableField[bool] = Field(default_factory=lambda: ChangeableField(current=False))
    keep_input_messages_ctrl: ChangeableField[bool] = Field(default_factory=lambda: ChangeableField(current=False))
    double_texting_strategy_ctrl: ChangeableField[DoubleTextingStrategy]

    def _update_from_control(self, control: BeforeCallSpriteControl, plugin_name: str) -> Self:
        new = {}
        if control.cancel is not UNSET:
            new['cancel_ctrl'] = self.cancel_ctrl._change(control.cancel, plugin_name)
        if control.keep_input_messages is not UNSET:
            new['keep_input_messages_ctrl'] = self.keep_input_messages_ctrl._change(control.keep_input_messages, plugin_name)
        if control.double_texting_strategy is not UNSET:
            new['double_texting_strategy_ctrl'] = self.double_texting_strategy_ctrl._change(control.double_texting_strategy, plugin_name)
        # 只进行浅拷贝
        return self.model_copy(update=new)



class OnCallSpriteControl(BaseControl):
    input_messages_patch: list[BaseMessage] = Field(default=UNSET)
    """call_sprite的input_messages补丁，将通过add_messages合并到input_messages"""

class OnCallSpriteInfo(BaseInfo):
    is_update_messages_only: bool = Field(default=False)
    """是否仅更新messages，而不调用sprite"""
    reason_of_update_messages_only: Optional[Literal['merged', 'before_call_sprite']] = None
    """仅更新messages的原因"""

    input_messages_ctrl: ChangeableField[list[BaseMessage]]

    def _update_from_control(self, control: OnCallSpriteControl, plugin_name: str, sprite_id: str) -> Self:
        new = {}
        if control.input_messages_patch is not UNSET:
            new['input_messages_ctrl'] = self.input_messages_ctrl._change(
                add_messages(
                    self.input_messages_ctrl.current,
                    control.input_messages_patch,
                    sprite_id=sprite_id
                ),
                plugin_name
            )
        # 只进行浅拷贝
        return self.model_copy(update=new)



class AfterCallSpriteInfo(BaseInfo):
    cancelled: bool = False
    cancelled_reason: Optional[Literal['sprite_running', 'interrupted', 'before_call_sprite', 'after_call_tools']] = None
    cancelled_by_plugin: Optional[str] = None
    new_messages: Optional[list[BaseMessage]] = None



# class BeforeCallModelControl(BaseControl):
#     """在调用sprite前的模型控制"""
#     core_prompt: tuple[str, str] = Field(default=UNSET, description='自定义core prompt标题和内容')
#     """自定义prompt标题和内容"""
#     secondary_prompt: tuple[str, str] = Field(default=UNSET, description='自定义secondary prompt标题和内容')
#     """自定义secondary prompt标题和内容"""

class BeforeCallModelInfo(BaseInfo):
    interrupted: bool = False



class AfterCallModelControl(BaseControl):
    """在调用模型后对消息的控制"""
    response: Optional[AIMessage] = Field(default=UNSET, description='修改过的模型响应，若为None则清除这条消息')
    """如果要修改模型响应，请深拷贝原响应，若为None则清除这条消息"""

class AfterCallModelInfo(BaseInfo):
    interrupted: bool = False
    response_ctrl: ChangeableField[Optional[AIMessage]] = Field(default_factory=lambda: ChangeableField(current=None))

    def _update_from_control(self, control: AfterCallModelControl, plugin_name: str) -> Self:
        new = {}
        if control.response is not UNSET:
            new['response_ctrl'] = self.response_ctrl._change(control.response, plugin_name)
        # 只进行浅拷贝
        return self.model_copy(update=new)



class AfterCallToolsControl(BaseControl):
    """在调用工具后对消息的控制"""
    exit_loop: bool = Field(default=UNSET)
    """是否主动跳出循环，取消call_sprite的执行"""
    tool_responses_patch: list[ToolMessage] = Field(default=UNSET)
    """修改过的工具响应"""
    #updated_new_messages_patch: list[BaseMessage] = Field(default=UNSET)
    #"""中途更新进来的修改过的消息"""

class AfterCallToolsInfo(BaseInfo):
    exit_loop_ctrl: ChangeableField[bool]
    tool_responses_ctrl: ChangeableField[list[ToolMessage]]

    def _update_from_control(self, control: AfterCallToolsControl, plugin_name: str, sprite_id: str) -> Self:
        new = {}
        if control.exit_loop is not UNSET:
            new['exit_loop_ctrl'] = self.exit_loop_ctrl._change(control.exit_loop, plugin_name)
        if control.tool_responses_patch is not UNSET:
            new['tool_responses_ctrl'] = self.tool_responses_ctrl._change(
                add_messages(self.tool_responses_ctrl.current, control.tool_responses_patch, sprite_id=sprite_id),
                plugin_name
            )
        # 只进行浅拷贝
        return self.model_copy(update=new)



class OnUpdateMessagesControl(BaseControl):
    """在更新消息后对消息的控制"""
    messages_patch: Optional[list[BaseMessage]] = Field(default=UNSET, description='修改过的消息')
    """如果要修改消息，请深拷贝原消息，若为None则清除这些消息"""

class OnUpdateMessagesInfo(BaseInfo):
    messages_ctrl: ChangeableField[list[BaseMessage]]

    def _update_from_control(self, control: OnUpdateMessagesControl, plugin_name: str, sprite_id: str) -> Self:
        new = {}
        if control.messages_patch is not UNSET:
            new['messages_ctrl'] = self.messages_ctrl._change(
                add_messages(self.messages_ctrl.current, control.messages_patch, sprite_id=sprite_id),
                plugin_name
            )
        # 只进行浅拷贝
        return self.model_copy(update=new)





class PluginPriority(BaseModel):
    phase: Literal['earliest', 'early', 'normal', 'late', 'last'] = Field(default='normal', description='插件优先级')
    offset: int = Field(default=0, ge=-100, le=100, description='插件优先级偏移量')

    model_config = ConfigDict(frozen=True)

    PHASE_MAP: ClassVar[dict[Literal['earliest', 'early', 'normal', 'late', 'last'], int]] = {
        'earliest': -600,
        'early': -300,
        'normal': 0,
        'late': 300,
        'last': 600,
    }

    def get_priority(self) -> int:
        """获取插件优先级，优先级越低=越早"""
        return self.PHASE_MAP[self.phase] + self.offset

    @staticmethod
    def sort_plugins_by_priority(
        plugins: list[Union[type['BasePlugin'], 'BasePlugin']]
    ) -> list[Union[type['BasePlugin'], 'BasePlugin']]:
        """根据插件优先级排序，优先级越低=越早=越靠前"""
        return sorted(plugins, key=lambda plugin: plugin.priority.get_priority())

class PluginDependency(BaseModel):
    name: str
    """依赖插件名称"""
    specifiers: str = Field(default='')
    """依赖插件版本规格"""

    @field_validator('name', mode='after')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError(f"Plugin name cannot be empty.")
        return v

    @field_validator('specifiers', mode='after')
    @classmethod
    def validate_specifiers(cls, v: str) -> str:
        try:
            SpecifierSet(v)
        except ValueError as e:
            raise ValueError(f"Plugin version specifiers {v} is invalid.") from e
        return v

    def is_dependency_met(self, plugins_with_name: dict[str, 'BasePlugin']) -> bool:
        """检查插件依赖是否满足"""
        if self.name not in plugins_with_name:
            return False
        plugin = plugins_with_name[self.name]
        return SpecifierSet(self.specifiers).contains(plugin.version)

    @staticmethod
    def check_dependencies(plugins_with_name: dict[str, 'BasePlugin']) -> None:
        """检查所有插件的依赖是否满足

        Raises:
            ValueError: 插件依赖不满足
        """
        for plugin in plugins_with_name.values():
            if hasattr(plugin, 'dependencies'):
                for dep in plugin.dependencies:
                    if not dep.is_dependency_met(plugins_with_name):
                        raise ValueError(f"Plugin {plugin.name} depends on {dep.name}{dep.specifiers} but "
                                         "it is not found." if dep.name not in plugins_with_name else
                                         f"found {dep.name}{plugins_with_name[dep.name].version}")

    @staticmethod
    def check_enbaled_plugins_dependencies(enabled_plugins: list[str], plugins_with_name: dict[str, 'BasePlugin']) -> None:
        """检查启用的插件的依赖是否满足

        Args:
            enabled_plugins: 启用的插件名称列表
            plugins_with_name: 所有插件名称到插件实例的映射

        Raises:
            ValueError: 插件依赖不满足
        """
        PluginDependency.check_dependencies({name: plugins_with_name[name] for name in set(enabled_plugins) if name in plugins_with_name})

class PluginPrompt(BaseModel):
    """插件提示"""
    title: str
    """插件提示名称"""
    content: str
    """插件提示内容"""

    @field_validator('title', mode='after')
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(f"Plugin prompt title cannot be empty.")
        return v

    @field_validator('content', mode='after')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(f"Plugin prompt content cannot be empty.")
        return v

class PluginPrompts(BaseModel):
    """插件提示"""
    core: Optional[PluginPrompt] = None
    """插件核心提示"""
    secondary: Optional[PluginPrompt] = None
    """插件次要提示"""
    role: Optional[PluginPrompt] = None
    """插件角色提示"""

class BasePlugin:
    """插件基类

    只有name属性是必须的"""
    name: str
    """插件识别名称，这是必须值，必须为类属性，必须是唯一的"""
    version: Version = Version('0.0.1')
    """插件版本，必须为类属性"""
    priority: PluginPriority = PluginPriority()
    """插件优先级，数值越大，优先级越高。必须为类属性，如无特殊需求请保持默认"""
    dependencies: list[PluginDependency]
    """插件依赖列表，必须为类属性，可选，也可手动检查依赖"""
    tools: list[Union[Callable, BaseTool, SpriteTool]]
    """插件提供的工具列表"""
    config: type[StoreModel]
    """插件配置存储模型，与data的区别是会出现在配置文件中"""
    data: type[StoreModel]
    """插件数据存储模型，与config的区别是不会出现在配置文件中"""
    commands: list[str] # TODO
    """插件提供的命令列表"""
    prompts: PluginPrompts
    """插件提示，会被注入到系统提示词"""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not hasattr(cls, 'name'):
            raise TypeError(f"Plugin {cls.__name__} must have a name.")
        if not isinstance(cls.name, str):
            raise TypeError(f"Plugin name {cls.name} must be a string.")
        if not cls.name:
            raise TypeError(f"Plugin name cannot be empty.")
        elif cls.name in ['settings', 'plugins', 'init_on_startup', 'prompts']:
            raise TypeError(f"Plugin name {cls.name} is reserved.")

        if not hasattr(cls, 'version'):
            raise TypeError(f"Plugin {cls.__name__} must have version.")
        if isinstance(cls.version, str):
            cls.version = Version(cls.version)
        elif not isinstance(cls.version, Version):
            raise TypeError(f"Plugin version {cls.version} must be a Version instance.")

        if not hasattr(cls, 'priority'):
            raise TypeError(f"Plugin {cls.__name__} must have priority.")
        if not isinstance(cls.priority, PluginPriority):
            raise TypeError(f"Plugin priority {cls.priority} must be a PluginPriority instance.")

        if hasattr(cls, 'dependencies'):
            if not isinstance(cls.dependencies, list):
                raise TypeError(f"Plugin dependencies {cls.dependencies} must be a list.")
            for dep in cls.dependencies:
                if not isinstance(dep, PluginDependency):
                    raise TypeError(f"Plugin dependency {dep} must be a PluginDependency instance.")

        if hasattr(cls, 'prompts'):
            if not isinstance(cls.prompts, PluginPrompts):
                raise TypeError(f"Plugin prompts {cls.prompts} must be a PluginPrompts instance.")


    async def on_manager_init(self) -> None:
        """插件在sprite_manager初始化时要调用的方法"""
        ...
    async def on_manager_close(self) -> None:
        """插件在sprite_manager关闭时要调用的方法"""
        ...
    async def on_sprite_init(self, sprite_id: str, /) -> None:
        """插件在每个sprite初始化时要调用的方法"""
        ...
    async def on_sprite_close(self, sprite_id: str, /) -> None:
        """插件在每个sprite关闭时要调用的方法"""
        ...
    async def before_call_sprite(
        self,
        request: CallSpriteRequest,
        info: BeforeCallSpriteInfo,
    /) -> Optional[BeforeCallSpriteControl]:
        """插件在每次call_sprite前要调用的方法"""
        ...
    async def after_call_sprite(
        self,
        request: CallSpriteRequest,
        info: AfterCallSpriteInfo,
    /) -> None:
        """插件在每次call_sprite后要调用的方法"""
        ...
    async def on_call_sprite(
        self,
        request: CallSpriteRequest,
        info: OnCallSpriteInfo,
    /) -> Optional[OnCallSpriteControl]:
        """插件在每次图将要进入ReAct前要调用的方法

        这个方法不会被打断"""
        ...
    async def before_call_model(
        self,
        request: CallSpriteRequest,
        info: BeforeCallModelInfo,
    /) -> None:
        """插件在每次调用模型前要调用的方法

        cancelled reason只可能是interrupted"""
        ...
    async def after_call_model(
        self,
        request: CallSpriteRequest,
        info: AfterCallModelInfo,
    /) -> Optional[AfterCallModelControl]:
        """插件在每次调用模型后要调用的方法

        一旦出现了response，就意味着当前进入到不可打断的状态了"""
        ...
    async def after_call_tools(
        self,
        request: CallSpriteRequest,
        info: AfterCallToolsInfo,
    /) -> Optional[AfterCallToolsControl]:
        """插件在每次调用工具后要调用的方法"""
        ...
    async def on_update_messages(
        self,
        sprite_id: str,
        info: OnUpdateMessagesInfo,
    /) -> Optional[OnUpdateMessagesControl]:
        """插件在每次sprite被调用update_messages后要调用的方法"""
        ...
    async def on_sprite_reset(self, sprite_id: str, /) -> None:
        """插件在每个sprite重置时要调用的方法，这会在close、删除所有schedule后，删除thread、store以及init前调用"""
        ...
