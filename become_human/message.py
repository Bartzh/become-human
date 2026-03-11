# 显而易见的，add_messages的代码来自langgraph（MIT许可），尽管目前没有修改add_messages的需求了
# https://github.com/langchain-ai/langgraph

from secrets import token_hex
from typing import Any, Optional, Literal, Self, Union, cast, ClassVar, override, overload
from abc import ABC, abstractmethod
import uuid
from copy import deepcopy
from pydantic import BaseModel, Field, ValidationError, field_validator
from loguru import logger
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
    AnyMessage,
    convert_to_messages,
    message_chunk_to_message,
    RemoveMessage,
    BaseMessageChunk,
    ContentBlock
)
from langgraph.graph.message import _add_messages_wrapper, _format_messages, Messages, REMOVE_ALL_MESSAGES
from become_human.times import Times, format_time
from become_human.utils import to_json_like_string, deep_dict_update, exclude_none_in_dict
from become_human.names import PROJECT_NAME
from become_human.store.manager import store_manager

SPRITES_MESSAGE_METADATA_KEY = PROJECT_NAME

MESSAGE_METADATAS_KEY = PROJECT_NAME + '_msg_metas'

DEFAULT_USER_MSG_TYPE = PROJECT_NAME + ':user'
DEFAULT_AI_MSG_TYPE = PROJECT_NAME + ':ai'
DEFAULT_TOOL_MSG_TYPE = PROJECT_NAME + ':tool'
DEFAULT_SYSTEM_MSG_TYPE = PROJECT_NAME + ':system'


class ABCMsgMeta(ABC, BaseModel):
    KEY: ClassVar[str]

    @classmethod
    @abstractmethod
    def parse(
        cls,
        message_or_additional_kwargs: Union[BaseMessage, dict, Self],
        /,
        *,
        is_artifact: bool = False
    ) -> Self:
        ...

    @abstractmethod
    def parse_with_default(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict, Self],
        /,
        *,
        is_artifact: bool = False
    ) -> Self:
        ...


    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: BaseMessage,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> BaseMessage:
        ...
    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: dict[str, Any],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> dict[str, Any]:
        ...
    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: Self,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> Self:
        ...

    @abstractmethod
    def fill_to(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict[str, Any], Self],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> Union[BaseMessage, dict[str, Any], Self]:
        ...


    @overload
    def update_to(
        self,
        message_or_additional_kwargs: BaseMessage,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> BaseMessage:
        ...
    @overload
    def update_to(
        self,
        message_or_additional_kwargs: dict[str, Any],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> dict[str, Any]:
        ...
    @overload
    def update_to(
        self,
        message_or_additional_kwargs: Self,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> Self:
        ...

    @abstractmethod
    def update_to(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict[str, Any], Self],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> Union[BaseMessage, dict[str, Any], Self]:
        ...


    @overload
    def set_to(
        self,
        message_or_additional_kwargs: BaseMessage,
        /,
        *,
        is_artifact: bool = False
    ) -> BaseMessage:
        ...
    @overload
    def set_to(
        self,
        message_or_additional_kwargs: dict[str, Any],
        /,
        *,
        is_artifact: bool = False
    ) -> dict[str, Any]:
        ...

    @abstractmethod
    def set_to(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict[str, Any]],
        /,
        *,
        is_artifact: bool = False
    ) -> Union[BaseMessage, dict[str, Any]]:
        ...

class BaseMsgMeta(ABCMsgMeta):
    """Base message metadata"""
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)  # 虽然 BaseModel 的 __init_subclass__ 默认无操作，但建议保留以兼容未来变更
        if not hasattr(cls, 'KEY'):
            raise TypeError(f"Class {cls.__name__} must define a 'KEY' class variable.")
        if not isinstance(getattr(cls, 'KEY'), str):
            raise TypeError(f"Class {cls.__name__}'s 'KEY' must be a string.")

    @override
    @classmethod
    def parse(cls, message_or_additional_kwargs: Union[BaseMessage, dict, ABCMsgMeta], /, *, is_artifact: bool = False) -> Self:
        """可输入消息或消息的additional_kwargs，解析为BaseMsgMeta（`cls.model_validate(additional_kwargs[cls.KEY], strict=True)`）

        也允许输入BaseMsgMeta是因为类型可能还没有被checkpointer转换为dict，如果没有手动model_dump的话

        Raises:
            TypeError: 输入类型错误
            KeyError: `cls.KEY not in additional_kwargs`
            pydantic.ValidationError: `additional_kwargs[cls.KEY]`的格式错误，验证失败"""
        return _parse(message_or_additional_kwargs, cls.KEY, cls, is_artifact=is_artifact)

    @override
    def parse_with_default(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict, ABCMsgMeta],
        /,
        *,
        is_artifact: bool = False
    ) -> Self:
        """可输入消息或消息的additional_kwargs，解析为BaseMsgMeta

        与parse的区别在于当`self.KEY not in additional_kwargs`时，返回自身实例的深拷贝而不是抛出KeyError"""
        try:
            return self.parse(message_or_additional_kwargs, is_artifact=is_artifact)
        except KeyError:
            return self.model_copy(deep=True)

    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: BaseMessage,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> BaseMessage:
        ...
    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: dict[str, Any],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> dict[str, Any]:
        ...
    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: Self,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> Self:
        ...
    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: 'DictMsgMeta',
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> 'DictMsgMeta':
        ...

    @override
    def fill_to(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict[str, Any], Self, 'DictMsgMeta'],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> Union[BaseMessage, dict[str, Any], Self, 'DictMsgMeta']:
        """将self的字段填充到目标中（会对填充后的结果进行严格验证，验证失败会抛出ValueError）

        这个方法会就地修改目标（所以这无法对为frozen的目标使用）

        在填充时，对于包括自己在内的所有BaseModel都会使用`exclude_none=True`的dump结果来填充，如果已经存在值则跳过，请注意这一点

        Args:
            message_or_additional_kwargs: 目标对象，可输入消息或消息的additional_kwargs，或BaseMsgMeta实例
            max_depth: 填充时对于dict（或BaseModel）的最大递归层数
                    0 = 仅顶层（等同于 dict.update）

                    -1 = 无限递归（完全深度合并）

                    N = 最多递归 N 层
            is_msgmeta: 当输入为dict时，是否直接将message_or_additional_kwargs视为BaseMsgMeta，而不是消息的additional_kwargs
            is_artifact: 当输入为dict时，是否直接将message_or_additional_kwargs视为artifact，而不是消息的additional_kwargs，不可与is_msgmeta同时为True

        Raises:
            TypeError: 输入类型错误
            ValueError: 输入值错误
        """
        return _fill_to(
            self,
            message_or_additional_kwargs,
            self.KEY,
            overwrite=False,
            max_depth=max_depth,
            is_artifact=is_artifact,
            is_msgmeta=is_msgmeta
        )


    @overload
    def update_to(
        self,
        message_or_additional_kwargs: BaseMessage,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> BaseMessage:
        ...
    @overload
    def update_to(
        self,
        message_or_additional_kwargs: dict[str, Any],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> dict[str, Any]:
        ...
    @overload
    def update_to(
        self,
        message_or_additional_kwargs: Self,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> Self:
        ...
    @overload
    def update_to(
        self,
        message_or_additional_kwargs: 'DictMsgMeta',
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> 'DictMsgMeta':
        ...

    @override
    def update_to(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict[str, Any], Self, 'DictMsgMeta'],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> Union[BaseMessage, dict[str, Any], Self, 'DictMsgMeta']:
        """将self的字段更新到目标中（会对填充后的结果进行严格验证，验证失败会抛出ValueError）

        这个方法会就地修改目标（所以这无法对为frozen的目标使用）

        在更新时，对于包括自己在内的所有BaseModel都会使用`exclude_none=True`的dump结果来更新，会覆盖所有已存在的值，请注意这一点

        Args:
            message_or_additional_kwargs: 目标对象，可输入消息或消息的additional_kwargs，或BaseMsgMeta实例
            max_depth: 填充时对于dict（或BaseModel）的最大递归层数
                    0 = 仅顶层（等同于 dict.update）

                    -1 = 无限递归（完全深度合并）

                    N = 最多递归 N 层
            is_msgmeta: 当输入为dict时，是否直接将message_or_additional_kwargs视为BaseMsgMeta，而不是消息的additional_kwargs
            is_artifact: 当输入为dict时，是否直接将message_or_additional_kwargs视为artifact，而不是消息的additional_kwargs，不可与is_msgmeta同时为True

        Raises:
            TypeError: 输入类型错误
            ValueError: 输入值错误
        """
        return _fill_to(
            self,
            message_or_additional_kwargs,
            self.KEY,
            overwrite=True,
            max_depth=max_depth,
            is_artifact=is_artifact,
            is_msgmeta=is_msgmeta
        )


    @overload
    def set_to(
        self,
        message_or_additional_kwargs: BaseMessage,
        /,
        *,
        is_artifact: bool = False
    ) -> BaseMessage:
        ...
    @overload
    def set_to(
        self,
        message_or_additional_kwargs: dict[str, Any],
        /,
        *,
        is_artifact: bool = False
    ) -> dict[str, Any]:
        ...

    @override
    def set_to(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict],
        /,
        *,
        is_artifact: bool = False
    ) -> Union[BaseMessage, dict]:
        """将self直接赋值到message_or_additional_kwargs中，就地修改"""
        return _set_to(self, message_or_additional_kwargs, self.KEY, is_artifact=is_artifact)


class DictMsgMeta(ABCMsgMeta):
    """从dict构建一个MsgMeta表示消息的metadata"""
    KEY: str = Field(description="metadata的key", frozen=True)
    value: dict[str, Any] = Field(default_factory=dict, description="metadata的value")

    @override
    def parse(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict, ABCMsgMeta],
        /,
        *,
        is_artifact: bool = False
    ) -> dict[str, Any]:
        """获取消息的metadata。这只会进行isinstance(v, dict)的验证，并返回深拷贝过的dict

        适用于在无法直接导入其他模块的情况下，需要获取消息的metadata的场景（如尝试获取其他插件定义的metadata）

        Args:
            message_or_additional_kwargs: 消息或additional_kwargs
            is_artifact: 是否从artifact中获取metadata

        Raises:
            TypeError: 输入类型错误
            KeyError: MsgMeta不存在
            pydantic.ValidationError: MsgMeta的格式错误，验证失败"""
        return _parse(message_or_additional_kwargs, self.KEY, is_artifact=is_artifact)

    @override
    def parse_with_default(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict, ABCMsgMeta],
    /) -> dict[str, Any]:
        """获取消息的metadata。如果metadata不存在，返回自身value的深拷贝"""
        try:
            return self.parse(message_or_additional_kwargs)
        except KeyError:
            return deepcopy(self.value)


    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: BaseMessage,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> BaseMessage:
        ...
    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: dict[str, Any],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> dict[str, Any]:
        ...
    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: BaseMsgMeta,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> BaseMsgMeta:
        ...
    @overload
    def fill_to(
        self,
        message_or_additional_kwargs: Self,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> Self:
        ...

    @override
    def fill_to(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict[str, Any], BaseMsgMeta, Self],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False,
    ) -> Union[BaseMessage, dict[str, Any], BaseMsgMeta, Self]:
        """将self.value填充到message_or_additional_kwargs中，就地修改"""
        return _fill_to(
            self.value,
            message_or_additional_kwargs,
            self.KEY,
            overwrite=False,
            max_depth=max_depth,
            is_artifact=is_artifact,
            is_msgmeta=is_msgmeta
        )


    @overload
    def update_to(
        self,
        message_or_additional_kwargs: BaseMessage,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> BaseMessage:
        ...
    @overload
    def update_to(
        self,
        message_or_additional_kwargs: dict[str, Any],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> dict[str, Any]:
        ...
    @overload
    def update_to(
        self,
        message_or_additional_kwargs: BaseMsgMeta,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> BaseMsgMeta:
        ...
    @overload
    def update_to(
        self,
        message_or_additional_kwargs: Self,
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False
    ) -> Self:
        ...

    @override
    def update_to(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict[str, Any], BaseMsgMeta, Self],
        /,
        *,
        max_depth: int = 0,
        is_artifact: bool = False,
        is_msgmeta: bool = False,
    ) -> Union[BaseMessage, dict[str, Any], BaseMsgMeta, Self]:
        """将self.value更新到message_or_additional_kwargs中，就地修改"""
        return _fill_to(
            self.value,
            message_or_additional_kwargs,
            self.KEY,
            overwrite=True,
            max_depth=max_depth,
            is_artifact=is_artifact,
            is_msgmeta=is_msgmeta
        )


    @overload
    def set_to(
        self,
        message_or_additional_kwargs: BaseMessage,
        /,
        *,
        is_artifact: bool = False
    ) -> BaseMessage:
        ...
    @overload
    def set_to(
        self,
        message_or_additional_kwargs: dict[str, Any],
        /,
        *,
        is_artifact: bool = False
    ) -> dict[str, Any]:
        ...

    @override
    def set_to(
        self,
        message_or_additional_kwargs: Union[BaseMessage, dict[str, Any]],
        /,
        *,
        is_artifact: bool = False
    ) -> Union[BaseMessage, dict[str, Any]]:
        """将self.value直接赋值到message_or_additional_kwargs中，就地修改"""
        return _set_to(self.value, message_or_additional_kwargs, self.KEY, is_artifact=is_artifact)


def _parse(
    target: Union[BaseMessage, dict[str, Any], ABCMsgMeta],
    key: str,
    source_type: Optional[type[BaseMsgMeta]] = None,
    /,
    *,
    is_artifact: bool = False
) -> Union[dict[str, Any], BaseMsgMeta]:
    _process_artifact(target)
    if isinstance(target, dict):
        if is_artifact:
            metas = target
        else:
            metas = target.get(MESSAGE_METADATAS_KEY, {})
    elif isinstance(target, BaseMessage):
        metas = target.additional_kwargs.get(MESSAGE_METADATAS_KEY, {})
    elif isinstance(target, BaseMsgMeta):
        if source_type is not None and not isinstance(target, source_type):
            raise TypeError(f"target must be a {source_type.__name__}")
        else:
            return target.model_copy(deep=True)
    elif isinstance(target, DictMsgMeta):
        return target.model_dump()['value']
    else:
        raise TypeError(f"target must be a BaseMessage or dict, or BaseMsgMeta")
    if key in metas:
        # pydantic.ValidationError
        meta = metas[key]
        if source_type is None:
            if isinstance(meta, BaseMsgMeta):
                return meta.model_dump(exclude_none=True)
            elif isinstance(meta, dict):
                dict_result = deepcopy(meta)
                exclude_none_in_dict(dict_result)
                return dict_result
            else:
                raise ValidationError(f"`metas[{key}]` is not a BaseMsgMeta or dict")
        else:
            # pydantic在验证时只会进行浅拷贝
            return source_type.model_validate(deepcopy(meta), strict=True)
    else:
        raise KeyError(f"No {key} found in metas")

def _fill_to(
    source: Union[BaseMsgMeta, dict[str, Any]],
    target: Union[BaseMessage, dict[str, Any], ABCMsgMeta],
    key: str,
    /,
    *,
    overwrite: bool = True,
    max_depth: int = 0,
    is_artifact: bool = False,
    is_msgmeta: bool = False,
) -> Union[BaseMessage, dict[str, Any], ABCMsgMeta]:
    _process_artifact(target)
    if is_artifact and is_msgmeta:
        raise ValueError("is_artifact and is_msgmeta cannot be True at the same time")
    if isinstance(target, dict):
        if is_msgmeta:
            current = target
        elif is_artifact:
            current = target.get(key, {})
        else:
            current = target.get(MESSAGE_METADATAS_KEY, {}).get(key, {})
    elif isinstance(target, BaseMessage):
        current = target.additional_kwargs.get(MESSAGE_METADATAS_KEY, {}).get(key, {})
    elif isinstance(target, BaseMsgMeta):
        current = target
    elif isinstance(target, DictMsgMeta):
        current = target.model_dump()['value']
    else:
        raise TypeError(f"target must be a BaseMessage or dict, or BaseMsgMeta or DictMsgMeta")

    source_is_basemsgmeta = isinstance(source, BaseMsgMeta)

    if isinstance(current, BaseMsgMeta):
        if source_is_basemsgmeta and not isinstance(current, source.__class__):
            raise ValueError(f"`additional_kwargs[{key}]` is another msgmeta: {current.__class__.__name__}")
        else:
            current = current.model_dump(exclude_none=True)
    elif not isinstance(current, dict):
        raise ValueError(f"`additional_kwargs[{key}]` is not a BaseMsgMeta or dict")
    else:
        exclude_none_in_dict(current)

    if source_is_basemsgmeta:
        dumped_source = source.model_dump(exclude_none=True)
    else:
        dumped_source = deepcopy(source)
        exclude_none_in_dict(dumped_source)

    # 一点小优化，当current为空时，直接填充self.model_dump()
    empty_current = not current
    if empty_current:
        current.update(dumped_source)
        result = current
    else:
        if overwrite:
            deep_dict_update(current, dumped_source, max_depth)
            result = current
        else:
            result = dumped_source
            deep_dict_update(result, current, max_depth)
        if source_is_basemsgmeta:
            try:
                result = source.__class__.model_validate(result, strict=True).model_dump(exclude_none=True)
            except ValidationError as e:
                raise ValueError(f"Invalid {source.__class__.__name__} data when fill_to/update_to: {e}")

    if isinstance(target, dict):
        if is_msgmeta:
            # 非overwrite情况下，result是一个新的dict
            # 将target的值替换为result，保持引用
            if not overwrite and not empty_current:
                target.clear()
                target.update(result)
            # overwrite或empty的情况下，target已经被就地修改了
            else:
                pass
        elif is_artifact:
            target[key] = result
        else:
            target.setdefault(MESSAGE_METADATAS_KEY, {})[key] = result
    elif isinstance(target, ABCMsgMeta):
        if not isinstance(target, DictMsgMeta):
            # 直接就地修改 Pydantic 模型
            for field_name, field_value in result.items():
                setattr(target, field_name, field_value)
            try:
                target.__class__.model_validate(target.__dict__, strict=True)
            except ValidationError as e:
                raise ValueError(f"Invalid {target.__class__.__name__} data when fill_to/update_to: {e}")
        else:
            target.value = result
    else:
        target.additional_kwargs.setdefault(MESSAGE_METADATAS_KEY, {})[key] = result
    return target

def _set_to(
    source: Union[BaseMsgMeta, dict[str, Any]],
    target: Union[BaseMessage, dict[str, Any]],
    key: str,
    /,
    *,
    is_artifact: bool = False
) -> Union[BaseMessage, dict[str, Any]]:
    _process_artifact(target)
    if isinstance(source, BaseMsgMeta):
        source = source.model_dump(exclude_none=True)
    else:
        source  = deepcopy(source)
        exclude_none_in_dict(source)
    if isinstance(target, dict):
        if is_artifact:
            target[key] = source
        else:
            target.setdefault(MESSAGE_METADATAS_KEY, {})[key] = source
    elif isinstance(target, BaseMessage):
        target.additional_kwargs.setdefault(MESSAGE_METADATAS_KEY, {})[key] = source
    else:
        raise TypeError(f"set_to target must be a BaseMessage or dict")
    return target

def _process_artifact(message: Any) -> None:
    """目前支持artifact为

    BaseMsgMeta, DictMsgMeta,

    list, tuple, set,

    dict[MESSAGE_METADATAS_KEY, Union[list, tuple, set]]"""
    if not isinstance(message, ToolMessage):
        return

    def _if_key_in_metas(message: BaseMessage, key: str) -> bool:
        if key in message.additional_kwargs.get(MESSAGE_METADATAS_KEY, {}):
            logger.error(f"ToolMessage 的 artifact 和 additional_kwargs 同时存在msgmeta {key}，这是不允许的。将忽略并删除 artifact 中的值。")
            return True
        return False

    artifact = message.artifact
    if isinstance(artifact, (list, tuple, set)):
        # 确保遍历的是副本
        artifacts = list(artifact)
        artifact_type = 'list'
    # UserDict 不是 dict 的子类，这样判断没有问题
    elif isinstance(artifact, dict):
        artifacts = artifact.get(MESSAGE_METADATAS_KEY, [])
        if not isinstance(artifacts, (list, tuple, set)):
            logger.error(f"ToolMessage 的 artifact 若为 dict，其中的 {MESSAGE_METADATAS_KEY} 必须是 list, tuple, set 类型，当前类型为 {type(artifacts)}，将直接移除该字段。")
            del message.artifact[MESSAGE_METADATAS_KEY]
            return
        # 确保遍历的是副本
        artifacts = list(artifacts)
        artifact_type = 'dict'
    else:
        artifacts = [artifact]
        artifact_type = 'meta'

    for i, artifact in enumerate(artifacts):
        if isinstance(artifact, (BaseMsgMeta, DictMsgMeta)):
            if isinstance(artifact, BaseMsgMeta):
                if not _if_key_in_metas(message, artifact.KEY):
                    message.additional_kwargs.setdefault(MESSAGE_METADATAS_KEY, {})[artifact.KEY] = artifact.model_dump(exclude_none=True)
            else:
                if not _if_key_in_metas(message, artifact.KEY):
                    dict_meta = deepcopy(dict(artifact))
                    exclude_none_in_dict(dict_meta)
                    message.additional_kwargs.setdefault(MESSAGE_METADATAS_KEY, {})[artifact.KEY] = dict_meta
            if artifact_type == 'list':
                del message.artifact[i]
            elif artifact_type == 'dict':
                del message.artifact[MESSAGE_METADATAS_KEY][i]
            else:
                message.artifact = None



class SpritesMsgMeta(BaseMsgMeta):
    """Metadata for a Sprites message."""

    creation_times: Times
    """消息创建时间"""
    message_type: Optional[str] = Field(default=None)
    """消息类型。建议保持 provider:type 的格式"""
    is_action_only_tool: Optional[bool] = Field(default=None)
    """sprite是否不用特意再看一遍此(工具)消息"""

    KEY: ClassVar = SPRITES_MESSAGE_METADATA_KEY


class SpritesMsgMetaOptionalTimes(SpritesMsgMeta):
    """主要作用是使SpritesMsgMeta可以不填写时间，由之后的节点来添加时间信息。同时还有一个默认的消息类型

    以及在parse时防止报错

    最终还是会使用SpritesMsgMeta，这个结构只是方便输入

    目前支持的场景有BaseTool、InitalAIMessage、InitalToolCall、construct_system_message"""
    creation_times: Optional[Times] = Field(default=None)




@_add_messages_wrapper
def add_messages(
    left: Messages,
    right: Messages,
    *,
    format: Optional[Literal["langchain-openai"]] = None,
    sprite_id: Optional[str] = None,
) -> Messages:
    """Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Args:
        left: The base list of messages.
        right: The list of messages (or single message) to merge
            into the base list.
        format: The format to return messages in. If None then messages will be
            returned as is. If 'langchain-openai' then messages will be returned as
            BaseMessage objects with their contents formatted to match OpenAI message
            format, meaning contents can be string, 'text' blocks, or 'image_url' blocks
            and tool responses are returned as their own ToolMessages.

            !!! important "Requirement"

                Must have ``langchain-core>=0.3.11`` installed to use this feature.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`.

    Example:
        ```python title="Basic usage"
        from langchain_core.messages import AIMessage, HumanMessage
        msgs1 = [HumanMessage(content="Hello", id="1")]
        msgs2 = [AIMessage(content="Hi there!", id="2")]
        add_messages(msgs1, msgs2)
        # [HumanMessage(content='Hello', id='1'), AIMessage(content='Hi there!', id='2')]
        ```

        ```python title="Overwrite existing message"
        msgs1 = [HumanMessage(content="Hello", id="1")]
        msgs2 = [HumanMessage(content="Hello again", id="1")]
        add_messages(msgs1, msgs2)
        # [HumanMessage(content='Hello again', id='1')]
        ```

        ```python title="Use in a StateGraph"
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph

        class State(TypedDict):
            messages: Annotated[list, add_messages]

        builder = StateGraph(State)
        builder.add_node("chatbot", lambda state: {"messages": [("assistant", "Hello")]})
        builder.set_entry_point("chatbot")
        builder.set_finish_point("chatbot")
        graph = builder.compile()
        graph.invoke({})
        # {'messages': [AIMessage(content='Hello', id=...)]}
        ```

        ```python title="Use OpenAI message format"
        from typing import Annotated
        from typing_extensions import TypedDict
        from langgraph.graph import StateGraph, add_messages

        class State(TypedDict):
            messages: Annotated[list, add_messages(format='langchain-openai')]

        def chatbot_node(state: State) -> list:
            return {"messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here's an image:",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "1234",
                            },
                        },
                    ]
                },
            ]}

        builder = StateGraph(State)
        builder.add_node("chatbot", chatbot_node)
        builder.set_entry_point("chatbot")
        builder.set_finish_point("chatbot")
        graph = builder.compile()
        graph.invoke({"messages": []})
        # {
        #     'messages': [
        #         HumanMessage(
        #             content=[
        #                 {"type": "text", "text": "Here's an image:"},
        #                 {
        #                     "type": "image_url",
        #                     "image_url": {"url": "data:image/jpeg;base64,1234"},
        #                 },
        #             ],
        #         ),
        #     ]
        # }
        ```

    """
    remove_all_idx = None
    # coerce to list
    if not isinstance(left, list):
        left = [left]  # type: ignore[assignment]
    if not isinstance(right, list):
        right = [right]  # type: ignore[assignment]
    # coerce to message
    left = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(left)
    ]
    right = [
        message_chunk_to_message(cast(BaseMessageChunk, m))
        for m in convert_to_messages(right)
    ]
    # assign missing ids
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for idx, m in enumerate(right):
        if m.id is None:
            m.id = str(uuid.uuid4())
        if isinstance(m, RemoveMessage) and m.id == REMOVE_ALL_MESSAGES:
            remove_all_idx = idx

    # 修改处
    messages_post_processing(right, sprite_id)

    if remove_all_idx is not None:
        return right[remove_all_idx + 1 :]

    # merge
    merged = left.copy()
    merged_by_id = {m.id: i for i, m in enumerate(merged)}
    ids_to_remove = set()
    for m in right:
        if (existing_idx := merged_by_id.get(m.id)) is not None:
            if isinstance(m, RemoveMessage):
                ids_to_remove.add(m.id)
            else:
                ids_to_remove.discard(m.id)
                merged[existing_idx] = m
        else:
            if isinstance(m, RemoveMessage):
                raise ValueError(
                    f"Attempting to delete a message with an ID that doesn't exist ('{m.id}')"
                )

            merged_by_id[m.id] = len(merged)
            merged.append(m)
    merged = [m for m in merged if m.id not in ids_to_remove]

    if format == "langchain-openai":
        merged = _format_messages(merged)
    elif format:
        msg = f"Unrecognized {format=}. Expected one of 'langchain-openai', None."
        raise ValueError(msg)
    else:
        pass

    return merged

def messages_post_processing(messages: list[BaseMessage], sprite_id: Optional[str] = None):
    for m in messages:
        for key, value in m.additional_kwargs.get(MESSAGE_METADATAS_KEY, {}).items():
            if isinstance(value, BaseMsgMeta):
                m.additional_kwargs[MESSAGE_METADATAS_KEY][key] = value.model_dump(exclude_none=True)
            elif isinstance(value, DictMsgMeta):
                dict_meta = value.model_dump()['value']
                exclude_none_in_dict(dict_meta)
                m.additional_kwargs[MESSAGE_METADATAS_KEY][key] = dict_meta
    if sprite_id:
        current_times = Times.from_time_settings(store_manager.get_settings(sprite_id).time_settings)
        default_meta = SpritesMsgMeta(
            creation_times=current_times
        )
        for m in messages:
            default_meta.fill_to(m)
    return messages



def format_human_message(message: HumanMessage) -> str:
    return '<others>\n' + "\n".join(extract_text_parts(message.content)) + '\n</others>'

def format_ai_message(message: AIMessage) -> str:
    message_string = "<AI>\n"
    if message.tool_calls:
        for tool_call in message.tool_calls:
            message_string += f'''<action name="{tool_call['name']}" datetime="{format_time(SpritesMsgMeta.parse(message).creation_times.sprite_world_datetime)}">
<args>
{to_json_like_string(tool_call['args'])}
</args>
</action>\n'''
    return message_string.strip() + '\n</AI>'

def format_tool_message(message: ToolMessage) -> str:
    metadata = SpritesMsgMeta.parse(message)
    feedback_content = '\n'.join(extract_text_parts(message.content))
    return f'''<action name="{message.name}" datetime="{format_time(metadata.creation_times.sprite_world_datetime)}>
<feedback>
{feedback_content}
</feedback>
</action>'''

def format_message(message: AnyMessage) -> str:
    if isinstance(message, HumanMessage):
        return format_human_message(message)
    elif isinstance(message, AIMessage):
        return format_ai_message(message)
    elif isinstance(message, ToolMessage):
        return format_tool_message(message)
    return "<unsupported_message_type />"

def format_messages(
    messages: list[AnyMessage]
) -> str:
    return '\n\n\n'.join([format_message(m) for m in messages])


def extract_text_parts(content: Union[list[str], list[ContentBlock], str]) -> list[str]:
    contents = []
    if isinstance(content, str):
        contents.append(content)
    elif isinstance(content, list):
        for c in content:
            if isinstance(c, str):
                contents.append(c)
            elif isinstance(c, dict):
                if c.get("type") == "text" and isinstance(c.get("text"), str):
                    contents.append(c["text"])
    return contents

def convert_to_content_blocks(content: Union[str, list[str], list[ContentBlock]]) -> list[ContentBlock]:
    """只会处理content为str和list[str]的情况"""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    elif isinstance(content, list):
        content = content.copy()
        for i, c in enumerate(content):
            if isinstance(c, str):
                content[i] = {"type": "text", "text": c}
        return content
    else:
        raise TypeError(f"Unsupported content type: {type(content)}")


def construct_system_message(
    content: str | list[ContentBlock],
    times: Times,
    message_type: str = DEFAULT_SYSTEM_MSG_TYPE,
    extra_metas: Optional[list[ABCMsgMeta]] = None
) -> HumanMessage:
    """sprites_msg_metas 默认包含 SpritesMsgMeta 中的 creation_times、message_type，以及 MemoryMsgMeta 中的 do_not_store

    extra_metas 可以包含刚才默认包含的meta，将会与默认值合并，默认包含的字段会被相应值覆盖（若有）"""
    default_metadata = SpritesMsgMeta(
        message_type=message_type,
        creation_times=times
    )
    memory_metadata = DictMsgMeta(
        KEY='bh_memory',
        value={
            'do_not_store': True
        }
    )
    system_prefix = "**这条消息来自系统（system）自动发送**\n"
    if isinstance(content, str):
        if not content.startswith(system_prefix):
            content = system_prefix + content
    else:
        for c in content:
            if c['type'] == 'text':
                if not c['text'].startswith(system_prefix):
                    c['text'] = system_prefix + c['text']
    message = HumanMessage(
        content=content,
        name="system",
        id=str(uuid.uuid4())
    )
    if extra_metas:
        for msg_meta in extra_metas:
            msg_meta.set_to(message)
    default_metadata.fill_to(message)
    memory_metadata.fill_to(message)

    return message


def _validate_msg_metas(v: Optional[list[ABCMsgMeta]]) -> Optional[list[ABCMsgMeta]]:
    if v is None:
        return None
    new_v = []
    for msg_meta in v:
        if isinstance(msg_meta, (BaseMsgMeta, DictMsgMeta)):
            new_v.append(msg_meta)
        elif isinstance(msg_meta, dict):
            new_v.append(DictMsgMeta.model_validate(msg_meta))
        else:
            raise ValueError(f"msg_metas 中发现未知类型 {type(msg_meta)}")
    return new_v

class InitalToolCall(BaseModel):
    name: str = Field(description="工具名称")
    args: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    result_content: Union[str, dict[str, dict]] = Field(default=None, description="工具调用结果content")
    result_msg_metas: Optional[list[ABCMsgMeta]] = Field(default=None, description="工具调用结果消息元数据，默认已包含SpritesMsgMeta")

    @field_validator('result_msg_metas', mode='plain')
    @classmethod
    def validate_result_msg_metas(cls, v: Optional[list[ABCMsgMeta]]) -> Optional[list[ABCMsgMeta]]:
        return _validate_msg_metas(v)

class InitalAIMessage(BaseModel):
    """至少需要其中一项"""
    content: Union[str, dict[str, dict]] = Field(default='', description="内容")
    tool_calls: list[InitalToolCall] = Field(default_factory=list, description="工具调用列表")
    msg_metas: Optional[list[ABCMsgMeta]] = Field(default=None, description="消息元数据，默认已包含SpritesMsgMeta")

    @field_validator('msg_metas', mode='plain')
    @classmethod
    def validate_msg_metas(cls, v: Optional[list[ABCMsgMeta]]) -> Optional[list[ABCMsgMeta]]:
        return _validate_msg_metas(v)

    def construct_messages(self, times: Times) -> list[Union[AIMessage, ToolMessage]]:
        tool_calls_with_id = [{
            'name': tool_call.name,
            'args': tool_call.args,
            'id': 'call_' + token_hex(12),
            'result_content': tool_call.result_content,
            'result_msg_metas': tool_call.result_msg_metas,
        } for tool_call in self.tool_calls]


        additional_kwargs = {
            'tool_calls': [{
                'index': i,
                'id': tool_call['id'],
                'function': {
                    'arguments': tool_call['args'],
                    'name': tool_call['name']
                },
                'type': 'function'
            } for i, tool_call in enumerate(tool_calls_with_id)]
        }
        default_metadata = SpritesMsgMeta(
            message_type=DEFAULT_AI_MSG_TYPE,
            creation_times=times
        )
        if self.msg_metas:
            for msg_meta in self.msg_metas:
                msg_meta.set_to(additional_kwargs)
        default_metadata.fill_to(additional_kwargs)

        messages = [AIMessage(
            content=self.content,
            additional_kwargs=additional_kwargs,
            response_metadata={
                'finish_reason': 'tool_calls',
                'model_name': 'qwen-plus-2025-04-28'
            },
            tool_calls=[{
                'name': tool_call['name'],
                'args': tool_call['args'],
                'id': tool_call['id'],
                'type': 'tool_call'
            } for tool_call in tool_calls_with_id],
            id=str(uuid.uuid4())
        )]


        default_tool_metadata = SpritesMsgMeta(
            message_type=DEFAULT_TOOL_MSG_TYPE,
            creation_times=times
        )
        for tool_call in tool_calls_with_id:
            tool_message = ToolMessage(
                content=tool_call['result_content'],
                name=tool_call['name'],
                tool_call_id=tool_call['id'],
                id=str(uuid.uuid4())
            )
            if tool_call['result_msg_metas']:
                for msg_meta in tool_call['result_msg_metas']:
                    msg_meta.set_to(tool_message)
            default_tool_metadata.fill_to(tool_message)
            messages.append(tool_message)


        return messages
