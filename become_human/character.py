from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Literal, Optional, Annotated, Any, get_origin, get_args, Union, Self
from abc import ABC, abstractmethod
from loguru import logger

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage
from langchain.chat_models import BaseChatModel

from sprited.constants import UNSET
from sprited.utils import to_json_like_string, add_indent
from sprited.manager import sprite_manager
from sprited.plugin import *
from sprited.times import Times, TimestampUs, SerializableTimeZone, format_time
from sprited.store.manager import store_manager
from sprited.store.base import StoreModel, StoreField
from sprited.types import CallSpriteRequest

@dataclass(frozen=True)
class FieldUse:
    """
    字段使用
    """
    in_prompt: bool = False

    memory_type: str = "reflective"
    content: Optional[str] = None
    similarity_threshold: float = 0.3
    #retrievability: float = 0.7
    ttl: Optional[int] = None

class MemoryField[T](BaseModel):
    value: T = Field(description='字段值')
    memory_creation_time: datetime = Field(description='当出现这个时，意味着该字段的值（value）将会被加入记忆数据库，memory_creation_time会作为该记忆的创建时间，这需要你考虑这件事可能是何时被记住的')

def create_times(sprite_id: str, sprite_time: datetime) -> Times:
    """
    从AI生成的时间创建Times对象，目前假定输入是没有时区的datetime，也可以把这个函数改成输入字符串什么的

    用于要加入记忆的字段，每个要加入记忆的字段都应生成一个记忆产生的时间，但add_memory需要的是一个Times对象，所以需要这个函数
    """
    time_settings = store_manager.get_settings(sprite_id).time_settings
    now_timestampus = TimestampUs.now()
    now_times = Times.from_time_settings(time_settings, now_timestampus)
    time_settings = time_settings.model_copy()
    if sprite_time.tzinfo is None:
        sprite_time = sprite_time.replace(tzinfo=time_settings.time_zone.tz())
    time_settings.world_real_anchor = now_timestampus
    time_settings.world_sprite_anchor = TimestampUs(sprite_time)
    time_settings.subjective_real_anchor = now_timestampus
    time_settings.subjective_sprite_anchor = now_times.sprite_subjective_tick
    return Times(
        real_world_timestampus=now_timestampus,
        real_world_time_zone=SerializableTimeZone(name='UTC'),
        sprite_time_settings=time_settings,
    )


class ABCStep(ABC, BaseModel):
    """
    步骤基类
    """

    #model_config = ConfigDict(validate_assignment=)

    @abstractmethod
    def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
        """
        获取步骤提示词
        """
        ...

    async def generate(self, current: "CharacterProfile", llm: BaseChatModel, sprite_id: str) -> Self:
        """
        生成步骤信息
        """
        # 检查是否有未填写的字段
        generated = True
        for key in self.__class__.model_fields.keys():
            if getattr(self, key, UNSET) is UNSET:
                generated = False
                break
        if generated:
            return self
        def _recreate_model(model: Any) -> Any:
            # 处理泛型类型如 List[BaseModel], Dict[str, BaseModel]
            origin = get_origin(model)
            args = get_args(model)
            if origin is not None and args:
                new_args = []
                for arg in args:
                    new_arg = _recreate_model(arg)
                    if not isinstance(new_arg, FieldUse):
                        new_args.append(new_arg)
                if len(new_args) == 1 and origin is Annotated:
                    return new_args[0]
                else:
                    return origin[tuple(new_args)]
            elif isinstance(model, type):
                if issubclass(model, MemoryField):
                    return MemoryField[_recreate_model(model.model_fields['value'].annotation)]
                if issubclass(model, BaseModel):
                    fields = {}
                    # 重建模型，如果默认值是UNSET则删除，表示这是必填字段
                    # 这样做是为了让用户可以预填模型中任意字段（不必全部填写）
                    # 而对于LLM来说，所有UNSET字段都是必填的
                    for key, value in model.model_fields.items():
                        field = value.asdict()
                        # 泛型BaseModel不会提取metadata，所以需要手动处理
                        if get_origin(field["annotation"]) is Annotated:
                            args = get_args(field["annotation"])
                            field["metadata"].extend(args[1:])
                            field["annotation"] = args[0]
                        field["annotation"] = _recreate_model(field["annotation"])
                        if field["metadata"]:
                            # 去掉FieldUse类型的metadata，只保留普通metadata
                            metadata = [m for m in field["metadata"] if not isinstance(m, FieldUse)]
                            if metadata:
                                field["annotation"] = Annotated[field["annotation"], *metadata]
                        if field["attributes"]["default"] is UNSET:
                            del field["attributes"]["default"]
                        fields[key] = (field["annotation"], Field(**field["attributes"]))
                    if not fields:
                        raise ValueError(f"{model.__name__} has no fields.")
                    return create_model(
                        model.__name__,
                        __doc__=model.__doc__,
                        __base__=model,
                        **fields
                    )
            return model

        #model_for_llm = _recreate_model(self.__class__)
        model_for_llm = self.__class__
        agent = create_agent(llm, response_format=ToolStrategy(model_for_llm))
        result = None
        for _ in range(3):
            try:
                result = await agent.ainvoke({"messages": [HumanMessage(content=self.step_prompt(current, sprite_id))]})
                result = result.get("structured_response")
                if isinstance(result, model_for_llm):
                    break
                else:
                    logger.warning(f"第 {_+1} 次生成失败，返回结果不是 {model_for_llm.__name__}")
            except Exception:
                logger.exception(f"第 {_+1} 次生成 {self.__class__.__name__} 失败")
        if isinstance(result, model_for_llm):
            return self.__class__.model_validate(result.model_dump())
        raise ValueError(f"生成 {self.__class__.__name__} 失败")


Gender = Literal["male", "female", "other"]

LanguageProficiency = Literal["native", "fluent", "intermediate", "basic"]

AssetType = Literal["physical", "financial", "sentimental"]

RelationshipType = Literal["parent", "sibling", "partner", "friend", "other"]


class CoreDrive(ABCStep):
    core_desire: str = Field(description="角色最核心的欲望或追求")
    core_weakness: str = Field(description="角色最致命的弱点或软肋")

    def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
        return f"""请根据以下既定的角色设定生成一个角色的核心驱动力：
{current.format_context(sprite_id=sprite_id)}

核心驱动力定义了角色的灵魂——他们最渴望什么，以及他们最容易在哪里失败。
请确保欲望与弱点之间存在内在联系，弱点往往是实现欲望的阻碍。
只输出核心驱动力，不要预设具体的身份、外貌或背景。"""


class LifeExperience(BaseModel):
    #period: str = Field(description="时间段（如 2010-2014）")
    event: Annotated[MemoryField[str], FieldUse(memory_type='original')] = Field(description="关键事件（要求客观叙述但保持第一人称视角）")
    impact: str = Field(description="对角色核心特质的影响")

class Experiences(ABCStep):
    life_experiences: List[LifeExperience] = Field(description="人生经历列表（2-4个关键事件）")

    def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
        return f"""已知角色的部分设定：
{current.format_context(sprite_id=sprite_id)}

请推断这个角色可能经历的关键人生事件。这些事件应该：
1. 能够解释其核心欲望的形成原因
2. 反映其弱点如何影响人生轨迹
3. 具备戏剧张力，适合作为故事背景

生成 2-4 个关键事件，涵盖不同人生阶段。"""


class ResidenceInfo(BaseModel):
    location: str = Field(description="地点")
    type: str = Field(description="居住类型（如公寓/别墅/宿舍等）")
    surrounding_facilities: List[str] = Field(description="周边设施")
    duration: Optional[str] = Field(default=None, description="居住时间段")

class Lifestyle(ABCStep):
    occupation: Annotated[str, FieldUse(in_prompt=True)] = Field(description="职业")
    # 暂时不加入prompt或记忆，可能影响其行为
    daily_routine: Dict[str, str] = Field(description="日常作息")
    hobbies: Annotated[List[MemoryField[str]], FieldUse(content='我的爱好有{value}')] = Field(description="爱好")
    likes: Annotated[List[MemoryField[str]], FieldUse(content='我喜欢{value}')] = Field(description="喜欢的事物")
    dislikes: Annotated[List[MemoryField[str]], FieldUse(content='我不喜欢{value}')] = Field(description="讨厌的事物")
    # 暂时取消健康史这一字段
    # backup: 请确定：职业、日常作息、爱好、喜欢/讨厌的事物、健康状况、以及居住环境。
    # health_history: Annotated[List[MemoryField[str]], FieldUse(content='我有{value}')] = Field(default_factory=list, description="健康史")
    residence: Annotated[List[MemoryField[ResidenceInfo]], FieldUse(content='我住在/我住过：\n{value}')] = Field(description="居住信息")

    def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
        return f"""已知角色的部分设定：
{current.format_context(sprite_id=sprite_id)}

请推断这个角色的日常生活方式。生活方式应该：
1. 与其核心欲望相呼应（追求欲望的方式会影响日常）
2. 反映其弱点带来的生活影响
3. 与经历塑造的背景相符

请确定：职业、日常作息、爱好、喜欢/讨厌的事物、以及居住环境。"""


class PersonalInfo(ABCStep):
    gender: Annotated[str, FieldUse(in_prompt=True)] = Field(description="性别（如Man、Woman、Other）")
    pronouns: Annotated[str, FieldUse(in_prompt=True)] = Field(description="人称代词（如He、She、They或其他Neopronouns）")
    name: Annotated[str, FieldUse(in_prompt=True)] = Field(description="姓名")
    age: Annotated[int, FieldUse(in_prompt=True)] = Field(ge=0, description="年龄")
    birthday: Annotated[date, FieldUse(in_prompt=True)] = Field(description="出生日期")
    #height: Annotated[float, FieldUse(in_prompt=True)] = Field(description="身高（厘米）")
    #weight: Annotated[float, FieldUse(in_prompt=True)] = Field(description="体重（千克）")
    nationality: Annotated[str, FieldUse(in_prompt=True)] = Field(description="国籍")
    native_language: Annotated[str, FieldUse(in_prompt=True)] = Field(description="母语")
    other_languages: Annotated[Dict[str, LanguageProficiency], FieldUse(in_prompt=True)] = Field(default_factory=dict, description="其他掌握语言及熟练程度（若有）")
    education: Annotated[str, FieldUse(in_prompt=True)] = Field(description="最高学历")
    major: Annotated[Optional[str], FieldUse(in_prompt=True)] = Field(default=None, description="专业（若有）")
    knowledge_domains: Annotated[List[str], FieldUse(in_prompt=True)] = Field(default_factory=list, description="擅长领域")

    def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
        return f"""已知角色的部分设定：
{current.format_context(sprite_id=sprite_id)}

请生成这个角色的基础身份信息。身份信息应该：
1. 与其生活方式和职业相符
2. 姓名、年龄、国籍等应与角色背景一致
3. 教育背景和专业应能支撑其职业选择
4. 语言能力和知识领域应与其追求和经历相关"""


class Appearance(ABCStep):
    text_description: Annotated[str, FieldUse(in_prompt=True)] = Field(description="整体外貌的文字描述")
    height: Annotated[float, FieldUse(in_prompt=True)] = Field(description="身高（厘米）")
    weight: Annotated[float, FieldUse(in_prompt=True)] = Field(description="体重（千克）")
    hairstyle: Annotated[str, FieldUse(in_prompt=True)] = Field(description="发型")
    eye_features: Annotated[str, FieldUse(in_prompt=True)] = Field(description="眼部特征")
    accessories: Annotated[List[str], FieldUse(in_prompt=True)] = Field(default_factory=list, description="配饰")
    body_measurements: Annotated[Dict[str, float], FieldUse(in_prompt=True)] = Field(description="身体尺寸")

    def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
        return f"""已知角色的部分设定：
{current.format_context(sprite_id=sprite_id)}

请生成这个角色的外貌特征。外貌应该：
1. 与其身份、年龄、职业相符
2. 反映其性格特点（外表是内心的投射）
3. 包含整体描述、发型、眼部特征、配饰等细节"""


class Relationship(BaseModel):
    type: RelationshipType = Field(description="关系类型")
    name: str = Field(description="姓名")
    relationship_quality: int = Field(ge=1, le=5, description="亲密度（1-5）")
    key_memories: Annotated[List[MemoryField[str]], FieldUse(memory_type='original')] = Field(default_factory=list, description="关键回忆（要求客观叙述但保持第一人称视角）")

class Relationships(ABCStep):
    relationships: List[Relationship] = Field(default_factory=list, description="人际关系列表")

    def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
        return f"""已知角色的部分设定：
{current.format_context(sprite_id=sprite_id)}

请生成这个角色的人际关系。人际关系应该：
1. 与其经历相关联（关键事件中的人物）
2. 反映其弱点和欲望（关系模式）
3. 具备1-2段关键回忆作为关系基础

关系类型包括：父母、兄弟姐妹、伴侣、朋友、其他。每个人物应有亲密度评分和关键回忆。"""


class Asset(BaseModel):
    name: str = Field(description="资产名称")
    type: AssetType = Field(description="资产类型")
    description: str = Field(description="资产描述")
    importance: int = Field(ge=1, le=5, description="重要程度评分（1-5）")

class Assets(ABCStep):
    assets: Annotated[List[MemoryField[Asset]], FieldUse(content='我的资产之一：{value}')] = Field(default_factory=list, description="资产列表（0-5个）")

    def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
        return f"""已知角色的完整背景：
{current.format_context(sprite_id=sprite_id)}

请生成这个角色的资产列表（0-5个）：包括实际资产（房产、金钱）、情感资产（纪念物）。资产是对角色细枝末节的补充，但能增添独特性。"""

class RareTraits(ABCStep):
    traits: list[str] = Field(default_factory=list, description="罕见特质列表")

    def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
        return f"""已知角色的完整背景：
{current.format_context(sprite_id=sprite_id)}

请生成这个角色的罕见特质如小癖好、恐惧症等（0-3个），为角色增添独特魅力。罕见特质是对角色细枝末节的补充，但能增添独特性。"""



# def infer_personality(core_drive: CoreDrive) -> str:
#     return f"渴望\"{core_drive.core_desire}\"，因\"{core_drive.core_weakness}\"而受限"



class CharacterProfile(BaseModel):
    user_prompt: str = Field(description="用户输入的对角色设定的要求与描述")
    #description: Annotated[str, FieldUse(in_prompt=True)] = Field(description="角色整体描述")
    in_another_world: bool = Field(description="该角色是否在另一个世界（非现实世界或现实世界的非现实时间），在考虑部分设定以及事件发生时间时应注意这一点")
    core_drive: CoreDrive = Field(default_factory=CoreDrive.model_construct, description="核心驱动力")
    experiences: Experiences = Field(default_factory=Experiences.model_construct, description="人生经历")
    lifestyle: Annotated[Lifestyle, FieldUse(in_prompt=True)] = Field(default_factory=Lifestyle.model_construct, description="生活方式")
    personal_info: Annotated[PersonalInfo, FieldUse(in_prompt=True)] = Field(default_factory=PersonalInfo.model_construct, description="个人信息")
    appearance: Annotated[Appearance, FieldUse(in_prompt=True)] = Field(default_factory=Appearance.model_construct, description="外貌")
    #relationships: Relationships = Field(default_factory=Relationships.model_construct, description="人际关系")
    #assets: Assets = Field(default_factory=Assets.model_construct, description="资产列表")
    #rare_traits: RareTraits = Field(default_factory=RareTraits.model_construct, description="特殊设定")

    def format_context(self, indent: int = 4, for_sprite_prompt: bool = False, sprite_id: Optional[str] = None) -> str:
        """格式化角色设定"""
        result = ''
        def _remove_comma() -> None:
            nonlocal result
            result = result.strip(',')
            if result.endswith('\n'):
                result = result[:-1]
                result = result.strip(',')
                result += '\n'
        def _format(model: BaseModel, spaces: int = 0) -> None:
            nonlocal result
            fields = model.__class__.model_fields
            for k, field_info in fields.items():
                v = getattr(model, k, UNSET)
                if not isinstance(v, (bool, int, float)) and (
                    not v or
                    (isinstance(v, BaseModel) and not any(True if isinstance(i, (bool, int, float)) else i for i in v.model_dump().values()))
                ):
                    continue
                if (
                    for_sprite_prompt and
                    (
                        not field_info.metadata or
                        not isinstance(field_info.metadata[-1], FieldUse) or
                        not field_info.metadata[-1].in_prompt
                    )
                ):
                    continue

                description = field_info.description or "无字段描述"
                result += f"{' '*spaces}# {description}\n"

                if isinstance(v, BaseModel):
                    result += f'{' '*spaces}"{k}": {{\n'
                    _format(v, spaces + indent)
                    _remove_comma()
                    result += f'{' '*spaces}}}'

                elif isinstance(v, (list, tuple, set, dict)):
                    is_dict = False
                    if isinstance(v, dict):
                        is_dict = True
                    if is_dict:
                        result += f'{' '*spaces}"{k}": {{\n'
                    else:
                        result += f'{' '*spaces}"{k}": [\n'
                    items = v.items() if is_dict else [('', i) for i in v]
                    for key, item in items:
                        if isinstance(item, BaseModel):
                            result += ' '*(spaces+indent)
                            if is_dict:
                                result += f'"{key}": '
                            result += '{\n'
                            _format(item, spaces + indent * 2)
                            _remove_comma()
                            result += ' '*(spaces+indent) + '}'
                        else:
                            result += add_indent(f'{f'"{key}": ' if is_dict else ''}' + to_json_like_string(item, indent=indent, support_multiline_str=True), spaces+indent)
                        result += ',\n'
                    _remove_comma()
                    if is_dict:
                        result += f'{' '*spaces}}}'
                    else:
                        result += f'{' '*spaces}]'

                else:
                    formatted_v = to_json_like_string(v, indent=indent, support_multiline_str=True)
                    result += add_indent(f'"{k}": {formatted_v}', spaces)

                result += ',\n'

        _format(self, indent)
        _remove_comma()
        result = result.strip()
        result = ' ' * indent + result
        if result:
            result = ('（以下以json格式呈现的角色设定中可能存在以"# "开头的注释（注释在上字段在下）和多行字符串，'
                      '需注意这只是为了方便阅读的伪代码，json实际上并不支持注释和多行字符串）\n```json\n{\n') + result + '\n}\n```'
        else:
            result = "无既定的角色设定"
        if not for_sprite_prompt and getattr(self, 'in_another_world', None) is not None:
            time_settings = store_manager.get_settings(sprite_id).time_settings
            if (
                not time_settings.is_world_time_identity() or
                (time_settings.is_world_time_identity() and not self.in_another_world)
            ):
                result += f'\n另附当前角色所在世界时间供参考对比：{format_time(Times.from_time_settings(time_settings).sprite_world_datetime)}'
        return result

    class FinalStep(ABCStep):
        description: str = Field(description="重写的角色整体描述")

        def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
            return f"""以下是正在设计的角色的所有设定：
{current.format_context(sprite_id=sprite_id)}

由于角色整体描述（description）是在一开始其他设定还未完善时定下的，因此现在要求你根据以上完整设定，重写角色整体描述，使其更能描述当下经过大量细化后的角色设定。
"""

    class WorldStep(ABCStep):
        in_another_world: bool = Field(description="该角色是否在另一个世界（非现实世界），这会影响部分设定以及事件发生时间的正确性")

        def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
            return f"""已知角色的部分设定：
{current.format_context(sprite_id=sprite_id)}

请根据这些已知设定，判断角色是否可能并不处于现实世界，而是在另一个世界（现实世界的非现实时间也算作另一个世界）。
"""

    class TimeStep(ABCStep):
        current_time: datetime = Field(description="角色所在世界的当前时间")
        time_scale: float = Field(default=1.0, description="角色所在世界的时间流速，默认值为1.0，即与现实世界时间流速一致")

        def step_prompt(self, current: "CharacterProfile", sprite_id: str) -> str:
            return f'''以下是正在设计的角色的所有设定：
{current.format_context(sprite_id=sprite_id)}

由于其中的in_another_world被设为了true，故现在需要考虑角色所在的那个世界的当前时间。
请你结合以上设定，推断这个角色所在的世界的当前时间大概是什么时候，以及可能存在的时间流速不一致的情况。
'''

    async def generate(self, llm: BaseChatModel, sprite_id: str):
        time_settings = store_manager.get_settings(sprite_id).time_settings
        if not time_settings.is_world_time_identity():
            self.in_another_world = True
        self.core_drive = await self.core_drive.generate(self, llm, sprite_id)
        if not hasattr(self, 'in_another_world'):
            world_step = CharacterProfile.WorldStep.model_construct()
            world_step = await world_step.generate(self, llm, sprite_id)
            self.in_another_world = world_step.in_another_world
        self.experiences = await self.experiences.generate(self, llm, sprite_id)
        self.lifestyle = await self.lifestyle.generate(self, llm, sprite_id)
        self.personal_info = await self.personal_info.generate(self, llm, sprite_id)
        self.appearance = await self.appearance.generate(self, llm, sprite_id)
        #self.relationships = await self.relationships.generate(self, llm, sprite_id)
        #self.assets = await self.assets.generate(self, llm, sprite_id)
        #self.rare_traits = await self.rare_traits.generate(self, llm, sprite_id)
        #final = CharacterProfile.FinalStep.model_construct()
        #final = await final.generate(self, llm, sprite_id)
        #self.description = final.description
        settings = store_manager.get_settings(sprite_id)
        if self.in_another_world and settings.time_settings.is_world_time_identity():
            time_step = CharacterProfile.TimeStep.model_construct()
            time_step = await time_step.generate(self, llm, sprite_id)
            time_settings = settings.time_settings.model_copy()
            time_settings.world_real_anchor = TimestampUs.now()
            time_settings.world_sprite_anchor = TimestampUs(time_step.current_time)
            time_settings.world_scale = time_step.time_scale
            await sprite_manager.set_time_settings(sprite_id, time_settings)

    async def add_to_memory(self, sprite_id: str):
        if not sprite_manager.is_plugin_enabled('bh_memory', sprite_id):
            return
        memory_plugin = sprite_manager.get_plugin("bh_memory")

        async def _add_to_memory(memory: MemoryField, field_info: FieldInfo, field_name: str, plugin: BasePlugin):
            # 获取 FieldUse metadata
            if field_info.metadata and isinstance(field_info.metadata[-1], FieldUse):
                field_use = field_info.metadata[-1]
            else:
                field_use = FieldUse()

            # 提取记忆内容
            content = _format_field_content(
                field_info.description or field_name, memory.value, field_use
            )

            # 写入记忆
            if content:
                await plugin.add_memory(
                    sprite_id=sprite_id,
                    type=field_use.memory_type,
                    content=content,
                    creation_times=create_times(sprite_id, memory.memory_creation_time),
                    previous_memory_id="",
                    next_memory_id="",
                    #retrievability=field_use.retrievability,
                    similarity_threshold=field_use.similarity_threshold,
                    ttl=field_use.ttl or 'max',
                )

        async def extract_memories_from_model(
            model: BaseModel,
            sprite_id: str,
            plugin: BasePlugin,
        ) -> None:  # (content, memory_type)
            """
            从 model 中提取需要写入记忆的内容
            """

            for field_name, field_info in model.__class__.model_fields.items():
                value = getattr(model, field_name, None)
                # 如果字段值为False，可能是默认值或未填写，跳过
                if not isinstance(value, (bool, int, float)) and not value:
                    continue

                # 如果不用添加到memory，跳过
                if not isinstance(value, MemoryField):
                    # 如果是嵌套模型，递归处理嵌套模型的内容
                    if isinstance(value, BaseModel):
                        await extract_memories_from_model(value, sprite_id, plugin)
                    elif isinstance(value, (list, tuple, set, dict)):
                        # 如果是列表，遍历处理每个元素
                        if isinstance(value, dict):
                            value = value.values()
                        for item in value:
                            if isinstance(item, MemoryField):
                                await _add_to_memory(item, field_info, field_name, plugin)
                            elif isinstance(item, BaseModel):
                                await extract_memories_from_model(item, sprite_id, plugin)
                    continue

                await _add_to_memory(value, field_info, field_name, plugin)

        await extract_memories_from_model(self, sprite_id, memory_plugin)

REFLECTIVE_TEMPLATES = {
    "reflective": "{name}是{value}",
    "original": "{value}"
}
def _format_field_content(
    name: str, 
    value: Any, 
    field_use: FieldUse,
) -> str:
    """将字段格式化为记忆内容"""

    # 获取模板
    template = field_use.content
    if not template:
        template = REFLECTIVE_TEMPLATES.get(field_use.memory_type, "{value}")

    def _format_value(value: Any) -> str:
        # 处理不同类型的 value
        if isinstance(value, (list, tuple, set)):
            # 如果是列表，遍历处理每个元素
            items = []
            for item in value:
                items.append(_format_value(item))
            return '\n'.join(items)
        elif isinstance(value, (BaseModel, dict)):
            fields = value.__class__.model_fields
            if isinstance(value, BaseModel):
                keys = [info.description or name for name, info in fields.items()]
                values = [getattr(value, k) for k in fields.keys()]
            else:
                keys = value.keys()
                values = value.values()
            items = []
            for k, v in zip(keys, values):
                items.append(f"- {k}：{_format_value(v)}")
            return '\n'.join(items)
        else:
            return str(value)

    return template.format(**{"name": name, "value": _format_value(value)})



NAME = "bh_character"

class CharacterConfig(StoreModel):
    gen_prompt: str = StoreField(title="角色设定初始提示", description="角色整体描述（作为接下来所有要生产的详细设定的初始提示）", default='一个有用的AI助手')
    gen_model: Literal['max', 'plus', 'flash'] = StoreField(title="角色设定生成模型", description="用于生成角色详细设定的模型", default='max')
    gen_model_thinking: bool = Field(default=False, title="角色设定生成模型是否开启思考模式")

class CharacterData(StoreModel):
    profile: Union[CharacterProfile, bool] = StoreField(description="角色配置", default=False)

class CharacterPlugin(BasePlugin):
    name = NAME
    config = CharacterConfig
    data = CharacterData
    priority = PluginPriority(phase='early')

    async def on_sprite_init(self, sprite_id: str, /) -> None:
        data_store = store_manager.get_model(sprite_id, CharacterData)
        if data_store.profile == False:
            # 防止重复生成角色设定
            data_store.profile = True
            config_store = store_manager.get_model(sprite_id, CharacterConfig)
            llm = getattr(sprite_manager, f"{config_store.gen_model}_model{'_thinking' if config_store.gen_model_thinking else ''}")
            profile = CharacterProfile.model_construct(user_prompt=config_store.gen_prompt)
            await sprite_manager.publish_sprite_log(
                sprite_id=sprite_id,
                log='开始生成角色详细设定，请等待生成完毕再进行交互'
            )
            await profile.generate(llm, sprite_id)
            await profile.add_to_memory(sprite_id)
            data_store.profile = profile
            await sprite_manager.publish_sprite_log(
                sprite_id=sprite_id,
                log='角色详细设定生成完毕'
            )

    async def before_call_model(self, request: CallSpriteRequest, info: BeforeCallModelInfo, /) -> None:
        data_store = store_manager.get_model(request.sprite_id, CharacterData)
        if isinstance(data_store.profile, CharacterProfile):
            self.prompts = PluginPrompts(
                role=PluginPrompt(
                    title="角色详细设定",
                    content=data_store.profile.format_context(for_sprite_prompt=True))
            )
        else:
            self.prompts = PluginPrompts()
