from pydantic import BaseModel, Field
from typing import Optional, Type
from datetime import date

from become_human.times import SpriteTimeSettings, Times, TimestampUs
from become_human.store.base import StoreField, StoreModel
from become_human.names import PROJECT_NAME


# 新想法，通过结构化然后让AI生成所有没有指定的细节
class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="姓名")
    age: Optional[int] = Field(default=None, description="年龄")
    sex: Optional[str] = Field(default=None, description="性别")
    birthday: Optional[date] = Field(default=None, description="生日")
    additional_info: Optional[dict[str, str]] = Field(default=None, description="其他信息")

class SpritedSettings(StoreModel):
    _namespace = (PROJECT_NAME,)
    _title = "sprited设置"
    _description = "sprited内置的一些设置"
    _is_config = True
    role_prompt: str = StoreField(default="你是一个对陌生人也抱有基本尊重的普通人。你与他人是通过一个无聊天记录（阅后即焚）的即时通讯软件远程交流的。", title="角色提示词")
    role_description: str = StoreField(default="应该是一个有用的助手吧。", title="展示用角色描述", description="直接向用户显示的一段文本，描述这个角色")
    time_settings: SpriteTimeSettings = StoreField(default_factory=SpriteTimeSettings, title="时间设置")
    character_settings: Person = StoreField(default_factory=Person, title="角色设定")

    def format_character_settings(self, indent: int = 4, prefix: str = '- ',) -> str:
        def _format_character_setting(model: Type[BaseModel], source: dict) -> dict:
            character_settings = {}
            for key, value in source.items():
                if value is None:
                    continue
                if key in model.model_fields.keys():
                    if model.model_fields[key].description:
                        cs_key = model.model_fields[key].description
                    else:
                        cs_key = key
                    if isinstance(value, dict) and issubclass(model.model_fields[key].annotation, BaseModel):
                        character_settings[cs_key] = _format_character_setting(model.model_fields[key].annotation, value)
                    else:
                        character_settings[cs_key] = value
                elif key not in character_settings.keys():
                    character_settings[key] = value
            return character_settings
        person = self.character_settings
        character_settings = _format_character_setting(Person, person.model_dump())
        # always_active的话就当它不会睡觉了
        #if self.sleep_time_range and not self.always_active:
        #    character_settings["睡觉时间段"] = f"{seconds_to_datetime(self.sleep_time_range[0]).time()} ~ {seconds_to_datetime(self.sleep_time_range[1]).time()}"
        def _dict_to_readable_string(d: dict, plus: int = 4, prefix: str = '- ', indent=0):
            result = ""
            for key, value in d.items():
                if isinstance(value, dict):
                    result += " " * indent + prefix + f"{key}:\n"
                    result += _dict_to_readable_string(value, plus, prefix, indent + plus)
                else:
                    result += " " * indent + prefix + f"{key}: {value}\n"
            return result.strip()
        return _dict_to_readable_string(character_settings, indent, prefix)
