from typing import Callable, Any, Union, Optional
from copy import deepcopy
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool

class SpriteTool:
    tool: BaseTool
    default_schema: Optional[dict[str, Any]]
    hide_by_default: bool
    _sprite_schemas: dict[str, Optional[dict[str, Any]]]


    def __init__(
        self,
        tool: Union[BaseTool, Callable],
        default_schema: Optional[dict[str, Any]] = None,
        hide_by_default: bool = False,
    ):
        if isinstance(tool, BaseTool):
            self.tool = tool
        else:
            self.tool = create_tool(tool, parse_docstring=True, error_on_invalid_docstring=False)
        self._sprite_schemas = {}
        self.default_schema = default_schema
        self.hide_by_default = hide_by_default

    def get_schema(self, sprite_id: str) -> Optional[dict[str, Any]]:
        if sprite_id not in self._sprite_schemas:
            self._sprite_schemas[sprite_id] = self.generate_default_schema()
        return self._sprite_schemas[sprite_id]

    def set_schema(self, sprite_id: str, schema: Optional[dict[str, Any]]):
        """设置该tool在sprite_id下的schema，设置为None则表示隐藏"""
        self._sprite_schemas[sprite_id] = schema

    def reset_schema(self, sprite_id: str):
        """重置该tool在sprite_id下的schema为默认schema，若hide_by_default为True，则schema为None"""
        self._sprite_schemas[sprite_id] = self.generate_default_schema()

    def generate_default_schema(self) -> Optional[dict[str, Any]]:
        """生成默认schema"""
        if self.hide_by_default:
            return None
        if self.default_schema is not None:
            return deepcopy(self.default_schema)
        schema = self.tool.tool_call_schema
        if not isinstance(schema, dict):
            schema = schema.model_json_schema()
        else:
            schema = deepcopy(schema)
        schema['title'] = self.tool.name
        schema['description'] = self.tool.description
        return schema
