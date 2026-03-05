from typing import Callable, Any, Union, Optional
from copy import deepcopy
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool

class SpriteTool:
    tool: BaseTool
    default_schema: Optional[dict[str, Any]]
    _sprite_schemas: dict[str, dict[str, Any]]

    def __init__(self, tool: Union[BaseTool, Callable], default_schema: Optional[dict[str, Any]] = None):
        if isinstance(tool, BaseTool):
            self.tool = tool
        else:
            self.tool = create_tool(tool, parse_docstring=True, error_on_invalid_docstring=False)
        self._sprite_schemas = {}
        self.default_schema = default_schema

    def get_schema(self, sprite_id: str) -> dict[str, Any]:
        if self._sprite_schemas.get(sprite_id) is None:
            self._sprite_schemas[sprite_id] = self.generate_schema()
        return self._sprite_schemas[sprite_id]

    def set_schema(self, sprite_id: str, schema: Optional[dict[str, Any]]):
        self._sprite_schemas[sprite_id] = schema

    def generate_schema(self) -> dict[str, Any]:
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
