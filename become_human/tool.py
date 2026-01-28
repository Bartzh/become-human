from typing import Callable, Any, Union
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool

class AgentTool:
    tool: BaseTool
    _agent_schemas: dict[str, dict[str, Any]]

    def __init__(self, tool: Union[BaseTool, Callable]):
        if isinstance(tool, BaseTool):
            self.tool = tool
        else:
            self.tool = create_tool(tool, parse_docstring=True, error_on_invalid_docstring=False)
        self._agent_schemas = {}

    def get_agent_tool_schema(self, agent_id: str) -> dict[str, Any]:
        if agent_id not in self._agent_schemas:
            schema = self.tool.tool_call_schema
            if not isinstance(schema, dict):
                schema = schema.model_json_schema()
            schema['title'] = self.tool.name
            schema['description'] = self.tool.description
            self._agent_schemas[agent_id] = schema
        return self._agent_schemas[agent_id]
