from typing import Callable, Any, Union, Self
from langchain_core.tools import BaseTool

class AgentTool:
    tool: BaseTool
    _agent_schemas: dict[str, dict[str, Any]]

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, 'tool'):
            raise TypeError(f"子类 {cls.__name__} 必须定义类属性 'tool'")
        cls._agent_schemas = {}

    @classmethod
    def get_agent_tool_schema(cls, agent_id: str) -> dict[str, Any]:
        if agent_id not in cls._agent_schemas:
            schema = cls.tool.tool_call_schema
            if not isinstance(schema, dict):
                schema = schema.model_json_schema()
            schema['title'] = cls.tool.name
            schema['description'] = cls.tool.description
            cls._agent_schemas[agent_id] = schema
        return cls._agent_schemas[agent_id]

    @classmethod
    def from_tool(cls, tool: BaseTool) -> type[Self]:
        """从已有的工具创建 AgentTool 子类"""
        class_name = f"{tool.name.replace('_', ' ').title().replace(' ', '')}Tool"
        return type(class_name, (cls,), {'tool': tool})

AnyTool = Union[Callable, BaseTool, type[AgentTool]]
