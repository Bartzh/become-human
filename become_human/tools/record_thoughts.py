from typing import Annotated

from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

from become_human.time import Times
from become_human.store.manager import store_manager
from become_human.types.main import MainContext, MainState

RECORD_THOUGHTS = "record_thoughts"
RECORD_THOUGHTS_CONTENT = "content"
RECORD_THOUGHTS_TOOL_CONTENT = "已记录心理活动。"

@tool(RECORD_THOUGHTS, description="""「即时工具」记录当下你所扮演的角色的心理活动。
这个动作是**必须**的，若未调用此工具，系统将会向你返回错误。
何时调用此工具？
- 每轮对话开始时或进行时，你都应先调用此工具记录心理活动，然后再调用其他需要调用的工具。
- 就算你什么工具都不想调用，也应至少调用此工具。该工具为「即时工具」，所以这不会导致你陷入无限的ReAct工具调用循环。
何时不调用此工具？
- 没有。任何情况下都必须调用一次。""")
async def record_thoughts(
    content: Annotated[str, '要记录的心理活动'],
    runtime: ToolRuntime[MainContext, MainState]
) -> Command:
    content = RECORD_THOUGHTS_TOOL_CONTENT
    artifact = {
        #"bh_do_not_store": True,
        "bh_streaming": True
    }
    agent_id = runtime.context.agent_id
    time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
    times = Times(setting=time_settings)
    tool_message = ToolMessage(
        name=RECORD_THOUGHTS,
        content=content,
        artifact=artifact,
        tool_call_id=runtime.tool_call_id,
        additional_kwargs={
            "bh_creation_agent_timeseconds": times.agent_timeseconds,
            "bh_creation_real_timeseconds": times.real_timeseconds,
        }
    )
    new_state = {"tool_messages": [tool_message]}
    return Command(update=new_state)
