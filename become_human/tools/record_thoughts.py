from typing import Annotated, Any

from langchain.tools import tool, ToolRuntime

from become_human.times import Times
from become_human.message import SpritesMsgMeta, DEFAULT_TOOL_MSG_TYPE
from become_human.store.manager import store_manager
from become_human.types.manager import CallSpriteRequest

RECORD_THOUGHTS = "record_thoughts"
RECORD_THOUGHTS_CONTENT = "content"
RECORD_THOUGHTS_TOOL_CONTENT = "已记录心理活动。"

@tool(RECORD_THOUGHTS, response_format='content_and_artifact', description="""「纯执行工具」记录当下你所扮演的角色的心理活动。
这个动作是**必须**的，若未调用此工具，系统将会向你返回错误。
何时调用此工具？
- 每轮对话开始时或进行时，你都应先调用此工具记录心理活动，然后再调用其他需要调用的工具。
- 就算你什么工具都不想调用，也应至少调用此工具。该工具为「纯执行工具」，所以这不会导致你陷入无限的ReAct工具调用循环。
何时不调用此工具？
- 没有。任何情况下都必须调用一次。""")
async def record_thoughts(
    content: Annotated[str, '要记录的心理活动'],
    runtime: ToolRuntime[CallSpriteRequest]
) -> tuple[str, dict[str, Any]]:
    content = RECORD_THOUGHTS_TOOL_CONTENT
    sprite_id = runtime.context.sprite_id
    time_settings = store_manager.get_settings(sprite_id).time_settings
    times = Times.from_time_settings(time_settings)
    artifact = SpritesMsgMeta(
        creation_times=times,
        message_type=DEFAULT_TOOL_MSG_TYPE,
        is_action_only_tool=True
    )
    return content, artifact
