from typing import Annotated

from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

from become_human.time import Times
from become_human.store_manager import store_manager
from become_human.types_main import MainContext, MainState

SEND_MESSAGE = "send_message"
SEND_MESSAGE_CONTENT = "content"

def send_message_tool_content(content: str) -> str:
    return f'消息“{content[:6]}{"..." if len(content) > 6 else ""}”发送成功。'
@tool(SEND_MESSAGE)
async def send_message(
    content: Annotated[str, '要发送的内容'],
    runtime: ToolRuntime[MainContext, MainState]
) -> Command:
    """「即时工具」发送一条消息，这是你唯一可以与用户交流的方式。
除非特别要求，不要使用如星号**加粗**、1. 或 - 这样的前缀等 Markdown 语法（因为没有人会那样说话）。
可以通过多次调用该工具的方式来分割内容，模拟真实的对话，如（示例为伪代码）：
send_message("你好！")
send_message("我是你的专属助手！")
或：
send_message("1. 进行思考")
send_message("2. 执行动作")
send_message("3. 返回结果")
何时使用此工具？
- 聊天时，你需要将要表述的内容传达给用户。
何时不使用此工具？
- 根据当前场景与你所扮演的角色设定，你“不应”回复，如：因生气或懒等原因不想回复，因身体机能或网络故障等原因无法回复。
- 你本来就无话可说。
- 用户不希望你说话，而你也接受此提议。"""
    content = send_message_tool_content(content)
    artifact = {"bh_do_not_store": True, "bh_streaming": True}
    agent_id = runtime.context.agent_id
    time_settings = (await store_manager.get_settings(agent_id)).main.time_settings
    times = Times(setting=time_settings)
    tool_message = ToolMessage(
        name=SEND_MESSAGE,
        content=content,
        artifact=artifact,
        tool_call_id=runtime.tool_call_id,
        additional_kwargs={
            "bh_creation_agent_time_seconds": times.agent_time_seconds,
            "bh_creation_time_seconds": times.real_time_seconds,
        }
    )
    new_state = {"tool_messages": [tool_message]}
    return Command(update=new_state)
