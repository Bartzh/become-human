from typing import Annotated
from langchain.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import InjectedToolCallId, tool
from langgraph.types import Command
from become_human.time import Times
from become_human.store_manager import store_manager

SEND_MESSAGE = "send_message"
SEND_MESSAGE_CONTENT = "content"

def send_message_tool_content(content: str) -> str:
    return f'消息“{content[:6]}{"..." if len(content) > 6 else ""}”发送成功。'
@tool(SEND_MESSAGE)
async def send_message(content: Annotated[str, '要发送的内容'], config: RunnableConfig, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """「即时工具」发送一条消息，这是你唯一可以与用户交流的方式
何时使用此工具？
- 聊天时，你需要将要表述的内容传达给用户
何时不使用此工具？
- 根据当前场景与你所扮演的角色设定，你“不应”回复，如：因生气或懒等原因不想回复，因身体机能或网络故障等原因无法回复
- 你本来就无话可说
- 用户不希望你说话，而你也接受"""
    content = send_message_tool_content(content)
    artifact = {"bh_do_not_store": True, "bh_streaming": True}
    thread_id = config["configurable"]["thread_id"]
    time_settings = (await store_manager.get_settings(thread_id)).main.time_settings
    times = Times(setting=time_settings)
    tool_message = ToolMessage(name=SEND_MESSAGE, content=content, artifact=artifact, tool_call_id=tool_call_id, additional_kwargs={
        "bh_creation_agent_time_seconds": times.agent_time_seconds,
        "bh_creation_time_seconds": times.real_time_seconds,
    })
    new_state = {"tool_messages": [tool_message]}
    return Command(update=new_state)
