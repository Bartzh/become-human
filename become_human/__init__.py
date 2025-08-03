from langchain_qwq import ChatQwen
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessageChunk, HumanMessage, RemoveMessage, BaseMessage, AIMessage, AnyMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages.utils import count_tokens_approximately

from become_human.graph_main import MainGraph
from become_human.graph_recycle import RecycleGraph
from become_human.graph_retrieve import RetrieveGraph
from become_human.memory import MemoryManager
from become_human.config import load_config, get_thread_configs_toml, set_config
from become_human.utils import parse_time, is_valid_json, parse_messages

from typing import Annotated, Optional, Union, Any
import os
import asyncio
import atexit
from datetime import datetime, timezone
from warnings import warn
from uuid import uuid4

event_queue = asyncio.Queue()

@tool(response_format="content_and_artifact")
async def retrieve_memories(search_string: Annotated[str, "要检索的内容"], config: RunnableConfig) -> tuple[str, dict[str, Any]]:
    """从数据库（大脑）中检索记忆"""
    result = await retrieve_graph.graph.ainvoke({"input": search_string, "type": "active"}, config)
    content = result["output"]
    artifact = {"bh_do_not_store": True}
    return content, artifact

# 缓冲用于当双发但还没调用graph时，最后一次调用可以连上之前的输入给agent，而前面的调用直接取消即可。
thread_user_input_buffers: dict[str, list[str]] = {}
thread_gathereds: dict[str, AIMessageChunk] = {}
thread_streaming_tool_messages: dict[str, list[ToolMessage]] = {}
async def stream_graph_updates(user_input: Union[str, list[str]], thread_id: str, user_name: Optional[str] = None, is_self_call: Optional[bool] = None):
    global thread_user_input_buffers, thread_gathereds, thread_streaming_tool_messages

    if not is_self_call:
        # 首先把user_input加入buffer
        if not thread_user_input_buffers.get(thread_id):
            thread_user_input_buffers[thread_id] = []
        if isinstance(user_input, list):
            thread_user_input_buffers[thread_id].extend(user_input)
        else:
            thread_user_input_buffers[thread_id].append(user_input)
    elif main_graph.thread_run_ids.get(thread_id):
        # 如果是自我调用同时已有运行，则不重复运行
        return

    # 生成运行id并加入运行id字典
    thread_run_id = str(uuid4())
    main_graph.thread_run_ids[thread_id] = thread_run_id

    # 当前线程有gathered，说明是在chatbot节点处打断了上次运行，则将上次运行结果加入中断数据字典
    if thread_gathereds.get(thread_id):
        interrupt_data = {}
        interrupt_data['chunk'] = thread_gathereds.pop(thread_id)
        if thread_streaming_tool_messages.get(thread_id):
            interrupt_data['called_tool_messages'] = thread_streaming_tool_messages.pop(thread_id)
        if interrupt_data:
            main_graph.thread_interrupt_datas[thread_id] = interrupt_data
    elif thread_streaming_tool_messages.get(thread_id):
        del thread_streaming_tool_messages[thread_id]
        warn('线程已结束但有未处理的工具消息，已清理')

    # 如果main_graph正在运行"tools"节点，则等待其运行完毕再打断。
    config = {"configurable": {"thread_id": thread_id}}
    main_graph_state = await main_graph.graph.aget_state(config)
    if main_graph_state.next:
        current_node = main_graph_state.next[0]
        while current_node == "tools" or current_node == "tool_node_post_process" or current_node == "begin":
            await asyncio.sleep(0.2)
            # 如果在等待期间又有新的调用，则取消这次调用
            if main_graph.thread_run_ids.get(thread_id, '') != thread_run_id:
                return
            main_graph_state = await main_graph.graph.aget_state(config)
            current_node = main_graph_state.next[0]

    if not is_self_call:
        user_input = thread_user_input_buffers.pop(thread_id)
    else:
        if thread_user_input_buffers.get(thread_id):
            del thread_user_input_buffers[thread_id]
            warn("在自我调用时意外发现存在残留未处理的用户输入，已删除。")

    config["configurable"]["thread_run_id"] = thread_run_id
    current_time = datetime.now()
    parsed_time = parse_time(current_time)
    current_timestamp = current_time.timestamp()
    if not is_self_call:
        if isinstance(user_name, str):
            user_name = user_name.strip()
        name = user_name or "未知姓名"
        if isinstance(user_input, str):
            input_content = f'[{parsed_time}]\n{name}: {user_input}'
        elif isinstance(user_input, list):
            if len(user_input) == 1:
                input_content = f'[{parsed_time}]\n{name}: {user_input[0]}'
            elif len(user_input) > 1:
                input_content = [{'type': 'text', 'text': f'[{parsed_time}]\n{name}: {message}'} for message in user_input]
            else:
                raise ValueError("Input list cannot be empty")
        else:
            raise ValueError("Invalid input type")
        graph_input = {"input_messages": HumanMessage(
            content=input_content,
            name=user_name,
            additional_kwargs={"bh_creation_timestamp": current_timestamp}
        )}
    else:
        config["configurable"]["is_self_call"] = True
        graph_input = {}

    first = True
    tool_index = 0
    last_message = ''
    canceled = False
    gathered = None
    streaming_tool_messages = []
    async for typ, msg in main_graph.graph.astream(graph_input, config, stream_mode=["updates", "messages"]):
        if typ == "updates":
            #print(msg)

            if not main_graph.thread_run_ids.get(thread_id) or main_graph.thread_run_ids.get(thread_id) != thread_run_id:
                canceled = True

            if not canceled and msg.get("chatbot"):
                first = True
                tool_index = 0
                gathered = None
                streaming_tool_messages = []
                last_message = ''
                del thread_gathereds[thread_id]
                del thread_streaming_tool_messages[thread_id]

            if msg.get("chatbot"):
                messages = msg.get("chatbot").get("messages")
            elif msg.get("tools"):
                messages = msg.get("tools").get("messages")
            else:
                continue
            if isinstance(messages, BaseMessage):
                messages = [messages]
            if messages:
                for message in messages:
                    message.pretty_print()
                    if isinstance(message, AIMessage) and message.additional_kwargs.get("reasoning_content"):
                        print("reasoning_content: " + message.additional_kwargs.get("reasoning_content", ""))



        elif typ == "messages":
            #print(msg[0], end="\n\n", flush=True)

            if canceled:
                continue

            if not main_graph.thread_run_ids.get(thread_id) or main_graph.thread_run_ids.get(thread_id) != thread_run_id:
                canceled = True
                continue

            if isinstance(msg[0], AIMessageChunk):

                chunk: AIMessageChunk = msg[0]

                if first:
                    first = False
                    gathered = chunk
                    thread_gathereds[thread_id] = chunk
                    streaming_tool_messages = []
                    thread_streaming_tool_messages[thread_id] = []
                else:
                    if chunk.id != gathered.id:
                        gathered = chunk
                        streaming_tool_messages= []
                        thread_streaming_tool_messages[thread_id] = []
                    else:
                        gathered += chunk
                    thread_gathereds[thread_id] = gathered

                tool_call_chunks = gathered.tool_call_chunks
                tool_calls = gathered.tool_calls


                #if chunk.response_metadata.get('finish_reason'):
                #    continue


                loop_once = True

                while loop_once:

                    loop_once = False
                    if 0 <= tool_index < len(tool_calls):

                        chunk_completed = is_valid_json(tool_call_chunks[tool_index]['args'])
                        tool_call_id = tool_calls[tool_index].get('id', 'run-' + str(tool_index))

                        if tool_calls[tool_index]['name'] == 'send_message':

                            new_message = tool_calls[tool_index]['args'].get('message')

                            if new_message:
                                #await event_queue.put({"thread_id": thread_id, "name": "send_message", "args": {"message": new_message.replace(last_message, '', 1)}, "not_completed": True})
                                event_item = {"thread_id": thread_id, "name": "send_message", "args": {"message": new_message}, "id": tool_call_id}
                                if not chunk_completed:
                                    event_item["not_completed"] = True
                                else:
                                    if tool_calls[tool_index].get('id'):
                                        streaming_tool_messages.append(ToolMessage(content="消息发送成功。", name="send_message", tool_call_id=tool_calls[tool_index]['id']))
                                        thread_streaming_tool_messages[thread_id] = streaming_tool_messages
                                    last_message = ''
                                await event_queue.put(event_item)
                                #print(new_message.replace(last_message, '', 1), end="", flush=True)
                                last_message = new_message


                        if chunk_completed:
                            #if tool_calls[tool_index]['name'] == 'send_message':
                            #    last_message = ''
                            #    await event_queue.put({"thread_id": thread_id, "name": "send_message", "args": {"message": ""}})
                                #print('', flush=True)
                            if hasattr(main_graph.streaming_tools, tool_calls[tool_index]['name']):
                                method = getattr(main_graph.streaming_tools, tool_calls[tool_index]['name'])
                                result = await method(tool_calls[tool_index]['args'])
                                if tool_calls[tool_index].get('id'):
                                    streaming_tool_messages.append(ToolMessage(content=result, name=tool_calls[tool_index]['name'], tool_call_id=tool_calls[tool_index]['id']))
                                    thread_streaming_tool_messages[thread_id] = streaming_tool_messages
                                await event_queue.put({"thread_id": thread_id, "name": tool_calls[tool_index]['name'], "args": tool_calls[tool_index]['args'], "id": tool_call_id})
                                #print(await method(tool_calls[tool_index]['args']), flush=True)
                            tool_index += 1
                            loop_once = True


    if main_graph.thread_run_ids.get(thread_id) and main_graph.thread_run_ids.get(thread_id) == thread_run_id:
        del main_graph.thread_run_ids[thread_id]
    return


class HeartbeatManager:

    interval: float
    thread_ids: dict[str, Any]
    is_running: bool
    task: Optional[asyncio.Task]

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.thread_ids = {}
        self.is_running = False
        self.task = None
        self.start()
        atexit.register(self.stop)

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self.heartbeat_task())

    async def heartbeat_task(self):
        if self.is_running and self.task is not None:
            return
        self.is_running = True
        while self.is_running:
            await self.trigger_threads()
            await asyncio.sleep(self.interval)
        print("Heartbeat task stopped.")

    async def trigger_threads(self):
        for thread_id in self.thread_ids.keys():
            await self.trigger_thread(thread_id)

    async def trigger_thread(self, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        await memory_manager.update_timer(thread_id)
        main_graph_state = await main_graph.graph.aget_state(config)
        current_timestamp = datetime.now(timezone.utc).timestamp()
        self_call_timestamps = main_graph_state.values.get("self_call_timestamps", [])
        wakeup_call_timestamp = main_graph_state.values.get("wakeup_call_timestamp")
        if self_call_timestamps:
            for timestamp in self_call_timestamps:
                if current_timestamp >= timestamp:
                    await stream_graph_updates('', thread_id, is_self_call=True) #TODO task
                    break
        if wakeup_call_timestamp:
            if current_timestamp >= wakeup_call_timestamp:
                await stream_graph_updates('', thread_id, is_self_call=True)
        else:
            if current_timestamp > (self.thread_ids[thread_id]["created_at"] + 1209600):
                close_thread(thread_id)


    async def init_thread(self, thread_id: str):
        self.thread_ids[thread_id] = {"created_at": datetime.now(timezone.utc).timestamp()}
        await self.trigger_thread(thread_id)

    def close_thread(self, thread_id: str):
        if self.thread_ids.get(thread_id):
            del self.thread_ids[thread_id]

    def stop(self):
        print("wait for the last heartbeat to stop")
        self.is_running = False

    def stop_force(self):
        if self.task:
            self.task.cancel()
            self.task = None
        self.is_running = False

async def init_thread(thread_id: str):
    await heartbeat_manager.init_thread(thread_id)

def close_thread(thread_id: str):
    heartbeat_manager.close_thread(thread_id)


async def init_graphs(heartbeat_interval: float = 5.0) -> tuple[BaseChatModel, BaseChatModel, Embeddings, MemoryManager, MainGraph, RecycleGraph, RetrieveGraph]:
    global llm_for_chat, llm_for_structured, embeddings, memory_manager, main_graph, recycle_graph, retrieve_graph, heartbeat_manager

    envs = ["CHAT_MODEL_NAME", "CHAT_MODEL_ENABLE_THINKING", "STRUCTURED_MODEL_NAME", "STRUCTURED_MODEL_ENABLE_THINKING"]
    for e in envs:
        if not os.getenv(e):
            raise Exception(f"{e} is not set")

    llm_for_chat = ChatQwen(
        model=os.getenv("CHAT_MODEL_NAME"),
        #top_p=0.8,
        max_retries=2,
        timeout=30.0,
        enable_thinking=True if os.getenv("CHAT_MODEL_ENABLE_THINKING").lower() == "true" else False,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #parallel_tool_calls=True
    )

    llm_for_structured = ChatQwen(
        model=os.getenv("STRUCTURED_MODEL_NAME"),
        #top_p=0.8,
        max_retries=2,
        timeout=30.0,
        enable_thinking=True if os.getenv("STRUCTURED_MODEL_ENABLE_THINKING").lower() == "true" else False,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    embeddings = DashScopeEmbeddings(model="text-embedding-v3")

    memory_manager = await MemoryManager.create(embeddings)

    retrieve_graph = await RetrieveGraph.create(llm_for_structured, memory_manager)
    recycle_graph = await RecycleGraph.create(llm_for_structured, memory_manager)
    main_graph = await MainGraph.create(llm_for_chat, retrieve_graph, recycle_graph, memory_manager, [retrieve_memories], llm_for_structured)

    heartbeat_manager = HeartbeatManager(heartbeat_interval)

    return llm_for_chat, llm_for_structured, embeddings, memory_manager, main_graph, recycle_graph, retrieve_graph

async def close_graphs():
    #memory_manager.stop_periodic_task()
    await main_graph.conn.close()
    await recycle_graph.conn.close()
    await retrieve_graph.conn.close()
    print("Graphs closed")


async def command_processing(thread_id: str, user_input: str):
    config = {"configurable": {"thread_id": thread_id}}
    message = 'invalid command'
    if user_input.startswith("/get_state ") or user_input == "/get_state":
        if user_input == "/get_state":
            message = '''Usage: /get_state <graph_name> [key]
graph_name: main, recycle, retrievn
key: (optional) specific key to get from state'''
        else:
            splited_input = user_input.split(" ")
            graph_name = splited_input[1]
            requested_key = None
            if len(splited_input) > 2:
                requested_key = splited_input[2]
            
            if graph_name == "main":
                state = await main_graph.graph.aget_state(config)
                if requested_key:
                    message = f"Main graph state[{requested_key}]: {state.values.get(requested_key, 'Key not found')}"
                else:
                    message = f"Main graph state: {state.values}"
            elif graph_name == "recycle":
                state = await recycle_graph.graph.aget_state(config)
                if requested_key:
                    message = f"Recycle graph state[{requested_key}]: {state.values.get(requested_key, 'Key not found')}"
                else:
                    message = f"Recycle graph state: {state.values}"
            elif graph_name == "retrieve":
                state = await retrieve_graph.graph.aget_state(config)
                if requested_key:
                    message = f"Retrieve graph state[{requested_key}]: {state.values.get(requested_key, 'Key not found')}"
                else:
                    message = f"Retrieve graph state: {state.values}"

    elif user_input.startswith("/delete_last_messages ") or user_input == "/delete_last_messages":
        if user_input == "/delete_last_messages":
            message = 'Usage: /delete_last_messages <message_count>'
        else:
            splited_input = user_input.split(" ")
            message_count = int(splited_input[1])
            if message_count > 0:
                _main_state = await main_graph.graph.aget_state(config)
                _main_messages: list[AnyMessage] = _main_state.values.get('messages')
                if _main_messages:
                    _last_messages = _main_messages[-message_count:]
                    remove_messages = [RemoveMessage(id=_message.id) for _message in _last_messages]
                    await main_graph.graph.aupdate_state(config, {"messages": remove_messages, "input_messages": remove_messages})
                    message = f"Last {len(remove_messages)} messages deleted."
                else:
                    message = "No messages found"

    elif user_input.startswith("/set_role_prompt ") or user_input == "/set_role_prompt":
        if user_input == "/set_role_prompt":
            message = """Usage: /set_role_prompt <role_prompt>
role_prompt: The role prompt to set for the user"""
        else:
            splited_input = user_input.split(" ")
            role_prompt = " ".join(splited_input[1:]).strip()
            if role_prompt:
                configs_toml = get_thread_configs_toml()
                configs_toml[thread_id]['main']['role_prompt'] = role_prompt
                set_config(configs_toml)
                message = "Role prompt set successfully"
            else:
                message = "Role prompt cannot be empty"

    elif user_input == "/reload_config":
        load_config()
        message = "配置文件已重新加载。"

    elif user_input == "/wakeup":
        await main_graph.graph.aupdate_state(config, {"active_timestamp": 0.0, "self_call_timestamps": [], "wakeup_call_timestamp": 0.0})
        message = "已唤醒agent（重置自我调用相关状态），这可能导致agent对prompt有些误解。"
    
    elif user_input == "/messages":
        main_state = await main_graph.graph.aget_state(config)
        if main_messages := main_state.values.get("messages"):
            message = parse_messages(main_messages)
        else:
            message = "消息为空。"
    
    elif user_input == "/tokens":
        main_state = await main_graph.graph.aget_state(config)
        if main_messages := main_state.values.get("messages"):
            message = count_tokens_approximately(main_messages)
        else:
            message = "消息为空。"

    #return {"name": "log", "args": {"message": message}}
    await event_queue.put({"thread_id": thread_id, "name": "log", "args": {"message": message}, "id": 'command-' + str(uuid4())})
