from langchain_qwq import ChatQwen, ChatQwQ
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessageChunk, HumanMessage, RemoveMessage, BaseMessage, AIMessage, AnyMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import InjectedState
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from become_human.graph_main import MainGraph
from become_human.graph_recycle import RecycleGraph
from become_human.graph_retrieve import RetrieveGraph
from become_human.memory import MemoryManager, parse_memory_documents
from become_human.config import load_config, get_thread_configs
from become_human.utils import is_valid_json, parse_messages
from become_human.time import datetime_to_seconds, real_time_to_agent_time, now_seconds, now_agent_seconds, parse_time
from become_human.store import store_setup, stop_listener, store_adelete_namespace
from become_human.store_manager import store_manager

from typing import Annotated, Optional, Union, Any, Literal
import os
import asyncio
from datetime import datetime
from warnings import warn
from uuid import uuid4

event_queue = asyncio.Queue()

@tool(response_format="content_and_artifact")
async def retrieve_memories(search_string: Annotated[str, "要检索的内容"], messages: Annotated[list[AnyMessage], InjectedState('messages')], config: RunnableConfig) -> tuple[str, dict[str, Any]]:
    """从数据库（大脑）中检索记忆"""
    thread_id = config["configurable"]["thread_id"]
    store_settings = await store_manager.get_settings(thread_id)
    time_settings = store_settings.main.time_settings
    result = await retrieve_graph.graph.ainvoke({"input": search_string, "type": "active"}, config)
    if result.get("error"):
        content = result.get("error")
    else:
        docs: list[Document] = result.get("output", [])
        message_ids = [m.id for m in messages if m.id]
        docs = [doc for doc in docs if doc.id not in message_ids]
        content = parse_memory_documents(docs, time_settings)
    artifact = {"bh_do_not_store": True}
    return content, artifact


# 缓冲用于当双发但还没调用graph时，最后一次调用可以连上之前的输入给agent，而前面的调用直接取消即可。
thread_user_input_buffers: dict[str, list[str]] = {}
thread_gathereds: dict[str, AIMessageChunk] = {}
thread_streaming_tool_messages: dict[str, list[ToolMessage]] = {}
async def stream_graph_updates(user_input: Union[str, list[str]], thread_id: str, user_name: Optional[str] = None, is_self_call: Optional[bool] = None, self_call_type: Literal['passive', 'active'] = 'passive'):
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
        while current_node == "tools" or current_node == "tool_node_post_process" or current_node == "prepare_to_recycle" or current_node == "begin":
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
    current_time_seconds = datetime_to_seconds(current_time)
    store_settings = await store_manager.get_settings(thread_id)
    time_settings = store_settings.main.time_settings
    current_agent_time = real_time_to_agent_time(current_time, time_settings)
    parsed_agent_time = parse_time(current_agent_time)
    if not is_self_call:
        if isinstance(user_name, str):
            user_name = user_name.strip()
        name = user_name or "未知姓名"
        if isinstance(user_input, str):
            input_content = f'[{parsed_agent_time}]\n{name}: {user_input}'
        elif isinstance(user_input, list):
            if len(user_input) == 1:
                input_content = f'[{parsed_agent_time}]\n{name}: {user_input[0]}'
            elif len(user_input) > 1:
                input_content = [{'type': 'text', 'text': f'[{parsed_agent_time}]\n{name}: {message}'} for message in user_input]
            else:
                del main_graph.thread_run_ids[thread_id]
                raise ValueError("Input list cannot be empty")
        else:
            del main_graph.thread_run_ids[thread_id]
            raise ValueError("Invalid input type")
        graph_input = {"input_messages": HumanMessage(
            content=input_content,
            name=user_name,
            additional_kwargs={"bh_creation_time_seconds": current_time_seconds}
        )}
    else:
        config["configurable"]["is_self_call"] = True
        config["configurable"]["self_call_type"] = self_call_type
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

    @classmethod
    async def create(cls, interval: float = 5.0):
        instance = cls(interval)
        await instance.start()
        return instance

    async def start(self):
        for key, value in get_thread_configs().items():
            if value.get('init_on_startup'):
                await self.init_thread(key)
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
        tasks = [self.trigger_thread(thread_id) for thread_id in self.thread_ids.keys()]
        await asyncio.gather(*tasks)

    async def trigger_thread(self, thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        # 更新记忆
        await memory_manager.update_timer(thread_id)

        main_graph_state = await main_graph.graph.aget_state(config)
        if main_graph_state.next:
            return

        # 处理自我调用
        current_time_seconds = now_seconds()
        store_settings = await store_manager.get_settings(thread_id)
        time_settings = store_settings.main.time_settings
        current_agent_time_seconds = now_agent_seconds(time_settings)
        self_call_time_secondses = main_graph_state.values.get("self_call_time_secondses", [])
        wakeup_call_time_seconds = main_graph_state.values.get("wakeup_call_time_seconds")
        active_self_call_time_secondses_and_notes = main_graph_state.values.get("active_self_call_time_secondses_and_notes", [])
        can_call = False
        if active_self_call_time_secondses_and_notes:
            for seconds, note in active_self_call_time_secondses_and_notes:
                if current_agent_time_seconds >= seconds:
                    can_call = True
                    self_call_type = 'active'
                    break
        if self_call_time_secondses and not can_call:
            for seconds in self_call_time_secondses:
                if current_agent_time_seconds >= seconds:
                    can_call = True
                    self_call_type = 'passive'
                    break
        if wakeup_call_time_seconds and not can_call:
            if current_agent_time_seconds >= wakeup_call_time_seconds:
                can_call = True
                self_call_type = 'passive'
        if can_call:
            await stream_graph_updates('', thread_id, is_self_call=True, self_call_type=self_call_type)

        # 自动回收闲置上下文
        elif main_graph_state.values.get("active_time_seconds") and current_agent_time_seconds > main_graph_state.values.get("active_time_seconds"):# 已超出活跃时间
            messages = await main_graph.get_messages(thread_id)
            passive_retrieve_messages = [m for m in messages if m.additional_kwargs.get("bh_type", '') == "passive_retrieve"]
            remove_messages = [RemoveMessage(id=m.id) for m in passive_retrieve_messages]
            not_passive_retrieve_messages = [m for m in messages if m.additional_kwargs.get("bh_type", '') != "passive_retrieve"]
            not_extracted_messages = [m for m in messages if not m.additional_kwargs.get("bh_extracted")]
            if [m for m in not_extracted_messages if isinstance(m, HumanMessage) and (not m.additional_kwargs.get("bh_from_system") or not m.additional_kwargs.get("bh_do_not_store"))]: # 有来自用户的消息
                await recycle_graph.graph.ainvoke({"input_messages": not_extracted_messages, "recycle_type": "extract"}, config)
                is_cleanup = store_settings.recycle.cleanup_on_non_active_recycle
                if is_cleanup:
                    #await main_graph.update_messages(thread_id, [RemoveMessage(id=m.id) for m in messages])
                    max_tokens = store_settings.recycle.cleanup_target_size
                    if max_tokens > 0:
                        new_messages: list[BaseMessage] = trim_messages(
                            messages=not_passive_retrieve_messages,
                            max_tokens=max_tokens,
                            token_counter=count_tokens_approximately,
                            strategy='last',
                            start_on=HumanMessage,
                            allow_partial=True,
                            text_splitter=RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
                        )
                        if not new_messages:
                            warn("Trim messages failed on cleanup.")
                            new_messages = []
                        excess_count = len(not_passive_retrieve_messages) - len(new_messages)
                        old_messages = not_passive_retrieve_messages[:excess_count]
                        remove_messages.extend([RemoveMessage(id=message.id) for message in old_messages])
                        for m in new_messages:
                            m.additional_kwargs["bh_extracted"] = True
                        await main_graph.update_messages(thread_id, new_messages + remove_messages)
                    else:
                        await main_graph.update_messages(thread_id, [RemoveMessage(id=REMOVE_ALL_MESSAGES)])
                else:
                    for m in not_extracted_messages:
                        m.additional_kwargs["bh_extracted"] = True
                    await main_graph.update_messages(thread_id, not_extracted_messages + remove_messages)

        # 闲置过久则关闭线程
        elif current_time_seconds > (self.thread_ids[thread_id]["created_at"] + 1209600):
            self.close_thread(thread_id)


    async def init_thread(self, thread_id: str):
        self.thread_ids[thread_id] = {"created_at": now_seconds()}
        await store_manager.init_thread(thread_id)
        await self.trigger_thread(thread_id)

    def close_thread(self, thread_id: str):
        if self.thread_ids.get(thread_id):
            del self.thread_ids[thread_id]
        store_manager.close_thread(thread_id)

    async def stop(self):
        print("wait for the last heartbeat to stop")
        self.is_running = False
        if self.task is not None:
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def stop_force(self):
        self.is_running = False
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

async def init_thread(thread_id: str):
    await heartbeat_manager.init_thread(thread_id)

def close_thread(thread_id: str):
    heartbeat_manager.close_thread(thread_id)


async def init_graphs(heartbeat_interval: float = 5.0) -> tuple[BaseChatModel, BaseChatModel, Embeddings, MemoryManager, MainGraph, RecycleGraph, RetrieveGraph]:
    global llm_for_chat, llm_for_structured, embeddings, memory_manager, main_graph, recycle_graph, retrieve_graph, heartbeat_manager

    req_envs = ["CHAT_MODEL_NAME", "STRUCTURED_MODEL_NAME"]
    for e in req_envs:
        if not os.getenv(e):
            raise Exception(f"{e} is not set")

    await store_setup()
    await load_config()

    def create_model(model_name: str, enable_thinking: Optional[bool] = None):
        if model_name.startswith('qwen-'):
            return ChatQwen(
                model=model_name,
                max_retries=2,
                timeout=60.0,
                enable_thinking=enable_thinking,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        elif model_name.startswith(('qwq-', 'qvq-')):
            return ChatQwQ(
                model=model_name,
                max_retries=2,
                timeout=60.0,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
        else:
            return ChatOpenAI(
                model_name=model_name,
                max_retries=2,
                timeout=60.0,
            )

    chat_enable_thinking = os.getenv("CHAT_MODEL_ENABLE_THINKING", '').lower()
    if chat_enable_thinking == "true":
        chat_enable_thinking = True
    elif chat_enable_thinking == "false":
        chat_enable_thinking = False
    else:
        chat_enable_thinking = None
    llm_for_chat = create_model(os.getenv("CHAT_MODEL_NAME", ""), chat_enable_thinking)

    structured_enable_thinking = os.getenv("STRUCTURED_MODEL_ENABLE_THINKING", '').lower()
    if structured_enable_thinking == "true":
        structured_enable_thinking = True
    elif structured_enable_thinking == "false":
        structured_enable_thinking = False
    else:
        structured_enable_thinking = None
    llm_for_structured = create_model(os.getenv("STRUCTURED_MODEL_NAME", ""), structured_enable_thinking)

    embeddings = DashScopeEmbeddings(model="text-embedding-v4")
    memory_manager = await MemoryManager.create(embeddings)

    retrieve_graph = await RetrieveGraph.create(llm_for_structured, memory_manager)
    recycle_graph = await RecycleGraph.create(llm_for_structured, memory_manager)
    main_graph = await MainGraph.create(llm_for_chat, retrieve_graph, recycle_graph, memory_manager, [retrieve_memories], llm_for_structured)

    heartbeat_manager = await HeartbeatManager.create(heartbeat_interval)

    return llm_for_chat, llm_for_structured, embeddings, memory_manager, main_graph, recycle_graph, retrieve_graph

async def close_graphs():
    await heartbeat_manager.stop()
    await main_graph.conn.close()
    await recycle_graph.conn.close()
    await retrieve_graph.conn.close()
    await stop_listener()
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
                _main_messages = await main_graph.get_messages(thread_id)
                if _main_messages:
                    _last_messages = _main_messages[-message_count:]
                    remove_messages = [RemoveMessage(id=_message.id) for _message in _last_messages if _message.id]
                    await main_graph.update_messages(thread_id, remove_messages)
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
                thread_settings = await store_manager.get_settings(thread_id)
                thread_settings.main.role_prompt = role_prompt
                message = "Role prompt set successfully"
            else:
                message = "Role prompt cannot be empty"

    elif user_input == "/load_config" or user_input.startswith("/load_config "):
        if user_input == "/load_config":
            result = await load_config(thread_id, force=True)
        else:
            splited_input = user_input.split(" ")
            if splited_input[1]:
                if splited_input[1] == "__all__":
                    result = await load_config(force=True)
                else:
                    result = await load_config(splited_input[1], force=True)
        if result:
            await store_manager.init_thread(thread_id)
            message = "配置文件已加载。"
        else:
            message = "不存在指定的线程ID。"

    elif user_input == "/wakeup":
        await main_graph.graph.aupdate_state(config, {"active_time_seconds": 0.0, "self_call_time_secondses": [], "wakeup_call_time_seconds": 0.0})
        message = "已唤醒agent（重置自我调用相关状态），这可能导致agent对prompt有些误解。"

    elif user_input == "/messages":
        main_messages = await main_graph.get_messages(thread_id)
        if main_messages:
            message = parse_messages(main_messages)
        else:
            message = "消息为空。"

    elif user_input == "/tokens":
        main_state = await main_graph.graph.aget_state(config)
        if main_messages := main_state.values.get("messages"):
            message = count_tokens_approximately(main_messages)
        else:
            message = "消息为空。"

    elif user_input.startswith("/memories ") or user_input == "/memories":
        if user_input == "/memories":
            message = "Usage: /memories <original|summary|semantic> [offset] [limit]"
        else:
            splited_input = user_input.split(" ")
            memory_type = splited_input[1]
            limit = int(splited_input[3]) if len(splited_input) > 3 else 6
            offset = int(splited_input[2]) if len(splited_input) > 2 else None
            get_result = await memory_manager.aget(thread_id=thread_id, memory_type=memory_type, limit=limit, offset=offset)
            message = '\n\n\n'.join([f'id: {get_result["ids"][i]}\n\ncontent: {get_result["documents"][i]}\n\nmetadata: {'\n'.join([f'{k}: {str(v)}' for k, v in get_result["metadatas"][i].items()])}' for i in range(len(get_result["ids"]))])
            if not message:
                message = "没有找到任何记忆。"

    elif user_input == "/reset" or user_input.startswith("/reset "):
        if user_input == "/reset":
            message = "Usage: /reset <all|config>"
        else:
            splited_input = user_input.split(" ")
            if len(splited_input) >= 2 and splited_input[1]:
                reset_type = splited_input[1]
                if reset_type == 'config':
                    await store_adelete_namespace((thread_id, 'model', 'settings'))
                    await store_manager.init_thread(thread_id)
                    message = "已重置该线程配置。"
                elif reset_type == 'all':
                    main_state = await main_graph.graph.aget_state(config)
                    if not main_state.next:
                        close_thread(thread_id)
                        await memory_manager.delete_timer_from_db(thread_id)
                        await main_graph.graph.checkpointer.adelete_thread(thread_id)
                        await recycle_graph.graph.checkpointer.adelete_thread(thread_id)
                        await retrieve_graph.graph.checkpointer.adelete_thread(thread_id)
                        memory_manager.delete_collection(thread_id, "original")
                        memory_manager.delete_collection(thread_id, "summary")
                        memory_manager.delete_collection(thread_id, "semantic")
                        await store_adelete_namespace((thread_id,))
                        await load_config(thread_id)
                        await init_thread(thread_id)
                        message = "已重置该线程所有数据。"
                    else:
                        message = "线程运行时无法重置所有数据。"


    #return {"name": "log", "args": {"message": message}}
    await event_queue.put({"thread_id": thread_id, "name": "log", "args": {"message": message}, "id": 'command-' + str(uuid4())})
