from langchain_qwq import ChatQwen
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessageChunk, HumanMessage, RemoveMessage, BaseMessage, AIMessage, AnyMessage

from become_human.graph_main import MainGraph
from become_human.graph_recycle import RecycleGraph
from become_human.graph_retrieve import RetrieveGraph
from become_human.memory import MemoryManager
from become_human.config import load_config, get_thread_configs_toml, set_config, verify_toml

from typing import Annotated
import os

@tool(response_format="content_and_artifact")
async def retrieve_memories(search_string: Annotated[str, "要检索的内容"], config: RunnableConfig) -> str:
    """从数据库（大脑）中检索记忆"""
    result = await retrieve_graph.graph.ainvoke({"input": search_string, "type": "active"}, config)
    content = result["output"]
    artifact = {"do_not_store": True}
    return content, artifact


async def init_graphs():
    global llm_for_chat, llm_for_structured, embeddings, memory_manager, main_graph, recycle_graph, retrieve_graph

    envs = ["CHAT_MODEL_NAME", "CHAT_MODEL_ENABLE_THINKING", "STRUCTURED_MODEL_NAME", "STRUCTURED_MODEL_ENABLE_THINKING"]
    for e in envs:
        if not os.getenv(e):
            raise Exception(f"{e} is not set")

    llm_for_chat = ChatQwen(
        #model="qwen-max-2025-01-25",
        model=os.getenv("CHAT_MODEL_NAME"),
        #top_p=0.8,
        max_retries=2,
        enable_thinking=True if os.getenv("CHAT_MODEL_ENABLE_THINKING") == "true" else False,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #parallel_tool_calls=True
    )

    llm_for_structured = ChatQwen(
        #model="qwen-max-2025-01-25",
        model=os.getenv("STRUCTURED_MODEL_NAME"),
        #top_p=0.8,
        max_retries=2,
        enable_thinking=True if os.getenv("STRUCTURED_MODEL_ENABLE_THINKING") == "true" else False,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    embeddings = DashScopeEmbeddings(model="text-embedding-v3")

    memory_manager = await MemoryManager.create(embeddings)

    retrieve_graph = await RetrieveGraph.create(llm_for_structured, memory_manager)
    main_graph = await MainGraph.create(llm_for_chat, retrieve_graph, memory_manager, [retrieve_memories], llm_for_structured)
    recycle_graph = await RecycleGraph.create(llm_for_structured, memory_manager)

    return llm_for_chat, llm_for_structured, embeddings, memory_manager, main_graph, recycle_graph, retrieve_graph

async def close_graphs():
    #memory_manager.stop_periodic_task()
    await main_graph.conn.close()
    await recycle_graph.conn.close()
    await retrieve_graph.conn.close()
    print("Graphs closed")


async def command_processing(thread_id: str, user_input: str) -> dict:
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
                    await main_graph.graph.aupdate_state(config, {"messages": remove_messages})
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

    return {"name": "log", "args": {"message": message}}
