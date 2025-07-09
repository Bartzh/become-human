from become_human.graph_main import MainGraph
from become_human.graph_recycle import RecycleGraph
from become_human.graph_retrieve import RetrieveGraph
from become_human.memory import MemoryManager
from become_human.utils import make_sure_path_exists
from become_human.config import load_config

from typing import Optional, Annotated, Literal, Union
from langchain_openai import ChatOpenAI
#from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_qwq import ChatQwen
from langchain_core.messages import AIMessageChunk, HumanMessage, RemoveMessage, BaseMessage, AIMessage, AnyMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig
from langchain_community.embeddings import DashScopeEmbeddings

import asyncio

make_sure_path_exists()

config = {"configurable": {"thread_id": "default"}}


@tool(response_format="content_and_artifact")
async def retrieve_memories(search_string: Annotated[str, "要检索的内容"], config: RunnableConfig) -> str:
    """从数据库（大脑）中检索记忆"""
    result = await retrieve_graph.graph.ainvoke({"input": search_string, "type": "active"}, config)
    content = result["output"]
    artifact = {"dont_store": True}
    return content, artifact



async def command_processing(thread_id: str, user_input: str, main_graph: MainGraph, recycle_graph: RecycleGraph, retrieve_graph: RetrieveGraph) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    message = 'invalid command'
    if user_input.startswith("@get_state ") or user_input == "@get_state":
        if user_input == "@get_state":
            message = '''Usage: @get_state <graph_name> [key]
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

    elif user_input.startswith("@delete_last_messages ") or user_input == "@delete_last_messages":
        if user_input == "@delete_last_messages":
            message_count = 1
        else:
            splited_input = user_input.split(" ")
            message_count = int(splited_input[1])
        if message_count > 0:
            _main_state = await main_graph.graph.aget_state(config)
            _main_messages: list[AnyMessage] = _main_state.values.get('messages')
            if _main_messages:
                _last_messages = _main_messages[-message_count:]
                remove_messages = [RemoveMessage(id=_message.id) for _message in _last_messages]
                print(remove_messages)
                await main_graph.graph.aupdate_state(config, {"messages": remove_messages})
                message = f"Last {message_count} messages deleted. (此功能目前无效，不知道为什么)"
            else:
                message = "No messages found"

    elif user_input == "@reload_config":
        load_config()
        message = "配置文件已重新加载。"

    return {"name": "log", "args": {"message": message}}

def _print(item: dict):
    if item['name'] == 'send_message' or item['name'] == 'log':
        if item['args'].get('isCompleted'):
            print(item['args']['message'], flush=True)
        else:
            print(item['args']['message'], end='', flush=True)




from langchain_core.messages.utils import count_tokens_approximately


async def main():
    global llm, embeddings, main_graph, recycle_graph, retrieve_graph

    llm = ChatQwen(
        #model="qwen-max-2025-01-25",
        model="qwen-plus-2025-04-28",
        #model="qwen3-235b-a22b",
        #top_p=0.8,
        max_retries=2,
        enable_thinking=True,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #parallel_tool_calls=True
    )

    llm_no_thinking = ChatQwen(
        #model="qwen-max-2025-01-25",
        model="qwen-plus-2025-04-28",
        #model="qwen3-235b-a22b",
        #top_p=0.8,
        max_retries=2,
        enable_thinking=False,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    embeddings = DashScopeEmbeddings(model="text-embedding-v3")

    memory_manager = await MemoryManager.create(embeddings)

    retrieve_graph = await RetrieveGraph.create(llm_no_thinking, memory_manager)
    main_graph = await MainGraph.create(llm, retrieve_graph, memory_manager, [retrieve_memories], llm_no_thinking)
    recycle_graph = await RecycleGraph.create(llm_no_thinking, memory_manager)

    await memory_manager.init_thread(config['configurable']['thread_id'])

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # 添加指令检测
        elif user_input.startswith("@"):
            log = await command_processing('default_thread', user_input, main_graph, recycle_graph, retrieve_graph)
            _print(log)
            continue


        async for item in main_graph.stream_graph_updates(user_input, config, user_name="Bart"):
            #print(item)
            _print(item)
        main_state = await main_graph.graph.aget_state(config)
        main_messages = main_state.values["messages"]
        print(f'{count_tokens_approximately(main_messages)} tokens')
        recycle_response = await recycle_graph.graph.ainvoke({"input_messages": main_messages}, config)
        if recycle_response.get("success"):
            print(recycle_response)
            await main_graph.graph.aupdate_state(config, {"messages": recycle_response["remove_messages"]})

    #memory_manager.stop_periodic_task()
    await main_graph.conn.close()
    await recycle_graph.conn.close()
    await retrieve_graph.conn.close()

if __name__ == "__main__":
    asyncio.run(main())

