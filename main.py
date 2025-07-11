from become_human import init_graphs, close_graphs, command_processing

import asyncio

config = {"configurable": {"thread_id": "default_thread"}}

def _print(item: dict):
    if item['name'] == 'send_message' or item['name'] == 'log':
        if item['args'].get('isCompleted'):
            print(item['args']['message'], flush=True)
        else:
            print(item['args']['message'], end='', flush=True)

from langchain_core.messages.utils import count_tokens_approximately

async def main():
    global llm_for_chat, llm_for_structured, embeddings, memory_manager, main_graph, recycle_graph, retrieve_graph
    llm_for_chat, llm_for_structured, embeddings, memory_manager, main_graph, recycle_graph, retrieve_graph = await init_graphs()

    await memory_manager.init_thread(config['configurable']['thread_id'])

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # 添加指令检测
        elif user_input.startswith("@"):
            log = await command_processing('default_thread', user_input)
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

    await close_graphs()

if __name__ == "__main__":
    asyncio.run(main())

