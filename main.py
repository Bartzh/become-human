from become_human import init_graphs, close_graphs, command_processing, init_thread, event_queue, stream_graph_updates
import os
import asyncio

thread_id = "default_thread"
config = {"configurable": {"thread_id": thread_id}}
user_name = os.getenv('USER_NAME')

last_message = ''
def _print(item: dict):
    global last_message
    if item['name'] == 'send_message' or item['name'] == 'log':
        if item.get('not_completed'):
            print(item['args']['message'].replace(last_message, '', 1), end='', flush=True)
            last_message = item['args']['message']
        else:
            print(item['args']['message'].replace(last_message, '', 1), flush=True)
            last_message = ''

async def main():
    await init_graphs(30)
    await init_thread(config['configurable']['thread_id'])
    task = asyncio.create_task(event_listener())

    while True:
        user_input = await asyncio.to_thread(input)
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        elif user_input.startswith("/"):
            await command_processing(thread_id, user_input)
            continue
        asyncio.create_task(stream_graph_updates(user_input, thread_id, user_name=user_name))

    task.cancel()
    await close_graphs()

async def event_listener():
    while True:
        event = await event_queue.get()
        _print(event)


if __name__ == "__main__":
    asyncio.run(main())
