from become_human import init_graphs, close_graphs, command_processing, init_thread, event_queue, stream_graph_updates
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT
import os
import asyncio

thread_id = "default_thread_1"
user_name = os.getenv('USER_NAME')

last_message = ''
def _print(item: dict):
    global last_message
    if item['name'] == SEND_MESSAGE or item['name'] == 'log':
        if item.get('not_completed'):
            print(item['args'][SEND_MESSAGE_CONTENT].replace(last_message, '', 1), end='', flush=True)
            last_message = item['args'][SEND_MESSAGE_CONTENT]
        else:
            print(item['args'][SEND_MESSAGE_CONTENT].replace(last_message, '', 1), flush=True)
            last_message = ''

async def main():
    await init_graphs(10)
    await init_thread(thread_id)
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
