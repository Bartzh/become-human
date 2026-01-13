import os, sys
import asyncio
from loguru import logger
from become_human.agent_manager import AgentManager
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stdout, level=log_level)
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="2 weeks",
    enqueue=True,
    level=log_level
)

agent_id = os.getenv('MAIN_AGENT_ID', "default_agent_1")
user_name = os.getenv('MAIN_USER_NAME')

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
    agent_manager = await AgentManager.create(10)
    await agent_manager.init_agent(agent_id)
    task = asyncio.create_task(event_listener(agent_manager.event_queue))

    while True:
        user_input = await asyncio.to_thread(input)
        if user_input:
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            else:
                asyncio.create_task(agent_manager.call_agent_for_user_with_command(
                    agent_id=agent_id,
                    user_input=user_input,
                    user_name=user_name,
                    is_admin=True
                ))

    task.cancel()
    await agent_manager.close_manager()

async def event_listener(queue: asyncio.Queue):
    while True:
        event = await queue.get()
        _print(event)


if __name__ == "__main__":
    asyncio.run(main())
