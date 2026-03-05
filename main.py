import os, sys
import asyncio
from loguru import logger
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT
from become_human.plugins import *
from become_human import sprite_manager

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

sprite_id = os.getenv('MAIN_SPRITE_ID', "default_sprite_1")
user_name = os.getenv('MAIN_USER_NAME')

last_message = ''
@sprite_manager.on_sprite_output
def print_message(name: str, args: dict, not_completed: bool = False):
    global last_message
    if name == SEND_MESSAGE or name == 'log':
        if not_completed:
            print(args[SEND_MESSAGE_CONTENT].replace(last_message, '', 1), end='', flush=True)
            last_message = args[SEND_MESSAGE_CONTENT]
        else:
            print(args[SEND_MESSAGE_CONTENT].replace(last_message, '', 1), flush=True)
            last_message = ''

async def main():
    await sprite_manager.init_manager(plugins=[
        PresencePlugin,
        MemoryPlugin,
        InstructionPlugin,
        ReminderPlugin,
        TimeIncrementerPlugin,
    ], heartbeat_interval=10)
    await sprite_manager.init_sprite(sprite_id)

    while True:
        user_input = await asyncio.to_thread(input)
        if user_input:
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            else:
                sprite_manager.call_sprite_for_user_with_command_nowait(
                    sprite_id=sprite_id,
                    user_input=user_input,
                    user_name=user_name,
                    is_admin=True
                )

    await sprite_manager.close_manager()


if __name__ == "__main__":
    asyncio.run(main())
