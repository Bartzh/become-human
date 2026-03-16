import os
import asyncio
from become_human.tools.send_message import SEND_MESSAGE, SEND_MESSAGE_CONTENT
from become_human.plugin import *
from become_human import sprite_manager

sprite_id = os.getenv('CLI_SPRITE_ID', "default_sprite_1")
user_name = os.getenv('CLI_USER_NAME')

last_message = ''
@sprite_manager.on_sprite_output
def print_message(method: str, params: dict, not_completed: bool = False, log: str = ''):
    global last_message
    if method == SEND_MESSAGE:
        if not_completed:
            print(params[SEND_MESSAGE_CONTENT].replace(last_message, '', 1), end='', flush=True)
            last_message = params[SEND_MESSAGE_CONTENT]
        else:
            print(params[SEND_MESSAGE_CONTENT].replace(last_message, '', 1), flush=True)
            last_message = ''
    if log:
        print(log, flush=True)

class SimpleCLI(BasePlugin):
    name = 'simple_cli'
    task: asyncio.Task | None

    def __init__(self):
        self.task = None

    async def on_manager_init(self) -> None:
        await sprite_manager.init_sprite(sprite_id)
        self.task = asyncio.create_task(input_task())

    async def on_manager_close(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

async def input_task():
    while True:
        try:
            user_input = await asyncio.to_thread(input)
        except EOFError:
            break
        if user_input:
            sprite_manager.call_sprite_for_user_with_command_nowait(
                sprite_id=sprite_id,
                user_input=user_input,
                user_name=user_name,
                is_admin=True
            )
