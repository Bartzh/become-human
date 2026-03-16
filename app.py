from become_human.plugins import *
from become_human import sprite_manager

if __name__ == '__main__':
    sprite_manager.run_standalone([
        PresencePlugin,
        MemoryPlugin,
        InstructionPlugin,
        ReminderPlugin,
        TimeIncrementerPlugin,
        NotePlugin,
        SimpleAPI
    ])
