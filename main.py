from become_human import *
from sprited.plugins import *
from sprited import sprite_manager

if __name__ == '__main__':
    sprite_manager.run_standalone([
        PresencePlugin,
        MemoryPlugin,
        CharacterPlugin,
        InstructionPlugin,
        ReminderPlugin,
        TimeIncrementerPlugin,
        NotePlugin,
        PlanningPlugin,
        SimpleCLI,
    ])
