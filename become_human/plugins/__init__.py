from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from become_human.plugins.reminder import ReminderPlugin
    from become_human.plugins.time_incrementer import TimeIncrementerPlugin
    from become_human.plugins.presence import PresencePlugin
    from become_human.plugins.memory import MemoryPlugin
    from become_human.plugins.instruction import InstructionPlugin
    from become_human.plugins.note import NotePlugin
    from become_human.plugins.cli import SimpleCLI
    from become_human.plugins.api import SimpleAPI

__all__ = [
    'ReminderPlugin',
    'TimeIncrementerPlugin',
    'PresencePlugin',
    'MemoryPlugin',
    'InstructionPlugin',
    'NotePlugin',
    'SimpleCLI',
    'SimpleAPI',
]

def __getattr__(name):
    if name == 'ReminderPlugin':
        from become_human.plugins.reminder import ReminderPlugin
        return ReminderPlugin
    if name == 'TimeIncrementerPlugin':
        from become_human.plugins.time_incrementer import TimeIncrementerPlugin
        return TimeIncrementerPlugin
    if name == 'PresencePlugin':
        from become_human.plugins.presence import PresencePlugin
        return PresencePlugin
    if name == 'MemoryPlugin':
        from become_human.plugins.memory import MemoryPlugin
        return MemoryPlugin
    if name == 'InstructionPlugin':
        from become_human.plugins.instruction import InstructionPlugin
        return InstructionPlugin
    if name == 'NotePlugin':
        from become_human.plugins.note import NotePlugin
        return NotePlugin
    if name =='SimpleCLI':
        from become_human.plugins.cli import SimpleCLI
        return SimpleCLI
    if name == 'SimpleAPI':
        from become_human.plugins.api import SimpleAPI
        return SimpleAPI
    raise AttributeError(f"module {__name__} has no attribute {name}")
