from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from become_human.plugins.reminder import ReminderPlugin
    from become_human.plugins.time_incrementer import TimeIncrementerPlugin
    from become_human.plugins.presence import PresencePlugin
    from become_human.plugins.memory import MemoryPlugin
    from become_human.plugins.instruction import InstructionPlugin

__all__ = [
    'ReminderPlugin',
    'TimeIncrementerPlugin',
    'PresencePlugin',
    'MemoryPlugin',
    'InstructionPlugin'
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
    raise AttributeError(f"module {__name__} has no attribute {name}")
