from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from become_human.presence import PresencePlugin
    from become_human.memory.plugin import MemoryPlugin
    from become_human.character import CharacterPlugin

__all__ = [
    'PresencePlugin',
    'MemoryPlugin',
    'CharacterPlugin',
]

def __getattr__(name):
    if name == 'PresencePlugin':
        from become_human.presence import PresencePlugin
        return PresencePlugin
    if name == 'MemoryPlugin':
        from become_human.memory.plugin import MemoryPlugin
        return MemoryPlugin
    if name == 'CharacterPlugin':
        from become_human.character import CharacterPlugin
        return CharacterPlugin
    raise AttributeError(f"module {__name__} has no attribute {name}")
