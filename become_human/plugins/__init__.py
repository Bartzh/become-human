from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from become_human.plugins.agent_reminder import AgentReminderPlugin
    from become_human.plugins.agent_time_incrementer import AgentTimeIncrementerPlugin

__all__ = ['AgentReminderPlugin', 'AgentTimeIncrementerPlugin']

def __getattr__(name):
    if name == 'AgentReminderPlugin':
        from become_human.plugins.agent_reminder import AgentReminderPlugin
        return AgentReminderPlugin
    if name == 'AgentTimeIncrementerPlugin':
        from become_human.plugins.agent_time_incrementer import AgentTimeIncrementerPlugin
        return AgentTimeIncrementerPlugin
    raise AttributeError(f"module {__name__} has no attribute {name}")
