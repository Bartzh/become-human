from become_human.store import StoreModel, store_asearch
from become_human.store_settings import AgentSettings
from become_human.store_timers import AgentTimers
from become_human.store_states import AgentStates

class AgentStore(StoreModel):
    _namespace = ('model',)
    _readable_name = "agent存储模型"
    settings: AgentSettings
    timers: AgentTimers
    states: AgentStates

class StoreManager:
    agents: dict[str, AgentStore]

    def __init__(self) -> None:
        self.agents = {}

    async def init_agent(self, agent_id: str) -> AgentStore:
        search_items = await store_asearch(('agents', agent_id) + AgentStore._namespace)
        model = AgentStore(agent_id, search_items)
        self.agents[agent_id] = model
        return model

    async def get_agent(self, agent_id: str) -> AgentStore:
        if agent_id not in self.agents:
            return await self.init_agent(agent_id)
        else:
            return self.agents[agent_id]

    def get_agent_sync(self, agent_id: str) -> AgentStore:
        if agent_id not in self.agents.keys():
            raise ValueError(f"agent {agent_id} 不存在")
        return self.agents[agent_id]

    def close_agent(self, agent_id: str) -> None:
        if agent_id in self.agents.keys():
            del self.agents[agent_id]

    async def get_settings(self, agent_id: str) -> AgentSettings:
        agent_store = await self.get_agent(agent_id)
        return agent_store.settings

    async def get_timers(self, agent_id: str) -> AgentTimers:
        agent_store = await self.get_agent(agent_id)
        return agent_store.timers

    async def get_states(self, agent_id: str) -> AgentStates:
        agent_store = await self.get_agent(agent_id)
        return agent_store.states

store_manager = StoreManager()
