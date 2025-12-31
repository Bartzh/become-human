from become_human.store.base import StoreModel, store_asearch
from become_human.store.settings import AgentSettings
from become_human.store.schedules import AgentSchedules
from become_human.store.states import AgentStates

class AgentStore(StoreModel):
    _namespace = ('model',)
    _readable_name = "agent存储模型"
    settings: AgentSettings
    schedules: AgentSchedules
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

    async def get_schedules(self, agent_id: str) -> AgentSchedules:
        agent_store = await self.get_agent(agent_id)
        return agent_store.schedules

    async def get_states(self, agent_id: str) -> AgentStates:
        agent_store = await self.get_agent(agent_id)
        return agent_store.states

store_manager = StoreManager()
