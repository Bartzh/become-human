from become_human.store.base import StoreModel, store_asearch
from become_human.store.settings import BuiltinSettings
from become_human.store.states import BuiltinStates

class BuiltinStore(StoreModel):
    _namespace = ('builtin',)
    _readable_name = "builtin存储模型"
    settings: BuiltinSettings
    states: BuiltinStates

class StoreManager:
    agents: dict[str, dict[type[StoreModel], StoreModel]]
    models: list[type[StoreModel]]

    def __init__(self) -> None:
        self.agents = {}
        self.models = [BuiltinStore]

    async def init_agent(self, agent_id: str) -> None:
        if agent_id not in self.agents.keys():
            self.agents[agent_id] = {}
        for model in self.models:
            search_items = await store_asearch(('agents', agent_id, 'models') + model._namespace)
            self.agents[agent_id][model] = model(agent_id, search_items)

    def close_agent(self, agent_id: str) -> None:
        if agent_id in self.agents.keys():
            del self.agents[agent_id]

    async def get_builtin(self, agent_id: str) -> BuiltinStore:
        if agent_id not in self.agents.keys():
            await self.init_agent(agent_id)
        return self.agents[agent_id][BuiltinStore]

    def get_builtin_sync(self, agent_id: str) -> BuiltinStore:
        if agent_id not in self.agents.keys():
            raise ValueError(f"agent {agent_id} 不存在")
        return self.agents[agent_id][BuiltinStore]

    async def get_settings(self, agent_id: str) -> BuiltinSettings:
        builtin_store = await self.get_builtin(agent_id)
        return builtin_store.settings

    async def get_states(self, agent_id: str) -> BuiltinStates:
        builtin_store = await self.get_builtin(agent_id)
        return builtin_store.states

store_manager = StoreManager()
