from become_human.store import StoreModel, store_asearch
from become_human.store_settings import ThreadSettings
from become_human.store_timers import ThreadTimers
from become_human.store_states import ThreadStates

class ThreadStore(StoreModel):
    _namespace = ('model',)
    _readable_name = "线程存储模型"
    settings: ThreadSettings
    timers: ThreadTimers
    states: ThreadStates

class StoreManager:
    threads: dict[str, ThreadStore]

    def __init__(self) -> None:
        self.threads = {}

    async def init_thread(self, thread_id: str) -> ThreadStore:
        search_items = await store_asearch(('threads', thread_id) + ThreadStore._namespace)
        model = ThreadStore(thread_id, search_items)
        self.threads[thread_id] = model
        return model

    async def get_thread(self, thread_id: str) -> ThreadStore:
        if thread_id not in self.threads:
            return await self.init_thread(thread_id)
        else:
            return self.threads[thread_id]
    
    def get_thread_sync(self, thread_id: str) -> ThreadStore:
        if thread_id not in self.threads.keys():
            raise ValueError(f"线程 {thread_id} 不存在")
        return self.threads[thread_id]

    def close_thread(self, thread_id: str) -> None:
        if thread_id in self.threads.keys():
            del self.threads[thread_id]

    async def get_settings(self, thread_id: str) -> ThreadSettings:
        thread_store = await self.get_thread(thread_id)
        return thread_store.settings

    async def get_timers(self, thread_id: str) -> ThreadTimers:
        thread_store = await self.get_thread(thread_id)
        return thread_store.timers

    async def get_states(self, thread_id: str) -> ThreadStates:
        thread_store = await self.get_thread(thread_id)
        return thread_store.states

store_manager = StoreManager()
