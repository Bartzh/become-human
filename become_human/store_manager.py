from become_human.store import StoreModel, store_asearch
from become_human.store_settings import ThreadSettings

class ThreadStore(StoreModel):
    _namespace = ('model',)
    _readable_name = "线程存储模型"
    settings: ThreadSettings

class StoreManager:
    threads: dict[str, ThreadStore]

    def __init__(self) -> None:
        self.threads = {}

    async def init_thread(self, thread_id: str) -> ThreadStore:
        search_items = await store_asearch((thread_id,) + ThreadStore._namespace, limit=99999)
        model = ThreadStore(thread_id, search_items)
        self.threads[thread_id] = model
        return model

    async def get_thread(self, thread_id: str) -> ThreadStore:
        if thread_id not in self.threads:
            return await self.init_thread(thread_id)
        else:
            return self.threads[thread_id]

    def close_thread(self, thread_id: str) -> None:
        if thread_id in self.threads.keys():
            del self.threads[thread_id]

    async def get_settings(self, thread_id: str) -> ThreadSettings:
        thread_store = await self.get_thread(thread_id)
        return thread_store.settings

store_manager = StoreManager()