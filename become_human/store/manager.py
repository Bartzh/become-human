from typing import TypeVar
from become_human.store.base import StoreModel, store_asearch
from become_human.store.settings import BuiltinSettings
from become_human.store.states import BuiltinStates

class BuiltinStore(StoreModel):
    _namespace = ('builtin',)
    _title = "builtin存储模型"
    settings: BuiltinSettings
    states: BuiltinStates

StoreType = TypeVar('StoreType', bound=StoreModel)
class StoreManager:
    sprites: dict[str, dict[type[StoreModel], StoreModel]]
    models: list[type[StoreModel]]

    def __init__(self) -> None:
        self.sprites = {}
        self.models = [BuiltinStore]

    async def _init_model(self, sprite_id: str, model: type[StoreModel]) -> None:
        search_items = await store_asearch(('sprites', sprite_id, 'models') + model._namespace)
        self.sprites[sprite_id][model] = model(items=search_items, sprite_id=sprite_id)

    async def init_sprite(self, sprite_id: str) -> None:
        if sprite_id not in self.sprites.keys():
            self.sprites[sprite_id] = {}
        for model in self.models:
            await self._init_model(sprite_id, model)

    async def register_model(self, model: type[StoreModel]) -> None:
        if model not in self.models:
            self.models.append(model)
            for sprite_id in self.sprites.keys():
                await self._init_model(sprite_id, model)

    def get_model(self, sprite_id: str, model: type[StoreType]) -> StoreType:
        if sprite_id not in self.sprites.keys():
            raise ValueError(f"sprite {sprite_id} 未被初始化")
        if model not in self.sprites[sprite_id].keys():
            raise ValueError(f"sprite {sprite_id} 未初始化模型 {model.__name__}")
        return self.sprites[sprite_id][model]

    def close_sprite(self, sprite_id: str) -> None:
        if sprite_id in self.sprites.keys():
            del self.sprites[sprite_id]

    def get_builtin(self, sprite_id: str) -> BuiltinStore:
        return self.get_model(sprite_id, BuiltinStore)

    def get_settings(self, sprite_id: str) -> BuiltinSettings:
        return self.get_builtin(sprite_id).settings

    def get_states(self, sprite_id: str) -> BuiltinStates:
        return self.get_builtin(sprite_id).states

store_manager = StoreManager()
