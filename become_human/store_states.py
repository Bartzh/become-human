from become_human.store import StoreModel, StoreField

class ThreadStates(StoreModel):
    _namespace = ('states',)
    _readable_name = "线程状态"
    last_update_time_seconds: float = StoreField(default=0.0, readable_name="最后更新时间")
    last_update_agent_time_seconds: float = StoreField(default=0.0, readable_name="最后更新agent时间")
    memory_types: set[str] = StoreField(default_factory=lambda: set(['original', 'summary', 'semantic']), readable_name="记忆类型")
