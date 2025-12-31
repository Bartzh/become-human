from become_human.store import StoreModel, StoreField

class AgentStates(StoreModel):
    _namespace = ('states',)
    _readable_name = "agent状态"
    last_update_real_timeseconds: float = StoreField(default=0.0, readable_name="最后更新时间")
    last_update_agent_timeseconds: float = StoreField(default=0.0, readable_name="最后更新agent时间")
    is_active: bool = StoreField(default=False, readable_name="是否活跃。一段时间无用户消息后会退出活跃状态，模拟agent去做别的事情了。agent设计不应依赖此状态，这只是过渡方案")
    is_first_time: bool = StoreField(default=True, readable_name="是否首次运行")
