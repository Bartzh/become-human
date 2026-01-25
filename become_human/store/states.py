from become_human.store.base import StoreModel, StoreField
from become_human.times import Times, AgentTimeSettings

class BuiltinStates(StoreModel):
    _namespace = ('states',)
    _readable_name = "builtin状态"
    last_updated_times: Times = StoreField(default_factory=lambda: Times.from_time_settings(AgentTimeSettings()), readable_name="最后更新时间Times")
    is_active: bool = StoreField(default=False, readable_name="是否活跃。一段时间无用户消息后会退出活跃状态，模拟agent去做别的事情了。agent设计不应依赖此状态，这只是过渡方案")
    is_first_time: bool = StoreField(default=True, readable_name="是否首次运行")
    #last_response_times: Times = StoreField(default_factory=Times, readable_name="最后回复时间Times")
    #active_timeseconds: float = StoreField(default=0.0, readable_name="活跃时间截止时间")
