from become_human.store.base import StoreModel, StoreField
from become_human.times import Times, SpriteTimeSettings, TimestampUs

class BuiltinStates(StoreModel):
    _namespace = ('states',)
    _readable_name = "builtin状态"
    first_init_timestampus: TimestampUs = StoreField(default_factory=TimestampUs.now, readable_name="首次初始化时间戳（微秒）", frozen=True)
    last_updated_times: Times = StoreField(default_factory=lambda: Times.from_time_settings(SpriteTimeSettings()), readable_name="最后更新时间Times")
    #last_response_times: Times = StoreField(default_factory=Times, readable_name="最后回复时间Times")
    #active_timeseconds: float = StoreField(default=0.0, readable_name="活跃时间截止时间")
