from become_human.store.base import StoreModel, StoreField
from become_human.times import Times, SpriteTimeSettings, TimestampUs
from become_human.names import PROJECT_NAME

class SpritedStates(StoreModel):
    _namespace = (PROJECT_NAME,)
    _title = "sprited状态"
    born_at: TimestampUs = StoreField(default_factory=TimestampUs.now, title="首次初始化现实时间戳（微秒）", frozen=True)
    last_updated_times: Times = StoreField(default_factory=lambda: Times.from_time_settings(SpriteTimeSettings()), title="最后更新时间Times")
