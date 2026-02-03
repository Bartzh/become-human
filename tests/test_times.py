"""time模块的全面单元测试

测试覆盖：
- 时间转换函数
- Agent时区模型
- 时间格式化函数
- 时间差解析函数
- Times工具类
"""

import pytest
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pydantic import ValidationError

from become_human.times import (
    nowtz, utcnow, timedelta_to_microseconds, TimestampUs,
    SerializableTimeZone, AgentTimeSettings,
    format_time, format_duration, parse_timedelta, Times
)


class TestBasicTimeFunctions:
    """基础时间函数测试类"""
    
    def test_nowtz(self):
        """测试本地时区当前时间获取"""
        # 测试返回值类型
        result = nowtz()
        assert isinstance(result, datetime)
        assert result.tzinfo is not None
    
    def test_utcnow(self):
        """测试UTC当前时间获取"""
        # 测试返回值类型
        result = utcnow()
        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc
    
    def test_timedelta_to_microseconds(self):
        """测试timedelta转微秒数功能"""
        # 测试正常timedelta转换
        td = timedelta(hours=2, minutes=30, seconds=15)
        microseconds = timedelta_to_microseconds(td)
        assert isinstance(microseconds, int)
        assert microseconds == 2 * 3600 * 1000000 + 30 * 60 * 1000000 + 15 * 1000000
        
        # 测试零timedelta
        td_zero = timedelta(0)
        assert timedelta_to_microseconds(td_zero) == 0

class TestAgentTimeZone:
    """AgentTimeZone类测试"""
    
    def test_agent_timezone_with_name_only(self):
        """测试仅使用名称创建时区"""
        tz = SerializableTimeZone(name="UTC")
        result = tz.tz()
        # 根据实际实现，ZoneInfo可能不支持UTC名称
        if isinstance(result, ZoneInfo):
            # 如果是ZoneInfo，检查是否有效
            assert hasattr(result, 'key')
        else:
            # 如果是timezone，应该是UTC
            assert result == timezone.utc
    
    def test_agent_timezone_with_offset(self):
        """测试使用偏移量创建时区"""
        tz = SerializableTimeZone(name="UTC+8", offset=28800.0)  # 8小时
        result = tz.tz()
        assert isinstance(result, timezone)
    
    def test_agent_timezone_invalid_offset(self):
        """测试无效偏移量"""
        # 测试超出范围的偏移量
        with pytest.raises(ValidationError):
            SerializableTimeZone(name="Test", offset=100000.0)  # 超过86400


class TestAgentTimeSettings:
    """AgentTimeSettings类测试"""
    
    def test_agent_time_settings_defaults(self):
        """测试默认设置创建"""
        settings = AgentTimeSettings()
        now_timestampus = TimestampUs.now()
        assert settings.world_agent_anchor == 0
        assert settings.world_real_anchor == 0
        assert settings.world_scale == 1
        assert settings.subjective_agent_anchor == 0
        assert abs(settings.subjective_real_anchor - now_timestampus) < 1_000_000
        assert settings.subjective_scale == 1
        assert isinstance(settings.time_zone, SerializableTimeZone)
    
    def test_agent_time_settings_custom_values(self):
        """测试自定义值设置"""
        tz = SerializableTimeZone(name="UTC")
        settings = AgentTimeSettings(
            world_agent_anchor=TimestampUs(1000),
            world_real_anchor=2000,
            world_scale=2,
            time_zone=tz
        )
        assert settings.world_agent_anchor == 1000
        assert settings.world_real_anchor == 2000
        assert settings.world_scale == 2


class TestTimeConversionFunctions:
    """时间转换函数测试类"""
    
    def setup_method(self):
        """设置测试用的时区设置"""
        self.settings_with_anchors = AgentTimeSettings(
            world_agent_anchor=1000,
            world_real_anchor=2000,
            world_scale=1,
            time_zone=SerializableTimeZone(name="UTC")
        )
        self.settings_without_anchors = AgentTimeSettings(
            time_zone=SerializableTimeZone(name="UTC")
        )
    
    def test_real_time_to_agent_time_with_anchors(self):
        """测试有锚点时的真实时间转agent时间"""
        real_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        agent_dt = self.settings_with_anchors.to_world_datetime(real_dt)
        assert isinstance(agent_dt, datetime)
        assert agent_dt.tzinfo is not None
    
    def test_real_time_to_agent_time_without_anchors(self):
        """测试无锚点时的真实时间转agent时间"""
        real_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        agent_dt = self.settings_without_anchors.to_world_datetime(real_dt)
        assert isinstance(agent_dt, datetime)
        assert agent_dt.tzinfo is not None
    
    def test_real_time_to_agent_time_with_timestampus(self):
        """测试使用TimestampUs输入的时间转换"""
        # 使用当前时间的秒数避免溢出
        current_timestampus = TimestampUs.now()
        agent_dt = self.settings_with_anchors.to_world_datetime(current_timestampus)
        assert isinstance(agent_dt, datetime)


class TestFormattingFunctions:
    """格式化函数测试类"""
    
    def test_format_time_none(self):
        """测试None时间格式化"""
        result = format_time(None)
        assert result == "未知时间"
    
    def test_format_time_datetime(self):
        """测试datetime格式化"""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = format_time(dt)
        assert "2024-01-01" in result
        assert "12:00:00" in result
        assert "Monday" in result
    
    def test_format_time_seconds(self):
        """测试TimestampUs格式化"""
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        timestampus = TimestampUs(dt)
        result = format_time(timestampus)
        assert "2024-01-01" in result
        assert "12:00:00" in result
        assert "Monday" in result
    
    def test_format_time_seconds_with_timezone(self):
        """测试秒数带时区格式化"""
        # 使用当前时间的秒数避免溢出
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        timestampus = TimestampUs(dt)
        result = format_time(timestampus)
        tz = timezone(timedelta(hours=8))
        result = format_time(timestampus, tz)
        assert "2024-01-01" in result
        assert "20:00:00" in result
        assert "Monday" in result
    
    def test_format_time_invalid_data(self):
        """测试无效时间数据格式化"""
        # 测试极大值或损坏的数据
        invalid_seconds = float('inf')
        result = format_time(invalid_seconds)
        assert result == "时间信息损坏"
    
    @pytest.mark.parametrize("input_value,expected_parts", [
        (timedelta(hours=2), ["2小时"]),
        (timedelta(minutes=30), ["30分"]),
        (timedelta(days=1), ["1天"]),
        (timedelta(weeks=1), ["7天"]),  # 1周 = 7天
        (3600000000, ["1小时"]),  # 3600秒 = 1小时
        (90000000, ["1分", "30秒"]),  # 90秒 = 1分30秒
        (-3600000000, ["负", "1小时"]),  # 负数
    ])
    def test_format_duration_parametrized(self, input_value, expected_parts):
        """测试时长格式化参数化测试"""
        result = format_duration(input_value)
        for part in expected_parts:
            assert part in result
    
    def test_format_duration_zero(self):
        """测试零时长格式化"""
        result = format_duration(0)
        assert result == ""
    
    def test_format_duration_complex(self):
        """测试复杂时长格式化"""
        delta = timedelta(days=2, hours=3, minutes=4, seconds=5)
        result = format_duration(delta)
        # timedelta(days=2) -> datetime(1,1,3) -> 减1 -> datetime(1,1,2) -> 2天
        assert "2天" in result  # 2天保持不变
        assert "3小时" in result
        assert "4分" in result
        assert "5秒" in result


class TestParseTimedelta:
    """parse_timedelta函数测试类"""
    
    @pytest.mark.parametrize("input_str,expected_delta", [
        ("2h", timedelta(hours=2)),
        ("30m", timedelta(minutes=30)),
        ("1d", timedelta(days=1)),
        ("1w", timedelta(weeks=1)),
        ("2h30m", timedelta(hours=2, minutes=30)),
        ("1d2h3m4s", timedelta(days=1, hours=2, minutes=3, seconds=4)),
        ("1.5h", timedelta(hours=1.5)),
    ])
    def test_parse_timedelta_valid(self, input_str, expected_delta):
        """测试有效时间字符串解析"""
        result = parse_timedelta(input_str)
        assert result == expected_delta
    
    def test_parse_timedelta_zero_string(self):
        """测试零值字符串应该抛出异常"""
        with pytest.raises(ValueError):
            parse_timedelta("0")
    
    def test_parse_timedelta_invalid_string(self):
        """测试无效时间字符串"""
        with pytest.raises(ValueError):
            parse_timedelta("invalid")
    
    def test_parse_timedelta_empty_string(self):
        """测试空字符串"""
        with pytest.raises(ValueError):
            parse_timedelta("")
    
    def test_parse_timedelta_case_insensitive(self):
        """测试大小写不敏感"""
        result = parse_timedelta("2H")  # 大写H
        expected = timedelta(hours=2)
        assert result == expected
    
    def test_parse_timedelta_partial_match(self):
        """测试部分匹配"""
        # 应该只匹配有效的部分，"30invalid"不匹配，只有"1h"被匹配
        result = parse_timedelta("1h30invalid")
        expected = timedelta(hours=1)  # 只有1小时被匹配
        assert result == expected


class TestTimesClass:
    """Times类测试"""
    
    def setup_method(self):
        """设置测试数据"""
        self.settings = AgentTimeSettings(
            world_agent_anchor=1000,
            world_real_anchor=2000,
            world_scale=1,
            time_zone=SerializableTimeZone(name="UTC")
        )
    
    def test_times_init_real_time(self):
        """测试使用真实时间初始化"""
        real_time = nowtz()
        times = Times.from_time_settings(settings=self.settings, real_time=real_time)
        
        assert times.real_world_datetime == real_time
        assert times.agent_time_settings == self.settings
        assert times.agent_world_datetime.tzinfo is not None
    
    def test_times_init_real_time_seconds(self):
        """测试使用真实时间秒数初始化"""
        current_datetime = nowtz()
        times = Times.from_time_settings(settings=self.settings, real_time=current_datetime)
        
        assert times.real_world_timestampus == TimestampUs(current_datetime)
        assert isinstance(times.real_world_datetime, datetime)
    
    def test_times_init_default_time(self):
        """测试使用默认时间初始化"""
        times = Times.from_time_settings(settings=self.settings)
        
        assert isinstance(times.real_world_datetime, datetime)
        assert isinstance(times.real_world_timestampus, TimestampUs)


class TestEdgeCasesAndBoundaryConditions:
    """边界条件和异常情况测试"""
    
    def test_datetime_conversion_extreme_values(self):
        """测试极值时间转换"""
        # 测试最早时间
        early_dt = datetime(1, 1, 1, tzinfo=timezone.utc)
        timestampus = TimestampUs(early_dt)
        assert timestampus == 0
        
        # 测试转换回来
        converted_dt = timestampus.to_datetime()
        assert converted_dt == early_dt
    
    def test_time_scale_zero(self):
        """测试时间缩放为零"""
        settings = AgentTimeSettings(
            world_agent_anchor=1000,
            world_real_anchor=2000,
            world_scale=0,
            time_zone=SerializableTimeZone(name="UTC")
        )
        
        real_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        agent_dt = settings.to_world_datetime(real_dt)
        # 时间缩放为零时，所有时间都应该等于锚点时间
        assert TimestampUs(agent_dt) == 1000
    
    def test_negative_time_scale(self):
        """测试负时间缩放"""
        settings = AgentTimeSettings(
            world_agent_anchor=1000,
            world_real_anchor=2000,
            world_scale=-1,
            time_zone=SerializableTimeZone(name="UTC")
        )
        
        agent_dt = settings.to_world_datetime(TimestampUs(3000))
        assert TimestampUs(agent_dt) == 0


class TestTimezoneHandling:
    """时区处理测试"""
    
    def test_various_timezones(self):
        """测试各种时区"""
        timezones = [
            "UTC",
            "Asia/Shanghai",
            "America/New_York",
            "Europe/London"
        ]
        
        for tz_name in timezones:
            settings = AgentTimeSettings(
                time_zone=SerializableTimeZone(name=tz_name)
            )
            
            real_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
            agent_dt = settings.to_world_datetime(real_dt)
            
            assert isinstance(agent_dt, datetime)
            assert agent_dt.tzinfo is not None
    
    def test_timezone_offset_conversion(self):
        """测试时区偏移量转换"""
        # 测试不同的时区偏移
        offsets = [
            0,      # UTC
            28800,  # UTC+8
            -14400, # UTC-4
            3600    # UTC+1
        ]
        
        for offset in offsets:
            tz = SerializableTimeZone(name=f"Offset{offset}", offset=offset)
            result = tz.tz()
            assert isinstance(result, timezone)
    
    def test_timezone_with_float_offset(self):
        """测试浮点数时区偏移"""
        # 测试半小时偏移
        tz = SerializableTimeZone(name="UTC+5:30", offset=19800.0)  # 5.5小时
        result = tz.tz()
        assert isinstance(result, timezone)


# Pytest fixtures
@pytest.fixture
def sample_agent_settings():
    """提供示例agent时间设置"""
    return AgentTimeSettings(
        world_agent_anchor=1000.0,
        world_real_anchor=2000.0,
        world_scale=1.0,
        time_zone=SerializableTimeZone(name="UTC")
    )

@pytest.fixture
def sample_datetime():
    """提供示例datetime。在我的测试中，微秒最高可以设置到999969，再往上会溢出"""
    return datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=timezone.utc)


class TestWithFixtures:
    """使用fixture的测试"""
    
    def test_sample_fixture_usage(self, sample_agent_settings, sample_datetime):
        """测试fixture使用"""
        agent_time = sample_agent_settings.to_world_datetime(sample_datetime)
        assert isinstance(agent_time, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
