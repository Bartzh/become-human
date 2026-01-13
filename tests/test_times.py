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
from tzlocal import get_localzone

from become_human.times import (
    nowtz, utcnow, datetime_to_seconds, seconds_to_datetime, now_seconds,
    SerializableTimeZone, AgentTimeSettings, AgentTimeSetting,
    real_time_to_agent_time,
    real_seconds_to_agent_seconds,
    agent_seconds_to_datetime, now_agent_time, now_agent_seconds,
    format_time, format_seconds, parse_timedelta, Times, AnyTz
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
    
    def test_datetime_to_seconds(self):
        """测试datetime转秒数功能"""
        # 测试正常datetime转换
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        seconds = datetime_to_seconds(dt)
        assert isinstance(seconds, float)
        assert seconds > 0
        
        # 测试无时区信息的datetime
        dt_naive = datetime(2024, 1, 1, 12, 0, 0)
        seconds_naive = datetime_to_seconds(dt_naive)
        assert isinstance(seconds_naive, float)
    
    def test_seconds_to_datetime(self):
        """测试秒数转datetime功能"""
        # 测试正常秒数转换，使用当前时间的秒数
        current_seconds = now_seconds()
        dt = seconds_to_datetime(current_seconds)
        assert isinstance(dt, datetime)
        assert dt.tzinfo == timezone.utc
        
        # 测试从零开始
        dt_zero = seconds_to_datetime(0.0)
        assert dt_zero.year == 1
        assert dt_zero.month == 1
        assert dt_zero.day == 1
        assert dt_zero.tzinfo == timezone.utc
    
    def test_now_seconds(self):
        """测试当前时间秒数获取"""
        result = now_seconds()
        assert isinstance(result, float)
        assert result > 0


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
        assert settings.world_time_setting.agent_time_anchor == 0.0
        assert settings.world_time_setting.real_time_anchor == 0.0
        assert settings.world_time_setting.agent_time_scale == 1.0
        assert settings.subjective_time_setting.agent_time_anchor == 0.0
        assert settings.subjective_time_setting.real_time_anchor == 0.0
        assert settings.subjective_time_setting.agent_time_scale == 1.0
        assert isinstance(settings.time_zone, SerializableTimeZone)
    
    def test_agent_time_settings_custom_values(self):
        """测试自定义值设置"""
        tz = SerializableTimeZone(name="UTC")
        settings = AgentTimeSettings(
            world_time_setting=AgentTimeSetting(
                agent_time_anchor=1000.0,
                real_time_anchor=2000.0,
                agent_time_scale=2.0,
            ),
            time_zone=tz
        )
        assert settings.world_time_setting.agent_time_anchor == 1000.0
        assert settings.world_time_setting.real_time_anchor == 2000.0
        assert settings.world_time_setting.agent_time_scale == 2.0


class TestTimeConversionFunctions:
    """时间转换函数测试类"""
    
    def setup_method(self):
        """设置测试用的时区设置"""
        self.settings_with_anchors = AgentTimeSettings(
            world_time_setting=AgentTimeSetting(
                agent_time_anchor=1000.0,
                real_time_anchor=2000.0,
                agent_time_scale=1.0,
            ),
            time_zone=SerializableTimeZone(name="UTC")
        )
        self.settings_without_anchors = AgentTimeSettings(
            time_zone=SerializableTimeZone(name="UTC")
        )
    
    def test_real_time_to_agent_time_with_anchors(self):
        """测试有锚点时的真实时间转agent时间"""
        real_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        agent_dt = real_time_to_agent_time(
            real_dt,
            self.settings_with_anchors.world_time_setting,
            self.settings_with_anchors.time_zone
        )
        assert isinstance(agent_dt, datetime)
        assert agent_dt.tzinfo is not None
    
    def test_real_time_to_agent_time_without_anchors(self):
        """测试无锚点时的真实时间转agent时间"""
        real_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        agent_dt = real_time_to_agent_time(
            real_dt,
            self.settings_without_anchors.world_time_setting,
            self.settings_without_anchors.time_zone
        )
        assert isinstance(agent_dt, datetime)
        assert agent_dt.tzinfo is not None
    
    def test_real_time_to_agent_time_with_seconds(self):
        """测试使用秒数输入的时间转换"""
        # 使用当前时间的秒数避免溢出
        current_seconds = now_seconds()
        agent_dt = real_time_to_agent_time(
            current_seconds,
            self.settings_with_anchors.world_time_setting,
            self.settings_with_anchors.time_zone
        )
        assert isinstance(agent_dt, datetime)
    
    def test_real_seconds_to_agent_seconds_with_anchors(self):
        """测试有锚点时的真实秒数转agent秒数"""
        current_seconds = now_seconds()
        agent_seconds = real_seconds_to_agent_seconds(current_seconds, self.settings_with_anchors.world_time_setting)
        assert isinstance(agent_seconds, float)
    
    def test_real_seconds_to_agent_seconds_without_anchors(self):
        """测试无锚点时的真实秒数转agent秒数"""
        current_seconds = now_seconds()
        agent_seconds = real_seconds_to_agent_seconds(current_seconds, self.settings_without_anchors.world_time_setting)
        assert agent_seconds == current_seconds
    
    def test_agent_seconds_to_datetime_function(self):
        """测试agent秒数转datetime函数"""
        current_seconds = now_seconds()
        dt = agent_seconds_to_datetime(current_seconds, self.settings_without_anchors.time_zone)
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None
    
    def test_now_agent_time(self):
        """测试获取当前agent时间"""
        agent_time = now_agent_time(self.settings_without_anchors.world_time_setting, self.settings_without_anchors.time_zone)
        assert isinstance(agent_time, datetime)
        assert agent_time.tzinfo is not None
    
    def test_now_agent_seconds(self):
        """测试获取当前agent秒数"""
        agent_seconds = now_agent_seconds(self.settings_without_anchors.world_time_setting)
        assert isinstance(agent_seconds, float)
        assert agent_seconds > 0


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
        """测试秒数格式化"""
        # 使用当前时间的秒数避免溢出
        from become_human.times import now_seconds
        seconds = now_seconds()
        result = format_time(seconds)
        # 检查包含日期部分和时间格式
        assert any(char.isdigit() for char in result)
        assert ":" in result  # 时间格式应该包含冒号
        # 检查包含月份格式（XX格式）
        import re
        month_pattern = r'\b(0[1-9]|1[0-2])\b'  # 匹配01-12格式
        assert re.search(month_pattern, result) is not None
    
    def test_format_time_seconds_with_timezone(self):
        """测试秒数带时区格式化"""
        # 使用当前时间的秒数避免溢出
        from become_human.times import now_seconds
        seconds = now_seconds()
        tz = timezone(timedelta(hours=8))
        result = format_time(seconds, tz)
        # 检查包含日期部分和时间格式
        assert any(char.isdigit() for char in result)
        assert ":" in result  # 时间格式应该包含冒号
    
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
        (3600.0, ["1小时"]),  # 3600秒 = 1小时
        (90.0, ["1分", "30秒"]),  # 90秒 = 1分30秒
        (-3600.0, ["负", "1小时"]),  # 负数
    ])
    def test_format_seconds_parametrized(self, input_value, expected_parts):
        """测试时间差格式化参数化测试"""
        result = format_seconds(input_value)
        for part in expected_parts:
            assert part in result
    
    def test_format_seconds_zero(self):
        """测试零时间差格式化"""
        result = format_seconds(0)
        assert result == ""
    
    def test_format_seconds_complex(self):
        """测试复杂时间差格式化"""
        delta = timedelta(days=2, hours=3, minutes=4, seconds=5)
        result = format_seconds(delta)
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
            world_time_setting=AgentTimeSetting(
                agent_time_anchor=1000.0,
                real_time_anchor=2000.0,
                agent_time_scale=1.0,
            ),
            time_zone=SerializableTimeZone(name="UTC")
        )
    
    def test_times_init_real_time(self):
        """测试使用真实时间初始化"""
        real_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        times = Times.from_time_settings(settings=self.settings, time=real_time)
        
        assert times.real_world_datetime == real_time
        assert times.agent_time_settings == self.settings
        assert times.agent_world_datetime.tzinfo is not None
    
    def test_times_init_real_time_seconds(self):
        """测试使用真实时间秒数初始化"""
        current_seconds = now_seconds()
        times = Times.from_time_settings(settings=self.settings, time=current_seconds)
        
        assert times.real_world_timeseconds == current_seconds
        assert isinstance(times.real_world_datetime, datetime)
    
    def test_times_init_default_time(self):
        """测试使用默认时间初始化"""
        times = Times.from_time_settings(settings=self.settings)
        
        assert isinstance(times.real_world_datetime, datetime)
        assert isinstance(times.real_world_timeseconds, float)
    
    def test_times_from_now(self):
        """测试from_now初始化"""
        local_tz = get_localzone()
        times = Times.from_now()
        
        assert times.real_world_time_zone.tz() == local_tz
        assert times.real_world_datetime.tzinfo is not None
        assert isinstance(times.real_world_timeseconds, float)
        assert times.agent_time_settings.time_zone.tz() == local_tz
        assert times.agent_world_datetime.tzinfo is not None
        assert times.agent_world_timeseconds == times.real_world_timeseconds
        assert times.agent_subjective_datetime.tzinfo is not None
        assert times.agent_subjective_timeseconds == times.real_world_timeseconds


class TestEdgeCasesAndBoundaryConditions:
    """边界条件和异常情况测试"""
    
    def test_datetime_conversion_extreme_values(self):
        """测试极值时间转换"""
        # 测试最早时间
        early_dt = datetime(1, 1, 1, tzinfo=timezone.utc)
        seconds = datetime_to_seconds(early_dt)
        assert seconds == 0.0
        
        # 测试转换回来
        converted_dt = seconds_to_datetime(seconds)
        assert converted_dt == early_dt
    
    def test_time_scale_zero(self):
        """测试时间缩放为零"""
        settings = AgentTimeSettings(
            world_time_setting=AgentTimeSetting(
                agent_time_anchor=1000.0,
                real_time_anchor=2000.0,
                agent_time_scale=0.0,
            ),
            time_zone=SerializableTimeZone(name="UTC")
        )
        
        real_dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        agent_dt = real_time_to_agent_time(real_dt, settings.world_time_setting, settings.time_zone)
        # 时间缩放为零时，所有时间都应该等于锚点时间
        assert isinstance(agent_dt, datetime)
    
    def test_negative_time_scale(self):
        """测试负时间缩放"""
        # 使用较小的锚点值避免溢出
        current_seconds = now_seconds()
        settings = AgentTimeSettings(
            world_time_setting=AgentTimeSetting(
                agent_time_anchor=current_seconds,
                real_time_anchor=current_seconds,
                agent_time_scale=-1.0,
            ),
            time_zone=SerializableTimeZone(name="UTC")
        )
        
        real_dt = datetime.now(timezone.utc)
        agent_dt = real_time_to_agent_time(real_dt, settings.world_time_setting, settings.time_zone)
        assert isinstance(agent_dt, datetime)
    
    def test_large_time_scale(self):
        """测试大时间缩放"""
        # 使用较小的锚点值避免溢出
        current_seconds = now_seconds()
        settings = AgentTimeSettings(
            world_time_setting=AgentTimeSetting(
                agent_time_anchor=current_seconds,
                real_time_anchor=current_seconds,
                agent_time_scale=100.0,
            ),
            time_zone=SerializableTimeZone(name="UTC")
        )
        
        real_dt = datetime.now(timezone.utc)
        agent_dt = real_time_to_agent_time(real_dt, settings.world_time_setting, settings.time_zone)
        assert isinstance(agent_dt, datetime)


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
            agent_dt = real_time_to_agent_time(real_dt, settings.subjective_time_setting, settings.time_zone)
            
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
        world_time_setting=AgentTimeSetting(
            agent_time_anchor=1000.0,
            real_time_anchor=2000.0,
            agent_time_scale=1.0,
        ),
        time_zone=SerializableTimeZone(name="UTC")
    )

@pytest.fixture
def sample_datetime():
    """提供示例datetime。在我的测试中，微秒最高可以设置到999969，再往上会溢出"""
    return datetime(9999, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

@pytest.fixture
def sample_seconds():
    """提供示例秒数"""
    return now_seconds()  # 使用当前时间避免溢出


class TestWithFixtures:
    """使用fixture的测试"""
    
    def test_sample_fixture_usage(self, sample_agent_settings, sample_datetime):
        """测试fixture使用"""
        agent_time = real_time_to_agent_time(
            sample_datetime, 
            sample_agent_settings.world_time_setting, 
            sample_agent_settings.time_zone
        )
        assert isinstance(agent_time, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
