"""
用于测试 time.py 模块的 Pytest 测试
这些测试避免导入需要 API 密钥的完整 become_human 包
"""
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import pytest

# time 模块命名空间缓存
_time_namespace = None

def get_time_namespace():
    """将 time 模块加载到独立的命名空间中以避免导入问题"""
    global _time_namespace
    if _time_namespace is not None:
        return _time_namespace
    
    # time.py 文件路径
    time_py_path = os.path.join(os.path.dirname(__file__), '..', 'become_human', 'time.py')
    
    # 读取并执行 time.py 文件
    with open(time_py_path, 'r', encoding='utf-8') as f:
        time_code = f.read()
    
    # 创建隔离的命名空间
    _time_namespace = {}
    exec(time_code, _time_namespace)
    return _time_namespace


class TestDatetimeConversions:
    """测试日期时间转换函数"""
    
    def test_datetime_to_seconds(self):
        """测试 datetime_to_seconds 函数"""
        time_ns = get_time_namespace()
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        seconds = time_ns['datetime_to_seconds'](dt)
        
        assert isinstance(seconds, float)
        # 应该大于 2020 年但小于 2100 年
        base_2020 = time_ns['datetime_to_seconds'](datetime(2020, 1, 1, tzinfo=timezone.utc))
        base_2100 = time_ns['datetime_to_seconds'](datetime(2100, 1, 1, tzinfo=timezone.utc))
        assert seconds > base_2020
        assert seconds < base_2100
    
    def test_seconds_to_datetime(self):
        """测试 seconds_to_datetime 函数"""
        time_ns = get_time_namespace()
        
        # 使用已知值进行测试
        test_dt = datetime(2025, 6, 15, 8, 30, 45, tzinfo=timezone.utc)
        seconds = time_ns['datetime_to_seconds'](test_dt)
        converted_dt = time_ns['seconds_to_datetime'](seconds)
        
        # 检查转换是否一致
        assert isinstance(converted_dt, datetime)
        assert converted_dt.tzinfo is not None
        assert converted_dt.utcoffset() == timedelta(0)
        
        # 允许由于浮点精度导致的小差异
        diff = abs((converted_dt - test_dt).total_seconds())
        assert diff < 1e-6


class TestNowFunctions:
    """测试与当前时间相关的函数"""
    
    def test_utcnow(self):
        """测试 utcnow 函数"""
        time_ns = get_time_namespace()
        utc_now = time_ns['utcnow']()
        
        assert isinstance(utc_now, datetime)
        assert utc_now.tzinfo is not None
        assert utc_now.utcoffset() == timedelta(0)
    
    def test_now_seconds(self):
        """测试 now_seconds 函数"""
        time_ns = get_time_namespace()
        now_secs = time_ns['now_seconds']()
        
        assert isinstance(now_secs, float)
        
        # 应该是一个合理的当前时间值
        base_2020 = time_ns['datetime_to_seconds'](datetime(2020, 1, 1, tzinfo=timezone.utc))
        base_2100 = time_ns['datetime_to_seconds'](datetime(2100, 1, 1, tzinfo=timezone.utc))
        assert now_secs > base_2020
        assert now_secs < base_2100


class TestAgentTimeZone:
    """测试 AgentTimeZone 类"""
    
    def test_create_with_name(self):
        """测试仅使用名称创建 AgentTimeZone"""
        time_ns = get_time_namespace()
        agent_tz_class = time_ns['AgentTimeZone']
        
        tz = agent_tz_class(name="Asia/Shanghai")
        assert tz.name == "Asia/Shanghai"
        assert tz.offset is None
    
    def test_tz_method_with_name(self):
        """测试使用名称的 tz() 方法"""
        time_ns = get_time_namespace()
        agent_tz_class = time_ns['AgentTimeZone']
        
        tz = agent_tz_class(name="Asia/Shanghai")
        tz_obj = tz.tz()
        assert isinstance(tz_obj, ZoneInfo)
    
    def test_create_with_offset(self):
        """测试使用偏移量创建 AgentTimeZone"""
        time_ns = get_time_namespace()
        agent_tz_class = time_ns['AgentTimeZone']
        
        tz = agent_tz_class(name="UTC+8", offset=8*3600)
        assert tz.name == "UTC+8"
        assert tz.offset == 8*3600
    
    def test_tz_method_with_offset(self):
        """测试使用偏移量的 tz() 方法"""
        time_ns = get_time_namespace()
        agent_tz_class = time_ns['AgentTimeZone']
        
        tz = agent_tz_class(name="UTC+8", offset=8*3600)
        tz_obj = tz.tz()
        assert isinstance(tz_obj, timezone)
        assert tz_obj.utcoffset(None) == timedelta(seconds=8*3600)


class TestFormatFunctions:
    """测试 format_time 和 format_seconds 函数"""
    
    def test_format_time_with_none(self):
        """测试 format_time 处理 None 值"""
        time_ns = get_time_namespace()
        result = time_ns['format_time'](None)
        assert result == "未知时间"
    
    def test_format_time_with_datetime(self):
        """测试 format_time 处理 datetime 对象"""
        time_ns = get_time_namespace()
        dt = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        result = time_ns['format_time'](dt)
        assert "2025-01-01 12:00:00" in result
        assert "Wednesday" in result  # 2025-01-01 实际上是星期三
    
    def test_format_seconds_with_timedelta(self):
        """测试 format_seconds 处理 timedelta 对象"""
        time_ns = get_time_namespace()
        delta = timedelta(days=1, hours=2, minutes=3, seconds=4)
        result = time_ns['format_seconds'](delta)
        assert ("1天" in result) or ("1天" in result)
        assert "2小时" in result
        assert "3分" in result
        assert "4秒" in result
    
    def test_format_seconds_with_float(self):
        """测试 format_seconds 处理浮点秒数"""
        time_ns = get_time_namespace()
        seconds = 3661.0  # 1小时1分钟1秒
        result = time_ns['format_seconds'](seconds)
        assert "1小时" in result
        assert "1分" in result
        assert "1秒" in result


class TestAgentTimeSettings:
    """测试 AgentTimeSettings 类"""
    
    def test_default_settings(self):
        """测试默认的 AgentTimeSettings"""
        time_ns = get_time_namespace()
        agent_settings_class = time_ns['AgentTimeSettings']
        
        settings = agent_settings_class()
        assert settings.agent_time_anchor == 0.0
        assert settings.real_time_anchor == 0.0
        assert settings.time_scale == 1.0
        assert isinstance(settings.time_zone, time_ns['AgentTimeZone'])


class TestTimeConversionWithSettings:
    """测试使用 AgentTimeSettings 的时间转换函数"""
    
    @pytest.fixture
    def sample_settings(self):
        """创建用于测试的示例 AgentTimeSettings"""
        time_ns = get_time_namespace()
        agent_settings_class = time_ns['AgentTimeSettings']
        agent_tz_class = time_ns['AgentTimeZone']
        
        return agent_settings_class(
            agent_time_anchor=time_ns['datetime_to_seconds'](datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
            real_time_anchor=time_ns['datetime_to_seconds'](datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)),
            time_scale=2.0,  # Agent 时间以 2 倍速度运行
            time_zone=agent_tz_class(name="Asia/Tokyo")
        )
    
    def test_real_time_to_agent_time(self, sample_settings):
        """测试 real_time_to_agent_time 函数"""
        time_ns = get_time_namespace()
        
        # 现实时间在锚点之后 1 小时
        real_dt = time_ns['seconds_to_datetime'](sample_settings.real_time_anchor + 3600)
        agent_dt = time_ns['real_time_to_agent_time'](real_dt, sample_settings)
        
        # Agent 时间应该在锚点之后 2 小时（由于 2 倍时间比例）
        expected_agent_seconds = sample_settings.agent_time_anchor + 7200
        actual_agent_seconds = time_ns['datetime_to_seconds'](agent_dt)
        
        # 允许由于浮点精度导致的小差异
        assert abs(actual_agent_seconds - expected_agent_seconds) < 1e-6
    
    def test_agent_time_to_real_time(self, sample_settings):
        """测试 agent_time_to_real_time 函数"""
        time_ns = get_time_namespace()
        
        # Agent 时间在锚点之后 2 小时
        agent_dt = time_ns['seconds_to_datetime'](sample_settings.agent_time_anchor + 7200)
        real_dt = time_ns['agent_time_to_real_time'](agent_dt, sample_settings)
        
        # 现实时间应该在锚点之后 1 小时（由于 2 倍时间比例）
        expected_real_seconds = sample_settings.real_time_anchor + 3600
        actual_real_seconds = time_ns['datetime_to_seconds'](real_dt)
        
        # 允许由于浮点精度导致的小差异
        assert abs(actual_real_seconds - expected_real_seconds) < 1e-6