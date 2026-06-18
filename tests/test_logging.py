"""
日志系统测试套件 | Tests for the logging system.
===================================================

验证所有核心功能：
- LogRecord 创建和序列化 | creation and serialization
- MetricTracker 聚合统计正确性 | aggregation correctness
- LogContext 作用域栈 | scope stacking
- Logger 路由到后端 | routing to backends
- FileBackend JSONL 输出 | JSONL output

运行时: PYTHONPATH="E:/.../AdaTile-FastSAM" pytest tests/test_logging.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from adatile.logging import (
    LogContext,
    LogLevel,
    LogRecord,
    MetricTracker,
    get_logger,
)
from adatile.logging.backends import ConsoleBackend, FileBackend


# ════════════════════════════════════════════════════════════════
# LogRecord 测试 | LogRecord Tests
# 验证日志记录容器的创建和序列化
# Verify creation and serialization of log records
# ════════════════════════════════════════════════════════════════

class TestLogRecord:
    """验证 LogRecord 构造和序列化 | Verify LogRecord construction and serialization."""

    def test_metric_factory(self) -> None:
        """测试 METRIC 工厂方法 | Test metric factory method."""
        r = LogRecord.metric("loss/total", 0.5, step=100, phase="train", scope="loss")
        assert r.key == "loss/total"
        assert r.value == 0.5
        assert r.step == 100
        assert r.phase == "train"
        assert r.scope == "loss"
        assert r.level == LogLevel.METRIC

    def test_info_factory(self) -> None:
        """测试 INFO 工厂方法 | Test info factory method."""
        r = LogRecord.info("checkpoint", "Saved at step 1000", step=1000)
        assert r.level == LogLevel.INFO
        assert isinstance(r.value, str)

    def test_warn_factory(self) -> None:
        """测试 WARN 工厂方法 | Test warn factory method."""
        r = LogRecord.warn("oom_risk", "Memory usage > 80%")
        assert r.level == LogLevel.WARN

    def test_to_dict(self) -> None:
        """测试序列化为字典 | Test serialization to dict."""
        r = LogRecord.metric("iou", 0.85, step=50, tags=["few-shot", "5-shot"])
        d = r.to_dict()
        assert d["key"] == "iou"
        assert d["value"] == 0.85
        assert d["step"] == 50
        assert "few-shot" in d["tags"]
        assert "timestamp" in d  # 自动生成的时间戳 | auto-generated timestamp

    def test_json_serializable(self) -> None:
        """测试 JSON 序列化不抛异常 | Test JSON serialization doesn't raise."""
        r = LogRecord.metric("loss", 0.3, metadata={"lr": 1e-3})
        json.dumps(r.to_dict())  # should not raise


# ════════════════════════════════════════════════════════════════
# MetricTracker 测试 | MetricTracker Tests
# 验证运行统计量的正确性
# Verify correctness of running statistics
# ════════════════════════════════════════════════════════════════

class TestMetricTracker:
    """验证运行统计计算 | Verify running statistics computation."""

    def test_single_update(self) -> None:
        """单次更新：均值=该值，std=0 | Single update: mean=value, std=0."""
        t = MetricTracker()
        t.update("loss", 0.5)
        stats = t.get("loss")
        assert stats["mean"] == 0.5
        assert stats["latest"] == 0.5
        assert stats["count"] == 1
        assert stats["std"] == 0.0

    def test_mean_convergence(self) -> None:
        """测试累计均值计算 | Test cumulative mean calculation."""
        t = MetricTracker(window_size=100)
        for i in range(10):
            t.update("x", float(i))
        stats = t.get("x")
        # mean of [0..9] = 4.5
        assert abs(stats["mean"] - 4.5) < 0.01
        assert stats["count"] == 10

    def test_min_max(self) -> None:
        """测试最小/最大值跟踪 | Test min/max tracking."""
        t = MetricTracker()
        for v in [3.0, 1.0, 5.0, 2.0, 4.0]:
            t.update("y", v)
        stats = t.get("y")
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0

    def test_ema(self) -> None:
        """测试 EMA 计算 | Test EMA computation.
        EMA = decay * old_ema + (1 - decay) * new_value
        decay=0.9: ema = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        """
        t = MetricTracker(ema_decay=0.9)
        t.update("z", 1.0)
        t.update("z", 2.0)
        assert abs(t.get("z")["ema"] - 1.1) < 0.01

    def test_window_sliding(self) -> None:
        """测试滑动窗口淘汰旧值 | Test window sliding evicts old values."""
        t = MetricTracker(window_size=3)
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            t.update("w", v)
        # 窗口应该是 [3, 4, 5] | Window should be [3, 4, 5]
        stats = t.get("w")
        assert stats["latest"] == 5.0

    def test_multiple_keys(self) -> None:
        """测试多 key 独立跟踪 | Test multiple keys tracked independently."""
        t = MetricTracker()
        t.update("a", 1.0)
        t.update("b", 2.0)
        assert set(t.keys) == {"a", "b"}
        assert t.get("a")["latest"] == 1.0
        assert t.get("b")["latest"] == 2.0

    def test_snapshot(self) -> None:
        """测试全量快照 | Test full snapshot."""
        t = MetricTracker()
        t.update("loss", 0.5)
        t.update("iou", 0.8)
        snap = t.snapshot()
        assert "loss" in snap
        assert "iou" in snap
        assert snap["loss"]["latest"] == 0.5

    def test_reset_key(self) -> None:
        """测试重置单个 key | Test reset single key."""
        t = MetricTracker()
        t.update("x", 1.0)
        t.reset("x")
        assert t.get("x")["count"] == 0

    def test_reset_all(self) -> None:
        """测试重置所有 key | Test reset all keys."""
        t = MetricTracker()
        t.update("x", 1.0)
        t.reset()
        assert t.keys == []


# ════════════════════════════════════════════════════════════════
# LogContext 测试 | LogContext Tests
# 验证上下文管理器的嵌套和恢复行为
# Verify context manager nesting and restore behavior
# ════════════════════════════════════════════════════════════════

class TestLogContext:
    """验证上下文管理器栈 | Verify context manager stacking."""

    def test_default_context(self) -> None:
        """默认上下文：phase=train, scope="" | Default context."""
        from adatile.logging.context import get_context as gc
        ctx = gc()
        assert ctx["phase"] == "train"
        assert ctx["scope"] == ""

    def test_phase_override(self) -> None:
        """测试 phase 覆盖和恢复 | Test phase override and restore."""
        with LogContext(phase="val"):
            from adatile.logging.context import get_context as gc
            assert gc()["phase"] == "val"
        # 退出后恢复 | Restored on exit
        from adatile.logging.context import get_context as gc
        assert gc()["phase"] == "train"

    def test_scope_override(self) -> None:
        """测试 scope 覆盖 | Test scope override."""
        with LogContext(scope="decoder"):
            from adatile.logging.context import get_context as gc
            assert gc()["scope"] == "decoder"

    def test_step_override(self) -> None:
        """测试 step 覆盖 | Test step override."""
        with LogContext(step=42):
            from adatile.logging.context import get_context as gc
            assert gc()["step"] == 42

    def test_tag_union(self) -> None:
        """测试嵌套 tags 取并集 | Test nested tags are unioned."""
        with LogContext(tags=["a", "b"]):
            with LogContext(tags=["b", "c"]):
                from adatile.logging.context import get_context as gc
                tags = set(gc()["tags"])
                assert tags == {"a", "b", "c"}

    def test_nested_phase_restore(self) -> None:
        """测试多层嵌套正确恢复 | Test multi-level nesting proper restore."""
        with LogContext(phase="val"):
            with LogContext(phase="test"):
                from adatile.logging.context import get_context as gc
                assert gc()["phase"] == "test"
            from adatile.logging.context import get_context as gc
            assert gc()["phase"] == "val"
        from adatile.logging.context import get_context as gc
        assert gc()["phase"] == "train"


# ════════════════════════════════════════════════════════════════
# Logger 测试 | Logger Tests
# 验证 Logger 路由和便捷方法
# Verify Logger routing and convenience methods
# ════════════════════════════════════════════════════════════════

class TestLogger:
    """验证 Logger 路由和便捷方法 | Verify Logger routing and convenience methods."""

    def test_log_metric_routes_to_backend(self) -> None:
        """测试 log_metric 正确路由到后端 | Test log_metric routes to backend."""
        logger = get_logger("test_metric")

        records: list[LogRecord] = []

        # 捕获后端：记录所有 write 调用 | Capture backend: record all writes
        class _CaptureBackend(ConsoleBackend):
            def write(self, r: LogRecord) -> None:
                records.append(r)

        backend = _CaptureBackend()
        logger.add_backend(backend)

        logger.log_metric("loss/total", 0.5, step=100)
        assert len(records) == 1
        assert records[0].key == "loss/total"
        assert records[0].value == 0.5

        logger.remove_backend(backend)

    def test_context_auto_merge(self) -> None:
        """测试上下文自动合并到记录 | Test context auto-merged into records."""
        logger = get_logger("test_context")

        records: list[LogRecord] = []

        class _CaptureBackend(ConsoleBackend):
            def write(self, r: LogRecord) -> None:
                records.append(r)

        backend = _CaptureBackend()
        logger.add_backend(backend)

        # 设置上下文后 log，记录应自动携带 phase 和 scope
        # After setting context, log should auto-carry phase and scope
        with logger.phase("val"):
            with logger.scope("decoder"):
                logger.log_metric("iou", 0.85, step=200)

        assert len(records) == 1
        assert records[0].phase == "val"
        assert records[0].scope == "decoder"

        logger.remove_backend(backend)

    def test_log_loss_shorthand(self) -> None:
        """测试 log_loss 快捷方法自动设置 scope='loss' | Test log_loss shorthand sets scope='loss'."""
        logger = get_logger("test_loss")

        records: list[LogRecord] = []

        class _CaptureBackend(ConsoleBackend):
            def write(self, r: LogRecord) -> None:
                records.append(r)

        backend = _CaptureBackend()
        logger.add_backend(backend)

        logger.log_loss("seg", 0.3)
        assert records[0].scope == "loss"
        assert records[0].key == "seg"

        logger.remove_backend(backend)

    def test_tracker_updated_on_log_metric(self) -> None:
        """测试 log_metric 同时更新 tracker | Test tracker updated on log_metric."""
        logger = get_logger("test_tracker")
        logger.log_metric("acc", 0.95)
        assert logger.get_stats("acc")["latest"] == 0.95

    def test_log_info_warn(self) -> None:
        """测试 log_info 和 log_warn | Test log_info and log_warn."""
        logger = get_logger("test_msgs")

        records: list[LogRecord] = []

        class _CaptureBackend(ConsoleBackend):
            def write(self, r: LogRecord) -> None:
                records.append(r)

        backend = _CaptureBackend()
        logger.add_backend(backend)

        logger.log_info("start", "Training started")
        logger.log_warn("mem", "High memory usage")

        assert records[0].level == LogLevel.INFO
        assert records[1].level == LogLevel.WARN

        logger.remove_backend(backend)


# ════════════════════════════════════════════════════════════════
# FileBackend 测试 | FileBackend Tests
# 验证 JSONL 文件输出的正确性
# Verify JSONL file output correctness
# ════════════════════════════════════════════════════════════════

class TestFileBackend:
    """验证 JSONL 文件输出 | Verify JSONL file output."""

    def test_writes_jsonl(self) -> None:
        """测试写入 JSONL 并验证行数 | Test write JSONL and verify line count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            backend = FileBackend(path)
            backend.write(LogRecord.metric("loss", 0.5, step=1))
            backend.write(LogRecord.metric("iou", 0.8, step=1))
            backend.close()

            lines = path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}"
            for line in lines:
                d = json.loads(line)
                assert "key" in d
                assert "value" in d
                assert "timestamp" in d

    def test_creates_parent_dir(self) -> None:
        """测试自动创建父目录 | Test auto-creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "deep" / "log.jsonl"
            backend = FileBackend(path)
            backend.write(LogRecord.info("test", "hello"))
            backend.close()
            assert path.exists()

    def test_min_level_filter(self) -> None:
        """测试 min_level 过滤 | Test min_level filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "filtered.jsonl"
            # 只接受 WARN 及以上级别 | Only accept WARN and above
            backend = FileBackend(path, min_level="WARN")
            backend.write(LogRecord.info("debug", "should not appear"))
            backend.write(LogRecord.warn("warning", "should appear"))
            backend.close()

            lines = path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1, f"Expected 1 line after filter, got {len(lines)}"
            assert "warning" in lines[0]


# ════════════════════════════════════════════════════════════════
# Registry 测试 | Registry Tests
# 验证全局注册表单例行为
# Verify global registry singleton behavior
# ════════════════════════════════════════════════════════════════

class TestRegistry:
    """验证全局注册表行为 | Verify global registry behavior."""

    def test_get_logger_singleton(self) -> None:
        """同一名称返回同一实例 | Same name returns same instance."""
        a = get_logger("foo")
        b = get_logger("foo")
        assert a is b

    def test_different_names(self) -> None:
        """不同名称返回不同实例 | Different names return different instances."""
        a = get_logger("train")
        b = get_logger("data")
        assert a is not b
