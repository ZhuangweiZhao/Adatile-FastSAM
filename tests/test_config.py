"""
Config 模块测试 | Config module tests.
=======================================

验证实验 ID 生成、超参数配置、结果记录器。
Verify experiment ID generation, hyperparameter config, results recorder.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend


# ════════════════════════════════════════════════════════════════
# Experiment ID 测试 | Experiment ID Tests
# ════════════════════════════════════════════════════════════════

class TestGenerateExpId:
    """验证实验 ID 生成 | Verify experiment ID generation."""

    def test_default_prefix(self) -> None:
        """默认前缀 "exp" | Default prefix "exp"."""
        eid = generate_exp_id()
        assert eid.startswith("exp_")
        # 格式: exp_YYYYMMDD_HHMMSS_xxxx | Format: exp_YYYYMMDD_HHMMSS_xxxx
        parts = eid.split("_")
        assert len(parts) == 4  # exp, date, time, microsecond
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 6  # HHMMSS
        assert len(parts[3]) == 4  # microsecond

    def test_custom_name(self) -> None:
        """自定义名称 | Custom name."""
        eid = generate_exp_id(name="baseline_test")
        assert eid.startswith("exp_baseline_test_")
        # exp_baseline_test_YYYYMMDD_HHMMSS
        parts = eid.split("_")
        assert len(parts) >= 3

    def test_unique_ids(self) -> None:
        """连续调用生成不重复 ID | Sequential calls produce unique IDs."""
        ids = {generate_exp_id() for _ in range(100)}
        assert len(ids) == 100, "All 100 IDs should be unique"

    def test_prefix_override(self) -> None:
        """自定义前缀 | Custom prefix."""
        eid = generate_exp_id(prefix="abl")
        assert eid.startswith("abl_")


# ════════════════════════════════════════════════════════════════
# ExperimentConfig 测试 | ExperimentConfig Tests
# ════════════════════════════════════════════════════════════════

class TestExperimentConfig:
    """验证超参数配置 dataclass | Verify hyperparameter config dataclass."""

    def test_default_values(self) -> None:
        """默认值正确 | Default values correct."""
        cfg = ExperimentConfig(exp_id="test_001")
        assert cfg.exp_id == "test_001"
        assert cfg.backbone_name == "FastSAM-x"
        assert cfg.image_size == (1024, 1024)
        assert cfg.batch_size == 1
        assert cfg.learning_rate == 1e-4
        assert cfg.max_epochs == 50
        assert cfg.seed == 42

    def test_to_dict(self) -> None:
        """序列化为字典 | Serialization to dict."""
        cfg = ExperimentConfig(exp_id="test_001", seed=123)
        d = cfg.to_dict()
        assert d["exp_id"] == "test_001"
        assert d["seed"] == 123
        assert d["backbone_name"] == "FastSAM-x"

    def test_from_dict(self) -> None:
        """从字典反序列化 | Deserialization from dict."""
        data = {"exp_id": "test_002", "seed": 99, "batch_size": 4}
        cfg = ExperimentConfig.from_dict(data)
        assert cfg.exp_id == "test_002"
        assert cfg.seed == 99
        assert cfg.batch_size == 4

    def test_to_dict_round_trip(self) -> None:
        """字典往返：to_dict → from_dict 一致性 | Dict round-trip consistency."""
        cfg1 = ExperimentConfig(exp_id="test_003", learning_rate=5e-5)
        cfg2 = ExperimentConfig.from_dict(cfg1.to_dict())
        assert cfg1.exp_id == cfg2.exp_id
        assert cfg1.learning_rate == cfg2.learning_rate
        assert cfg1.image_size == cfg2.image_size

    def test_invalid_exp_id(self) -> None:
        """exp_id 不能为空 | exp_id must not be empty."""
        with pytest.raises(ValueError, match="exp_id"):
            ExperimentConfig(exp_id="")

    def test_invalid_batch_size(self) -> None:
        """batch_size 必须为正整数 | batch_size must be positive."""
        with pytest.raises(ValueError, match="batch_size"):
            ExperimentConfig(exp_id="test", batch_size=0)

    def test_yaml_round_trip(self) -> None:
        """YAML 文件往返 | YAML file round-trip."""
        cfg1 = ExperimentConfig(
            exp_id="test_yaml",
            backbone_name="FastSAM-s",
            image_size=(512, 512),
            learning_rate=1e-3,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "config.yaml"
            cfg1.to_yaml(str(yaml_path))
            assert yaml_path.exists()

            cfg2 = ExperimentConfig.from_yaml(str(yaml_path))
            assert cfg1.exp_id == cfg2.exp_id
            assert cfg1.backbone_name == cfg2.backbone_name
            assert cfg1.image_size == cfg2.image_size
            assert cfg1.learning_rate == cfg2.learning_rate


# ════════════════════════════════════════════════════════════════
# ExperimentRecorder 测试 | ExperimentRecorder Tests
# ════════════════════════════════════════════════════════════════

class TestExperimentRecorder:
    """验证结果记录器 | Verify results recorder."""

    @pytest.fixture
    def tmp_config(self) -> ExperimentConfig:
        """创建临时配置 | Create temporary config."""
        return ExperimentConfig(
            exp_id=f"test_recorder_{generate_exp_id().split('_')[-1]}",
            output_dir=None,  # 会用 temp dir 覆盖 | Will override with temp dir
        )

    def test_creates_output_dir(self, tmp_config: ExperimentConfig) -> None:
        """自动创建输出目录 | Auto-creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(
                exp_id=tmp_config.exp_id,
                output_dir=tmpdir,
            )
            recorder = ExperimentRecorder(cfg)
            exp_dir = Path(tmpdir) / cfg.exp_id
            assert exp_dir.exists()
            assert exp_dir.is_dir()
            recorder.close()

    def test_record_metric_via_logger(self, tmp_config: ExperimentConfig) -> None:
        """记录指标通过日志系统路由 | Metrics routed through logging system."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(exp_id=tmp_config.exp_id, output_dir=tmpdir)
            recorder = ExperimentRecorder(cfg)

            # 捕获后端验证日志路由 | Capture backend to verify log routing
            records: list = []

            class _CaptureBackend(ConsoleBackend):
                def write(self, r):
                    records.append(r)

            backend = _CaptureBackend()
            recorder.logger.add_backend(backend)

            recorder.record_metric("miou/val", 0.85, step=100)
            assert len(records) == 1
            assert records[0].key == "miou/val"
            assert records[0].value == 0.85

            recorder.close()

    def test_config_logged_on_init(self, tmp_config: ExperimentConfig) -> None:
        """初始化时自动记录配置 | Config auto-logged on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(exp_id=tmp_config.exp_id, output_dir=tmpdir)
            records: list = []

            class _CaptureBackend(ConsoleBackend):
                def write(self, r):
                    records.append(r)

            recorder = ExperimentRecorder(cfg)
            recorder.logger.add_backend(_CaptureBackend())

            recorder.record_config()
            # 应该有配置相关的记录（backbone_name, image_size, etc.）
            config_keys = [r.key for r in records]
            assert any("config" in k for k in config_keys) or len(records) > 0

            recorder.close()

    def test_finalize_writes_summary(self, tmp_config: ExperimentConfig) -> None:
        """finalize() 写入 summary.json | finalize() writes summary.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(exp_id=tmp_config.exp_id, output_dir=tmpdir)
            recorder = ExperimentRecorder(cfg)

            recorder.record_metric("miou/val", 0.85, step=100)
            recorder.record_metric("dice/val", 0.78, step=100)
            recorder.finalize()

            summary_path = Path(tmpdir) / cfg.exp_id / "summary.json"
            assert summary_path.exists()

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            assert "exp_id" in summary
            assert summary["exp_id"] == tmp_config.exp_id

            recorder.close()

    def test_logger_has_backends(self, tmp_config: ExperimentConfig) -> None:
        """Recorder 的 logger 默认有 Console + File 后端 | Default backends include Console + File."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ExperimentConfig(exp_id=tmp_config.exp_id, output_dir=tmpdir)
            recorder = ExperimentRecorder(cfg)
            # 至少有一个后端 | At least one backend
            assert len(recorder.logger.backends) >= 1
            recorder.close()
