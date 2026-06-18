"""
Metrics 模块测试 | Metrics module tests.
===========================================

验证 mIoU、Dice、FPS、Params 四个评测指标。
Key tests for v1 lessons:
    - Dice batch>1 不广播爆炸 | no broadcast explosion
    - {0, 255} 二值掩码正确检测 | correct binary detection
"""

from __future__ import annotations

import pytest
import torch

from adatile.metrics import (
    FPSMeter,
    compute_dice,
    compute_miou,
    count_params,
    format_param_count,
)


# ════════════════════════════════════════════════════════════════
# mIoU 测试 | mIoU Tests
# ════════════════════════════════════════════════════════════════

class TestMIoU:
    """验证 mIoU 计算 | Verify mIoU computation."""

    def test_perfect_prediction(self) -> None:
        """完全相同 → mIoU = 1.0 | Perfect match → mIoU = 1.0."""
        pred = torch.tensor([[[0, 1], [1, 0]]])  # [1, H, W]
        target = torch.tensor([[[0, 1], [1, 0]]])
        result = compute_miou(pred, target, num_classes=2)
        assert result["miou"] == pytest.approx(1.0, abs=1e-5)

    def test_zero_overlap(self) -> None:
        """完全不相交 → mIoU ≈ 0.0 | Completely disjoint → mIoU ≈ 0.0."""
        pred = torch.tensor([[[0, 0], [0, 0]]])
        target = torch.tensor([[[1, 1], [1, 1]]])
        result = compute_miou(pred, target, num_classes=2)
        # class 0: pred=4 target=0 → IoU=0
        # class 1: pred=0 target=4 → IoU=0
        assert result["miou"] == pytest.approx(0.0, abs=1e-5)

    def test_partial_overlap(self) -> None:
        """已知重叠 → IoU 可计算 | Known overlap → computable IoU."""
        # 2/4 pixels overlap for class 1
        pred = torch.tensor([[[0, 1], [0, 1]]])
        target = torch.tensor([[[0, 1], [1, 1]]])
        # class 0: pred=2 target=1 union=3 → IoU=1/3≈0.333
        # class 1: pred=2 target=3 union=3 → IoU=2/3≈0.667
        result = compute_miou(pred, target, num_classes=2)
        assert 0.4 < result["miou"] < 0.6  # (0.333+0.667)/2 = 0.5

    def test_multi_class(self) -> None:
        """多类别（>2 类）| Multi-class (>2 classes)."""
        pred = torch.tensor([[[0, 1], [2, 0]]])
        target = torch.tensor([[[0, 1], [2, 0]]])
        result = compute_miou(pred, target, num_classes=3)
        assert result["miou"] == pytest.approx(1.0, abs=1e-5)

    def test_batch(self) -> None:
        """批处理 >1 | Batch > 1."""
        pred = torch.randint(0, 3, (4, 64, 64))
        target = pred.clone()
        result = compute_miou(pred, target, num_classes=3)
        assert result["miou"] == pytest.approx(1.0, abs=1e-5)

    def test_ignore_index(self) -> None:
        """ignore_index 像素被排除 | ignore_index pixels excluded."""
        pred = torch.tensor([[[0, 1, 1], [0, 1, 1]]])
        target = torch.tensor([[[0, 1, 255], [0, 1, 255]]])
        result = compute_miou(pred, target, num_classes=2, ignore_index=255)
        # 只计算前2列（255被忽略） | Only first 2 cols counted (255 ignored)
        assert result["miou"] == pytest.approx(1.0, abs=1e-5)

    def test_one_hot_input(self) -> None:
        """one-hot 格式输入 [B,C,H,W] | one-hot format input [B,C,H,W]."""
        pred_oh = torch.zeros(1, 2, 4, 4)
        pred_oh[:, 0, :, :] = 1  # all class 0
        target_oh = torch.zeros(1, 2, 4, 4)
        target_oh[:, 0, :, :] = 1
        result = compute_miou(pred_oh, target_oh, num_classes=2)
        assert result["miou"] == pytest.approx(1.0, abs=1e-5)


# ════════════════════════════════════════════════════════════════
# Dice 测试 | Dice Tests
# ════════════════════════════════════════════════════════════════

class TestDice:
    """验证 Dice 系数计算 | Verify Dice coefficient computation."""

    def test_perfect(self) -> None:
        """完全相同 → Dice = 1.0 | Perfect match → Dice = 1.0."""
        pred = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
        target = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.float32)
        d = compute_dice(pred, target)
        assert d.item() == pytest.approx(1.0, abs=1e-5)

    def test_zero(self) -> None:
        """完全不相交 → Dice = 0.0 | Completely disjoint → Dice = 0.0."""
        pred = torch.tensor([[[1, 1], [1, 1]]], dtype=torch.float32)
        target = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.float32)
        d = compute_dice(pred, target)
        assert d.item() == pytest.approx(0.0, abs=1e-5)

    def test_batch_no_broadcast_bug(self) -> None:
        """
        V1 教训：batch>1 无广播爆炸 | V1 lesson: no broadcast explosion with batch>1.
        验证 unsqueeze(0) 问题已被修复。| Verify unsqueeze(0) issue is fixed.
        """
        pred = torch.ones(4, 1, 64, 64)
        target = torch.ones(4, 1, 64, 64)
        d = compute_dice(pred, target)
        # 所有 batch 的 Dice 都应该 ≤ 1.0（不会出现 >100 的爆炸值）
        # All batch Dice should be ≤ 1.0 (no >100 explosion)
        assert d.item() <= 1.0 + 1e-5, f"Dice exploded: {d.item()}"  # 不能爆炸 | must not explode
        assert d.item() >= 0.0, f"Dice negative: {d.item()}"

    def test_255_mask_handling(self) -> None:
        """
        V1 教训：{0, 255} 掩码正确处理 | V1 lesson: {0, 255} masks handled correctly.
        唯一值 ≤ 2 → 二值 | n_unique ≤ 2 → binary.
        """
        # 模拟 {0, 255} 掩码 | Simulate {0, 255} masks
        pred = torch.tensor([[[0, 255], [255, 0]]], dtype=torch.float32) / 255.0
        target = torch.tensor([[[0, 255], [255, 0]]], dtype=torch.float32) / 255.0
        d = compute_dice(pred, target)
        assert d.item() == pytest.approx(1.0, abs=1e-5)

    def test_binary_detection(self) -> None:
        """n_unique ≤ 2 检测正常工作 | n_unique ≤ 2 detection works properly."""
        # 二值掩码应正确计算 | Binary mask should compute correctly
        pred = (torch.rand(1, 1, 32, 32) > 0.5).float()
        target = pred.clone()
        d = compute_dice(pred, target)
        assert d.item() == pytest.approx(1.0, abs=1e-5)

    def test_smooth_prevents_nan(self) -> None:
        """smooth 参数防止除零 NaN | smooth param prevents div-zero NaN."""
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        d = compute_dice(pred, target, smooth=1e-6)
        # 两个都是零 → 没有交集也没有并集 → Dice=1.0 (smooth 使得)
        # Both zero → no intersection, no union → Dice=1.0 (due to smooth)
        assert not torch.isnan(d), "Dice is NaN"
        assert not torch.isinf(d), "Dice is Inf"

    def test_per_class_dice(self) -> None:
        """每个类别独立 Dice | Per-class Dice."""
        # [B, C, H, W] one-hot format, 3 classes
        pred = torch.zeros(2, 3, 16, 16)
        pred[:, 0] = 1  # all class 0
        target = torch.zeros(2, 3, 16, 16)
        target[:, 0] = 1
        d = compute_dice(pred, target, per_class=True)
        assert d.shape[0] == 3  # 3 classes
        assert d[0].item() == pytest.approx(1.0, abs=1e-5)  # class 0 perfect
        assert d[1].item() == pytest.approx(1.0, abs=1e-5)  # class 1 (all zero → 1.0 via smooth)


# ════════════════════════════════════════════════════════════════
# FPS 测试 | FPS Tests
# ════════════════════════════════════════════════════════════════

class TestFPS:
    """验证 FPS 测量 | Verify FPS measurement."""

    def test_returns_positive(self) -> None:
        """compute() 返回正数 | compute() returns positive."""
        meter = FPSMeter(warmup=2, num_runs=5)
        for _ in range(10):
            with meter:
                _ = torch.randn(1, 3, 512, 512) * 2
        fps = meter.compute()
        assert fps > 0, f"FPS should be positive, got {fps}"

    def test_context_manager(self) -> None:
        """支持 with 语句 | Supports context manager."""
        meter = FPSMeter(warmup=0, num_runs=3)
        with meter:
            _ = torch.randn(100, 100) @ torch.randn(100, 100)
        # 至少有一次计时 | At least one timing recorded
        assert meter.compute() > 0

    def test_warmup_excluded(self) -> None:
        """预热运行不计入统计 | Warmup runs excluded from stats."""
        meter = FPSMeter(warmup=3, num_runs=5)
        assert len(meter._times) == 0  # 初始为空 | Initially empty
        for _ in range(8):
            with meter:
                _ = torch.randn(1) + 1
        # 8 - 3 warmup = 5 次有效计时 | 5 valid timings
        assert len(meter._times) == 5

    def test_reset(self) -> None:
        """reset() 清空计时 | reset() clears timings."""
        meter = FPSMeter(warmup=0, num_runs=100)
        with meter:
            _ = torch.randn(1) + 1
        meter.reset()
        assert len(meter._times) == 0


# ════════════════════════════════════════════════════════════════
# Params 测试 | Params Tests
# ════════════════════════════════════════════════════════════════

class TestParams:
    """验证参数计数 | Verify parameter counting."""

    @pytest.fixture
    def simple_model(self) -> torch.nn.Module:
        """简单测试模型 | Simple test model."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 1, 1),
        )

    def test_total_positive(self, simple_model: torch.nn.Module) -> None:
        """总参数量为正 | Total params positive."""
        info = count_params(simple_model)
        assert info["total"] > 0

    def test_trainable_total_relation(self, simple_model: torch.nn.Module) -> None:
        """可训练 ≤ 总数 | Trainable ≤ total."""
        info = count_params(simple_model)
        assert info["trainable"] <= info["total"]

    def test_frozen_total_relation(self, simple_model: torch.nn.Module) -> None:
        """冻结 + 可训练 = 总数 | Frozen + trainable = total."""
        info = count_params(simple_model)
        assert info["frozen"] + info["trainable"] == info["total"]

    def test_per_module_breakdown(self, simple_model: torch.nn.Module) -> None:
        """每个子模块参数和 = 总数 | Per-module sum = total."""
        info = count_params(simple_model)
        per_module_sum = sum(info["per_module"].values())
        assert per_module_sum == info["total"]

    def test_format_param_count(self) -> None:
        """格式化输出正确 | Format output correct."""
        assert format_param_count(1_200_000) == "1.20M"
        assert format_param_count(45_300) == "45.30K"
        assert format_param_count(999) == "999"
        assert format_param_count(0) == "0"

    def test_all_frozen(self, simple_model: torch.nn.Module) -> None:
        """全部冻结后 trainable=0 | trainable=0 when all frozen."""
        for p in simple_model.parameters():
            p.requires_grad = False
        info = count_params(simple_model)
        assert info["trainable"] == 0
        assert info["frozen"] == info["total"]
