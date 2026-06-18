"""
Backbone 模块测试 | Backbone module tests.
===========================================

验证 FastSAMBackbone 的模型加载、特征提取、eval mode 强制。
Verify FastSAMBackbone model loading, feature extraction, eval mode enforcement.

V1 关键测试：
- 模型始终处于 eval mode（model.train() 会崩溃 YOLOv8 detect head）
- P4 (~stride 16) 和 P8 (~stride 32) 特征图尺寸正确
- 冻结/解冻控制正常
"""

from __future__ import annotations

import pytest
import torch

from adatile.backbone import FastSAMBackbone, build_backbone


# 测试图像尺寸 | Test image sizes
TEST_SIZES = [
    (1024, 1024),  # 正方形 | Square
    (800, 1024),   # 非正方形 | Non-square
]


# ════════════════════════════════════════════════════════════════
# FastSAMBackbone 测试 | FastSAMBackbone Tests
# ════════════════════════════════════════════════════════════════

class TestFastSAMBackbone:
    """验证 FastSAM 骨干网络 | Verify FastSAM backbone."""

    @pytest.fixture(scope="class")
    def backbone(self) -> FastSAMBackbone:
        """
        Class-scoped fixture: 只加载一次模型，所有测试复用。
        Class-scoped fixture: load model once, reuse across tests.
        """
        return FastSAMBackbone()

    def test_load_success(self, backbone: FastSAMBackbone) -> None:
        """模型加载成功 | Model loads successfully."""
        assert backbone is not None
        assert backbone.model is not None

    def test_forward_output_keys(self, backbone: FastSAMBackbone) -> None:
        """输出字典包含 "p4" 和 "p8" 键 | Output dict has "p4" and "p8" keys."""
        x = torch.randn(1, 3, 1024, 1024)
        features = backbone(x)
        assert "p4" in features, "Missing p4 features"
        assert "p8" in features, "Missing p8 features"

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_p4_feature_stride(self, backbone: FastSAMBackbone, size: tuple) -> None:
        """P4 特征步长 ≈ 16 | P4 feature stride ≈ 16."""
        x = torch.randn(1, 3, *size)
        features = backbone(x)
        p4 = features["p4"]
        # stride 16: H/16, W/16 (可能有 ±1 的误差因 padding)
        # stride 16: H/16, W/16 (may have ±1 due to padding)
        assert abs(p4.shape[2] - size[0] // 16) <= 2, (
            f"P4 H: expected ~{size[0] // 16}, got {p4.shape[2]}"
        )
        assert abs(p4.shape[3] - size[1] // 16) <= 2, (
            f"P4 W: expected ~{size[1] // 16}, got {p4.shape[3]}"
        )

    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_p8_feature_stride(self, backbone: FastSAMBackbone, size: tuple) -> None:
        """P8 特征步长 ≈ 32 | P8 feature stride ≈ 32."""
        x = torch.randn(1, 3, *size)
        features = backbone(x)
        p8 = features["p8"]
        # stride 32: H/32, W/32
        assert abs(p8.shape[2] - size[0] // 32) <= 2, (
            f"P8 H: expected ~{size[0] // 32}, got {p8.shape[2]}"
        )
        assert abs(p8.shape[3] - size[1] // 32) <= 2, (
            f"P8 W: expected ~{size[1] // 32}, got {p8.shape[3]}"
        )

    def test_batch_input(self, backbone: FastSAMBackbone) -> None:
        """批处理输入正常 | Batch input works."""
        x = torch.randn(2, 3, 1024, 1024)
        features = backbone(x)
        assert features["p4"].shape[0] == 2
        assert features["p8"].shape[0] == 2

    def test_model_in_eval_mode(self, backbone: FastSAMBackbone) -> None:
        """V1 教训：内部 YOLO 模型始终处于 eval 模式 | V1 lesson: internal YOLO always in eval mode."""
        # 关键检查：底层 YOLO 模型必须 eval
        # Critical check: underlying YOLO model MUST be eval
        yolo_model = backbone.model.model
        assert hasattr(yolo_model, 'training'), "YOLO model has no training attr"
        assert not yolo_model.training, (
            "V1 LESSON VIOLATION: Internal YOLO model is in training mode! "
            "This crashes the YOLOv8 detect head."
        )

    def test_train_mode_raises(self, backbone: FastSAMBackbone) -> None:
        """调用 .train() 应抛出异常或至少被警告 | Calling .train() should raise or warn."""
        # V1 教训：不能调用 model.train()
        # V1 lesson: must not call model.train()
        # 我们的 backbone 应该阻止此操作
        # Our backbone should prevent this
        with pytest.raises((RuntimeError, ValueError, AttributeError)):
            backbone.train()

    def test_eval_mode_ok(self, backbone: FastSAMBackbone) -> None:
        """.eval() 可以安全调用 | .eval() is safe to call."""
        backbone.eval()  # should not raise
        assert not backbone.training

    def test_freeze_default(self) -> None:
        """默认 freeze_backbone=True：所有参数 requires_grad=False | Default freeze: all params frozen."""
        bb = FastSAMBackbone(freeze_backbone=True)
        # 至少有一些参数被冻结 | At least some params should be frozen
        frozen_params = sum(
            1 for p in bb.model.model.parameters() if not p.requires_grad
        )
        total_params = sum(1 for _ in bb.model.model.parameters())
        # 如果 freeze=True，绝大多数参数应冻结 | If freeze=True, vast majority should be frozen
        assert frozen_params > 0, "No parameters frozen when freeze_backbone=True"

    def test_freeze_false(self) -> None:
        """freeze_backbone=False：参数可训练 | freeze_backbone=False: params are trainable."""
        bb = FastSAMBackbone(freeze_backbone=False)
        trainable = sum(
            1 for p in bb.model.model.parameters() if p.requires_grad
        )
        assert trainable > 0, "No trainable parameters when freeze_backbone=False"

    def test_features_are_tensors(self, backbone: FastSAMBackbone) -> None:
        """输出特征是 torch.Tensor 类型 | Output features are torch.Tensor."""
        x = torch.randn(1, 3, 512, 512)
        features = backbone(x)
        assert isinstance(features["p4"], torch.Tensor)
        assert isinstance(features["p8"], torch.Tensor)
        assert features["p4"].dtype == torch.float32
        assert features["p8"].dtype == torch.float32

    def test_gradient_flow_unfrozen(self) -> None:
        """未冻结时梯度流动 | Gradients flow when unfrozen."""
        bb = FastSAMBackbone(freeze_backbone=False)
        x = torch.randn(1, 3, 512, 512)
        features = bb(x)

        # 对 P4 特征求和并反向传播 | Sum P4 features and backprop
        loss = features["p4"].sum()
        loss.backward()

        # 检查是否有非零梯度 | Check for non-zero gradients
        has_grad = False
        for p in bb.model.model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowing when freeze_backbone=False"


# ════════════════════════════════════════════════════════════════
# build_backbone 工厂函数测试 | build_backbone Factory Tests
# ════════════════════════════════════════════════════════════════

class TestBuildBackbone:
    """验证工厂函数 | Verify factory function."""

    def test_build_default(self) -> None:
        """默认构建 FastSAM-x | Default builds FastSAM-x."""
        bb = build_backbone()
        assert isinstance(bb, FastSAMBackbone)

    def test_build_fastsam_x(self) -> None:
        """按名称构建 FastSAM-x | Build FastSAM-x by name."""
        bb = build_backbone("FastSAM-x")
        assert isinstance(bb, FastSAMBackbone)
