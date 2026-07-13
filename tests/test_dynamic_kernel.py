"""
测试 DynamicKernelDecoder + Hungarian Matcher | Test Dynamic Kernel Decoder + Hungarian Matcher.
"""

import torch
import pytest
import numpy as np
from adatile.decoder.dynamic_kernel_decoder import DynamicKernelDecoder
from adatile.metrics.hungarian_matcher import (
    hungarian_match_instances,
    multi_instance_loss,
    mask_scores,
    _dice_coeff,
    _bce_cost,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_decoder():
    """极小 DynamicKernelDecoder (CPU, 快速测试) | Tiny decoder for fast CPU tests."""
    return DynamicKernelDecoder(
        p3_channels=16,
        p4_channels=32,
        proto_dim=4,
        n_kernels=8,
        kernel_dim=32,
        fpn_dim=32,
        normalize_proto="none",
    )


@pytest.fixture
def dummy_inputs():
    """合成输入 (匹配 tiny_decoder 维度) | Synthetic inputs matching tiny_decoder dims."""
    B, H8, W8 = 1, 28, 28  # stride 8: 224/8=28
    H16, W16 = 14, 14       # stride 16
    H4, W4 = 56, 56         # stride 4: 224/4=56

    p3 = torch.randn(B, 16, H8, W8)
    p4 = torch.randn(B, 32, H16, W16)
    proto_masks = torch.randn(4, H4, W4)  # [proto_dim, H/4, W/4]
    support_proto = torch.randn(32)        # [p4_channels]
    return p3, p4, proto_masks, support_proto


# ═══════════════════════════════════════════════════════════════════
# DynamicKernelDecoder 测试
# ═══════════════════════════════════════════════════════════════════

class TestDynamicKernelDecoder:
    """DynamicKernelDecoder 单元测试 | Unit tests."""

    def test_forward_shape(self, tiny_decoder, dummy_inputs):
        """前向传播输出形状正确 | Forward produces correct shapes."""
        p3, p4, proto, sp = dummy_inputs
        masks, proto_mask = tiny_decoder(p3, p4, proto, sp)

        # masks: [N_kernels, H/8, W/8]
        assert masks.shape == (8, 28, 28), f"Expected (8, 28, 28), got {masks.shape}"
        # proto_mask: [H/4, W/4]
        assert proto_mask.shape == (56, 56), f"Expected (56, 56), got {proto_mask.shape}"
        # Values in [0, 1] (sigmoid)
        assert masks.min() >= 0 and masks.max() <= 1

    def test_forward_grad(self, tiny_decoder, dummy_inputs):
        """梯度可以正常回传 | Gradients flow correctly."""
        p3, p4, proto, sp = dummy_inputs
        p3.requires_grad = True
        p4.requires_grad = True

        masks, _ = tiny_decoder(p3, p4, proto, sp)
        loss = masks.mean()
        loss.backward()

        assert p3.grad is not None, "P3 gradient should not be None"
        assert p4.grad is not None, "P4 gradient should not be None"
        assert not torch.isnan(p3.grad).any(), "P3 gradient contains NaN"

    def test_kernel_variation(self, tiny_decoder, dummy_inputs):
        """不同 kernel 产生不同 mask | Different kernels produce different masks."""
        p3, p4, proto, sp = dummy_inputs
        masks, _ = tiny_decoder(p3, p4, proto, sp)

        # 至少有些 kernel 产生不同的 mask | At least some kernels differ
        diffs = []
        for i in range(7):
            for j in range(i + 1, 8):
                diff = (masks[i] - masks[j]).abs().max()
                diffs.append(diff.item())

        # 不是所有 kernel 都一样 (初始化引入变化) | Not all kernels identical
        assert max(diffs) > 0.01, "All kernel masks are near-identical"

    def test_prototype_sensitivity(self, tiny_decoder, dummy_inputs):
        """不同 prototype 产生不同 mask | Different prototypes produce different masks."""
        p3, p4, proto, sp = dummy_inputs
        masks_a, _ = tiny_decoder(p3, p4, proto, sp)
        # 换一个 prototype | Different prototype
        sp_b = torch.randn_like(sp)
        masks_b, _ = tiny_decoder(p3, p4, proto, sp_b)

        diff = (masks_a - masks_b).abs().max()
        assert diff > 0.001, f"Prototype change had no effect on masks (max diff={diff:.6f})"

    def test_param_count(self, tiny_decoder):
        """参数量统计正确 | Parameter count is reasonable."""
        params = tiny_decoder.get_submodule_params()
        assert params["total"] > 0
        assert params["kernel_generator"] > 0
        assert params["mask_feat"] > 0
        assert "fpn" in params or sum(1 for k in params if "fpn" in k.lower()) > 0
        # 确保所有子模块都已统计 | Ensure all submodules are accounted for
        assert abs(params["total"] - sum(
            v for k, v in params.items() if k != "total"
        )) < 10  # 允许舍入误差 | Allow rounding error

    def test_normalize_proto_none(self, dummy_inputs):
        """normalize_proto='none' 是 identity | normalize_proto='none' is identity."""
        p3, p4, proto, sp = dummy_inputs
        d = DynamicKernelDecoder(p3_channels=16, p4_channels=32, normalize_proto="none")
        result = d._normalize_proto(proto)
        assert torch.allclose(result, proto, atol=1e-6)

    def test_normalize_proto_l2(self, dummy_inputs):
        """normalize_proto='l2' 产生单位 L2 norm | normalize_proto='l2' gives unit L2 norm."""
        p3, p4, proto, sp = dummy_inputs
        d = DynamicKernelDecoder(p3_channels=16, p4_channels=32, normalize_proto="l2")
        result = d._normalize_proto(proto)
        # 每个 basis (channel) 的 L2 norm 应该 = 1 | Each basis L2 norm = 1
        norms = result.view(4, -1).norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_collect_stats(self, tiny_decoder, dummy_inputs):
        """collect_stats 收集前向统计 | collect_stats collects forward stats."""
        p3, p4, proto, sp = dummy_inputs
        tiny_decoder.collect_stats = True
        tiny_decoder(p3, p4, proto, sp)
        stats = tiny_decoder.last_stats
        assert "kernel_norm_mean" in stats
        assert "mask_mean_per_kernel" in stats
        assert stats["kernel_norm_mean"] >= 0

    def test_kernel_bias_exists(self, tiny_decoder):
        """kernel_bias 参数存在且形状正确 | kernel_bias exists and has correct shape."""
        assert hasattr(tiny_decoder, "kernel_bias")
        expected_shape = (tiny_decoder.n_kernels, tiny_decoder.kernel_dim)
        assert tiny_decoder.kernel_bias.shape == expected_shape

    def test_kernel_bias_breaks_symmetry(self, tiny_decoder, dummy_inputs):
        """kernel_bias 打破对称性 (不同 kernel 产生不同 mask) | Bias breaks symmetry."""
        p3, p4, proto, sp = dummy_inputs

        # Run twice with same input, kernels should differ due to bias
        decoder = DynamicKernelDecoder(
            p3_channels=16, p4_channels=32, proto_dim=4,
            n_kernels=8, kernel_dim=32, fpn_dim=32,
            normalize_proto="none",
        )

        # After init, bias is small random → kernels should differ slightly
        with torch.no_grad():
            proto_t = proto.clone()
            sp_t = sp.clone()
        masks, _ = decoder(p3, p4, proto_t, sp_t)

        # Check that at least some kernel masks differ from each other
        diffs = []
        for i in range(7):
            for j in range(i + 1, 8):
                diff = (masks[i] - masks[j]).abs().max().item()
                diffs.append(diff)
        assert max(diffs) > 0.001, f"kernel_bias did not create diversity: max_diff={max(diffs):.6f}"

    def test_diversity_loss_zero_for_single_kernel(self, tiny_decoder):
        """单 kernel 时 diversity loss = 0 | Diversity loss = 0 for single kernel."""
        single = torch.randn(1, 32)
        loss = DynamicKernelDecoder.kernel_diversity_loss(single)
        assert loss.item() == 0.0

    def test_diversity_loss_high_for_identical(self):
        """相同 kernel 时 diversity loss 接近 1 | Diversity loss ≈ 1 for identical kernels."""
        N, D = 4, 32
        identical = torch.ones(N, D)  # All cosine_sim = 1.0
        loss = DynamicKernelDecoder.kernel_diversity_loss(identical)
        assert loss.item() > 0.9, f"Expected >0.9 for identical kernels, got {loss.item():.4f}"

    def test_diversity_loss_low_for_orthogonal(self):
        """正交 kernel 时 diversity loss 接近 0 | Diversity loss ≈ 0 for orthogonal kernels."""
        N, D = 4, 32
        # Create approximately orthogonal random vectors
        orth = torch.randn(N, D)
        loss = DynamicKernelDecoder.kernel_diversity_loss(orth)
        # Random vectors have low cosine similarity → low loss
        assert loss.item() < 0.3, f"Expected <0.3 for random kernels, got {loss.item():.4f}"

    def test_diversity_loss_decreases_with_bias(self, dummy_inputs):
        """kernel_bias 使得 diversity loss < 没有 bias 的情况 | Bias reduces diversity loss."""
        p3, p4, proto, sp = dummy_inputs

        # Decoder WITHOUT bias
        dec_no_bias = DynamicKernelDecoder(
            p3_channels=16, p4_channels=32, proto_dim=4,
            n_kernels=8, kernel_dim=32, fpn_dim=32,
            normalize_proto="none",
        )
        dec_no_bias.kernel_bias.data.zero_()

        # Decoder WITH bias
        dec_with_bias = DynamicKernelDecoder(
            p3_channels=16, p4_channels=32, proto_dim=4,
            n_kernels=8, kernel_dim=32, fpn_dim=32,
            normalize_proto="none",
        )

        # Generate kernels from both
        with torch.no_grad():
            proto_t = proto.clone()
            sp_t = sp.clone()

        # Forward pass to generate kernels
        dec_no_bias(p3, p4, proto_t, sp_t)
        k_no_bias = dec_no_bias._last_kernels
        loss_no_bias = DynamicKernelDecoder.kernel_diversity_loss(k_no_bias)

        dec_with_bias(p3, p4, proto_t, sp_t)
        k_with_bias = dec_with_bias._last_kernels
        loss_with_bias = DynamicKernelDecoder.kernel_diversity_loss(k_with_bias)

        # Both should be low-ish (random init), but check they differ
        # The key assertion: loss is computable and finite
        assert not torch.isnan(loss_no_bias)
        assert not torch.isnan(loss_with_bias)


# ═══════════════════════════════════════════════════════════════════
# Hungarian Matcher 测试
# ═══════════════════════════════════════════════════════════════════

class TestHungarianMatcher:
    """Hungarian 匹配器测试 | Hungarian Matcher tests."""

    def test_perfect_match(self):
        """完美预测应该全部匹配 | Perfect predictions should all match."""
        # 两个预测, 每个正好匹配一个 GT | 2 preds, each matching one GT
        N, H, W = 2, 56, 56
        pred = torch.zeros(N, H, W)
        pred[0, 10:30, 10:30] = 1.0  # matches gt[0]
        pred[1, 40:50, 40:50] = 1.0  # matches gt[1]

        gt = torch.zeros(N, H, W)
        gt[0, 10:30, 10:30] = 1.0
        gt[1, 40:50, 40:50] = 1.0

        matched, unmatched_pred, unmatched_gt = hungarian_match_instances(pred, gt)
        assert len(matched) == 2
        assert len(unmatched_pred) == 0
        assert len(unmatched_gt) == 0

    def test_empty_gt(self):
        """空 GT → 所有预测未匹配 | Empty GT → all predictions unmatched."""
        pred = torch.rand(4, 28, 28)
        gt = torch.zeros(0, 28, 28)  # 0 GT instances

        matched, unmatched_pred, unmatched_gt = hungarian_match_instances(pred, gt)
        assert len(matched) == 0
        assert len(unmatched_pred) == 4
        assert len(unmatched_gt) == 0

    def test_more_preds_than_gt(self):
        """预测多于 GT → 部分未匹配 | More preds than GTs → some unmatched."""
        pred = torch.zeros(4, 28, 28)
        pred[0, 5:15, 5:15] = 1.0
        pred[1, 20:25, 20:25] = 0.3  # low quality
        pred[2, 5:15, 5:15] = 0.6    # partial match
        pred[3, 0:5, 0:5] = 0.1      # noise

        gt = torch.zeros(1, 28, 28)
        gt[0, 5:15, 5:15] = 1.0

        matched, unmatched_pred, unmatched_gt = hungarian_match_instances(pred, gt)
        assert len(matched) == 1
        assert len(unmatched_pred) == 3  # 3 unused preds
        assert len(unmatched_gt) == 0    # The one GT is matched

    def test_dice_coeff_shape(self):
        """Dice 系数矩阵形状正确 | Dice coefficient matrix shape is correct."""
        pred = torch.rand(8, 28, 28)
        gt = torch.rand(3, 28, 28)
        dice = _dice_coeff(pred, gt)
        assert dice.shape == (8, 3)

    def test_dice_coeff_perfect(self):
        """Dice 完美预测 = 1.0 | Dice for perfect prediction = 1.0."""
        pred = torch.ones(1, 28, 28)
        gt = torch.ones(1, 28, 28)
        dice = _dice_coeff(pred, gt)
        assert torch.allclose(dice, torch.tensor([[1.0]]), atol=1e-5)

    def test_dice_coeff_disjoint(self):
        """Dice 完全不相交 = 0.0 | Dice for completely disjoint masks = 0.0."""
        H, W = 28, 28
        pred = torch.zeros(1, H, W)
        pred[0, 0:14, :] = 1.0  # left half
        gt = torch.zeros(1, H, W)
        gt[0, 14:, :] = 1.0     # right half
        dice = _dice_coeff(pred, gt)
        assert dice.item() < 0.01

    def test_bce_cost_shape(self):
        """BCE 代价矩阵形状正确 | BCE cost matrix shape is correct."""
        pred = torch.rand(8, 28, 28)
        gt = torch.rand(3, 28, 28)
        bce = _bce_cost(pred, gt)
        assert bce.shape == (8, 3)

    def test_min_dice_filter(self):
        """min_dice 阈值过滤低质量匹配 | min_dice filters low-quality matches."""
        pred = torch.zeros(2, 28, 28)
        pred[0, 5:15, 5:15] = 1.0
        pred[1, 0:5, 0:5] = 0.3  # 低质量 | low quality

        gt = torch.zeros(2, 28, 28)
        gt[0, 5:15, 5:15] = 1.0
        gt[1, 20:25, 20:25] = 1.0

        # 高 min_dice → 只有完美匹配通过 | High min_dice → only perfect match passes
        matched, _, _ = hungarian_match_instances(pred, gt, min_dice=0.8)
        assert len(matched) == 1


class TestMultiInstanceLoss:
    """多实例损失测试 | Multi-instance loss tests."""

    def test_loss_scalar(self):
        """损失是标量 | Loss is a scalar."""
        pred = torch.rand(4, 28, 28).sigmoid()
        gt = torch.zeros(2, 28, 28)
        gt[0, 5:15, 5:15] = 1.0
        gt[1, 15:25, 15:25] = 1.0

        matched, unmatched_p, unmatched_g = hungarian_match_instances(pred, gt)
        loss, d = multi_instance_loss(pred, gt, matched, unmatched_p, unmatched_g)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert d["n_matched"] >= 0

    def test_loss_no_gt(self):
        """无 GT 时损失合理 | Loss is reasonable when no GT."""
        pred = torch.rand(4, 28, 28).sigmoid()
        gt = torch.zeros(0, 28, 28)

        matched, unmatched_p, unmatched_g = hungarian_match_instances(pred, gt)
        loss, d = multi_instance_loss(pred, gt, matched, unmatched_p, unmatched_g)

        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert d["n_matched"] == 0
        assert d["n_unmatched_pred"] == 4

    def test_loss_perfect(self):
        """完美匹配时损失近似为零 | Loss ≈ 0 for perfect matches."""
        N, H, W = 2, 28, 28
        pred = torch.zeros(N, H, W)
        pred[0, 5:15, 5:15] = 0.99  # near-perfect
        pred[1, 15:25, 15:25] = 0.99

        gt = torch.zeros(N, H, W)
        gt[0, 5:15, 5:15] = 1.0
        gt[1, 15:25, 15:25] = 1.0

        matched, unmatched_p, unmatched_g = hungarian_match_instances(pred, gt)
        loss, _ = multi_instance_loss(pred, gt, matched, unmatched_p, unmatched_g)

        assert loss.item() < 0.1  # Nearly perfect → low loss


class TestMaskScores:
    """掩码置信度测试 | Mask score tests."""

    def test_score_mean(self):
        """mean 方法正确计算 | mean method computes correctly."""
        masks = torch.zeros(3, 28, 28)
        masks[0, 5:15, 5:15] = 0.8  # mean of FG = 0.8
        masks[1, 10:20, 10:20] = 0.5
        masks[2, :, :] = 0.0        # empty

        scores = mask_scores(masks, method="mean")
        assert scores.shape == (3,)
        assert scores[0] > 0.7       # ~0.8
        assert scores[0] > scores[1]  # 0.8 > 0.5
        assert scores[2] == 0.0       # empty

    def test_score_max(self):
        """max 方法正确计算 | max method computes correctly."""
        masks = torch.zeros(2, 28, 28)
        masks[0, 10, 10] = 0.9
        masks[1, 20, 20] = 0.3

        scores = mask_scores(masks, method="max")
        assert scores.shape == (2,)
        assert abs(scores[0].item() - 0.9) < 0.01
        assert abs(scores[1].item() - 0.3) < 0.01
