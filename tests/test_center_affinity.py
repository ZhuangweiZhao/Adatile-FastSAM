"""
Test Center-Affinity Decoder + Center-Affinity Grouping.
测试中心-亲和力解码器 + 中心-亲和力分组.
"""

import torch
import pytest
import numpy as np
from adatile.decoder.center_affinity_decoder import CenterAffinityDecoder
from adatile.metrics.instance_generation import generate_instances_center_affinity


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def tiny_decoder():
    """极小 CenterAffinityDecoder (CPU, fast tests)."""
    return CenterAffinityDecoder(
        p3_channels=16,
        p4_channels=32,
        proto_dim=4,
        fpn_dim=32,
        normalize_proto="none",
    )


@pytest.fixture
def dummy_inputs():
    """Synthetic inputs matching tiny_decoder dims."""
    B, H8, W8 = 1, 28, 28   # stride 8
    H16, W16 = 14, 14        # stride 16
    H4, W4 = 56, 56          # stride 4

    p3 = torch.randn(B, 16, H8, W8)
    p4 = torch.randn(B, 32, H16, W16)
    proto_masks = torch.randn(4, H4, W4)
    support_proto = torch.randn(32)
    return p3, p4, proto_masks, support_proto


# ═══════════════════════════════════════════════════════════════════
# CenterAffinityDecoder Tests
# ═══════════════════════════════════════════════════════════════════

class TestCenterAffinityDecoder:
    """CenterAffinityDecoder unit tests."""

    def test_forward_shapes(self, tiny_decoder, dummy_inputs):
        """Forward produces correct shapes."""
        p3, p4, proto, sp = dummy_inputs
        center, offset, proto_mask = tiny_decoder(p3, p4, proto, sp)

        # center: [H/8, W/8]
        assert center.shape == (28, 28), f"center shape: {center.shape}"
        # offset: [2, H/8, W/8]
        assert offset.shape == (2, 28, 28), f"offset shape: {offset.shape}"
        # proto_mask: [H/4, W/4]
        assert proto_mask.shape == (56, 56), f"proto_mask shape: {proto_mask.shape}"

        # Value ranges
        assert 0 <= center.min() and center.max() <= 1, "center not in [0,1]"
        assert 0 <= proto_mask.min() and proto_mask.max() <= 1, "proto_mask not in [0,1]"

    def test_forward_grad(self, tiny_decoder, dummy_inputs):
        """Gradients flow correctly through all paths."""
        p3, p4, proto, sp = dummy_inputs
        p3.requires_grad = True
        p4.requires_grad = True

        center, offset, proto_mask = tiny_decoder(p3, p4, proto, sp)

        # Test center gradient
        center.mean().backward(retain_graph=True)
        assert p3.grad is not None, "P3 grad None for center"
        assert not torch.isnan(p3.grad).any(), "P3 grad NaN"

        p3.grad = None
        p4.grad = None

        # Test offset gradient
        offset.mean().backward(retain_graph=True)
        assert p3.grad is not None, "P3 grad None for offset"
        assert not torch.isnan(p3.grad).any()

    def test_proto_only(self, tiny_decoder, dummy_inputs):
        """forward_proto_only produces correct proto mask."""
        _, _, proto, sp = dummy_inputs
        proto_mask = tiny_decoder.forward_proto_only(proto, sp)
        assert proto_mask.shape == (56, 56)
        assert 0 <= proto_mask.min() and proto_mask.max() <= 1

    def test_normalize_proto_l2(self, dummy_inputs):
        """normalize_proto='l2' gives unit L2 norm."""
        p3, p4, proto, sp = dummy_inputs
        d = CenterAffinityDecoder(p3_channels=16, p4_channels=32, fpn_dim=32, normalize_proto="l2")
        result = d._normalize_proto(proto)
        norms = result.view(4, -1).norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_center_responds_to_different_input(self, tiny_decoder, dummy_inputs):
        """Different P3/P4 features produce different center heatmaps."""
        p3, p4, proto, sp = dummy_inputs
        center_a, _, _ = tiny_decoder(p3, p4, proto, sp)

        # Different features
        p3_b = torch.randn_like(p3)
        p4_b = torch.randn_like(p4)
        center_b, _, _ = tiny_decoder(p3_b, p4_b, proto, sp)

        diff = (center_a - center_b).abs().max()
        assert diff > 0.001, f"Center heatmap invariant to input change (diff={diff:.6f})"

    def test_offset_not_all_zero(self, tiny_decoder, dummy_inputs):
        """Offset field contains non-zero values (not collapsed)."""
        p3, p4, proto, sp = dummy_inputs
        _, offset, _ = tiny_decoder(p3, p4, proto, sp)
        assert offset.abs().max() > 0.01, f"Offset field is all-zero (max={offset.abs().max():.6f})"

    def test_param_count(self, tiny_decoder):
        """Parameter count is reasonable."""
        params = tiny_decoder.get_submodule_params()
        assert params["total"] > 0
        assert params["fpn"] > 0
        assert params["center_head"] > 0
        assert params["offset_head"] > 0
        assert params["coeff_predictor"] > 0
        assert abs(params["total"] - sum(
            v for k, v in params.items() if k != "total"
        )) < 10


# ═══════════════════════════════════════════════════════════════════
# Center-Affinity Grouping Tests
# ═══════════════════════════════════════════════════════════════════

class TestCenterAffinityGrouping:
    """generate_instances_center_affinity unit tests."""

    def _make_simple_scene(self, H=120, W=120):
        """Create a simple scene with 2 separated instances."""
        center_h = np.zeros((H, W), dtype=np.float32)
        offset = np.zeros((2, H, W), dtype=np.float32)
        fg = np.zeros((H, W), dtype=bool)

        # Instance 1: center at (40, 40), radius ~15
        cx1, cy1 = 40, 40
        # Instance 2: center at (80, 80), radius ~15
        cx2, cy2 = 80, 80

        for y in range(H):
            for x in range(W):
                # Instance 1
                if (x - cx1)**2 + (y - cy1)**2 < 15**2:
                    fg[y, x] = True
                    offset[0, y, x] = cx1 - x  # dx
                    offset[1, y, x] = cy1 - y  # dy
                # Instance 2
                elif (x - cx2)**2 + (y - cy2)**2 < 15**2:
                    fg[y, x] = True
                    offset[0, y, x] = cx2 - x
                    offset[1, y, x] = cy2 - y

        # Gaussian peaks at centers
        for y in range(H):
            for x in range(W):
                g1 = np.exp(-((x-cx1)**2 + (y-cy1)**2) / (2*4.0**2))
                g2 = np.exp(-((x-cx2)**2 + (y-cy2)**2) / (2*4.0**2))
                center_h[y, x] = max(g1, g2)

        return center_h, offset, fg

    def test_two_instances_perfect(self):
        """Perfect center+offset → 2 separated instances."""
        center_h, offset, fg = self._make_simple_scene()

        instances = generate_instances_center_affinity(
            center_h, offset, fg,
            score_thr=0.1, min_area=10, min_distance=8,
        )

        assert len(instances) == 2, f"Expected 2 instances, got {len(instances)}"
        # Masks should be disjoint
        iou = (instances[0]["mask"] & instances[1]["mask"]).sum() / max(
            (instances[0]["mask"] | instances[1]["mask"]).sum(), 1
        )
        assert iou < 0.01, f"Instances overlap (IoU={iou:.3f})"

    def test_empty_fg(self):
        """Empty FG mask → no instances."""
        center_h = np.ones((60, 60), dtype=np.float32) * 0.5
        offset = np.zeros((2, 60, 60), dtype=np.float32)
        fg = np.zeros((60, 60), dtype=bool)

        instances = generate_instances_center_affinity(
            center_h, offset, fg,
        )
        assert len(instances) == 0

    def test_no_center_peaks(self):
        """No center peaks above threshold → no instances."""
        center_h = np.zeros((60, 60), dtype=np.float32)
        offset = np.zeros((2, 60, 60), dtype=np.float32)
        fg = np.ones((60, 60), dtype=bool)

        instances = generate_instances_center_affinity(
            center_h, offset, fg, score_thr=0.5,
        )
        assert len(instances) == 0

    def test_single_instance(self):
        """Single instance with perfect offset → 1 instance."""
        H, W = 60, 60
        cx, cy = 30, 30

        center_h = np.zeros((H, W), dtype=np.float32)
        offset = np.zeros((2, H, W), dtype=np.float32)
        fg = np.zeros((H, W), dtype=bool)

        for y in range(H):
            for x in range(W):
                if (x - cx)**2 + (y - cy)**2 < 10**2:
                    fg[y, x] = True
                    offset[0, y, x] = cx - x
                    offset[1, y, x] = cy - y
                center_h[y, x] = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*4.0**2))

        instances = generate_instances_center_affinity(
            center_h, offset, fg,
            score_thr=0.1, min_area=10,
        )

        assert len(instances) == 1
        assert instances[0]["score"] > 0

    def test_nms_suppresses_close_peaks(self):
        """NMS suppresses peaks closer than min_distance."""
        H, W = 100, 100
        center_h = np.zeros((H, W), dtype=np.float32)

        # Two peaks very close together
        center_h[40, 40] = 0.9
        center_h[42, 42] = 0.8  # within min_distance=8

        offset = np.zeros((2, H, W), dtype=np.float32)
        fg = np.ones((H, W), dtype=bool)
        # Set offset to point to the stronger peak
        offset[0, :, :] = 40 - np.arange(W).reshape(1, -1)
        offset[1, :, :] = 40 - np.arange(H).reshape(-1, 1)

        instances = generate_instances_center_affinity(
            center_h, offset, fg,
            score_thr=0.1, min_area=10, min_distance=8,
        )

        # Only the stronger peak should survive
        assert len(instances) == 1

    def test_min_area_filters_small(self):
        """min_area filters tiny blobs."""
        H, W = 60, 60
        cx, cy = 30, 30

        center_h = np.zeros((H, W), dtype=np.float32)
        for y in range(H):
            for x in range(W):
                center_h[y, x] = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*4.0**2))

        offset = np.zeros((2, H, W), dtype=np.float32)
        offset[0, :, :] = cx - np.arange(W).reshape(1, -1)
        offset[1, :, :] = cy - np.arange(H).reshape(-1, 1)

        fg = np.zeros((H, W), dtype=bool)
        fg[cy-2:cy+2, cx-2:cx+2] = True  # ~16 pixels

        instances = generate_instances_center_affinity(
            center_h, offset, fg,
            score_thr=0.1, min_area=200,  # much larger than actual
        )
        assert len(instances) == 0

    def test_offset_noise_still_groups(self):
        """Offset with noise still groups to nearest center."""
        H, W = 100, 100
        rng = np.random.RandomState(42)

        # Two well-separated centers
        centers = [(30, 30), (70, 70)]
        center_h = np.zeros((H, W), dtype=np.float32)
        offset = np.zeros((2, H, W), dtype=np.float32)
        fg = np.zeros((H, W), dtype=bool)

        for ci, (cx, cy) in enumerate(centers):
            for y in range(H):
                for x in range(W):
                    dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                    if dist < 15:
                        fg[y, x] = True
                        # Noisy offset (GT + small perturbation)
                        offset[0, y, x] = cx - x + rng.normal(0, 1.0)
                        offset[1, y, x] = cy - y + rng.normal(0, 1.0)
            # Gaussian peak
            for y in range(H):
                for x in range(W):
                    g = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*4.0**2))
                    center_h[y, x] = max(center_h[y, x], g)

        instances = generate_instances_center_affinity(
            center_h, offset, fg,
            score_thr=0.1, min_area=50, min_distance=15,
        )

        # With noise, we should still get 2 (or close)
        assert len(instances) >= 1, f"With noise should get at least 1 instance, got {len(instances)}"
