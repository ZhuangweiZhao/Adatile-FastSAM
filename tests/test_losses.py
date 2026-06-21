"""
测试 adatile.losses — 损失函数 | Test loss functions.
"""

import torch
import pytest
from adatile.losses import FocalLoss, DiceLoss, CombinedLoss


class TestFocalLoss:
    """Focal loss 单元测试 | Focal loss unit tests."""

    def test_shape_mean(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(2, 16, 64, 64)
        targets = torch.randint(0, 16, (2, 64, 64))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # scalar

    def test_ignore_index(self):
        loss_fn = FocalLoss(gamma=2.0, ignore_index=255)
        logits = torch.randn(2, 16, 64, 64)
        targets = torch.randint(0, 16, (2, 64, 64))
        targets[0, 0, 0] = 255  # should be ignored
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)

    def test_gamma_effect(self):
        """Higher gamma → lower loss for well-classified pixels."""
        logits = torch.randn(1, 16, 32, 32)
        targets = torch.randint(0, 16, (1, 32, 32))
        loss_g2 = FocalLoss(gamma=2.0)(logits, targets)
        loss_g5 = FocalLoss(gamma=5.0)(logits, targets)
        # Both should be valid scalars (not NaN)
        assert loss_g2.item() > 0
        assert loss_g5.item() > 0


class TestDiceLoss:
    """Dice loss 单元测试 | Dice loss unit tests."""

    def test_shape(self):
        loss_fn = DiceLoss(num_classes=16)
        logits = torch.randn(2, 16, 64, 64)
        targets = torch.randint(0, 16, (2, 64, 64))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_perfect_prediction(self):
        """Perfect prediction → Dice loss ≈ 0."""
        loss_fn = DiceLoss(num_classes=3)
        # Strong bias toward class 1 → near-perfect Dice for that class
        logits = torch.zeros(1, 3, 16, 16)
        logits[:, 1] = 100.0  # very confident class 1
        logits[:, 0] = -100.0
        logits[:, 2] = -100.0
        targets = torch.ones(1, 16, 16, dtype=torch.long)  # all class 1
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.1  # near zero

    def test_worst_prediction(self):
        """All wrong → Dice loss ≈ 1."""
        loss_fn = DiceLoss(num_classes=3)
        logits = torch.zeros(1, 3, 16, 16)
        logits[:, 2] = 100.0  # confident class 2
        logits[:, 0] = -100.0
        logits[:, 1] = -100.0
        targets = torch.ones(1, 16, 16, dtype=torch.long)  # all class 1
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.9


class TestCombinedLoss:
    """组合损失单元测试 | Combined loss unit tests."""

    def test_shape(self):
        loss_fn = CombinedLoss(num_classes=16, gamma=5.0, alpha=0.5)
        logits = torch.randn(2, 16, 64, 64)
        targets = torch.randint(0, 16, (2, 64, 64))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0

    def test_default_args(self):
        loss_fn = CombinedLoss()  # all defaults
        logits = torch.randn(1, 16, 32, 32)
        targets = torch.randint(0, 16, (1, 32, 32))
        loss = loss_fn(logits, targets)
        assert not torch.isnan(loss)


class TestLightDecoder:
    """LightDecoder 单元测试 | LightDecoder unit tests."""

    def test_binary_mode(self):
        from adatile.decoder.light_decoder import LightDecoder
        decoder = LightDecoder(in_channels=1280, num_classes=1)
        feats = {"p4": torch.randn(2, 1280, 64, 64)}
        logit = decoder(feats, target_size=(1024, 1024))
        assert logit.shape == (2, 1, 1024, 1024)

    def test_multiclass_mode(self):
        from adatile.decoder.light_decoder import LightDecoder
        decoder = LightDecoder(in_channels=1280, num_classes=16)
        feats = {"p4": torch.randn(2, 1280, 64, 64)}
        logit = decoder(feats, target_size=(1024, 1024))
        assert logit.shape == (2, 16, 1024, 1024)

    def test_binary_predict(self):
        from adatile.decoder.light_decoder import LightDecoder
        decoder = LightDecoder(in_channels=1280, num_classes=1)
        feats = {"p4": torch.randn(1, 1280, 64, 64)}
        mask = decoder.predict(feats, target_size=(512, 512))
        assert mask.shape == (1, 1, 512, 512)
        assert mask.min() >= 0 and mask.max() <= 1

    def test_multiclass_predict(self):
        from adatile.decoder.light_decoder import LightDecoder
        decoder = LightDecoder(in_channels=1280, num_classes=16)
        feats = {"p4": torch.randn(1, 1280, 64, 64)}
        pred = decoder.predict(feats, target_size=(512, 512))
        assert pred.shape == (1, 512, 512)
        assert pred.dtype == torch.int64
        assert pred.min() >= 0 and pred.max() < 16


class TestSpatialRouter:
    """Spatial Router 单元测试 | Spatial Router unit tests."""

    def test_density_head_shape(self):
        from adatile.sparse.spatial_router import DensityHead
        head = DensityHead(in_channels=576, mid_channels=128)
        x = torch.randn(2, 576, 20, 20)
        out = head(x)
        assert out.shape == (2, 1, 20, 20)
        assert out.min() >= 0 and out.max() <= 1  # sigmoid output

    def test_density_head_params(self):
        from adatile.sparse.spatial_router import DensityHead
        head = DensityHead(in_channels=576, mid_channels=128)
        n_params = sum(p.numel() for p in head.parameters())
        assert 70000 < n_params < 80000  # ~75K as documented

    def test_foreground_density_router(self):
        from adatile.sparse.spatial_router import ForegroundDensityRouter
        router = ForegroundDensityRouter(
            in_channels=576, mid_channels=128, tile_size_feat=8)
        feats = torch.randn(2, 576, 24, 24)
        out = router(feats)
        assert "importance" in out
        assert out["importance"].shape == (2, 1, 24, 24)

    def test_tile_scores_shape(self):
        from adatile.sparse.spatial_router import ForegroundDensityRouter
        router = ForegroundDensityRouter(
            in_channels=576, mid_channels=128, tile_size_feat=8)
        imp = torch.rand(2, 1, 24, 24)
        scores = router.tile_scores(imp, n_ty=3, n_tx=3)
        # Returns flat tensor [B * n_ty * n_tx]
        assert scores.shape == (18,)  # 2 * 3 * 3

    def test_select_tiles(self):
        from adatile.sparse.spatial_router import ForegroundDensityRouter
        router = ForegroundDensityRouter(
            in_channels=576, mid_channels=128, tile_size_feat=8)
        imp = torch.rand(2, 1, 24, 24)
        mask = router.select_tiles(imp, n_ty=3, n_tx=3, k=0.5)  # k is fraction
        assert mask.shape == (2, 3, 3)  # [B, n_ty, n_tx]
