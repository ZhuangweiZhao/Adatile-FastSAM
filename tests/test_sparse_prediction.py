"""Tests for SparsePrediction dataclass and interface contract enforcement."""

import pytest
import torch
from torch import Tensor


# ── SparsePrediction construction & validation ──────────────────────


class TestSparsePrediction:
    """Test SparsePrediction dataclass construction and __post_init__ validation."""

    @staticmethod
    def make_importance(batch=2, h=32, w=32):
        return torch.rand(batch, 1, h, w)

    def test_basic_construction(self):
        from adatile.core import SparsePrediction
        imp = self.make_importance()
        sp = SparsePrediction(importance=imp, density=imp)
        assert sp.importance is imp
        assert sp.density is imp
        assert sp.granularity_soft is None
        assert sp.granularity_hard is None

    def test_full_construction(self):
        from adatile.core import SparsePrediction
        imp = self.make_importance()
        den = self.make_importance()
        gs = torch.rand(2, 4, 32, 32).softmax(dim=1)
        gh = torch.randint(0, 4, (2, 1, 32, 32))
        sp = SparsePrediction(importance=imp, density=den, granularity_soft=gs, granularity_hard=gh)
        assert sp.importance.shape == (2, 1, 32, 32)
        assert sp.granularity_soft.shape == (2, 4, 32, 32)
        assert sp.granularity_hard.shape == (2, 1, 32, 32)

    def test_importance_wrong_dim(self):
        from adatile.core import SparsePrediction
        imp = torch.rand(32, 32)
        with pytest.raises(ValueError, match="4D"):
            SparsePrediction(importance=imp, density=imp.unsqueeze(0).unsqueeze(0))

    def test_importance_wrong_channels(self):
        from adatile.core import SparsePrediction
        imp = torch.rand(2, 3, 32, 32)
        with pytest.raises(ValueError, match="channel dim"):
            SparsePrediction(importance=imp, density=imp)

    def test_density_shape_mismatch(self):
        from adatile.core import SparsePrediction
        imp = self.make_importance()
        den = torch.rand(2, 1, 16, 16)
        with pytest.raises(ValueError, match="must match"):
            SparsePrediction(importance=imp, density=den)

    def test_importance_wrong_type(self):
        from adatile.core import SparsePrediction
        with pytest.raises(TypeError, match="Tensor"):
            SparsePrediction(importance=[1, 2, 3], density=torch.rand(2, 1, 4, 4))

    def test_granularity_soft_wrong_dim(self):
        from adatile.core import SparsePrediction
        imp = self.make_importance()
        gs = torch.rand(4, 32, 32)
        with pytest.raises(ValueError, match="4D"):
            SparsePrediction(importance=imp, density=imp, granularity_soft=gs)

    def test_granularity_soft_batch_mismatch(self):
        from adatile.core import SparsePrediction
        imp = self.make_importance(batch=2)
        gs = torch.rand(4, 4, 32, 32)
        with pytest.raises(ValueError, match="batch"):
            SparsePrediction(importance=imp, density=imp, granularity_soft=gs)

    def test_granularity_hard_wrong_channels(self):
        from adatile.core import SparsePrediction
        imp = self.make_importance()
        gh = torch.randint(0, 4, (2, 4, 32, 32))
        with pytest.raises(ValueError, match="channel dim"):
            SparsePrediction(importance=imp, density=imp, granularity_hard=gh)

    def test_granularity_hard_wrong_dim(self):
        from adatile.core import SparsePrediction
        imp = self.make_importance()
        gh = torch.randint(0, 4, (32, 32))
        with pytest.raises(ValueError, match="4D"):
            SparsePrediction(importance=imp, density=imp, granularity_hard=gh)

    def test_frozen_prevents_reassignment(self):
        from dataclasses import FrozenInstanceError
        from adatile.core import SparsePrediction
        imp = self.make_importance()
        sp = SparsePrediction(importance=imp, density=imp)
        with pytest.raises(FrozenInstanceError):
            sp.importance = torch.zeros(2, 1, 32, 32)

    def test_tensor_mutation_still_possible(self):
        from adatile.core import SparsePrediction
        imp = torch.ones(2, 1, 32, 32)
        sp = SparsePrediction(importance=imp, density=imp.clone())
        sp.importance.zero_()
        assert sp.importance.sum() == 0.0

    def test_to_dict_full(self):
        from adatile.core import SparsePrediction
        imp = self.make_importance()
        den = self.make_importance()
        gs = torch.rand(2, 4, 32, 32).softmax(dim=1)
        gh = torch.randint(0, 4, (2, 1, 32, 32))
        sp = SparsePrediction(importance=imp, density=den, granularity_soft=gs, granularity_hard=gh)
        d = sp.to_dict()
        assert d["importance"] is imp
        assert d["density"] is den
        assert d["granularity_soft"] is gs
        assert d["granularity_hard"] is gh

    def test_to_dict_without_granularity(self):
        from adatile.core import SparsePrediction
        imp = self.make_importance()
        sp = SparsePrediction(importance=imp, density=imp)
        d = sp.to_dict()
        assert "granularity_soft" not in d
        assert "granularity_hard" not in d


# ── SPM implementation interface compliance ─────────────────────────


class TestUniformImportance:
    """UniformImportance must return SparsePrediction."""

    def test_returns_sparse_prediction(self):
        from adatile.core import SparsePrediction
        from adatile.sparse.base import UniformImportance
        model = UniformImportance()
        features = {"res2": torch.rand(2, 256, 64, 64)}
        result = model(features)
        assert isinstance(result, SparsePrediction)
        assert result.importance.shape == (2, 1, 64, 64)
        assert result.density.shape == (2, 1, 64, 64)
        assert result.granularity_soft is None
        assert result.granularity_hard is None


class TestAdaSPMInterface:
    """AdaSPM and variants must return SparsePrediction."""

    @staticmethod
    def make_features(batch=1, h=128, w=128):
        return {
            "res2": torch.randn(batch, 256, h // 4, w // 4),
            "res3": torch.randn(batch, 256, h // 8, w // 8),
            "res4": torch.randn(batch, 256, h // 16, w // 16),
            "res5": torch.randn(batch, 256, h // 32, w // 32),
        }

    def test_adspm_returns_sparse_prediction(self):
        from adatile.core import SparsePrediction
        from adatile.sparse import AdaSPM
        model = AdaSPM(in_channels_list=[256, 256, 256, 256], use_transformer=False)
        features = self.make_features(batch=2)
        result = model(features)
        assert isinstance(result, SparsePrediction)
        assert result.importance.shape == (2, 1, 32, 32)
        assert result.density.shape == (2, 1, 32, 32)
        assert result.granularity_soft.shape == (2, 4, 32, 32)
        assert result.granularity_hard.shape == (2, 1, 32, 32)

    def test_density_only_spm_returns_sparse_prediction(self):
        from adatile.core import SparsePrediction
        from adatile.sparse import DensityOnlySPM
        model = DensityOnlySPM(in_channels_list=[256, 256, 256, 256], use_transformer=False)
        features = self.make_features()
        result = model(features)
        assert isinstance(result, SparsePrediction)
        assert result.granularity_soft is not None

    def test_sparselite_inherits_interface(self):
        from adatile.core import SparsePrediction
        from adatile.sparse import AdaSPMLite
        model = AdaSPMLite(in_channels_list=[256, 256, 256, 256])
        result = model(self.make_features())
        assert isinstance(result, SparsePrediction)

    def test_spmfull_inherits_interface(self):
        from adatile.core import SparsePrediction
        from adatile.sparse import AdaSPMFull
        model = AdaSPMFull(in_channels_list=[256, 256, 256, 256])
        result = model(self.make_features())
        assert isinstance(result, SparsePrediction)


# ── Gradient flow ───────────────────────────────────────────────────


class TestSparsePredictionGradients:
    """Gradients must flow through SparsePrediction fields."""

    @staticmethod
    def make_features(batch=1, h=128, w=128):
        return {
            "res2": torch.randn(batch, 256, h // 4, w // 4),
            "res3": torch.randn(batch, 256, h // 8, w // 8),
            "res4": torch.randn(batch, 256, h // 16, w // 16),
            "res5": torch.randn(batch, 256, h // 32, w // 32),
        }

    def test_adspm_importance_gradient(self):
        from adatile.sparse import AdaSPM
        model = AdaSPM(in_channels_list=[256, 256, 256, 256], use_transformer=False)
        features = self.make_features()
        result = model(features)
        loss = result.importance.sum()
        loss.backward()
        grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
        assert any(grads), "No gradients flowed to AdaSPM parameters"

    def test_adspm_density_gradient(self):
        from adatile.sparse import AdaSPM
        model = AdaSPM(in_channels_list=[256, 256, 256, 256], use_transformer=False)
        features = self.make_features()
        result = model(features)
        loss = result.density.sum()
        loss.backward()
        grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
        assert any(grads), "No gradients flowed to AdaSPM parameters"


# ── Device & batch consistency ──────────────────────────────────────


class TestSparsePredictionConsistency:
    """Device and batch consistency checks."""

    def test_device_consistency(self):
        from adatile.core import SparsePrediction
        imp = torch.rand(1, 1, 32, 32)
        den = torch.rand(1, 1, 32, 32)
        gs = torch.rand(1, 4, 32, 32)
        gh = torch.randint(0, 4, (1, 1, 32, 32))
        sp = SparsePrediction(importance=imp, density=den, granularity_soft=gs, granularity_hard=gh)
        assert sp.importance.device == sp.density.device
        assert sp.importance.device == sp.granularity_soft.device
        assert sp.importance.device == sp.granularity_hard.device

    def test_batch_consistency(self):
        from adatile.core import SparsePrediction
        for b in [1, 2, 4]:
            imp = torch.rand(b, 1, 32, 32)
            sp = SparsePrediction(importance=imp, density=imp)
            assert sp.importance.shape[0] == b
            assert sp.density.shape[0] == b

    def test_spatial_consistency_across_batch(self):
        from adatile.core import SparsePrediction
        imp = torch.rand(3, 1, 64, 64)
        den = torch.rand(3, 1, 64, 64)
        sp = SparsePrediction(importance=imp, density=den)
        assert sp.importance.shape[2] == sp.density.shape[2]
        assert sp.importance.shape[3] == sp.density.shape[3]
