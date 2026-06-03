"""End-to-end pipeline smoke test — verifies the full forward pass.

This test MUST pass before any paper submission.
It validates:
    1. Config → model construction
    2. Full forward pass (standard)
    3. Few-shot forward pass
    4. Gradient flow through all components
    5. SparsePrediction contract enforcement
    6. Router gradient integrity
"""

import pytest
import torch

from adatile.config import Config


def _build_model(cfg=None):
    """Build a minimal model for smoke testing."""
    if cfg is None:
        cfg = Config()
        cfg.experiment_name = "test_e2e"
        cfg.backbone.name = "ResNet50Backbone"
        cfg.backbone.pretrained = False
        cfg.backbone.embed_dim = 256
        cfg.backbone.depth = 4
        cfg.backbone.num_heads = 4
        cfg.backbone.patch_size = 16
        cfg.backbone.output_scales = [4, 8, 16, 32]
        cfg.sparse.name = "ada_spm"
        cfg.sparse.num_scales = 4
        cfg.tokenizer.name = "dynamic_tile"
        cfg.tokenizer.tile_sizes = [384, 768, 1536]
        cfg.tokenizer.max_tokens_per_image = 256
        cfg.tokenizer.skip_mode = "threshold"
        cfg.router.name = "DTRv2Router"
        cfg.router.embed_dim = 256
        cfg.decoder.name = "fastsam_decoder"
        cfg.decoder.mask_dim = 256
        cfg.decoder.num_mask_tokens = 4
        cfg.prototype.name = "masked_avg"

    from adatile.modeling import build_adatile_fastsam
    return build_adatile_fastsam(cfg)


class TestEndToEndPipeline:
    """Full pipeline smoke tests."""

    def test_model_builds(self):
        """Model constructs without errors."""
        model = _build_model()
        assert model is not None
        assert hasattr(model, "pipeline")
        assert model.pipeline.backbone is not None
        assert model.pipeline.sparse_predictor is not None
        assert model.pipeline.tokenizer is not None
        assert model.pipeline.router is not None
        assert model.pipeline.decoder is not None

    def test_forward_standard(self):
        """Standard forward pass completes without errors."""
        model = _build_model()
        model.eval()
        image = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            output, aux = model(image)
        assert output is not None
        assert hasattr(output, "masks")
        assert hasattr(output, "scores")
        assert "importance" in aux
        assert "density" in aux
        assert "routing_weights" in aux

    def test_forward_fewshot(self):
        """Few-shot forward pass completes without errors."""
        model = _build_model()
        model.eval()
        support = torch.randn(2, 3, 1024, 1024)
        support_masks = torch.randint(0, 2, (2, 1024, 1024)).float()
        query = torch.randn(1, 3, 1024, 1024)
        with torch.no_grad():
            output, aux = model(
                query,
                support_images=support,
                support_masks=support_masks,
                class_ids=[0, 0],
            )
        assert output is not None
        assert "prototypes" in aux

    def test_gradient_flow_to_adaspm(self):
        """Gradients flow back to Ada-SPM from downstream losses."""
        model = _build_model()
        model.train()
        image = torch.randn(1, 3, 1024, 1024)

        output, aux = model(image)

        # Gradients should reach Ada-SPM parameters via:
        # 1. Density loss on importance
        # 2. Planning alignment loss
        # 3. Router importance extraction
        spm_params = list(model.pipeline.sparse_predictor.parameters())
        trainable_spm = [p for p in spm_params if p.requires_grad]

        if trainable_spm:
            loss = aux["importance"].sum() + aux["density"].sum()
            if aux.get("granularity_soft") is not None:
                loss = loss + aux["granularity_soft"].sum()

            model.zero_grad()
            loss.backward()

            grads = [
                p.grad is not None and p.grad.abs().sum() > 0
                for p in trainable_spm
            ]
            assert any(grads), (
                "No gradients flowed to Ada-SPM parameters. "
                "The gradient connection between Ada-SPM and downstream "
                "losses is broken."
            )

    def test_gradient_flow_through_router(self):
        """Gradients flow through DTRv2 router to routing head."""
        model = _build_model()
        model.train()
        image = torch.randn(1, 3, 1024, 1024)

        output, aux = model(image)

        router_params = list(model.pipeline.router.parameters())
        trainable_router = [p for p in router_params if p.requires_grad]

        if trainable_router:
            routed = aux.get("routed_tokens")
            if routed is not None and routed.numel() > 0:
                model.zero_grad()
                loss = routed.sum()
                loss.backward()
                grads = [
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in trainable_router
                ]
                assert any(grads), (
                    "No gradients flowed to DTRv2 router parameters!"
                )

    def test_sparse_prediction_contract(self):
        """Ada-SPM returns a valid SparsePrediction."""
        from adatile.core import SparsePrediction

        model = _build_model()
        model.eval()
        image = torch.randn(1, 3, 1024, 1024)

        # Access sparse predictor directly
        features = model.pipeline.backbone(image)
        spm_output = model.pipeline.sparse_predictor(features)

        # Handle conditional return type
        if isinstance(spm_output, tuple):
            spm_output = spm_output[0]

        assert isinstance(spm_output, SparsePrediction), (
            f"Ada-SPM must return SparsePrediction, got {type(spm_output)}"
        )
        assert spm_output.importance.shape[1] == 1
        assert spm_output.density.shape == spm_output.importance.shape

    def test_granularity_connected(self):
        """Granularity from Ada-SPM reaches the tokenizer."""
        model = _build_model()
        model.eval()
        image = torch.randn(1, 3, 1024, 1024)

        output, aux = model(image)

        # After the fix, the tokenizer receives granularity_hard
        # from Ada-SPM through the pipeline
        assert "granularity_hard" in aux, (
            "granularity_hard not found in pipeline aux output. "
            "The GranularityHead output is not being captured."
        )

    def test_planner_stats_nonzero(self):
        """PlannerStats FLOPs/memory estimates are non-zero by default."""
        from adatile.tokenizer.tile_planner import TilePlanner

        planner = TilePlanner(
            tile_sizes=[384, 768, 1536, 3072],
            importance_threshold=0.5,
            skip_mode="threshold",
        )
        imp = torch.full((8, 8), 0.1)  # All low → should skip most
        plan = planner.plan(imp, image_size=(1024, 1024))

        assert plan.planner_stats is not None
        # With defaults, skipped FLOPs should be non-zero when cells are skipped
        if plan.skipped_regions > 0:
            assert plan.planner_stats.estimated_flops_saved > 0, (
                "estimated_flops_saved is zero despite skipped cells. "
                "Default calibration values are broken."
            )
