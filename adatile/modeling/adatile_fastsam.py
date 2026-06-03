"""AdaTile-FastSAM main model builder.

Constructs a full pipeline from config:
    Config → individual modules → AdaTileFastSAMPipeline
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch.nn as nn

from adatile.config import Config
from adatile.backbone import build_backbone
from adatile.sparse import build_sparse
from adatile.tokenizer import build_tokenizer
from adatile.routing import build_router
from adatile.decoder import build_decoder
from adatile.prototype import build_prototype
from adatile.segmentation.base import AdaTileFastSAMPipeline
from adatile.core import SegmentationOutput


class AdaTileFastSAM(nn.Module):
    """High-level wrapper around the AdaTile-FastSAM pipeline.

    Provides a clean API for training and inference with automatic
    pipeline construction from a Config object.

    Usage:
        cfg = Config.from_yaml("configs/default.py")
        model = AdaTileFastSAM(cfg)
        model = AdaTileFastSAM.from_config(cfg)

        # Training
        output, aux = model(images)

        # Few-shot
        output, aux = model(query_images, support_images, support_masks)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pipeline = self._build_pipeline(cfg)

    def _build_pipeline(self, cfg: Config) -> AdaTileFastSAMPipeline:
        """Assemble pipeline from config."""
        backbone = build_backbone(cfg.backbone.name, **self._backbone_kwargs(cfg))
        sparse_pred = build_sparse(cfg.sparse.name, **self._sparse_kwargs(cfg))
        tokenizer = build_tokenizer(cfg.tokenizer.name, **self._tokenizer_kwargs(cfg))
        router = build_router(cfg.router.name, **self._router_kwargs(cfg))
        decoder = build_decoder(cfg.decoder.name, **self._decoder_kwargs(cfg))

        prototype_memory = None
        if cfg.prototype.name:
            prototype_memory = build_prototype(
                cfg.prototype.name, **self._prototype_kwargs(cfg)
            )

        return AdaTileFastSAMPipeline(
            backbone=backbone,
            sparse_predictor=sparse_pred,
            tokenizer=tokenizer,
            router=router,
            decoder=decoder,
            prototype_memory=prototype_memory,
            global_context=None,
        )

    def _backbone_kwargs(self, cfg: Config) -> Dict[str, Any]:
        b = cfg.backbone
        return {
            "embed_dim": b.embed_dim,
            "depth": b.depth,
            "num_heads": b.num_heads,
            "patch_size": b.patch_size,
            "out_scales": tuple(b.output_scales),
        }

    def _sparse_kwargs(self, cfg: Config) -> Dict[str, Any]:
        s = cfg.sparse
        return {
            "num_tile_sizes": getattr(s, "num_scales", 4),
            "fusion_dim": 256,
            "hidden_dim": 128,
            "importance_threshold": getattr(s, "importance_threshold", 0.15),
            "use_transformer": getattr(s, "use_transformer", True),
            "dropout": getattr(s, "dropout", 0.1),
        }

    def _tokenizer_kwargs(self, cfg: Config) -> Dict[str, Any]:
        t = cfg.tokenizer
        return {
            "tile_sizes": t.tile_sizes,
            "strides": t.stride_ratios,
            "max_tokens": t.max_tokens_per_image,
            "skip_mode": t.skip_mode,
            "hard_skip_multiplier": t.hard_skip_multiplier,
            "importance_threshold": cfg.sparse.importance_threshold,
        }

    def _router_kwargs(self, cfg: Config) -> Dict[str, Any]:
        r = cfg.router
        return {
            "embed_dim": r.embed_dim,
            "max_full_ratio": r.max_full_ratio,
            "max_skip_ratio": r.max_skip_ratio,
            "min_linear_ratio": r.min_linear_ratio,
            "load_balance_weight": r.aux_loss_weight,
        }

    def _decoder_kwargs(self, cfg: Config) -> Dict[str, Any]:
        d = cfg.decoder
        return {
            "mask_dim": d.mask_dim,
            "num_mask_tokens": d.num_mask_tokens,
            "iou_prediction": d.iou_prediction,
        }

    def _prototype_kwargs(self, cfg: Config) -> Dict[str, Any]:
        p = cfg.prototype
        return {
            "prototype_dim": p.prototype_dim,
            "temperature": p.temperature,
        }

    def forward(
        self,
        image,
        support_images=None,
        support_masks=None,
        class_ids=None,
    ):
        return self.pipeline(
            image,
            support_images=support_images,
            support_masks=support_masks,
            class_ids=class_ids,
        )

    @classmethod
    def from_config(cls, cfg: Config) -> AdaTileFastSAM:
        """Create model from config."""
        return cls(cfg)


def build_adatile_fastsam(cfg: Config) -> AdaTileFastSAM:
    """Build AdaTile-FastSAM model from configuration.

    This is the primary entry point for model construction.
    """
    return AdaTileFastSAM.from_config(cfg)
