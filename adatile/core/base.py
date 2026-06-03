"""Core abstract interfaces for AdaTile-FastSAM.

All six core components are defined here as abstract base classes (ABCs).
Concrete implementations register themselves via the Registry pattern and
inherit from these interfaces.

Interfaces:
    - SparseImportancePredictor
    - DynamicTileTokenizer
    - BaseRouter
    - PrototypeMemory
    - GlobalContextBranch
    - SegmentationDecoder
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ── Data Structures ──────────────────────────────────────────────────


@dataclass
class TileInfo:
    """Metadata for a single tile."""

    tile_id: str
    image_id: str
    x1: int
    y1: int
    x2: int
    y2: int
    tile_size: int
    object_density: float = 0.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass
class RoutingOutput:
    """Standardized output from any token router.

    Fields:
        routed_tokens: [N_active, C] processed/attention-refined token features.
        assignments: [N_active, 1] integer level assignments (1=linear, 2=lowrank, 3=full).
        routing_weights: [N_active, 1] routing probability/magnitude weights.
        skipped_mask: [N_total] boolean mask — True if token was pruned (Level-0).
        aux_loss: optional scalar auxiliary loss (load-balance + entropy).
        stats: optional dict with per-level distribution, confidence, etc.
    """

    routed_tokens: Tensor
    assignments: Tensor
    routing_weights: Tensor
    skipped_mask: Tensor
    aux_loss: Optional[Tensor] = None
    stats: Optional[Dict[str, Any]] = None


@dataclass
class SegmentationOutput:
    """Standardized output from any segmentation decoder."""

    masks: Tensor  # [N_inst, H, W]  — instance masks
    scores: Tensor  # [N_inst]         — confidence scores
    boxes: Optional[Tensor] = None  # [N_inst, 4]       — bounding boxes
    classes: Optional[Tensor] = None  # [N_inst]           — class labels
    iou_preds: Optional[Tensor] = None  # [N_inst]        — predicted IoU
    protos: Optional[Tensor] = None  # [N_inst, C, H, W] — optional prototypes

    def to_coco(self, image_id: int) -> List[dict]:
        """Convert to COCO-format annotation dicts."""
        results = []
        for i in range(len(self.scores)):
            ann = {
                "image_id": image_id,
                "category_id": int(self.classes[i]) if self.classes is not None else 0,
                "score": float(self.scores[i]),
                "segmentation": [],  # RLE or polygon
                "bbox": self.boxes[i].tolist() if self.boxes is not None else [],
            }
            results.append(ann)
        return results


# ── Abstract Interfaces ──────────────────────────────────────────────


@dataclass(frozen=True)
class SparsePrediction:
    """Typed container for sparse importance prediction outputs.

    Replaces the ad-hoc Dict[str, Tensor] return from Ada-SPM forward()
    with a statically-typed, validated data structure.

    Fields:
        importance: [B, 1, H_s, W_s] float in [0, 1].
                    Combined importance score for routing/tiling decisions.
        density: [B, 1, H_s, W_s] float in [0, 1].
                 Raw predicted object density.
        granularity_soft: Optional [B, K, H_s, W_s] float.
                          Softmax probabilities over K tile-size categories.
        granularity_hard: Optional [B, 1, H_s, W_s] long.
                          Argmax tile-size index per spatial cell.
    """

    importance: Tensor
    density: Tensor
    granularity_soft: Optional[Tensor] = None
    granularity_hard: Optional[Tensor] = None

    def __post_init__(self):
        # importance: must be 4D [B, 1, H, W]
        if not isinstance(self.importance, Tensor):
            raise TypeError(f"importance must be a Tensor, got {type(self.importance).__name__}")
        if self.importance.dim() != 4:
            raise ValueError(f"importance must be 4D [B,1,H,W], got shape {tuple(self.importance.shape)}")
        if self.importance.shape[1] != 1:
            raise ValueError(f"importance channel dim must be 1, got {self.importance.shape[1]}")

        # density: must be 4D, same shape as importance
        if not isinstance(self.density, Tensor):
            raise TypeError(f"density must be a Tensor, got {type(self.density).__name__}")
        if self.density.dim() != 4:
            raise ValueError(f"density must be 4D [B,1,H,W], got shape {tuple(self.density.shape)}")
        if self.density.shape != self.importance.shape:
            raise ValueError(
                f"density shape {tuple(self.density.shape)} must match "
                f"importance shape {tuple(self.importance.shape)}"
            )

        # granularity_soft: optional 4D [B, K, H, W], batch-matched
        if self.granularity_soft is not None:
            if not isinstance(self.granularity_soft, Tensor):
                raise TypeError(f"granularity_soft must be a Tensor or None, got {type(self.granularity_soft).__name__}")
            if self.granularity_soft.dim() != 4:
                raise ValueError(f"granularity_soft must be 4D, got shape {tuple(self.granularity_soft.shape)}")
            if self.granularity_soft.shape[0] != self.importance.shape[0]:
                raise ValueError(
                    f"granularity_soft batch {self.granularity_soft.shape[0]} "
                    f"!= importance batch {self.importance.shape[0]}"
                )

        # granularity_hard: optional 4D [B, 1, H, W], channel=1, integer dtype
        if self.granularity_hard is not None:
            if not isinstance(self.granularity_hard, Tensor):
                raise TypeError(f"granularity_hard must be a Tensor or None, got {type(self.granularity_hard).__name__}")
            if self.granularity_hard.dim() != 4:
                raise ValueError(f"granularity_hard must be 4D, got shape {tuple(self.granularity_hard.shape)}")
            if self.granularity_hard.shape[1] != 1:
                raise ValueError(f"granularity_hard channel dim must be 1, got {self.granularity_hard.shape[1]}")
            if self.granularity_hard.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                raise TypeError(
                    f"granularity_hard must have integer dtype (long/int), "
                    f"got {self.granularity_hard.dtype}"
                )
            # Check values are in valid tile-size-index range
            max_val = self.granularity_hard.max().item()
            if max_val > 32:  # reasonable upper bound for tile size categories
                raise ValueError(
                    f"granularity_hard values out of range: max={max_val}, "
                    f"expected tile-size indices (0..K-1)"
                )

    def to_dict(self) -> Dict[str, Tensor]:
        """Convert to dict (backward compat during migration)."""
        d: Dict[str, Tensor] = {"importance": self.importance, "density": self.density}
        if self.granularity_soft is not None:
            d["granularity_soft"] = self.granularity_soft
        if self.granularity_hard is not None:
            d["granularity_hard"] = self.granularity_hard
        return d


class SparseImportancePredictor(nn.Module, ABC):
    """Predicts per-region importance scores for adaptive sparse tiling.

    Takes multi-scale backbone features and outputs a SparsePrediction
    with importance, density, and optional granularity tensors.

    Alias: Ada-SPM (Adaptive Spatial Partition Module).
    """

    @abstractmethod
    def forward(
        self, features: Dict[str, Tensor]
    ) -> "SparsePrediction":
        """Predict importance scores.

        Args:
            features: Multi-scale backbone features, e.g.
                {"c2": [B,C,H/4,W/4], "c3": [B,C,H/8,W/8], ...}

        Returns:
            SparsePrediction with importance, density, and optional granularity.
        """

    @abstractmethod
    def compute_loss(
        self,
        pred_importance: Tensor,
        target_density: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute density-supervised importance loss.

        Args:
            pred_importance: [B, 1, H, W] predicted importance.
            target_density: [B, 1, H, W] ground-truth instance density.
            valid_mask: optional validity mask.

        Returns:
            Scalar loss tensor.
        """


class DynamicTileTokenizer(nn.Module, ABC):
    """Dynamically partitions a high-resolution image into variable-size tiles.

    Uses importance predictions from SparseImportancePredictor to decide
    where and at what resolution to extract tokens.
    """

    @abstractmethod
    def forward(
        self,
        image: Tensor,
        features: Optional[Dict[str, Tensor]] = None,
        importance: Optional[Tensor] = None,
        granularity_hard: Optional[Tensor] = None,
    ) -> Tuple[List[TileInfo], Tensor]:
        """Partition image into tiles and extract token features.

        Args:
            image: [B, 3, H, W] input image.
            features: Optional pre-computed backbone features.
            importance: Optional [B, 1, H_s, W_s] importance map.
            granularity_hard: Optional [B, 1, H_s, W_s] learned tile-size index per cell.

        Returns:
            tiles: List of TileInfo for each generated tile.
            tokens: [B, N_tiles, C] token features for each tile.
        """

    @abstractmethod
    def prepare_cache(
        self,
        image: Tensor,
        tile_size: int,
        stride: int,
        save_dir: str,
    ) -> None:
        """Pre-compute and cache tile features for an image.

        Args:
            image: [3, H, W] single input image.
            tile_size: Tile side length.
            stride: Sliding window stride.
            save_dir: Output cache directory.
        """


class BaseRouter(nn.Module, ABC):
    """Routes tile tokens through adaptive processing paths.

    Canonical base class for all token routers in AdaTile-FastSAM.
    Implementations: DTRv2Router, UniformRouter, IdentityRouter.
    """

    @abstractmethod
    def forward(
        self,
        tokens: Tensor,
        metadata: Optional[Dict] = None,
    ) -> RoutingOutput:
        """Route and process tokens.

        Args:
            tokens: [B, N, C] tile token features.
            metadata: optional dict with keys like "importance" ([B,N]),
                      "prototypes" (dict of class_id→vector).

        Returns:
            RoutingOutput with processed tokens, assignments, weights, skip mask.
        """

    @abstractmethod
    def compute_load_balance_loss(self, routing_weights: Tensor) -> Tensor:
        """Compute auxiliary load-balancing loss.

        Args:
            routing_weights: [N_active, 1] routing probability weights.

        Returns:
            Scalar loss encouraging uniform level utilization.
        """


class PrototypeMemory(nn.Module, ABC):
    """Stores, retrieves, and updates class prototypes for few-shot segmentation.

    Computes class prototypes from support images and masks,
    then retrieves similarity scores for query features.
    """

    @abstractmethod
    def forward(
        self,
        support_features: Tensor,
        support_masks: Tensor,
        class_ids: Optional[List[int]] = None,
    ) -> Dict[int, Tensor]:
        """Generate class prototypes from support set.

        Args:
            support_features: [B_s, C, H, W] support image features.
            support_masks: [B_s, H, W] binary support masks.
            class_ids: Optional class label for each support sample.

        Returns:
            prototypes: class_id → prototype [C] (or [K, C] for multiple).
        """

    @abstractmethod
    def retrieve(
        self,
        query_features: Tensor,
        prototypes: Dict[int, Tensor],
        temperature: Optional[float] = None,
    ) -> Tensor:
        """Compute query-to-prototype similarity.

        Args:
            query_features: [B_q, C, H, W] query image features.
            prototypes: class_id → prototype from forward().
            temperature: Softmax temperature override.

        Returns:
            similarity: [B_q, N_classes, H, W] similarity maps.
        """

    @abstractmethod
    def update(
        self,
        prototypes: Dict[int, Tensor],
        cache_path: Optional[str] = None,
    ) -> None:
        """Persist or update prototype cache.

        Args:
            prototypes: Updated class prototypes.
            cache_path: Where to persist (uses configured path if None).
        """


class GlobalContextBranch(nn.Module, ABC):
    """Extracts global scene context to complement local tile features.

    Processes a downsampled view of the full image and injects
    global context embeddings into each tile's feature stream.
    """

    @abstractmethod
    def forward(
        self,
        image: Tensor,
        features: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Extract global context.

        Args:
            image: [B, 3, H, W] input image (may be resized/downsampled).
            features: Optional backbone features.

        Returns:
            global_embed: [B, C_g] global context embedding.
            global_features: Multi-scale global feature maps.
        """

    @abstractmethod
    def fuse(
        self,
        tile_features: Tensor,
        global_embed: Tensor,
    ) -> Tensor:
        """Fuse global context into tile features.

        Args:
            tile_features: [B, N, C] local tile token features.
            global_embed: [B, C_g] global context embedding.

        Returns:
            fused: [B, N, C] context-augmented tile features.
        """


class SegmentationDecoder(nn.Module, ABC):
    """Decodes per-tile features into final instance masks.

    Combines tile-level predictions into a unified full-image output
    via non-maximum suppression and tile-boundary merging.
    """

    @abstractmethod
    def forward(
        self,
        tile_features: Tensor,
        tile_infos: List[List[TileInfo]],
        prototypes: Optional[Dict[int, Tensor]] = None,
        global_features: Optional[Dict[str, Tensor]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        skipped_indices: Optional[Tensor] = None,
    ) -> SegmentationOutput:
        """Decode tile features to instance masks.

        Args:
            tile_features: [B, N_tiles, C] features from tokenizer + router.
            tile_infos: Per-batch list of TileInfo for coordinate mapping.
            prototypes: Optional class prototypes for few-shot.
            global_features: Optional global context features.
            image_size: (H, W) of original image for output sizing.
            skipped_indices: Optional indices of tokens that were skipped
                by the router, so the decoder can filter tile_infos accordingly.

        Returns:
            SegmentationOutput with masks, scores, boxes, classes.
        """

    @abstractmethod
    def merge_tiles(
        self,
        tile_predictions: List[SegmentationOutput],
        tile_infos: List[TileInfo],
        image_size: Tuple[int, int],
        iou_threshold: float = 0.6,
    ) -> SegmentationOutput:
        """Merge overlapping tile predictions with tile-boundary NMS.

        Args:
            tile_predictions: Per-tile segmentation outputs.
            tile_infos: Tile metadata for coordinate mapping.
            image_size: (H, W) full image size.
            iou_threshold: NMS IoU threshold.

        Returns:
            Merged full-image SegmentationOutput.
        """


class LossFunction(nn.Module, ABC):
    """Abstract base class for loss functions.

    All loss functions registered in the LOSS registry must inherit
    from this class and implement forward(). This ensures consistent
    interface: forward(predictions, targets, **kwargs) -> Tensor.
    """

    @abstractmethod
    def forward(self, predictions: Tensor, targets: Tensor, **kwargs) -> Tensor:
        """Compute loss.

        Args:
            predictions: Model output tensor.
            targets: Ground truth tensor.
            **kwargs: Additional loss-specific parameters.

        Returns:
            Scalar loss tensor.
        """
