"""Global thumbnail context branch and tile merging utilities.

Components:
    - GlobalThumbnailBranch: heavily downsampled full-image context extraction
    - TileMerger: overlap-aware merging with soft blending and NMS dedup
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from adatile.core import GlobalContextBranch, TileInfo


# ── Global Thumbnail Branch ────────────────────────────────────────────


class GlobalThumbnailBranch(GlobalContextBranch):
    """Extracts global scene context from a heavily downsampled full image.

    NOTE: This class is NOT registered in the TOKENIZER registry because
    it implements GlobalContextBranch, not DynamicTileTokenizer. It is
    instantiated directly by DynamicTileTokenizerImpl when
    use_global_branch=True.

    Architecture:
        Downsample → Lightweight ConvNet → Global AvgPool → embed
        + Multi-scale feature pyramid for cross-scale fusion.

    The global context complements local tile features by providing:
        - Scene-level semantics (indoor/outdoor, scale priors)
        - Long-range spatial relationships between tiles
        - Background class context

    Args:
        thumbnail_size: Full image is resized to this size before encoding.
        in_channels: Input channels (3 for RGB).
        embed_dim: Output global embedding dimension.
        hidden_dims: Channel dimensions for each conv stage.
        out_dims: Multi-scale output channel dimensions.
    """

    def __init__(
        self,
        thumbnail_size: int = 512,
        in_channels: int = 3,
        embed_dim: int = 256,
        hidden_dims: Tuple[int, ...] = (64, 128, 256, 512),
        out_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.thumbnail_size = thumbnail_size
        self.embed_dim = embed_dim

        # Lightweight conv backbone for the thumbnail
        dims = (in_channels,) + hidden_dims
        stages = []
        for i in range(len(dims) - 1):
            stages.extend([
                nn.Conv2d(dims[i], dims[i + 1], 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(dims[i + 1]),
                nn.ReLU(inplace=True),
            ])
        self.backbone = nn.Sequential(*stages)

        # Multi-scale feature dimensions
        self.out_dims = out_dims or hidden_dims

        # Global pooling → embedding
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_dims[-1], embed_dim)

        # Cross-attention fusion: tile features attend to global context
        self.fuse_norm = nn.LayerNorm(embed_dim)
        self.fuse_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            batch_first=True,
        )
        self.fuse_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.fuse_ffn_norm = nn.LayerNorm(embed_dim)

        self._last_global_features: Dict[str, Tensor] = {}

    def forward(
        self,
        image: Tensor,
        features: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Extract global context from downsampled full image.

        Args:
            image: [B, 3, H, W] input image at full resolution.
            features: Optional backbone features (unused, but accepted for API compat).

        Returns:
            global_embed: [B, embed_dim] global context vector.
            global_features: Multi-scale feature maps from the thumbnail.
        """
        B, _, H, W = image.shape

        # Downsample to thumbnail
        thumbnail = F.interpolate(
            image, size=(self.thumbnail_size, self.thumbnail_size),
            mode="bilinear", align_corners=False,
        )

        # Extract features stage by stage
        x = thumbnail
        global_features = {}
        stage_sizes = [self.thumbnail_size]
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            _, _, h, w = x.shape
            if h != stage_sizes[-1]:
                stage_sizes.append(h)

        # The final output is the backbone's last feature map
        global_embed = self.global_pool(x).view(B, -1)  # [B, hidden_dim]
        global_embed = self.proj(global_embed)           # [B, embed_dim]

        # Store multi-scale features (last stage)
        global_features["global_p5"] = x  # highest semantic level

        self._last_global_features = global_features
        return global_embed, global_features

    def fuse(
        self,
        tile_features: Tensor,
        global_embed: Tensor,
    ) -> Tensor:
        """Fuse global context into tile features via cross-attention.

        Each tile feature attends to the global context embedding,
        then passes through a small FFN with residual connection.

        Args:
            tile_features: [B, N, C] local tile token features.
            global_embed: [B, C] global context embedding.

        Returns:
            fused: [B, N, C] context-augmented tile features.
        """
        # Cross-attention: tile features query global context
        g = global_embed.unsqueeze(1)  # [B, 1, C] as key/value
        t = self.fuse_norm(tile_features)
        attended, _ = self.fuse_cross_attn(
            query=t, key=g, value=g,
        )
        t2 = tile_features + attended
        t3 = t2 + self.fuse_ffn(self.fuse_ffn_norm(t2))
        return t3


# ── Tile Merging ───────────────────────────────────────────────────────


class TileMerger(nn.Module):
    """Overlap-aware tile merging with soft blending and NMS deduplication.

    Handles the common case where overlapping tiles produce duplicate
    instance predictions near tile boundaries.

    Strategy:
        1. Map each tile's predictions to full-image coordinates.
        2. Apply distance-weighted soft blending in overlap regions.
        3. Run class-aware NMS to remove duplicate instances.
        4. (Optional) Boundary artifact suppression.

    Args:
        iou_threshold: NMS IoU threshold for deduplication.
        blend_sigma: Soft blending falloff sigma (in pixels).
        score_threshold: Minimum score to keep.
        boundary_margin: Pixels near tile edges to suppress.
    """

    def __init__(
        self,
        iou_threshold: float = 0.6,
        blend_sigma: float = 32.0,
        score_threshold: float = 0.05,
        boundary_margin: int = 16,
    ):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.blend_sigma = blend_sigma
        self.score_threshold = score_threshold
        self.boundary_margin = boundary_margin

    def forward(
        self,
        tile_masks: List[Tensor],
        tile_scores: List[Tensor],
        tile_boxes: List[Optional[Tensor]],
        tile_classes: List[Optional[Tensor]],
        tile_infos: List[TileInfo],
        image_size: Tuple[int, int],
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """Merge per-tile instance predictions into full-image output.

        Args:
            tile_masks: List of [N_i, H_t, W_t] per-tile instance masks.
            tile_scores: List of [N_i] confidence scores.
            tile_boxes: List of [N_i, 4] per-tile bboxes (x1,y1,x2,y2).
            tile_classes: List of [N_i] class indices.
            tile_infos: Tile metadata for coordinate mapping.
            image_size: (H, W) of the full image.

        Returns:
            masks: [N_final, H, W] full-image instance masks.
            scores: [N_final] confidence scores.
            boxes: [N_final, 4] or None.
            classes: [N_final] or None.
        """
        H, W = image_size
        device = tile_masks[0].device if tile_masks else torch.device("cpu")
        if not tile_masks:
            return (
                torch.empty(0, H, W, device=device),
                torch.empty(0, device=device),
                None,
                None,
            )

        all_masks = []
        all_scores = []
        all_boxes = []
        all_classes = []

        for i, (masks, scores, boxes, classes, info) in enumerate(
            zip(tile_masks, tile_scores, tile_boxes, tile_classes, tile_infos)
        ):
            if masks.shape[0] == 0:
                continue

            N_i, H_t, W_t = masks.shape
            # Resize each instance mask to full image and place at tile position
            for j in range(N_i):
                full_mask = torch.zeros(H, W, device=device, dtype=masks.dtype)
                y1, y2 = info.y1, info.y2
                x1, x2 = info.x1, info.x2

                # Resize mask to tile's image footprint
                mask_j = masks[j:j + 1, :, :]  # [1, H_t, W_t]
                mask_resized = F.interpolate(
                    mask_j.unsqueeze(0),
                    size=(y2 - y1, x2 - x1),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)  # [H_roi, W_roi]

                # Apply boundary soft suppression
                mask_resized = self._suppress_boundary(
                    mask_resized, y2 - y1, x2 - x1,
                )

                full_mask[y1:y2, x1:x2] = mask_resized
                all_masks.append(full_mask)
                all_scores.append(scores[j])

                if boxes is not None:
                    b = boxes[j].clone()
                    # Shift box from tile-local to full-image coords
                    b[0] += info.x1
                    b[1] += info.y1
                    b[2] += info.x1
                    b[3] += info.y1
                    all_boxes.append(b)
                if classes is not None:
                    all_classes.append(classes[j])

        if not all_masks:
            return (
                torch.empty(0, H, W, device=device),
                torch.empty(0, device=device),
                None,
                None,
            )

        masks = torch.stack(all_masks, dim=0)  # [M, H, W]
        scores = torch.stack(all_scores, dim=0)  # [M]
        boxes_t = torch.stack(all_boxes, dim=0) if all_boxes else None
        classes_t = torch.stack(all_classes, dim=0) if all_classes else None

        # Soft-blend overlap regions
        masks = self._soft_blend(masks)

        # Apply NMS deduplication
        keep = self._class_aware_nms(masks, scores, boxes_t, classes_t)
        masks = masks[keep]
        scores = scores[keep]
        if boxes_t is not None:
            boxes_t = boxes_t[keep]
        if classes_t is not None:
            classes_t = classes_t[keep]

        # Filter by score
        score_keep = scores > self.score_threshold
        masks = masks[score_keep]
        scores = scores[score_keep]
        if boxes_t is not None:
            boxes_t = boxes_t[score_keep]
        if classes_t is not None:
            classes_t = classes_t[score_keep]

        return masks, scores, boxes_t, classes_t

    def _suppress_boundary(
        self, mask: Tensor, h: int, w: int,
    ) -> Tensor:
        """Attenuate mask values near tile edges to reduce boundary artifacts."""
        if self.boundary_margin <= 0:
            return mask

        m = self.boundary_margin
        # Create spatial weight map: 1 in center, fading to 0 at edges
        wy = torch.ones(h, device=mask.device)
        wx = torch.ones(w, device=mask.device)

        if h > 2 * m:
            wy[:m] = torch.linspace(0, 1, m, device=mask.device)
            wy[-m:] = torch.linspace(1, 0, m, device=mask.device)
        if w > 2 * m:
            wx[:m] = torch.linspace(0, 1, m, device=mask.device)
            wx[-m:] = torch.linspace(1, 0, m, device=mask.device)

        weight = wy.unsqueeze(1) * wx.unsqueeze(0)  # [H, W]
        return mask * weight

    def _soft_blend(self, masks: Tensor) -> Tensor:
        """Soft blending: weight each pixel by a Gaussian centered on the mask.

        For overlapping masks, reduces double-counting by attenuating
        pixels that are also covered by other masks.
        """
        if masks.shape[0] <= 1:
            return masks

        # Per-pixel: max-pool across masks for hard merging, or mean for soft
        # Research code: use element-wise max as the simplest soft merge
        # More advanced: pairwise IoU-weighted blending
        overlap = (masks.sum(dim=0, keepdim=True) > 1.0).float()
        scale = 1.0 / (1.0 + overlap * 0.5)  # attenuate where >1 mask active

        return masks * scale

    def _class_aware_nms(
        self,
        masks: Tensor,
        scores: Tensor,
        boxes: Optional[Tensor],
        classes: Optional[Tensor],
    ) -> Tensor:
        """Class-aware NMS using mask IoU.

        Args:
            masks: [N, H, W] binary/logit masks.
            scores: [N] confidence scores.
            boxes: Optional [N, 4] bboxes for fast pre-filter.
            classes: Optional [N] class indices.

        Returns:
            keep: Boolean indices of kept instances.
        """
        N = masks.shape[0]
        if N <= 1:
            return torch.ones(N, dtype=torch.bool, device=masks.device)

        # Sort by score descending
        order = scores.argsort(descending=True)
        masks_sorted = masks[order]
        classes_sorted = classes[order] if classes is not None else None

        keep = torch.ones(N, dtype=torch.bool, device=masks.device)

        for i in range(N):
            if not keep[order[i]]:
                continue
            for j in range(i + 1, N):
                if not keep[order[j]]:
                    continue
                # Skip if different classes
                if classes_sorted is not None and classes_sorted[i] != classes_sorted[j]:
                    continue

                # Compute mask IoU efficiently via downsample
                iou = self._mask_iou(masks_sorted[i], masks_sorted[j])
                if iou > self.iou_threshold:
                    keep[order[j]] = False

        return keep

    def _mask_iou(self, mask_a: Tensor, mask_b: Tensor) -> float:
        """Compute IoU between two full-image masks efficiently.

        Uses downsampled masks for speed on large images.
        """
        # Downsample for efficiency on large images
        max_dim = max(mask_a.shape[-2], mask_a.shape[-1])
        if max_dim > 1024:
            scale = 1024 / max_dim
            new_h = int(mask_a.shape[-2] * scale)
            new_w = int(mask_a.shape[-1] * scale)
            a = F.interpolate(
                mask_a.unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w), mode="bilinear", align_corners=False,
            ).squeeze()
            b = F.interpolate(
                mask_b.unsqueeze(0).unsqueeze(0),
                size=(new_h, new_w), mode="bilinear", align_corners=False,
            ).squeeze()
        else:
            a, b = mask_a, mask_b

        a_bin = (a > 0.5).float()
        b_bin = (b > 0.5).float()
        intersection = (a_bin * b_bin).sum()
        union = (a_bin + b_bin).clamp(0, 1).sum()

        if union == 0:
            return 0.0
        return float(intersection / union)


# ── Overlap Blend Kernel ────────────────────────────────────────────────


def gaussian_blend_kernel(
    tile_size: int,
    sigma: float = 0.125,
) -> Tensor:
    """Generate a 2D Gaussian blend weight for a square tile.

    Used to smoothly fade tile boundaries when stitching predictions.

    Args:
        tile_size: Tile side length in pixels.
        sigma: Gaussian sigma as fraction of tile_size.

    Returns:
        weight: [tile_size, tile_size] blend weights in [0, 1].
    """
    sigma_px = tile_size * sigma
    coords = torch.linspace(-1, 1, tile_size)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    d2 = x ** 2 + y ** 2
    weight = torch.exp(-d2 / (2 * (2 * sigma_px / tile_size) ** 2))
    return weight


def soft_merge_overlap(
    predictions: List[Tensor],
    tile_infos: List[TileInfo],
    image_size: Tuple[int, int],
    sigma: float = 0.125,
) -> Tensor:
    """Merge tile predictions into a unified full-image output with soft blending.

    Each prediction is weighted by a Gaussian centered on its tile,
    producing smooth transitions at tile boundaries.

    Args:
        predictions: List of [C, H_t, W_t] per-tile feature/logit maps.
        tile_infos: Tile metadata (positions in full image).
        image_size: (H, W) full image size.
        sigma: Blend sigma as fraction of tile_size.

    Returns:
        merged: [C, H, W] blended full-image output.
    """
    if not predictions:
        raise ValueError("No predictions to merge")

    C = predictions[0].shape[0]
    H, W = image_size
    device = predictions[0].device

    accumulator = torch.zeros(C, H, W, device=device)
    weight_sum = torch.zeros(H, W, device=device)

    for pred, info in zip(predictions, tile_infos):
        _, H_t, W_t = pred.shape
        y1, y2 = info.y1, info.y2
        x1, x2 = info.x1, info.x2

        # Resize prediction to tile footprint
        pred_resized = F.interpolate(
            pred.unsqueeze(0),
            size=(y2 - y1, x2 - x1),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        # Generate blend weight
        blend = gaussian_blend_kernel(
            max(y2 - y1, x2 - x1),
            sigma=sigma,
        ).to(device)
        blend = blend[:y2 - y1, :x2 - x1]  # crop to actual size
        if blend.dim() == 2:
            blend = blend.unsqueeze(0)  # [1, H_roi, W_roi]

        accumulator[:, y1:y2, x1:x2] += pred_resized * blend
        weight_sum[y1:y2, x1:x2] += blend.squeeze(0)

    # Normalize by weight sum
    weight_sum = weight_sum.clamp(min=1e-8)
    return accumulator / weight_sum.unsqueeze(0)
