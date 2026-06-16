"""Per-Tile FastSAM Encoder with native YOLOv8-Seg Head.

Restores the full YOLOv8-seg detection pipeline (backbone + neck + head)
that was previously bypassed via hook-based extraction.

Architecture:
    tile_image [B_t, 3, H_t, W_t]
        ↓
    YOLOv8 Backbone + Neck (FastSAM-x weights, frozen)
        ↓
    P3/P4/P5 features
        ↓
    YOLOv8-Seg Head (class + bbox + mask_coefficients)
        ↓
    Per-tile instances: {class, box(tile-relative), mask_coef}

Key difference from fastsam_hook.py:
  - Runs the FULL YOLOv8 model (not just hooking backbone layers)
  - Outputs detection results (cls, box, mask) not feature dicts
  - Operates on TILES, not full images
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PerTileFastSAM(nn.Module):
    """FastSAM running on individual tiles with native YOLOv8-seg head.

    Loads the same FastSAM-x.pt checkpoint, but keeps the full model
    (detection head included) instead of discarding it.

    Args:
        weight_file: Path to FastSAM-x.pt.
        conf_threshold: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.
        max_det: Maximum detections per tile.
        tile_size: Default tile size (used for coordinate normalization).
    """

    def __init__(
        self,
        weight_file: str = "FastSAM-x.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_det: int = 300,
    ):
        super().__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det

        # Load via ultralytics YOLO API (includes NMS post-processing)
        try:
            from ultralytics import YOLO
            self.model = YOLO(weight_file)
        except ImportError:
            # Fallback: raw checkpoint (no NMS — won't produce boxes)
            ckpt = torch.load(weight_file, map_location="cpu", weights_only=False)
            self.model = ckpt["model"].float()
            print("[PerTileFastSAM] WARNING: ultralytics not found, raw model loaded (no NMS)")

        # Access inner DetectionModel (bypass YOLO wrapper's train() trigger)
        self.inner = self.model.model if hasattr(self.model, 'model') else self.model
        self.inner.eval()
        for p in self.inner.parameters():
            p.requires_grad = False

    def forward(
        self, tile_images: Tensor
    ) -> List[Dict[str, Tensor]]:
        """Run FastSAM on batched tiles.

        Args:
            tile_images: [N_tiles, 3, H_t, W_t] batched tile images in [0, 1].

        Returns:
            List of N_tiles dicts, each with:
                'boxes':   [M, 4]   xyxy format, in tile-local coordinates
                'scores':  [M]      confidence scores
                'classes': [M]      class indices
                'masks':   [M, H, W] binary instance masks (tile-local)
        """
        if tile_images.max() <= 1.0:
            tile_images = tile_images * 255.0  # YOLO expects [0, 255]

        N = tile_images.shape[0]

        # ── Pad to multiple of 32 (YOLOv8 requirement) ──────
        _, _, H, W = tile_images.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            tile_images = F.pad(tile_images, (0, pad_w, 0, pad_h), mode="reflect")

        # ── Run full YOLOv8 inference (with NMS) ────────────
        # Convert tensor → numpy (YOLO.predict expects numpy or PIL)
        np_images = [
            (tile_images[i].permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            for i in range(N)
        ]

        with torch.no_grad():
            results = self.model.predict(
                source=np_images,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_det,
                verbose=False,
                stream=False,
            )

        # ── Parse Results objects (NMS already applied) ─────
        outputs = []
        for i in range(N):
            r = results[i]

            if r is None or r.boxes is None:
                outputs.append({
                    'boxes': torch.empty(0, 4, device=tile_images.device),
                    'scores': torch.empty(0, device=tile_images.device),
                    'classes': torch.empty(0, dtype=torch.long, device=tile_images.device),
                    'masks': torch.empty(0, H, W, device=tile_images.device),
                })
                continue

            # .predict() already applied NMS + confidence filtering
            boxes = r.boxes.xyxy       # [M, 4] in tile coordinates
            scores = r.boxes.conf      # [M]
            classes = r.boxes.cls.long() if r.boxes.cls is not None else torch.zeros(len(scores), dtype=torch.long)

            # Get masks if available
            if hasattr(r, 'masks') and r.masks is not None and r.masks.data is not None:
                masks_data = r.masks.data  # [M, H_t, W_t]
            else:
                masks_data = torch.empty(0, H, W, device=tile_images.device)

            outputs.append({
                'boxes': boxes,
                'scores': scores,
                'classes': classes,
                'masks': masks_data,
            })

        return outputs

    def extract_features(self, tile_images: Tensor) -> Dict[str, Tensor]:
        """Extract P3/P4/P5 features for prototype adapter use.

        Uses the internal YOLOv8 model (self.model.model) for hook-based extraction.
        """
        if tile_images.max() <= 1.0:
            tile_images = tile_images * 255.0

        # Access the internal DetectionModel via YOLO wrapper
        inner = self.model.model if hasattr(self.model, 'model') else self.model

        features = {}
        hooks = []
        children = list(inner.model.children())

        def make_hook(name):
            def hook(_, __, outp):
                if isinstance(outp, Tensor):
                    features[name] = outp
            return hook

        for name, idx in [("P3", 4), ("P4", 6), ("P5", 8)]:
            if idx < len(children):
                h = children[idx].register_forward_hook(make_hook(name))
                hooks.append(h)

        with torch.no_grad():
            _ = inner(tile_images)

        for h in hooks:
            h.remove()

        return features
