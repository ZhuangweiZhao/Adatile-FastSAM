"""Segmentation decoder — Sparse tile-level mask generation.

CRITICAL: Masks are NEVER materialized at full-image resolution.
Each instance is stored as (crop_mask, bbox, score) — a sparse
representation where the mask lives inside the tile ROI only.

Memory: O(Σ h_i·w_i) for instance ROIs, NOT O(N_inst · H_img · W_img).

This is the key efficiency innovation — it's what makes AdaTile-FastSAM
truly adaptive and sparse.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms

from adatile.core import SegmentationDecoder, SegmentationOutput, TileInfo
from adatile.registry import DECODER


# ── Sparse Instance Representation ────────────────────────────────────


class TileProtoModule(nn.Module):
    """Per-tile mask prototype generator (operates inside tile only)."""

    def __init__(self, in_channels: int = 256, proto_dim: int = 32):
        super().__init__()
        self.proto_dim = proto_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, proto_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, target_h: int, target_w: int) -> Tensor:
        x = self.conv(x)
        if x.shape[-2:] != (target_h, target_w):
            x = F.interpolate(x, size=(target_h, target_w),
                              mode='bilinear', align_corners=False)
        return x.squeeze(0)


@DECODER.register()
class FastSAMDecoder(SegmentationDecoder):
    """Sparse tile-level segmentation decoder.

    Outputs a SPARSE instance list — masks are crop-level, not full-image.
    `SegmentationOutput.masks` is filled on-demand via `materialize_masks()`.
    """

    def __init__(
        self,
        in_channels: int = 256,
        proto_dim: int = 32,
        fpn_channels: int = 256,
        score_threshold: float = 0.3,
        nms_iou_threshold: float = 0.6,
        max_instances: int = 100,
        mask_dim: int = 256,
        num_mask_tokens: int = 4,
        iou_prediction: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.proto_dim = proto_dim
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_instances = max_instances

        self.tile_proto = TileProtoModule(in_channels, proto_dim)
        self.det_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 5, 1),
        )
        self.coeff_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, proto_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        tile_features: Tensor,
        tile_infos: List[List[TileInfo]],
        prototypes: Optional[Dict[int, Tensor]] = None,
        global_features: Optional[Dict[str, Tensor]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        skipped_indices: Optional[Tensor] = None,
    ) -> SegmentationOutput:
        """Decode masks PER TILE, return SPARSE instance list.

        Output masks are CROP-LEVEL (ROI-sized), NOT full-image.
        Use materialize_masks() if full-image masks are needed for eval.
        """
        if tile_features.dim() == 3:
            tokens = tile_features.reshape(-1, tile_features.shape[-1])
        else:
            tokens = tile_features

        if isinstance(tile_infos, list) and len(tile_infos) > 0:
            if isinstance(tile_infos[0], list):
                infos = [info for batch in tile_infos for info in batch]
            else:
                infos = tile_infos
        else:
            infos = []

        H_img, W_img = image_size or (1024, 1024)

        if len(infos) == 0 or tokens.numel() == 0:
            return SegmentationOutput(
                masks=torch.zeros(0, 64, 64, device=tokens.device),
                scores=torch.zeros(0, device=tokens.device),
                boxes=torch.empty(0, 4, device=tokens.device),
            )

        # ── Per-tile decode → sparse instance candidates ──────────
        all_boxes = []
        all_scores = []
        all_mask_patches = []  # [(tensor_hw, y1_img, x1_img), ...]

        for i, info in enumerate(infos):
            if i >= tokens.shape[0]:
                break

            token_i = tokens[i]
            tw = info.x2 - info.x1
            th = info.y2 - info.y1
            grid_h = max(th // 32, 4)
            grid_w = max(tw // 32, 4)

            feat_map = token_i.view(1, self.in_channels, 1, 1).expand(
                1, self.in_channels, grid_h, grid_w
            )
            proto_tile = self.tile_proto(feat_map, th // 4, tw // 4)
            det = self.det_head(feat_map).sigmoid()
            scores_map = det[0, 0]

            high_conf = (scores_map > self.score_threshold).nonzero(as_tuple=False)
            n_det = min(high_conf.shape[0], 4)

            for j in range(n_det):
                gy, gx = high_conf[j].tolist()
                score = scores_map[gy, gx].item()
                bx = det[0, 1:, gy, gx]

                cell_x = gx / grid_w
                cell_y = gy / grid_h
                cell_w = 1.0 / grid_w
                cell_h = 1.0 / grid_h

                x1_img = info.x1 + (cell_x + bx[0].item() * cell_w) * tw
                y1_img = info.y1 + (cell_y + bx[1].item() * cell_h) * th
                x2_img = info.x1 + (cell_x + bx[2].item() * cell_w) * tw
                y2_img = info.y1 + (cell_y + bx[3].item() * cell_h) * th

                coeff = self.coeff_head(feat_map)
                mask_tile = (coeff.view(self.proto_dim, 1, 1) *
                             proto_tile).sum(dim=0).sigmoid()
                mask_bin = (mask_tile > 0.5).float()  # [th/4, tw/4]

                all_mask_patches.append((
                    mask_bin,
                    int(info.y1 // 4), int(info.x1 // 4),  # placement origin
                ))
                all_scores.append(score)
                all_boxes.append(torch.tensor(
                    [x1_img, y1_img, x2_img, y2_img],
                    device=tokens.device, dtype=torch.float32,
                ))

        if not all_mask_patches:
            return SegmentationOutput(
                masks=torch.zeros(0, 64, 64, device=tokens.device),
                scores=torch.zeros(0, device=tokens.device),
                boxes=torch.empty(0, 4, device=tokens.device),
            )

        # ── NMS on bboxes (NO full-image mask materialization) ────
        boxes = torch.stack(all_boxes, dim=0)
        scores_t = torch.tensor(all_scores, device=tokens.device)
        cls = torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
        keep = batched_nms(boxes, scores_t, cls, self.nms_iou_threshold)

        if keep.shape[0] > self.max_instances:
            _, topk = scores_t[keep].topk(self.max_instances)
            keep = keep[topk]

        # ── Build sparse masks: crop-level, NOT full-image ────────
        # Each mask is the ROI-cropped binary tensor.
        # For COCO eval / loss, use materialize_masks() to expand.
        sparse_masks = []
        sparse_boxes = []
        sparse_scores = []

        for idx in keep.tolist():
            patch, y1, x1 = all_mask_patches[idx]
            h_p, w_p = patch.shape
            y2 = y1 + h_p
            x2 = x1 + w_p

            # Store mask at its NATIVE crop resolution
            # Pad to max crop size for batch stacking compatibility
            max_crop = 256
            if h_p > max_crop or w_p > max_crop:
                patch = F.interpolate(
                    patch.unsqueeze(0).unsqueeze(0),
                    size=(min(h_p, max_crop), min(w_p, max_crop)),
                    mode='nearest',
                ).squeeze()

            sparse_masks.append(patch)
            sparse_boxes.append(boxes[idx])
            sparse_scores.append(scores_t[idx])

        # Stack as variable-size masks (padded to max)
        max_h = max(m.shape[0] for m in sparse_masks)
        max_w = max(m.shape[1] for m in sparse_masks)
        masks_padded = torch.zeros(
            len(sparse_masks), max_h, max_w,
            device=tokens.device, dtype=torch.float32,
        )
        for i, m in enumerate(sparse_masks):
            masks_padded[i, :m.shape[0], :m.shape[1]] = m

        return SegmentationOutput(
            masks=masks_padded,  # [N, max_h, max_w] crop-level masks
            scores=torch.stack(sparse_scores, dim=0) if sparse_scores else torch.zeros(0, device=tokens.device),
            boxes=torch.stack(sparse_boxes, dim=0) if sparse_boxes else torch.empty(0, 4, device=tokens.device),
            classes=None,
        )

    def merge_tiles(
        self,
        tile_predictions: List[SegmentationOutput],
        tile_infos: List[TileInfo],
        image_size: Tuple[int, int],
        iou_threshold: float = 0.6,
    ) -> SegmentationOutput:
        """Merge via NMS (crop-level masks remain sparse)."""
        all_masks, all_scores, all_boxes = [], [], []
        for pred in tile_predictions:
            if pred.masks.shape[0] > 0:
                all_masks.append(pred.masks)
                all_scores.append(pred.scores)
                if pred.boxes is not None:
                    all_boxes.append(pred.boxes)

        if not all_masks:
            return SegmentationOutput(
                masks=torch.empty(0, 64, 64), scores=torch.empty(0),
            )

        masks = torch.cat(all_masks, dim=0)
        scores = torch.cat(all_scores, dim=0)
        boxes = torch.cat(all_boxes, dim=0) if all_boxes else None

        if boxes is not None and boxes.shape[0] > 0:
            cls = torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
            keep = batched_nms(boxes, scores, cls, iou_threshold)
            masks = masks[keep]
            scores = scores[keep]
            boxes = boxes[keep]

        return SegmentationOutput(masks=masks, scores=scores, boxes=boxes)
