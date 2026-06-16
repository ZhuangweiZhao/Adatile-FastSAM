"""Tile Instance Fusion — SAHI-style instance merging with Mask IoU.

Problem: A single instance spanning two adjacent tiles is detected as
two separate instances. Standard NMS (box IoU only) cannot correctly
merge their masks.

Solution: Mask IoU Matching + Weighted Mask Averaging + coordinate
mapping from tile-local → full-image coordinates.

Reference: SAHI (Akyon et al., 2022) — Slicing Aided Hyper Inference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from adatile.sparse.tile_router import TileSpec


def tile_to_full_coordinates(
    tile_box: Tensor,       # [M, 4] xyxy in tile-local coordinates
    tile_spec: TileSpec,    # tile position and size
    image_size: Tuple[int, int],  # (H, W) full image
) -> Tensor:
    """Convert tile-local boxes to full-image coordinates.

    Args:
        tile_box: [M, 4] boxes in tile coordinate system.
        tile_spec: Tile specification with x,y (SPM grid) and size (pixels).
        image_size: (H_img, W_img) of the original image.

    Returns:
        [M, 4] boxes in full-image xyxy coordinates.
    """
    H_img, W_img = image_size

    # SPM cell → pixel center
    cx = tile_spec.x * 32 * W_img // max(W_img, 1)
    cy = tile_spec.y * 32 * H_img // max(H_img, 1)

    half = tile_spec.size // 2
    tile_x1 = max(0, cx - half)
    tile_y1 = max(0, cy - half)

    # Offset tile-local boxes to full-image coordinates
    full_box = tile_box.clone()
    full_box[:, 0] += tile_x1  # x1
    full_box[:, 1] += tile_y1  # y1
    full_box[:, 2] += tile_x1  # x2
    full_box[:, 3] += tile_y1  # y2

    # Clamp to image bounds
    full_box[:, 0].clamp_(0, W_img)
    full_box[:, 1].clamp_(0, H_img)
    full_box[:, 2].clamp_(0, W_img)
    full_box[:, 3].clamp_(0, H_img)

    return full_box


def mask_iou(mask_a: Tensor, mask_b: Tensor) -> float:
    """Compute IoU between two binary masks.

    Args:
        mask_a: [H, W] binary.
        mask_b: [H, W] binary.

    Returns:
        IoU in [0, 1].
    """
    # Resize to common size if needed
    if mask_a.shape != mask_b.shape:
        mask_b = F.interpolate(
            mask_b.float().unsqueeze(0).unsqueeze(0),
            size=mask_a.shape, mode="nearest",
        ).squeeze(0).squeeze(0)

    inter = (mask_a.bool() & mask_b.bool()).sum().float()
    union = (mask_a.bool() | mask_b.bool()).sum().float()
    return float((inter / (union + 1e-8)).item())


def merge_instances(
    instances_a: Dict[str, Tensor],  # one instance
    instances_b: Dict[str, Tensor],  # another instance
) -> Dict[str, Tensor]:
    """Merge two instance dicts into one (weighted average).

    Args:
        instances_a: {'boxes': [4], 'scores': scalar, 'classes': scalar, 'masks': [H,W]}
        instances_b: Same format.

    Returns:
        Merged instance dict.
    """
    w_a = instances_a['scores'] / (instances_a['scores'] + instances_b['scores'] + 1e-8)
    w_b = 1 - w_a

    merged = {}
    merged['boxes'] = instances_a['boxes'] * w_a + instances_b['boxes'] * w_b
    merged['scores'] = max(instances_a['scores'], instances_b['scores'])
    merged['classes'] = instances_a['classes']  # keep dominant class

    # Resize mask_b to match mask_a if needed
    mask_a = instances_a['masks']
    mask_b = instances_b['masks']
    if mask_a.shape != mask_b.shape:
        mask_b = F.interpolate(
            mask_b.float().unsqueeze(0).unsqueeze(0),
            size=mask_a.shape, mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)
    merged['masks'] = (mask_a * w_a + mask_b * w_b) > 0.5

    return merged


def fuse_tile_instances(
    tile_results: List[Dict[str, Tensor]],   # per-tile YOLOv8 outputs
    tile_specs: List[TileSpec],               # per-tile specifications
    image_size: Tuple[int, int],              # (H, W) full image
    mask_iou_threshold: float = 0.5,
    box_iou_threshold: float = 0.6,
) -> Dict[str, Tensor]:
    """Fuse per-tile instances into full-image predictions.

    Algorithm (SAHI-style):
      1. Convert all tile-local boxes → full-image coordinates
      2. Place all masks on a full-image canvas
      3. For each pair of instances with box IoU > threshold:
         - If mask IoU > 0.5 → merge (same instance across tiles)
         - Otherwise → keep both (different instances)
      4. Run final NMS on merged instances

    Args:
        tile_results: List of N_tiles dicts from PerTileFastSAM.
        tile_specs: List of N_tiles TileSpec with position info.
        image_size: (H, W) of original full image.
        mask_iou_threshold: Threshold for mask-based merging.
        box_iou_threshold: Threshold for box-based NMS.

    Returns:
        {'boxes': [M, 4], 'scores': [M], 'classes': [M], 'masks': [M, H, W]}
    """
    H_img, W_img = image_size

    # ── Step 1: Convert all instances to full-image coordinates ─
    all_boxes: List[Tensor] = []
    all_scores: List[Tensor] = []
    all_classes: List[Tensor] = []
    all_masks: List[Tensor] = []
    all_ids: List[int] = []  # which tile each instance came from

    for tile_idx, (result, spec) in enumerate(zip(tile_results, tile_specs)):
        if result['boxes'].numel() == 0:
            continue

        full_boxes = tile_to_full_coordinates(
            result['boxes'], spec, image_size
        )

        # Place masks on full-image canvas
        tile_x1 = int(full_boxes[:, 0].min().item()) if len(full_boxes) > 0 else 0
        tile_y1 = int(full_boxes[:, 1].min().item()) if len(full_boxes) > 0 else 0

        for i in range(len(result['boxes'])):
            box = full_boxes[i]
            score = result['scores'][i]
            cls_id = result['classes'][i]

            # Create full-image mask for this instance
            full_mask = torch.zeros(H_img, W_img, dtype=torch.bool,
                                    device=box.device)
            if result['masks'].numel() > 0 and i < len(result['masks']):
                tile_mask = result['masks'][i]  # [H_t, W_t]
                # Place tile mask at correct position
                x1 = max(0, int(box[0].item()))
                y1 = max(0, int(box[1].item()))
                x2 = min(W_img, int(box[2].item()))
                y2 = min(H_img, int(box[3].item()))

                out_h, out_w = y2 - y1, x2 - x1
                if out_h <= 0 or out_w <= 0:
                    all_boxes.append(box); all_scores.append(score)
                    all_classes.append(cls_id); all_masks.append(torch.zeros(H_img, W_img, dtype=torch.bool, device=box.device))
                    all_ids.append(tile_idx)
                    continue

                mask_resized = F.interpolate(
                    tile_mask.float().unsqueeze(0).unsqueeze(0),
                    size=(out_h, out_w), mode="bilinear",
                    align_corners=False,
                ).squeeze() > 0.5

                crop_h = min(out_h, full_mask[y1:y2, x1:x2].shape[0])
                crop_w = min(out_w, full_mask[y1:y2, x1:x2].shape[1])
                if crop_h > 0 and crop_w > 0:
                    full_mask[y1:y1 + crop_h, x1:x1 + crop_w] = \
                        mask_resized[:crop_h, :crop_w]

            all_boxes.append(box)
            all_scores.append(score)
            all_classes.append(cls_id)
            all_masks.append(full_mask)
            all_ids.append(tile_idx)

    if len(all_boxes) == 0:
        return {
            'boxes': torch.empty(0, 4), 'scores': torch.empty(0),
            'classes': torch.empty(0, dtype=torch.long),
            'masks': torch.empty(0, H_img, W_img, dtype=torch.bool),
        }

    # ── Step 2: Mask IoU Matching ───────────────────────────
    N = len(all_boxes)
    merged = [True] * N  # True = still active (not absorbed by merge)
    final_instances: List[Dict[str, Tensor]] = []

    for i in range(N):
        if not merged[i]:
            continue
        current = {
            'boxes': all_boxes[i], 'scores': all_scores[i],
            'classes': all_classes[i], 'masks': all_masks[i],
        }

        for j in range(i + 1, N):
            if not merged[j]:
                continue
            # Skip same-tile instances
            if all_ids[i] == all_ids[j]:
                continue

            # Check box IoU first (cheap)
            box_iou_val = _box_iou(all_boxes[i], all_boxes[j])
            if box_iou_val < box_iou_threshold:
                continue

            # Check mask IoU (expensive, only if box IoU passes)
            miou = mask_iou(all_masks[i], all_masks[j])
            if miou > mask_iou_threshold:
                # Same instance across tiles → merge
                other = {
                    'boxes': all_boxes[j], 'scores': all_scores[j],
                    'classes': all_classes[j], 'masks': all_masks[j],
                }
                current = merge_instances(current, other)
                merged[j] = False

        final_instances.append(current)

    # ── Step 3: Box-level NMS on merged instances ──────────
    if len(final_instances) > 1:
        boxes_t = torch.stack([inst['boxes'] for inst in final_instances])
        scores_t = torch.stack([inst['scores'] for inst in final_instances])

        keep = _nms(boxes_t, scores_t, box_iou_threshold)

        final_instances = [final_instances[i] for i in keep]

    return {
        'boxes': torch.stack([inst['boxes'] for inst in final_instances]),
        'scores': torch.stack([inst['scores'] for inst in final_instances]),
        'classes': torch.stack([inst['classes'] for inst in final_instances]).long(),
        'masks': torch.stack([inst['masks'] for inst in final_instances]),
    }


# ── Helpers ─────────────────────────────────────────────────────

def _box_iou(box_a: Tensor, box_b: Tensor) -> float:
    """Compute IoU of two boxes."""
    x1 = max(box_a[0].item(), box_b[0].item())
    y1 = max(box_a[1].item(), box_b[1].item())
    x2 = min(box_a[2].item(), box_b[2].item())
    y2 = min(box_a[3].item(), box_b[3].item())
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return float(inter / (area_a + area_b - inter + 1e-8))


def _nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> List[int]:
    """Standard NMS — returns indices to keep."""
    if boxes.numel() == 0:
        return []

    # Sort by score descending
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        idx = order[0].item()
        keep.append(idx)

        if order.numel() == 1:
            break

        # IoU of best box with rest
        best_box = boxes[idx]
        other_boxes = boxes[order[1:]]
        ious = []
        for ob in other_boxes:
            ious.append(_box_iou(best_box, ob))
        ious = torch.tensor(ious, device=boxes.device)

        mask = ious < iou_threshold
        order = order[1:][mask]

    return keep
