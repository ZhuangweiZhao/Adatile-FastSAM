"""Evaluation metrics: COCO AP, few-shot mIoU, sparse efficiency.

Wraps pycocotools for standard COCO evaluation and implements
custom few-shot segmentation metrics.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from adatile.config import Config
from adatile.core import SegmentationOutput


class COCOEvaluator:
    """COCO-style evaluation (bbox + mask AP).

    Wraps pycocotools.cocoeval for standard benchmark evaluation.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.iou_thresholds = cfg.eval.iou_thresholds
        self.max_dets = cfg.eval.max_dets
        self._predictions: List[Dict] = []
        self._targets: List[Dict] = []

    def process(
        self,
        output: SegmentationOutput,
        batch: Dict[str, Any],
    ) -> None:
        """Accumulate predictions and targets for one batch."""
        predictions = output.to_coco(batch.get("image_id", 0))
        self._predictions.extend(predictions)
        # Targets accumulated from batch annotations
        for anns, img_id in zip(
            batch.get("annotations", []), batch.get("image_ids", [])
        ):
            for ann in anns:
                ann_copy = dict(ann)
                ann_copy["image_id"] = img_id
                self._targets.append(ann_copy)

    def evaluate(self) -> Dict[str, float]:
        """Run COCO evaluation on accumulated predictions.

        Returns:
            Dict of metric_name → float value.
            Keys: "coco_bbox_ap", "coco_bbox_ap50", "coco_bbox_ap75",
                  "coco_mask_ap", "coco_mask_ap50", "coco_mask_ap75",
                  "coco_bbox_ar", "coco_mask_ar".
        """
        metrics = {}
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            if self._predictions:
                coco_gt = COCO()
                coco_gt.dataset = {"annotations": self._targets}
                coco_gt.createIndex()

                coco_dt = coco_gt.loadRes(self._predictions)
                coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                metrics["coco_bbox_ap"] = coco_eval.stats[0]
                metrics["coco_bbox_ap50"] = coco_eval.stats[1]
                metrics["coco_bbox_ap75"] = coco_eval.stats[2]
                metrics["coco_bbox_ar"] = coco_eval.stats[6]

                # Mask evaluation
                coco_eval = COCOeval(coco_gt, coco_dt, "segm")
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                metrics["coco_mask_ap"] = coco_eval.stats[0]
                metrics["coco_mask_ap50"] = coco_eval.stats[1]
                metrics["coco_mask_ap75"] = coco_eval.stats[2]
                metrics["coco_mask_ar"] = coco_eval.stats[6]
        except ImportError:
            metrics["pycocotools_error"] = 1.0

        # Reset accumulators
        self._predictions = []
        self._targets = []

        return metrics


class FewShotEvaluator:
    """Few-shot segmentation evaluation metrics.

    Computes:
        - mIoU: Mean intersection-over-union per novel class.
        - FB-IoU: Foreground-background IoU.
        - Binary IoU: Per-class binary IoU averaged over episodes.
    """

    def __init__(self, novel_classes: List[int]):
        self.novel_classes = novel_classes
        self._ious: List[Dict[int, float]] = []
        self._fb_ious: List[float] = []

    def process(
        self,
        pred_masks: Tensor,
        gt_masks: Tensor,
        class_ids: List[int],
    ) -> None:
        """Compute IoU for one episode.

        Args:
            pred_masks: [N, H, W] binary predictions.
            gt_masks: [N, H, W] ground truth masks.
            class_ids: Class ID for each mask.
        """
        episode_ious = {}
        for i, cls_id in enumerate(class_ids):
            pred = pred_masks[i].bool()
            gt = gt_masks[i].bool()
            intersection = (pred & gt).sum().float()
            union = (pred | gt).sum().float()
            iou = (intersection / (union + 1e-8)).item()
            episode_ious[cls_id] = iou

        self._ious.append(episode_ious)

        # FB-IoU: treat all novel classes as foreground
        pred_fg = torch.any(pred_masks.bool(), dim=0)
        gt_fg = torch.any(gt_masks.bool(), dim=0)
        intersection = (pred_fg & gt_fg).sum().float()
        union = (pred_fg | gt_fg).sum().float()
        fb_iou = (intersection / (union + 1e-8)).item()
        self._fb_ious.append(fb_iou)

    def evaluate(self) -> Dict[str, float]:
        """Aggregate results over all episodes.

        Returns:
            Dict with "fewshot_miou", "fewshot_fb_iou", and per-class IoUs.
        """
        if not self._ious:
            return {"fewshot_miou": 0.0, "fewshot_fb_iou": 0.0}

        # Per-class mIoU
        class_ious = defaultdict(list)
        for episode in self._ious:
            for cls_id, iou in episode.items():
                class_ious[cls_id].append(iou)

        metrics = {}
        for cls_id, ious in class_ious.items():
            metrics[f"fewshot_iou_cls{cls_id}"] = np.mean(ious)

        # Mean over all classes and episodes
        all_ious = [iou for ep in self._ious for iou in ep.values()]
        metrics["fewshot_miou"] = np.mean(all_ious)
        metrics["fewshot_fb_iou"] = np.mean(self._fb_ious)

        self._ious = []
        self._fb_ious = []

        return metrics


class SparseEfficiencyMetrics:
    """Efficiency metrics for sparse routing analysis.

    Tracks:
        - Skip ratio: fraction of tiles skipped
        - Token reduction: fraction of tokens pruned vs. dense baseline
        - Expert utilization: load balance across routing experts
        - Tile size distribution: proportion of small/medium/large tiles
    """

    def __init__(self):
        self._skip_ratios: List[float] = []
        self._token_reductions: List[float] = []
        self._expert_loads: List[np.ndarray] = []
        self._tile_size_counts: defaultdict = defaultdict(int)

    def update(
        self,
        total_tiles: int,
        skipped_tiles: int,
        dense_tokens: int,
        routed_tokens: int,
        expert_weights: Optional[Tensor] = None,
        tile_sizes: Optional[List[int]] = None,
    ) -> None:
        """Record efficiency stats for one batch.

        Args:
            total_tiles: Total number of tiles considered.
            skipped_tiles: Number of tiles skipped by sparsity.
            dense_tokens: Token count without sparsity (baseline).
            routed_tokens: Actual token count after routing.
            expert_weights: [N, num_experts] routing weights.
            tile_sizes: Tile sizes used in this batch.
        """
        if total_tiles > 0:
            self._skip_ratios.append(skipped_tiles / total_tiles)
        if dense_tokens > 0:
            self._token_reductions.append(1.0 - routed_tokens / dense_tokens)
        if expert_weights is not None:
            self._expert_loads.append(expert_weights.mean(dim=0).cpu().numpy())
        if tile_sizes is not None:
            for ts in tile_sizes:
                self._tile_size_counts[ts] += 1

    def evaluate(self) -> Dict[str, float]:
        """Aggregate efficiency statistics.

        Returns:
            Dict with skip_ratio, token_reduction, expert_utilization, tile distribution.
        """
        metrics = {}

        if self._skip_ratios:
            metrics["skip_ratio"] = np.mean(self._skip_ratios)
        if self._token_reductions:
            metrics["token_reduction"] = np.mean(self._token_reductions)
        if self._expert_loads:
            avg_load = np.mean(np.stack(self._expert_loads), axis=0)
            metrics["expert_utilization"] = float(
                1.0 - np.std(avg_load) / (np.mean(avg_load) + 1e-8)
            )

        total_tiles = sum(self._tile_size_counts.values())
        for size, count in self._tile_size_counts.items():
            metrics[f"tile_{size}_ratio"] = count / max(total_tiles, 1)

        return metrics
