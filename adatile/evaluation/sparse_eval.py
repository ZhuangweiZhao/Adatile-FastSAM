"""Sparse evaluation utilities — full vs sparse inference metrics.

Extracted from tools/train_as_fastsam.py.

Usage:
    from adatile.evaluation.sparse_eval import sparse_eval, split_support_query
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def split_support_query(
    dataset,
    n_shot: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Randomly split dataset indices into support and query sets.

    Args:
        dataset: Dataset with __len__.
        n_shot: Number of support samples.
        seed: Random seed for reproducibility.

    Returns:
        (support_indices, query_indices) tuple.
    """
    rng = random.Random(seed)
    idx = list(range(len(dataset)))
    rng.shuffle(idx)
    return idx[:n_shot], idx[n_shot:]


def sparse_eval(
    backbone,
    decoder,
    spm,
    val_loader,
    device: torch.device,
    args,
    use_tile_inference: bool = False,
) -> Dict[str, float]:
    """Evaluate full + sparse segmentation metrics.

    Runs backbone+decoder on full images, then optionally applies
    SPM-guided sparse masking (post-hoc) or tile-based sparse inference.

    Args:
        backbone: Feature extractor.
        decoder: Segmentation decoder.
        spm: Sparse Perception Module (or None).
        val_loader: Validation DataLoader.
        device: Target device.
        args: Namespace with use_planner, keep_ratio attributes.
        use_tile_inference: If True, use tile-based decoder (True sparse).
                           If False, use post-hoc mask zeroing.

    Returns:
        Dict with mean values for: full_iou, sparse_iou, sparse_dice,
        imp_mean, keep_ratio, coverage.
    """
    decoder.eval()
    if spm is not None:
        spm.eval()

    results: Dict[str, list] = {
        "full_iou": [],
        "sparse_iou": [],
        "sparse_dice": [],
        "imp_mean": [],
        "keep_ratio": [],
        "coverage": [],
    }

    if use_tile_inference:
        # True tile-based sparse inference
        try:
            from adatile.inference.tile_inference import tile_sparse_forward
        except ImportError:
            raise ImportError(
                "Tile-based sparse inference requires adatile.inference.tile_inference. "
                "Run Phase 3 setup first."
            )
        with torch.no_grad():
            for vb in val_loader:
                vi = vb["images"].to(device)
                vg = vb["masks"].to(device)
                keep_ratio = getattr(args, "keep_ratio", 0.15)
                full_mask, sparse_mask, tile_metrics = tile_sparse_forward(
                    vi, backbone, decoder, spm, keep_ratio,
                )
                gt_bin = _prepare_gt(vg, full_mask.shape[-2:], device)
                eps = 1e-8

                # Full metrics
                fi = (full_mask * gt_bin).sum()
                fu = (full_mask + gt_bin).clamp(0, 1).sum()
                results["full_iou"].append((fi / (fu + eps)).item())

                # Sparse metrics
                si = (sparse_mask * gt_bin).sum()
                su = (sparse_mask + gt_bin).clamp(0, 1).sum()
                results["sparse_iou"].append((si / (su + eps)).item())
                results["sparse_dice"].append(
                    (2 * si / (sparse_mask.sum() + gt_bin.sum() + eps)).item()
                )

                results["imp_mean"].append(tile_metrics.get("imp_mean", 0))
                results["keep_ratio"].append(tile_metrics.get("keep_ratio", 0))
                results["coverage"].append(tile_metrics.get("coverage", 0))
    else:
        # Original post-hoc mask zeroing
        with torch.no_grad():
            for vb in val_loader:
                vi = vb["images"].to(device)
                vg = vb["masks"].to(device)
                feats = backbone(vi)
                lgs = decoder(features=feats)

                full_bin = (lgs.sigmoid() > 0.5).float()
                gt4d = vg.float()
                # Fix: correctly handle [H,W] (2D) vs [B,H,W] (3D) vs [B,1,H,W] (4D)
                if gt4d.dim() == 2:
                    gt4d = gt4d.unsqueeze(0).unsqueeze(0)  # [H,W] → [1,1,H,W]
                elif gt4d.dim() == 3:
                    gt4d = gt4d.unsqueeze(1)                # [B,H,W] → [B,1,H,W]
                if gt4d.shape[1] > 4:
                    gt4d = gt4d.mean(dim=1, keepdim=True)
                gt4d = F.interpolate(gt4d, size=full_bin.shape[-2:], mode="nearest")
                gt_bin = (gt4d > 0.5).float()
                eps = 1e-8

                fi = (full_bin * gt_bin).sum()
                fu = (full_bin + gt_bin).clamp(0, 1).sum()
                results["full_iou"].append((fi / (fu + eps)).item())

                if spm is not None and getattr(args, "use_planner", False):
                    imp = spm(feats)
                    # Support SparsePrediction from AdaSPM
                    if hasattr(imp, 'importance'):
                        imp = imp.importance
                    imp_flat = imp[0, 0].reshape(-1)
                    n_total = imp_flat.shape[0]
                    k = max(1, int(n_total * args.keep_ratio))
                    _, idx = imp_flat.topk(k)
                    keep = torch.zeros(n_total, dtype=torch.bool, device=device)
                    keep[idx] = True
                    keep = keep.reshape(imp.shape[-2:])
                    keep_large = (
                        F.interpolate(
                            keep.float().unsqueeze(0).unsqueeze(0),
                            size=full_bin.shape[-2:],
                            mode="nearest",
                        ).squeeze()
                        > 0.5
                    )
                    sparse_bin = full_bin.clone()
                    sparse_bin[:, :, ~keep_large] = 0
                    results["imp_mean"].append(imp.mean().item())
                    results["keep_ratio"].append(keep.float().mean().item())
                    gt_in_kept = gt_bin * keep_large.float().unsqueeze(0).unsqueeze(0)
                    results["coverage"].append(
                        (gt_in_kept.sum() / (gt_bin.sum() + eps)).item()
                        if gt_bin.sum() > 0
                        else 0.0
                    )
                else:
                    sparse_bin = full_bin

                si = (sparse_bin * gt_bin).sum()
                su = (sparse_bin + gt_bin).clamp(0, 1).sum()
                results["sparse_iou"].append((si / (su + eps)).item())
                results["sparse_dice"].append(
                    (2 * si / (sparse_bin.sum() + gt_bin.sum() + eps)).item()
                )

    return {k: float(np.mean(v)) for k, v in results.items() if v}


def _prepare_gt(masks: torch.Tensor, target_size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    """Prepare ground truth masks for IoU computation.

    Args:
        masks: Raw masks tensor.
        target_size: (H, W) target size.
        device: Target device.

    Returns:
        Binary float tensor [B, 1, H, W].
    """
    gt = masks.float().to(device)
    # Fix: correctly handle [H,W] (2D) vs [B,H,W] (3D) vs [B,1,H,W] (4D)
    if gt.dim() == 2:
        gt = gt.unsqueeze(0).unsqueeze(0)  # [H,W] → [1,1,H,W]
    elif gt.dim() == 3:
        gt = gt.unsqueeze(1)                # [B,H,W] → [B,1,H,W]
    if gt.shape[1] > 4:
        gt = gt.mean(dim=1, keepdim=True)
    gt = F.interpolate(gt, size=target_size, mode="nearest")
    return (gt > 0.5).float()
