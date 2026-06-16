#!/usr/bin/env python
"""Sparse Inference Ablation: Post-Hoc Masking vs True Tile-Based Inference.

Compares three inference modes:
    Full        — backbone + decoder on entire image (no sparsity)
    Post-Hoc    — full backbone + decoder, then zero out low-importance masks
    Tile-Based  — full backbone, decoder ONLY on high-importance tiles

Key insight: Post-hoc masking reports "sparse IoU" but doesn't save any
computation. Tile-based inference actually reduces decoder FLOPs.

Metrics:
    Dice, IoU, Coverage, FLOPs reduction, tiles processed, keep_ratio

Usage:
    python tools/ablation_sparse_inference.py --quick
    python tools/ablation_sparse_inference.py --max-steps 500
"""

import argparse, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adatile.datasets.universal import UniversalDataset
from adatile.logging.run_logger import RunLogger
from adatile.engine import build_components
from adatile.evaluation import sparse_eval
from adatile.inference import tile_sparse_forward, estimate_flops_saved


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_full(backbone, decoder, val_loader, device):
    """Full-image evaluation (no sparsity)."""
    decoder.eval()
    dices, ious = [], []
    with torch.no_grad():
        for vb in val_loader:
            vi = vb["images"].to(device)
            vg = vb["masks"].to(device)
            feats = backbone(vi)
            lgs = decoder(features=feats)
            pb = (lgs.sigmoid() > 0.5).float()
            gt = vg.float()
            if gt.dim() == 2:
                gt = gt.unsqueeze(0).unsqueeze(0)  # [H,W] → [1,1,H,W]
            elif gt.dim() == 3:
                gt = gt.unsqueeze(1)                # [B,H,W] → [B,1,H,W]
            gt = F.interpolate(gt, size=pb.shape[-2:], mode="nearest")
            gt_bin = (gt > 0.5).float()
            eps = 1e-8
            inter = (pb * gt_bin).sum()
            union = (pb + gt_bin).clamp(0, 1).sum()
            ious.append((inter / (union + eps)).item())
            dices.append((2 * inter / (pb.sum() + gt_bin.sum() + eps)).item())
    decoder.train()
    return {"iou": np.mean(ious), "dice": np.mean(dices)}


def evaluate_tile_based(backbone, decoder, spm, val_loader, device, keep_ratio):
    """Tile-based sparse evaluation — actually reduces decoder FLOPs."""
    decoder.eval()
    spm.eval()
    dices, ious, covers, imps, reductions, n_tiles = [], [], [], [], [], []
    with torch.no_grad():
        for vb in val_loader:
            vi = vb["images"].to(device)
            vg = vb["masks"].to(device)
            full_mask, sparse_mask, metrics = tile_sparse_forward(
                vi, backbone, decoder, spm,
                keep_ratio=keep_ratio,
                tile_size=32,  # smaller tiles = more fine-grained selection
            )
            gt = vg.float()
            if gt.dim() == 2:
                gt = gt.unsqueeze(0).unsqueeze(0)  # [H,W] → [1,1,H,W]
            elif gt.dim() == 3:
                gt = gt.unsqueeze(1)                # [B,H,W] → [B,1,H,W]
            gt = F.interpolate(gt, size=sparse_mask.shape[-2:], mode="nearest")
            gt_bin = (gt > 0.5).float()
            eps = 1e-8
            inter = (sparse_mask * gt_bin).sum()
            union = (sparse_mask + gt_bin).clamp(0, 1).sum()
            ious.append((inter / (union + eps)).item())
            dices.append((2 * inter / (sparse_mask.sum() + gt_bin.sum() + eps)).item())
            covers.append(metrics["coverage"])
            imps.append(metrics["imp_mean"])
            reductions.append(metrics["compute_reduction"])
            n_tiles.append(metrics["tiles_processed"])
    decoder.train()
    spm.train()
    return {
        "iou": np.mean(ious), "dice": np.mean(dices),
        "coverage": np.mean(covers), "imp_mean": np.mean(imps),
        "compute_reduction": np.mean(reductions),
        "tiles_processed": np.mean(n_tiles),
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Sparse Inference Ablation")
    p.add_argument("--dataset", type=str,
                   default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/isaid_dota")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[1.0, 0.50, 0.25, 0.15, 0.10, 0.05])
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log = RunLogger("outputs", "sparse_inference", vars(args))
    log.start()

    img_sz = None if args.image_size == 0 else (args.image_size, args.image_size)
    val_ds = UniversalDataset(
        args.dataset, split="val", image_size=img_sz, num_classes=None,
    )
    vl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Dataset: {val_ds.layout}, classes={val_ds.num_classes}, val={len(val_ds)}")

    # Build model (needs to be trained first — load checkpoint)
    # For this ablation, we assume a trained model is available
    class BA:
        image_size = args.image_size
        unfreeze_layers = 2
        use_spm = True
        spm_type = "light"

    backbone, decoder, spm = build_components(BA(), device, val_ds.num_classes)

    # ── Full inference baseline ──────────────────────────────────
    full = evaluate_full(backbone, decoder, vl, device)
    print(f"Full inference:          Dice={full['dice']:.4f}  IoU={full['iou']:.4f}")

    # ── FLOPs estimate ───────────────────────────────────────────
    image_hw = (args.image_size, args.image_size) if args.image_size else (640, 640)
    flops_est = estimate_flops_saved(image_hw, keep_ratio=1.0)
    print(f"Decoder FLOPs (full):    {flops_est['full_decoder_flops']/1e6:.1f}M")

    # ── Ratio sweep ──────────────────────────────────────────────
    print(f"\n{'Ratio':>7} | {'Full Dice':>9} | {'Sparse Dice':>10} | "
          f"{'Coverage':>8} | {'Comp.Red.':>9} | {'Tiles':>6} | {'Mode':>12}")
    print("-" * 78)

    all_res = []
    for ratio in args.ratios:
        if ratio >= 1.0:
            # Full inference (no sparsity)
            r = {
                "keep_ratio": 1.0,
                "dice": full["dice"],
                "iou": full["iou"],
                "coverage": 1.0,
                "compute_reduction": 0.0,
                "tiles_processed": "all",
                "mode": "full",
            }
            print(f"{1.0:7.0%} | {full['dice']:9.4f} | {'-':>10} | "
                  f"{1.0:8.1%} | {0.0:9.1%} | {'all':>6} | {'full':>12}")
        else:
            # Post-hoc masking
            class EvalArgs:
                use_planner = True
                keep_ratio = ratio

            posthoc = sparse_eval(backbone, decoder, spm, vl, device, EvalArgs())
            r = {
                "keep_ratio": ratio,
                "dice": posthoc.get("sparse_dice", 0),
                "iou": posthoc.get("sparse_iou", 0),
                "coverage": posthoc.get("coverage", 0),
                "compute_reduction": 0.0,  # post-hoc doesn't reduce compute
                "tiles_processed": "all (post-hoc)",
                "mode": "post-hoc",
            }

            # Tile-based sparse inference
            tb = evaluate_tile_based(backbone, decoder, spm, vl, device, ratio)
            r_tb = {
                "keep_ratio": ratio,
                "dice": tb["dice"],
                "iou": tb["iou"],
                "coverage": tb["coverage"],
                "compute_reduction": tb["compute_reduction"],
                "tiles_processed": int(tb["tiles_processed"]),
                "mode": "tile-based",
            }
            all_res.append(r_tb)

            print(f"{ratio:7.0%} | {full['dice']:9.4f} | {r['dice']:10.4f} | "
                  f"{r['coverage']:8.1%} | {r['compute_reduction']:9.1%} | "
                  f"{'post':>6} | {'post-hoc':>12}")
            print(f"{'':>7} | {'':>9} | {r_tb['dice']:10.4f} | "
                  f"{r_tb['coverage']:8.1%} | {r_tb['compute_reduction']:9.1%} | "
                  f"{r_tb['tiles_processed']:>6} | {'tile-based':>12}")

        all_res.append(r)

    # ── Analysis ─────────────────────────────────────────────────
    if len(all_res) > 1:
        ref_full = all_res[0]["dice"]
        ref_15 = next(
            (r for r in all_res if abs(r.get("keep_ratio", 0) - 0.15) < 0.02
             and r.get("mode") == "post-hoc"),
            None,
        )
        if ref_15:
            retention = ref_15["dice"] / max(ref_full, 1e-8)
            print(f"\n  Key Result: 15% tiles → {retention:.1%} Dice retention "
                  f"(post-hoc masking)")

        tb_15 = next(
            (r for r in all_res if abs(r.get("keep_ratio", 0) - 0.15) < 0.02
             and r.get("mode") == "tile-based"),
            None,
        )
        if tb_15:
            print(f"  Tile-based:  {tb_15['compute_reduction']:.0%} FLOPs reduction, "
                  f"Dice={tb_15['dice']:.4f}, Coverage={tb_15['coverage']:.1%}")

        # Paper-ready table
        print(f"\n  Paper-ready table:")
        print(f"  | keep_ratio | Full Dice | PostHoc Dice | Tile Dice | "
              f"Coverage | FLOPs Saved |")
        print(f"  |------------|-----------|--------------|-----------|"
              f"----------|-------------|")
        for r in all_res:
            if r.get("mode") == "full":
                print(f"  | {1.0:.0%}        | {r['dice']:.4f}   | -            | "
                      f"-         | {r['coverage']:.1%}    | {r['compute_reduction']:.0%}           |")
            elif r.get("mode") == "post-hoc":
                print(f"  | {r['keep_ratio']:.0%}        | -         | {r['dice']:.4f}      | "
                      f"-         | {r['coverage']:.1%}    | {r['compute_reduction']:.0%}           |")
            elif r.get("mode") == "tile-based":
                print(f"  | {r['keep_ratio']:.0%}        | -         | -            | "
                      f"{r['dice']:.4f}  | {r['coverage']:.1%}    | "
                      f"{r['compute_reduction']:.0%}           |")

    log.log_table("sparse_inference", all_res)
    log.finish()
    print(f"\nLog: {log.run_dir}")


if __name__ == "__main__":
    a = parse_args()
    if a.quick:
        a.ratios = [1.0, 0.25, 0.15, 0.05]
    main()
