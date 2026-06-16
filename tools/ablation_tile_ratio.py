#!/usr/bin/env python
"""Tile Retention Ratio Ablation.

Trains ONE model with ExperimentRunner, then evaluates with different
sparse tile keep ratios.

Usage:
    python tools/ablation_tile_ratio.py --quick
    python tools/ablation_tile_ratio.py --max-steps 500
"""

import argparse, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adatile.engine import build_components, ExperimentRunner
from adatile.losses import UnifiedLoss
from adatile.evaluation import sparse_eval
from adatile.logging.run_logger import RunLogger
from adatile.datasets.universal import UniversalDataset


class EvalArgs:
    """Mock args for sparse_eval (reads use_planner, keep_ratio)."""
    use_planner = True
    def __init__(self, keep_ratio=0.15):
        self.keep_ratio = keep_ratio


def parse_args():
    p = argparse.ArgumentParser(description="Tile Ratio Ablation")
    p.add_argument("--dataset", type=str,
                   default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/isaid_dota")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[1.0, 0.75, 0.50, 0.25, 0.15, 0.10, 0.05])
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    log = RunLogger("outputs", "tile_ablation", vars(args))
    log.start()

    img_sz = None if args.image_size == 0 else args.image_size
    img_sz_t = None if img_sz is None else (img_sz, img_sz)
    full_ds = UniversalDataset(args.dataset, split="train", image_size=img_sz_t)
    val_ds = UniversalDataset(args.dataset, split="val", image_size=img_sz_t)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    batch_sz = 1 if img_sz is None else 4

    # ── Build model ───────────────────────────────────────────────
    print("Training model (SPM only)...")
    build_args = EvalArgs(keep_ratio=0.15)
    build_args.image_size = args.image_size
    build_args.unfreeze_layers = args.unfreeze_layers
    build_args.use_spm = True
    build_args.spm_type = "light"

    backbone, decoder, spm = build_components(build_args, device, 1)
    loss_fn = UnifiedLoss(use_spm=True)

    # ── Setup ExperimentRunner ────────────────────────────────────
    runner = ExperimentRunner(
        device=device, max_steps=args.max_steps, lr=args.lr,
        patience=15, mode="max",
    )
    runner.setup(backbone, decoder, spm, loss_fn, full_ds, vl)

    # Wire logging via callbacks
    def make_log_cb(runner, log):
        def cb(step, val_metrics):
            log.log(step,
                    loss=val_metrics.get("train_loss", 0),
                    iou=val_metrics.get("train_iou", 0),
                    dice=val_metrics.get("train_dice", 0),
                    imp_mean=val_metrics.get("train_imp_mean", 0),
                    val_iou=val_metrics["val_iou"],
                    val_dice=val_metrics["val_dice"],
                    lr=runner.optimizer.param_groups[0]["lr"])
        return cb
    runner.on_eval(make_log_cb(runner, log))

    # ── Run training ──────────────────────────────────────────────
    results = runner.run_epoch_based(epochs=500, batch_size=batch_sz)
    log.log_best(train_loss=0)  # placeholder

    # ── Ablation sweep (post-training) ────────────────────────────
    all_res = []
    print(f"\n{'Ratio':>6} | {'Full IoU':>8} | {'Sparse IoU':>10} | "
          f"{'Sparse Dice':>10} | {'Coverage':>8} | {'Tiles':>6} | {'Reduction':>9}")
    print("-" * 75)

    for ratio in args.ratios:
        ea = EvalArgs(keep_ratio=ratio)
        r = sparse_eval(backbone, decoder, spm, vl, device, ea)
        r["keep_ratio"] = ratio
        r["reduction"] = 1.0 - ratio
        all_res.append(r)
        log.log_best(**{
            f"ratio_{ratio:.0f}_iou": r.get("full_iou", 0),
            f"ratio_{ratio:.0f}_sparse_iou": r.get("sparse_iou", 0),
            f"ratio_{ratio:.0f}_coverage": r.get("coverage", 0),
        })
        print(f"{ratio:6.1%} | {r.get('full_iou', 0):8.4f} | "
              f"{r.get('sparse_iou', 0):10.4f} | {r.get('sparse_dice', 0):10.4f} | "
              f"{r.get('coverage', 0):8.1%} | {r.get('keep_ratio', 0)*100:5.0f} | "
              f"{1-ratio:9.1%}")

    # Key finding
    if len(all_res) > 1:
        ref_full = all_res[0]
        ref_15 = next(
            (r for r in all_res if abs(r["keep_ratio"] - 0.15) < 0.02), all_res[3]
        )
        retention = ref_15.get("sparse_iou", 0) / (ref_full.get("full_iou", 0) + 1e-8)
        print(f"\nKey Result: {ref_15['keep_ratio']:.0%} tiles → "
              f"{retention:.1%} accuracy, {ref_15.get('coverage', 0):.1%} coverage")
        log.log_best(retention=float(retention),
                     coverage=float(ref_15.get("coverage", 0)))

    log.log_table("tile_ablation", all_res)
    log.finish()

    del backbone, decoder, spm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    a = parse_args()
    if a.quick:
        a.max_steps = 300; a.image_size = 640
        a.ratios = [1.0, 0.25, 0.15, 0.05]
    main()
