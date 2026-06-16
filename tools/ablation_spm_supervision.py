#!/usr/bin/env python
"""SPM Supervision Ablation: Density Regression vs Top-K Ranking.

L_total = L_seg + L_spm + λ * L_budget
L_budget = (imp.mean() - keep_ratio)²   ← fully differentiable

Experiment:
  Density Regression  — gt_d = 0.01 + 0.89 * binary_fg
  Top-K λ=0.5         — per-image top-K ranking, budget weight 0.5
  Top-K λ=2.0         — stronger budget constraint
  Top-K λ=5.0         — even stronger

Key Metrics: Dice, Coverage, imp_mean

Usage:
  python tools/ablation_spm_supervision.py --quick
  python tools/ablation_spm_supervision.py --max-steps 500
"""

import argparse, sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adatile.datasets.universal import UniversalDataset
from adatile.logging.run_logger import RunLogger
from adatile.losses import UnifiedLoss
from adatile.engine import build_components, StepRunner


def parse_args():
    p = argparse.ArgumentParser(description="SPM Supervision Ablation")
    p.add_argument("--dataset", type=str,
                   default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/isaid_dota")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--keep-ratio", type=float, default=0.15)
    p.add_argument("--keep-ratios", type=float, nargs="*", default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--budget-mode", type=str, default="ratio",
                   choices=["ratio", "entropy", "learnable"])
    p.add_argument("--lambda-reg", type=float, default=0.1)
    p.add_argument("--lambda-budgets", type=float, nargs="+",
                   default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--fast", action="store_true")
    return p.parse_args()


def train_one(backbone, decoder, spm, train_loader, val_loader, device, args,
              spm_mode, label, num_classes, lambda_budget=0.5):
    """Train one config using StepRunner. Returns best {dice, coverage, imp_mean}."""
    loss_fn = UnifiedLoss(
        use_spm=True, num_classes=num_classes,
        spm_mode=spm_mode, budget_target=args.keep_ratio,
        lambda_budget=lambda_budget,
        budget_mode=args.budget_mode,
        lambda_reg=args.lambda_reg,
    )

    runner = StepRunner(
        device=device, max_steps=args.max_steps, lr=args.lr,
        patience=9999,  # no early stopping — run all steps
    )
    runner.setup(backbone, decoder, spm, loss_fn, train_loader, val_loader)

    best_dice, best_cover, best_imp = 0.0, 0.0, 0.0
    log_interval = max(1, args.max_steps // 15)

    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    # StepRunner manual integration — we need per-step eval printing
    train_iter = iter(train_loader)
    for step in range(args.max_steps):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        img = batch["images"].to(device)
        gt = batch["masks"].to(device)
        feats = backbone(img)
        lgs = decoder(features=feats)
        imp = spm(feats)
        loss, tm = loss_fn(lgs, gt, imp)
        loss.backward()
        nn = torch.nn
        params = [p for mod in [backbone, decoder, spm, loss_fn]
                  for p in mod.parameters() if p.requires_grad]
        nn.utils.clip_grad_norm_(params, 1.0)
        runner.optimizer.step()
        runner.scheduler.step()
        runner.optimizer.zero_grad()

        if step % log_interval == 0 or step == args.max_steps - 1:
            val_m = runner.evaluate()
            val_dice = val_m["val_dice"]
            val_cover = val_m.get("val_coverage", 0)
            val_imp = val_m.get("val_imp_mean", 0)
            print(f"  [{step:4d}] dice={val_dice:.3f}  cover={val_cover:.2%}  "
                  f"imp={val_imp:.4f}")

            if val_dice > best_dice:
                best_dice = val_dice
                best_cover = val_cover
                best_imp = val_imp

    return {"dice": best_dice, "coverage": best_cover, "imp_mean": best_imp}


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    log = RunLogger("outputs", "spm_supervision", vars(args))
    log.start()

    img_sz = None if args.image_size == 0 else (args.image_size, args.image_size)
    n = args.max_samples if args.max_samples > 0 else -1
    full_train = UniversalDataset(args.dataset, split="train", image_size=img_sz,
                                  num_classes=args.num_classes, max_samples=n)
    full_val = UniversalDataset(args.dataset, split="val", image_size=img_sz,
                                num_classes=full_train.num_classes, max_samples=max(1, n // 5))
    print(f"Dataset: {full_train.layout}, num_classes={full_train.num_classes}, "
          f"train={len(full_train)}, val={len(full_val)}")

    train_loader = torch.utils.data.DataLoader(
        full_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        full_val, batch_size=1, shuffle=False, num_workers=0)

    class BA:
        image_size = args.image_size
        unfreeze_layers = 2
        use_spm = True
        spm_type = "light"

    results = {}
    sweep_ratios = args.keep_ratios is not None
    ratios = args.keep_ratios if sweep_ratios else [args.keep_ratio]
    lambdas = [5.0] if sweep_ratios else args.lambda_budgets

    # Density baseline — run ONCE
    if sweep_ratios:
        print("\n--- Density baseline (keep_ratio independent) ---")
        backbone, decoder, spm = build_components(BA(), device, full_train.num_classes)
        d = train_one(backbone, decoder, spm, train_loader, val_loader,
                      device, args, "density", "Density",
                      full_train.num_classes, lambda_budget=0.5)
        results["density"] = d
        del backbone, decoder, spm; torch.cuda.empty_cache()

    for kr in ratios:
        args.keep_ratio = kr
        for lb in lambdas:
            backbone, decoder, spm = build_components(BA(), device, full_train.num_classes)
            tag = "learnable" if args.budget_mode == "learnable" else f"λ={lb}"
            label = f"Top-K kr={kr:.0%} {tag}"
            r = train_one(backbone, decoder, spm, train_loader, val_loader,
                          device, args, "topk", label,
                          full_train.num_classes, lambda_budget=lb)
            results[f"topk_kr{kr}_lb{lb}"] = r
            del backbone, decoder, spm; torch.cuda.empty_cache()

    # ── Results Table (same output format as before) ──────────────
    if sweep_ratios:
        print(f"\n{'='*80}")
        print(f"SPM Supervision: keep_ratio sweep (λ=5.0)")
        print(f"{'='*80}")
        print(f"{'keep_ratio':<12} {'Method':<10} {'Dice':>8} {'Coverage':>10} {'imp_mean':>10}")
        print("-" * 54)
        d = results["density"]
        print(f"{'N/A':<12} {'Density':<10} {d['dice']:8.4f} {d['coverage']:10.2%} {d['imp_mean']:10.4f}")
        for kr in ratios:
            t = results[f"topk_kr{kr}_lb5.0"]
            print(f"{kr:<12.0%} {'Top-K':<10} {t['dice']:8.4f} {t['coverage']:10.2%} {t['imp_mean']:10.4f}")

        print(f"\n  Paper-ready table (keep_ratio sweep):")
        print(f"  | keep_ratio | Density Dice | Top-K Dice | Top-K Coverage | Top-K imp_mean |")
        print(f"  |------------|--------------|------------|----------------|----------------|")
        for kr in ratios:
            t = results[f"topk_kr{kr}_lb5.0"]
            print(f"  | {kr:.0%}        | {d['dice']:.4f}      | {t['dice']:.4f}    | "
                  f"{t['coverage']:.1%}          | {t['imp_mean']:.4f}        |")
        table_data = []
        for kr in ratios:
            t = results[f"topk_kr{kr}_lb5.0"]
            table_data.append({"keep_ratio": kr, "method": "Density", **d})
            table_data.append({"keep_ratio": kr, "method": "Top-K", **t})
        log.log_table("spm_supervision", table_data)
    else:
        kr = args.keep_ratio
        if "density" not in results:
            backbone, decoder, spm = build_components(BA(), device, full_train.num_classes)
            d = train_one(backbone, decoder, spm, train_loader, val_loader,
                          device, args, "density", "Density",
                          full_train.num_classes, lambda_budget=0.5)
            results["density"] = d
            del backbone, decoder, spm; torch.cuda.empty_cache()
        else:
            d = results["density"]
        print(f"\n{'='*70}")
        print(f"SPM Supervision Ablation (keep_ratio={kr:.0%})")
        print(f"L_budget = (imp.mean() - {kr:.0%})²")
        print(f"{'='*70}")
        print(f"{'Method':<28} {'Dice':>8} {'Coverage':>10} {'imp_mean':>10}")
        print("-" * 58)
        print(f"{'Density Regression':<28} {d['dice']:8.4f} {d['coverage']:10.2%} {d['imp_mean']:10.4f}")
        for lb in args.lambda_budgets:
            r = results[f"topk_kr{kr}_lb{lb}"]
            print(f"{f'Top-K λ_budget={lb}':<28} {r['dice']:8.4f} {r['coverage']:10.2%} {r['imp_mean']:10.4f}")

        best_key, best_gap = "", 999
        print(f"\n  Analysis: Target imp_mean = {kr:.0%}")
        for lb in args.lambda_budgets:
            r = results[f"topk_kr{kr}_lb{lb}"]
            gap = abs(r["imp_mean"] - kr)
            marker = " ← best" if gap < 0.1 else ""
            print(f"    λ={lb:<4.1f} imp_mean={r['imp_mean']:.4f}  "
                  f"Coverage={r['coverage']:5.1%}  Δ={gap:+.4f}{marker}")
            if gap < best_gap:
                best_gap, best_key = gap, lb
        br = results[f"topk_kr{kr}_lb{best_key}"]
        print(f"\n  Closest to target: λ={best_key}  "
              f"vs Density: ΔCover={br['coverage']-d['coverage']:+.2%}")

        print(f"\n  | Method                    | Dice   | Coverage | imp_mean |")
        print(f"  |---------------------------|--------|----------|----------|")
        print(f"  | Density Regression        | {d['dice']:.4f} |  {d['coverage']:.1%}   |  {d['imp_mean']:.4f} |")
        for lb in args.lambda_budgets:
            r = results[f"topk_kr{kr}_lb{lb}"]
            star = " ★" if lb == best_key else ""
            print(f"  | Top-K Ranking λ={lb:<6}     | {r['dice']:.4f} |  {r['coverage']:.1%}   |  {r['imp_mean']:.4f} |{star}")
        table_data = [{"method": "Density Regression", **d}]
        for lb in args.lambda_budgets:
            r = results[f"topk_kr{kr}_lb{lb}"]
            table_data.append({"method": f"Top-K λ_budget={lb}", **r})
        log.log_table("spm_supervision", table_data)

    log.finish()


if __name__ == "__main__":
    a = parse_args()
    if a.quick:
        a.max_steps = 100; a.image_size = 640
        a.max_samples = 100
        a.lambda_budgets = [1.0, 5.0, 10.0]
    main()
