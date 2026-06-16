#!/usr/bin/env python
"""Ada-SPM Few-Shot Advantage Experiment.

Hypothesis: SPM importance map provides spatial prior — the fewer training
examples, the more valuable this prior is to the Decoder.

Contrast: Baseline (Backbone+Decoder) vs SPM-only (Backbone+Decoder+SPM)
at 1, 3, 5, 10, full-shot × 3 seeds, episodic training.

Key output: ΔDice vs shot count — if ΔDice shrinks as shot↑, SPM few-shot
advantage is confirmed.

       Baseline = Backbone + Decoder
       SPM-only = Backbone + Decoder + LightSPM (Ada-SPM)

Usage:
  python tools/exp_fewshot.py --quick                    # fast sweep
  python tools/exp_fewshot.py                            # full run
  python tools/exp_fewshot.py --baseline                 # baseline only
"""

import argparse, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adatile.datasets.universal import UniversalDataset
from adatile.logging.run_logger import RunLogger
from adatile.utils.early_stop import EarlyStopping
from adatile.losses import UnifiedLoss
from adatile.engine import build_components, collect_params, ExperimentRunner
from adatile.evaluation import split_support_query


# ═══════════════════════════════════════════════════════════════════
# Evaluation (local — few-shot specific metrics via UnifiedLoss)
# ═══════════════════════════════════════════════════════════════════

def evaluate(backbone, decoder, spm, val_loader, device):
    """Evaluate on validation set. Returns dice, iou, coverage, imp_mean."""
    decoder.eval()
    if spm is not None:
        spm.eval()
    dices, ious, covers, imps = [], [], [], []

    with torch.no_grad():
        for vb in val_loader:
            vi = vb["images"].to(device)
            vg = vb["masks"].to(device)
            feats = backbone(vi)
            logits = decoder(features=feats)
            imp = spm(feats) if spm is not None else None
            _, vm = UnifiedLoss(use_spm=spm is not None)(logits, vg, imp)
            dices.append(vm["dice"])
            ious.append(vm["iou"])
            if "coverage" in vm:
                covers.append(vm["coverage"])
            if "imp_mean" in vm:
                imps.append(vm["imp_mean"])

    decoder.train()
    if spm is not None:
        spm.train()
    return {
        "dice": np.mean(dices), "iou": np.mean(ious),
        "coverage": np.mean(covers) if covers else 0.0,
        "imp_mean": np.mean(imps) if imps else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════
# Training (episodic, one (shot, seed, method) combination)
# ═══════════════════════════════════════════════════════════════════

def train_one(backbone, decoder, spm, full_ds, val_loader, device, args,
              n_shot, seed, log, label, step_offset=0):
    """Episodic training for one (shot, seed, method) combination."""
    has_spm = spm is not None
    loss_fn = UnifiedLoss(use_spm=has_spm, num_classes=1, spm_mode="topk",
                          budget_target=0.15)

    # Use ExperimentRunner for opt/sch/stopper setup
    runner = ExperimentRunner(
        device=device, max_steps=args.max_steps, lr=args.lr,
        patience=args.patience, mode="max",
    )
    runner.setup(backbone, decoder, spm, loss_fn, full_ds, val_loader)
    opt = runner.optimizer
    sch = runner.scheduler
    stopper = runner.stopper
    params = collect_params(backbone, decoder, spm, loss_fn)
    log_interval = max(1, args.max_steps // 15)

    decoder.train()
    if has_spm:
        spm.train()

    best_dice, gs = 0.0, 0

    for epoch in range(500):
        s_idx, q_idx = split_support_query(full_ds, n_shot, seed * 1000 + epoch)
        q_subset = torch.utils.data.Subset(full_ds, q_idx)
        dl = torch.utils.data.DataLoader(q_subset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=0)

        for batch in dl:
            img = batch["images"].to(device)
            gt = batch["masks"].to(device)
            feats = backbone(img)
            logits = decoder(features=feats)
            imp = spm(feats) if has_spm else None
            loss, tm = loss_fn(logits, gt, imp)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sch.step()
            opt.zero_grad()
            gs += 1

            if gs % log_interval == 0 or gs == 1:
                m = evaluate(backbone, decoder, spm, val_loader, device)
                log.log(step_offset + gs,
                        loss=loss.item(),
                        train_iou=tm["iou"], train_dice=tm["dice"],
                        val_iou=m["iou"], val_dice=m["dice"],
                        coverage=m["coverage"], imp_mean=m["imp_mean"],
                        lr=opt.param_groups[0]["lr"])
                if gs % (log_interval * 5) == 0:
                    log.log_memory(step_offset + gs)

                if m["dice"] > best_dice:
                    best_dice = m["dice"]
                    best_coverage = m["coverage"]
                    best_imp = m["imp_mean"]
                    log.log_best(**{
                        f"{label}_best_dice": best_dice,
                        f"{label}_best_step": gs,
                        f"{label}_best_coverage": best_coverage,
                    })
                if stopper.step(m["dice"]):
                    break
            if gs >= args.max_steps:
                break
        if gs >= args.max_steps or stopper.should_stop:
            break

    final_m = evaluate(backbone, decoder, spm, val_loader, device)
    return {
        "dice": final_m["dice"], "iou": final_m["iou"],
        "coverage": final_m["coverage"], "imp_mean": final_m["imp_mean"],
        "best_dice": best_dice, "best_coverage": best_coverage, "best_imp": best_imp,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Ada-SPM Few-Shot Advantage")
    p.add_argument("--dataset", type=str,
                   default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/isaid_dota")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5, 10, -1])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--baseline", action="store_true",
                   help="Only run Baseline (no SPM)")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    mode_label = "baseline" if args.baseline else "full"
    log = RunLogger("outputs", f"fewshot_{mode_label}", vars(args))
    log.start()

    img_sz = None if args.image_size == 0 else (args.image_size, args.image_size)
    full_ds = UniversalDataset(args.dataset, split="train", image_size=img_sz, num_classes=None)
    val_ds = UniversalDataset(args.dataset, split="val", image_size=img_sz, num_classes=full_ds.num_classes)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Dataset: {full_ds.layout}, classes={full_ds.num_classes}, "
          f"train={len(full_ds)}, val={len(val_ds)}")

    # Build args-compatible namespace for build_components
    class BuildArgs:
        image_size = args.image_size
        unfreeze_layers = args.unfreeze_layers
        use_spm = True  # will be set per method
        spm_type = "light"

    methods = [("Baseline", False)]
    if not args.baseline:
        methods.append(("SPM-only", True))

    all_results = []
    step_offset = 0

    for n_shot in args.shots:
        shot_label = "full" if n_shot < 0 else f"{n_shot}-shot"
        actual_shot = n_shot if n_shot > 0 else len(full_ds) - 1
        actual_shot = min(actual_shot, len(full_ds) - 1)

        for method_name, use_spm in methods:
            key = f"{shot_label}_{method_name}"
            label = f"{shot_label}_{method_name}"
            print(f"\n{'='*55}\n  {key}  (n={actual_shot})\n{'='*55}")

            shot_results = []
            for seed in range(args.seeds):
                torch.manual_seed(seed)

                # Build components using shared builder
                ba = BuildArgs()
                ba.use_spm = use_spm
                backbone, decoder, spm = build_components(ba, device, 1)

                r = train_one(backbone, decoder, spm, full_ds, val_loader,
                              device, args, actual_shot, seed,
                              log=log, label=f"{label}_s{seed}",
                              step_offset=step_offset)
                step_offset += args.max_steps
                shot_results.append(r)
                print(f"  [{key}] seed={seed} dice={r['dice']:.4f} iou={r['iou']:.4f} "
                      f"cover={r['coverage']:.2%} imp={r['imp_mean']:.4f}")

                del backbone, decoder
                if spm is not None:
                    del spm
                torch.cuda.empty_cache()

            dices = [r["dice"] for r in shot_results]
            ious = [r["iou"] for r in shot_results]
            covers = [r["coverage"] for r in shot_results]
            imps = [r["imp_mean"] for r in shot_results]

            all_results.append({
                "setting": key,
                "shot": shot_label,
                "method": method_name,
                "use_spm": use_spm,
                "dice_mean": np.mean(dices),
                "dice_std": np.std(dices),
                "iou_mean": np.mean(ious),
                "iou_std": np.std(ious),
                "coverage_mean": np.mean(covers),
                "imp_mean": np.mean(imps),
            })

    # ═══ Results Table ═══════════════════════════════════════════
    print(f"\n{'='*78}")
    print(f"Ada-SPM Few-Shot Advantage Experiment")
    print(f"{'='*78}")

    pairs = {}
    for r in all_results:
        k = (r["shot"], r["method"])
        pairs[k] = r

    if not args.baseline:
        print(f"\n  {'Shot':<10} {'Baseline Dice':>13} {'SPM Dice':>10} {'ΔDice':>10} "
              f"{'SPM Cover':>10} {'SPM imp':>9} {'Advantage':>12}")
        print(f"  {'-'*72}")

        for shot_label in [f"{s}-shot" if s > 0 else "full" for s in args.shots]:
            b = pairs.get((shot_label, "Baseline"))
            s = pairs.get((shot_label, "SPM-only"))
            if b is None or s is None:
                continue
            delta = s["dice_mean"] - b["dice_mean"]
            advantage = "★★★" if delta > 0.03 else ("★★" if delta > 0.01 else ("★" if delta > 0 else "-"))
            print(f"  {shot_label:<10} {b['dice_mean']:>13.4f} {s['dice_mean']:>10.4f} "
                  f"{delta:>+10.4f} {s['coverage_mean']:>10.2%} {s['imp_mean']:>9.4f} "
                  f"{advantage:>12}")

        # Paper-ready table
        print(f"\n  Paper-ready table:")
        print(f"  | Shot   | Baseline | SPM-only | ΔDice  | Coverage |")
        print(f"  |--------|----------|----------|--------|----------|")
        for shot_label in [f"{s}-shot" if s > 0 else "full" for s in args.shots]:
            b = pairs.get((shot_label, "Baseline"))
            s = pairs.get((shot_label, "SPM-only"))
            if b is None or s is None:
                continue
            delta = s["dice_mean"] - b["dice_mean"]
            print(f"  | {shot_label:<6} | {b['dice_mean']:.4f}   | {s['dice_mean']:.4f}   "
                  f"| {delta:+.4f} | {s['coverage_mean']:.1%}    |")

        # Analysis
        print(f"\n{'='*78}")
        print(f"Analysis: SPM Few-Shot Advantage Pattern")
        print(f"{'='*78}")

        deltas = []
        for shot_label in [f"{s}-shot" if s > 0 else "full" for s in args.shots]:
            b = pairs.get((shot_label, "Baseline"))
            s = pairs.get((shot_label, "SPM-only"))
            if b and s:
                deltas.append((shot_label, s["dice_mean"] - b["dice_mean"]))

        if len(deltas) >= 2:
            trend_holds = all(
                deltas[i][1] >= deltas[i + 1][1]
                for i in range(len(deltas) - 1)
            )
            print(f"  ΔDice vs shot: {[(s, f'{d:+.4f}') for s, d in deltas]}")
            print(f"  Expected: advantage ↓ as shot ↑")
            print(f"  Observed: {'✅ Confirmed (monotonic decrease)' if trend_holds else '⚠️ Not monotonic — check individual seeds'}")

            if deltas[0][1] > 0.02:
                print(f"  1-shot ΔDice = {deltas[0][1]:+.4f} → SPM provides meaningful few-shot advantage")
            else:
                print(f"  1-shot ΔDice = {deltas[0][1]:+.4f} → SPM few-shot advantage is small/absent")
    else:
        print(f"\n  {'Shot':<10} {'Dice':>10} {'IoU':>10}")
        print(f"  {'-'*30}")
        for shot_label in [f"{s}-shot" if s > 0 else "full" for s in args.shots]:
            b = pairs.get((shot_label, "Baseline"))
            if b:
                print(f"  {shot_label:<10} {b['dice_mean']:>10.4f} {b['iou_mean']:>10.4f}")

    log.log_table("fewshot_results", all_results)
    log.finish()
    print(f"\nLog: {log.run_dir}")


if __name__ == "__main__":
    a = parse_args()
    if a.quick:
        a.max_steps = 200
        a.shots = [1, 5, -1]
        a.seeds = 2
        a.image_size = 640
    main()
