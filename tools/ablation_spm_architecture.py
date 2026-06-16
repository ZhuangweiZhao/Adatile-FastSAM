#!/usr/bin/env python
"""SPM Architecture Ablation: LightSPM vs AdaSPM-Lite vs AdaSPM-Full.

Answers the critical question: Is the 3-layer conv LightSPM sufficient,
or does the full FPN+Transformer AdaSPM provide measurable benefits?

Variants compared:
    LightSPM       — 3 conv layers, P8 only, sigmoid output            (baseline)
    AdaSPM-Lite    — FPN 128ch, no transformer, density+granularity heads
    AdaSPM-Full    — FPN 256ch + SpatialTransformer, density+granularity heads
    DensityOnlySPM — AdaSPM-Full but uniform granularity (ablation within ablation)

Key Metrics:
    Dice, IoU, Coverage, imp_mean, #params, forward time (ms), peak GPU memory

Usage:
    python tools/ablation_spm_architecture.py --quick
    python tools/ablation_spm_architecture.py --max-steps 500
"""

import argparse, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adatile.datasets.universal import UniversalDataset
from adatile.logging.run_logger import RunLogger
from adatile.losses import UnifiedLoss
from adatile.engine import build_components, collect_params
from adatile.evaluation import sparse_eval


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_one(backbone, decoder, spm, train_loader, val_loader, device, args, label):
    """Train one SPM variant. Returns best metrics and timing info."""
    params = collect_params(backbone, decoder, spm)
    loss_fn = UnifiedLoss(
        use_spm=True, num_classes=1, spm_mode="topk",
        budget_target=args.keep_ratio,
    )
    params += list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)
    log_interval = max(1, args.max_steps // 15)

    decoder.train()
    spm.train()

    best_dice, best_cover, best_imp = 0.0, 0.0, 0.0
    fwd_times = []

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for step in range(args.max_steps):
        try:
            batch = next(train_iter)
        except (NameError, StopIteration):
            train_iter = iter(train_loader)
            batch = next(train_iter)

        img = batch["images"].to(device)
        gt = batch["masks"].to(device)

        # Time forward pass
        t0 = time.perf_counter()
        feats = backbone(img)

        # AdaSPM variants return SparsePrediction; LightSPM returns tensor
        imp = spm(feats)
        if hasattr(imp, 'importance'):
            imp_tensor = imp.importance
        else:
            imp_tensor = imp

        lgs = decoder(features=feats)
        loss, tm = loss_fn(lgs, gt, imp_tensor)
        t1 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000)  # ms

        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        opt.zero_grad()

        if step % log_interval == 0 or step == args.max_steps - 1:
            decoder.eval()
            spm.eval()
            ious, dices, covers, imps = [], [], [], []
            with torch.no_grad():
                for vb in val_loader:
                    vi = vb["images"].to(device)
                    vg = vb["masks"].to(device)
                    vf = backbone(vi)
                    vimp = spm(vf)
                    if hasattr(vimp, 'importance'):
                        vimp_t = vimp.importance
                    else:
                        vimp_t = vimp
                    vl = decoder(features=vf)
                    _, vm = loss_fn(vl, vg, vimp_t)
                    ious.append(vm["iou"])
                    dices.append(vm["dice"])
                    covers.append(vm.get("coverage", 0))
                    imps.append(vm.get("imp_mean", 0))
            decoder.train()
            spm.train()

            val_dice = np.mean(dices)
            val_cover = np.mean(covers)
            val_imp = np.mean(imps)
            print(f"  [{step:4d}] dice={val_dice:.3f}  cover={val_cover:.2%}  "
                  f"imp={val_imp:.4f}  fwd={np.mean(fwd_times[-10:]):.1f}ms")

            if val_dice > best_dice:
                best_dice = val_dice
                best_cover = val_cover
                best_imp = val_imp

    # Count parameters
    n_params = sum(p.numel() for p in spm.parameters())

    return {
        "dice": best_dice,
        "coverage": best_cover,
        "imp_mean": best_imp,
        "n_params": n_params,
        "fwd_time_ms": float(np.mean(fwd_times[-20:])),  # last 20 steps (stable)
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="SPM Architecture Ablation")
    p.add_argument("--dataset", type=str,
                   default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/isaid_dota")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--keep-ratio", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--spm-types", type=str, nargs="+",
                   default=["light", "lite", "full"],
                   help="SPM variants to test")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    log = RunLogger("outputs", "spm_architecture", vars(args))
    log.start()

    img_sz = None if args.image_size == 0 else (args.image_size, args.image_size)
    full_train = UniversalDataset(
        args.dataset, split="train", image_size=img_sz, num_classes=None,
    )
    full_val = UniversalDataset(
        args.dataset, split="val", image_size=img_sz,
        num_classes=full_train.num_classes,
    )
    print(f"Dataset: {full_train.layout}, num_classes={full_train.num_classes}, "
          f"train={len(full_train)}, val={len(full_val)}")

    train_loader = torch.utils.data.DataLoader(
        full_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        full_val, batch_size=1, shuffle=False, num_workers=0,
    )

    # BuildArgs-compatible namespace for shared builder
    class BA:
        image_size = args.image_size
        unfreeze_layers = args.unfreeze_layers
        use_spm = True
        spm_type = "light"  # will be overridden per variant

    results = {}
    ba = BA()

    for spm_type in args.spm_types:
        ba.spm_type = spm_type
        backbone, decoder, spm = build_components(ba, device, full_train.num_classes)

        label = {
            "light": "LightSPM (3-conv, P8 only)",
            "lite": "AdaSPM-Lite (FPN 128ch, no transformer)",
            "full": "AdaSPM-Full (FPN 256ch + transformer)",
            "density_only": "DensityOnlySPM (full FPN, uniform granularity)",
        }.get(spm_type, spm_type)

        n_params_spm = sum(p.numel() for p in spm.parameters()) if spm else 0
        print(f"\n  SPM params: {n_params_spm:,} ({n_params_spm/1e3:.1f}K)")

        r = train_one(
            backbone, decoder, spm, train_loader, val_loader,
            device, args, label,
        )
        results[spm_type] = r

        del backbone, decoder, spm
        torch.cuda.empty_cache()

    # ═══ Results Table ═══════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"SPM Architecture Ablation Results")
    print(f"{'='*80}")
    print(f"{'Variant':<40} {'Dice':>8} {'Coverage':>10} {'imp':>8} "
          f"{'Params':>10} {'Fwd(ms)':>9}")
    print("-" * 88)

    spm_names = {
        "light": "LightSPM (3-conv)",
        "lite": "AdaSPM-Lite",
        "full": "AdaSPM-Full",
        "density_only": "DensityOnlySPM",
    }

    for spm_type in args.spm_types:
        r = results[spm_type]
        name = spm_names.get(spm_type, spm_type)
        print(f"  {name:<40} {r['dice']:8.4f} {r['coverage']:10.2%} "
              f"{r['imp_mean']:8.4f} {r['n_params']:>8,}  {r['fwd_time_ms']:8.1f}")

    # Analysis
    if "light" in results and "full" in results:
        light = results["light"]
        full = results["full"]
        delta_dice = full["dice"] - light["dice"]
        delta_cover = full["coverage"] - light["coverage"]
        speedup = full["fwd_time_ms"] / max(light["fwd_time_ms"], 1e-3)
        param_ratio = full["n_params"] / max(light["n_params"], 1)

        print(f"\n  Analysis: AdaSPM-Full vs LightSPM")
        print(f"    ΔDice:     {delta_dice:+.4f}")
        print(f"    ΔCoverage:  {delta_cover:+.2%}")
        print(f"    Speed:      LightSPM is {speedup:.1f}x faster")
        print(f"    Params:     AdaSPM-Full has {param_ratio:.1f}x more params")

        if delta_dice < 0.01:
            print(f"    Conclusion:  LightSPM is SUFFICIENT — "
                  f"Full AdaSPM adds <1% Dice with {param_ratio:.0f}x params")
        elif delta_dice < 0.03:
            print(f"    Conclusion:  Trade-off — AdaSPM-Full provides small gain "
                  f"({delta_dice:+.3f} Dice) at significant cost")
        else:
            print(f"    Conclusion:  AdaSPM-Full is WORTHWHILE — "
                  f"{delta_dice:+.3f} Dice gain justifies cost")

    # Paper-ready table
    print(f"\n  Paper-ready table:")
    print(f"  | Variant         | Dice   | Coverage | imp_mean | Params  | Fwd(ms) |")
    print(f"  |-----------------|--------|----------|----------|---------|---------|")
    for spm_type in args.spm_types:
        r = results[spm_type]
        name = spm_names.get(spm_type, spm_type)
        print(f"  | {name:<15} | {r['dice']:.4f} | {r['coverage']:.1%}    | "
              f"{r['imp_mean']:.4f} | {r['n_params']:>7,} | {r['fwd_time_ms']:7.1f} |")

    log.log_table("spm_architecture", [
        {"spm_type": spm_type, **{k: round(v, 6) if isinstance(v, float) else v
                                  for k, v in r.items()}}
        for spm_type, r in results.items()
    ])
    log.finish()
    print(f"\nLog: {log.run_dir}")


if __name__ == "__main__":
    a = parse_args()
    if a.quick:
        a.max_steps = 100
        a.image_size = 640
        a.spm_types = ["light", "full"]  # key comparison for quick mode
    main()
