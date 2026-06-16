#!/usr/bin/env python
"""Domain Shift Ablation — verify Ada-SPM learns object density, not dataset texture.

Core Hypothesis:
  If Ada-SPM learns object density → importance maps generalize across domains
  If Ada-SPM learns dataset texture → importance maps fail under domain shift

Experiment (4 combinations):
  A: Train Urban  → Test Urban   (in-domain baseline)
  B: Train Urban  → Test Rural   (cross-domain)
  C: Train Rural  → Test Rural   (in-domain baseline)
  D: Train Rural  → Test Urban   (cross-domain)

Key Metrics:
  Dice, IoU, Coverage, Activation Ratio, Mean Importance, Pearson Correlation

Interpretation:
  Small ΔCoverage across domain → Ada-SPM learns object density ✅
  Large ΔCoverage across domain → Ada-SPM overfits to texture ❌

Usage:
  python tools/ablation_domain_shift.py --quick
  python tools/ablation_domain_shift.py --max-steps 1000 --batch-size 4
"""

import argparse, sys, os
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adatile.datasets.universal import UniversalDataset
from adatile.logging.run_logger import RunLogger
from adatile.losses import UnifiedLoss
from adatile.engine import build_components, collect_params, StepRunner

try:
    from scipy.stats import pearsonr
except ImportError:
    def pearsonr(x, y):
        xm, ym = x - x.mean(), y - y.mean()
        r = (xm * ym).sum() / (np.sqrt((xm ** 2).sum()) * np.sqrt((ym ** 2).sum()) + 1e-8)
        return r, 0


def parse_args():
    p = argparse.ArgumentParser(description="Domain Shift Ablation")
    p.add_argument("--dataset", type=str, default="E:/A_postgraduate_stude/AdaTile-FastSAM/datasets/loveda")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--keep-ratio", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--unfreeze-layers", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


# ─── Domain filtering ────────────────────────────────────────────────

def filter_by_domain(dataset, domain):
    """Filter UniversalDataset to only include Urban or Rural images."""
    indices = []
    for i in range(len(dataset)):
        name = dataset.pairs[i][2]
        if name.lower().startswith(domain.lower()):
            indices.append(i)
    return torch.utils.data.Subset(dataset, indices)


# ─── Training ────────────────────────────────────────────────────────

def train_spm(backbone, decoder, spm, train_loader, val_loader, device, args, log):
    loss_fn = UnifiedLoss(use_spm=True, num_classes=1, spm_mode="topk", budget_target=args.keep_ratio)
    runner = StepRunner(
        device=device, max_steps=args.max_steps, lr=args.lr,
        patience=9999,  # no early stopping
    )
    runner.setup(backbone, decoder, spm, loss_fn, train_loader, val_loader)
    opt = runner.optimizer
    sch = runner.scheduler
    params = collect_params(backbone, decoder, spm, loss_fn)
    log_interval = max(1, args.max_steps // 20)
    decoder.train(); spm.train()
    best_dice = 0.0

    for step in range(args.max_steps):
        try:
            batch = next(train_iter)
        except (NameError, StopIteration):
            train_iter = iter(train_loader)
            batch = next(train_iter)

        img = batch["images"].to(device)
        gt = batch["masks"].to(device)
        feats = backbone(img)
        lgs = decoder(features=feats)
        imp = spm(feats)
        loss, tm = loss_fn(lgs, gt, imp)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sch.step(); opt.zero_grad()

        if step % log_interval == 0 or step == args.max_steps - 1:
            decoder.eval(); spm.eval()
            ious, dices, imps, covers = [], [], [], []
            with torch.no_grad():
                for vb in val_loader:
                    vi = vb["images"].to(device); vg = vb["masks"].to(device)
                    vf = backbone(vi)
                    vl = decoder(features=vf)
                    vi_imp = spm(vf)
                    _, vm = loss_fn(vl, vg, vi_imp)
                    ious.append(vm["iou"]); dices.append(vm["dice"])
                    imps.append(vm.get("imp_mean", 0))
                    covers.append(vm.get("coverage", 0))
            decoder.train(); spm.train()
            val_iou = np.mean(ious); val_dice = np.mean(dices)
            val_imp = np.mean(imps); val_cover = np.mean(covers)
            log.log(step, loss=loss.item(), iou=tm["iou"], dice=tm["dice"],
                    val_iou=val_iou, val_dice=val_dice,
                    imp_mean=tm.get("imp_mean", 0), coverage=tm.get("coverage", 0),
                    val_imp=val_imp, val_cover=val_cover, lr=opt.param_groups[0]['lr'])
            print(f"  [{step:4d}] loss={loss.item():.3f} val_dice={val_dice:.3f}"
                  f" imp={val_imp:.3f} cover={val_cover:.2%}")
            if val_dice > best_dice: best_dice = val_dice

    return best_dice


# ─── Evaluation (with unique Pearson correlation) ─────────────────────

def evaluate_domain(backbone, decoder, spm, test_loader, device, keep_ratio):
    """Compute all metrics on a test domain, including Pearson correlation."""
    decoder.eval(); spm.eval()
    eps = 1e-8

    all_dices, all_ious, all_coverages, all_actives, all_imps, all_corrs = [], [], [], [], [], []

    with torch.no_grad():
        for vb in test_loader:
            vi = vb["images"].to(device); vg = vb["masks"].to(device)
            feats = backbone(vi)
            lgs = decoder(features=feats)
            imp = spm(feats)

            # GT → binary foreground
            gt_bin = vg.float()
            if gt_bin.dim() == 2:
                gt_bin = gt_bin.unsqueeze(0).unsqueeze(0)
            elif gt_bin.dim() == 3:
                gt_bin = gt_bin.unsqueeze(1)
            if gt_bin.max() > 1:
                gt_bin = (gt_bin > 0).float().clamp(0, 1)

            full_bin = (lgs.sigmoid() > 0.5).float()
            gt_resized = F.interpolate(gt_bin, size=full_bin.shape[-2:], mode="nearest")

            # Full metrics
            fi = (full_bin * gt_resized).sum()
            fu = (full_bin + gt_resized).clamp(0, 1).sum()
            all_dices.append((2 * fi / (full_bin.sum() + gt_resized.sum() + eps)).item())
            all_ious.append((fi / (fu + eps)).item())

            # Top-K sparse by importance
            imp_f = imp[0, 0].reshape(-1)
            n_total = imp_f.shape[0]
            k = max(1, int(n_total * keep_ratio))
            _, idx = imp_f.topk(k)
            keep_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
            keep_mask[idx] = True
            keep_mask = keep_mask.reshape(imp.shape[-2:])
            keep_large = F.interpolate(
                keep_mask.float().unsqueeze(0).unsqueeze(0),
                size=full_bin.shape[-2:], mode="nearest"
            ).squeeze() > 0.5

            gt_in_kept = gt_resized * keep_large.float().unsqueeze(0).unsqueeze(0)
            coverage = (gt_in_kept.sum() / (gt_resized.sum() + eps)).item() if gt_resized.sum() > 0 else 1.0
            all_coverages.append(coverage)
            all_actives.append(keep_mask.float().mean().item())
            all_imps.append(imp.mean().item())

            # Pearson: importance vs GT density
            gt_density = F.interpolate(gt_bin, size=imp.shape[-2:], mode="area")
            imp_np = imp.cpu().squeeze().numpy().flatten()
            gt_np = gt_density.cpu().squeeze().numpy().flatten()
            if np.std(imp_np) > 0 and np.std(gt_np) > 0:
                corr, _ = pearsonr(imp_np, gt_np)
            else:
                corr = 0.0
            all_corrs.append(corr)

    return {
        "dice": np.mean(all_dices), "iou": np.mean(all_ious),
        "coverage": np.mean(all_coverages), "activation": np.mean(all_actives),
        "imp_mean": np.mean(all_imps), "corr": np.mean(all_corrs),
    }


# ─── Main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    log = RunLogger("outputs", "domain_shift", vars(args))
    log.start()

    img_sz = None if args.image_size == 0 else args.image_size
    img_sz_t = None if img_sz is None else (img_sz, img_sz)

    full_train = UniversalDataset(args.dataset, split="train", image_size=img_sz_t, num_classes=None)
    full_val = UniversalDataset(args.dataset, split="val", image_size=img_sz_t, num_classes=full_train.num_classes)
    args.num_classes = full_train.num_classes

    domains = ["Urban", "Rural"]
    domain_loaders = {}
    for domain in domains:
        train_sub = filter_by_domain(full_train, domain)
        val_sub = filter_by_domain(full_val, domain)
        domain_loaders[domain] = {
            "train": torch.utils.data.DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0),
            "val": torch.utils.data.DataLoader(val_sub, batch_size=1, shuffle=False, num_workers=0),
        }
        print(f"Domain [{domain}]: train={len(train_sub)}, val={len(val_sub)}")

    results = []

    # BuildArgs-compatible namespace for shared builder
    class BA:
        image_size = args.image_size
        unfreeze_layers = args.unfreeze_layers
        use_spm = True
        spm_type = "light"

    for train_domain in domains:
        print(f"\n{'='*60}")
        print(f"Training on {train_domain}")
        print(f"{'='*60}")

        backbone, decoder, spm = build_components(BA(), device, 1)

        train_loader = domain_loaders[train_domain]["train"]
        val_loader = domain_loaders[train_domain]["val"]
        best = train_spm(backbone, decoder, spm, train_loader, val_loader, device, args, log)

        for test_domain in domains:
            test_loader = domain_loaders[test_domain]["val"]
            r = evaluate_domain(backbone, decoder, spm, test_loader, device, args.keep_ratio)
            r["train"] = train_domain
            r["test"] = test_domain
            r["best_train_dice"] = best
            results.append(r)
            in_or_cross = "IN-domain" if train_domain == test_domain else "CROSS-domain"
            print(f"  [{in_or_cross}] Train {train_domain} → Test {test_domain}: "
                  f"Dice={r['dice']:.4f} Cover={r['coverage']:.2%} "
                  f"Active={r['activation']:.2%} Corr={r['corr']:.3f}")

        del backbone, decoder, spm
        torch.cuda.empty_cache()

    # ═══ Results Table ═══════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"Domain Shift Ablation Results")
    print(f"{'='*70}")
    print(f"{'Train':<10} {'Test':<10} {'Dice':>8} {'IoU':>8} {'Cover':>8} {'Active':>8} {'Corr':>8}")
    print("-" * 56)
    for r in results:
        print(f"{r['train']:<10} {r['test']:<10} {r['dice']:8.4f} {r['iou']:8.4f} "
              f"{r['coverage']:8.2%} {r['activation']:8.2%} {r['corr']:8.3f}")

    # ═══ Cross-domain Analysis ═══════════════════════════════════════
    urban_in = next(r for r in results if r['train'] == 'Urban' and r['test'] == 'Urban')
    rural_in = next(r for r in results if r['train'] == 'Rural' and r['test'] == 'Rural')
    urban2rural = next(r for r in results if r['train'] == 'Urban' and r['test'] == 'Rural')
    rural2urban = next(r for r in results if r['train'] == 'Rural' and r['test'] == 'Urban')

    drop_u2r = urban_in['coverage'] - urban2rural['coverage']
    drop_r2u = rural_in['coverage'] - rural2urban['coverage']
    drop_dice_u2r = urban_in['dice'] - urban2rural['dice']
    drop_dice_r2u = rural_in['dice'] - rural2urban['dice']
    drop_corr_u2r = urban_in['corr'] - urban2rural['corr']
    drop_corr_r2u = rural_in['corr'] - rural2urban['corr']

    print(f"\n{'='*70}")
    print(f"Cross-Domain Degradation")
    print(f"{'='*70}")
    print(f"  Urban → Rural:  ΔCover={drop_u2r:+.2%}  ΔDice={drop_dice_u2r:+.4f}  ΔCorr={drop_corr_u2r:+.3f}")
    print(f"  Rural  → Urban: ΔCover={drop_r2u:+.2%}  ΔDice={drop_dice_r2u:+.4f}  ΔCorr={drop_corr_r2u:+.3f}")
    avg_drop = (drop_u2r + drop_r2u) / 2
    avg_drop_dice = (drop_dice_u2r + drop_dice_r2u) / 2
    print(f"  Average:         ΔCover={avg_drop:+.2%}  ΔDice={avg_drop_dice:+.4f}")
    print(f"\n{'='*70}")
    if abs(avg_drop) < 0.05:
        print(f"✅ Ada-SPM learns OBJECT DENSITY (ΔCover < 5%)")
        print(f"   Importance maps generalize across domains — core claim supported.")
    elif abs(avg_drop) < 0.15:
        print(f"⚠️  Partial generalization (5% < ΔCover < 15%)")
        print(f"   Ada-SPM captures some density signal but also some dataset bias.")
    else:
        print(f"❌ Ada-SPM overfits to DATASET TEXTURE (ΔCover > 15%)")
        print(f"   Importance maps do NOT generalize. SPM needs redesign.")

    log.log_table("domain_shift", [{k: round(v, 6) if isinstance(v, float) else v
                                      for k, v in r.items()} for r in results])
    log.log_best(avg_coverage_drop=float(avg_drop), avg_dice_drop=float(avg_drop_dice),
                 drop_u2r=float(drop_u2r), drop_r2u=float(drop_r2u))
    log.finish()


if __name__ == "__main__":
    a = parse_args()
    if a.quick:
        a.max_steps = 300; a.image_size = 640
    main()
