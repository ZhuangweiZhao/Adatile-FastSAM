#!/usr/bin/env python3
"""
E005-D: CosineAnnealingLR 学习率调度 | Cosine Annealing LR Schedule.
======================================================================

E005 单变量对照实验：唯一变量 = LR Schedule。
Single-variable controlled experiment: only variable = LR Schedule.

E005 (fixed lr=5e-4) vs E005-D (CosineAnnealingLR, same initial lr).

假设 | Hypothesis:
    - E005 后期 Dice 振荡 (0.55~0.63) 部分来自 LR 过大导致过拟合
    - CosineAnnealingLR 平滑降低 LR → 稳定收敛 → +0.02~0.04 Dice
    - E005 后期 Dice oscillation partially from LR too high → overfitting
    - CosineAnnealingLR smooths LR decay → stabilizes convergence

用法 | Usage:
    python tools/eval_e005d_lr_schedule.py
    python tools/eval_e005d_lr_schedule.py --epochs 50 --lr 5e-4
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.metrics import compute_dice, compute_miou, format_param_count
from adatile.backbone import FastSAMBackbone
from adatile.decoder.light_decoder import LightDecoder


def parse_args():
    p = argparse.ArgumentParser(description="E005-D: CosineAnnealingLR vs Fixed LR")
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Initial learning rate (CosineAnnealingLR start)")
    p.add_argument("--lr-min", type=float, default=1e-6,
                   help="Minimum learning rate (CosineAnnealingLR eta_min)")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def train_one_epoch(decoder, backbone, ds, optimizer, device):
    """训练一个 epoch | Train one epoch. (与 E005 完全相同 | Identical to E005)"""
    decoder.train()
    total_loss = 0.0
    for idx in tqdm(range(len(ds)), desc="  Train", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        with torch.no_grad():
            features = backbone(image)
        logit = decoder.forward(features, target_size=tuple(gt_mask.shape[1:]))
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(1), gt_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(ds)


@torch.no_grad()
def evaluate(decoder, backbone, ds, device):
    """评测 | Evaluate. (与 E005 完全相同 | Identical to E005)"""
    decoder.eval()
    dices, mious = [], []
    for idx in tqdm(range(len(ds)), desc="  Eval", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        features = backbone(image)
        pred = decoder.predict(features, target_size=tuple(gt_mask.shape[1:]))
        dices.append(compute_dice(pred, gt_mask.unsqueeze(0)).item())
        pred_lbl = pred.squeeze(0).squeeze(0).long()
        gt_lbl = gt_mask.squeeze(0).long()
        mious.append(compute_miou(pred_lbl, gt_lbl, num_classes=2)["miou"])
    return {"dice_mean": float(np.mean(dices)), "dice_std": float(np.std(dices)),
            "miou_mean": float(np.mean(mious)), "miou_std": float(np.std(mious))}


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  E005-D: CosineAnnealingLR — 单变量对照 | Single-Variable Control")
    print("  唯一变量 | Only variable: LR Schedule (fixed → CosineAnnealing)")
    print("=" * 70)

    exp_id = generate_exp_id(name=args.name or "e005d_cosine_lr")
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()

    # ── Frozen Backbone (与 E005 完全相同) ──
    print("\n[1/5] Frozen FastSAM")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()
    bb_total = sum(p.numel() for p in backbone.model.model.parameters())
    print(f"  FastSAM: {format_param_count(bb_total)} (frozen)")

    # ── LightDecoder (与 E005 完全相同) ──
    print("\n[2/5] LightDecoder")
    decoder = LightDecoder(in_channels=1280).to(device)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder params: {format_param_count(n_dec)} ({n_dec:,})")
    print(f"  Trainable/Total: {n_dec:,} / {bb_total + n_dec:,} = {100*n_dec/(bb_total+n_dec):.2f}%")
    recorder.logger.log_info("e005d/params",
        f"decoder={n_dec}, backbone={bb_total}, ratio={n_dec/(bb_total+n_dec):.6f}")

    # ── Data (与 E005 完全相同) ──
    print("\n[3/5] Data")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")

    # ── Optimizer + Scheduler (唯一变量) ──
    print(f"\n[4/5] Optimizer + CosineAnnealingLR (lr={args.lr}→{args.lr_min})")
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    # 余弦退火调度器 | Cosine annealing scheduler
    # T_max=epochs: 完整余弦周期 | Full cosine cycle over all epochs
    # eta_min: 最终学习率下限 | Final LR floor
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min
    )
    recorder.logger.log_info("e005d/scheduler",
        f"CosineAnnealingLR(T_max={args.epochs}, eta_min={args.lr_min})")

    print(f"  Scheduler: CosineAnnealingLR(T_max={args.epochs}, eta_min={args.lr_min})")
    print(f"  LR path: {args.lr:.1e} → ... → {args.lr_min:.1e}")

    # ── Train ──
    print(f"\n[5/5] Train ({args.epochs} epochs)")
    best_dice, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        # 记录当前 LR | Log current LR
        current_lr = scheduler.get_last_lr()[0]

        loss = train_one_epoch(decoder, backbone, train_ds, optimizer, device)
        m = evaluate(decoder, backbone, val_ds, device)

        # 记录指标 | Record metrics
        recorder.record_metric("loss/train", loss, step=epoch, phase="train")
        recorder.record_metric("dice/val", m["dice_mean"], step=epoch, phase="val")
        recorder.record_metric("miou/val", m["miou_mean"], step=epoch, phase="val")
        recorder.record_metric("lr", current_lr, step=epoch, phase="train")

        # 更新 best | Update best
        if m["dice_mean"] > best_dice:
            best_dice = m["dice_mean"]
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}

        # Step scheduler
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{args.epochs}  lr={current_lr:.2e}  loss={loss:.4f}  "
              f"Dice={m['dice_mean']:.4f}±{m['dice_std']:.4f}  "
              f"mIoU={m['miou_mean']:.4f}"
              f"{' *' if m['dice_mean'] == best_dice else ''}")

    # ── Final evaluation with best state ──
    decoder.load_state_dict(best_state)
    final = evaluate(decoder, backbone, val_ds, device)

    # ── Comparison ──
    e001, e002, e005_best, e005_final = 0.12, 0.40, 0.628, 0.598
    delta_vs_e005_best = final["dice_mean"] - e005_best
    delta_vs_e005_final = final["dice_mean"] - e005_final

    print(f"\n{'=' * 70}")
    print(f"  E005-D 结果 | Results (CosineAnnealingLR)")
    print(f"  {'─' * 50}")
    print(f"  Best Dice (val):     {final['dice_mean']:.4f} ± {final['dice_std']:.4f}")
    print(f"  Best mIoU (val):     {final['miou_mean']:.4f} ± {final['miou_std']:.4f}")
    print(f"  ─────────────────────")
    print(f"  E001 Random:               {e001:.2f}")
    print(f"  E002 LinearProbe:          {e002:.2f}")
    print(f"  E005 LightDecoder (best):  {e005_best:.3f}")
    print(f"  E005 LightDecoder (final): {e005_final:.3f}")
    print(f"  E005-D CosineLR (best):    {final['dice_mean']:.3f}")
    print(f"  ─────────────────────")
    print(f"  Δ vs E005 best:   {delta_vs_e005_best:+.4f}")
    print(f"  Δ vs E005 final:  {delta_vs_e005_final:+.4f}")
    print(f"  ─────────────────────")

    # 解读 | Interpretation
    if delta_vs_e005_best > 0.02:
        print(f"  ✓ CosineLR 显著优于固定 LR → LR schedule 是过拟合因素之一")
    elif delta_vs_e005_best > 0.005:
        print(f"  △ CosineLR 微弱改善 → LR schedule 有帮助但非主要因素")
    elif delta_vs_e005_final > 0.01:
        print(f"  △ 未超 E005 best，但 final 更稳定 → LR schedule 减轻过拟合")
    else:
        print(f"  → 过拟合主要来自数据集大小 (137)，LR schedule 无法根治")

    print(f"{'=' * 70}")

    # 记录总结 | Record summary
    recorder.record_metric("e005d/dice_mean", final["dice_mean"], phase="val",
                           tags=["e005d", "summary"])
    recorder.record_metric("e005d/delta_vs_e005_best", delta_vs_e005_best,
                           phase="val", tags=["e005d", "summary"])
    recorder.record_metric("e005d/delta_vs_e005_final", delta_vs_e005_final,
                           phase="val", tags=["e005d", "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {config.output_dir}/{exp_id}/")


if __name__ == "__main__":
    main()
