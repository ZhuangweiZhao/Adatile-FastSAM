#!/usr/bin/env python3
"""
E003: P4 + P8 Fusion Probe
=============================

实验目的 | Experiment purpose:
    验证深层语义（P8）是否对建筑分割有额外贡献。
    Verify whether deeper semantics (P8) contribute beyond P4 alone.

结构 | Architecture (~10K params):
    P4 [1280, H/16, W/16]          P8 [1280, H/32, W/32]
     │                               │
     │                         Upsample → H/16
     │                               │
    Conv(1280→4)               Conv(1280→4)
     │                               │
     └─────── Concat ────────────────┘
                 │
           Conv(8→1)
                 │
          Upsample → H, W
                 │
          Binary Mask

实验逻辑 | Experimental logic:
    E002: P4 alone  → Dice=0.40
    E003: P4 + P8   → Dice=?

    δ > 0.15 (Dice > 0.55): 深层语义有显著贡献 | deep semantics matter
    δ ≈ 0.02 (Dice ≈ 0.42): 问题不在语义深度，在解码能力 | issue is decoder, not semantics

用法 | Usage:
    python tools/eval_e003_fusion_probe.py
    python tools/eval_e003_fusion_probe.py --epochs 30 --hidden 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.metrics import compute_dice, compute_miou, format_param_count
from adatile.backbone import FastSAMBackbone
from adatile.decoder.fusion_probe import FusionProbe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E003: P4 + P8 Fusion Probe")
    parser.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=4,
                        help="中间通道数 (每分支) | Hidden channels per branch (default: 4 → ~10K params)")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def train_one_epoch(
    probe: nn.Module,
    backbone: FastSAMBackbone,
    ds: MassachusettsBuildingsDataset,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """训练一轮 | Train one epoch."""
    probe.train()
    total_loss = 0.0

    for idx in tqdm(range(len(ds)), desc="  Train", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)  # [1, H, W]

        with torch.no_grad():
            features = backbone(image)

        # FusionProbe 前向 | Forward
        prob = probe.forward(features)  # [1, 1, H/16, W/16]

        # 上采样到 GT 尺寸 + BCE loss | Upsample + BCE
        prob_up = F.interpolate(
            prob, size=gt_mask.shape[1:], mode="bilinear", align_corners=False,
        )
        # BCE: prob_up 是概率，用 binary_cross_entropy (不是 with_logits)
        loss = F.binary_cross_entropy(prob_up.squeeze(1), gt_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(ds)


@torch.no_grad()
def evaluate(
    probe: nn.Module,
    backbone: FastSAMBackbone,
    ds: MassachusettsBuildingsDataset,
    device: str,
) -> dict:
    """评测 | Evaluate."""
    probe.eval()
    all_dices, all_mious = [], []

    for idx in tqdm(range(len(ds)), desc="  Eval", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)  # [1, H, W]

        features = backbone(image)
        pred = probe.predict(features, target_size=tuple(gt_mask.shape[1:]))  # [1, 1, H, W]

        dice_val = compute_dice(pred, gt_mask.unsqueeze(0))
        all_dices.append(dice_val.item())

        pred_labels = pred.squeeze(0).squeeze(0).long()
        gt_labels = gt_mask.squeeze(0).long()
        miou_result = compute_miou(pred_labels, gt_labels, num_classes=2)
        all_mious.append(miou_result["miou"])

    return {
        "dice_mean": float(np.mean(all_dices)),
        "dice_std": float(np.std(all_dices)),
        "miou_mean": float(np.mean(all_mious)),
        "miou_std": float(np.std(all_mious)),
        "dice_samples": all_dices,
        "miou_samples": all_mious,
    }


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  E003: P4 + P8 Fusion Probe")
    print(f"  hidden={args.hidden} → ~{1280*args.hidden*2 + args.hidden*2*1:,} params")
    print("=" * 70)

    # ── Config ─────────────────────────────────────────────
    exp_id = generate_exp_id(name=args.name or "e003_fusion")
    config = ExperimentConfig(
        exp_id=exp_id, output_dir=args.output_dir,
        dataset_name="Massachusetts_Buildings", dataset_root=args.data_root,
    )
    recorder = ExperimentRecorder(config)
    recorder.record_config()

    # ── Frozen Backbone ────────────────────────────────────
    print("\n[1/4] Frozen FastSAM")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()
    bb_total = sum(p.numel() for p in backbone.model.model.parameters())
    print(f"  FastSAM: {format_param_count(bb_total)} (frozen)")

    # ── FusionProbe ────────────────────────────────────────
    print("\n[2/4] FusionProbe")
    probe = FusionProbe(in_channels=1280, hidden_channels=args.hidden).to(device)
    n_probe = sum(p.numel() for p in probe.parameters())
    p4_n = sum(p.numel() for p in probe.p4_conv.parameters())
    p8_n = sum(p.numel() for p in probe.p8_conv.parameters())
    fusion_n = sum(p.numel() for p in probe.fusion.parameters())
    print(f"  Total: {n_probe:,} params  (P4={p4_n:,}, P8={p8_n:,}, Fusion={fusion_n:,})")
    print(f"  Trainable ratio: {n_probe:,} / {bb_total + n_probe:,} = {100*n_probe/(bb_total+n_probe):.4f}%")

    recorder.logger.log_info(
        "e003/params",
        f"P4={p4_n}, P8={p8_n}, Fusion={fusion_n}, Total={n_probe}, "
        f"ratio={n_probe/(bb_total+n_probe):.6f}",
    )

    # ── Data ───────────────────────────────────────────────
    print("\n[3/4] 数据 | Data")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # ── Train ──────────────────────────────────────────────
    print(f"\n[4/4] 训练 | Training ({args.epochs} epochs)")
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    best_dice, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(probe, backbone, train_ds, optimizer, device)
        metrics = evaluate(probe, backbone, val_ds, device)

        recorder.record_metric("loss/train", train_loss, step=epoch, phase="train")
        recorder.record_metric("dice/val", metrics["dice_mean"], step=epoch, phase="val")
        recorder.record_metric("miou/val", metrics["miou_mean"], step=epoch, phase="val")

        if metrics["dice_mean"] > best_dice:
            best_dice = metrics["dice_mean"]
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  "
              f"Dice={metrics['dice_mean']:.4f}±{metrics['dice_std']:.4f}  "
              f"mIoU={metrics['miou_mean']:.4f}"
              f"{' *' if metrics['dice_mean'] == best_dice else ''}")

    # ── Final Eval ─────────────────────────────────────────
    probe.load_state_dict(best_state)
    final = evaluate(probe, backbone, val_ds, device)

    # ── Comparison ─────────────────────────────────────────
    e001_dice = 0.12   # P4 mean baseline
    e002_dice = 0.40   # P4 + 1×1 Conv LinearProbe
    # E003 结果 | E003 results
    delta_vs_e001 = final["dice_mean"] - e001_dice
    delta_vs_e002 = final["dice_mean"] - e002_dice

    print(f"\n{'=' * 70}")
    print(f"  E003 结果 | E003 Results")
    print(f"  {'─' * 50}")
    print(f"  FusionProbe params:     {n_probe:,} ({100*n_probe/(bb_total+n_probe):.4f}%)")
    print(f"  Dice (val):             {final['dice_mean']:.4f} ± {final['dice_std']:.4f}")
    print(f"  mIoU (val):             {final['miou_mean']:.4f} ± {final['miou_std']:.4f}")
    print(f"  ─────────────────────")
    print(f"  E001 (P4 mean):         Dice = {e001_dice:.4f}")
    print(f"  E002 (P4 LinearProbe):  Dice = {e002_dice:.4f}")
    print(f"  E003 (P4+P8 Fusion):    Dice = {final['dice_mean']:.4f}")
    print(f"  ─────────────────────")
    print(f"  Δ vs E001:  {delta_vs_e001:+.4f}")
    print(f"  Δ vs E002:  {delta_vs_e002:+.4f}")
    print(f"  ─────────────────────")

    if delta_vs_e002 > 0.10:
        conclusion = (
            "P8 深层语义有显著贡献 ✓ | P8 deep semantics contribute significantly ✓\n"
            "→ 方向：继续探索多尺度特征融合 | Direction: continue multi-scale fusion"
        )
    elif delta_vs_e002 > 0.03:
        conclusion = (
            "P8 有微弱贡献 | P8 has marginal contribution\n"
            "→ 问题可能不在语义深度，在解码器容量 | Issue may be decoder capacity, not semantic depth"
        )
    else:
        conclusion = (
            "P8 几乎无额外贡献 | P8 contributes almost nothing\n"
            "→ 问题不在语义层次：要么解码器太弱，要么 16× 上采样损失太大\n"
            "  Issue is NOT semantic depth: either decoder too weak, or 16× upsample loss dominant"
        )
    print(f"  结论 | Conclusion: {conclusion}")
    print(f"{'=' * 70}")

    # ── Record ─────────────────────────────────────────────
    recorder.record_metric("e003/dice_mean", final["dice_mean"], phase="val", tags=["e003", "summary"])
    recorder.record_metric("e003/miou_mean", final["miou_mean"], phase="val", tags=["e003", "summary"])
    recorder.record_metric("e003/delta_vs_e001", delta_vs_e001, phase="val", tags=["e003", "summary"])
    recorder.record_metric("e003/delta_vs_e002", delta_vs_e002, phase="val", tags=["e003", "summary"])

    for i, (d, m) in enumerate(zip(final["dice_samples"], final["miou_samples"])):
        recorder.record_metric("e003/dice_per_sample", d, step=i, phase="val", tags=["e003", "final"])
        recorder.record_metric("e003/miou_per_sample", m, step=i, phase="val", tags=["e003", "final"])

    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {config.output_dir}/{exp_id}/")


if __name__ == "__main__":
    main()
