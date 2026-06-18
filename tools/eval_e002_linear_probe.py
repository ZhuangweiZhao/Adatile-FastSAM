#!/usr/bin/env python3
"""
E002: P4 + 1×1 Conv Linear Probe
===================================

实验目的 | Experiment purpose:
    验证冻结的 FastSAM P4 特征是否已经包含足够的分割信息。
    Verify whether frozen FastSAM P4 features already contain sufficient segmentation info.

假设 | Hypothesis:
    如果只用 1281 个可训练参数就能达到 Dice > 0.6，
    则 FastSAM backbone 本身已经很有价值，后续重点放在 Decoder。
    If 1281 trainable params achieve Dice > 0.6,
    FastSAM backbone is valuable, future work on Decoder.

架构 | Architecture:
    FastSAM (Frozen, 72M) → P4 [B, 1280, H/16, W/16]
                                   │
                              1×1 Conv (1280→1)  ← 1281 trainable params
                                   │
                              Upsample → [B, 1, H, W]
                                   │
                              Sigmoid → Binary Mask

对比 | Comparison:
    E001: P4 mean → threshold          Dice=0.12 (随机 | random)
    E002: P4 + 1×1 Conv Linear Probe   Dice=?    (本次实验 | this experiment)

用法 | Usage:
    python tools/eval_e002_linear_probe.py
    python tools/eval_e002_linear_probe.py --epochs 30 --lr 0.001
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

# 将项目根目录加入路径 | Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.metrics import compute_dice, compute_miou, format_param_count
from adatile.backbone import FastSAMBackbone
from adatile.decoder import LinearProbe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E002: P4 + 1x1 Conv Linear Probe")
    parser.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮次 | Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率 | Learning rate")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def train_one_epoch(
    probe: nn.Module,
    backbone: FastSAMBackbone,
    ds: MassachusettsBuildingsDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """训练一轮 | Train one epoch. 返回平均 loss | Returns average loss."""
    probe.train()
    total_loss = 0.0

    for idx in tqdm(range(len(ds)), desc="  Train", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)  # [1, H, W]

        # 冻结的 backbone 前向 | Frozen backbone forward
        with torch.no_grad():
            features = backbone(image)

        # LinearProbe 前向 | LinearProbe forward
        logit = probe.conv(features["p4"])  # [1, 1, H/16, W/16]

        # 上采样 logit 到 GT 尺寸 | Upsample logit to GT size
        logit_up = F.interpolate(
            logit, size=gt_mask.shape[1:], mode="bilinear", align_corners=False,
        )

        # BCE loss（直接对 logit，不用 sigmoid）
        # BCE loss (on logit, no sigmoid)
        # gt_mask: [1, H, W], logit_up: [1, 1, H, W] → squeeze to match
        loss = criterion(logit_up.squeeze(1), gt_mask)

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
) -> dict[str, float]:
    """评测 | Evaluate. 返回 {dice_mean, miou_mean, dice_per_sample, miou_per_sample}."""
    probe.eval()
    all_dices: list[float] = []
    all_mious: list[float] = []

    for idx in tqdm(range(len(ds)), desc="  Eval", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)  # [1, H, W]

        features = backbone(image)

        # 预测 | Predict
        pred = probe.predict(features, target_size=tuple(gt_mask.shape[1:]))  # [1, 1, H, W]

        # Dice | Dice coefficient
        dice_val = compute_dice(pred, gt_mask.unsqueeze(0))
        all_dices.append(dice_val.item())

        # mIoU (二分类) | mIoU (binary)
        pred_labels = pred.squeeze(0).squeeze(0).long()  # [H, W]
        gt_labels = gt_mask.squeeze(0).long()
        miou_result = compute_miou(pred_labels, gt_labels, num_classes=2)
        all_mious.append(miou_result["miou"])

    return {
        "dice_mean": float(np.mean(all_dices)),
        "dice_std": float(np.std(all_dices)),
        "dice_min": float(np.min(all_dices)),
        "dice_max": float(np.max(all_dices)),
        "miou_mean": float(np.mean(all_mious)),
        "miou_std": float(np.std(all_mious)),
        "dice_samples": all_dices,
        "miou_samples": all_mious,
    }


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  E002: P4 + 1×1 Conv Linear Probe")
    print("  假设 | Hypothesis: 1281 参数能否让 Dice 突破 0.6?")
    print("=" * 70)

    # ── 实验配置 | Experiment Config ──────────────────────────
    exp_id = generate_exp_id(name=args.name or "e002_linear_probe")
    config = ExperimentConfig(
        exp_id=exp_id,
        output_dir=args.output_dir,
        dataset_name="Massachusetts_Buildings",
        dataset_root=args.data_root,
    )
    recorder = ExperimentRecorder(config)
    recorder.record_config()

    # ── Backbone (Frozen) ────────────────────────────────────
    print("\n[1/4] 加载 Frozen FastSAM | Loading Frozen FastSAM")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()
    bb_total = sum(p.numel() for p in backbone.model.model.parameters())
    bb_trainable = sum(
        p.numel() for p in backbone.model.model.parameters() if p.requires_grad
    )
    print(f"  FastSAM: {format_param_count(bb_total)} total, "
          f"{format_param_count(bb_trainable)} trainable")

    # ── LinearProbe ──────────────────────────────────────────
    print("\n[2/4] 创建 LinearProbe | Creating LinearProbe")
    p4_channels = 1280  # FastSAM-x P4
    probe = LinearProbe(in_channels=p4_channels).to(device)
    probe_params = sum(p.numel() for p in probe.parameters())
    print(f"  LinearProbe: {probe_params} params (1280×1 Conv + bias)")
    print(f"  Trainable/Total: {probe_params:,} / {bb_total + probe_params:,} "
          f"= {100 * probe_params / (bb_total + probe_params):.4f}%")

    recorder.logger.log_info(
        "e002/params",
        f"FastSAM={bb_total}, LinearProbe={probe_params}, "
        f"ratio={probe_params/(bb_total+probe_params):.6f}",
    )

    # ── 数据集 | Datasets ─────────────────────────────────────
    print("\n[3/4] 加载数据 | Loading Data")
    train_ds = MassachusettsBuildingsDataset(
        root_dir=args.data_root, split="train", tile_size=None,
    )
    val_ds = MassachusettsBuildingsDataset(
        root_dir=args.data_root, split="val", tile_size=None,
    )
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # ── 训练 | Training ───────────────────────────────────────
    print(f"\n[4/4] 训练 LinearProbe | Training LinearProbe ({args.epochs} epochs)")
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_dice = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            probe, backbone, train_ds, optimizer, criterion, device,
        )

        # 每个 epoch 在验证集上评测 | Evaluate on val every epoch
        metrics = evaluate(probe, backbone, val_ds, device)

        recorder.record_metric("loss/train", train_loss, step=epoch, phase="train")
        recorder.record_metric("dice/val", metrics["dice_mean"], step=epoch, phase="val")
        recorder.record_metric("miou/val", metrics["miou_mean"], step=epoch, phase="val")

        # 保存最佳 | Save best
        if metrics["dice_mean"] > best_dice:
            best_dice = metrics["dice_mean"]
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"loss={train_loss:.4f}  "
              f"Dice={metrics['dice_mean']:.4f}±{metrics['dice_std']:.4f}  "
              f"mIoU={metrics['miou_mean']:.4f}"
              f"{' *' if metrics['dice_mean'] == best_dice else ''}")

    # ── 最佳模型评测 | Best Model Evaluation ──────────────────
    probe.load_state_dict(best_state)
    final_metrics = evaluate(probe, backbone, val_ds, device)

    # ── E001 对比 | E001 Comparison ───────────────────────────
    e001_dice = 0.12  # 上一轮实验结果 | Previous experiment result
    delta = final_metrics["dice_mean"] - e001_dice

    print(f"\n{'=' * 70}")
    print(f"  E002 结果 | E002 Results")
    print(f"  {'─' * 50}")
    print(f"  LinearProbe params:     {probe_params} (0.0018% of total)")
    print(f"  Best epoch Dice:        {best_dice:.4f}")
    print(f"  Final Dice (val):       {final_metrics['dice_mean']:.4f} ± {final_metrics['dice_std']:.4f}")
    print(f"  Final mIoU (val):       {final_metrics['miou_mean']:.4f} ± {final_metrics['miou_std']:.4f}")
    print(f"  ─────────────────────")
    print(f"  E001 Dice (baseline):   {e001_dice:.4f}")
    print(f"  Δ Dice (E002 - E001):   {delta:+.4f}")
    if delta > 0.48:
        conclusion = "FastSAM P4 特征已含分割信息 ✓ | FastSAM P4 features encode segmentation info ✓"
    elif delta > 0.1:
        conclusion = "P4 有一定信息但不够 → E003 P4+P8 融合 | P4 has some info but not enough → E003 fusion"
    else:
        conclusion = "P4 不适合建筑分割 → 需要更高层特征或不同 backbone | P4 insufficient → need higher-level features"
    print(f"  ─────────────────────")
    print(f"  结论 | Conclusion: {conclusion}")
    print(f"{'=' * 70}")

    # ── 记录最终结果 | Record Final Results ───────────────────
    for i, (d, m) in enumerate(zip(final_metrics["dice_samples"], final_metrics["miou_samples"])):
        recorder.record_metric("e002/dice_per_sample", d, step=i, phase="val", tags=["e002", "final"])
        recorder.record_metric("e002/miou_per_sample", m, step=i, phase="val", tags=["e002", "final"])

    recorder.record_metric("e002/dice_mean", final_metrics["dice_mean"], phase="val", tags=["e002", "summary"])
    recorder.record_metric("e002/miou_mean", final_metrics["miou_mean"], phase="val", tags=["e002", "summary"])
    recorder.record_metric("e002/delta_vs_e001", delta, phase="val", tags=["e002", "summary"])
    recorder.finalize()
    recorder.close()

    print(f"\n  Results saved to: {config.output_dir}/{exp_id}/")


if __name__ == "__main__":
    main()
