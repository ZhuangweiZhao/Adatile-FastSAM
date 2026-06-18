#!/usr/bin/env python3
"""
E003.5: P4 + 3×3 Conv → 1×1 Conv (空间建模探针)
====================================================

实验目的 | Experiment purpose:
    隔离测试"局部空间建模"这一单一变量的贡献。
    Isolate the contribution of local spatial modeling alone.

控制变量 | Controlled variables (unchanged from E002):
    ✓ P4 特征层         | P4 feature level
    ✓ 16× bilinear 上采样 | 16× bilinear upsampling
    ✓ 训练数据/epoch/lr  | Training data/epoch/lr

唯一变化 | Only change:
    E002:  1×1 Conv(1280→1)             ← 逐点分类，无空间上下文
    E003.5: 3×3 Conv(1280→4) → 1×1      ← 3×3 邻域，有局部空间建模

参数对比 | Parameter comparison:
    E002:   1,281  (1280×1×1 + 1)
    E003.5: ~46K   (1280×4×9 + 4 + 4×1 + 1)

预期 | Hypothesis:
    如果 Dice: 0.40 → 0.50+  → P4 包含局部空间结构，1×1 浪费了它
    如果 Dice: 0.40 → 0.40   → P4 逐点特征已经足够，瓶颈在 16× 上采样

用法 | Usage:
    python tools/eval_e003_5_spatial.py
    python tools/eval_e003_5_spatial.py --hidden 8
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


class SpatialProbe(nn.Module):
    """
    空间探针 | Spatial Probe.

    E002 的 LinearProbe 用 1×1 Conv → 逐点分类，无空间上下文。
    这个 SpatialProbe 在 1×1 之前加一个 3×3 Conv，
    给每个像素 3×3 邻域信息（在 stride-16 特征层上约为 48×48 像素的感受野）。

    E002's LinearProbe uses 1×1 Conv → point-wise, no spatial context.
    This SpatialProbe adds a 3×3 Conv before the 1×1,
    giving each pixel a 3×3 neighborhood (≈48×48 pixel receptive field at stride-16).

    结构 | Architecture:
        P4 [1280, H/16, W/16]
             │
        3×3 Conv(1280→hidden, padding=1)  ← 局部空间建模 | local spatial modeling
             │
        1×1 Conv(hidden→1)                ← 逐点投影 | point-wise projection
             │
        Bilinear Upsample                  ← 与 E002 相同的 16× | same as E002
             │
        Binary Mask
    """

    def __init__(self, in_channels: int = 1280, hidden_channels: int = 4):
        super().__init__()
        self.spatial = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.proj = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

        n_spatial = sum(p.numel() for p in self.spatial.parameters())
        n_proj = sum(p.numel() for p in self.proj.parameters())
        n_total = n_spatial + n_proj
        print(f"  SpatialProbe: 3×3({in_channels}→{hidden_channels}) = {n_spatial:,} params")
        print(f"                1×1({hidden_channels}→1) = {n_proj:,} params")
        print(f"                Total = {n_total:,} params")

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        x = features["p4"]                       # [B, 1280, H/16, W/16]
        x = self.spatial(x)                       # [B, hidden, H/16, W/16]
        x = torch.sigmoid(self.proj(x))           # [B, 1, H/16, W/16]
        return x

    def predict(self, features: dict[str, torch.Tensor], target_size: tuple[int, int]) -> torch.Tensor:
        prob = self.forward(features)
        prob_up = F.interpolate(prob, size=target_size, mode="bilinear", align_corners=False)
        return (prob_up > 0.5).float()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E003.5: 3×3 Spatial Probe")
    parser.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=4,
                        help="3×3 Conv 隐藏通道 (default: 4 → ~46K params)")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def train_one_epoch(probe, backbone, ds, optimizer, device):
    probe.train()
    total_loss = 0.0
    for idx in tqdm(range(len(ds)), desc="  Train", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        with torch.no_grad():
            features = backbone(image)
        prob = probe.forward(features)
        prob_up = F.interpolate(prob, size=gt_mask.shape[1:], mode="bilinear", align_corners=False)
        loss = F.binary_cross_entropy(prob_up.squeeze(1), gt_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(ds)


@torch.no_grad()
def evaluate(probe, backbone, ds, device):
    probe.eval()
    dices, mious = [], []
    for idx in tqdm(range(len(ds)), desc="  Eval", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        features = backbone(image)
        pred = probe.predict(features, target_size=tuple(gt_mask.shape[1:]))
        dices.append(compute_dice(pred, gt_mask.unsqueeze(0)).item())
        pred_lbl = pred.squeeze(0).squeeze(0).long()
        gt_lbl = gt_mask.squeeze(0).long()
        mious.append(compute_miou(pred_lbl, gt_lbl, num_classes=2)["miou"])
    return {
        "dice_mean": float(np.mean(dices)), "dice_std": float(np.std(dices)),
        "miou_mean": float(np.mean(mious)), "miou_std": float(np.std(mious)),
        "dice_samples": dices, "miou_samples": mious,
    }


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  E003.5: P4 + 3×3 Conv → 1×1 Conv (Spatial Probe)")
    print("  单变量测试: 局部空间建模 | Single variable: local spatial modeling")
    print("=" * 70)

    # Config
    exp_id = generate_exp_id(name=args.name or "e003_5_spatial")
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()

    # Frozen Backbone
    print("\n[1/3] Frozen FastSAM")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()
    bb_total = sum(p.numel() for p in backbone.model.model.parameters())
    print(f"  FastSAM: {format_param_count(bb_total)} (frozen)")

    # SpatialProbe
    print("\n[2/3] SpatialProbe")
    probe = SpatialProbe(in_channels=1280, hidden_channels=args.hidden).to(device)
    n_probe = sum(p.numel() for p in probe.parameters())
    recorder.logger.log_info("e003_5/params", f"params={n_probe}, hidden={args.hidden}")

    # Data
    print("\n[3/3] Train + Eval")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")
    print(f"  Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # Train
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    best_dice, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(probe, backbone, train_ds, optimizer, device)
        m = evaluate(probe, backbone, val_ds, device)
        recorder.record_metric("loss/train", loss, step=epoch, phase="train")
        recorder.record_metric("dice/val", m["dice_mean"], step=epoch, phase="val")
        recorder.record_metric("miou/val", m["miou_mean"], step=epoch, phase="val")
        if m["dice_mean"] > best_dice:
            best_dice = m["dice_mean"]
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}
        print(f"  Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  "
              f"Dice={m['dice_mean']:.4f}±{m['dice_std']:.4f}  "
              f"mIoU={m['miou_mean']:.4f}"
              f"{' *' if m['dice_mean'] == best_dice else ''}")

    probe.load_state_dict(best_state)
    final = evaluate(probe, backbone, val_ds, device)

    # Comparison
    e001_dice, e002_dice = 0.12, 0.40
    delta_vs_e002 = final["dice_mean"] - e002_dice

    print(f"\n{'=' * 70}")
    print(f"  E003.5 结果 | Results")
    print(f"  {'─' * 50}")
    print(f"  Dice (val):  {final['dice_mean']:.4f} ± {final['dice_std']:.4f}")
    print(f"  mIoU (val):  {final['miou_mean']:.4f} ± {final['miou_std']:.4f}")
    print(f"  ─────────────────────")
    print(f"  E001 (P4 mean):         Dice = {e001_dice:.2f}")
    print(f"  E002 (1×1 Conv):        Dice = {e002_dice:.2f}")
    print(f"  E003.5 (3×3→1×1 Conv):  Dice = {final['dice_mean']:.2f}")
    print(f"  ─────────────────────")
    print(f"  Δ vs E002:  {delta_vs_e002:+.4f}")
    print(f"  ─────────────────────")
    if delta_vs_e002 > 0.05:
        print(f"  结论: 局部空间建模有贡献 ✓ → P4 包含可利用的空间结构")
    elif delta_vs_e002 > 0.01:
        print(f"  结论: 局部空间建模有微弱贡献 → 可能有帮助但非主要瓶颈")
    else:
        print(f"  结论: 局部空间建模无贡献 → 瓶颈在 16× 上采样损失或逐点特征已够用")
    print(f"{'=' * 70}")

    recorder.record_metric("e003_5/dice_mean", final["dice_mean"], phase="val", tags=["e003_5", "summary"])
    recorder.record_metric("e003_5/delta_vs_e002", delta_vs_e002, phase="val", tags=["e003_5", "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {config.output_dir}/{exp_id}/")


if __name__ == "__main__":
    main()
