#!/usr/bin/env python3
"""
E004: 渐进上采样 vs 单步上采样（修复版 | Fixed version）
=========================================================

修复 | Fix: 训练使用 BCEWithLogitsLoss（上采样 logit，非 prob），与 E002 完全一致。
     Training uses BCEWithLogitsLoss (upsample logits, not probs), identical to E002.

单变量测试 | Single-variable test:
    E004-A: logit → 4× 2× bilinear → BCEWithLogitsLoss
    E004-B: logit → 1× 16× bilinear → BCEWithLogitsLoss  (= 与 E002 相同)
    E002:   logit → 1× 16× bilinear → BCEWithLogitsLoss

控制变量 | Controlled (all identical to E002):
    ✓ 1×1 Conv(1280→1)
    ✓ BCEWithLogitsLoss (logit upsampling, not prob)
    ✓ Adam lr=1e-3, 20 epochs
    ✓ P4 only, no P8, no spatial conv

唯一变化 | Only change:
    E004-A: upsampling = 4×2 bilinear
    E004-B: upsampling = 1×16 bilinear  (E002 复现 | E002 reproduction)

用法 | Usage:
    python tools/eval_e004a_upsample.py              # E004-A
    python tools/eval_e004a_upsample.py --single-step  # E004-B
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.metrics import compute_dice, compute_miou, format_param_count
from adatile.backbone import FastSAMBackbone


class UpsampleProbe(nn.Module):
    """
    上采样探针 | Upsample Probe.
    1×1 Conv（与 E002 参数完全相同），支持两种上采样策略。
    1×1 Conv (same params as E002), supports two upsampling strategies.
    """

    def __init__(self, in_channels: int = 1280):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)

    def forward_logit(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """返回 stride-16 的原始 logit | Return raw logit at stride-16."""
        return self.conv(features["p4"])  # [B, 1, H/16, W/16], NO sigmoid

    def upsample_gradual(self, logit: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """
        E004-A: 4 步渐进上采样 logit | 4-step gradual upsampling of logits.
        logit 上采样后仍然是 logit → 可直接用于 BCEWithLogitsLoss。
        """
        h_src, w_src = logit.shape[2], logit.shape[3]
        h_tgt, w_tgt = target_size

        # 4 步，每步约 2× | 4 steps, ~2× each
        h1, w1 = h_src * 2, w_src * 2
        logit = F.interpolate(logit, size=(h1, w1), mode="bilinear", align_corners=False)
        h2, w2 = h1 * 2, w1 * 2
        logit = F.interpolate(logit, size=(h2, w2), mode="bilinear", align_corners=False)
        h3, w3 = h2 * 2, w2 * 2
        logit = F.interpolate(logit, size=(h3, w3), mode="bilinear", align_corners=False)
        # 最后一步到精确目标尺寸 | Final step to exact target
        logit = F.interpolate(logit, size=(h_tgt, w_tgt), mode="bilinear", align_corners=False)
        return logit

    def upsample_single(self, logit: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """
        E004-B: 单步 16× 上采样 logit（与 E002 完全相同）
        Single-step 16× upsampling of logits (identical to E002).
        """
        return F.interpolate(logit, size=target_size, mode="bilinear", align_corners=False)


def parse_args():
    p = argparse.ArgumentParser(description="E004: Upsampling Strategy Test (FIXED)")
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--single-step", action="store_true",
                   help="E004-B: single 16x (E002 reproduction)")
    return p.parse_args()


def train_one_epoch(probe, backbone, ds, optimizer, device, gradual: bool):
    probe.train()
    total_loss = 0.0
    for idx in tqdm(range(len(ds)), desc="  Train", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        with torch.no_grad():
            features = backbone(image)

        # 与 E002 一致：上采样 logit + BCEWithLogitsLoss
        # Same as E002: upsample logit + BCEWithLogitsLoss
        logit = probe.forward_logit(features)  # [B, 1, H/16, W/16]
        tgt_size = tuple(gt_mask.shape[1:])
        if gradual:
            logit_up = probe.upsample_gradual(logit, tgt_size)
        else:
            logit_up = probe.upsample_single(logit, tgt_size)

        # BCEWithLogitsLoss = sigmoid + BCE，数值稳定 | numerically stable
        loss = F.binary_cross_entropy_with_logits(logit_up.squeeze(1), gt_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(ds)


@torch.no_grad()
def evaluate(probe, backbone, ds, device, gradual: bool):
    probe.eval()
    dices, mious = [], []
    for idx in tqdm(range(len(ds)), desc="  Eval", leave=False):
        sample = ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        features = backbone(image)

        logit = probe.forward_logit(features)
        tgt_size = tuple(gt_mask.shape[1:])
        if gradual:
            logit_up = probe.upsample_gradual(logit, tgt_size)
        else:
            logit_up = probe.upsample_single(logit, tgt_size)

        # sigmoid + 二值化 | sigmoid + binarize
        pred = (torch.sigmoid(logit_up) > 0.5).float()

        dices.append(compute_dice(pred, gt_mask.unsqueeze(0)).item())
        pred_lbl = pred.squeeze(0).squeeze(0).long()
        gt_lbl = gt_mask.squeeze(0).long()
        mious.append(compute_miou(pred_lbl, gt_lbl, num_classes=2)["miou"])
    return {"dice_mean": float(np.mean(dices)), "dice_std": float(np.std(dices)),
            "miou_mean": float(np.mean(mious)), "miou_std": float(np.std(mious))}


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    gradual = not args.single_step
    mode = "gradual 4×2" if gradual else "single 16×"
    tag = "e004a" if gradual else "e004b"

    print("=" * 70)
    print(f"  E004-{'A' if gradual else 'B'}: logit upsample ({mode}) + BCEWithLogitsLoss")
    print("  单变量: 上采样策略 | Variable: upsampling strategy")
    print("  (修复版: BCEWithLogitsLoss, 与 E002 一致 | Fixed: matches E002)")
    print("=" * 70)

    exp_id = generate_exp_id(name=args.name or tag)
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()

    print("\n[1/3] Frozen FastSAM")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    print(f"\n[2/3] UpsampleProbe (1×1 Conv, 1281 params)")
    probe = UpsampleProbe().to(device)
    print(f"  Loss: BCEWithLogitsLoss (same as E002)")

    print(f"\n[3/3] Train ({args.epochs} epochs, {mode})")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")

    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    best_dice, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(probe, backbone, train_ds, optimizer, device, gradual)
        m = evaluate(probe, backbone, val_ds, device, gradual)
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
    final = evaluate(probe, backbone, val_ds, device, gradual)

    # Comparison
    e001, e002 = 0.12, 0.40
    delta = final["dice_mean"] - e002

    print(f"\n{'=' * 70}")
    print(f"  E004-{'A' if gradual else 'B'} 结果 | Results (FIXED)")
    print(f"  {'─' * 50}")
    print(f"  Upsample:     {mode}")
    print(f"  Loss:         BCEWithLogitsLoss (= E002)")
    print(f"  Dice (val):   {final['dice_mean']:.4f} ± {final['dice_std']:.4f}")
    print(f"  mIoU (val):   {final['miou_mean']:.4f} ± {final['miou_std']:.4f}")
    print(f"  ─────────────────────")
    print(f"  E001 (P4 mean):         Dice = {e001:.2f}")
    print(f"  E002 (1×1, 1×16):       Dice = {e002:.2f}")
    print(f"  E004-{'A' if gradual else 'B'} (1×1, {mode}): Dice = {final['dice_mean']:.2f}")
    print(f"  ─────────────────────")
    print(f"  Δ vs E002:  {delta:+.4f}")
    print(f"  ─────────────────────")
    if not gradual:
        # E004-B 应该等于 E002（复现测试 | reproduction test）
        if abs(delta) < 0.03:
            print(f"  E002 复现成功 ✓ | E002 reproduced (|Δ| < 0.03)")
        else:
            print(f"  E002 复现失败 ✗ | E002 NOT reproduced (|Δ| = {abs(delta):.4f})")
    else:
        if delta > 0.08:
            print(f"  结论: 上采样是主要瓶颈 ✓ | Upsampling IS the bottleneck")
        elif delta > 0.02:
            print(f"  结论: 上采样有微弱贡献 | Upsampling has minor effect")
        else:
            print(f"  结论: 上采样不是瓶颈 → 问题在 P4 特征表达")
    print(f"{'=' * 70}")

    recorder.record_metric(f"{tag}/dice_mean", final["dice_mean"], phase="val", tags=[tag, "summary"])
    recorder.record_metric(f"{tag}/delta_vs_e002", delta, phase="val", tags=[tag, "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {config.output_dir}/{exp_id}/")


if __name__ == "__main__":
    main()
