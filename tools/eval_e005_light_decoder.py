#!/usr/bin/env python3
"""
E005: LightDecoder Baseline
=============================

真正的 Decoder：P4 + 多层 Conv + 渐进上采样。
A real Decoder: P4 + multi-layer Conv + gradual upsampling.

回答核心问题 | Answers the core question:
    FastSAM P4 特征 + 合理 Decoder 后，Dice 能到多少？
    With a proper decoder, what Dice can FastSAM P4 achieve?

结构 | Architecture (~800K params):
    P4 [1280, H/16] → Conv(1280→64)+BN+ReLU
        → 2× up → Conv(64→64)+BN+ReLU
        → 2× up → Conv(64→32)+BN+ReLU
        → 2× up → Conv(32→32)+BN+ReLU
        → 2× up → Conv(32→1) → Mask

对比 | Comparison:
    E002 (LinearProbe, 1.2K):  Dice = 0.40
    E005 (LightDecoder, 800K): Dice = ?

如果 Dice > 0.65 → P4 特征很有价值，Decoder 是瓶颈，后续做 Proto/SPM
如果 Dice ≈ 0.45 → P4 特征本身有限，需要回头研究特征层

用法 | Usage:
    python tools/eval_e005_light_decoder.py
    python tools/eval_e005_light_decoder.py --epochs 50 --lr 0.0005
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
    p = argparse.ArgumentParser(description="E005: LightDecoder Baseline")
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def train_one_epoch(decoder, backbone, ds, optimizer, device):
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
    print("  E005: LightDecoder Baseline")
    print("  核心问题: FastSAM P4 + 合理 Decoder → Dice 上限是多少?")
    print("=" * 70)

    exp_id = generate_exp_id(name=args.name or "e005_light_decoder")
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()

    # Frozen Backbone
    print("\n[1/4] Frozen FastSAM")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()
    bb_total = sum(p.numel() for p in backbone.model.model.parameters())
    print(f"  FastSAM: {format_param_count(bb_total)} (frozen)")

    # LightDecoder
    print("\n[2/4] LightDecoder")
    decoder = LightDecoder(in_channels=1280).to(device)
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder params: {format_param_count(n_dec)} ({n_dec:,})")
    print(f"  Trainable/Total: {n_dec:,} / {bb_total + n_dec:,} = {100*n_dec/(bb_total+n_dec):.2f}%")
    recorder.logger.log_info("e005/params", f"decoder={n_dec}, backbone={bb_total}, ratio={n_dec/(bb_total+n_dec):.6f}")

    # Data
    print("\n[3/4] Data")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")

    # Train
    print(f"\n[4/4] Train ({args.epochs} epochs, lr={args.lr})")
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    best_dice, best_state = 0.0, None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(decoder, backbone, train_ds, optimizer, device)
        m = evaluate(decoder, backbone, val_ds, device)
        recorder.record_metric("loss/train", loss, step=epoch, phase="train")
        recorder.record_metric("dice/val", m["dice_mean"], step=epoch, phase="val")
        recorder.record_metric("miou/val", m["miou_mean"], step=epoch, phase="val")
        if m["dice_mean"] > best_dice:
            best_dice = m["dice_mean"]
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
        print(f"  Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  "
              f"Dice={m['dice_mean']:.4f}±{m['dice_std']:.4f}  "
              f"mIoU={m['miou_mean']:.4f}"
              f"{' *' if m['dice_mean'] == best_dice else ''}")

    decoder.load_state_dict(best_state)
    final = evaluate(decoder, backbone, val_ds, device)

    # Comparison
    e001, e002 = 0.12, 0.40
    delta = final["dice_mean"] - e002

    print(f"\n{'=' * 70}")
    print(f"  E005 结果 | Results")
    print(f"  {'─' * 50}")
    print(f"  Decoder params:  {format_param_count(n_dec)}")
    print(f"  Dice (val):      {final['dice_mean']:.4f} ± {final['dice_std']:.4f}")
    print(f"  mIoU (val):      {final['miou_mean']:.4f} ± {final['miou_std']:.4f}")
    print(f"  ─────────────────────")
    print(f"  E002 (LinearProbe, 1.2K):  Dice = {e002:.2f}")
    print(f"  E005 (LightDecoder, 800K): Dice = {final['dice_mean']:.2f}")
    print(f"  ─────────────────────")
    print(f"  Δ vs E002:  {delta:+.4f}")
    print(f"  ─────────────────────")
    if delta > 0.25:
        print(f"  ✓ P4 特征有巨大潜力 → Decoder 是瓶颈 → Proto/SPM 有空间")
    elif delta > 0.10:
        print(f"  ✓ P4 特征有价值，Decoder 显著改善 → 继续优化 Decoder")
    elif delta > 0.03:
        print(f"  △ P4 特征有限，Decoder 有微弱帮助 → 考虑特征层改进")
    else:
        print(f"  ✗ P4 特征已到上限，Decoder 无法突破 → 需要 Proto 或更强的特征")
    print(f"{'=' * 70}")

    recorder.record_metric("e005/dice_mean", final["dice_mean"], phase="val", tags=["e005", "summary"])
    recorder.record_metric("e005/delta_vs_e002", delta, phase="val", tags=["e005", "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {config.output_dir}/{exp_id}/")


if __name__ == "__main__":
    main()
