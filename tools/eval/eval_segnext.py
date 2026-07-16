#!/usr/bin/env python3
"""
SegNeXt 评估 — NEU_Seg 数据集 | SegNeXt Evaluation — NEU_Seg Dataset.
=====================================================================

用法 | Usage::

    python tools/eval/eval_segnext.py --checkpoint runs/segnext_tiny_NEUSeg_0716_1600/best_model.pt
    python tools/eval/eval_segnext.py --checkpoint best_model.pt --save-vis
"""

from __future__ import annotations

import sys, argparse, json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from adatile.backbone.mscan import MSCAN
from adatile.decoder.ham_head import LightHamHead
from adatile.datasets.neu_seg import NEUSegDataset

NUM_CLASSES = 4
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]

# ImageNet 归一化常数 (与 train_segnext.py 一致) | Must match train_segnext.py
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_img(img: torch.Tensor, mode: str) -> torch.Tensor:
    """图像归一化 (必须与训练一致) | Image normalization (must match training)."""
    if mode == "imagenet":
        mean = img.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = img.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (img - mean) / std
    return img


@torch.no_grad()
def evaluate(backbone: MSCAN, head: LightHamHead,
             dataset: NEUSegDataset, device: torch.device,
             img_norm: str = "unit") -> dict:
    """完整评估 | Full evaluation."""
    backbone.eval()
    head.eval()

    per_class_intersection = np.zeros(NUM_CLASSES, dtype=np.float64)
    per_class_union = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_correct = 0
    total_pixels = 0
    per_sample_ious = []

    for idx in tqdm(range(len(dataset)), desc="Evaluating"):
        try:
            sample = dataset[idx]
        except (ValueError, OSError, FileNotFoundError):
            continue

        img = sample["image"].unsqueeze(0).to(device)
        img = normalize_img(img, img_norm)
        gt_raw = sample["masks"]
        if isinstance(gt_raw, torch.Tensor):
            gt_raw = gt_raw.numpy()
        gt = gt_raw.astype(np.int64)
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt.squeeze(0)                           # [1, H, W] → [H, W]

        feats = backbone(img)
        logits = head(feats[1:])
        pred = F.softmax(logits, dim=1)
        pred_up = F.interpolate(pred, size=tuple(gt.shape), mode="bilinear",
                                align_corners=False)
        pred_cls = pred_up.argmax(dim=1).squeeze(0).cpu().numpy()

        for c in range(NUM_CLASSES):
            pred_c = (pred_cls == c)
            gt_c = (gt == c)
            per_class_intersection[c] += (pred_c & gt_c).sum()
            per_class_union[c] += (pred_c | gt_c).sum()

        total_correct += (pred_cls == gt).sum()
        total_pixels += gt.size

        # Per-sample IoU
        sample_ious = []
        for c in range(NUM_CLASSES):
            inter = (pred_cls == c) & (gt == c)
            union = (pred_cls == c) | (gt == c)
            if union.sum() > 0:  # class present in this sample's GT or prediction
                sample_ious.append(inter.sum() / union.sum())
        if sample_ious:
            per_sample_ious.append(np.mean(sample_ious))

    ious = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        union = per_class_union[c]
        ious[c] = per_class_intersection[c] / max(union, 1)

    return {
        "mIoU": round(float(np.mean(ious)), 6),
        "pixel_accuracy": round(float(total_correct / max(total_pixels, 1)), 6),
        "per_class_IoU": {name: round(float(ious[i]), 4)
                          for i, name in enumerate(CLASS_NAMES)},
        "sample_mIoU_mean": round(float(np.mean(per_sample_ious)), 6)
               if per_sample_ious else 0.0,
        "sample_mIoU_median": round(float(np.median(per_sample_ious)), 6)
               if per_sample_ious else 0.0,
        "n_evaluated": len(per_sample_ious),
    }


def main():
    p = argparse.ArgumentParser(description="SegNeXt NEU_Seg Evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-vis", action="store_true")
    p.add_argument("--output-dir", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ── 输出目录 | Output Dir ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ckpt_path.parent / "eval_segnext"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 加载 Checkpoint | Load Checkpoint ──
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_size = ckpt.get("model_size", "tiny")
    train_args = ckpt.get("args", {}) or {}
    # 归一化模式跟随训练 (旧 ckpt 无此字段 → unit) | Follow training normalization
    img_norm = train_args.get("img_norm", "unit")
    print(f"  SegNeXt-{model_size}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  img_norm: {img_norm}")
    print(f"  Best Dice (train-time): {ckpt.get('best_Dice', ckpt.get('Dice', 'N/A'))}")

    # ── 数据集 | Dataset ──
    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    print(f"  Val samples: {len(val_ds)}")

    # ── 构建模型 | Build Model ──
    backbone = MSCAN(model_size=model_size).to(device)
    head = LightHamHead(
        in_channels=[backbone.embed_dims[1], backbone.embed_dims[2],
                      backbone.embed_dims[3]],
        num_classes=NUM_CLASSES,
        ham_channels=train_args.get("ham_channels", 256),
        channels=256,
        ham_kwargs=dict(MD_R=train_args.get("md_r", 16)),
        dropout_ratio=0.1,
    ).to(device)

    # ── 加载权重 | Load Weights ──
    backbone_state = ckpt.get("backbone_state_dict", {})
    if backbone_state:
        backbone.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded backbone weights: {len(backbone_state)} keys")
    else:
        print("  [WARN] No backbone_state_dict in checkpoint!")

    head_state = ckpt.get("head_state_dict", {})
    if head_state:
        head.load_state_dict(head_state, strict=False)
        print(f"  Loaded head weights: {len(head_state)} keys")
    else:
        print("  [WARN] No head_state_dict in checkpoint!")

    # ── 评估 | Evaluation ──
    print()
    print("=" * 60)
    print(f"  SegNeXt-{model_size} Multi-class Evaluation")
    print("=" * 60)

    metrics = evaluate(backbone, head, val_ds, device, img_norm=img_norm)

    print(f"  mIoU:          {metrics['mIoU']:.4f}")
    print(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  Sample mIoU:   mean={metrics['sample_mIoU_mean']:.4f} "
          f"median={metrics['sample_mIoU_median']:.4f}")
    print(f"  Per-class IoU:")
    for name in CLASS_NAMES:
        print(f"      {name:>12s}: {metrics['per_class_IoU'][name]:.4f}")

    # ── 保存 | Save ──
    results_path = out_dir / "eval_results.json"
    metrics["checkpoint"] = str(ckpt_path)
    metrics["model_size"] = model_size
    metrics["img_norm"] = img_norm
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_path}")
    print("[Done] Evaluation complete.")


if __name__ == "__main__":
    main()
