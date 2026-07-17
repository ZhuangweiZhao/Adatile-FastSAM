#!/usr/bin/env python3
"""
BoundaryRefiner 可视化 | BoundaryRefiner Visualization.
=========================================================

对比 Coarse vs Refined mask，展示 boundary attention map + per-class IoU delta。
Compare Coarse vs Refined mask, show boundary attention + per-class IoU delta.

用法 | Usage::

    python tools/viz/viz_boundary_refine.py \
        --checkpoint runs/neuseg_Refine_*/best_model.pt --num-samples 6
"""

from __future__ import annotations

import sys, argparse, json
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from adatile.backbone import FastSAMBackbone
from adatile.decoder.pure_cnn_decoder import PureDecoderP3P4
from adatile.refine import BoundaryRefiner
from adatile.datasets.neu_seg import NEUSegDataset

NUM_CLASSES = 4
CLASS_NAMES = ["BG", "Inclusion", "Patch", "Scratch"]
CLASS_COLORS = [
    [0, 0, 0],           # BG
    [1.0, 0.3, 0.3],     # Inclusion — red
    [0.3, 0.6, 1.0],     # Patch — blue
    [0.3, 1.0, 0.3],     # Scratch — green
]


def build_label_image(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(NUM_CLASSES):
        rgb[mask == c] = CLASS_COLORS[c]
    return rgb


def build_diff_image(coarse: np.ndarray, refined: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """差异图: 绿色=修正正确, 红色=修正错误, 黄色=未变错误"""
    H, W = coarse.shape
    diff = np.zeros((H, W, 3), dtype=np.float32)
    coarse_correct = (coarse == gt)
    refined_correct = (refined == gt)

    # 修正正确 (coarse wrong → refined correct) → 绿色
    diff[~coarse_correct & refined_correct] = [0, 1, 0]
    # 修正错误 (coarse correct → refined wrong) → 红色
    diff[coarse_correct & ~refined_correct] = [1, 0, 0]
    # 未变错误 (both wrong) → 黄色
    diff[~coarse_correct & ~refined_correct] = [1, 1, 0]
    # 未变正确 (both correct) → 灰色底
    diff[coarse_correct & refined_correct] = [0.3, 0.3, 0.3]

    return diff


@torch.no_grad()
def infer_with_refine(decoder, refiner, backbone, image, device):
    """推理 + 细化, 返回 coarse, refined, boundary"""
    H, W = image.shape[1:]
    img = image.unsqueeze(0).to(device)
    pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

    feats = backbone(img, extract_proto=True)
    coarse = decoder(feats["p3"], feats["p4"])  # [C, H/4, W/4]
    coarse_4d = coarse.unsqueeze(0)
    refined_4d, boundary = refiner(coarse_4d, feats["p2"])  # [1,C,H/4,W/4], [1,1,H/4,W/4]

    # Upsample
    coarse_full = F.interpolate(coarse_4d, size=(H, W),
                                mode="bilinear", align_corners=False).squeeze(0)
    refined_full = F.interpolate(refined_4d, size=(H, W),
                                 mode="bilinear", align_corners=False).squeeze(0)
    boundary_full = F.interpolate(boundary, size=(H, W),
                                  mode="bilinear", align_corners=False).squeeze()

    coarse_cls = torch.argmax(coarse_full, dim=0).cpu().numpy()
    refined_cls = torch.argmax(refined_full, dim=0).cpu().numpy()
    boundary_np = boundary_full.cpu().numpy()

    return coarse_cls, refined_cls, boundary_np


def plot_compare(checkpoint_path, output_path, dataset, device, num_samples=6, seed=42):
    """对比 Coarse vs Refined mask."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_ckpt = ckpt.get("args", {})

    backbone = FastSAMBackbone(freeze_backbone=True).to(device); backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels

    decoder = PureDecoderP3P4(p3_channels=ch["p3"], p4_channels=ch["p4"], out_channels=NUM_CLASSES).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"]); decoder.eval()

    refiner = BoundaryRefiner(feat_channels=ch["p2"], num_classes=NUM_CLASSES, hidden=64).to(device)
    refiner.load_state_dict(ckpt["refiner_state_dict"]); refiner.eval()

    epoch = ckpt.get("epoch", "?")

    # Select samples with defects
    rng = np.random.RandomState(seed)
    defect_idx = [i for i in range(len(dataset)) if (dataset[i]["masks"] > 0).sum() > 0]
    selected = list(rng.choice(defect_idx, size=min(num_samples, len(defect_idx)), replace=False))

    n_rows = len(selected)
    n_cols = 5  # Image, GT, Coarse, Refined, Diff
    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.2))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.1, hspace=0.2)

    col_titles = ["Image", "GT", "Coarse", "Refined", "Diff (G=fix R=break)"]
    for row_idx, ds_idx in enumerate(tqdm(selected, desc="Inferring")):
        sample = dataset[ds_idx]
        image = sample["image"]; gt = sample["masks"].squeeze(0).numpy()
        coarse_cls, refined_cls, boundary_np = infer_with_refine(decoder, refiner, backbone, image, device)
        img_np = np.clip(image.permute(1, 2, 0).cpu().numpy(), 0, 1)

        for col_idx, (title, data) in enumerate([
            ("Image", img_np),
            ("GT", build_label_image(gt)),
            ("Coarse", build_label_image(coarse_cls)),
            ("Refined", build_label_image(refined_cls)),
            ("Diff", build_diff_image(coarse_cls, refined_cls, gt)),
        ]):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if col_idx <= 3:
                ax.imshow(data)
            else:
                ax.imshow(data)
            ax.set_xticks([]); ax.set_yticks([])
            if row_idx == 0: ax.set_title(title, fontsize=9, fontweight="bold")

    fig.suptitle(f"BoundaryRefiner: Coarse vs Refined — Epoch {epoch}", fontsize=11, fontweight="bold", y=0.995)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_boundary_detail(checkpoint_path, output_path, dataset, device, num_samples=4, seed=42):
    """详细展示: Image + Coarse + Refined + Boundary overlay + Boundary map."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    backbone = FastSAMBackbone(freeze_backbone=True).to(device); backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels
    decoder = PureDecoderP3P4(p3_channels=ch["p3"], p4_channels=ch["p4"], out_channels=NUM_CLASSES).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"]); decoder.eval()
    refiner = BoundaryRefiner(feat_channels=ch["p2"], num_classes=NUM_CLASSES, hidden=64).to(device)
    refiner.load_state_dict(ckpt["refiner_state_dict"]); refiner.eval()

    rng = np.random.RandomState(seed)
    defect_idx = [i for i in range(len(dataset)) if (dataset[i]["masks"] > 0).sum() > 0]
    selected = list(rng.choice(defect_idx, size=min(num_samples, len(defect_idx)), replace=False))

    # 选一个 Inclusion + 一个 Scratch + 一个 Patch
    n_rows = len(selected)
    n_cols = 5
    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.2))
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.1, hspace=0.2)

    for row_idx, ds_idx in enumerate(tqdm(selected, desc="Detail view")):
        sample = dataset[ds_idx]
        image = sample["image"]; gt = sample["masks"].squeeze(0).numpy()
        coarse_cls, refined_cls, boundary_np = infer_with_refine(decoder, refiner, backbone, image, device)
        img_np = np.clip(image.permute(1, 2, 0).cpu().numpy(), 0, 1)

        # Col 0: Image
        ax = fig.add_subplot(gs[row_idx, 0]); ax.imshow(img_np)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Image", fontsize=9, fontweight="bold")

        # Col 1: Coarse (overlay boundary as contour)
        ax = fig.add_subplot(gs[row_idx, 1]); ax.imshow(build_label_image(coarse_cls))
        # Overlay GT boundary as contour
        gt_boundary = np.abs(np.gradient(gt.astype(float)))
        gt_edge = (gt_boundary[0] + gt_boundary[1]) > 0
        ax.contour(gt_edge, colors='white', linewidths=0.5, alpha=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Coarse (GT edge)", fontsize=9, fontweight="bold")

        # Col 2: Refined
        ax = fig.add_subplot(gs[row_idx, 2]); ax.imshow(build_label_image(refined_cls))
        ax.contour(gt_edge, colors='white', linewidths=0.5, alpha=0.5)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Refined (GT edge)", fontsize=9, fontweight="bold")

        # Col 3: Boundary attention (learned)
        ax = fig.add_subplot(gs[row_idx, 3])
        ax.imshow(img_np.mean(axis=-1), cmap="gray", vmin=0, vmax=1)
        ax.imshow(boundary_np, cmap="hot", alpha=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Boundary Attention", fontsize=9, fontweight="bold")

        # Col 4: Diff map
        ax = fig.add_subplot(gs[row_idx, 4])
        ax.imshow(build_diff_image(coarse_cls, refined_cls, gt))
        ax.set_xticks([]); ax.set_yticks([])
        if row_idx == 0: ax.set_title("Fix(G)/Break(R)", fontsize=9, fontweight="bold")

    fig.suptitle(f"BoundaryRefiner Detail — Epoch {ckpt.get('epoch','?')}", fontsize=11, fontweight="bold", y=0.995)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_class_delta(checkpoint_path, output_path, dataset, device, max_samples=100, seed=42):
    """Per-class mIoU delta: Coarse vs Refined."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone = FastSAMBackbone(freeze_backbone=True).to(device); backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels
    decoder = PureDecoderP3P4(p3_channels=ch["p3"], p4_channels=ch["p4"], out_channels=NUM_CLASSES).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"]); decoder.eval()
    refiner = BoundaryRefiner(feat_channels=ch["p2"], num_classes=NUM_CLASSES, hidden=64).to(device)
    refiner.load_state_dict(ckpt["refiner_state_dict"]); refiner.eval()

    rng = np.random.RandomState(seed)
    defect_idx = [i for i in range(len(dataset)) if (dataset[i]["masks"] > 0).sum() > 0]
    selected = list(rng.choice(defect_idx, size=min(max_samples, len(defect_idx)), replace=False))

    coarse_inter = np.zeros(NUM_CLASSES); coarse_union = np.zeros(NUM_CLASSES)
    refined_inter = np.zeros(NUM_CLASSES); refined_union = np.zeros(NUM_CLASSES)

    for ds_idx in tqdm(selected, desc="Computing delta"):
        sample = dataset[ds_idx]
        gt = sample["masks"].squeeze(0).numpy()
        coarse_cls, refined_cls, _ = infer_with_refine(decoder, refiner, backbone, sample["image"], device)
        for c in range(NUM_CLASSES):
            for arr_cls, arr_inter, arr_union in [
                (coarse_cls, coarse_inter, coarse_union),
                (refined_cls, refined_inter, refined_union),
            ]:
                pc = (arr_cls == c); gc = (gt == c)
                arr_inter[c] += (pc & gc).sum(); arr_union[c] += (pc | gc).sum()

    coarse_iou = {CLASS_NAMES[c]: coarse_inter[c] / max(coarse_union[c], 1) for c in range(NUM_CLASSES)}
    refined_iou = {CLASS_NAMES[c]: refined_inter[c] / max(refined_union[c], 1) for c in range(NUM_CLASSES)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar chart: per-class IoU
    ax = axes[0]; x = np.arange(NUM_CLASSES); w = 0.35
    ax.bar(x - w/2, [coarse_iou[cn] for cn in CLASS_NAMES], w, label="Coarse", color="steelblue", alpha=0.8)
    ax.bar(x + w/2, [refined_iou[cn] for cn in CLASS_NAMES], w, label="Refined", color="coral", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_ylabel("IoU"); ax.set_title("Per-Class IoU: Coarse vs Refined")
    ax.legend()

    # Delta
    ax = axes[1]
    deltas = [refined_iou[cn] - coarse_iou[cn] for cn in CLASS_NAMES]
    colors = ["#999" if d >= 0 else "#f66" for d in deltas]
    ax.bar(range(NUM_CLASSES), deltas, color=colors)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASS_NAMES, fontsize=9)
    ax.set_ylabel("Δ IoU"); ax.set_title("Refinement Delta (+ = improved)")
    ax.axhline(0, color="black", linewidth=0.5)
    for i, d in enumerate(deltas):
        ax.text(i, d + (0.002 if d >= 0 else -0.008), f"{d:+.4f}", ha="center", fontsize=9)

    fig.suptitle(f"BoundaryRefiner Impact — Epoch {ckpt.get('epoch','?')}", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize BoundaryRefiner")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent

    dataset = NEUSegDataset(root=args.data_root, split="test", binary=False)
    print(f"Dataset: {len(dataset)} test samples")

    plot_compare(args.checkpoint, str(out_dir / "viz_refine_compare.png"), dataset, device,
                 num_samples=args.num_samples, seed=args.seed)
    plot_boundary_detail(args.checkpoint, str(out_dir / "viz_refine_detail.png"), dataset, device,
                         num_samples=max(4, args.num_samples // 2), seed=args.seed)
    plot_per_class_delta(args.checkpoint, str(out_dir / "viz_refine_delta.png"), dataset, device, seed=args.seed)

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__": main()
