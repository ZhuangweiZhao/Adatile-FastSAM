#!/usr/bin/env python3
"""
NEU_Seg 多类别分割可视化 | NEU_Seg Multi-class Segmentation Visualization.
=========================================================================

生成数据集概览图、前景覆盖率统计、模型预测对比 (4 类)。
Generate dataset overview, FG coverage stats, model prediction comparison (4 classes).

三种模式 | Three Modes:
    1. dataset-only:  数据集概览 + 前景分布统计 (不需要 checkpoint)
    2. predict:        加载模型, 并排对比 GT vs Pred (需要 checkpoint)
    3. threshold-sweep: 不同阈值下 IoU/Dice 曲线 (需要 checkpoint, 基于 FG 通道)

用法 | Usage::

    # 数据集概览
    python tools/viz/viz_neuseg.py --mode dataset

    # 数据集概览 + 模型预测
    python tools/viz/viz_neuseg.py --mode predict \
        --checkpoint runs/neuseg_xxx/best_model.pt

    # 阈值扫描
    python tools/viz/viz_neuseg.py --mode threshold-sweep \
        --checkpoint runs/neuseg_xxx/best_model.pt
"""

from __future__ import annotations

import sys, argparse
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

import torch
import torch.nn.functional as F

from adatile.datasets.neu_seg import NEUSegDataset

# ═══════════════════════════════════════════════════════════════════
# 样式常量 | Style Constants
# ═══════════════════════════════════════════════════════════════════

# 多类别颜色 | Multi-class colors
CLASS_COLORS = {
    0: "#808080",  # BG: gray
    1: "#E63946",  # Inclusion: red
    2: "#2A9D8F",  # Patch: teal
    3: "#457B9D",  # Scratch: blue
}
CLASS_NAMES = ["Background", "Inclusion", "Patch", "Scratch"]
NUM_CLASSES = 4

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
})


# ═══════════════════════════════════════════════════════════════════
# 辅助函数 | Helpers
# ═══════════════════════════════════════════════════════════════════

def pad_to_32(image, mask=None):
    """Pad image to multiple of 32 (FastSAM backbone requirement)."""
    if image.dim() == 4:
        H, W = image.shape[2], image.shape[3]
    else:
        H, W = image.shape[1], image.shape[2]
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h == 0 and pad_w == 0:
        return image, mask, (H, W)
    pad_dims = (0, pad_w, 0, pad_h)
    image_padded = F.pad(image, pad_dims, mode='constant', value=0)
    mask_padded = F.pad(mask, pad_dims, mode='constant', value=0) if mask is not None else None
    return image_padded, mask_padded, (H, W)


def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def overlay_mask(image_rgb, mask, color, alpha=0.45):
    """在 RGB 图像上叠加彩色掩码 | Overlay colored mask on RGB image."""
    img = image_rgb.astype(np.float32) / 255.0
    overlay = img.copy()
    for c in range(3):
        overlay[:, :, c][mask] = overlay[:, :, c][mask] * (1 - alpha) + color[c] * alpha
    return (overlay * 255).astype(np.uint8)


def _collect_fg_ratios(dataset):
    """收集所有样本的 FG ratio (%) | Collect FG ratios for all samples (%)."""
    return [(dataset[i]["masks"] > 0).sum().item() / dataset[i]["masks"].numel() * 100
            for i in range(len(dataset))]


def _collect_fg_pixels(dataset):
    """收集所有样本的 FG 像素数 | Collect FG pixel counts."""
    return [int((dataset[i]["masks"] > 0).sum().item()) for i in range(len(dataset))]


def _decoder_forward(decoder, feats, support_cache):
    """统一的 decoder 前向传播 | Unified decoder forward."""
    from adatile.decoder.adaptive_sparse_decoder import (
        ProtoOnlyDecoder, ProtoOnlyDecoderP3P4,
    )
    from adatile.decoder.pure_cnn_decoder import PureDecoder, PureDecoderP3P4, PureDecoderP2P3P4

    if isinstance(decoder, PureDecoderP2P3P4):
        p2 = feats.get("p2")
        if p2 is None:
            return None
        return decoder(p2, feats["p3"], feats["p4"])
    elif isinstance(decoder, PureDecoderP3P4):
        return decoder(feats["p3"], feats["p4"])
    elif isinstance(decoder, PureDecoder):
        return decoder(feats["p4"])
    elif isinstance(decoder, ProtoOnlyDecoderP3P4):
        proto = feats.get("proto")
        if proto is None:
            return None
        sp3, sp4 = support_cache
        return decoder(proto, sp3, sp4)
    elif isinstance(decoder, ProtoOnlyDecoder):
        proto = feats.get("proto")
        if proto is None:
            return None
        return decoder(proto, support_cache)
    else:
        proto = feats.get("proto")
        if proto is None:
            return None
        return decoder(feats["p4"], proto, support_cache)


# ═══════════════════════════════════════════════════════════════════
# 图 1: 数据集全景概览 | Figure 1: Dataset Panorama Overview
# ═══════════════════════════════════════════════════════════════════

def fig_dataset_overview(dataset, output_dir: Path, max_samples: int = 20):
    """训练集样本网格：原图 + GT overlay + FG ratio 标注."""
    n = min(max_samples, len(dataset))
    n_cols = min(5, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.0))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    gt_rgb = hex_to_rgb("#2A9D8F")  # teal

    for i in range(n_rows * n_cols):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        if i >= n:
            ax.axis("off")
            continue

        sample = dataset[i]
        img = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = sample["masks"].squeeze(0).numpy() > 0
        fg_ratio = mask.sum() / mask.size * 100

        overlayed = overlay_mask(img, mask, gt_rgb, alpha=0.45)
        ax.imshow(overlayed)
        ax.set_title(f"{sample['image_id']}\nFG={fg_ratio:.3f}%", fontsize=7, family="monospace")
        ax.axis("off")

    fig.suptitle("NEU_Seg — Dataset Overview", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = output_dir / "01_dataset_overview.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 图 2: 前景覆盖率分析 | Figure 2: FG Coverage Analysis
# ═══════════════════════════════════════════════════════════════════

def fig_fg_coverage_analysis(train_ds, val_ds, output_dir: Path):
    """FG 覆盖率分布 (直方图 + 箱线图 + 统计表)."""
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # A: Train 直方图
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_fg_histogram(ax_a, train_ds, "Train")

    # B: Val 直方图
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_fg_histogram(ax_b, val_ds, "Val")

    # C: Train vs Val 箱线图
    ax_c = fig.add_subplot(gs[0, 2])
    _plot_fg_boxplot(ax_c, train_ds, val_ds)

    # D: Per-sample FG bar
    ax_d = fig.add_subplot(gs[1, :2])
    _plot_per_sample_bar(ax_d, train_ds, val_ds)

    # E: 统计摘要表
    ax_e = fig.add_subplot(gs[1, 2])
    _plot_summary_table(ax_e, train_ds, val_ds)

    fig.suptitle("NEU_Seg — Foreground Coverage Analysis", fontsize=14, fontweight="bold", y=1.01)

    path = output_dir / "02_fg_coverage.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


def _plot_fg_histogram(ax, dataset, label):
    """FG 覆盖率直方图."""
    ratios = _collect_fg_ratios(dataset)
    bins = np.linspace(0, max(ratios) * 1.1, 30) if ratios else 20
    ax.hist(ratios, bins=bins, alpha=0.7, color="#457B9D",
            edgecolor="white", linewidth=0.3)
    if ratios:
        mean_v = np.mean(ratios)
        ax.axvline(x=mean_v, color="#E63946", linestyle="--", alpha=0.8, linewidth=1.2)
        ax.text(mean_v * 1.05, ax.get_ylim()[1] * 0.9,
                f"mean={mean_v:.3f}%", fontsize=8, color="#E63946", fontweight="bold")
    ax.set_xlabel("FG Coverage (%)")
    ax.set_ylabel("Sample Count")
    ax.set_title(f"{label} Set FG Coverage Distribution", fontweight="bold")
    ax.grid(True, alpha=0.3)


def _plot_fg_boxplot(ax, train_ds, val_ds):
    """Train vs Val 箱线图."""
    tr = _collect_fg_ratios(train_ds)
    vl = _collect_fg_ratios(val_ds)
    data = [tr, vl]
    colors = ["#457B9D", "#E63946"]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True,
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.5})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for median in bp["medians"]:
        median.set_color("#333333")
        median.set_linewidth(1.5)

    ax.set_xticklabels(["Train", "Val"], fontsize=8)
    ax.set_ylabel("FG Coverage (%)")
    ax.set_title("FG Coverage — Split Comparison", fontweight="bold")
    if max(max(d) for d in data if d) > 50:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    for i, d in enumerate(data):
        if d:
            ax.scatter(i + 1, np.mean(d), marker="D", color="#333333", s=25, zorder=5)


def _plot_per_sample_bar(ax, train_ds, val_ds):
    """Per-sample FG bar (按 FG 降序)."""
    all_data = []
    for ds, split_name in [(train_ds, "Train"), (val_ds, "Val")]:
        for i in range(len(ds)):
            fg = _collect_fg_ratios(ds)[i]  # reuse per-index
            all_data.append({"name": ds[i]["image_id"], "fg": fg, "split": split_name})

    all_data.sort(key=lambda x: x["fg"], reverse=True)
    fg_values = [d["fg"] for d in all_data]
    colors = ["#457B9D" if d["split"] == "Train" else "#E63946" for d in all_data]

    x = np.arange(len(all_data))
    ax.bar(x, fg_values, color=colors, alpha=0.75, width=0.7)

    train_count = sum(1 for d in all_data if d["split"] == "Train")
    if 0 < train_count < len(all_data):
        ax.axvline(x=train_count - 0.5, color="#333333", linestyle="--", linewidth=1.5, alpha=0.6)

    step = max(1, len(all_data) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([all_data[i]["name"] for i in range(0, len(all_data), step)],
                       rotation=45, ha="right", fontsize=6, family="monospace")
    ax.set_ylabel("FG Coverage (%)")
    ax.set_title("Per-Sample FG Coverage (Sorted)", fontweight="bold")
    if max(fg_values) > 50:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_summary_table(ax, train_ds, val_ds):
    """统计摘要表."""
    ax.axis("off")
    tr = _collect_fg_ratios(train_ds)
    vl = _collect_fg_ratios(val_ds)

    def _fmt(v):
        return f"{v:.3f}%" if isinstance(v, float) else str(v)

    rows = [
        ["", "Train", "Val"],
        ["Samples", str(len(train_ds)), str(len(val_ds))],
        ["FG% Mean", _fmt(np.mean(tr)) if tr else "-", _fmt(np.mean(vl)) if vl else "-"],
        ["FG% Median", _fmt(np.median(tr)) if tr else "-", _fmt(np.median(vl)) if vl else "-"],
        ["FG% Min", _fmt(np.min(tr)) if tr else "-", _fmt(np.min(vl)) if vl else "-"],
        ["FG% Max", _fmt(np.max(tr)) if tr else "-", _fmt(np.max(vl)) if vl else "-"],
    ]

    table = ax.table(cellText=rows, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.05, 1.4)
    for j in range(len(rows[0])):
        table[(0, j)].set_facecolor("#333333")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
        table[(1, j)].set_facecolor("#555555")
        table[(1, j)].set_text_props(color="white", fontweight="bold")
    ax.set_title("Dataset Statistics Summary", fontweight="bold", fontsize=10, pad=10)


# ═══════════════════════════════════════════════════════════════════
# 图 3: 模型预测对比 | Figure 3: Model Prediction Comparison
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def fig_prediction_comparison(
    decoder, backbone, support_cache, dataset,
    device: torch.device, output_dir: Path,
    max_samples: int = 9,
):
    """
    多类别 GT vs Pred 对比, 选择 FG 最大的 Top-N 样本.
    Multi-class GT vs Pred comparison, top-N by FG coverage.
    """
    decoder.eval()

    mc_colors = {k: np.array(hex_to_rgb(v)) * 255 for k, v in CLASS_COLORS.items()}

    samples_info = sorted(
        [(i, (dataset[i]["masks"] > 0).sum().item() / dataset[i]["masks"].numel())
         for i in range(len(dataset))],
        key=lambda x: x[1], reverse=True
    )[:max_samples]

    n = len(samples_info)
    fig, axes = plt.subplots(n, 4, figsize=(13, n * 3.2))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, (idx, fg_ratio) in enumerate(samples_info):
        sample = dataset[idx]
        img = (sample["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt_mask = sample["masks"].squeeze(0).numpy()  # int64 {0,1,2,3}
        H, W = gt_mask.shape

        # ── 模型预测 ──
        query_img = sample["image"].unsqueeze(0).to(device)
        query_img, _, _ = pad_to_32(query_img)
        feats = backbone(query_img, extract_proto=True)
        pred_prob = _decoder_forward(decoder, feats, support_cache)

        if pred_prob is None:
            pred_class = np.zeros((H, W), dtype=np.int64)
        else:
            pred_full = F.interpolate(
                pred_prob.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0).cpu().numpy()
            pred_class = np.argmax(pred_full, axis=0)

        gt_binary = gt_mask > 0

        # ── Col 1: 原图 | Original Image ──
        ax_orig = axes[row, 0]
        ax_orig.imshow(img)
        ax_orig.set_title(f"{sample['image_id']}\nFG={fg_ratio*100:.3f}%",
                          fontsize=7, family="monospace")
        ax_orig.axis("off")
        if row == 0:
            ax_orig.set_ylabel("Original", fontsize=9, fontweight="bold",
                               rotation=90, labelpad=15, va="center")

        # ── Col 2: 原图 + GT overlay ──
        ax_img = axes[row, 1]
        gt_rgb = hex_to_rgb("#2A9D8F")
        ax_img.imshow(overlay_mask(img, gt_binary, gt_rgb, alpha=0.4))
        ax_img.set_title(f"Image + GT", fontsize=7, family="monospace")
        ax_img.axis("off")
        if row == 0:
            ax_img.set_ylabel("Image + GT", fontsize=9, fontweight="bold",
                              rotation=90, labelpad=15, va="center")

        # ── Col 3: GT mask (color-coded) ──
        ax_gt = axes[row, 2]
        gt_rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
        for c, color in mc_colors.items():
            gt_rgb_img[gt_mask.astype(np.int64) == c] = color
        ax_gt.imshow(gt_rgb_img.astype(np.uint8))
        ax_gt.set_title("GT", fontsize=7, family="monospace")
        ax_gt.axis("off")
        if row == 0:
            ax_gt.set_ylabel("GT Mask", fontsize=9, fontweight="bold",
                             rotation=90, labelpad=15, va="center")

        # ── Col 4: Pred mask (color-coded) ──
        ax_pred = axes[row, 3]
        pred_rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
        for c, color in mc_colors.items():
            pred_rgb_img[pred_class == c] = color
        acc = (pred_class == gt_mask.astype(np.int64)).mean()
        ax_pred.imshow(pred_rgb_img.astype(np.uint8))
        ax_pred.set_title(f"Pred\nPixel Acc={acc:.4f}", fontsize=7, family="monospace")
        ax_pred.axis("off")
        if row == 0:
            ax_pred.set_ylabel("Prediction", fontsize=9, fontweight="bold",
                               rotation=90, labelpad=15, va="center")

    # Legend
    legend_patches = [mpatches.Patch(color=color, label=name)
                      for name, color in CLASS_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.8)

    fig.suptitle("Multi-class Prediction Comparison", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    path = output_dir / "03_prediction_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 图 4: 阈值扫描 (基于 FG 通道) | Figure 4: Threshold Sweep (FG channel)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def fig_threshold_sweep(
    decoder, backbone, support_cache, dataset,
    device: torch.device, output_dir: Path, n_thresholds: int = 50,
):
    """
    IoU / Dice / Precision / Recall vs Confidence Threshold 曲线.
    对多类别模型使用 argmax 确定二值 FG/BG，扫不同置信度阈值 (参数化 softmax 对阈值不敏感, 此处提供参考).
    """
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    metrics = {t: {"inter": 0, "union": 0, "tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    decoder.eval()

    for idx in tqdm(range(len(dataset)), desc="Threshold sweep"):
        sample = dataset[idx]
        gt_mask = sample["masks"].squeeze(0).numpy() > 0
        H, W = gt_mask.shape

        query_img = sample["image"].unsqueeze(0).to(device)
        query_img, _, _ = pad_to_32(query_img)
        feats = backbone(query_img, extract_proto=True)
        pred_prob = _decoder_forward(decoder, feats, support_cache)
        if pred_prob is None:
            continue

        # 多类别 → 取 FG 通道概率 (argmax 后对阈值不敏感, 这里用 softmax 的 FG sum)
        pred_full = F.interpolate(
            pred_prob.unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(0).cpu().numpy()  # [C, H, W]
        # FG probability = sum of class 1,2,3 softmax
        fg_prob = pred_full[1:].sum(axis=0)

        for t in thresholds:
            pred_bin = fg_prob > t
            metrics[t]["inter"] += (pred_bin & gt_mask).sum()
            metrics[t]["union"] += (pred_bin | gt_mask).sum()
            metrics[t]["tp"] += (pred_bin & gt_mask).sum()
            metrics[t]["fp"] += (pred_bin & ~gt_mask).sum()
            metrics[t]["fn"] += (~pred_bin & gt_mask).sum()

    iou_curve, dice_curve, prec_curve, rec_curve = [], [], [], []
    for t in thresholds:
        m = metrics[t]
        iou = m["inter"] / m["union"] if m["union"] > 0 else 0.0
        dice = 2 * m["tp"] / (2 * m["tp"] + m["fp"] + m["fn"] + 1e-6)
        prec = m["tp"] / (m["tp"] + m["fp"] + 1e-6)
        rec = m["tp"] / (m["tp"] + m["fn"] + 1e-6)
        iou_curve.append(iou)
        dice_curve.append(dice)
        prec_curve.append(prec)
        rec_curve.append(rec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: IoU + Dice
    ax1.plot(thresholds, iou_curve, "o-", color="#2A9D8F", markersize=2.5, linewidth=1.8, label="mIoU")
    ax1.plot(thresholds, dice_curve, "s-", color="#F4A261", markersize=2.5, linewidth=1.8, label="Dice")

    best_iou_idx = np.argmax(iou_curve)
    best_dice_idx = np.argmax(dice_curve)
    ax1.scatter([thresholds[best_iou_idx]], [iou_curve[best_iou_idx]], color="#2A9D8F",
                s=80, zorder=5, edgecolors="white", linewidth=1.5)
    ax1.scatter([thresholds[best_dice_idx]], [dice_curve[best_dice_idx]], color="#F4A261",
                s=80, zorder=5, edgecolors="white", linewidth=1.5)
    ax1.set_xlabel("Confidence Threshold")
    ax1.set_ylabel("Score")
    ax1.set_title("IoU & Dice vs Threshold", fontweight="bold")
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max(iou_curve), max(dice_curve)) * 1.15)

    # Right: Precision + Recall
    ax2.plot(thresholds, prec_curve, "-", color="#E63946", linewidth=1.8, label="Precision")
    ax2.plot(thresholds, rec_curve, "-", color="#457B9D", linewidth=1.8, label="Recall")
    diff = np.abs(np.array(prec_curve) - np.array(rec_curve))
    cross_idx = np.argmin(diff)
    ax2.scatter([thresholds[cross_idx]], [prec_curve[cross_idx]], color="#333333",
                s=80, zorder=5, edgecolors="white", linewidth=1.5)
    ax2.set_xlabel("Confidence Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision & Recall vs Threshold", fontweight="bold")
    ax2.legend(fontsize=8, loc="center right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Threshold Sensitivity Analysis (FG channel)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    path = output_dir / "04_threshold_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    print(f"\n  Best IoU:  {iou_curve[best_iou_idx]:.4f} @ theta={thresholds[best_iou_idx]:.2f}")
    print(f"  Best Dice: {dice_curve[best_dice_idx]:.4f} @ theta={thresholds[best_dice_idx]:.2f}")
    print(f"  P~R:       P=R={prec_curve[cross_idx]:.4f} @ theta={thresholds[cross_idx]:.2f}")
    return path


# ═══════════════════════════════════════════════════════════════════
# 参数解析 & 主函数 | Arg Parsing & Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NEU_Seg Multi-class Visualization")
    p.add_argument("--mode", type=str, default="dataset",
                   choices=["dataset", "predict", "threshold-sweep", "all"])
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=9)
    return p.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"runs/viz_neuseg_{args.mode}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── 加载数据集 (始终多类别) | Load Datasets (always multi-class) ──
    train_ds = NEUSegDataset(root=args.data_root, split="train", binary=False)
    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)} (NEU_Seg, multi-class)")

    # ═══════════════════════════════════════════════════════════════
    # 模式 1: 数据集概览
    # ═══════════════════════════════════════════════════════════════
    if args.mode in ("dataset", "all"):
        print("\n" + "=" * 50)
        print("  [1/2] Dataset Visualization")
        print("=" * 50)
        fig_dataset_overview(train_ds, out_dir, max_samples=20)
        fig_fg_coverage_analysis(train_ds, val_ds, out_dir)

    # ═══════════════════════════════════════════════════════════════
    # 模式 2/3/4: 需要模型
    # ═══════════════════════════════════════════════════════════════
    if args.mode in ("predict", "threshold-sweep", "all"):
        if args.checkpoint is None:
            print("\n[ERROR] --checkpoint required")
            sys.exit(1)

        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"\n[ERROR] Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        print(f"\nLoading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        ckpt_args = ckpt.get("args", {})
        decoder_type = ckpt_args.get("decoder_type", "adaptive")
        proto_source = ckpt_args.get("proto_source", "p4")
        feat_dim = 960 if proto_source == "p3" else 1280

        from adatile.backbone import FastSAMBackbone
        from adatile.decoder.adaptive_sparse_decoder import (
            AdaptiveSparseDecoder, ProtoOnlyDecoder, ProtoOnlyDecoderP3P4,
        )
        from adatile.decoder.pure_cnn_decoder import PureDecoder, PureDecoderP3P4, PureDecoderP2P3P4

        backbone = FastSAMBackbone(freeze_backbone=True).to(device)
        backbone.eval()

        decoder_map = {
            "pure": lambda: PureDecoder(in_channels=1280, out_channels=NUM_CLASSES),
            "pure_p3p4": lambda: PureDecoderP3P4(p3_channels=960, p4_channels=1280,
                                                  out_channels=NUM_CLASSES),
            "pure_p2p3p4": lambda: PureDecoderP2P3P4(
                p2_channels=160, p3_channels=960, p4_channels=1280,
                out_channels=NUM_CLASSES,
            ),
        }
        decoder = decoder_map.get(decoder_type,
            lambda: AdaptiveSparseDecoder(in_channels=1280, proto_dim=32, use_fdr=False,
                                          out_channels=NUM_CLASSES))().to(device)

        decoder_state = ckpt.get("decoder_state_dict", {})
        if decoder_state:
            try:
                decoder.load_state_dict(decoder_state, strict=False)
                print(f"  Loaded decoder: {len(decoder_state)} keys")
            except Exception as e:
                print(f"  [WARN] Failed to load decoder: {e}")
        decoder.eval()

        # Support prototype
        if decoder_type == "proto_only_p3p4":
            sp3 = ckpt.get("support_proto_p3", torch.zeros(960, device=device))
            sp4 = ckpt.get("support_proto_p4", torch.zeros(1280, device=device))
            support_cache = (sp3.to(device), sp4.to(device))
        else:
            sp = ckpt.get("support_proto")
            if sp is None or not isinstance(sp, torch.Tensor):
                sp = torch.zeros(1280, device=device)
            support_cache = sp.to(device)

        if args.mode in ("predict", "all"):
            print(f"\n{'=' * 50}")
            print(f"  Multi-class Prediction Comparison")
            print(f"{'=' * 50}")
            fig_prediction_comparison(
                decoder, backbone, support_cache, val_ds, device, out_dir,
                max_samples=args.max_samples,
            )

        if args.mode in ("threshold-sweep", "all"):
            print(f"\n{'=' * 50}")
            print(f"  Threshold Sweep")
            print(f"{'=' * 50}")
            fig_threshold_sweep(decoder, backbone, support_cache, val_ds, device, out_dir)

    # ── 汇总 ──
    print(f"\n{'=' * 60}")
    print(f"  Done — Output: {out_dir}")
    for f in sorted(out_dir.glob("*.png")):
        print(f"    {f.name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
