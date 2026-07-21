#!/usr/bin/env python3
"""
DA-FRN 可视化 | DA-FRN Visualization.
=======================================

可视化 DA-FRN 的三个方面:
    1. 分割预测对比 (Input / GT / Prediction)
    2. FDE 学习的频率剖面 (径向 profile — 低频→高频的增强曲线)
    3. DCR 通道权重分布 (哪些通道被增强/抑制)
    4. CDF 动态融合权重分布 (w3 vs w4 分布)

Visualizes three aspects of DA-FRN:
    1. Segmentation prediction comparison (Input / GT / Prediction)
    2. FDE learned frequency profile (radial profile — low→high freq boost curve)
    3. DCR channel weight distribution (which channels are boosted/suppressed)
    4. CDF dynamic fusion weight distribution (w3 vs w4 distribution)

用法 | Usage::

    python tools/viz/viz_dafrn.py \
        --checkpoint runs/neuseg_DAFRN_DCR+FDE+CDF_0717_2338/best_model.pt \
        --num-samples 6 --device cuda
"""

from __future__ import annotations

import sys, argparse
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
from adatile.rectify import DA_FRN
from adatile.datasets.neu_seg import NEUSegDataset

CLASS_NAMES = ["BG", "Inclusion", "Patch", "Scratch"]
CLASS_COLORS = {
    0: [0, 0, 0],         # BG: black
    1: [220, 50, 50],     # Inclusion: red
    2: [50, 180, 50],     # Patch: green
    3: [50, 50, 220],     # Scratch: blue
}
NUM_CLASSES = len(CLASS_NAMES)


# ═══════════════════════════════════════════════════════════════════
# 分割预测 | Segmentation Prediction
# ═══════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device):
    """加载训练好的 DA-FRN 模型 | Load trained DA-FRN model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args_dict = ckpt.get("args", {})

    # Backbone
    backbone_name = args_dict.get("backbone", "fastsam-x")
    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint=f"thirdLibrary/FastSAM/weights/FastSAM-{backbone_name.split('-')[-1]}.pt",
    ).to(device)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels

    # DA-FRN
    frn = DA_FRN(
        p3_channels=ch["p3"],
        p4_channels=ch["p4"],
        dcr_reduction=args_dict.get("dcr_reduction", 16),
        fde_num_bins=args_dict.get("fde_bins", 32),
        cdf_hidden=args_dict.get("cdf_hidden", 64),
        enable_dcr=args_dict.get("enable_dcr", True),
        enable_fde=args_dict.get("enable_fde", True),
        enable_cdf=args_dict.get("enable_cdf", True),
    ).to(device)
    frn.load_state_dict(ckpt["frn_state_dict"])
    frn.eval()

    # Decoder
    decoder = PureDecoderP3P4(
        p3_channels=ch["p3"],
        p4_channels=ch["p4"],
        out_channels=NUM_CLASSES,
    ).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()

    return backbone, frn, decoder, ckpt


@torch.no_grad()
def predict(backbone, frn, decoder, image: torch.Tensor, device: torch.device):
    """单张图像推理 | Single image inference."""
    img = image.unsqueeze(0).to(device)
    _, _, H, W = img.shape
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)

    feats = backbone(img, extract_proto=False)
    p3, p4 = feats["p3"], feats["p4"]
    p3_r, p4_r = frn(p3, p4)
    pred = decoder(p3_r, p4_r)  # [C, H/4, W/4]

    # Resize to original
    pred_full = F.interpolate(pred.unsqueeze(0), size=(H, W),
                              mode="bilinear", align_corners=False).squeeze(0)
    pred_class = torch.argmax(pred_full, dim=0).cpu().numpy()
    return pred_class


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """类别索引 → RGB 彩色图 | Class indices → RGB color map."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        rgb[mask == c] = color
    return rgb


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_segmentation_grid(dataset, backbone, frn, decoder, device,
                           indices: list[int], save_path: Path):
    """分割预测网格 | Segmentation prediction grid."""
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        sample = dataset[idx]
        img_t = sample["image"]  # [3, H, W]
        gt = sample["masks"].squeeze(0).long().numpy()  # [H, W]

        # Input image
        img_np = img_t.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # Prediction
        pred = predict(backbone, frn, decoder, img_t, device)
        pred_rgb = mask_to_rgb(pred)
        gt_rgb = mask_to_rgb(gt)

        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f"Input (#{idx})", fontsize=11)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_rgb)
        axes[row, 1].set_title("Ground Truth", fontsize=11)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_rgb)
        axes[row, 2].set_title("DA-FRN Prediction", fontsize=11)
        axes[row, 2].axis("off")

    # Legend
    legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=np.array(c) / 255, ec="gray", lw=0.5)
                      for c in CLASS_COLORS.values()]
    fig.legend(legend_patches, CLASS_NAMES, loc="lower center",
               ncol=NUM_CLASSES, fontsize=9, frameon=False)

    plt.suptitle("DA-FRN Segmentation Results", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Segmentation grid saved → {save_path}")


def plot_fde_profile(frn: DA_FRN, save_path: Path):
    """FDE 空间域增强可视化 | FDE spatial enhancement visualization.

    显示: (1) 各核大小的 α 强度, (2) 通道权重分布.
    Shows: (1) α strength per kernel size, (2) channel weight distribution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (name, fde, color) in enumerate([
        ("P3 (stride 8, 960ch)", frn.fde_p3, "#E74C3C"),
        ("P4 (stride 16, 1280ch)", frn.fde_p4, "#2980B9"),
    ]):
        if fde is None:
            axes[0, col].text(0.5, 0.5, "FDE disabled", ha="center", va="center",
                              transform=axes[0, col].transAxes, fontsize=14, color="gray")
            axes[1, col].text(0.5, 0.5, "FDE disabled", ha="center", va="center",
                              transform=axes[1, col].transAxes, fontsize=14, color="gray")
            continue

        # ── Row 0: Alpha values per kernel size (bar chart) ──
        alphas = fde.get_alphas()  # {ks: alpha_val}
        ks_list = list(alphas.keys())
        alpha_vals = list(alphas.values())
        bars = axes[0, col].bar(ks_list, alpha_vals, color=color, alpha=0.7,
                                edgecolor="white", lw=0.5, width=1.5)
        axes[0, col].axhline(y=0.1, color="gray", ls="--", lw=1, alpha=0.5,
                             label="Initial α=0.1")
        # Annotate bars
        for bar, val in zip(bars, alpha_vals):
            axes[0, col].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                              f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        axes[0, col].set_title(f"{name} — α per Kernel Size\n"
                               f"(small=texture, mid=coarse, large=BG suppress)",
                               fontsize=10, fontweight="bold")
        axes[0, col].set_xlabel("Kernel Size"); axes[0, col].set_ylabel("α (enhancement strength)")
        axes[0, col].legend(fontsize=8); axes[0, col].grid(True, alpha=0.3, axis="y")
        axes[0, col].set_ylim(0, max(max(alpha_vals) * 1.5, 0.3))

        # ── Row 1: Channel weight distribution (histogram) ──
        ch_weights = fde.get_channel_weights_mean().detach().cpu().numpy()
        axes[1, col].hist(ch_weights, bins=80, color=color, alpha=0.7,
                          edgecolor="white", lw=0.2)
        axes[1, col].axvline(x=1.0, color="gray", ls="--", lw=1.2, alpha=0.6,
                             label="Init (1.0)")
        axes[1, col].axvline(x=ch_weights.mean(), color="darkred", ls="-", lw=1.5,
                             label=f"Mean={ch_weights.mean():.4f}")

        # Top/bottom 5 channels
        top5 = np.argsort(ch_weights)[-5:]
        bot5 = np.argsort(ch_weights)[:5]
        axes[1, col].set_title(f"{name} — Channel Enhancement Weights\n"
                               f"Top-5 boosted: {top5.tolist()}, "
                               f"Suppressed: {bot5.tolist()}",
                               fontsize=9, fontweight="bold")
        axes[1, col].set_xlabel("Weight (1.0=identity, >1=boost, <1=suppress)")
        axes[1, col].set_ylabel("Num Channels"); axes[1, col].legend(fontsize=8)
        axes[1, col].grid(True, alpha=0.3)

        std_val = ch_weights.std()
        axes[1, col].text(0.95, 0.95, f"σ={std_val:.4f}\n"
                          f"{'✓ Learned (>0.02)' if std_val > 0.02 else '⚠ Static (<0.02)'}",
                          transform=axes[1, col].transAxes, fontsize=9, ha="right", va="top",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("FDE — Spatial High-Frequency Enhancement (New)", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  FDE profile plot saved → {save_path}")


def plot_dcr_weights(frn: DA_FRN, backbone, device: torch.device,
                     dataset, save_path: Path, num_samples: int = 10):
    """DCR 通道权重分布 | DCR channel weight distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, (name, dcr) in enumerate([
        ("P3 (960 channels)", frn.dcr_p3),
        ("P4 (1280 channels)", frn.dcr_p4),
    ]):
        if dcr is None:
            axes[ax_idx].text(0.5, 0.5, "DCR disabled", ha="center", va="center",
                              transform=axes[ax_idx].transAxes, fontsize=14, color="gray")
            axes[ax_idx].set_title(f"{name} — DCR Disabled")
            continue

        # Collect channel weights across multiple samples
        all_weights = []
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        for idx in indices:
            img = dataset[idx]["image"].unsqueeze(0).to(device)
            feats = backbone(img, extract_proto=False)
            x = feats["p3"] if ax_idx == 0 else feats["p4"]
            weights = dcr.get_channel_weights(x).squeeze(0).detach().cpu().numpy()
            all_weights.append(weights)

        all_weights = np.array(all_weights)  # [N, C]
        mean_w = all_weights.mean(axis=0)     # [C] — 平均通道权重
        std_w = all_weights.std(axis=0)       # [C] — 通道权重方差

        # Channel weight distribution
        axes[ax_idx].hist(mean_w, bins=50, color="#8E44AD" if ax_idx == 0 else "#2980B9",
                          alpha=0.7, edgecolor="white", lw=0.3)

        # Statistics
        top5_idx = np.argsort(mean_w)[-5:]
        bot5_idx = np.argsort(mean_w)[:5]

        axes[ax_idx].axvline(x=0.5, color="gray", ls="--", lw=1.2, alpha=0.6,
                             label="Initial (0.5)")
        axes[ax_idx].axvline(x=mean_w.mean(), color="red", ls="-", lw=1.5, alpha=0.8,
                             label=f"Mean={mean_w.mean():.4f}")

        axes[ax_idx].set_title(f"{name} — DCR Channel Weights\n"
                               f"Top-5 boosted ch: {top5_idx.tolist()}\n"
                               f"Top-5 suppressed ch: {bot5_idx.tolist()}",
                               fontsize=10, fontweight="bold")
        axes[ax_idx].set_xlabel("Channel Weight (0=suppress, 1=boost)")
        axes[ax_idx].set_ylabel("Number of Channels")
        axes[ax_idx].legend(fontsize=8)
        axes[ax_idx].grid(True, alpha=0.3)

    plt.suptitle("DCR — Learned Channel Reweighting Distribution", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  DCR weight distribution saved → {save_path}")


def plot_cdf_weights(frn: DA_FRN, backbone, device: torch.device,
                     dataset, save_path: Path, num_samples: int = 50):
    """CDF 动态融合权重分布 | CDF dynamic fusion weight distribution."""
    if frn.cdf is None:
        print("  CDF disabled — skipping CDF weight plot")
        return

    cdf = frn.cdf
    w3_list, w4_list = [], []
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    for idx in tqdm(indices, desc="CDF stats", leave=False):
        img = dataset[idx]["image"].unsqueeze(0).to(device)
        feats = backbone(img, extract_proto=False)
        p3, p4 = feats["p3"], feats["p4"]

        # Forward through DCR+FDE first (CDF is last in chain)
        if frn.enable_dcr:
            with torch.no_grad():
                p3 = frn.dcr_p3(p3)
                p4 = frn.dcr_p4(p4)
        if frn.enable_fde:
            with torch.no_grad():
                p3 = frn.fde_p3(p3)
                p4 = frn.fde_p4(p4)

        # Extract CDF weights
        B = p3.shape[0]
        p3_gap = F.adaptive_avg_pool2d(p3, 1).view(B, -1)
        p4_gap = F.adaptive_avg_pool2d(p4, 1).view(B, -1)
        combined = torch.cat([p3_gap, p4_gap], dim=1)
        raw = cdf.predictor(combined)  # [B, 2]
        w_sum = raw.sum(dim=1, keepdim=True).clamp(min=1e-6)
        w3 = (raw[:, 0] / w_sum.squeeze(1)) * 2.0
        w4 = (raw[:, 1] / w_sum.squeeze(1)) * 2.0
        w3_list.append(w3.item())
        w4_list.append(w4.item())

    w3_arr = np.array(w3_list)
    w4_arr = np.array(w4_list)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Histogram: w3
    axes[0].hist(w3_arr, bins=30, color="#E74C3C", alpha=0.7, edgecolor="white", lw=0.3)
    axes[0].axvline(x=1.0, color="gray", ls="--", lw=1.2, alpha=0.6, label="Equal weight")
    axes[0].axvline(x=w3_arr.mean(), color="darkred", ls="-", lw=1.5,
                    label=f"Mean={w3_arr.mean():.4f}")
    axes[0].set_title(f"P3 Weight (w3)\nMean={w3_arr.mean():.4f}±{w3_arr.std():.4f}",
                      fontweight="bold")
    axes[0].set_xlabel("Weight"); axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # Histogram: w4
    axes[1].hist(w4_arr, bins=30, color="#2980B9", alpha=0.7, edgecolor="white", lw=0.3)
    axes[1].axvline(x=1.0, color="gray", ls="--", lw=1.2, alpha=0.6, label="Equal weight")
    axes[1].axvline(x=w4_arr.mean(), color="darkblue", ls="-", lw=1.5,
                    label=f"Mean={w4_arr.mean():.4f}")
    axes[1].set_title(f"P4 Weight (w4)\nMean={w4_arr.mean():.4f}±{w4_arr.std():.4f}",
                      fontweight="bold")
    axes[1].set_xlabel("Weight"); axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # Scatter: w3 vs w4
    axes[2].scatter(w3_arr, w4_arr, alpha=0.6, s=30, c="#27AE60", edgecolors="white", lw=0.3)
    axes[2].axhline(y=1.0, color="gray", ls="--", lw=0.8, alpha=0.4)
    axes[2].axvline(x=1.0, color="gray", ls="--", lw=0.8, alpha=0.4)
    axes[2].plot([0.9, 1.1], [0.9, 1.1], color="gray", ls=":", lw=0.8, alpha=0.5,
                 label="w3=w4")
    axes[2].set_xlabel("w3 (P3 weight)"); axes[2].set_ylabel("w4 (P4 weight)")
    axes[2].set_title(f"CDF Dynamic Weights\nw3+w4={w3_arr.mean()+w4_arr.mean():.2f}",
                      fontweight="bold")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    # Variability analysis
    variability = w3_arr.std() + w4_arr.std()
    axes[2].text(0.95, 0.05, f"Total variability: {variability:.4f}\n"
                 f"{'✓ Weights adapt per-sample' if variability > 0.05 else '⚠ Weights nearly static'}",
                 transform=axes[2].transAxes, fontsize=9, ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("CDF — Dynamic Cross-scale Fusion Weights", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CDF weight distribution saved → {save_path}")
    print(f"  CDF stats: w3={w3_arr.mean():.4f}±{w3_arr.std():.4f}, "
          f"w4={w4_arr.mean():.4f}±{w4_arr.std():.4f}, variability={variability:.4f}")


# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="DA-FRN Visualization")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="最佳模型 checkpoint | Best model checkpoint")
    parser.add_argument("--data-root", type=str, default="data/NEU_Seg")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: checkpoint 同目录下的 viz/)")
    parser.add_argument("--num-samples", type=int, default=6,
                        help="分割可视化样本数 | Number of segmentation samples")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── Output dir ──
    ckpt_path = Path(args.checkpoint)
    if args.output_dir is None:
        out_dir = ckpt_path.parent / "viz"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── Load model ──
    print("Loading model...")
    backbone, frn, decoder, ckpt = load_model(args.checkpoint, device)
    miou = ckpt.get("mIoU", "N/A")
    per_class = ckpt.get("per_class_IoU", {})
    print(f"  mIoU: {miou}")
    if per_class:
        for cls_name, iou in per_class.items():
            print(f"    {cls_name}: {iou:.4f}")

    # ── Data ──
    dataset = NEUSegDataset(root=args.data_root, split="test", binary=False)
    print(f"Test samples: {len(dataset)}")

    # ── 1. Segmentation Grid ──
    print("\n=== 1. Segmentation Predictions ===")
    sample_indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)),
                                      replace=False).tolist()
    plot_segmentation_grid(dataset, backbone, frn, decoder, device,
                           sample_indices, out_dir / "segmentation_grid.png")

    # ── 2. FDE Frequency Profile ──
    print("\n=== 2. FDE Frequency Profile ===")
    plot_fde_profile(frn, out_dir / "fde_frequency_profile.png")

    # ── 3. DCR Channel Weights ──
    print("\n=== 3. DCR Channel Weights ===")
    plot_dcr_weights(frn, backbone, device, dataset,
                     out_dir / "dcr_channel_weights.png", num_samples=10)

    # ── 4. CDF Dynamic Weights ──
    print("\n=== 4. CDF Dynamic Weights ===")
    plot_cdf_weights(frn, backbone, device, dataset,
                     out_dir / "cdf_dynamic_weights.png", num_samples=50)

    # ── Summary ──
    frn_stats = frn.get_stats()
    print(f"\n{'='*60}")
    print(f"  DA-FRN Visualization — Complete")
    print(f"  mIoU: {miou}")
    print(f"  FRN Stats: {frn_stats}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
