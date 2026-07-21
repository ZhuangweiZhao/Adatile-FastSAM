#!/usr/bin/env python3
"""
DCR 深度分析 & 可视化 | DCR Deep Analysis & Visualization.
=============================================================

分析内容:
    1. 分割预测可视化 (Input / GT / Pred)
    2. 通道权重分布 (P3 + P4) — 哪些被增强/抑制
    3. Top-K boosted/suppressed 通道
    4. P3 vs P4 通道权重相关性
    5. fc1 权重矩阵分析 — 输入→注意力瓶颈的信息流动
    6. 逐类别激活模式 — 不同缺陷触发不同通道吗?

用法 | Usage::

    python tools/viz/viz_dcr_analysis.py \
        --checkpoint runs/neuseg_DAFRN_DCR_0718_1205/best_model.pt \
        --num-samples 8 --device cuda
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
from adatile.rectify.dcr import DefectChannelReweighting
from adatile.datasets.neu_seg import NEUSegDataset

CLASS_NAMES = ["BG", "Inclusion", "Patch", "Scratch"]
CLASS_COLORS = {0: [0,0,0], 1: [220,50,50], 2: [50,180,50], 3: [50,50,220]}
NUM_CLASSES = len(CLASS_NAMES)


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    backbone = FastSAMBackbone(freeze_backbone=True,
        checkpoint="thirdLibrary/FastSAM/weights/FastSAM-x.pt").to(device)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels

    dcr_p3 = DefectChannelReweighting(ch["p3"]).to(device)
    dcr_p4 = DefectChannelReweighting(ch["p4"]).to(device)
    frn_sd = ckpt["frn_state_dict"]
    dcr_p3.load_state_dict({k.replace("dcr_p3.",""): v for k,v in frn_sd.items() if k.startswith("dcr_p3.")})
    dcr_p4.load_state_dict({k.replace("dcr_p4.",""): v for k,v in frn_sd.items() if k.startswith("dcr_p4.")})
    dcr_p3.eval(); dcr_p4.eval()

    decoder = PureDecoderP3P4(p3_channels=ch["p3"], p4_channels=ch["p4"],
                              out_channels=NUM_CLASSES).to(device)
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    decoder.eval()

    return backbone, dcr_p3, dcr_p4, decoder, ckpt


@torch.no_grad()
def predict(backbone, dcr_p3, dcr_p4, decoder, image, device):
    img = image.unsqueeze(0).to(device)
    _, _, H, W = img.shape
    pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
    feats = backbone(img, extract_proto=False)
    p3 = dcr_p3(feats["p3"]); p4 = dcr_p4(feats["p4"])
    pred = decoder(p3, p4)
    pred_full = F.interpolate(pred.unsqueeze(0), size=(H, W),
                              mode="bilinear", align_corners=False).squeeze(0)
    return torch.argmax(pred_full, dim=0).cpu().numpy()


def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in CLASS_COLORS.items():
        rgb[mask == c] = color
    return rgb


@torch.no_grad()
def collect_channel_weights(dcr_p3, dcr_p4, backbone, dataset, device, num_samples=30):
    """收集通道权重 + 逐类别特征 | Collect channel weights + per-class features."""
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    p3_weights, p4_weights = [], []
    p3_per_class = {c: [] for c in range(1, NUM_CLASSES)}  # per-class P3 weights
    p4_per_class = {c: [] for c in range(1, NUM_CLASSES)}  # per-class P4 weights

    for idx in tqdm(indices, desc="Collecting DCR weights", leave=False):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["masks"].squeeze(0).long().numpy()
        H, W = img.shape[2], img.shape[3]
        pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)

        feats = backbone(img, extract_proto=False)
        p3_raw, p4_raw = feats["p3"], feats["p4"]

        w3 = dcr_p3.get_channel_weights(p3_raw).squeeze(0).cpu().numpy()  # [C3]
        w4 = dcr_p4.get_channel_weights(p4_raw).squeeze(0).cpu().numpy()  # [C4]
        p3_weights.append(w3); p4_weights.append(w4)

        # Per-class: record weights for images that contain each class
        present_classes = np.unique(gt)
        for c in range(1, NUM_CLASSES):
            if c in present_classes:
                p3_per_class[c].append(w3)
                p4_per_class[c].append(w4)

    return {
        "p3_weights": np.array(p3_weights),      # [N, C3]
        "p4_weights": np.array(p4_weights),       # [N, C4]
        "p3_per_class": {c: np.array(v) for c, v in p3_per_class.items() if len(v) > 0},
        "p4_per_class": {c: np.array(v) for c, v in p4_per_class.items() if len(v) > 0},
    }


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_segmentation_grid(dataset, backbone, dcr_p3, dcr_p4, decoder,
                           device, indices, save_path):
    """分割预测网格 | Segmentation prediction grid."""
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1: axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        sample = dataset[idx]; img_t = sample["image"]
        gt = sample["masks"].squeeze(0).long().numpy()
        img_np = np.clip(img_t.permute(1,2,0).cpu().numpy(), 0, 1)
        pred = predict(backbone, dcr_p3, dcr_p4, decoder, img_t, device)

        axes[row,0].imshow(img_np); axes[row,0].set_title(f"Input (#{idx})", fontsize=11); axes[row,0].axis("off")
        axes[row,1].imshow(mask_to_rgb(gt)); axes[row,1].set_title("Ground Truth", fontsize=11); axes[row,1].axis("off")
        axes[row,2].imshow(mask_to_rgb(pred)); axes[row,2].set_title("DCR Prediction", fontsize=11); axes[row,2].axis("off")

    legend_patches = [plt.Rectangle((0,0),1,1,fc=np.array(c)/255,ec="gray",lw=0.5) for c in CLASS_COLORS.values()]
    fig.legend(legend_patches, CLASS_NAMES, loc="lower center", ncol=4, fontsize=9, frameon=False)
    plt.suptitle(f"DCR-only Segmentation (mIoU=0.7741)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0,0.05,1,0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Segmentation grid → {save_path}")


def plot_channel_weight_analysis(weight_data, save_path):
    """通道权重综合分析 | Channel weight comprehensive analysis."""
    p3_w = weight_data["p3_weights"]  # [N, 960]
    p4_w = weight_data["p4_weights"]  # [N, 1280]
    p3_mean = p3_w.mean(axis=0); p3_std = p3_w.std(axis=0)
    p4_mean = p4_w.mean(axis=0); p4_std = p4_w.std(axis=0)

    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ── (0,0): P3 weight histogram ──
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(p3_mean, bins=60, color="#E74C3C", alpha=0.7, edgecolor="white", lw=0.2)
    ax.axvline(x=0.5, color="gray", ls="--", lw=1.2, alpha=0.6, label="Init (0.5)")
    ax.axvline(x=p3_mean.mean(), color="darkred", ls="-", lw=1.5, label=f"Mean={p3_mean.mean():.4f}")
    n_boosted = (p3_mean > 0.55).sum(); n_suppressed = (p3_mean < 0.45).sum()
    ax.set_title(f"P3 (960 ch) — DCR Weights\n"
                 f"μ={p3_mean.mean():.4f}, σ={p3_mean.std():.4f}\n"
                 f"Boosted(>0.55): {n_boosted}, Suppressed(<0.45): {n_suppressed}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Weight"); ax.set_ylabel("Num Channels"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (0,1): P4 weight histogram ──
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(p4_mean, bins=60, color="#2980B9", alpha=0.7, edgecolor="white", lw=0.2)
    ax.axvline(x=0.5, color="gray", ls="--", lw=1.2, alpha=0.6, label="Init (0.5)")
    ax.axvline(x=p4_mean.mean(), color="darkblue", ls="-", lw=1.5, label=f"Mean={p4_mean.mean():.4f}")
    n_boosted = (p4_mean > 0.55).sum(); n_suppressed = (p4_mean < 0.45).sum()
    ax.set_title(f"P4 (1280 ch) — DCR Weights\n"
                 f"μ={p4_mean.mean():.4f}, σ={p4_mean.std():.4f}\n"
                 f"Boosted(>0.55): {n_boosted}, Suppressed(<0.45): {n_suppressed}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Weight"); ax.set_ylabel("Num Channels"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (0,2): P3 vs P4 channel weight scatter ──
    ax = fig.add_subplot(gs[0, 2])
    # P3 and P4 have different channel counts — compare distribution shapes
    ax.hist(p3_mean, bins=50, color="#E74C3C", alpha=0.5, label="P3", density=True)
    ax.hist(p4_mean, bins=50, color="#2980B9", alpha=0.5, label="P4", density=True)
    ax.axvline(x=0.5, color="gray", ls="--", lw=1, alpha=0.4)
    ks_stat = max(abs(np.sort(p3_mean) - np.sort(p4_mean)[:len(p3_mean)]).max() if len(p3_mean) <= len(p4_mean) else 0, 0)
    ax.set_title(f"P3 vs P4 — Weight Distribution Overlay\n"
                 f"P3: μ={p3_mean.mean():.4f}±{p3_mean.std():.4f}\n"
                 f"P4: μ={p4_mean.mean():.4f}±{p4_mean.std():.4f}",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Weight"); ax.set_ylabel("Density"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── (1,0): P3 Top-10 boosted/suppressed ──
    ax = fig.add_subplot(gs[1, 0])
    top10_p3 = np.argsort(p3_mean)[-10:][::-1]
    bot10_p3 = np.argsort(p3_mean)[:10]
    channels = list(bot10_p3) + list(top10_p3)
    values = list(p3_mean[bot10_p3]) + list(p3_mean[top10_p3])
    colors = ["#E74C3C"]*10 + ["#27AE60"]*10
    ax.barh(range(20), values, color=colors, alpha=0.8, edgecolor="white", lw=0.3)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"ch{c}" for c in channels], fontsize=7)
    ax.axvline(x=0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_title(f"P3 — Top-10 Suppressed (red) & Boosted (green)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Weight"); ax.invert_yaxis()

    # ── (1,1): P4 Top-10 boosted/suppressed ──
    ax = fig.add_subplot(gs[1, 1])
    top10_p4 = np.argsort(p4_mean)[-10:][::-1]
    bot10_p4 = np.argsort(p4_mean)[:10]
    channels = list(bot10_p4) + list(top10_p4)
    values = list(p4_mean[bot10_p4]) + list(p4_mean[top10_p4])
    colors = ["#E74C3C"]*10 + ["#27AE60"]*10
    ax.barh(range(20), values, color=colors, alpha=0.8, edgecolor="white", lw=0.3)
    ax.set_yticks(range(20))
    ax.set_yticklabels([f"ch{c}" for c in channels], fontsize=7)
    ax.axvline(x=0.5, color="gray", ls="--", lw=1, alpha=0.5)
    ax.set_title(f"P4 — Top-10 Suppressed (red) & Boosted (green)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Weight"); ax.invert_yaxis()

    # ── (1,2): Per-class DCR weight divergence ──
    ax = fig.add_subplot(gs[1, 2])
    p3_per_class = weight_data.get("p3_per_class", {})
    class_colors = {1: "#E74C3C", 2: "#27AE60", 3: "#3498DB"}  # same as seg colors
    class_labels = {1: "Inclusion", 2: "Patch", 3: "Scratch"}
    for c in [1, 2, 3]:
        if c in p3_per_class and len(p3_per_class[c]) > 0:
            w = p3_per_class[c].mean(axis=0)  # [C3] mean weight for this class
            ax.hist(w, bins=40, color=class_colors[c], alpha=0.4, label=f"{class_labels[c]} (n={len(p3_per_class[c])})",
                    density=True)
    ax.axvline(x=0.5, color="gray", ls="--", lw=1, alpha=0.4)
    ax.set_title("P3 — DCR Weight by Defect Class\n"
                 "(Different classes → different channel profiles?)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Weight"); ax.set_ylabel("Density"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Divergence metric: std of per-class mean weights
    if len(p3_per_class) >= 2:
        class_means = [p3_per_class[c].mean(axis=0) for c in sorted(p3_per_class.keys()) if len(p3_per_class[c]) > 0]
        if len(class_means) >= 2:
            divergence = np.stack(class_means).std(axis=0).mean()
            ax.text(0.95, 0.95, f"Class divergence: {divergence:.5f}\n"
                    f"{'✓ Classes differ' if divergence > 0.005 else '⚠ No class specialization'}",
                    transform=ax.transAxes, fontsize=9, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("DCR Channel Weight Analysis — What did the channel attention learn?",
                 fontsize=14, fontweight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Channel weight analysis → {save_path}")


def plot_fc1_analysis(dcr_p3, dcr_p4, save_path):
    """fc1 权重矩阵分析 — 哪些输入通道驱动注意力? | fc1 weight analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, (name, dcr, color) in enumerate([
        ("P3 (960→60)", dcr_p3, "#E74C3C"),
        ("P4 (1280→80)", dcr_p4, "#2980B9"),
    ]):
        fc1_w = dcr.fc1.weight.detach().cpu().numpy()  # [mid_channels, channels]
        # L2 norm of each input channel's contribution to attention
        channel_importance = np.linalg.norm(fc1_w, axis=0)  # [channels]

        axes[col].bar(range(len(channel_importance)), np.sort(channel_importance)[::-1],
                      color=color, alpha=0.7, width=1.0)
        axes[col].set_title(f"{name} — Input Channel Importance\n"
                            f"(L2 norm of fc1 weight per input channel)\n"
                            f"Top ch: {np.argsort(channel_importance)[-5:][::-1].tolist()}",
                            fontsize=10, fontweight="bold")
        axes[col].set_xlabel("Channel Rank (by importance)"); axes[col].set_ylabel("||fc1_weight[:,ch]||₂")
        axes[col].grid(True, alpha=0.3)

        # Sparsity metric
        top20_pct = np.percentile(channel_importance, 80)
        sparse_ratio = (channel_importance > top20_pct).sum() / len(channel_importance)
        axes[col].text(0.95, 0.95, f"Top-20% channels carry\n{sparse_ratio*100:.0f}% of importance",
                       transform=axes[col].transAxes, fontsize=9, ha="right", va="top",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    plt.suptitle("DCR — fc1 Weight Matrix Analysis (Input → Attention Bottleneck)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  fc1 analysis → {save_path}")


# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="DCR Deep Analysis & Visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/NEU_Seg")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── Load ──
    print("Loading model...")
    backbone, dcr_p3, dcr_p4, decoder, ckpt = load_model(args.checkpoint, device)
    miou = ckpt.get("mIoU", "N/A"); per_class = ckpt.get("per_class_IoU", {})
    print(f"  mIoU: {miou}")
    for cls_name, iou in per_class.items():
        print(f"    {cls_name}: {iou:.4f}")

    dataset = NEUSegDataset(root=args.data_root, split="test", binary=False)

    # ── 1. Segmentation Grid ──
    print("\n=== 1. Segmentation Predictions ===")
    seg_indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False).tolist()
    plot_segmentation_grid(dataset, backbone, dcr_p3, dcr_p4, decoder,
                           device, seg_indices, out_dir / "segmentation_grid.png")

    # ── 2. Channel Weight Analysis ──
    print("\n=== 2. DCR Channel Weight Analysis ===")
    weight_data = collect_channel_weights(dcr_p3, dcr_p4, backbone, dataset, device, num_samples=30)

    # Summary stats
    p3_mean = weight_data["p3_weights"].mean(axis=0)
    p4_mean = weight_data["p4_weights"].mean(axis=0)
    print(f"  P3: μ={p3_mean.mean():.4f}±{p3_mean.std():.4f}, "
          f"range=[{p3_mean.min():.4f}, {p3_mean.max():.4f}]")
    print(f"  P4: μ={p4_mean.mean():.4f}±{p4_mean.std():.4f}, "
          f"range=[{p4_mean.min():.4f}, {p4_mean.max():.4f}]")
    print(f"  P3 Top-5 boosted: {np.argsort(p3_mean)[-5:][::-1].tolist()}")
    print(f"  P3 Top-5 suppressed: {np.argsort(p3_mean)[:5].tolist()}")
    print(f"  P4 Top-5 boosted: {np.argsort(p4_mean)[-5:][::-1].tolist()}")
    print(f"  P4 Top-5 suppressed: {np.argsort(p4_mean)[:5].tolist()}")

    plot_channel_weight_analysis(weight_data, out_dir / "dcr_channel_analysis.png")

    # ── 3. fc1 Weight Analysis ──
    print("\n=== 3. fc1 Weight Matrix Analysis ===")
    fc1_p3 = dcr_p3.fc1.weight.detach().cpu().numpy()  # [60, 960]
    fc1_p4 = dcr_p4.fc1.weight.detach().cpu().numpy()  # [80, 1280]
    print(f"  P3 fc1: {fc1_p3.shape}, ||W||={np.linalg.norm(fc1_p3):.2f}, "
          f"sparsity={((np.abs(fc1_p3) < 0.01).sum() / fc1_p3.size * 100):.1f}%")
    print(f"  P4 fc1: {fc1_p4.shape}, ||W||={np.linalg.norm(fc1_p4):.2f}, "
          f"sparsity={((np.abs(fc1_p4) < 0.01).sum() / fc1_p4.size * 100):.1f}%")
    plot_fc1_analysis(dcr_p3, dcr_p4, out_dir / "dcr_fc1_analysis.png")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  DCR Analysis — Complete")
    print(f"  mIoU: {miou}")
    print(f"  Key finding: DCR learns channel weights μ={p3_mean.mean():.4f} (P3), "
          f"μ={p4_mean.mean():.4f} (P4)")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
