#!/usr/bin/env python3
"""
Foundation Feature Utilization Analysis | 基础模型特征利用率分析.
===================================================================

严格基于 DCR 学到的通道权重, 分析 FastSAM SA-1B 特征在工业缺陷上的利用模式.
Rigorously analyze FastSAM SA-1B feature utilization patterns on industrial defects.

分析内容 | Analysis:
    1. FUR (Feature Utilization Ratio) — P3/P4 的有用通道比例
    2. Per-class Top-K Channel Jaccard Overlap — 不同缺陷共享多少通道
    3. Per-sample channel weight consistency — 跨样本的权重一致性
    4. Weight distribution statistics — 双峰特性量化

原则 | Principles:
    - 不声称 "information content", 只声称 "utilization"
    - 不声称 "only N channels contain info", 只声称 "N channels are consistently utilized"
    - 区分 "decoder utility" vs "intrinsic information"

用法 | Usage::

    python tools/diag/diag_foundation_feature_analysis.py \
        --checkpoint runs/neuseg_DAFRN_DCR_0718_1205/best_model.pt \
        --num-samples 100 --top-k 50 --device cuda
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
from tqdm import tqdm

from adatile.backbone import FastSAMBackbone
from adatile.rectify.dcr import DefectChannelReweighting
from adatile.datasets.neu_seg import NEUSegDataset

CLASS_NAMES = ["BG", "Inclusion", "Patch", "Scratch"]


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

    return backbone, dcr_p3, dcr_p4, ckpt


@torch.no_grad()
def collect_per_sample_weights(dcr_p3, dcr_p4, backbone, dataset, device,
                               num_samples: int = 100):
    """
    逐样本收集通道权重, 按类别分组. | Per-sample channel weights, grouped by class.

    Returns:
        p3_all: [N, C3] — all samples
        p4_all: [N, C4] — all samples
        p3_per_class: {class_id: [n_i, C3]} — per-class weights
        p4_per_class: {class_id: [n_i, C4]}
        class_presence: [N, 3] — which classes are present in each sample
    """
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    p3_all, p4_all = [], []
    p3_per_class = {1: [], 2: [], 3: []}  # Inclusion, Patch, Scratch
    p4_per_class = {1: [], 2: [], 3: []}
    class_presence = []

    for idx in tqdm(indices, desc="Collecting weights", leave=False):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["masks"].squeeze(0).long().numpy()
        H, W = img.shape[2], img.shape[3]
        pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)

        feats = backbone(img, extract_proto=False)
        w3 = dcr_p3.get_channel_weights(feats["p3"]).squeeze(0).cpu().numpy()  # [C3]
        w4 = dcr_p4.get_channel_weights(feats["p4"]).squeeze(0).cpu().numpy()  # [C4]

        p3_all.append(w3); p4_all.append(w4)

        present = np.unique(gt)
        presence_vec = [1 if c in present else 0 for c in [1, 2, 3]]
        class_presence.append(presence_vec)

        for c in [1, 2, 3]:
            if c in present:
                p3_per_class[c].append(w3)
                p4_per_class[c].append(w4)

    return {
        "p3_all": np.array(p3_all),           # [N, C3]
        "p4_all": np.array(p4_all),           # [N, C4]
        "p3_per_class": {c: np.array(v) for c, v in p3_per_class.items() if len(v) > 0},
        "p4_per_class": {c: np.array(v) for c, v in p4_per_class.items() if len(v) > 0},
        "class_presence": np.array(class_presence),  # [N, 3]
    }


# ═══════════════════════════════════════════════════════════════════
# 分析 1: FUR (Feature Utilization Ratio) | 特征利用率
# ═══════════════════════════════════════════════════════════════════

def compute_fur(weights: np.ndarray, threshold: float = 0.55) -> dict:
    """
    Compute FUR = fraction of channels with mean weight > threshold.

    :param weights: [N, C] per-sample channel weights.
    :param threshold: weights above this are "utilized".
    :return: dict with fur, mean_weight, num_utilized, etc.
    """
    mean_w = weights.mean(axis=0)  # [C]
    n_total = len(mean_w)
    n_utilized = int((mean_w > threshold).sum())
    n_suppressed = int((mean_w < 0.45).sum())
    n_neutral = n_total - n_utilized - n_suppressed

    return {
        "FUR": round(n_utilized / n_total, 4),
        "FUR_pct": round(n_utilized / n_total * 100, 2),
        "suppressed_pct": round(n_suppressed / n_total * 100, 2),
        "neutral_pct": round(n_neutral / n_total * 100, 2),
        "n_total": n_total,
        "n_utilized": n_utilized,
        "n_suppressed": n_suppressed,
        "mean_weight": round(float(mean_w.mean()), 4),
        "std_weight": round(float(mean_w.std()), 4),
        "range": [round(float(mean_w.min()), 4), round(float(mean_w.max()), 4)],
    }


# ═══════════════════════════════════════════════════════════════════
# 分析 2: Per-class Top-K Jaccard Overlap | 类别间 Top-K 重叠率
# ═══════════════════════════════════════════════════════════════════

def compute_class_overlap(per_class_weights: dict, top_k: int = 50) -> dict:
    """
    计算不同缺陷类别之间的 Top-K 通道重叠 (Jaccard Index).
    Compute Top-K channel Jaccard overlap between defect classes.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B| at Top-K.

    :param per_class_weights: {class_id: [n_samples, C]} mean weight per class.
    :param top_k: number of top channels to consider.
    :return: dict with pairwise Jaccard and annotation.
    """
    classes = sorted(per_class_weights.keys())
    if len(classes) < 2:
        return {"error": "Need at least 2 classes for overlap analysis"}

    # Per-class mean weight → Top-K channel indices
    class_topk = {}
    for c in classes:
        mean_w = per_class_weights[c].mean(axis=0)  # [C]
        topk_idx = set(np.argsort(mean_w)[-top_k:].tolist())
        class_topk[c] = topk_idx

    # Pairwise Jaccard
    pairwise = {}
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            a, b = class_topk[c1], class_topk[c2]
            intersection = len(a & b)
            union = len(a | b)
            jaccard = intersection / union if union > 0 else 0
            key = f"{CLASS_NAMES[c1]}_{CLASS_NAMES[c2]}"
            pairwise[key] = {
                "intersection": intersection,
                "union": union,
                "jaccard": round(jaccard, 4),
                "overlap_pct": round(intersection / top_k * 100, 1),
            }

    # Mean Jaccard across all pairs
    jaccard_vals = [v["jaccard"] for v in pairwise.values()]
    mean_jaccard = np.mean(jaccard_vals) if jaccard_vals else 0

    # Three-way overlap (all classes)
    if len(classes) >= 3:
        three_way = class_topk[classes[0]] & class_topk[classes[1]] & class_topk[classes[2]]
    else:
        three_way = set()

    return {
        "top_k": top_k,
        "pairwise": pairwise,
        "mean_jaccard": round(float(mean_jaccard), 4),
        "three_way_overlap": len(three_way),
        "three_way_overlap_pct": round(len(three_way) / top_k * 100, 1),
        "class_topk_channels": {CLASS_NAMES[c]: sorted(list(s)) for c, s in class_topk.items()},
    }


# ═══════════════════════════════════════════════════════════════════
# 分析 3: Per-sample weight consistency | 跨样本权重一致性
# ═══════════════════════════════════════════════════════════════════

def compute_sample_consistency(weights: np.ndarray, top_k: int = 50) -> dict:
    """
    计算跨样本的 Top-K 通道选择一致性.
    Compute cross-sample Top-K channel selection consistency.

    对每对样本计算 Top-K Jaccard → 取均值 → 衡量"选择稳定性".
    Pairwise Top-K Jaccard across samples → mean → "selection stability."

    :param weights: [N, C] per-sample weights.
    :param top_k: number of top channels.
    :return: dict with consistency metrics.
    """
    N = weights.shape[0]
    if N < 2:
        return {"error": "Need at least 2 samples"}

    # Per-sample Top-K
    sample_topk = []
    for i in range(N):
        topk = set(np.argsort(weights[i])[-top_k:].tolist())
        sample_topk.append(topk)

    # Pairwise Jaccard between samples
    jaccards = []
    for i in range(min(N, 50)):  # Limit pairs to avoid O(N²)
        for j in range(i + 1, min(N, 50)):
            a, b = sample_topk[i], sample_topk[j]
            inter = len(a & b)
            union = len(a | b)
            jaccards.append(inter / union if union > 0 else 0)

    jaccards = np.array(jaccards)

    # Channel selection frequency: how often does each channel appear in Top-K?
    channel_freq = np.zeros(weights.shape[1])
    for s in sample_topk:
        for ch in s:
            channel_freq[ch] += 1
    channel_freq /= N  # [C] — selection frequency per channel

    # "Core channels": appear in >80% of samples
    core_mask = channel_freq > 0.8
    core_count = int(core_mask.sum())

    return {
        "mean_pairwise_jaccard": round(float(jaccards.mean()), 4),
        "std_pairwise_jaccard": round(float(jaccards.std()), 4),
        "core_channels_count": core_count,
        "core_channels_pct": round(core_count / weights.shape[1] * 100, 2),
        "top_k": top_k,
        "n_samples": N,
    }


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_fur_summary(fur_p3: dict, fur_p4: dict, save_path: Path):
    """FUR 可视化 | FUR visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, (name, fur, color) in enumerate([
        ("P3 (960 channels)", fur_p3, "#E74C3C"),
        ("P4 (1280 channels)", fur_p4, "#2980B9"),
    ]):
        # Pie chart: utilized vs suppressed vs neutral
        sizes = [fur["FUR_pct"], fur["suppressed_pct"], fur["neutral_pct"]]
        labels = [f"Utilized\n(>{fur.get('threshold', 0.55)}): {fur['n_utilized']} ch",
                  f"Suppressed\n(<0.45): {fur['n_suppressed']} ch",
                  f"Neutral\n(0.45-0.55): {fur['n_total']-fur['n_utilized']-fur['n_suppressed']} ch"]
        colors_pie = ["#27AE60", "#E74C3C", "#95A5A6"]
        explode = (0.05, 0, 0)

        axes[col].pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                      autopct='%1.1f%%', startangle=90, textprops={'fontsize': 9})
        axes[col].set_title(f"{name} — Feature Utilization Ratio (FUR)\n"
                            f"μ={fur['mean_weight']}, σ={fur['std_weight']}\n"
                            f"FUR = {fur['FUR_pct']}% ({fur['n_utilized']}/{fur['n_total']})",
                            fontsize=10, fontweight="bold")

    plt.suptitle("Foundation Feature Utilization — How many SA-1B channels are useful for defects?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  FUR plot → {save_path}")


def plot_class_overlap(p3_overlap: dict, p4_overlap: dict, save_path: Path):
    """Per-class overlap visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col, (name, overlap, color) in enumerate([
        ("P3", p3_overlap, "#E74C3C"),
        ("P4", p4_overlap, "#2980B9"),
    ]):
        pairwise = overlap.get("pairwise", {})
        if not pairwise:
            axes[col].text(0.5, 0.5, "No overlap data", ha="center", va="center",
                           transform=axes[col].transAxes)
            continue

        pairs = list(pairwise.keys())
        jaccards = [pairwise[p]["jaccard"] for p in pairs]
        overlaps = [pairwise[p]["overlap_pct"] for p in pairs]

        x = np.arange(len(pairs))
        width = 0.35
        bars1 = axes[col].bar(x - width/2, [pairwise[p]["intersection"] for p in pairs],
                              width, label="Intersection |A∩B|", color="#27AE60", alpha=0.8)
        bars2 = axes[col].bar(x + width/2, [pairwise[p]["union"] for p in pairs],
                              width, label="Union |A∪B|", color="#3498DB", alpha=0.5)

        # Annotate bars with Jaccard
        for i, (bar, jac) in enumerate(zip(bars1, jaccards)):
            axes[col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f"J={jac:.3f}", ha="center", fontsize=9, fontweight="bold")

        axes[col].set_xticks(x)
        axes[col].set_xticklabels([p.replace("_", "\n") for p in pairs], fontsize=8)
        axes[col].set_title(f"{name} — Per-Class Top-{overlap['top_k']} Channel Overlap\n"
                            f"Mean Jaccard={overlap['mean_jaccard']:.4f}, "
                            f"3-way overlap={overlap.get('three_way_overlap', 0)}/{overlap['top_k']}",
                            fontsize=10, fontweight="bold")
        axes[col].set_ylabel("Channel Count"); axes[col].legend(fontsize=8); axes[col].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Do Different Defect Types Use Different Foundation Feature Channels?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Class overlap plot → {save_path}")


def plot_sample_consistency(p3_cons: dict, p4_cons: dict, weight_data: dict, save_path: Path):
    """Sample consistency & channel frequency visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (name, cons, weights, color) in enumerate([
        ("P3", p3_cons, weight_data["p3_all"], "#E74C3C"),
        ("P4", p4_cons, weight_data["p4_all"], "#2980B9"),
    ]):
        # ── Row 0: Per-channel selection frequency ──
        N, C = weights.shape
        top_k = cons["top_k"]
        channel_freq = np.zeros(C)
        for i in range(N):
            topk = set(np.argsort(weights[i])[-top_k:].tolist())
            for ch in topk:
                channel_freq[ch] += 1
        channel_freq /= N

        axes[0, col].bar(range(C), np.sort(channel_freq)[::-1], color=color, alpha=0.7, width=1.0)
        axes[0, col].axhline(y=0.8, color="gray", ls="--", lw=1, alpha=0.5, label="Core (>80%)")
        axes[0, col].axhline(y=0.5, color="gray", ls=":", lw=1, alpha=0.3, label="Frequent (>50%)")
        core_n = int((channel_freq > 0.8).sum())
        axes[0, col].set_title(f"{name} — Channel Selection Frequency (Top-{top_k})\n"
                               f"Core ch (>80% samples): {core_n}/{C} ({core_n/C*100:.1f}%)",
                               fontsize=10, fontweight="bold")
        axes[0, col].set_xlabel("Channel Rank (by frequency)"); axes[0, col].set_ylabel("Selection Frequency")
        axes[0, col].legend(fontsize=8); axes[0, col].grid(True, alpha=0.3)

        # ── Row 1: Pairwise sample Jaccard histogram ──
        # Already computed in cons
        axes[1, col].text(0.5, 0.5,
                          f"Mean pairwise Jaccard: {cons['mean_pairwise_jaccard']:.4f}\n"
                          f"Std: {cons['std_pairwise_jaccard']:.4f}\n"
                          f"Core channels (>80%): {cons['core_channels_count']}\n"
                          f"{'✓ Stable selection' if cons['mean_pairwise_jaccard'] > 0.3 else '⚠ Variable selection'}",
                          ha="center", va="center", fontsize=14,
                          transform=axes[1, col].transAxes,
                          bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        axes[1, col].set_title(f"{name} — Cross-Sample Consistency", fontsize=11, fontweight="bold")

    plt.suptitle("How Consistent is DCR's Channel Selection Across Samples?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Sample consistency plot → {save_path}")


# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Foundation Feature Utilization Analysis")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/NEU_Seg")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent / "foundation_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── Load ──
    print("Loading model...")
    backbone, dcr_p3, dcr_p4, ckpt = load_model(args.checkpoint, device)
    print(f"  mIoU: {ckpt.get('mIoU', 'N/A')}")
    dataset = NEUSegDataset(root=args.data_root, split="test", binary=False)

    # ── Collect data ──
    print(f"\nCollecting weights from {args.num_samples} samples...")
    data = collect_per_sample_weights(dcr_p3, dcr_p4, backbone, dataset, device,
                                      num_samples=args.num_samples)

    # ═══════════════════════════════════════════════════════════════
    # Analysis 1: FUR
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Analysis 1: Feature Utilization Ratio (FUR)")
    print("  Definition: fraction of channels with mean DCR weight > 0.55")
    print("=" * 60)

    fur_p3 = compute_fur(data["p3_all"])
    fur_p4 = compute_fur(data["p4_all"])

    print(f"\n  P3 (960 ch):")
    print(f"    FUR = {fur_p3['FUR_pct']}% ({fur_p3['n_utilized']}/{fur_p3['n_total']} channels)")
    print(f"    Suppressed (<0.45): {fur_p3['suppressed_pct']}% ({fur_p3['n_suppressed']} channels)")
    print(f"    Mean weight: {fur_p3['mean_weight']} ± {fur_p3['std_weight']}")
    print(f"    Range: [{fur_p3['range'][0]}, {fur_p3['range'][1]}]")

    print(f"\n  P4 (1280 ch):")
    print(f"    FUR = {fur_p4['FUR_pct']}% ({fur_p4['n_utilized']}/{fur_p4['n_total']} channels)")
    print(f"    Suppressed (<0.45): {fur_p4['suppressed_pct']}% ({fur_p4['n_suppressed']} channels)")
    print(f"    Mean weight: {fur_p4['mean_weight']} ± {fur_p4['std_weight']}")
    print(f"    Range: [{fur_p4['range'][0]}, {fur_p4['range'][1]}]")

    plot_fur_summary(fur_p3, fur_p4, out_dir / "fur_summary.png")

    # ═══════════════════════════════════════════════════════════════
    # Analysis 2: Per-class Top-K Overlap
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  Analysis 2: Per-Class Top-{args.top_k} Channel Jaccard Overlap")
    print("  Definition: Jaccard(A,B) = |A∩B| / |A∪B| of per-class Top-K")
    print("=" * 60)

    p3_overlap = compute_class_overlap(data["p3_per_class"], top_k=args.top_k)
    p4_overlap = compute_class_overlap(data["p4_per_class"], top_k=args.top_k)

    for level, overlap in [("P3", p3_overlap), ("P4", p4_overlap)]:
        print(f"\n  {level}:")
        print(f"    Mean pairwise Jaccard: {overlap['mean_jaccard']}")
        print(f"    3-way overlap: {overlap['three_way_overlap']}/{args.top_k} "
              f"({overlap['three_way_overlap_pct']}%)")
        for pair, info in overlap.get("pairwise", {}).items():
            print(f"    {pair}: Jaccard={info['jaccard']}, "
                  f"overlap={info['overlap_pct']}% ({info['intersection']}/{args.top_k})")

    plot_class_overlap(p3_overlap, p4_overlap, out_dir / "class_overlap.png")

    # ═══════════════════════════════════════════════════════════════
    # Analysis 3: Cross-sample consistency
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print(f"  Analysis 3: Cross-Sample Channel Selection Consistency")
    print(f"  Definition: pairwise Top-{args.top_k} Jaccard between samples")
    print("=" * 60)

    p3_cons = compute_sample_consistency(data["p3_all"], top_k=args.top_k)
    p4_cons = compute_sample_consistency(data["p4_all"], top_k=args.top_k)

    for level, cons in [("P3", p3_cons), ("P4", p4_cons)]:
        print(f"\n  {level}:")
        print(f"    Mean pairwise Jaccard: {cons['mean_pairwise_jaccard']}")
        print(f"    Core channels (>80% samples): {cons['core_channels_count']} "
              f"({cons['core_channels_pct']}%)")

    plot_sample_consistency(p3_cons, p4_cons, data, out_dir / "sample_consistency.png")

    # ═══════════════════════════════════════════════════════════════
    # Save JSON results
    # ═══════════════════════════════════════════════════════════════
    results = {
        "checkpoint": str(args.checkpoint),
        "num_samples": args.num_samples,
        "top_k": args.top_k,
        "fur": {
            "p3": {k: v for k, v in fur_p3.items()},
            "p4": {k: v for k, v in fur_p4.items()},
        },
        "class_overlap": {
            "p3": {k: v for k, v in p3_overlap.items() if k != "class_topk_channels"},
            "p4": {k: v for k, v in p4_overlap.items() if k != "class_topk_channels"},
        },
        "sample_consistency": {
            "p3": p3_cons,
            "p4": p4_cons,
        },
    }
    with open(out_dir / "foundation_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_dir / 'foundation_analysis.json'}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Foundation Feature Analysis — Complete")
    print(f"  Key findings:")
    print(f"    FUR P3: {fur_p3['FUR_pct']}%  |  FUR P4: {fur_p4['FUR_pct']}%")
    print(f"    Class overlap (mean Jaccard): P3={p3_overlap['mean_jaccard']}, P4={p4_overlap['mean_jaccard']}")
    print(f"    Cross-sample consistency: P3={p3_cons['mean_pairwise_jaccard']}, P4={p4_cons['mean_pairwise_jaccard']}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
