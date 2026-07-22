#!/usr/bin/env python3
"""
Prototype 质量诊断 — Severstal | Prototype Quality Diagnosis — Severstal.
===========================================================================

对比 Frozen vs LoRA backbone: LoRA 是否真正改善了 Prototype 的质量？
Compare Frozen vs LoRA backbone: does LoRA genuinely improve prototype quality?

核心问题 | Core Question:
    LoRA 的收益是来自 "Prototype 变好了" 还是仅仅 "模型参数变多了"？
    Is LoRA's gain from "better prototypes" or just "more parameters"?

指标 | Metrics:
    1. 同类 Support-Query Cosine Similarity → 越高越好 (Prototype 代表性强)
    2. 异类 Support-Query Cosine Similarity → 越低越好 (类别区分度高)
    3. 同类区分度比率 (Sim_same / Sim_diff) → 越高越好
    4. Intra-class Variance (类内方差) → 越低越好 (同类 Prototype 聚集)
    5. Inter-class Distance (类间距离) → 越高越好 (不同类分离)
    6. Silhouette Score → [-1, 1], 综合衡量聚类质量
    7. t-SNE 可视化 → 直观判断类别分离

输出 | Output:
    diag_output/severstal_proto_quality_{ts}/
    ├── proto_quality.json          # 全部指标
    ├── tsne_prototypes.png          # t-SNE 可视化
    ├── cosine_similarity_heatmap.png # 同类/异类余弦相似度热力图
    └── proto_distance_bar.png       # 每类 Intra/Inter 距离对比

用法 | Usage::

    # 对比 Frozen vs LoRA (推荐) | Compare Frozen vs LoRA (recommended)
    python tools/diag/diag_severstal_proto_quality.py \
        --checkpoint-a runs/severstal_episodic_multi_NoLoRA_.../K1_S42/best_model.pt \
        --checkpoint-b runs/severstal_episodic_multi_LoRA_r4_.../K1_S42/best_model.pt \
        --label-a Frozen --label-b LoRA_r4 \
        --data-root data/severstal-steel-defect-detection \
        --device cuda

    # 单 checkpoint 分析 | Single checkpoint analysis
    python tools/diag/diag_severstal_proto_quality.py \
        --checkpoint-b runs/.../best_model.pt --label-b LoRA_r4 \
        --data-root data/severstal-steel-defect-detection \
        --device cuda
"""

from __future__ import annotations

import sys, argparse, json, random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from adatile.backbone import FastSAMBackbone
from adatile.datasets.severstal import SeverstalDataset

# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

IMG_H, IMG_W = 256, 1600  # Severstal native
DEFECT_CLASSES = [1, 2, 3, 4]
CLASS_NAMES = {1: "Class1", 2: "Class2", 3: "Class3", 4: "Class4"}
CLASS_COLORS = {1: "#E74C3C", 2: "#3498DB", 3: "#2ECC71", 4: "#F39C12"}


# ═══════════════════════════════════════════════════════════════════
# Prototype 计算 (与训练脚本一致) | Prototype Computation (same as training)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_prototype(
    backbone: FastSAMBackbone,
    images: torch.Tensor,           # [N, 3, H, W]
    masks: torch.Tensor,            # [N, H, W] binary FG mask
    device: torch.device,
) -> torch.Tensor:
    """
    从 N 张图像计算 L2-normalized FG prototype (诊断模式: 始终 no_grad).
    Compute L2-normalized FG prototype from N images (diagnostic mode: always no_grad).

    :return: [1280] L2-normalized prototype vector.
    """
    N = images.shape[0]
    feats_list = []

    for i in range(N):
        img = images[i:i + 1].to(device)
        mask = masks[i].to(device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        feats = backbone(img)
        p4 = feats["p4"]  # [1, 1280, H/16, W/16]

        _, _, H_p4, W_p4 = p4.shape
        mask_p4 = F.interpolate(
            mask.unsqueeze(0).float(), size=(H_p4, W_p4), mode="nearest"
        ).squeeze(0)  # [1, H_p4, W_p4]

        fg_area = mask_p4.sum()
        if fg_area > 0:
            proto = (p4.squeeze(0) * mask_p4).sum(dim=(1, 2)) / (fg_area + 1e-8)
            feats_list.append(proto)

    if not feats_list:
        return torch.zeros(1280, device=device)

    proto = torch.stack(feats_list).mean(dim=0)  # [1280]
    return F.normalize(proto, dim=0, p=2)


def get_class_mask(sample: dict, class_id: int) -> torch.Tensor:
    """从数据集样本中提取指定类别的二值掩码。"""
    mask = sample["masks"].squeeze(0)  # [H, W]
    if mask.max() > 1:
        return (mask == class_id).float()
    return mask.float()


# ═══════════════════════════════════════════════════════════════════
# Backbone 加载器 (支持 LoRA checkpoint 恢复) | Backbone Loader
# ═══════════════════════════════════════════════════════════════════

def load_backbone(
    checkpoint_path: str | None,
    device: torch.device,
    logger,
) -> FastSAMBackbone:
    """
    加载 backbone, 如果 checkpoint 含 LoRA 权重则恢复。
    Load backbone, restore LoRA weights if present in checkpoint.

    :param checkpoint_path: 训练 checkpoint 路径 (可为 None → 返回纯 Frozen backbone).
    :return: (backbone, has_lora, lora_rank)
    """
    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint="thirdLibrary/FastSAM/weights/FastSAM-x.pt",
    ).to(device)
    backbone.eval()

    if checkpoint_path is None:
        logger("Backbone: FastSAM-x (Frozen, no checkpoint)")
        return backbone

    ckpt = torch.load(checkpoint_path, map_location=device)
    lora_rank = ckpt.get("lora_rank", 0)
    has_lora = lora_rank > 0 and "lora_state_dict" in ckpt

    if has_lora:
        # ── 注入 ConvLoRA + 恢复权重 | Inject ConvLoRA + restore weights ──
        backbone.apply_conv_lora(rank=lora_rank, alpha=1.0)
        backbone.model.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
        backbone.eval()
        lora_params = sum(p.numel() for p in backbone.get_lora_parameters()
                         ) if hasattr(backbone, 'get_lora_parameters') else 0
        logger(
            f"Backbone: FastSAM-x + ConvLoRA r={lora_rank} restored "
            f"(+{lora_params/1e3:.1f}K LoRA params)"
        )
    else:
        logger("Backbone: FastSAM-x (Frozen, no LoRA in checkpoint)")

    # ── Warm-up ──
    with torch.no_grad():
        backbone(torch.randn(1, 3, IMG_H, IMG_W, device=device), extract_proto=False)

    return backbone


# ═══════════════════════════════════════════════════════════════════
# Prototype 采集 | Prototype Collection
# ═══════════════════════════════════════════════════════════════════

def collect_prototypes(
    backbone: FastSAMBackbone,
    dataset: SeverstalDataset,
    device: torch.device,
    k_shot: int = 1,
    num_queries_per_class: int = 30,
    seed: int = 42,
) -> dict:
    """
    采集所有类的 Support Prototype + Query Prototypes。
    Collect Support Prototype + Query Prototypes for all classes.

    :return: {
        "support": {cls_id: np.ndarray [1280]},       # support prototype
        "queries": {cls_id: np.ndarray [N, 1280]},     # per-query prototypes
        "query_labels": {cls_id: [int]},                # class labels (for t-SNE)
        "query_class_ids": [int],                       # class IDs for each query proto
        "all_protos": np.ndarray [M, 1280],             # all prototypes (support+query)
        "all_labels": [str],                            # labels for each proto
    }
    """
    rng = random.Random(seed)

    support_protos: dict[int, np.ndarray] = {}
    query_protos: dict[int, list[np.ndarray]] = defaultdict(list)

    for cls_id in DEFECT_CLASSES:
        pool = dataset.class_to_images(cls_id)
        n_available = len(pool)
        n_needed = k_shot + num_queries_per_class

        if n_available < n_needed:
            print(f"  ⚠ Class {cls_id}: only {n_available} images, "
                  f"need {n_needed}. Using all available.")
            sampled = pool[:]
        else:
            sampled = rng.sample(pool, n_needed)

        support_indices = sampled[:k_shot]
        query_indices = sampled[k_shot:k_shot + num_queries_per_class]

        # ── Support Prototype | K 张 → 1 个 prototype ──
        s_imgs, s_masks = [], []
        for idx in support_indices:
            s = dataset[idx]
            s_imgs.append(s["image"])
            s_masks.append(get_class_mask(s, cls_id))
        s_imgs_t = torch.stack(s_imgs)
        s_masks_t = torch.stack(s_masks)
        sp = compute_prototype(backbone, s_imgs_t, s_masks_t, device)
        support_protos[cls_id] = sp.cpu().numpy()

        # ── Query Prototypes | 每张 query → 1 个 prototype (使用 GT mask) ──
        for idx in query_indices:
            q = dataset[idx]
            q_img = q["image"].unsqueeze(0)           # [1, 3, H, W]
            q_mask = get_class_mask(q, cls_id).unsqueeze(0)  # [1, H, W]
            qp = compute_prototype(backbone, q_img, q_mask, device)
            query_protos[cls_id].append(qp.cpu().numpy())

    # ── 汇总 all_protos + labels (用于 t-SNE) | Aggregate for t-SNE ──
    all_protos_list = []
    all_labels_list = []
    query_class_ids = []

    # Support prototypes
    for cls_id in DEFECT_CLASSES:
        all_protos_list.append(support_protos[cls_id])
        all_labels_list.append(f"S_{CLASS_NAMES[cls_id]}")

    # Query prototypes
    for cls_id in DEFECT_CLASSES:
        for qp in query_protos[cls_id]:
            all_protos_list.append(qp)
            all_labels_list.append(f"Q_{CLASS_NAMES[cls_id]}")
            query_class_ids.append(cls_id)

    all_protos = np.stack(all_protos_list, axis=0)  # [M, 1280]

    return {
        "support": support_protos,
        "queries": {k: np.stack(v) for k, v in query_protos.items()},
        "all_protos": all_protos,
        "all_labels": all_labels_list,
        "query_class_ids": query_class_ids,
    }


# ═══════════════════════════════════════════════════════════════════
# 指标计算 | Metrics Computation
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(proto_data: dict) -> dict:
    """
    计算 Prototype 质量指标 | Compute Prototype Quality Metrics.

    :return: {
        "same_class_cosine": {cls_id: float},
        "cross_class_cosine": {cls_id: float},
        "discrimination_ratio": {cls_id: float},
        "intra_class_variance": {cls_id: float},
        "inter_class_distance": float,
        "silhouette_score": float,
        "prototype_norm": {cls_id: float},
    }
    """
    support = proto_data["support"]
    queries = proto_data["queries"]
    all_protos = proto_data["all_protos"]  # [M, 1280]
    all_labels = proto_data["all_labels"]

    # ── 1. Support-Query Cosine Similarity | 同类 vs 异类 ──
    same_class_cos = {}
    cross_class_cos = {}

    for cls_id in DEFECT_CLASSES:
        sp = support[cls_id]  # [1280]
        qp_same = queries[cls_id]  # [N, 1280]

        # 同类: support vs same-class queries
        same_sim = np.dot(qp_same, sp) / (
            np.linalg.norm(qp_same, axis=1) * np.linalg.norm(sp) + 1e-8
        )
        same_class_cos[cls_id] = float(np.mean(same_sim))

        # 异类: support vs other-class queries
        cross_sims = []
        for other_cls in DEFECT_CLASSES:
            if other_cls == cls_id:
                continue
            qp_other = queries[other_cls]  # [N, 1280]
            cross_sim = np.dot(qp_other, sp) / (
                np.linalg.norm(qp_other, axis=1) * np.linalg.norm(sp) + 1e-8
            )
            cross_sims.extend(cross_sim.tolist())
        cross_class_cos[cls_id] = float(np.mean(cross_sims))

    # ── 2. 区分度比率 | Discrimination Ratio ──
    disc_ratio = {}
    for cls_id in DEFECT_CLASSES:
        same = same_class_cos[cls_id]
        cross = cross_class_cos[cls_id]
        disc_ratio[cls_id] = float(same / (cross + 1e-8))

    # ── 3. Intra-class Variance (仅 query prototypes) ──
    intra_var = {}
    for cls_id in DEFECT_CLASSES:
        qp = queries[cls_id]  # [N, 1280]
        mean = qp.mean(axis=0)  # [1280]
        var = np.mean(np.sum((qp - mean) ** 2, axis=1))
        intra_var[cls_id] = float(var)

    # ── 4. Inter-class Distance (support prototypes 之间的平均欧氏距离) ──
    support_protos = np.stack([support[c] for c in DEFECT_CLASSES], axis=0)  # [4, 1280]
    inter_dists = []
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.linalg.norm(support_protos[i] - support_protos[j])
            inter_dists.append(float(d))
    inter_class_dist = float(np.mean(inter_dists))

    # ── 5. Silhouette Score (所有 prototype, 按 class label) ──
    # 构建 labels: support 用 cls_id, query 用 cls_id
    proto_labels = []
    for cls_id in DEFECT_CLASSES:
        proto_labels.append(cls_id)  # Support
    for cls_id in DEFECT_CLASSES:
        proto_labels.extend([cls_id] * len(queries[cls_id]))  # Queries
    proto_labels = np.array(proto_labels)

    try:
        sil = float(silhouette_score(all_protos, proto_labels, metric="cosine"))
    except (ValueError, Exception):
        sil = float("nan")

    # ── 6. Prototype L2 Norm (support) ──
    proto_norms = {}
    for cls_id in DEFECT_CLASSES:
        proto_norms[cls_id] = float(np.linalg.norm(support[cls_id]))

    return {
        "same_class_cosine": same_class_cos,
        "cross_class_cosine": cross_class_cos,
        "discrimination_ratio": disc_ratio,
        "intra_class_variance": intra_var,
        "inter_class_distance": inter_class_dist,
        "silhouette_score": sil,
        "prototype_norm": proto_norms,
    }


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_tsne(
    proto_data_a: dict,
    proto_data_b: dict,
    label_a: str,
    label_b: str,
    output_path: Path,
):
    """t-SNE 可视化: 左右对比 Frozen vs LoRA | t-SNE side-by-side comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, proto_data, title in [
        (axes[0], proto_data_a, label_a),
        (axes[1], proto_data_b, label_b),
    ]:
        if proto_data is None:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=14)
            continue

        all_protos = proto_data["all_protos"]
        all_labels = proto_data["all_labels"]

        # t-SNE 降维 | t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_protos) - 1))
        proto_2d = tsne.fit_transform(all_protos)

        # 按类别绘制 | Plot by class
        for cls_id in DEFECT_CLASSES:
            # Support
            mask_s = [f"S_{CLASS_NAMES[cls_id]}" == lb for lb in all_labels]
            if any(mask_s):
                pts_s = proto_2d[mask_s]
                ax.scatter(pts_s[:, 0], pts_s[:, 1], c=CLASS_COLORS[cls_id],
                          marker="*", s=250, edgecolors="black", linewidths=1.5,
                          label=f"{CLASS_NAMES[cls_id]} (Support)",
                          zorder=5)

            # Queries
            mask_q = [f"Q_{CLASS_NAMES[cls_id]}" == lb for lb in all_labels]
            if any(mask_q):
                pts_q = proto_2d[mask_q]
                ax.scatter(pts_q[:, 0], pts_q[:, 1], c=CLASS_COLORS[cls_id],
                          marker="o", s=30, alpha=0.6,
                          label=f"{CLASS_NAMES[cls_id]} (Query)",
                          zorder=3)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.legend(loc="lower right", fontsize=7, markerscale=0.8)

    fig.suptitle("Prototype t-SNE: Frozen vs LoRA", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ t-SNE plot saved: {output_path}")


def plot_cosine_heatmap(
    metrics_a: dict,
    metrics_b: dict,
    label_a: str,
    label_b: str,
    output_path: Path,
):
    """余弦相似度对比热力图 | Cosine Similarity comparison heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metrics, title in [
        (axes[0], metrics_a, label_a),
        (axes[1], metrics_b, label_b),
    ]:
        if metrics is None:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)
            ax.set_title(title, fontsize=14)
            continue

        # 构建 4×4 余弦相似度矩阵 (support class i vs query class j)
        cos_matrix = np.zeros((4, 4))
        for i, ci in enumerate(DEFECT_CLASSES):
            for j, cj in enumerate(DEFECT_CLASSES):
                if ci == cj:
                    cos_matrix[i, j] = metrics["same_class_cosine"].get(ci, 0)
                else:
                    # 取每对异类的原始值 (已按 "support ci vs queries cj" 计算)
                    cos_matrix[i, j] = 0  # Placeholder, filled below

        # 更精确的计算: 重构 4×4 矩阵
        # 行=Support class, 列=Query class
        # 我们需要重新计算，但 metrics 中没有存储完整的 4×4
        # 简化: 对角线=same_class_cos, 非对角线=所有异类的平均值

        for i, ci in enumerate(DEFECT_CLASSES):
            for j, cj in enumerate(DEFECT_CLASSES):
                if ci == cj:
                    cos_matrix[i, j] = metrics["same_class_cosine"].get(ci, 0)
                else:
                    cos_matrix[i, j] = metrics["cross_class_cosine"].get(ci, 0)

        im = ax.imshow(cos_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels([CLASS_NAMES[c] for c in DEFECT_CLASSES])
        ax.set_yticks(range(4))
        ax.set_yticklabels([CLASS_NAMES[c] for c in DEFECT_CLASSES])
        ax.set_xlabel("Query Class")
        ax.set_ylabel("Support Class")
        ax.set_title(title, fontsize=14, fontweight="bold")

        # 添加数值标注 | Add value annotations
        for i in range(4):
            for j in range(4):
                text_color = "white" if cos_matrix[i, j] < 0.5 else "black"
                ax.text(j, i, f"{cos_matrix[i, j]:.3f}", ha="center",
                       va="center", color=text_color, fontsize=10)

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Support-Query Cosine Similarity Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Cosine heatmap saved: {output_path}")


def plot_metrics_bar(
    metrics_a: dict,
    metrics_b: dict,
    label_a: str,
    label_b: str,
    output_path: Path,
):
    """指标条形图对比 | Metrics bar chart comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cls_labels = [CLASS_NAMES[c] for c in DEFECT_CLASSES]
    colors = [CLASS_COLORS[c] for c in DEFECT_CLASSES]

    # ── (a) Same-Class Cosine Similarity ──
    ax = axes[0, 0]
    x = np.arange(4)
    w = 0.35
    vals_a = [metrics_a["same_class_cosine"].get(c, 0) for c in DEFECT_CLASSES] if metrics_a else []
    vals_b = [metrics_b["same_class_cosine"].get(c, 0) for c in DEFECT_CLASSES] if metrics_b else []
    if vals_a:
        ax.bar(x - w/2, vals_a, w, label=label_a, color="#E74C3C", alpha=0.8)
    if vals_b:
        ax.bar(x + w/2, vals_b, w, label=label_b, color="#3498DB", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_labels)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Same-Class Support-Query Cosine ↑")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # ── (b) Discrimination Ratio ──
    ax = axes[0, 1]
    vals_a = [metrics_a["discrimination_ratio"].get(c, 0) for c in DEFECT_CLASSES] if metrics_a else []
    vals_b = [metrics_b["discrimination_ratio"].get(c, 0) for c in DEFECT_CLASSES] if metrics_b else []
    if vals_a:
        ax.bar(x - w/2, vals_a, w, label=label_a, color="#E74C3C", alpha=0.8)
    if vals_b:
        ax.bar(x + w/2, vals_b, w, label=label_b, color="#3498DB", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_labels)
    ax.set_ylabel("Same / Cross Ratio")
    ax.set_title("Discrimination Ratio (Same/Cross) ↑")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── (c) Intra-class Variance ──
    ax = axes[1, 0]
    vals_a = [metrics_a["intra_class_variance"].get(c, 0) for c in DEFECT_CLASSES] if metrics_a else []
    vals_b = [metrics_b["intra_class_variance"].get(c, 0) for c in DEFECT_CLASSES] if metrics_b else []
    if vals_a:
        ax.bar(x - w/2, vals_a, w, label=label_a, color="#E74C3C", alpha=0.8)
    if vals_b:
        ax.bar(x + w/2, vals_b, w, label=label_b, color="#3498DB", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cls_labels)
    ax.set_ylabel("Variance")
    ax.set_title("Intra-class Variance ↓")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # ── (d) Summary: Silhouette + Inter-class ──
    ax = axes[1, 1]
    summary_metrics = ["Silhouette Score ↑", "Inter-class Distance ↑"]
    vals_a_sum = [
        metrics_a.get("silhouette_score", 0) if metrics_a else 0,
        metrics_a.get("inter_class_distance", 0) if metrics_a else 0,
    ]
    vals_b_sum = [
        metrics_b.get("silhouette_score", 0) if metrics_b else 0,
        metrics_b.get("inter_class_distance", 0) if metrics_b else 0,
    ]
    x2 = np.arange(2)
    if metrics_a:
        ax.bar(x2 - w/2, vals_a_sum, w, label=label_a, color="#E74C3C", alpha=0.8)
    if metrics_b:
        ax.bar(x2 + w/2, vals_b_sum, w, label=label_b, color="#3498DB", alpha=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels(summary_metrics, fontsize=9)
    ax.set_title("Global Separation Metrics")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Prototype Quality: Frozen vs LoRA", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Metrics bar chart saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# 报告生成 | Report Generation
# ═══════════════════════════════════════════════════════════════════

def print_comparison_table(metrics_a, metrics_b, label_a, label_b):
    """打印对比表格 | Print comparison table."""
    print(f"\n{'═'*80}")
    print(f"  Prototype Quality Comparison: {label_a} vs {label_b}")
    print(f"{'═'*80}")

    hdr = f"{'Metric':<35} {'Class':<10} {label_a:>12} {label_b:>12} {'Δ':>10} {'Better':>8}"
    print(hdr)
    print("-" * len(hdr))

    def print_row(metric, cls_name, va, vb):
        if va is None or vb is None:
            return
        delta = vb - va
        better = "← LoRA" if delta > 0 else "← Frozen" if delta < 0 else "="
        print(f"{metric:<35} {cls_name:<10} {va:>12.4f} {vb:>12.4f} {delta:>+10.4f} {better:>8}")

    # Same-class cosine
    for cls_id in DEFECT_CLASSES:
        cn = CLASS_NAMES[cls_id]
        print_row("Same-Class Cosine Sim ↑", cn,
                  metrics_a["same_class_cosine"].get(cls_id) if metrics_a else None,
                  metrics_b["same_class_cosine"].get(cls_id) if metrics_b else None)

    print("-" * len(hdr))

    # Cross-class cosine
    for cls_id in DEFECT_CLASSES:
        cn = CLASS_NAMES[cls_id]
        print_row("Cross-Class Cosine Sim ↓", cn,
                  metrics_a["cross_class_cosine"].get(cls_id) if metrics_a else None,
                  metrics_b["cross_class_cosine"].get(cls_id) if metrics_b else None)

    print("-" * len(hdr))

    # Discrimination ratio
    for cls_id in DEFECT_CLASSES:
        cn = CLASS_NAMES[cls_id]
        print_row("Discrimination Ratio ↑", cn,
                  metrics_a["discrimination_ratio"].get(cls_id) if metrics_a else None,
                  metrics_b["discrimination_ratio"].get(cls_id) if metrics_b else None)

    print("-" * len(hdr))

    # Intra-class variance
    for cls_id in DEFECT_CLASSES:
        cn = CLASS_NAMES[cls_id]
        print_row("Intra-class Variance ↓", cn,
                  metrics_a["intra_class_variance"].get(cls_id) if metrics_a else None,
                  metrics_b["intra_class_variance"].get(cls_id) if metrics_b else None)

    print("-" * len(hdr))

    # Global
    for gname, gkey, gdir in [
        ("Inter-class Distance", "inter_class_distance", "↑"),
        ("Silhouette Score", "silhouette_score", "↑"),
    ]:
        va = metrics_a.get(gkey) if metrics_a else None
        vb = metrics_b.get(gkey) if metrics_b else None
        if va is not None and vb is not None:
            delta = vb - va
            better = "← LoRA" if delta > 0 else "← Frozen" if delta < 0 else "="
            print(f"{gname+' '+gdir:<35} {'(global)':<10} {va:>12.4f} {vb:>12.4f} {delta:>+10.4f} {better:>8}")

    print(f"{'─'*80}")


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Prototype Quality Diagnosis — Severstal"
    )
    p.add_argument("--data-root", type=str, required=True,
                   help="Severstal 数据集路径 | Severstal dataset path")
    p.add_argument("--checkpoint-a", type=str, default=None,
                   help="对照 checkpoint (如 Frozen/NoLoRA)")
    p.add_argument("--checkpoint-b", type=str, default=None,
                   help="实验 checkpoint (如 LoRA_r4)")
    p.add_argument("--label-a", type=str, default="Frozen",
                   help="对照标签 | Control label")
    p.add_argument("--label-b", type=str, default="LoRA",
                   help="实验标签 | Experiment label")
    p.add_argument("--k-shot", type=int, default=1,
                   help="Support 图像数 (default: 1)")
    p.add_argument("--num-queries", type=int, default=30,
                   help="每类 Query 图像数 (default: 30)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 输出目录 | Output Dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"diag_output/severstal_proto_quality_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] {msg}")

    log(f"Output: {out_dir}")
    log(f"Data: {args.data_root}")
    log(f"K-shot: {args.k_shot}, Queries/class: {args.num_queries}")

    # ── 数据集 | Dataset (仅用 train split 做诊断) ──
    train_ds = SeverstalDataset(root=args.data_root, split="train", binary=False, seed=args.seed)
    log(f"Dataset: {len(train_ds)} training images")
    for cls_id in DEFECT_CLASSES:
        log(f"  Class {cls_id}: {len(train_ds.class_to_images(cls_id))} images")

    # ── 采集 Prototypes | Collect Prototypes ──
    proto_data_a, metrics_a = None, None
    proto_data_b, metrics_b = None, None

    if args.checkpoint_a:
        log(f"\n{'─'*50}")
        log(f"Loading checkpoint A: {args.label_a}")
        log(f"  Path: {args.checkpoint_a}")
        backbone_a = load_backbone(args.checkpoint_a, device, log)
        log(f"  Collecting prototypes...")
        proto_data_a = collect_prototypes(
            backbone_a, train_ds, device,
            k_shot=args.k_shot, num_queries_per_class=args.num_queries,
            seed=args.seed,
        )
        metrics_a = compute_metrics(proto_data_a)
        log(f"  ✓ {args.label_a}: Silhouette={metrics_a.get('silhouette_score', float('nan')):.4f}, "
            f"InterDist={metrics_a.get('inter_class_distance', 0):.4f}")
        del backbone_a
        torch.cuda.empty_cache()

    if args.checkpoint_b:
        log(f"\n{'─'*50}")
        log(f"Loading checkpoint B: {args.label_b}")
        log(f"  Path: {args.checkpoint_b}")
        backbone_b = load_backbone(args.checkpoint_b, device, log)
        log(f"  Collecting prototypes...")
        proto_data_b = collect_prototypes(
            backbone_b, train_ds, device,
            k_shot=args.k_shot, num_queries_per_class=args.num_queries,
            seed=args.seed,
        )
        metrics_b = compute_metrics(proto_data_b)
        log(f"  ✓ {args.label_b}: Silhouette={metrics_b.get('silhouette_score', float('nan')):.4f}, "
            f"InterDist={metrics_b.get('inter_class_distance', 0):.4f}")
        del backbone_b
        torch.cuda.empty_cache()

    # ── 打印对比表 | Print Comparison ──
    if metrics_a and metrics_b:
        print_comparison_table(metrics_a, metrics_b, args.label_a, args.label_b)
    elif metrics_b:
        log(f"\n{args.label_b} metrics:")
        for k, v in metrics_b.items():
            log(f"  {k}: {v}")

    # ── 可视化 | Visualizations ──
    log(f"\n{'─'*50}")
    log("Generating visualizations...")

    if proto_data_a is not None or proto_data_b is not None:
        plot_tsne(
            proto_data_a, proto_data_b,
            args.label_a, args.label_b,
            out_dir / "tsne_prototypes.png",
        )

    if metrics_a is not None or metrics_b is not None:
        plot_cosine_heatmap(
            metrics_a, metrics_b,
            args.label_a, args.label_b,
            out_dir / "cosine_similarity_heatmap.png",
        )

        plot_metrics_bar(
            metrics_a, metrics_b,
            args.label_a, args.label_b,
            out_dir / "proto_metrics_bar.png",
        )

    # ── 保存 JSON | Save JSON ──
    result = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "data_root": args.data_root,
            "k_shot": args.k_shot,
            "num_queries_per_class": args.num_queries,
            "seed": args.seed,
        },
        "checkpoint_a": {"label": args.label_a, "path": args.checkpoint_a,
                         "metrics": metrics_a} if metrics_a else None,
        "checkpoint_b": {"label": args.label_b, "path": args.checkpoint_b,
                         "metrics": metrics_b} if metrics_b else None,
    }

    # 保存 prototype 数据 (可选, 较大)
    if proto_data_a:
        result["proto_data_a"] = {
            "support": {str(k): v.tolist() for k, v in proto_data_a["support"].items()},
        }
    if proto_data_b:
        result["proto_data_b"] = {
            "support": {str(k): v.tolist() for k, v in proto_data_b["support"].items()},
        }

    json_path = out_dir / "proto_quality.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    log(f"\n✓ Results saved: {json_path}")

    # ── 最终判定 | Final Verdict ──
    if metrics_a and metrics_b:
        sil_a = metrics_a.get("silhouette_score", float("nan"))
        sil_b = metrics_b.get("silhouette_score", float("nan"))
        inter_a = metrics_a.get("inter_class_distance", 0)
        inter_b = metrics_b.get("inter_class_distance", 0)

        log(f"\n{'═'*60}")
        log("  VERDICT: Does LoRA Improve Prototype Quality?")
        log(f"{'═'*60}")
        log(f"  Silhouette:  {args.label_a}={sil_a:.4f} → {args.label_b}={sil_b:.4f} "
            f"(Δ={sil_b-sil_a:+.4f})")
        log(f"  Inter-Dist:  {args.label_a}={inter_a:.4f} → {args.label_b}={inter_b:.4f} "
            f"(Δ={inter_b-inter_a:+.4f})")

        if sil_b > sil_a + 0.01 and inter_b > inter_a * 1.05:
            log("  ✅ LoRA IMPROVES prototype quality (higher Silhouette + larger inter-dist)")
        elif sil_b < sil_a - 0.01:
            log("  ⚠️ LoRA DEGRADES prototype separation — overfitting?")
        elif abs(sil_b - sil_a) <= 0.01:
            log("  ➡️ LoRA has NEGLIGIBLE effect on prototype quality")
            log("     → Performance gain may come from Query feature adaptation, not Prototype")
        log(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
