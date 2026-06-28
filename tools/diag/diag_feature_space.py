#!/usr/bin/env python3
"""
Feature Space Diagnosis — FastSAM P4 特征是否适合 Prototype-based FSS？
=========================================================================
Feature Space Diagnosis: Is FastSAM P4 suitable for Prototype-based FSS?

诊断维度 | Diagnosis Dimensions:
    1. t-SNE 可视化 — 各类前景像素的特征分布 (定性)
    2. 类内距离 vs 类间距离 — 同类聚集性、异类分离性 (定量)
    3. Silhouette Score — 聚类质量 (定量)
    4. 类均值 Cosine Similarity Matrix — 类间混淆度 (定量)
    5. Base→Novel 泛化差距 — 特征空间在 base/novel 间是否一致

核心问题 | Core Question:
    FastSAM P4 feature space 是否天然支持 "同类聚集、异类分离" 的 prototype 学习？
    如果不支持，需要 Feature Adaptation (Adapter/LoRA) 来重构特征空间。

用法 | Usage:
    python tools/diag/diag_feature_space.py \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --fold 0 --device cuda --max-samples-per-class 500

输出 | Output:
    - runs/diag_feature_space/{timestamp}/feature_diagnosis.json  — 定量指标
    - runs/diag_feature_space/{timestamp}/tsne_{base,novel,all}.png  — t-SNE 可视化
    - runs/diag_feature_space/{timestamp}/class_sim_matrix.png  — 类间相似度矩阵
"""

import sys, json, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.backbone.fastsam_backbone import FastSAMBackbone
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, get_isaid5i_novel_classes, get_isaid5i_base_classes
from adatile.utils.seed import set_seed

# ── 预计算: 不从 sklearn 导入大型库，用 numpy 实现核心算法
# Pre-compute: avoid importing heavy sklearn, implement core algorithms in numpy

def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix. [N, D] → [N, N]."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = vectors / norms
    return normalized @ normalized.T


def silhouette_score_custom(features: np.ndarray, labels: np.ndarray,
                             n_samples: int = 5000) -> dict:
    """
    Compute silhouette score from scratch.
    从零实现 Silhouette Score.

    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) = mean intra-cluster distance, b(i) = min mean inter-cluster distance

    :return: dict with per-class scores and overall mean
    """
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)
    if n_labels < 2:
        return {"overall": 0.0, "per_class": {}}

    # Subsample for speed
    n_total = len(features)
    if n_total > n_samples:
        indices = np.random.RandomState(42).choice(n_total, n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]

    N = len(features)
    # Precompute pairwise distances (chunked for memory)
    dists = np.zeros((N, N), dtype=np.float32)
    chunk_size = 500
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        for j in range(0, N, chunk_size):
            end_j = min(j + chunk_size, N)
            # Cosine distance: 1 - cosine_sim
            norms_i = np.linalg.norm(features[i:end_i], axis=1, keepdims=True)
            norms_j = np.linalg.norm(features[j:end_j], axis=1, keepdims=True)
            norms_i = np.maximum(norms_i, 1e-10)
            norms_j = np.maximum(norms_j, 1e-10)
            sim = (features[i:end_i] / norms_i) @ (features[j:end_j] / norms_j).T
            dists[i:end_i, j:end_j] = 1.0 - sim

    # Per-sample silhouette
    silhouettes = np.zeros(N)
    for i in range(N):
        label_i = labels[i]
        # Intra-cluster: mean distance to same-label points
        same_mask = labels == label_i
        same_mask[i] = False
        if same_mask.sum() == 0:
            a_i = 0.0
        else:
            a_i = dists[i, same_mask].mean()

        # Inter-cluster: min mean distance to other labels
        b_i = float("inf")
        for other_label in unique_labels:
            if other_label == label_i:
                continue
            other_mask = labels == other_label
            if other_mask.sum() == 0:
                continue
            mean_dist = dists[i, other_mask].mean()
            b_i = min(b_i, mean_dist)

        if b_i == float("inf"):
            silhouettes[i] = 0.0
        else:
            silhouettes[i] = (b_i - a_i) / max(a_i, b_i)

    overall = float(np.mean(silhouettes))
    per_class = {}
    for label in unique_labels:
        mask = labels == label
        if mask.sum() > 0:
            per_class[int(label)] = float(np.mean(silhouettes[mask]))

    return {"overall": round(overall, 4), "per_class": per_class}


def davies_bouldin_index(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Davies-Bouldin Index (lower = better separation).
    DB = (1/K) * sum_k max_{j≠k} ( (σ_k + σ_j) / d(μ_k, μ_j) )
    """
    unique_labels = sorted(set(labels))
    K = len(unique_labels)
    if K < 2:
        return 0.0

    centroids = []
    dispersions = []
    for label in unique_labels:
        mask = labels == label
        cluster_feats = features[mask]
        centroid = cluster_feats.mean(axis=0)
        # Dispersion: mean cosine distance to centroid
        centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-10)
        sims = cluster_feats @ centroid_norm
        dists = 1.0 - sims
        dispersions.append(float(np.mean(dists)))
        centroids.append(centroid)

    centroids = np.array(centroids)
    dispersions = np.array(dispersions)

    db_sum = 0.0
    for k in range(K):
        max_ratio = 0.0
        for j in range(K):
            if j == k:
                continue
            # Cosine distance between centroids
            c_k_norm = centroids[k] / max(np.linalg.norm(centroids[k]), 1e-10)
            c_j_norm = centroids[j] / max(np.linalg.norm(centroids[j]), 1e-10)
            d_centroids = 1.0 - float(np.dot(c_k_norm, c_j_norm))
            ratio = (dispersions[k] + dispersions[j]) / max(d_centroids, 1e-10)
            max_ratio = max(max_ratio, ratio)
        db_sum += max_ratio

    return round(float(db_sum / K), 4)


# ═══════════════════════════════════════════════════════════════════
# Feature Extraction | 特征提取
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_class_features(
    tile_root: str, class_ids: list[int], max_per_class: int,
    backbone, device: torch.device, split: str = "val",
) -> dict:
    """
    逐类收集前景像素的 P4 特征向量。
    Collect P4 feature vectors of foreground pixels, per class.

    对每个类的每个 tile:
    1. 加载图像 → backbone → P4 [C, H/16, W/16]
    2. 加载该类 mask → resize 到 P4 尺度
    3. 提取 mask 内的所有 feature vector → [N_fg, C]
    4. 每类最多收集 max_per_class 个向量

    :return: {class_id: {"features": [N, C] np.ndarray, "n_tiles": int, "n_vectors": int}}
    """
    from tools.train.train_fewshot import PreCutTileAdapter

    adapter = PreCutTileAdapter(tile_root, split)
    results = {}

    for cls_id in tqdm(class_ids, desc=f"  [{split}] Extracting features"):
        cls_name = ISAID5I_CATEGORIES.get(cls_id, f"class_{cls_id}")
        candidates = adapter.class_to_images(cls_id)
        if not candidates:
            print(f"    [{cls_name}] No tiles — skipping")
            continue

        all_feats = []
        n_tiles_used = 0

        # Shuffle to avoid bias
        rng = np.random.RandomState(42)
        tile_order = rng.permutation(candidates)

        for tile_idx in tile_order:
            if len(all_feats) >= max_per_class:
                break

            # Load image
            img_tensor = adapter.load_image(int(tile_idx)).unsqueeze(0).to(device)
            mask_tensor = adapter.render_class_mask(int(tile_idx), cls_id)

            if mask_tensor.sum() < 16:
                continue  # 前景太少，跳过

            # Backbone
            feats = backbone(img_tensor)
            p4 = feats["p4"]  # [1, C, H/16, W/16]

            # Resize mask to P4 spatial size
            mask_p4 = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=p4.shape[2:], mode="nearest"
            ).squeeze() > 0.5  # [H/16, W/16] bool

            if mask_p4.sum() < 4:
                continue

            # Extract FG features: [C, H, W] → [C, N_fg] → [N_fg, C]
            fg_vectors = p4[0, :, mask_p4].permute(1, 0).cpu().numpy()  # [N_fg, C]
            all_feats.append(fg_vectors)
            n_tiles_used += 1

        if not all_feats:
            print(f"    [{cls_name}] No valid FG vectors — skipping")
            continue

        combined = np.concatenate(all_feats, axis=0)
        # Subsample
        if len(combined) > max_per_class:
            indices = rng.choice(len(combined), max_per_class, replace=False)
            combined = combined[indices]

        results[cls_id] = {
            "features": combined,
            "n_tiles": n_tiles_used,
            "n_vectors": len(combined),
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# 诊断指标计算 | Diagnostic Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_class_statistics(class_features: dict) -> dict:
    """
    计算每个类的统计量：类内距离、类中心、分散度。
    Compute per-class statistics: intra-class distance, centroid, dispersion.
    """
    stats = {}
    for cls_id, data in class_features.items():
        feats = data["features"]  # [N, C]
        centroid = feats.mean(axis=0)
        centroid_norm = centroid / max(np.linalg.norm(centroid), 1e-10)

        # Intra-class distance (mean pairwise cosine distance)
        # 近似: mean distance to centroid
        sims = feats @ centroid_norm
        intra_dist = float(np.mean(1.0 - sims))

        stats[cls_id] = {
            "n_vectors": len(feats),
            "intra_dist": round(intra_dist, 4),
            "centroid": centroid_norm,  # numpy array, L2-normalized
        }

    return stats


def compute_pairwise_metrics(stats: dict, class_ids: list[int]) -> dict:
    """
    计算类间距离和相似度矩阵。
    Compute inter-class distances and similarity matrix.
    """
    n = len(class_ids)
    sim_matrix = np.zeros((n, n))
    inter_dists = {}

    for i, ci in enumerate(class_ids):
        if ci not in stats:
            continue
        for j, cj in enumerate(class_ids):
            if cj not in stats:
                continue
            sim = float(np.dot(stats[ci]["centroid"], stats[cj]["centroid"]))
            sim_matrix[i, j] = sim
            if i < j:
                inter_dists[(ci, cj)] = round(1.0 - sim, 4)

    return {
        "similarity_matrix": sim_matrix.round(4).tolist(),
        "inter_dists": {f"{k[0]}_{k[1]}": v for k, v in inter_dists.items()},
        "mean_inter_dist": round(np.mean(list(inter_dists.values())), 4) if inter_dists else 0.0,
    }


def compute_base_novel_gap(base_stats: dict, novel_stats: dict) -> dict:
    """
    Base vs Novel 特征空间差距。
    Compare feature space quality between Base and Novel.
    """
    base_intra = [s["intra_dist"] for s in base_stats.values()]
    novel_intra = [s["intra_dist"] for s in novel_stats.values()]

    return {
        "base_intra_dist_mean": round(np.mean(base_intra), 4) if base_intra else 0,
        "novel_intra_dist_mean": round(np.mean(novel_intra), 4) if novel_intra else 0,
        "base_novel_intra_ratio": round(np.mean(novel_intra) / max(np.mean(base_intra), 1e-10), 4),
        "note": "ratio>1 means Novel classes LESS compact than Base (harder for prototype)",
    }


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_tsne(features_dict: dict, class_names: dict, title: str,
              save_path: str, max_points: int = 3000,
              novel_ids: set = None):
    """
    t-SNE 可视化特征分布。
    t-SNE visualization of feature distribution.
    """
    # Collect data
    all_feats = []
    all_labels = []
    all_cls_ids = []

    for cls_id, data in features_dict.items():
        feats = data["features"]
        n_take = min(len(feats), max_points // len(features_dict))
        indices = np.random.RandomState(42).choice(len(feats), n_take, replace=False)
        all_feats.append(feats[indices])
        all_labels.extend([class_names.get(cls_id, f"c{cls_id}")] * n_take)
        all_cls_ids.extend([cls_id] * n_take)

    if not all_feats:
        print(f"  No data for t-SNE: {title}")
        return

    X = np.concatenate(all_feats, axis=0)
    labels = np.array(all_labels)

    # Use sklearn TSNE if available, else PCA fallback
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000, verbose=0)
        X_2d = tsne.fit_transform(X)
        method = "t-SNE"
    except ImportError:
        # PCA fallback
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_2d = (X_centered @ Vt[:2].T)
        # Scale
        X_2d = X_2d / np.std(X_2d, axis=0)
        method = "PCA (sklearn not available)"

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    unique_labels = sorted(set(labels))

    # Color by novel/base
    cmap = plt.cm.tab20
    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = cmap(i % 20)
        marker = 'o'
        alpha = 0.4
        # Highlight novel classes
        cls_id_for_label = [cid for cid, name in class_names.items() if name == label]
        if cls_id_for_label and novel_ids and cls_id_for_label[0] in novel_ids:
            marker = '^'
            alpha = 0.7
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=label,
                   marker=marker, alpha=alpha, s=5)

    ax.set_title(f"{title} ({method}, {len(X)} points)", fontsize=14, fontweight="bold")
    ax.legend(markerscale=3, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1),
              ncol=max(1, len(unique_labels) // 15))
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_similarity_matrix(sim_matrix: np.ndarray, class_names_list: list,
                           title: str, save_path: str):
    """绘制类间 cosine similarity 矩阵 | Plot inter-class cosine similarity matrix."""
    n = len(class_names_list)
    fig, ax = plt.subplots(1, 1, figsize=(max(10, n * 0.7), max(8, n * 0.6)))
    im = ax.imshow(sim_matrix, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = sim_matrix[i, j]
            color = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names_list, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names_list, fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════
# 主逻辑 | Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Feature Space Diagnosis")
    p.add_argument("--tile-root", type=str, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max-samples-per-class", type=int, default=500,
                   help="Max FG vectors per class (default 500)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="runs/diag_feature_space")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ts = datetime.now().strftime("%m%d_%H%M")
    out_dir = Path(args.output_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    base_classes = get_isaid5i_base_classes(args.fold)
    novel_classes = get_isaid5i_novel_classes(args.fold)
    all_classes = base_classes + novel_classes
    novel_set = set(novel_classes)

    print(f"\n{'='*70}")
    print(f"  Feature Space Diagnosis | 特征空间诊断")
    print(f"  {'─'*60}")
    print(f"  Tile root:          {args.tile_root}")
    print(f"  Fold:               {args.fold}")
    print(f"  Max samples/class:  {args.max_samples_per_class}")
    print(f"  Base classes:       {len(base_classes)}")
    print(f"  Novel classes:      {len(novel_classes)}")
    print(f"  Output:             {out_dir}")
    print(f"{'='*70}\n")

    # ── [1] 加载 Backbone ──
    print("[1] Loading FastSAM backbone...")
    backbone = FastSAMBackbone(freeze_backbone=True).to(device).eval()
    print(f"    Backbone: FastSAM-x (frozen) on {device}")

    # ── [2] 收集特征 ──
    print(f"\n[2] Collecting features ({len(all_classes)} classes)...")
    class_features = collect_class_features(
        args.tile_root, all_classes, args.max_samples_per_class,
        backbone, device, split="val",
    )

    valid_classes = sorted(class_features.keys())
    print(f"    Valid classes with features: {len(valid_classes)}/{len(all_classes)}")

    # ── [3] 计算统计量 ──
    print(f"\n[3] Computing statistics...")
    stats = compute_class_statistics(class_features)

    # Category names
    cat_names = ISAID5I_CATEGORIES

    # Per-class intra distance
    print(f"\n  Per-Class Intra-Cluster Distance (mean cosine dist to centroid):")
    print(f"  {'Class':<22s} {'Type':<6s} {'N_vectors':>10s} {'IntraDist':>10s} {'Quality'}")
    print(f"  {'─'*55}")
    for cls_id in valid_classes:
        s = stats[cls_id]
        ctype = "NOVEL" if cls_id in novel_set else "BASE"
        quality = "🟢" if s["intra_dist"] < 0.15 else ("🟡" if s["intra_dist"] < 0.25 else "🔴")
        print(f"  {cat_names.get(cls_id, f'c{cls_id}'):<22s} {ctype:<6s} "
              f"{s['n_vectors']:>10d} {s['intra_dist']:>10.4f}  {quality}")

    # ── [4] 计算类间指标 ──
    print(f"\n[4] Computing inter-class metrics...")
    pairwise = compute_pairwise_metrics(stats, valid_classes)

    base_stats = {k: v for k, v in stats.items() if k in base_classes}
    novel_stats = {k: v for k, v in stats.items() if k in novel_classes}
    gap = compute_base_novel_gap(base_stats, novel_stats)

    # Silhouette + DB
    print(f"\n  Computing Silhouette Score & Davies-Bouldin Index...")
    all_feats = []
    all_labels_list = []
    for cls_id in valid_classes:
        feats = class_features[cls_id]["features"]
        all_feats.append(feats)
        all_labels_list.extend([cls_id] * len(feats))
    X_all = np.concatenate(all_feats, axis=0)
    y_all = np.array(all_labels_list)

    silhouette = silhouette_score_custom(X_all, y_all, n_samples=5000)
    db_index = davies_bouldin_index(X_all, y_all)

    # ── [5] 输出诊断结果 ──
    print(f"\n{'='*70}")
    print(f"  FEATURE SPACE DIAGNOSIS RESULTS | 特征空间诊断结果")
    print(f"{'='*70}")

    print(f"\n  ── Global Metrics ──")
    print(f"  Classes analyzed:        {len(valid_classes)}")
    print(f"  Silhouette Score:        {silhouette['overall']:.4f}")
    print(f"    (>0.25=separable, >0.5=good, <0.1=random)")
    print(f"  Davies-Bouldin Index:    {db_index:.4f}")
    print(f"    (<1.0=good separation, >2.0=poor)")

    print(f"\n  ── Base vs Novel Gap ──")
    print(f"  Base intra-class dist:   {gap['base_intra_dist_mean']:.4f}")
    print(f"  Novel intra-class dist:  {gap['novel_intra_dist_mean']:.4f}")
    print(f"  Ratio (Novel/Base):      {gap['base_novel_intra_ratio']:.2f}")
    if gap["base_novel_intra_ratio"] > 1.2:
        print(f"  ⚠ Novel classes LESS compact — harder for prototype to capture")

    print(f"\n  ── Mean Inter-Class Cosine Distance ──")
    print(f"  Mean inter-class dist:   {pairwise['mean_inter_dist']:.4f}")
    print(f"    (>0.3=well separated, <0.15=high confusion)")

    print(f"\n  ── Per-Class Silhouette ──")
    for cls_id in valid_classes:
        s = silhouette["per_class"].get(cls_id, 0)
        ctype = "NOVEL" if cls_id in novel_set else "BASE"
        bar = "█" * int(max(0, s * 20))
        print(f"  {cat_names.get(cls_id, f'c{cls_id}'):<22s} {ctype:<6s} {s:+.4f} {bar}")

    # ── Verdict ──
    print(f"\n{'█'*70}")
    print(f"  VERDICT | 诊断结论")
    print(f"{'█'*70}")

    issues = []
    if silhouette["overall"] < 0.15:
        issues.append(f"🔴 Silhouette={silhouette['overall']:.3f} < 0.15: "
                      f"feature space has POOR class separation")
    elif silhouette["overall"] < 0.25:
        issues.append(f"🟡 Silhouette={silhouette['overall']:.3f} < 0.25: "
                      f"feature space is WEAKLY structured")
    else:
        issues.append(f"🟢 Silhouette={silhouette['overall']:.3f}: "
                      f"feature space has adequate structure")

    if db_index > 2.0:
        issues.append(f"🔴 Davies-Bouldin={db_index:.2f} > 2.0: clusters overlap heavily")
    elif db_index > 1.0:
        issues.append(f"🟡 Davies-Bouldin={db_index:.2f} > 1.0: moderate cluster overlap")

    if pairwise["mean_inter_dist"] < 0.15:
        issues.append(f"🔴 Mean inter-class cos-dist={pairwise['mean_inter_dist']:.4f} < 0.15: "
                      f"class prototypes are too similar")
    elif pairwise["mean_inter_dist"] < 0.25:
        issues.append(f"🟡 Mean inter-class cos-dist={pairwise['mean_inter_dist']:.4f}: "
                      f"moderate prototype confusion")

    if gap["base_novel_intra_ratio"] > 1.3:
        issues.append(f"🔴 Novel classes {gap['base_novel_intra_ratio']:.2f}x "
                      f"less compact than Base: prototype harder for novel")

    for issue in issues:
        print(f"  {issue}")

    if silhouette["overall"] < 0.15 and db_index > 2.0:
        print(f"\n  ★ VERDICT: FastSAM P4 feature space is NOT suitable for")
        print(f"    prototype-based few-shot learning WITHOUT adaptation.")
        print(f"    → Priority: Feature Adaptation (Adapter/LoRA) before Prototype improvement.")
    elif silhouette["overall"] < 0.25:
        print(f"\n  ★ VERDICT: Feature space is marginal. Adaptation recommended.")
    else:
        print(f"\n  ★ VERDICT: Feature space is acceptable. Focus on Prototype/Decoder.")

    print(f"{'█'*70}\n")

    # ── [6] 保存结果 ──
    diagnosis = {
        "config": {
            "tile_root": args.tile_root,
            "fold": args.fold,
            "max_samples_per_class": args.max_samples_per_class,
        },
        "global_metrics": {
            "silhouette": silhouette["overall"],
            "davies_bouldin": db_index,
            "mean_inter_class_distance": pairwise["mean_inter_dist"],
            "n_classes": len(valid_classes),
        },
        "base_novel_gap": gap,
        "per_class": {
            str(cls_id): {
                "name": cat_names.get(cls_id, f"class_{cls_id}"),
                "type": "NOVEL" if cls_id in novel_set else "BASE",
                "n_vectors": stats[cls_id]["n_vectors"],
                "intra_dist": stats[cls_id]["intra_dist"],
                "silhouette": silhouette["per_class"].get(cls_id, 0),
            }
            for cls_id in valid_classes
        },
        "similarity_matrix": pairwise["similarity_matrix"],
        "class_ids_ordered": valid_classes,
        "class_names_ordered": [cat_names.get(c, f"c{c}") for c in valid_classes],
        "verdict": {
            "silhouette_poor": silhouette["overall"] < 0.15,
            "db_poor": db_index > 2.0,
            "inter_class_poor": pairwise["mean_inter_dist"] < 0.15,
            "novel_less_compact": gap["base_novel_intra_ratio"] > 1.3,
            "needs_adaptation": silhouette["overall"] < 0.15 and db_index > 2.0,
        },
    }

    diag_path = out_dir / "feature_diagnosis.json"
    with open(diag_path, "w") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)

    # ── [7] 可视化 ──
    print(f"\n[7] Generating visualizations...")

    # t-SNE: All classes
    plot_tsne(class_features, cat_names, "FastSAM P4 Feature Space (All 15 Classes)",
              str(out_dir / "tsne_all.png"), max_points=2000, novel_ids=novel_set)

    # t-SNE: Base only
    base_features = {k: v for k, v in class_features.items() if k in base_classes}
    if base_features:
        plot_tsne(base_features, cat_names, "FastSAM P4 Feature Space (Base Classes Only)",
                  str(out_dir / "tsne_base.png"), max_points=2000, novel_ids=set())

    # t-SNE: Novel only
    novel_features = {k: v for k, v in class_features.items() if k in novel_classes}
    if novel_features:
        plot_tsne(novel_features, cat_names, "FastSAM P4 Feature Space (Novel Classes Only)",
                  str(out_dir / "tsne_novel.png"), max_points=2000, novel_ids=novel_set)

    # Similarity matrix
    plot_similarity_matrix(
        np.array(pairwise["similarity_matrix"]),
        [cat_names.get(c, f"c{c}") for c in valid_classes],
        "Class Prototype Cosine Similarity Matrix",
        str(out_dir / "class_sim_matrix.png"),
    )

    print(f"\n{'='*70}")
    print(f"  ✅ Feature space diagnosis saved → {diag_path}")
    print(f"  📊 Visualizations → {out_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
