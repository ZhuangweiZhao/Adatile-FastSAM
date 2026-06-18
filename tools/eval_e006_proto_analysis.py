#!/usr/bin/env python3
"""
E006: Proto Analysis — P4 特征空间是否存在建筑 Prototype？
=============================================================

诊断实验 | Diagnostic experiment (不追求提分 | NOT about improving Dice).

核心问题 | Core question:
    FastSAM P4 特征空间里，建筑和背景是否天然形成可分离的聚类？
    Does the P4 feature space naturally contain separable building/background prototypes?

分析流水线 | Analysis pipeline:
    1. 冻结 Backbone → 提取 P4 特征 | Extract P4 features
    2. 按 GT mask 分离建筑/背景像素特征 | Separate building/background pixel features
    3. K-Means 聚类 → 找原型中心 | K-Means → find prototype centers
    4. PCA + t-SNE 降维 → 可视化 | PCA + t-SNE → visualization
    5. Cosine similarity → Proto vs GT 相似度分布 | Proto vs GT similarity distribution

如果建筑/背景在 P4 空间中明显可分 → SPM 和 AdaTile 有坚实的理论基础。
If building/background are clearly separable → SPM and AdaTile have solid theoretical foundation.

用法 | Usage:
    python tools/eval_e006_proto_analysis.py
    python tools/eval_e006_proto_analysis.py --max-samples 20 --n-clusters 4
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.backbone import FastSAMBackbone


def parse_args():
    p = argparse.ArgumentParser(description="E006: Proto Analysis — P4 feature space diagnosis")
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--max-samples", type=int, default=10,
                   help="Max images to analyze (P4 features are large, limit memory)")
    p.add_argument("--max-pixels-per-image", type=int, default=5000,
                   help="Max pixels to sample per image per class (avoid memory explosion)")
    p.add_argument("--n-clusters", type=int, default=3,
                   help="Number of K-Means clusters per class (default 3 → sub-prototypes)")
    p.add_argument("--tsne-samples", type=int, default=3000,
                   help="Max samples for t-SNE (slow for >5000)")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def extract_p4_features(
    backbone: FastSAMBackbone,
    dataset: MassachusettsBuildingsDataset,
    max_samples: int,
    max_pixels_per_class: int,
    device: str,
):
    """
    从多张图像提取 P4 特征，按 GT mask 分离建筑/背景。
    Extract P4 features from multiple images, separate by GT mask.

    Returns:
        features_building: [N_b, 1280]  建筑像素特征 | Building pixel features
        features_background: [N_bg, 1280] 背景像素特征 | Background pixel features
    """
    all_building = []
    all_background = []

    n_images = min(len(dataset), max_samples)
    indices = np.linspace(0, len(dataset) - 1, n_images, dtype=int)

    print(f"  Extracting P4 features from {n_images} images...")
    for idx in tqdm(indices, desc="  Extracting"):
        sample = dataset[int(idx)]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)  # [H, W], values 0 or 1

        # 提取 P4 特征 | Extract P4 features
        with torch.no_grad():
            features = backbone(image)
            p4 = features["p4"]  # [1, 1280, H/16, W/16]

        # 将 GT mask 降采样到 P4 分辨率 | Downsample GT to P4 resolution
        gt_p4 = F.interpolate(
            gt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=p4.shape[2:],
            mode="nearest",
        ).squeeze(0).squeeze(0)  # [H/16, W/16]

        # 展平为像素特征 | Flatten to pixel features
        p4_flat = p4.squeeze(0).reshape(1280, -1).permute(1, 0)  # [N_total, 1280]
        gt_flat = gt_p4.reshape(-1).long()  # [N_total]

        # 建筑像素索引 | Building pixel indices
        building_idx = (gt_flat == 1).nonzero(as_tuple=True)[0]
        background_idx = (gt_flat == 0).nonzero(as_tuple=True)[0]

        # 随机采样以控制数量 | Random sample to control count
        if len(building_idx) > max_pixels_per_class:
            building_idx = building_idx[torch.randperm(len(building_idx))[:max_pixels_per_class]]
        if len(background_idx) > max_pixels_per_class:
            background_idx = background_idx[torch.randperm(len(background_idx))[:max_pixels_per_class]]

        if len(building_idx) > 0:
            all_building.append(p4_flat[building_idx].cpu().numpy())
        if len(background_idx) > 0:
            all_background.append(p4_flat[background_idx].cpu().numpy())

    features_building = np.concatenate(all_building, axis=0) if all_building else np.array([])
    features_background = np.concatenate(all_background, axis=0) if all_background else np.array([])

    print(f"  Building pixels: {features_building.shape[0]:,}")
    print(f"  Background pixels: {features_background.shape[0]:,}")
    return features_building, features_background


def run_clustering(features: np.ndarray, n_clusters: int, label: str):
    """
    K-Means 聚类 → 找原型中心 + 轮廓系数。
    K-Means clustering → find prototype centers + silhouette score.

    Returns:
        centers: [n_clusters, 1280]  聚类中心（原型）| Cluster centers (prototypes)
        labels:  [N,]                 每个像素的聚类标签 | Cluster label per pixel
        sil_score: float              轮廓系数 | Silhouette score
    """
    print(f"\n  [{label}] K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_

    # 轮廓系数 | Silhouette score
    # 采样以加速 | Subsample for speed
    n_eval = min(10000, len(features))
    idx = np.random.choice(len(features), n_eval, replace=False)
    sil = silhouette_score(features[idx], cluster_labels[idx])

    for i in range(n_clusters):
        count = (cluster_labels == i).sum()
        print(f"    Cluster {i}: {count:,} pixels ({100*count/len(features):.1f}%)")
    print(f"    Silhouette score: {sil:.4f}")

    return centers, cluster_labels, sil


def compute_proto_similarity(
    proto_centers: np.ndarray,
    features: np.ndarray,
    gt_labels: np.ndarray,
    label: str,
):
    """
    计算每个 Prototype 与 GT 区域的 Cosine Similarity 分布。
    Compute Cosine Similarity distribution between prototypes and GT regions.

    Args:
        proto_centers: [K, 1280]  原型中心 | Prototype centers
        features:      [N, 1280]  像素特征 | Pixel features
        gt_labels:     [N,]       GT 标签 (0=bg, 1=building) | GT labels
        label:         str         标签 | Label
    """
    # 归一化以计算 cosine similarity | Normalize for cosine similarity
    proto_norm = proto_centers / (np.linalg.norm(proto_centers, axis=1, keepdims=True) + 1e-8)
    feat_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

    # 所有像素 vs 所有原型的 cosine similarity | All pixels vs all prototypes
    sim_matrix = np.dot(feat_norm, proto_norm.T)  # [N, K]

    print(f"\n  [{label}] Cosine Similarity: Pixels vs Prototypes")
    print(f"    {'Proto':<8} {'BG mean':>10} {'BG std':>10} {'Build mean':>10} {'Build std':>10} {'Δ(BG-Build)':>14}")
    print(f"    {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*14}")

    proto_stats = []
    for i in range(proto_centers.shape[0]):
        sim_to_bg = sim_matrix[gt_labels == 0, i] if (gt_labels == 0).sum() > 0 else np.array([0])
        sim_to_build = sim_matrix[gt_labels == 1, i] if (gt_labels == 1).sum() > 0 else np.array([0])

        delta = sim_to_build.mean() - sim_to_bg.mean()
        print(f"    P{i:<7} {sim_to_bg.mean():>10.4f} {sim_to_bg.std():>10.4f} "
              f"{sim_to_build.mean():>10.4f} {sim_to_build.std():>10.4f} {delta:>+14.4f}")

        proto_stats.append({
            "proto_id": i,
            "sim_to_bg_mean": float(sim_to_bg.mean()),
            "sim_to_build_mean": float(sim_to_build.mean()),
            "delta": float(delta),
        })

    return proto_stats


def visualize(
    features_building: np.ndarray,
    features_background: np.ndarray,
    proto_building: np.ndarray,
    proto_background: np.ndarray,
    tsne_samples: int,
    output_path: str,
):
    """
    PCA + t-SNE 降维可视化 | PCA + t-SNE dimensionality reduction visualization.

    生成两张图 | Generate two figures:
        1. PCA: Building vs Background 分布
        2. t-SNE: Building vs Background + Prototype centers
    """
    print(f"\n  Running PCA + t-SNE visualization...")

    # 合并特征，控制采样量 | Combine features, control sample count
    n_b = min(tsne_samples // 2, len(features_building))
    n_bg = min(tsne_samples // 2, len(features_background))

    idx_b = np.random.choice(len(features_building), n_b, replace=False)
    idx_bg = np.random.choice(len(features_background), n_bg, replace=False)

    X = np.concatenate([features_building[idx_b], features_background[idx_bg]], axis=0)
    labels = np.concatenate([
        np.ones(n_b, dtype=int),       # 1 = building
        np.zeros(n_bg, dtype=int),     # 0 = background
    ])

    # Combine prototypes | Combine prototype centers
    proto_all = np.concatenate([proto_building, proto_background], axis=0) if \
        len(proto_building) > 0 and len(proto_background) > 0 else \
        (proto_building if len(proto_building) > 0 else proto_background)
    proto_labels = np.concatenate([
        np.ones(len(proto_building), dtype=int) * 2,       # 2 = building proto
        np.ones(len(proto_background), dtype=int) * 3,     # 3 = bg proto
    ]) if len(proto_building) > 0 and len(proto_background) > 0 else np.array([])

    # ── PCA ──
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    proto_pca = pca.transform(proto_all) if len(proto_all) > 0 else None

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # PCA plot
    ax = axes[0]
    for cls, color, name in [(0, "tab:blue", "Background"), (1, "tab:red", "Building")]:
        mask = labels == cls
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name,
                   alpha=0.3, s=3, rasterized=True)
    if proto_pca is not None:
        ax.scatter(proto_pca[:, 0], proto_pca[:, 1], c="black", marker="X",
                   s=200, edgecolors="white", linewidth=2, label="Prototypes", zorder=10)
    ax.set_title(f"PCA: Building vs Background Features\n"
                 f"(PC1={pca.explained_variance_ratio_[0]:.1%}, "
                 f"PC2={pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(loc="upper right")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # ── t-SNE ──
    # 进一步降采样以加速 t-SNE | Further downsample for t-SNE speed
    n_tsne = min(2000, len(X))
    idx_tsne = np.random.choice(len(X), n_tsne, replace=False)
    X_tsne_sub = X[idx_tsne]
    labels_tsne_sub = labels[idx_tsne]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_tsne_sub)

    # t-SNE on the same subset for prototypes
    proto_tsne = None
    if len(proto_all) > 0:
        # t-SNE doesn't have transform(), we fit on combined data
        X_with_proto = np.concatenate([X_tsne_sub, proto_all], axis=0)
        tsne_all = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_with_proto_tsne = tsne_all.fit_transform(X_with_proto)
        X_tsne = X_with_proto_tsne[:n_tsne]
        proto_tsne = X_with_proto_tsne[n_tsne:]

    ax = axes[1]
    for cls, color, name in [(0, "tab:blue", "Background"), (1, "tab:red", "Building")]:
        mask = labels_tsne_sub == cls
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name,
                   alpha=0.4, s=5, rasterized=True)
    if proto_tsne is not None:
        ax.scatter(proto_tsne[:, 0], proto_tsne[:, 1], c="black", marker="X",
                   s=200, edgecolors="white", linewidth=2, label="Prototypes", zorder=10)
    ax.set_title("t-SNE: Building vs Background Features + Prototypes")
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Visualization saved to: {output_path}")


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  E006: Proto Analysis — P4 Feature Space Diagnosis")
    print("  核心问题: P4 特征空间是否存在天然建筑 Prototype?")
    print("=" * 70)

    exp_id = generate_exp_id(name=args.name or "e006_proto")
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)

    # ── [1/5] Frozen Backbone ──
    print("\n[1/5] Frozen FastSAM Backbone")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()
    print(f"  Backbone: FastSAM-x (frozen)")

    # ── [2/5] Load Data ──
    print("\n[2/5] Load Data")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")
    print(f"  Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    # ── [3/5] Extract P4 Features ──
    print("\n[3/5] Extract P4 Features (Building vs Background)")
    features_b, features_bg = extract_p4_features(
        backbone, train_ds, args.max_samples, args.max_pixels_per_image, device
    )
    n_b = features_b.shape[0]
    n_bg = features_bg.shape[0]
    print(f"  Total: Building={n_b:,}, Background={n_bg:,} pixels")

    recorder.logger.log_info("e006/features",
        f"building_pixels={n_b}, background_pixels={n_bg}, "
        f"n_images={min(len(train_ds), args.max_samples)}")

    # ── [4/5] Clustering & Prototype Discovery ──
    print(f"\n[4/5] K-Means Clustering (k={args.n_clusters} per class)")
    print(f"  {'─' * 50}")

    # 建筑特征聚类 | Building feature clustering
    proto_b = np.array([])
    sil_b = 0.0
    if n_b > args.n_clusters:
        proto_b, labels_b, sil_b = run_clustering(features_b, args.n_clusters, "Building")
    else:
        print(f"  [Building] Insufficient samples ({n_b}) for k={args.n_clusters}")

    # 背景特征聚类 | Background feature clustering
    proto_bg = np.array([])
    sil_bg = 0.0
    if n_bg > args.n_clusters:
        proto_bg, labels_bg, sil_bg = run_clustering(features_bg, args.n_clusters, "Background")
    else:
        print(f"  [Background] Insufficient samples ({n_bg}) for k={args.n_clusters}")

    # 全局聚类(混合建筑+背景,看是否自然分离) | Global clustering (mixed, check natural separation)
    print(f"\n  [Global] K-Means on mixed Building+Background (k={args.n_clusters * 2})...")
    features_all = np.concatenate([features_b, features_bg], axis=0)
    gt_all = np.concatenate([np.ones(n_b), np.zeros(n_bg)])
    kmeans_global = KMeans(n_clusters=args.n_clusters * 2, random_state=42, n_init=10)
    labels_global = kmeans_global.fit_predict(features_all)
    proto_global = kmeans_global.cluster_centers_

    # 报告每个全局聚类中建筑/背景的混合比例 | Report building/bg ratio per cluster
    print(f"    {'Cluster':<8} {'Total':>8} {'Building':>10} {'BG':>10} {'Build%':>10} {'Dominant':>10}")
    print(f"    {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(args.n_clusters * 2):
        mask = labels_global == i
        n_total = mask.sum()
        n_build = (gt_all[mask] == 1).sum()
        n_back = n_total - n_build
        pct = 100 * n_build / n_total if n_total > 0 else 0
        dominant = "Building" if pct >= 50 else "Background"
        print(f"    C{i:<7} {n_total:>8} {n_build:>10} {n_back:>10} {pct:>9.1f}% {dominant:>10}")

    recorder.logger.log_info("e006/clustering",
        f"k={args.n_clusters}, sil_building={sil_b:.4f}, sil_background={sil_bg:.4f}")

    # ── [5/5] Proto-GT Similarity Analysis ──
    print(f"\n[5/5] Proto-GT Similarity Analysis")
    print(f"  {'─' * 50}")

    if len(proto_global) > 0:
        proto_stats = compute_proto_similarity(proto_global, features_all, gt_all, "Global")
        recorder.logger.log_info("e006/proto_similarity",
            f"n_protos={len(proto_global)}, "
            f"max_delta={max(s['delta'] for s in proto_stats):.4f}")
    else:
        proto_stats = []

    # ── Visualization ──
    print(f"\n{'─' * 50}")
    print("  Generating visualizations...")
    vis_path = str(output_path / "proto_analysis.png")
    visualize(features_b, features_bg, proto_b, proto_bg, args.tsne_samples, vis_path)

    # ── Summary ──
    # 判断 Proto 是否天然存在 | Determine if prototypes naturally exist
    # 条件1: 建筑特征聚类轮廓系数 > 0.3 | Condition 1: Building cluster sil score > 0.3
    # 条件2: 全局聚类中至少有一个 Building-dominant (>70%) 和一个 BG-dominant (<30%)
    # Condition 2: At least one Building-dominant (>70%) and one BG-dominant (<30%) global cluster

    build_dominant = sum(1 for i in range(args.n_clusters * 2)
                         if (gt_all[labels_global == i] == 1).sum() / max(
                             (labels_global == i).sum(), 1) > 0.7)
    bg_dominant = sum(1 for i in range(args.n_clusters * 2)
                      if (gt_all[labels_global == i] == 1).sum() / max(
                          (labels_global == i).sum(), 1) < 0.3)

    has_natural_clusters = sil_b > 0.3 and sil_bg > 0.3
    has_pure_clusters = build_dominant >= 1 and bg_dominant >= 1

    print(f"\n{'=' * 70}")
    print(f"  E006 结论 | Conclusions")
    print(f"  {'─' * 50}")
    print(f"  Building Silhouette:        {sil_b:.4f}  {'✓ >0.3' if sil_b > 0.3 else '✗ <0.3'}")
    print(f"  Background Silhouette:      {sil_bg:.4f}  {'✓ >0.3' if sil_bg > 0.3 else '✗ <0.3'}")
    print(f"  Building-dominant clusters: {build_dominant}/{args.n_clusters * 2}")
    print(f"  Background-dominant clusters: {bg_dominant}/{args.n_clusters * 2}")
    print(f"  {'─' * 50}")

    if has_natural_clusters and has_pure_clusters:
        print(f"  ✅ P4 特征空间存在天然建筑 Prototype！")
        print(f"     Building 和 Background 在 P4 空间中明显可分离。")
        print(f"     → SPM + AdaTile 有坚实的理论基础。")
        verdict = "strong_proto_exists"
    elif has_pure_clusters:
        print(f"  △ 全局聚类可分离但内部轮廓系数较弱。")
        print(f"     → Proto 存在但需要更好的特征提炼（SPM 的任务）。")
        verdict = "proto_exists_weak"
    elif sil_b > 0.2 or sil_bg > 0.2:
        print(f"  △ P4 特征有微弱聚类趋势，但不足以形成清晰 Prototype。")
        print(f"     → 需要更强的特征提炼或 Decoder 支持。")
        verdict = "proto_hint"
    else:
        print(f"  ✗ P4 特征空间中建筑/背景无明显聚类。")
        print(f"     → SPM/Proto 需要从零学习原型，而非从 P4 继承。")
        verdict = "no_proto"

    print(f"{'=' * 70}")

    # 记录 | Record
    recorder.record_metric("e006/silhouette_building", sil_b, phase="val", tags=["e006", "summary"])
    recorder.record_metric("e006/silhouette_background", sil_bg, phase="val", tags=["e006", "summary"])
    recorder.record_metric("e006/build_dominant_clusters", build_dominant,
                           phase="val", tags=["e006", "summary"])
    recorder.record_metric("e006/bg_dominant_clusters", bg_dominant,
                           phase="val", tags=["e006", "summary"])
    recorder.logger.log_info("e006/verdict", verdict, tags=["e006", "summary"])
    recorder.finalize()
    recorder.close()

    print(f"\n  Results saved to: {output_path}/")


if __name__ == "__main__":
    main()
