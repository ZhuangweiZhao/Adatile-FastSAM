#!/usr/bin/env python3
"""
特征可视化诊断 | Feature Visualization Diagnosis.
===================================================

t-SNE/UMAP 可视化 P3/P4/P8 特征聚类 + Cosine Similarity + Silhouette Score。
t-SNE/UMAP visualization of P3/P4/P8 feature clustering + cosine similarity + silhouette.

核心问题 | Core Questions:
    Q1: 不同层的特征是否按类别聚类？→ 支撑"三层分工"设计
       Do features at different levels cluster by class? → justifies three-layer design
    Q2: P8 的语义聚类是否强于 P3？→ 证明 P8→Prototype 合理性
       Is P8 semantic clustering stronger than P3? → justifies P8→Prototype
    Q3: Roundabout/Bridge 等困难类的特征分布？→ 诊断 ZS>FT 的原因
       Feature distribution of hard classes (Roundabout, Bridge)? → diagnose ZS>FT

用法 | Usage::

    # 使用 frozen backbone (默认)
    python tools/diag/diag_feature_vis.py --device cuda

    # 使用 uf=8 checkpoint
    python tools/diag/diag_feature_vis.py \\
        --checkpoint runs/train_fewshot_allcls_K1_adaptive_uf8_*/best_model.pt \\
        --unfreeze-layers 8 --device cuda

    # 只分析特定类
    python tools/diag/diag_feature_vis.py --classes 13,15 --device cuda

输出 | Output:
    runs/diag/feature_vis/
    ├── tsne_p3.png / tsne_p4.png / tsne_p8.png    # t-SNE per level
    ├── tsne_combined.png                           # 三层并排
    ├── cosine_similarity.png                       # Cosine 相似度矩阵
    ├── silhouette.csv                              # Silhouette scores
    └── feature_stats.json                          # 统计摘要
"""

from __future__ import annotations

import sys, argparse, json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from tools.train.train_fewshot_allclass import (
    _build_class_index_instance,
    load_instance_tile_and_mask,
    extract_features,
    CATEGORY_NAMES,
    _resolve_paths,
)

OUT_DIR = _PROJECT_ROOT / "runs" / "diag" / "feature_vis"

# 论文级配色 (15 类 + 背景) | Paper-quality colormap (15 classes)
CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def load_model(device: str, checkpoint: str | None = None,
               unfreeze_layers: int = 0):
    """加载 FastSAM + 可选恢复 backbone 权重 | Load FastSAM + optionally restore backbone."""
    from ultralytics import FastSAM
    model_path = _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"
    model = FastSAM(str(model_path))
    model.model.cuda().eval()

    for p in model.model.parameters():
        p.requires_grad = False

    if checkpoint and unfreeze_layers > 0:
        ckpt = torch.load(checkpoint, map_location=device)
        if "backbone" in ckpt:
            seq = model.model.model
            n_total = len(seq)
            start = max(0, n_total - unfreeze_layers)
            for i_str, state in ckpt["backbone"].items():
                i = int(i_str)
                seq[i].load_state_dict(state)
            print(f"  Backbone: loaded {unfreeze_layers} layers (indices {start}-{n_total-1})")
        else:
            print(f"  [WARN] Checkpoint has no backbone weights, using frozen backbone")
    elif checkpoint:
        print(f"  [NOTE] No unfreeze_layers specified, using frozen backbone for feature extraction")

    return model


def extract_feature_vectors(model, images: list[np.ndarray],
                            device: str = "cuda") -> dict[str, np.ndarray]:
    """
    提取 P3/P4/P8 特征向量 (spatial avg pool) | Extract P3/P4/P8 feature vectors (spatial avg).

    :param model: FastSAM model.
    :param images: List of uint8 numpy arrays [H, W, 3].
    :param device: "cuda" or "cpu".
    :return: {"p3": np[N, 960], "p4": np[N, 1280], "p8": np[N, 1280]}
    """
    feats_list = extract_features(model, images, device, no_grad=True)

    p3_vecs, p4_vecs, p8_vecs = [], [], []
    for f in feats_list:
        # Spatial average pool → feature vector
        p3_vecs.append(f["p3"].mean(dim=(2, 3)).squeeze(0).cpu().numpy())  # [960]
        p4_vecs.append(f["p4"].mean(dim=(2, 3)).squeeze(0).cpu().numpy())  # [1280]
        p8_vecs.append(f["p8"].mean(dim=(2, 3)).squeeze(0).cpu().numpy())  # [1280]

    return {
        "p3": np.stack(p3_vecs, axis=0),
        "p4": np.stack(p4_vecs, axis=0),
        "p8": np.stack(p8_vecs, axis=0),
    }


def compute_cosine_matrix(feats: dict[str, np.ndarray],
                          labels: np.ndarray,
                          n_classes: int) -> dict[str, np.ndarray]:
    """
    计算每层的类间 Cosine 相似度矩阵 | Compute inter-class cosine similarity per level.

    :param feats: {"p3": [N, D3], "p4": [N, D4], "p8": [N, D8]}
    :param labels: [N] class indices (0-based).
    :param n_classes: total number of classes.
    :return: {"p3": [C, C], "p4": [C, C], "p8": [C, C]} cosine similarity matrices.
    """
    results = {}
    for level in ["p3", "p4", "p8"]:
        X = feats[level]  # [N, D]
        # L2-normalize
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # Per-class mean prototype
        prototypes = np.zeros((n_classes, X.shape[1]))
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() > 0:
                prototypes[c] = X_norm[mask].mean(axis=0)
                prototypes[c] /= np.linalg.norm(prototypes[c]) + 1e-8

        # Cosine similarity matrix
        cos_matrix = prototypes @ prototypes.T  # [C, C]
        results[level] = cos_matrix

    return results


def compute_silhouette(feats: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, float]:
    """
    计算每层的 Silhouette Score | Compute silhouette score per level.

    :return: {"p3": float, "p4": float, "p8": float}
    """
    from sklearn.metrics import silhouette_score

    results = {}
    for level in ["p3", "p4", "p8"]:
        X = feats[level]
        # L2 normalize for meaningful euclidean distance
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        try:
            score = silhouette_score(X_norm, labels, random_state=42)
        except Exception:
            score = float('nan')
        results[level] = score
    return results


def compute_intra_inter_ratio(feats: dict[str, np.ndarray],
                              labels: np.ndarray,
                              n_classes: int) -> dict[str, dict]:
    """
    计算类内/类间距离比 | Compute intra/inter-class distance ratio.

    :return: {"p3": {"intra": float, "inter": float, "ratio": float}, ...}
    """
    results = {}
    for level in ["p3", "p4", "p8"]:
        X = feats[level]
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        intra_dists = []
        inter_dists = []
        for c in range(n_classes):
            mask_c = labels == c
            if mask_c.sum() < 2:
                continue
            X_c = X_norm[mask_c]
            # Intra-class: all pairwise cosine distances within class
            cos_sim = X_c @ X_c.T  # [n_c, n_c]
            # Upper triangle (exclude diagonal)
            triu = cos_sim[np.triu_indices_from(cos_sim, k=1)]
            intra_dists.extend(1.0 - triu)

            # Inter-class: distance to other classes' centroids
            for d in range(c + 1, n_classes):
                mask_d = labels == d
                if mask_d.sum() == 0:
                    continue
                X_d = X_norm[mask_d]
                centroid_d = X_d.mean(axis=0)
                centroid_d /= np.linalg.norm(centroid_d) + 1e-8
                # Each sample in c vs centroid of d
                for v in X_c:
                    inter_dists.append(1.0 - float(np.dot(v, centroid_d)))

        intra_mean = float(np.mean(intra_dists)) if intra_dists else 0.0
        inter_mean = float(np.mean(inter_dists)) if inter_dists else 0.0
        ratio = inter_mean / max(intra_mean, 1e-8)

        results[level] = {
            "intra_mean": intra_mean,
            "inter_mean": inter_mean,
            "ratio": ratio,  # >1 means inter-class > intra-class (good)
        }
    return results


# ═══════════════════════════════════════════════════════════════════
# Plotting | 可视化
# ═══════════════════════════════════════════════════════════════════

def plot_tsne(feats: dict[str, np.ndarray], labels: np.ndarray,
              class_names: list[str], out_dir: Path, title_prefix: str = ""):
    """
    为 P3/P4/P8 分别绘制 t-SNE + 合并图 | Plot t-SNE for each level + combined figure.
    """
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_classes = len(class_names)
    colors = CLASS_COLORS[:n_classes]

    # ── Combined figure (1 row, 3 cols) ──
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    tsne_results = {}

    for ax_idx, level in enumerate(["p3", "p4", "p8"]):
        X = feats[level]
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # t-SNE
        perplexity = min(30, X.shape[0] // 4)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    max_iter=1000, metric='euclidean')
        X_2d = tsne.fit_transform(X_norm)
        tsne_results[level] = X_2d

        ax = axes[ax_idx]
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=colors[c], label=class_names[c],
                      s=12, alpha=0.7, edgecolors='none')

        stride = {"p3": 8, "p4": 16, "p8": 32}[level]
        dim = {"p3": 960, "p4": 1280, "p8": 1280}[level]
        ax.set_title(f"{level.upper()} (stride-{stride}, {dim}D)", fontsize=13, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared legend
    handles, labels_unique = [], []
    for c in range(n_classes):
        handles.append(Line2D([0], [0], marker='o', color='w',
                              markerfacecolor=colors[c], markersize=8))
        labels_unique.append(class_names[c])
    fig.legend(handles, labels_unique, loc='center left', bbox_to_anchor=(1.01, 0.5),
              fontsize=8, ncol=1, frameon=False)

    fig.suptitle(f"{title_prefix}Feature Space Visualization (t-SNE) — P3/P4/P8",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(out_dir / "tsne_combined.png", dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  [OK] t-SNE combined → {out_dir / 'tsne_combined.png'}")

    return tsne_results


def plot_tsne_single_class(feats: dict[str, np.ndarray], labels: np.ndarray,
                           class_names: list[str], target_classes: list[int],
                           out_dir: Path):
    """
    对指定类高亮显示 t-SNE (其他类灰色) | Highlight specific classes in t-SNE.
    """
    from sklearn.manifold import TSNE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_classes = len(class_names)
    colors = CLASS_COLORS[:n_classes]

    for level in ["p3", "p4", "p8"]:
        X = feats[level]
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        perplexity = min(30, X.shape[0] // 4)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    max_iter=1000, metric='euclidean')
        X_2d = tsne.fit_transform(X_norm)

        for target in target_classes:
            fig, ax = plt.subplots(figsize=(8, 7))

            # All classes in gray
            mask_gray = ~np.isin(labels, [target])
            ax.scatter(X_2d[mask_gray, 0], X_2d[mask_gray, 1],
                      c='lightgray', s=6, alpha=0.3, edgecolors='none')

            # Target class highlighted
            mask_target = labels == target
            ax.scatter(X_2d[mask_target, 0], X_2d[mask_target, 1],
                      c=colors[target], s=20, alpha=0.9, edgecolors='black',
                      linewidths=0.5)

            stride = {"p3": 8, "p4": 16, "p8": 32}[level]
            ax.set_title(f"{level.upper()} (stride-{stride}) — "
                        f"{class_names[target]} (class {target+1})",
                        fontsize=14, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            name_clean = class_names[target].replace(' ', '_')
            fig.savefig(out_dir / f"tsne_{level}_class{target+1}_{name_clean}.png",
                       dpi=200, bbox_inches='tight', facecolor='white')
            plt.close(fig)
    print(f"  [OK] Single-class t-SNE → {out_dir}")


def plot_cosine_matrix(cosine_matrices: dict[str, np.ndarray],
                       class_names: list[str], out_dir: Path):
    """绘制 Cosine 相似度矩阵 | Plot cosine similarity matrix."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    for ax_idx, level in enumerate(["p3", "p4", "p8"]):
        matrix = cosine_matrices[level]
        ax = axes[ax_idx]

        im = ax.imshow(matrix, cmap='RdYlBu_r', vmin=0.0, vmax=1.0, aspect='equal')

        # Annotate
        n = len(class_names)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(range(1, n + 1), fontsize=7)
        ax.set_yticklabels([f"{i+1}:{class_names[i][:8]}" for i in range(n)], fontsize=7)

        # Highlight diagonal
        for i in range(n):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                       fill=False, edgecolor='black', linewidth=1.5))

        diag_mean = float(np.diag(matrix).mean()) if n > 0 else 0
        off_diag = matrix[~np.eye(n, dtype=bool)].mean() if n > 1 else 0
        ax.set_title(f"{level.upper()} (cosine similarity)\n"
                    f"diag={diag_mean:.3f}, off-diag={off_diag:.3f}",
                    fontsize=11, fontweight='bold')

    fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    fig.suptitle("Inter-Class Prototype Cosine Similarity", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / "cosine_similarity.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  [OK] Cosine matrix → {out_dir / 'cosine_similarity.png'}")


def plot_per_class_compactness(feats: dict[str, np.ndarray],
                               labels: np.ndarray, class_names: list[str],
                               out_dir: Path):
    """
    绘制每类特征紧凑度 (类内方差) | Plot per-class feature compactness (intra-class variance).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_classes = len(class_names)
    data = {}
    for level in ["p3", "p4", "p8"]:
        X = feats[level]
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        intra_vars = []
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() < 2:
                intra_vars.append(0)
                continue
            X_c = X_norm[mask]
            centroid = X_c.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            # Average cosine distance to centroid
            dists = 1.0 - (X_c @ centroid)
            intra_vars.append(float(np.mean(dists)))

        data[level] = intra_vars

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_classes)
    width = 0.25

    for i, (level, color) in enumerate([("p3", "#1f77b4"), ("p4", "#ff7f0e"), ("p8", "#2ca02c")]):
        bars = ax.bar(x + i * width, data[level], width, label=level.upper(), color=color, alpha=0.85)
        # Annotate highest
        vals = data[level]
        top3 = sorted(range(len(vals)), key=lambda j: vals[j], reverse=True)[:3]
        for j in top3:
            if vals[j] > 0:
                ax.annotate(f'{class_names[j][:6]}', (x[j] + i * width, vals[j]),
                          textcoords="offset points", xytext=(0, 3),
                          fontsize=6, ha='center', rotation=90)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{i+1}:{name[:10]}" for i, name in enumerate(class_names)],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Intra-Class Cosine Distance (lower = more compact)", fontsize=11)
    ax.set_title("Per-Class Feature Compactness", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "per_class_compactness.png", dpi=200, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print(f"  [OK] Compactness → {out_dir / 'per_class_compactness.png'}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Feature Visualization Diagnosis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="训练 checkpoint (可选, 用于恢复 uf=8 backbone)")
    parser.add_argument("--unfreeze-layers", type=int, default=0,
                        help="解冻层数 (需与 checkpoint 匹配)")
    parser.add_argument("--data-root", type=str, default=None,
                        help="数据根目录 | Data root")
    parser.add_argument("--data-format", type=str, default="isaid_instance",
                        choices=["isaid_instance"],
                        help="数据格式 | Data format")
    parser.add_argument("--split", type=str, default="val",
                        help="数据划分 | Data split (train/val)")
    parser.add_argument("--samples-per-class", type=int, default=30,
                        help="每类采样数 | Samples per class")
    parser.add_argument("--classes", type=str, default=None,
                        help="指定类 ID (逗号分隔), None=全部 | Comma-separated class IDs, None=all")
    parser.add_argument("--highlight-classes", type=str, default="13,15",
                        help="高亮显示的单类 ID | Highlight class IDs (for single-class t-SNE)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    ts = datetime.now().strftime("%m%d_%H%M")

    if args.output_dir is None:
        tag = f"uf{args.unfreeze_layers}" if args.unfreeze_layers > 0 else "frozen"
        args.output_dir = str(OUT_DIR / f"{tag}_{ts}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  Feature Visualization Diagnosis")
    print(f"  Backbone: {'uf=' + str(args.unfreeze_layers) if args.unfreeze_layers > 0 else 'frozen'}")
    print(f"  Samples/class: {args.samples_per_class}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")

    # ── 1. Load model | 加载模型 ──
    print(f"\n[1/4] Loading model...")
    model = load_model(device, args.checkpoint, args.unfreeze_layers)
    print(f"  Model ready on {device}")

    # ── 2. Sample tiles | 采样 tile ──
    print(f"\n[2/4] Sampling tiles per class...")
    data_root, _, _, val_split = _resolve_paths(args)
    eval_split = args.split if args.split else val_split

    class_index = _build_class_index_instance(data_root, eval_split)

    # 过滤类 | Filter classes
    if args.classes:
        target_classes = [int(c.strip()) for c in args.classes.split(",")]
        class_index = {k: v for k, v in class_index.items() if k in target_classes}

    # 采样 | Sampling
    import random
    rng = random.Random(42)
    sampled_tiles = defaultdict(list)  # cls_id → [(stem, img, gt_mask)]
    for cls_id in sorted(class_index.keys()):
        src_to_tiles = class_index[cls_id]
        all_tiles = []
        for src, tiles in src_to_tiles.items():
            all_tiles.extend(tiles)

        n_sample = min(args.samples_per_class, len(all_tiles))
        selected = rng.sample(all_tiles, n_sample)

        for stem in selected:
            try:
                img, mask = load_instance_tile_and_mask(stem, eval_split, data_root)
                sampled_tiles[cls_id].append((stem, img, mask))
            except Exception:
                continue

        name = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")
        print(f"  Class {cls_id:>2d} ({name:<18s}): {len(sampled_tiles[cls_id])} tiles sampled")

    total = sum(len(v) for v in sampled_tiles.values())
    print(f"  Total: {total} tiles across {len(sampled_tiles)} classes")

    # ── 3. Extract features | 提取特征 ──
    print(f"\n[3/4] Extracting features...")
    all_images = []
    all_labels = []
    all_stems = []

    for cls_id in sorted(sampled_tiles.keys()):
        for stem, img, mask in sampled_tiles[cls_id]:
            all_images.append(img)
            all_labels.append(cls_id - 1)  # 0-based
            all_stems.append(stem)

    # 批量提取 (分批处理避免 OOM) | Batch extraction (avoid OOM)
    batch_size = 32
    p3_all, p4_all, p8_all = [], [], []
    for i in tqdm(range(0, len(all_images), batch_size), desc="  Extracting features"):
        batch = all_images[i:i + batch_size]
        feats = extract_feature_vectors(model, batch, device)
        p3_all.append(feats["p3"])
        p4_all.append(feats["p4"])
        p8_all.append(feats["p8"])

    feats = {
        "p3": np.concatenate(p3_all, axis=0),
        "p4": np.concatenate(p4_all, axis=0),
        "p8": np.concatenate(p8_all, axis=0),
    }
    labels = np.array(all_labels, dtype=np.int32)

    # 构建类名列表 | Build class name list
    sorted_cls_ids = sorted(sampled_tiles.keys())
    class_names = [CATEGORY_NAMES[c] for c in sorted_cls_ids]

    print(f"\n  Feature shapes:")
    for level in ["p3", "p4", "p8"]:
        print(f"    {level}: {feats[level].shape}")

    # ── 4. Analysis & Visualization | 分析与可视化 ──
    print(f"\n[4/4] Analysis & visualization...")

    # 4a. Silhouette scores
    sil_scores = compute_silhouette(feats, labels)
    print(f"\n  Silhouette Scores (higher = better clustering):")
    for level in ["p3", "p4", "p8"]:
        print(f"    {level}: {sil_scores[level]:.4f}")

    # 4b. Intra/Inter ratio
    ratios = compute_intra_inter_ratio(feats, labels, len(class_names))
    print(f"\n  Intra/Inter-Class Distance Ratio (>1 = inter > intra, good):")
    for level in ["p3", "p4", "p8"]:
        r = ratios[level]
        print(f"    {level}: intra={r['intra_mean']:.4f}, inter={r['inter_mean']:.4f}, "
              f"ratio={r['ratio']:.2f}×")

    # 4c. t-SNE
    print(f"\n  [Plot] t-SNE...")
    plot_tsne(feats, labels, class_names, out_dir,
              title_prefix=f"{'uf=' + str(args.unfreeze_layers) if args.unfreeze_layers > 0 else 'Frozen'} ")

    # 4d. Single-class t-SNE
    if args.highlight_classes:
        highlight_ids = [int(c.strip()) - 1 for c in args.highlight_classes.split(",")
                        if int(c.strip()) - 1 < len(class_names)]
        if highlight_ids:
            print(f"\n  [Plot] Single-class t-SNE...")
            plot_tsne_single_class(feats, labels, class_names, highlight_ids, out_dir)

    # 4e. Cosine similarity matrix
    print(f"\n  [Plot] Cosine similarity...")
    cos_matrices = compute_cosine_matrix(feats, labels, len(class_names))
    plot_cosine_matrix(cos_matrices, class_names, out_dir)

    # 4f. Per-class compactness
    print(f"\n  [Plot] Per-class compactness...")
    plot_per_class_compactness(feats, labels, class_names, out_dir)

    # ── Save statistics | 保存统计 ──
    stats = {
        "backbone": f"uf={args.unfreeze_layers}" if args.unfreeze_layers > 0 else "frozen",
        "n_samples": total,
        "n_classes": len(class_names),
        "classes": {str(cid): CATEGORY_NAMES[cid] for cid in sorted_cls_ids},
        "silhouette_scores": {k: float(v) for k, v in sil_scores.items()},
        "intra_inter_ratio": {k: {kk: float(vv) for kk, vv in v.items()}
                              for k, v in ratios.items()},
        "per_class_compactness": {},
        "cosine_diag_mean": {k: float(np.diag(v).mean()) for k, v in cos_matrices.items()},
        "cosine_offdiag_mean": {k: float(v[~np.eye(len(class_names), dtype=bool)].mean())
                                 for k, v in cos_matrices.items()},
    }

    # Per-class compactness detail
    for level in ["p3", "p4", "p8"]:
        X = feats[level]
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        for i, cid in enumerate(sorted_cls_ids):
            mask = labels == i
            if mask.sum() < 2:
                continue
            X_c = X_norm[mask]
            cent = X_c.mean(axis=0)
            cent /= np.linalg.norm(cent) + 1e-8
            dists = 1.0 - (X_c @ cent)
            stats["per_class_compactness"][f"{cid}_{level}"] = {
                "name": CATEGORY_NAMES[cid],
                "n": int(mask.sum()),
                "intra_cosine_dist": float(np.mean(dists)),
            }

    with open(out_dir / "feature_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] Statistics → {out_dir / 'feature_stats.json'}")

    # ── 论文关键发现 | Paper-Key Findings ──
    print(f"\n  {'=' * 60}")
    print(f"  KEY FINDINGS")
    print(f"  {'=' * 60}")
    best_level = max(sil_scores, key=lambda k: sil_scores[k])
    print(f"  1. Best clustering: {best_level.upper()} "
          f"(Silhouette={sil_scores[best_level]:.4f})")
    print(f"     → {'P8 is best for prototype' if best_level == 'p8' else 'Reconsider prototype source'}")

    worst = min(sil_scores, key=lambda k: sil_scores[k])
    print(f"  2. Worst clustering: {worst.upper()} "
          f"(Silhouette={sil_scores[worst]:.4f})")
    print(f"     → {'P3 is best for boundary (not semantics)' if worst == 'p3' else 'Reconsider'}")

    ratio_best = max(ratios, key=lambda k: ratios[k]["ratio"])
    print(f"  3. Best intra/inter ratio: {ratio_best.upper()} "
          f"({ratios[ratio_best]['ratio']:.2f}×)")
    print(f"     → Inter-class distance is {ratios[ratio_best]['ratio']:.1f}× intra-class distance")

    print(f"  4. Output: {out_dir}")
    print(f"  {'=' * 60}")


if __name__ == "__main__":
    main()
