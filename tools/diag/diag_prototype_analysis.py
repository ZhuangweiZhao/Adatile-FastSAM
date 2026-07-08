#!/usr/bin/env python3
"""
Prototype 塌缩与判别力诊断 | Prototype Collapse & Diversity Diagnosis.
======================================================================

核心问题 | Core Question:
    uf=8 将 mIoU 从 0.285 提升到 0.418 (+46.7%)，
    但 class prototype cosine 从 ~0.87 塌缩到 ~0.999 (几乎平行)。
    那 prototype 到底还有没有用？Coeff Predictor 在干什么？

分析维度 | Analysis Dimensions:
    Q1: Prototype Collapse — 同类/跨类 prototype 的 cosine 分布
    Q2: Coefficient Diversity — 塌缩的 prototype 是否仍能产生不同的 32D coefficients
    Q3: Prototype Stability — P4 vs P8 prototype 的跨采样稳定性
    Q4: Ablation — 随机 prototype vs 真实 prototype 的 mask 质量差异

用法 | Usage::

    # Frozen backbone
    python tools/diag/diag_prototype_analysis.py --device cuda

    # uf=8 backbone
    python tools/diag/diag_prototype_analysis.py \
        --checkpoint runs/.../best_model.pt --unfreeze-layers 8 --device cuda

输出 | Output:
    runs/diag/proto_analysis/
    ├── proto_collapse.png          # Proto collapse: frozen vs uf=8 histogram
    ├── coeff_diversity.png         # 15×15 coefficient cosine matrix
    ├── proto_stability.png         # P4 vs P8 per-class stability
    ├── ablation_random_proto.png   # Real vs random prototype mask quality
    └── proto_stats.json
"""

from __future__ import annotations

import sys, argparse, json, random
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
    compute_support_prototype,
    CATEGORY_NAMES,
    _resolve_paths,
)

from adatile.sparse.coefficient_predictor import ProtoCoeffPredictor

OUT_DIR = _PROJECT_ROOT / "runs" / "diag" / "proto_analysis"


def load_model(device: str, checkpoint: str | None = None,
               unfreeze_layers: int = 0):
    """加载 FastSAM + 可选恢复 backbone | Load model."""
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
            print(f"  Backbone: loaded {unfreeze_layers} layers")

    return model


def load_coeff_predictor(checkpoint: str, device: str) -> ProtoCoeffPredictor | None:
    """从 checkpoint 加载 ProtoCoeffPredictor | Load ProtoCoeffPredictor from checkpoint."""
    ckpt = torch.load(checkpoint, map_location=device)
    decoder_state = ckpt.get("decoder", {})

    # 检测 predictor 的架构参数 | Detect predictor architecture
    # coeff_predictor.0.weight shape: [hidden, feat, 1] or [hidden, feat]
    coeff_keys = [k for k in decoder_state if k.startswith("coeff_predictor.")]
    if not coeff_keys:
        # Try alternate keys for P3P4 decoder
        print("  [WARN] No coeff_predictor in checkpoint, trying alternate keys")
        return None

    # Infer dimensions
    w0 = decoder_state.get("coeff_predictor.0.weight")  # Linear: [hidden, feat]
    if w0 is None:
        return None

    hidden_dim, feat_dim = w0.shape
    # Infer proto_dim from last layer
    w_last = None
    for k in coeff_keys:
        if k.endswith(".weight") and "predictor" not in k:
            w_last = decoder_state[k]
    if w_last is not None:
        proto_dim = w_last.shape[0]
    else:
        proto_dim = 32

    print(f"  CoeffPredictor: feat={feat_dim}, hidden={hidden_dim}, proto={proto_dim}")

    predictor = ProtoCoeffPredictor(
        proto_dim=proto_dim, feat_dim=feat_dim, hidden_dim=hidden_dim
    ).to(device)

    # Load predictor weights only
    pred_state = {k.replace("coeff_predictor.", ""): v
                  for k, v in decoder_state.items()
                  if k.startswith("coeff_predictor.")}
    predictor.load_state_dict(pred_state)
    predictor.eval()
    return predictor


# ═══════════════════════════════════════════════════════════════════
# Q1: Prototype Collapse Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_prototype_collapse(model, class_index: dict, eval_split: str,
                                data_root: Path, device: str,
                                proto_source: str = "p4",
                                n_samples: int = 10) -> dict:
    """
    分析 Prototype 塌缩 | Analyze prototype collapse.

    对每类采样 n_samples 个不同的 support tile，
    各计算 prototype，统计类内/类间 cosine 分布。
    For each class, sample n_samples different support tiles,
    compute prototype each, analyze intra/inter cosine distribution.
    """
    rng = random.Random(42)
    all_protos = {}  # cls_id → [n_samples, feat_dim]
    feat_dim = None

    print(f"\n  [Q1] Prototype Collapse (source={proto_source})...")
    for cls_id in tqdm(sorted(class_index.keys()), desc="  Computing prototypes"):
        src_to_tiles = class_index[cls_id]
        all_tiles = []
        for src, tiles in src_to_tiles.items():
            all_tiles.extend(tiles)

        n_sample = min(n_samples, len(all_tiles))
        selected = rng.sample(all_tiles, n_sample)

        protos = []
        for stem in selected:
            try:
                img, _ = load_instance_tile_and_mask(stem, eval_split, data_root)
                feats = extract_features(model, [img], device, no_grad=True)
                proto = compute_support_prototype(feats, source=proto_source)
                protos.append(proto.squeeze(0).cpu().numpy())
            except Exception:
                continue

        if protos:
            all_protos[cls_id] = np.stack(protos, axis=0)
            if feat_dim is None:
                feat_dim = all_protos[cls_id].shape[1]

    if not all_protos:
        return {}

    # Compute intra-class cosine distance
    intra_dists = []
    for cls_id, protos in all_protos.items():
        protos_norm = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
        cos_sim = protos_norm @ protos_norm.T
        triu = cos_sim[np.triu_indices_from(cos_sim, k=1)]
        intra_dists.extend(1.0 - triu)

    # Compute inter-class cosine distance (between class centroids)
    centroids = {}
    for cls_id, protos in all_protos.items():
        c = protos.mean(axis=0)
        centroids[cls_id] = c / (np.linalg.norm(c) + 1e-8)

    inter_dists = []
    cls_ids = sorted(centroids.keys())
    for i in range(len(cls_ids)):
        for j in range(i + 1, len(cls_ids)):
            cos = float(np.dot(centroids[cls_ids[i]], centroids[cls_ids[j]]))
            inter_dists.append(1.0 - cos)

    # Compute centroid cosine matrix
    n_cls = len(cls_ids)
    cos_matrix = np.zeros((n_cls, n_cls))
    for i, ci in enumerate(cls_ids):
        for j, cj in enumerate(cls_ids):
            cos_matrix[i, j] = float(np.dot(centroids[ci], centroids[cj]))

    diag_mean = float(np.diag(cos_matrix).mean())
    off_diag = cos_matrix[~np.eye(n_cls, dtype=bool)].mean()

    results = {
        "intra_mean": float(np.mean(intra_dists)) if intra_dists else 0,
        "intra_std": float(np.std(intra_dists)) if intra_dists else 0,
        "intra_min": float(np.min(intra_dists)) if intra_dists else 0,
        "intra_max": float(np.max(intra_dists)) if intra_dists else 0,
        "inter_mean": float(np.mean(inter_dists)) if inter_dists else 0,
        "inter_std": float(np.std(inter_dists)) if inter_dists else 0,
        "inter_min": float(np.min(inter_dists)) if inter_dists else 0,
        "inter_max": float(np.max(inter_dists)) if inter_dists else 0,
        "cos_matrix_diag_mean": diag_mean,
        "cos_matrix_offdiag_mean": float(off_diag),
        "collapse_ratio": float(np.mean(intra_dists) / max(np.mean(inter_dists), 1e-8))
                          if intra_dists and inter_dists else 0,
        "cos_matrix": cos_matrix.tolist(),
        "class_ids": cls_ids,
        "class_names": [CATEGORY_NAMES[c] for c in cls_ids],
    }

    print(f"    Intra: {results['intra_mean']:.4f} ± {results['intra_std']:.4f}")
    print(f"    Inter: {results['inter_mean']:.4f} ± {results['inter_std']:.4f}")
    print(f"    Diag cos: {diag_mean:.4f}, Off-diag cos: {off_diag:.4f}")
    print(f"    Collapse ratio: {results['collapse_ratio']:.3f} (<1 = collapsed)")

    return results


# ═══════════════════════════════════════════════════════════════════
# Q2: Coefficient Diversity Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_coefficient_diversity(model, class_index: dict, eval_split: str,
                                   data_root: Path, device: str,
                                   coeff_predictor: ProtoCoeffPredictor,
                                   proto_source: str = "p4",
                                   n_samples: int = 5) -> dict:
    """
    分析 Coefficient 多样性 | Analyze coefficient diversity.

    对每类采样多个 support → prototype → coeff_predictor → 32D coefficient。
    统计 15×15 coefficient cosine matrix。
    For each class: support → prototype → 32D coefficients → diversity analysis.
    """
    rng = random.Random(42)
    all_coeffs = {}  # cls_id → [n_samples, 32]

    print(f"\n  [Q2] Coefficient Diversity...")
    for cls_id in tqdm(sorted(class_index.keys()), desc="  Computing coefficients"):
        src_to_tiles = class_index[cls_id]
        all_tiles = []
        for src, tiles in src_to_tiles.items():
            all_tiles.extend(tiles)

        n_sample = min(n_samples, len(all_tiles))
        selected = rng.sample(all_tiles, n_sample)

        coeffs_list = []
        for stem in selected:
            try:
                img, _ = load_instance_tile_and_mask(stem, eval_split, data_root)
                feats = extract_features(model, [img], device, no_grad=True)
                proto = compute_support_prototype(feats, source=proto_source)

                with torch.no_grad():
                    coeffs = coeff_predictor(proto)  # [1, 32]
                coeffs_list.append(coeffs.squeeze(0).cpu().numpy())
            except Exception as e:
                continue

        if coeffs_list:
            all_coeffs[cls_id] = np.stack(coeffs_list, axis=0)

    if not all_coeffs:
        return {}

    # Centroid coefficients
    cls_ids = sorted(all_coeffs.keys())
    n_cls = len(cls_ids)
    centroids = {}
    for cid in cls_ids:
        c = all_coeffs[cid].mean(axis=0)
        centroids[cid] = c / (np.linalg.norm(c) + 1e-8)

    # Coefficient cosine matrix
    coeff_cos = np.zeros((n_cls, n_cls))
    for i, ci in enumerate(cls_ids):
        for j, cj in enumerate(cls_ids):
            coeff_cos[i, j] = float(np.dot(centroids[ci], centroids[cj]))

    diag = float(np.diag(coeff_cos).mean())
    off = float(coeff_cos[~np.eye(n_cls, dtype=bool)].mean())

    print(f"    Coeff diag cos: {diag:.4f}, Off-diag cos: {off:.4f}")
    print(f"    Diversity ratio: {off:.4f} ({'GOOD' if off < 0.95 else 'COLLAPSED'})")

    return {
        "coeff_cos_diag": diag,
        "coeff_cos_offdiag": off,
        "coeff_cos_matrix": coeff_cos.tolist(),
        "class_ids": cls_ids,
        "class_names": [CATEGORY_NAMES[c] for c in cls_ids],
    }


# ═══════════════════════════════════════════════════════════════════
# Q3: Prototype Stability (P4 vs P8)
# ═══════════════════════════════════════════════════════════════════

def analyze_prototype_stability(model, class_index: dict, eval_split: str,
                                 data_root: Path, device: str,
                                 n_samples: int = 10) -> dict:
    """
    对比 P4 vs P8 prototype 跨采样稳定性 | Compare P4 vs P8 stability across samples.

    对每类, 用同一组 tile 分别提取 P4/P8 prototype，
    统计 Intra-class cosine variance。
    Same tiles → P4 and P8 prototypes → compare intra-class variance.
    """
    rng = random.Random(42)

    results = {"p4": defaultdict(list), "p8": defaultdict(list)}

    print(f"\n  [Q3] Prototype Stability (P4 vs P8)...")
    for cls_id in tqdm(sorted(class_index.keys()), desc="  Computing stability"):
        src_to_tiles = class_index[cls_id]
        all_tiles = []
        for src, tiles in src_to_tiles.items():
            all_tiles.extend(tiles)

        n_sample = min(n_samples, len(all_tiles))
        selected = rng.sample(all_tiles, n_sample)

        for stem in selected:
            try:
                img, _ = load_instance_tile_and_mask(stem, eval_split, data_root)
                feats = extract_features(model, [img], device, no_grad=True)
                proto_p4 = compute_support_prototype(feats, source="p4").squeeze(0).cpu().numpy()
                proto_p8 = compute_support_prototype(feats, source="p8").squeeze(0).cpu().numpy()
                results["p4"][cls_id].append(proto_p4)
                results["p8"][cls_id].append(proto_p8)
            except Exception:
                continue

    # Compute per-class intra-class cosine variance
    stats = {"p4": {}, "p8": {}}
    for source in ["p4", "p8"]:
        intra_vars = []
        for cls_id, protos in results[source].items():
            if len(protos) < 2:
                continue
            protos = np.stack(protos)
            protos_norm = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
            centroid = protos_norm.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            # Average cosine similarity to centroid
            cos_to_centroid = protos_norm @ centroid
            mean_sim = float(np.mean(cos_to_centroid))
            std_sim = float(np.std(cos_to_centroid))
            intra_vars.append(1.0 - mean_sim)
            stats[source][cls_id] = {
                "name": CATEGORY_NAMES.get(cls_id, "?"),
                "n": len(protos),
                "intra_cos_mean": mean_sim,
                "intra_cos_std": std_sim,
                "intra_dist": 1.0 - mean_sim,
            }

        mean_intra = float(np.mean(intra_vars)) if intra_vars else 0
        print(f"    {source.upper()}: mean intra-cos-dist = {mean_intra:.4f} "
              f"({'MORE STABLE' if mean_intra < 0.05 else 'LESS STABLE'})")

    return stats


# ═══════════════════════════════════════════════════════════════════
# Q4: Random Prototype Ablation
# ═══════════════════════════════════════════════════════════════════

def analyze_random_proto_ablation(model, class_index: dict, eval_split: str,
                                   data_root: Path, device: str,
                                   coeff_predictor: ProtoCoeffPredictor | None,
                                   n_samples: int = 20) -> dict:
    """
    随机 prototype 消融 | Random prototype ablation.

    对 query tile:
    1. 真实 prototype → coeffs → proto_mask → IoU
    2. 随机 prototype (L2-normalized randn) → coeffs → proto_mask → IoU
    3. 零 prototype (all zeros) → coeffs → proto_mask → IoU

    如果随机 proto 的 IoU ≈ 真实 proto，说明 prototype 已失效。
    If random proto IoU ≈ real proto IoU, prototype is dead.
    """
    if coeff_predictor is None:
        print("  [Q4] SKIP: no coeff_predictor")
        return {}

    rng = random.Random(42)
    feat_dim = 640  # P4/P8 dim for FastSAM

    print(f"\n  [Q4] Random Prototype Ablation...")
    real_ious = []
    random_ious = []
    zero_ious = []

    for cls_id in tqdm(sorted(class_index.keys()), desc="  Ablation"):
        src_to_tiles = class_index[cls_id]
        all_tiles = []
        for src, tiles in src_to_tiles.items():
            all_tiles.extend(tiles)

        n_sample = min(n_samples, len(all_tiles))
        selected = rng.sample(all_tiles, n_sample)

        # Support tile (different from query)
        for stem in selected:
            try:
                img, gt_mask = load_instance_tile_and_mask(stem, eval_split, data_root)
                feats = extract_features(model, [img], device, no_grad=True)[0]
                proto_masks = feats["proto"].to(device)  # [1, 32, H/4, W/4]
                H, W = gt_mask.shape

                # Real prototype
                real_proto = compute_support_prototype([feats], source="p4")

                # Random prototype
                rand_proto = torch.randn(1, feat_dim, device=device)
                rand_proto = F.normalize(rand_proto, p=2, dim=-1)

                # Zero prototype
                zero_proto = torch.zeros(1, feat_dim, device=device)

                with torch.no_grad():
                    for proto, iou_list in [(real_proto, real_ious),
                                             (rand_proto, random_ious),
                                             (zero_proto, zero_ious)]:
                        coeffs = coeff_predictor(proto)  # [1, 32]
                        mask = coeff_predictor.generate_mask(coeffs, proto_masks.squeeze(0))
                        # [H/4, W/4] → original resolution
                        mask_up = F.interpolate(
                            mask.unsqueeze(0).unsqueeze(0),
                            size=(H, W), mode='bilinear', align_corners=False
                        ).squeeze()
                        pred_bin = (mask_up > 0.5).float().cpu().numpy()
                        gt_bin = (gt_mask > 0.5).astype(np.float32)
                        inter = (pred_bin * gt_bin).sum()
                        union = (pred_bin + gt_bin).clip(0, 1).sum()
                        iou = float(inter / max(union, 1))
                        iou_list.append(iou)

            except Exception:
                continue

    real_mean = float(np.mean(real_ious)) if real_ious else 0
    rand_mean = float(np.mean(random_ious)) if random_ious else 0
    zero_mean = float(np.mean(zero_ious)) if zero_ious else 0

    print(f"    Real proto IoU:   {real_mean:.4f}")
    print(f"    Random proto IoU: {rand_mean:.4f} (Δ={rand_mean - real_mean:+.4f})")
    print(f"    Zero proto IoU:   {zero_mean:.4f} (Δ={zero_mean - real_mean:+.4f})")

    # Interpretation
    if rand_mean > real_mean * 0.8:
        print(f"    [VERDICT] Prototype is NEAR-DEAD — random ≈ real")
    elif rand_mean > real_mean * 0.5:
        print(f"    [VERDICT] Prototype has MODERATE contribution")
    else:
        print(f"    [VERDICT] Prototype is ESSENTIAL — random << real")

    return {
        "real_proto_iou": real_mean,
        "random_proto_iou": rand_mean,
        "zero_proto_iou": zero_mean,
        "n_samples": len(real_ious),
    }


# ═══════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_proto_collapse(frozen_stats: dict, uf8_stats: dict, out_dir: Path):
    """绘制 Proto collapse 对比直方图 | Plot collapse comparison histogram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not frozen_stats or not uf8_stats:
        print("  [SKIP] proto_collapse — missing data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, stats, color) in [
        (axes[0], "Frozen", frozen_stats, "#1f77b4"),
        (axes[1], "uf=8", uf8_stats, "#d62728"),
    ]:
        intra = stats.get("intra_mean", 0)
        inter = stats.get("inter_mean", 0)
        diag = stats.get("cos_matrix_diag_mean", 0)
        off = stats.get("cos_matrix_offdiag_mean", 0)

        # Bar chart: intra vs inter
        categories = ["Intra-Class\n(proto distance)", "Inter-Class\n(proto distance)",
                      "Diag Cos\n(同类相似度)", "Off-Diag Cos\n(跨类相似度)"]
        values = [intra, inter, diag, off]
        colors = [color, color, "#2ca02c", "#ff7f0e"]

        bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')

        # Collapse warning
        if off > 0.95:
            ax.text(0.5, 0.95, "WARNING: Prototype Collapse\n(off-diag cos > 0.95)",
                   transform=ax.transAxes, ha='center', fontsize=12,
                   color='red', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_title(f"{label} Backbone", fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Prototype Collapse Analysis: Frozen vs uf=8", fontsize=16, fontweight='bold')
    plt.tight_layout()
    fig.savefig(out_dir / "proto_collapse.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] proto_collapse.png")


def plot_coeff_diversity(coeff_stats: dict, out_dir: Path):
    """绘制 Coefficient diversity heatmap | Plot coefficient cosine matrix."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not coeff_stats:
        print("  [SKIP] coeff_diversity — no coeff predictor")
        return

    matrix = np.array(coeff_stats["coeff_cos_matrix"])
    names = coeff_stats["class_names"]
    n = len(names)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='RdYlBu_r', vmin=0.0, vmax=1.0, aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"{i+1}" for i in range(n)], fontsize=8)
    ax.set_yticklabels([f"{i+1}:{names[i][:10]}" for i in range(n)], fontsize=8)

    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                                   fill=False, edgecolor='black', linewidth=1.5))

    diag = coeff_stats["coeff_cos_diag"]
    off = coeff_stats["coeff_cos_offdiag"]

    title_color = 'red' if off > 0.95 else 'green'
    verdict = "COLLAPSED" if off > 0.95 else "DIVERSE"
    ax.set_title(f"Coefficient Cosine Matrix\ndiag={diag:.4f}, off-diag={off:.4f} → {verdict}",
                fontsize=13, fontweight='bold', color=title_color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_dir / "coeff_diversity.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] coeff_diversity.png")


def plot_proto_stability(stability_stats: dict, out_dir: Path):
    """绘制 P4 vs P8 稳定性对比 | Plot P4 vs P8 stability."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not stability_stats:
        print("  [SKIP] proto_stability — no data")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    cls_ids = sorted([int(k) for k in stability_stats.get("p4", {}).keys()])
    # Normalize key access (can be int or str)
    def _get(d, key):
        return d.get(key, d.get(str(key), d.get(int(key) if isinstance(key, str) else key, {})))

    names = [_get(stability_stats["p4"], c)["name"] for c in cls_ids]

    x = np.arange(len(cls_ids))
    width = 0.35

    p4_dists = [_get(stability_stats["p4"], c).get("intra_dist", 0) for c in cls_ids]
    p8_dists = [_get(stability_stats["p8"], c).get("intra_dist", 0) for c in cls_ids]

    bars1 = ax.bar(x - width / 2, p4_dists, width, label="P4 Prototype", color="#1f77b4", alpha=0.85)
    bars2 = ax.bar(x + width / 2, p8_dists, width, label="P8 Prototype", color="#2ca02c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{cid}:{name[:8]}" for cid, name in zip(cls_ids, names)],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Intra-Class Cosine Distance\n(lower = more stable)", fontsize=11)
    ax.set_title("Prototype Stability: P4 vs P8", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Highlight better source
    p4_mean = float(np.mean(p4_dists)) if p4_dists else 0
    p8_mean = float(np.mean(p8_dists)) if p8_dists else 0
    ax.text(0.02, 0.95, f"P4 mean: {p4_mean:.4f}\nP8 mean: {p8_mean:.4f}\n"
            f"Winner: {'P8' if p8_mean < p4_mean else 'P4'}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_dir / "proto_stability.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] proto_stability.png")


def plot_ablation(ablation: dict, out_dir: Path):
    """绘制 Random Proto Ablation 结果 | Plot random proto ablation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not ablation:
        print("  [SKIP] ablation — no data")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["Real Proto", "Random Proto", "Zero Proto"]
    values = [ablation["real_proto_iou"], ablation["random_proto_iou"], ablation["zero_proto_iou"]]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
               f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel("IoU (proto-only, no refinement)", fontsize=11)
    ax.set_title("Prototype Ablation: Real vs Random vs Zero", fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Verdict
    real = ablation["real_proto_iou"]
    rand = ablation["random_proto_iou"]
    if rand > real * 0.8:
        verdict = "Prototype is NEAR-DEAD"
        vc = 'red'
    elif rand > real * 0.5:
        verdict = "Prototype has MODERATE contribution"
        vc = 'orange'
    else:
        verdict = "Prototype is ESSENTIAL"
        vc = 'green'

    ax.text(0.5, 0.92, verdict, transform=ax.transAxes, ha='center',
           fontsize=13, fontweight='bold', color=vc,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_dir / "ablation_random_proto.png", dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [OK] ablation_random_proto.png")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Prototype Collapse & Diversity Diagnosis")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--unfreeze-layers", type=int, default=0)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--data-format", type=str, default="isaid_instance")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    ts = datetime.now().strftime("%m%d_%H%M")
    tag = f"uf{args.unfreeze_layers}" if args.unfreeze_layers > 0 else "frozen"

    if args.output_dir is None:
        args.output_dir = str(OUT_DIR / f"{tag}_{ts}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  Prototype Collapse & Diversity Diagnosis")
    print(f"  Backbone: {'uf=' + str(args.unfreeze_layers) if args.unfreeze_layers > 0 else 'frozen'}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")

    # Load model
    print(f"\n[1/3] Loading model...")
    model = load_model(device, args.checkpoint, args.unfreeze_layers)

    # Load coeff predictor if checkpoint available
    coeff_predictor = None
    if args.checkpoint:
        coeff_predictor = load_coeff_predictor(args.checkpoint, device)

    # Build class index
    print(f"\n[2/3] Building class index...")
    data_root, _, _, val_split = _resolve_paths(args)
    class_index = _build_class_index_instance(data_root, val_split if args.split == "val" else args.split)

    # ═══════════════════════════════════════════════════════════════
    # Q1: Prototype Collapse
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[3/3] Analysis...")
    print(f"\n{'=' * 60}")
    print(f"  Q1: PROTOTYPE COLLAPSE")
    print(f"{'=' * 60}")

    collapse_p4 = analyze_prototype_collapse(model, class_index, val_split, data_root, device, "p4")
    collapse_p8 = analyze_prototype_collapse(model, class_index, val_split, data_root, device, "p8")

    # ═══════════════════════════════════════════════════════════════
    # Q2: Coefficient Diversity
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Q2: COEFFICIENT DIVERSITY")
    print(f"{'=' * 60}")

    coeff_p4 = {}
    coeff_p8 = {}
    if coeff_predictor is not None:
        coeff_p4 = analyze_coefficient_diversity(model, class_index, val_split, data_root,
                                                  device, coeff_predictor, "p4")
        coeff_p8 = analyze_coefficient_diversity(model, class_index, val_split, data_root,
                                                  device, coeff_predictor, "p8")
    else:
        print("  [SKIP] No coeff_predictor available (need --checkpoint)")

    # ═══════════════════════════════════════════════════════════════
    # Q3: Prototype Stability
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Q3: PROTOTYPE STABILITY (P4 vs P8)")
    print(f"{'=' * 60}")

    stability = analyze_prototype_stability(model, class_index, val_split, data_root, device)

    # ═══════════════════════════════════════════════════════════════
    # Q4: Random Prototype Ablation
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  Q4: RANDOM PROTOTYPE ABLATION")
    print(f"{'=' * 60}")

    ablation = {}
    if coeff_predictor is not None:
        ablation = analyze_random_proto_ablation(model, class_index, val_split, data_root,
                                                  device, coeff_predictor)
    else:
        print("  [SKIP] No coeff_predictor available")

    # ═══════════════════════════════════════════════════════════════
    # Plotting
    # ═══════════════════════════════════════════════════════════════
    print(f"\n[Plotting]...")

    # Q1 plot: collapse comparison (would need frozen+uf8 in same run)
    # For single run, just save the stats
    print(f"  [NOTE] proto_collapse comparison plot requires both frozen and uf=8 runs")
    print(f"         Run this script twice (frozen + uf=8), then compare stats.")

    # Q2 plot: coefficient diversity
    if coeff_p4:
        plot_coeff_diversity(coeff_p4, out_dir)
    if coeff_p8:
        # Save P8 separately
        p8_dir = out_dir / "p8_proto"
        p8_dir.mkdir(exist_ok=True)
        plot_coeff_diversity(coeff_p8, p8_dir)

    # Q3 plot: stability
    if stability:
        plot_proto_stability(stability, out_dir)

    # Q4 plot: ablation
    if ablation:
        plot_ablation(ablation, out_dir)

    # ═══════════════════════════════════════════════════════════════
    # Save all statistics
    # ═══════════════════════════════════════════════════════════════
    stats = {
        "backbone": f"uf={args.unfreeze_layers}" if args.unfreeze_layers > 0 else "frozen",
        "q1_prototype_collapse": {
            "p4": collapse_p4,
            "p8": collapse_p8,
        },
        "q2_coefficient_diversity": {
            "p4_proto": coeff_p4,
            "p8_proto": coeff_p8,
        },
        "q3_prototype_stability": stability,
        "q4_random_ablation": ablation,
    }

    # Clean up for JSON (remove large matrices from stats if needed)
    stats_clean = json.loads(json.dumps(stats, default=str))

    with open(out_dir / "proto_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_clean, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] Statistics → {out_dir / 'proto_stats.json'}")

    # ═══════════════════════════════════════════════════════════════
    # Key Findings Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print(f"  KEY FINDINGS")
    print(f"{'=' * 60}")

    if collapse_p4:
        off = collapse_p4["cos_matrix_offdiag_mean"]
        if off > 0.95:
            print(f"  1. Prototype COLLAPSE confirmed: off-diag cos = {off:.4f}")
            print(f"     → Class prototypes are nearly parallel")
        else:
            print(f"  1. Prototype SEPARABLE: off-diag cos = {off:.4f}")

    if coeff_p4:
        off = coeff_p4["coeff_cos_offdiag"]
        if off > 0.95:
            print(f"  2. Coefficient COLLAPSED: off-diag cos = {off:.4f}")
            print(f"     → Even coefficients don't distinguish classes → PROTOTYPE IS DEAD")
        else:
            print(f"  2. Coefficient DIVERSE: off-diag cos = {off:.4f}")
            print(f"     → CoeffPredictor rescues discriminability from collapsed prototypes")

    if ablation:
        real = ablation["real_proto_iou"]
        rand = ablation["random_proto_iou"]
        if rand > real * 0.8:
            print(f"  3. ABLATION: Random ≈ Real ({rand:.4f} vs {real:.4f})")
            print(f"     → PROTOTYPE IS NEAR-DEAD. Decoder works without it.")
        else:
            print(f"  3. ABLATION: Real > Random ({real:.4f} vs {rand:.4f})")
            print(f"     → Prototype provides meaningful conditioning.")

    print(f"  4. Output: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
