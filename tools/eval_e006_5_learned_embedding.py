#!/usr/bin/env python3
"""
E006.5: Learned Embedding — P4 → 64-dim 投影后是否可聚类？
=============================================================

单变量对照 | Single-variable control:
    E006  (Raw P4 1280-dim):      Silhouette ≈ 0.03, 不可聚类
    E006.5 (Learned 64-dim):      Silhouette = ?

核心问题 | Core question:
    学习后的低维嵌入是否比原始 P4 更适合聚类？
    Does a learned low-dim embedding become more clusterable than raw P4?

假设 | Hypothesis:
    Raw P4 是可线性分离的 (E002, Dice=0.40)，但不是 Prototype-Friendly 的。
    Raw P4 is linearly separable but not prototype-friendly.
    一个简单的 1×1 Conv 投影 + 分割训练可能使特征空间更可聚类。
    A simple 1×1 Conv projection + seg training may make the space more clusterable.

实验设计 | Experiment design:
    1. P4 (1280-dim) → 1×1 Conv → 64-dim → ReLU → Embedding
    2. 加上 1×1 Head: Embedding → 1-dim logit → BCE loss 训练
    3. 训练后提取 64-dim Embedding
    4. 重做 KMeans / Silhouette / Cosine Δ 分析

如果 Silhouette 从 0.03 → 0.20+:
    → 非常漂亮的论文故事:
      Raw P4 → 不可聚类 → Learned Embedding → 可聚类 → Proto/SPM 有理论依据

用法 | Usage:
    python tools/eval_e006_5_learned_embedding.py
    python tools/eval_e006_5_learned_embedding.py --epochs 20 --embed-dim 64
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
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
from adatile.metrics import compute_dice, compute_miou, format_param_count


def parse_args():
    p = argparse.ArgumentParser(description="E006.5: Learned Embedding clustering analysis")
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--epochs", type=int, default=15,
                   help="Training epochs for the embedding projector")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=64,
                   help="Embedding dimension (default 64)")
    p.add_argument("--max-samples", type=int, default=20,
                   help="Max images for clustering analysis")
    p.add_argument("--max-pixels-per-image", type=int, default=5000)
    p.add_argument("--n-clusters", type=int, default=5,
                   help="K-Means clusters per class")
    p.add_argument("--tsne-samples", type=int, default=3000)
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Embedding Projector — 最简投影器 | Minimal embedding projector
# ═══════════════════════════════════════════════════════════════════

class EmbeddingProjector(nn.Module):
    """
    P4 特征 → 低维嵌入 + 分割头 | P4 features → low-dim embedding + seg head.

    P4 [B, 1280, H/16, W/16]
         │
    1×1 Conv(1280 → embed_dim)
         │
    ReLU
         │
    Embedding [B, embed_dim, H/16, W/16]    ← 用于聚类分析 | For clustering analysis
         │
    1×1 Conv(embed_dim → 1)
         │
    Logit [B, 1, H/16, W/16]                ← 用于分割训练 | For segmentation training
    """

    def __init__(self, in_channels: int = 1280, embed_dim: int = 64):
        super().__init__()
        # 投影到低维嵌入 | Project to low-dim embedding
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 分割头 | Segmentation head
        self.head = nn.Conv2d(embed_dim, 1, kernel_size=1, bias=True)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  EmbeddingProjector: {format_param_count(n_params)} ({n_params:,}) "
              f"(project={sum(p.numel() for p in self.project.parameters()):,} + "
              f"head={sum(p.numel() for p in self.head.parameters()):,})")

    def forward(self, p4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            embedding: [B, embed_dim, H/16, W/16]  低维嵌入 | Low-dim embedding
            logit:     [B, 1, H/16, W/16]          分割 logit | Segmentation logit
        """
        embedding = self.project(p4)
        logit = self.head(embedding)
        return embedding, logit


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Train Embedding Projector
# ═══════════════════════════════════════════════════════════════════

def train_projector(projector, backbone, train_ds, val_ds, args, device, recorder):
    """训练投影器（只训练 projector，backbone 冻结）| Train projector (backbone frozen)."""
    projector.train()
    optimizer = torch.optim.Adam(projector.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    best_dice = 0.0
    best_state = None

    print(f"\n  Training {args.epochs} epochs (lr={args.lr}, CosineLR)...")
    for epoch in range(1, args.epochs + 1):
        # Train
        projector.train()
        total_loss = 0.0
        for idx in tqdm(range(len(train_ds)), desc=f"  Epoch {epoch}/{args.epochs}", leave=False):
            sample = train_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            gt_mask = sample["masks"].to(device)
            # 确保 gt_mask 是 [H, W] | Ensure gt_mask is [H, W]
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.squeeze(0)
            elif gt_mask.dim() == 4:
                gt_mask = gt_mask.squeeze(0).squeeze(0)

            with torch.no_grad():
                features = backbone(image)
            p4 = features["p4"]

            embedding, logit = projector(p4)
            # 将 logit 上采样到 GT 分辨率 | Upsample logit to GT resolution
            logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                     mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(logit_up.squeeze(), gt_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        # Eval
        projector.eval()
        dices = []
        with torch.no_grad():
            for idx in range(len(val_ds)):
                sample = val_ds[idx]
                image = sample["image"].unsqueeze(0).to(device)
                gt_mask = sample["masks"].to(device)
                if gt_mask.dim() == 3:
                    gt_mask = gt_mask.squeeze(0)
                elif gt_mask.dim() == 4:
                    gt_mask = gt_mask.squeeze(0).squeeze(0)

                features = backbone(image)
                _, logit = projector(features["p4"])
                logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                         mode="bilinear", align_corners=False)
                pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)  # [1, H, W]
                dices.append(compute_dice(pred, gt_mask.unsqueeze(0)).item())

        dice_mean = float(np.mean(dices))
        recorder.record_metric("loss/train", avg_loss, step=epoch, phase="train")
        recorder.record_metric("dice/val", dice_mean, step=epoch, phase="val")

        if dice_mean > best_dice:
            best_dice = dice_mean
            best_state = {k: v.clone() for k, v in projector.state_dict().items()}

        print(f"    loss={avg_loss:.4f}  Dice={dice_mean:.4f}"
              f"{' *' if dice_mean == best_dice else ''}")

    projector.load_state_dict(best_state)
    print(f"  Best val Dice: {best_dice:.4f}")
    return best_dice


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Extract Embeddings & Cluster Analysis (复用 E006 逻辑)
# ═══════════════════════════════════════════════════════════════════

def extract_embeddings(projector, backbone, dataset, max_samples, max_pixels, device, embed_dim):
    """提取训练后的嵌入 | Extract trained embeddings."""
    all_building = []
    all_background = []

    n_images = min(len(dataset), max_samples)
    indices = np.linspace(0, len(dataset) - 1, n_images, dtype=int)

    projector.eval()
    for idx in tqdm(indices, desc="  Extracting embeddings"):
        sample = dataset[int(idx)]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        with torch.no_grad():
            features = backbone(image)
            embedding, _ = projector(features["p4"])  # [1, D, H/16, W/16]

        gt_p4 = F.interpolate(
            gt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=embedding.shape[2:], mode="nearest",
        ).squeeze(0).squeeze(0)

        emb_flat = embedding.squeeze(0).reshape(embed_dim, -1).permute(1, 0)
        gt_flat = gt_p4.reshape(-1).long()

        building_idx = (gt_flat == 1).nonzero(as_tuple=True)[0]
        background_idx = (gt_flat == 0).nonzero(as_tuple=True)[0]

        if len(building_idx) > max_pixels:
            building_idx = building_idx[torch.randperm(len(building_idx))[:max_pixels]]
        if len(background_idx) > max_pixels:
            background_idx = background_idx[torch.randperm(len(background_idx))[:max_pixels]]

        if len(building_idx) > 0:
            all_building.append(emb_flat[building_idx].cpu().numpy())
        if len(background_idx) > 0:
            all_background.append(emb_flat[background_idx].cpu().numpy())

    features_b = np.concatenate(all_building, axis=0) if all_building else np.array([])
    features_bg = np.concatenate(all_background, axis=0) if all_background else np.array([])
    return features_b, features_bg


def cluster_and_evaluate(features_b, features_bg, n_clusters, recorder):
    """K-Means + Silhouette + Cosine Δ. (复用 E006 逻辑 | Reuse E006 logic)."""
    results = {}

    # Per-class clustering
    for name, feats in [("Building", features_b), ("Background", features_bg)]:
        if len(feats) < n_clusters:
            print(f"  [{name}] Insufficient samples ({len(feats)})")
            results[f"sil_{name.lower()}"] = 0.0
            continue

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feats)
        n_eval = min(10000, len(feats))
        idx = np.random.choice(len(feats), n_eval, replace=False)
        sil = silhouette_score(feats[idx], labels[idx])
        results[f"sil_{name.lower()}"] = sil

        print(f"  [{name}] K-Means (k={n_clusters}): Silhouette = {sil:.4f}")

    # Global mixed clustering
    features_all = np.concatenate([features_b, features_bg], axis=0)
    gt_all = np.concatenate([np.ones(len(features_b)), np.zeros(len(features_bg))])
    kmeans_global = KMeans(n_clusters=n_clusters * 2, random_state=42, n_init=10)
    labels_global = kmeans_global.fit_predict(features_all)
    proto_global = kmeans_global.cluster_centers_

    build_dominant = 0
    bg_dominant = 0
    for i in range(n_clusters * 2):
        mask = labels_global == i
        n_total = mask.sum()
        if n_total == 0:
            continue
        pct_build = (gt_all[mask] == 1).sum() / n_total
        if pct_build >= 0.7:
            build_dominant += 1
        if pct_build <= 0.3:
            bg_dominant += 1

    # Cosine similarity
    proto_norm = proto_global / (np.linalg.norm(proto_global, axis=1, keepdims=True) + 1e-8)
    feat_norm = features_all / (np.linalg.norm(features_all, axis=1, keepdims=True) + 1e-8)
    sim_matrix = np.dot(feat_norm, proto_norm.T)
    max_delta = 0.0
    for i in range(len(proto_global)):
        sim_bg = sim_matrix[gt_all == 0, i].mean() if (gt_all == 0).sum() > 0 else 0
        sim_build = sim_matrix[gt_all == 1, i].mean() if (gt_all == 1).sum() > 0 else 0
        max_delta = max(max_delta, sim_build - sim_bg)

    results["build_dominant"] = build_dominant
    results["bg_dominant"] = bg_dominant
    results["max_cos_delta"] = max_delta

    return results, features_all, gt_all, proto_global


def visualize_comparison(features_raw, features_learned, labels_raw, labels_learned,
                         tsne_samples, output_path):
    """并排对比 Raw P4 vs Learned Embedding 的 PCA/t-SNE."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for row, (feats, labels, title) in enumerate([
        (features_raw, labels_raw, "Raw P4 (1280-dim) — E006"),
        (features_learned, labels_learned, "Learned Embedding (64-dim) — E006.5"),
    ]):
        n_total = min(tsne_samples, len(feats))
        idx = np.random.choice(len(feats), n_total, replace=False)
        X = feats[idx]
        y = labels[idx]

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        ax = axes[row, 0]
        for cls, color, name in [(0, "tab:blue", "Background"), (1, "tab:red", "Building")]:
            mask = y == cls
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name,
                       alpha=0.3, s=3, rasterized=True)
        ax.set_title(f"{title}\nPCA (PC1={pca.explained_variance_ratio_[0]:.1%}, "
                     f"PC2={pca.explained_variance_ratio_[1]:.1%})")
        ax.legend(loc="upper right")

        # t-SNE
        n_tsne = min(2000, n_total)
        idx_tsne = np.random.choice(n_total, n_tsne, replace=False)
        X_tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                      max_iter=1000).fit_transform(X[idx_tsne])
        ax = axes[row, 1]
        for cls, color, name in [(0, "tab:blue", "Background"), (1, "tab:red", "Building")]:
            mask = y[idx_tsne] == cls
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color, label=name,
                       alpha=0.4, s=5, rasterized=True)
        ax.set_title(f"{title}\nt-SNE")
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Comparison visualization saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  E006.5: Learned Embedding — P4 → 64-dim 后可聚类?")
    print("  核心问题: 学习后的低维嵌入是否更适合 Prototype?")
    print("=" * 70)

    exp_id = generate_exp_id(name=args.name or "e006_5_embedding")
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)

    # ── [1/6] Backbone ──
    print("\n[1/6] Frozen FastSAM Backbone")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    # ── [2/6] Data ──
    print("\n[2/6] Load Data")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── [3/6] Embedding Projector + Train ──
    print(f"\n[3/6] Embedding Projector (1280 → {args.embed_dim} → 1)")
    projector = EmbeddingProjector(in_channels=1280, embed_dim=args.embed_dim).to(device)
    best_dice = train_projector(projector, backbone, train_ds, val_ds, args, device, recorder)
    recorder.logger.log_info("e006.5/train", f"best_dice={best_dice:.4f}")

    # ── [4/6] Extract Raw P4 Features (baseline, 复用 E006 逻辑) ──
    print("\n[4/6] Extract Raw P4 Features (1280-dim, for comparison)")
    from tools.eval_e006_proto_analysis import extract_p4_features as extract_p4
    # Run inline to avoid import complexity
    features_raw_b, features_raw_bg = extract_p4_raw(
        backbone, train_ds, args.max_samples, args.max_pixels_per_image, device
    )
    print(f"  Raw P4: Building={features_raw_b.shape[0]:,}, BG={features_raw_bg.shape[0]:,}")

    # ── [5/6] Extract Learned Embeddings ──
    print(f"\n[5/6] Extract Learned Embeddings ({args.embed_dim}-dim)")
    features_learned_b, features_learned_bg = extract_embeddings(
        projector, backbone, train_ds, args.max_samples, args.max_pixels_per_image, device,
        args.embed_dim
    )
    print(f"  Learned: Building={features_learned_b.shape[0]:,}, BG={features_learned_bg.shape[0]:,}")

    # ── [6/6] Clustering Comparison ──
    print(f"\n[6/6] Clustering: Raw P4 vs Learned Embedding")
    print(f"  {'─' * 55}")

    # Raw P4 clustering
    print("\n  >>> Raw P4 (1280-dim) — E006 Baseline <<<")
    results_raw, feats_raw_all, labels_raw_all, proto_raw = cluster_and_evaluate(
        features_raw_b, features_raw_bg, args.n_clusters, recorder
    )

    # Learned embedding clustering
    print(f"\n  >>> Learned Embedding ({args.embed_dim}-dim) — E006.5 <<<")
    results_learned, feats_learned_all, labels_learned_all, proto_learned = cluster_and_evaluate(
        features_learned_b, features_learned_bg, args.n_clusters, recorder
    )

    # ── Comparison Summary ──
    sil_raw_b = results_raw.get("sil_building", 0)
    sil_raw_bg = results_raw.get("sil_background", 0)
    sil_learned_b = results_learned.get("sil_building", 0)
    sil_learned_bg = results_learned.get("sil_background", 0)

    delta_sil_b = sil_learned_b - sil_raw_b
    delta_sil_bg = sil_learned_bg - sil_raw_bg

    print(f"\n{'=' * 70}")
    print(f"  E006.5 结果 | Results: Raw P4 vs Learned Embedding")
    print(f"  {'─' * 55}")
    print(f"  {'Metric':<30} {'Raw P4':>10} {'Learned':>10} {'Δ':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Building Silhouette':<30} {sil_raw_b:>10.4f} {sil_learned_b:>10.4f} {delta_sil_b:>+10.4f}")
    print(f"  {'Background Silhouette':<30} {sil_raw_bg:>10.4f} {sil_learned_bg:>10.4f} {delta_sil_bg:>+10.4f}")
    print(f"  {'Building-dominant clusters':<30} {results_raw.get('build_dominant',0):>10} {results_learned.get('build_dominant',0):>10}")
    print(f"  {'BG-dominant clusters':<30} {results_raw.get('bg_dominant',0):>10} {results_learned.get('bg_dominant',0):>10}")
    print(f"  {'Max Cosine Δ (Build-BG)':<30} {results_raw.get('max_cos_delta',0):>10.4f} {results_learned.get('max_cos_delta',0):>10.4f}")
    print(f"  {'─'*55}")
    print(f"  Embedding Dice (val): {best_dice:.4f}  (LinearProbe baseline: 0.40)")

    print(f"  {'─'*55}")

    # 判决 | Verdict
    if sil_learned_b > 0.20:
        print(f"  ✅ Silhouette {sil_raw_b:.3f} → {sil_learned_b:.3f}，聚类结构显著改善！")
        print(f"     Learned Embedding 是 Prototype-Friendly 的。")
        print(f"     → Raw P4 不可聚类 → Learned Embedding 可聚类 → Proto/SPM 有理论依据")
        verdict = "strong_improvement"
    elif sil_learned_b > 0.10:
        print(f"  △ Silhouette {sil_raw_b:.3f} → {sil_learned_b:.3f}，有改善但不够显著。")
        print(f"     → 需要更强的投影（更多层、更高维）或更多训练。")
        verdict = "moderate_improvement"
    elif delta_sil_b > 0.03:
        print(f"  △ Silhouette 改善 {delta_sil_b:+.3f}，微弱但方向正确。")
        print(f"     → 方向对，但当前投影太弱。")
        verdict = "weak_improvement"
    else:
        print(f"  → Silhouette 无显著改善 ({delta_sil_b:+.3f})。")
        print(f"     → 简单的 1×1 Conv 投影不足以改变聚类结构。")
        print(f"     → 需要 SPM 级别的学习来重排特征空间。")
        verdict = "no_improvement"

    print(f"{'=' * 70}")

    # ── Visualization ──
    print("\n  Generating comparison visualization...")
    vis_path = str(output_path / "embedding_comparison.png")
    visualize_comparison(feats_raw_all, feats_learned_all,
                         labels_raw_all, labels_learned_all,
                         args.tsne_samples, vis_path)

    # ── Record ──
    recorder.record_metric("e006.5/sil_raw_building", sil_raw_b, phase="val", tags=["e006.5", "summary"])
    recorder.record_metric("e006.5/sil_learned_building", sil_learned_b, phase="val", tags=["e006.5", "summary"])
    recorder.record_metric("e006.5/delta_sil_building", delta_sil_b, phase="val", tags=["e006.5", "summary"])
    recorder.record_metric("e006.5/embedding_dice", best_dice, phase="val", tags=["e006.5", "summary"])
    recorder.logger.log_info("e006.5/verdict", verdict, tags=["e006.5", "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {output_path}/")


# ═══════════════════════════════════════════════════════════════════
# Raw P4 extraction helper (inline to avoid import issues)
# ═══════════════════════════════════════════════════════════════════

def extract_p4_raw(backbone, dataset, max_samples, max_pixels, device):
    """Extract raw P4 features split by GT mask (same logic as E006)."""
    all_building = []
    all_background = []
    n_images = min(len(dataset), max_samples)
    indices = np.linspace(0, len(dataset) - 1, n_images, dtype=int)

    for idx in tqdm(indices, desc="  Extracting raw P4"):
        sample = dataset[int(idx)]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        with torch.no_grad():
            features = backbone(image)
            p4 = features["p4"]

        gt_p4 = F.interpolate(
            gt_mask.unsqueeze(0).unsqueeze(0).float(),
            size=p4.shape[2:], mode="nearest",
        ).squeeze(0).squeeze(0)

        p4_flat = p4.squeeze(0).reshape(1280, -1).permute(1, 0)
        gt_flat = gt_p4.reshape(-1).long()

        building_idx = (gt_flat == 1).nonzero(as_tuple=True)[0]
        background_idx = (gt_flat == 0).nonzero(as_tuple=True)[0]

        if len(building_idx) > max_pixels:
            building_idx = building_idx[torch.randperm(len(building_idx))[:max_pixels]]
        if len(background_idx) > max_pixels:
            background_idx = background_idx[torch.randperm(len(background_idx))[:max_pixels]]

        if len(building_idx) > 0:
            all_building.append(p4_flat[building_idx].cpu().numpy())
        if len(background_idx) > 0:
            all_background.append(p4_flat[background_idx].cpu().numpy())

    features_b = np.concatenate(all_building, axis=0) if all_building else np.array([])
    features_bg = np.concatenate(all_background, axis=0) if all_background else np.array([])
    return features_b, features_bg


if __name__ == "__main__":
    main()
