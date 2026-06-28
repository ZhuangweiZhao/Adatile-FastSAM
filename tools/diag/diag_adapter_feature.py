#!/usr/bin/env python3
"""
Adapter Feature Space 对比实验 | Adapter Feature Space Comparison Experiment.
=============================================================================
验证核心假设: ConvAdapter 能否降低 intra-class variance，改善 Silhouette/DB？

Validate core hypothesis: Can ConvAdapter reduce intra-class variance
and improve Silhouette/DB in FastSAM P4 feature space?

实验设计 | Experiment Design:
    1. 提取原始 FastSAM P4 特征 → 测量 Silhouette/DB (Baseline)
    2. 在 P4 后插入 ConvAdapter (channel attention + residual)
    3. 用监督对比损失训练 Adapter (frozen backbone)
    4. 提取 Adapter 后的 P4 特征 → 重新测量 Silhouette/DB
    5. 对比: ΔSilhouette, ΔDB, ΔIntra-dist

训练目标 | Training Objective:
    - 最小化类内 cosine 距离 (同类点聚集)
    - 最大化类间 cosine 距离 (异类点推远)
    - 使用 supervised contrastive-like loss

用法 | Usage:
    python tools/diag/diag_adapter_feature.py \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --fold 0 --adapter-epochs 10 --device cuda

输出 | Output:
    - runs/diag_adapter/{timestamp}/comparison.json  — Before/After 定量对比
    - runs/diag_adapter/{timestamp}/tsne_before.png   — 原始特征 t-SNE
    - runs/diag_adapter/{timestamp}/tsne_after.png    — Adapter 后 t-SNE
"""

import sys, json, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
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

# ═══════════════════════════════════════════════════════════════════
# 复用 Feature Space 诊断的核心函数 | Reuse core functions from diag_feature_space
# ═══════════════════════════════════════════════════════════════════

from tools.diag.diag_feature_space import (
    collect_class_features, compute_class_statistics,
    compute_pairwise_metrics, compute_base_novel_gap,
    silhouette_score_custom, davies_bouldin_index,
    plot_tsne, plot_similarity_matrix,
)


# ═══════════════════════════════════════════════════════════════════
# Lightweight Feature Adapter | 轻量特征适配器
# ═══════════════════════════════════════════════════════════════════

class P4FeatureAdapter(nn.Module):
    """
    轻量 P4 特征适配器: 1×1 Conv → Channel Attention → Residual.
    Lightweight P4 feature adapter.

    设计原则 | Design principles:
    - 保持维度不变 (1280 → 1280)
    - 少量参数 (避免过拟合)
    - 残差连接 (保护原始特征中的有用信息)
    """

    def __init__(self, in_channels: int = 1280, reduction: int = 8):
        super().__init__()
        mid = in_channels // reduction  # 160

        # Channel attention (SE-style)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # 1×1 projection with small bottleneck
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )

        # Learnable residual weight (starts near 0.1 — adapter is subtle at first)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # Channel attention
        attn = self.channel_attn(x)
        x_attn = x * attn

        # Projection
        x_proj = self.proj(x_attn)

        # Residual connection with learnable weight
        return x + self.alpha * x_proj


class P3P4Adapter(nn.Module):
    """P3 + P4 特征适配器 | P3 + P4 feature adapter."""

    def __init__(self, p3_dim: int = 960, p4_dim: int = 1280):
        super().__init__()
        self.p3_adapter = P4FeatureAdapter(p3_dim) if p3_dim else None
        self.p4_adapter = P4FeatureAdapter(p4_dim)

    def forward(self, feats: dict) -> dict:
        out = dict(feats)
        if self.p3_adapter is not None and "p3" in feats:
            out["p3"] = self.p3_adapter(feats["p3"])
        if "p4" in feats:
            out["p4"] = self.p4_adapter(feats["p4"])
        return out


# ═══════════════════════════════════════════════════════════════════
# Simple Supervised Contrastive Training | 简单监督对比训练
# ═══════════════════════════════════════════════════════════════════

def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor,
                                 temperature: float = 0.07) -> torch.Tensor:
    """
    监督对比损失 — 让同类点靠近、异类点远离。
    Supervised contrastive loss — pull same-class points together, push others apart.

    :param features: [N, C] L2-normalized feature vectors
    :param labels: [N] class labels
    :param temperature: softmax temperature
    :return: scalar loss
    """
    N = features.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=features.device)

    # Cosine similarity matrix [N, N]
    sim = features @ features.T  # already L2-normalized

    # Positive mask: same class, exclude self
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~torch.eye(
        N, dtype=torch.bool, device=features.device)

    # For each anchor, compute: -log( Σ_pos exp(sim/τ) / Σ_all exp(sim/τ) )
    sim = sim / temperature

    # Numerical stability: subtract max per row
    sim_max = sim.max(dim=1, keepdim=True)[0].detach()
    sim = sim - sim_max

    exp_sim = torch.exp(sim)

    # Sum of positives per row
    pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
    # Sum of all (excluding self)
    all_sum = exp_sim.sum(dim=1) - torch.exp(sim.diag())  # remove self

    # Avoid division by zero
    valid = pos_sum > 1e-8
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)

    loss = -torch.log(pos_sum[valid] / all_sum[valid].clamp(min=1e-8))
    return loss.mean()


def train_adapter(
    backbone, adapter, class_features: dict, class_ids: list[int],
    epochs: int, device: torch.device, lr: float = 0.001,
) -> dict:
    """
    用监督对比损失训练 Adapter。| Train adapter with supervised contrastive loss.

    每轮: 从各类随机采样 N 个点 → Adapter → L2-norm → 计算 SupCon loss → backward.
    """
    adapter.train()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Pre-compute per-class features on CPU for fast sampling
    class_vectors = {}
    for cls_id in class_ids:
        if cls_id not in class_features:
            continue
        class_vectors[cls_id] = class_features[cls_id]["features"]  # numpy [N, C]

    n_per_class = 64  # samples per class per batch
    history = {"loss": [], "intra_dist": [], "inter_dist": []}

    print(f"\n  Training adapter ({epochs} epochs, {len(class_ids)} classes)...")

    for epoch in range(epochs):
        # Sample balanced batch
        batch_feats = []
        batch_labels = []
        available = sorted(class_vectors.keys())
        rng = np.random.RandomState(epoch)

        for cls_id in available:
            vecs = class_vectors[cls_id]
            indices = rng.choice(len(vecs), min(n_per_class, len(vecs)), replace=False)
            batch_feats.append(torch.from_numpy(vecs[indices]).float().to(device))
            batch_labels.append(torch.full((len(indices),), cls_id, dtype=torch.long, device=device))

        x = torch.cat(batch_feats, dim=0)  # [N, 1280]
        labels = torch.cat(batch_labels, dim=0)

        # Forward through adapter
        # Adapter expects [N, C, 1, 1] format (avg-pooled P4 features)
        x_2d = x.unsqueeze(-1).unsqueeze(-1)  # [N, 1280, 1, 1]
        x_adapted = adapter(x_2d).squeeze(-1).squeeze(-1)  # [N, 1280]

        # L2 normalize
        x_norm = F.normalize(x_adapted, p=2, dim=1)

        # Supervised contrastive loss
        loss = supervised_contrastive_loss(x_norm, labels, temperature=0.1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        history["loss"].append(round(loss.item(), 6))

        # Evaluate intra/inter dist every 2 epochs
        if epoch % 2 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                # Per-class intra distance
                intra_dists = []
                for cls_id in available:
                    vecs = class_vectors[cls_id]
                    idx = rng.choice(len(vecs), min(32, len(vecs)), replace=False)
                    x_cls = torch.from_numpy(vecs[idx]).float().to(device)
                    x_cls_2d = x_cls.unsqueeze(-1).unsqueeze(-1)
                    x_cls_adapted = adapter(x_cls_2d).squeeze(-1).squeeze(-1)
                    x_cls_norm = F.normalize(x_cls_adapted, p=2, dim=1)
                    centroid = F.normalize(x_cls_norm.mean(0), p=2, dim=0)
                    sims = (x_cls_norm @ centroid).clamp(-1, 1)
                    intra_dists.append((1.0 - sims).mean().item())

                history["intra_dist"].append(round(np.mean(intra_dists), 4))

                # Mean inter-class distance
                centroids = {}
                for cls_id in available:
                    vecs = class_vectors[cls_id]
                    idx = rng.choice(len(vecs), min(32, len(vecs)), replace=False)
                    x_cls = torch.from_numpy(vecs[idx]).float().to(device)
                    x_2d = x_cls.unsqueeze(-1).unsqueeze(-1)
                    x_adapted = adapter(x_2d).squeeze(-1).squeeze(-1)
                    centroids[cls_id] = F.normalize(x_adapted.mean(0), p=2, dim=0)

                inter_dists = []
                cls_list = sorted(centroids.keys())
                for i in range(len(cls_list)):
                    for j in range(i + 1, len(cls_list)):
                        d = 1.0 - centroids[cls_list[i]].dot(centroids[cls_list[j]]).clamp(-1, 1).item()
                        inter_dists.append(d)
                history["inter_dist"].append(round(np.mean(inter_dists), 4))

            print(f"    E{epoch:3d}: loss={loss.item():.6f}  "
                  f"intra={history['intra_dist'][-1]:.4f}  "
                  f"inter={history['inter_dist'][-1]:.4f}")

    return history


# ═══════════════════════════════════════════════════════════════════
# Extract adapted features | 提取 Adapter 后的特征
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_adapted_features(
    tile_root: str, class_ids: list[int], max_per_class: int,
    backbone, adapter, device: torch.device, split: str = "val",
) -> dict:
    """
    与 collect_class_features 相同，但通过 adapter 变换特征。
    Same as collect_class_features, but transforms features through adapter.
    """
    from tools.train.train_fewshot import PreCutTileAdapter

    tile_adapter = PreCutTileAdapter(tile_root, split)
    results = {}

    for cls_id in tqdm(class_ids, desc=f"  [{split}] Adapted features"):
        candidates = tile_adapter.class_to_images(cls_id)
        if not candidates:
            continue

        all_feats = []
        n_tiles_used = 0
        rng = np.random.RandomState(42)
        tile_order = rng.permutation(candidates)

        for tile_idx in tile_order:
            if len(all_feats) >= max_per_class:
                break

            img_tensor = tile_adapter.load_image(int(tile_idx)).unsqueeze(0).to(device)
            mask_tensor = tile_adapter.render_class_mask(int(tile_idx), cls_id)
            if mask_tensor.sum() < 16:
                continue

            feats = backbone(img_tensor)
            # Apply adapter
            feats = adapter(feats)
            p4 = feats["p4"]

            mask_p4 = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0).float(),
                size=p4.shape[2:], mode="nearest"
            ).squeeze() > 0.5

            if mask_p4.sum() < 4:
                continue

            fg_vectors = p4[0, :, mask_p4].permute(1, 0).cpu().numpy()
            all_feats.append(fg_vectors)
            n_tiles_used += 1

        if not all_feats:
            continue

        combined = np.concatenate(all_feats, axis=0)
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
# Run full diagnosis | 运行完整诊断
# ═══════════════════════════════════════════════════════════════════

def run_diagnosis(features_dict: dict, base_ids: list, novel_ids: list,
                  cat_names: dict, label: str):
    """运行完整特征空间诊断并返回指标 | Run full feature space diagnosis."""
    valid_ids = sorted(features_dict.keys())
    stats = compute_class_statistics(features_dict)
    pairwise = compute_pairwise_metrics(stats, valid_ids)

    base_stats = {k: v for k, v in stats.items() if k in base_ids}
    novel_stats = {k: v for k, v in stats.items() if k in novel_ids}
    gap = compute_base_novel_gap(base_stats, novel_stats)

    # Silhouette + DB
    all_feats_list, all_labels_list = [], []
    for cls_id in valid_ids:
        feats = features_dict[cls_id]["features"]
        all_feats_list.append(feats)
        all_labels_list.extend([cls_id] * len(feats))
    X_all = np.concatenate(all_feats_list, axis=0)
    y_all = np.array(all_labels_list)

    silhouette = silhouette_score_custom(X_all, y_all, n_samples=5000)
    db_index = davies_bouldin_index(X_all, y_all)

    # Per-class metrics
    per_class = {}
    for cls_id in valid_ids:
        s = stats[cls_id]
        per_class[str(cls_id)] = {
            "name": cat_names.get(cls_id, f"c{cls_id}"),
            "type": "NOVEL" if cls_id in novel_ids else "BASE",
            "n_vectors": s["n_vectors"],
            "intra_dist": s["intra_dist"],
            "silhouette": silhouette["per_class"].get(cls_id, 0),
        }

    return {
        "label": label,
        "n_classes": len(valid_ids),
        "silhouette": silhouette["overall"],
        "db_index": db_index,
        "intra_dist_mean": round(np.mean([s["intra_dist"] for s in stats.values()]), 4),
        "inter_dist_mean": pairwise["mean_inter_dist"],
        "base_intra": gap["base_intra_dist_mean"],
        "novel_intra": gap["novel_intra_dist_mean"],
        "per_class": per_class,
    }


# ═══════════════════════════════════════════════════════════════════
# Main | 主逻辑
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Adapter Feature Space Comparison")
    p.add_argument("--tile-root", type=str, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--max-samples-per-class", type=int, default=500)
    p.add_argument("--adapter-epochs", type=int, default=10,
                   help="Adapter training epochs (10-20 recommended)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="runs/diag_adapter")
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
    cat_names = ISAID5I_CATEGORIES

    print(f"\n{'='*70}")
    print(f"  Adapter Feature Space Comparison | Adapter 特征空间对比")
    print(f"  {'─'*60}")
    print(f"  Tile root:          {args.tile_root}")
    print(f"  Fold:               {args.fold}")
    print(f"  Adapter epochs:     {args.adapter_epochs}")
    print(f"  Base: {len(base_classes)} classes, Novel: {len(novel_classes)} classes")
    print(f"  Output:             {out_dir}")
    print(f"{'='*70}\n")

    # ── [1] Baseline: 提取原始特征 | Extract raw features ──
    print("[1/5] Baseline: Extracting raw FastSAM P4 features...")
    backbone = FastSAMBackbone(freeze_backbone=True).to(device).eval()

    raw_features = collect_class_features(
        args.tile_root, all_classes, args.max_samples_per_class,
        backbone, device, split="val",
    )
    print(f"      Valid classes: {len(raw_features)}")

    # ── [2] Baseline 诊断 | Baseline diagnosis ──
    print("\n[2/5] Baseline: Running feature space diagnosis...")
    baseline_result = run_diagnosis(
        raw_features, base_classes, novel_classes, cat_names, "Baseline (Raw FastSAM P4)"
    )

    # ── [3] 训练 Adapter | Train adapter ──
    print(f"\n[3/5] Training P4 Feature Adapter...")

    # Probe backbone for P3/P4 dims
    with torch.no_grad():
        probe = backbone(torch.randn(1, 3, 896, 896).to(device))
        p3_dim = probe["p3"].shape[1]
        p4_dim = probe["p4"].shape[1]
    print(f"      P3 dim: {p3_dim}, P4 dim: {p4_dim}")

    adapter = P3P4Adapter(p3_dim=p3_dim, p4_dim=p4_dim).to(device)
    n_params = sum(p.numel() for p in adapter.parameters())
    n_trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"      Adapter: {n_params:,} params ({n_trainable:,} trainable)")

    train_history = train_adapter(
        backbone, adapter.p4_adapter, raw_features, all_classes,
        epochs=args.adapter_epochs, device=device,
    )

    # ── [4] Adapter 后诊断 | Post-adapter diagnosis ──
    print(f"\n[4/5] Post-adapter: Extracting adapted features...")
    adapter.eval()

    adapted_features = collect_adapted_features(
        args.tile_root, all_classes, args.max_samples_per_class,
        backbone, adapter, device, split="val",
    )
    print(f"      Valid classes: {len(adapted_features)}")

    print("\n      Running feature space diagnosis...")
    adapted_result = run_diagnosis(
        adapted_features, base_classes, novel_classes, cat_names,
        "Adapter (Post-Adaptation)"
    )

    # ── [5] 对比输出 | Comparison output ──
    print(f"\n{'='*70}")
    print(f"  BEFORE → AFTER COMPARISON | 对比结果")
    print(f"{'='*70}")

    metrics = [
        ("Silhouette Score", "silhouette", ">0.25 good, <0.1 random", True),
        ("Davies-Bouldin", "db_index", "<1.0 good, >2.0 poor", False),
        ("Intra-Class Dist", "intra_dist_mean", "lower = more compact", False),
        ("Inter-Class Dist", "inter_dist_mean", ">0.3 well separated", True),
        ("Base Intra Dist", "base_intra", "lower = more compact", False),
        ("Novel Intra Dist", "novel_intra", "lower = more compact", False),
    ]

    for name, key, note, higher_better in metrics:
        b = baseline_result[key]
        a = adapted_result[key]
        delta = a - b
        if higher_better:
            arrow = "↑" if delta > 0 else "↓"
        else:
            arrow = "↓" if delta < 0 else "↑"
            delta = -delta  # flip sign for display
        direction = "better" if (
            (higher_better and a > b) or (not higher_better and a < b)
        ) else "worse"
        print(f"  {name:<20s}: {b:>8.4f} → {a:>8.4f}  "
              f"({arrow}{abs(delta):.4f} {direction})  [{note}]")

    # Per-class intra dist comparison
    print(f"\n  ── Per-Class Intra Dist Change ──")
    print(f"  {'Class':<22s} {'Before':>8s} {'After':>8s} {'Δ':>8s}")
    print(f"  {'─'*48}")
    for cls_id in sorted(raw_features.keys()):
        if cls_id not in adapted_features:
            continue
        b_name = cat_names.get(cls_id, f"c{cls_id}")
        b_val = baseline_result["per_class"].get(str(cls_id), {}).get("intra_dist", 0)
        a_val = adapted_result["per_class"].get(str(cls_id), {}).get("intra_dist", 0)
        delta = a_val - b_val
        sign = "↓" if delta < 0 else "↑"
        print(f"  {b_name:<22s} {b_val:>8.4f} {a_val:>8.4f} {sign}{abs(delta):>7.4f}")

    # ── Verdict ──
    print(f"\n{'█'*70}")
    print(f"  VERDICT | 对比结论")
    print(f"{'█'*70}")

    sil_before = baseline_result["silhouette"]
    sil_after = adapted_result["silhouette"]
    db_before = baseline_result["db_index"]
    db_after = adapted_result["db_index"]

    if sil_after - sil_before > 0.05:
        print(f"  ✅ Adapter SIGNIFICANTLY improves Silhouette ({sil_before:.3f}→{sil_after:.3f})")
        print(f"     Feature space IS adaptable — FastSAM P4 can be reshaped for FSS.")
    elif sil_after - sil_before > 0.01:
        print(f"  🟡 Adapter MODERATELY improves Silhouette ({sil_before:.3f}→{sil_after:.3f})")
        print(f"     Marginal improvement — consider stronger adaptation (LoRA, more epochs).")
    else:
        print(f"  🔴 Adapter has MINIMAL effect on Silhouette ({sil_before:.3f}→{sil_after:.3f})")
        print(f"     Simple ConvAdapter insufficient — need stronger feature transformation.")
        print(f"     Options: (1) LoRA on backbone, (2) Cross-Attention bypass, (3) Deeper adapter")

    if db_after < db_before:
        print(f"  ✅ DB Index improved ({db_before:.2f}→{db_after:.2f})")
    print(f"{'█'*70}\n")

    # ── 保存结果 | Save results ──
    comparison = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "baseline": {k: v for k, v in baseline_result.items() if k != "per_class"},
        "adapted": {k: v for k, v in adapted_result.items() if k != "per_class"},
        "baseline_per_class": baseline_result["per_class"],
        "adapted_per_class": adapted_result["per_class"],
        "training_history": train_history,
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)

    # ── 可视化 | Visualizations ──
    print("\n[6] Generating visualizations...")

    plot_tsne(raw_features, cat_names, "FastSAM P4 — Before Adapter (Baseline)",
              str(out_dir / "tsne_before.png"), max_points=2000, novel_ids=novel_set)

    plot_tsne(adapted_features, cat_names, f"FastSAM P4 — After Adapter ({args.adapter_epochs} epochs)",
              str(out_dir / "tsne_after.png"), max_points=2000, novel_ids=novel_set)

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_history["loss"])
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("SupCon Loss")
    axes[1].plot(train_history["intra_dist"], label="Intra-class dist")
    axes[1].plot(train_history["inter_dist"], label="Inter-class dist")
    axes[1].set_title("Intra/Inter Class Distance")
    axes[1].set_xlabel("Eval step")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(str(out_dir / "training_curves.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved → {out_dir}/")
    print(f"\n{'='*70}")
    print(f"  ✅ Adapter comparison saved → {out_dir}/comparison.json")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
