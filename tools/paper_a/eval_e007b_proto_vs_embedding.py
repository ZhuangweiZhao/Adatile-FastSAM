#!/usr/bin/env python3
"""
E007-B: Proto vs Embedding 公平对照 | Fair Head-to-Head Comparison
===================================================================

核心问题 | Core question:
    固定 embedding 维度、参数量、训练配置的条件下，
    Proto Head 相比 Embedding Head 是否:
    1. 保持分割精度 (Dice 持平)?
    2. 构建更结构化的嵌入空间 (Silhouette 更高)?

设计原则 | Design principle:
    - 相同 Backbone (Frozen FastSAM)
    - 相同 Project 层 (1280→128 Conv+ReLU, bias=False)
    - 相同训练配置 (lr=1e-3, CosineLR, epochs=30, seed=42)
    - 仅 Head 机制不同：
        Embedding Head: 128-dim → Conv(128→1) → logit       (163,969 params)
        Proto Head:    128-dim → CosineSim(N protos) → Conv(N→1) → logit  (164,873 params)
    - Δ params = 904 (0.55%) — 可忽略

假设 | Hypothesis:
    H0: Proto 约束不损害 Dice，但显著提升 Silhouette
    H1: Proto 是构建可解释、可稀疏化表示空间的关键，不是提分工具

与 E007 的区别 | Diff from E007:
    E007:  Proto 绝对值 (Dice=0.459, 有语义分化)
    E007-B: Proto vs Embedding 对照 (证明语义分化来自 Proto 约束，非偶然)

用法 | Usage:
    python tools/eval_e007b_proto_vs_embedding.py
    python tools/eval_e007b_proto_vs_embedding.py --n-protos 8 --epochs 30
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.backbone import FastSAMBackbone
from adatile.metrics import compute_dice, format_param_count
from adatile.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="E007-B: Proto vs Embedding Fair Comparison")
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--n-protos", type=int, default=8,
                   help="Number of prototypes (Proto Head only)")
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="e007b")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42,
                   help="Fixed seed for reproducibility")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Head Definitions
# ═══════════════════════════════════════════════════════════════════

class EmbeddingHead(nn.Module):
    """
    嵌入头 | Embedding Head (Baseline).

    P4 → Conv(1280→128) → ReLU → Conv(128→1) → logit.

    无结构约束 | No structural constraint:
        128-dim embedding 直接通过 1×1 Conv 映射到 logit。
        128-dim embedding directly mapped to logit via 1×1 Conv.
    """

    def __init__(self, in_channels: int = 1280, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # 特征投影 | Feature projection (shared with ProtoHead)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 直接分割头 | Direct segmentation head
        self.head = nn.Conv2d(embed_dim, 1, kernel_size=1, bias=True)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  EmbeddingHead: {format_param_count(n_params)} ({n_params:,})")
        print(f"    project={sum(p.numel() for p in self.project.parameters()):,}")
        print(f"    head={sum(p.numel() for p in self.head.parameters()):,}")

    def forward(self, p4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            p4: [B, 1280, H/16, W/16]

        Returns:
            embedding: [B, D, H, W]  低维嵌入 | Low-dim embedding
            logit:     [B, 1, H, W]  分割 logit | Segmentation logit
        """
        embedding = self.project(p4)  # [B, D, H, W]
        logit = self.head(embedding)  # [B, 1, H, W]
        return embedding, logit


class ProtoHead(nn.Module):
    """
    原型头 | Proto Head (Treatment).

    P4 → Conv(1280→128) → ReLU → CosineSim(N protos) → Conv(N→1) → logit.

    结构约束 | Structural constraint:
        分割必须通过 N 个原型向量的 cosine similarity 完成。
        Segmentation must pass through cosine similarity to N prototypes.
    """

    def __init__(self, in_channels: int = 1280, embed_dim: int = 128, n_protos: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_protos = n_protos

        # 特征投影 | Feature projection (shared architecture with EmbeddingHead)
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 可学习原型向量 | Learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)

        # 分割头：原型响应的 1×1 线性组合 | Seg head: 1×1 linear combo
        self.head = nn.Conv2d(n_protos, 1, kernel_size=1, bias=True)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  ProtoHead: {format_param_count(n_params)} ({n_params:,})")
        print(f"    project={sum(p.numel() for p in self.project.parameters()):,}")
        print(f"    prototypes={self.prototypes.numel():,}")
        print(f"    head={sum(p.numel() for p in self.head.parameters()):,}")

    def forward(self, p4: torch.Tensor, temperature: float = 0.1
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            p4:          [B, 1280, H/16, W/16]
            temperature: softmax temperature

        Returns:
            embedding: [B, D, H, W]  低维嵌入 | Low-dim embedding
            sim_maps:  [B, N, H, W]  proto 相似度图 | Proto similarity maps
            logit:     [B, 1, H, W]  分割 logit | Segmentation logit
        """
        embedding = self.project(p4)  # [B, D, H, W]

        # L2 normalize for cosine similarity
        emb_norm = F.normalize(embedding, dim=1, p=2)          # [B, D, H, W]
        proto_norm = F.normalize(self.prototypes, dim=1, p=2)  # [N, D]

        # Cosine similarity: [B, D, H, W] × [N, D] → [B, N, H, W]
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_norm, proto_norm)
        sim_maps = sim_maps / temperature

        logit = self.head(sim_maps)  # [B, 1, H, W]
        return embedding, sim_maps, logit

    def get_hard_assignment(self, p4: torch.Tensor) -> torch.Tensor:
        """硬分配 | Hard assignment per pixel → prototype index [B, H, W]."""
        _, sim_maps, _ = self.forward(p4, temperature=0.01)
        return sim_maps.argmax(dim=1)


# ═══════════════════════════════════════════════════════════════════
# Silhouette Score Computation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_silhouette(embedding: torch.Tensor, gt_mask: torch.Tensor,
                       n_samples: int = 5000) -> float:
    """
    计算 Building vs Background 的 Silhouette Score。
    Compute Silhouette Score for Building vs Background clustering.

    对 embedding 空间中的像素随机采样，用 GT 标签评估聚类质量。
    Randomly sample pixels from embedding space, evaluate cluster
    quality using GT labels (building=1, background=0).

    Args:
        embedding: [1, D, H, W] 低维嵌入 | Low-dim embedding (pre-head)
        gt_mask:   [H, W]        二值 GT 掩码 | Binary GT mask
        n_samples: 采样像素数 | Number of pixels to sample

    Returns:
        Silhouette score ∈ [-1, 1]. Higher = better separated clusters.
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return float("nan")

    D = embedding.shape[1]

    # 下采样 GT 到 embedding 分辨率 (节省内存 | save memory)
    # Downsample GT to embedding resolution instead of upsampling embedding
    # (avoids ~1.15 GB tensor for 1500×1500 upsampled embedding)
    gt_down = F.interpolate(
        gt_mask.unsqueeze(0).unsqueeze(0).float(),
        size=(embedding.shape[2], embedding.shape[3]),
        mode="nearest",
    ).squeeze()  # [H_emb, W_emb]

    emb_sq = embedding.squeeze(0)  # [D, H_emb, W_emb]

    # 随机采样 | Random sampling
    gt_flat = gt_down.flatten()
    n_total = gt_flat.numel()
    idx = torch.randperm(n_total, device=embedding.device)[:n_samples]

    labels = gt_flat[idx].cpu().numpy()
    features = emb_sq.reshape(D, -1).T[idx].cpu().numpy()  # [n_samples, D]

    # 确保有两类 | Ensure both classes present
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return float("nan")

    return float(silhouette_score(features, labels))


@torch.no_grad()
def compute_val_silhouette(head, backbone, val_ds, device, n_samples: int = 5000,
                           is_proto: bool = True) -> float:
    """
    在整个 val 集上计算平均 Silhouette Score。
    Compute average Silhouette Score across val set.
    """
    head.eval()
    scores = []
    for idx in range(min(8, len(val_ds))):  # 8 张图足够估计 | 8 images enough for reliable estimate
        sample = val_ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        # 统一 mask 维度 | Normalize mask dimensions
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        features = backbone(image)
        if is_proto:
            embedding, _, _ = head(features["p4"], temperature=0.1)
        else:
            embedding, _ = head(features["p4"])

        s = compute_silhouette(embedding, gt_mask, n_samples=n_samples)
        if not np.isnan(s):
            scores.append(s)

    return float(np.mean(scores)) if scores else float("nan")


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_head(head, backbone, train_ds, val_ds, args, device, recorder,
               head_name: str, is_proto: bool) -> dict:
    """
    训练一个 Head | Train one head variant.

    Returns:
        results dict with keys: best_dice, final_dice, best_silhouette,
        final_silhouette, loss_history, dice_history
    """
    head.train()
    # 确保可训练参数仅来自 head | Ensure only head params are trainable
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    best_dice, best_state = 0.0, None
    best_sil = float("nan")
    loss_history, dice_history = [], []

    print(f"\n  [{head_name}] Training {args.epochs} epochs (lr={args.lr}, CosineLR)...")
    for epoch in range(1, args.epochs + 1):
        # ── 训练阶段 | Training phase ──
        head.train()
        total_loss = 0.0
        # 逐样本 episodic training | Per-sample episodic training (no batch accumulation)
        pbar = tqdm(range(len(train_ds)), desc=f"  [{head_name}] Epoch {epoch}/{args.epochs}",
                    leave=False)
        for idx in pbar:
            sample = train_ds[idx]
            image = sample["image"].unsqueeze(0).to(device)
            gt_mask = sample["masks"].to(device)
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.squeeze(0)
            elif gt_mask.dim() == 4:
                gt_mask = gt_mask.squeeze(0).squeeze(0)

            with torch.no_grad():
                features = backbone(image)
            p4 = features["p4"]

            if is_proto:
                _, _, logit = head(p4, temperature=args.temperature)
            else:
                _, logit = head(p4)

            # 上采样 + BCE 损失 | Upsample + BCE loss
            logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                     mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(logit_up.squeeze(), gt_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/(idx+1):.4f}"})
        scheduler.step()

        avg_loss = total_loss / len(train_ds)
        loss_history.append(avg_loss)

        # ── 验证 Dice | Validation Dice ──
        head.eval()
        dices = []
        with torch.no_grad():
            for idx in range(len(val_ds)):
                sample = val_ds[idx]
                image = sample["image"].unsqueeze(0).to(device)
                gt_mask = sample["masks"].to(device)
                # 统一 mask 维度 | Normalize mask dimensions (handle various shapes)
                if gt_mask.dim() == 3:
                    gt_mask = gt_mask.squeeze(0)
                elif gt_mask.dim() == 4:
                    gt_mask = gt_mask.squeeze(0).squeeze(0)
                features = backbone(image)
                if is_proto:
                    _, _, logit = head(features["p4"], temperature=args.temperature)
                else:
                    _, logit = head(features["p4"])
                # 上采样到 GT 分辨率 | Upsample to GT resolution
                logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                         mode="bilinear", align_corners=False)
                # 二值化预测 | Binarize prediction at threshold 0.5
                pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)
                # 确保 batch 维度 | Ensure batch dimension for compute_dice
                if pred.dim() == 2:
                    pred = pred.unsqueeze(0)
                if gt_mask.dim() == 2:
                    gt_mask = gt_mask.unsqueeze(0)
                dices.append(compute_dice(pred, gt_mask).item())

        dice_mean = float(np.mean(dices))
        dice_history.append(dice_mean)

        is_best = dice_mean > best_dice
        if is_best:
            best_dice = dice_mean
            best_state = {k: v.clone() for k, v in head.state_dict().items()}

        marker = " *" if is_best else ""
        print(f"    [{head_name}] Epoch {epoch:2d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  Dice(val)={dice_mean:.4f}{marker}")

        # Log to recorder
        recorder.record_metric(f"loss/train", avg_loss, step=epoch,
                               phase="train", tags=[head_name])
        recorder.record_metric(f"dice/val", dice_mean, step=epoch,
                               phase="val", tags=[head_name])

    # ── Load best state ──
    if best_state is not None:
        head.load_state_dict(best_state)

    # ── Val Silhouette (on best model) ──
    val_sil = compute_val_silhouette(head, backbone, val_ds, device,
                                     is_proto=is_proto)

    # ── Final epoch Dice (last state, for convergence check) ──
    final_dice = dice_history[-1] if dice_history else 0.0

    results = {
        "head_name": head_name,
        "n_params": sum(p.numel() for p in head.parameters()),
        "best_dice": best_dice,
        "final_dice": final_dice,
        "silhouette": val_sil,
        "best_epoch": dice_history.index(best_dice) + 1 if dice_history else 0,
        "loss_history": loss_history,
        "dice_history": dice_history,
    }
    return results


# ═══════════════════════════════════════════════════════════════════
# Proto Semantic Analysis (Proto Head only)
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_proto_semantics(proto_head, backbone, dataset, device, n_protos):
    """
    分析每个 Proto 的建筑/背景倾向 | Analyze per-proto building/background ratio.
    """
    proto_head.eval()
    proto_build_pct = np.zeros(n_protos)
    proto_activate_count = np.zeros(n_protos)

    for idx in range(min(20, len(dataset))):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        features = backbone(image)
        hard_assign = proto_head.get_hard_assignment(features["p4"])  # [1, H/16, W/16]
        hard_up = F.interpolate(
            hard_assign.unsqueeze(1).float(),
            size=tuple(gt_mask.shape), mode="nearest",
        ).squeeze(1).long().squeeze(0)

        for p in range(n_protos):
            proto_mask = (hard_up == p)
            n_pixels = proto_mask.sum().item()
            if n_pixels > 0:
                n_building = (gt_mask[proto_mask] == 1).sum().item()
                proto_build_pct[p] += n_building
                proto_activate_count[p] += n_pixels

    for p in range(n_protos):
        if proto_activate_count[p] > 0:
            proto_build_pct[p] /= proto_activate_count[p]

    return proto_build_pct, proto_activate_count


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device

    print("=" * 70)
    print(f"  E007-B: Proto vs Embedding 公平对照 | Fair Head-to-Head")
    print(f"  N_protos={args.n_protos}, D={args.embed_dim}, Epochs={args.epochs}")
    print(f"  Seed={args.seed}")
    print("  核心问题: Proto 是构建可解释空间的关键，还是单纯的提分工具?")
    print("=" * 70)

    # ── Set seed ──
    set_seed(args.seed)

    exp_id = generate_exp_id(name=args.name)
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings",
                              dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)

    # ── [1] Backbone ──
    print("\n[1] Frozen FastSAM Backbone")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    # ── [2] Data ──
    print("\n[2] Load Data")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── [3] Fair Comparison ──
    print(f"\n[3] Head-to-Head Comparison")
    print(f"  {'─'*50}")
    print(f"  Shared: Frozen FastSAM backbone")
    print(f"  Shared: Conv(1280→{args.embed_dim}) + ReLU (project layer, bias=False)")
    print(f"  Shared: lr={args.lr}, epochs={args.epochs}, CosineLR, seed={args.seed}")
    print(f"  Diff:   Classification mechanism only")
    print(f"  {'─'*50}")

    # ── A: Embedding Head ──
    print(f"\n  [3A] Embedding Head (Baseline) — no structural constraint")
    set_seed(args.seed)
    embed_head = EmbeddingHead(in_channels=1280, embed_dim=args.embed_dim).to(device)

    t0 = time.time()
    embed_results = train_head(
        embed_head, backbone, train_ds, val_ds, args, device, recorder,
        head_name="Embedding", is_proto=False
    )
    embed_time = time.time() - t0

    # ── B: Proto Head ──
    print(f"\n  [3B] Proto Head (Treatment) — cosine similarity constraints")
    set_seed(args.seed)
    proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                            n_protos=args.n_protos).to(device)

    t0 = time.time()
    proto_results = train_head(
        proto_head, backbone, train_ds, val_ds, args, device, recorder,
        head_name="Proto", is_proto=True
    )
    proto_time = time.time() - t0

    # ── Proto Semantic Analysis ──
    proto_build_pct, proto_activate_count = analyze_proto_semantics(
        proto_head, backbone, val_ds, device, args.n_protos
    )
    n_build = sum(1 for p in range(args.n_protos) if proto_build_pct[p] > 0.5)
    n_bg = sum(1 for p in range(args.n_protos) if proto_build_pct[p] < 0.3)
    n_mixed = args.n_protos - n_build - n_bg

    # ── [4] Summary ──
    print(f"\n{'=' * 70}")
    print(f"  E007-B 结果 | Results: Proto vs Embedding Fair Comparison")
    print(f"  {'=' * 70}")
    print(f"  {'Metric':<30} {'Embedding':>12} {'Proto':>12} {'Δ':>12}")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'Parameters':<30} {embed_results['n_params']:>12,} "
          f"{proto_results['n_params']:>12,} "
          f"{proto_results['n_params'] - embed_results['n_params']:>+12,}")
    print(f"  {'Best Dice (val)':<30} {embed_results['best_dice']:>12.4f} "
          f"{proto_results['best_dice']:>12.4f} "
          f"{proto_results['best_dice'] - embed_results['best_dice']:>+12.4f}")
    print(f"  {'Final Dice (val)':<30} {embed_results['final_dice']:>12.4f} "
          f"{proto_results['final_dice']:>12.4f} "
          f"{proto_results['final_dice'] - embed_results['final_dice']:>+12.4f}")
    print(f"  {'Silhouette Score':<30} "
          f"{embed_results['silhouette']:>12.4f} "
          f"{proto_results['silhouette']:>12.4f} "
          f"{proto_results['silhouette'] - embed_results['silhouette']:>+12.4f}")
    print(f"  {'Best Epoch':<30} {embed_results['best_epoch']:>12} "
          f"{proto_results['best_epoch']:>12}")
    print(f"  {'Training Time (s)':<30} {embed_time:>11.1f}s "
          f"{proto_time:>11.1f}s")
    print(f"  {'─'*30} {'─'*12} {'─'*12} {'─'*12}")
    print(f"\n  Proto 语义分析 | Proto Semantic Analysis ({args.n_protos} protos):")
    print(f"  {'Proto':<8} {'Build%':>10} {'Pixels':>12} {'Semantic':>12}")
    print(f"  {'─'*8} {'─'*10} {'─'*12} {'─'*12}")
    for p in range(args.n_protos):
        semantic = ("Building" if proto_build_pct[p] > 0.5 else
                    ("Background" if proto_build_pct[p] < 0.3 else "Mixed"))
        print(f"  P{p:<7} {proto_build_pct[p]:>10.1%} "
              f"{proto_activate_count[p]:>12,.0f} {semantic:>12}")
    print(f"  {'─'*8} {'─'*10} {'─'*12} {'─'*12}")
    print(f"  Building: {n_build}  Background: {n_bg}  Mixed: {n_mixed}")

    # ── 结论 | Conclusion ──
    delta_dice = proto_results['best_dice'] - embed_results['best_dice']
    delta_sil = proto_results['silhouette'] - embed_results['silhouette']

    print(f"\n  {'─'*60}")
    if abs(delta_dice) < 0.02 and delta_sil > 0.03:
        print(f"  ✅ PROTO HYPOTHESIS SUPPORTED")
        print(f"     Dice 持平 (|Δ|={abs(delta_dice):.3f} < 0.02)")
        print(f"     Silhouette 显著更高 (Δ=+{delta_sil:.3f})")
        print(f"     → Proto 不是为了提分，是为了构建可解释、可稀疏化的表示空间。")
        print(f"     → Proto is not about improving Dice — it builds an")
        print(f"       interpretable, sparsifiable representation space.")
        verdict = "proto_hypothesis_supported"
    elif delta_dice > 0.02 and delta_sil > 0.03:
        print(f"  ✅ PROTO WINS ON BOTH")
        print(f"     Dice 更高 (Δ=+{delta_dice:.4f})")
        print(f"     Silhouette 更高 (Δ=+{delta_sil:.4f})")
        print(f"     → Proto 既提分又结构化。")
        verdict = "proto_wins_both"
    elif abs(delta_dice) < 0.02 and abs(delta_sil) <= 0.03:
        print(f"  ⚠️  NO SIGNIFICANT DIFFERENCE")
        print(f"     Dice 持平 (|Δ|={abs(delta_dice):.3f})")
        print(f"     Silhouette 持平 (|Δ|={abs(delta_sil):.3f})")
        print(f"     → Proto 约束在 embedding=128 时无显著效果。")
        verdict = "no_difference"
    else:
        print(f"  → Mixed results. See detailed analysis above.")
        verdict = "mixed"
    print(f"  {'─'*60}")

    # ── Record ──
    recorder.record_metric("e007b/embed_dice", embed_results['best_dice'],
                           phase="val", tags=["e007b", "embedding"])
    recorder.record_metric("e007b/proto_dice", proto_results['best_dice'],
                           phase="val", tags=["e007b", "proto"])
    recorder.record_metric("e007b/embed_silhouette", embed_results['silhouette'],
                           phase="val", tags=["e007b", "embedding"])
    recorder.record_metric("e007b/proto_silhouette", proto_results['silhouette'],
                           phase="val", tags=["e007b", "proto"])
    recorder.record_metric("e007b/delta_dice", delta_dice,
                           phase="val", tags=["e007b", "delta"])
    recorder.record_metric("e007b/delta_silhouette", delta_sil,
                           phase="val", tags=["e007b", "delta"])
    recorder.logger.log_info("e007b/verdict", verdict, tags=["e007b", "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {output_path}/")

    return embed_results, proto_results


if __name__ == "__main__":
    main()
