#!/usr/bin/env python3
"""
E008: SPM Sparsity Validation — Proto 激活是否天然稀疏？
==========================================================

三个子实验 | Three sub-experiments:

  E008-A (mode=post-hoc-sim):
    Train with ALL protos, test Top-K by raw similarity.
    → 已跑，Dice 崩塌。证明：sim 大小 ≠ 重要性。

  E008-B (mode=post-hoc-weighted):
    Train with ALL protos, test Top-K by |w * sim| (head-weighted importance).
    → 排序指标从 sim 改为 head 学习的贡献度。
    → 预期：K=2 Dice ≈ 0.44, K=4 Dice ≈ 0.46。

  E008-C (mode=train-sparse):
    Train AND test with Top-K (head-weighted).
    → 消除 distribution shift。Head 和 Protos 都适配稀疏推理。
    → 预期：K=2 Dice 接近全量。

用法 | Usage:
    python tools/eval_e008_spm_sparsity.py                          # E008-A (default)
    python tools/eval_e008_spm_sparsity.py --mode post-hoc-weighted  # E008-B
    python tools/eval_e008_spm_sparsity.py --mode train-sparse --train-k 2  # E008-C
"""

from __future__ import annotations
import sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.backbone import FastSAMBackbone
from adatile.metrics import compute_dice


def parse_args():
    p = argparse.ArgumentParser(description="E008: SPM Sparsity Validation")
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--n-protos", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--mode", type=str, default="post-hoc-sim",
                   choices=["post-hoc-sim", "post-hoc-weighted", "train-sparse"],
                   help="post-hoc-sim: train full, test top-K by sim (E008-A) | "
                        "post-hoc-weighted: train full, test top-K by |w*sim| (E008-B) | "
                        "train-sparse: train AND test with top-K (E008-C)")
    p.add_argument("--train-k", type=int, default=2,
                   help="K for train-sparse mode (E008-C). Number of protos per position during training.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to pre-trained ProtoHead .pt (skip training)")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default="e008_sparsity")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# ProtoHead (extended with ranking-aware sparse forward)
# ═══════════════════════════════════════════════════════════════════

class ProtoHead(nn.Module):
    """
    原型头 | Proto Head.

    P4 → Conv(1280→128)→ReLU → CosineSim(N protos) → Conv(N→1) → logit.
    """

    def __init__(self, in_channels: int = 1280, embed_dim: int = 128, n_protos: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_protos = n_protos

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)
        self.head = nn.Conv2d(n_protos, 1, kernel_size=1, bias=True)

    def forward(self, p4: torch.Tensor, temperature: float = 0.1):
        """Returns: embedding, sim_maps, logit."""
        embedding = self.project(p4)
        emb_norm = F.normalize(embedding, dim=1, p=2)
        proto_norm = F.normalize(self.prototypes, dim=1, p=2)
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_norm, proto_norm) / temperature
        logit = self.head(sim_maps)
        return embedding, sim_maps, logit

    def forward_sparse(self, p4: torch.Tensor, k: int, temperature: float = 0.1,
                       ranking: str = "sim"):
        """
        稀疏前向 | Sparse forward: 每个位置只保留 Top-K Proto。

        Args:
            p4:     [1, 1280, H/16, W/16]
            k:      number of prototypes to keep per position
            ranking: "sim" = top-K by raw similarity (E008-A)
                     "head-weighted" = top-K by |w * sim| (E008-B, E008-C)

        Returns:
            logit_sparse: [1, 1, H/16, W/16]
            sparsity_mask: [1, N, H/16, W/16]
        """
        embedding, sim_maps, _ = self.forward(p4, temperature)
        B, N, H, W = sim_maps.shape

        # Flatten spatial dims | 展平空间维度
        sim_flat = sim_maps.permute(0, 2, 3, 1).reshape(-1, N)  # [B*H*W, N]

        # 选择排序指标 | Choose ranking metric
        if ranking == "head-weighted":
            # 真实贡献 | True contribution to logit: w_i * sim_i
            head_w = self.head.weight.squeeze()  # [N]
            importance = sim_flat * head_w.unsqueeze(0)  # [B*H*W, N]
            _, topk_idx = importance.abs().topk(k, dim=1)
        else:
            # 原始相似度 | Raw similarity (E008-A)
            _, topk_idx = sim_flat.topk(k, dim=1)

        # 构建稀疏掩码 | Build sparsity mask
        mask_flat = torch.zeros_like(sim_flat)  # [B*H*W, N]
        mask_flat.scatter_(1, topk_idx, 1.0)
        mask = mask_flat.reshape(B, H, W, N).permute(0, 3, 1, 2)  # [B, N, H, W]

        # 零化非 Top-K 通道 | Zero out non-top-K channels
        sim_sparse = sim_maps * mask
        logit_sparse = self.head(sim_sparse)

        return logit_sparse, mask


# ═══════════════════════════════════════════════════════════════════
# Training — Full (E008-A, E008-B)
# ═══════════════════════════════════════════════════════════════════

def train_full(proto_head, backbone, train_ds, val_ds, args, device, recorder):
    """
    训练全量 ProtoHead | Train with ALL protos (post-hoc modes).

    Same as E007-B training recipe.
    """
    proto_head.train()
    optimizer = torch.optim.Adam(proto_head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    best_dice, best_state = 0.0, None

    print(f"  Training {args.epochs} epochs (FULL proto, lr={args.lr}, CosineLR)...")
    for epoch in range(1, args.epochs + 1):
        proto_head.train()
        total_loss = 0.0
        pbar = tqdm(range(len(train_ds)), desc=f"  Epoch {epoch}/{args.epochs}",
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
            _, _, logit = proto_head(features["p4"], temperature=args.temperature)
            logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                     mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(logit_up.squeeze(), gt_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/(idx+1):.4f}"})
        scheduler.step()

        # Val (full proto) | 验证（全量 proto）
        proto_head.eval()
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
                _, _, logit = proto_head(features["p4"], temperature=args.temperature)
                logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                         mode="bilinear", align_corners=False)
                pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)
                if pred.dim() == 2:
                    pred = pred.unsqueeze(0)
                if gt_mask.dim() == 2:
                    gt_mask = gt_mask.unsqueeze(0)
                dices.append(compute_dice(pred, gt_mask).item())

        dice_mean = float(np.mean(dices))
        is_best = dice_mean > best_dice
        if is_best:
            best_dice = dice_mean
            best_state = {k: v.clone() for k, v in proto_head.state_dict().items()}
        marker = " *" if is_best else ""
        print(f"    Epoch {epoch:2d}/{args.epochs}  loss={total_loss/len(train_ds):.4f}  "
              f"Dice={dice_mean:.4f}{marker}")
        recorder.record_metric("dice/val_full", dice_mean, step=epoch, phase="val")

    if best_state is not None:
        proto_head.load_state_dict(best_state)
    print(f"  Best val Dice (full): {best_dice:.4f}")
    return best_dice


# ═══════════════════════════════════════════════════════════════════
# Training — Sparse (E008-C)
# ═══════════════════════════════════════════════════════════════════

def train_sparse(proto_head, backbone, train_ds, val_ds, args, device, recorder):
    """
    训练稀疏 ProtoHead | Train WITH Top-K sparsity (E008-C).

    训练和验证都使用 Top-K（head-weighted ranking）。
    梯度只通过保留的 K 个 Proto 通道回传。
    Both train and val use Top-K (head-weighted). Gradients flow
    only through kept proto channels.
    """
    proto_head.train()
    optimizer = torch.optim.Adam(proto_head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)
    best_dice, best_state = 0.0, None

    train_k = args.train_k
    print(f"  Training {args.epochs} epochs (SPARSE K={train_k}/{args.n_protos}, "
          f"lr={args.lr}, CosineLR)...")

    for epoch in range(1, args.epochs + 1):
        # ── Train with Top-K ──
        proto_head.train()
        total_loss = 0.0
        pbar = tqdm(range(len(train_ds)),
                    desc=f"  Epoch {epoch}/{args.epochs} [K={train_k}]", leave=False)
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

            # Sparse forward during training | 训练时就用稀疏前向
            logit_k, _ = proto_head.forward_sparse(
                features["p4"], k=train_k, temperature=args.temperature,
                ranking="head-weighted"
            )
            logit_up = F.interpolate(logit_k, size=tuple(gt_mask.shape),
                                     mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(logit_up.squeeze(), gt_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/(idx+1):.4f}"})
        scheduler.step()

        # ── Val with Top-K ──
        proto_head.eval()
        dices_full, dices_sparse = [], []
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

                # Val: full proto (reference) | 全量 proto（参考）
                _, _, logit_full = proto_head(features["p4"], temperature=args.temperature)

                # Val: sparse K=train_k (main metric) | 稀疏 K=train_k（主指标）
                logit_k, _ = proto_head.forward_sparse(
                    features["p4"], k=train_k, temperature=args.temperature,
                    ranking="head-weighted"
                )

                for logit, dlist in [(logit_full, dices_full), (logit_k, dices_sparse)]:
                    logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                             mode="bilinear", align_corners=False)
                    pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)
                    if pred.dim() == 2:
                        pred = pred.unsqueeze(0)
                    gm = gt_mask
                    if gm.dim() == 2:
                        gm = gm.unsqueeze(0)
                    dlist.append(compute_dice(pred, gm).item())

        dice_full = float(np.mean(dices_full))
        dice_sparse = float(np.mean(dices_sparse))

        is_best = dice_sparse > best_dice
        if is_best:
            best_dice = dice_sparse
            best_state = {k: v.clone() for k, v in proto_head.state_dict().items()}
        marker = " *" if is_best else ""
        print(f"    Epoch {epoch:2d}/{args.epochs}  loss={total_loss/len(train_ds):.4f}  "
              f"Dice(sparse K={train_k})={dice_sparse:.4f}  "
              f"Dice(full)={dice_full:.4f}{marker}")

        recorder.record_metric("dice/val_sparse", dice_sparse, step=epoch, phase="val")
        recorder.record_metric("dice/val_full_ref", dice_full, step=epoch, phase="val")

    if best_state is not None:
        proto_head.load_state_dict(best_state)
    print(f"  Best val Dice (sparse K={train_k}): {best_dice:.4f}")
    return best_dice


# ═══════════════════════════════════════════════════════════════════
# Sparsity Analysis
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_sparsity(proto_head, backbone, val_ds, device, args, ranking: str):
    """
    Top-K 稀疏推理 + 能量统计.

    Args:
        ranking: "sim" (E008-A) or "head-weighted" (E008-B, E008-C)
    """
    proto_head.eval()
    K_values = [1, 2, 3, 4, 6, args.n_protos]

    dice_by_k = {k: [] for k in K_values}
    energy_by_k = {k: [] for k in K_values}

    n_val = len(val_ds)
    ranking_label = {"sim": "E008-A (sim)", "head-weighted": "E008-B/C (|w·sim|)"}[ranking]
    print(f"\n  Analyzing sparsity on {n_val} val images [{ranking_label}]...")

    for idx in tqdm(range(n_val), desc="  Sparse analysis"):
        sample = val_ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        features = backbone(image)

        # Full forward (baseline)
        embedding, sim_maps, logit_full = proto_head(features["p4"],
                                                      temperature=args.temperature)
        N = sim_maps.shape[1]

        for k in K_values:
            if k == args.n_protos:
                logit_k = logit_full
            else:
                logit_k, _ = proto_head.forward_sparse(
                    features["p4"], k=k, temperature=args.temperature,
                    ranking=ranking
                )

            logit_up = F.interpolate(logit_k, size=tuple(gt_mask.shape),
                                     mode="bilinear", align_corners=False)
            pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)
            gm = gt_mask
            if gm.dim() == 2:
                gm = gm.unsqueeze(0)
            dice_by_k[k].append(compute_dice(pred, gm).item())

            # Energy: fraction of total |w*sim| captured by top-K
            head_w = proto_head.head.weight.squeeze()  # [N]
            sim_flat = sim_maps.squeeze(0).reshape(N, -1)  # [N, BHW]
            importance = (sim_flat * head_w.unsqueeze(1)).abs()  # [N, BHW], |w*sim|
            total_energy = importance.sum(dim=0) + 1e-8
            topk_vals, _ = importance.topk(k, dim=0)
            topk_energy = topk_vals.sum(dim=0)
            energy_ratio = (topk_energy / total_energy).mean().item()
            energy_by_k[k].append(energy_ratio)

    # ── Aggregate ──
    results = {}
    for k in K_values:
        results[k] = {
            "dice_mean": float(np.mean(dice_by_k[k])),
            "dice_std": float(np.std(dice_by_k[k])),
            "energy_mean": float(np.mean(energy_by_k[k])),
            "energy_std": float(np.std(energy_by_k[k])),
        }

    # Top-1 dominance (head-weighted energy)
    dominance_vals = energy_by_k[1]
    dominance_mean = float(np.mean(dominance_vals))
    dominance_std = float(np.std(dominance_vals))

    return results, dominance_mean, dominance_std


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_sparsity_results(results, dominance_mean, dominance_std, n_protos,
                          output_path, mode_label: str):
    """绘制稀疏度-性能 trade-off 曲线."""
    K_values = sorted(results.keys())
    dice_means = [results[k]["dice_mean"] for k in K_values]
    dice_stds = [results[k]["dice_std"] for k in K_values]
    energy_means = [results[k]["energy_mean"] for k in K_values]
    full_dice = results[n_protos]["dice_mean"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Dice vs K
    ax = axes[0]
    ax.errorbar(K_values, dice_means, yerr=dice_stds, marker="o", capsize=4,
                color="tab:blue", linewidth=2, markersize=8)
    ax.axhline(y=full_dice, color="gray", linestyle="--", alpha=0.5,
               label=f"Full (K={n_protos}) = {full_dice:.4f}")
    ax.set_xlabel("K (Top-K Protos per pixel)", fontsize=11)
    ax.set_ylabel("Dice (val)", fontsize=11)
    ax.set_title(f"Dice vs Sparse Proto Usage [{mode_label}]\n"
                 f"K=1: {dice_means[0]:.4f}  |  K={n_protos}: {full_dice:.4f}  |  "
                 f"Δ(K=1→full)={full_dice - dice_means[0]:.4f}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Energy (|w·sim|) vs K
    ax = axes[1]
    energy_stds = [results[k]["energy_std"] for k in K_values]
    ax.errorbar(K_values, energy_means, yerr=energy_stds, marker="s", capsize=4,
                color="tab:orange", linewidth=2, markersize=8)
    ax.fill_between(K_values, 0, energy_means, alpha=0.15, color="tab:orange")
    ax.set_xlabel("K (Top-K Protos per pixel)", fontsize=11)
    ax.set_ylabel("Fraction of Total |w·sim| Energy", fontsize=11)
    ax.set_title(f"Head-Weighted Energy Captured by Top-K\n"
                 f"K=1: {energy_means[0]:.1%}  |  K=2: {energy_means[1]:.1%}  |  "
                 f"K=4: {energy_means[3]:.1%}", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Plot 3: Dice vs Computation
    ax = axes[2]
    for i, k in enumerate(K_values):
        computation_pct = k / n_protos * 100
        ax.annotate(f"K={k}", (computation_pct, dice_means[i]),
                    textcoords="offset points", xytext=(5, -5), fontsize=9)
    ax.plot([k / n_protos * 100 for k in K_values], dice_means,
            marker="D", color="tab:green", linewidth=2, markersize=8)
    ax.axhline(y=full_dice, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Computation (% of full)", fontsize=11)
    ax.set_ylabel("Dice (val)", fontsize=11)
    ax.set_title(f"Dice vs Computation Trade-off\n"
                 f"K=2: {dice_means[1]:.4f} @ {2/n_protos*100:.0f}% compute", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"E008 [{mode_label}]: SPM Sparsity Validation — "
                 f"Top-1 |w·sim| Dominance = {dominance_mean:.1%} ± {dominance_std:.1%}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "sparsity_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {output_path / 'sparsity_tradeoff.png'}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device

    mode_names = {
        "post-hoc-sim": "E008-A: Post-hoc Top-K by raw similarity",
        "post-hoc-weighted": "E008-B: Post-hoc Top-K by |w·sim| (head-weighted)",
        "train-sparse": f"E008-C: Train & Test with Top-K={args.train_k}",
    }

    print("=" * 70)
    print(f"  E008: SPM Sparsity Validation")
    print(f"  {mode_names[args.mode]}")
    print(f"  N_protos={args.n_protos}, D={args.embed_dim}, Epochs={args.epochs}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 区分实验名 | Distinguish experiment names
    suffix = {"post-hoc-sim": "a", "post-hoc-weighted": "b",
              "train-sparse": f"c_k{args.train_k}"}[args.mode]
    exp_id = generate_exp_id(name=f"{args.name}_{suffix}")
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

    # ── [3] ProtoHead ──
    print(f"\n[3] ProtoHead (1280→{args.embed_dim}→{args.n_protos} protos→1)")
    proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                            n_protos=args.n_protos).to(device)
    print(f"  Params: {sum(p.numel() for p in proto_head.parameters()):,}")

    if args.checkpoint:
        print(f"  Loading checkpoint: {args.checkpoint}")
        proto_head.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                               weights_only=True))
        print("  ✅ Loaded pre-trained weights, skipping training.")
    elif args.mode == "train-sparse":
        print(f"\n[3a] Train ProtoHead WITH Top-K={args.train_k} sparsity")
        train_sparse(proto_head, backbone, train_ds, val_ds, args, device, recorder)
        ckpt_path = output_path / "proto_head_sparse.pt"
        torch.save(proto_head.state_dict(), ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")
    else:
        print(f"\n[3a] Train ProtoHead (FULL)")
        train_full(proto_head, backbone, train_ds, val_ds, args, device, recorder)
        ckpt_path = output_path / "proto_head.pt"
        torch.save(proto_head.state_dict(), ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # ── [4] Sparsity Analysis ──
    # 选择 ranking 指标 | Choose ranking metric
    if args.mode == "post-hoc-sim":
        ranking = "sim"
    else:
        ranking = "head-weighted"  # E008-B and E008-C

    print(f"\n[4] Sparsity Analysis (ranking={ranking})")
    results, dominance_mean, dominance_std = analyze_sparsity(
        proto_head, backbone, val_ds, device, args, ranking
    )

    # ── [5] Visualization ──
    print(f"\n[5] Visualization")
    plot_sparsity_results(results, dominance_mean, dominance_std, args.n_protos,
                          output_path, mode_names[args.mode])

    # ── [6] Summary ──
    full_dice = results[args.n_protos]["dice_mean"]
    K_values = sorted(results.keys())

    print(f"\n{'=' * 70}")
    print(f"  E008 结果 [{args.mode}] | Results: SPM Sparsity Validation")
    print(f"  {'=' * 70}")
    print(f"  {'K':<8} {'Dice':>10} {'ΔDice':>10} {'Δ%(rel)':>10} "
          f"{'|w·sim|%':>10} {'Compute%':>10}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for k in K_values:
        delta = full_dice - results[k]["dice_mean"]
        delta_rel = delta / (full_dice + 1e-8) * 100
        compute_pct = k / args.n_protos * 100
        print(f"  K={k:<6} {results[k]['dice_mean']:>10.4f} {delta:>10.4f} "
              f"{delta_rel:>9.1f}% {results[k]['energy_mean']:>9.1%} {compute_pct:>9.0f}%")

    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    print(f"\n  Top-1 |w·sim| Dominance: {dominance_mean:.1%} ± {dominance_std:.1%}")
    print(f"    (平均每个像素的 Top-1 Proto 占 |w·sim| 总能量的比例)")

    # ── Verdict ──
    k2_dice = results[2]["dice_mean"]
    k2_delta = full_dice - k2_dice
    k4_dice = results[4]["dice_mean"]
    k4_delta = full_dice - k4_dice

    print(f"\n  {'─'*60}")
    if args.mode == "train-sparse":
        # E008-C: the real test
        if k2_delta < 0.02:
            print(f"  ✅ SPARSE TRAINING VIABLE")
            print(f"     Train K={args.train_k}, Test K={args.train_k}: "
                  f"Dice={k2_dice:.4f} (Δ={k2_delta:.4f} from full)")
            print(f"     → ProtoHead 可以从头用稀疏约束训练")
            print(f"     → SPM 稀疏激活在训练层面可行")
            verdict = "sparse_training_viable"
        else:
            print(f"  → Sparse training needs more work")
            print(f"     Train K={args.train_k}, Test K={args.train_k}: "
                  f"Dice={k2_dice:.4f} (Δ={k2_delta:.4f} from full)")
            verdict = "sparse_training_insufficient"
    elif k2_delta < 0.01:
        print(f"  ✅ SPARSE INFERENCE VIABLE")
        print(f"     K=2 Dice={k2_dice:.4f} (Δ={k2_delta:.4f} from full)")
        print(f"     → Post-hoc Top-2 即可保持全量性能")
        verdict = "sparsity_viable"
    elif k2_delta < 0.02:
        print(f"  ✅ SPARSITY PROMISING")
        print(f"     K=2 Dice={k2_dice:.4f} (Δ={k2_delta:.4f} from full)")
        verdict = "sparsity_promising"
    elif k4_delta < 0.015:
        print(f"  △ SPARSITY AT K=4")
        print(f"     K=4 Dice={k4_dice:.4f} (Δ={k4_delta:.4f} from full)")
        verdict = "sparsity_at_k4"
    else:
        print(f"  → Proto 激活不够稀疏 (ranking={ranking})")
        print(f"     K=2 Dice={k2_dice:.4f} (Δ={k2_delta:.4f})")
        print(f"     K=4 Dice={k4_dice:.4f} (Δ={k4_delta:.4f})")
        verdict = "sparsity_insufficient"
    print(f"  {'─'*60}")

    # Record
    for k in K_values:
        recorder.record_metric(f"sparse/dice_k{k}", results[k]["dice_mean"],
                               phase="val", tags=["e008", args.mode, f"k{k}"])
        recorder.record_metric(f"sparse/energy_k{k}", results[k]["energy_mean"],
                               phase="val", tags=["e008", args.mode, f"k{k}"])
    recorder.record_metric("sparse/dominance", dominance_mean,
                           phase="val", tags=["e008", args.mode, "summary"])
    recorder.logger.log_info("e008/verdict", verdict, tags=["e008", args.mode, "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {output_path}/")

    return results


if __name__ == "__main__":
    main()
