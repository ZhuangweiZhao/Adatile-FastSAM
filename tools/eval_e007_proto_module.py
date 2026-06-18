#!/usr/bin/env python3
"""
E007: Proto Module — 原型是否自发形成语义意义？
===================================================

诊断实验 | Diagnostic experiment (不追求提分 | NOT about improving Dice).

核心问题 | Core question:
    学习到的 Prototype 是否自发对应建筑的不同语义区域？
    Do learned prototypes spontaneously correspond to semantically meaningful regions?

Proto Module 设计 | Proto Module Design:
    P4 [B, 1280, H/16]
         │
    1×1 Conv(1280 → 128) + ReLU
         │
    Embedding [B, 128, H/16]
         │
    Cosine Similarity with N learnable Prototype vectors
         │
    Similarity Maps [B, N, H/16]
         │
    1×1 Conv(N → 1) → Segmentation Logit (BCE training signal)
         │
    可视化: Proto-GT overlap, Top-K activation maps, Winner-take-all map

假设 | Hypothesis (from E006.5):
    学习投影后的特征空间有聚类结构 (Sil 0.225)。
    Proto Module 的显式模板匹配是否能:
    1. 保持分割能力 (Dice ≈ 0.40-0.44)?
    2. 自发形成语义分工 (边缘/内部/背景 Proto)?

用法 | Usage:
    python tools/eval_e007_proto_module.py
    python tools/eval_e007_proto_module.py --n-protos 12 --epochs 30
"""

from __future__ import annotations
import argparse, sys
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
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.backbone import FastSAMBackbone
from adatile.metrics import compute_dice, format_param_count


def parse_args():
    p = argparse.ArgumentParser(description="E007: Proto Module visualization")
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--n-protos", type=int, default=8,
                   help="Number of prototype vectors to learn")
    p.add_argument("--temperature", type=float, default=0.1,
                   help="Temperature for softmax over prototypes (lower = sharper assignment)")
    p.add_argument("--output-dir", type=str, default="runs")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Proto Module
# ═══════════════════════════════════════════════════════════════════

class ProtoModule(nn.Module):
    """
    原型模块 | Prototype Module.

    P4 → Embedding → Cosine Sim w/ Learnable Prototypes → Seg Logit.

    结构约束 | Structural constraint:
        分割必须通过 N 个固定原型向量的 cosine similarity 完成。
        Segmentation must go through cosine similarity to N fixed prototypes.
        这迫使网络学习有意义的原型，而非任意线性组合。
        This forces the network to learn meaningful prototypes, not arbitrary linear combos.
    """

    def __init__(self, in_channels: int = 1280, embed_dim: int = 128, n_protos: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_protos = n_protos

        # 特征投影 | Feature projection
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 可学习原型向量 | Learnable prototype vectors
        # 随机初始化，训练中学习 | Random init, learned during training
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)

        # 分割头：原型响应的 1×1 线性组合 | Seg head: 1×1 linear combo of proto responses
        self.head = nn.Conv2d(n_protos, 1, kernel_size=1, bias=True)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  ProtoModule: {format_param_count(n_params)} ({n_params:,})")
        print(f"    project={sum(p.numel() for p in self.project.parameters()):,}")
        print(f"    prototypes={self.prototypes.numel():,}")
        print(f"    head={sum(p.numel() for p in self.head.parameters()):,}")

    def forward(self, p4: torch.Tensor, temperature: float = 0.1
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            p4: [B, 1280, H/16, W/16]
            temperature: softmax temperature (lower = sharper)

        Returns:
            embedding:  [B, D, H, W]  低维嵌入 | Low-dim embedding
            sim_maps:   [B, N, H, W]  proto 相似度图 | Proto similarity maps
            logit:      [B, 1, H, W]  分割 logit | Segmentation logit
        """
        # 投影 | Project
        embedding = self.project(p4)  # [B, D, H, W]

        # L2 normalize embedding and prototypes for cosine similarity
        emb_norm = F.normalize(embedding, dim=1, p=2)          # [B, D, H, W]
        proto_norm = F.normalize(self.prototypes, dim=1, p=2)  # [N, D]

        # Cosine similarity via einsum: [B, D, H, W] × [N, D] → [B, N, H, W]
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_norm, proto_norm)

        # 可选的温度缩放 | Optional temperature scaling
        sim_maps = sim_maps / temperature

        # 分割 logit: proto 响应的线性组合 | Seg logit: linear combo of proto responses
        logit = self.head(sim_maps)  # [B, 1, H, W]

        return embedding, sim_maps, logit

    def get_soft_assignment(self, p4: torch.Tensor, temperature: float = 0.1
                            ) -> torch.Tensor:
        """
        获取每个像素的软分配（softmax over prototypes）。
        Get soft assignment per pixel (softmax over prototypes).

        Returns:
            [B, N, H, W] softmax probabilities over prototypes.
        """
        _, sim_maps, _ = self.forward(p4, temperature)
        return F.softmax(sim_maps, dim=1)

    def get_hard_assignment(self, p4: torch.Tensor) -> torch.Tensor:
        """
        获取每个像素的硬分配（argmax over prototypes）。
        Get hard assignment per pixel (argmax over prototypes).

        Returns:
            [B, H, W] prototype index per pixel (0 to N-1).
        """
        _, sim_maps, _ = self.forward(p4, temperature=0.01)  # sharp for hard assignment
        return sim_maps.argmax(dim=1)  # [B, H, W]


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train_proto(proto_module, backbone, train_ds, val_ds, args, device, recorder):
    """训练 Proto Module | Train Proto Module."""
    proto_module.train()
    optimizer = torch.optim.Adam(proto_module.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    best_dice, best_state = 0.0, None

    print(f"\n  Training {args.epochs} epochs (lr={args.lr}, CosineLR)...")
    for epoch in range(1, args.epochs + 1):
        proto_module.train()
        total_loss = 0.0
        for idx in tqdm(range(len(train_ds)), desc=f"  Epoch {epoch}/{args.epochs}", leave=False):
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

            _, _, logit = proto_module(p4, temperature=args.temperature)
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
        proto_module.eval()
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
                _, _, logit = proto_module(features["p4"], temperature=args.temperature)
                logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                         mode="bilinear", align_corners=False)
                pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)
                dices.append(compute_dice(pred, gt_mask.unsqueeze(0)).item())

        dice_mean = float(np.mean(dices))
        recorder.record_metric("loss/train", avg_loss, step=epoch, phase="train")
        recorder.record_metric("dice/val", dice_mean, step=epoch, phase="val")

        if dice_mean > best_dice:
            best_dice = dice_mean
            best_state = {k: v.clone() for k, v in proto_module.state_dict().items()}

        print(f"    loss={avg_loss:.4f}  Dice={dice_mean:.4f}"
              f"{' *' if dice_mean == best_dice else ''}")

    proto_module.load_state_dict(best_state)
    print(f"  Best val Dice: {best_dice:.4f}")
    return best_dice


# ═══════════════════════════════════════════════════════════════════
# Proto Analysis & Visualization
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_protos(proto_module, backbone, dataset, device, n_protos, output_path):
    """
    分析每个 Prototype 与 Building/Background 的关联。
    Analyze each prototype's association with Building/Background.

    对多张图像，统计每个 proto 激活最强的像素中 building 占比。
    Across multiple images, compute building ratio among pixels
    where each proto has the strongest activation.
    """
    proto_module.eval()

    # 累积统计 | Accumulate statistics
    proto_build_pct = np.zeros(n_protos)    # 每个 proto 的建筑像素占比
    proto_activate_count = np.zeros(n_protos)  # 每个 proto 被选中的次数

    for idx in tqdm(range(min(20, len(dataset))), desc="  Analyzing protos"):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        features = backbone(image)
        hard_assign = proto_module.get_hard_assignment(features["p4"])  # [1, H/16, W/16]

        # 上采样到 GT 分辨率 | Upsample to GT resolution
        hard_up = F.interpolate(
            hard_assign.unsqueeze(1).float(),
            size=tuple(gt_mask.shape),
            mode="nearest",
        ).squeeze(1).long()  # [1, H, W]

        for p in range(n_protos):
            proto_mask = (hard_up == p)
            n_pixels = proto_mask.sum().item()
            if n_pixels > 0:
                n_building = (gt_mask[proto_mask.squeeze(0)] == 1).sum().item()
                proto_build_pct[p] += n_building
                proto_activate_count[p] += n_pixels

    # 归一化 | Normalize
    for p in range(n_protos):
        if proto_activate_count[p] > 0:
            proto_build_pct[p] /= proto_activate_count[p]

    return proto_build_pct, proto_activate_count


@torch.no_grad()
def visualize_proto_maps(proto_module, backbone, dataset, device, args, output_path):
    """
    可视化 Proto 激活图 + Winner-take-all 图。
    Visualize Proto activation maps + Winner-take-all map.

    选 2 张 val 图像，绘制:
    - 原图 + GT
    - 每个 Proto 的 similarity heatmap (N 个子图)
    - Winner-take-all map (每个像素颜色 = 激活最强的 proto)
    - Per-proto 建筑占比条形图
    """
    proto_module.eval()
    n_vis = min(2, len(dataset))
    indices = [len(dataset) - 1, len(dataset) // 2][:n_vis]  # 选最后和中间的图

    # 计算全局 proto 语义 | Compute global proto semantics
    proto_build_pct, proto_activate_count = analyze_protos(
        proto_module, backbone, dataset, device, args.n_protos, output_path
    )

    n_protos = args.n_protos
    n_cols = min(6, n_protos)
    n_rows_proto = (n_protos + n_cols - 1) // n_cols

    for fig_idx, ds_idx in enumerate(indices):
        sample = dataset[ds_idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].to(device)
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        features = backbone(image)
        _, sim_maps, _ = proto_module(features["p4"], temperature=args.temperature)

        # 上采样 sim_maps 和 hard_assign | Upsample
        sim_up = F.interpolate(sim_maps, size=tuple(gt_mask.shape),
                               mode="bilinear", align_corners=False)  # [1, N, H, W]
        hard_up = F.interpolate(
            sim_maps.argmax(dim=1, keepdim=True).float(),
            size=tuple(gt_mask.shape), mode="nearest",
        ).squeeze(1).long().squeeze(0)  # [H, W]

        # 转换为 numpy | Convert to numpy
        img_np = sample["image"].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
        img_np = np.clip(img_np, 0, 1)
        gt_np = gt_mask.cpu().numpy()
        sim_np = sim_up.squeeze(0).cpu().numpy()  # [N, H, W]
        hard_np = hard_up.cpu().numpy()

        # ── Figure 1: Proto similarity maps ──
        fig, axes = plt.subplots(2 + n_rows_proto, n_cols,
                                 figsize=(3 * n_cols, 3 * (2 + n_rows_proto)))
        axes = np.atleast_2d(axes)

        # Row 0: Original image + GT
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original Image", fontsize=9)
        axes[0, 0].axis("off")
        axes[0, 1].imshow(gt_np, cmap="gray")
        axes[0, 1].set_title("GT Mask", fontsize=9)
        axes[0, 1].axis("off")
        for c in range(2, n_cols):
            axes[0, c].axis("off")

        # Row 1: Winner-take-all + building ratio
        # Color-code each proto
        cmap_proto = plt.cm.tab10
        axes[1, 0].imshow(img_np)
        axes[1, 0].imshow(hard_np, alpha=0.5, cmap=cmap_proto, vmin=0, vmax=max(9, n_protos - 1))
        axes[1, 0].set_title("Winner-Take-All Map", fontsize=9)
        axes[1, 0].axis("off")

        # Building ratio bar chart
        colors = [cmap_proto(i) for i in range(n_protos)]
        bars = axes[1, 1].bar(range(n_protos), proto_build_pct, color=colors)
        axes[1, 1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        axes[1, 1].set_xticks(range(n_protos))
        axes[1, 1].set_xticklabels([f"P{i}" for i in range(n_protos)], fontsize=7)
        axes[1, 1].set_ylabel("Building Ratio", fontsize=8)
        axes[1, 1].set_title("Per-Proto Building Ratio", fontsize=9)
        axes[1, 1].set_ylim(0, 1)
        for p, pct in enumerate(proto_build_pct):
            axes[1, 1].text(p, pct + 0.03, f"{pct:.1%}", ha="center", fontsize=6)
        for c in range(2, n_cols):
            axes[1, c].axis("off")

        # Rows 2+: Per-proto similarity heatmaps
        for p in range(n_protos):
            r = 2 + p // n_cols
            c = p % n_cols
            im = axes[r, c].imshow(sim_np[p], cmap="hot", vmin=-1, vmax=1)
            label = "Build" if proto_build_pct[p] >= 0.5 else "BG"
            axes[r, c].set_title(f"P{p} ({label}, {proto_build_pct[p]:.0%})", fontsize=8)
            axes[r, c].axis("off")
            plt.colorbar(im, ax=axes[r, c], fraction=0.046)

        # Hide unused subplots
        for r in range(2 + n_rows_proto):
            for c in range(n_cols):
                if r >= 2 and (r - 2) * n_cols + c >= n_protos:
                    axes[r, c].axis("off")

        fig.suptitle(f"E007 Proto Analysis — Image {ds_idx} (N={n_protos} protos, "
                     f"D={args.embed_dim})",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(f"{output_path}/proto_maps_img{fig_idx}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Proto maps saved: proto_maps_img{fig_idx}.png")

    return proto_build_pct, proto_activate_count


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"  E007: Proto Module — N={args.n_protos} prototypes, D={args.embed_dim}")
    print("  核心问题: Prototype 是否自发形成语义意义?")
    print("=" * 70)

    exp_id = generate_exp_id(name=args.name or "e007_proto")
    config = ExperimentConfig(exp_id=exp_id, output_dir=args.output_dir,
                              dataset_name="Massachusetts_Buildings", dataset_root=args.data_root)
    recorder = ExperimentRecorder(config)
    recorder.record_config()
    output_path = Path(config.output_dir) / exp_id
    output_path.mkdir(parents=True, exist_ok=True)

    # ── [1/5] Backbone ──
    print("\n[1/5] Frozen FastSAM Backbone")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    # ── [2/5] Data ──
    print("\n[2/5] Load Data")
    train_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir=args.data_root, split="val")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── [3/5] Proto Module ──
    print(f"\n[3/5] Proto Module (1280 → {args.embed_dim} → {args.n_protos} protos → 1)")
    proto_module = ProtoModule(
        in_channels=1280, embed_dim=args.embed_dim, n_protos=args.n_protos
    ).to(device)

    # ── [4/5] Train ──
    print(f"\n[4/5] Train Proto Module")
    best_dice = train_proto(proto_module, backbone, train_ds, val_ds, args, device, recorder)
    recorder.logger.log_info("e007/train", f"best_dice={best_dice:.4f}")

    # ── [5/5] Proto Semantic Analysis ──
    print(f"\n[5/5] Proto Semantic Analysis & Visualization")
    proto_build_pct, proto_activate_count = visualize_proto_maps(
        proto_module, backbone, val_ds, device, args, str(output_path)
    )

    # ── Summary ──
    # 判断是否出现语义分工 | Check for semantic specialization
    sorted_idx = np.argsort(proto_build_pct)
    bg_protos = [i for i in sorted_idx if proto_build_pct[i] < 0.3]
    build_protos = [i for i in sorted_idx if proto_build_pct[i] > 0.7]
    mixed_protos = [i for i in sorted_idx if 0.3 <= proto_build_pct[i] <= 0.7]

    has_specialization = len(build_protos) >= 1 and len(bg_protos) >= 1

    print(f"\n{'=' * 70}")
    print(f"  E007 结果 | Results: Proto Semantic Analysis")
    print(f"  {'─' * 50}")
    print(f"  Dice (val):  {best_dice:.4f}")
    print(f"  {'─' * 50}")
    print(f"  {'Proto':<8} {'Build%':>10} {'Pixels':>12} {'Semantic':>12}")
    print(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*12}")
    for p in range(args.n_protos):
        semantic = "Building" if proto_build_pct[p] > 0.7 else \
                   ("Background" if proto_build_pct[p] < 0.3 else "Mixed")
        print(f"  P{p:<7} {proto_build_pct[p]:>10.1%} "
              f"{proto_activate_count[p]:>12,.0f} {semantic:>12}")
    print(f"  {'─'*50}")
    print(f"  Building-specialized:  {len(build_protos)}/{args.n_protos}  "
          f"({sorted(build_protos)})")
    print(f"  Background-specialized: {len(bg_protos)}/{args.n_protos}  "
          f"({sorted(bg_protos)})")
    print(f"  Mixed:                 {len(mixed_protos)}/{args.n_protos}  "
          f"({sorted(mixed_protos)})")
    print(f"  {'─'*50}")

    if has_specialization:
        print(f"  ✅ Prototype 自发形成了语义分工！")
        print(f"     → Building-specific 和 Background-specific 原型自然分化。")
        print(f"     → 论文叙事：特征重组后自发形成具有语义意义的原型结构。")
        verdict = "semantic_specialization"
    elif len(build_protos) >= 1 or len(bg_protos) >= 1:
        print(f"  △ 部分语义分化，但不够清晰。")
        print(f"     → 可能需要更多原型或更长的训练。")
        verdict = "partial_specialization"
    else:
        print(f"  → 无显著语义分化。")
        print(f"     → Proto 约束可能需要更强的训练信号。")
        verdict = "no_specialization"

    print(f"{'=' * 70}")

    # Record
    recorder.record_metric("e007/dice", best_dice, phase="val", tags=["e007", "summary"])
    recorder.record_metric("e007/n_build_protos", len(build_protos),
                           phase="val", tags=["e007", "summary"])
    recorder.record_metric("e007/n_bg_protos", len(bg_protos),
                           phase="val", tags=["e007", "summary"])
    recorder.logger.log_info("e007/verdict", verdict, tags=["e007", "summary"])
    recorder.finalize()
    recorder.close()
    print(f"\n  Results saved to: {output_path}/")


if __name__ == "__main__":
    main()
