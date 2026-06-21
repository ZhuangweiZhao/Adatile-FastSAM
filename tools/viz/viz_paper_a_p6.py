#!/usr/bin/env python3
"""
E007.5: Proto Activation Analysis — 每个 Proto 到底在看什么？
===============================================================

诊断实验 | Diagnostic experiment.

核心问题 | Core question:
    P2 (83.9% building / 70K px)  → 建筑主体？| Building interior?
    P6 (93.7% building / 4K px)   → 建筑边缘？| Building edge?
    P8 (0.8% building / 3.1M px)  → 纯背景？  | Pure background?

验证方法 | Verification:
    1. 计算每个 Proto 高激活区与 GT 边缘/内部/背景的 overlap
    2. 边缘 = Sobel(GT), 内部 = GT ∧ ¬edge, 背景 = ¬GT
    3. 多图平均统计 + 可视化

如果 P2≈内部、P6≈边缘、P8≈背景 → Proto 不仅是有效的，而是可解释的。
If P2≈interior, P6≈edge, P8≈background → Protos are not just effective, but interpretable.

用法 | Usage:
    python tools/viz_e007_p6_analysis.py
    python tools/viz_e007_p6_analysis.py --n-protos 12 --epochs 25
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.config import ExperimentConfig, ExperimentRecorder, generate_exp_id
from adatile.datasets import MassachusettsBuildingsDataset
from adatile.backbone import FastSAMBackbone
from adatile.metrics import compute_dice, format_param_count


def compute_edge(mask: np.ndarray) -> np.ndarray:
    """用 Sobel 算子提取建筑边缘 | Extract building edges via Sobel."""
    gy = sobel(mask.astype(float), axis=0)
    gx = sobel(mask.astype(float), axis=1)
    edge = np.sqrt(gx**2 + gy**2)
    return (edge > 0.1).astype(np.uint8)


@torch.no_grad()
def analyze_p6(backbone, proto_module, dataset, device, output_dir):
    """
    可视化 P2/P6/P8 的激活模式 + 计算与边缘的 overlap。
    Visualize P2/P6/P8 activation patterns + compute edge overlap.
    """
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 选择几张有建筑的 val 图 | Select val images with buildings
    proto_module.eval()

    # 先找到 proto_module 的 proto 索引 (P2=建筑, P6=建筑, P8=背景)
    # 需要先跑一次 analyze_protos
    proto_build_pct, proto_activate_count = _quick_analyze(backbone, proto_module, dataset, device)

    # 按 building ratio 排序 | Sort by building ratio
    sorted_by_build = sorted(range(len(proto_build_pct)),
                             key=lambda p: proto_build_pct[p], reverse=True)
    sorted_by_bg = sorted(range(len(proto_build_pct)),
                          key=lambda p: proto_build_pct[p])

    p_build_main = sorted_by_build[0]       # 建筑率最高的 proto | Highest building ratio
    p_build_detail = sorted_by_build[1] if len(sorted_by_build) > 1 else sorted_by_build[0]  # 次高 | Second highest
    p_bg = sorted_by_bg[0]                  # 建筑率最低的 proto | Lowest building ratio

    build_protos = [p for p in range(len(proto_build_pct)) if proto_build_pct[p] > 0.5]
    bg_protos = [p for p in range(len(proto_build_pct)) if proto_build_pct[p] < 0.3]

    print(f"  Building-leaning protos (>50%): {build_protos}")
    print(f"  Background-leaning protos (<30%): {bg_protos}")
    print(f"  Selected: P{p_build_main}=highest-build, P{p_build_detail}=2nd-build, P{p_bg}=lowest-build")
    for p in sorted_by_build:
        print(f"    P{p}: build%={proto_build_pct[p]:.1%}, px={proto_activate_count[p]:,.0f}")

    # 选 4 张图可视化 | Visualize 4 images
    val_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["masks"]
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        building_pct = mask.mean().item()
        if building_pct > 0.05:  # 至少有 5% 建筑
            val_indices.append(i)
        if len(val_indices) >= 4:
            break

    # 每个 proto 的详细分析 | Detailed per-proto analysis
    for vis_idx, ds_idx in enumerate(val_indices):
        sample = dataset[ds_idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"]
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        features = backbone(image)
        embedding, sim_maps, logit = proto_module(features["p4"], temperature=0.1)

        # 上采样 | Upsample
        sim_up = F.interpolate(sim_maps, size=tuple(gt_mask.shape),
                               mode="bilinear", align_corners=False)  # [1, N, H, W]
        sim_np = sim_up.squeeze(0).cpu().numpy()  # [N, H, W]

        img_np = sample["image"].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        gt_np = gt_mask.cpu().numpy().astype(np.uint8)
        edge_np = compute_edge(gt_np)

        # ── Per-proto analysis ──
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))

        # Row 1: Image, GT, GT Edge, Segmentation
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Image", fontsize=9)
        axes[0, 0].axis("off")

        axes[0, 1].imshow(gt_np, cmap="gray")
        axes[0, 1].set_title("GT Mask", fontsize=9)
        axes[0, 1].axis("off")

        axes[0, 2].imshow(edge_np, cmap="hot")
        axes[0, 2].set_title("GT Edges (Sobel)", fontsize=9)
        axes[0, 2].axis("off")

        logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                 mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze().cpu().numpy()
        axes[0, 3].imshow(pred, cmap="gray")
        axes[0, 3].set_title("Prediction", fontsize=9)
        axes[0, 3].axis("off")

        # Row 2: P_build_main, P_build_detail, P_bg, Overlay
        def plot_proto(ax, proto_idx, title, cmap="hot"):
            vmin, vmax = -1.0, 1.0
            im = ax.imshow(sim_np[proto_idx], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
            return im

        plot_proto(axes[1, 0], p_build_main,
                   f"P{p_build_main} (Building Interior)\n"
                   f"Build%={proto_build_pct[p_build_main]:.1%}")
        plot_proto(axes[1, 1], p_build_detail,
                   f"P{p_build_detail} (Building Detail)\n"
                   f"Build%={proto_build_pct[p_build_detail]:.1%}, "
                   f"{proto_activate_count[p_build_detail]:,.0f} px")
        plot_proto(axes[1, 2], p_bg,
                   f"P{p_bg} (Background)\n"
                   f"Build%={proto_build_pct[p_bg]:.1%}")

        # Overlay: P_build_detail on GT edges
        ax = axes[1, 3]
        ax.imshow(img_np)
        # P_build_detail 的高激活区 (top 20%) | High activation regions (top 20%)
        p_detail = sim_np[p_build_detail]
        threshold = np.percentile(p_detail, 80)
        high_act = p_detail > threshold
        ax.imshow(high_act, alpha=0.5, cmap="Reds")
        ax.set_title(f"P{p_build_detail} Top-20% Activation\noverlayed on image", fontsize=9)
        ax.axis("off")

        # Row 3: Quantitative analysis
        # P_detail vs Edge overlap
        p_act_binary = (p_detail > threshold).astype(np.uint8)
        edge_flat = edge_np.flatten()
        act_flat = p_act_binary.flatten()

        # Precision: of activated pixels, what fraction are edges?
        edge_in_act = edge_flat[act_flat == 1]
        prec = edge_in_act.mean() if len(edge_in_act) > 0 else 0

        # Recall: of edge pixels, what fraction are activated?
        act_in_edge = act_flat[edge_flat == 1]
        rec = act_in_edge.mean() if len(act_in_edge) > 0 else 0

        # 随机 baseline | Random baseline
        edge_total = edge_flat.mean()
        prec_random = edge_total

        # Bar chart: Precision, Recall vs random
        ax = axes[2, 0]
        ax.bar(["Precision", "Recall", "Edge% (random)"],
               [prec, rec, edge_total],
               color=["tab:red", "tab:blue", "gray"])
        ax.set_ylim(0, max(1.0, prec + 0.1))
        ax.set_title(f"P{p_build_detail} vs GT Edge Overlap\n"
                     f"Prec={prec:.3f} Rec={rec:.3f}", fontsize=9)
        ax.axhline(y=edge_total, color="gray", linestyle="--", alpha=0.3)

        # P_build_main activation histogram
        ax = axes[2, 1]
        p_main_vals = sim_np[p_build_main].flatten()
        ax.hist(p_main_vals[gt_np.flatten() == 1], bins=50, alpha=0.5,
                density=True, color="tab:red", label="Building")
        ax.hist(p_main_vals[gt_np.flatten() == 0], bins=50, alpha=0.5,
                density=True, color="tab:blue", label="Background")
        ax.set_title(f"P{p_build_main} Activation Distribution\n"
                     f"Building vs Background", fontsize=9)
        ax.legend(fontsize=7)

        # P_detail activation histogram
        ax = axes[2, 2]
        p_detail_vals = p_detail.flatten()
        ax.hist(p_detail_vals[gt_np.flatten() == 1], bins=50, alpha=0.5,
                density=True, color="tab:red", label="Building")
        ax.hist(p_detail_vals[gt_np.flatten() == 0], bins=50, alpha=0.5,
                density=True, color="tab:blue", label="Background")
        ax.set_title(f"P{p_build_detail} Activation Distribution\n"
                     f"Building vs Background", fontsize=9)
        ax.legend(fontsize=7)

        # All proto activations in building vs bg regions
        ax = axes[2, 3]
        x = np.arange(len(proto_build_pct))
        width = 0.35
        ax.bar(x, proto_build_pct, width, color=[
            "tab:red" if v >= 0.5 else "tab:blue" for v in proto_build_pct
        ])
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{i}" for i in range(len(proto_build_pct))], fontsize=7)
        ax.set_ylabel("Building Ratio", fontsize=8)
        ax.set_title("Per-Prototype Building Ratio", fontsize=9)
        ax.set_ylim(0, 1)

        fig.suptitle(f"P6 Analysis — Image {ds_idx} (Building Area = {gt_np.mean():.1%})",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(save_dir / f"p6_img{vis_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  img{vis_idx} (idx={ds_idx}): "
              f"P{p_build_detail} Precision(edge)={prec:.3f}, Recall(edge)={rec:.3f}, "
              f"Random={edge_total:.3f}")

    # ── 多图平均统计 | Multi-image aggregate ──
    print(f"\n  All building protos: {build_protos}")
    for p in build_protos:
        print(f"    P{p}: build%={proto_build_pct[p]:.1%}, "
              f"pixels={proto_activate_count[p]:,.0f}")


def _quick_analyze(backbone, proto_module, dataset, device):
    """Quick per-proto building ratio (reuse E007 logic)."""
    n_protos = proto_module.n_protos
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
        hard_assign = proto_module.get_hard_assignment(features["p4"])  # [1, H/16, W/16]
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "runs/p6_analysis"

    print("=" * 60)
    print("  P6 Activation Map Analysis")
    print("=" * 60)

    # Load model
    print("\n[1/3] Load Backbone")
    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    # Load or train Proto Module
    print("\n[2/3] Train Proto Module (quick)")
    sys.path.insert(0, str(_PROJECT_ROOT / "tools"))
    from eval_e007_proto_module import ProtoModule

    proto_module = ProtoModule(in_channels=1280, embed_dim=128, n_protos=12).to(device)

    # Quick train
    train_ds = MassachusettsBuildingsDataset(root_dir="data/Massachusetts_Buildings", split="train")
    val_ds = MassachusettsBuildingsDataset(root_dir="data/Massachusetts_Buildings", split="val")

    proto_module.train()
    optimizer = torch.optim.Adam(proto_module.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    best_dice = 0.0

    print("  Training 20 epochs (lr=1e-3, CosineLR)...")
    for epoch in range(1, 21):
        # ── Train ──
        proto_module.train()
        total_loss = 0.0
        pbar = tqdm(range(len(train_ds)), desc=f"  Epoch {epoch}/20 [train]", leave=False)
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
            _, _, logit = proto_module(features["p4"], temperature=0.1)
            logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                     mode="bilinear", align_corners=False)
            loss = F.binary_cross_entropy_with_logits(logit_up.squeeze(), gt_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/(idx+1):.4f}"})
        scheduler.step()

        # ── Val ──
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
                _, _, logit = proto_module(features["p4"], temperature=0.1)
                logit_up = F.interpolate(logit, size=tuple(gt_mask.shape),
                                         mode="bilinear", align_corners=False)
                pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze(1)
                # 统一维度: pred=[1,H,W], gt=[H,W] → 各加 batch 维后用 compute_dice | Align dims
                if pred.dim() == 2:
                    pred = pred.unsqueeze(0)
                if gt_mask.dim() == 2:
                    gt_mask = gt_mask.unsqueeze(0)
                dices.append(compute_dice(pred, gt_mask).item())

        dice_mean = float(np.mean(dices))
        is_best = dice_mean > best_dice
        if is_best:
            best_dice = dice_mean
        marker = " *" if is_best else ""
        print(f"    Epoch {epoch:2d}/20  loss={total_loss/len(train_ds):.4f}  "
              f"Dice(val)={dice_mean:.4f}{marker}")

    print("\n[3/3] P6 Analysis & Visualization")
    analyze_p6(backbone, proto_module, val_ds, device, output_dir)
    print(f"\n  Visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    main()
