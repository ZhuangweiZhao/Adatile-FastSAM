#!/usr/bin/env python3
"""
E009-C: Router Visualization — Router 学会了什么路由策略？
===========================================================

核心问题 | Core question:
    conv3x3 Router 在不同的空间区域（建筑内部/边缘/背景）
    分别选择哪些 Proto？路由是否具有空间语义？

验证假设 | Hypotheses:
    H1: 建筑内部 → 特定 Proto 组合 (Building Body protos)
    H2: 建筑边缘 → 不同 Proto 组合 (Edge/Detail protos)
    H3: 背景     → 抑制 Proto 组合 (Background protos)
    H4: 路由具有空间连续性（相邻像素路由相似）

与 E007.5 的关系 | Relation to E007.5:
    E007.5: 每个 Proto 看什么区域？（Proto → Region）
    E009-C: 每个区域用什么 Proto？（Region → Proto）
    两者互补，形成完整的可解释性论证。

用法 | Usage:
    python tools/viz_e009_router.py --checkpoint runs/.../spm_head_s2.pt
"""

import sys, argparse, glob as _glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from collections import Counter

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.datasets import MassachusettsBuildingsDataset
from adatile.backbone import FastSAMBackbone
from eval_e009_spm_router import ProtoHead, SPMHead


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data/Massachusetts_Buildings")
    p.add_argument("--n-protos", type=int, default=8)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--router-k", type=int, default=4)
    p.add_argument("--n-images", type=int, default=4,
                   help="Number of test images to visualize")
    p.add_argument("--split", type=str, default="test",
                   help="Dataset split to use (test has more images)")
    p.add_argument("--output-dir", type=str, default="runs/router_viz")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def compute_edge(mask: np.ndarray) -> np.ndarray:
    """Sobel 边缘 | Sobel edge detection."""
    gy = sobel(mask.astype(float), axis=0)
    gx = sobel(mask.astype(float), axis=1)
    edge = np.sqrt(gx**2 + gy**2)
    return (edge > 0.1).astype(np.uint8)


def region_mask(gt: np.ndarray) -> dict:
    """
    三分区掩码 | Three-region mask.

    Returns:
        interior: GT=1 AND edge=0 (建筑内部)
        edge:     GT=1 AND edge=1 (建筑边缘)
        bg:       GT=0            (背景)
    """
    edge_map = compute_edge(gt)
    return {
        "interior": (gt == 1) & (edge_map == 0),
        "edge":     (gt == 1) & (edge_map == 1),
        "bg":       gt == 0,
    }, edge_map


@torch.no_grad()
def analyze_routing(spm_head, backbone, dataset, device, args, output_dir):
    """
    可视化 Router 的空间路由决策 | Visualize Router's spatial routing decisions.
    """
    spm_head.eval()
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    k = args.router_k
    n_protos = args.n_protos

    # 累积跨图像统计 | Accumulate cross-image statistics
    # per_region[region][proto] = count of times this proto is in top-K
    per_region = {r: np.zeros(n_protos) for r in ["interior", "edge", "bg"]}
    region_pixel_counts = {r: 0 for r in ["interior", "edge", "bg"]}

    # 每个 proto 组合的统计 | Per-combination statistics
    combo_counter = Counter()

    # 选图 | Select images with buildings
    img_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["masks"]
        if mask.dim() == 3:
            mask = mask.squeeze(0)
        building_pct = mask.mean().item()
        if building_pct > 0.03:
            img_indices.append(i)
        if len(img_indices) >= args.n_images:
            break

    print(f"  Selected {len(img_indices)} images from {args.split} split")
    print(f"  Analyzing router decisions (K={k}/{n_protos})...")

    for vis_idx, ds_idx in enumerate(tqdm(img_indices, desc="  Router viz")):
        sample = dataset[ds_idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"]
        if gt_mask.dim() == 3:
            gt_mask = gt_mask.squeeze(0)
        elif gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(0).squeeze(0)

        features = backbone(image)
        p4 = features["p4"]

        # ── Full forward ──
        logit_full, sim_maps, embedding = spm_head.forward_full(p4)
        _, _, _, router_logits = spm_head.forward_routed(p4, mode="learned", k=k)

        # ── Router Top-K per pixel ──
        # router_logits: [1, N, H/16, W/16]
        router_flat = router_logits.squeeze(0).reshape(n_protos, -1)  # [N, H/16*W/16]
        _, topk_idx = router_flat.topk(k, dim=0)  # [K, H/16*W/16]
        topk_idx = topk_idx.cpu().numpy()  # [K, H/16*W/16]

        # 上采样路由选择到 GT 分辨率 | Upsample routing to GT resolution
        H_emb, W_emb = router_logits.shape[2], router_logits.shape[3]
        H_gt, W_gt = int(gt_mask.shape[0]), int(gt_mask.shape[1])

        # 对每个 proto，创建路由掩码并上采样
        routing_up = np.zeros((n_protos, H_gt, W_gt), dtype=bool)
        for p in range(n_protos):
            mask_p = (topk_idx == p).any(axis=0).reshape(H_emb, W_emb)  # [H/16, W/16]
            mask_p_t = torch.from_numpy(mask_p).float().unsqueeze(0).unsqueeze(0).to(device)
            mask_up = F.interpolate(mask_p_t, size=(H_gt, W_gt), mode="nearest")
            routing_up[p] = mask_up.squeeze().cpu().numpy() > 0.5

        # ── Region analysis ──
        gt_np = gt_mask.cpu().numpy().astype(np.uint8)
        regions, edge_np = region_mask(gt_np)

        # GT 上采样到路由分辨率用于 combo 统计
        gt_down = F.interpolate(
            torch.from_numpy(gt_np).float().unsqueeze(0).unsqueeze(0).to(device),
            size=(H_emb, W_emb), mode="nearest"
        ).squeeze().cpu().numpy() > 0.5
        edge_down = compute_edge(gt_down.astype(np.uint8))
        regions_down = {
            "interior": (gt_down == 1) & (edge_down == 0),
            "edge":     (gt_down == 1) & (edge_down == 1),
            "bg":       gt_down == 0,
        }

        for region_name, region_m in regions_down.items():
            if region_m.sum() == 0:
                continue
            region_pixel_counts[region_name] += region_m.sum()
            for p in range(n_protos):
                proto_present = (topk_idx == p).any(axis=0)  # [H/16*W/16]
                per_region[region_name][p] += proto_present[region_m.flatten()].sum()

            # Count routing combinations in this region
            for pixel_i in np.where(region_m.flatten())[0]:
                combo = tuple(sorted(topk_idx[:, pixel_i].tolist()))
                combo_counter[(region_name, combo)] += 1

        # ── Per-image visualization ──
        img_np = sample["image"].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)

        # Prediction
        logit_up = F.interpolate(logit_full, size=(H_gt, W_gt),
                                 mode="bilinear", align_corners=False)
        pred = (torch.sigmoid(logit_up) > 0.5).float().squeeze().cpu().numpy()

        # ── Figure ──
        fig = plt.figure(figsize=(20, 14))

        # Row 1: Image, GT, Prediction, Edge Map
        ax = plt.subplot(3, 4, 1)
        ax.imshow(img_np); ax.set_title("Image", fontsize=9); ax.axis("off")

        ax = plt.subplot(3, 4, 2)
        ax.imshow(gt_np, cmap="gray"); ax.set_title("GT Mask", fontsize=9); ax.axis("off")

        ax = plt.subplot(3, 4, 3)
        ax.imshow(pred, cmap="gray"); ax.set_title("Prediction (Full)", fontsize=9); ax.axis("off")

        ax = plt.subplot(3, 4, 4)
        ax.imshow(edge_np, cmap="hot"); ax.set_title("GT Edges", fontsize=9); ax.axis("off")

        # Row 2: Region heatmaps — which protos activated at each pixel
        # For each pixel, show the dominant proto index as a color
        dominant = np.argmax(
            np.array([routing_up[p].astype(float) for p in range(n_protos)]), axis=0
        )
        # Color by dominant proto
        ax = plt.subplot(3, 4, 5)
        ax.imshow(img_np)
        im = ax.imshow(dominant, alpha=0.5, cmap="tab10", vmin=0, vmax=9)
        ax.set_title("Dominant Proto per Pixel\n(color = proto index)", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, ticks=range(n_protos))

        # Routing map: each pixel color-coded by proto combination hash
        combo_hash = np.zeros((H_emb, W_emb), dtype=int)
        for h in range(H_emb):
            for w in range(W_emb):
                combo_hash[h, w] = hash(tuple(sorted(topk_idx[:, h * W_emb + w].tolist()))) % 1000
        combo_hash_up = F.interpolate(
            torch.from_numpy(combo_hash).float().unsqueeze(0).unsqueeze(0).to(device),
            size=(H_gt, W_gt), mode="nearest"
        ).squeeze().cpu().numpy()

        ax = plt.subplot(3, 4, 6)
        ax.imshow(img_np)
        ax.imshow(combo_hash_up, alpha=0.4, cmap="tab20")
        ax.set_title(f"Routing Combination Map\n"
                     f"(unique combos = {len(set(combo_hash.flatten()))})", fontsize=9)
        ax.axis("off")

        # Per-region proto frequency (this image)
        img_per_region = {r: np.zeros(n_protos) for r in ["interior", "edge", "bg"]}
        for r in img_per_region:
            if regions[r].sum() > 0:
                for p in range(n_protos):
                    img_per_region[r][p] = routing_up[p][regions[r]].mean()

        colors_bar = plt.cm.tab10(np.linspace(0, 1, n_protos))
        for i, (r_name, r_label) in enumerate([("interior", "Interior"),
                                                ("edge", "Edge"), ("bg", "Background")]):
            ax = plt.subplot(3, 4, 7 + i)
            ax.bar(range(n_protos), img_per_region[r_name], color=colors_bar)
            ax.set_title(f"{r_label} — Proto Activation Rate", fontsize=9)
            ax.set_xlabel("Proto", fontsize=8)
            ax.set_ylabel("Activation Rate", fontsize=8)
            ax.set_xticks(range(n_protos))
            ax.set_xticklabels([f"P{p}" for p in range(n_protos)], fontsize=7)
            ax.set_ylim(0, 1)

        # Routing spatial coherence
        # % of adjacent pixel pairs with shared proto selection
        routing_onehot = np.stack([routing_up[p].astype(float) for p in range(n_protos)], axis=0)
        # Jaccard similarity of adjacent pixels
        jaccard_h = (routing_onehot[:, :-1, :] * routing_onehot[:, 1:, :]).sum(axis=0)
        union_h = (routing_onehot[:, :-1, :] + routing_onehot[:, 1:, :]).clip(0, 1).sum(axis=0)
        coherence_h = (jaccard_h / (union_h + 1e-8)).mean()

        jaccard_w = (routing_onehot[:, :, :-1] * routing_onehot[:, :, 1:]).sum(axis=0)
        union_w = (routing_onehot[:, :, :-1] + routing_onehot[:, :, 1:]).clip(0, 1).sum(axis=0)
        coherence_w = (jaccard_w / (union_w + 1e-8)).mean()

        ax = plt.subplot(3, 4, 10)
        ax.text(0.5, 0.5, f"Spatial Coherence\n\n"
                f"Horizontal: {coherence_h:.3f}\n"
                f"Vertical:   {coherence_w:.3f}\n\n"
                f"Top Region Combos:\n"
                + "\n".join(f"  {r}: {tuple(c)} ({cnt}x)"
                           for (r, c), cnt in combo_counter.most_common(6)),
                transform=ax.transAxes, fontsize=8, va="center", ha="center",
                family="monospace")
        ax.set_title("Routing Statistics", fontsize=9)
        ax.axis("off")

        # Learned vs Fixed routing difference map
        _, _, _, _ = spm_head.forward_routed(p4, mode="fixed", k=k)
        # Compute learned routing again and compare
        # Show where Learned ≠ Fixed
        router_flat_l = router_logits.squeeze(0).reshape(n_protos, -1)
        _, topk_l = router_flat_l.topk(k, dim=0)

        # Fixed top-K
        head_w = spm_head.proto_head.head.weight.squeeze()
        sim_flat = sim_maps.squeeze(0).reshape(n_protos, -1)
        importance = (sim_flat * head_w.unsqueeze(1)).abs()
        _, topk_f = importance.topk(k, dim=0)

        agree_mask = np.zeros(H_emb * W_emb, dtype=float)
        for pi in range(H_emb * W_emb):
            agree_mask[pi] = len(set(topk_l[:, pi].cpu().numpy()) &
                                 set(topk_f[:, pi].cpu().numpy())) / k
        agree_map = agree_mask.reshape(H_emb, W_emb)
        agree_up = F.interpolate(
            torch.from_numpy(agree_map).float().unsqueeze(0).unsqueeze(0).to(device),
            size=(H_gt, W_gt), mode="nearest"
        ).squeeze().cpu().numpy()

        ax = plt.subplot(3, 4, 11)
        ax.imshow(img_np)
        im = ax.imshow(agree_up, alpha=0.5, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"Learned vs Fixed Agreement\n"
                     f"(mean={agree_mask.mean():.2f})", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Full prediction with routing overlay
        ax = plt.subplot(3, 4, 12)
        ax.imshow(img_np)
        # Show where router disagrees with fixed
        disagreement = agree_up < 0.5
        ax.imshow(disagreement, alpha=0.3, cmap="Reds")
        ax.set_title("Regions where Router ≠ |w·sim|\n(red overlay)", fontsize=9)
        ax.axis("off")

        fig.suptitle(f"E009-C Router Visualization — Image {ds_idx} "
                     f"(K={k}/{n_protos}, arch=conv3x3, Building={gt_np.mean():.1%})",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        fig.savefig(save_dir / f"router_viz_img{vis_idx}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ── Cross-image summary ──
    print(f"\n  ── Cross-Image Proto Selection by Region ──")
    print(f"  {'Proto':<8} {'Interior':>12} {'Edge':>12} {'Background':>12}")
    print(f"  {'─'*8} {'─'*12} {'─'*12} {'─'*12}")
    for p in range(n_protos):
        ir = per_region["interior"][p] / max(region_pixel_counts["interior"], 1)
        er = per_region["edge"][p] / max(region_pixel_counts["edge"], 1)
        br = per_region["bg"][p] / max(region_pixel_counts["bg"], 1)
        print(f"  P{p:<7} {ir:>11.1%} {er:>11.1%} {br:>11.1%}")

    # ── Per-region proto preference bar chart ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (r_name, r_label) in enumerate([("interior", "Interior"),
                                            ("edge", "Edge"), ("bg", "Background")]):
        rates = per_region[r_name] / max(region_pixel_counts[r_name], 1)
        colors = ["tab:red" if v >= 0.5 else "tab:blue" for v in rates]
        axes[i].bar(range(n_protos), rates, color=colors)
        axes[i].axhline(y=k/n_protos, color="gray", linestyle="--", alpha=0.5,
                        label=f"Uniform ({k/n_protos:.1%})")
        axes[i].set_title(f"{r_label} — Proto Selection Rate\n"
                          f"({region_pixel_counts[r_name]:,} px)")
        axes[i].set_xticks(range(n_protos))
        axes[i].set_xticklabels([f"P{p}" for p in range(n_protos)])
        axes[i].set_ylabel("Selection Rate")
        axes[i].set_ylim(0, 1)
        axes[i].legend(fontsize=7)

    fig.suptitle(f"E009-C: Per-Region Proto Selection (K={k}/{n_protos})", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_dir / "region_proto_selection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Top routing combinations ──
    print(f"\n  ── Top Routing Combinations by Region ──")
    for r_name in ["interior", "edge", "bg"]:
        r_combos = [(c, cnt) for (r, c), cnt in combo_counter.items() if r == r_name]
        r_combos.sort(key=lambda x: -x[1])
        print(f"  [{r_name:9s}]")
        for combo, cnt in r_combos[:5]:
            pct = cnt / max(sum(c for _, c in r_combos), 1)
            print(f"    {combo}  ({cnt:6d} px, {pct:.1%})")

    print(f"\n  Visualizations saved to: {save_dir}/")
    return per_region, region_pixel_counts


def main():
    args = parse_args()
    device = args.device

    # Load checkpoint
    ckpt_path = args.checkpoint
    if "*" in ckpt_path:
        matches = _glob.glob(ckpt_path)
        ckpt_path = matches[0] if matches else ckpt_path
    print(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    router_arch = ckpt.get("router_arch", "conv3x3")
    print(f"  Router arch: {router_arch}")

    proto_head = ProtoHead(in_channels=1280, embed_dim=args.embed_dim,
                            n_protos=args.n_protos).to(device)
    proto_head.load_state_dict(ckpt["proto_head"])

    spm_head = SPMHead(proto_head, n_protos=args.n_protos,
                        router_k=args.router_k,
                        router_arch=router_arch).to(device)
    spm_head.router.load_state_dict(ckpt["router"])
    spm_head.eval()

    backbone = FastSAMBackbone(freeze_backbone=True)
    backbone.eval()

    dataset = MassachusettsBuildingsDataset(root_dir=args.data_root, split=args.split)
    print(f"Dataset: {args.split} ({len(dataset)} images)")

    analyze_routing(spm_head, backbone, dataset, device, args, args.output_dir)


if __name__ == "__main__":
    main()
