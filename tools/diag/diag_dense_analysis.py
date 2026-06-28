#!/usr/bin/env python3
"""
Dense Matching 深度分析: Oracle Upper Bound + Shot Scaling.
==============================================================
Deep analysis: what is the ceiling of dense matching, and how does it scale with shots?

实验 1: Oracle Dense Matching — 用 GT mask 替代 support→query matching 的上界
    对每个 query FG 像素，在所有 support FG 像素中找最匹配的 → 理论最优 dense matching IoU

实验 2: Shot Scaling — Dense Matching 是否比 Prototype 更高效地利用更多 support?
    1-shot → 3-shot → 5-shot → 10-shot 对比 Proto vs Dense

输出 | Output:
    - runs/diag_dense_analysis/{timestamp}/oracle_dense.json
    - runs/diag_dense_analysis/{timestamp}/shot_scaling.json

用法 | Usage:
    python tools/diag/diag_dense_analysis.py \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --fold 0 --device cuda
"""

import sys, json, argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
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

# Reuse dense_matching and compute_best_iou from diag_dense_matching
from tools.diag.diag_dense_matching import (
    dense_matching, prototype_matching, compute_best_iou,
)


# ═══════════════════════════════════════════════════════════════════
# Experiment 1: Oracle Dense Matching
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def oracle_dense_episode(
    dataset, backbone, class_id: int, shot: int,
    device: torch.device, rng: np.random.RandomState,
) -> dict:
    """
    Oracle dense matching: 对每个 query FG 像素，
    在所有 support FG 像素中找最佳匹配 → 理论上界。

    Oracle dense matching: for each query FG pixel,
    find the best-matching support FG pixel → theoretical upper bound.
    """
    candidates = dataset.class_to_images(class_id)
    if len(candidates) < shot + 1:
        return None

    indices = rng.choice(candidates, shot + 1, replace=False)
    support_idxs = indices[:shot]
    query_idx = int(indices[shot])

    # Load support
    support_imgs = []
    support_masks_orig = []
    for si in support_idxs:
        img = dataset.load_image(int(si)).to(device)
        mask = dataset.render_class_mask(int(si), class_id).to(device)
        if dataset.crop_support and mask.sum() > 64:
            img, mask = dataset._roi_crop(img, mask)
        support_imgs.append(img)
        support_masks_orig.append(mask)

    support_batch = torch.stack(support_imgs)

    # Load query
    query_img = dataset.load_image(int(query_idx)).unsqueeze(0).to(device)
    query_mask = dataset.render_class_mask(int(query_idx), class_id).to(device)
    H_orig, W_orig = query_mask.shape

    # Backbone
    s_feats = backbone(support_batch)
    q_feats = backbone(query_img)
    s_p4 = s_feats["p4"]
    q_p4 = q_feats["p4"]

    # Resize masks to P4 scale
    s_masks_list = []
    for k in range(shot):
        m = F.interpolate(
            support_masks_orig[k].unsqueeze(0).unsqueeze(0).float(),
            size=s_p4.shape[2:], mode="nearest"
        ).squeeze() > 0.5
        s_masks_list.append(m)
    s_mask_batch = torch.stack(s_masks_list).to(device)

    q_mask_p4 = F.interpolate(
        query_mask.unsqueeze(0).unsqueeze(0).float(),
        size=q_p4.shape[2:], mode="nearest"
    ).squeeze() > 0.5

    fg_count = s_mask_batch.sum().item()
    if fg_count < 10 or q_mask_p4.sum() < 4:
        return None

    # Standard dense matching strategies
    proto_sim_p4 = prototype_matching(s_p4, s_mask_batch, q_p4)
    dense_p4 = dense_matching(s_p4, s_mask_batch, q_p4, strategy="softmax")

    # ── Oracle Dense: query FG pixels only match support FG pixels ──
    # This is what an ideal Cross-Attention with GT guidance could achieve
    s_norm = F.normalize(s_p4, p=2, dim=1)
    q_norm = F.normalize(q_p4, p=2, dim=1)

    # Collect support FG vectors
    s_fg = []
    for k in range(shot):
        m = s_mask_batch[k] > 0.5
        if m.sum() > 0:
            s_fg.append(s_norm[k, :, m])  # [C, N_k]
    if not s_fg:
        return None
    s_fg_all = torch.cat(s_fg, dim=1)  # [C, N_total]

    # Query FG vectors
    q_fg_vecs = q_norm[0, :, q_mask_p4].permute(1, 0)  # [N_q_fg, C]

    # Oracle: for each query FG pixel, best match among support FG
    # sim_oracle[0, q_pos] = max over support FG of cos(q_vec, s_vec)
    oracle_sim = torch.zeros(1, q_p4.shape[2] * q_p4.shape[3], device=device)
    q_mask_flat = q_mask_p4.flatten()

    chunk_size = 128
    for q_start in range(0, q_fg_vecs.shape[0], chunk_size):
        q_end = min(q_start + chunk_size, q_fg_vecs.shape[0])
        q_chunk = q_fg_vecs[q_start:q_end].T  # [C, chunk]
        sim_chunk = s_fg_all.T @ q_chunk  # [N_total, chunk]
        best_sim = sim_chunk.max(dim=0)[0]  # [chunk]

        # Place back into flat array
        fg_indices = torch.where(q_mask_flat)[0][q_start:q_end]
        oracle_sim[0, fg_indices] = best_sim

    oracle_p4 = oracle_sim.reshape(1, 1, q_p4.shape[2], q_p4.shape[3])

    # Resize all to original spatial size
    proto_sim = F.interpolate(proto_sim_p4, size=(H_orig, W_orig),
                              mode="bilinear", align_corners=False).squeeze()
    dense_sim = F.interpolate(dense_p4, size=(H_orig, W_orig),
                              mode="bilinear", align_corners=False).squeeze()
    oracle_sim_resized = F.interpolate(oracle_p4, size=(H_orig, W_orig),
                                        mode="bilinear", align_corners=False).squeeze()

    return {
        "class_id": class_id,
        "proto_iou": compute_best_iou(proto_sim, query_mask)["best_iou"],
        "dense_iou": compute_best_iou(dense_sim, query_mask)["best_iou"],
        "oracle_iou": compute_best_iou(oracle_sim_resized, query_mask)["best_iou"],
    }


# ═══════════════════════════════════════════════════════════════════
# Experiment 2: Shot Scaling
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def shot_scaling_episode(
    dataset, backbone, class_id: int, max_shot: int,
    device: torch.device, rng: np.random.RandomState,
) -> dict:
    """
    同一 episode 用不同 shot 数对比 Proto vs Dense.
    Same episode, compare Proto vs Dense at different shot counts.
    """
    candidates = dataset.class_to_images(class_id)
    if len(candidates) < max_shot + 1:
        return None

    indices = rng.choice(candidates, max_shot + 1, replace=False)
    query_idx = int(indices[-1])

    # Load query once
    query_img = dataset.load_image(int(query_idx)).unsqueeze(0).to(device)
    query_mask = dataset.render_class_mask(int(query_idx), class_id).to(device)
    H_orig, W_orig = query_mask.shape

    q_feats = backbone(query_img)
    q_p4 = q_feats["p4"]

    results_by_shot = {}
    shot_list = [1, 3, 5, 10] if max_shot >= 10 else [1, 3, 5]
    shot_list = [s for s in shot_list if s <= max_shot]

    for shot in shot_list:
        support_idxs = indices[:shot]

        # Load support
        support_imgs = []
        support_masks_orig = []
        for si in support_idxs:
            img = dataset.load_image(int(si)).to(device)
            mask = dataset.render_class_mask(int(si), class_id).to(device)
            if dataset.crop_support and mask.sum() > 64:
                img, mask = dataset._roi_crop(img, mask)
            support_imgs.append(img)
            support_masks_orig.append(mask)

        support_batch = torch.stack(support_imgs)
        s_feats = backbone(support_batch)
        s_p4 = s_feats["p4"]

        s_masks_list = []
        for k in range(shot):
            m = F.interpolate(
                support_masks_orig[k].unsqueeze(0).unsqueeze(0).float(),
                size=s_p4.shape[2:], mode="nearest"
            ).squeeze() > 0.5
            s_masks_list.append(m)
        s_mask_batch = torch.stack(s_masks_list).to(device)

        if s_mask_batch.sum() < 10:
            continue

        # Proto
        proto_sim_p4 = prototype_matching(s_p4, s_mask_batch, q_p4)
        proto_sim = F.interpolate(proto_sim_p4, size=(H_orig, W_orig),
                                  mode="bilinear", align_corners=False).squeeze()
        proto_iou = compute_best_iou(proto_sim, query_mask)["best_iou"]

        # Dense Softmax
        dense_p4 = dense_matching(s_p4, s_mask_batch, q_p4, strategy="softmax")
        dense_sim = F.interpolate(dense_p4, size=(H_orig, W_orig),
                                  mode="bilinear", align_corners=False).squeeze()
        dense_iou = compute_best_iou(dense_sim, query_mask)["best_iou"]

        results_by_shot[shot] = {"proto_iou": proto_iou, "dense_iou": dense_iou}

    return {
        "class_id": class_id,
        "by_shot": results_by_shot,
    }


# ═══════════════════════════════════════════════════════════════════
# Main | 主逻辑
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Dense Matching Deep Analysis")
    p.add_argument("--tile-root", type=str, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n-episodes-per-class", type=int, default=6)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="runs/diag_dense_analysis")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ts = datetime.now().strftime("%m%d_%H%M")
    out_dir = Path(args.output_dir) / ts
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    base_classes = get_isaid5i_base_classes(args.fold)
    novel_classes = get_isaid5i_novel_classes(args.fold)
    novel_set = set(novel_classes)
    cat_names = ISAID5I_CATEGORIES

    print(f"\n{'='*70}")
    print(f"  Dense Matching Deep Analysis")
    print(f"  Exp 1: Oracle Dense Upper Bound")
    print(f"  Exp 2: Shot Scaling (1/3/5/10)")
    print(f"{'='*70}\n")

    # ── Load models ──
    print("[1] Loading backbone...")
    backbone = FastSAMBackbone(freeze_backbone=True).to(device).eval()

    # ── Load datasets ──
    print("[2] Loading datasets...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from adatile.datasets.fewshot_dataset import FewShotEpisodeDataset
    from tools.train.train_fewshot import PreCutTileAdapter

    train_tiles = PreCutTileAdapter(args.tile_root, "train")
    val_tiles = PreCutTileAdapter(args.tile_root, "val")

    base_ds = FewShotEpisodeDataset(
        train_tiles, fold=args.fold, shot=10, split="train",
        episodes_per_epoch=args.n_episodes_per_class, seed=args.seed,
        crop_support=True, crop_margin=0.2, novel_classes=base_classes,
    )
    novel_ds = FewShotEpisodeDataset(
        val_tiles, fold=args.fold, shot=10, split="val",
        episodes_per_epoch=args.n_episodes_per_class, seed=args.seed + 1,
        crop_support=False, novel_classes=novel_classes,
    )

    rng = np.random.RandomState(args.seed + 99)

    # ═════════════════════════════════════════════════════════════
    # Experiment 1: Oracle Dense
    # ═════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  Experiment 1: Oracle Dense Matching")
    print(f"{'─'*60}")

    oracle_results = []
    oracle_per_class = defaultdict(list)

    for cls_id in novel_classes:
        cls_name = cat_names.get(cls_id, f"c{cls_id}")
        ds = novel_ds
        n_candidates = len(ds.class_to_images(cls_id))
        if n_candidates < 6:
            continue

        for ep_i in tqdm(range(args.n_episodes_per_class), desc=f"  [{cls_name:>20s}]"):
            r = oracle_dense_episode(ds, backbone, cls_id, 5, device, rng)
            if r:
                oracle_results.append(r)
                oracle_per_class[cls_id].append(r)

    # Aggregate
    def avg(lst):
        return np.mean(lst) if lst else 0.0

    print(f"\n  Oracle Dense Results ({len(oracle_results)} eps):")
    print(f"  {'Method':<20s} {'IoU':>8s}")
    print(f"  {'─'*30}")
    print(f"  {'Proto (MAP)':<20s} {avg([r['proto_iou'] for r in oracle_results]):>8.4f}")
    print(f"  {'Dense-Softmax':<20s} {avg([r['dense_iou'] for r in oracle_results]):>8.4f}")
    print(f"  {'Oracle Dense':<20s} {avg([r['oracle_iou'] for r in oracle_results]):>8.4f}  ← upper bound")

    # Per-class
    print(f"\n  {'Class':<22s} {'Proto':>7s} {'Dense':>7s} {'Oracle':>7s} {'Gap':>7s}")
    print(f"  {'─'*50}")
    for cls_id in sorted(oracle_per_class.keys()):
        items = oracle_per_class[cls_id]
        p = avg([r["proto_iou"] for r in items])
        d = avg([r["dense_iou"] for r in items])
        o = avg([r["oracle_iou"] for r in items])
        print(f"  {cat_names.get(cls_id, f'c{cls_id}'):<22s} {p:>7.4f} {d:>7.4f} {o:>7.4f} {o-d:>+7.4f}")

    # ═════════════════════════════════════════════════════════════
    # Experiment 2: Shot Scaling
    # ═════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  Experiment 2: Shot Scaling")
    print(f"{'─'*60}")

    shot_results = []
    shot_per_class = defaultdict(list)
    shot_list = [1, 3, 5, 10]

    for cls_id in novel_classes:
        cls_name = cat_names.get(cls_id, f"c{cls_id}")
        ds = novel_ds
        n_candidates = len(ds.class_to_images(cls_id))
        if n_candidates < 11:
            continue

        for ep_i in tqdm(range(args.n_episodes_per_class), desc=f"  [{cls_name:>20s}]"):
            r = shot_scaling_episode(ds, backbone, cls_id, 10, device, rng)
            if r:
                shot_results.append(r)
                shot_per_class[cls_id].append(r)

    # Aggregate by shot
    print(f"\n  Shot Scaling Results ({len(shot_results)} eps):")
    print(f"  {'Shot':<8s} {'Proto':>8s} {'Dense':>8s} {'Δ':>8s} {'D/P Ratio':>10s}")
    print(f"  {'─'*45}")

    shot_agg = {}
    for shot in shot_list:
        proto_vals = []
        dense_vals = []
        for r in shot_results:
            if shot in r["by_shot"]:
                proto_vals.append(r["by_shot"][shot]["proto_iou"])
                dense_vals.append(r["by_shot"][shot]["dense_iou"])
        p_avg = avg(proto_vals)
        d_avg = avg(dense_vals)
        delta = d_avg - p_avg
        ratio = d_avg / max(p_avg, 1e-4)
        shot_agg[shot] = {"proto": p_avg, "dense": d_avg, "delta": delta, "ratio": ratio}
        print(f"  {shot:<8d} {p_avg:>8.4f} {d_avg:>8.4f} {delta:>+8.4f} {ratio:>9.2f}×")

    # ── Visualization ──
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Oracle plot
    ax = axes[0]
    methods = ["Proto\n(MAP)", "Dense\nSoftmax", "Oracle\nDense"]
    ious = [
        avg([r["proto_iou"] for r in oracle_results]),
        avg([r["dense_iou"] for r in oracle_results]),
        avg([r["oracle_iou"] for r in oracle_results]),
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(methods, ious, color=colors, width=0.5)
    for bar, val in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", fontsize=12, fontweight="bold")
    ax.set_title("Oracle Dense Matching (Novel, 5-shot)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Best IoU")
    ax.set_ylim(0, max(ious) * 1.2)

    # Shot scaling plot
    ax = axes[1]
    shots = list(shot_agg.keys())
    proto_line = [shot_agg[s]["proto"] for s in shots]
    dense_line = [shot_agg[s]["dense"] for s in shots]
    ax.plot(shots, proto_line, "o-", color="#e74c3c", linewidth=2, markersize=8, label="Proto (MAP)")
    ax.plot(shots, dense_line, "s-", color="#3498db", linewidth=2, markersize=8, label="Dense-Softmax")
    ax.set_xlabel("Shot", fontsize=12)
    ax.set_ylabel("Best IoU", fontsize=12)
    ax.set_title("Shot Scaling: Proto vs Dense (Novel)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(shots)

    plt.tight_layout()
    fig.savefig(str(vis_dir / "dense_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Verdict ──
    print(f"\n{'█'*70}")
    print(f"  FINAL VERDICT")
    print(f"{'█'*70}")

    oracle_gap = avg([r["oracle_iou"] - r["dense_iou"] for r in oracle_results])
    print(f"\n  [Oracle] Oracle Dense − Dense-Softmax = {oracle_gap:+.4f}")
    if oracle_gap > 0.15:
        print(f"    → Large headroom: trained Cross-Attn can substantially improve over raw cosine")
    elif oracle_gap > 0.05:
        print(f"    → Moderate headroom: Cross-Attn can improve but not drastically")
    else:
        print(f"    → Low headroom: raw cosine matching is near-optimal for this feature space")

    if len(shot_agg) >= 3:
        slope_dense = (dense_line[-1] - dense_line[0]) / (shots[-1] - shots[0])
        slope_proto = (proto_line[-1] - proto_line[0]) / (shots[-1] - shots[0])
        print(f"\n  [Shot Scaling] Proto slope={slope_proto:.4f}/shot, Dense slope={slope_dense:.4f}/shot")
        if slope_dense > slope_proto * 1.5:
            print(f"    → Dense benefits MORE from additional support than Proto")
            print(f"    → Dense aggregation is more sample-efficient")
        else:
            print(f"    → Both benefit similarly from additional support")

    print(f"{'█'*70}\n")

    # Save
    analysis = {
        "config": {"fold": args.fold, "n_episodes": args.n_episodes_per_class},
        "oracle": {
            "proto_mean": avg([r["proto_iou"] for r in oracle_results]),
            "dense_mean": avg([r["dense_iou"] for r in oracle_results]),
            "oracle_mean": avg([r["oracle_iou"] for r in oracle_results]),
            "oracle_headroom": oracle_gap,
            "per_class": {str(k): {
                "name": cat_names.get(k, f"c{k}"),
                "proto": avg([r["proto_iou"] for r in items]),
                "dense": avg([r["dense_iou"] for r in items]),
                "oracle": avg([r["oracle_iou"] for r in items]),
            } for k, items in oracle_per_class.items()},
        },
        "shot_scaling": {
            str(s): v for s, v in shot_agg.items()
        },
    }
    with open(out_dir / "dense_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

    print(f"  ✅ Saved → {out_dir}/\n")


if __name__ == "__main__":
    main()
