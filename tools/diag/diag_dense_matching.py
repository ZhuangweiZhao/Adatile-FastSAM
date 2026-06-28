#!/usr/bin/env python3
"""
Dense Feature Matching Diagnosis — 零训练验证 Feature Map 匹配能力
=====================================================================
Training-free diagnosis: Can FastSAM P4 feature maps do dense matching?

核心问题 | Core Question:
    Support FG pixel ↔ Query FG pixel 的 cosine similarity
    能否在 query 图上产生清晰的前景响应？

    这测试的不是 Prototype (MAP → vector) 路径，
    而是 Dense Matching (Feature Map → Feature Map) 路径。

与 Cross-Attention 的关系 | Relationship to Cross-Attention:
    Cross-Attention 本质上就是学习如何加权聚合这些 pairwise similarities。
    如果 raw cosine matching 已经能产生信号，Cross-Attention 几乎一定有效。
    如果 raw cosine matching 是纯噪声，Cross-Attention 也无从学习。

三种 Matching 策略 | Three matching strategies:
    A. Prototype (Baseline): MAP(support) → 1280 → cosine(query)
    B. Dense-Max: 每个 query position 与所有 support FG position 取 max cosine
    C. Dense-Mean: 每个 query position 对所有 support FG position 取 mean cosine

输出 | Output:
    - runs/diag_dense_match/{timestamp}/diagnosis.json
    - 对比: Proto Sim IoU vs Dense-Max Sim IoU vs Dense-Mean Sim IoU

用法 | Usage:
    python tools/diag/diag_dense_matching.py \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --fold 0 --shot 5 --device cuda
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
from adatile.utils.prototype import compute_fg_prototype
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, get_isaid5i_novel_classes, get_isaid5i_base_classes
from adatile.utils.seed import set_seed


# ═══════════════════════════════════════════════════════════════════
# Dense Matching 计算 | Dense Matching Computation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def dense_matching(
    support_p4: torch.Tensor,   # [K, C, H, W]
    support_mask: torch.Tensor,  # [K, H, W] binary, same spatial size as P4
    query_p4: torch.Tensor,      # [1, C, H_q, W_q]
    strategy: str = "max",       # "max" | "mean" | "topk"
    topk: int = 10,
) -> torch.Tensor:
    """
    Dense feature matching: support FG pixels → query pixels.
    密集特征匹配: 对每个 query 位置，与所有 support FG 像素计算 cosine 相似度。

    Strategy A (max):     每个 query 位置 = max_{support FG} cos(q_i, s_j)
    Strategy B (mean):    每个 query 位置 = mean_{support FG} cos(q_i, s_j)
    Strategy C (topk):    每个 query 位置 = mean of top-K support FG cosines
    Strategy D (softmax): 每个 query 位置 = cos(q_i, Σ_j softmax(cos(q_i,s_j)/τ)·s_j)
                          ↑ 最接近 Cross-Attention 的实际行为

    :return: [1, 1, H_q, W_q] 相似度图 (未归一化)
    """
    K, C, H, W = support_p4.shape
    _, _, Hq, Wq = query_p4.shape

    # L2-normalize all features
    s_norm = F.normalize(support_p4, p=2, dim=1)  # [K, C, H, W]
    q_norm = F.normalize(query_p4, p=2, dim=1)    # [1, C, Hq, Wq]

    # Extract support FG pixel vectors
    # support_mask: [K, H, W] → [K, 1, H, W]
    mask = support_mask.unsqueeze(1).float()  # [K, 1, H, W]

    # For each support image, extract FG feature vectors
    s_fg_list = []
    for k in range(K):
        m = support_mask[k] > 0.5  # [H, W]
        if m.sum() < 4:
            continue
        fg_vecs = s_norm[k, :, m]  # [C, N_fg]
        s_fg_list.append(fg_vecs)

    if not s_fg_list:
        # Fallback: use all support pixels
        s_fg_list = [s_norm[k].reshape(C, -1) for k in range(K)]

    s_fg = torch.cat(s_fg_list, dim=1)  # [C, N_total]
    N_total = s_fg.shape[1]

    # For memory efficiency, process query in chunks
    chunk_size = 512  # process this many query positions at a time
    q_flat = q_norm[0].reshape(C, -1)  # [C, Hq*Wq]
    N_q = q_flat.shape[1]

    sim_flat = torch.zeros(1, N_q, device=support_p4.device)

    for q_start in range(0, N_q, chunk_size):
        q_end = min(q_start + chunk_size, N_q)
        q_chunk = q_flat[:, q_start:q_end]  # [C, chunk]

        # Cosine similarity: [C, N_total]^T @ [C, chunk] = [N_total, chunk]
        sim_chunk = s_fg.T @ q_chunk  # [N_total, chunk]

        if strategy == "max":
            sim_flat[0, q_start:q_end] = sim_chunk.max(dim=0)[0]  # [chunk]
        elif strategy == "topk":
            k_eff = min(topk, N_total)
            topk_vals, _ = sim_chunk.topk(k_eff, dim=0)
            sim_flat[0, q_start:q_end] = topk_vals.mean(dim=0)
        elif strategy == "softmax":
            # 最接近 Cross-Attention: softmax-weighted sum of support → cos with query
            # Most similar to Cross-Attention: softmax-weighted support → cos with query
            tau = 0.1  # temperature (same as ProtoRefineDecoder)
            attn = (sim_chunk / tau).softmax(dim=0)  # [N_total, chunk]
            # Weighted sum of support features: [C, N_total] @ [N_total, chunk] = [C, chunk]
            s_weighted = s_fg @ attn  # [C, chunk]
            s_weighted = F.normalize(s_weighted, p=2, dim=0)  # L2-normalize
            # Cosine similarity between query and attention-weighted support
            sim_flat[0, q_start:q_end] = (q_chunk * s_weighted).sum(dim=0)  # [chunk]
        else:  # mean
            sim_flat[0, q_start:q_end] = sim_chunk.mean(dim=0)

    sim_map = sim_flat.reshape(1, 1, Hq, Wq)  # [1, 1, Hq, Wq]
    return sim_map


@torch.no_grad()
def prototype_matching(
    support_p4: torch.Tensor,   # [K, C, H, W]
    support_mask: torch.Tensor,  # [K, H, W]
    query_p4: torch.Tensor,      # [1, C, H_q, W_q]
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Prototype matching (Baseline): MAP → cosine sim with query.
    原型匹配 (基准): MAP → 与 query 逐位置 cosine 相似度。
    """
    K = support_p4.shape[0]
    s_feats_list = [support_p4[i] for i in range(K)]
    s_masks_list = [support_mask[i] for i in range(K)]

    proto = compute_fg_prototype(s_feats_list, s_masks_list)  # [C]

    q_norm = F.normalize(query_p4, p=2, dim=1)  # [1, C, Hq, Wq]
    p_norm = proto  # already L2-normalized

    sim = (q_norm * p_norm.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # [1, 1, Hq, Wq]
    sim = sim / temperature
    return sim


# ═══════════════════════════════════════════════════════════════════
# 度量计算 (复用 diag_prototype_quality 中的实现)
# ═══════════════════════════════════════════════════════════════════

def compute_best_iou(sim_map: torch.Tensor, gt_mask: torch.Tensor,
                     n_thresholds: int = 50) -> dict:
    """对相似度图做最优阈值分割 → best IoU."""
    sim = sim_map.float()
    gt = gt_mask.float()
    fg_area = gt.sum().item()
    if fg_area == 0:
        return {"best_iou": 0.0, "peak_in_gt": False, "auc": 0.0}

    # Peak in GT
    peak_idx = sim.flatten().argmax().item()
    peak_in_gt = gt.flatten()[peak_idx].item() > 0.5

    # PR curve
    sim_prob = torch.sigmoid(sim)
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best_iou, recalls, precisions = 0.0, [], []

    for t in thresholds:
        pred = (sim_prob >= t).float()
        intersection = (pred * gt).sum().item()
        union = ((pred + gt) > 0).sum().item()
        iou = intersection / union if union > 0 else 0.0

        tp = intersection
        fp = (pred * (1 - gt)).sum().item()
        fn = ((1 - pred) * gt).sum().item()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if iou > best_iou:
            best_iou = iou
        recalls.append(rec)
        precisions.append(prec)

    # PR AUC
    sorted_pairs = sorted(zip(recalls, precisions))
    auc = 0.0
    for i in range(1, len(sorted_pairs)):
        r0, p0 = sorted_pairs[i - 1]
        r1, p1 = sorted_pairs[i]
        auc += (r1 - r0) * (p0 + p1) / 2

    return {
        "best_iou": round(best_iou, 4),
        "peak_in_gt": peak_in_gt,
        "auc": round(auc, 4),
    }


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

def colormap_sim(sim: np.ndarray) -> np.ndarray:
    vmin, vmax = sim.min(), sim.max()
    if vmax - vmin < 1e-8:
        normalized = np.zeros_like(sim)
    else:
        normalized = (sim - vmin) / (vmax - vmin)
    colored = plt.cm.jet(normalized)[:, :, :3]
    return (colored * 255).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.tile(image, (1, 1, 3))
    overlay = image.copy()
    mask_bool = mask > 0.5
    for c in range(3):
        overlay[:, :, c][mask_bool] = (
            overlay[:, :, c][mask_bool] * (1 - alpha) + color[c] * alpha
        ).astype(np.uint8)
    return overlay


def make_comparison_figure(
    query_img_np, gt_np,
    proto_sim_np, dense_max_sim_np, dense_mean_sim_np,
    class_name: str, ep_idx: int,
    proto_iou: float, dense_iou: float,
) -> plt.Figure:
    """6-panel comparison: Query+GT | Proto Sim | Dense-Max | Dense-Mean | Best Dense Threshold | Threshold Pred"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1
    axes[0, 0].imshow(overlay_mask(query_img_np, gt_np))
    axes[0, 0].set_title("Query + GT", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(proto_sim_np, cmap="jet")
    axes[0, 1].set_title(f"Proto Sim (bestIoU={proto_iou:.3f})", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(dense_max_sim_np, cmap="jet")
    axes[0, 2].set_title(f"Dense-Max Sim (bestIoU={dense_iou:.3f})", fontsize=11)
    axes[0, 2].axis("off")

    # Row 2
    axes[1, 0].imshow(dense_mean_sim_np, cmap="jet")
    axes[1, 0].set_title("Dense-Mean Sim", fontsize=11)
    axes[1, 0].axis("off")

    # Best threshold of Dense-Max
    best_t = _find_best_threshold(dense_max_sim_np, gt_np)
    axes[1, 1].imshow(dense_max_sim_np >= best_t, cmap="gray")
    axes[1, 1].set_title(f"Dense-Max @ best thresh={best_t:.2f}", fontsize=11)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(overlay_mask(query_img_np, (dense_max_sim_np >= best_t).astype(float), color=(255, 0, 0)))
    axes[1, 2].set_title("Dense Pred (red) + GT (green)", fontsize=11)
    axes[1, 2].axis("off")

    fig.suptitle(f"{class_name} Ep{ep_idx} | Proto={proto_iou:.3f} Dense={dense_iou:.3f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def _find_best_threshold(sim_np: np.ndarray, gt_np: np.ndarray) -> float:
    best_t, best_iou = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 20):
        pred = (sim_np >= t).astype(float)
        inter = (pred * gt_np).sum()
        union = ((pred + gt_np) > 0).sum()
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best_t = iou, t
    return round(best_t, 2)


# ═══════════════════════════════════════════════════════════════════
# Episode 执行 | Episode Execution
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_dense_episode(
    dataset, backbone, class_id: int, shot: int,
    device: torch.device, rng: np.random.RandomState,
) -> dict:
    """执行 episode: 对比 Proto vs Dense-Max vs Dense-Mean."""

    candidates = dataset.class_to_images(class_id)
    if len(candidates) < shot + 1:
        return None

    indices = rng.choice(candidates, shot + 1, replace=False)
    support_idxs = indices[:shot]
    query_idx = int(indices[shot])

    # Load support
    support_imgs, support_masks_p4 = [], []
    for si in support_idxs:
        img = dataset.load_image(int(si)).to(device)
        mask_orig = dataset.render_class_mask(int(si), class_id).to(device)
        if dataset.crop_support and mask_orig.sum() > 64:
            img, mask_orig = dataset._roi_crop(img, mask_orig)
        support_imgs.append(img)
        support_masks_p4.append(mask_orig)

    support_batch = torch.stack(support_imgs)  # [K, 3, H, W]

    # Load query
    query_img = dataset.load_image(int(query_idx)).unsqueeze(0).to(device)
    query_mask_orig = dataset.render_class_mask(int(query_idx), class_id).to(device)
    H_orig, W_orig = query_mask_orig.shape

    # Backbone
    s_feats = backbone(support_batch)
    q_feats = backbone(query_img)

    s_p4 = s_feats["p4"]  # [K, 1280, H/16, W/16]
    q_p4 = q_feats["p4"]  # [1, 1280, H/16, W/16]

    # Resize support masks to P4 spatial size
    s_masks_resized = []
    for k in range(shot):
        m = F.interpolate(
            support_masks_p4[k].unsqueeze(0).unsqueeze(0).float(),
            size=s_p4.shape[2:], mode="nearest"
        ).squeeze() > 0.5
        s_masks_resized.append(m)
    s_mask_batch = torch.stack(s_masks_resized).to(device)  # [K, H_p4, W_p4]

    fg_count = s_mask_batch.sum().item()
    if fg_count < 10:
        return None

    # ── Strategy A: Prototype Matching ──
    proto_sim_p4 = prototype_matching(s_p4, s_mask_batch, q_p4)

    # ── Strategy B: Dense-Max ──
    dense_max_p4 = dense_matching(s_p4, s_mask_batch, q_p4, strategy="max")

    # ── Strategy C: Dense-Mean ──
    dense_mean_p4 = dense_matching(s_p4, s_mask_batch, q_p4, strategy="mean")

    # ── Strategy D: Dense-Softmax ── (closest to Cross-Attention)
    dense_softmax_p4 = dense_matching(s_p4, s_mask_batch, q_p4, strategy="softmax")

    # Resize to original spatial size
    proto_sim = F.interpolate(proto_sim_p4, size=(H_orig, W_orig),
                              mode="bilinear", align_corners=False).squeeze()
    dense_max_sim = F.interpolate(dense_max_p4, size=(H_orig, W_orig),
                                  mode="bilinear", align_corners=False).squeeze()
    dense_mean_sim = F.interpolate(dense_mean_p4, size=(H_orig, W_orig),
                                   mode="bilinear", align_corners=False).squeeze()
    dense_softmax_sim = F.interpolate(dense_softmax_p4, size=(H_orig, W_orig),
                                       mode="bilinear", align_corners=False).squeeze()

    # Metrics
    proto_metrics = compute_best_iou(proto_sim, query_mask_orig)
    dense_max_metrics = compute_best_iou(dense_max_sim, query_mask_orig)
    dense_mean_metrics = compute_best_iou(dense_mean_sim, query_mask_orig)
    dense_softmax_metrics = compute_best_iou(dense_softmax_sim, query_mask_orig)

    # Visualization data
    def to_display(t):
        return np.clip(t.cpu().permute(1, 2, 0).numpy(), 0, 1)

    return {
        "class_id": class_id,
        "proto": proto_metrics,
        "dense_max": dense_max_metrics,
        "dense_mean": dense_mean_metrics,
        "dense_softmax": dense_softmax_metrics,
        "fg_pixels": int(query_mask_orig.sum().item()),
        "query_img": to_display(query_img.squeeze(0)),
        "gt_mask": query_mask_orig.cpu().numpy(),
        "proto_sim": proto_sim.cpu().numpy(),
        "dense_max_sim": dense_max_sim.cpu().numpy(),
        "dense_mean_sim": dense_mean_sim.cpu().numpy(),
        "dense_softmax_sim": dense_softmax_sim.cpu().numpy(),
    }


# ═══════════════════════════════════════════════════════════════════
# Main | 主逻辑
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Dense Feature Matching Diagnosis")
    p.add_argument("--tile-root", type=str, required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--shot", type=int, default=5)
    p.add_argument("--n-episodes-per-class", type=int, default=8)
    p.add_argument("--novel-only", action="store_true")
    p.add_argument("--save-vis", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="runs/diag_dense_match")
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

    target_classes = novel_classes if args.novel_only else (base_classes + novel_classes)
    class_type = "NOVEL" if args.novel_only else "ALL"

    print(f"\n{'='*70}")
    print(f"  Dense Feature Matching Diagnosis | 密集特征匹配诊断")
    print(f"  {'─'*60}")
    print(f"  Fold: {args.fold}, Shot: {args.shot}, Classes: {len(target_classes)} ({class_type})")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}\n")

    # ── Load backbone ──
    print("[1] Loading FastSAM backbone...")
    backbone = FastSAMBackbone(freeze_backbone=True).to(device).eval()

    # ── Load dataset ──
    print("[2] Loading dataset...")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from adatile.datasets.fewshot_dataset import FewShotEpisodeDataset
    from tools.train.train_fewshot import PreCutTileAdapter

    train_tiles = PreCutTileAdapter(args.tile_root, "train")
    val_tiles = PreCutTileAdapter(args.tile_root, "val")

    base_ds = FewShotEpisodeDataset(
        train_tiles, fold=args.fold, shot=args.shot, split="train",
        episodes_per_epoch=args.n_episodes_per_class, seed=args.seed,
        crop_support=True, crop_margin=0.2, novel_classes=base_classes,
    )
    novel_ds = FewShotEpisodeDataset(
        val_tiles, fold=args.fold, shot=args.shot, split="val",
        episodes_per_epoch=args.n_episodes_per_class, seed=args.seed + 1,
        crop_support=False, novel_classes=novel_classes,
    )

    # ── Run episodes ──
    print(f"\n[3] Running dense matching episodes...")
    rng = np.random.RandomState(args.seed + 99)
    all_results = []
    per_class = defaultdict(lambda: {
        "proto_ious": [], "dense_max_ious": [], "dense_mean_ious": [],
    })

    for cls_id in target_classes:
        cls_name = cat_names.get(cls_id, f"c{cls_id}")
        ds = novel_ds if cls_id in novel_set else base_ds

        n_candidates = len(ds.class_to_images(cls_id))
        if n_candidates < args.shot + 1:
            print(f"  [{cls_name}] SKIP: {n_candidates} tiles")
            continue

        cls_results = []
        n_success = 0
        pbar = tqdm(range(args.n_episodes_per_class), desc=f"  [{cls_name:>20s}]", unit="ep")

        for ep_i in pbar:
            result = run_dense_episode(ds, backbone, cls_id, args.shot, device, rng)
            if result is None:
                continue

            cls_results.append(result)
            n_success += 1

            if n_success <= args.save_vis:
                fig = make_comparison_figure(
                    result["query_img"], result["gt_mask"],
                    result["proto_sim"], result["dense_max_sim"], result["dense_mean_sim"],
                    cls_name, ep_i,
                    result["proto"]["best_iou"], result["dense_max"]["best_iou"],
                )
                fig.savefig(vis_dir / f"{cls_name}_ep{ep_i:02d}.png", dpi=100, bbox_inches="tight")
                plt.close(fig)

            avg_proto = np.mean([r["proto"]["best_iou"] for r in cls_results])
            avg_dense = np.mean([r["dense_max"]["best_iou"] for r in cls_results])
            pbar.set_postfix(Proto=f"{avg_proto:.3f}", Dense=f"{avg_dense:.3f}")

        if cls_results:
            proto_ious = [r["proto"]["best_iou"] for r in cls_results]
            dense_ious = [r["dense_max"]["best_iou"] for r in cls_results]
            softmax_ious = [r["dense_softmax"]["best_iou"] for r in cls_results]
            per_class[cls_id] = {
                "name": cls_name,
                "n_episodes": len(cls_results),
                "proto_iou_mean": round(np.mean(proto_ious), 4),
                "dense_max_iou_mean": round(np.mean(dense_ious), 4),
                "dense_mean_iou_mean": round(np.mean([r["dense_mean"]["best_iou"] for r in cls_results]), 4),
                "dense_softmax_iou_mean": round(np.mean(softmax_ious), 4),
                "proto_peak_gt": round(np.mean([r["proto"]["peak_in_gt"] for r in cls_results]), 4),
                "dense_peak_gt": round(np.mean([r["dense_max"]["peak_in_gt"] for r in cls_results]), 4),
            }
            all_results.extend(cls_results)

    # ── Output ──
    print(f"\n{'='*70}")
    print(f"  DENSE MATCHING RESULTS | 密集匹配结果")
    print(f"{'='*70}")

    def avg(lst):
        return np.mean(lst) if lst else 0.0

    base_results = [r for r in all_results if r["class_id"] not in novel_set]
    novel_results = [r for r in all_results if r["class_id"] in novel_set]

    for label, results in [("BASE", base_results), ("NOVEL", novel_results)]:
        if not results:
            continue
        p = avg([r["proto"]["best_iou"] for r in results])
        d = avg([r["dense_max"]["best_iou"] for r in results])
        dm = avg([r["dense_mean"]["best_iou"] for r in results])
        ds = avg([r["dense_softmax"]["best_iou"] for r in results])
        pp = avg([r["proto"]["peak_in_gt"] for r in results])
        dp = avg([r["dense_max"]["peak_in_gt"] for r in results])
        gain = d - p
        gain_softmax = ds - p
        print(f"\n  ── {label} ({len(results)} eps) ──")
        print(f"  Proto bestIoU:         {p:.4f}  (peak={pp:.1%})")
        print(f"  Dense-Max bestIoU:     {d:.4f}  (peak={dp:.1%})  ← upper bound")
        print(f"  Dense-Softmax bestIoU: {ds:.4f}                    ← ≈Cross-Attn")
        print(f"  Dense-Mean bestIoU:    {dm:.4f}")
        print(f"  Δ Max−Proto:    {gain:+.4f}")
        print(f"  Δ Softmax−Proto:{gain_softmax:+.4f}")
        if gain_softmax > 0.03:
            print(f"  ✅ Dense-Softmax SIGNIFICANTLY better — Cross-Attn almost certain to help")
        elif gain_softmax > 0.01:
            print(f"  🟡 Dense-Softmax moderately better — Cross-Attn worth trying")
        else:
            print(f"  🔴 Dense-Softmax NOT better — even spatial correspondence is weak")

    # Per-class
    print(f"\n  {'Class':<22s} {'Type':<6s} {'Proto':>7s} {'DenseM':>7s} {'SoftMx':>7s} {'ΔMax':>7s} {'ΔSoft':>7s}")
    print(f"  {'─'*70}")
    for cls_id in sorted(target_classes):
        if cls_id not in per_class:
            continue
        c = per_class[cls_id]
        ct = "NOVEL" if cls_id in novel_set else "BASE"
        delta_max = c["dense_max_iou_mean"] - c["proto_iou_mean"]
        delta_soft = c["dense_softmax_iou_mean"] - c["proto_iou_mean"]
        print(f"  {c['name']:<22s} {ct:<6s} "
              f"{c['proto_iou_mean']:>7.4f} {c['dense_max_iou_mean']:>7.4f} "
              f"{c['dense_softmax_iou_mean']:>7.4f} {delta_max:>+7.4f} {delta_soft:>+7.4f}")

    # ── Verdict ──
    print(f"\n{'█'*70}")
    print(f"  VERDICT | 诊断结论")
    print(f"{'█'*70}")

    novel_gain = avg([r["dense_max"]["best_iou"] - r["proto"]["best_iou"]
                       for r in novel_results]) if novel_results else 0.0
    novel_gain_soft = avg([r["dense_softmax"]["best_iou"] - r["proto"]["best_iou"]
                            for r in novel_results]) if novel_results else 0.0

    if novel_gain_soft > 0.05:
        print(f"  ✅ Dense-Softmax provides SIGNIFICANT gain over prototype")
        print(f"     (Novel ΔSoft={novel_gain_soft:+.3f})")
        print(f"     → Cross-Attention is strongly validated by this diagnostic.")
    elif novel_gain_soft > 0.02:
        print(f"  🟡 Dense-Softmax provides MODERATE gain (Novel ΔSoft={novel_gain_soft:+.3f})")
        print(f"     → Cross-Attention worth trying but not guaranteed.")
    else:
        print(f"  🔴 Dense-Softmax does NOT improve over prototype")
        print(f"     (Novel ΔSoft={novel_gain_soft:+.3f})")
        print(f"     → Even dense spatial correspondence is weak in FastSAM P4.")
        print(f"     → Consider: LoRA fine-tune backbone to improve feature matching quality.")

    print(f"{'█'*70}\n")

    # Save
    diagnosis = {
        "config": {"fold": args.fold, "shot": args.shot},
        "summary": {
            "base": {"proto_iou": avg([r["proto"]["best_iou"] for r in base_results]),
                      "dense_max_iou": avg([r["dense_max"]["best_iou"] for r in base_results]),
                      "dense_softmax_iou": avg([r["dense_softmax"]["best_iou"] for r in base_results])},
            "novel": {"proto_iou": avg([r["proto"]["best_iou"] for r in novel_results]),
                       "dense_max_iou": avg([r["dense_max"]["best_iou"] for r in novel_results]),
                       "dense_softmax_iou": avg([r["dense_softmax"]["best_iou"] for r in novel_results])},
            "novel_softmax_gain": novel_gain_soft,
            "base_softmax_gain": avg([r["dense_softmax"]["best_iou"] - r["proto"]["best_iou"]
                                       for r in base_results]) if base_results else 0.0,
        },
        "per_class": {str(k): v for k, v in per_class.items()},
    }
    with open(out_dir / "diagnosis.json", "w") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False, default=str)

    print(f"  ✅ Saved → {out_dir}/\n")


if __name__ == "__main__":
    main()
