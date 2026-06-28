#!/usr/bin/env python3
"""
Prototype Quality Diagnosis — Similarity Map vs Decoder
========================================================
诊断 Prototype 质量：相似度图能否定位目标？Decoder 是否有效利用了 Prototype？

Diagnosis: Can the similarity map localize objects?
           Is the Decoder effectively using the Prototype?

核心诊断逻辑 | Core Diagnosis Logic:
    1. 手动计算 query P4 ⊙ prototype 的 cosine similarity map
    2. 对 similarity map 做最优阈值分割，得到 best_sim_IoU（Prototype 的上限）
    3. 对比 Decoder 输出的 pred_IoU
    4. Δ = best_sim_IoU - pred_IoU:
       - Δ >> 0  → Decoder 是瓶颈（prototype 信息足够，decoder 没用好）
       - Δ ≈ 0, both low → Prototype 是瓶颈（相似度图本身就是噪声）
       - Δ ≈ 0, both high → 都工作良好

用法 | Usage:
    # 使用最新训练的 checkpoint
    python tools/diag/diag_prototype_quality.py \
        --ckpt runs/fewshot_f0_k5_0628_1935/decoder_p3p4film_5shot_best.pt \
        --tile-root /root/autodl-tmp/iSAID5i_tiles/tile_896 \
        --fold 0 --shot 5 --device cuda

    # 只诊断 novel 类
    python tools/diag/diag_prototype_quality.py \
        --ckpt runs/.../decoder_p3p4film_5shot_best.pt \
        --tile-root ... --fold 0 --shot 5 \
        --novel-only --device cuda

输出 | Output:
    - runs/diag_proto_quality/{timestamp}/diagnosis.json  — 详细诊断数据
    - runs/diag_proto_quality/{timestamp}/vis/            — 可视化 grid 图
    - 终端输出：清晰的瓶颈判断
"""

import sys, json, argparse, os
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
from adatile.utils.prototype import compute_fg_prototype
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, get_isaid5i_novel_classes, get_isaid5i_base_classes
from adatile.utils.seed import set_seed


# ═══════════════════════════════════════════════════════════════════
# Decoder 构建（从 c04 复制，保持与训练一致）
# Decoder builder (copied from c04, keep consistent with training)
# ═══════════════════════════════════════════════════════════════════

def build_decoder_for_diag(decoder_type: str, feat_dim: int = 1280, num_prototypes: int = 1):
    """构建 Decoder | Build decoder (mirrors eval_c04_full_fewshot.build_decoder)."""
    # 动态导入避免循环依赖 | Dynamic import to avoid circular deps
    sys.path.insert(0, str(_PROJECT_ROOT / "tools" / "instance"))
    from eval_c04_full_fewshot import build_decoder as _build
    return _build(decoder_type, feat_dim=feat_dim, num_prototypes=num_prototypes)


# ═══════════════════════════════════════════════════════════════════
# Similarity Map 计算 | Similarity Map Computation
# ═══════════════════════════════════════════════════════════════════

def compute_similarity_map(query_p4: torch.Tensor, prototype: torch.Tensor,
                           temperature: float = 0.1) -> torch.Tensor:
    """
    计算 query P4 特征与 prototype 的 cosine similarity map。
    Compute cosine similarity map between query P4 features and prototype.

    与 ProtoRefineDecoder 中的计算方式完全一致：
    Identical to the computation in ProtoRefineDecoder.

    :param query_p4: [1, C, H, W] query P4 features
    :param prototype: [C] or [K, C] prototype(s), L2-normalized
    :param temperature: similarity temperature (default 0.1, same as ProtoRefineDecoder)
    :return: [1, 1, H, W] similarity map (higher = more similar)
    """
    # L2-normalize query features
    q_norm = F.normalize(query_p4, p=2, dim=1)  # [1, C, H, W]

    if prototype.dim() == 1:
        # 单原型 | Single prototype: [C]
        p_norm = prototype  # already L2-normalized
        sim = (q_norm * p_norm.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    else:
        # 多原型 | Multi-prototype: [K, C]
        # einsum: bchw,kc -> bkhw, then max over K
        sim = torch.einsum("bchw,kc->bkhw", q_norm, prototype)  # [1, K, H, W]
        sim = sim.max(dim=1, keepdim=True)[0]  # [1, 1, H, W]

    sim = sim / temperature
    return sim  # [1, 1, H, W]


def compute_similarity_metrics(sim_map: torch.Tensor, gt_mask: torch.Tensor,
                               n_thresholds: int = 50) -> dict:
    """
    计算 similarity map 相对于 GT 的各种指标。
    Compute similarity map metrics against ground truth.

    :param sim_map: [H, W] similarity scores (raw, before sigmoid)
    :param gt_mask: [H, W] binary ground truth (0/1)
    :return: dict with keys:
        - peak_in_gt: bool, 最大相似度是否在 GT 内
        - best_iou: float, 最优阈值下的 best IoU
        - best_thresh: float, 最优阈值 (sigmoid 后的值)
        - best_recall: float, 最优阈值下的 recall
        - best_precision: float, 最优阈值下的 precision
        - auc: float, PR 曲线下面积 (higher = better score map)
        - thresholds: list[float], 采样的阈值
        - ious: list[float], 各阈值的 IoU
    """
    sim = sim_map.float()
    gt = gt_mask.float()

    # Flatten
    sim_flat = sim.flatten()
    gt_flat = gt.flatten()
    fg_area = gt_flat.sum().item()

    if fg_area == 0:
        return {
            "peak_in_gt": False, "best_iou": 0.0, "best_thresh": 0.0,
            "best_recall": 0.0, "best_precision": 0.0, "auc": 0.0,
            "thresholds": [], "ious": [],
        }

    # ── Peak in GT? ──
    peak_idx = sim_flat.argmax().item()
    peak_in_gt = gt_flat[peak_idx].item() > 0.5

    # ── PR curve & best IoU ──
    # 对 similarity 做 sigmoid 得到概率 | Apply sigmoid to get probabilities
    sim_prob = torch.sigmoid(sim)

    # 采样阈值 | Sample thresholds
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best_iou, best_thresh, best_rec, best_prec = 0.0, 0.0, 0.0, 0.0
    precisions, recalls, ious_list = [], [], []

    for t in thresholds:
        pred = (sim_prob >= t).float()
        pred_flat = pred.flatten()
        intersection = (pred_flat * gt_flat).sum().item()
        union = ((pred_flat + gt_flat) > 0).sum().item()
        iou = intersection / union if union > 0 else 0.0

        tp = intersection
        fp = (pred_flat * (1 - gt_flat)).sum().item()
        fn = ((1 - pred_flat) * gt_flat).sum().item()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        ious_list.append(iou)
        precisions.append(prec)
        recalls.append(rec)

        if iou > best_iou:
            best_iou, best_thresh, best_rec, best_prec = iou, t, rec, prec

    # ── PR AUC (trapezoidal) ──
    # Sort by recall ascending
    sorted_pairs = sorted(zip(recalls, precisions))
    auc = 0.0
    for i in range(1, len(sorted_pairs)):
        r0, p0 = sorted_pairs[i - 1]
        r1, p1 = sorted_pairs[i]
        auc += (r1 - r0) * (p0 + p1) / 2

    return {
        "peak_in_gt": peak_in_gt,
        "best_iou": round(best_iou, 4),
        "best_thresh": round(best_thresh, 3),
        "best_recall": round(best_rec, 4),
        "best_precision": round(best_prec, 4),
        "auc": round(auc, 4),
        "thresholds": thresholds.tolist(),
        "ious": ious_list,
    }


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

def colormap_sim(sim: np.ndarray) -> np.ndarray:
    """相似度 → RGB 热力图 (红=高, 蓝=低) | Similarity → RGB heatmap (red=high, blue=low)."""
    import matplotlib.cm as cm
    vmin, vmax = sim.min(), sim.max()
    if vmax - vmin < 1e-8:
        normalized = np.zeros_like(sim)
    else:
        normalized = (sim - vmin) / (vmax - vmin)
    colored = cm.jet(normalized)[:, :, :3]  # RGBA -> RGB
    return (colored * 255).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.4):
    """将 mask 半透明叠加到图像上 | Overlay mask on image with transparency."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.shape[2] == 1:
        image = np.tile(image, (1, 1, 3))

    overlay_img = image.copy()
    mask_bool = mask > 0.5
    for c in range(3):
        ch = overlay_img[:, :, c].astype(np.float32)
        ch[mask_bool] = ch[mask_bool] * (1 - alpha) + color[c] * alpha
        overlay_img[:, :, c] = ch.astype(np.uint8)
    return overlay_img


def make_episode_figure(support_img: np.ndarray, support_mask: np.ndarray,
                         query_img: np.ndarray, gt_mask: np.ndarray,
                         sim_map: np.ndarray, pred_mask: np.ndarray,
                         class_name: str, shot: int, ep_idx: int,
                         sim_metrics: dict, pred_iou: float) -> np.ndarray:
    """生成单个 episode 的诊断图 | Generate diagnosis figure for one episode.

    布局 | Layout (2 rows × 3 cols):
        Row 1: Support Image | Support Mask     | Query + GT Overlay
        Row 2: Similarity Map | Prediction       | Pred + GT Overlay
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"{class_name} | Sim bestIoU={sim_metrics['best_iou']:.3f} "
        f"Pred IoU={pred_iou:.3f} | Δ={sim_metrics['best_iou'] - pred_iou:+.3f}",
        fontsize=14, fontweight="bold",
    )

    # Row 1
    axes[0, 0].imshow(support_img)
    axes[0, 0].set_title("Support Image", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(support_mask, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Support Mask", fontsize=11)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(overlay_mask(query_img, gt_mask, color=(0, 255, 0)))
    axes[0, 2].set_title("Query + GT (green)", fontsize=11)
    axes[0, 2].axis("off")

    # Row 2
    axes[1, 0].imshow(sim_map, cmap="jet")
    axes[1, 0].set_title(
        f"Similarity Map (bestIoU={sim_metrics['best_iou']:.3f})", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Prediction (IoU={pred_iou:.3f})", fontsize=11)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(overlay_mask(query_img, pred_mask, color=(255, 0, 0)))
    axes[1, 2].set_title("Pred (red) + GT (green)", fontsize=11)
    axes[1, 2].axis("off")

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# Episode 执行 | Episode Execution
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_diagnosis_episode(
    dataset, backbone, decoder, class_id: int, shot: int,
    feature_level: str, device: torch.device, num_proto: int = 1,
    rng: np.random.RandomState = None,
    p3_adapter: nn.Module = None, p4_adapter: nn.Module = None,
) -> dict:
    """
    执行单个诊断 episode | Run a single diagnosis episode.

    同时计算：
    - Similarity map (手动 cosine sim，不依赖 decoder)
    - Decoder prediction

    使用与训练完全一致的采样逻辑 | Uses same sampling logic as training.

    :return: dict with all metrics, features, and masks
    """
    if rng is None:
        rng = np.random.RandomState()

    # ── 采样 episode（与 FewShotEpisodeDataset.sample_episode 逻辑一致）
    # Sample episode (same logic as FewShotEpisodeDataset.sample_episode)
    candidates = dataset.class_to_images(class_id)
    if len(candidates) < shot + 1:
        return None

    indices = rng.choice(candidates, shot + 1, replace=False)
    support_idxs = indices[:shot]
    query_idx = int(indices[shot])

    # ── Load support ──
    support_imgs, support_masks = [], []
    for si in support_idxs:
        img = dataset.load_image(int(si))
        mask = dataset.render_class_mask(int(si), class_id)
        # ROI crop for train split | 训练集支持图像做 ROI 裁剪
        if dataset.crop_support and mask.sum() > 64:
            img, mask = dataset._roi_crop(img, mask)
        support_imgs.append(img)
        support_masks.append(mask)

    support_batch = torch.stack(support_imgs).to(device)  # [K, 3, H, W]

    # ── Load query ──
    query_img = dataset.load_image(int(query_idx)).unsqueeze(0).to(device)
    query_mask = dataset.render_class_mask(int(query_idx), class_id)
    H_orig, W_orig = query_mask.shape

    # ── Backbone: support → prototype ──
    s_feats = backbone(support_batch)
    if feature_level == "p3p4":
        s_feats_list = [s_feats["p4"][i] for i in range(len(support_imgs))]
    else:
        s_feats_list = [s_feats[feature_level][i] for i in range(len(support_imgs))]

    if num_proto > 1:
        from tools.instance.eval_c03_catsam_fewshot import compute_multi_prototype
        proto = compute_multi_prototype(s_feats_list, support_masks, num_prototypes=num_proto)
    else:
        proto = compute_fg_prototype(s_feats_list, support_masks)

    fg_count = sum(m.sum().item() for m in support_masks)
    if fg_count < 10:
        return None  # 前景太少，跳过 | Too little foreground, skip

    # ── Backbone: query ──
    q_feats = backbone(query_img)
    q_p4 = q_feats["p4"]  # [1, 1280, H/16, W/16]

    # ── Similarity Map (manual) ──
    sim_map_raw = compute_similarity_map(q_p4, proto, temperature=0.1)  # [1, 1, h, w]

    # Resize similarity to original size for metric computation
    sim_map_resized = F.interpolate(
        sim_map_raw, size=(H_orig, W_orig), mode="bilinear", align_corners=False
    ).squeeze()  # [H_orig, W_orig]

    sim_metrics = compute_similarity_metrics(sim_map_resized, query_mask)

    # ── Decoder prediction ──
    # 应用 adapter 处理维度不匹配 | Apply adapter for dimension mismatch
    q_p3 = q_feats["p3"]
    q_p4_dec = q_feats["p4"]
    if p3_adapter is not None:
        q_p3 = p3_adapter(q_p3)
    if p4_adapter is not None:
        q_p4_dec = p4_adapter(q_p4_dec)

    if feature_level == "p3p4":
        logit = decoder(q_p3, q_p4_dec, proto, target_size=(H_orig, W_orig))
    else:
        logit = decoder(q_p4_dec, proto, target_size=(H_orig, W_orig))

    pred_mask = (torch.sigmoid(logit.squeeze()) > 0.5).float().cpu()  # [H_orig, W_orig]

    # Pred IoU
    intersection = (pred_mask * query_mask).sum().item()
    union = ((pred_mask + query_mask) > 0).sum().item()
    pred_iou = intersection / union if union > 0 else 0.0

    # ── 准备可视化数据 | Prepare visualization data ──
    def to_display(img_tensor):
        img = img_tensor.cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        return img

    return {
        "class_id": class_id,
        "pred_iou": round(pred_iou, 4),
        "sim_metrics": sim_metrics,
        "delta": round(sim_metrics["best_iou"] - pred_iou, 4),
        "fg_pixels": int(query_mask.sum().item()),
        # Visualization data (numpy)
        "support_img": to_display(support_imgs[0]),
        "support_mask": support_masks[0].cpu().numpy(),
        "query_img": to_display(query_img.squeeze(0)),
        "gt_mask": query_mask.cpu().numpy(),
        "sim_map": sim_map_resized.cpu().numpy(),
        "pred_mask": pred_mask.numpy(),
    }


# ═══════════════════════════════════════════════════════════════════
# 主逻辑 | Main Logic
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Prototype Quality Diagnosis")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Decoder checkpoint path (.pt)")
    p.add_argument("--tile-root", type=str, required=True,
                   help="Path to pre-cut tiles")
    p.add_argument("--fold", type=int, default=0,
                   help="iSAID-5i fold (0/1/2)")
    p.add_argument("--shot", type=int, default=5,
                   help="Shot number")
    p.add_argument("--decoder", type=str, default="p3p4film",
                   help="Decoder type used during training")
    p.add_argument("--feature-level", type=str, default="p3p4",
                   help="Feature level: p4, p3p4")
    p.add_argument("--num-proto", type=int, default=1,
                   help="Number of prototypes")
    p.add_argument("--n-episodes-per-class", type=int, default=8,
                   help="Episodes per class for diagnosis")
    p.add_argument("--novel-only", action="store_true",
                   help="Only diagnose novel classes")
    p.add_argument("--base-only", action="store_true",
                   help="Only diagnose base classes")
    p.add_argument("--save-vis", type=int, default=3,
                   help="Number of episodes per class to save visualization (0=none)")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--output-dir", type=str, default="runs/diag_proto_quality",
                   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── 输出目录 | Output directory ──
    ts = datetime.now().strftime("%m%d_%H%M")
    out_dir = Path(args.output_dir) / ts
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Prototype Quality Diagnosis | 原型质量诊断")
    print(f"  {'─'*60}")
    print(f"  Checkpoint:     {args.ckpt}")
    print(f"  Tile root:      {args.tile_root}")
    print(f"  Fold:           {args.fold}")
    print(f"  Shot:           {args.shot}")
    print(f"  Decoder type:   {args.decoder}")
    print(f"  Feature level:  {args.feature_level}")
    print(f"  Num proto:      {args.num_proto}")
    print(f"  Output:         {out_dir}")
    print(f"{'='*70}\n")

    # ── 加载模型 | Load models ──
    print("[1/5] Loading models...")
    backbone = FastSAMBackbone(freeze_backbone=True).to(device).eval()

    # 探测 backbone 实际输出维度 | Probe backbone actual output dims
    with torch.no_grad():
        probe_feats = backbone(torch.randn(1, 3, 896, 896).to(device))
        actual_p3_dim = probe_feats["p3"].shape[1]
        actual_p4_dim = probe_feats["p4"].shape[1]
        print(f"       Backbone P3 dim: {actual_p3_dim}, P4 dim: {actual_p4_dim}")

    # 从 checkpoint 自动检测维度 | Auto-detect dimensions from checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)

    if "proj_p3.0.weight" in ckpt:
        ckpt_p3_dim = ckpt["proj_p3.0.weight"].shape[1]  # [out, in, 1, 1]
        print(f"       Checkpoint P3 dim: {ckpt_p3_dim}")
    else:
        ckpt_p3_dim = actual_p3_dim

    if "proj_p4.0.weight" in ckpt:
        ckpt_p4_dim = ckpt["proj_p4.0.weight"].shape[1]
        print(f"       Checkpoint P4 dim: {ckpt_p4_dim}")
    else:
        ckpt_p4_dim = actual_p4_dim

    # 构建 decoder 并加载权重 | Build decoder and load weights
    if args.decoder == "p3p4film":
        from tools.instance.eval_c04_full_fewshot import P3P4FiLMFusionDecoder
        decoder = P3P4FiLMFusionDecoder(
            feat_dim_p3=ckpt_p3_dim, feat_dim_p4=ckpt_p4_dim
        ).to(device)
    else:
        feat_dim = 640 if args.feature_level == "p3" else 1280
        decoder = build_decoder_for_diag(args.decoder, feat_dim=feat_dim,
                                          num_prototypes=args.num_proto).to(device)

    decoder.load_state_dict(ckpt)
    decoder.eval()

    # 如果 backbone P3 ≠ checkpoint P3，添加投影适配器
    # If backbone P3 ≠ checkpoint P3, add projection adapter
    p3_adapter = None
    if actual_p3_dim != ckpt_p3_dim:
        print(f"       ⚠ P3 dim mismatch: backbone={actual_p3_dim} vs ckpt={ckpt_p3_dim}")
        print(f"       Adding projection adapter {actual_p3_dim}→{ckpt_p3_dim}")
        p3_adapter = nn.Conv2d(actual_p3_dim, ckpt_p3_dim, 1, bias=False).to(device)
        # 如果 ckpt 维度是 backbone+其他，尝试用 identity 初始化部分
        # If ckpt dim is backbone+something else, try partial identity init
        nn.init.kaiming_normal_(p3_adapter.weight)

    p4_adapter = None
    if actual_p4_dim != ckpt_p4_dim:
        print(f"       ⚠ P4 dim mismatch: backbone={actual_p4_dim} vs ckpt={ckpt_p4_dim}")
        print(f"       Adding projection adapter {actual_p4_dim}→{ckpt_p4_dim}")
        p4_adapter = nn.Conv2d(actual_p4_dim, ckpt_p4_dim, 1, bias=False).to(device)
        nn.init.kaiming_normal_(p4_adapter.weight)

    print(f"       Backbone: FastSAM-x (frozen)")
    print(f"       Decoder:  {args.decoder} ({sum(p.numel() for p in decoder.parameters()):,} params)")
    print(f"       Checkpoint loaded: {len(ckpt)} keys")

    # ── 数据集 | Dataset ──
    print("\n[2/5] Loading dataset...")
    base_classes = get_isaid5i_base_classes(args.fold)
    novel_classes = get_isaid5i_novel_classes(args.fold)

    # Determine which classes to diagnose
    if args.novel_only:
        target_classes = novel_classes
        class_type = "NOVEL"
    elif args.base_only:
        target_classes = base_classes
        class_type = "BASE"
    else:
        target_classes = base_classes + novel_classes
        class_type = "ALL"

    # 导入适配器 | Import adapter (same as train_fewshot.py)
    from tools.train.train_fewshot import PreCutTileAdapter

    # 导入 episode dataset | Import episode dataset
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from adatile.datasets.fewshot_dataset import FewShotEpisodeDataset

    # 构建数据集: Base 类用 train split + base_ids，Novel 类用 val split + novel_ids
    # Build datasets: Base classes → train split + base_ids, Novel classes → val split + novel_ids
    train_tiles = PreCutTileAdapter(args.tile_root, "train")
    val_tiles = PreCutTileAdapter(args.tile_root, "val")

    base_ds = FewShotEpisodeDataset(
        train_tiles, fold=args.fold, shot=args.shot, split="train",
        episodes_per_epoch=args.n_episodes_per_class, seed=args.seed,
        crop_support=True, crop_margin=0.2,
        novel_classes=base_classes,  # ← 采样 Base 类
    )
    novel_ds = FewShotEpisodeDataset(
        val_tiles, fold=args.fold, shot=args.shot, split="val",
        episodes_per_epoch=args.n_episodes_per_class, seed=args.seed + 1,
        crop_support=False,
        novel_classes=novel_classes,  # ← 采样 Novel 类
    )

    print(f"       Base classes:  {len(base_classes)}")
    print(f"       Novel classes: {len(novel_classes)}")
    print(f"       Diagnosing:    {len(target_classes)} classes ({class_type})")
    print(f"       Episodes/class: {args.n_episodes_per_class}")

    # ── 运行诊断 | Run diagnosis ──
    print(f"\n[3/5] Running diagnosis episodes...")

    all_results = []  # list of episode dicts
    per_class = defaultdict(lambda: {
        "sim_ious": [], "pred_ious": [], "deltas": [],
        "sim_aucs": [], "peak_in_gt": [],
    })

    novel_set = set(novel_classes)
    rng = np.random.RandomState(args.seed + 42)

    for cls_id in target_classes:
        cls_name = ISAID5I_CATEGORIES.get(cls_id, f"class_{cls_id}")
        ds = novel_ds if cls_id in novel_set else base_ds

        # 检查该类的样本数 | Check sample count for this class
        n_candidates = len(ds.class_to_images(cls_id))
        if n_candidates < args.shot + 1:
            print(f"  [{cls_name}] SKIP: only {n_candidates} candidates (need {args.shot + 1})")
            continue

        cls_results = []
        n_success = 0
        pbar = tqdm(range(args.n_episodes_per_class), desc=f"  [{cls_name:>20s}]", unit="ep")

        for ep_i in pbar:
            result = run_diagnosis_episode(
                ds, backbone, decoder, cls_id, args.shot,
                feature_level=args.feature_level, device=device,
                num_proto=args.num_proto, rng=rng,
                p3_adapter=p3_adapter, p4_adapter=p4_adapter,
            )
            if result is None:
                continue

            cls_results.append(result)
            n_success += 1

            # 保存可视化 | Save visualization
            if n_success <= args.save_vis:
                fig = make_episode_figure(
                    support_img=result["support_img"],
                    support_mask=result["support_mask"],
                    query_img=result["query_img"],
                    gt_mask=result["gt_mask"],
                    sim_map=result["sim_map"],
                    pred_mask=result["pred_mask"],
                    class_name=cls_name,
                    shot=args.shot,
                    ep_idx=ep_i,
                    sim_metrics=result["sim_metrics"],
                    pred_iou=result["pred_iou"],
                )
                fig.savefig(vis_dir / f"{cls_name}_ep{ep_i:02d}.png", dpi=100, bbox_inches="tight")
                plt.close(fig)

            # 更新进度条 | Update progress bar
            avg_sim_iou = np.mean([r["sim_metrics"]["best_iou"] for r in cls_results])
            avg_pred_iou = np.mean([r["pred_iou"] for r in cls_results])
            pbar.set_postfix(
                sim_IoU=f"{avg_sim_iou:.3f}",
                pred_IoU=f"{avg_pred_iou:.3f}",
                Δ=f"{avg_sim_iou - avg_pred_iou:+.3f}",
            )

        if not cls_results:
            print(f"  [{cls_name}] FAILED: no valid episodes")
            continue

        # 汇总该类 | Aggregate per class
        sim_ious = [r["sim_metrics"]["best_iou"] for r in cls_results]
        pred_ious = [r["pred_iou"] for r in cls_results]
        deltas = [r["delta"] for r in cls_results]
        sim_aucs = [r["sim_metrics"]["auc"] for r in cls_results]
        peak_gts = [r["sim_metrics"]["peak_in_gt"] for r in cls_results]

        per_class[cls_id] = {
            "name": cls_name,
            "n_episodes": len(cls_results),
            "sim_iou_mean": round(np.mean(sim_ious), 4),
            "sim_iou_std": round(np.std(sim_ious), 4),
            "pred_iou_mean": round(np.mean(pred_ious), 4),
            "pred_iou_std": round(np.std(pred_ious), 4),
            "delta_mean": round(np.mean(deltas), 4),
            "delta_std": round(np.std(deltas), 4),
            "sim_auc_mean": round(np.mean(sim_aucs), 4),
            "peak_in_gt_rate": round(np.mean(peak_gts), 4),
        }
        all_results.extend(cls_results)

    # ── 汇总诊断 | Aggregate diagnosis ──
    print(f"\n[4/5] Computing diagnosis...")

    # 分 base / novel 汇总 | Separate base/novel aggregation
    base_results = [r for r in all_results if r["class_id"] in base_classes]
    novel_results = [r for r in all_results if r["class_id"] in novel_classes]

    def summarize(results, label):
        if not results:
            return {"label": label, "n_episodes": 0, "error": "no data"}
        sim_ious = [r["sim_metrics"]["best_iou"] for r in results]
        pred_ious = [r["pred_iou"] for r in results]
        deltas = [r["delta"] for r in results]
        aucs = [r["sim_metrics"]["auc"] for r in results]
        peaks = [r["sim_metrics"]["peak_in_gt"] for r in results]

        return {
            "label": label,
            "n_episodes": len(results),
            "sim_iou_mean": round(np.mean(sim_ious), 4),
            "sim_iou_std": round(np.std(sim_ious), 4),
            "pred_iou_mean": round(np.mean(pred_ious), 4),
            "pred_iou_std": round(np.std(pred_ious), 4),
            "delta_mean": round(np.mean(deltas), 4),
            "delta_std": round(np.std(deltas), 4),
            "sim_auc_mean": round(np.mean(aucs), 4),
            "peak_in_gt_rate": round(np.mean(peaks), 4),
        }

    summary_base = summarize(base_results, "BASE")
    summary_novel = summarize(novel_results, "NOVEL")
    summary_all = summarize(all_results, "ALL")

    # ── 瓶颈判断 | Bottleneck verdict ──
    print(f"\n[5/5] Bottleneck Diagnosis...")

    def verdict(summary):
        if "error" in summary:
            return "NO DATA"
        sim = summary["sim_iou_mean"]
        pred = summary["pred_iou_mean"]
        delta = summary["delta_mean"]

        if sim < 0.05:
            return "🔴 Prototype 完全失效 (sim<5%) — prototype 无法定位目标"
        elif sim < 0.10:
            return "🔴 Prototype 极弱 (sim<10%) — 优先改进 prototype"
        elif delta > 0.05 and pred < sim * 0.5:
            return "🟡 Decoder 是瓶颈 — 相似度图可用但 decoder 未充分利用"
        elif delta > 0.03:
            return "🟠 轻微 Decoder 瓶颈 — decoder 有优化空间"
        elif pred < 0.05:
            return "🔴 双弱点 — 相似度图和 decoder 都极弱"
        else:
            return "🟢 Decoder 有效 — prototype 和 decoder 配合良好"

    # ── 打印结果 | Print results ──
    print(f"\n{'='*70}")
    print(f"  DIAGNOSIS RESULTS | 诊断结果")
    print(f"{'='*70}")

    def print_summary(s):
        if "error" in s:
            print(f"\n  [{s['label']}] {s['error']}")
            return
        print(f"\n  ── {s['label']} ({s['n_episodes']} episodes) ──")
        print(f"  Similarity bestIoU:  {s['sim_iou_mean']:.4f} ± {s['sim_iou_std']:.4f}")
        print(f"  Similarity AUC:      {s['sim_auc_mean']:.4f}")
        print(f"  Peak in GT rate:     {s['peak_in_gt_rate']:.2%}")
        print(f"  Prediction IoU:      {s['pred_iou_mean']:.4f} ± {s['pred_iou_std']:.4f}")
        print(f"  Δ (Sim − Pred):      {s['delta_mean']:+.4f}")
        print(f"  Verdict: {verdict(s)}")

    print_summary(summary_base)
    print_summary(summary_novel)
    print_summary(summary_all)

    # ── 逐类详情 | Per-class details ──
    print(f"\n{'─'*70}")
    print(f"  Per-Class Breakdown | 逐类详情")
    print(f"  {'Class':<22s} {'Type':<6s} {'SimIoU':>8s} {'PredIoU':>8s} {'Δ':>8s} {'PeakGT':>7s} {'Verdict'}")
    print(f"  {'─'*60}")

    for cls_id in sorted(target_classes):
        if cls_id not in per_class:
            continue
        c = per_class[cls_id]
        ctype = "NOVEL" if cls_id in novel_classes else "BASE"
        v = "🔴" if c["sim_iou_mean"] < 0.05 else (
            "🟡" if c["delta_mean"] > 0.03 else "🟢")
        print(f"  {c['name']:<22s} {ctype:<6s} "
              f"{c['sim_iou_mean']:>8.4f} {c['pred_iou_mean']:>8.4f} "
              f"{c['delta_mean']:>+8.4f} {c['peak_in_gt_rate']:>6.1%}  {v}")

    # ── 保存结果 | Save results ──
    diagnosis = {
        "config": {
            "ckpt": args.ckpt,
            "tile_root": args.tile_root,
            "fold": args.fold,
            "shot": args.shot,
            "decoder": args.decoder,
            "feature_level": args.feature_level,
            "num_proto": args.num_proto,
            "n_episodes_per_class": args.n_episodes_per_class,
        },
        "summary": {
            "base": summary_base,
            "novel": summary_novel,
            "all": summary_all,
        },
        "per_class": {str(k): v for k, v in per_class.items()},
        "verdicts": {
            "base": verdict(summary_base),
            "novel": verdict(summary_novel),
            "all": verdict(summary_all),
        },
        "n_visualizations_saved": min(args.save_vis, args.n_episodes_per_class),
    }

    diagnosis_path = out_dir / "diagnosis.json"
    with open(diagnosis_path, "w") as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  ✅ Diagnosis saved → {diagnosis_path}")
    print(f"  📊 Visualizations → {vis_dir}/ ({args.save_vis} per class)")
    print(f"{'='*70}")

    # ── 最终结论 | Final conclusion ──
    print(f"\n{'█'*70}")
    print(f"  FINAL VERDICT | 最终诊断结论")
    print(f"{'█'*70}")
    print(f"  BASE:  {verdict(summary_base)}")
    print(f"  NOVEL: {verdict(summary_novel)}")

    if summary_novel.get("sim_iou_mean", 0) > 0.10 and summary_novel.get("delta_mean", 0) > 0.03:
        print(f"\n  ★ 建议：优先改善 Decoder")
        print(f"    Novel 类的 similarity map 已具有一定定位能力")
        print(f"    (bestIoU={summary_novel['sim_iou_mean']:.3f})，")
        print(f"    但 Decoder 输出明显更差 (IoU={summary_novel['pred_iou_mean']:.3f})。")
        print(f"    考虑 Cross-Attention 替代 FiLM，让 support 空间信息")
        print(f"    更充分地传递到 query。")
    elif summary_novel.get("sim_iou_mean", 0) < 0.05:
        print(f"\n  ★ 建议：优先改善 Prototype")
        print(f"    Novel 类的 similarity map 基本无定位能力")
        print(f"    (bestIoU={summary_novel['sim_iou_mean']:.3f})。")
        print(f"    考虑 Multi-Prototype 或学习式原型（如 PerSAM 的")
        print(f"    learnable prototype adaptation）。")
    print(f"{'█'*70}\n")


if __name__ == "__main__":
    main()
