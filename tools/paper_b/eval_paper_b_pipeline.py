#!/usr/bin/env python3
"""
Paper B 统一验证管线 | Unified Paper B Validation Pipeline.
=============================================================

单脚本完成: Decoder训练 → Oracle Importance → Decoder诊断 → Tile尺寸消融。
Single script: Decoder training → Oracle Importance → Decoder Diag → Tile Size Ablation.

支持数据集 | Supported datasets:
    iSAID (tile)       — 15-class, 空tile多, fg_ratio有效
    Vaihingen (tile)   — 5-7 class, 密集标注, fg_ratio可能失效
    MassBuildings (tile)— binary, 建筑提取

设计原则 | Design principle:
    - 统一命令行接口 | Unified CLI
    - 数据集无关 | Dataset-agnostic
    - 零外部依赖（除adatile + torch）| No external deps beyond adatile + torch

用法 | Usage:
    # iSAID
    python tools/paper_b/eval_paper_b_pipeline.py \
        --tile-root data/iSAID_tiles --dataset isaid \
        --epochs 20 --output-dir runs/paper_b_isaid

    # Vaihingen
    python tools/paper_b/eval_paper_b_pipeline.py \
        --tile-root data/vaihingen_tiles --dataset vaihingen \
        --epochs 30 --output-dir runs/paper_b_vaihingen --skip-oracle-contrib

输出 | Output:
    {output_dir}/
    ├── decoder_best.pt              # 最佳 Decoder 权重
    ├── decoder_metrics.jsonl         # Per-epoch 训练指标
    ├── oracle_results.json           # Oracle ranking 结果
    ├── tile_size_ablation.json       # Tile 尺寸消融
    ├── decoder_diag.json             # Decoder 诊断统计
    └── paper_b_pipeline.png          # 综合图表 (4-panel)
"""

import sys, argparse, json, datetime, time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体 | Chinese font support
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.decoder.light_decoder import LightDecoder

# ═══════════════════════════════════════════════════
# Dataset configs | 数据集配置
# ═══════════════════════════════════════════════════

DATASET_CONFIGS = {
    "isaid":     {"num_classes": 16},
    "vaihingen": {"num_classes": 7},
}

TILE_SIZE = 1024
K_VALUES = [10, 20, 30, 40, 50, 70, 100]
TILE_SIZES_TO_TEST = [256, 512, 1024]  # ablation tile sizes


# ═══════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════

def load_tile_dataset(tile_root, dataset_name, split="train"):
    """加载 tile 数据集 | Load tile dataset by name."""
    from adatile.datasets.isaid_tiles import FastISAIDTileDataset
    from adatile.datasets.vaihingen_tiles import VaihingenTileDataset

    ds_map = {
        "isaid": FastISAIDTileDataset,
        "vaihingen": VaihingenTileDataset,
    }
    DSClass = ds_map.get(dataset_name)
    if DSClass is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DSClass(tile_root, split=split, semantic=True)


def compute_miou(pred_mask, gt_mask, num_classes):
    """Per-class mIoU (foreground only)."""
    miou_v, valid = 0.0, 0
    per_cls = {}
    for c in range(1, num_classes):
        pc = (pred_mask == c); tc = (gt_mask == c)
        inter = (pc & tc).sum(); union = (pc | tc).sum()
        if union > 0:
            iou_c = float(inter / union)
            per_cls[c] = iou_c
            miou_v += iou_c; valid += 1
    return miou_v / max(valid, 1), per_cls, valid


def compute_binary_iou(pred_mask, gt_mask):
    """Binary IoU (foreground only). Type-safe: tensor or numpy input."""
    if hasattr(pred_mask, "dtype") and pred_mask.dtype in (torch.float32, torch.float64):
        pc = pred_mask > 0.5
        tc = gt_mask > 0
        inter = (pc & tc).sum().float()
        union = (pc | tc).sum().float()
        return float((inter / max(union, 1)).item())
    else:
        pc = np.asarray(pred_mask) > 0.5
        tc = np.asarray(gt_mask) > 0
        inter = (pc & tc).sum()
        union = (pc | tc).sum()
        return float(inter / max(union, 1))


# ═══════════════════════════════════════════════════
# Phase 1: Train Decoder | 训练解码器
# ═══════════════════════════════════════════════════

def train_decoder(args, device, log):
    """Phase 1: Train LightDecoder with Focal+Dice loss on FG>5% filtered tiles.
    Phase 1: 训练LightDecoder, Focal+Dice loss, FG>5%过滤.
    
    Returns best checkpoint and validation mIoU.
    返回最佳checkpoint和验证mIoU."""
    from PIL import Image

    num_classes = args.num_classes
    log("decoder", f"Training LightDecoder ({args.epochs} epochs, {num_classes-1} classes)")

    # Auto-detect val split | 自动检测验证集划分
    val_split = "val" if (Path(args.tile_root) / "val").exists() else "test"
    train_ds = load_tile_dataset(args.tile_root, args.dataset, "train")
    val_ds = load_tile_dataset(args.tile_root, args.dataset, val_split)

    log("decoder", f"Train tiles: {len(train_ds)}, Val({val_split}) tiles: {len(val_ds)}")

    # Detect mask extension | 检测mask文件扩展名
    mask_ext = ".png"
    if train_ds._tiles:
        first_name = train_ds._tiles[0]
        if (train_ds._mask_dir / f"{first_name}.png").exists():
            mask_ext = ".png"
        elif (train_ds._mask_dir / f"{first_name}.tif").exists():
            mask_ext = ".tif"

    # Filter FG>5% tiles for training | FG>5% 过滤
    # iSAID: filters ~60pct empty tiles. Vaihingen: keeps almost all tiles (dense).
    # iSAID: 过滤约60pct空tile. Vaihingen: 保留几乎所有tile(密集标注).
    fg5_tiles = []
    for fname in tqdm(train_ds._tiles, desc="  Filter FG>5%", leave=False):
        mask_path = train_ds._mask_dir / f"{fname}{mask_ext}"
        mask = np.array(Image.open(mask_path))
        fg_r = (mask > 0).sum() / mask.size
        if fg_r > 0.05:
            fg5_tiles.append(fname)
    n_train_orig = len(train_ds._tiles)
    train_ds._tiles = fg5_tiles if fg5_tiles else train_ds._tiles
    log("decoder", f"Train FG>5% tiles: {len(train_ds._tiles)} ({len(train_ds._tiles)/max(n_train_orig,1)*100:.0f}pct of original {n_train_orig})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=min(2, args.num_workers), pin_memory=True)

    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    decoder = LightDecoder(1280, num_classes).to(device)
    n_p = sum(p.numel() for p in decoder.parameters())
    log("decoder", f"Decoder: {n_p:,} params")

    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    best_miou, best_state = 0.0, None
    metrics_path = Path(args.output_dir) / "decoder_metrics.jsonl"

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        total_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"  E{epoch}", leave=False):
            img = batch["image"].to(device)
            tgt = batch["mask"].to(device)
            feats = backbone(img)
            logit = decoder(feats, target_size=tgt.shape[1:])

            ce = F.cross_entropy(logit, tgt, ignore_index=255, reduction="none")
            focal_loss = ((1 - torch.exp(-ce)) ** 5.0 * ce).mean()

            probs = F.softmax(logit, dim=1)
            dice_sum, vd = 0.0, 0
            for c in range(1, num_classes):
                p_c = probs[:, c]; t_c = (tgt == c).float()
                inter = (p_c * t_c).sum()
                union = p_c.sum() + t_c.sum() + 1e-8
                if t_c.sum() > 0: dice_sum += (2*inter/union); vd += 1
            dice_loss = 1.0 - (dice_sum / max(vd, 1))

            loss = 0.5 * focal_loss + 0.5 * dice_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1

        sch.step()
        avg_loss = total_loss / max(n, 1)

        # Validation
        decoder.eval()
        inter = torch.zeros(num_classes, device=device)
        union = torch.zeros(num_classes, device=device)
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(device)
                tgt = batch["mask"].to(device)
                feats = backbone(img)
                logit = decoder(feats, target_size=tgt.shape[1:])
                pred = logit.argmax(dim=1)
                for c in range(1, num_classes):
                    pc = (pred == c); tc = (tgt == c)
                    inter[c] += (pc & tc).sum().float()
                    union[c] += (pc | tc).sum().float()

        miou_v, vc = 0.0, 0
        for c in range(1, num_classes):
            if union[c] > 0: miou_v += (inter[c]/union[c]).item(); vc += 1
        miou_val = miou_v / max(vc, 1)

        epoch_metrics = {"epoch": epoch, "loss": round(avg_loss, 6),
                         "miou_val": round(miou_val, 6)}
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps(epoch_metrics) + "\n"); mf.flush()

        marker = ""
        if miou_val > best_miou:
            best_miou = miou_val
            best_state = {k: v.clone() for k, v in decoder.state_dict().items()}
            marker = " *"
        log("decoder", f"E{epoch:2d}/{args.epochs} loss={avg_loss:.4f} val={miou_val:.4f}{marker}")

    if best_state:
        decoder.load_state_dict(best_state)
        torch.save(best_state, str(Path(args.output_dir) / "decoder_best.pt"))

    log("decoder", f"Best val mIoU={best_miou:.4f} | saved to decoder_best.pt")
    return decoder, backbone, best_miou


# ═══════════════════════════════════════════════════
# Phase 2: Oracle Importance | Oracle 重要性分析
# ═══════════════════════════════════════════════════

@torch.no_grad()
def oracle_importance(args, decoder, backbone, device, log):
    """Phase 2: Oracle tile importance analysis.
    Phase 2: Oracle tile重要性分析.
    
    Computes Spearman r(fg_ratio, per-tile IoU) and oracle Top-K rankings.
    计算fg_ratio与per-tile IoU的Spearman相关系数及Oracle Top-K排序.
    
    NOTE: Tile-level random sampling => global Spearman mixes inter/intra-image effects.
    注意：tile级随机采样意味着全局Spearman混淆了图间和图内效应."""
    from PIL import Image

    log("oracle", "Oracle Importance Analysis...")
    val_ds = load_tile_dataset(args.tile_root, args.dataset, "val" if (Path(args.tile_root)/"val").exists() else "test")
    num_classes = args.num_classes
    n_samples = min(args.oracle_samples, len(val_ds))
    log("oracle", f"Analyzing {n_samples} tiles from val set")

    # Sample tiles | 采样tile
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(val_ds), n_samples, replace=False)

    tile_stats = []
    for idx in tqdm(indices, desc="  Oracle tiles", leave=False):
        sample = val_ds[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["mask"]
        if gt.dim() == 3: gt = gt.squeeze(0)
        H, W = gt.shape

        feats = backbone(img)
        logit = decoder(feats, target_size=(H, W))
        pred = logit.argmax(dim=1).cpu().numpy()[0]

        # Tile-level metrics
        if num_classes > 2:
            tile_miou, per_cls_iou, n_valid = compute_miou(pred, gt.cpu().numpy(), num_classes)
        else:
            tile_miou = compute_binary_iou(
                (torch.sigmoid(logit) > 0.5).float().cpu().numpy()[0, 0],
                gt.cpu().numpy())
            n_valid = 1

        fg_ratio = float((gt > 0).sum() / max(gt.numel(), 1))

        # Dominant class
        gt_fg = gt[gt > 0]
        dominant_class = 0
        if len(gt_fg) > 0:
            vals, counts = np.unique(gt_fg.numpy(), return_counts=True)
            dominant_class = int(vals[counts.argmax()])

        tile_stats.append({
            "miou": float(tile_miou),
            "fg_ratio": fg_ratio,
            "dominant_class": dominant_class,
            "n_valid_classes": n_valid,
        })

    all_ious = np.array([t["miou"] for t in tile_stats])
    all_fgs = np.array([t["fg_ratio"] for t in tile_stats])

    # Spearman r: fg_ratio vs IoU
    sr_fg, _ = spearmanr(all_fgs, all_ious)
    log("oracle", f"Spearman r(fg_ratio, IoU) = {sr_fg:.4f}")

    # Oracle rankings (simulated on tile-level data)
    # L1: Random baseline
    rng = np.random.RandomState(args.seed)
    n = len(tile_stats)
    order_random = rng.permutation(n)

    # L2: fg_ratio
    order_fg = np.argsort(all_fgs)[::-1]

    # L3: Per-tile IoU
    order_iou = np.argsort(all_ious)[::-1]

    results = {}
    for name, order in [("random", order_random), ("fg_ratio", order_fg),
                         ("tile_iou", order_iou)]:
        results[name] = {}
        for k in K_VALUES:
            nk = max(1, int(n * k / 100))
            selected = order[:nk]
            miou_k = float(np.mean(all_ious[selected]))
            results[name][k] = {
                "miou_mean": miou_k,
                "retention_mean": miou_k / max(all_ious.mean(), 1e-8),
            }

    # Print table
    log("oracle", f"  {'Strategy':<15} {'K=100%':>8} {'K=50%':>8} {'K=30%':>8} {'K=10%':>8}  Ret@50%")
    baseline = results["fg_ratio"][100]["miou_mean"]
    for name in ["random", "fg_ratio", "tile_iou"]:
        r = results[name]
        log("oracle",
            f"  {name:<15} {r[100]['miou_mean']*100:>7.2f}% "
            f"{r[50]['miou_mean']*100:>7.2f}% "
            f"{r[30]['miou_mean']*100:>7.2f}% "
            f"{r[10]['miou_mean']*100:>7.2f}% "
            f"{r[50]['retention_mean']*100:>6.1f}%")

    # Diagnosis: validate fg_ratio as importance proxy.
    # If IoU >> fg_ratio gap > 0.15: contribution routing needed instead.
    # Dense datasets (Vaihingen) typically show larger gaps.
    iou_ret50 = results["tile_iou"][50]["retention_mean"]
    fg_ret50 = results["fg_ratio"][50]["retention_mean"]
    if iou_ret50 - fg_ret50 > 0.15:
        log("oracle", "⚠️  IoU ranking >> fg_ratio: fg_ratio is suboptimal for this dataset")
    elif iou_ret50 - fg_ret50 > 0.05:
        log("oracle", "△  IoU marginally better than fg_ratio")
    else:
        log("oracle", "✅  fg_ratio is a valid importance proxy for this dataset")

    return {
        "n_tiles": n,
        "per_image_spearman_note": "Not computed; global Spearman mixes inter/intra-image effects. Use with caution.",
        "spearman_fg_iou": float(sr_fg),
        "fg_ratio_ret50": float(fg_ret50),
        "tile_iou_ret50": float(iou_ret50),
        "results": {name: {str(k): v for k, v in r.items()}
                    for name, r in results.items()},
        "tile_stats": [{"miou": t["miou"], "fg_ratio": t["fg_ratio"],
                        "dominant_class": t["dominant_class"]}
                       for t in tile_stats],
    }


# ═══════════════════════════════════════════════════
# Phase 3: Decoder Diagnostic | Decoder 诊断
# ═══════════════════════════════════════════════════

def decoder_diag(oracle_data, args, log):
    """Phase 3: Decoder diagnostic statistics.
    Phase 3: Decoder诊断统计.
    
    IoU distribution, fg_ratio vs IoU scatter, per-class IoU breakdown.
    IoU分布, IoU vs fg_ratio散点, 每类IoU分解."""
    tile_stats = oracle_data["tile_stats"]
    all_ious = np.array([t["miou"] for t in tile_stats])
    all_fgs = np.array([t["fg_ratio"] for t in tile_stats])

    # FG ratio bins
    bins_fg = [(0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.20),
               (0.20, 0.50), (0.50, 1.0)]
    log("diag", "IoU by FG ratio bin:")
    for lo, hi in bins_fg:
        group = [t for t in tile_stats if lo <= t["fg_ratio"] < hi]
        if group:
            avg_iou = np.mean([t["miou"] for t in group])
            log("diag", f"  FG [{lo:.2f},{hi:.2f}): {len(group):>4d} tiles, avg IoU={avg_iou:.4f}")

    # Per-class avg IoU
    per_cls = defaultdict(list)
    for t in tile_stats:
        if t["dominant_class"] > 0:
            per_cls[t["dominant_class"]].append(t["miou"])

    log("diag", "Per-class avg IoU:")
    for c in sorted(per_cls.keys()):
        log("diag", f"  Class {c}: {np.mean(per_cls[c]):.4f} ({len(per_cls[c])} tiles)")

    return {
        "iou_mean": float(np.mean(all_ious)),
        "iou_median": float(np.median(all_ious)),
        "iou_std": float(np.std(all_ious)),
        "spearman_fg_iou": oracle_data["spearman_fg_iou"],
        "per_class_avg_iou": {str(c): float(np.mean(v)) for c, v in per_cls.items()},
    }


# ═══════════════════════════════════════════════════
# Phase 4: Tile Size Ablation | Tile 尺寸消融
# ═══════════════════════════════════════════════════

@torch.no_grad()
def tile_size_ablation(args, decoder, backbone, device, log):
    """Phase 4: Tile size ablation with Oracle fg_ratio ranking.
    Phase 4: Tile尺寸消融, Oracle fg_ratio排序.
    
    Re-cuts 1024 tiles into sub-tiles (256, 512, 1024) and simulates
    oracle Top-K selection to measure accuracy vs compute tradeoff.
    将1024 tile重新切割成子tile并模拟Oracle Top-K选择来度量准确率vs计算量的权衡."""
    from PIL import Image

    log("ts_ablation", "Tile Size Ablation...")
    val_ds = load_tile_dataset(args.tile_root, args.dataset, "val" if (Path(args.tile_root)/"val").exists() else "test")
    num_classes = args.num_classes
    n_images = min(args.ts_images, len(val_ds))
    log("ts_ablation", f"Testing {n_images} val tiles × {TILE_SIZES_TO_TEST} sizes")

    all_results = {}
    for ts in TILE_SIZES_TO_TEST:
        mious_at_k = {k: [] for k in K_VALUES}
        tile_counts = []
        for idx in tqdm(range(n_images), desc=f"  TS={ts}", leave=False):
            sample = val_ds[idx]
            img_full = sample["image"]  # [3, 1024, 1024]
            gt_full = sample["mask"].squeeze() if sample["mask"].dim() == 3 else sample["mask"]
            H, W = gt_full.shape

            n_ty = (H + ts - 1) // ts
            n_tx = (W + ts - 1) // ts
            tile_counts.append(n_ty * n_tx)
            tiles = []

            for ty in range(n_ty):
                for tx in range(n_tx):
                    y0, y1 = ty * ts, min(ty * ts + ts, H)
                    x0, x1 = tx * ts, min(tx * ts + ts, W)
                    th, tw = y1 - y0, x1 - x0

                    # Extract + pad tile
                    tile_rgb = img_full[:, y0:y1, x0:x1]
                    if th < ts or tw < ts:
                        p = torch.zeros(3, ts, ts)
                        p[:, :th, :tw] = tile_rgb
                        tile_rgb = p

                    # Pad to 32
                    ph = (32 - ts % 32) % 32; pw = (32 - ts % 32) % 32
                    if ph > 0 or pw > 0:
                        tile_rgb = F.pad(tile_rgb, (0, pw, 0, ph))

                    tile_t = tile_rgb.unsqueeze(0).to(device)
                    feats = backbone(tile_t)
                    logit = decoder(feats, target_size=(ts+ph, ts+pw))
                    pred_tile = logit.argmax(dim=1).cpu().numpy()[0, :th, :tw]
                    gt_tile = gt_full[y0:y1, x0:x1].cpu().numpy()
                    fg_ratio = float((gt_tile > 0).sum() / max(th * tw, 1))

                    tiles.append({
                        "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                        "th": th, "tw": tw, "pred": pred_tile,
                        "fg_ratio": fg_ratio, "gt_tile": gt_tile,
                    })

            # Rank by fg_ratio + stitch at each K%
            for k in K_VALUES:
                nk = max(1, int(len(tiles) * k / 100))
                order = np.argsort([t["fg_ratio"] for t in tiles])[::-1]
                selected = set(order[:nk])

                pred_full = np.zeros((H, W), dtype=np.int64)
                gt_full_np = np.zeros((H, W), dtype=np.uint8)
                for i, t in enumerate(tiles):
                    gt_full_np[t["y0"]:t["y0"]+t["th"], t["x0"]:t["x0"]+t["tw"]] = t["gt_tile"]
                    if i in selected:
                        pred_full[t["y0"]:t["y0"]+t["th"], t["x0"]:t["x0"]+t["tw"]] = t["pred"]

                if num_classes > 2:
                    miou_k, _, _ = compute_miou(pred_full, gt_full_np, num_classes)
                else:
                    miou_k = compute_binary_iou(pred_full, gt_full_np)
                mious_at_k[k].append(miou_k)

        # Aggregate
        miou_100 = np.mean(mious_at_k[100]) if mious_at_k[100] else 1.0
        all_results[ts] = {
            "avg_tiles": round(float(np.mean(tile_counts)), 1),
            **{str(k): {
                "miou_mean": float(np.mean(mious_at_k[k])),
                "retention_mean": float(np.mean(mious_at_k[k]) / max(miou_100, 1e-8)),
            } for k in K_VALUES},
        }

    # Print
    log("ts_ablation", f"  {'TS':<8} {'Tiles':>7} {'K=100%':>8} {'K=50%':>8} {'K=30%':>8}  Ret@50%")
    for ts in TILE_SIZES_TO_TEST:
        r = all_results[ts]
        log("ts_ablation",
            f"  {ts:<8} {r['avg_tiles']:>7} "
            f"{r[str(100)]['miou_mean']*100:>7.2f}% "
            f"{r[str(50)]['miou_mean']*100:>7.2f}% "
            f"{r[str(30)]['miou_mean']*100:>7.2f}% "
            f"{r[str(50)]['retention_mean']*100:>6.1f}%")

    return all_results


# ═══════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════

def plot_pipeline_results(oracle_data, ts_data, diag_data, args, output_path):
    """4-panel summary figure."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))

    tile_stats = oracle_data["tile_stats"]
    all_ious = np.array([t["miou"] for t in tile_stats])
    all_fgs = np.array([t["fg_ratio"] for t in tile_stats])

    # Panel 1: IoU vs fg_ratio scatter
    ax = axes[0, 0]
    ax.scatter(all_fgs, all_ious, c="steelblue", alpha=0.5, s=15, edgecolors="none")
    ax.set_xlabel("fg_ratio (Foreground Density)", fontsize=11)
    ax.set_ylabel("Per-Tile IoU", fontsize=11)
    ax.set_title(f"fg_ratio vs IoU (Spearman r={oracle_data['spearman_fg_iou']:.3f})", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel 2: Oracle ranking comparison
    ax = axes[0, 1]
    colors = {"random": "gray", "fg_ratio": "#E74C3C", "tile_iou": "#3498DB"}
    for name in ["random", "fg_ratio", "tile_iou"]:
        data = oracle_data["results"][name]
        xs, ys = [], []
        for k in K_VALUES:
            xs.append(k)
            ys.append(data[str(k)]["miou_mean"] * 100)
        ax.plot(xs, ys, "o-", color=colors[name], linewidth=2, markersize=7, label=name)
    ax.set_xlabel("K% (Tiles Selected)", fontsize=11)
    ax.set_ylabel("Mean IoU (%)", fontsize=11)
    ax.set_title(f"Oracle Importance (Ret@50%: fg={oracle_data['fg_ratio_ret50']*100:.0f}% vs IoU={oracle_data['tile_iou_ret50']*100:.0f}%)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: IoU histogram
    ax = axes[1, 0]
    ax.hist(all_ious, bins=30, color="#3498DB", edgecolor="white", alpha=0.8)
    ax.axvline(x=np.mean(all_ious), color="red", linestyle="--", label=f"Mean={np.mean(all_ious):.3f}")
    ax.set_xlabel("Per-Tile IoU", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Tile IoU Distribution", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Panel 4: Tile size ablation
    ax = axes[1, 1]
    ts_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ts_data.keys())))
    for ti, ts in enumerate(TILE_SIZES_TO_TEST):
        data = ts_data[ts]
        xs, ys = [], []
        for k in K_VALUES:
            xs.append(k)
            ys.append(data[str(k)]["miou_mean"] * 100)
        ax.plot(xs, ys, "o-", color=ts_colors[ti], linewidth=2, markersize=7,
                label=f"{ts}px ({data['avg_tiles']} tiles)")  # avg per image tile count approximation
    ax.set_xlabel("K% (Tiles Selected)", fontsize=11)
    ax.set_ylabel("FG-mIoU (%)", fontsize=11)
    ax.set_title("Tile Size Ablation (Oracle fg_ratio)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Paper B Pipeline: {args.dataset.upper()} ({args.num_classes-1} classes)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "paper_b_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    # Dataset | 数据集
    p.add_argument("--tile-root", type=str, required=True,
                   help="Tile dataset root directory | tile数据集根目录")
    p.add_argument("--dataset", type=str, default="isaid",
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Dataset name | 数据集名称")
    p.add_argument("--num-classes", type=int, default=None,
                   help="Override num_classes (auto-detect if not set) | 类别数覆盖")
    # Training | 训练
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    # Oracle | Oracle实验
    p.add_argument("--oracle-samples", type=int, default=200,
                   help="Tiles to sample for oracle analysis")
    # Tile size ablation | Tile尺寸消融
    p.add_argument("--ts-images", type=int, default=50,
                   help="Tiles for tile size ablation")
    # Output | 输出
    p.add_argument("--skip-oracle-contrib", action="store_true",
                   help="Skip oracle contribution analysis (save compute)")
    p.add_argument("--output-dir", type=str, default="runs/paper_b_pipeline")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    # Auto-detect num_classes from config
    if args.num_classes is None:
        args.num_classes = DATASET_CONFIGS[args.dataset]["num_classes"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("paper_b")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "pipeline.jsonl")))

    def log(tag, msg):
        logger.log_info(f"paper_b/{tag}", msg)
        print(f"  {msg}")

    log("start", f"{'='*60}")
    log("start", f"Paper B Pipeline: {args.dataset} ({args.num_classes-1} classes)")
    log("start", f"Output: {output_dir}")
    log("start", f"{'='*60}")

    # ═══ Phase 1: Train Decoder ═══
    t0 = time.perf_counter()
    decoder, backbone, best_miou = train_decoder(args, device, log)
    dt = time.perf_counter() - t0
    log("done", f"Phase 1 (Decoder): {dt:.0f}s, best mIoU={best_miou:.4f}")

    # ═══ Phase 2: Oracle Importance ═══
    t0 = time.perf_counter()
    oracle_data = oracle_importance(args, decoder, backbone, device, log)
    dt = time.perf_counter() - t0
    log("done", f"Phase 2 (Oracle): {dt:.0f}s")

    # ═══ Phase 3: Decoder Diagnostic ═══
    diag_data = decoder_diag(oracle_data, args, log)

    # ═══ Phase 4: Tile Size Ablation ═══
    t0 = time.perf_counter()
    ts_data = tile_size_ablation(args, decoder, backbone, device, log)
    dt = time.perf_counter() - t0
    log("done", f"Phase 4 (Tile Size): {dt:.0f}s")

    # ═══ Visualization ═══
    plot_pipeline_results(oracle_data, ts_data, diag_data, args, output_dir)

    # ═══ Save Summary ═══
    summary = {
        "experiment": "Paper B Pipeline",
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": args.dataset,
        "num_classes": args.num_classes,
        "config": vars(args),
        "decoder": {"best_miou": float(best_miou)},
        "oracle": {
            "spearman_fg_iou": oracle_data["spearman_fg_iou"],
            "fg_ratio_ret50": oracle_data["fg_ratio_ret50"],
            "tile_iou_ret50": oracle_data["tile_iou_ret50"],
        },
        "diagnostic": diag_data,
        "tile_size_ablation": {str(ts): ts_data[ts] for ts in TILE_SIZES_TO_TEST},
    }
    with open(output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log("done", f"All results saved to {output_dir}/")
    log("done", f"  - decoder_best.pt (mIoU={best_miou:.4f})")
    log("done", f"  - pipeline_summary.json")
    log("done", f"  - paper_b_pipeline.png")


if __name__ == "__main__":
    main()
