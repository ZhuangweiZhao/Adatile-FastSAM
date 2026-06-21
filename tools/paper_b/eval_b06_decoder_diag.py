#!/usr/bin/env python3
"""
B-06: Decoder Diagnostic — 为什么 Tile-mIoU=48% → Full-mIoU=30%？
=====================================================================

B-05 Oracle 实验发现: K=10% 贡献型 tile 几乎等于 K=100% 全量。
说明大部分 tile 的 Decoder 预测质量极差，甚至贡献为负。

本诊断回答三个问题 | Three questions:
    1. IoU 分布: 是"少数好多数差"还是"普遍差"？
       IoU distribution: "few good, many bad" or "uniformly bad"?
    2. 好/差 tile 的特征: 类别、前景占比、目标尺寸有何不同？
       Best/worst tile characteristics: class, fg_ratio, object size?
    3. fg_ratio 与 IoU 的真实关系: 高前景密度是否意味着高 IoU？
       The real relationship between fg_ratio and IoU.

输出 | Output:
    runs/b06_decoder_diag/
    ├── decoder_diag.png            # 5-panel figure
    ├── decoder_diag.json           # Summary statistics
    └── tiles/                      # 最好/最差 tile 可视化
        ├── best_01.png ... best_10.png
        └── worst_01.png ... worst_10.png

用法 | Usage:
    python tools/paper_b/eval_b06_decoder_diag.py \
        --src-root /root/autodl-tmp/iSAID_processed \
        --decoder-ckpt runs/b04_v3/decoder_best.pt \
        --n-images 30 --output-dir runs/b06_decoder_diag
"""

import sys, argparse, json, datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.utils.label_mapping import _ID_TO_NAME as ISAID_NAMES

TILE_SIZE = 1024
NUM_CLASSES = 15
NUM_OUT_CH = 16
_ISAID_NAMES = {0: "bg", **ISAID_NAMES}

# ═══════════════════════════════════════════════════════════════
# LightDecoder (与 train_b04.py 内联版一致)
# ═══════════════════════════════════════════════════════════════

class LightDecoder(nn.Module):
    """FastSAM P4 → 16-class, mirrors train_b04.py."""

    def __init__(self, in_channels=1280, num_classes=16):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, num_classes, 1, bias=True)

    def forward(self, p4, target_size=None):
        x = self.stage1(p4)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage3(x)
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


# ═══════════════════════════════════════════════════════════════
# 语义掩码渲染 | Mask Rendering
# ═══════════════════════════════════════════════════════════════

def render_semantic_mask(annotations, h, w):
    import cv2
    sem = np.zeros((h, w), dtype=np.uint8)
    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        if cat_id <= 0: continue
        seg = ann.get("segmentation", [])
        if not seg:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            sem[max(0, int(bbox[1])):min(h, int(bbox[1]+bbox[3])),
                max(0, int(bbox[0])):min(w, int(bbox[0]+bbox[2]))] = cat_id
            continue
        if isinstance(seg, dict): continue
        for poly in (seg if isinstance(seg[0], list) else [seg]):
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w-1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h-1)
            cv2.fillPoly(sem, [pts], int(cat_id))
    return sem


# ═══════════════════════════════════════════════════════════════
# Per-tile IoU 计算
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_per_tile_iou(pred, gt):
    """逐前景类 mIoU (15类) | Per-foreground-class mIoU."""
    miou_v, valid = 0.0, 0
    per_cls = {}
    for c in range(1, NUM_OUT_CH):
        pc = (pred == c); tc = (gt == c)
        inter = (pc & tc).sum(); union = (pc | tc).sum()
        if union > 0:
            iou_c = (inter / union).item()
            per_cls[c] = iou_c
            miou_v += iou_c; valid += 1
    return miou_v / max(valid, 1), per_cls, valid


# ═══════════════════════════════════════════════════════════════
# 主诊断 | Main Diagnostics
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--decoder-ckpt", type=str, required=True)
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--output-dir", type=str, default="runs/b06_decoder_diag")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_viz_dir = output_dir / "tiles"
    tile_viz_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("b06")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "b06.jsonl")))

    # ── 1. Load Data | 加载数据 ──
    src_root = Path(args.src_root)
    with open(src_root / "train" / "annotations" / "instances_train.json") as f:
        coco = json.load(f)
    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    img_dir = src_root / "train" / "images"
    from PIL import Image
    rng = np.random.RandomState(args.seed)
    candidates = [(img_info, str(img_dir / img_info["file_name"]),
                   img_id_to_anns.get(img_info["id"], []))
                  for img_info in coco["images"]
                  if (img_dir / img_info["file_name"]).exists()
                  and img_id_to_anns.get(img_info["id"], [])]
    rng.shuffle(candidates)
    images = candidates[:args.n_images]
    logger.log_info("b06/data", f"Images: {len(images)}")

    # ── 2. Load Model | 加载模型 ──
    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    decoder = LightDecoder(1280, 16).to(device)
    decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
    decoder.eval()
    logger.log_info("b06/model", f"Model loaded from {args.decoder_ckpt}")

    # ═══ 3. Collect per-tile statistics | 收集逐 tile 统计 ═══
    all_tile_stats = []  # list of dicts: iou, fg_ratio, dominant_class, per_cls_iou, ...

    for img_info, img_path, anns in tqdm(images, desc="  Analyzing tiles"):
        img_np = np.array(Image.open(img_path).convert("RGB"))
        H, W = img_np.shape[:2]
        gt_full = render_semantic_mask(anns, H, W)

        n_ty = (H + TILE_SIZE - 1) // TILE_SIZE
        n_tx = (W + TILE_SIZE - 1) // TILE_SIZE

        for ty in range(n_ty):
            for tx in range(n_tx):
                y0, y1 = ty * TILE_SIZE, min(ty * TILE_SIZE + TILE_SIZE, H)
                x0, x1 = tx * TILE_SIZE, min(tx * TILE_SIZE + TILE_SIZE, W)
                gt_tile = gt_full[y0:y1, x0:x1]
                th, tw = y1 - y0, x1 - x0

                # Extract tile image + pad | 提取 tile 图像 + 填充
                tile_rgb = img_np[y0:y1, x0:x1]
                if th < TILE_SIZE or tw < TILE_SIZE:
                    p = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
                    p[:th, :tw] = tile_rgb; tile_rgb = p

                tile_t = torch.from_numpy(tile_rgb.astype(np.float32) / 255.0)
                tile_t = tile_t.permute(2, 0, 1).unsqueeze(0).to(device)

                # Decoder forward | Decoder 前向
                feats = backbone(tile_t)
                logit = decoder(feats["p4"], target_size=(TILE_SIZE, TILE_SIZE))
                pred_tile = logit.argmax(dim=1).cpu().numpy()[0][:th, :tw]

                # Compute metrics | 计算指标
                tile_miou, per_cls_iou, n_valid = compute_per_tile_iou(pred_tile, gt_tile)
                fg_ratio = float((gt_tile > 0).sum() / max(th * tw, 1))

                # Dominant class (GT foreground class with most pixels) | GT 主导前景类
                gt_fg = gt_tile[gt_tile > 0]
                dominant_class = 0
                if len(gt_fg) > 0:
                    vals, counts = np.unique(gt_fg, return_counts=True)
                    dominant_class = int(vals[counts.argmax()])

                # Object size proxy: median connected component area per dominant class
                from scipy import ndimage
                obj_sizes = []
                if dominant_class > 0:
                    labeled, n_objs = ndimage.label(gt_tile == dominant_class)
                    if n_objs > 0:
                        obj_sizes = [int((labeled == i).sum())
                                    for i in range(1, n_objs + 1)]

                all_tile_stats.append({
                    "miou": float(tile_miou),
                    "fg_ratio": fg_ratio,
                    "dominant_class": dominant_class,
                    "n_valid_classes": n_valid,
                    "per_cls_iou": {str(k): float(v) for k, v in per_cls_iou.items()},
                    "n_objects": len(obj_sizes),
                    "median_obj_size": float(np.median(obj_sizes)) if obj_sizes else 0,
                    "tile_h": th, "tile_w": tw,
                    "grid_pos": f"{ty},{tx}",
                    "edge_tile": (ty == 0 or ty == n_ty-1 or tx == 0 or tx == n_tx-1),
                    # Save tile data for visualization | 保存用于可视化
                    "_tile_rgb": tile_rgb,  # [1024,1024,3] uint8
                    "_gt_tile": gt_tile,    # [th,tw] uint8
                    "_pred_tile": pred_tile, # [th,tw] int64
                    "_img_id": img_info["file_name"],
                })

    logger.log_info("b06/stats", f"Total tiles analyzed: {len(all_tile_stats)}")

    # ── 4. Aggregate Statistics | 汇总统计 ──
    ious = np.array([t["miou"] for t in all_tile_stats])
    fg_ratios = np.array([t["fg_ratio"] for t in all_tile_stats])
    n_valid_arr = np.array([t["n_valid_classes"] for t in all_tile_stats])

    # Per-class average IoU | 每类平均 IoU
    per_cls_all = {c: [] for c in range(1, NUM_OUT_CH)}
    for t in all_tile_stats:
        for c_str, iou_v in t["per_cls_iou"].items():
            per_cls_all[int(c_str)].append(iou_v)
    per_cls_avg = {c: float(np.mean(v)) if v else 0.0
                   for c, v in per_cls_all.items()}

    # Tile groups by FG ratio | 按 FG 占比分组
    bins_fg = [(0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.2),
               (0.2, 0.5), (0.5, 1.0)]
    for lo, hi in bins_fg:
        group = [t for t in all_tile_stats if lo <= t["fg_ratio"] < hi]
        if group:
            avg_iou = np.mean([t["miou"] for t in group])
            logger.log_info("b06/fg_group",
                           f"  FG [{lo:.2f},{hi:.2f}): {len(group):>5d} tiles, "
                           f"avg IoU={avg_iou:.4f}")

    # Best / Worst tiles | 最好/最差 tile
    sorted_by_iou = sorted(all_tile_stats, key=lambda t: t["miou"], reverse=True)
    best_10 = sorted_by_iou[:10]
    worst_10 = sorted_by_iou[-10:]

    logger.log_info("b06/best", "Top 10 Tiles:")
    for i, t in enumerate(best_10):
        cname = _ISAID_NAMES.get(t["dominant_class"], "?")
        logger.log_info("b06/best",
                       f"  #{i+1}: IoU={t['miou']:.4f}, "
                       f"FG={t['fg_ratio']:.3f}, "
                       f"cls={t['dominant_class']}({cname}), "
                       f"objs={t['n_objects']}, "
                       f"med_size={t['median_obj_size']:.0f}px, "
                       f"edge={t['edge_tile']}")

    logger.log_info("b06/worst", "Bottom 10 Tiles:")
    for i, t in enumerate(worst_10):
        cname = _ISAID_NAMES.get(t["dominant_class"], "?")
        logger.log_info("b06/worst",
                       f"  #{i+1}: IoU={t['miou']:.4f}, "
                       f"FG={t['fg_ratio']:.3f}, "
                       f"cls={t['dominant_class']}({cname}), "
                       f"objs={t['n_objects']}, "
                       f"med_size={t['median_obj_size']:.0f}px, "
                       f"edge={t['edge_tile']}")

    # Spearman r: IoU vs fg_ratio | 排序相关性
    from scipy.stats import spearmanr
    sr, _ = spearmanr(ious, fg_ratios)
    logger.log_info("b06/correlation",
                   f"Spearman r(IoU, fg_ratio) = {sr:.4f}")

    # ═══ 5. Visualization | 可视化 ═══
    logger.log_info("b06/viz", "Generating figure...")
    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Panel 1: IoU Histogram | IoU 分布直方图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(ious, bins=40, color="#3498DB", edgecolor="white", alpha=0.8)
    ax1.axvline(x=np.mean(ious), color="#E74C3C", linestyle="--", linewidth=2,
                label=f"Mean={np.mean(ious):.3f}")
    ax1.axvline(x=np.median(ious), color="#F39C12", linestyle="--", linewidth=2,
                label=f"Median={np.median(ious):.3f}")
    ax1.set_xlabel("Per-Tile FG-mIoU", fontsize=11)
    ax1.set_ylabel("Tile Count", fontsize=11)
    ax1.set_title(f"Tile IoU Distribution (n={len(ious)})\ntile级 IoU 分布", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Annotate zeros | 标注零值
    zero_count = int((ious == 0).sum())
    if zero_count > 0:
        ax1.annotate(f"{zero_count} tiles\nIoU=0", xy=(0, zero_count),
                    textcoords="offset points", xytext=(30, 30),
                    arrowprops=dict(arrowstyle="->", color="#E74C3C"),
                    fontsize=9, color="#E74C3C", fontweight="bold")

    # Panel 2: IoU vs fg_ratio scatter | 散点图
    ax2 = fig.add_subplot(gs[0, 1])
    # Color by valid class count | 按有效类数着色
    colors = np.clip(n_valid_arr, 0, 5)
    sc = ax2.scatter(fg_ratios, ious, c=colors, cmap="plasma",
                     alpha=0.5, s=15, edgecolors="none")
    ax2.set_xlabel("Foreground Ratio (fg_ratio)", fontsize=11)
    ax2.set_ylabel("Per-Tile FG-mIoU", fontsize=11)
    ax2.set_title(f"IoU vs Foreground Density\nSpearman r={sr:.3f}", fontsize=11)
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("# Valid Classes", fontsize=9)

    # Vertical lines for FG thresholds
    ax2.axvline(x=0.01, color="gray", linestyle=":", alpha=0.5)
    ax2.axvline(x=0.05, color="gray", linestyle=":", alpha=0.5)
    ax2.text(0.005, ax2.get_ylim()[1] * 0.95, "FG<1%", fontsize=7, ha="center")
    ax2.text(0.03, ax2.get_ylim()[1] * 0.95, "FG<5%", fontsize=7, ha="center")

    # Panel 3: Per-Class Avg IoU | 每类平均 IoU
    ax3 = fig.add_subplot(gs[0, 2])
    classes = list(range(1, NUM_OUT_CH))
    cls_names = [_ISAID_NAMES[c] for c in classes]
    cls_avgs = [per_cls_avg[c] for c in classes]
    cls_colors = ["#27AE60" if v > 0.3 else "#F39C12" if v > 0.1 else "#E74C3C"
                  for v in cls_avgs]
    bars = ax3.barh(range(len(classes)), cls_avgs, color=cls_colors, edgecolor="white")
    ax3.set_yticks(range(len(classes)))
    ax3.set_yticklabels(cls_names, fontsize=8)
    ax3.set_xlabel("Avg Per-Tile IoU", fontsize=11)
    ax3.set_title("Per-Class Average Tile IoU\n各类别平均 Tile IoU", fontsize=11)
    ax3.axvline(x=0.3, color="gray", linestyle="--", alpha=0.3)
    ax3.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, cls_avgs):
        if v > 0.01:
            ax3.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", fontsize=7, va="center")

    # Panel 4: IoU by FG ratio group (boxplot) | 按 FG 分组 boxplot
    ax4 = fig.add_subplot(gs[1, 0])
    group_data, group_labels = [], []
    for lo, hi in bins_fg:
        group = [t["miou"] for t in all_tile_stats if lo <= t["fg_ratio"] < hi]
        if group:
            group_data.append(group)
            group_labels.append(f"[{lo:.2f},{hi:.2f})")
    bp = ax4.boxplot(group_data, labels=group_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"],
                            plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(group_data)))):
        patch.set_facecolor(color)
    ax4.set_xlabel("FG Ratio Bin", fontsize=11)
    ax4.set_ylabel("Per-Tile IoU", fontsize=11)
    ax4.set_title("IoU Distribution by Foreground Ratio\n按前景占比分组的 IoU 分布", fontsize=11)
    ax4.grid(axis="y", alpha=0.3)
    ax4.tick_params(axis="x", rotation=45)

    # Panel 5: IoU vs Object Size (dominant class) | IoU vs 目标尺寸
    ax5 = fig.add_subplot(gs[1, 1])
    obj_sizes_raw = np.array([t["median_obj_size"] for t in all_tile_stats])
    has_objs = obj_sizes_raw > 0
    if has_objs.sum() > 0:
        ax5.scatter(np.log10(obj_sizes_raw[has_objs] + 1),
                   ious[has_objs],
                   c="steelblue", alpha=0.4, s=10, edgecolors="none")
        ax5.set_xlabel("log10(Median Object Size px)", fontsize=11)
        ax5.set_ylabel("Per-Tile IoU", fontsize=11)
        ax5.set_title("IoU vs Object Size\n(dominant class objects)", fontsize=11)
        ax5.grid(True, alpha=0.3)

    # Panel 6: Best/Worst tile summary (text + small thumbnails) | 文本摘要
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    summary_lines = [
        "DECODER DIAGNOSTIC SUMMARY",
        "=" * 35,
        "",
        f"Total tiles: {len(all_tile_stats)}",
        f"Mean IoU:   {np.mean(ious):.4f}",
        f"Median IoU: {np.median(ious):.4f}",
        f"Std IoU:    {np.std(ious):.4f}",
        f"IoU=0 tiles: {zero_count} ({zero_count/len(ious)*100:.1f}%)",
        f"IoU>0.5:     {(ious>0.5).sum()} ({(ious>0.5).sum()/len(ious)*100:.1f}%)",
        "",
        f"Spearman r(IoU, fg_ratio) = {sr:.3f}",
        "",
        "BEST tile characteristics:",
        f"  FG: {np.mean([t['fg_ratio'] for t in best_10]):.3f}",
        f"  Objs: {np.mean([t['n_objects'] for t in best_10]):.0f}",
        f"  Med size: {np.mean([t['median_obj_size'] for t in best_10]):.0f}px",
        f"  Edge: {sum(1 for t in best_10 if t['edge_tile'])}/10",
        "",
        "WORST tile characteristics:",
        f"  FG: {np.mean([t['fg_ratio'] for t in worst_10]):.3f}",
        f"  Objs: {np.mean([t['n_objects'] for t in worst_10]):.0f}",
        f"  Med size: {np.mean([t['median_obj_size'] for t in worst_10]):.0f}px",
        f"  Edge: {sum(1 for t in worst_10 if t['edge_tile'])}/10",
        "",
        "KEY FINDING:",
    ]

    # Diagnose the gap | 诊断 gap 原因
    if np.mean(ious) < 0.2 and (ious > 0.5).sum() / len(ious) < 0.1:
        summary_lines.append("  MOST tiles have low IoU (<0.2)")
        summary_lines.append("  → Decoder weak on majority of tiles")
    elif np.mean(ious) > 0.3:
        summary_lines.append("  Decoder performs reasonably")
    if sr < 0.3:
        summary_lines.append("  → fg_ratio weakly correlated with IoU")
    else:
        summary_lines.append("  → fg_ratio moderately predicts IoU")

    for i, line in enumerate(summary_lines):
        ypos = 9.5 - i * 0.42
        if line.startswith("DECODER") or line.startswith("KEY"):
            ax6.text(0.0, ypos, line, fontsize=11, fontweight="bold",
                    fontfamily="monospace", va="top")
        elif line.startswith("="):
            ax6.text(0.0, ypos, line, fontsize=9, fontfamily="monospace",
                    va="top", color="gray")
        elif "BEST" in line or "WORST" in line:
            ax6.text(0.0, ypos, line, fontsize=9, fontweight="bold",
                    fontfamily="monospace", va="top")
        else:
            ax6.text(0.0, ypos, line, fontsize=9, fontfamily="monospace", va="top")

    fig.suptitle("B-06: Decoder Diagnostic — Why 48% (Tile) → 30% (Full)?",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.savefig(output_dir / "decoder_diag.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.log_info("b06/viz", f"Figure saved: {output_dir / 'decoder_diag.png'}")

    # ═══ 6. Save Tile Visualizations (Best/Worst 10) | 保存 tile 可视化 ═══
    for tag, tiles in [("best", best_10), ("worst", worst_10)]:
        for i, t in enumerate(tiles):
            fig_t, axes_t = plt.subplots(1, 3, figsize=(15, 5))
            # RGB
            axes_t[0].imshow(t["_tile_rgb"])
            axes_t[0].set_title("Input Tile", fontsize=10)
            axes_t[0].axis("off")
            # GT
            axes_t[1].imshow(t["_gt_tile"], cmap="tab20", vmin=0, vmax=15)
            axes_t[1].set_title("GT Mask", fontsize=10)
            axes_t[1].axis("off")
            # Pred
            axes_t[2].imshow(t["_pred_tile"], cmap="tab20", vmin=0, vmax=15)
            axes_t[2].set_title(f"Pred (IoU={t['miou']:.3f})", fontsize=10)
            axes_t[2].axis("off")
            cls_name = _ISAID_NAMES.get(t["dominant_class"], "?")
            fig_t.suptitle(
                f"{tag.upper()} #{i+1} | IoU={t['miou']:.3f} | "
                f"FG={t['fg_ratio']:.3f} | Dominant={cls_name} | "
                f"Objs={t['n_objects']} | Size={t['median_obj_size']:.0f}px",
                fontsize=12, fontweight="bold")
            fig_t.tight_layout()
            fig_t.savefig(tile_viz_dir / f"{tag}_{i+1:02d}.png", dpi=100)
            plt.close(fig_t)

    logger.log_info("b06/viz", f"Tile visualizations saved: {tile_viz_dir}/")

    # ═══ 7. Save JSON Summary | 保存 JSON ═══
    summary = {
        "experiment": "B-06 Decoder Diagnostic",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {"n_images": len(images), "tile_size": TILE_SIZE},
        "statistics": {
            "n_tiles": len(all_tile_stats),
            "iou_mean": float(np.mean(ious)),
            "iou_median": float(np.median(ious)),
            "iou_std": float(np.std(ious)),
            "iou_zero_count": int(zero_count),
            "iou_gt_05_count": int((ious > 0.5).sum()),
            "fg_ratio_mean": float(np.mean(fg_ratios)),
            "spearman_r_iou_vs_fg": float(sr),
            "per_class_avg_iou": {str(c): v for c, v in per_cls_avg.items()},
        },
        "best_10": [{"miou": t["miou"], "fg_ratio": t["fg_ratio"],
                      "dominant_class": t["dominant_class"],
                      "n_objects": t["n_objects"],
                      "median_obj_size": t["median_obj_size"],
                      "edge_tile": t["edge_tile"],
                      "grid_pos": t["grid_pos"],
                      "img_id": t["_img_id"]} for t in best_10],
        "worst_10": [{"miou": t["miou"], "fg_ratio": t["fg_ratio"],
                       "dominant_class": t["dominant_class"],
                       "n_objects": t["n_objects"],
                       "median_obj_size": t["median_obj_size"],
                       "edge_tile": t["edge_tile"],
                       "grid_pos": t["grid_pos"],
                       "img_id": t["_img_id"]} for t in worst_10],
    }
    with open(output_dir / "decoder_diag.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.log_info("b06/done", f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
