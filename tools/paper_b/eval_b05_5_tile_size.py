#!/usr/bin/env python3
"""
B-05.5: Tile Size Ablation — 选择粒度对 mIoU 保留率的影响
============================================================

B-04/B-05 在 1024px tile 下: K=50% → 仅 6 tile/图, 保留率 ~68%。
如果 tile 更小 (512, 256), 选择单元更多, 能否显著提升保留率？

核心假设 | Core hypothesis:
    更细粒度的 tile (更多选择单元) → K% 下浪费更少 → mIoU 保留率更高。
    Finer tile granularity (more selection units) → less waste at K% → higher retention.

实验设计 | Design:
    对每个 tile 尺寸 [256, 384, 512, 768, 1024]:
        1. 切全图 → 逐个 tile 过 Decoder
        2. 用 Oracle fg_ratio 排序 (隔离 tile 尺寸效应)
        3. K=10/20/30/40/50/70/100% → 拼接 → 算 mIoU 保留率
        4. 对比: 相同 K%, 不同 tile 尺寸 → 保留率差多少?

关键对比 | Key comparison:
    | Tile Size | Tiles/Img | K=50% (n tiles) | K=50% 保留率? |
    |-----------|-----------|------------------|---------------|
    | 1024      | ~12       | 6                | 68% (B-05)    |
    | 512       | ~48       | 24               | ???           |
    | 256       | ~192      | 96               | ???           |

输出 | Output:
    runs/b05_5_tile_size/
    ├── tile_size_ablation.png      # 4-panel figure
    └── tile_size_ablation.json     # Summary

用法 | Usage::
    python tools/paper_b/eval_b05_5_tile_size.py \
        --src-root data/iSAID_processed \
        --decoder-ckpt runs/b04_v3/decoder_best.pt \
        --n-images 20 --tile-sizes 256,512,1024
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.decoder.light_decoder import LightDecoder

NUM_CLASSES = 15
NUM_OUT_CH = 16
K_VALUES = [10, 20, 30, 40, 50, 70, 100]
DEFAULT_TILE_SIZES = [256, 384, 512, 768, 1024]
STRIDE = 32  # FastSAM stride


# ═══════════════════════════════════════════════════════════════
# Helpers
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


def compute_miou(pred_mask, gt_mask):
    miou_v, valid = 0.0, 0
    for c in range(1, NUM_OUT_CH):
        pc = (pred_mask == c); tc = (gt_mask == c)
        inter = (pc & tc).sum(); union = (pc | tc).sum()
        if union > 0: miou_v += inter / union; valid += 1
    return miou_v / max(valid, 1)


# ═══════════════════════════════════════════════════════════════
# 核心: 按给定 tile_size 切图 → 逐 tile 推理 → 按 fg_ratio 排序
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def analyze_one_image(img_np, gt_full, tile_size, backbone, decoder, device):
    """
    按 tile_size 切全图 → 逐 tile 过 Decoder → 记录 fg_ratio + pred。

    :param img_np: [H, W, 3] uint8

    :param gt_full: [H, W] uint8

    :param tile_size: int backbone, decoder: models

    :param device: torch device

    :return: tiles: list of dicts with keys: y0, y1, x0, x1, th, tw (tile 在全图坐标 | full-image coords) pred       [th, tw] int64 prediction fg_ratio   float foreground pixel fraction gt_tile    [th, tw] uint8 GT for this tile n_ty, n_tx: int
    """
    H, W = img_np.shape[:2]
    n_ty = (H + tile_size - 1) // tile_size
    n_tx = (W + tile_size - 1) // tile_size

    tiles = []
    for ty in range(n_ty):
        for tx in range(n_tx):
            y0, y1 = ty * tile_size, min(ty * tile_size + tile_size, H)
            x0, x1 = tx * tile_size, min(tx * tile_size + tile_size, W)
            th, tw = y1 - y0, x1 - x0

            # 提取 tile RGB | Extract tile RGB
            tile_rgb = img_np[y0:y1, x0:x1]  # [th, tw, 3]

            # Pad to tile_size | 填充到 tile_size
            tile_padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            tile_padded[:th, :tw] = tile_rgb

            # Pad to multiple of 32 for FastSAM
            ph = (STRIDE - tile_size % STRIDE) % STRIDE
            pw = (STRIDE - tile_size % STRIDE) % STRIDE
            if ph > 0 or pw > 0:
                tile_padded = np.pad(tile_padded, ((0, ph), (0, pw), (0, 0)),
                                     mode="constant")
            pad_h, pad_w = tile_padded.shape[:2]

            # Forward through decoder | 推理
            tile_t = torch.from_numpy(
                tile_padded.astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)

            feats = backbone(tile_t)
            logit = decoder(feats, target_size=(pad_h, pad_w))
            pred_full = logit.argmax(dim=1).cpu().numpy()[0]  # [pad_h, pad_w]
            pred_tile = pred_full[:th, :tw]  # crop back to actual tile

            # 计算 fg_ratio | Compute fg_ratio
            gt_tile = gt_full[y0:y1, x0:x1]
            fg_ratio = float((gt_tile > 0).sum() / max(th * tw, 1))

            tiles.append({
                "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                "th": th, "tw": tw,
                "pred": pred_tile,
                "fg_ratio": fg_ratio,
                "gt_tile": gt_tile,
            })

    return tiles, n_ty, n_tx


# ═══════════════════════════════════════════════════════════════
# 动态选择 + 拼接 → mIoU
# ═══════════════════════════════════════════════════════════════

def evaluate_at_k(tiles, H, W, K_pct):
    """按 fg_ratio 排 top-K% tiles → 拼接全图 → mIoU."""
    n = len(tiles)
    nk = max(1, int(n * K_pct / 100))
    scores = np.array([t["fg_ratio"] for t in tiles])
    order = np.argsort(scores)[::-1]
    selected = set(order[:nk])

    pred_full = np.zeros((H, W), dtype=np.int64)
    gt_full = np.zeros((H, W), dtype=np.uint8)

    for i, t in enumerate(tiles):
        y0, y1 = t["y0"], t["y1"]
        x0, x1 = t["x0"], t["x1"]
        th, tw = t["th"], t["tw"]
        # GT 总是完整拼接 | GT always fully stitched
        gt_full[y0:y0+th, x0:x0+tw] = t["gt_tile"]
        if i in selected:
            pred_full[y0:y0+th, x0:x0+tw] = t["pred"]

    return compute_miou(pred_full, gt_full), len(selected)


# ═══════════════════════════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════════════════════════

def plot_tile_size_ablation(all_results, tile_sizes, output_path):
    """4-panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tile_sizes)))

    # Panel 1: mIoU vs K% (每种 tile size 一条线) | mIoU 保留率曲线
    ax = axes[0, 0]
    for ti, ts in enumerate(tile_sizes):
        r = all_results[ts]
        mious = [r[k]["miou_mean"] * 100 for k in K_VALUES]
        ax.plot(K_VALUES, mious, "o-", color=colors[ti], linewidth=2.5,
                markersize=9, label=f"{ts}px ({r['avg_tiles']:.0f} tiles/img)")

    # 标注 K=100% 基准 | Baseline
    ax.axhline(y=all_results[tile_sizes[0]][100]["miou_mean"] * 100,
               color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("K% (Tiles Selected)", fontsize=12)
    ax.set_ylabel("FG-mIoU (%)", fontsize=12)
    ax.set_title("mIoU vs Tile Selection Rate\n"
                 "不同 Tile 尺寸下的精度-选择率曲线", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    # Panel 2: Retention Rate vs K% | 保留率
    ax = axes[0, 1]
    for ti, ts in enumerate(tile_sizes):
        r = all_results[ts]
        rets = [r[k]["retention_mean"] * 100 for k in K_VALUES]
        ax.plot(K_VALUES, rets, "s-", color=colors[ti], linewidth=2.5,
                markersize=9, label=f"{ts}px")

    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=50, color="red", linestyle=":", alpha=0.4)
    ax.set_xlabel("K% (Tiles Selected)", fontsize=12)
    ax.set_ylabel("Retention (% of K=100% mIoU)", fontsize=12)
    ax.set_title("mIoU Retention vs Tile Selection Rate\n"
                 "归一化保留率 (K=100% = 100%)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 标注 K=50% 关键点 | Mark K=50% key point
    y_50 = max([all_results[ts][50]["retention_mean"] * 100 for ts in tile_sizes])
    ax.annotate(f"K=50%: best={y_50:.0f}%",
                xy=(50, y_50), xytext=(60, y_50 + 5),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red", fontweight="bold")

    # Panel 3: Tiles per Image vs K=50% Retention | 选择单元数 vs 保留率
    ax = axes[1, 0]
    avg_tiles_arr = [all_results[ts]["avg_tiles"] for ts in tile_sizes]
    ret50_arr = [all_results[ts][50]["retention_mean"] * 100 for ts in tile_sizes]
    ret70_arr = [all_results[ts][70]["retention_mean"] * 100 for ts in tile_sizes]
    ret30_arr = [all_results[ts][30]["retention_mean"] * 100 for ts in tile_sizes]

    ax.plot(avg_tiles_arr, ret70_arr, "D-", color="#27AE60", linewidth=2,
            markersize=10, label="K=70%")
    ax.plot(avg_tiles_arr, ret50_arr, "o-", color="#E67E22", linewidth=2.5,
            markersize=12, label="K=50%")
    ax.plot(avg_tiles_arr, ret30_arr, "s-", color="#E74C3C", linewidth=2,
            markersize=10, label="K=30%")

    for i, ts in enumerate(tile_sizes):
        ax.annotate(f"{ts}px", (avg_tiles_arr[i], ret50_arr[i]),
                    textcoords="offset points", xytext=(5, -15),
                    fontsize=8, color="#E67E22")
    ax.set_xlabel("Avg Tiles per Image (more = finer granularity)", fontsize=12)
    ax.set_ylabel("Retention @ K% (%)", fontsize=12)
    ax.set_title("Retention vs Selection Granularity\n"
                 "更多选择单元 → 更高保留率?", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(avg_tiles_arr) * 1.1)
    ax.set_ylim(0, 105)

    # Panel 4: Compute vs Retention trade-off | 计算量 vs 保留率
    ax = axes[1, 1]
    for ti, ts in enumerate(tile_sizes):
        r = all_results[ts]
        # x: tiles selected (normalized to K=100%) | y: retention
        n_tiles_k100 = r[100]["n_tiles_mean"]
        xs = [r[k]["n_tiles_mean"] / max(n_tiles_k100, 1) * 100 for k in K_VALUES]
        ys = [r[k]["retention_mean"] * 100 for k in K_VALUES]
        ax.plot(xs, ys, "o-", color=colors[ti], linewidth=2.5,
                markersize=9, label=f"{ts}px")

    ax.plot([0, 100], [0, 100], ":", color="gray", alpha=0.4, label="Linear")
    ax.set_xlabel("Tiles Processed (% of full)", fontsize=12)
    ax.set_ylabel("mIoU Retention (% of full)", fontsize=12)
    ax.set_title("Accuracy vs Compute Trade-off\n"
                 "(上方 = 更好 | higher = better)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    fig.suptitle("B-05.5: Tile Size Ablation — Finer Granularity → Higher Retention?",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path / "tile_size_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=str, default="data/iSAID_processed")
    p.add_argument("--decoder-ckpt", type=str, required=True)
    p.add_argument("--n-images", type=int, default=20)
    p.add_argument("--tile-sizes", type=str, default="256,384,512,768,1024")
    p.add_argument("--output-dir", type=str, default="runs/b05_5_tile_size")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device
    tile_sizes = [int(x.strip()) for x in args.tile_sizes.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("b05_5")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "b05_5.jsonl")))

    logger.log_info("b05_5/start",
                    f"B-05.5 Tile Size Ablation: {tile_sizes}")

    # ── 1. Load Data | 加载数据 ──
    from PIL import Image
    src_root = Path(args.src_root)
    with open(src_root / "train" / "annotations" / "instances_train.json") as f:
        coco = json.load(f)
    img_id_to_anns = {}
    for ann in coco["annotations"]:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    img_dir = src_root / "train" / "images"
    rng = np.random.RandomState(args.seed)
    candidates = [(img_info, str(img_dir / img_info["file_name"]),
                   img_id_to_anns.get(img_info["id"], []))
                  for img_info in coco["images"]
                  if (img_dir / img_info["file_name"]).exists()
                  and img_id_to_anns.get(img_info["id"], [])]
    rng.shuffle(candidates)
    images = candidates[:args.n_images]
    logger.log_info("b05_5/data", f"Images: {len(images)}")

    # ── 2. Load Model | 加载模型 ──
    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    decoder = LightDecoder(1280, 16).to(device)
    decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
    decoder.eval()
    logger.log_info("b05_5/model", f"Loaded decoder from {args.decoder_ckpt}")

    # ═══ 3. 对每个 tile size: 逐图处理所有 tile | Per tile size: process all tiles ═══
    all_results = {}

    for ts in tile_sizes:
        logger.log_info("b05_5/ts",
                        f"\n{'='*60}\n"
                        f"  Tile Size: {ts}px\n"
                        f"  {'='*60}")

        # 收集每张图的 tile 数据 | Collect per-image tile data
        all_image_data = []  # list of (tiles, H, W)
        total_tiles = 0

        for img_info, img_path, anns in tqdm(images, desc=f"  TS={ts}"):
            img_np = np.array(Image.open(img_path).convert("RGB"))
            H, W = img_np.shape[:2]
            gt_full = render_semantic_mask(anns, H, W)

            tiles, n_ty, n_tx = analyze_one_image(
                img_np, gt_full, ts, backbone, decoder, device
            )
            all_image_data.append((tiles, H, W))
            total_tiles += len(tiles)

        avg_tiles = total_tiles / len(images)
        logger.log_info("b05_5/ts",
                        f"Total tiles: {total_tiles}, avg/img: {avg_tiles:.0f}")

        # 对每个 K%: 逐图选择 + 拼接 → mIoU
        results_k = {k: {"mious": [], "retentions": [], "n_tiles": []}
                     for k in K_VALUES}

        for tiles, H, W in tqdm(all_image_data, desc=f"  Eval K%"):
            # K=100% 作为该图基线 | per-image baseline
            miou_100, n_full = evaluate_at_k(tiles, H, W, 100)
            results_k[100]["mious"].append(miou_100)
            results_k[100]["retentions"].append(1.0)
            results_k[100]["n_tiles"].append(n_full)

            for k in K_VALUES:
                if k == 100:
                    continue
                miou_k, n_sel = evaluate_at_k(tiles, H, W, k)
                results_k[k]["mious"].append(miou_k)
                results_k[k]["retentions"].append(miou_k / max(miou_100, 1e-8))
                results_k[k]["n_tiles"].append(n_sel)

        # 汇总 | Aggregate
        aggregated = {}
        for k in K_VALUES:
            aggregated[k] = {
                "miou_mean": float(np.mean(results_k[k]["mious"])),
                "miou_std": float(np.std(results_k[k]["mious"])),
                "retention_mean": float(np.mean(results_k[k]["retentions"])),
                "n_tiles_mean": float(np.mean(results_k[k]["n_tiles"])),
            }

        aggregated["avg_tiles"] = avg_tiles
        all_results[ts] = aggregated

        # 打印 | Print
        logger.log_info("b05_5/ts",
                        f"  {'K%':<8} {'mIoU':>10} {'Retention':>10} {'Tiles':>8}")
        for k in K_VALUES:
            a = aggregated[k]
            logger.log_info("b05_5/ts",
                            f"  {k:<8}% {a['miou_mean']*100:>9.2f}% "
                            f"{a['retention_mean']*100:>9.1f}% {a['n_tiles_mean']:>7.0f}")

    # ═══ 4. Visualization ═══
    logger.log_info("b05_5/viz", "Generating figure...")
    plot_tile_size_ablation(all_results, tile_sizes, output_dir)

    # ═══ 5. Summary Table | 汇总表 ═══
    logger.log_info("b05_5/summary",
                    f"\n{'='*80}\n"
                    f"  TILE SIZE ABLATION — KEY FINDINGS\n"
                    f"  {'='*80}")
    logger.log_info("b05_5/summary",
                    f"  {'Tile Size':<12} {'Tiles/Img':>12} {'K=50% Ret':>14} "
                    f"{'K=30% Ret':>12} {'K=100% mIoU':>12}")
    logger.log_info("b05_5/summary",
                    f"  {'─'*12} {'─'*12} {'─'*14} {'─'*12} {'─'*12}")

    best_ret_ts = max(tile_sizes, key=lambda ts: all_results[ts][50]["retention_mean"])
    best_ret = all_results[best_ret_ts][50]["retention_mean"] * 100

    for ts in tile_sizes:
        r = all_results[ts]
        logger.log_info("b05_5/summary",
                        f"  {ts:>6}px     {r['avg_tiles']:>8.0f}      "
                        f"{r[50]['retention_mean']*100:>8.1f}%      "
                        f"{r[30]['retention_mean']*100:>8.1f}%     "
                        f"{r[100]['miou_mean']*100:>8.2f}%")

    logger.log_info("b05_5/conclusion",
                    f"\n  Best K=50% retention: {best_ret:.1f}% @ {best_ret_ts}px\n"
                    f"  → Finer granularity {'DOES' if best_ret_ts < 1024 else 'does NOT'} "
                    f"significantly improve retention")

    # ═══ 6. Save JSON ═══
    summary_json = {
        "experiment": "B-05.5 Tile Size Ablation",
        "timestamp": datetime.datetime.now().isoformat(),
        "config": {"n_images": len(images), "tile_sizes": tile_sizes},
        "results": {
            str(ts): {
                "avg_tiles_per_image": all_results[ts]["avg_tiles"],
                **{str(k): {
                    "miou_mean": all_results[ts][k]["miou_mean"],
                    "retention_mean": all_results[ts][k]["retention_mean"],
                    "n_tiles_mean": all_results[ts][k]["n_tiles_mean"],
                } for k in K_VALUES},
            } for ts in tile_sizes
        },
    }
    with open(output_dir / "tile_size_ablation.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    logger.log_info("b05_5/done", f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
