#!/usr/bin/env python3
"""
B-07: Contribution Ground Truth — 定义并计算每个 tile 的真实贡献
==================================================================

此前"Contribution"概念一直被松散使用。本实验正式定义并计算:

    Contribution(tile_i) = mIoU(all tiles) - mIoU(remove tile_i)

即: 删除 tile_i 导致的全图 mIoU 损失.
正值 = 该 tile 贡献为正(删除后变差). 负值 = 该 tile 贡献为负(删除后变好).

输出:
    {output_dir}/
    ├── contribution_labels.json       # 每个 tile 的贡献标签
    └── contribution_analysis.png      # 分布 + vs fg_ratio 散点

用法::
    # Vaihingen
    python tools/paper_b/eval_b07_contribution_gt.py \
        --tile-root data/Vaihingen --dataset vaihingen \
        --decoder-ckpt runs/paper_b_vaihingen/decoder_best.pt

    # iSAID
    python tools/paper_b/eval_b07_contribution_gt.py \
        --tile-root data/iSAID_tiles --dataset isaid \
        --decoder-ckpt runs/b04_v3/decoder_best.pt
"""

import sys, argparse, json, datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.backbone import FastSAMBackbone
from adatile.decoder.light_decoder import LightDecoder

TILE_SIZE = 1024
STRIDE = 32
DATASET_CONFIGS = {"isaid": 16, "vaihingen": 7}

try:
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


def load_dataset(tile_root, dataset_name, split="val"):
    if (Path(tile_root) / split).exists() or (Path(tile_root) / "images" / split).exists():
        use_split = split
    else:
        use_split = "test"
    if dataset_name == "isaid":
        from adatile.datasets.isaid_tiles import FastISAIDTileDataset
        return FastISAIDTileDataset(tile_root, split=use_split, dense_labels=True)
    from adatile.datasets.vaihingen_tiles import VaihingenTileDataset
    return VaihingenTileDataset(tile_root, split=use_split, dense_labels=True)


def compute_miou(pred, gt, num_classes):
    miou_v, valid = 0.0, 0
    for c in range(1, num_classes):
        pc = (pred == c); tc = (gt == c)
        inter = (pc & tc).sum(); union = (pc | tc).sum()
        if union > 0: miou_v += inter / union; valid += 1
    return miou_v / max(valid, 1)


@torch.no_grad()
def compute_contribution_labels(images, backbone, decoder, num_classes, device):
    """
    对每张图: 逐 tile 跑 Decoder → Leave-One-Out → Contribution_i = ΔmIoU.

    :return: labels: list of dicts, each with {img_idx, tile_idx, y0,y1,x0,x1, tile_iou, fg_ratio, contribution, n_tiles_in_image}
    """
    labels = []

    for img_idx in tqdm(range(len(images)), desc="  Contribution GT"):
        sample = images[img_idx]
        img = sample["image"]  # [3, H, W]
        gt = sample["mask"].squeeze() if sample["mask"].dim() == 3 else sample["mask"]
        H, W = gt.shape
        n_ty = (H + TILE_SIZE - 1) // TILE_SIZE
        n_tx = (W + TILE_SIZE - 1) // TILE_SIZE
        n_tiles = n_ty * n_tx

        # 逐 tile 过 Decoder (只做一次)
        tiles = []
        for ty in range(n_ty):
            for tx in range(n_tx):
                y0, y1 = ty * TILE_SIZE, min(ty * TILE_SIZE + TILE_SIZE, H)
                x0, x1 = tx * TILE_SIZE, min(tx * TILE_SIZE + TILE_SIZE, W)
                th, tw = y1 - y0, x1 - x0

                tile_rgb = img[:, y0:y1, x0:x1]
                if th < TILE_SIZE or tw < TILE_SIZE:
                    p = torch.zeros(3, TILE_SIZE, TILE_SIZE)
                    p[:, :th, :tw] = tile_rgb; tile_rgb = p
                ph = (STRIDE - TILE_SIZE % STRIDE) % STRIDE
                pw = (STRIDE - TILE_SIZE % STRIDE) % STRIDE
                if ph > 0 or pw > 0:
                    tile_rgb = F.pad(tile_rgb, (0, pw, 0, ph))

                tile_t = tile_rgb.unsqueeze(0).to(device)
                feats = backbone(tile_t)
                logit = decoder(feats, target_size=(TILE_SIZE + ph, TILE_SIZE + pw))
                pred_tile = logit.argmax(dim=1).cpu().numpy()[0, :th, :tw]
                gt_tile = gt[y0:y1, x0:x1].cpu().numpy()

                per_tile_iou = compute_miou(pred_tile, gt_tile, num_classes)
                fg_ratio = float((gt_tile > 0).sum() / max(th * tw, 1))

                tiles.append({
                    "pred": pred_tile, "gt": gt_tile,
                    "y0": y0, "y1": y1, "x0": x0, "x1": x1,
                    "th": th, "tw": tw,
                    "per_tile_iou": per_tile_iou,
                    "fg_ratio": fg_ratio,
                })

        # K=100% 基线: 所有 tile 拼接
        gt_full = np.zeros((H, W), dtype=np.uint8)
        pred_all = np.zeros((H, W), dtype=np.int64)
        for t in tiles:
            gt_full[t["y0"]:t["y0"]+t["th"], t["x0"]:t["x0"]+t["tw"]] = t["gt"]
            pred_all[t["y0"]:t["y0"]+t["th"], t["x0"]:t["x0"]+t["tw"]] = t["pred"]
        miou_all = compute_miou(pred_all, gt_full, num_classes)

        # Leave-One-Out: 逐个 tile 删除 → 算 ΔmIoU
        for i, t in enumerate(tiles):
            pred_loo = np.zeros((H, W), dtype=np.int64)
            for j, tj in enumerate(tiles):
                if j == i:
                    continue
                pred_loo[tj["y0"]:tj["y0"]+tj["th"],
                         tj["x0"]:tj["x0"]+tj["tw"]] = tj["pred"]
            miou_loo = compute_miou(pred_loo, gt_full, num_classes)
            contribution = miou_all - miou_loo

            labels.append({
                "img_idx": img_idx,
                "tile_idx": i,
                "y0": t["y0"], "y1": t["y1"],
                "x0": t["x0"], "x1": t["x1"],
                "per_tile_iou": float(t["per_tile_iou"]),
                "fg_ratio": float(t["fg_ratio"]),
                "contribution": float(contribution),
                "miou_all": float(miou_all),
                "n_tiles": n_tiles,
            })

    return labels


def analyze_and_plot(labels, dataset_name, output_dir, logger):
    """分析与可视化."""
    contribs = np.array([l["contribution"] for l in labels])
    per_tile_ious = np.array([l["per_tile_iou"] for l in labels])
    fg_ratios = np.array([l["fg_ratio"] for l in labels])

    # 基本统计
    logger.log_info("stats", f"Total tiles: {len(labels)}")
    logger.log_info("stats", f"Contribution range: [{contribs.min():.4f}, {contribs.max():.4f}]")
    logger.log_info("stats", f"Contribution mean±std: {contribs.mean():.4f}±{contribs.std():.4f}")
    logger.log_info("stats", f"Positive contrib: {(contribs>0).sum()} ({(contribs>0).mean()*100:.1f}%)")
    logger.log_info("stats", f"Negative contrib: {(contribs<0).sum()} ({(contribs<0).mean()*100:.1f}%)")

    # 相关性
    sr_contrib_fg, _ = spearmanr(contribs, fg_ratios)
    sr_contrib_iou, _ = spearmanr(contribs, per_tile_ious)
    sr_fg_iou, _ = spearmanr(fg_ratios, per_tile_ious)
    logger.log_info("corr", f"Spearman r(Contribution, fg_ratio)  = {sr_contrib_fg:.4f}")
    logger.log_info("corr", f"Spearman r(Contribution, per_tile_IoU) = {sr_contrib_iou:.4f}")
    logger.log_info("corr", f"Spearman r(fg_ratio, per_tile_IoU)     = {sr_fg_iou:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.hist(contribs, bins=40, color="#3498DB", edgecolor="white", alpha=0.8)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(x=contribs.mean(), color="orange", linestyle="--", label=f"Mean={contribs.mean():.4f}")
    ax.set_xlabel("Contribution (ΔmIoU)", fontsize=11)
    ax.set_ylabel("Tile Count", fontsize=11)
    ax.set_title("Contribution Distribution\n贡献分布 (正=删除后变差, 负=删除后变好)", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.scatter(fg_ratios, contribs, c="steelblue", alpha=0.4, s=12, edgecolors="none")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("fg_ratio", fontsize=11)
    ax.set_ylabel("Contribution (ΔmIoU)", fontsize=11)
    ax.set_title(f"Contribution vs fg_ratio\nSpearman r={sr_contrib_fg:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(per_tile_ious, contribs, c="darkorange", alpha=0.4, s=12, edgecolors="none")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Per-Tile IoU", fontsize=11)
    ax.set_ylabel("Contribution (ΔmIoU)", fontsize=11)
    ax.set_title(f"Contribution vs Per-Tile IoU\nSpearman r={sr_contrib_iou:.3f}", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"B-07: Contribution Ground Truth — {dataset_name}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "contribution_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 关键结论
    if sr_contrib_iou > sr_contrib_fg + 0.15:
        logger.log_info("conclusion",
            "Per-tile IoU is a STRONGER proxy for true contribution than fg_ratio.")
        logger.log_info("conclusion",
            "→ Training a Contribution Predictor on per-tile IoU labels is justified.")
    elif abs(sr_contrib_iou - sr_contrib_fg) < 0.1:
        logger.log_info("conclusion",
            "Per-tile IoU and fg_ratio are similarly correlated with true contribution.")
    logger.log_info("conclusion",
        f"{(contribs<0).mean()*100:.0f}% of tiles have negative contribution "
        f"(deleting them IMPROVES mIoU).")

    return {
        "n_tiles": len(labels),
        "contribution_mean": float(contribs.mean()),
        "contribution_std": float(contribs.std()),
        "spearman_contrib_fg": float(sr_contrib_fg),
        "spearman_contrib_iou": float(sr_contrib_iou),
        "spearman_fg_iou": float(sr_fg_iou),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-root", type=str, required=True)
    p.add_argument("--dataset", type=str, default="vaihingen", choices=["isaid", "vaihingen"])
    p.add_argument("--decoder-ckpt", type=str, required=True)
    p.add_argument("--n-images", type=int, default=50)
    p.add_argument("--output-dir", type=str, default="runs/b07_contribution_gt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    num_classes = DATASET_CONFIGS[args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("b07")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(output_dir / "b07.jsonl")))

    # 加载模型
    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    decoder = LightDecoder(1280, num_classes).to(device)
    decoder.load_state_dict(torch.load(args.decoder_ckpt, map_location=device))
    decoder.eval()
    logger.log_info("model", f"Loaded decoder from {args.decoder_ckpt}")

    # 加载数据
    ds = load_dataset(args.tile_root, args.dataset)
    n_imgs = min(args.n_images, len(ds))
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(ds), n_imgs, replace=False)
    images = [ds[i] for i in indices]
    logger.log_info("data", f"Loaded {n_imgs} images from {args.dataset}")

    # 计算贡献标签
    labels = compute_contribution_labels(images, backbone, decoder, num_classes, device)

    # 分析 + 可视化
    stats = analyze_and_plot(labels, args.dataset, output_dir, logger)

    # 保存标签
    label_data = {
        "experiment": "B-07 Contribution Ground Truth",
        "dataset": args.dataset,
        "timestamp": datetime.datetime.now().isoformat(),
        "n_images": n_imgs,
        "n_tiles": len(labels),
        "statistics": stats,
        "labels": [{k: v for k, v in l.items()} for l in labels],
    }
    with open(output_dir / "contribution_labels.json", "w") as f:
        json.dump(label_data, f, indent=2)

    logger.log_info("done", f"Saved {len(labels)} contribution labels to {output_dir}/")


if __name__ == "__main__":
    main()
