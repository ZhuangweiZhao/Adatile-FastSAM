#!/usr/bin/env python3
"""
B-04 诊断实验 1 & 2 | Diagnostics Exp 1 & 2
============================================

实验 1: Meaningful tiles (FG>5%), 验证类别学习; 每类 IoU
实验 2: Binary Segmentation (FG/BG), 验证定位能力
日志: Console + File(JSONL), 每5 epoch, 含per-class breakdown

用法::
    python tools/diag_b04_exp12.py --tile-root /root/autodl-tmp/iSAID_tiles
"""

import sys, json, datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
from adatile.datasets.isaid_tiles import FastISAIDTileDataset
from adatile.backbone import FastSAMBackbone
from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.decoder.light_decoder import LightDecoder

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 颜色表: 15 个前景类别配色 | Color map: 15 foreground class colors
COLOR_MAP = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
             (0,255,255),(128,0,0),(0,128,0),(0,0,128),(128,128,0),
             (128,0,128),(0,128,128),(255,128,0),(255,0,128),(128,255,0)]


def train_eval(decoder, backbone, loader, epochs, device, binary, num_classes, log):
    """
    训练与评估循环 | Training and evaluation loop.

    每 5 epoch 评估一次 per-class IoU，记录完整历史。
        Evaluates per-class IoU every 5 epochs, records full history.

    :param decoder: LightDecoder 模型 | LightDecoder model

    :param backbone: 冻结的 FastSAM backbone | Frozen FastSAM backbone

    :param loader: DataLoader

    :param epochs: 训练 epoch 数 | Number of training epochs

    :param device: cuda/cpu

    :param binary: True → FG/BG 二分类 | True → FG/BG binary classification

    :param num_classes: 输出通道数 | Number of output channels

    :param log: 日志回调函数 | Logging callback function

    :return: history: list of dict with keys [epoch, loss, miou, bg_pred, per_class_iou]
    """
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    history = []

    for epoch in range(1, epochs + 1):
        decoder.train()
        total_loss, n = 0.0, 0

        for batch in loader:
            img = batch["image"].to(device)
            tgt = batch["mask"].to(device)
            if binary:
                tgt = (tgt > 0).long()

            feats = backbone(img)
            logit = decoder(feats, target_size=tgt.shape[1:])

            ce = F.cross_entropy(logit, tgt, ignore_index=255, reduction="none")
            loss = ((1 - torch.exp(-ce)) ** 5.0 * ce).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1

        # ── 评估: 每 5 epoch 或 epoch 1 | Evaluate: every 5 or epoch 1 ──
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            decoder.eval()
            per_class_inter = torch.zeros(num_classes, device=device)
            per_class_union = torch.zeros(num_classes, device=device)
            bg_px, total_px = 0.0, 0
            all_pred_vals, all_tgt_vals = set(), set()

            with torch.no_grad():
                for batch in loader:
                    img = batch["image"].to(device)
                    tgt = batch["mask"].to(device)
                    if binary: tgt = (tgt > 0).long()
                    feats = backbone(img)
                    logit = decoder(feats, target_size=tgt.shape[1:])
                    pred = logit.argmax(dim=1)
                    probs = F.softmax(logit, dim=1)

                    all_pred_vals.update(pred.unique().cpu().tolist())
                    all_tgt_vals.update(tgt.unique().cpu().tolist())
                    bg_px += (pred == 0).float().sum().item()
                    total_px += pred.numel()

                    nc = 2 if binary else num_classes
                    for c in range(1, nc):  # skip bg
                        pc = (pred == c); tc = (tgt == c)
                        per_class_inter[c] += (pc & tc).sum().float()
                        per_class_union[c] += (pc | tc).sum().float()

            # Per-class + global IoU
            per_cls_iou = {}
            miou_sum, valid = 0.0, 0
            for c in range(1, num_classes if not binary else 2):
                iou_c = (per_class_inter[c] / (per_class_union[c] + 1e-8)).item()
                if per_class_union[c] > 0:
                    per_cls_iou[c] = round(iou_c, 4)
                    miou_sum += iou_c
                    valid += 1

            miou = miou_sum / max(valid, 1)
            bg_pct = bg_px / max(total_px, 1)

            # 日志: 主行 + per-class 细节 | Log: main line + per-class detail
            mode = "BIN" if binary else "MC"
            log("exp12/epoch",
                f"[{mode}] E{epoch:3d} loss={total_loss/n:.4f} "
                f"mIoU={miou:.4f} bg_pred={bg_pct:.3f} "
                f"n_classes_active={valid}/{num_classes} "
                f"pred={sorted(all_pred_vals)} gt={sorted(all_tgt_vals)}")
            log("exp12/per_class",
                f"[{mode}] E{epoch:3d} IoU: {per_cls_iou}")

            history.append({"epoch": epoch, "loss": total_loss/n,
                           "miou": miou, "bg_pred": bg_pct,
                           "per_class_iou": per_cls_iou})
            decoder.train()

    # 最终 per-class 详情 | Final per-class details
    final_pc = history[-1]["per_class_iou"]
    log("exp12/final_per_class", f"Final per-class IoU | 最终各类别 IoU: {final_pc}")

    return history


def viz_pred(model, backbone, loader, output_path, device, binary, log):
    """保存拼接可视化 (5 列对照图) + 日志 | Save composite visualization (5-column comparison) + log.

    输出 5 列: 原图 | GT 掩码 | 预测掩码 | GT 叠加 | 预测叠加
    Output 5 columns: Image | GT Mask | Pred Mask | GT Overlay | Pred Overlay
    """
    model.eval()
    for i, batch in enumerate(loader):
        if i >= 3: break  # 最多 3 张 | Max 3 images
        img = batch["image"][:1].to(device)
        tgt = batch["mask"][:1].to(device)
        tgt_disp = (tgt > 0).long() if binary else tgt  # 二分类模式下只区分 FG/BG | Binary mode: FG/BG only

        feats = backbone(img)
        logit = model(feats, target_size=tgt.shape[1:])
        pred = logit.argmax(dim=1)[0].cpu().numpy()
        gt = tgt_disp[0].cpu().numpy()
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)

        # 上色: GT 和 Pred 各自着色 | Colorize: GT and Pred separately
        nc = 2 if binary else 16
        gt_c, pred_c = (np.zeros((*gt.shape, 3), dtype=np.uint8) for _ in range(2))
        for c in range(1, nc):
            gt_c[gt == c] = COLOR_MAP[c % len(COLOR_MAP)]
            pred_c[pred == c] = COLOR_MAP[c % len(COLOR_MAP)]
        # 叠加: 50% 原图 + 50% 着色掩码 | Overlay: 50% original + 50% colorized mask
        gt_ov = img_np * 0.5 + gt_c.astype(np.float32) / 255.0 * 0.5
        pred_ov = img_np * 0.5 + pred_c.astype(np.float32) / 255.0 * 0.5

        # 5 列子图 | 5-column subplot
        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        for ax, title, im in zip(axes,
                                  ["Image", "GT", "Pred", "GT Overlay", "Pred Overlay"],
                                  [img_np, gt_c, pred_c, gt_ov, pred_ov]):
            ax.imshow(im); ax.set_title(title); ax.axis("off")

        gt_cls = sorted(set(gt.flatten()) - {0})
        pred_cls = sorted(set(pred.flatten()) - {0})
        mode = "BIN" if binary else "MC"
        fig.suptitle(f"[{mode}] GT={gt_cls} Pred={pred_cls}", fontsize=10, fontfamily="monospace")
        fig.tight_layout()
        fig.savefig(output_path.replace(".png", f"_s{i}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

        log("exp12/viz",
            f"Sample{i}: GT={gt_cls} Pred={pred_cls} "
            f"GT_bg={(gt==0).mean():.3f} Pred_bg={(pred==0).mean():.3f}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tile-root", type=str, default="/root/autodl-tmp/iSAID_tiles")
    p.add_argument("--output-dir", type=str, default="/root/autodl-tmp/b04_exp12")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ── Logger: Console + File | 日志: 终端 + 文件 ──
    log_path = str(output_dir / "exp12.jsonl")
    logger = get_logger("b04_exp12")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(log_path))

    def log(key, msg):
        logger.log_info(key, msg)
        print(f"  {msg}")  # 双保险: 确保终端可见 | Belt-and-suspenders: ensure visible

    log("exp12/start", f"B-04 Exp1&2 | output={output_dir} | log={log_path}")
    log("exp12/device", f"Device: {device}")

    train_ds = FastISAIDTileDataset(args.tile_root, split="train", semantic=True)
    log("exp12/data", f"Train tiles: {len(train_ds)}")

    # ═══════════════════════════════════════════════════════════════
    # 共同数据准备: 筛选 FG>5% tile | Common data: filter FG>5%
    # ═══════════════════════════════════════════════════════════════
    log("exp12/filter", "Filtering tiles with FG > 5%...")
    fg5_tiles, fg_ratios = [], []
    for fname in tqdm(train_ds._tiles, desc="  Filter FG>5%"):
        mask = np.array(Image.open(train_ds._mask_dir / fname))
        fg_r = (mask > 0).sum() / mask.size
        fg_ratios.append(fg_r)
        if fg_r > 0.05:
            fg5_tiles.append(fname)

    fg_arr = np.array(fg_ratios)
    log("exp12/filter",
        f"FG>5%: {len(fg5_tiles)}/{len(train_ds._tiles)} "
        f"({len(fg5_tiles)/len(train_ds._tiles)*100:.1f}%) | "
        f"All-FG: mean={fg_arr.mean():.4f} median={np.median(fg_arr):.4f} "
        f"max={fg_arr.max():.4f}")

    if len(fg5_tiles) > 1000:
        np.random.seed(42)
        fg5_tiles = list(np.random.choice(fg5_tiles, 1000, replace=False))
        log("exp12/filter", f"Sampled 1000 for speed")

    # 统计这些 tile 里的类别分布 | Class distribution in selected tiles
    all_cls = set()
    for fname in tqdm(fg5_tiles[:500], desc="  Class stats"):
        mask = np.array(Image.open(train_ds._mask_dir / fname))
        all_cls.update(mask.flatten().tolist())
    log("exp12/classes",
        f"Classes in FG>5% tiles: {sorted(all_cls)} "
        f"({len(all_cls)}/{16} present)")

    train_ds._tiles = fg5_tiles
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    backbone = FastSAMBackbone(freeze_backbone=True).eval()

    # ═══════════════════════════════════════════════════════════════
    # 实验 1: Multi-class, FG>5% tiles
    # ═══════════════════════════════════════════════════════════════
    log("exp12/exp1", "=" * 60)
    log("exp12/exp1", "Experiment 1: Multi-class (16ch) on FG>5% tiles × 100 epochs")
    log("exp12/exp1", "=" * 60)

    decoder1 = LightDecoder(1280, 16).to(device)
    n_p = sum(p.numel() for p in decoder1.parameters())
    log("exp12/exp1", f"Decoder: {n_p:,} params, 16 output channels")

    history1 = train_eval(decoder1, backbone, loader, 100, device, False, 16, log)

    h1 = history1[-1]
    log("exp12/exp1", f"FINAL: FG-mIoU={h1['miou']:.4f} loss={h1['loss']:.4f}")

    # Viz
    val_ds = FastISAIDTileDataset(args.tile_root, split="val", semantic=True)
    val_fg = []
    for fname in val_ds._tiles:
        mask = np.array(Image.open(val_ds._mask_dir / fname))
        if (mask > 0).sum() / mask.size > 0.05:
            val_fg.append(fname)
    val_ds._tiles = val_fg[:10]
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    viz_pred(decoder1, backbone, val_loader, str(output_dir / "exp1_viz"), device, False, log)

    # ═══════════════════════════════════════════════════════════════
    # 实验 2: Binary, same tiles
    # ═══════════════════════════════════════════════════════════════
    log("exp12/exp2", "=" * 60)
    log("exp12/exp2", "Experiment 2: Binary (2ch FG/BG) on same FG>5% tiles × 100 epochs")
    log("exp12/exp2", "=" * 60)

    train_ds._tiles = fg5_tiles
    loader_bin = DataLoader(train_ds, batch_size=8, shuffle=True)
    decoder2 = LightDecoder(1280, 2).to(device)
    log("exp12/exp2", f"Decoder: {n_p:,} params, 2 output channels (BG, FG)")

    history2 = train_eval(decoder2, backbone, loader_bin, 100, device, True, 2, log)

    h2 = history2[-1]
    log("exp12/exp2", f"FINAL: Binary-IoU={h2['miou']:.4f} loss={h2['loss']:.4f}")

    viz_pred(decoder2, backbone, val_loader, str(output_dir / "exp2_viz"), device, True, log)

    # ═══════════════════════════════════════════════════════════════
    # 汇总 | Summary
    # ═══════════════════════════════════════════════════════════════
    log("exp12/summary", "=" * 60)
    log("exp12/summary", "SUMMARY | 汇总")
    log("exp12/summary", "=" * 60)

    miou1, miou2 = h1["miou"], h2["miou"]

    # 过拟合基线 | Overfit reference
    log("exp12/summary", f"  {'Experiment':<40} {'mIoU':>8}  {'Note':<20}")
    log("exp12/summary", f"  {'─'*70}")
    log("exp12/summary", f"  {'Previous: 20-tile overfit (multi-class)':<40} {0.692:>8.3f}  {'Decoder CAPABLE'}")
    log("exp12/summary", f"  {'Exp1: Multi-class, FG>5% tiles':<40} {miou1:>8.4f}  {'Class learning test'}")
    log("exp12/summary", f"  {'Exp2: Binary, FG>5% tiles':<40} {miou2:>8.4f}  {'Localization test'}")

    if miou2 > 0.6 and miou1 < 0.3:
        log("exp12/verdict",
            f"✅ LOCALIZATION WORKS (Binary IoU={miou2:.3f}), "
            f"CLASS LEARNING FAILS (Multi-class={miou1:.3f}). "
            f"Problem = class imbalance, NOT feature resolution. | 问题 = 类别不均衡，非特征分辨率")
    elif miou2 > 0.6 and miou1 >= 0.3:
        log("exp12/verdict",
            f"✅ BOTH work. FG>5% filter + Focal γ=5.0 fix the issue. "
            f"Train B-04 with FG>5% filter. | 两者均正常，用 FG>5% 过滤训练 B-04")
    elif miou2 < 0.3:
        log("exp12/verdict",
            f"❌ EVEN BINARY fails ({miou2:.3f}). Decoder or feature is fundamentally broken. | 二分类都失败，Decoder 或特征根本有问题")
    else:
        log("exp12/verdict",
            f"⚠️ Mixed: binary={miou2:.3f} multi={miou1:.3f}. Need further diagnosis. | 混合结果，需进一步诊断")

    # 画对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, hist, title, color in zip(axes,
                                       [history1, history2],
                                       ["Exp1: Multi-class (FG>5%)", "Exp2: Binary (FG>5%)"],
                                       ["#E74C3C", "#27AE60"]):
        epochs = [h["epoch"] for h in hist]
        mious = [h["miou"] for h in hist]
        ax.plot(epochs, mious, "o-", color=color, lw=2.5, ms=6)
        # 标注最终值 | Label final value
        ax.annotate(f"{mious[-1]:.3f}", (epochs[-1], mious[-1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("mIoU")
        ax.set_title(title); ax.grid(alpha=0.3)
        ax.set_ylim(0, max(1.0, max(mious) * 1.15))

    fig.suptitle("B-04 Diagnostics: Can the model segment objects?",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "exp12_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 保存完整历史 | Save full history
    with open(output_dir / "exp12_history.json", "w") as f:
        json.dump({"exp1": history1, "exp2": history2, "timestamp": datetime.datetime.now().isoformat()}, f, indent=2)

    log("exp12/done", f"All results: {output_dir}/")


if __name__ == "__main__":
    main()
