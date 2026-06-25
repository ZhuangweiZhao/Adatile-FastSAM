#!/usr/bin/env python3
"""
B-04 诊断: 可视化 + Overfit 测试 | Diagnostics: Visualization + Overfit
======================================================================

① 可视化: 保存 GT vs Pred 的对照图, 看模型到底预测了什么
② Overfit: 只训练 20 tile, 100 epoch, 验证模型能力上限

用法 | Usage::
    python tools/diag_b04_overfit.py --tile-root /root/autodl-tmp/iSAID_tiles
"""

import sys
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
from adatile.logging.backends import ConsoleBackend
from adatile.decoder.light_decoder import LightDecoder
from adatile.utils.seed import get_worker_init_fn

logger = get_logger("b04_diag")
logger.add_backend(ConsoleBackend())

PIL_AVAILABLE = True
try:
    from PIL import Image
except ImportError:
    PIL_AVAILABLE = False

NUM_OUT_CH = 16  # 输出通道数 (含背景) | Output channels (including background)
NUM_CLASSES = 15  # 前景类别数 | Foreground classes count


# ═══════════════════════════════════════════════════════════════════
# ① 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize(model, backbone, loader, output_dir, device, n=5):
    """保存拼接对照图: 原图 | GT | Pred | GT Overlay | Pred Overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0
    for batch in loader:
        if saved >= n:
            break
        img = batch["image"][:1].to(device)
        tgt = batch["mask"][:1].to(device)

        feats = backbone(img)
        logit = model(feats, target_size=tgt.shape[1:])
        pred = logit.argmax(dim=1)[0].cpu().numpy()
        gt = tgt[0].cpu().numpy()
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)

        # 上色 | Colorize
        gt_color = np.zeros((*gt.shape, 3), dtype=np.uint8)
        pred_color = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for c in range(1, 16):
            gt_color[gt == c] = COLOR_MAP[c % len(COLOR_MAP)]
            pred_color[pred == c] = COLOR_MAP[c % len(COLOR_MAP)]

        gt_overlay = (img_np * 0.5 + gt_color.astype(np.float32) / 255.0 * 0.5)
        pred_overlay = (img_np * 0.5 + pred_color.astype(np.float32) / 255.0 * 0.5)

        # 拼接 5 列: 原图 | GT mask | Pred mask | GT Overlay | Pred Overlay
        fig, axes = plt.subplots(1, 5, figsize=(22, 5))

        titles = ["Image", "GT Mask", "Pred Mask", "GT Overlay", "Pred Overlay"]
        images = [img_np, gt_color, pred_color, gt_overlay, pred_overlay]

        gt_cls = sorted(set(gt.flatten()) - {0})
        pred_cls = sorted(set(pred.flatten()) - {0})
        gt_bg = (gt == 0).mean()
        pred_bg = (pred == 0).mean()

        for ax, title, im in zip(axes, titles, images):
            ax.imshow(im)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        fig.suptitle(f"Sample {saved} | GT classes: {gt_cls} → Pred classes: {pred_cls}\n"
                     f"GT bg={gt_bg:.1%} | Pred bg={pred_bg:.1%}",
                     fontsize=10, fontfamily="monospace")
        fig.tight_layout()
        fig.savefig(viz_dir / f"sample{saved:02d}_comparison.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)

        logger.log_info("diag",
                        f"Sample {saved}: GT={gt_cls} Pred={pred_cls} "
                        f"GT_bg={gt_bg:.3f} Pred_bg={pred_bg:.3f}")

        saved += 1

    logger.log_info("diag", f"Saved {saved} comparison images to {viz_dir}/")


# 颜色表 | Color map for visualization
COLOR_MAP = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
]


# ═══════════════════════════════════════════════════════════════════
# ② Overfit 测试 | Overfit Test
# ═══════════════════════════════════════════════════════════════════

def overfit_test(args, device):
    """
    20 tile × 100 epoch 过拟合测试 — 验证模型能力上限 | Overfit test — verify model capacity upper bound.

    如果 Decoder 无法过拟合 20 个 tile，说明架构本身有问题。
        If the Decoder cannot overfit 20 tiles, the architecture itself is the problem.

    :return: (decoder, backbone) trained models.
    """
    logger.log_info("diag", "=" * 50)
    logger.log_info("diag", "Overfit Test: 20 tiles × 100 epochs | 过拟合测试: 20 tile × 100 epoch")

    train_ds = FastISAIDTileDataset(args.tile_root, split="train", dense_labels=True)

    # 选 20 个有前景的 tile | Pick 20 tiles with foreground
    fg_tiles = []
    for fname in train_ds._tiles:
        mask = np.array(Image.open(train_ds._mask_dir / fname))
        if (mask > 0).sum() / mask.size > 0.01:
            fg_tiles.append(fname)
        if len(fg_tiles) >= 20:
            break

    if len(fg_tiles) < 20:
        logger.log_info("diag", f"Only {len(fg_tiles)} tiles with FG, using all")
    train_ds._tiles = fg_tiles
    logger.log_info("diag", f"Overfit set: {len(train_ds)} tiles")

    _wif = get_worker_init_fn(42)  # 固定种子保证数据增强可复现 | Fixed seed for reproducible augmentation
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, worker_init_fn=_wif)
    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    decoder = LightDecoder(1280, NUM_OUT_CH).to(device)
    n_p = sum(p.numel() for p in decoder.parameters())
    logger.log_info("diag", f"Decoder: {n_p:,} params")

    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    # ═══ 训练循环: 100 epoch | Training loop: 100 epochs ═══
    for epoch in range(1, 101):
        decoder.train()
        total_loss, n = 0.0, 0

        for batch in loader:
            img = batch["image"].to(device)
            tgt = batch["mask"].to(device)
            feats = backbone(img)
            logit = decoder(feats, target_size=tgt.shape[1:])

            # Focal Loss (γ=5.0): 针对极端不均衡 | For extreme class imbalance
            ce = F.cross_entropy(logit, tgt, ignore_index=255, reduction="none")
            pt = torch.exp(-ce)
            focal_loss = ((1 - pt) ** 5.0 * ce).mean()
            loss = focal_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1

        # 每 20 epoch 或首轮评估 | Evaluate every 20 epochs or first epoch
        if epoch % 20 == 0 or epoch == 1:
            # 计算 train FG-mIoU (忽略背景) | Compute train FG-mIoU (ignore background)
            decoder.eval()
            miou_v, valid = 0.0, 0
            bg_ratio = 0.0
            with torch.no_grad():
                for batch in loader:
                    img = batch["image"].to(device)
                    tgt = batch["mask"].to(device)
                    feats = backbone(img)
                    logit = decoder(feats, target_size=tgt.shape[1:])
                    pred = logit.argmax(dim=1)
                    bg_ratio += (pred == 0).float().mean().item()
                    # Per-class IoU: 只统计前景类 | Only foreground classes
                    for c in range(1, NUM_CLASSES + 1):
                        pc = (pred == c); tc = (tgt == c)
                        inter = (pc & tc).sum().float()
                        union = (pc | tc).sum().float()
                        if union > 0: miou_v += inter / union; valid += 1

            train_miou = miou_v / valid if valid > 0 else 0.0
            pred_bg = bg_ratio / max(n, 1)
            pred_vals = set()
            for batch in loader:
                pred_vals.update(logit.argmax(dim=1).unique().cpu().tolist())

            logger.log_info("diag",
                            f"E{epoch:3d} loss={total_loss/n:.4f} "
                            f"Train-FG-mIoU={train_miou:.4f} "
                            f"pred_bg={pred_bg:.3f} "
                            f"pred_unique={sorted(pred_vals)}")
            decoder.train()

        # 最终诊断结论 | Final diagnostic verdict
        if epoch == 100:
            if train_miou > 0.8:
                verdict = "✅ DECODER CAPABLE — problem is in data/generalization, not architecture"
            elif train_miou > 0.3:
                verdict = "⚠️ DECODER MARGINAL — can learn but struggles even on 20 tiles"
            else:
                verdict = "❌ DECODER BROKEN — cannot overfit 20 tiles, architecture is the bottleneck"
            logger.log_info("diag", verdict)

    return decoder, backbone


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    """主入口: Overfit + 可视化 | Main entry: Overfit + Visualization."""
    import argparse
    p = argparse.ArgumentParser(description="B-04 Diagnostic: Overfit + Visualization | B-04 诊断: 过拟合 + 可视化")
    p.add_argument("--tile-root", type=str, default="/root/autodl-tmp/iSAID_tiles")
    p.add_argument("--output-dir", type=str, default="runs/b04_diag")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    logger.log_info("diag", f"Starting B-04 diagnostics | 启动 B-04 诊断. Device={device}, Output={output_dir}")

    # ═══ 20 tile × 100 epoch overfit | 过拟合测试 ═══
    decoder, backbone = overfit_test(args, device)

    # ═══ 可视化预测结果 | Visualize predictions ═══
    val_ds = FastISAIDTileDataset(args.tile_root, split="val", dense_labels=True)
    # 选取验证集中有前景的 tile | Select val tiles with foreground
    fg_val_tiles = []
    for fname in val_ds._tiles:
        mask = np.array(Image.open(val_ds._mask_dir / fname))
        if (mask > 0).sum() / mask.size > 0.01:
            fg_val_tiles.append(fname)
        if len(fg_val_tiles) >= 10:
            break
    val_ds._tiles = fg_val_tiles
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, worker_init_fn=_wif)
    visualize(decoder, backbone, val_loader, output_dir, device, n=5)

    logger.log_info("diag", f"Done | 完成. Check {output_dir}/")


if __name__ == "__main__":
    main()
