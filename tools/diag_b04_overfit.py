#!/usr/bin/env python3
"""
B-04 诊断: 可视化 + Overfit 测试 | Diagnostics: Visualization + Overfit
======================================================================

① 可视化: 保存 GT vs Pred 的对照图, 看模型到底预测了什么
② Overfit: 只训练 20 tile, 100 epoch, 验证模型能力上限

用法 | Usage:
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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from adatile.datasets.isaid_tiles import FastISAIDTileDataset
from adatile.backbone import FastSAMBackbone
from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend

logger = get_logger("b04_diag")
logger.add_backend(ConsoleBackend())

PIL_AVAILABLE = True
try:
    from PIL import Image
except ImportError:
    PIL_AVAILABLE = False

NUM_OUT_CH = 16
NUM_CLASSES = 15


# ═══════════════════════════════════════════════════════════════════
# Decoder (与 B-04 完全一致) | Decoder (identical to B-04)
# ═══════════════════════════════════════════════════════════════════

class LightDecoder(nn.Module):
    def __init__(self, in_channels=1280, num_classes=16):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.stage2_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.stage3_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, num_classes, 1, bias=True)

    def forward(self, p4, target_size=None):
        x = self.stage1(p4)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2_conv(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage3_conv(x)
        x = self.head(x)
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return x


# ═══════════════════════════════════════════════════════════════════
# ① 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize(model, backbone, loader, output_dir, device, n=5):
    """保存 GT vs Pred 对照图."""
    if not PIL_AVAILABLE:
        logger.log_info("diag", "PIL not available, skipping viz")
        return

    viz_dir = Path(output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved = 0
    for imgs, gt_scores, fg_px, n_ty_arr, n_tx_arr in loader:
        if saved >= n:
            break
        img = imgs[:1].to(device)
        tgt = gt_scores[:1].to(device)

        feats = backbone(img)
        logit = model(feats["p4"], target_size=tgt.shape[1:])
        pred = logit.argmax(dim=1)[0].cpu().numpy()
        gt = tgt[0].cpu().numpy()
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)

        # 上色: 背景=黑色, 前景=彩色 | Color: bg=black, fg=colored
        cmap = plt_get_cmap() if 'plt_get_cmap' in dir() else None

        # 原图
        Image.fromarray((img_np * 255).astype(np.uint8)).save(
            viz_dir / f"sample{saved:02d}_image.png")

        # GT: 前景用颜色, 背景=0
        gt_color = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
        for c in range(1, 16):
            gt_color[gt == c] = COLOR_MAP[c % len(COLOR_MAP)]
        Image.fromarray(gt_color).save(viz_dir / f"sample{saved:02d}_gt.png")

        # Pred
        pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        for c in range(1, 16):
            pred_color[pred == c] = COLOR_MAP[c % len(COLOR_MAP)]
        Image.fromarray(pred_color).save(viz_dir / f"sample{saved:02d}_pred.png")

        # Overlay: image + GT + pred
        gt_overlay = (img_np * 0.5 + gt_color.astype(np.float32) / 255.0 * 0.5)
        pred_overlay = (img_np * 0.5 + pred_color.astype(np.float32) / 255.0 * 0.5)
        Image.fromarray((gt_overlay * 255).astype(np.uint8)).save(
            viz_dir / f"sample{saved:02d}_overlay_gt.png")
        Image.fromarray((pred_overlay * 255).astype(np.uint8)).save(
            viz_dir / f"sample{saved:02d}_overlay_pred.png")

        # 统计 | Stats
        gt_bg = (gt == 0).mean()
        pred_bg = (pred == 0).mean()
        gt_classes = set(gt.flatten().tolist())
        pred_classes = set(pred.flatten().tolist())
        logger.log_info("diag",
                        f"Sample {saved}: gt_classes={sorted(gt_classes)} "
                        f"pred_classes={sorted(pred_classes)} "
                        f"gt_bg={gt_bg:.3f} pred_bg={pred_bg:.3f}")

        saved += 1

    logger.log_info("diag", f"Saved {saved} viz samples to {viz_dir}/")


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
    """20 tile × 100 epoch — 验证模型能力上限."""
    logger.log_info("diag", "=" * 50)
    logger.log_info("diag", "Overfit Test: 20 tiles × 100 epochs")

    train_ds = FastISAIDTileDataset(args.tile_root, split="train", semantic=True)

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

    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    decoder = LightDecoder(1280, NUM_OUT_CH).to(device)
    n_p = sum(p.numel() for p in decoder.parameters())
    logger.log_info("diag", f"Decoder: {n_p:,} params")

    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    for epoch in range(1, 101):
        decoder.train()
        total_loss, n = 0.0, 0

        for batch in loader:
            img = batch["image"].to(device)
            tgt = batch["mask"].to(device)
            feats = backbone(img)
            logit = decoder(feats["p4"], target_size=tgt.shape[1:])

            ce = F.cross_entropy(logit, tgt, ignore_index=255, reduction="none")
            pt = torch.exp(-ce)
            focal_loss = ((1 - pt) ** 5.0 * ce).mean()
            loss = focal_loss
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1

        if epoch % 20 == 0 or epoch == 1:
            # 计算 train IoU | Compute train IoU
            decoder.eval()
            miou_v, valid = 0.0, 0
            bg_ratio = 0.0
            with torch.no_grad():
                for batch in loader:
                    img = batch["image"].to(device)
                    tgt = batch["mask"].to(device)
                    feats = backbone(img)
                    logit = decoder(feats["p4"], target_size=tgt.shape[1:])
                    pred = logit.argmax(dim=1)
                    bg_ratio += (pred == 0).float().mean().item()
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

        if epoch == 100:
            # 最终 | Final
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
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tile-root", type=str, default="/root/autodl-tmp/iSAID_tiles")
    p.add_argument("--output-dir", type=str, default="runs/b04_diag")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ── 20 tile × 100 epoch overfit ──
    decoder, backbone = overfit_test(args, device)

    # ── 可视化 ──
    val_ds = FastISAIDTileDataset(args.tile_root, split="val", semantic=True)
    fg_val_tiles = []
    for fname in val_ds._tiles:
        mask = np.array(Image.open(val_ds._mask_dir / fname))
        if (mask > 0).sum() / mask.size > 0.01:
            fg_val_tiles.append(fname)
        if len(fg_val_tiles) >= 10:
            break
    val_ds._tiles = fg_val_tiles
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    visualize(decoder, backbone, val_loader, output_dir, device, n=5)

    logger.log_info("diag", f"Done. Check {output_dir}/")


if __name__ == "__main__":
    main()
