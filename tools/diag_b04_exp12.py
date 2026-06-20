#!/usr/bin/env python3
"""
B-04 诊断实验 1 & 2 | Diagnostics Exp 1 & 2
============================================

实验 1: 只训练 Meaningful tiles (FG > 5%), 验证类别学习是否改善
实验 2: Binary Segmentation (FG/BG), 与 Multi-class 对比, 验证定位能力

用法:
    python tools/diag_b04_exp12.py --tile-root /root/autodl-tmp/iSAID_tiles
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

logger = get_logger("b04_exp12")
logger.add_backend(ConsoleBackend())

from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_MAP = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),
             (0,255,255),(128,0,0),(0,128,0),(0,0,128),(128,128,0),
             (128,0,128),(0,128,128),(255,128,0),(255,0,128),(128,255,0)]


class LightDecoder(nn.Module):
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


def train_eval(decoder, backbone, loader, epochs, device, binary=False, num_classes=16):
    """训练 + 返回最终指标."""
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    history = []

    for epoch in range(1, epochs + 1):
        decoder.train()
        total_loss, n = 0.0, 0

        for batch in loader:
            img = batch["image"].to(device)
            tgt = batch["mask"].to(device)

            if binary:
                tgt = (tgt > 0).long()  # 0=bg, 1=fg

            feats = backbone(img)
            logit = decoder(feats["p4"], target_size=tgt.shape[1:])

            ce = F.cross_entropy(logit, tgt, ignore_index=255, reduction="none")
            pt = torch.exp(-ce)
            loss = ((1 - pt) ** 5.0 * ce).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n += 1

        if epoch % 20 == 0 or epoch == 1:
            decoder.eval()
            miou_v, dices, valid = 0.0, 0.0, 0
            bg_ratio, total_px = 0.0, 0
            all_pred_vals, all_tgt_vals = set(), set()

            with torch.no_grad():
                for batch in loader:
                    img = batch["image"].to(device)
                    tgt = batch["mask"].to(device)
                    if binary:
                        tgt = (tgt > 0).long()

                    feats = backbone(img)
                    logit = decoder(feats["p4"], target_size=tgt.shape[1:])
                    pred = logit.argmax(dim=1)
                    probs = F.softmax(logit, dim=1)

                    all_pred_vals.update(pred.unique().cpu().tolist())
                    all_tgt_vals.update(tgt.unique().cpu().tolist())
                    bg_ratio += (pred == 0).float().sum().item()
                    total_px += pred.numel()

                    nc = 2 if binary else num_classes
                    start_c = 1 if not binary else 1  # skip bg
                    for c in range(start_c, nc):
                        pc = (pred == c); tc = (tgt == c)
                        inter = (pc & tc).sum().float()
                        union = (pc | tc).sum().float()
                        if union > 0:
                            miou_v += inter / union; valid += 1
                            # Dice
                            p_c = probs[:, c]; t_c = (tgt == c).float()
                            inter_d = (p_c * t_c).sum()
                            dices += (2 * inter_d / (p_c.sum() + t_c.sum() + 1e-8)).item()

            miou = miou_v / max(valid, 1)
            dice = dices / max(valid, 1)
            bg_pct = bg_ratio / max(total_px, 1)

            logger.log_info("exp12",
                            f"E{epoch:3d} loss={total_loss/n:.4f} "
                            f"mIoU={miou:.4f} Dice={dice:.4f} "
                            f"bg_pred={bg_pct:.3f} "
                            f"pred={sorted(all_pred_vals)} gt={sorted(all_tgt_vals)}")
            history.append((epoch, total_loss/n, miou, dice, bg_pct))
            decoder.train()

    return history


def viz_pred(model, backbone, loader, output_path, device, binary=False):
    """保存拼接可视化."""
    model.eval()
    for i, batch in enumerate(loader):
        if i >= 2: break
        img = batch["image"][:1].to(device)
        tgt = batch["mask"][:1].to(device)
        tgt_disp = (tgt > 0).long() if binary else tgt

        feats = backbone(img)
        logit = model(feats["p4"], target_size=tgt.shape[1:])
        pred = logit.argmax(dim=1)[0].cpu().numpy()
        gt = tgt_disp[0].cpu().numpy()
        img_np = img[0].cpu().numpy().transpose(1, 2, 0)

        nc = 2 if binary else 16
        gt_c = np.zeros((*gt.shape, 3), dtype=np.uint8)
        pred_c = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for c in range(1, nc):
            gt_c[gt == c] = COLOR_MAP[c % len(COLOR_MAP)]
            pred_c[pred == c] = COLOR_MAP[c % len(COLOR_MAP)]
        gt_ov = img_np * 0.5 + gt_c.astype(np.float32) / 255.0 * 0.5
        pred_ov = img_np * 0.5 + pred_c.astype(np.float32) / 255.0 * 0.5

        fig, axes = plt.subplots(1, 5, figsize=(22, 5))
        for ax, title, im in zip(axes,
                                  ["Image", "GT", "Pred", "GT Overlay", "Pred Overlay"],
                                  [img_np, gt_c, pred_c, gt_ov, pred_ov]):
            ax.imshow(im); ax.set_title(title); ax.axis("off")

        gt_cls = sorted(set(gt.flatten()) - {0})
        pred_cls = sorted(set(pred.flatten()) - {0})
        fig.suptitle(f"GT={gt_cls} Pred={pred_cls}", fontsize=10, fontfamily="monospace")
        fig.tight_layout()
        fig.savefig(output_path.replace(".png", f"_s{i}.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)


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

    train_ds = FastISAIDTileDataset(args.tile_root, split="train", semantic=True)

    # ═══════════════════════════════════════════════════════════════
    # 实验 1: Meaningful tiles only (FG > 5%)
    # ═══════════════════════════════════════════════════════════════
    logger.log_info("exp12", "=" * 60)
    logger.log_info("exp12", "Experiment 1: Meaningful tiles (FG > 5%)")
    logger.log_info("exp12", "=" * 60)

    fg5_tiles = []
    for fname in tqdm(train_ds._tiles, desc="  Filtering FG>5%"):
        mask = np.array(Image.open(train_ds._mask_dir / fname))
        if (mask > 0).sum() / mask.size > 0.05:
            fg5_tiles.append(fname)

    logger.log_info("exp12",
                    f"FG>5% tiles: {len(fg5_tiles)}/{len(train_ds._tiles)} "
                    f"({len(fg5_tiles)/len(train_ds._tiles)*100:.1f}%)")

    if len(fg5_tiles) > 1000:
        np.random.seed(42)
        fg5_tiles = list(np.random.choice(fg5_tiles, 1000, replace=False))
        logger.log_info("exp12", f"Sampled 1000 tiles for speed")

    train_ds._tiles = fg5_tiles
    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    backbone = FastSAMBackbone(freeze_backbone=True).eval()
    decoder = LightDecoder(1280, 16).to(device)

    logger.log_info("exp12", f"Training 100 epochs on {len(fg5_tiles)} meaningful tiles...")
    history1 = train_eval(decoder, backbone, loader, 100, device, binary=False, num_classes=16)

    final_epoch, _, miou1, dice1, bg1 = history1[-1]
    logger.log_info("exp12",
                    f"Exp1 Final: FG-mIoU={miou1:.4f} Dice={dice1:.4f} bg_pred={bg1:.3f}")

    # Viz
    val_ds = FastISAIDTileDataset(args.tile_root, split="val", semantic=True)
    val_fg = []
    for fname in val_ds._tiles:
        mask = np.array(Image.open(val_ds._mask_dir / fname))
        if (mask > 0).sum() / mask.size > 0.05:
            val_fg.append(fname)
    val_ds._tiles = val_fg[:10]
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    viz_pred(decoder, backbone, val_loader, str(output_dir / "exp1_viz.png"), device)

    # ═══════════════════════════════════════════════════════════════
    # 实验 2: Binary Segmentation
    # ═══════════════════════════════════════════════════════════════
    logger.log_info("exp12", "")
    logger.log_info("exp12", "=" * 60)
    logger.log_info("exp12", "Experiment 2: Binary Segmentation (FG vs BG)")
    logger.log_info("exp12", "=" * 60)

    # 复用 Exp1 的 meaningful tiles for fair comparison
    train_ds._tiles = fg5_tiles
    loader_bin = DataLoader(train_ds, batch_size=8, shuffle=True)
    decoder_bin = LightDecoder(1280, 2).to(device)  # 2 output channels: bg, fg

    logger.log_info("exp12", f"Training 100 epochs on {len(fg5_tiles)} tiles (binary)...")
    history2 = train_eval(decoder_bin, backbone, loader_bin, 100, device, binary=True, num_classes=2)

    _, _, miou2, dice2, bg2 = history2[-1]
    logger.log_info("exp12",
                    f"Exp2 Final: Binary-IoU={miou2:.4f} Dice={dice2:.4f} bg_pred={bg2:.3f}")

    viz_pred(decoder_bin, backbone, val_loader, str(output_dir / "exp2_viz.png"), device, binary=True)

    # ═══════════════════════════════════════════════════════════════
    # 汇总 | Summary
    # ═══════════════════════════════════════════════════════════════
    logger.log_info("exp12", "")
    logger.log_info("exp12", "=" * 60)
    logger.log_info("exp12", "SUMMARY")
    logger.log_info("exp12", "=" * 60)
    logger.log_info("exp12",
                    f"Exp1 (Multi-class, FG>5%):  FG-mIoU={miou1:.4f}  Dice={dice1:.4f}")
    logger.log_info("exp12",
                    f"Exp2 (Binary,    FG>5%):  Binary-IoU={miou2:.4f}  Dice={dice2:.4f}")
    logger.log_info("exp12",
                    f"Exp1-20tile-overfit:      FG-mIoU=0.692  (reference)")

    if miou2 > 0.6 and miou1 < 0.2:
        verdict = ("✅ VERDICT: Object LOCALIZATION works (IoU={:.2f}), "
                   "CLASS learning fails ({:.3f}).\n"
                   "Problem is multi-class imbalance, not feature resolution.").format(miou2, miou1)
    elif miou1 > 0.2:
        verdict = (f"✅ VERDICT: FG>5% filter helps. Multi-class FG-mIoU={miou1:.3f} "
                   f"vs binary IoU={miou2:.3f}. Class imbalance is the main bottleneck.")
    else:
        verdict = (f"⚠️ VERDICT: Both multi-class ({miou1:.3f}) and binary ({miou2:.3f}) "
                   f"are low. Decoder or feature issues remain.")
    logger.log_info("exp12", verdict)

    # 画对比图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, hist, title in zip(axes,
                                [history1, history2],
                                ["Exp1: Multi-class (FG>5%)", "Exp2: Binary (FG>5%)"]):
        epochs = [h[0] for h in hist]
        mious = [h[2] for h in hist]
        dices = [h[3] for h in hist]
        ax.plot(epochs, mious, "o-", label="mIoU", lw=2)
        ax.plot(epochs, dices, "s-", label="Dice", lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("B-04 Diagnostics: Exp1 vs Exp2", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "exp12_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.log_info("exp12", f"Done. Check {output_dir}/")


if __name__ == "__main__":
    main()
