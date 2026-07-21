#!/usr/bin/env python3
"""
DA-FRN 训练脚本 | DA-FRN Training Script.
===========================================

训练 Defect-Aware Feature Rectification Network — Backbone 和 Decoder 之间的
即插即用特征翻译器。
Trains DA-FRN — plug-and-play feature translator between Backbone and Decoder.

三阶段训练流程 | Three-Stage Training Protocol:
    Stage 1: 冻结 Backbone, 训练 DA-FRN + Decoder (当前脚本)
    Stage 2: 选择性解冻 Backbone 最后 1-2 层 (lr=1e-6)  [可选 | optional]
    Stage 3: 全系统微调 (lr=1e-6)  [可选 | optional]

    本脚本实现 Stage 1 + 可选的 Stage 2.
    This script implements Stage 1 + optional Stage 2.

消融实验 | Ablation::

    # Baseline (无 DA-FRN, 只有 PureDecoderP3P4)
    python tools/train/train_dafrn.py --no-dcr --no-fde --no-cdf

    # DCR only
    python tools/train/train_dafrn.py --no-fde --no-cdf

    # FDE only
    python tools/train/train_dafrn.py --no-dcr --no-cdf

    # CDF only
    python tools/train/train_dafrn.py --no-dcr --no-fde

    # Full DA-FRN
    python tools/train/train_dafrn.py

用法 | Usage::

    # 基础训练 | Basic training
    python tools/train/train_dafrn.py --epochs 200

    # 指定模块 | Specify modules
    python tools/train/train_dafrn.py --dcr --no-fde --cdf

    # 高分辨率 Backbone | High-res backbone
    python tools/train/train_dafrn.py --backbone fastsam-x
"""

from __future__ import annotations

import sys, argparse, random
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.decoder.pure_cnn_decoder import PureDecoderP3P4
from adatile.rectify import DA_FRN
from adatile.rectify.hdn import HeatmapDenoiser
from adatile.datasets.neu_seg import NEUSegDataset

# ═══════════════════════════════════════════════════════════════════
NUM_CLASSES = 4
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]


# ═══════════════════════════════════════════════════════════════════
# 数据增强 | Data Augmentation
# ═══════════════════════════════════════════════════════════════════

class BasicAugmentation:
    """极简增强: flip + rotate + brightness + noise."""
    def __init__(self, p_flip=0.5, p_rotate=0.5, brightness=0.2, contrast=0.2, noise_std=0.02):
        self.p_flip = p_flip; self.p_rotate = p_rotate
        self.brightness = brightness; self.contrast = contrast; self.noise_std = noise_std

    def __call__(self, image, mask):
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[-1]); mask = torch.flip(mask, dims=[-1])
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[-2]); mask = torch.flip(mask, dims=[-2])
        if torch.rand(1).item() < self.p_rotate:
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1]); mask = torch.rot90(mask, k, dims=[-2, -1])
        if torch.rand(1).item() < 0.7:
            b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            image = torch.clamp(image * b, 0.0, 1.0)
            c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            m_val = image.mean(dim=(-2, -1), keepdim=True)
            image = torch.clamp((image - m_val) * c + m_val, 0.0, 1.0)
        if torch.rand(1).item() < 0.5:
            image = torch.clamp(image + torch.randn_like(image) * self.noise_std, 0.0, 1.0)
        return image, mask


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════

def multiclass_dice_loss(pred, target, smooth=1e-6, ignore_bg=True):
    """多类别 Dice 损失 | Multi-class Dice loss."""
    C = pred.shape[1]; dice_sum = 0.0; count = 0
    for c in range(1 if ignore_bg else 0, C):
        pred_c = pred[:, c]; target_c = (target == c).float()
        if target_c.sum() > 0:
            inter = (pred_c * target_c).sum(); union = pred_c.sum() + target_c.sum()
            dice_sum += (2.0 * inter + smooth) / (union + smooth); count += 1
    if count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return 1.0 - dice_sum / count


# ═══════════════════════════════════════════════════════════════════
# 评估 | Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(decoder, frn, backbone, dataset, device, max_samples=0):
    """多类别分割评估 | Multi-class segmentation evaluation."""
    decoder.eval()
    if frn is not None:
        frn.eval()
    per_class_inter = torch.zeros(NUM_CLASSES, device=device)
    per_class_union = torch.zeros(NUM_CLASSES, device=device)
    indices = list(range(len(dataset)))
    if max_samples > 0:
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Eval", leave=False):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["masks"].squeeze(0).to(device).long()
        H, W = gt.shape

        # Pad to multiple of 32 (FastSAM requirement)
        pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)

        feats = backbone(img, extract_proto=False)
        p3, p4 = feats["p3"], feats["p4"]

        # DA-FRN rectification
        if frn is not None:
            p3, p4 = frn(p3, p4)

        pred = decoder(p3, p4)
        pred_full = F.interpolate(pred.unsqueeze(0), size=(H, W),
                                  mode="bilinear", align_corners=False).squeeze(0)
        pred_class = torch.argmax(pred_full, dim=0)

        for c in range(NUM_CLASSES):
            pc = (pred_class == c); gc = (gt == c)
            per_class_inter[c] += (pc & gc).sum()
            per_class_union[c] += (pc | gc).sum()

    per_class_iou = {}
    for c in range(NUM_CLASSES):
        inter = per_class_inter[c].item(); union = per_class_union[c].item()
        per_class_iou[CLASS_NAMES[c]] = round(inter / union, 6) if union > 0 else float("nan")
    valid = [v for v in per_class_iou.values() if not (v != v)]
    return {"mIoU": round(float(np.mean(valid)), 6) if valid else 0.0,
            "per_class_IoU": per_class_iou}


# ═══════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="DA-FRN Training")

    # ── 数据 | Data ──
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    p.add_argument("--no-augment", action="store_true")

    # ── 模型 | Model ──
    p.add_argument("--backbone", type=str, default="fastsam-x",
                   choices=["fastsam-x", "fastsam-s"])
    p.add_argument("--dcr", dest="enable_dcr", action="store_true", default=True,
                   help="启用 DCR (Defect-aware Channel Reweighting)")
    p.add_argument("--no-dcr", dest="enable_dcr", action="store_false",
                   help="禁用 DCR")
    p.add_argument("--fde", dest="enable_fde", action="store_true", default=True,
                   help="启用 FDE (Frequency-aware Defect Enhancement)")
    p.add_argument("--no-fde", dest="enable_fde", action="store_false",
                   help="禁用 FDE")
    p.add_argument("--cdf", dest="enable_cdf", action="store_true", default=True,
                   help="启用 CDF (Cross-scale Defect Fusion)")
    p.add_argument("--no-cdf", dest="enable_cdf", action="store_false",
                   help="禁用 CDF")

    # ── DCR 参数 | DCR params ──
    p.add_argument("--dcr-reduction", type=int, default=16,
                   help="DCR FC 压缩比 | DCR FC reduction ratio")

    # ── FDE 参数 | FDE params ──
    p.add_argument("--fde-alpha", type=float, default=0.1,
                   help="FDE 初始增强强度 | FDE initial enhancement strength")

    # ── CDF 参数 | CDF params ──
    p.add_argument("--cdf-hidden", type=int, default=64,
                   help="CDF 权重预测 MLP 隐层维度 | CDF weight predictor hidden dim")

    # ── HDN 参数 | HDN (Heatmap Denoiser) params ──
    p.add_argument("--heatmap-denoise", action="store_true",
                   help="启用 HDN 热力图去噪 (Sobel gradient → learnable denoise)")
    p.add_argument("--hdn-weight", type=float, default=0.1,
                   help="HDN 辅助 loss 权重 | HDN auxiliary loss weight")

    # ── 训练 | Training ──
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-frn", type=float, default=None,
                   help="DA-FRN learning rate (默认与 --lr 相同)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "balanced"])
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=5)

    # ── Stage 2 (可选): backbone 解冻 | Stage 2 (optional): backbone unfreeze ──
    p.add_argument("--stage2-epoch", type=int, default=0,
                   help="Stage 2 开始 epoch (0=跳过) | Stage 2 start epoch (0=skip)")
    p.add_argument("--stage2-lr", type=float, default=1e-6,
                   help="Stage 2 backbone 解冻学习率 | Stage 2 backbone unfreeze lr")
    p.add_argument("--stage2-unfreeze", type=int, default=1,
                   help="Stage 2 解冻 backbone 层数 | Stage 2 unfreeze layers")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# HDN 辅助: PyTorch Sobel 梯度算子 | HDN Helper: PyTorch Sobel Gradient
# ═══════════════════════════════════════════════════════════════════

def _rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    """RGB → Gray: 0.299R + 0.587G + 0.114B.  img: [B, 3, H, W] → [B, 1, H, W]."""
    return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]


def _sobel_gradient(gray: torch.Tensor) -> torch.Tensor:
    """
    PyTorch Sobel 梯度幅值 (可微) | PyTorch Sobel gradient magnitude (differentiable).

    使用固定 3×3 Sobel 核，梯度可回传到 HDN 参数。
    Uses fixed 3×3 Sobel kernels, gradients flow back to HDN params.

    :param gray: [B, 1, H, W] 灰度图 | Grayscale image.
    :return: [B, 1, H, W] 梯度幅值 | Gradient magnitude, same shape.
    """
    device = gray.device
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sobel_x)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sobel_y)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    # Robust normalize per sample (percentile clip)
    B = mag.shape[0]
    for b in range(B):
        m = mag[b].flatten()
        p_low = torch.quantile(m, 0.02)
        p_high = torch.quantile(m, 0.98)
        if p_high > p_low:
            mag[b] = torch.clamp(mag[b], p_low, p_high)
            mag[b] = (mag[b] - p_low) / (p_high - p_low + 1e-8)
    return mag


# ═══════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ── 生成模块标签 (用于实验命名) | Generate module tag (for experiment naming) ──
    modules = []
    if args.enable_dcr: modules.append("DCR")
    if args.enable_fde: modules.append("FDE")
    if args.enable_cdf: modules.append("CDF")
    if args.heatmap_denoise: modules.append("HDN")
    module_tag = "+".join(modules) if modules else "Baseline"

    # ── Output dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"runs/neuseg_DAFRN_{module_tag}_{ts}"
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ──
    logger = get_logger("train_dafrn")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))
    logger.log_info("config", f"DA-FRN Modules: {module_tag}")
    logger.log_info("config", f"Output: {out_dir}")

    # ── Data ──
    train_ds = NEUSegDataset(root=args.data_root, split="train", binary=False)
    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    ce_weight = None
    if args.class_weights == "balanced":
        stats = train_ds.get_class_stats()
        pc = [stats[cn]["pixels"] for cn in CLASS_NAMES]; total = sum(pc)
        raw = [total / max(p, 1) for p in pc]; mean_w = sum(raw) / len(raw)
        ce_weight = torch.tensor([w / mean_w for w in raw], dtype=torch.float32, device=device)

    augment = None if args.no_augment else BasicAugmentation()
    logger.log_info("data", f"Aug: {'none' if args.no_augment else 'basic'}")

    # ── Backbone (frozen) ──
    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint=f"thirdLibrary/FastSAM/weights/FastSAM-{args.backbone.split('-')[-1]}.pt",
    ).to(device)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, 224, 224, device=device), extract_proto=False)
    ch = backbone.channels
    logger.log_info("model", f"Backbone: P3={ch['p3']}ch, P4={ch['p4']}ch")

    # ── DA-FRN (Feature Rectifier) ──
    frn = DA_FRN(
        p3_channels=ch["p3"],
        p4_channels=ch["p4"],
        dcr_reduction=args.dcr_reduction,
        fde_kernel_sizes=(3, 7, 15),
        cdf_hidden=args.cdf_hidden,
        enable_dcr=args.enable_dcr,
        enable_fde=args.enable_fde,
        enable_cdf=args.enable_cdf,
    ).to(device)

    # ── Decoder ──
    decoder = PureDecoderP3P4(
        p3_channels=ch["p3"],
        p4_channels=ch["p4"],
        out_channels=NUM_CLASSES,
    ).to(device)

    # ── HDN (Heatmap Denoiser) — 可选 | Optional ──
    hdn = None
    if args.heatmap_denoise:
        hdn = HeatmapDenoiser(in_channels=1).to(device)
        logger.log_info("model", f"HDN: {sum(p.numel() for p in hdn.parameters())/1e3:.1f}K params")

    # ── 参数量统计 | Parameter Count ──
    frn_params = sum(p.numel() for p in frn.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    hdn_params = sum(p.numel() for p in hdn.parameters()) if hdn else 0
    total_trainable = frn_params + dec_params + hdn_params
    logger.log_info("model", f"DA-FRN: {frn_params/1e3:.1f}K | "
                    f"Decoder: {dec_params/1e3:.1f}K | "
                    f"HDN: {hdn_params/1e3:.1f}K | "
                    f"Total trainable: {total_trainable/1e3:.1f}K ({total_trainable:,})")
    logger.log_info("model", f"DA-FRN sub-params: {frn.get_submodule_params()}")

    # ── Optimizer (Stage 1: DA-FRN + Decoder + HDN) ──
    frn_lr = args.lr_frn if args.lr_frn is not None else args.lr
    optim_params = [
        {"params": decoder.parameters(), "lr": args.lr},
        {"params": frn.parameters(), "lr": frn_lr},
    ]
    if hdn is not None:
        optim_params.append({"params": hdn.parameters(), "lr": frn_lr})
    optimizer = torch.optim.AdamW(optim_params, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch)

    # ── Training ──
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train", f"DA-FRN Training — Modules: {module_tag}")
    logger.log_info("train", f"Trainable params: {total_trainable:,} (DA-FRN: {frn_params:,}, "
                    f"Decoder: {dec_params:,})")
    logger.log_info("train", f"Stage 2: {'epoch ' + str(args.stage2_epoch) if args.stage2_epoch > 0 else 'disabled'}")
    logger.log_info("train", f"{'='*60}")

    best_miou = 0.0; best_epoch = 0; global_step = 0; nan_count = 0
    stage2_active = False
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        # ── Stage 2 切换: 解冻 backbone 最后 N 层 | Stage 2 switch: unfreeze last N backbone layers ──
        if args.stage2_epoch > 0 and epoch == args.stage2_epoch and not stage2_active:
            stage2_active = True
            yolo_model = backbone.model.model
            sequential = yolo_model.model
            total_layers = len(sequential)
            start = max(0, total_layers - 1 - args.stage2_unfreeze)
            unfreeze_range = list(range(start, total_layers - 1))
            for idx in unfreeze_range:
                for param in sequential[idx].parameters():
                    param.requires_grad = True
            # 降低 lr | Reduce lr
            for pg in optimizer.param_groups:
                pg["lr"] = args.stage2_lr
            logger.log_info("train", f"  Stage 2: unfroze layers {unfreeze_range}, lr={args.stage2_lr}")

        epoch_losses = []; epoch_ce = []; epoch_dice = []

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch:3d}/{args.epochs}", unit="step")
        for _ in pbar:
            q_idx = rng.randint(0, len(train_ds) - 1)
            try:
                sample = train_ds[q_idx]
            except Exception:
                continue
            img = sample["image"]; mask = sample["masks"]
            if (mask > 0).sum() < 1.0:
                continue
            if augment:
                img, mask = augment(img, mask)

            # Get H,W AFTER augmentation (rot90 can swap spatial dims)
            H, W = mask.shape[1:]

            img_dev = img.unsqueeze(0).to(device); mask_dev = mask.to(device)
            pad_h = (32 - H % 32) % 32; pad_w = (32 - W % 32) % 32
            if pad_h > 0 or pad_w > 0:
                img_dev = F.pad(img_dev, (0, pad_w, 0, pad_h), mode='constant', value=0)

            # ── Forward ──
            decoder.train()
            frn.train()
            feats = backbone(img_dev, extract_proto=False)
            p3, p4 = feats["p3"], feats["p4"]

            # DA-FRN: 特征校正 | Feature Rectification
            p3_r, p4_r = frn(p3, p4)

            # Decoder: 掩码预测 | Mask Prediction
            pred_prob = decoder(p3_r, p4_r)

            pred_full = F.interpolate(pred_prob.unsqueeze(0), size=(H, W),
                                      mode="bilinear", align_corners=False).squeeze(0)
            target = mask_dev.squeeze(0).long()

            # ── HDN: 热力图去噪辅助 loss | Heatmap Denoising Auxiliary Loss ──
            hdn_loss = torch.tensor(0.0, device=device)
            if hdn is not None:
                gray = _rgb_to_gray(img_dev)         # [1, 1, H_pad, W_pad]
                raw_grad = _sobel_gradient(gray)     # [1, 1, H_pad, W_pad]
                denoised = hdn(raw_grad)             # [1, 1, H_pad, W_pad]

                # BCE loss: denoised heatmap vs GT binary mask
                gt_binary = (target > 0).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                # Resize denoised to match GT size
                denoised_resized = F.interpolate(denoised, size=(H, W),
                                                  mode="bilinear", align_corners=False)
                hdn_loss = F.binary_cross_entropy(
                    denoised_resized.clamp(1e-7, 1 - 1e-7), gt_binary, reduction='mean'
                )

            # ── Loss ──
            log_pred = torch.log(pred_full.unsqueeze(0).clamp(1e-7, 1))
            ce = F.nll_loss(log_pred, target.unsqueeze(0), weight=ce_weight, reduction='mean')
            dice = multiclass_dice_loss(pred_full.unsqueeze(0), target.unsqueeze(0))
            loss_val = 0.5 * ce + 0.5 * dice + args.hdn_weight * hdn_loss

            if torch.isnan(loss_val) or torch.isinf(loss_val):
                nan_count += 1; continue

            # ── Backward ──
            optimizer.zero_grad()
            loss_val.backward()

            # NaN grad check
            all_params = list(decoder.parameters()) + list(frn.parameters())
            if hdn is not None:
                all_params += list(hdn.parameters())
            grad_nan = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in all_params
            )
            if grad_nan:
                optimizer.zero_grad(); nan_count += 1; continue

            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step(); scheduler.step(); global_step += 1

            epoch_losses.append(loss_val.item()); epoch_ce.append(ce.item()); epoch_dice.append(dice.item())

            if epoch_losses:
                postfix = {"loss": f"{np.mean(epoch_losses[-50:]):.4f}",
                           "ce": f"{np.mean(epoch_ce[-50:]):.4f}",
                           "dice": f"{np.mean(epoch_dice[-50:]):.4f}"}
                pbar.set_postfix(postfix)

        # ── Epoch summary ──
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        log_msg = f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f}"
        if epoch_ce: log_msg += f" ce={np.mean(epoch_ce):.4f}"
        if epoch_dice: log_msg += f" dice={np.mean(epoch_dice):.4f}"
        log_msg += f" | lr={scheduler.get_last_lr()[0]:.2e} | NaN={nan_count}"

        # DA-FRN stats
        frn_stats = frn.get_stats()
        if frn_stats.get("cdf_enabled"):
            log_msg += f" | w3={frn_stats.get('w3_mean', 0):.3f} w4={frn_stats.get('w4_mean', 0):.3f}"

        logger.log_info("epoch", log_msg)
        logger.log_metric("loss", avg_loss, step=epoch, tags=["dafrn"])
        if epoch_ce: logger.log_metric("ce", np.mean(epoch_ce), step=epoch, tags=["dafrn"])
        if epoch_dice: logger.log_metric("dice", np.mean(epoch_dice), step=epoch, tags=["dafrn"])

        # ── Evaluation ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"{'─'*50}")
            logger.log_info("eval", f"Evaluation @ Epoch {epoch}")
            result = evaluate(decoder, frn, backbone, val_ds, device)
            miou = result["mIoU"]
            logger.log_info("eval", f"  mIoU={miou:.4f}  best={best_miou:.4f} (epoch {best_epoch})")
            for cls_name, iou_c in result["per_class_IoU"].items():
                logger.log_info("eval", f"    {cls_name}: IoU={iou_c:.4f}")
            logger.log_metric("mIoU", miou, step=epoch, tags=["dafrn"])

            if miou > best_miou:
                best_miou = miou; best_epoch = epoch
                ckpt = {
                    "epoch": epoch, "global_step": global_step,
                    "frn_state_dict": {k: v.clone() for k, v in frn.state_dict().items()},
                    "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mIoU": miou, "per_class_IoU": result["per_class_IoU"],
                    "args": vars(args), "frn_stats": frn_stats,
                    "num_classes": NUM_CLASSES, "module_tag": module_tag,
                }
                if hdn is not None:
                    ckpt["hdn_state_dict"] = {k: v.clone() for k, v in hdn.state_dict().items()}
                torch.save(ckpt, str(out_dir / "best_model.pt"))
                logger.log_info("eval", f"  New best: mIoU={best_miou:.4f} @ epoch {best_epoch}")

    # ── Final save ──
    final = {
        "epoch": args.epochs, "global_step": global_step,
        "frn_state_dict": {k: v.clone() for k, v in frn.state_dict().items()},
        "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mIoU": best_miou, "best_epoch": best_epoch,
        "args": vars(args), "frn_stats": frn.get_stats(),
        "num_classes": NUM_CLASSES, "nan_skip_count": nan_count,
    }
    if hdn is not None:
        final["hdn_state_dict"] = {k: v.clone() for k, v in hdn.state_dict().items()}
    torch.save(final, str(out_dir / "last_model.pt"))

    # ── Done ──
    print(f"\n{'='*60}")
    print(f"  DA-FRN Training — Complete")
    print(f"  Modules: {module_tag}")
    print(f"  Trainable: {total_trainable:,} (DA-FRN: {frn_params:,}, Decoder: {dec_params:,})")
    print(f"  Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    print(f"  NaN skips: {nan_count}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")
    logger.log_info("done", f"Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
