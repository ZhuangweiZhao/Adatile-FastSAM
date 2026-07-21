#!/usr/bin/env python3
"""
DA-FRN 训练脚本 — Severstal 钢铁缺陷检测 | DA-FRN Training — Severstal Steel Defect.
======================================================================================

适配 Severstal 钢铁表面缺陷检测数据集。支持二值和多类别两种模式。
Adapted for Severstal steel surface defect detection. Supports binary and multi-class modes.

三阶段训练流程 | Three-Stage Training Protocol:
    Stage 1: 冻结 Backbone, 训练 DA-FRN + Decoder (当前脚本)
    Stage 2: 选择性解冻 Backbone 最后 1-2 层 (lr=1e-6)  [可选 | optional]
    Stage 3: 全系统微调 (lr=1e-6)  [可选 | optional]

消融实验 | Ablation::

    # Baseline (无 DA-FRN, 只有 PureDecoderP3P4 / NoLoRA)
    python tools/train/train_dafrn_severstal.py --no-dcr --no-fde --no-cdf

    # DCR only
    python tools/train/train_dafrn_severstal.py --no-fde --no-cdf

    # FDE only
    python tools/train/train_dafrn_severstal.py --no-dcr --no-cdf

    # Full DA-FRN (DCR + FDE + CDF)
    python tools/train/train_dafrn_severstal.py

用法 | Usage::

    # 基础训练 (二值模式, FG/BG) | Basic training (binary mode)
    python tools/train/train_dafrn_severstal.py --epochs 200 --binary

    # 多类别训练 (5 类: BG + 4 种缺陷) | Multi-class training
    python tools/train/train_dafrn_severstal.py --epochs 200

    # 指定模块 | Specify modules
    python tools/train/train_dafrn_severstal.py --dcr --no-fde --cdf --binary
"""

from __future__ import annotations

import sys, argparse
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.decoder.pure_cnn_decoder import PureDecoderP3P4
from adatile.rectify import DA_FRN
from adatile.rectify.hdn import HeatmapDenoiser
from adatile.datasets.severstal import SeverstalDataset

# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

# 多类别: 5 类 (BG + Class1-4) | Multi-class: 5 classes
MULTI_NUM_CLASSES = 5
MULTI_CLASS_NAMES = ["background", "Class1", "Class2", "Class3", "Class4"]

# 二值: 2 类 (BG + FG) | Binary: 2 classes
BINARY_NUM_CLASSES = 2
BINARY_CLASS_NAMES = ["background", "foreground"]

# 图像尺寸 (pad 后) | Image size (after padding)
IMG_H = 256
IMG_W = 1600
# Pad to multiple of 32: (256→256, 1600→1600) — both are already multiples of 32!
# 256 = 8 × 32, 1600 = 50 × 32
PAD_H = 0  # No padding needed
PAD_W = 0


# ═══════════════════════════════════════════════════════════════════
# 数据增强 | Data Augmentation
# ═══════════════════════════════════════════════════════════════════

class BasicAugmentation:
    """
    极简增强: flip + rotate + brightness + noise.
    Simple augmentation: flip + rotate + brightness + noise.

    对 Severstal 钢带图像的特殊处理:
    - 水平翻转 (flip dim=-1) 更自然 (钢带方向)
    - 垂直翻转可能不太合理，概率减半
    Special handling for Severstal steel strips:
    - Horizontal flip more natural (strip direction)
    - Vertical flip less reasonable, half probability
    """
    def __init__(self, p_flip=0.5, p_rotate=0.3, brightness=0.2, contrast=0.2, noise_std=0.02):
        self.p_hflip = p_flip
        self.p_vflip = p_flip * 0.5  # Vertical flip less likely for steel strips
        self.p_rotate = p_rotate
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, image, mask):
        # Horizontal flip (across width — natural for steel strips)
        if torch.rand(1).item() < self.p_hflip:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        # Vertical flip (across height — less natural)
        if torch.rand(1).item() < self.p_vflip:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        # Rotation (more conservative — steel strips are directional)
        if torch.rand(1).item() < self.p_rotate:
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
        # Brightness jitter
        if torch.rand(1).item() < 0.7:
            b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            image = torch.clamp(image * b, 0.0, 1.0)
            c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            m_val = image.mean(dim=(-2, -1), keepdim=True)
            image = torch.clamp((image - m_val) * c + m_val, 0.0, 1.0)
        # Gaussian noise
        if torch.rand(1).item() < 0.5:
            image = torch.clamp(image + torch.randn_like(image) * self.noise_std, 0.0, 1.0)
        return image, mask


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════

def multiclass_dice_loss(pred, target, smooth=1e-6, ignore_bg=True):
    """
    多类别 Dice 损失 | Multi-class Dice loss.
    Severstal 有极端的 FG/BG 不平衡 (~3.3% FG), Dice loss 对不平衡较鲁棒.
    """
    C = pred.shape[1]
    dice_sum = 0.0
    count = 0
    for c in range(1 if ignore_bg else 0, C):
        pred_c = pred[:, c]
        target_c = (target == c).float()
        if target_c.sum() > 0:
            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_sum += (2.0 * inter + smooth) / (union + smooth)
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return 1.0 - dice_sum / count


def binary_dice_loss(pred, target, smooth=1e-6):
    """
    二值 Dice 损失 | Binary Dice loss.
    pred: [B, 2, H, W] (softmax), target: [B, H, W] long.
    取 FG channel (index 1) 计算.
    """
    pred_fg = pred[:, 1]  # [B, H, W]
    target_fg = (target > 0).float()
    if target_fg.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    inter = (pred_fg * target_fg).sum()
    union = pred_fg.sum() + target_fg.sum()
    return 1.0 - (2.0 * inter + smooth) / (union + smooth)


# ═══════════════════════════════════════════════════════════════════
# 评估 | Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(decoder, frn, backbone, dataset, device, num_classes, class_names,
             max_samples=0):
    """
    分割评估 | Segmentation evaluation.
    Returns: {"mIoU": float, "per_class_IoU": dict}
    """
    decoder.eval()
    backbone.eval()
    if frn is not None:
        frn.eval()

    per_class_inter = torch.zeros(num_classes, device=device)
    per_class_union = torch.zeros(num_classes, device=device)

    indices = list(range(len(dataset)))
    if max_samples > 0:
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Eval", leave=False):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt = sample["masks"].squeeze(0).to(device).long()
        H, W = gt.shape

        # FastSAM forward (no extra padding needed — 256 and 1600 are multiples of 32)
        feats = backbone(img, extract_proto=False)
        p3, p4 = feats["p3"], feats["p4"]

        # DA-FRN rectification
        if frn is not None:
            p3, p4 = frn(p3, p4)

        pred = decoder(p3, p4)
        # Interpolate back to original size
        pred_full = F.interpolate(pred.unsqueeze(0), size=(H, W),
                                  mode="bilinear", align_corners=False).squeeze(0)
        pred_class = torch.argmax(pred_full, dim=0)

        for c in range(num_classes):
            pc = (pred_class == c)
            gc = (gt == c)
            per_class_inter[c] += (pc & gc).sum()
            per_class_union[c] += (pc | gc).sum()

    per_class_iou = {}
    for c in range(num_classes):
        inter = per_class_inter[c].item()
        union = per_class_union[c].item()
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        per_class_iou[name] = round(inter / union, 6) if union > 0 else float("nan")

    valid = [v for v in per_class_iou.values() if not (v != v)]  # filter NaN
    return {
        "mIoU": round(float(np.mean(valid)), 6) if valid else 0.0,
        "per_class_IoU": per_class_iou,
    }


# ═══════════════════════════════════════════════════════════════════
# HDN 辅助: PyTorch Sobel 梯度算子 | HDN Helper: PyTorch Sobel Gradient
# ═══════════════════════════════════════════════════════════════════

def _rgb_to_gray(img: torch.Tensor) -> torch.Tensor:
    """RGB → Gray: 0.299R + 0.587G + 0.114B.  img: [B, 3, H, W] → [B, 1, H, W]."""
    return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]


def _sobel_gradient(gray: torch.Tensor) -> torch.Tensor:
    """
    PyTorch Sobel 梯度幅值 (可微) | PyTorch Sobel gradient magnitude (differentiable).
    """
    device = gray.device
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=device).view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sobel_x)
    gy = F.conv2d(F.pad(gray, (1, 1, 1, 1), mode='reflect'), sobel_y)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    # Robust normalize per sample
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
# 命令行参数 | Command-line Arguments
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="DA-FRN Training — Severstal Steel Defect")

    # ── 数据 | Data ──
    p.add_argument("--data-root", type=str, default="data/severstal-steel-defect-detection")
    p.add_argument("--binary", action="store_true",
                   help="二值模式 (FG/BG) | Binary mode (FG/BG). 默认多类别 (5 类).")
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
    p.add_argument("--fde-alpha", type=float, default=0.5,
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
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size | 批次大小 (default: 1)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-frn", type=float, default=None,
                   help="DA-FRN learning rate (默认与 --lr 相同)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "balanced"])
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 = main process only)")
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

    # ── 断点恢复 | Resume ──
    p.add_argument("--resume", type=str, default=None,
                   help="从 checkpoint 恢复训练 | Resume training from checkpoint")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 训练主函数 | Main Training Function
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ── 根据模式选择类别配置 | Select class config based on mode ──
    if args.binary:
        NUM_CLASSES = BINARY_NUM_CLASSES
        CLASS_NAMES = BINARY_CLASS_NAMES
        mode_str = "binary"
    else:
        NUM_CLASSES = MULTI_NUM_CLASSES
        CLASS_NAMES = MULTI_CLASS_NAMES
        mode_str = "multi"

    # ── 生成模块标签 | Generate module tag ──
    modules = []
    if args.enable_dcr:
        modules.append("DCR")
    if args.enable_fde:
        modules.append("FDE")
    if args.enable_cdf:
        modules.append("CDF")
    if args.heatmap_denoise:
        modules.append("HDN")
    module_tag = "+".join(modules) if modules else "Baseline"

    # ── Output dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"runs/severstal_DAFRN_{module_tag}_{mode_str}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ──
    logger = get_logger("train_dafrn_severstal")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))
    logger.log_info("config", f"Dataset: Severstal ({mode_str})")
    logger.log_info("config", f"DA-FRN Modules: {module_tag}")
    logger.log_info("config", f"Output: {out_dir}")

    # ── Data ──
    train_ds = SeverstalDataset(
        root=args.data_root, split="train", binary=args.binary, seed=args.seed,
    )
    val_ds = SeverstalDataset(
        root=args.data_root, split="val", binary=args.binary, seed=args.seed,
    )
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}, Mode: {mode_str}")

    # 类别权重 | Class weights
    ce_weight = None
    if args.class_weights == "balanced":
        stats = train_ds.get_class_stats()
        pc = [stats.get(cn, {}).get("pixels", 1) for cn in CLASS_NAMES]
        # Convert from percentage
        total = sum(pc)
        if total > 0:
            raw = [total / max(p, 0.01) for p in pc]
            mean_w = sum(raw) / len(raw)
            ce_weight = torch.tensor([w / mean_w for w in raw], dtype=torch.float32, device=device)

    augment = None if args.no_augment else BasicAugmentation()
    logger.log_info("data", f"Aug: {'none' if args.no_augment else 'basic'}")

    # ── DataLoader: full-epoch iteration (standard training protocol) ──
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    logger.log_info("data", f"DataLoader: batch={args.batch_size}, workers={args.num_workers}, "
                    f"batches/epoch={len(train_loader)}")

    # ── Backbone (frozen) ──
    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint=f"thirdLibrary/FastSAM/weights/FastSAM-{args.backbone.split('-')[-1]}.pt",
    ).to(device)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, IMG_H, IMG_W, device=device), extract_proto=False)
    ch = backbone.channels
    logger.log_info("model", f"Backbone: P3={ch['p3']}ch, P4={ch['p4']}ch")

    # ── DA-FRN (Feature Rectifier) ──
    frn = DA_FRN(
        p3_channels=ch["p3"],
        p4_channels=ch["p4"],
        dcr_reduction=args.dcr_reduction,
        fde_kernel_sizes=(3, 7, 15),
        fde_alpha_init=args.fde_alpha,
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
                    f"Total: {total_trainable/1e3:.1f}K ({total_trainable:,})")
    if args.enable_dcr or args.enable_fde or args.enable_cdf:
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
        optimizer, T_max=args.epochs * len(train_loader))

    # ── Resume (断点恢复) ──
    start_epoch = 1
    global_step = 0
    best_miou = 0.0
    best_epoch = 0
    nan_count = 0
    stage2_active = False
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        frn.load_state_dict(ckpt["frn_state_dict"])
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if args.heatmap_denoise and hdn is not None and "hdn_state_dict" in ckpt:
            hdn.load_state_dict(ckpt["hdn_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_miou = ckpt.get("mIoU", ckpt.get("best_mIoU", 0.0))
        best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", 0))
        stage2_active = ckpt.get("stage2_active", False)
        nan_count = ckpt.get("nan_skip_count", 0)
        # If resuming from Stage 2, unfreeze backbone immediately
        if stage2_active and args.stage2_epoch > 0:
            yolo_model = backbone.model.model
            sequential = yolo_model.model
            total_layers = len(sequential)
            start_l = max(0, total_layers - 1 - args.stage2_unfreeze)
            for idx in range(start_l, total_layers - 1):
                for param in sequential[idx].parameters():
                    param.requires_grad = True
        logger.log_info("resume", f"Resumed from epoch {start_epoch}, "
                        f"global_step={global_step}, best_mIoU={best_miou:.4f}, "
                        f"stage2_active={stage2_active}")

    # ── Training ──
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train", f"DA-FRN Training — Severstal | Modules: {module_tag}")
    logger.log_info("train", f"Trainable: {total_trainable:,} (DA-FRN: {frn_params:,}, "
                    f"Decoder: {dec_params:,})")
    logger.log_info("train", f"Classes: {NUM_CLASSES} ({mode_str}), Image: {IMG_H}×{IMG_W}")
    logger.log_info("train", f"Stage 2: {'epoch ' + str(args.stage2_epoch) if args.stage2_epoch > 0 else 'disabled'}")
    logger.log_info("train", f"{'='*60}")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── Stage 2 switch ──
        if args.stage2_epoch > 0 and epoch == args.stage2_epoch and not stage2_active:
            stage2_active = True
            yolo_model = backbone.model.model
            sequential = yolo_model.model
            total_layers = len(sequential)
            start = max(0, total_layers - 1 - args.stage2_unfreeze)
            unfreeze_range = list(range(start, total_layers - 1))

            # Collect unfrozen params + add to optimizer (BUG FIX: they were unfrozen
            # but never added to optimizer.param_groups, so optimizer.step() skipped them.)
            stage2_params = []
            for idx in unfreeze_range:
                for param in sequential[idx].parameters():
                    param.requires_grad = True
                    stage2_params.append(param)
            optimizer.add_param_group({
                "params": stage2_params,
                "lr": args.stage2_lr,
            })
            # Also lower LR for existing groups to avoid destabilizing decoder/FRN
            for pg in optimizer.param_groups:
                pg["lr"] = args.stage2_lr
            logger.log_info("train", f"  Stage 2: unfroze layers {unfreeze_range} "
                            f"({sum(p.numel() for p in stage2_params)/1e6:.2f}M params), "
                            f"lr={args.stage2_lr}")

        epoch_losses = []
        epoch_ce = []
        epoch_dice = []

        # Full-epoch DataLoader iteration (standard training protocol)
        # Decoder 内部有 .squeeze(0), 因此 B>1 时逐样本循环累积梯度
        # Decoder internally squeezes dim 0, so for B>1 we loop per-sample with grad accumulation
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", unit="batch")
        for batch in pbar:
            img_batch = batch["image"]        # [B, 3, H_img, W_img]
            mask_batch = batch["masks"]       # [B, 1, H_img, W_img]
            mask_batch = mask_batch.squeeze(1)  # -> [B, H_img, W_img]

            B = img_batch.shape[0]

            decoder.train()
            frn.train()

            optimizer.zero_grad()
            batch_loss = 0.0
            batch_ce = 0.0
            batch_dice = 0.0
            valid_samples = 0

            for b_idx in range(B):
                img = img_batch[b_idx]          # [3, H, W]
                mask = mask_batch[b_idx]        # [H, W]

                # Apply augmentation per-sample
                if augment:
                    img, mask = augment(img, mask)

                H, W = mask.shape
                img_dev = img.unsqueeze(0).to(device)    # [1, 3, H, W]
                mask_dev = mask.unsqueeze(0).to(device)  # [1, H, W]

                # ── Forward ──
                feats = backbone(img_dev, extract_proto=False)
                p3, p4 = feats["p3"], feats["p4"]

                # DA-FRN: 特征校正 | Feature Rectification
                p3_r, p4_r = frn(p3, p4)

                # Decoder (returns [num_classes, H/4, W/4] — no batch dim)
                pred_prob = decoder(p3_r, p4_r)

                pred_full = F.interpolate(pred_prob.unsqueeze(0), size=(H, W),
                                          mode="bilinear", align_corners=False).squeeze(0)
                target = mask_dev.squeeze(0).long()

                # ── HDN auxiliary loss ──
                hdn_loss = torch.tensor(0.0, device=device)
                if hdn is not None:
                    gray = _rgb_to_gray(img_dev)
                    raw_grad = _sobel_gradient(gray)
                    denoised = hdn(raw_grad)
                    gt_binary = (target > 0).float().unsqueeze(0).unsqueeze(0)
                    denoised_resized = F.interpolate(denoised, size=(H, W),
                                                      mode="bilinear", align_corners=False)
                    hdn_loss = F.binary_cross_entropy(
                        denoised_resized.clamp(1e-7, 1 - 1e-7), gt_binary, reduction='mean'
                    )

                # ── Loss ──
                log_pred = torch.log(pred_full.unsqueeze(0).clamp(1e-7, 1))
                ce = F.nll_loss(log_pred, target.unsqueeze(0), weight=ce_weight, reduction='mean')

                if args.binary:
                    dice = binary_dice_loss(pred_full.unsqueeze(0), target.unsqueeze(0))
                else:
                    dice = multiclass_dice_loss(pred_full.unsqueeze(0), target.unsqueeze(0))

                loss_val = (0.5 * ce + 0.5 * dice + args.hdn_weight * hdn_loss) / B

                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    nan_count += 1
                    continue

                # Accumulate gradients (no step() until batch is done)
                loss_val.backward()
                batch_loss += loss_val.item() * B
                batch_ce += ce.item()
                batch_dice += dice.item()
                valid_samples += 1

            if valid_samples == 0:
                continue

            # ── Post-batch: gradient check + clip + step ──
            all_params = list(decoder.parameters()) + list(frn.parameters())
            if hdn is not None:
                all_params += list(hdn.parameters())
            if stage2_active:
                all_params += stage2_params
            grad_nan = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in all_params
            )
            if grad_nan:
                optimizer.zero_grad()
                nan_count += valid_samples
                continue

            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_losses.append(batch_loss / valid_samples)
            epoch_ce.append(batch_ce / valid_samples)
            epoch_dice.append(batch_dice / valid_samples)

            if epoch_losses:
                postfix = {
                    "loss": f"{np.mean(epoch_losses[-50:]):.4f}",
                    "ce": f"{np.mean(epoch_ce[-50:]):.4f}",
                    "dice": f"{np.mean(epoch_dice[-50:]):.4f}",
                }
                pbar.set_postfix(postfix)

        # ── Epoch summary ──
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        log_msg = f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f}"
        if epoch_ce:
            log_msg += f" ce={np.mean(epoch_ce):.4f}"
        if epoch_dice:
            log_msg += f" dice={np.mean(epoch_dice):.4f}"
        log_msg += f" | lr={scheduler.get_last_lr()[0]:.2e} | NaN={nan_count}"

        # DA-FRN stats
        frn_stats = frn.get_stats()
        if frn_stats.get("cdf_enabled"):
            log_msg += f" | w3={frn_stats.get('w3_mean', 0):.3f} w4={frn_stats.get('w4_mean', 0):.3f}"

        logger.log_info("epoch", log_msg)
        logger.log_metric("loss", avg_loss, step=epoch, tags=["severstal"])
        if epoch_ce:
            logger.log_metric("ce", np.mean(epoch_ce), step=epoch, tags=["severstal"])
        if epoch_dice:
            logger.log_metric("dice", np.mean(epoch_dice), step=epoch, tags=["severstal"])

        # ── Evaluation ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"{'─'*50}")
            logger.log_info("eval", f"Evaluation @ Epoch {epoch}")
            result = evaluate(decoder, frn, backbone, val_ds, device,
                            NUM_CLASSES, CLASS_NAMES)
            miou = result["mIoU"]
            logger.log_info("eval", f"  mIoU={miou:.4f}  best={best_miou:.4f} (epoch {best_epoch})")
            for cls_name, iou_c in result["per_class_IoU"].items():
                logger.log_info("eval", f"    {cls_name}: IoU={iou_c:.4f}")
            logger.log_metric("mIoU", miou, step=epoch, tags=["severstal"])

            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "frn_state_dict": {k: v.clone() for k, v in frn.state_dict().items()},
                    "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "stage2_active": stage2_active,
                    "mIoU": miou,
                    "per_class_IoU": result["per_class_IoU"],
                    "args": vars(args),
                    "frn_stats": frn_stats,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "module_tag": module_tag,
                    "mode": mode_str,
                }
                if hdn is not None:
                    ckpt["hdn_state_dict"] = {k: v.clone() for k, v in hdn.state_dict().items()}
                torch.save(ckpt, str(out_dir / "best_model.pt"))
                logger.log_info("eval", f"  New best: mIoU={best_miou:.4f} @ epoch {best_epoch}")

    # ── Final save ──
    final = {
        "epoch": args.epochs,
        "global_step": global_step,
        "frn_state_dict": {k: v.clone() for k, v in frn.state_dict().items()},
        "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "stage2_active": stage2_active,
        "best_mIoU": best_miou,
        "best_epoch": best_epoch,
        "args": vars(args),
        "frn_stats": frn.get_stats(),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "nan_skip_count": nan_count,
        "module_tag": module_tag,
        "mode": mode_str,
    }
    if hdn is not None:
        final["hdn_state_dict"] = {k: v.clone() for k, v in hdn.state_dict().items()}
    torch.save(final, str(out_dir / "last_model.pt"))

    # ── Done ──
    print(f"\n{'='*60}")
    print(f"  DA-FRN Training — Severstal — Complete")
    print(f"  Dataset: Severstal ({mode_str}, {IMG_H}×{IMG_W})")
    print(f"  Modules: {module_tag}")
    print(f"  Trainable: {total_trainable:,} (DA-FRN: {frn_params:,}, Decoder: {dec_params:,})")
    print(f"  Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    print(f"  NaN skips: {nan_count}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")
    logger.log_info("done", f"Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
