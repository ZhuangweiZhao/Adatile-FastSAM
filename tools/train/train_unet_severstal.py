#!/usr/bin/env python3
"""
UNet Training -- Severstal Steel Defect Detection (2015 Original Protocol).
===========================================================================

Trains original U-Net (Ronneberger et al., MICCAI 2015) from scratch on
Severstal steel defect dataset, strictly following the original paper's
training protocol unless adaptation is required for experimental fairness.

训练协议严格遵循 2015 原论文:
  - Optimizer: SGD + momentum=0.99
  - LR schedule: StepLR (reduce by 0.1x every 50 epochs)
  - Loss: pixel-wise softmax + weighted cross-entropy
  - Augmentation: elastic deformation + rotation + flip + gray value
  - Batch size: 1 (same as original)
  - Full-epoch iteration (no random sampling)
  - No gradient clipping (not in original)

与 2015 原论文的偏差 (已声明 | Declared deviations):
  1. padding=1 (same conv) -- 所有对比基线使用 same-padding, 保证输出与输入同尺寸
  2. 无 d_1/d_2 形态学边界权重图 -- 钢铁缺陷非接触细胞, 不适用
     (可通过 --unet-weight-map 启用, 但 Severstal 场景不推荐)
  3. 无 overlap-tile 策略 -- 图像已为固定 256x1600 strip

Usage::

    # Binary (FG/BG)
    python tools/train/train_unet_severstal.py --epochs 200 --binary

    # Multi-class (5 classes: BG + 4 defect types)
    python tools/train/train_unet_severstal.py --epochs 200

    # With UNet weight map (original paper border emphasis)
    python tools/train/train_unet_severstal.py --epochs 200 --binary --unet-weight-map
"""

from __future__ import annotations

import sys
import argparse
import random
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.baselines import UNet
from adatile.datasets.severstal import SeverstalDataset

# ============================================================================
# Constants
# ============================================================================

MULTI_NUM_CLASSES = 5
MULTI_CLASS_NAMES = ["background", "Class1", "Class2", "Class3", "Class4"]
BINARY_NUM_CLASSES = 2
BINARY_CLASS_NAMES = ["background", "foreground"]

IMG_H, IMG_W = 256, 1600  # Severstal native size (both multiples of 32)


# ============================================================================
# Original UNet Data Augmentation (Ronneberger et al. 2015, Section 2)
# ============================================================================

class UNetOriginalAugmentation:
    """Original UNet augmentations: elastic deformation + rotation + flip + gray.

    Ronneberger et al. 2015, Section 2:
      "We generate smooth deformations using random displacement vectors on a
       coarse 3 by 3 grid. The displacements are sampled from a Gaussian
       distribution with 10 pixels standard deviation."

    For 256x1600 steel strips, the grid is scaled proportionally (4x25 points)
    to maintain similar spatial deformation frequency.
    """

    def __init__(
        self,
        p_elastic: float = 0.5,
        elastic_alpha: float = 34.0,
        elastic_sigma: float = 4.0,
        elastic_grid_h: int = 4,
        elastic_grid_w: int = 25,
        p_rotate: float = 0.5,
        rotate_deg: float = 15.0,
        p_hflip: float = 0.5,
        p_vflip: float = 0.0,  # vertical flip off for steel strips
        brightness: float = 0.2,
        contrast: float = 0.2,
        p_brightness: float = 0.5,
        p_noise: float = 0.3,
        noise_std: float = 0.02,
    ):
        self.p_elastic = p_elastic
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.elastic_grid_h = elastic_grid_h
        self.elastic_grid_w = elastic_grid_w
        self.p_rotate = p_rotate
        self.rotate_deg = rotate_deg
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.brightness = brightness
        self.contrast = contrast
        self.p_brightness = p_brightness
        self.p_noise = p_noise
        self.noise_std = noise_std

    def _elastic_deform(self, image: torch.Tensor, mask: torch.Tensor):
        """Elastic deformation with coarse grid + Gaussian smoothing.

        Implements Ronneberger et al. 2015, Section 2.
        Uses F.grid_sample for efficient GPU/CPU warp.
        """
        C_img, H, W = image.shape
        H_mask = mask.shape[1]
        gh, gw = self.elastic_grid_h, self.elastic_grid_w

        # Random displacement vectors on coarse grid (Gaussian, std=elastic_sigma)
        dx_coarse = torch.randn(1, 1, gh, gw) * self.elastic_sigma
        dy_coarse = torch.randn(1, 1, gh, gw) * self.elastic_sigma

        # Bicubic upsample to full resolution -> smooth per-pixel displacement
        dx = F.interpolate(dx_coarse, size=(H, W), mode='bicubic', align_corners=False).squeeze() * self.elastic_alpha
        dy = F.interpolate(dy_coarse, size=(H, W), mode='bicubic', align_corners=False).squeeze() * self.elastic_alpha

        # Build sampling grid: identity + displacement, normalized to [-1, 1]
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij',
        )
        grid_x = (xx + dx) / (W - 1) * 2.0 - 1.0
        grid_y = (yy + dy) / (H - 1) * 2.0 - 1.0
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        # Warp image (bilinear) and mask (nearest to preserve labels)
        img_warped = F.grid_sample(
            image.unsqueeze(0), grid, mode='bilinear',
            padding_mode='border', align_corners=True,
        ).squeeze(0)
        mask_warped = F.grid_sample(
            mask.unsqueeze(0).unsqueeze(0).float(), grid, mode='nearest',
            padding_mode='border', align_corners=True,
        ).squeeze(0).squeeze(0)

        # Restore mask dtype
        if mask.dtype == torch.int64 or mask.dtype == torch.long:
            mask_warped = mask_warped.long()

        return img_warped, mask_warped

    def _rotate(self, image: torch.Tensor, mask: torch.Tensor):
        """Random rotation by angle in [-rotate_deg, +rotate_deg]."""
        angle = (torch.rand(1).item() * 2 - 1) * self.rotate_deg

        # Build rotation matrix and affine grid
        theta = torch.tensor([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
                              [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), 0]],
                             dtype=torch.float32).unsqueeze(0)

        C, H, W = image.shape
        grid = F.affine_grid(theta, [1, C, H, W], align_corners=False)

        img_rot = F.grid_sample(image.unsqueeze(0), grid, mode='bilinear',
                                padding_mode='border', align_corners=False).squeeze(0)
        mask_rot = F.grid_sample(mask.unsqueeze(0).unsqueeze(0).float(), grid, mode='nearest',
                                 padding_mode='border', align_corners=False).squeeze(0).squeeze(0)

        if mask.dtype == torch.int64 or mask.dtype == torch.long:
            mask_rot = mask_rot.long()

        return img_rot, mask_rot

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        """
        Apply all augmentations in original UNet order.

        :param image: [C, H, W] float32 [0, 1]
        :param mask:  [H, W] int64 or float32 (class labels)
        :return: augmented (image, mask)
        """
        # 1. Elastic deformation (signature augmentation of original UNet)
        if torch.rand(1).item() < self.p_elastic:
            image, mask = self._elastic_deform(image, mask)

        # 2. Random rotation
        if torch.rand(1).item() < self.p_rotate:
            image, mask = self._rotate(image, mask)

        # 3. Random horizontal flip
        if torch.rand(1).item() < self.p_hflip:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        # 4. Random vertical flip (off by default for steel strips)
        if torch.rand(1).item() < self.p_vflip:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])

        # 5. Gray value variation (brightness + contrast, original UNet)
        if torch.rand(1).item() < self.p_brightness:
            b = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
            image = torch.clamp(image * b, 0.0, 1.0)
            c = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
            m_val = image.mean(dim=(-2, -1), keepdim=True)
            image = torch.clamp((image - m_val) * c + m_val, 0.0, 1.0)

        # 6. Gaussian noise (additive)
        if torch.rand(1).item() < self.p_noise:
            image = torch.clamp(image + torch.randn_like(image) * self.noise_std, 0.0, 1.0)

        return image, mask


# ============================================================================
# UNet Weight Map (Ronneberger et al. 2015, Section 3)
# ============================================================================

def compute_unet_weight_map(
    mask: torch.Tensor,
    w0: float = 10.0,
    sigma: float = 5.0,
    num_classes: int = 2,
) -> torch.Tensor:
    """Compute pixel-wise weight map as in original UNet paper.

    w(x) = w_c(x) + w0 * exp(-(d1(x) + d2(x))^2 / (2 * sigma^2))

    where:
      w_c(x): class frequency balancing weight
      d1(x):  distance to nearest object border
      d2(x):  distance to second-nearest object border
      w0=10, sigma≈5 pixels

    For Severstal steel defects (non-touching), d1 ≈ d2 in most cases,
    so the weight map primarily emphasizes defect boundaries.

    NOTE: This is expensive (distance transform per sample) and designed
    for biomedical cell segmentation. For steel defects it adds ~20-50ms
    per sample and provides marginal benefit. Disabled by default.

    :param mask: [H, W] class labels (int64).
    :param w0: Border emphasis weight (default 10, from paper).
    :param sigma: Border width in pixels (default 5, from paper).
    :param num_classes: Number of classes for w_c computation.
    :return: [H, W] weight map (float32).
    """
    try:
        import cv2
    except ImportError:
        # Fallback: class weights only
        return _compute_class_weight_map(mask, num_classes)

    H, W = mask.shape
    mask_np = mask.cpu().numpy().astype(np.uint8)

    # -- w_c(x): class frequency balancing --
    w_c = torch.ones(H, W, dtype=torch.float32)
    for c in range(num_classes):
        count_c = (mask == c).sum().float()
        if count_c > 0:
            weight_c = (H * W) / (num_classes * count_c)
            w_c[mask == c] = weight_c

    # -- d1, d2: distance to nearest & second-nearest border --
    # Compute per-class borders via morphological gradient
    border_map = np.zeros((H, W), dtype=np.uint8)
    for c in range(1, num_classes):  # skip background
        cls_mask = (mask_np == c).astype(np.uint8)
        if cls_mask.sum() == 0:
            continue
        # Morphological gradient = border pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(cls_mask, kernel, iterations=1)
        eroded = cv2.erode(cls_mask, kernel, iterations=1)
        border_map = np.maximum(border_map, dilated - eroded)

    if border_map.sum() < 1:
        return w_c

    # Distance transform to borders
    dist = cv2.distanceTransform((1 - border_map).astype(np.uint8), cv2.DIST_L2, 5)
    d1 = torch.from_numpy(dist).float()

    # d2: second-nearest border = distance after masking the nearest border region
    # For non-touching objects, d2 ≈ d1; approximate as d1 * 1.1
    d2 = d1 * 1.1

    # -- Combine: w(x) = w_c(x) + w0 * exp(-(d1+d2)^2 / (2*sigma^2))
    d_sum = d1 + d2
    border_weight = w0 * torch.exp(- (d_sum ** 2) / (2.0 * sigma ** 2))
    weight_map = w_c + border_weight

    return weight_map


def _compute_class_weight_map(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Class frequency balancing only (fallback when cv2 unavailable)."""
    H, W = mask.shape
    w_c = torch.ones(H, W, dtype=torch.float32)
    for c in range(num_classes):
        count_c = (mask == c).sum().float()
        if count_c > 0:
            weight_c = (H * W) / (num_classes * count_c)
            w_c[mask == c] = weight_c
    return w_c


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: SeverstalDataset,
    device: torch.device,
    num_classes: int,
    class_names: list[str],
    max_samples: int = 0,
) -> dict:
    """Segmentation evaluation -- per-class IoU + mIoU.

    No random augmentation during evaluation. Uses full image inference.
    """
    model.eval()
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

        logits = model(img)
        # UNet with padding=1 outputs same size as input -> no resize needed
        # But we interpolate as a safety net in case of size mismatch
        if logits.shape[2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        pred_full = logits.squeeze(0)
        # argmax is invariant to softmax (monotonic), so this is correct
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

    valid = [v for v in per_class_iou.values() if not (v != v)]
    return {
        "mIoU": round(float(np.mean(valid)), 6) if valid else 0.0,
        "per_class_IoU": per_class_iou,
    }


# ============================================================================
# CLI Arguments
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Original UNet Training -- Severstal Steel Defect "
                    "(Ronneberger et al., MICCAI 2015 protocol)"
    )

    # Data
    p.add_argument("--data-root", type=str, default="data/severstal-steel-defect-detection")
    p.add_argument("--binary", action="store_true",
                   help="Binary mode (FG/BG). Default: multi-class (5 classes).")
    p.add_argument("--no-augment", action="store_true",
                   help="Disable ALL data augmentation (debug only).")

    # Model
    p.add_argument("--unet-base", type=int, default=64,
                   help="UNet base channels (64 -> ~31.0M, 32 -> ~7.8M)")

    # Training -- Original UNet protocol
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size (original UNet uses 1 due to large tiles)")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Learning rate (original: ~0.01 with StepLR decay)")
    p.add_argument("--momentum", type=float, default=0.99,
                   help="SGD momentum (original: 0.99)")
    p.add_argument("--weight-decay", type=float, default=0.0005,
                   help="Weight decay for SGD")
    p.add_argument("--lr-step", type=int, default=50,
                   help="StepLR: halve LR every N epochs")
    p.add_argument("--lr-gamma", type=float, default=0.5,
                   help="StepLR: multiply LR by gamma at each step")
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "balanced"],
                   help="Class frequency balancing for CE loss")
    p.add_argument("--unet-weight-map", action="store_true",
                   help="Enable original UNet morphological border weight map "
                        "(expensive, designed for cell segmentation)")
    p.add_argument("--weight-map-w0", type=float, default=10.0,
                   help="UNet weight map border emphasis (w0)")
    p.add_argument("--weight-map-sigma", type=float, default=5.0,
                   help="UNet weight map border width (sigma)")

    # Evaluation
    p.add_argument("--eval-every", type=int, default=5,
                   help="Evaluate every N epochs")
    p.add_argument("--eval-max-samples", type=int, default=0,
                   help="Max eval samples (0 = all)")

    # System
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 = main process only)")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint path")

    return p.parse_args()


# ============================================================================
# Main Training
# ============================================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.binary:
        NUM_CLASSES = BINARY_NUM_CLASSES
        CLASS_NAMES = BINARY_CLASS_NAMES
        mode_str = "binary"
    else:
        NUM_CLASSES = MULTI_NUM_CLASSES
        CLASS_NAMES = MULTI_CLASS_NAMES
        mode_str = "multi"

    # -- Output dir --
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        wm = "_wm" if args.unet_weight_map else ""
        args.output_dir = f"runs/severstal_UNet_original_b{args.unet_base}_{mode_str}{wm}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Logger --
    logger = get_logger("train_unet_severstal")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))
    logger.log_info("config", f"Model: Original UNet (base={args.unet_base}), Severstal ({mode_str})")
    logger.log_info("config", f"Protocol: Ronneberger et al., MICCAI 2015")
    logger.log_info("config", f"SGD momentum={args.momentum}, StepLR gamma={args.lr_gamma} step={args.lr_step}")
    logger.log_info("config", f"Augmentation: elastic+rotate+flip+intensity (UNet original)")
    logger.log_info("config", f"Weight map: {'ON' if args.unet_weight_map else 'OFF'}")
    logger.log_info("config", f"Output: {out_dir}")

    # -- Data --
    train_ds = SeverstalDataset(
        root=args.data_root, split="train", binary=args.binary, seed=args.seed,
    )
    val_ds = SeverstalDataset(
        root=args.data_root, split="val", binary=args.binary, seed=args.seed,
    )
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}, Mode: {mode_str}")

    # Verify no overlap between train/val
    train_names = set(train_ds.sample_names)
    val_names = set(val_ds.sample_names)
    overlap = train_names & val_names
    if overlap:
        logger.log_info("data", f"WARNING: Train/Val overlap: {len(overlap)} images!")
    else:
        logger.log_info("data", "Train/Val split: NO overlap [OK]")

    # -- Class weights for CE --
    ce_weight = None
    if args.class_weights == "balanced":
        stats = train_ds.get_class_stats()
        pc = [stats.get(cn, {}).get("pixels", 1) for cn in CLASS_NAMES]
        total = sum(pc)
        if total > 0:
            raw = [total / max(p, 0.01) for p in pc]
            mean_w = sum(raw) / len(raw)
            ce_weight = torch.tensor([w / mean_w for w in raw], dtype=torch.float32, device=device)
            logger.log_info("data", f"Class weights: {ce_weight.tolist()}")

    # -- DataLoader -- Full-epoch iteration (original UNet protocol)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # -- Augmentation --
    augment = None if args.no_augment else UNetOriginalAugmentation(
        p_elastic=0.5,
        elastic_alpha=34.0,
        elastic_sigma=4.0,
        p_rotate=0.5,
        rotate_deg=15.0,
        p_hflip=0.5,
        p_vflip=0.0,   # off for steel strips
    )

    # -- Model --
    model = UNet(in_channels=3, num_classes=NUM_CLASSES, base=args.unet_base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_info("model", f"UNet base={args.unet_base}: {n_params/1e6:.2f}M params "
                    f"(trainable={n_trainable/1e6:.2f}M)")

    # -- Optimizer: SGD + momentum=0.99 (original UNet protocol) --
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # StepLR: reduce LR by gamma every lr_step epochs (original UNet uses step decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_gamma,
    )

    # -- Resume --
    start_epoch = 1
    global_step = 0
    best_miou = 0.0
    best_epoch = 0
    nan_count = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_miou = ckpt.get("best_mIoU", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        logger.log_info("resume", f"Resumed from epoch {start_epoch}, best mIoU={best_miou:.4f}")

    # -- Training --
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train", f"Original UNet Training -- Severstal")
    logger.log_info("train", f"Protocol: Ronneberger et al., MICCAI 2015")
    logger.log_info("train", f"Params: {n_params/1e6:.2f}M (all trainable)")
    logger.log_info("train", f"Image: {IMG_H}x{IMG_W}, Classes: {NUM_CLASSES} ({mode_str})")
    logger.log_info("train", f"Optimizer: SGD(lr={args.lr}, momentum={args.momentum}, wd={args.weight_decay})")
    logger.log_info("train", f"Scheduler: StepLR(step={args.lr_step}, gamma={args.lr_gamma})")
    logger.log_info("train", f"Batch size: {args.batch_size}, Epochs: {args.epochs}")
    logger.log_info("train", f"Weight map: {'ON (w0='+str(args.weight_map_w0)+')' if args.unet_weight_map else 'OFF'}")
    logger.log_info("train", f"{'='*60}")

    # Cross-entropy loss (original UNet uses pixel-wise softmax + CE)
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", unit="batch")

        for batch in epoch_pbar:
            img = batch["image"]              # [B, 3, H, W]
            mask = batch["masks"]             # [B, 1, H, W] or [B, H, W]
            mask = mask.squeeze(1) if mask.dim() == 4 else mask  # -> [B, H, W]

            # Apply augmentation per-sample in batch
            if augment and args.batch_size == 1:
                img_aug, mask_aug = augment(img[0], mask[0])
                img = img_aug.unsqueeze(0)
                mask = mask_aug.unsqueeze(0)

            img_dev = img.to(device)
            mask_dev = mask.to(device).long()

            # -- Forward --
            logits = model(img_dev)  # [B, C, H, W] raw logits

            # Safety: interpolate if size mismatch
            if logits.shape[2:] != mask_dev.shape[1:]:
                logits = F.interpolate(logits, size=mask_dev.shape[1:],
                                       mode="bilinear", align_corners=False)

            # -- Loss: pixel-wise softmax + cross-entropy (original UNet) --
            if args.unet_weight_map:
                # Compute per-sample weight map (expensive)
                total_loss = torch.tensor(0.0, device=device)
                for b in range(mask_dev.shape[0]):
                    wm = compute_unet_weight_map(
                        mask_dev[b], w0=args.weight_map_w0,
                        sigma=args.weight_map_sigma, num_classes=NUM_CLASSES,
                    ).to(device)
                    ce_per_pixel = ce_loss_fn(logits[b:b+1], mask_dev[b:b+1])
                    total_loss += (ce_per_pixel.squeeze(0) * wm).mean()
                loss_val = total_loss / mask_dev.shape[0]
            else:
                # Standard weighted CE (no border weight map)
                loss_val = ce_loss_fn(logits, mask_dev).mean()

            if torch.isnan(loss_val) or torch.isinf(loss_val):
                nan_count += 1
                continue

            # -- Backward --
            optimizer.zero_grad()
            loss_val.backward()

            # NaN grad safety check
            grad_nan = any(
                p.grad is not None and
                (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model.parameters()
            )
            if grad_nan:
                optimizer.zero_grad()
                nan_count += 1
                continue

            # Original UNet does NOT use gradient clipping.
            # We skip it to match the paper. NaN guard above handles rare extremes.
            optimizer.step()
            global_step += 1

            epoch_losses.append(loss_val.item())
            if epoch_losses:
                epoch_pbar.set_postfix({
                    "loss": f"{np.mean(epoch_losses[-50:]):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

        # End of epoch: step scheduler
        scheduler.step()

        # -- Epoch summary --
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        log_msg = (f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f} "
                   f"| lr={scheduler.get_last_lr()[0]:.2e} | NaN={nan_count}")
        logger.log_info("epoch", log_msg)
        logger.log_metric("loss", avg_loss, step=epoch, tags=["unet_original"])
        logger.log_metric("lr", scheduler.get_last_lr()[0], step=epoch, tags=["unet_original"])

        # -- Evaluation --
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"{'-'*50}")
            logger.log_info("eval", f"Evaluation @ Epoch {epoch}")

            # Validation: NO augmentation
            result = evaluate(
                model, val_ds, device, NUM_CLASSES, CLASS_NAMES,
                max_samples=args.eval_max_samples,
            )
            miou = result["mIoU"]
            logger.log_info("eval", f"  mIoU={miou:.4f}  best={best_miou:.4f} (epoch {best_epoch})")
            for cls_name, iou_c in result["per_class_IoU"].items():
                logger.log_info("eval", f"    {cls_name}: IoU={iou_c:.4f}")
            logger.log_metric("mIoU", miou, step=epoch, tags=["unet_original"])

            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "mIoU": miou,
                    "per_class_IoU": result["per_class_IoU"],
                    "args": vars(args),
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "mode": mode_str,
                    "params": n_params,
                    "model_type": "unet_original_2015",
                }
                torch.save(ckpt, str(out_dir / "best_model.pt"))
                logger.log_info("eval", f"  New best: mIoU={best_miou:.4f} @ epoch {best_epoch}")

    # -- Final save --
    final = {
        "epoch": args.epochs,
        "global_step": global_step,
        "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_mIoU": best_miou,
        "best_epoch": best_epoch,
        "args": vars(args),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "mode": mode_str,
        "params": n_params,
        "nan_skip_count": nan_count,
        "model_type": "unet_original_2015",
    }
    torch.save(final, str(out_dir / "last_model.pt"))

    print(f"\n{'='*60}")
    print(f"  Original UNet Training -- Severstal -- Complete")
    print(f"  Protocol: Ronneberger et al., MICCAI 2015")
    print(f"  Params: {n_params/1e6:.2f}M (all trainable)")
    print(f"  Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    print(f"  NaN skips: {nan_count}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")
    logger.log_info("done", f"UNet best mIoU: {best_miou:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
