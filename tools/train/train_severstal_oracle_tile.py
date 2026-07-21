#!/usr/bin/env python3
"""
Oracle Tile + Episodic Few-Shot — Severstal 钢铁缺陷检测.
==========================================================

Step 2 (P1): GT Oracle Tile 选择 + Episodic Few-Shot 消融实验.

核心问题 | Core Question:
    如果 Oracle (GT mask) 完美知道哪些 Tile 有缺陷,
    只对 Top-K Tile 做 Backbone + Decoder 计算,
    精度能保持吗？计算量能省多少？

实验设计 | Design:
    Baseline: 全图 256×1600 → Backbone → Decoder → Loss
    Oracle Tile: 切 W=100 Tile (16 个) → GT 选 Top-K → 每 Tile Backbone → Decoder → Loss

对比维度 | Comparison:
    - mIoU (精度是否持平?)
    - FLOPs ratio (计算量省多少?)
    - Prototype quality (只从缺陷 Tile 提取 vs 全图)

零训练成本前提 | Zero-Training-Cost Premise:
    不需要先训练 SPM. GT mask 直接当 Oracle Importance Map.

用法 | Usage::

    # 快速验证: K=1, W=100, Top-2 tiles, 少量 epoch
    python tools/train/train_severstal_oracle_tile.py \
        --k-shot 1 --seeds 42 --epochs 5 \
        --tile-width 100 --top-k-tiles 2 --device cuda

    # 完整对比: baseline vs oracle, K=1/5, Top-2 vs Top-3 tiles
    python tools/train/train_severstal_oracle_tile.py \
        --k-shot 1 5 --seeds 42 123 456 \
        --epochs 50 --tile-width 100 --top-k-tiles 2 \
        --device cuda --mode both
"""

from __future__ import annotations

import sys, json, argparse, random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

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
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
from adatile.datasets.severstal import SeverstalDataset


# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

IMG_H, IMG_W = 256, 1600  # Severstal native (multiples of 32)
DEFECT_CLASSES = [1, 2, 3, 4]
STRIDE_MULTIPLE = 32  # FastSAM requires input dims to be multiples of 32
P4_STRIDE = 16         # FastSAM P4 downsample factor


# ═══════════════════════════════════════════════════════════════════
# Tile 工具函数 | Tile Utility Functions
# ═══════════════════════════════════════════════════════════════════

def cut_into_tiles(
    image: torch.Tensor,    # [3, H, W]
    mask: torch.Tensor,      # [H, W]
    tile_w: int = 100,
) -> list[dict]:
    """
    将 256×1600 图像沿宽度方向切为 Tile (高度固定 256).
    Cut 256×1600 image into tiles along width (height fixed at 256).

    每个 Tile 自动 pad 到 32 的倍数 (Backbone 要求).
    Each tile auto-padded to multiple of 32 (Backbone requirement).

    :param image: [3, 256, 1600] float tensor.
    :param mask: [256, 1600] long/int tensor (category labels).
    :param tile_w: Tile width (default 100 → 16 tiles/image).
    :return: List of {
        "image": [3, 256, tile_w_padded],
        "mask": [256, tile_w_padded],
        "orig_w": int,     # unpadded width
        "x_offset": int,   # x position in original image
        "tile_idx": int,   # 0-based tile index
    }
    """
    tiles = []
    tile_idx = 0
    for x in range(0, IMG_W, tile_w):
        tx = min(tile_w, IMG_W - x)
        tile_img = image[:, :, x:x + tx].clone()       # [3, 256, tx]
        tile_mask = mask[:, x:x + tx].clone()            # [256, tx]

        # Pad width to multiple of 32
        pad_w = (STRIDE_MULTIPLE - tx % STRIDE_MULTIPLE) % STRIDE_MULTIPLE
        if pad_w > 0:
            tile_img = F.pad(tile_img, (0, pad_w), value=0.0)
            tile_mask = F.pad(tile_mask, (0, pad_w), value=0)

        tiles.append({
            "image": tile_img,
            "mask": tile_mask,
            "orig_w": tx,
            "x_offset": x,
            "tile_idx": tile_idx,
        })
        tile_idx += 1

    return tiles


def oracle_select_tiles(
    tiles: list[dict],
    class_id: int,
    top_k: int = 2,
) -> list[dict]:
    """
    Oracle (GT mask) 选择包含指定缺陷类的 Top-K Tile.
    Oracle (GT mask) selects Top-K tiles containing the specified defect class.

    按每 Tile 内该类别的像素数降序排列，取 Top-K.
    Sorted by pixel count of target class per tile, descending → Top-K.

    :param tiles: cut_into_tiles() 的输出 | Output from cut_into_tiles().
    :param class_id: 目标缺陷类 | Target defect class (1-4).
    :param top_k: 保留的 Tile 数量 | Number of tiles to keep.
    :return: Top-K tiles sorted by defect pixel count (descending).
              Returns empty list if no tile has the target class.
    """
    tile_scores = []
    for t in tiles:
        mask = t["mask"]  # [256, tile_w_padded]
        # 只统计原始区域 (排除 padding) | Only count original region (exclude padding)
        orig_mask = mask[:, :t["orig_w"]]
        score = int((orig_mask == class_id).sum().item())
        if score > 0:
            tile_scores.append((score, t))

    if not tile_scores:
        return []

    # 按缺陷像素数降序 | Sort by defect pixel count descending
    tile_scores.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in tile_scores[:top_k]]


def estimate_flops_ratio(
    tile_w: int,
    top_k: int,
    n_tiles_per_image: int,
) -> dict:
    """
    估算 Oracle Tile 模式相对于 Baseline 的计算量比例.
    Estimate FLOPs ratio of Oracle Tile mode vs Baseline.

    假设 Backbone 和 Decoder 的计算量与输入像素数成正比.
    Assumes Backbone/Decoder FLOPs ∝ input pixel count.

    :return: {
        "backbone_ratio": float,   # Oracle / Baseline backbone pixels
        "decoder_ratio": float,    # Oracle / Baseline decoder spatial positions
        "total_ratio": float,      # Weighted average (0.7 backbone + 0.3 decoder)
        "savings_pct": float,      # % computation saved
    }
    """
    # Baseline: full image
    baseline_backbone = IMG_H * IMG_W  # 409,600 px
    baseline_decoder = (IMG_H // P4_STRIDE) * (IMG_W // P4_STRIDE)  # 16 × 100 = 1600

    # Oracle Tile: K tiles, each padded to multiple of 32
    padded_w = tile_w + (STRIDE_MULTIPLE - tile_w % STRIDE_MULTIPLE) % STRIDE_MULTIPLE
    oracle_backbone = top_k * IMG_H * padded_w
    oracle_decoder = top_k * (IMG_H // P4_STRIDE) * (padded_w // P4_STRIDE)

    backbone_ratio = oracle_backbone / baseline_backbone
    decoder_ratio = oracle_decoder / baseline_decoder
    total_ratio = 0.7 * backbone_ratio + 0.3 * decoder_ratio
    savings_pct = (1.0 - total_ratio) * 100

    return {
        "backbone_ratio": round(backbone_ratio, 4),
        "decoder_ratio": round(decoder_ratio, 4),
        "total_ratio": round(total_ratio, 4),
        "savings_pct": round(savings_pct, 1),
        "oracle_pixels": oracle_backbone,
        "baseline_pixels": baseline_backbone,
    }


# ═══════════════════════════════════════════════════════════════════
# 数据增强 | Data Augmentation (copied from baseline)
# ═══════════════════════════════════════════════════════════════════

class BasicAugmentation:
    """极简增强: flip + rotate + brightness + noise (steel strip aware)."""

    def __init__(self, p_flip=0.5, p_rotate=0.3, brightness=0.2, contrast=0.2, noise_std=0.02):
        self.p_hflip = p_flip
        self.p_vflip = p_flip * 0.5
        self.p_rotate = p_rotate
        self.brightness = brightness
        self.contrast = contrast
        self.noise_std = noise_std

    def __call__(self, image, mask):
        if torch.rand(1).item() < self.p_hflip:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        if torch.rand(1).item() < self.p_vflip:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        if torch.rand(1).item() < self.p_rotate:
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])
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
# Episode 采样器 | Episode Sampler (copied from baseline)
# ═══════════════════════════════════════════════════════════════════

class EpisodeSampler:
    """每步: 随机类 → K support + 1 query (同类别，不同图像)."""

    def __init__(self, dataset, class_ids=None, k_shot=1, seed=42):
        self.dataset = dataset
        self.class_ids = class_ids or DEFECT_CLASSES
        self.k_shot = k_shot
        self.rng = random.Random(seed)
        self._class_pool: dict[int, list[int]] = {}
        for cls_id in self.class_ids:
            pool = dataset.class_to_images(cls_id)
            if len(pool) <= k_shot:
                raise ValueError(
                    f"Class {cls_id} has only {len(pool)} images, "
                    f"need at least {k_shot + 1} (K={k_shot} support + 1 query)."
                )
            self._class_pool[cls_id] = pool

    def sample(self) -> dict:
        cls_id = self.rng.choice(self.class_ids)
        pool = self._class_pool[cls_id]
        sampled = self.rng.sample(pool, self.k_shot + 1)
        return {
            "class_id": cls_id,
            "support_indices": sampled[:self.k_shot],
            "query_index": sampled[self.k_shot],
        }


# ═══════════════════════════════════════════════════════════════════
# 工具函数 | Utility Functions (copied/modified from baseline)
# ═══════════════════════════════════════════════════════════════════

def get_class_mask(sample: dict, class_id: int) -> torch.Tensor:
    """从数据集样本中提取指定类别的二值掩码."""
    mask = sample["masks"].squeeze(0)  # [H, W]
    if mask.max() > 1:
        return (mask == class_id).float()
    return mask.float()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """二值 Dice 损失."""
    pred_f = pred.flatten()
    target_f = target.flatten()
    if target_f.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    inter = (pred_f * target_f).sum()
    union = pred_f.sum() + target_f.sum()
    return 1.0 - (2.0 * inter + smooth) / (union + smooth)


def compute_prototype_full_image(
    backbone: FastSAMBackbone,
    support_images: torch.Tensor,    # [K, 3, H, W]
    support_masks: torch.Tensor,     # [K, H, W]
    device: torch.device,
) -> torch.Tensor:
    """
    Baseline: 从全图 support 图像计算 L2-normalized FG prototype.
    """
    K = support_images.shape[0]
    feats_list = []
    for i in range(K):
        img = support_images[i:i + 1].to(device)
        mask = support_masks[i].to(device)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        with torch.no_grad():
            feats = backbone(img)
            p4 = feats["p4"]
        _, _, H_p4, W_p4 = p4.shape
        mask_p4 = F.interpolate(
            mask.unsqueeze(0).float(), size=(H_p4, W_p4), mode="nearest"
        ).squeeze(0)
        fg_area = mask_p4.sum()
        if fg_area > 0:
            proto = (p4.squeeze(0) * mask_p4).sum(dim=(1, 2)) / (fg_area + 1e-8)
            feats_list.append(proto)
    if not feats_list:
        return torch.zeros(1280, device=device)
    proto = torch.stack(feats_list).mean(dim=0)
    return F.normalize(proto, dim=0, p=2)


def compute_prototype_oracle_tile(
    backbone: FastSAMBackbone,
    support_images: torch.Tensor,    # [K, 3, H, W]
    support_masks: torch.Tensor,     # [K, H, W] category labels
    class_id: int,
    device: torch.device,
    tile_w: int = 100,
    top_k_tiles: int = 2,
) -> torch.Tensor:
    """
    Oracle Tile: 只从包含目标类别的 Tile 中提取 Prototype.
    Only extract prototype from tiles containing the target class.

    每张 support: 切 Tile → Oracle 选 Top-K → Backbone → P4 → FG pool.
    This gives a CLEANER prototype (no background tile noise).
    """
    K = support_images.shape[0]
    feats_list = []

    for i in range(K):
        img = support_images[i]       # [3, H, W]
        mask = support_masks[i]        # [H, W] category labels

        # Cut into tiles → Oracle select
        tiles = cut_into_tiles(img, mask, tile_w=tile_w)
        selected = oracle_select_tiles(tiles, class_id, top_k=top_k_tiles)

        for t in selected:
            tile_img = t["image"].unsqueeze(0).to(device)        # [1, 3, 256, W']
            tile_mask = t["mask"].to(device)                      # [256, W']
            fg_binary = (tile_mask == class_id).float()           # [256, W']

            if fg_binary.sum() == 0:
                continue

            with torch.no_grad():
                feats = backbone(tile_img)
                p4 = feats["p4"]  # [1, 1280, 16, W'/16]

            # Resize mask to P4 resolution
            _, _, H_p4, W_p4 = p4.shape
            mask_p4 = F.interpolate(
                fg_binary.unsqueeze(0).unsqueeze(0).float(),
                size=(H_p4, W_p4), mode="nearest"
            ).squeeze(0).squeeze(0)  # [H_p4, W_p4]

            fg_area = mask_p4.sum()
            if fg_area > 0:
                proto = (p4.squeeze(0) * mask_p4.unsqueeze(0)).sum(dim=(1, 2)) / (fg_area + 1e-8)
                feats_list.append(proto)

    if not feats_list:
        return torch.zeros(1280, device=device)

    proto = torch.stack(feats_list).mean(dim=0)
    return F.normalize(proto, dim=0, p=2)


def _get_gt_binary(gt: torch.Tensor, class_id: int, binary: bool) -> torch.Tensor:
    """获取指定类的 GT binary mask."""
    if binary:
        return (gt > 0) if class_id == 1 else (gt == 0)
    return (gt == class_id)


# ═══════════════════════════════════════════════════════════════════
# 训练: Baseline 模式 (全图) | Training: Baseline Mode (Full Image)
# ═══════════════════════════════════════════════════════════════════

def _train_epoch_baseline(
    decoder, backbone, sampler, train_ds, device, optimizer, scheduler,
    epoch, args, augment, logger,
) -> dict:
    """Baseline episodic training epoch: full image → backbone → decoder."""
    decoder.train()
    epoch_losses, epoch_ce_list, epoch_dice_list = [], [], []
    nan_count = 0
    total_backbone_px = 0
    total_decoder_pos = 0

    pbar = tqdm(range(args.episodes_per_epoch), desc=f"Base E{epoch:3d}",
                unit="ep", leave=False)
    for _ in pbar:
        episode = sampler.sample()
        cls_id = episode["class_id"]

        # Support
        s_imgs, s_masks = [], []
        for si in episode["support_indices"]:
            s = train_ds[si]
            s_imgs.append(s["image"])
            s_masks.append(get_class_mask(s, cls_id))
        s_imgs = torch.stack(s_imgs)
        s_masks = torch.stack(s_masks)

        # Query
        q = train_ds[episode["query_index"]]
        q_img = q["image"]
        q_mask = get_class_mask(q, cls_id)
        H, W = q_mask.shape

        if augment:
            q_img, q_mask = augment(q_img, q_mask)
            H, W = q_mask.shape

        q_img = q_img.unsqueeze(0).to(device)
        q_mask = q_mask.unsqueeze(0).to(device)

        # Prototype
        support_proto = compute_prototype_full_image(backbone, s_imgs, s_masks, device)

        # Query forward
        feats = backbone(q_img, extract_proto=True)
        p4 = feats["p4"]
        proto_masks = feats["proto"]

        pred_prob = decoder(p4, proto_masks, support_proto)
        pred_full = F.interpolate(
            pred_prob.unsqueeze(0), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)

        target = q_mask.squeeze(0)

        # Loss
        ce = F.binary_cross_entropy(pred_full.clamp(1e-7, 1 - 1e-7), target, reduction="mean")
        dice = dice_loss(pred_full, target)
        loss_val = 0.5 * ce + 0.5 * dice

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            nan_count += 1
            continue

        optimizer.zero_grad()
        loss_val.backward()

        grad_nan = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in decoder.parameters()
        )
        if grad_nan:
            optimizer.zero_grad()
            nan_count += 1
            continue

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss_val.item())
        epoch_ce_list.append(ce.item())
        epoch_dice_list.append(dice.item())
        total_backbone_px += IMG_H * IMG_W
        total_decoder_pos += (IMG_H // P4_STRIDE) * (IMG_W // P4_STRIDE)

        if epoch_losses:
            pbar.set_postfix({
                "loss": f"{np.mean(epoch_losses[-20:]):.4f}",
                "dice": f"{np.mean(epoch_dice_list[-20:]):.4f}",
                "cls": cls_id,
            })

    return {
        "avg_loss": np.mean(epoch_losses) if epoch_losses else 0.0,
        "nan_count": nan_count,
        "backbone_mpx": total_backbone_px / 1e6,
        "decoder_kpos": total_decoder_pos / 1e3,
    }


# ═══════════════════════════════════════════════════════════════════
# 训练: Oracle Tile 模式 | Training: Oracle Tile Mode
# ═══════════════════════════════════════════════════════════════════

def _train_epoch_oracle_tile(
    decoder, backbone, sampler, train_ds, device, optimizer, scheduler,
    epoch, args, augment, logger,
) -> dict:
    """
    Oracle Tile episodic training epoch.
    每步: 切 Tile → Oracle 选 Top-K → per-tile Backbone + Decoder → Loss.
    """
    decoder.train()
    epoch_losses, epoch_ce_list, epoch_dice_list = [], [], []
    nan_count, skip_count = 0, 0
    total_backbone_px = 0
    total_decoder_pos = 0

    tile_w = args.tile_width
    top_k = args.top_k_tiles
    padded_w = tile_w + (STRIDE_MULTIPLE - tile_w % STRIDE_MULTIPLE) % STRIDE_MULTIPLE

    pbar = tqdm(range(args.episodes_per_epoch), desc=f"Tile E{epoch:3d}",
                unit="ep", leave=False)
    for _ in pbar:
        episode = sampler.sample()
        cls_id = episode["class_id"]

        # ── 加载 query | Load query ──
        q = train_ds[episode["query_index"]]
        q_img = q["image"]      # [3, H, W]
        q_mask_full = q["masks"].squeeze(0)  # [H, W] category labels

        if augment:
            q_img, q_mask_full = augment(q_img, q_mask_full)

        # ── 切 query Tile + Oracle 选择 | Cut query tiles + Oracle select ──
        q_tiles = cut_into_tiles(q_img, q_mask_full, tile_w=tile_w)
        q_selected = oracle_select_tiles(q_tiles, cls_id, top_k=top_k)

        if not q_selected:
            skip_count += 1
            continue

        # ── Prototype (Oracle Tile) | Support prototype from defect tiles only ──
        s_imgs_list, s_masks_list = [], []
        for si in episode["support_indices"]:
            s = train_ds[si]
            s_imgs_list.append(s["image"])
            s_masks_list.append(s["masks"].squeeze(0))  # category labels
        s_imgs = torch.stack(s_imgs_list)
        s_masks = torch.stack(s_masks_list)

        support_proto = compute_prototype_oracle_tile(
            backbone, s_imgs, s_masks, cls_id, device,
            tile_w=tile_w, top_k_tiles=top_k,
        )

        # ── Per-Tile Query Forward | Process each selected tile ──
        tile_losses, tile_ces, tile_dices = [], [], []
        for t in q_selected:
            tile_img = t["image"].unsqueeze(0).to(device)        # [1, 3, 256, W_pad]
            tile_mask = t["mask"].to(device)                      # [256, W_pad]
            orig_w = t["orig_w"]

            # GT: binary mask for this class, crop to original width
            fg_binary = (tile_mask == cls_id).float()  # [256, W_pad]
            fg_binary = fg_binary[:, :orig_w]           # [256, orig_w]

            if fg_binary.sum() == 0:
                continue

            # Backbone
            feats = backbone(tile_img, extract_proto=True)
            p4 = feats["p4"]              # [1, 1280, 16, W_pad/16]
            proto_masks = feats["proto"]  # [1, 32, 64, W_pad/4]

            # Decoder
            pred_prob = decoder(p4, proto_masks, support_proto)  # [1, 64, W_pad/4]

            # Resize to tile resolution → crop to original width
            H_tile, W_tile_pad = tile_img.shape[2], tile_img.shape[3]
            pred_full_pad = F.interpolate(
                pred_prob.unsqueeze(0), size=(H_tile, W_tile_pad),
                mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)  # [256, W_tile_pad]

            pred_full = pred_full_pad[:, :orig_w]  # [256, orig_w]

            # Loss (只在有 FG 像素的 Tile 上计算 | Only on tiles with FG pixels)
            ce = F.binary_cross_entropy(
                pred_full.clamp(1e-7, 1 - 1e-7), fg_binary, reduction="mean"
            )
            dice = dice_loss(pred_full, fg_binary)
            loss_tile = 0.5 * ce + 0.5 * dice

            tile_losses.append(loss_tile)
            tile_ces.append(ce.item())
            tile_dices.append(dice.item())

            total_backbone_px += IMG_H * padded_w
            total_decoder_pos += (IMG_H // P4_STRIDE) * (padded_w // P4_STRIDE)

        if not tile_losses:
            skip_count += 1
            continue

        # ── 平均所有选中 Tile 的 Loss | Average loss across selected tiles ──
        loss_val = torch.stack(tile_losses).mean()
        avg_ce = np.mean(tile_ces)
        avg_dice = np.mean(tile_dices)

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            nan_count += 1
            continue

        optimizer.zero_grad()
        loss_val.backward()

        grad_nan = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in decoder.parameters()
        )
        if grad_nan:
            optimizer.zero_grad()
            nan_count += 1
            continue

        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss_val.item())
        epoch_ce_list.append(avg_ce)
        epoch_dice_list.append(avg_dice)

        if epoch_losses:
            pbar.set_postfix({
                "loss": f"{np.mean(epoch_losses[-20:]):.4f}",
                "dice": f"{np.mean(epoch_dice_list[-20:]):.4f}",
                "cls": cls_id, "t": len(q_selected),
            })

    return {
        "avg_loss": np.mean(epoch_losses) if epoch_losses else 0.0,
        "nan_count": nan_count,
        "skip_count": skip_count,
        "backbone_mpx": total_backbone_px / 1e6,
        "decoder_kpos": total_decoder_pos / 1e3,
    }


# ═══════════════════════════════════════════════════════════════════
# 评估 | Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_baseline(
    decoder: AdaptiveSparseDecoder,
    backbone: FastSAMBackbone,
    val_ds: SeverstalDataset,
    device: torch.device,
    class_ids: list[int],
    support_cache: dict,
    binary: bool = False,
    max_samples: int = 0,
) -> dict:
    """
    Baseline evaluation: full image → backbone → per-class prototype → per-class prediction.
    Same as evaluate_with_prototypes() in the baseline script.
    """
    decoder.eval()
    backbone.eval()

    # Pre-compute prototypes
    prototypes: dict[int, torch.Tensor] = {}
    for cls_id in class_ids:
        sc = support_cache[cls_id]
        proto = compute_prototype_full_image(
            backbone,
            sc["images"].to(device) if isinstance(sc["images"], torch.Tensor)
                else sc["images"].clone().to(device),
            sc["masks"].to(device) if isinstance(sc["masks"], torch.Tensor)
                else sc["masks"].clone().to(device),
            device,
        )
        prototypes[cls_id] = proto

    per_class_inter = {c: 0.0 for c in [0] + class_ids}
    per_class_union = {c: 0.0 for c in [0] + class_ids}

    indices = list(range(len(val_ds)))
    if max_samples > 0:
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Eval-Base", leave=False):
        sample = val_ds[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt_full = sample["masks"].squeeze(0).to(device)
        H, W = gt_full.shape

        feats = backbone(img, extract_proto=True)
        p4 = feats["p4"]
        proto_masks = feats["proto"]

        pred_class = torch.zeros(H, W, dtype=torch.long, device=device)
        for cls_id in class_ids:
            proto = prototypes[cls_id]
            pred_prob = decoder(p4, proto_masks, proto)
            pred_full = F.interpolate(
                pred_prob.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)
            pred_binary = (pred_full > 0.5).long()
            pred_class[(pred_binary > 0) & (pred_class == 0)] = cls_id

        for c in [0] + class_ids:
            pc = (pred_class == c)
            gc = _get_gt_binary(gt_full, c, binary)
            inter = (pc & gc).sum().item()
            union = (pc | gc).sum().item()
            per_class_inter[c] += inter
            per_class_union[c] += union

    class_names = {0: "background", 1: "Class1", 2: "Class2", 3: "Class3", 4: "Class4"}
    per_class_iou = {}
    for c in [0] + class_ids:
        name = class_names.get(c, f"class_{c}")
        per_class_iou[name] = (
            round(per_class_inter[c] / per_class_union[c], 6)
            if per_class_union[c] > 0 else float("nan")
        )

    valid = [v for v in per_class_iou.values() if v == v]
    return {
        "mIoU": round(float(np.mean(valid)), 6) if valid else 0.0,
        "per_class_IoU": per_class_iou,
    }


@torch.no_grad()
def evaluate_oracle_tile(
    decoder: AdaptiveSparseDecoder,
    backbone: FastSAMBackbone,
    val_ds: SeverstalDataset,
    device: torch.device,
    class_ids: list[int],
    support_cache: dict,
    tile_w: int,
    top_k: int,
    binary: bool = False,
    max_samples: int = 0,
) -> dict:
    """
    Oracle Tile evaluation:
    每张 val 图切 Tile → Oracle 选 Top-K → per-tile prediction → stitch.
    Prototype 也只用缺陷 Tile.
    """
    decoder.eval()
    backbone.eval()

    # Pre-compute prototypes (Oracle Tile version)
    prototypes: dict[int, torch.Tensor] = {}
    for cls_id in class_ids:
        sc = support_cache[cls_id]
        proto = compute_prototype_oracle_tile(
            backbone,
            sc["images"].to(device) if isinstance(sc["images"], torch.Tensor)
                else sc["images"].clone().to(device),
            sc["masks"].to(device) if isinstance(sc["masks"], torch.Tensor)
                else sc["masks"].clone().to(device),
            cls_id, device,
            tile_w=tile_w, top_k_tiles=top_k,
        )
        prototypes[cls_id] = proto

    per_class_inter = {c: 0.0 for c in [0] + class_ids}
    per_class_union = {c: 0.0 for c in [0] + class_ids}

    indices = list(range(len(val_ds)))
    if max_samples > 0:
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Eval-Tile", leave=False):
        sample = val_ds[idx]
        img = sample["image"]        # [3, H, W]
        gt_full = sample["masks"].squeeze(0)  # [H, W] category labels
        H, W = gt_full.shape

        # Cut into tiles
        tiles = cut_into_tiles(img, gt_full, tile_w=tile_w)

        # Per-class prediction (stitched from tiles)
        pred_class = torch.zeros(H, W, dtype=torch.long, device=device)

        for cls_id in class_ids:
            proto = prototypes[cls_id]
            selected = oracle_select_tiles(tiles, cls_id, top_k=top_k)

            for t in selected:
                tile_img = t["image"].unsqueeze(0).to(device)
                x0 = t["x_offset"]
                orig_w = t["orig_w"]

                feats = backbone(tile_img, extract_proto=True)
                p4 = feats["p4"]
                proto_masks = feats["proto"]

                pred_prob = decoder(p4, proto_masks, proto)  # [1, 64, W_pad/4]
                H_tile, W_tile_pad = tile_img.shape[2], tile_img.shape[3]
                pred_full = F.interpolate(
                    pred_prob.unsqueeze(0), size=(H_tile, W_tile_pad),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)  # [256, W_tile_pad]

                # Crop to original width and stitch
                pred_crop = pred_full[:, :orig_w]  # [256, orig_w]
                pred_binary = (pred_crop > 0.5).long()

                # Write to full prediction map (不覆盖已有预测 | Don't overwrite)
                roi = pred_class[:, x0:x0 + orig_w]
                mask = (pred_binary > 0) & (roi == 0)
                roi[mask] = cls_id
                pred_class[:, x0:x0 + orig_w] = roi

        # IoU
        for c in [0] + class_ids:
            pc = (pred_class == c)
            gc = _get_gt_binary(gt_full, c, binary)
            inter = (pc & gc).sum().item()
            union = (pc | gc).sum().item()
            per_class_inter[c] += inter
            per_class_union[c] += union

    class_names = {0: "background", 1: "Class1", 2: "Class2", 3: "Class3", 4: "Class4"}
    per_class_iou = {}
    for c in [0] + class_ids:
        name = class_names.get(c, f"class_{c}")
        per_class_iou[name] = (
            round(per_class_inter[c] / per_class_union[c], 6)
            if per_class_union[c] > 0 else float("nan")
        )

    valid = [v for v in per_class_iou.values() if v == v]
    return {
        "mIoU": round(float(np.mean(valid)), 6) if valid else 0.0,
        "per_class_IoU": per_class_iou,
    }


# ═══════════════════════════════════════════════════════════════════
# 单次 (mode, K, seed) 训练 | Single (mode, K, seed) Training Run
# ═══════════════════════════════════════════════════════════════════

def train_one_run(
    train_ds, val_ds, backbone, device, args, logger, run_dir,
    k: int, seed: int, mode: str,
) -> dict:
    """
    对一个 (mode, K, seed) 组合执行 episodic 训练.
    """
    set_seed(seed)

    class_ids = [1] if args.binary else DEFECT_CLASSES

    # Episode sampler
    sampler = EpisodeSampler(train_ds, class_ids=class_ids, k_shot=k, seed=seed)

    logger.log_info("fewshot/run",
                    f"Mode={mode} K={k} S={seed}: {args.episodes_per_epoch} episodes/epoch")

    # Model (fresh init per run)
    decoder = AdaptiveSparseDecoder(
        in_channels=1280, proto_dim=32, hidden_dim=256,
        use_fdr=False, normalize_proto="none", out_channels=1,
    ).to(device)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.episodes_per_epoch)

    # Fixed support cache for eval
    eval_support_cache: dict[int, dict] = {}
    eval_rng = random.Random(seed + 10000)
    for cls_id in class_ids:
        pool = train_ds.class_to_images(cls_id)
        picked = eval_rng.sample(pool, min(k, len(pool)))
        s_imgs, s_masks = [], []
        for idx in picked:
            s = train_ds[idx]
            s_imgs.append(s["image"])
            # For eval support cache, store full category mask for Oracle Tile mode
            s_masks.append(s["masks"].squeeze(0))
        eval_support_cache[cls_id] = {
            "images": torch.stack(s_imgs),
            "masks": torch.stack(s_masks),
        }

    augment = None if args.no_augment else BasicAugmentation()

    # Select train/eval functions based on mode
    if mode == "baseline":
        train_epoch_fn = _train_epoch_baseline
        eval_fn = evaluate_baseline
    else:  # oracle_tile
        train_epoch_fn = _train_epoch_oracle_tile
        eval_fn = evaluate_oracle_tile

    # Training loop
    best_miou = 0.0
    best_epoch = 0
    best_per_class = {}
    total_nan = 0
    total_skip = 0
    total_backbone_mpx = 0.0
    total_decoder_kpos = 0.0

    for epoch in range(1, args.epochs + 1):
        if mode == "oracle_tile":
            epoch_info = _train_epoch_oracle_tile(
                decoder, backbone, sampler, train_ds, device,
                optimizer, scheduler, epoch, args, augment, logger,
            )
            total_skip += epoch_info.get("skip_count", 0)
        else:
            epoch_info = _train_epoch_baseline(
                decoder, backbone, sampler, train_ds, device,
                optimizer, scheduler, epoch, args, augment, logger,
            )

        total_nan += epoch_info["nan_count"]
        total_backbone_mpx += epoch_info["backbone_mpx"]
        total_decoder_kpos += epoch_info["decoder_kpos"]

        # Evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            if mode == "oracle_tile":
                result = eval_fn(
                    decoder, backbone, val_ds, device,
                    class_ids=class_ids, support_cache=eval_support_cache,
                    tile_w=args.tile_width, top_k=args.top_k_tiles,
                    binary=args.binary,
                )
            else:
                result = eval_fn(
                    decoder, backbone, val_ds, device,
                    class_ids=class_ids, support_cache=eval_support_cache,
                    binary=args.binary,
                )

            miou = result["mIoU"]
            logger.log_info("fewshot/eval",
                            f"Mode={mode} K={k} S={seed} E{epoch:3d}: mIoU={miou:.4f}")

            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                best_per_class = result["per_class_IoU"]
                torch.save({
                    "epoch": epoch, "mode": mode,
                    "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                    "mIoU": miou, "per_class_IoU": best_per_class,
                    "k": k, "seed": seed,
                }, str(run_dir / "best_model.pt"))

    # FLOPs estimation
    if mode == "oracle_tile":
        flops = estimate_flops_ratio(args.tile_width, args.top_k_tiles,
                                     n_tiles_per_image=IMG_W // args.tile_width)
    else:
        flops = {"total_ratio": 1.0, "savings_pct": 0.0}

    logger.log_info("fewshot/run_done",
                    f"Mode={mode} K={k} S={seed}: best mIoU={best_miou:.4f} "
                    f"@ E{best_epoch}, NaN={total_nan}, Skip={total_skip}")

    return {
        "mode": mode, "k": k, "seed": seed,
        "best_mIoU": best_miou, "best_epoch": best_epoch,
        "per_class_IoU": best_per_class,
        "nan_count": total_nan,
        "skip_count": total_skip,
        "flops_ratio": flops["total_ratio"],
        "flops_savings_pct": flops["savings_pct"],
        "backbone_mpx": round(total_backbone_mpx, 1),
        "decoder_kpos": round(total_decoder_kpos, 1),
    }


# ═══════════════════════════════════════════════════════════════════
# 命令行参数 | CLI Arguments
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Oracle Tile + Few-Shot — Severstal Steel Defect Detection"
    )
    p.add_argument("--data-root", type=str, default="data/severstal-steel-defect-detection")
    p.add_argument("--binary", action="store_true")
    p.add_argument("--no-augment", action="store_true")

    # Mode
    p.add_argument("--mode", type=str, default="both",
                   choices=["baseline", "oracle_tile", "both"],
                   help="baseline=全图, oracle_tile=GT Tile选择, both=两者对比 (default: both)")

    # Tile
    p.add_argument("--tile-width", type=int, default=100,
                   help="Tile 宽度 (default: 100 → 16 tiles/image)")
    p.add_argument("--top-k-tiles", type=int, default=2,
                   help="Oracle 每张图保留的 Tile 数 (default: 2 → ~12.5 pct)")

    # Few-shot
    p.add_argument("--k-shot", type=int, nargs="+", default=[1, 5],
                   help="每类 support 图像数 (default: 1 5)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--episodes-per-epoch", type=int, default=200)

    # Training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--eval-every", type=int, default=5)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device(args.device)

    class_ids = [1] if args.binary else DEFECT_CLASSES
    class_names = {0: "BG", 1: "FG"} if args.binary \
        else {0: "BG", 1: "C1", 2: "C2", 3: "C3", 4: "C4"}

    # Output dir
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"runs/severstal_oracle_tile_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = get_logger("train_severstal_oracle")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))

    # FLOPs estimate
    n_tiles = IMG_W // args.tile_width
    flops = estimate_flops_ratio(args.tile_width, args.top_k_tiles, n_tiles)
    logger.log_info("config", f"Oracle Tile Config: W={args.tile_width} "
                    f"({n_tiles} tiles/img), Top-K={args.top_k_tiles}")
    logger.log_info("config", f"FLOPs estimate: backbone {flops['backbone_ratio']:.1%}, "
                    f"decoder {flops['decoder_ratio']:.1%}, "
                    f"total {flops['total_ratio']:.1%} "
                    f"→ save ~{flops['savings_pct']:.0f}%")
    logger.log_info("config",
                    f"Mode={args.mode} | K={args.k_shot} | seeds={args.seeds} | "
                    f"Epochs={args.epochs}")

    # Data
    train_ds = SeverstalDataset(root=args.data_root, split="train",
                                binary=args.binary, seed=42)
    val_ds = SeverstalDataset(root=args.data_root, split="val",
                              binary=args.binary, seed=42)
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    for cls_id in class_ids:
        logger.log_info("data",
                        f"  Class {cls_id}: {len(train_ds.class_to_images(cls_id))} train images")

    # Backbone (frozen, shared)
    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint="thirdLibrary/FastSAM/weights/FastSAM-x.pt",
    ).to(device)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, IMG_H, IMG_W, device=device), extract_proto=True)
    logger.log_info("model", "Backbone: FastSAM-x (frozen, shared)")

    # Determine modes to run
    modes = ["baseline", "oracle_tile"] if args.mode == "both" else [args.mode]

    # Sweep
    all_results = []
    runs = [(mode, k, s) for mode in modes for k in args.k_shot for s in args.seeds]
    logger.log_info("sweep", f"Starting sweep: {len(runs)} runs "
                    f"({len(modes)} modes × {len(args.k_shot)} K × {len(args.seeds)} seeds)")

    for run_idx, (mode, k, seed) in enumerate(runs):
        logger.log_info("sweep", f"[{run_idx+1}/{len(runs)}] Mode={mode}, K={k}, seed={seed}")

        run_dir = out_dir / f"{mode}_K{k}_S{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = train_one_run(
                train_ds=train_ds, val_ds=val_ds,
                backbone=backbone, device=device,
                args=args, logger=logger, run_dir=run_dir,
                k=k, seed=seed, mode=mode,
            )
            all_results.append(result)
        except ValueError as e:
            logger.log_info("sweep", f"  SKIP: {e}")
            continue

    if not all_results:
        logger.log_info("done", "No results.")
        return

    # ── 汇总 | Summary ──
    _print_comparison_table(all_results, class_names, args, flops, logger)

    # results.json
    results_json = {
        "experiment": "Severstal Oracle Tile + Few-Shot",
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "flops_estimate": flops,
        "runs": all_results,
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False, default=str)
    logger.log_info("done", f"Results → {json_path}")
    print(f"\nResults: {json_path}")


def _print_comparison_table(
    all_results: list, class_names: dict, args, flops: dict, logger,
):
    """打印 baseline vs oracle_tile 对比表."""
    # Group by mode → K
    from collections import defaultdict
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["mode"], r["k"])].append(r)

    print(f"\n{'='*95}")
    print(f"  Severstal Oracle Tile vs Baseline — Episodic Few-Shot")
    print(f"  Tile: W={args.tile_width} ({IMG_W//args.tile_width}/img), "
          f"Top-K={args.top_k_tiles}, FLOPs save ~{flops['savings_pct']:.0f}%")
    print(f"{'='*95}")

    # Table header
    col_names = [class_names.get(c, f"c{c}") for c in [0] + ([1] if args.binary else DEFECT_CLASSES)]
    header = (f"{'Mode':>12} {'K':>3} | {'mIoU':>7} ± {'std':>6} | "
              + " | ".join(f"{n:>7}" for n in col_names)
              + f" | {'FLOPs':>6} | {'Ep':>5}")
    print(header)
    print("-" * len(header.expandtabs()))

    for (mode, k), results in sorted(groups.items()):
        mious = [r["best_mIoU"] for r in results]
        mean_miou = np.mean(mious)
        std_miou = np.std(mious) if len(mious) > 1 else 0.0

        # Average per-class IoU across seeds
        avg_per_class = {}
        pc_keys = [class_names.get(c, f"c{c}") for c in [0] + ([1] if args.binary else DEFECT_CLASSES)]
        for key in pc_keys:
            vals = [r["per_class_IoU"].get(key, float("nan")) for r in results]
            valid_vals = [v for v in vals if v == v]
            avg_per_class[key] = np.mean(valid_vals) if valid_vals else float("nan")

        flops_val = results[0].get("flops_ratio", 1.0)
        flops_str = f"{flops_val*100:.0f}%" if flops_val < 1.0 else "100%"
        best_ep = min(r["best_epoch"] for r in results)

        vals_str = " | ".join(f"{avg_per_class.get(k, 0):7.4f}" for k in pc_keys)
        mode_tag = f"[{mode}]"
        print(f"{mode_tag:>12} {k:3d} | {mean_miou:7.4f} ± {std_miou:.4f} | "
              f"{vals_str} | {flops_str:>6} | {best_ep:5d}")

    print("-" * len(header.expandtabs()))

    # Comparison summary
    baseline_results = [r for r in all_results if r["mode"] == "baseline"]
    oracle_results = [r for r in all_results if r["mode"] == "oracle_tile"]

    if baseline_results and oracle_results:
        print(f"\n  ── Comparison Summary ──")
        for k in sorted(set(r["k"] for r in all_results)):
            b_runs = [r for r in baseline_results if r["k"] == k]
            o_runs = [r for r in oracle_results if r["k"] == k]
            if not b_runs or not o_runs:
                continue
            b_miou = np.mean([r["best_mIoU"] for r in b_runs])
            o_miou = np.mean([r["best_mIoU"] for r in o_runs])
            delta = o_miou - b_miou
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            print(f"  K={k:2d}: Baseline mIoU={b_miou:.4f} → Oracle Tile mIoU={o_miou:.4f} "
                  f"(Δ={delta_str}) | FLOPs={flops['savings_pct']:.0f}% saved")

    print(f"\n{'='*95}\n")


if __name__ == "__main__":
    main()
