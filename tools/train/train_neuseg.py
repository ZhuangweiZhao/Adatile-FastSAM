#!/usr/bin/env python3
"""
Neu_seg 多类别分割训练脚本 | Neu_seg Multi-class Segmentation Training Script.
==============================================================================

4 类分割训练 (BG + Inclusion + Patch + Scratch)。
4-class segmentation training (BG + Inclusion + Patch + Scratch).

训练策略 | Training Strategy:
    每步: 随机选 K support tiles + 1 query tile
    → Support prototype (FG pooling) → Decoder → Mask → Loss
    Per step: random K support + 1 query tile → prototype → decoder → mask → loss.

支持 decoder | Supported decoders:
    pure / pure_p3p4 / adaptive
    (ProtoOnly 系列不支持多类别, 架构限制)

用法 | Usage::

    # 基础训练
    python tools/train/train_neuseg.py --decoder-type pure_p3p4 --epochs 50

    # 带数据增强 + 类别权重
    python tools/train/train_neuseg.py --decoder-type adaptive --augment --class-weights balanced

    # 快速验证
    python tools/train/train_neuseg.py --epochs 10 --steps-per-epoch 100 --device cpu
"""

from __future__ import annotations

import sys, argparse, json, random
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone import FastSAMBackbone
from adatile.decoder.adaptive_sparse_decoder import (
    AdaptiveSparseDecoder, ProtoOnlyDecoder, ProtoOnlyDecoderP3P4,
)
from adatile.decoder.pure_cnn_decoder import PureDecoder, PureDecoderP3P4, PureDecoderP2P3P4
from adatile.adapter import MultiScaleAdapter
from adatile.frequency import (
    MultiScaleSpectralAttention,
    FrequencyGuidedFusion,
    spectral_combined_loss,
)
from adatile.datasets.neu_seg import NEUSegDataset


# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

DEFAULT_DATA_ROOT = "data/NEU_Seg"
NUM_CLASSES = 4  # BG + Inclusion + Patch + Scratch
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]

# 支持多类别的 decoder | Multi-class capable decoders
SUPPORTED_DECODERS = {"adaptive", "pure", "pure_p3p4"}


# ═══════════════════════════════════════════════════════════════════
# 工具函数 | Utility Functions
# ═══════════════════════════════════════════════════════════════════

def pad_to_32(image: torch.Tensor, mask: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, int]]:
    """
    将图像 pad 到 32 的倍数 (FastSAM backbone 要求) | Pad image to multiple of 32.

    :param image: [C, H, W] or [B, C, H, W].
    :param mask: [1, H, W] or [B, 1, H, W] or None.
    :return: (padded_image, padded_mask, (orig_H, orig_W)).
    """
    if image.dim() == 4:
        H, W = image.shape[2], image.shape[3]
    else:
        H, W = image.shape[1], image.shape[2]

    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32

    if pad_h == 0 and pad_w == 0:
        return image, mask, (H, W)

    pad_dims = (0, pad_w, 0, pad_h)  # for [..., H, W]
    image_padded = F.pad(image, pad_dims, mode='constant', value=0)
    mask_padded = F.pad(mask, pad_dims, mode='constant', value=0) if mask is not None else None

    return image_padded, mask_padded, (H, W)


# ═══════════════════════════════════════════════════════════════════
# Support Prototype 计算 | Support Prototype Computation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_support_prototype(
    backbone: FastSAMBackbone,
    support_images: torch.Tensor,       # [K, 3, H, W]
    support_masks: torch.Tensor,        # [K, 1, H, W] binary masks (FG>0.5 → 1)
    device: torch.device,
    proto_source: str = "p4",          # "p3" or "p4"
) -> torch.Tensor:
    """
    从 K 张 support 图像计算 FG prototype | Compute FG prototype from K support images.

    :param backbone: FastSAM backbone (eval mode, frozen).
    :param support_images: [K, 3, H, W] support images in [0, 1].
    :param support_masks: [K, 1, H, W] binary GT masks (FG > 0.5).
    :param device: 计算设备 | Compute device.
    :param proto_source: 特征来源 "p3"(960-dim, H/8) or "p4"(1280-dim, H/16).
    :return: [feat_dim] L2-normalized prototype vector.
    """
    K = support_images.shape[0]
    prototypes = []
    feat_dim = 960 if proto_source == "p3" else 1280

    for i in range(K):
        img = support_images[i:i + 1].to(device)
        mask = support_masks[i:i + 1].to(device)

        img, mask, _ = pad_to_32(img, mask)

        feats = backbone(img)
        feat = feats[proto_source]  # [1, feat_dim, H/s, W/s]

        mask_ds = F.interpolate(
            mask.float(), size=feat.shape[2:], mode="nearest"
        ).squeeze(1)  # [1, H/s, W/s]

        fg = mask_ds > 0.5
        if fg.sum() > 0:
            proto = feat[:, :, fg.squeeze(0)].mean(dim=-1).squeeze(0)
            prototypes.append(proto)

    if not prototypes:
        return torch.zeros(feat_dim, device=device)

    proto = torch.stack(prototypes).mean(dim=0)
    return F.normalize(proto, dim=0)


def compute_support_prototype_dual(
    backbone: FastSAMBackbone,
    support_images: torch.Tensor,
    support_masks: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 P3 和 P4 两个层级计算 FG prototype | Compute FG prototypes from both P3 and P4.

    :return: (p3_proto [960], p4_proto [1280]) L2-normalized.
    """
    p3 = compute_support_prototype(
        backbone, support_images, support_masks, device, proto_source="p3"
    )
    p4 = compute_support_prototype(
        backbone, support_images, support_masks, device, proto_source="p4"
    )
    return p3, p4


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════

def multiclass_dice_loss(
    pred: torch.Tensor,          # [B, C, H, W] softmax probs
    target: torch.Tensor,        # [B, H, W] int64 class labels
    smooth: float = 1e-6,
    ignore_bg: bool = True,
) -> torch.Tensor:
    """
    多类别 Dice Loss | Multi-class Dice Loss.

    对每个前景类别计算二值 Dice，取平均。
    Compute binary Dice per foreground class, average.

    :param pred: [B, C, H, W] softmax probabilities.
    :param target: [B, H, W] integer class labels.
    :param smooth: smoothing term.
    :param ignore_bg: 是否忽略背景类 | Whether to ignore background class.
    :return: 1 - mean(Dice over FG classes).
    """
    C = pred.shape[1]
    start_class = 1 if ignore_bg else 0
    dice_sum = 0.0
    count = 0
    for c in range(start_class, C):
        pred_c = pred[:, c, :, :]
        target_c = (target == c).float()
        if target_c.sum() > 0:
            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_sum += (2.0 * inter + smooth) / (union + smooth)
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return 1.0 - dice_sum / count


def multiclass_combined_loss(
    pred: torch.Tensor,          # [B, C, H, W] softmax probs
    target: torch.Tensor,        # [B, H, W] int64 class labels
    ce_weight: torch.Tensor | None = None,  # [C] class weights for CE
    ce_alpha: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    多类别组合损失: ce_alpha * CE + (1 - ce_alpha) * multi-Dice.
    Multi-class combined loss: ce_alpha * CE + (1 - ce_alpha) * multi-Dice.

    :param pred: [B, C, H, W] softmax probabilities.
    :param target: [B, H, W] int64 class labels.
    :param ce_weight: [C] per-class weights for cross-entropy.
    :param ce_alpha: CE vs Dice weight.
    :return: (total_loss, {"ce": float, "dice": float}).
    """
    log_pred = torch.log(pred + 1e-7)
    ce = F.nll_loss(log_pred, target, weight=ce_weight)
    md = multiclass_dice_loss(pred, target)
    return ce_alpha * ce + (1 - ce_alpha) * md, {"ce": ce.item(), "dice": md.item()}


# ═══════════════════════════════════════════════════════════════════
# Boundary-aware Loss (Lovász-Softmax) | 边界感知损失
# ═══════════════════════════════════════════════════════════════════

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Lovász 扩展的梯度 | Gradient of the Lovász extension."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if jaccard.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def lovasz_softmax(
    prob: torch.Tensor,       # [B, C, H, W] softmax probs
    label: torch.Tensor,      # [B, H, W] int64 class labels
    classes: str = "present",  # "present" or "all"
    ignore: int | None = None,
) -> torch.Tensor:
    """
    多类别 Lovász-Softmax 损失 | Multi-class Lovász-Softmax loss.

    Lovász-Softmax 是 IoU (Jaccard index) 的凸代理损失, 直接优化 mIoU。
    对边界敏感, 适合薄长缺陷 (Scratch/Inclusion)。
    Convex surrogate for the Jaccard index — directly optimizes mIoU.
    Boundary-sensitive, suitable for thin/long defects.

    :param prob: [B, C, H, W] softmax probabilities.
    :param label: [B, H, W] integer ground truth labels.
    :param classes: "present" → only classes in batch; "all" → all C classes.
    :param ignore: class index to ignore (e.g. background).
    :return: scalar Lovász-Softmax loss.
    """
    B, C, H, W = prob.shape
    loss = torch.tensor(0.0, device=prob.device)

    # 展平空间维度 | Flatten spatial dims
    prob = prob.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
    label = label.view(-1)                                      # [B*H*W]

    if classes == "present":
        active = label.unique().long().tolist()
    else:
        active = list(range(C))

    for c in active:
        if c == ignore:
            continue
        # 二值化该类别 | Binarize for this class
        fg = (label == c).float()
        if fg.sum() == 0:
            continue
        # 该类别的 softmax 概率 | Softmax prob for this class
        prob_c = prob[:, c]
        # 按错误排序 (高置信度前景 → 先考虑) | Sort by errors (high conf FG first)
        errors = (fg - prob_c).abs()
        prob_c_sorted, idx = prob_c.sort(descending=True)
        fg_sorted = fg[idx]
        errors_sorted = errors[idx]
        # Lovász hinge
        grad = _lovasz_grad(fg_sorted)
        loss += (grad * (1.0 - prob_c_sorted)).sum()

    return loss / B


def boundary_weight_map(label: torch.Tensor, sigma: float = 3.0) -> torch.Tensor:
    """
    生成边界权重图 (高斯距离加权) | Generate boundary weight map (Gaussian distance).

    标签边界附近像素获得更高权重, 强调边缘区域的学习。
    Pixels near label boundaries get higher weight, emphasizing edge regions.

    :param label: [B, H, W] int64 class labels.
    :param sigma: 高斯衰减的 sigma | Gaussian decay sigma (pixels).
    :return: [B, H, W] normalized weight map (mean ≈ 1.0).
    """
    B, H, W = label.shape
    weight = torch.ones(B, H, W, device=label.device)

    for b in range(B):
        lbl = label[b]  # [H, W]
        # 检测每个类别的边界 | Detect boundaries for each class
        boundary = torch.zeros(H, W, device=label.device)
        classes = lbl.unique().tolist()
        for c in classes:
            if c == 0:  # skip background
                continue
            c_mask = (lbl == c).float()
            # Sobel 梯度检测边界 | Sobel gradient for boundary detection
            # 使用简单的 Laplacian | Simple Laplacian
            padded = F.pad(c_mask.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
            laplacian = torch.abs(
                padded[:, :, 2:, 1:-1].squeeze() + padded[:, :, :-2, 1:-1].squeeze() +
                padded[:, :, 1:-1, 2:].squeeze() + padded[:, :, 1:-1, :-2].squeeze() -
                4 * padded[:, :, 1:-1, 1:-1].squeeze()
            )
            boundary = torch.maximum(boundary, (laplacian > 0).float())

        # 高斯距离加权 | Gaussian distance weighting
        if boundary.sum() > 0:
            # 近似: 使用边界膨胀 + 高斯核 | Approx: boundary dilation + Gaussian kernel
            kernel_size = int(sigma * 3) | 1  # ensure odd
            kernel = torch.ones(1, 1, kernel_size, kernel_size, device=label.device)
            boundary_expanded = F.conv2d(
                boundary.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2
            ).squeeze()
            dist_weight = 1.0 + torch.exp(-boundary_expanded / (2 * sigma ** 2))
            weight[b] = dist_weight

    # 归一化使 mean=1 | Normalize to mean=1
    weight = weight / weight.mean()
    return weight


def lovasz_combined_loss(
    pred: torch.Tensor,          # [B, C, H, W] softmax probs
    target: torch.Tensor,        # [B, H, W] int64 class labels
    ce_weight: torch.Tensor | None = None,
    ce_alpha: float = 0.3,      # CE weight
    dice_alpha: float = 0.3,    # Dice weight
    lovasz_alpha: float = 0.4,  # Lovász weight
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    组合损失 (Focal CE + Dice + Lovász) | Combined loss (Focal CE + Dice + Lovász).

    CE 保证像素级正确, Dice 处理类别不平衡, Lovász 优化边界 mIoU。
    CE ensures per-pixel correctness, Dice handles class imbalance,
    Lovász optimizes boundary mIoU.

    :param pred: [B, C, H, W] softmax probabilities.
    :param target: [B, H, W] int64 class labels.
    :param ce_weight: [C] per-class weights for CE.
    :param ce_alpha: CE loss weight.
    :param dice_alpha: Dice loss weight.
    :param lovasz_alpha: Lovász-Softmax loss weight.
    :return: (total_loss, {"ce": float, "dice": float, "lovasz": float}).
    """
    log_pred = torch.log(pred + 1e-7)
    ce = F.nll_loss(log_pred, target, weight=ce_weight)
    dice = multiclass_dice_loss(pred, target)
    lovasz = lovasz_softmax(pred, target, classes="present", ignore=None)
    total = ce_alpha * ce + dice_alpha * dice + lovasz_alpha * lovasz
    return total, {"ce": ce.item(), "dice": dice.item(), "lovasz": lovasz.item()}


# ═══════════════════════════════════════════════════════════════════
# 数据增强 (基于分析结果) | Data Augmentation (based on analysis)
# ═══════════════════════════════════════════════════════════════════

class NEUSegAugmentation:
    """
    基于数据集分析的增强策略 | Augmentation strategy based on dataset analysis.

    分析发现 | Analysis Findings:
        - Train/Test brightness shift: KS p=0.003 → RandomBrightnessContrast
        - Train/Test blur shift: KS p<0.001 (50.8% blurry) → GaussianNoise
        - 52.9% overexposed → RandomGamma
        - Objects have no canonical orientation → RandomFlip + RandomRotate90
    """

    def __init__(self, p_flip: float = 0.5, p_rotate: float = 0.5,
                 brightness_range: float = 0.2, contrast_range: float = 0.2,
                 noise_std: float = 0.02, p_color: float = 0.7):
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.p_color = p_color

    def __call__(self, image: torch.Tensor, mask: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对 image [C,H,W] 和 mask [1,H,W] 应用相同空间变换，独立颜色变换。
        Apply same spatial transforms to image+mask, independent color transforms.

        :param image: [3, H, W] float32 in [0, 1].
        :param mask: [1, H, W] int64.
        :return: augmented (image, mask).
        """
        # ── 空间变换 (同步) | Spatial transforms (synced) ──
        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        if torch.rand(1).item() < self.p_flip:
            image = torch.flip(image, dims=[-2])
            mask = torch.flip(mask, dims=[-2])

        if torch.rand(1).item() < self.p_rotate:
            k = torch.randint(0, 4, (1,)).item()
            image = torch.rot90(image, k, dims=[-2, -1])
            mask = torch.rot90(mask, k, dims=[-2, -1])

        # ── 颜色变换 (仅图像) | Color transforms (image only) ──
        if torch.rand(1).item() < self.p_color:
            brightness = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness_range
            image = torch.clamp(image * brightness, 0.0, 1.0)

            contrast = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast_range
            mean_val = image.mean(dim=(-2, -1), keepdim=True)
            image = torch.clamp((image - mean_val) * contrast + mean_val, 0.0, 1.0)

        # ── 高斯噪声 (模拟模糊鲁棒性) | Gaussian noise (blur robustness) ──
        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(image) * self.noise_std
            image = torch.clamp(image + noise, 0.0, 1.0)

        return image, mask


# ═══════════════════════════════════════════════════════════════════
# Decoder 前向传播 | Decoder Forward
# ═══════════════════════════════════════════════════════════════════

def _decoder_forward(decoder, feats, support_cache, freq_fusion=None):
    """
    统一的 decoder 前向传播 | Unified decoder forward.

    :param freq_fusion: 可选 FrequencyGuidedFusion 模块.
    :return: prediction tensor or None if proto unavailable.
    """
    if isinstance(decoder, PureDecoderP2P3P4):
        p2 = feats.get("p2")
        if p2 is None:
            return None
        return decoder(p2, feats["p3"], feats["p4"], freq_fusion=freq_fusion)
    elif isinstance(decoder, PureDecoderP3P4):
        return decoder(feats["p3"], feats["p4"])
    elif isinstance(decoder, PureDecoder):
        return decoder(feats["p4"])
    elif isinstance(decoder, ProtoOnlyDecoderP3P4):
        proto_masks = feats.get("proto")
        if proto_masks is None:
            return None
        support_p3, support_p4 = support_cache
        return decoder(proto_masks, support_p3, support_p4)
    elif isinstance(decoder, ProtoOnlyDecoder):
        proto_masks = feats.get("proto")
        if proto_masks is None:
            return None
        return decoder(proto_masks, support_cache)
    else:
        # AdaptiveSparseDecoder
        p4 = feats["p4"]
        proto_masks = feats.get("proto")
        if proto_masks is None:
            return None
        return decoder(p4, proto_masks, support_cache)


# ═══════════════════════════════════════════════════════════════════
# 评估 | Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    decoder: nn.Module,
    backbone: FastSAMBackbone,
    support_cache,
    dataset: NEUSegDataset,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    max_samples: int = 0,
    adapter: nn.Module | None = None,
    spectral_attn: nn.Module | None = None,
    freq_fusion: nn.Module | None = None,
) -> dict:
    """
    多类别评估 — per-class mIoU + Dice | Multi-class evaluation — per-class mIoU + Dice.

    :param dataset: val/test split dataset.
    :param support_cache: prototype tensor or (p3_proto, p4_proto) tuple.
    :param num_classes: total classes (default 4).
    :param max_samples: 最多评估样本数 (0=全部) | Max eval samples (0=all).
    :return: dict with mIoU, per_class_IoU, per_class_Dice.
    """
    decoder.eval()

    per_class_inter = torch.zeros(num_classes, device=device)
    per_class_union = torch.zeros(num_classes, device=device)
    per_class_tp = torch.zeros(num_classes, device=device)
    per_class_fp = torch.zeros(num_classes, device=device)
    per_class_fn = torch.zeros(num_classes, device=device)
    per_sample = []

    indices = list(range(len(dataset)))
    if max_samples > 0:
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Eval", leave=False):
        sample = dataset[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt_mask = sample["masks"].squeeze(0).to(device)
        H, W = gt_mask.shape

        img, _, _ = pad_to_32(img)

        feats = backbone(img, extract_proto=True)

        # ── Adapter (CAT-SAM style) | 特征域适配 ──
        if adapter is not None:
            adapted = adapter(p3=feats.get("p3"), p4=feats.get("p4"), p8=feats.get("p8"))
            feats.update(adapted)

        # ── DCT Spectral Attention | 频域注意力 ──
        if spectral_attn is not None:
            spec_feats = spectral_attn(
                p2=feats.get("p2"), p3=feats.get("p3"), p4=feats.get("p4"),
            )
            feats.update(spec_feats)

        pred_prob = _decoder_forward(decoder, feats, support_cache,
                                     freq_fusion=freq_fusion)
        if pred_prob is None:
            continue

        # [C, H/4, W/4] → [C, H, W]
        pred_full = F.interpolate(
            pred_prob.unsqueeze(0),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(0)

        pred_class = torch.argmax(pred_full, dim=0)
        gt_class = gt_mask.long()

        sample_iou = 0.0
        n_classes_present = 0
        for c in range(num_classes):
            pred_c = (pred_class == c)
            gt_c = (gt_class == c)
            inter = (pred_c & gt_c).sum()
            union = (pred_c | gt_c).sum()
            per_class_inter[c] += inter
            per_class_union[c] += union
            if gt_c.sum() > 0:
                tp = inter
                fp = (pred_c & ~gt_c).sum()
                fn = (~pred_c & gt_c).sum()
                per_class_tp[c] += tp
                per_class_fp[c] += fp
                per_class_fn[c] += fn
                if union > 0:
                    sample_iou += (inter / union).item()
                    n_classes_present += 1

        sample_miou = sample_iou / max(n_classes_present, 1)
        per_sample.append({
            "image_id": sample["image_id"],
            "mIoU": round(sample_miou, 6),
            "n_classes_present": n_classes_present,
        })

    # ── Per-class IoU ──
    per_class_iou = {}
    for c in range(num_classes):
        inter = per_class_inter[c].item()
        union = per_class_union[c].item()
        iou_c = inter / union if union > 0 else float("nan")
        per_class_iou[CLASS_NAMES[c]] = round(iou_c, 6)

    valid_ious = [v for v in per_class_iou.values() if not (v != v)]  # NaN check
    miou = np.mean(valid_ious) if valid_ious else 0.0

    # ── Per-class Dice ──
    per_class_dice = {}
    for c in range(num_classes):
        tp = per_class_tp[c].item()
        fp = per_class_fp[c].item()
        fn = per_class_fn[c].item()
        dice_c = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        per_class_dice[CLASS_NAMES[c]] = round(dice_c, 6)

    return {
        "mIoU": round(float(miou), 6),
        "per_class_IoU": per_class_iou,
        "per_class_Dice": per_class_dice,
        "per_sample": per_sample,
        "n_evaluated": len(per_sample),
    }


# ═══════════════════════════════════════════════════════════════════
# 模型构建 | Model Construction
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def _probe_backbone_channels(backbone: FastSAMBackbone, device: torch.device) -> dict[str, int]:
    """
    触发 backbone 探测，返回所有层的通道数 | Trigger backbone probing, return all channel counts.

    送一张 dummy 224×224 图像走一遍 backbone，触发布局探测，
    然后从 backbone.channels 属性获取检测到的通道数。
    Feed a dummy 224×224 image through backbone to trigger stride probing,
    then read detected channel counts from backbone.channels property.
    """
    dummy = torch.randn(1, 3, 224, 224, device=device)
    backbone(dummy, extract_proto=False)
    return backbone.channels  # {"p2": int, "p3": int, "p4": int, "p8": int}


def build_decoder(decoder_type: str, proto_source: str, logger,
                  backbone: FastSAMBackbone | None = None,
                  device: torch.device = torch.device("cpu")) -> nn.Module:
    """
    根据参数构建多类别 decoder | Build multi-class decoder based on parameters.

    通道数自动从 backbone 探测获取 | Channel counts auto-detected from backbone.

    :param decoder_type: "adaptive", "pure", "pure_p3p4", "pure_p2p3p4".
    :param proto_source: "p3" or "p4" (for adaptive).
    :param logger: logger instance.
    :param backbone: FastSAMBackbone (for auto-detecting channel counts).
    :param device: torch device.
    :return: decoder module.
    """
    # 自动探测通道数 | Auto-detect channel counts
    ch = {"p2": 160, "p3": 960, "p4": 1280, "p8": 1280}  # 默认 FastSAM-x | default
    if backbone is not None:
        ch = _probe_backbone_channels(backbone, device)
        logger.log_info("model", f"Auto-detected channels: {ch}")

    out_channels = NUM_CLASSES

    if decoder_type == "pure":
        decoder = PureDecoder(in_channels=ch["p4"], out_channels=out_channels)
    elif decoder_type == "pure_p3p4":
        decoder = PureDecoderP3P4(p3_channels=ch["p3"], p4_channels=ch["p4"],
                                  out_channels=out_channels)
    elif decoder_type == "pure_p2p3p4":
        decoder = PureDecoderP2P3P4(
            p2_channels=ch["p2"], p3_channels=ch["p3"], p4_channels=ch["p4"],
            out_channels=out_channels, mid_channels=128,
        )
    else:  # adaptive
        decoder = AdaptiveSparseDecoder(
            in_channels=ch["p4"], proto_dim=32, use_fdr=False,
            out_channels=out_channels,
        )

    params = sum(p.numel() for p in decoder.parameters())
    logger.log_info("model",
        f"{decoder.__class__.__name__}: {params/1e3:.1f}K params, "
        f"out_channels={out_channels}")
    return decoder


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Neu_seg Multi-class Segmentation Training"
    )

    # ── 数据 | Data ──
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                   help="数据集根目录 | Data root")

    # ── 训练 | Training ──
    p.add_argument("--epochs", type=int, default=50,
                   help="训练轮数 | Training epochs")
    p.add_argument("--steps-per-epoch", type=int, default=200,
                   help="每轮训练步数 | Training steps per epoch")
    p.add_argument("--k-support", type=int, default=3,
                   help="每步 support tile 数 | Support tiles per step")
    p.add_argument("--batch-size", type=int, default=1,
                   help="批次大小 (当前仅支持 bs=1) | Batch size (only bs=1)")

    # ── 模型 | Model ──
    p.add_argument("--decoder-type", type=str, default="adaptive",
                   choices=["adaptive", "pure", "pure_p3p4", "pure_p2p3p4"],
                   help="Decoder 类型 | Decoder type")
    p.add_argument("--backbone", type=str, default="fastsam-x",
                   choices=["fastsam-x", "fastsam-s"],
                   help="Backbone 模型: fastsam-x (YOLOv8x/68M) / fastsam-s (YOLOv8s/~14M)")
    p.add_argument("--proto-source", type=str, default="p4",
                   choices=["p3", "p4"],
                   help="Support prototype 特征来源 (adaptive only)")
    p.add_argument("--freeze-backbone", action="store_true", default=True,
                   help="冻结 backbone | Freeze backbone")
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone",
                   action="store_false",
                   help="解冻 backbone | Unfreeze backbone")

    # ── CAT-SAM Adapter | 特征域适配器 ──
    p.add_argument("--use-adapter", action="store_true", default=False,
                   help="插入 MultiScaleAdapter (CAT-SAM 风格) 在 backbone 和 decoder 之间 | "
                        "Insert MultiScaleAdapter between backbone and decoder")
    p.add_argument("--use-spectral", action="store_true", default=False,
                   help="插入 MultiScaleSpectralAttention (DCT 频域注意力) | "
                        "Insert DCT spectral attention")
    p.add_argument("--use-freq-fusion", action="store_true", default=False,
                   help="使用 FrequencyGuidedFusion 替代 BiFPN 固定权重 (仅 pure_p2p3p4) | "
                        "Use frequency-guided fusion instead of BiFPN")

    # ── 数据增强 | Data Augmentation ──
    p.add_argument("--augment", action="store_true", default=False,
                   help="启用数据增强 (brightness/contrast/noise/flip)")

    # ── 类别权重 | Class Weights ──
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "balanced", "inverse"],
                   help="类别权重策略: 'none'=均匀, 'balanced'=1/freq, "
                        "'inverse'=中频加权 | Class weight strategy")
    p.add_argument("--loss-type", type=str, default="ce_dice",
                   choices=["ce_dice", "lovasz", "spectral"],
                   help="损失函数: 'ce_dice' (CE+Dice) / 'lovasz' (CE+Dice+Lovász) / 'spectral' (CE+Dice+Spectral)")

    # ── 优化器 | Optimizer ──
    p.add_argument("--lr", type=float, default=1e-4,
                   help="学习率 | Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="权重衰减 | Weight decay")

    # ── 硬件 | Hardware ──
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # ── 输出 | Output ──
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=5,
                   help="每 N epochs 评估一次 | Evaluate every N epochs")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ── 输出目录 | Output Directory ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        dec_short = {"adaptive": "Adapt", "pure": "Pure",
                     "pure_p3p4": "PureP3P4",
                     "pure_p2p3p4": "PureP2P3P4"}[args.decoder_type]
        adapter_suffix = "_Ada" if args.use_adapter else ""
        spectral_suffix = "_Spec" if args.use_spectral else ""
        freq_fusion_suffix = "_FreqFuse" if args.use_freq_fusion else ""
        loss_suffix = f"_{args.loss_type}" if args.loss_type != "ce_dice" else ""
        args.output_dir = f"runs/neuseg_{dec_short}{adapter_suffix}{spectral_suffix}{freq_fusion_suffix}{loss_suffix}_{args.backbone}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logger ──
    logger = get_logger("train_neuseg")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))

    logger.log_info("config", f"Multi-class mode, num_classes={NUM_CLASSES}")
    logger.log_info("config", f"Backbone: {args.backbone}")
    logger.log_info("config", f"Decoder: {args.decoder_type}, "
                    f"proto_source={args.proto_source}, augment={args.augment}")
    logger.log_info("config", f"Training: {args.epochs} epochs × "
                    f"{args.steps_per_epoch} steps, K={args.k_support}, "
                    f"lr={args.lr}, device={args.device}")

    # ── 数据集 (始终多类别) | Datasets (always multi-class) ──
    logger.log_info("data", "Loading NEU_Seg datasets (multi-class)...")
    train_ds = NEUSegDataset(root=args.data_root, split="train", binary=False)
    val_ds = NEUSegDataset(root=args.data_root, split="test", binary=False)
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── 类别权重 | Class Weights ──
    ce_weight = None
    if args.class_weights != "none":
        if hasattr(train_ds, 'get_class_stats'):
            stats = train_ds.get_class_stats()
            pixel_counts = [stats[cn]["pixels"] for cn in CLASS_NAMES]
            total = sum(pixel_counts)
            if args.class_weights == "balanced":
                raw_weights = [total / max(p, 1) for p in pixel_counts]
                mean_w = sum(raw_weights) / len(raw_weights)
                weights = [w / mean_w for w in raw_weights]
            elif args.class_weights == "inverse":
                freqs = [p / total for p in pixel_counts]
                raw_weights = [1.0 / max(f, 1e-6)**0.5 for f in freqs]
                mean_w = sum(raw_weights) / len(raw_weights)
                weights = [w / mean_w for w in raw_weights]
            ce_weight = torch.tensor(weights, dtype=torch.float32, device=device)
            logger.log_info("data", f"Class weights ({args.class_weights}): "
                            f"{dict(zip(CLASS_NAMES, weights))}")

    # ── 数据增强 | Data Augmentation ──
    augment = NEUSegAugmentation() if args.augment else None
    if augment:
        logger.log_info("data", "Augmentation enabled: flip, rotate, brightness, noise")

    # ── 模型 | Models ──
    logger.log_info("model", f"Building backbone ({args.backbone})...")
    backbone = FastSAMBackbone(
        freeze_backbone=args.freeze_backbone,
        checkpoint=f"thirdLibrary/FastSAM/weights/FastSAM-{args.backbone.split('-')[-1]}.pt",
    ).to(device)
    backbone.eval()

    decoder = build_decoder(args.decoder_type, args.proto_source, logger,
                            backbone=backbone, device=device).to(device)

    # ── Auto-detect channels from backbone (after probing via build_decoder) | 从 backbone 自动获取通道数 ──
    ch = backbone.channels if all(v > 0 for v in backbone.channels.values()) else \
         {"p2": 160, "p3": 960, "p4": 1280, "p8": 1280}  # fallback
    logger.log_info("model", f"Backbone channels: {ch}")

    # ── CAT-SAM Adapter | 特征域适配器 ──
    adapter = None
    if args.use_adapter:
        adapter = MultiScaleAdapter(
            p3_channels=ch["p3"],
            p4_channels=ch["p4"],
            p8_channels=ch["p8"],
            reduction=4,
        ).to(device)
        adapter_params = sum(p.numel() for p in adapter.parameters())
        logger.log_info("model", f"MultiScaleAdapter: {adapter_params:,} params (CAT-SAM style)")

    # ── DCT Spectral Attention | 频域注意力 ──
    spectral_attn = None
    freq_fusion = None
    if args.use_spectral:
        spectral_attn = MultiScaleSpectralAttention(
            p2_channels=ch["p2"], p3_channels=ch["p3"], p4_channels=ch["p4"],
            reduction=4, n_freq=16,
        ).to(device)
        spec_params = sum(p.numel() for p in spectral_attn.parameters())
        logger.log_info("model", f"MultiScaleSpectralAttention: {spec_params:,} params (DCT 16-band)")

    if args.use_freq_fusion and args.decoder_type == "pure_p2p3p4":
        freq_fusion = FrequencyGuidedFusion(mid_channels=128, patch_size=8).to(device)
        ff_params = sum(p.numel() for p in freq_fusion.parameters())
        logger.log_info("model", f"FrequencyGuidedFusion: {ff_params:,} params (dynamic weights)")

    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    if adapter is not None:
        trainable_params += sum(p.numel() for p in adapter.parameters())
    if spectral_attn is not None:
        trainable_params += sum(p.numel() for p in spectral_attn.parameters())
    if freq_fusion is not None:
        trainable_params += sum(p.numel() for p in freq_fusion.parameters())
    logger.log_info("model", f"Total trainable params: {trainable_params:,}")

    # ── 优化器 | Optimizer ──
    optim_params = list(decoder.parameters())
    if adapter is not None:
        optim_params += list(adapter.parameters())
    if spectral_attn is not None:
        optim_params += list(spectral_attn.parameters())
    if freq_fusion is not None:
        optim_params += list(freq_fusion.parameters())
    optimizer = torch.optim.AdamW(
        optim_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch,
    )

    # ── 构建固定 Support Cache | Build Fixed Support Cache ──
    rng = random.Random(args.seed)
    support_indices = rng.sample(range(len(train_ds)),
                                  min(args.k_support, len(train_ds)))

    support_imgs_list, support_masks_list = [], []
    for idx in support_indices:
        try:
            s = train_ds[idx]
        except (ValueError, OSError, FileNotFoundError):
            continue
        if (s["masks"] > 0).sum() < 1.0:
            continue
        support_imgs_list.append(s["image"])
        # 转换为二值 mask 用于 FG pooling
        support_masks_list.append((s["masks"] > 0).float())

    if len(support_imgs_list) == 0:
        logger.log_info("error", "No support samples with FG pixels! Check dataset.")
        return

    support_imgs = torch.stack(support_imgs_list)
    support_masks = torch.stack(support_masks_list)

    support_cache = None
    if args.decoder_type == "adaptive":
        support_proto = compute_support_prototype(
            backbone, support_imgs, support_masks, device,
            proto_source=args.proto_source
        )
        support_cache = support_proto
        logger.log_info("data",
            f"Support cache: {len(support_imgs_list)} tiles, "
            f"|p|={support_proto.norm().item():.4f}"
        )
    else:
        # Pure decoder — no prototype needed
        support_cache = torch.zeros(1280, device=device)
        logger.log_info("data", "Pure decoder: no support prototype needed")

    # ── 训练循环 | Training Loop ──
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train",
        f"Starting training: {args.epochs} epochs × {args.steps_per_epoch} steps")
    logger.log_info("train", f"{'='*60}")

    best_miou = 0.0
    best_epoch = 0
    global_step = 0
    nan_skip_count = 0

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        epoch_ce_vals = []
        epoch_dice_vals = []
        epoch_lovasz_vals = []
        epoch_pred_mean = []

        pbar = tqdm(range(args.steps_per_epoch),
                    desc=f"Epoch {epoch:3d}/{args.epochs}", unit="step")
        for _ in pbar:
            # ── 随机采样 query tile | Random sample query tile ──
            q_idx = rng.randint(0, len(train_ds) - 1)
            try:
                q_sample = train_ds[q_idx]
            except (ValueError, OSError, FileNotFoundError) as e:
                # 跳过损坏文件 | Skip corrupted files
                continue
            query_img = q_sample["image"]
            query_mask = q_sample["masks"]
            H, W = query_mask.shape[1:]

            # 跳过无 FG 样本 | Skip empty samples
            if (query_mask > 0).sum() < 1.0:
                continue

            # ── 数据增强 | Data Augmentation ──
            if augment:
                query_img_aug, query_mask_aug = augment(query_img, query_mask)
            else:
                query_img_aug, query_mask_aug = query_img, query_mask

            query_img_dev = query_img_aug.unsqueeze(0).to(device)
            query_mask_dev = query_mask_aug.to(device)

            # ── Pad to 32 倍数 | Pad to multiple of 32 ──
            query_img_dev, _, _ = pad_to_32(query_img_dev)

            # ── Decoder Forward ──
            decoder.train()
            if adapter is not None:
                adapter.train()
            feats = backbone(query_img_dev, extract_proto=True)

            # ── Adapter (CAT-SAM style) | 特征域适配 ──
            if adapter is not None:
                adapted = adapter(p3=feats.get("p3"), p4=feats.get("p4"), p8=feats.get("p8"))
                feats.update(adapted)

            # ── DCT Spectral Attention | 频域注意力 ──
            if spectral_attn is not None:
                spec_feats = spectral_attn(
                    p2=feats.get("p2"), p3=feats.get("p3"), p4=feats.get("p4"),
                )
                feats.update(spec_feats)

            pred_prob = _decoder_forward(decoder, feats, support_cache,
                                         freq_fusion=freq_fusion)
            if pred_prob is None:
                continue

            # ── 上采样到原图 | Upsample to original → [C, H, W] ──
            pred_full = F.interpolate(
                pred_prob.unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(0)  # [C, H, W]

            # ── Loss ──
            target = query_mask_dev.squeeze(0).long()
            if args.loss_type == "lovasz":
                loss_val, loss_dict = lovasz_combined_loss(
                    pred_full.unsqueeze(0), target.unsqueeze(0),
                    ce_weight=ce_weight,
                )
            elif args.loss_type == "spectral":
                loss_val, loss_dict = spectral_combined_loss(
                    pred_full.unsqueeze(0), target.unsqueeze(0),
                    ce_weight=ce_weight,
                )
            else:
                loss_val, loss_dict = multiclass_combined_loss(
                    pred_full.unsqueeze(0), target.unsqueeze(0),
                    ce_weight=ce_weight,
                )

            if torch.isnan(loss_val) or torch.isinf(loss_val):
                logger.log_info("nan_diag",
                    f"NaN/Inf loss at step {global_step}, skipping")
                nan_skip_count += 1
                continue

            optimizer.zero_grad()
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

            # 梯度 NaN 检测 | Gradient NaN detection
            grad_nan = False
            all_params = list(decoder.named_parameters())
            if adapter is not None:
                all_params += list(adapter.named_parameters())
            if spectral_attn is not None:
                all_params += list(spectral_attn.named_parameters())
            if freq_fusion is not None:
                all_params += list(freq_fusion.named_parameters())
            for name, param in all_params:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        grad_nan = True
                        break
            if grad_nan:
                optimizer.zero_grad()
                nan_skip_count += 1
                continue

            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_losses.append(loss_val.item())
            epoch_ce_vals.append(loss_dict["ce"])
            epoch_dice_vals.append(loss_dict["dice"])
            if "lovasz" in loss_dict:
                epoch_lovasz_vals.append(loss_dict["lovasz"])
            if "spectral" in loss_dict:
                epoch_lovasz_vals.append(loss_dict["spectral"])  # reuse same list
            epoch_pred_mean.append(pred_full[1:].mean().item())  # mean over FG classes

            if len(epoch_losses) > 0:
                postfix = {"loss": f"{np.mean(epoch_losses[-50:]):.4f}"}
                if epoch_ce_vals:
                    postfix["ce"] = f"{np.mean(epoch_ce_vals[-50:]):.4f}"
                if epoch_dice_vals:
                    postfix["dice"] = f"{np.mean(epoch_dice_vals[-50:]):.4f}"
                postfix["pred"] = f"{np.mean(epoch_pred_mean[-50:]):.4f}"
                pbar.set_postfix(postfix)

        # ── Epoch 汇总 | Epoch Summary ──
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        log_msg = (f"Epoch {epoch:3d}/{args.epochs} | loss={avg_loss:.4f}")
        if epoch_ce_vals:
            avg_ce = np.mean(epoch_ce_vals)
            log_msg += f" ce={avg_ce:.4f}"
            logger.log_metric("ce", avg_ce, step=epoch, tags=["neuseg_train"])
        if epoch_dice_vals:
            avg_dice = np.mean(epoch_dice_vals)
            log_msg += f" dice={avg_dice:.4f}"
            logger.log_metric("dice", avg_dice, step=epoch, tags=["neuseg_train"])
        if epoch_lovasz_vals:
            avg_lovasz = np.mean(epoch_lovasz_vals)
            log_msg += f" lovasz={avg_lovasz:.4f}"
            logger.log_metric("lovasz", avg_lovasz, step=epoch, tags=["neuseg_train"])
        log_msg += (f" | lr={scheduler.get_last_lr()[0]:.2e} | NaN={nan_skip_count}")
        logger.log_info("epoch", log_msg)
        logger.log_metric("loss", avg_loss, step=epoch, tags=["neuseg_train"])

        # ── 评估 (定期) | Evaluation (periodic) ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"{'─'*50}")
            logger.log_info("eval", f"Evaluation @ Epoch {epoch}")

            eval_result = evaluate(
                decoder, backbone, support_cache, val_ds, device,
                num_classes=NUM_CLASSES, adapter=adapter,
                spectral_attn=spectral_attn, freq_fusion=freq_fusion,
            )
            miou = eval_result["mIoU"]
            dice = np.mean(list(eval_result.get("per_class_Dice", {}).values()))

            logger.log_info("eval",
                f"  mIoU={miou:.4f}  dice={dice:.4f}  "
                f"n={eval_result['n_evaluated']}  "
                f"best_mIoU={best_miou:.4f} (epoch {best_epoch})"
            )
            for cls_name, iou_c in eval_result.get("per_class_IoU", {}).items():
                logger.log_info("eval", f"    {cls_name}: IoU={iou_c:.4f}")
            logger.log_metric("mIoU", miou, step=epoch, tags=["neuseg_eval"])
            logger.log_metric("Dice", dice, step=epoch, tags=["neuseg_eval"])

            # ── 按 Val mIoU 保存最佳 (与 SegNeXt 基线选择准则一致) ──
            # Save best by Val mIoU (same model-selection criterion as SegNeXt baseline)
            if miou > best_miou:
                best_miou = miou
                best_epoch = epoch
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "decoder_state_dict": {k: v.clone() for k, v
                                           in decoder.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mIoU": miou,
                    "Dice": dice,
                    "args": vars(args),
                    "num_classes": NUM_CLASSES,
                }
                if adapter is not None:
                    checkpoint["adapter_state_dict"] = {k: v.clone() for k, v
                        in adapter.state_dict().items()}
                if spectral_attn is not None:
                    checkpoint["spectral_attn_state_dict"] = {k: v.clone() for k, v
                        in spectral_attn.state_dict().items()}
                if freq_fusion is not None:
                    checkpoint["freq_fusion_state_dict"] = {k: v.clone() for k, v
                        in freq_fusion.state_dict().items()}
                if isinstance(support_cache, tuple):
                    checkpoint["support_proto_p3"] = support_cache[0].clone()
                    checkpoint["support_proto_p4"] = support_cache[1].clone()
                else:
                    checkpoint["support_proto"] = support_cache.clone() \
                        if isinstance(support_cache, torch.Tensor) else support_cache
                torch.save(checkpoint, str(out_dir / "best_model.pt"))
                logger.log_info("eval",
                    f"  ✓ New best: mIoU={best_miou:.4f} @ epoch {best_epoch}")

    # ── 最终保存 | Final Save ──
    final_checkpoint = {
        "epoch": args.epochs,
        "global_step": global_step,
        "decoder_state_dict": {k: v.clone() for k, v
                               in decoder.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mIoU": best_miou,
        "best_epoch": best_epoch,
        "args": vars(args),
        "num_classes": NUM_CLASSES,
        "nan_skip_count": nan_skip_count,
    }
    if adapter is not None:
        final_checkpoint["adapter_state_dict"] = {k: v.clone() for k, v
            in adapter.state_dict().items()}
    if spectral_attn is not None:
        final_checkpoint["spectral_attn_state_dict"] = {k: v.clone() for k, v
            in spectral_attn.state_dict().items()}
    if freq_fusion is not None:
        final_checkpoint["freq_fusion_state_dict"] = {k: v.clone() for k, v
            in freq_fusion.state_dict().items()}
    if isinstance(support_cache, tuple):
        final_checkpoint["support_proto_p3"] = support_cache[0].clone()
        final_checkpoint["support_proto_p4"] = support_cache[1].clone()
    else:
        final_checkpoint["support_proto"] = support_cache.clone() \
            if isinstance(support_cache, torch.Tensor) else support_cache
    torch.save(final_checkpoint, str(out_dir / "last_model.pt"))

    # ── 保存结果 JSON | Save Results JSON ──
    results = {
        "experiment": "Neu_seg Multi-class Training",
        "backbone": args.backbone,
        "decoder_type": args.decoder_type,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "k_support": args.k_support,
        "best_mIoU": round(best_miou, 6),
        "best_epoch": best_epoch,
        "nan_skip_count": nan_skip_count,
        "trainable_params": trainable_params,
        "augment": args.augment,
        "class_weights": args.class_weights,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 摘要 | Summary ──
    print(f"\n{'='*60}")
    print(f"  Neu_seg Multi-class Training — Complete")
    print(f"  Decoder: {args.decoder_type}")
    print(f"  Epochs: {args.epochs}, Steps: {global_step}")
    print(f"  Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")
    print(f"  NaN skips: {nan_skip_count}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    logger.log_info("done", f"Output: {out_dir}")
    logger.log_info("done", f"Best mIoU: {best_miou:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
