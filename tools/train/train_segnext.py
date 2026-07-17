#!/usr/bin/env python3
"""
SegNeXt 复现训练 — NEU_Seg 数据集 | SegNeXt Reproduction — NEU_Seg Dataset.
=============================================================================

SegNeXt (Guo et al., NeurIPS 2022):
    "Rethinking Convolutional Attention Design for Semantic Segmentation"
    https://arxiv.org/abs/2209.08575

模型 | Model:
    MSCAN (backbone) + LightHamHead (decoder), 独立于 mmseg 框架。
    MSCAN backbone + LightHamHead decoder, standalone (no mmseg dependency).

4 类工业缺陷分割 | 4-class industrial defect segmentation:
    BG(0) + Inclusion(1) + Patch(2) + Scratch(3)

训练配方对齐官方 mmseg 实现 | Training recipe mirrors official mmseg configs:
    - 数据增广: RandomResize(0.5~2.0) + RandomCrop(cat_max_ratio=0.75) + HFlip(0.5)
      + PhotoMetricDistortion (官方 train pipeline)
    - 归一化: ImageNet mean/std (官方 img_norm_cfg, 配合 IN-1K 预训练权重)
    - 优化器: AdamW lr=6e-5, head lr×10, norm 参数 weight_decay=0 (官方 paramwise_cfg)
    - 损失: 纯 CrossEntropy, ignore_index=255 (官方 loss_decode)
    - 调度: linear warmup 1500 iters → poly power=1.0 (官方 lr_config)

用法 | Usage::

    # SegNeXt-Tiny + IN-1K 预训练 (官方配方, 推荐)
    python tools/train/train_segnext.py --model-size tiny --pretrained pretrained/mscan_t.pth

    # SegNeXt-Small + 预训练权重
    python tools/train/train_segnext.py --model-size small --pretrained pretrained/mscan_s.pth

    # 旧配方 (兼容历史 run: CE+Dice, 无增广, [0,1] 归一化)
    python tools/train/train_segnext.py --loss ce_dice --no-augment --img-norm unit --lr 6e-4

    # 快速验证
    python tools/train/train_segnext.py --epochs 10 --steps-per-epoch 100 --device cpu
"""

from __future__ import annotations

import sys, argparse, json, random
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.backbone.mscan import MSCAN, SEGNEXT_CONFIGS
from adatile.decoder.ham_head import LightHamHead

# ── 数据集 | Dataset ──
from adatile.datasets.neu_seg import NEUSegDataset


NUM_CLASSES = 4
CLASS_NAMES = ["background", "Inclusion", "Patch", "Scratch"]

# 忽略像素值 (pad 区域) | Ignore label for padded regions (official seg_pad_val)
IGNORE_INDEX = 255

# ImageNet 归一化常数 (官方 mmseg img_norm_cfg: mean=[123.675,116.28,103.53]/255 等)
# ImageNet normalization constants (official mmseg img_norm_cfg, rescaled to [0,1] input)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_img(img: torch.Tensor, mode: str) -> torch.Tensor:
    """
    图像归一化 | Image normalization.

    :param img: [3, H, W] 或 [B, 3, H, W], float32 [0,1]。
    :param mode: "imagenet" → ImageNet mean/std (官方, 配合 IN-1K 预训练);
                 "unit" → 保持 [0,1] (旧行为 | legacy behavior)。
    """
    if mode == "imagenet":
        shape = (3, 1, 1) if img.dim() == 3 else (1, 3, 1, 1)
        mean = img.new_tensor(IMAGENET_MEAN).view(shape)
        std = img.new_tensor(IMAGENET_STD).view(shape)
        return (img - mean) / std
    return img


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════

def multi_class_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                          smooth: float = 1e-5,
                          ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """
    多类别 Dice 损失 | Multi-class Dice Loss.

    :param pred: [B, C, H, W] softmax 概率 | softmax probabilities.
    :param target: [B, H, W] 类别索引 (long), 可含 ignore_index | class index (long).
    :return: scalar Dice loss.
    """
    B, C, H, W = pred.shape
    # 屏蔽 ignore 像素 (pad 区域) | Mask out ignored pixels (padded regions)
    valid = (target != ignore_index)
    target_safe = target.clone()
    target_safe[~valid] = 0
    valid_f = valid.unsqueeze(1).float()                              # [B, 1, H, W]
    target_one_hot = F.one_hot(target_safe, num_classes=C).permute(0, 3, 1, 2).float()
    target_one_hot = target_one_hot * valid_f
    pred = pred * valid_f
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def compute_loss(pred: torch.Tensor, target: torch.Tensor,
                 loss_type: str = "ce",
                 ce_weight: torch.Tensor | None = None,
                 ce_alpha: float = 0.5, dice_alpha: float = 0.5) -> dict[str, float]:
    """
    损失计算 | Loss computation.

    :param loss_type: "ce" → 纯 CrossEntropy (官方 mmseg loss_decode);
                      "ce_dice" → 0.5·CE + 0.5·Dice (旧配方 | legacy recipe)。
    :return: dict with "loss", "ce", "dice" for logging.
    """
    # CE loss: pred is softmax → use NLL; ignore_index 排除 pad 像素
    log_pred = torch.log(pred + 1e-7)
    ce = F.nll_loss(log_pred, target, weight=ce_weight, ignore_index=IGNORE_INDEX)

    if loss_type == "ce":
        return {"loss": ce, "ce": ce.item(), "dice": 0.0}

    dice = multi_class_dice_loss(pred, target)
    loss = ce_alpha * ce + dice_alpha * dice
    return {"loss": loss, "ce": ce.item(), "dice": dice.item()}


# ═══════════════════════════════════════════════════════════════════
# 官方数据增广 | Official Data Augmentation
# ═══════════════════════════════════════════════════════════════════

class OfficialAugment:
    """
    官方 mmseg SegNeXt 训练增广管线 | Official mmseg SegNeXt train pipeline.

    镜像官方 train_pipeline (local_configs/_base_/datasets):
        RandomResize(scale × 0.5~2.0, keep_ratio)
        → RandomCrop(crop², cat_max_ratio=0.75)
        → RandomFlip(p=0.5)
        → PhotoMetricDistortion(brightness=32, contrast/saturation=0.5~1.5, hue=18)
        → Pad(crop², img_pad_val=0, seg_pad_val=255)

    输入/输出均为 numpy: image float32 RGB [H,W,3] (0~255), mask int64 [H,W]。
    Pad 区域 mask=255, 由 ignore_index 排除出损失。
    Operates on numpy arrays; padded pixels get mask=255 (excluded via ignore_index).
    """

    def __init__(self, crop_size: int = 200,
                 ratio_range: tuple[float, float] = (0.5, 2.0),
                 cat_max_ratio: float = 0.75,
                 flip_prob: float = 0.5,
                 brightness_delta: int = 32,
                 contrast_range: tuple[float, float] = (0.5, 1.5),
                 saturation_range: tuple[float, float] = (0.5, 1.5),
                 hue_delta: int = 18):
        self.crop_size = crop_size
        self.ratio_range = ratio_range
        self.cat_max_ratio = cat_max_ratio
        self.flip_prob = flip_prob
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_delta = hue_delta

    # ── RandomResize ──
    def _random_resize(self, img: np.ndarray, mask: np.ndarray):
        """keep_ratio 缩放 (方形输入 → 等比) | keep-ratio resize."""
        ratio = np.random.uniform(*self.ratio_range)
        new_h = max(int(img.shape[0] * ratio + 0.5), 1)
        new_w = max(int(img.shape[1] * ratio + 0.5), 1)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (new_w, new_h),
                          interpolation=cv2.INTER_NEAREST).astype(np.int64)
        return img, mask

    # ── RandomCrop (cat_max_ratio) ──
    def _random_crop(self, img: np.ndarray, mask: np.ndarray):
        """
        随机裁剪, 单类占比 >cat_max_ratio 时重采样 (最多 10 次, 官方语义)。
        Random crop; resample location if one class dominates (>cat_max_ratio), 10 attempts.
        """
        c = self.crop_size
        h, w = mask.shape

        def rand_bbox():
            y = np.random.randint(0, max(h - c, 0) + 1)
            x = np.random.randint(0, max(w - c, 0) + 1)
            return y, x

        y, x = rand_bbox()
        if self.cat_max_ratio < 1.0:
            for _ in range(10):
                crop_m = mask[y:y + c, x:x + c]
                labels, cnt = np.unique(crop_m, return_counts=True)
                cnt = cnt[labels != IGNORE_INDEX]
                if len(cnt) > 1 and cnt.max() / cnt.sum() < self.cat_max_ratio:
                    break
                y, x = rand_bbox()
        return img[y:y + c, x:x + c], mask[y:y + c, x:x + c]

    # ── PhotoMetricDistortion ──
    def _photometric(self, img: np.ndarray) -> np.ndarray:
        """亮度/对比度/饱和度/色调扰动 (镜像 mmseg 语义, 各步 p=0.5)。"""
        def rand2() -> bool:
            return bool(np.random.randint(2))

        if rand2():  # brightness
            img = np.clip(img + np.random.uniform(-self.brightness_delta,
                                                  self.brightness_delta), 0, 255)
        contrast_first = rand2()  # 对比度在饱和度前/后随机 | contrast before/after saturation
        if contrast_first and rand2():
            img = np.clip(img * np.random.uniform(*self.contrast_range), 0, 255)

        do_sat, do_hue = rand2(), rand2()
        if do_sat or do_hue:
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            if do_sat:
                s = hsv[:, :, 1].astype(np.float32) * np.random.uniform(*self.saturation_range)
                hsv[:, :, 1] = np.clip(s, 0, 255).astype(np.uint8)
            if do_hue:
                h = (hsv[:, :, 0].astype(int)
                     + np.random.randint(-self.hue_delta, self.hue_delta + 1)) % 180
                hsv[:, :, 0] = h.astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

        if not contrast_first and rand2():
            img = np.clip(img * np.random.uniform(*self.contrast_range), 0, 255)
        return img

    # ── Pad ──
    def _pad(self, img: np.ndarray, mask: np.ndarray):
        """不足 crop² 时右下 pad (img→0, mask→255) | Bottom-right pad to crop²."""
        c = self.crop_size
        pad_h, pad_w = max(c - img.shape[0], 0), max(c - img.shape[1], 0)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
            mask = cv2.copyMakeBorder(mask.astype(np.int32), 0, pad_h, 0, pad_w,
                                      cv2.BORDER_CONSTANT,
                                      value=IGNORE_INDEX).astype(np.int64)
        return img, mask

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        """:param img: [H,W,3] float32 RGB 0~255. :param mask: [H,W] int64."""
        img, mask = self._random_resize(img, mask)
        img, mask = self._random_crop(img, mask)
        if np.random.rand() < self.flip_prob:  # RandomFlip (horizontal)
            img, mask = np.fliplr(img).copy(), np.fliplr(mask).copy()
        img = self._photometric(img.astype(np.float32))
        img, mask = self._pad(img, mask)
        return img, mask


# ═══════════════════════════════════════════════════════════════════
# 官方优化器参数分组 | Official Optimizer Param Groups
# ═══════════════════════════════════════════════════════════════════

def build_official_param_groups(backbone: nn.Module, head: nn.Module,
                                lr: float, weight_decay: float) -> list[dict]:
    """
    官方 paramwise_cfg 参数分组 | Official paramwise_cfg param groups.

    镜像 mmcv DefaultOptimizerConstructor 的 custom_keys 语义
    (参数全名子串匹配, 按字典序首个命中生效):
        'head' → lr×10;  'norm' → weight_decay=0;  'pos_block' → weight_decay=0。
    Mirrors mmcv custom_keys semantics: substring match on the full param name,
    first hit in sorted key order wins.
    """
    custom_keys = {
        "head": dict(lr_mult=10.0, decay_mult=1.0),
        "norm": dict(lr_mult=1.0, decay_mult=0.0),
        "pos_block": dict(lr_mult=1.0, decay_mult=0.0),  # MSCAN 无此模块 (保留官方键)
    }
    sorted_keys = sorted(custom_keys)

    named = [(f"backbone.{n}", p) for n, p in backbone.named_parameters()] + \
            [(f"decode_head.{n}", p) for n, p in head.named_parameters()]

    groups: dict[tuple[float, float], dict] = {}
    for name, param in named:
        if not param.requires_grad:
            continue
        lr_mult, decay_mult = 1.0, 1.0
        for key in sorted_keys:
            if key in name:
                lr_mult = custom_keys[key].get("lr_mult", 1.0)
                decay_mult = custom_keys[key].get("decay_mult", 1.0)
                break
        gkey = (lr_mult, decay_mult)
        if gkey not in groups:
            groups[gkey] = {"params": [], "lr": lr * lr_mult,
                            "weight_decay": weight_decay * decay_mult,
                            "lr_mult": lr_mult, "decay_mult": decay_mult}
        groups[gkey]["params"].append(param)
    return list(groups.values())




# ═══════════════════════════════════════════════════════════════════
# 评估 | Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_segnext(backbone: MSCAN, head: LightHamHead,
                     val_ds: NEUSegDataset, device: torch.device,
                     img_norm: str = "unit") -> dict:
    """
    评估 SegNeXt 模型 | Evaluate SegNeXt model.

    :param img_norm: 图像归一化模式, 必须与训练一致 | Must match training normalization.
    :return: dict with mIoU, per_class IoU, Dice, pixel_accuracy, per_sample info.
    """
    backbone.eval()
    head.eval()

    per_class_intersection = np.zeros(NUM_CLASSES, dtype=np.float64)
    per_class_union = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_correct = 0
    total_pixels = 0
    per_sample_ious = []

    for idx in range(len(val_ds)):
        try:
            sample = val_ds[idx]
        except (ValueError, OSError, FileNotFoundError):
            continue

        img = sample["image"].unsqueeze(0).to(device)     # [1, 3, H, W]
        img = normalize_img(img, img_norm)
        gt_raw = sample["masks"]
        if isinstance(gt_raw, torch.Tensor):
            gt_raw = gt_raw.numpy()
        gt = gt_raw.astype(np.int64)
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt.squeeze(0)                           # [1, H, W] → [H, W]

        feats = backbone(img)
        logits = head(feats[1:])  # [1, C, H/8, W/8]
        pred = F.softmax(logits, dim=1)
        pred_up = F.interpolate(pred, size=tuple(gt.shape), mode="bilinear",
                                align_corners=False)
        pred_cls = pred_up.argmax(dim=1).squeeze(0).cpu().numpy()  # [H, W]

        # Compute intersection/union
        for c in range(NUM_CLASSES):
            pred_c = (pred_cls == c)
            gt_c = (gt == c)
            per_class_intersection[c] += (pred_c & gt_c).sum()
            per_class_union[c] += (pred_c | gt_c).sum()

        total_correct += (pred_cls == gt).sum()
        total_pixels += gt.size

        # Per-sample mIoU (only classes present in GT or prediction)
        sample_ious = []
        for c in range(NUM_CLASSES):
            inter = (pred_cls == c) & (gt == c)
            union = (pred_cls == c) | (gt == c)
            if union.sum() > 0:  # class present in this sample's GT or prediction
                sample_ious.append(inter.sum() / union.sum())
        if sample_ious:
            per_sample_ious.append(np.mean(sample_ious))

    # Compute metrics
    ious = np.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        union = per_class_union[c]
        ious[c] = per_class_intersection[c] / max(union, 1)

    mIoU = float(np.mean(ious))
    pixel_acc = total_correct / max(total_pixels, 1)
    # Macro Dice: 逐类 2I/(I+U) ≡ 2TP/(2TP+FP+FN) 后取均值 (全仓库统一定义)
    # Macro Dice: per-class 2I/(I+U), then mean — unified definition repo-wide
    dices = 2 * per_class_intersection / np.maximum(
        per_class_intersection + per_class_union, 1)
    dice = float(np.mean(dices))

    return {
        "mIoU": round(mIoU, 6),
        "Dice": round(dice, 6),
        "pixel_accuracy": round(float(pixel_acc), 6),
        "per_class_IoU": {name: round(float(ious[i]), 4)
                          for i, name in enumerate(CLASS_NAMES)},
        "sample_mIoU_mean": round(float(np.mean(per_sample_ious)), 6)
               if per_sample_ious else 0.0,
        "sample_mIoU_median": round(float(np.median(per_sample_ious)), 6)
               if per_sample_ious else 0.0,
        "n_evaluated": len(per_sample_ious),
    }


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="SegNeXt Training on NEU_Seg")
    # ── 模型 | Model ──
    p.add_argument("--model-size", type=str, default="tiny",
                   choices=["tiny", "small", "base", "large"],
                   help="SegNeXt 模型大小 | Model size")
    p.add_argument("--pretrained", type=str, default=None,
                   help="MSCAN 预训练权重路径 (mmseg checkpoint) | Pretrained MSCAN weights")
    p.add_argument("--ham-channels", type=int, default=256,
                   help="Hamburger 内部通道数 | Hamburger internal channels")
    p.add_argument("--md-r", type=int, default=16,
                   help="NMF 字典基数量 (MD_R)")

    # ── 数据 | Data ──
    p.add_argument("--data-root", type=str, default="data/NEU_Seg")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16,
                   help="批次大小 | Batch size (官方 1×bs16)")
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="官方增广管线 (Resize/Crop/Flip/PhotoMetric), --no-augment 关闭")
    p.add_argument("--img-norm", type=str, default="imagenet",
                   choices=["imagenet", "unit"],
                   help="图像归一化: imagenet=官方 mean/std, unit=[0,1] (旧行为)")

    # ── 损失 | Loss ──
    p.add_argument("--loss", type=str, default="ce",
                   choices=["ce", "ce_dice"],
                   help="损失: ce=纯 CrossEntropy (官方), ce_dice=0.5CE+0.5Dice (旧配方)")
    p.add_argument("--class-weights", type=str, default="none",
                   choices=["none", "balanced", "inverse"],
                   help="类别权重策略 | Class weight strategy")

    # ── 优化器 | Optimizer ──
    p.add_argument("--lr", type=float, default=6e-5,
                   help="基础学习率 | Base LR (官方 6e-5, head 自动 ×10)")
    p.add_argument("--min-lr", type=float, default=0.0,
                   help="poly 调度学习率下限 | Minimum LR for poly schedule (官方 0.0)")
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="权重衰减 | Weight decay")
    p.add_argument("--warmup-iters", type=int, default=1500,
                   help="warmup 迭代数 | Warmup iterations (官方 1500)")

    # ── 硬件 | Hardware ──
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=10,
                   help="每 N epochs 评估一次 | Evaluate every N epochs")
    p.add_argument("--output-dir", type=str, default=None)

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
        pretrained_tag = "_pt" if args.pretrained else ""
        args.output_dir = f"runs/segnext_{args.model_size}{pretrained_tag}_NEUSeg_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logger ──
    logger = get_logger("train_segnext")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))

    cfg = SEGNEXT_CONFIGS[args.model_size]
    logger.log_info("config", f"SegNeXt-{args.model_size}: {cfg}")
    logger.log_info("config", f"Training: {args.epochs} epochs × "
                    f"{args.steps_per_epoch} steps, lr={args.lr}, bs={args.batch_size}")
    logger.log_info("config", f"Pretrained: {args.pretrained or 'None'}")
    logger.log_info("config", f"Recipe: loss={args.loss}, augment={args.augment}, "
                    f"img_norm={args.img_norm}, warmup={args.warmup_iters}")

    # ── 数据集 | Datasets ──
    logger.log_info("data", "Loading NEU_Seg datasets...")
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

    # ── 模型 | Models ──
    logger.log_info("model", f"Building SegNeXt-{args.model_size}...")
    backbone = MSCAN(model_size=args.model_size, pretrained=args.pretrained,
                     drop_path_rate=0.1).to(device)

    # Decoder 通道数从 backbone config 推导
    embed_dims = cfg["embed_dims"]
    in_channels = [embed_dims[1], embed_dims[2], embed_dims[3]]  # C2, C3, C4
    head = LightHamHead(
        in_channels=in_channels,
        num_classes=NUM_CLASSES,
        ham_channels=args.ham_channels,
        channels=256,
        ham_kwargs=dict(MD_R=args.md_r),
        dropout_ratio=0.1,
    ).to(device)

    n_backbone = sum(p.numel() for p in backbone.parameters())
    n_head = sum(p.numel() for p in head.parameters())
    logger.log_info("model",
        f"Backbone: {n_backbone/1e6:.3f}M, Head: {n_head/1e3:.1f}K, "
        f"Total: {(n_backbone+n_head)/1e6:.3f}M")

    # ── 优化器 (官方配方: AdamW lr=6e-5, head lr×10, norm no-decay, poly) | Optimizer ──
    param_groups = build_official_param_groups(backbone, head, args.lr, args.weight_decay)
    for g in param_groups:
        n = sum(p.numel() for p in g["params"])
        logger.log_info("optim",
            f"  group lr_mult={g['lr_mult']:g} decay_mult={g['decay_mult']:g}: "
            f"{len(g['params'])} tensors / {n/1e6:.3f}M params, "
            f"lr={g['lr']:.1e}, wd={g['weight_decay']}")
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
                                  betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # ── 数据增广 | Data Augmentation ──
    augment = OfficialAugment(crop_size=200) if args.augment else None

    # Poly schedule: lr = lr0 * (1 - iter/max_iter)^power (官方 power=1.0, warmup linear)
    max_iters = args.epochs * args.steps_per_epoch
    min_factor = args.min_lr / max(args.lr, 1e-12)

    def poly_lambda(current_iter):
        if current_iter < args.warmup_iters:
            return current_iter / max(args.warmup_iters, 1) * 1.0
        factor = (1 - (current_iter - args.warmup_iters) /
                  max(max_iters - args.warmup_iters, 1)) ** 1.0
        return max(factor, min_factor)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, poly_lambda)

    # ── 训练循环 | Training Loop ──
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train",
        f"Starting training: {args.epochs} epochs × {args.steps_per_epoch} steps")
    logger.log_info("train", f"{'='*60}")

    best_miou = 0.0
    best_dice = 0.0
    best_epoch = 0
    global_step = 0
    nan_skip_count = 0

    # 创建随机采样索引池（无限循环用）| Create random index pool (for infinite sampling)
    n_train = len(train_ds)
    indices_pool = list(range(n_train))

    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()

        epoch_losses = []
        epoch_ces = []
        epoch_dices = []

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch:3d}/{args.epochs}")
        for step in pbar:
            # ── 批量采样 | Batch Sampling ──
            batch_idxs = random.sample(indices_pool, min(args.batch_size, n_train))

            imgs_list, masks_list = [], []
            for idx in batch_idxs:
                try:
                    s = train_ds[idx]
                except (ValueError, OSError, FileNotFoundError):
                    continue
                img_t = s["image"]                                # [3, H, W] float [0,1]
                m = s["masks"]
                if isinstance(m, np.ndarray):
                    m = torch.from_numpy(m.copy()).long()
                else:
                    m = m.long()
                # Ensure [H, W] (no channel dim)
                if m.dim() == 3 and m.shape[0] == 1:
                    m = m.squeeze(0)

                # ── 官方增广 (numpy RGB 0~255) | Official augmentation ──
                if augment is not None:
                    img_np = img_t.permute(1, 2, 0).numpy() * 255.0
                    img_np, m_np = augment(img_np, m.numpy())
                    img_t = torch.from_numpy(
                        np.ascontiguousarray(img_np / 255.0)).permute(2, 0, 1).float()
                    m = torch.from_numpy(m_np).long()

                imgs_list.append(img_t)
                masks_list.append(m)

            if len(imgs_list) == 0:
                continue

            imgs = torch.stack(imgs_list).to(device)          # [B, 3, H, W]
            imgs = normalize_img(imgs, args.img_norm)
            gts = torch.stack(masks_list).to(device)           # [B, H, W]
            if gts.dim() == 4 and gts.shape[1] == 1:
                gts = gts.squeeze(1)                          # [B, 1, H, W] → [B, H, W]

            # ── 前向传播 | Forward ──
            feats = backbone(imgs)
            logits = head(feats[1:])                           # [B, C, H/8, W/8]
            pred = F.softmax(logits, dim=1)

            # ── 上采样到原图尺寸 | Upsample to original resolution ──
            pred_up = F.interpolate(pred, size=tuple(gts.shape[1:]), mode="bilinear",
                                    align_corners=False)

            # ── 损失计算 | Loss Computation ──
            loss_dict = compute_loss(pred_up, gts, loss_type=args.loss,
                                     ce_weight=ce_weight)

            if torch.isnan(loss_dict["loss"]) or torch.isinf(loss_dict["loss"]):
                nan_skip_count += 1
                optimizer.zero_grad()
                continue

            epoch_losses.append(loss_dict["loss"].item())
            epoch_ces.append(loss_dict["ce"])
            epoch_dices.append(loss_dict["dice"])

            # ── 反向传播 | Backward ──
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            pbar.set_postfix(
                loss=f"{np.mean(epoch_losses[-10:]):.4f}",
                ce=f"{np.mean(epoch_ces[-10:]):.4f}",
                dice=f"{np.mean(epoch_dices[-10:]):.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

            # ── 日志记录 | Logging ──
            if global_step % 20 == 0:
                logger.log_metric("loss", loss_dict["loss"].item(), step=global_step,
                                  tags=["neuseg_train"])
                logger.log_metric("ce", loss_dict["ce"], step=global_step,
                                  tags=["neuseg_train"])
                logger.log_metric("dice", loss_dict["dice"], step=global_step,
                                  tags=["neuseg_train"])

        # ── Epoch 总结 | Epoch Summary ──
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_ce = np.mean(epoch_ces) if epoch_ces else 0.0
        avg_dice = np.mean(epoch_dices) if epoch_dices else 0.0
        logger.log_info("epoch",
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} ce={avg_ce:.4f} dice={avg_dice:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | NaN={nan_skip_count}")

        # ── 评估 | Evaluation ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"{'─'*60}")
            logger.log_info("eval", f"Evaluation @ Epoch {epoch}")

            metrics = evaluate_segnext(backbone, head, val_ds, device,
                                       img_norm=args.img_norm)
            miou = metrics["mIoU"]

            logger.log_info("eval",
                f"  mIoU={miou:.4f}  dice={metrics['Dice']:.4f}  "
                f"n={metrics['n_evaluated']}  best_mIoU={max(best_miou, miou):.4f} (epoch {best_epoch})")
            for name in CLASS_NAMES:
                logger.log_info("eval",
                    f"    {name:>12s}: IoU={metrics['per_class_IoU'][name]:.4f}")

            logger.log_metric("mIoU", metrics["mIoU"], step=epoch,
                              tags=["neuseg_eval"])
            logger.log_metric("Dice", metrics["Dice"], step=epoch,
                              tags=["neuseg_eval"])

            # ── 保存最佳模型 (按 mIoU) | Save Best Model (by mIoU) ──
            if miou > best_miou:
                best_miou = miou
                best_dice = metrics["Dice"]
                best_epoch = epoch

                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_size": args.model_size,
                    "backbone_state_dict": {k: v.clone() for k, v
                                            in backbone.state_dict().items()},
                    "head_state_dict": {k: v.clone() for k, v
                                        in head.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                    "num_classes": NUM_CLASSES,
                }
                torch.save(checkpoint, str(out_dir / "best_model.pt"))
                logger.log_info("eval",
                    f"  ✓ New best: mIoU={best_miou:.4f} @ epoch {best_epoch}")

    # ── 最终保存 | Final Save ──
    final_checkpoint = {
        "epoch": args.epochs,
        "global_step": global_step,
        "model_size": args.model_size,
        "backbone_state_dict": {k: v.clone() for k, v
                                in backbone.state_dict().items()},
        "head_state_dict": {k: v.clone() for k, v
                            in head.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "best_mIoU": best_miou,
        "best_Dice": best_dice,
        "best_epoch": best_epoch,
        "args": vars(args),
        "num_classes": NUM_CLASSES,
    }
    torch.save(final_checkpoint, str(out_dir / "last_model.pt"))

    # ── 保存结果 JSON | Save Results JSON ──
    results = {
        "experiment": "SegNeXt NEU_Seg Reproduction",
        "model_size": args.model_size,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "best_mIoU": round(best_miou, 6),
        "best_Dice": round(best_dice, 6),
        "best_epoch": best_epoch,
        "nan_skip_count": nan_skip_count,
        "backbone_params": n_backbone,
        "head_params": n_head,
        "class_weights": args.class_weights,
        "pretrained": args.pretrained,
        "loss": args.loss,
        "augment": args.augment,
        "img_norm": args.img_norm,
        "lr": args.lr,
        "warmup_iters": args.warmup_iters,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 完成 | Done ──
    print()
    print("=" * 60)
    print(f"  SegNeXt-{args.model_size} NEU_Seg Training -- Complete")
    print(f"  Epochs: {args.epochs}, Steps/epoch: {args.steps_per_epoch}")
    print(f"  Best mIoU: {best_miou:.4f} (Dice: {best_dice:.4f}) @ epoch {best_epoch}")
    print(f"  NaN skips: {nan_skip_count}")
    print(f"  Output: {out_dir}")
    print("=" * 60)

    return out_dir, best_miou


if __name__ == "__main__":
    main()
