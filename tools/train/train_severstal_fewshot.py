#!/usr/bin/env python3
"""
Episodic Few-Shot 训练 — Severstal 钢铁缺陷检测 | Episodic Few-Shot — Severstal Steel.
========================================================================================

标准 FSS 协议: 每步随机采样一个缺陷类 → K support + 1 query → prototype → decoder → loss。
一 Epoch = N 个 random episodes，不遍历全部图像。

Standard FSS protocol: each step samples a defect class → K support + 1 query
→ prototype → decoder → loss. One epoch = N random episodes.

用法 | Usage::

    # LoRA + Few-Shot (推荐 | Recommended)
    python tools/train/train_severstal_fewshot.py \
        --k-shot 1 3 5 10 20 --seeds 42 123 456 \
        --lora-rank 4 --epochs 50 --device cuda

    # 纯 Few-Shot 基线 (无 LoRA) | Pure Few-Shot baseline (no LoRA)
    python tools/train/train_severstal_fewshot.py --k-shot 1 --seeds 42 --epochs 5 --device cuda

    # 二值模式 (所有缺陷合并为 FG) | Binary mode
    python tools/train/train_severstal_fewshot.py --k-shot 5 --binary --device cuda
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
from adatile.backbone.fastsam_backbone import _collect_lora_modules
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
from adatile.datasets.severstal import SeverstalDataset


# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

IMG_H, IMG_W = 256, 1600  # Severstal native (multiples of 32)
DEFECT_CLASSES = [1, 2, 3, 4]  # 四类缺陷 | Four defect classes


# ═══════════════════════════════════════════════════════════════════
# LoRA 工具 | LoRA Utilities
# ═══════════════════════════════════════════════════════════════════

def reset_conv_lora_weights(backbone: FastSAMBackbone):
    """
    将所有 ConvLoRA 权重重置为零初始化状态 | Reset all ConvLoRA weights to zero-init state.

    每次 (K, seed) run 开始时调用，确保 LoRA 从原始行为开始。
    Called at the start of each (K, seed) run to ensure LoRA starts from original behavior.

    lora_down: Kaiming uniform (保留多样性 | preserve diversity)
    lora_up:   Zero (初始不改变特征 | initially no perturbation)
    """
    lora_modules = _collect_lora_modules(backbone.model.model.model)
    for m in lora_modules:
        torch.nn.init.kaiming_uniform_(m.lora_down.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(m.lora_up.weight)


# ═══════════════════════════════════════════════════════════════════
# 数据增强 | Data Augmentation
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
# Prototype 计算 | Prototype Computation
# ═══════════════════════════════════════════════════════════════════

def compute_prototype(
    backbone: FastSAMBackbone,
    support_images: torch.Tensor,    # [K, 3, H, W]
    support_masks: torch.Tensor,     # [K, H, W] binary FG mask
    device: torch.device,
    allow_grad: bool = False,        # 启用梯度回传 (LoRA 训练时需要)
) -> torch.Tensor:
    """
    从 K 张 support 图像计算 L2-normalized FG prototype。
    Compute L2-normalized FG prototype from K support images.

    每张 support: 提取 P4 特征 → FG masked average pool → mean over K → L2 norm.

    allow_grad=True: 移除 no_grad → 梯度可通过 prototype → backbone 回传到 LoRA 参数。
    allow_grad=True: removes no_grad → gradients flow through prototype → backbone to LoRA params.

    :return: [1280] L2-normalized prototype vector.
    """
    K = support_images.shape[0]
    feats_list = []

    # 上下文管理器: 根据 allow_grad 决定是否阻断梯度
    # Context manager: enable/disable gradient based on allow_grad
    ctx = torch.enable_grad() if allow_grad else torch.no_grad()
    with ctx:
        for i in range(K):
            img = support_images[i:i + 1].to(device)
            mask = support_masks[i].to(device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)

            feats = backbone(img)
            p4 = feats["p4"]  # [1, 1280, H/16, W/16]

            # Resize mask to P4 resolution
            _, _, H_p4, W_p4 = p4.shape
            mask_p4 = F.interpolate(
                mask.unsqueeze(0).float(), size=(H_p4, W_p4), mode="nearest"
            ).squeeze(0)  # [1, H_p4, W_p4]

            fg_area = mask_p4.sum()
            if fg_area > 0:
                proto = (p4.squeeze(0) * mask_p4).sum(dim=(1, 2)) / (fg_area + 1e-8)
                feats_list.append(proto)

    if not feats_list:
        return torch.zeros(1280, device=device)

    proto = torch.stack(feats_list).mean(dim=0)  # [1280]
    return F.normalize(proto, dim=0, p=2)


# ═══════════════════════════════════════════════════════════════════
# Episode 采样器 | Episode Sampler
# ═══════════════════════════════════════════════════════════════════

class EpisodeSampler:
    """
    每步: 随机类 → K support + 1 query (同类别，不同图像，0% overlap)。
    Each step: random class → K support + 1 query (same class, distinct images).

    Parameters
    ----------
    dataset : SeverstalDataset
        训练集 (split="train")。
    class_ids : list[int]
        参与训练的缺陷类 ID | Defect class IDs to sample from.
    k_shot : int
        每次 episode 的 support 图像数 | Support images per episode.
    seed : int
        随机种子 | Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: SeverstalDataset,
        class_ids: list[int] = None,
        k_shot: int = 1,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.class_ids = class_ids or DEFECT_CLASSES
        self.k_shot = k_shot
        self.rng = random.Random(seed)

        # ── 预建每类候选池 | Pre-build per-class candidate pool ──
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
        """
        采样一个 episode | Sample one episode.

        :return: {
            "class_id": int,
            "support_indices": list[int],  # K indices
            "query_index": int,             # 1 index
        }
        """
        cls_id = self.rng.choice(self.class_ids)
        pool = self._class_pool[cls_id]

        # K support + 1 query, all distinct
        sampled = self.rng.sample(pool, self.k_shot + 1)

        return {
            "class_id": cls_id,
            "support_indices": sampled[:self.k_shot],
            "query_index": sampled[self.k_shot],
        }


# ═══════════════════════════════════════════════════════════════════
# 获取每类的 binary mask | Get Per-Class Binary Mask
# ═══════════════════════════════════════════════════════════════════

def get_class_mask(sample: dict, class_id: int) -> torch.Tensor:
    """
    从数据集样本中提取指定类别的二值掩码。
    Extract binary mask for a specific class from a dataset sample.

    多类别模式下 mask 值为 {0, 1, 2, 3, 4} → (mask == class_id) → {0, 1}。
    二值模式下 mask 已是 {0, 1}，直接返回。

    :param sample: dataset[idx] 返回值，包含 "masks" key [1, H, W].
    :param class_id: 缺陷类 ID | Defect class ID.
    :return: [H, W] binary float tensor.
    """
    mask = sample["masks"].squeeze(0)  # [H, W]
    if mask.max() > 1:
        # 多类别模式: 提取指定类 | Multi-class mode: extract specific class
        return (mask == class_id).float()
    else:
        # 二值模式: 直接使用 | Binary mode: use directly
        return mask.float()


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════

def binary_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 5.0,
    alpha: float = 0.75,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    二值 Focal Loss — 极端 FG/BG 不平衡场景 (Severstal: FG ≈ 3%).
    Binary Focal Loss for extreme FG/BG imbalance.

    FL = -α·(1-pt)^γ·log(pt)  for FG
         -(1-α)·pt^γ·log(1-pt) for BG

    γ=5.0 (iSAID 经验值): 极度压低 easy-negative (背景) 的梯度.
    α=0.75: FG 样本权重更高.
    eps=1e-4: 防止 log(0) 梯度爆炸 (非 1e-8，否则梯度可达 1e8).

    :param pred: [...] sigmoid probability ∈ [0, 1].
    :param target: [...] binary float {0, 1} (same shape as pred).
    """
    pred = pred.clamp(eps, 1.0 - eps)

    # pt: probability of the target class
    pt = pred * target + (1.0 - pred) * (1.0 - target)

    # α_t: class-balanced weight
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)

    # Focal modulation: (1 - pt)^γ
    focal_weight = (1.0 - pt) ** gamma

    # Binary cross-entropy: -log(pt)
    bce = -torch.log(pt)

    loss = alpha_t * focal_weight * bce
    return loss.mean()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """二值 Dice 损失 | Binary Dice loss."""
    pred_f = pred.flatten()
    target_f = target.flatten()
    if target_f.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    inter = (pred_f * target_f).sum()
    union = pred_f.sum() + target_f.sum()
    return 1.0 - (2.0 * inter + smooth) / (union + smooth)


# ═══════════════════════════════════════════════════════════════════
# Random Crop (偏向目标类 | biased toward target class)
# ═══════════════════════════════════════════════════════════════════

def random_crop_with_class(
    image: torch.Tensor,     # [3, H, W]
    mask: torch.Tensor,       # [H, W] category labels
    class_id: int,
    crop_h: int = 256,
    crop_w: int = 256,
    max_attempts: int = 30,
    rng: random.Random = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    随机裁剪, 优先选择包含目标类的区域 | Random crop biased toward target class.

    在 max_attempts 次尝试内寻找包含 class_id 的 crop.
    找到 → 返回该 crop. 未找到 → fallback 到随机位置.

    Find a crop containing class_id within max_attempts tries.
    Found → return crop. Not found → fallback to random position.

    :return: (cropped_image [3, crop_h, crop_w], cropped_mask [crop_h, crop_w])
    """
    if rng is None:
        rng = random

    H, W = mask.shape
    h_eff = min(crop_h, H)
    w_eff = min(crop_w, W)

    # 如果图像小于 crop size, 直接返回 | If image smaller than crop, return as-is
    if H <= crop_h and W <= crop_w:
        return image.clone(), mask.clone()

    for _ in range(max_attempts):
        y = rng.randint(0, H - h_eff) if H > h_eff else 0
        x = rng.randint(0, W - w_eff) if W > w_eff else 0
        crop_mask = mask[y:y + h_eff, x:x + w_eff]
        if (crop_mask == class_id).sum() > 0:
            return image[:, y:y + h_eff, x:x + w_eff].clone(), crop_mask.clone()

    # Fallback: random position
    y = rng.randint(0, H - h_eff) if H > h_eff else 0
    x = rng.randint(0, W - w_eff) if W > w_eff else 0
    return image[:, y:y + h_eff, x:x + w_eff].clone(), mask[y:y + h_eff, x:x + w_eff].clone()


# ═══════════════════════════════════════════════════════════════════
# 评估: 基于 prototype 的 per-class IoU + Dice | Evaluation w/ Prototype
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_with_prototypes(
    decoder: AdaptiveSparseDecoder,
    backbone: FastSAMBackbone,
    val_ds: SeverstalDataset,
    device: torch.device,
    class_ids: list[int],
    support_cache: dict[int, dict],  # {cls_id: {"images": [K,3,H,W], "masks": [K,H,W]}}
    binary: bool = False,
    max_samples: int = 0,
) -> dict:
    """
    使用固定 support prototype 在验证集上评估每类 IoU。
    Evaluate per-class IoU on validation set using fixed support prototypes.

    对每类: support → prototype → 所有 val 图预测 binary mask → IoU vs GT。
    For each class: support → prototype → predict on all val images → IoU vs GT.

    :param support_cache: 预先采样的 support 数据 (与训练时一致)。
    :return: {"mIoU": float, "per_class_IoU": dict, "background_IoU": float}
    """
    decoder.eval()
    backbone.eval()

    # ── 预计算所有类的 prototype | Pre-compute prototypes ──
    prototypes: dict[int, torch.Tensor] = {}
    for cls_id in class_ids:
        sc = support_cache[cls_id]
        proto = compute_prototype(
            backbone,
            sc["images"].to(device) if isinstance(sc["images"], torch.Tensor)
                else sc["images"].clone().to(device),
            sc["masks"].to(device) if isinstance(sc["masks"], torch.Tensor)
                else sc["masks"].clone().to(device),
            device,
        )
        prototypes[cls_id] = proto

    num_classes = len(class_ids) + 1  # +1 for background
    per_class_inter = {c: 0.0 for c in [0] + class_ids}
    per_class_union = {c: 0.0 for c in [0] + class_ids}

    indices = list(range(len(val_ds)))
    if max_samples > 0:
        indices = indices[:max_samples]

    for idx in tqdm(indices, desc="Eval", leave=False):
        sample = val_ds[idx]
        img = sample["image"].unsqueeze(0).to(device)
        gt_full = sample["masks"].squeeze(0).to(device)
        H, W = gt_full.shape

        # Extract features once
        feats = backbone(img, extract_proto=True)
        p4 = feats["p4"]         # [1, 1280, H/16, W/16]
        proto_masks = feats["proto"]  # [1, 32, H/4, W/4]

        # ── 每类预测 binary mask | Predict binary mask per class ──
        pred_class = torch.zeros(H, W, dtype=torch.long, device=device)
        for cls_id in class_ids:
            proto = prototypes[cls_id]
            pred_prob = decoder(p4, proto_masks, proto)  # [1, H/4, W/4]
            pred_full = F.interpolate(
                pred_prob.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)  # [H, W]
            pred_binary = (pred_full > 0.5).long()
            # 后到的类不覆盖先到的（按面积排序，大缺陷优先）
            area = pred_binary.sum().item()
            # 冲突区域: 取置信度更高的类 | Conflict: higher confidence wins
            overlap_mask = (pred_binary > 0) & (pred_class > 0)
            if overlap_mask.any():
                # 在重叠区域取置信度更高的类
                overlap_pixels = overlap_mask.nonzero(as_tuple=True)
                for py, px in zip(overlap_pixels[0].tolist(), overlap_pixels[1].tolist()):
                    # 简化: 大缺陷优先 (大面积缺陷更可能置信度高)
                    pass  # 当前简单策略: 后预测的不覆盖先预测的
            pred_class[(pred_binary > 0) & (pred_class == 0)] = cls_id

        # ── 计算 per-class IoU | Compute per-class IoU ──
        for c in [0] + class_ids:
            pc = (pred_class == c)
            gc = _get_gt_binary(gt_full, c, binary)
            inter = (pc & gc).sum().item()
            union = (pc | gc).sum().item()
            per_class_inter[c] += inter
            per_class_union[c] += union

    # ── 汇总 | Summarize ──
    class_names = {0: "background", 1: "Class1", 2: "Class2", 3: "Class3", 4: "Class4"}
    per_class_iou = {}
    for c in [0] + class_ids:
        name = class_names.get(c, f"class_{c}")
        per_class_iou[name] = (
            round(per_class_inter[c] / per_class_union[c], 6)
            if per_class_union[c] > 0 else float("nan")
        )

    valid = [v for v in per_class_iou.values() if v == v]  # filter NaN
    miou = round(float(np.mean(valid)), 6) if valid else 0.0

    # ── Dice (from IoU: Dice = 2*IoU / (1+IoU)) ──
    per_class_dice = {}
    for c in [0] + class_ids:
        name = class_names.get(c, f"class_{c}")
        iou = per_class_iou.get(name, float("nan"))
        per_class_dice[name] = round(2 * iou / (1 + iou), 6) if iou == iou and iou > 0 else 0.0
    fg_dice_vals = [per_class_dice[class_names.get(c, f"class_{c}")]
                    for c in class_ids
                    if per_class_dice[class_names.get(c, f"class_{c}")] > 0]
    mDice = round(float(np.mean(fg_dice_vals)), 6) if fg_dice_vals else 0.0

    return {
        "mIoU": miou,
        "mDice": mDice,
        "per_class_IoU": per_class_iou,
        "per_class_Dice": per_class_dice,
    }


def _get_gt_binary(gt: torch.Tensor, class_id: int, binary: bool) -> torch.Tensor:
    """获取指定类的 GT binary mask。"""
    if binary:
        if class_id == 0:
            return (gt == 0)
        return (gt > 0)
    return (gt == class_id)


# ═══════════════════════════════════════════════════════════════════
# 单次 (K, seed) 训练 | Single (K, seed) Training Run
# ═══════════════════════════════════════════════════════════════════

def train_one_run(
    train_ds: SeverstalDataset,
    val_ds: SeverstalDataset,
    backbone: FastSAMBackbone,
    device: torch.device,
    args: argparse.Namespace,
    logger,
    run_dir: Path,
    k: int,
    seed: int,
    binary: bool,
) -> dict:
    """
    对一个 (K, seed) 组合执行 episodic 训练 | Train one (K, seed) with episodic protocol.

    :return: results dict with best_mIoU, per_class_IoU, etc.
    """
    set_seed(seed)

    class_ids = DEFECT_CLASSES if not binary else [1]  # binary: single FG class
    num_classes = 2 if binary else 5

    # ── Episode sampler ──
    sampler = EpisodeSampler(train_ds, class_ids=class_ids, k_shot=k, seed=seed)
    episodes_per_epoch = args.episodes_per_epoch

    logger.log_info("fewshot/run",
                    f"K={k} seed={seed}: {episodes_per_epoch} episodes/epoch, "
                    f"classes={class_ids}")

    # ── 模型 | Model (fresh init per run) ──
    decoder = AdaptiveSparseDecoder(
        in_channels=1280, proto_dim=32, hidden_dim=256,
        use_fdr=False, normalize_proto="none", out_channels=1,  # binary per-class
    ).to(device)
    decoder_params = sum(p.numel() for p in decoder.parameters())
    logger.log_info("fewshot/model", f"AdaptiveSparseDecoder: {decoder_params/1e3:.1f}K params")

    # ── LoRA 权重重置 (每次 run 从零开始) | Reset LoRA weights (fresh start per run) ──
    lora_rank = getattr(args, 'lora_rank', 0)
    use_lora = lora_rank > 0
    if use_lora:
        reset_conv_lora_weights(backbone)
        lora_params_count = sum(p.numel() for p in backbone.get_lora_parameters())
        logger.log_info("fewshot/lora",
                        f"ConvLoRA rank={lora_rank}: {lora_params_count/1e3:.1f}K params (re-initialized)")

    # ── Optimizer (Decoder + optional LoRA) ──
    optim_params = list(decoder.parameters())
    if use_lora:
        optim_params += backbone.get_lora_parameters()
    total_trainable = sum(p.numel() for p in optim_params)
    logger.log_info("fewshot/model",
                    f"Total trainable: {total_trainable/1e3:.1f}K params "
                    f"(Decoder: {decoder_params/1e3:.1f}K, LoRA: {(total_trainable - decoder_params)/1e3:.1f}K)"
                    if use_lora else
                    f"Total trainable: {total_trainable/1e3:.1f}K params (Decoder only)")

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * episodes_per_epoch)

    # ── 固定 support cache (eval 用) | Fixed support cache (for eval) ──
    eval_support_cache: dict[int, dict] = {}
    eval_rng = random.Random(seed + 10000)  # 独立 RNG | Independent RNG for eval
    for cls_id in class_ids:
        pool = train_ds.class_to_images(cls_id)
        picked = eval_rng.sample(pool, min(k, len(pool)))
        s_imgs, s_masks = [], []
        for idx in picked:
            s = train_ds[idx]
            s_imgs.append(s["image"])
            s_masks.append(get_class_mask(s, cls_id))
        eval_support_cache[cls_id] = {
            "images": torch.stack(s_imgs),
            "masks": torch.stack(s_masks),
        }

    augment = None if args.no_augment else BasicAugmentation()

    # ── 训练循环 | Training Loop ──
    global_step = 0
    best_miou = 0.0
    best_mdice = 0.0
    best_epoch = 0
    best_per_class = {}
    nan_count = 0

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        epoch_losses, epoch_focal_list, epoch_dice_list = [], [], []

        pbar = tqdm(range(episodes_per_epoch), desc=f"K{k}_S{seed} E{epoch:3d}/{args.epochs}",
                    unit="ep", leave=False)
        for _ in pbar:
            # ── 采样 episode | Sample episode ──
            episode = sampler.sample()
            cls_id = episode["class_id"]

            # ── 加载 support (始终全图 — 与 eval 保持一致) | Load support (always full — consistent w/ eval) ──
            s_imgs, s_masks = [], []
            for si in episode["support_indices"]:
                s = train_ds[si]
                s_imgs.append(s["image"])                 # [3, 256, 1600] full image
                s_masks.append(get_class_mask(s, cls_id))  # [256, 1600] binary for target class
            s_imgs = torch.stack(s_imgs)
            s_masks = torch.stack(s_masks)

            # ── 加载 query (crop 提高 FG 密度 | crop to boost FG density) ──
            q = train_ds[episode["query_index"]]
            use_crop = args.crop_size > 0
            if use_crop:
                q_img_full = q["image"]                    # [3, 256, 1600]
                q_mask_full = q["masks"].squeeze(0)         # [256, 1600] category labels
                q_img, q_mask_cat = random_crop_with_class(
                    q_img_full, q_mask_full, cls_id, args.crop_size, args.crop_size)
                q_mask = (q_mask_cat == cls_id).float()
            else:
                q_img = q["image"]
                q_mask = get_class_mask(q, cls_id)
            H, W = q_mask.shape

            # Augment query only
            if augment:
                q_img, q_mask = augment(q_img, q_mask)
                H, W = q_mask.shape

            q_img = q_img.unsqueeze(0).to(device)   # [1, 3, H, W]
            q_mask = q_mask.unsqueeze(0).to(device)  # [1, H, W]

            # ── Prototype (梯度回传当 LoRA 激活时 | Grad enabled when LoRA active) ──
            support_proto = compute_prototype(
                backbone, s_imgs, s_masks, device,
                allow_grad=use_lora,
            )

            # ── Query forward | Query forward ──
            feats = backbone(q_img, extract_proto=True)
            p4 = feats["p4"]              # [1, 1280, H/16, W/16]
            proto_masks = feats["proto"]  # [1, 32, H/4, W/4]

            pred_prob = decoder(p4, proto_masks, support_proto)  # [1, H/4, W/4]
            pred_full = F.interpolate(
                pred_prob.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)  # [H, W]

            target = q_mask.squeeze(0)  # [H, W]

            # ── Loss (Focal + Dice — 极端不平衡 | extreme imbalance) ──
            focal = binary_focal_loss(pred_full, target, gamma=5.0, alpha=0.75)
            dice = dice_loss(pred_full, target)
            loss_val = 0.5 * focal + 0.5 * dice

            if torch.isnan(loss_val) or torch.isinf(loss_val):
                nan_count += 1
                continue

            optimizer.zero_grad()
            loss_val.backward()

            grad_nan = any(
                p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in optim_params  # 检查 Decoder + LoRA | Check Decoder + LoRA
            )
            if grad_nan:
                optimizer.zero_grad()
                nan_count += 1
                continue

            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_losses.append(loss_val.item())
            epoch_focal_list.append(focal.item())
            epoch_dice_list.append(dice.item())

            if epoch_losses:
                pbar.set_postfix({
                    "loss": f"{np.mean(epoch_losses[-20:]):.4f}",
                    "dice": f"{np.mean(epoch_dice_list[-20:]):.4f}",
                    "cls": cls_id,
                })

        # ── Epoch summary ──
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        # ── Evaluation (every eval_every epochs) ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            result = evaluate_with_prototypes(
                decoder, backbone, val_ds, device,
                class_ids=class_ids, support_cache=eval_support_cache,
                binary=binary,
            )
            miou = result["mIoU"]
            mdice = result.get("mDice", 0.0)
            logger.log_info("fewshot/eval",
                            f"K={k} S={seed} E{epoch:3d}: mIoU={miou:.4f} mDice={mdice:.4f}")
            logger.log_metric(f"miou_K{k}_S{seed}", miou, step=epoch, tags=["fewshot"])
            logger.log_metric(f"mdice_K{k}_S{seed}", mdice, step=epoch, tags=["fewshot"])

            if miou > best_miou:
                best_miou = miou
                best_mdice = mdice
                best_epoch = epoch
                best_per_class = result["per_class_IoU"]
                ckpt = {
                    "epoch": epoch, "global_step": global_step,
                    "decoder_state_dict": {
                        k: v.clone() for k, v in decoder.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "mIoU": miou, "mDice": mdice, "per_class_IoU": best_per_class,
                    "k": k, "seed": seed, "lora_rank": lora_rank,
                }
                # ── 保存 LoRA 权重 (用于恢复/分析) | Save LoRA weights (for resume/analysis) ──
                if use_lora:
                    ckpt["lora_state_dict"] = {
                        k: v.clone() for k, v in backbone.model.model.state_dict().items()
                        if any(x in k for x in ["lora_down", "lora_up"])
                    }
                torch.save(ckpt, str(run_dir / "best_model.pt"))

    logger.log_info("fewshot/run_done",
                    f"K={k} S={seed}: best mIoU={best_miou:.4f} @ epoch {best_epoch}, NaN={nan_count}")

    return {
        "k": k, "seed": seed,
        "best_mIoU": best_miou, "best_mDice": best_mdice,
        "best_epoch": best_epoch,
        "per_class_IoU": best_per_class,
        "nan_count": nan_count, "episodes_per_epoch": episodes_per_epoch,
    }


# ═══════════════════════════════════════════════════════════════════
# 命令行参数 | CLI Arguments
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Episodic Few-Shot — Severstal Steel Defect Detection"
    )

    # ── 数据 | Data ──
    p.add_argument("--data-root", type=str, default="data/severstal-steel-defect-detection")
    p.add_argument("--binary", action="store_true",
                   help="二值模式 (所有缺陷合并为 FG) | Binary mode (all defects → FG).")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--crop-size", type=int, default=256,
                   help="训练时随机裁剪尺寸 (default: 256, 0=全图)")

    # ── LoRA | Low-Rank Adaptation ──
    p.add_argument("--lora-rank", type=int, default=4,
                   help="ConvLoRA 秩 (0=禁用, default: 4) | ConvLoRA rank (0=disabled).")
    p.add_argument("--lora-alpha", type=float, default=1.0,
                   help="LoRA 缩放因子 (default: 1.0) | LoRA scaling factor (alpha/rank).")

    # ── 少样本 | Few-Shot ──
    p.add_argument("--k-shot", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                   help="每类 support 图像数 (default: 1 3 5 10 20)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                   help="随机种子 (default: 42 123 456)")
    p.add_argument("--epochs", type=int, default=50,
                   help="训练 epoch 数 (default: 50)")
    p.add_argument("--episodes-per-epoch", type=int, default=200,
                   help="每 epoch 的 episode 数 (default: 200)")

    # ── 训练 | Training ──
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--eval-every", type=int, default=5,
                   help="每 N epoch 评估一次 (default: 5)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(42)
    device = torch.device(args.device)

    mode_str = "binary" if args.binary else "multi"
    class_ids = [1] if args.binary else DEFECT_CLASSES
    class_names = {0: "background", 1: "foreground"} if args.binary \
        else {0: "background", 1: "Class1", 2: "Class2", 3: "Class3", 4: "Class4"}

    # ── 输出目录 | Output dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        lora_tag = f"_LoRA_r{args.lora_rank}" if args.lora_rank > 0 else "_NoLoRA"
        args.output_dir = f"runs/severstal_episodic_{mode_str}{lora_tag}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ──
    logger = get_logger("train_severstal_fewshot")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))
    logger.log_info("config",
                    f"Severstal Episodic Few-Shot | K={args.k_shot} | seeds={args.seeds}")
    logger.log_info("config",
                    f"Mode={mode_str} | Crop={args.crop_size if args.crop_size>0 else 'full'} | "
                    f"Epochs={args.epochs} | "
                    f"Episodes/epoch={args.episodes_per_epoch} | lr={args.lr} | "
                    f"LoRA rank={args.lora_rank}")

    # ── 数据 | Data ──
    train_ds = SeverstalDataset(root=args.data_root, split="train",
                                binary=args.binary, seed=42)
    val_ds = SeverstalDataset(root=args.data_root, split="val",
                              binary=args.binary, seed=42)
    logger.log_info("data", f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    for cls_id in class_ids:
        logger.log_info("data",
                        f"  Class {cls_id}: {len(train_ds.class_to_images(cls_id))} images")

    # ── Backbone (frozen, shared) ──
    backbone = FastSAMBackbone(
        freeze_backbone=True,
        checkpoint="thirdLibrary/FastSAM/weights/FastSAM-x.pt",
    ).to(device)
    backbone.eval()
    with torch.no_grad():
        backbone(torch.randn(1, 3, IMG_H, IMG_W, device=device), extract_proto=True)
    logger.log_info("model", "Backbone: FastSAM-x (frozen)")

    # ── ConvLoRA 注入 (一次性架构修改, 每次 run 重置权重) ──
    # Inject ConvLoRA once (architectural change), weights reset per run
    if args.lora_rank > 0:
        lora_n = backbone.apply_conv_lora(rank=args.lora_rank, alpha=args.lora_alpha)
        logger.log_info("model",
                        f"ConvLoRA injected: rank={args.lora_rank}, alpha={args.lora_alpha}, "
                        f"+{lora_n:,} trainable params ({lora_n/1e3:.1f}K)")
    else:
        logger.log_info("model", "LoRA disabled (--lora-rank 0) — pure frozen backbone")

    # ── 验证 K 值 | Validate K values ──
    for k in args.k_shot:
        for cls_id in class_ids:
            n = len(train_ds.class_to_images(cls_id))
            if k + 1 > n:  # need K+1 for distinct support+query
                logger.log_info("warning",
                                f"K={k} needs {k+1} images for class {cls_id}, "
                                f"but only {n} available. Skipping.")
                # Remove from sweep
                args.k_shot = [x for x in args.k_shot if x != k]
                break

    if not args.k_shot:
        logger.log_info("error", "No valid K values after validation. Exiting.")
        return

    # ── 扫参 | Sweep ──
    runs = [(k, s) for k in args.k_shot for s in args.seeds]
    all_results = []
    logger.log_info("sweep", f"Starting sweep: {len(runs)} runs "
                    f"({len(args.k_shot)} K × {len(args.seeds)} seeds)")

    for run_idx, (k, seed) in enumerate(runs):
        logger.log_info("sweep", f"[{run_idx+1}/{len(runs)}] K={k}, seed={seed}")

        run_dir = out_dir / f"K{k}_S{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = train_one_run(
                train_ds=train_ds, val_ds=val_ds,
                backbone=backbone, device=device,
                args=args, logger=logger, run_dir=run_dir,
                k=k, seed=seed, binary=args.binary,
            )
            all_results.append(result)
        except ValueError as e:
            logger.log_info("sweep", f"  SKIP: {e}")
            continue

    if not all_results:
        logger.log_info("done", "No results to report.")
        return

    # ── 汇总 | Summary ──
    summary_by_k = {}
    for k in sorted(set(r["k"] for r in all_results)):
        k_results = [r for r in all_results if r["k"] == k]
        mious = [r["best_mIoU"] for r in k_results]
        mdices = [r.get("best_mDice", 0.0) for r in k_results]
        summary_by_k[str(k)] = {
            "mean_mIoU": round(float(np.mean(mious)), 4),
            "std_mIoU": round(float(np.std(mious)), 4) if len(mious) > 1 else 0.0,
            "min_mIoU": round(float(np.min(mious)), 4),
            "max_mIoU": round(float(np.max(mious)), 4),
            "mean_mDice": round(float(np.mean(mdices)), 4),
            "n_runs": len(k_results),
        }

    # ── 打印汇总表 | Print Table ──
    col_names = [class_names.get(c, f"c{c}") for c in [0] + class_ids]
    header = (f"{'K':>3} {'Seed':>4} | {'mIoU':>7} {'mDice':>7} | "
              + " | ".join(f"{n:>7}" for n in col_names)
              + f" | {'Ep':>5}")
    sep = "-" * len(header.expandtabs())

    print(f"\n{'='*len(header.expandtabs())}")
    print(f"  Severstal Episodic Few-Shot — AdaptiveSparseDecoder")
    print(f"  Crop: {args.crop_size if args.crop_size > 0 else 'full image'} | "
          f"LoRA rank: {args.lora_rank}")
    print(f"{'='*len(header.expandtabs())}")
    print(header)
    print(sep)

    for r in all_results:
        p = r.get("per_class_IoU", r.get("per_class_IoU", {}))
        mdice = r.get("best_mDice", 0.0)
        vals = " | ".join(f"{p.get(class_names.get(c, f'c{c}'), float('nan')):7.4f}"
                          for c in [0] + class_ids)
        print(f"{r['k']:3d} {r['seed']:4d} | {r['best_mIoU']:7.4f} {mdice:7.4f} | {vals} | {r['best_epoch']:5d}")

    print(sep)
    for k_str, s in summary_by_k.items():
        mean_dice = s.get("mean_mDice", 0.0)
        print(f"{k_str:>3} {'mean':>4} | {s['mean_mIoU']:7.4f} {mean_dice:7.4f} | ({s['std_mIoU']:.4f})")
    print(f"{'='*len(header.expandtabs())}\n")

    # ── results.json ──
    results_json = {
        "experiment": "Severstal Episodic Few-Shot",
        "timestamp": datetime.now().isoformat(),
        "mode": mode_str,
        "config": vars(args),
        "runs": all_results,
        "summary_by_k": summary_by_k,
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False, default=str)
    logger.log_info("done", f"Results saved to {json_path}")
    print(f"Results: {json_path}")


if __name__ == "__main__":
    main()
