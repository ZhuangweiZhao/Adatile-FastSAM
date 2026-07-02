#!/usr/bin/env python3
"""
全监督语义分割训练 | Fully-Supervised Semantic Segmentation Training.
=====================================================================

在 iSAID-5i 标准数据集上，使用冻结的 FastSAM Backbone + LightDecoder 进行全监督训练。
目的：测量 FastSAM P4 特征的全监督分割性能上限 (Experiment A)。

Fully-supervised training on iSAID-5i with frozen FastSAM backbone + LightDecoder.
Purpose: measure the upper bound of FastSAM P4 features for semantic segmentation (Exp A).

支持两种模式 | Two modes:
    Full mode (--fold -1):       所有 15 类，所有 train tiles 参与训练
                                 All 15 classes, all train tiles for training
    Base-only mode (--fold 0/1/2): 仅 Base 类，Novel 类像素设为 ignore (255)
                                   Only Base classes, Novel pixels set to ignore (255)

与 Few-Shot 训练的对比价值 | Comparison value vs Few-Shot:
    A. 全监督 Base mIoU → P4 特征上限 | P4 feature upper bound
    B. Episode Base mIoU → Few-shot 机制效率 | Few-shot mechanism efficiency
    C. Episode Novel mIoU → Novel 泛化能力 | Novel generalization ability

用法 | Usage::
    # Full mode: 所有类全监督 | All classes, full supervision
    python tools/train/train_supervised.py --fold -1

    # Base-only mode: Fold 0, 只训练 Base 类 | Base-only: Fold 0
    python tools/train/train_supervised.py --fold 0 --epochs 50 --batch-size 16

    # 指定设备 + 输出目录 | Specify device + output dir
    python tools/train/train_supervised.py --fold -1 --device cuda:0 --output-dir runs/supervised_expA
"""

from __future__ import annotations

import sys, argparse, json, os
from pathlib import Path
from collections import defaultdict
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed, get_worker_init_fn
from adatile.utils.env import get_env_info, save_env_info
from adatile.backbone import FastSAMBackbone
from adatile.decoder.light_decoder import LightDecoder, LightDecoderP3P4
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, ISAID5I_FOLDS

# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

NUM_CLASSES = 15       # 前景类别数 | Number of foreground classes
NUM_OUT_CH = 16        # 输出通道数 (15 FG + 1 BG) | Output channels
IGNORE_INDEX = 255     # 忽略标签 | Ignore label (for CrossEntropyLoss)

# 默认数据根目录 | Default data root
DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "data" / "iSAID-5i" / "iSAID")


# ═══════════════════════════════════════════════════════════════════
# 数据集 | Dataset
# ═══════════════════════════════════════════════════════════════════

class SupervisedSegDataset(Dataset):
    """
    iSAID-5i 全监督分割数据集 | iSAID-5i Fully-Supervised Segmentation Dataset.

    从 iSAID-5i 标准 split 文件加载 256×256 tiles，直接返回 (image, mask) 对。
    Loads 256×256 tiles from iSAID-5i standard split files, returns (image, mask) pairs.

    Parameters
    ----------
    root : str
        iSAID-5i 数据根目录 (e.g. "data/iSAID-5i/iSAID").
    split : str
        "train" 或 "val".
    fold : int
        Fold ID (0/1/2) 或 -1 (全量模式 | full mode, 不使用 Base/Novel 过滤).
    novel_ids : list[int] | None
        Novel 类 ID 列表。训练时这些类别的像素会被设为 IGNORE_INDEX。
        List of Novel class IDs. Pixels of these classes are set to IGNORE_INDEX during training.
        None → 不忽略任何类别 (全量模式 | full mode).
    """

    def __init__(
        self,
        root: str = DEFAULT_DATA_ROOT,
        split: str = "train",
        fold: int = -1,
        novel_ids: list[int] | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.fold = fold
        self.novel_ids = set(novel_ids) if novel_ids else set()

        # ── 路径设置 | Path setup ──
        self._img_dir = self.root / split / "images"
        self._mask_dir = self.root / split / "semantic_png"

        if not self._img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self._img_dir}")
        if not self._mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self._mask_dir}")

        # ── 加载 split 文件 | Load split file ──
        list_dir = self.root / split / f"{split}_list"
        if fold >= 0:
            list_file = list_dir / f"split{fold}_{split}.txt"
        else:
            # Full mode: 不使用 split 文件，直接扫描 images/ 目录
            # Full mode: scan images/ directory directly
            list_file = None

        if list_file and list_file.exists():
            with open(list_file) as f:
                raw_names = [line.strip() for line in f if line.strip()]
            self._tile_names = []
            for raw in raw_names:
                clean = self._clean_tile_name(raw)
                if clean:
                    self._tile_names.append(clean)
        else:
            # Full mode: 扫描所有 .png 文件 | Full mode: scan all .png files
            self._tile_names = sorted([
                p.stem for p in self._img_dir.glob("*.png")
                if not p.stem.endswith("_instance_color_RGB")
                and not p.stem.endswith("_instance_id_RGB")
            ])

        # ── 验证图像-掩码一致性 | Validate image-mask consistency ──
        valid_names = []
        missing_img, missing_mask = 0, 0
        for name in self._tile_names:
            img_path = self._get_img_path(name)
            mask_path = self._get_mask_path(name)
            if not img_path.exists():
                missing_img += 1
                continue
            if not mask_path.exists():
                missing_mask += 1
                continue
            valid_names.append(name)

        if missing_img > 0 or missing_mask > 0:
            print(f"[SupervisedSegDataset] WARNING: {missing_img} missing images, "
                  f"{missing_mask} missing masks (skipped)")

        self._tile_names = valid_names

        # ── 日志 | Log ──
        mode_str = f"fold={fold}, base-only" if fold >= 0 else "full (all 15 classes)"
        print(f"[SupervisedSegDataset] {split}: {len(self._tile_names)} tiles, "
              f"{mode_str}, novel_ids={sorted(self.novel_ids) if self.novel_ids else 'none'}")

    # ── 文件名处理 | Filename handling ──

    @staticmethod
    def _clean_tile_name(raw: str) -> str | None:
        """从 split 文件行提取干净的 tile 名 | Extract clean tile name from split file line."""
        raw = raw.strip()
        for suffix in ["_instance_color_RGB.png", "_instance_id_RGB.png", ".png"]:
            idx = raw.find(suffix)
            if idx > 0:
                return raw[:idx]
        return raw.rsplit(".", 1)[0] if "." in raw else raw

    def _get_img_path(self, tile_name: str) -> Path:
        """获取图像路径 | Get image path."""
        p = self._img_dir / f"{tile_name}.png"
        if p.exists():
            return p
        return self._img_dir / f"{tile_name}.jpg"

    def _get_mask_path(self, tile_name: str) -> Path:
        """获取语义掩码路径 | Get semantic mask path."""
        # 优先查找 _instance_color_RGB.png 后缀 | Prefer _instance_color_RGB.png suffix
        p = self._mask_dir / f"{tile_name}_instance_color_RGB.png"
        if p.exists():
            return p
        p = self._mask_dir / f"{tile_name}.png"
        if p.exists():
            return p
        return self._mask_dir / f"{tile_name}_instance_color_RGB.png"

    # ── 数据加载 | Data loading ──

    def __len__(self) -> int:
        return len(self._tile_names)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        返回单个 (image, mask) 对 | Returns a single (image, mask) pair.

        :return: {"image": [3, 256, 256] float32, "mask": [256, 256] int64}
        """
        tile_name = self._tile_names[idx]

        # ── 加载图像 | Load image ──
        img_path = self._get_img_path(tile_name)
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # [3, H, W]

        # ── 加载掩码 | Load mask ──
        mask_path = self._get_mask_path(tile_name)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")
        if mask.ndim == 3:
            # RGB mask → 取第一个通道 (所有通道值相同) | RGB mask → take first channel
            mask = mask[:, :, 0]
        mask_tensor = torch.from_numpy(mask.astype(np.int64))  # [H, W]

        # ── Novel 类像素 → IGNORE_INDEX | Novel class pixels → ignore ──
        if self.novel_ids:
            for nid in self.novel_ids:
                mask_tensor[mask_tensor == nid] = IGNORE_INDEX

        return {"image": img_tensor, "mask": mask_tensor, "tile_name": tile_name}


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════

def compute_class_weights(
    dataset: Dataset,
    num_classes: int = NUM_OUT_CH,
    cap: float = 10.0,
) -> torch.Tensor:
    """
    从数据集计算类别平衡权重 (Inverse Sqrt Frequency + Cap)。
    Compute class-balanced weights from dataset (Inverse Sqrt Frequency + Cap).

    公式 | Formula:
        w_c = min(1 / sqrt(freq_c + ε), cap)
        归一化到 mean=1.0 | Normalized to mean=1.0.

    对于极端长尾分布 (如 helicopter 1380px vs ship 43M px)，
    sqrt 提供平滑 + cap 防止稀有类权重爆炸。
    For extreme long-tail (e.g. helicopter 1380px vs ship 43M px),
    sqrt provides smoothing + cap prevents rare-class weight explosion.

    :param dataset: 训练数据集 | Training dataset.
    :param num_classes: 类别总数 (含 BG) | Total classes (incl. BG).
    :param cap: 权重上限 (默认 10.0) | Weight cap (default 10.0).
    :return: [num_classes] 类别权重张量 (BG=0.0) | Class weight tensor (BG=0.0).
    """
    # 统计每类前景像素 | Count per-class foreground pixels
    cls_pixels = defaultdict(int)
    total_fg = 0
    print(f"[ClassWeights] Scanning {len(dataset)} tiles for class distribution...")
    for i in tqdm(range(len(dataset)), desc="  Computing class weights"):
        mask = dataset[i]["mask"].numpy()
        for c in range(1, num_classes):
            count = int((mask == c).sum())
            if count > 0:
                cls_pixels[c] += count
                total_fg += count

    # Inverse sqrt frequency | 逆 sqrt 频率
    weights = torch.ones(num_classes)
    weights[0] = 0.0  # BG 权重临时占位，后面统一设置 | BG weight placeholder, set below

    eps = 1e-8
    raw_weights = {}
    for c in range(1, num_classes):
        if cls_pixels.get(c, 0) > 0:
            freq = cls_pixels[c] / max(total_fg, 1)
            raw_weights[c] = 1.0 / (freq ** 0.5 + eps)
        else:
            raw_weights[c] = cap  # 无数据 → 最大权重 | No data → max weight

    # 归一化：使用中位数频率类的权重作为锚点 (median anchor = 1.0)
    # Normalize: use median-frequency class weight as anchor (anchor = 1.0)
    sorted_weights = sorted(raw_weights.values())
    median_raw = sorted_weights[len(sorted_weights) // 2]

    for c in range(1, num_classes):
        w = raw_weights.get(c, cap) / max(median_raw, eps)
        weights[c] = min(w, cap)

    # BG 权重: 设为中位数类的权重 (1.0)，保证模型仍然学到"什么是背景"
    # BG weight: set to median class weight (1.0), ensuring model still learns background
    # 如果设为 0，CrossEntropy 对 BG 像素无惩罚 → 模型乱猜前景 → pixel_acc 崩溃
    # If weight=0, CE has no penalty for BG pixels → model guesses random FG → pixel_acc collapse
    weights[0] = 1.0

    print(f"[ClassWeights] Weight range: "
          f"min={weights[1:].min().item():.2f}, "
          f"max={weights[1:].max().item():.2f}, "
          f"mean={weights[1:].mean().item():.2f}")

    # 打印每类权重 (仅显示极端值) | Print per-class weights (highlight extremes)
    for c in range(1, num_classes):
        if weights[c] >= cap or weights[c] <= 1.0 / cap:
            marker = " ***" if weights[c] >= cap else " (low)"
            print(f"  cls{c}: freq={cls_pixels.get(c, 0)/max(total_fg,1)*100:.4f}% "
                  f"weight={weights[c]:.2f}{marker}")

    return weights


def compute_loss(
    logit: torch.Tensor,    # [B, 16, H, W] raw logits
    target: torch.Tensor,   # [B, H, W] class indices, 255=ignore
    num_classes: int = NUM_OUT_CH,
    ignore_index: int = IGNORE_INDEX,
    focal_gamma: float = 5.0,
    class_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    组合损失: 0.5 * Focal(γ=5) + 0.5 * Dice (支持类别平衡)。
    Combined loss: 0.5 * Focal(γ=5) + 0.5 * Dice (class-balanced).

    Focal Loss 对难例加权，缓解遥感场景的极端类别不平衡。
    Dice Loss 直接优化前景类的重叠区域。
    Focal Loss weights hard examples, alleviates extreme class imbalance in remote sensing.
    Dice Loss directly optimizes foreground class overlap.

    :param class_weight: [num_classes] 类别权重 (BG 应为 0.0)。None → 无平衡。
        [num_classes] class weights (BG should be 0.0). None → no balancing.
    :return: (total_loss, {"focal": float, "dice": float})
    """
    # ── Focal Loss | 焦点损失 ──
    # CrossEntropy 计算逐像素 CE (支持 class_weight) | CrossEntropy with class_weight
    ce = F.cross_entropy(
        logit, target,
        weight=class_weight.to(logit.device) if class_weight is not None else None,
        ignore_index=ignore_index,
        reduction="none",
    )
    # Focal weight: (1 - exp(-ce))^γ 加重难例 | upweight hard examples
    focal_loss = ((1.0 - torch.exp(-ce)) ** focal_gamma * ce).mean()

    # ── Dice Loss | Dice 损失 ──
    # 逐前景类 (c=1..15) 计算 Dice，忽略 BG (c=0) | Per foreground class, ignore BG
    # 稀有类在 Dice 中也获得更高权重 | Rare classes also get higher Dice weight
    probs = F.softmax(logit, dim=1)  # [B, C, H, W]
    dice_sum, valid_dice = 0.0, 0
    for c in range(1, num_classes):
        p_c = probs[:, c]                       # [B, H, W]
        t_c = (target == c).float()              # [B, H, W]
        inter = (p_c * t_c).sum()
        union = p_c.sum() + t_c.sum() + 1e-8
        if t_c.sum() > 0:
            # 类别平衡 Dice: 稀有类获得更高权重 | Class-balanced Dice: rare classes upweighted
            cls_w = class_weight[c].item() if class_weight is not None else 1.0
            dice_sum += cls_w * 2.0 * inter / union
            valid_dice += 1  # 不计权重，保持 valid count 为实际类数

    dice_loss = 1.0 - (dice_sum / max(valid_dice, 1))

    # ── 组合 (1:1 权重) | Combined (1:1 weight) ──
    total_loss = 0.5 * focal_loss + 0.5 * dice_loss

    return total_loss, {"focal": focal_loss.item(), "dice": dice_loss.item()}


# ═══════════════════════════════════════════════════════════════════
# 评估 | Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(
    logit: torch.Tensor,   # [B, 16, H, W]
    target: torch.Tensor,  # [B, H, W], 255=ignore
    num_classes: int = NUM_OUT_CH,
) -> dict[str, float]:
    """
    计算多类别分割指标 | Compute multi-class segmentation metrics.

    :return: dict with keys:
        - "miou": 前景类 mean IoU (仅计算 GT 中存在的类别) | FG mean IoU
        - "per_class_iou": dict{cid: IoU} 每类 IoU
        - "pixel_acc": 像素准确率 (排除 ignore) | Pixel accuracy (excl. ignore)
        - "fg_mean_iou": 前景类 mean IoU (同 miou) | Same as miou
    """
    pred = logit.argmax(dim=1)  # [B, H, W]

    per_class_inter = defaultdict(float)
    per_class_union = defaultdict(float)
    correct, total = 0, 0

    for c in range(1, num_classes):
        pc = (pred == c)
        tc = (target == c)
        per_class_inter[c] += (pc & tc).sum().float().item()
        per_class_union[c] += (pc | tc).sum().float().item()

    # 像素准确率 (排除 ignore) | Pixel accuracy (excl. ignore)
    valid_mask = (target != IGNORE_INDEX)
    correct = (pred[valid_mask] == target[valid_mask]).sum().float().item()
    total = valid_mask.sum().float().item()

    # 计算每个前景类的 IoU | Compute per-foreground-class IoU
    per_class_iou = {}
    miou_sum, miou_valid = 0.0, 0
    for c in range(1, num_classes):
        if per_class_union[c] > 0:
            iou_c = per_class_inter[c] / per_class_union[c]
            per_class_iou[c] = iou_c
            miou_sum += iou_c
            miou_valid += 1

    miou = miou_sum / max(miou_valid, 1)
    pixel_acc = correct / max(total, 1)

    return {
        "miou": miou,
        "fg_mean_iou": miou,
        "per_class_iou": per_class_iou,
        "pixel_acc": pixel_acc,
        "n_valid_classes": miou_valid,
    }


# ═══════════════════════════════════════════════════════════════════
# 训练与验证循环 | Training & Validation Loops
# ═══════════════════════════════════════════════════════════════════

def train_epoch(
    decoder: nn.Module,
    backbone: FastSAMBackbone,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    logger,
    class_weight: torch.Tensor | None = None,
    tag: str = "supervised/train",
) -> float:
    """
    训练一个 epoch | Train one epoch.

    :param class_weight: [num_classes] 类别平衡权重 (BG=0) | Class-balanced weights (BG=0).
    :return: 平均损失 | Average loss.
    """
    decoder.train()
    total_loss, total_focal, total_dice, n_batches = 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"  Train E{epoch}/{total_epochs}", leave=False)
    for batch in pbar:
        img = batch["image"].to(device, non_blocking=True)     # [B, 3, H, W]
        tgt = batch["mask"].to(device, non_blocking=True)      # [B, H, W]

        # 前向传播 | Forward pass
        feats = backbone(img)                                  # {"p4": [B, 1280, H/16, W/16], ...}
        logit = decoder(feats, target_size=tgt.shape[1:])      # [B, 16, H, W]

        # 计算损失 | Compute loss
        loss, loss_dict = compute_loss(logit, tgt, class_weight=class_weight)

        # 反向传播 | Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_focal += loss_dict["focal"]
        total_dice += loss_dict["dice"]
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    avg_focal = total_focal / max(n_batches, 1)
    avg_dice = total_dice / max(n_batches, 1)

    logger.log_info(tag,
        f"E{epoch:3d}/{total_epochs} train | loss={avg_loss:.4f} "
        f"focal={avg_focal:.4f} dice={avg_dice:.4f}",
    )
    return avg_loss


@torch.no_grad()
def validate(
    decoder: nn.Module,
    backbone: FastSAMBackbone,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    eval_class_ids: list[int],
    category_names: dict[int, str],
    logger,
    tag: str = "supervised/val",
) -> dict:
    """
    验证 | Validate.

    计算 mIoU、per-class IoU、pixel accuracy。
    Computes mIoU, per-class IoU, pixel accuracy.

    :param eval_class_ids: 需要评估的前景类 ID 列表 (Base 模式则仅含 Base 类)
                           List of foreground class IDs to evaluate (Base-only uses only Base classes)
    :param category_names: {class_id: name} 映射 | Category name mapping.
    :return: {"miou": float, "per_class_iou": dict, "pixel_acc": float}
    """
    decoder.eval()

    # 累积 per-class inter/union (仅评估指定的类别) | Accumulate per-class inter/union (evaluated classes only)
    eval_set = set(eval_class_ids)
    per_class_inter = {c: 0.0 for c in eval_set}
    per_class_union = {c: 0.0 for c in eval_set}
    correct, total = 0, 0

    for batch in tqdm(loader, desc=f"  Val  E{epoch}/{total_epochs}", leave=False):
        img = batch["image"].to(device, non_blocking=True)
        tgt = batch["mask"].to(device, non_blocking=True)

        feats = backbone(img)
        logit = decoder(feats, target_size=tgt.shape[1:])
        pred = logit.argmax(dim=1)

        # 仅评估指定类别 | Only evaluate specified classes
        for c in eval_set:
            pc = (pred == c)
            tc = (tgt == c)
            per_class_inter[c] += (pc & tc).sum().float().item()
            per_class_union[c] += (pc | tc).sum().float().item()

        # 像素准确率 (排除 ignore + Novel 类) | Pixel accuracy (excl. ignore + Novel classes)
        valid_mask = (tgt != IGNORE_INDEX) & torch.isin(tgt, torch.tensor(
            [0] + list(eval_set), device=device))
        correct += (pred[valid_mask] == tgt[valid_mask]).sum().float().item()
        total += valid_mask.sum().float().item()

    # ── 计算 mIoU | Compute mIoU ──
    per_class_iou = {}
    miou_sum, miou_valid = 0.0, 0
    for c in eval_set:
        if per_class_union[c] > 0:
            iou_c = per_class_inter[c] / per_class_union[c]
            per_class_iou[c] = iou_c
            miou_sum += iou_c
            miou_valid += 1

    miou = miou_sum / max(miou_valid, 1)
    pixel_acc = correct / max(total, 1)

    # ── 日志 | Log ──
    cls_str = ", ".join(
        f"{category_names.get(c, f'cls{c}')}={per_class_iou.get(c, 0):.3f}"
        for c in sorted(eval_set)
    )
    logger.log_info(tag,
        f"E{epoch:3d}/{total_epochs} val | mIoU={miou:.4f} ({miou_valid}/{len(eval_set)} classes) "
        f"acc={pixel_acc:.4f} | {cls_str}",
    )

    return {
        "miou": miou,
        "fg_mean_iou": miou,
        "per_class_iou": per_class_iou,
        "pixel_acc": pixel_acc,
        "n_valid_classes": miou_valid,
    }


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数 | Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="全监督语义分割训练 | Fully-Supervised Semantic Segmentation Training")

    # ── 数据 | Data ──
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                   help=f"iSAID-5i 数据根目录 | iSAID-5i data root (default: {DEFAULT_DATA_ROOT})")
    p.add_argument("--fold", type=int, default=-1,
                   choices=[-1, 0, 1, 2],
                   help="Fold ID: -1=全量所有类, 0/1/2=仅 Base 类 | "
                        "-1=all classes, 0/1/2=Base-only")

    # ── 模型 | Model ──
    p.add_argument("--freeze-backbone", action="store_true", default=True,
                   help="冻结 FastSAM backbone (默认: True) | Freeze FastSAM backbone (default: True)")
    p.add_argument("--no-freeze-backbone", action="store_false", dest="freeze_backbone",
                   help="解冻 backbone | Unfreeze backbone")
    p.add_argument("--load-ckpt", type=str, default=None,
                   help="从 checkpoint 恢复训练 | Resume training from checkpoint")

    # ── 训练 | Training ──
    p.add_argument("--epochs", type=int, default=50,
                   help="训练轮数 | Training epochs (default: 50)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="批次大小 | Batch size (default: 16)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="学习率 | Learning rate (default: 1e-3)")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="权重衰减 | Weight decay (default: 1e-4)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers (default: 4)")
    p.add_argument("--focal-gamma", type=float, default=5.0,
                   help="Focal Loss gamma (default: 5.0, 遥感场景推荐 | RS recommended)")
    p.add_argument("--no-class-balance", action="store_true",
                   help="禁用类别平衡损失 | Disable class-balanced loss")
    p.add_argument("--class-weight-cap", type=float, default=10.0,
                   help="类别权重上限 (default: 10.0) | Class weight cap (default: 10.0)")
    p.add_argument("--use-p3", action="store_true",
                   help="使用 P3+P4 多尺度融合解码器 (LightDecoderP3P4) | Use P3+P4 multi-scale decoder")

    # ── 硬件 | Hardware ──
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="运行设备 | Device")
    p.add_argument("--amp", action="store_true",
                   help="启用混合精度训练 | Enable AMP mixed precision training")

    # ── 输出 | Output ──
    p.add_argument("--output-dir", type=str, default=None,
                   help="输出目录 | Output directory (default: runs/supervised_<timestamp>)")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子 | Random seed (default: 42)")
    p.add_argument("--save-every", type=int, default=0,
                   help="每 N epoch 保存一次 checkpoint (0=仅最佳) | Save ckpt every N epochs (0=best only)")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ── 输出目录 | Output directory ──
    if args.output_dir is None:
        mode_str = f"fold{args.fold}" if args.fold >= 0 else "full"
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"runs/supervised_{mode_str}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志系统 | Logging system ──
    logger = get_logger("supervised")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))

    # ── 类别配置 | Category configuration ──
    if args.fold >= 0:
        # Base-only 模式 | Base-only mode
        novel_ids = ISAID5I_FOLDS[args.fold]["novel"]
        base_ids = ISAID5I_FOLDS[args.fold]["base"]
        eval_class_ids = base_ids  # 仅评估 Base 类 | Only evaluate Base classes
        mode_desc = f"Base-only (fold={args.fold})"
    else:
        # 全量模式 | Full mode
        novel_ids = []
        base_ids = list(range(1, NUM_CLASSES + 1))  # 所有 15 类 | All 15 classes
        eval_class_ids = base_ids
        mode_desc = "Full (all 15 classes)"

    logger.log_info("supervised/config",
        f"Mode: {mode_desc} | 模式: {mode_desc}")
    logger.log_info("supervised/config",
        f"Base classes ({len(base_ids)}): "
        f"{[ISAID5I_CATEGORIES[c] for c in base_ids if c in ISAID5I_CATEGORIES]}")
    if novel_ids:
        logger.log_info("supervised/config",
            f"Novel classes ({len(novel_ids)}): "
            f"{[ISAID5I_CATEGORIES[c] for c in novel_ids if c in ISAID5I_CATEGORIES]} "
            f"→ pixels set to IGNORE ({IGNORE_INDEX})")

    # ── 构建数据集 | Build datasets ──
    logger.log_info("supervised/data", "Loading datasets...")

    train_ds = SupervisedSegDataset(
        root=args.data_root, split="train", fold=args.fold,
        novel_ids=novel_ids,
    )
    val_ds = SupervisedSegDataset(
        root=args.data_root, split="val", fold=args.fold,
        novel_ids=novel_ids,
    )

    # ── 类别平衡权重 | Class-balanced weights ──
    class_weight = None
    if not args.no_class_balance:
        logger.log_info("supervised/data", "Computing class-balanced weights...")
        class_weight = compute_class_weights(train_ds, num_classes=NUM_OUT_CH, cap=args.class_weight_cap)
        logger.log_info("supervised/data",
            f"Class-balanced weights enabled (cap={args.class_weight_cap}) | "
            f"类别平衡损失已启用")
    else:
        logger.log_info("supervised/data", "Class-balanced weights DISABLED | 类别平衡已禁用")

    # DataLoader 配置 | DataLoader config
    # Windows multiprocessing uses "spawn" → closures can't be pickled.
    # Windows 多进程使用 "spawn" → 闭包无法 pickle，自动回退到单进程。
    is_windows = (sys.platform == "win32")
    if is_windows and args.num_workers > 0:
        logger.log_info("supervised/data",
            f"Windows detected, overriding num_workers {args.num_workers} -> 0 "
            f"(spawn cannot pickle closures) | Windows 检测到，回退到单进程 num_workers=0")
        args.num_workers = 0

    if is_windows:
        worker_init_fn = None  # Windows spawn cannot pickle closures
    else:
        worker_init_fn = get_worker_init_fn(args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        worker_init_fn=worker_init_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.num_workers, 2), pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    logger.log_info("supervised/data",
        f"Train: {len(train_ds)} tiles, Val: {len(val_ds)} tiles, "
        f"batch_size={args.batch_size}")

    # ── 构建模型 | Build model ──
    logger.log_info("supervised/model", "Building model...")
    backbone = FastSAMBackbone(freeze_backbone=args.freeze_backbone).to(device).eval()

    if args.use_p3:
        # P3+P4 多尺度融合解码器 | P3+P4 multi-scale fusion decoder
        # 自动探测 P3/P4 通道数 | Auto-detect P3/P4 channel counts
        with torch.no_grad():
            probe = backbone(torch.randn(1, 3, 256, 256).to(device))
            p3_ch = probe["p3"].shape[1]
            p4_ch = probe["p4"].shape[1]
        decoder = LightDecoderP3P4(
            p3_channels=p3_ch, p4_channels=p4_ch, num_classes=NUM_OUT_CH,
        ).to(device)
        logger.log_info("supervised/model",
            f"Using LightDecoderP3P4: p3_ch={p3_ch}, p4_ch={p4_ch}")
    else:
        decoder = LightDecoder(in_channels=1280, num_classes=NUM_OUT_CH).to(device)

    n_decoder_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    n_backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.log_info("supervised/model",
        f"Decoder: {n_decoder_params:,} trainable params | "
        f"Backbone: {n_backbone_params:,} trainable params | "
        f"Total: {n_decoder_params + n_backbone_params:,} trainable params")

    # ── 恢复训练 | Resume from checkpoint ──
    start_epoch = 1
    if args.load_ckpt:
        logger.log_info("supervised/model", f"Loading checkpoint: {args.load_ckpt}")
        ckpt = torch.load(args.load_ckpt, map_location=device)
        decoder.load_state_dict(ckpt["decoder_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.log_info("supervised/model", f"Resumed from epoch {start_epoch}")

    # ── 优化器 + 调度器 | Optimizer + Scheduler ──
    # 收集所有可训练参数 | Collect all trainable params
    trainable_params = list(decoder.parameters())
    if not args.freeze_backbone:
        trainable_params += [p for p in backbone.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )

    # ── AMP 混合精度 | AMP Mixed Precision ──
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.amp:
        logger.log_info("supervised/config", "AMP mixed precision enabled")

    # ── 训练循环 | Training Loop ──
    logger.log_info("supervised/start",
        f"Training {args.epochs} epochs, device={device}, seed={args.seed}")
    logger.log_info("supervised/start", f"Output: {out_dir}")

    best_miou = 0.0
    best_epoch = 0
    best_state = None
    metrics_path = out_dir / "metrics.jsonl"

    for epoch in range(start_epoch, args.epochs + 1):
        # ── 训练 | Train ──
        train_loss = train_epoch(
            decoder, backbone, train_loader, optimizer, device,
            epoch, args.epochs, logger, class_weight=class_weight, tag="supervised/train",
        )

        scheduler.step()

        # ── 验证 | Validate ──
        val_metrics = validate(
            decoder, backbone, val_loader, device,
            epoch, args.epochs, eval_class_ids, ISAID5I_CATEGORIES,
            logger, tag="supervised/val",
        )

        # ── 保存最佳模型 | Save best model ──
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            best_epoch = epoch
            best_state = {
                "epoch": epoch,
                "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "miou": best_miou,
                "per_class_iou": val_metrics["per_class_iou"],
                "args": vars(args),
            }
            torch.save(best_state, str(out_dir / "best_model.pt"))
            logger.log_info("supervised/best",
                f"New best model: E{epoch} mIoU={best_miou:.4f}")

        # ── 定期保存 | Periodic save ──
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "miou": val_metrics["miou"],
            }, str(out_dir / f"checkpoint_e{epoch}.pt"))

        # ── 保存指标 (增量) | Save metrics (append) ──
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_miou": round(val_metrics["miou"], 6),
            "val_pixel_acc": round(val_metrics["pixel_acc"], 6),
            "val_n_valid_classes": val_metrics["n_valid_classes"],
            "per_class_iou": {str(k): round(v, 6) for k, v in val_metrics["per_class_iou"].items()},
            "lr": scheduler.get_last_lr()[0],
        }
        with open(metrics_path, "a") as mf:
            mf.write(json.dumps(epoch_metrics) + "\n")
            mf.flush()

    # ── 最终结果 | Final Results ──
    logger.log_info("supervised/done",
        f"{'='*60}\n"
        f"Training complete | 训练完成\n"
        f"Best mIoU: {best_miou:.4f} @ Epoch {best_epoch}\n"
        f"Output: {out_dir}/\n"
        f"{'='*60}")

    # ── 保存最终报告 | Save final report ──
    final_report = {
        "experiment": "Fully-Supervised Semantic Segmentation (Experiment A)",
        "mode": mode_desc,
        "fold": args.fold,
        "eval_classes": {str(c): ISAID5I_CATEGORIES.get(c, f"cls{c}") for c in eval_class_ids},
        "timestamp": datetime.now().isoformat(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": get_env_info(),
        "results": {
            "best_miou": round(best_miou, 6),
            "best_epoch": best_epoch,
            "best_per_class_iou": {str(k): round(v, 6) for k, v in best_state["per_class_iou"].items()} if best_state else {},
            "n_train_tiles": len(train_ds),
            "n_val_tiles": len(val_ds),
            "decoder_params": n_decoder_params,
            "backbone_trainable_params": n_backbone_params,
        },
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    save_env_info(str(out_dir / "env_info.json"))

    logger.log_info("supervised/done",
        f"Results saved → {out_dir}/results.json | 结果已保存")
    logger.log_info("supervised/done",
        f"Metrics log → {out_dir}/metrics.jsonl | 指标日志已保存")

    return best_miou


if __name__ == "__main__":
    main()
