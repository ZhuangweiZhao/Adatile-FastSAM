#!/usr/bin/env python3
"""
少样本微调训练 | Few-Shot Fine-Tuning Training.
================================================

在 Base 类全监督预训练的基础上，用 K-shot Novel 类样本进行微调，
使模型快速适配新类别。微调后直接推理，无需 Support 输入。

Few-shot fine-tuning on Novel classes after Base-class pre-training.
Fine-tuned model infers directly — no Support image needed at inference.

范式 | Paradigm:
    1. Pre-train:  Base 类全量数据训练 → 冻结保存
    2. Fine-tune:  Novel 类 K-shot 样本微调
    3. Evaluate:   直接推理 Novel 类测试图像

与 Few-Shot Segmentation (FSS) 的区别 | Difference from FSS:
    - FSS: Support → Prototype → Query Matching → Prediction (推理时需 Support)
    - 本脚本: K-shot Fine-tune → 直接推理 (推理时不需要 Support)
    - FSS: requires Support at inference. This script: no Support at inference.

用法 | Usage::
    # K=5 shot fine-tuning on Novel classes
    python tools/train/train_fewshot_finetune.py \\
        --load-ckpt runs/supervised_F0_P4_Frz_nocb_256/best_model.pt \\
        --k-shot 5 --finetune-epochs 20

    # Zero-shot evaluation only (no fine-tuning)
    python tools/train/train_fewshot_finetune.py \\
        --load-ckpt runs/supervised_F0_P4_Frz_nocb_256/best_model.pt \\
        --eval-zero-shot-only

    # Multi-seed evaluation (1/3/5/10 shot)
    python tools/train/train_fewshot_finetune.py \\
        --load-ckpt runs/supervised_F0_P4_Frz_nocb_256/best_model.pt \\
        --k-shot 1,3,5,10 --k-shot-seed 42,123,456
"""

from __future__ import annotations

import sys, argparse, json, os, random
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
from adatile.utils.seed import set_seed
from adatile.utils.env import get_env_info, save_env_info
from adatile.backbone import FastSAMBackbone
from adatile.decoder.light_decoder import LightDecoder, LightDecoderP3, LightDecoderP3P4
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
# K-shot Novel 类采样 | K-shot Novel Class Sampling
# ═══════════════════════════════════════════════════════════════════

def build_tile_to_classes(
    tile_names: list[str],
    mask_dir: Path,
    novel_ids: set[int],
) -> dict[str, set[int]]:
    """
    扫描所有 tile，建立 {tile_name: {包含的 Novel 类 ID}} 映射。
    Scan all tiles, build {tile_name: {Novel class IDs contained}} mapping.

    只记录 Novel 类（用于 K-shot 采样），忽略 Base 类和 BG。
    Only records Novel classes (for K-shot sampling), ignores Base classes and BG.

    :param tile_names: tile 名列表 | List of tile names.
    :param mask_dir: 掩码目录 | Mask directory.
    :param novel_ids: Novel 类 ID 集合 | Set of Novel class IDs.
    :return: {tile_name: set of Novel class IDs present}.
    """
    tile_to_classes: dict[str, set[int]] = {}
    for name in tile_names:
        # 查找掩码文件 | Find mask file
        mask_path = mask_dir / f"{name}_instance_color_RGB.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{name}.png"
        if not mask_path.exists():
            tile_to_classes[name] = set()
            continue

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            tile_to_classes[name] = set()
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        classes = set(np.unique(mask).tolist()) - {0}
        tile_to_classes[name] = classes & novel_ids  # 只保留 Novel 类 | Keep only Novel classes

    return tile_to_classes


def sample_k_shot_novel(
    tile_names: list[str],
    tile_to_classes: dict[str, set[int]],
    novel_ids: set[int],
    k: int,
    seed: int = 42,
) -> list[str]:
    """
    为每个 Novel 类采样 K 个包含该类的 tile。
    For each Novel class, sample K tiles containing that class.

    取所有被选中 tile 的并集，确保小类别有机会被表示。
    Take the union of all selected tiles, ensuring rare classes are represented.

    :param tile_names: 所有候选 tile 名 | All candidate tile names.
    :param tile_to_classes: {tile_name: {Novel class IDs}} 映射.
    :param novel_ids: Novel 类 ID 集合.
    :param k: 每类最多采样 K 个 tile | At most K tiles per class.
    :param seed: 随机种子 | Random seed.
    :return: 被选中的 tile 名列表 | Selected tile name list.
    """
    rng = random.Random(seed)

    selected: set[str] = set()
    for cls_id in sorted(novel_ids):
        candidates = sorted([
            name for name in tile_names
            if cls_id in tile_to_classes.get(name, set())
        ])
        if not candidates:
            print(f"  [WARNING] Novel class {cls_id} "
                  f"({ISAID5I_CATEGORIES.get(cls_id, '?')}): 0 tiles available!")
            continue
        n_pick = min(k, len(candidates))
        picked = rng.sample(candidates, n_pick)
        selected.update(picked)

    print(f"[sample_k_shot_novel] k={k}, seed={seed}: "
          f"selected {len(selected)} tiles across {len(novel_ids)} Novel classes")

    # 打印每类实际采样数 | Print actual samples per class
    class_counts = {c: 0 for c in sorted(novel_ids)}
    for name in selected:
        for c in tile_to_classes.get(name, set()):
            if c in class_counts:
                class_counts[c] += 1
    cls_lines = []
    for cls_id in sorted(novel_ids):
        n_full = sum(1 for classes in tile_to_classes.values() if cls_id in classes)
        cls_name = ISAID5I_CATEGORIES.get(cls_id, f"cls{cls_id}")
        cls_lines.append(f"  {cls_name} (cls{cls_id}): {class_counts[cls_id]}/{n_full}")
    print(f"[sample_k_shot_novel] Per-class tile coverage:\n" + "\n".join(cls_lines))

    return sorted(selected)


# ═══════════════════════════════════════════════════════════════════
# 少样本微调数据集 | Few-Shot Fine-Tuning Dataset
# ═══════════════════════════════════════════════════════════════════

class FewShotFinetuneDataset(Dataset):
    """
    K-shot Novel 类微调数据集 | K-shot Novel Class Fine-Tuning Dataset.

    从 iSAID-5i 标准 split 文件中加载 256×256 tiles。
    继承自 SupervisedSegDataset 的核心理念，但专门为 Novel 类 K-shot 微调设计。

    Loads 256×256 tiles from iSAID-5i standard split files.
    Inherits core concepts from SupervisedSegDataset but designed for Novel-class K-shot fine-tuning.

    关键区别 | Key differences:
    - K-shot 采样 Novel 类（而非 Base 类）| Samples Novel classes (not Base)
    - 训练时 Base 类像素设为 IGNORE_INDEX | Base class pixels → IGNORE during training
    - 验证时评估所有 15 类（检测 Base 类遗忘）| Evaluates all 15 classes on val (detects forgetting)

    Parameters
    ----------
    root : str
        iSAID-5i 数据根目录 | iSAID-5i data root.
    split : str
        "train" 或 "val".
    fold : int
        Fold ID (0/1/2). 必须 ≥0 (Base/Novel 划分需要).
    novel_ids : list[int]
        Novel 类 ID 列表 | Novel class ID list.
    base_ids : list[int]
        Base 类 ID 列表 | Base class ID list.
    k_shot : int
        K-shot 采样数: 每类最多 K 个 tile (0=全量 | full data).
    k_shot_seed : int
        K-shot 采样随机种子 | Random seed for K-shot sampling.
    """

    def __init__(
        self,
        root: str = DEFAULT_DATA_ROOT,
        split: str = "train",
        fold: int = 0,
        novel_ids: list[int] | None = None,
        base_ids: list[int] | None = None,
        k_shot: int = 0,
        k_shot_seed: int = 42,
    ):
        self.root = Path(root)
        self.split = split
        self.fold = fold
        self.novel_ids = set(novel_ids) if novel_ids else set()
        self.base_ids = set(base_ids) if base_ids else set()
        self.k_shot = k_shot
        self.k_shot_seed = k_shot_seed

        # ── 图像和掩码目录 | Image and mask directories ──
        # 目录结构: {root}/{split}/images/ 和 {root}/{split}/semantic_png/
        # Directory structure: {root}/{split}/images/ and {root}/{split}/semantic_png/
        self._img_dir = self.root / split / "images"
        self._mask_dir = self.root / split / "semantic_png"

        if not self._img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self._img_dir}")
        if not self._mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self._mask_dir}")

        # ── 读取 split 文件 | Read split file ──
        # Split 文件位于 {root}/{split}/{split}_list/split{fold}_{split}.txt
        # Split files are at {root}/{split}/{split}_list/split{fold}_{split}.txt
        list_dir = self.root / split / f"{split}_list"
        split_file = list_dir / f"split{fold}_{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"  Expected: split{fold}_{split}.txt in {list_dir}"
            )

        with open(split_file, "r") as f:
            self._tile_names = [
                self._clean_tile_name(line) for line in f
                if self._clean_tile_name(line)
            ]

        # ── K-shot 采样 (仅 train split) | K-shot sampling (train split only) ──
        if k_shot > 0 and split == "train" and self.novel_ids:
            print(f"[FewShotFinetuneDataset] K-shot Novel sampling: k={k_shot}, seed={k_shot_seed}")
            tile_to_classes = build_tile_to_classes(
                self._tile_names, self._mask_dir, self.novel_ids,
            )
            k_shot_tiles = sample_k_shot_novel(
                self._tile_names, tile_to_classes, self.novel_ids,
                k=k_shot, seed=k_shot_seed,
            )
            k_set = set(k_shot_tiles)
            n_before = len(self._tile_names)
            self._tile_names = [n for n in self._tile_names if n in k_set]
            print(f"[FewShotFinetuneDataset] Filtered: {n_before} → {len(self._tile_names)} tiles "
                  f"(K={k_shot}/Novel class, retained {len(self._tile_names)/max(n_before,1)*100:.1f}%)")

        # ── 日志 | Log ──
        mode_str = f"fold={fold}, Novel K-shot fine-tuning"
        print(f"[FewShotFinetuneDataset] {split}: {len(self._tile_names)} tiles, "
              f"{mode_str}, novel_ids={sorted(self.novel_ids) if self.novel_ids else 'none'}")

    # ── 文件名处理 | Filename handling ──

    @staticmethod
    def _clean_tile_name(raw: str) -> str | None:
        """从 split 文件行提取干净的 tile 名 | Extract clean tile name from split file line."""
        raw = raw.strip()
        if not raw:
            return None
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
            mask = mask[:, :, 0]
        mask_tensor = torch.from_numpy(mask.astype(np.int64))  # [H, W]

        # ── 微调模式: Base 类像素 → IGNORE (仅训练集) | Fine-tune mode: Base pixels → ignore (train only) ──
        # 训练时只关心 Novel 类，Base 类像素被忽略；验证时保留所有类以进行多维度评估
        # During training, only care about Novel classes; Base pixels are ignored.
        # During validation, keep all classes for multi-dimensional evaluation.
        if self.split == "train" and self.base_ids:
            for bid in self.base_ids:
                mask_tensor[mask_tensor == bid] = IGNORE_INDEX

        return {"image": img_tensor, "mask": mask_tensor, "tile_name": tile_name}


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════

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

    与 train_supervised.py 中 compute_loss() 完全相同的实现。
    Identical implementation to compute_loss() in train_supervised.py.

    :param class_weight: [num_classes] 类别权重 (BG 应为 0.0)。None → 无平衡。
        [num_classes] class weights (BG should be 0.0). None → no balancing.
    :return: (total_loss, {"focal": float, "dice": float})
    """
    # ── Focal Loss | 焦点损失 ──
    ce = F.cross_entropy(
        logit, target,
        weight=class_weight.to(logit.device) if class_weight is not None else None,
        ignore_index=ignore_index,
        reduction="none",
    )
    focal_loss = ((1.0 - torch.exp(-ce)) ** focal_gamma * ce).mean()

    # ── Dice Loss | Dice 损失 ──
    probs = F.softmax(logit, dim=1)  # [B, C, H, W]
    dice_sum, valid_dice = 0.0, 0
    for c in range(1, num_classes):
        p_c = probs[:, c]                       # [B, H, W]
        t_c = (target == c).float()              # [B, H, W]
        inter = (p_c * t_c).sum()
        union = p_c.sum() + t_c.sum() + 1e-8
        if t_c.sum() > 0:
            cls_w = class_weight[c].item() if class_weight is not None else 1.0
            dice_sum += cls_w * 2.0 * inter / union
            valid_dice += 1

    dice_loss = 1.0 - (dice_sum / max(valid_dice, 1))
    total_loss = 0.5 * focal_loss + 0.5 * dice_loss

    return total_loss, {"focal": focal_loss.item(), "dice": dice_loss.item()}


# ═══════════════════════════════════════════════════════════════════
# 类别平衡权重 | Class-Balanced Weights
# ═══════════════════════════════════════════════════════════════════

def compute_class_weights(
    dataset: FewShotFinetuneDataset,
    num_classes: int = NUM_OUT_CH,
    cap: float = 10.0,
) -> torch.Tensor:
    """
    计算逆频率类别平衡权重 | Compute inverse-frequency class-balanced weights.

    对 Novel 类计算类平衡权重。权重 = median_freq / (cls_freq + eps)，上限为 cap。
    BG (c=0) 权重 = 0.0。

    Compute class-balanced weights for Novel classes.
    weight = median_freq / (cls_freq + eps), capped at cap. BG (c=0) weight = 0.0.

    与 train_supervised.py 中 compute_class_weights() 完全相同的实现。
    """
    cls_pixels = defaultdict(float)
    total_fg = 0

    print("[compute_class_weights] Scanning masks...")
    for i in tqdm(range(len(dataset)), desc="  Scanning masks"):
        sample = dataset[i]
        mask = sample["mask"]
        unique, counts = torch.unique(mask, return_counts=True)
        for u, cnt in zip(unique.tolist(), counts.tolist()):
            if u != IGNORE_INDEX and u != 0:
                cls_pixels[u] += cnt
                total_fg += cnt

    if total_fg == 0:
        print("[compute_class_weights] WARNING: No foreground pixels found!")
        return torch.ones(num_classes)

    freqs = {}
    for c in range(1, num_classes):
        freqs[c] = cls_pixels.get(c, 0) / max(total_fg, 1)

    valid_freqs = [v for v in freqs.values() if v > 0]
    median_freq = np.median(valid_freqs) if valid_freqs else 1.0

    weights = torch.ones(num_classes)
    weights[0] = 0.0  # BG weight = 0
    for c in range(1, num_classes):
        if freqs.get(c, 0) > 0:
            w = median_freq / (freqs[c] + 1e-8)
            weights[c] = min(w, cap)
        else:
            weights[c] = cap  # 未见类别给最大权重 | Unseen class gets max weight

    # 打印权重 | Print weights
    print(f"[compute_class_weights] median_freq={median_freq:.6f}, cap={cap}")
    for c in range(1, num_classes):
        if weights[c] >= cap or weights[c] <= 1.0 / cap:
            marker = " ***" if weights[c] >= cap else " (low)"
            print(f"  cls{c}: freq={freqs.get(c, 0)*100:.4f}% "
                  f"weight={weights[c]:.2f}{marker}")

    return weights


# ═══════════════════════════════════════════════════════════════════
# 评估 | Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    decoder: nn.Module,
    backbone: FastSAMBackbone,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    eval_class_ids: list[int],
    category_names: dict[int, str],
    logger,
    tag: str = "fewshot/val",
) -> dict:
    """
    多类别分割评估 | Multi-class segmentation evaluation.

    与 train_supervised.py 中 validate() 完全相同的实现。
    Identical implementation to validate() in train_supervised.py.

    :param eval_class_ids: 需要评估的前景类 ID 列表
                           List of foreground class IDs to evaluate.
    :param category_names: {class_id: name} 映射 | Category name mapping.
    :return: {"miou": float, "per_class_iou": dict, "pixel_acc": float}
    """
    decoder.eval()

    eval_set = set(eval_class_ids)
    per_class_inter = {c: 0.0 for c in eval_set}
    per_class_union = {c: 0.0 for c in eval_set}
    correct, total = 0, 0

    for batch in tqdm(loader, desc=f"  Eval E{epoch}/{total_epochs}", leave=False):
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

        # 像素准确率 (排除 ignore + 非评估类别) | Pixel accuracy (excl. ignore + non-eval classes)
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
        f"E{epoch:3d}/{total_epochs} eval | mIoU={miou:.4f} ({miou_valid}/{len(eval_set)} classes) "
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
# 训练 Epoch | Training Epoch
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
    tag: str = "fewshot/train",
) -> float:
    """
    微调训练一个 epoch | Fine-tune for one epoch.

    与 train_supervised.py 中 train_epoch() 完全相同的实现。
    Identical implementation to train_epoch() in train_supervised.py.

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
        feats = backbone(img)                                  # {"p4": ..., "p3": ..., "p8": ...}
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


# ═══════════════════════════════════════════════════════════════════
# 构建 Backbone + Decoder | Build Backbone + Decoder
# ═══════════════════════════════════════════════════════════════════

def build_model_from_pretrained(
    pretrained_args: dict,
    device: torch.device,
    logger,
) -> tuple[FastSAMBackbone, nn.Module]:
    """
    根据预训练参数构建 backbone + decoder。
    Build backbone + decoder from pre-trained args dict.

    读取 pretrained_args 中的 freeze_backbone, use_p3, use_p3_only 等参数，
    构建与预训练时相同的模型架构。

    Reads freeze_backbone, use_p3, use_p3_only from pretrained_args,
    builds the same model architecture as during pre-training.

    :param pretrained_args: checkpoint 中保存的 args dict | Args dict saved in checkpoint.
    :param device: 运行设备 | Device.
    :param logger: 日志器 | Logger.
    :return: (backbone, decoder)
    """
    freeze_backbone = pretrained_args.get("freeze_backbone", True)
    logger.log_info("fewshot/model",
        f"Building model from pretrained config: "
        f"freeze_backbone={freeze_backbone}, "
        f"use_p3={pretrained_args.get('use_p3', False)}, "
        f"use_p3_only={pretrained_args.get('use_p3_only', False)}")

    backbone = FastSAMBackbone(freeze_backbone=freeze_backbone).to(device).eval()

    if pretrained_args.get("use_p3_only", False):
        # 纯 P3 解码器 | P3-only decoder
        with torch.no_grad():
            probe = backbone(torch.randn(1, 3, 256, 256).to(device))
            p3_ch = probe["p3"].shape[1]
        decoder = LightDecoderP3(in_channels=p3_ch, num_classes=NUM_OUT_CH).to(device)
        logger.log_info("fewshot/model",
            f"Using LightDecoderP3 (P3-only): p3_ch={p3_ch}")

    elif pretrained_args.get("use_p3", False):
        # P3+P4 多尺度融合解码器 | P3+P4 multi-scale fusion decoder
        with torch.no_grad():
            probe = backbone(torch.randn(1, 3, 256, 256).to(device))
            p3_ch = probe["p3"].shape[1]
            p4_ch = probe["p4"].shape[1]
        decoder = LightDecoderP3P4(
            p3_channels=p3_ch, p4_channels=p4_ch, num_classes=NUM_OUT_CH,
        ).to(device)
        logger.log_info("fewshot/model",
            f"Using LightDecoderP3P4: p3_ch={p3_ch}, p4_ch={p4_ch}")

    else:
        # 默认 P4-only | Default P4-only
        decoder = LightDecoder(in_channels=1280, num_classes=NUM_OUT_CH).to(device)
        logger.log_info("fewshot/model", "Using LightDecoder (P4-only)")

    return backbone, decoder


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    """解析命令行参数 | Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="少样本微调训练 | Few-Shot Fine-Tuning Training")

    # ── 数据 | Data ──
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                   help=f"iSAID-5i 数据根目录 | iSAID-5i data root (default: {DEFAULT_DATA_ROOT})")
    p.add_argument("--tile-root", type=str, default=None,
                   help="预切 tile 目录 (896×896) | Pre-cut tile root")

    # ── 预训练 Checkpoint | Pre-trained Checkpoint ──
    p.add_argument("--load-ckpt", type=str, required=True,
                   help="预训练 Base 类 checkpoint 路径 (必需) | "
                        "Pre-trained Base-class checkpoint path (required)")

    # ── K-shot 微调 | K-shot Fine-Tuning ──
    p.add_argument("--k-shot", type=str, default="1,3,5,10",
                   help="K-shot 采样数，逗号分隔 (default: 1,3,5,10) | "
                        "K-shot values, comma-separated")
    p.add_argument("--k-shot-seed", type=str, default="42",
                   help="K-shot 随机种子，逗号分隔 (default: 42) | "
                        "K-shot random seeds, comma-separated (e.g. '42,123,456')")
    p.add_argument("--finetune-epochs", type=int, default=20,
                   help="微调 epoch 数 (default: 20) | Fine-tuning epochs")
    p.add_argument("--finetune-lr", type=float, default=1e-4,
                   help="微调学习率 (default: 1e-4, 为预训练 lr 的 1/10) | "
                        "Fine-tuning learning rate")

    # ── 微调策略 | Fine-Tuning Strategy ──
    p.add_argument("--finetune-backbone", action="store_true",
                   help="同时微调 backbone (默认只微调 decoder) | "
                        "Also fine-tune backbone (default: decoder only)")
    p.add_argument("--partial-finetune", type=int, default=0,
                   help="部分解冻: 仅解冻 backbone 最后 N 层 (0=仅 decoder) | "
                        "Partial unfreeze: unfreeze last N backbone layers (0=decoder only)")
    p.add_argument("--backbone-lr", type=float, default=1e-5,
                   help="Backbone 学习率 (default: 1e-5) | Backbone LR")

    # ── 评估 | Evaluation ──
    p.add_argument("--eval-zero-shot-only", action="store_true",
                   help="仅 zero-shot 评估 (不微调) | Only zero-shot evaluation (no fine-tuning)")

    # ── 训练参数 | Training Parameters ──
    p.add_argument("--batch-size", type=int, default=16,
                   help="批次大小 | Batch size (default: 16)")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="权重衰减 | Weight decay (default: 1e-4)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader workers (default: 4)")
    p.add_argument("--focal-gamma", type=float, default=5.0,
                   help="Focal Loss gamma (default: 5.0)")
    p.add_argument("--no-class-balance", action="store_true",
                   help="禁用类别平衡损失 | Disable class-balanced loss")
    p.add_argument("--class-weight-cap", type=float, default=10.0,
                   help="类别权重上限 (default: 10.0) | Class weight cap")

    # ── 硬件 | Hardware ──
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="运行设备 | Device")
    p.add_argument("--amp", action="store_true",
                   help="启用混合精度训练 | Enable AMP mixed precision training")

    # ── 输出 | Output ──
    p.add_argument("--output-dir", type=str, default=None,
                   help="输出目录 | Output directory (default: runs/fewshot_<timestamp>)")
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

    # ── 解析 K-shot 值和种子 | Parse K-shot values and seeds ──
    k_shot_values = [int(k.strip()) for k in args.k_shot.split(",")]
    k_shot_seeds = [int(s.strip()) for s in args.k_shot_seed.split(",")]

    # ── 加载预训练 checkpoint | Load pre-trained checkpoint ──
    ckpt_path = Path(args.load_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    pretrained_args = ckpt.get("args", {})
    pretrained_fold = pretrained_args.get("fold", 0)
    pretrained_miou = ckpt.get("miou", 0.0)

    if pretrained_fold < 0:
        print(f"[ERROR] Pre-trained checkpoint was trained in full mode (fold=-1).")
        print(f"  Few-shot fine-tuning requires a Base-only checkpoint (fold=0/1/2).")
        print(f"  Please re-train with --fold 0 (or 1/2) first.")
        sys.exit(1)

    # ── 输出目录 | Output directory ──
    if args.output_dir is None:
        ckpt_tag = ckpt_path.parent.name if ckpt_path.parent.name != "." else "ckpt"
        ts = datetime.now().strftime("%m%d_%H%M")
        args.output_dir = f"runs/fewshot_{ckpt_tag}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志系统 | Logging system ──
    logger = get_logger("fewshot_finetune")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))

    # ── 类别配置 | Category configuration ──
    fold = pretrained_fold
    novel_ids = ISAID5I_FOLDS[fold]["novel"]
    base_ids = ISAID5I_FOLDS[fold]["base"]

    logger.log_info("fewshot/config",
        f"Fold {fold} | Base-only → Novel K-shot Fine-tuning")
    logger.log_info("fewshot/config",
        f"Pre-trained checkpoint: {ckpt_path} (Base mIoU={pretrained_miou:.4f})")
    logger.log_info("fewshot/config",
        f"Base classes ({len(base_ids)}): "
        f"{[ISAID5I_CATEGORIES[c] for c in base_ids if c in ISAID5I_CATEGORIES]}")
    logger.log_info("fewshot/config",
        f"Novel classes ({len(novel_ids)}): "
        f"{[ISAID5I_CATEGORIES[c] for c in novel_ids if c in ISAID5I_CATEGORIES]}")
    logger.log_info("fewshot/config",
        f"K-shot values: {k_shot_values}, seeds: {k_shot_seeds}")
    logger.log_info("fewshot/config",
        f"Fine-tune epochs: {args.finetune_epochs}, LR: {args.finetune_lr}")
    logger.log_info("fewshot/config",
        f"Fine-tune backbone: {args.finetune_backbone}, "
        f"partial={args.partial_finetune}")

    # ── 构建模型 | Build model ──
    backbone, decoder = build_model_from_pretrained(pretrained_args, device, logger)

    # ── 加载预训练 Decoder 权重 | Load pre-trained decoder weights ──
    decoder.load_state_dict(ckpt["decoder_state_dict"])
    logger.log_info("fewshot/model",
        f"Loaded pre-trained decoder weights (epoch={ckpt.get('epoch', '?')})")

    # ── 构建验证数据集 | Build validation dataset ──
    logger.log_info("fewshot/data", "Building validation dataset...")
    if args.tile_root:
        raise NotImplementedError("Pre-cut tile dataset not yet supported for few-shot fine-tuning")
    else:
        val_ds = FewShotFinetuneDataset(
            root=args.data_root, split="val", fold=fold,
            novel_ids=novel_ids, base_ids=base_ids,
        )

    # DataLoader 配置 (val) | DataLoader config (val)
    is_windows = (sys.platform == "win32")
    if is_windows and args.num_workers > 0:
        args.num_workers = 0

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(args.num_workers, 2), pin_memory=True,
    )

    logger.log_info("fewshot/data",
        f"Val: {len(val_ds)} tiles, batch_size={args.batch_size}")

    # ═══════════════════════════════════════════════════════════════════
    # Zero-Shot 评估 | Zero-Shot Evaluation
    # ═══════════════════════════════════════════════════════════════════
    # 微调前：用预训练权重评估 Novel 类 mIoU
    # Before fine-tuning: evaluate Novel class mIoU with pre-trained weights

    logger.log_info("fewshot/zero_shot", "=" * 60)
    logger.log_info("fewshot/zero_shot", "Zero-Shot Evaluation (pre-trained weights)")

    # 评估 Novel 类 (仅 Novel) | Evaluate Novel classes only
    zero_shot_novel = evaluate(
        decoder, backbone, val_loader, device,
        0, 0, novel_ids, ISAID5I_CATEGORIES,
        logger, tag="fewshot/zero_shot_novel",
    )

    # 评估 Base 类 (检测遗忘基线) | Evaluate Base classes (forgetting baseline)
    zero_shot_base = evaluate(
        decoder, backbone, val_loader, device,
        0, 0, base_ids, ISAID5I_CATEGORIES,
        logger, tag="fewshot/zero_shot_base",
    )

    # 评估全类 | Evaluate all classes
    all_ids = sorted(base_ids + novel_ids)
    zero_shot_all = evaluate(
        decoder, backbone, val_loader, device,
        0, 0, all_ids, ISAID5I_CATEGORIES,
        logger, tag="fewshot/zero_shot_all",
    )

    logger.log_info("fewshot/zero_shot",
        f"Zero-Shot Results: Novel mIoU={zero_shot_novel['miou']:.4f}, "
        f"Base mIoU={zero_shot_base['miou']:.4f}, "
        f"All mIoU={zero_shot_all['miou']:.4f}")

    # 如果是 zero-shot-only 模式，保存结果后退出 | If zero-shot-only mode, save and exit
    if args.eval_zero_shot_only:
        zero_shot_results = {
            "experiment": "Few-Shot Fine-Tuning — Zero-Shot Evaluation Only",
            "fold": fold,
            "pretrained_ckpt": str(ckpt_path),
            "pretrained_base_miou": round(pretrained_miou, 6),
            "novel_ids": novel_ids,
            "base_ids": base_ids,
            "zero_shot": {
                "novel_miou": round(zero_shot_novel["miou"], 6),
                "base_miou": round(zero_shot_base["miou"], 6),
                "all_miou": round(zero_shot_all["miou"], 6),
                "novel_per_class_iou": {str(k): round(v, 6) for k, v in zero_shot_novel["per_class_iou"].items()},
                "base_per_class_iou": {str(k): round(v, 6) for k, v in zero_shot_base["per_class_iou"].items()},
            },
            "config": vars(args),
            "timestamp": datetime.now().isoformat(),
        }
        with open(out_dir / "zero_shot_results.json", "w") as f:
            json.dump(zero_shot_results, f, indent=2, ensure_ascii=False)
        logger.log_info("fewshot/done", f"Zero-shot results saved to {out_dir / 'zero_shot_results.json'}")
        return

    # ═══════════════════════════════════════════════════════════════════
    # K-Shot 微调扫描 | K-Shot Fine-Tuning Sweep
    # ═══════════════════════════════════════════════════════════════════
    # 对每个 (K, seed) 组合进行微调 | Fine-tune for each (K, seed) combination

    all_results = []
    best_overall = {"novel_miou": 0.0, "k": 0, "seed": 0}

    for k in k_shot_values:
        for ks_seed in k_shot_seeds:
            run_id = f"K{k}_S{ks_seed}"
            logger.log_info("fewshot/run", "")
            logger.log_info("fewshot/run", "=" * 60)
            logger.log_info("fewshot/run",
                f"Fine-Tuning Run: K={k}-shot, seed={ks_seed} | "
                f"微调运行: K={k}-shot, 种子={ks_seed}")
            logger.log_info("fewshot/run", "=" * 60)

            # ── 构建 K-shot 训练数据集 | Build K-shot training dataset ──
            train_ds = FewShotFinetuneDataset(
                root=args.data_root, split="train", fold=fold,
                novel_ids=novel_ids, base_ids=base_ids,
                k_shot=k, k_shot_seed=ks_seed,
            )

            n_train_tiles = len(train_ds)
            logger.log_info("fewshot/data",
                f"K={k}, seed={ks_seed}: {n_train_tiles} training tiles "
                f"(Novel classes only)")

            if n_train_tiles == 0:
                logger.log_info("fewshot/data",
                    f"WARNING: 0 training tiles for K={k}, seed={ks_seed} — skipping!")
                continue

            # ── 类别平衡权重 | Class-balanced weights ──
            class_weight = None
            if not args.no_class_balance:
                logger.log_info("fewshot/data", "Computing class-balanced weights...")
                class_weight = compute_class_weights(train_ds, num_classes=NUM_OUT_CH, cap=args.class_weight_cap)
            else:
                logger.log_info("fewshot/data", "Class-balanced weights DISABLED")

            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True,
                drop_last=True,
            )

            # ── 重新加载预训练权重 (每次运行从零开始) | Reload pre-trained weights (fresh start each run) ──
            decoder.load_state_dict(ckpt["decoder_state_dict"])

            # ── Backbone 微调策略 | Backbone fine-tuning strategy ──
            if args.partial_finetune != 0:
                n_unfrozen = backbone.unfreeze_last_n_layers(args.partial_finetune)
                logger.log_info("fewshot/model",
                    f"Partial fine-tune: unfroze last {args.partial_finetune} backbone layers "
                    f"({n_unfrozen:,} trainable params)")
            elif args.finetune_backbone:
                backbone.unfreeze()
                n_total_bb = sum(p.numel() for p in backbone.parameters())
                logger.log_info("fewshot/model",
                    f"Full backbone unfrozen: {n_total_bb:,} trainable params")
            else:
                # 默认: 冻结 backbone, 仅微调 decoder | Default: freeze backbone, only fine-tune decoder
                backbone.freeze()
                logger.log_info("fewshot/model", "Backbone frozen — decoder-only fine-tuning")

            # ── 优化器 + 调度器 | Optimizer + Scheduler ──
            backbone_params = [p for p in backbone.parameters() if p.requires_grad]
            decoder_params = list(decoder.parameters())

            param_groups = [
                {"params": decoder_params, "lr": args.finetune_lr},
            ]
            if backbone_params:
                param_groups.append({
                    "params": backbone_params,
                    "lr": args.backbone_lr,
                })
                logger.log_info("fewshot/optim",
                    f"Separate LRs: decoder={args.finetune_lr}, "
                    f"backbone={args.backbone_lr} ({len(backbone_params)} params)")

            n_params = sum(p.numel() for pg in param_groups for p in pg["params"])
            logger.log_info("fewshot/model",
                f"Total trainable: {n_params:,} params")

            optimizer = torch.optim.AdamW(
                param_groups,
                lr=args.finetune_lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.finetune_epochs,
                eta_min=args.finetune_lr * 0.01,
            )

            # ── 微调训练循环 | Fine-Tuning Loop ──
            best_novel_miou = 0.0
            best_epoch = 0
            best_state = None
            run_dir = out_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            for epoch in range(1, args.finetune_epochs + 1):
                # 训练 | Train
                train_loss = train_epoch(
                    decoder, backbone, train_loader, optimizer, device,
                    epoch, args.finetune_epochs, logger,
                    class_weight=class_weight,
                    tag=f"fewshot/{run_id}/train",
                )

                scheduler.step()

                # 评估 Novel 类 | Evaluate Novel classes
                val_novel = evaluate(
                    decoder, backbone, val_loader, device,
                    epoch, args.finetune_epochs,
                    novel_ids, ISAID5I_CATEGORIES,
                    logger, tag=f"fewshot/{run_id}/val_novel",
                )

                # 保存最佳模型 | Save best model
                if val_novel["miou"] > best_novel_miou:
                    best_novel_miou = val_novel["miou"]
                    best_epoch = epoch
                    best_state = {
                        "epoch": epoch,
                        "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                        "optimizer_state_dict": optimizer.state_dict(),
                        "novel_miou": best_novel_miou,
                        "per_class_iou": val_novel["per_class_iou"],
                        "k_shot": k,
                        "k_shot_seed": ks_seed,
                        "finetune_args": vars(args),
                    }
                    torch.save(best_state, str(run_dir / "best_model.pt"))
                    logger.log_info(f"fewshot/{run_id}/best",
                        f"New best: E{epoch} Novel mIoU={best_novel_miou:.4f}")

                # 定期保存 | Periodic save
                if args.save_every > 0 and epoch % args.save_every == 0:
                    torch.save({
                        "epoch": epoch,
                        "decoder_state_dict": decoder.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "novel_miou": val_novel["miou"],
                    }, str(run_dir / f"checkpoint_e{epoch}.pt"))

            # ── 最终评估 (最佳模型) | Final evaluation (best model) ──
            decoder.load_state_dict(best_state["decoder_state_dict"])

            final_novel = evaluate(
                decoder, backbone, val_loader, device,
                best_epoch, args.finetune_epochs,
                novel_ids, ISAID5I_CATEGORIES,
                logger, tag=f"fewshot/{run_id}/final_novel",
            )
            final_base = evaluate(
                decoder, backbone, val_loader, device,
                best_epoch, args.finetune_epochs,
                base_ids, ISAID5I_CATEGORIES,
                logger, tag=f"fewshot/{run_id}/final_base",
            )
            final_all = evaluate(
                decoder, backbone, val_loader, device,
                best_epoch, args.finetune_epochs,
                all_ids, ISAID5I_CATEGORIES,
                logger, tag=f"fewshot/{run_id}/final_all",
            )

            # ── 记录结果 | Record results ──
            run_result = {
                "k_shot": k,
                "k_shot_seed": ks_seed,
                "n_train_tiles": n_train_tiles,
                "best_epoch": best_epoch,
                "zero_shot_novel_miou": round(zero_shot_novel["miou"], 6),
                "final_novel_miou": round(final_novel["miou"], 6),
                "final_base_miou": round(final_base["miou"], 6),
                "final_all_miou": round(final_all["miou"], 6),
                "delta_novel": round(final_novel["miou"] - zero_shot_novel["miou"], 6),
                "novel_per_class_iou": {str(k): round(v, 6) for k, v in final_novel["per_class_iou"].items()},
                "base_per_class_iou": {str(k): round(v, 6) for k, v in final_base["per_class_iou"].items()},
            }
            all_results.append(run_result)

            logger.log_info(f"fewshot/{run_id}/result",
                f"K={k}, seed={ks_seed}: "
                f"Novel {zero_shot_novel['miou']:.4f} → {final_novel['miou']:.4f} "
                f"(Δ={final_novel['miou'] - zero_shot_novel['miou']:+.4f}), "
                f"Base {final_base['miou']:.4f}, "
                f"All {final_all['miou']:.4f}")

            # 更新最佳 | Update best
            if final_novel["miou"] > best_overall["novel_miou"]:
                best_overall = {
                    "novel_miou": final_novel["miou"],
                    "k": k,
                    "seed": ks_seed,
                }

    # ═══════════════════════════════════════════════════════════════════
    # 最终报告 | Final Report
    # ═══════════════════════════════════════════════════════════════════

    logger.log_info("fewshot/done", "")
    logger.log_info("fewshot/done", "=" * 60)
    logger.log_info("fewshot/done", "Few-Shot Fine-Tuning Complete | 少样本微调完成")
    logger.log_info("fewshot/done", "=" * 60)

    # 打印结果表格 | Print results table
    logger.log_info("fewshot/done", "")
    logger.log_info("fewshot/done", f"{'K':>4s}  {'Seed':>5s}  {'Tiles':>6s}  "
        f"{'ZeroShot':>10s}  {'FineTune':>10s}  {'Delta':>10s}  {'Base':>10s}  {'All':>10s}")
    logger.log_info("fewshot/done", "-" * 80)
    for r in all_results:
        logger.log_info("fewshot/done",
            f"{r['k_shot']:4d}  {r['k_shot_seed']:5d}  {r['n_train_tiles']:6d}  "
            f"{r['zero_shot_novel_miou']:10.4f}  {r['final_novel_miou']:10.4f}  "
            f"{r['delta_novel']:+10.4f}  {r['final_base_miou']:10.4f}  {r['final_all_miou']:10.4f}")

    logger.log_info("fewshot/done", "")
    logger.log_info("fewshot/done",
        f"Best: K={best_overall['k']}, seed={best_overall['seed']}, "
        f"Novel mIoU={best_overall['novel_miou']:.4f}")

    # ── 保存最终报告 | Save final report ──
    final_report = {
        "experiment": "Few-Shot Fine-Tuning (Experiment B)",
        "fold": fold,
        "pretrained_ckpt": str(ckpt_path),
        "pretrained_base_miou": round(pretrained_miou, 6),
        "novel_ids": novel_ids,
        "base_ids": base_ids,
        "novel_names": {c: ISAID5I_CATEGORIES.get(c, f"cls{c}") for c in novel_ids},
        "base_names": {c: ISAID5I_CATEGORIES.get(c, f"cls{c}") for c in base_ids},
        "zero_shot": {
            "novel_miou": round(zero_shot_novel["miou"], 6),
            "base_miou": round(zero_shot_base["miou"], 6),
            "all_miou": round(zero_shot_all["miou"], 6),
            "novel_per_class_iou": {str(k): round(v, 6) for k, v in zero_shot_novel["per_class_iou"].items()},
            "base_per_class_iou": {str(k): round(v, 6) for k, v in zero_shot_base["per_class_iou"].items()},
        },
        "fine_tuning_results": all_results,
        "best_overall": best_overall,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": get_env_info(),
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    logger.log_info("fewshot/done", f"Final report saved to {out_dir / 'results.json'}")
    logger.log_info("fewshot/done", f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
