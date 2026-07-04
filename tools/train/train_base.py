#!/usr/bin/env python3
"""
V3-05: Base 类预训练 | Base Class Pre-Training.
=================================================

v3 协议 Phase 3 Step 1: 在 Base 10 类上预训练 AdaptiveSparseDecoder。
v3 Protocol Phase 3 Step 1: Pre-train AdaptiveSparseDecoder on 10 Base classes.

训练策略 | Training Strategy:
    每步采样一个 Base 类 → K support tiles + 1 query tile
    → Support prototype → Decoder(coeffs + P4 refine) → Mask
    → Focal(γ=5.0) + Dice loss → Backward
    Each step: sample Base class → K support tiles + 1 query tile
    → Support prototype → Decoder → Mask → Loss → Backward

核心设计 | Core Design:
    - Decoder 学习"prototype → coefficients → mask"映射
    - Backbone 冻结 (FastSAM SA-1B 预训练权重)
    - SPM 暂不训练 (遵循"Decoder-SPM decoupled"原则)
    - 使用 v3 Instance Few-Shot Split (896² tiles, COCO format)

与 v2 的关键区别 | Key Differences from v2:
    - v2: Episode-based FSS meta-learning (iSAID-5i, 256²)
    - v3: Base pre-training → Novel fine-tune → Direct inference (896²)

用法 | Usage::

    # 快速验证 (1 epoch, 100 steps)
    python tools/train/train_base.py --fold 0 --epochs 1 --steps-per-epoch 100

    # 正式训练
    python tools/train/train_base.py --fold 0 --epochs 50 --device cuda

    # 云服务器后台
    nohup python tools/train/train_base.py --fold 0 --epochs 50 \
        --data-root /root/autodl-tmp/iSAID_instance_fewshot \
        --device cuda > /root/autodl-tmp/v3_05_base_train.log 2>&1 &
"""

from __future__ import annotations

import sys, argparse, json, os, random
from pathlib import Path
from collections import defaultdict
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
from adatile.sparse import SparsePerceptionModule
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder, ProtoOnlyDecoder
from adatile.datasets.isaid_instance_fewshot import (
    ISAIDInstanceFewShotDataset,
    sample_k_shot,
    instances_to_dense_mask,
    ISAID_CATEGORIES as ISAID_CAT_NAMES,
)
from adatile.metrics.coco_eval import COCOInstanceEvaluator


# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

DEFAULT_DATA_ROOT = "data/iSAID_instance_fewshot"
IGNORE_INDEX = 255


# ═══════════════════════════════════════════════════════════════════
# Support Prototype 计算 | Support Prototype Computation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_support_prototype(
    backbone: FastSAMBackbone,
    support_images: torch.Tensor,       # [K, 3, H, W]
    support_masks: torch.Tensor,        # [K, H, W] binary per-class masks
    device: torch.device,
) -> torch.Tensor:
    """
    从 K 张 support 图像计算 FG prototype | Compute FG prototype from K support images.

    对每张 support 图像提取 P4 特征，用 binary mask 做 FG 平均池化，
    然后对 K 个 prototype 取平均并 L2 归一化。
    Extract P4 features from each support image, FG average pool with binary mask,
    then average K prototypes and L2-normalize.

    :param backbone: FastSAM backbone (eval mode, frozen).
    :param support_images: [K, 3, H, W] support images in [0, 1].
    :param support_masks: [K, H, W] binary per-class GT masks.
    :param device: 计算设备 | Compute device.
    :return: [1280] L2-normalized prototype vector.
    """
    K = support_images.shape[0]
    prototypes = []

    for i in range(K):
        img = support_images[i:i + 1].to(device)   # [1, 3, H, W]
        mask = support_masks[i:i + 1].to(device)    # [1, H, W]

        feats = backbone(img)
        p4 = feats["p4"]  # [1, 1280, H/16, W/16]

        # 下采样 mask 到 P4 分辨率 | Downsample mask to P4 resolution
        mask_ds = F.interpolate(
            mask.unsqueeze(0).float(), size=p4.shape[2:], mode="nearest"
        ).squeeze(1)  # [1, H/16, W/16]

        fg = mask_ds > 0.5
        if fg.sum() > 0:
            proto = p4[:, :, fg.squeeze(0)].mean(dim=-1).squeeze(0)  # [1280]
            prototypes.append(proto)

    if not prototypes:
        return torch.zeros(1280, device=device)

    proto = torch.stack(prototypes).mean(dim=0)  # [1280]
    return F.normalize(proto, dim=0)


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════

def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 5.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Focal Loss (binary) — 遥感极端类别不平衡专用.
    Focal Loss (binary) — for extreme remote sensing class imbalance.

    航拍 tile 中 FG 占比常 <5%，γ=5.0 相比 γ=2.0 能更有效地
    抑制易分 BG 样本的梯度。eps=1e-4 (非 1e-8) 防止梯度爆炸。
    FG ratio <5% in aerial tiles, γ=5.0 better suppresses easy BG gradients.
    eps=1e-4 (not 1e-8) prevents gradient explosion.

    :param pred: [*] 预测概率 [0, 1] | Predicted probability.
    :param target: [*] 二值 GT (0 or 1) | Binary GT.
    :param gamma: Focal gamma (default 5.0 for remote sensing).
    :param eps: 数值稳定 epsilon | Numerical stability epsilon.
    :return: scalar loss.
    """
    pred = torch.clamp(pred, eps, 1.0 - eps)
    bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    pt = pred * target + (1 - pred) * (1 - target)
    return ((1 - pt) ** gamma * bce).mean()


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Dice 损失 (binary) | Dice Loss (binary).

    :param pred: [*] 预测概率 [0, 1] | Predicted probability.
    :param target: [*] 二值 GT | Binary GT.
    :param smooth: 平滑项 | Smoothing term.
    :return: 1 - Dice (scalar).
    """
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1.0 - (2.0 * inter + smooth) / (union + smooth)


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    focal_gamma: float = 5.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    组合损失: alpha * Focal + (1 - alpha) * Dice.
    Combined loss: alpha * Focal + (1 - alpha) * Dice.

    :param pred: [H, W] 预测概率 | Predicted probability.
    :param target: [H, W] 二值 GT | Binary GT.
    :param alpha: Focal 权重 | Focal weight (Dice = 1 - alpha).
    :param focal_gamma: Focal gamma 参数 | Focal gamma parameter.
    :return: (total_loss, {"focal": float, "dice": float}).
    """
    fl = focal_loss(pred, target, gamma=focal_gamma)
    dl = dice_loss(pred, target)
    return alpha * fl + (1 - alpha) * dl, {"focal": fl.item(), "dice": dl.item()}


# ═══════════════════════════════════════════════════════════════════
# 训练一步 | One Training Step
# ═══════════════════════════════════════════════════════════════════

def train_step(
    decoder: AdaptiveSparseDecoder | ProtoOnlyDecoder,
    backbone: FastSAMBackbone,
    spm: SparsePerceptionModule | None,
    support_imgs: torch.Tensor,          # [K, 3, H, W]
    support_masks: torch.Tensor,         # [K, H, W]
    query_img: torch.Tensor,             # [1, 3, H, W]
    query_mask: torch.Tensor,            # [H, W] binary per-class GT
    class_id: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_spm: bool = False,
) -> tuple[float, dict]:
    """
    训练一步 | Train one step.

    梯度 NaN 时自动跳过 (zero_grad + return 999.0)。
    Auto-skips gradient NaN (zero_grad + return 999.0).

    :return: (loss_value, metrics_dict).
    """
    decoder.train()

    # ── 1. Support → Prototype | Support → Prototype ──
    support_proto = compute_support_prototype(
        backbone, support_imgs, support_masks, device
    )

    # ── 2. Query → Backbone Features + Proto | Query → Backbone + Proto ──
    feats = backbone(query_img, extract_proto=True)
    p4 = feats["p4"]          # [1, 1280, H/16, W/16]
    proto_masks = feats.get("proto")  # [1, 32, H/4, W/4] or None

    # ── 3. SPM (可选) | SPM (optional) ──
    spm_map = None
    if use_spm and spm is not None:
        spm_out = spm(feats["p8"])  # [1, 1, H/32, W/32]
        spm_map = spm_out

    # ── 4. Decoder → Mask | Decoder → Mask ──
    if isinstance(decoder, ProtoOnlyDecoder):
        if proto_masks is None:
            raise RuntimeError("ProtoOnlyDecoder requires proto_masks but backbone returned None. "
                               "Ensure extract_proto=True.")
        pred_mask = decoder(proto_masks, support_proto)
    else:
        if proto_masks is None:
            raise RuntimeError("AdaptiveSparseDecoder requires proto_masks but backbone returned None. "
                               "Ensure extract_proto=True.")
        pred_mask = decoder(p4, proto_masks, support_proto, spm_map)

    # ── 去 batch 维度 + 上采样到 GT 分辨率 | Remove batch dim + upsample to GT resolution ──
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.squeeze(0)  # [H/4, W/4]
    pred_full = F.interpolate(
        pred_mask.unsqueeze(0).unsqueeze(0),
        size=query_mask.shape,
        mode="bilinear", align_corners=False,
    ).squeeze(0).squeeze(0)  # [H, W]

    # ── 5. 损失 | Loss ──
    loss, loss_dict = combined_loss(pred_full, query_mask.float().to(device))

    # ── 6. NaN 诊断 + 安全跳过 | NaN Diagnostic + Safe Skip ──
    if torch.isnan(loss) or torch.isinf(loss):
        logger = get_logger("train_base")
        logger.log_info("nan_diag", f"NaN/Inf loss at class_id={class_id}!")
        logger.log_info("nan_diag", f"  pred_full: nan={pred_full.isnan().any().item()}, "
                        f"range=[{pred_full.min().item():.4f}, {pred_full.max().item():.4f}]")
        logger.log_info("nan_diag", f"  query_mask sum={query_mask.sum().item()}")
        return 999.0, {"loss": 999.0, "focal": 999.0, "dice": 999.0,
                       "pred_mean": 0.0, "class_id": class_id}

    optimizer.zero_grad()
    loss.backward()

    # 梯度裁剪 | Gradient clipping
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

    # 检测梯度 NaN → 跳过 | Detect gradient NaN → skip
    grad_nan = False
    for name, param in decoder.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                logger = get_logger("train_base")
                logger.log_info("nan_diag", f"  GRAD NaN in {name}")
                grad_nan = True
                break
    if grad_nan:
        optimizer.zero_grad()
        return 999.0, {"loss": 999.0, "focal": 999.0, "dice": 999.0,
                       "pred_mean": 0.0, "class_id": class_id}

    optimizer.step()

    return loss.item(), {
        "loss": loss.item(),
        "focal": loss_dict["focal"],
        "dice": loss_dict["dice"],
        "pred_mean": pred_full.mean().item(),
        "class_id": class_id,
    }


# ═══════════════════════════════════════════════════════════════════
# 训练采样器 | Training Sampler
# ═══════════════════════════════════════════════════════════════════

class BaseClassSampler:
    """
    Base 类训练采样器 | Base Class Training Sampler.

    每步: 随机选 Base 类 → K support tiles + 1 query tile.
    Each step: random Base class → K support tiles + 1 query tile.

    Parameters
    ----------
    dataset : ISAIDInstanceFewShotDataset (mode="base")
    class_ids : list[int]
        Base 类 ID 列表 | List of Base class IDs.
    k_support : int
        每步 support tile 数量 | Number of support tiles per step.
    min_tiles : int
        每类最少 tile 数。低于此值的类自动排除（防 NaN）。
        Minimum tiles per class. Classes below this are auto-excluded (prevents NaN).
    seed : int
        随机种子 | Random seed.
    """

    def __init__(
        self,
        dataset: ISAIDInstanceFewShotDataset,
        class_ids: list[int],
        k_support: int = 5,
        min_tiles: int = 30,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.k_support = k_support
        self.rng = random.Random(seed)

        # 预构建每类 tile 列表 + 过滤数据过少的类 | Pre-build per-class tile list + filter scarce
        self._class_tiles: dict[int, list[int]] = {}
        self._excluded: dict[int, int] = {}

        for cls_id in class_ids:
            tiles = dataset.class_to_tiles(cls_id)
            if tiles and len(tiles) >= min_tiles:
                self._class_tiles[cls_id] = tiles
            elif tiles:
                self._excluded[cls_id] = len(tiles)

        if self._excluded:
            excluded_str = ", ".join(
                f"{ISAID_CAT_NAMES.get(c, f'cls{c}')}({c}:{n}tiles)"
                for c, n in self._excluded.items()
            )
            print(f"[BaseClassSampler] ⚠ Excluded {len(self._excluded)} scarce classes "
                  f"(<{min_tiles} tiles): {excluded_str}")

        valid = len(self._class_tiles)
        print(f"[BaseClassSampler] {valid}/{len(class_ids)} Base classes ready "
              f"(k_support={k_support}, min_tiles={min_tiles})")
        if valid == 0:
            raise ValueError(f"No classes have >= {min_tiles} tiles! "
                             f"Check dataset or lower --min-tiles.")

    def sample(self) -> dict:
        """
        采样一步 | Sample one step.

        :return: {
            "class_id": int,
            "support_indices": list[int],   # K indices into dataset
            "query_index": int,              # 1 index into dataset
        }
        """
        cls_id = self.rng.choice(list(self._class_tiles.keys()))
        candidates = self._class_tiles[cls_id]

        if len(candidates) <= self.k_support:
            support_idx = candidates[:]
            query_idx = self.rng.choice(candidates)
        else:
            sampled = self.rng.sample(candidates, self.k_support + 1)
            support_idx = sampled[:self.k_support]
            query_idx = sampled[self.k_support]

        return {
            "class_id": cls_id,
            "support_indices": support_idx,
            "query_index": query_idx,
        }

    @property
    def active_classes(self) -> list[int]:
        """活跃的 (有足够数据的) 类别 | Active (sufficient data) classes."""
        return sorted(self._class_tiles.keys())


# ═══════════════════════════════════════════════════════════════════
# 评估 | Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(
    decoder: nn.Module,
    backbone: FastSAMBackbone,
    spm: SparsePerceptionModule | None,
    dataset: ISAIDInstanceFewShotDataset,
    class_ids: list[int],
    device: torch.device,
    use_spm: bool = False,
    max_samples_per_class: int = 20,
) -> dict:
    """
    快速评估: per-class IoU + 可选 COCO AP | Quick eval: per-class IoU + optional COCO AP.

    :param dataset: val split dataset.
    :param class_ids: 要评估的类别 | Classes to evaluate.
    :param max_samples_per_class: 每类最多评估 tile 数 | Max tiles per class (0 = all).
    :return: {"miou": float, "per_class_iou": dict, "n_eval": int}.
    """
    decoder.eval()

    per_class_iou: dict[int, list[float]] = defaultdict(list)
    n_eval = 0

    for cls_id in tqdm(class_ids, desc="Eval", leave=False):
        tile_indices = dataset.class_to_tiles(cls_id)
        if not tile_indices:
            continue
        if max_samples_per_class > 0 and len(tile_indices) > max_samples_per_class:
            tile_indices = random.Random(42).sample(tile_indices, max_samples_per_class)

        for idx in tile_indices:
            sample = dataset[idx]
            instances = sample["instances"]

            # 过滤目标类实例 | Filter target class instances
            cls_instances = [i for i in instances if i["category_id"] == cls_id]
            if not cls_instances:
                continue

            # GT union mask (合并同类实例) | GT union mask (merge same-class instances)
            H, W = sample["image"].shape[1:]
            gt_union = torch.zeros(H, W, dtype=torch.bool)
            for inst in cls_instances:
                gt_union = torch.logical_or(gt_union, inst["mask"])

            query_img = sample["image"].unsqueeze(0).to(device)

            # ── Backbone ──
            feats = backbone(query_img, extract_proto=True)
            p4 = feats["p4"]
            proto_masks = feats.get("proto")

            # ── Support prototype (self-support: use query's own FG for eval) ──
            mask_ds = F.interpolate(
                gt_union.float().unsqueeze(0).unsqueeze(0),
                size=p4.shape[2:], mode="nearest",
            ).squeeze(1)  # [1, H/16, W/16]

            fg = mask_ds > 0.5
            if fg.sum() < 16:
                continue

            proto = p4[:, :, fg.squeeze(0)].mean(dim=-1).squeeze(0)
            proto = F.normalize(proto, dim=0)

            # ── SPM ──
            spm_map = None
            if use_spm and spm is not None:
                spm_map = spm(feats["p8"])

            # ── Decoder ──
            if isinstance(decoder, ProtoOnlyDecoder):
                if proto_masks is None:
                    continue
                pred_prob = decoder(proto_masks, proto)
            else:
                if proto_masks is None:
                    continue
                pred_prob = decoder(p4, proto_masks, proto, spm_map)
            if pred_prob.dim() == 3:
                pred_prob = pred_prob.squeeze(0)

            pred_full = F.interpolate(
                pred_prob.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)

            # Binary IoU
            pred_bin = (pred_full > 0.5).float()
            inter = (pred_bin * gt_union.float().to(device)).sum()
            union = (pred_bin + gt_union.float().to(device)).clamp(0, 1).sum()
            if union > 0:
                per_class_iou[cls_id].append((inter / union).item())
            n_eval += 1

    # ── 汇总 | Aggregate ──
    cls_means = {}
    for cls_id, ious in per_class_iou.items():
        cls_means[cls_id] = np.mean(ious) if ious else 0.0
    miou = np.mean(list(cls_means.values())) if cls_means else 0.0

    return {
        "miou": round(float(miou), 6),
        "per_class_iou": {k: round(float(v), 6) for k, v in cls_means.items()},
        "n_eval": n_eval,
    }


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="V3-05: Base Class Pre-Training for Few-Shot Instance Segmentation"
    )

    # 数据 | Data
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                   help="v3 数据根目录 | v3 data root")
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2],
                   help="Fold ID (决定 Base/Novel 划分)")

    # 训练 | Training
    p.add_argument("--epochs", type=int, default=50,
                   help="训练轮数 (每轮 = steps_per_epoch 步) | Training epochs")
    p.add_argument("--steps-per-epoch", type=int, default=500,
                   help="每轮训练步数 | Training steps per epoch")
    p.add_argument("--k-support", type=int, default=5,
                   help="每步 support tile 数 | Support tiles per step")

    # 模型 | Model
    p.add_argument("--decoder-type", type=str, default="adaptive",
                   choices=["adaptive", "proto_only"],
                   help="Decoder 类型: 'adaptive' (coeff + P4 refine) or 'proto_only'")
    p.add_argument("--use-spm", action="store_true",
                   help="启用 SPM 联合训练 (默认关闭, Decoder-SPM 解耦) | Enable SPM joint training")
    p.add_argument("--freeze-backbone", action="store_true", default=True,
                   help="冻结 backbone (默认) | Freeze backbone (default)")
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                   help="解冻 backbone | Unfreeze backbone")

    # 优化器 | Optimizer
    p.add_argument("--lr", type=float, default=1e-4,
                   help="学习率 | Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="权重衰减 | Weight decay")
    p.add_argument("--min-tiles", type=int, default=30,
                   help="每类最少 tile 数 (过滤稀有类) | Min tiles per class (filter rare)")

    # 硬件 | Hardware
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # 输出 | Output
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=5,
                   help="每 N epochs 评估一次 | Evaluate every N epochs")
    p.add_argument("--eval-max-samples", type=int, default=20,
                   help="评估时每类最多 tile 数 (0=全部) | Max eval tiles per class")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# 主函数 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # ── 输出目录 | Output ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        dec = "Adaptive" if args.decoder_type == "adaptive" else "ProtoOnly"
        args.output_dir = f"runs/v3_05_base_pretrain_F{args.fold}_{dec}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logger ──
    logger = get_logger("train_base")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))

    # ── 类别配置 | Class Config ──
    # 从数据集 fold 定义读取 Base/Novel 划分
    # Read Base/Novel split from dataset fold definition
    fold_path = Path(args.data_root) / "folds" / f"fold_{args.fold}.json"
    if fold_path.exists():
        with open(fold_path) as f:
            fold_data = json.load(f)
        base_ids = fold_data["base"]
        novel_ids = fold_data["novel"]
    else:
        # 回退到硬编码 | Fallback to hardcoded
        from adatile.datasets.isaid_instance_fewshot import DEFAULT_FOLDS
        fd = DEFAULT_FOLDS[args.fold]
        base_ids = fd["base"]
        novel_ids = fd["novel"]

    base_names = [ISAID_CAT_NAMES.get(c, f"cls{c}") for c in base_ids]
    novel_names = [ISAID_CAT_NAMES.get(c, f"cls{c}") for c in novel_ids]

    logger.log_info("config", f"Fold {args.fold}: Base={base_ids} ({base_names})")
    logger.log_info("config", f"Fold {args.fold}: Novel={novel_ids} ({novel_names})")
    logger.log_info("config", f"Decoder={args.decoder_type}, SPM={args.use_spm}, "
                    f"K_support={args.k_support}, epochs={args.epochs}, "
                    f"steps/epoch={args.steps_per_epoch}, lr={args.lr}")

    # ── 数据集 | Datasets ──
    logger.log_info("data", "Loading v3 Instance Few-Shot datasets...")
    train_ds = ISAIDInstanceFewShotDataset(
        root=args.data_root, split="train", fold=args.fold, mode="base",
    )
    val_ds = ISAIDInstanceFewShotDataset(
        root=args.data_root, split="val", fold=args.fold, mode="base",
    )
    logger.log_info("data", f"Train: {train_ds.tile_count} tiles, "
                    f"Val: {val_ds.tile_count} tiles")

    # ── 模型 | Models ──
    logger.log_info("model", "Building backbone...")
    backbone = FastSAMBackbone(freeze_backbone=args.freeze_backbone).to(device)
    backbone.eval()

    if args.decoder_type == "proto_only":
        decoder = ProtoOnlyDecoder(proto_dim=32, feat_dim=1280).to(device)
        logger.log_info("model", f"ProtoOnlyDecoder: ~{sum(p.numel() for p in decoder.parameters())/1e3:.1f}K params")
    else:
        decoder = AdaptiveSparseDecoder(
            in_channels=1280, proto_dim=32, use_fdr=args.use_spm,
        ).to(device)
        params = decoder.get_submodule_params()
        logger.log_info("model", f"AdaptiveSparseDecoder: {params}")

    spm = None
    if args.use_spm:
        spm = SparsePerceptionModule(in_channels=1280).to(device)
        # SPM 训练需要 density focal + top-K BCE + budget loss 三件套
        # 暂不实现完整 SPM 训练，仅作为 decoder 的 gating signal
        # Full SPM training needs 3-loss combo — not implemented yet
        logger.log_info("model", "SPM loaded (inference-only, not trained)")
        spm.eval()

    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    logger.log_info("model", f"Trainable params: {trainable_params:,}")

    # ── 优化器 | Optimizer ──
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.steps_per_epoch,
    )

    # ── 采样器 | Sampler ──
    sampler = BaseClassSampler(
        train_ds, base_ids, k_support=args.k_support,
        min_tiles=args.min_tiles, seed=args.seed,
    )
    active_base_ids = sampler.active_classes
    logger.log_info("data", f"Active Base classes: {active_base_ids}")

    # ── 训练循环 | Training Loop ──
    logger.log_info("train", f"{'='*60}")
    logger.log_info("train", f"Starting Base pre-training: {args.epochs} epochs × {args.steps_per_epoch} steps")
    logger.log_info("train", f"{'='*60}")

    best_miou = 0.0
    global_step = 0
    nan_skip_count = 0

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        epoch_focal = []
        epoch_dice = []
        epoch_pred_mean = []

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch:3d}/{args.epochs}", unit="step")
        for _ in pbar:
            # ── 采样 | Sample ──
            episode = sampler.sample()
            cls_id = episode["class_id"]

            # ── 加载 support tiles | Load support tiles ──
            support_imgs = []
            support_masks = []
            for s_idx in episode["support_indices"]:
                s_sample = train_ds[s_idx]
                support_imgs.append(s_sample["image"])
                # 构建 per-class binary mask | Build per-class binary mask
                H_s, W_s = s_sample["image"].shape[1:]
                s_mask = torch.zeros(H_s, W_s, dtype=torch.float32)
                for inst in s_sample["instances"]:
                    if inst["category_id"] == cls_id:
                        s_mask = torch.logical_or(s_mask, inst["mask"]).float()
                support_masks.append(s_mask)

            support_imgs = torch.stack(support_imgs)   # [K, 3, H, W]
            support_masks = torch.stack(support_masks)  # [K, H, W]

            # ── 加载 query tile | Load query tile ──
            q_sample = train_ds[episode["query_index"]]
            query_img = q_sample["image"].unsqueeze(0)  # [1, 3, H, W]
            H_q, W_q = q_sample["image"].shape[1:]
            q_mask = torch.zeros(H_q, W_q, dtype=torch.float32)
            for inst in q_sample["instances"]:
                if inst["category_id"] == cls_id:
                    q_mask = torch.logical_or(q_mask, inst["mask"]).float()

            if q_mask.sum() == 0:
                continue  # 跳过无 FG 的 query | Skip empty query

            # ── 训练一步 | Train step ──
            loss_val, metrics = train_step(
                decoder, backbone, spm,
                support_imgs, support_masks,
                query_img, q_mask,
                cls_id, optimizer, device,
                use_spm=args.use_spm,
            )

            if loss_val >= 999.0:
                nan_skip_count += 1
                continue

            scheduler.step()
            global_step += 1

            epoch_losses.append(loss_val)
            epoch_focal.append(metrics["focal"])
            epoch_dice.append(metrics["dice"])
            epoch_pred_mean.append(metrics["pred_mean"])

            # 更新进度条 | Update progress bar
            if len(epoch_losses) > 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(epoch_losses[-50:]):.4f}",
                    "dice": f"{np.mean(epoch_dice[-50:]):.4f}",
                    "pred": f"{np.mean(epoch_pred_mean[-50:]):.4f}",
                })

        # ── Epoch 汇总 | Epoch Summary ──
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_focal = np.mean(epoch_focal) if epoch_focal else 0.0
        avg_dice = np.mean(epoch_dice) if epoch_dice else 0.0
        avg_pred = np.mean(epoch_pred_mean) if epoch_pred_mean else 0.0

        logger.log_info("epoch",
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} focal={avg_focal:.4f} dice={avg_dice:.4f} "
            f"pred_mean={avg_pred:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"NaN_skip={nan_skip_count}"
        )
        logger.log_metric("loss", avg_loss, step=epoch, tags=["base_train"])
        logger.log_metric("dice", avg_dice, step=epoch, tags=["base_train"])
        logger.log_metric("pred_mean", avg_pred, step=epoch, tags=["base_train"])

        # ── 定期评估 | Periodic Evaluation ──
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            logger.log_info("eval", f"--- Evaluation @ Epoch {epoch} ---")
            eval_result = evaluate(
                decoder, backbone, spm, val_ds, active_base_ids, device,
                use_spm=args.use_spm, max_samples_per_class=args.eval_max_samples,
            )
            miou = eval_result["miou"]
            logger.log_info("eval",
                f"Epoch {epoch:3d}: mIoU={miou:.4f} (eval on {eval_result['n_eval']} tiles) "
                f"best={best_miou:.4f}")
            logger.log_metric("miou", miou, step=epoch, tags=["base_eval"])

            # 保存最佳模型 | Save best model
            if miou > best_miou:
                best_miou = miou
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "miou": miou,
                    "per_class_iou": eval_result["per_class_iou"],
                    "args": vars(args),
                    "base_ids": base_ids,
                    "novel_ids": novel_ids,
                }
                torch.save(checkpoint, str(out_dir / "best_model.pt"))
                logger.log_info("eval", f"  ✓ New best: mIoU={best_miou:.4f}")

    # ── 最终保存 | Final Save ──
    final_checkpoint = {
        "epoch": args.epochs,
        "global_step": global_step,
        "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "best_miou": best_miou,
        "args": vars(args),
        "base_ids": base_ids,
        "novel_ids": novel_ids,
        "nan_skip_count": nan_skip_count,
    }
    torch.save(final_checkpoint, str(out_dir / "last_model.pt"))

    # ── 保存结果 JSON | Save Results JSON ──
    results = {
        "experiment": "V3-05 Base Pre-Training",
        "fold": args.fold,
        "decoder_type": args.decoder_type,
        "use_spm": args.use_spm,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "k_support": args.k_support,
        "best_miou": round(best_miou, 6),
        "nan_skip_count": nan_skip_count,
        "base_classes": {c: ISAID_CAT_NAMES.get(c, str(c)) for c in base_ids},
        "novel_classes": {c: ISAID_CAT_NAMES.get(c, str(c)) for c in novel_ids},
        "trainable_params": trainable_params,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 摘要 | Summary ──
    print(f"\n{'='*60}")
    print(f"  V3-05 Base Pre-Training — Complete")
    print(f"  Fold: {args.fold}, Decoder: {args.decoder_type}")
    print(f"  Epochs: {args.epochs}, Steps: {global_step}")
    print(f"  Best mIoU: {best_miou:.4f}")
    print(f"  NaN skips: {nan_skip_count}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    logger.log_info("done", f"Output: {out_dir}")
    logger.log_info("done", f"Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()
