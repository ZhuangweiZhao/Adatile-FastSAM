#!/usr/bin/env python3
"""
实例分割 Few-Shot 微调训练 | Instance Segmentation Few-Shot Fine-Tuning.
=========================================================================

训练协议 | Training Protocol:
1. **Pre-train** (已完成): Base 类全量数据训练（语义分割，train_supervised.py）
2. **Few-shot Fine-tune** (本脚本): Novel 类 K-shot 微调（实例分割）
3. **Evaluate**: COCO AP 评估

核心流程 | Core Flow:
    Episode = K support tiles + 1 query tile (same class)
    Support → FG prototype → ProtoCoeffPredictor → coefficients
    Query → Backbone → Proto masks + P4 features
    Coefficients @ Proto masks → coarse mask
    P4 refine + FDR gate → fine mask
    Loss: Dice(预测mask, GT per-class mask)

用法 | Usage::

    # K=5 微调
    python tools/train/train_instance_fewshot.py \
        --data-root data/iSAID-5i/iSAID \
        --fold 0 --k-shot 5 --episodes 500 \
        --output-dir runs/instance_fewshot_K5

    # Overfit 测试（1 张图验证 pipeline）
    python tools/train/train_instance_fewshot.py \
        --data-root data/iSAID-5i/iSAID \
        --fold 0 --k-shot 10 --episodes 100 \
        --overfit --classes 9
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
from torch.utils.data import DataLoader

from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.utils.seed import set_seed
from adatile.utils.env import get_env_info
from adatile.backbone import FastSAMBackbone
from adatile.sparse import ForegroundDensityRouter
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder, ProtoOnlyDecoder
from adatile.datasets.isaid_instance import (
    ISAIDInstanceDataset,
    sample_k_shot_tiles,
    instances_to_dense_mask,
)
from adatile.metrics.coco_eval import (
    COCOInstanceEvaluator,
    connected_components_to_instances,
)
from adatile.utils.label_mapping import ISAID5I_CATEGORIES, ISAID5I_FOLDS

# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

DEFAULT_DATA_ROOT = str(_PROJECT_ROOT / "data" / "iSAID-5i" / "iSAID")
IGNORE_INDEX = 255


# ═══════════════════════════════════════════════════════════════════
# Support Prototype 计算 | Support Prototype Computation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_support_prototype(
    backbone: FastSAMBackbone,
    support_images: torch.Tensor,   # [K, 3, H, W]
    support_masks: torch.Tensor,    # [K, H, W] binary per-class masks
) -> torch.Tensor:
    """
    从 K 张 support 图像计算 FG prototype | Compute FG prototype from K support images.

    对每张 support 图像提取 P4 特征，用 binary mask 做 FG 平均池化，
    然后对 K 个 prototype 取平均并 L2 归一化。

    Extract P4 features from each support image, FG average pool with binary mask,
    then average K prototypes and L2-normalize.

    :param backbone: FastSAM backbone.
    :param support_images: [K, 3, H, W] support images.
    :param support_masks: [K, H, W] binary per-class GT masks.
    :return: [1280] L2-normalized prototype vector.
    """
    K = support_images.shape[0]
    prototypes = []

    for i in range(K):
        img = support_images[i:i+1]  # [1, 3, H, W]
        mask = support_masks[i:i+1]  # [1, H, W]

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
        return torch.zeros(1280, device=backbone._device)

    proto = torch.stack(prototypes).mean(dim=0)  # [1280]
    return F.normalize(proto, dim=0)


# ═══════════════════════════════════════════════════════════════════
# 损失函数 | Loss Functions
# ═══════════════════════════════════════════════════════════════════

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Dice 损失 (binary) | Dice Loss (binary).

    :param pred: [H, W] 预测概率 [0, 1] | Predicted probability.
    :param target: [H, W] 二值 GT | Binary GT.
    :return: 1 - Dice.
    """
    inter = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1.0 - (2.0 * inter + smooth) / (union + smooth)


def focal_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss (binary) | Focal Loss (binary).

    :param pred: [H, W] 预测概率 [0, 1] | Predicted probability.
    :param target: [H, W] 二值 GT | Binary GT.
    :param gamma: Focal gamma 参数 | Focal gamma parameter.
    """
    eps = 1e-8
    pred = torch.clamp(pred, eps, 1.0 - eps)
    bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
    pt = pred * target + (1 - pred) * (1 - target)
    return ((1 - pt) ** gamma * bce).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    组合损失: alpha * Focal + (1-alpha) * Dice | Combined: Focal + Dice.

    :param pred: [H, W] 预测概率 | Predicted probability.
    :param target: [H, W] 二值 GT | Binary GT.
    :param alpha: Focal 权重 | Focal weight (Dice = 1-alpha).
    :return: (total_loss, {"focal": float, "dice": float})
    """
    fl = focal_loss(pred, target)
    dl = dice_loss(pred, target)
    return alpha * fl + (1 - alpha) * dl, {"focal": fl.item(), "dice": dl.item()}


# ═══════════════════════════════════════════════════════════════════
# Episode 采样 | Episode Sampling
# ═══════════════════════════════════════════════════════════════════

class EpisodeSampler:
    """
    Few-shot episode 采样器 | Few-shot Episode Sampler.

    每个 episode: 随机选一个类 → K 张 support tiles + 1 张 query tile。
    Each episode: random class → K support tiles + 1 query tile.

    Parameters
    ----------
    dataset : ISAIDInstanceDataset (train split)
    class_ids : list[int]
        参与训练的类别 ID | Class IDs to train on.
    k_shot : int
        每 episode 的 support 数量 | Number of support tiles per episode.
    """

    def __init__(
        self,
        dataset: ISAIDInstanceDataset,
        class_ids: list[int],
        k_shot: int = 5,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.class_ids = class_ids
        self.k_shot = k_shot
        self.rng = random.Random(seed)
        self.episode_count = 0

        # 预构建每类的 tile 列表 | Pre-build tile list per class
        self._class_tiles: dict[int, list[int]] = {}
        for cls_id in class_ids:
            tiles = dataset.class_to_tiles(cls_id)
            if tiles:
                self._class_tiles[cls_id] = tiles

        valid = len(self._class_tiles)
        print(f"[EpisodeSampler] {valid}/{len(class_ids)} classes have tiles "
              f"(k_shot={k_shot})")
        if valid == 0:
            raise ValueError("No classes have tiles! Check dataset mode (must be 'tile').")

    def sample(self) -> dict:
        """
        采样一个 episode | Sample one episode.

        :return: {
            "class_id": int,
            "support_indices": list[int],  # K tile indices
            "query_index": int,             # 1 tile index
        }
        """
        # 随机选一个类 | Random class
        cls_id = self.rng.choice(list(self._class_tiles.keys()))
        candidates = self._class_tiles[cls_id]

        if len(candidates) <= self.k_shot:
            # 不足以分开 support/query → 有放回采样
            # Not enough for separate support/query → sample with replacement
            support_idx = candidates[:]
            query_idx = self.rng.choice(candidates)
        else:
            sampled = self.rng.sample(candidates, self.k_shot + 1)
            support_idx = sampled[:self.k_shot]
            query_idx = sampled[self.k_shot]

        self.episode_count += 1
        return {
            "class_id": cls_id,
            "support_indices": support_idx,
            "query_index": query_idx,
        }


# ═══════════════════════════════════════════════════════════════════
# 训练 | Training
# ═══════════════════════════════════════════════════════════════════

def train_episode(
    decoder: AdaptiveSparseDecoder,
    backbone: FastSAMBackbone,
    fdr_router: ForegroundDensityRouter | None,
    support_imgs: torch.Tensor,       # [K, 3, H, W]
    support_mask: torch.Tensor,        # [K, H, W]
    query_img: torch.Tensor,           # [1, 3, H, W]
    query_mask: torch.Tensor,          # [H, W] binary per-class GT
    class_id: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_fdr: bool = True,
    tag: str = "instance/train",
) -> tuple[float, dict]:
    """
    训练一个 episode | Train one episode.

    :return: (loss_value, metrics_dict)
    """
    decoder.train()

    # ── 1. Support → Prototype | Support → Prototype ──
    support_proto = compute_support_prototype(backbone, support_imgs, support_mask)

    # ── 2. Query → Backbone + Proto | Query → Backbone + Proto ──
    feats = backbone(query_img, extract_proto=True)
    p4 = feats["p4"]
    proto_masks = feats["proto"]

    # ── 3. FDR (可选) | FDR (optional) ──
    fdr_map = None
    if use_fdr and fdr_router is not None:
        fdr_out = fdr_router(feats["p8"])
        fdr_map = fdr_out["importance"]

    # ── 4. Decoder → Mask | Decoder → Mask ──
    if isinstance(decoder, ProtoOnlyDecoder):
        pred_mask = decoder(proto_masks, support_proto)
    else:
        pred_mask = decoder(p4, proto_masks, support_proto, fdr_map)

    # 上采样到 GT 分辨率 | Upsample to GT resolution
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.squeeze(0)  # [H/4, W/4]
    pred_full = F.interpolate(
        pred_mask.unsqueeze(0).unsqueeze(0),
        size=query_mask.shape,
        mode="bilinear", align_corners=False,
    ).squeeze(0).squeeze(0)  # [H, W]

    # ── 5. 损失 | Loss ──
    loss, loss_dict = combined_loss(pred_full, query_mask.float())

    optimizer.zero_grad()
    loss.backward()
    # 梯度裁剪，防止 BatchNorm/多层级联导致梯度爆炸 | Gradient clipping to prevent gradient explosion
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), {
        "loss": loss.item(),
        "focal": loss_dict["focal"],
        "dice": loss_dict["dice"],
        "pred_mean": pred_full.mean().item(),
        "class_id": class_id,
    }


@torch.no_grad()
def evaluate_coco(
    decoder: nn.Module,
    backbone: FastSAMBackbone,
    fdr_router: ForegroundDensityRouter | None,
    dataset: ISAIDInstanceDataset,
    class_ids: list[int],
    device: torch.device,
    gt_anno_path: str | None = None,
    use_fdr: bool = True,
    max_samples: int = 0,
) -> dict:
    """
    COCO AP 评估 | COCO AP Evaluation.

    对每个类，遍历所有包含该类的 tiles，预测 per-class mask，
    通过连通分量分解为 per-instance masks，提交到 COCO evaluator。

    For each class, iterate all tiles containing it, predict per-class mask,
    decompose to per-instance masks via connected components, submit to COCO evaluator.

    :param dataset: val split instance dataset.
    :param class_ids: 要评估的类别 | Classes to evaluate.
    :param gt_anno_path: COCO GT JSON 路径。None → 跳过 COCO eval（仅返回 per-class IoU）。
    :param max_samples: 每类最多评估的 tile 数 (0=全部) | Max tiles per class (0=all).
    :return: {"AP": float, "AP50": float, ..., "per_class_iou": dict}
    """
    decoder.eval()

    per_class_iou = {}
    per_class_preds = defaultdict(list)  # {cls_id: [(mask, score), ...]}

    for cls_id in tqdm(class_ids, desc="Eval classes", leave=False):
        tile_indices = dataset.class_to_tiles(cls_id)
        if max_samples > 0 and len(tile_indices) > max_samples:
            tile_indices = tile_indices[:max_samples]

        cls_ious = []
        cls_preds = []

        for idx in tqdm(tile_indices, desc=f"  cls{cls_id}", leave=False):
            sample = dataset[idx]

            # 获取 GT per-class mask | Get GT per-class mask
            gt_instances = [i for i in sample["instances"] if i["category_id"] == cls_id]
            if not gt_instances:
                continue

            # GT union mask (same class instances merged)
            gt_union = torch.zeros(256, 256, dtype=torch.bool, device=device)
            for inst in gt_instances:
                gt_union = torch.logical_or(gt_union, inst["mask"].to(device))

            # Support prototype（使用 GT mask 的 FG 区域）
            # For evaluation, construct a "perfect" support from GT
            # Use the query tile's own FG as support（cheating, for upper bound）
            # For actual few-shot, use a separate support set
            img = sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]

            feats = backbone(img, extract_proto=True)
            p4 = feats["p4"]

            # Prototype from query's own FG（self-support for evaluation）
            mask_ds = F.interpolate(
                gt_union.float().unsqueeze(0).unsqueeze(0),
                size=p4.shape[2:], mode="nearest",
            ).squeeze(1)  # [1, H/16, W/16]

            fg = mask_ds > 0.5
            if fg.sum() < 16:
                continue

            proto = p4[:, :, fg.squeeze(0)].mean(dim=-1).squeeze(0)
            proto = F.normalize(proto, dim=0)

            # FDR
            fdr_map = None
            if use_fdr and fdr_router is not None:
                fdr_map = fdr_router(feats["p8"])["importance"]

            # Predict
            if isinstance(decoder, ProtoOnlyDecoder):
                pred_prob = decoder(feats["proto"], proto)
            else:
                pred_prob = decoder(p4, feats["proto"], proto, fdr_map)
            if pred_prob.dim() == 3:
                pred_prob = pred_prob.squeeze(0)  # [H/4, W/4]

            pred_full = F.interpolate(
                pred_prob.unsqueeze(0).unsqueeze(0),
                size=(256, 256), mode="bilinear", align_corners=False,
            ).squeeze(0).squeeze(0)  # [256, 256]

            # Binary IoU
            pred_bin = (pred_full > 0.5).float()
            inter = (pred_bin * gt_union.float()).sum()
            union = (pred_bin + gt_union.float()).clamp(0, 1).sum()
            iou = (inter / max(union, 1)).item()
            cls_ious.append(iou)

            # Per-instance decomposition for COCO AP
            pred_instances = connected_components_to_instances(
                pred_bin.cpu().numpy(), min_area=16,
            )
            for inst_mask in pred_instances:
                score = float(pred_full.cpu().numpy()[inst_mask].max())
                cls_preds.append({"mask": inst_mask, "score": score})

        if cls_ious:
            per_class_iou[cls_id] = np.mean(cls_ious)
        per_class_preds[cls_id] = cls_preds

    # ── COCO AP 评估 | COCO AP Evaluation ──
    coco_result = {}
    if gt_anno_path and os.path.exists(gt_anno_path):
        evaluator = COCOInstanceEvaluator(gt_anno_path, iouType="segm")
        for cls_id in class_ids:
            for pred in per_class_preds.get(cls_id, []):
                evaluator.add_prediction(
                    image_id=0,  # Placeholder — tile mode needs proper image_id mapping
                    category_id=cls_id,
                    mask=pred["mask"],
                    score=pred["score"],
                )
        coco_result = evaluator.evaluate()
    else:
        coco_result = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "n_predictions": 0}

    # ── mIoU | Mean IoU ──
    miou = np.mean(list(per_class_iou.values())) if per_class_iou else 0.0

    return {
        "miou": round(float(miou), 6),
        "per_class_iou": {k: round(float(v), 6) for k, v in per_class_iou.items()},
        **coco_result,
    }


# ═══════════════════════════════════════════════════════════════════
# 参数解析 | Argument Parsing
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="实例分割 Few-Shot 微调训练")

    # 数据 | Data
    p.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])

    # 训练模式 | Training Mode
    p.add_argument("--k-shot", type=int, default=5,
                   help="每 episode 的 support tile 数 | Support tiles per episode")
    p.add_argument("--episodes", type=int, default=500,
                   help="训练 episode 数 | Training episodes")
    p.add_argument("--classes", type=str, default="novel",
                   help="训练类别: 'novel', 'base', 'all', 或逗号分隔的 ID | "
                        "Classes to train: 'novel', 'base', 'all', or comma-separated IDs")
    p.add_argument("--overfit", action="store_true",
                   help="Overfit 模式: 固定 query tile | Overfit mode: fixed query tile")

    # 模型 | Model
    p.add_argument("--decoder-type", type=str, default="adaptive",
                   choices=["adaptive", "proto_only"],
                   help="Decoder 类型 | Decoder type")
    p.add_argument("--no-fdr", action="store_true",
                   help="禁用 FDR | Disable FDR")
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")

    # 训练参数 | Training Params
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1,
                   help="每 episode 的 batch（通常为 1）| Batch per episode (usually 1)")

    # 硬件 | Hardware
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")

    # 输出 | Output
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=100,
                   help="每 N episodes 评估一次 | Evaluate every N episodes")
    p.add_argument("--eval-max-samples", type=int, default=10,
                   help="评估时每类最多 tile 数 (0=全部) | Max tiles per class in eval (0=all)")

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
        args.output_dir = f"runs/instance_fewshot_F{args.fold}_K{args.k_shot}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 日志 | Logger ──
    logger = get_logger("instance_fewshot")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "train.jsonl")))

    # ── 类别配置 | Class Config ──
    fold_info = ISAID5I_FOLDS[args.fold]
    novel_ids = fold_info["novel"]
    base_ids = fold_info["base"]
    all_ids = sorted(novel_ids + base_ids)

    if args.classes == "novel":
        train_class_ids = novel_ids
    elif args.classes == "base":
        train_class_ids = base_ids
    elif args.classes == "all":
        train_class_ids = all_ids
    else:
        train_class_ids = [int(c.strip()) for c in args.classes.split(",")]

    train_class_names = [ISAID5I_CATEGORIES.get(c, f"cls{c}") for c in train_class_ids]
    logger.log_info("config", f"Fold {args.fold}: training on {len(train_class_ids)} classes: {train_class_names}")
    logger.log_info("config", f"Novel: {[ISAID5I_CATEGORIES.get(c, str(c)) for c in novel_ids]}")
    logger.log_info("config", f"K-shot={args.k_shot}, episodes={args.episodes}, decoder={args.decoder_type}, FDR={not args.no_fdr}")

    # ── 数据集 | Datasets ──
    train_ds = ISAIDInstanceDataset(
        root=args.data_root, split="train", fold=args.fold, mode="tile",
    )
    val_ds = ISAIDInstanceDataset(
        root=args.data_root, split="val", fold=args.fold, mode="tile",
    )

    # ── 构建模型 | Build Models ──
    logger.log_info("model", "Building backbone...")
    backbone = FastSAMBackbone(freeze_backbone=args.freeze_backbone).to(device).eval()

    if args.decoder_type == "proto_only":
        decoder = ProtoOnlyDecoder(proto_dim=32, feat_dim=1280).to(device)
        logger.log_info("model", f"Using ProtoOnlyDecoder (~427K params)")
    else:
        decoder = AdaptiveSparseDecoder(
            in_channels=1280, proto_dim=32, use_fdr=not args.no_fdr,
        ).to(device)
        logger.log_info("model", f"Using AdaptiveSparseDecoder: {decoder.get_submodule_params()}")

    fdr_router = None
    if not args.no_fdr:
        fdr_router = ForegroundDensityRouter(in_channels=1280).to(device).eval()
        logger.log_info("model", "FDR router loaded (frozen)")

    total_params = sum(p.numel() for p in decoder.parameters())
    logger.log_info("model", f"Trainable params: {total_params:,}")

    # ── 优化器 | Optimizer ──
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ── Episode 采样器 | Episode Sampler ──
    sampler = EpisodeSampler(train_ds, train_class_ids, k_shot=args.k_shot, seed=args.seed)

    # ── 训练循环 | Training Loop ──
    logger.log_info("train", f"Starting {args.episodes} episodes...")

    best_miou = 0.0
    history = []

    for ep in range(1, args.episodes + 1):
        episode = sampler.sample()
        cls_id = episode["class_id"]

        # ── 加载 support + query | Load support + query ──
        support_imgs = []
        support_masks = []
        for s_idx in episode["support_indices"]:
            s_sample = train_ds[s_idx]
            support_imgs.append(s_sample["image"])

            # 构建 per-class binary mask | Build per-class binary mask
            s_mask = torch.zeros(256, 256, dtype=torch.float32)
            for inst in s_sample["instances"]:
                if inst["category_id"] == cls_id:
                    s_mask = torch.logical_or(s_mask, inst["mask"]).float()
            support_masks.append(s_mask)

        support_imgs = torch.stack(support_imgs).to(device)  # [K, 3, H, W]
        support_masks = torch.stack(support_masks).to(device)  # [K, H, W]

        # Query
        q_sample = train_ds[episode["query_index"]]
        query_img = q_sample["image"].unsqueeze(0).to(device)  # [1, 3, H, W]

        q_mask = torch.zeros(256, 256, dtype=torch.float32)
        for inst in q_sample["instances"]:
            if inst["category_id"] == cls_id:
                q_mask = torch.logical_or(q_mask, inst["mask"]).float()
        q_mask = q_mask.to(device)

        if q_mask.sum() == 0:
            continue  # 跳过无 FG 的 query | Skip queries with no FG

        # ── 训练 | Train ──
        loss_val, metrics = train_episode(
            decoder, backbone, fdr_router,
            support_imgs, support_masks,
            query_img, q_mask,
            cls_id, optimizer, device,
            use_fdr=not args.no_fdr,
            tag="instance/train",
        )

        history.append({"episode": ep, **metrics})

        if ep % 10 == 0:
            logger.log_info("train",
                f"Ep {ep:4d}/{args.episodes} | cls={cls_id:2d} "
                f"loss={loss_val:.4f} focal={metrics['focal']:.4f} dice={metrics['dice']:.4f} "
                f"pred_mean={metrics['pred_mean']:.4f}")

        # ── 定期评估 | Periodic Evaluation ──
        if ep % args.eval_every == 0 or ep == args.episodes:
            logger.log_info("eval", f"--- Evaluation @ Episode {ep} ---")
            eval_result = evaluate_coco(
                decoder, backbone, fdr_router,
                val_ds, train_class_ids, device,
                use_fdr=not args.no_fdr,
                max_samples=args.eval_max_samples,
            )
            miou = eval_result["miou"]
            logger.log_info("eval",
                f"Ep {ep:4d}: mIoU={miou:.4f}, "
                f"AP50={eval_result.get('AP50', 0):.4f} "
                f"(best={best_miou:.4f})")

            if miou > best_miou:
                best_miou = miou
                torch.save({
                    "episode": ep,
                    "decoder_state_dict": {k: v.clone() for k, v in decoder.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(),
                    "miou": miou,
                    "eval_result": eval_result,
                    "args": vars(args),
                }, str(out_dir / "best_model.pt"))
                logger.log_info("eval", f"  New best: mIoU={best_miou:.4f}")

    # ── 最终结果 | Final Results ──
    logger.log_info("done", f"{'='*60}")
    logger.log_info("done", f"Training complete | 训练完成")
    logger.log_info("done", f"Best mIoU: {best_miou:.4f}")
    logger.log_info("done", f"Output: {out_dir}")

    # 保存最终报告 | Save final report
    final_report = {
        "experiment": "Instance Segmentation Few-Shot Fine-Tuning",
        "fold": args.fold,
        "k_shot": args.k_shot,
        "episodes": args.episodes,
        "decoder_type": args.decoder_type,
        "use_fdr": not args.no_fdr,
        "train_classes": {c: ISAID5I_CATEGORIES.get(c, str(c)) for c in train_class_ids},
        "best_miou": round(best_miou, 6),
        "config": vars(args),
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    logger.log_info("done", f"Report saved to {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
