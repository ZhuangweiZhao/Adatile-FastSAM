#!/usr/bin/env python3
"""
全监督训练 — 使用全部训练数据, 确定架构上限.
Full Supervision Training — use ALL training data to determine architecture ceiling.
====================================================================================

与 train_fewshot_allclass.py 的核心区别 | Key differences:
    Few-shot:  episodic (support/query 分离), K-shot 限制, per-class prototype from K images
    Full:      标准 DataLoader 遍历, 无 shot 限制, per-class prototype from ALL images

用途 | Purpose:
    跑出当前架构的全监督 mIoU, 作为 few-shot 的上限参照.
    If full-sup = 0.55: 0.48 few-shot = 87% ceiling → 空间有限, 优先修上限.
    If full-sup = 0.72: 0.48 few-shot = 67% ceiling → 空间巨大, 优先修 few-shot 机制.

用法 | Usage::

    # Pure decoder (最干净的上限, 无 prototype 依赖)
    python tools/train/train_supervised_full.py \
        --decoder pure --unfreeze-layers 12 --epochs 100

    # Adaptive decoder (与 few-shot 同架构, proto 来自全量数据)
    python tools/train/train_supervised_full.py \
        --decoder adaptive --unfreeze-layers 12 --epochs 100

    # Adaptive decoder + P3P4 + uf=12
    python tools/train/train_supervised_full.py \
        --decoder adaptive-p3p4 --unfreeze-layers 12 --epochs 100

输出 | Output:
    runs/train_supervised_full_*/
    ├── best_model.pt
    ├── last_model.pt
    └── train_log.json
"""

from __future__ import annotations

import sys, argparse, json, random, os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adatile.utils.seed import set_seed
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
from adatile.decoder.adaptive_decoder_p3p4 import AdaptiveDecoderP3P4
from adatile.decoder.pure_cnn_decoder import PureDecoder, PureDecoderP3P4
from adatile.datasets.isaid_instance_fewshot import ISAIDInstanceFewShotDataset

# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

CATEGORY_NAMES = {
    1: "small_vehicle", 2: "large_vehicle", 3: "plane",
    4: "storage_tank", 5: "ship", 6: "harbor",
    7: "ground_track_field", 8: "soccer_ball_field", 9: "tennis_court",
    10: "swimming_pool", 11: "baseball_diamond", 12: "basketball_court",
    13: "bridge", 14: "helicopter", 15: "roundabout",
}

IS_FG_CLASS = {7, 8, 9, 10, 11, 12, 13, 14, 15}  # 大目标类 | Large-object classes
IS_SMALL_CLASS = {1, 2, 3, 4, 5, 6}                 # 小目标类 | Small-object classes


# ═══════════════════════════════════════════════════════════════════
# Dataset Wrapper — 标准训练用 | Standard Training Dataset Wrapper
# ═══════════════════════════════════════════════════════════════════

class SupervisedTileDataset(torch.utils.data.Dataset):
    """
    将 ISAIDInstanceFewShotDataset 包装为标准 (image, mask, class_id) 格式.
    Wraps ISAIDInstanceFewShotDataset into standard (image, mask, class_id) format.

    每个 tile:
        - image: [3, H, W] float32 [0, 1]
        - mask:  [H, W] float32 binary FG mask
        - dominant_class: int, 面积最大的类别 ID

    Parameters
    ----------
    fewshot_dataset : ISAIDInstanceFewShotDataset
        底层数据集 | Underlying dataset.
    target_class_only : bool
        True → GT mask 仅包含主导类实例 (SCACS 协议: Pred(target) vs GT(target)).
        True → GT mask only contains dominant class instances (SCACS protocol).
        False → GT mask 包含全部可见类实例 (pure decoder 用).
        False → GT mask contains all visible class instances (for pure decoder).
    """

    def __init__(self, fewshot_dataset: ISAIDInstanceFewShotDataset,
                 target_class_only: bool = False):
        self._ds = fewshot_dataset
        self._target_class_only = target_class_only

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int):
        data = self._ds[idx]
        image = data["image"]                    # [3, H, W] float32
        instances = data["instances"]            # list of per-instance dicts

        H, W = image.shape[1], image.shape[2]

        # ── 计算主导类别 (最大面积) | Compute dominant class (largest area) ──
        dominant_class = 0
        max_area = 0.0
        for inst in instances:
            area = inst["area"]
            if area > max_area:
                max_area = area
                dominant_class = inst["category_id"]

        # ── 渲染 GT mask | Render GT mask ──
        # SCACS 协议: target_class_only=True → 仅主导类实例 → FG=1
        # SCACS protocol: target_class_only=True → only dominant class instances → FG=1
        # Pure decoder: target_class_only=False → 全部可见类实例 → FG=1
        # Pure decoder: target_class_only=False → all visible class instances → FG=1
        mask = np.zeros((H, W), dtype=np.float32)
        for inst in instances:
            # 类过滤: 当 target_class_only=True 时跳过非主导类
            # Class filter: skip non-dominant classes when target_class_only=True
            if self._target_class_only and inst["category_id"] != dominant_class:
                continue
            inst_mask = inst["mask"].numpy().astype(np.float32)  # [H, W] bool → float32
            mask = np.maximum(mask, inst_mask)

        # ── BG tile (无实例) → dominant_class=0, mask 全零 ──
        mask_tensor = torch.from_numpy(mask).float()  # [H, W]

        return {
            "image": image,
            "mask": mask_tensor,
            "dominant_class": dominant_class,
            "tile_id": data["tile_id"],
            "has_objects": len(instances) > 0,
        }


def collate_tiles(batch: list[dict]) -> dict:
    """
    Batch collate: 确保所有 tile 同尺寸 (当前全部 896×896).
    Batch collate: ensures all tiles same size (currently all 896×896).
    """
    images = torch.stack([b["image"] for b in batch], dim=0)           # [B, 3, H, W]
    masks = torch.stack([b["mask"] for b in batch], dim=0)             # [B, H, W]
    dominant_classes = [b["dominant_class"] for b in batch]
    tile_ids = [b["tile_id"] for b in batch]
    has_objects = [b["has_objects"] for b in batch]

    return {
        "image": images,
        "mask": masks,
        "dominant_class": dominant_classes,
        "tile_id": tile_ids,
        "has_objects": has_objects,
    }


# ═══════════════════════════════════════════════════════════════════
# 特征提取 | Feature Extraction (复用自 train_fewshot_allclass)
# ═══════════════════════════════════════════════════════════════════

def extract_features_batch(model, images: torch.Tensor,
                           device: str = "cuda", no_grad: bool = True) -> dict:
    """
    批量提取 FastSAM backbone 特征.
    Batch extract FastSAM backbone features.

    与 train_fewshot_allclass.extract_features() 逻辑相同, 但接受 batch tensor.
    Same logic as train_fewshot_allclass.extract_features(), but accepts batch tensor.

    :param model: FastSAM model (ultralytics).
    :param images: [B, 3, H, W] float32 [0, 1] tensor.
    :param device: "cuda" or "cpu".
    :param no_grad: If True, use torch.no_grad().
    :return: {"p3": [B,C,H/8,W/8], "p4": [B,C,H/16,W/16],
              "proto": [B,32,H/4,W/4], "p8": [B,C,H/32,W/32]}
    """
    seg = model.model          # SegmentationModel
    seq = seg.model            # Sequential[23]
    save_set = set(seg.save)   # {4, 6, 9, 12, 15, 18, 21}
    segment = seq[22]          # Segment head

    tensor = images.to(device)

    # ── 注册 hooks | Register hooks ──
    hooked = {}

    def _hook(name, _no_grad=no_grad):
        def _fn(m, inp, outp):
            hooked[name] = outp.detach() if _no_grad else outp
        return _fn

    handles = [
        seq[15].register_forward_hook(_hook("p3")),
        seq[18].register_forward_hook(_hook("p4")),
        seq[21].register_forward_hook(_hook("p8")),
    ]

    # ── Forward with Concat handling ──
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        x = tensor
        y = []
        for i, m in enumerate(seq):
            if hasattr(m, 'f') and m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if i in save_set else None)

    for h in handles:
        h.remove()

    p3 = hooked.get("p3")
    p4 = hooked.get("p4")
    p8 = hooked.get("p8")
    if p3 is None or p4 is None:
        raise RuntimeError(f"Hook failed: p3={p3 is not None}, p4={p4 is not None}")

    # ── Proto masks ──
    with ctx:
        proto = segment.proto(p3)  # [B, 32, H/4, W/4]

    return {
        "p3": p3,
        "p4": p4,
        "proto": proto,
        "p8": p8,
    }


# ═══════════════════════════════════════════════════════════════════
# 类别 Prototype 预计算 | Class Prototype Pre-computation
# ═══════════════════════════════════════════════════════════════════

def compute_class_prototypes(
    model,
    dataset: SupervisedTileDataset,
    device: str = "cuda",
    source: str = "p4",
    batch_size: int = 8,
) -> dict[int, torch.Tensor]:
    """
    从全部训练数据预计算每个类别的 prototype.
    Pre-compute prototype for each class from ALL training tiles.

    对每个类: 收集所有包含该类的 tile 的 spatial-averaged P4/P8, 取均值 + L2-norm.
    For each class: collect spatial-averaged P4/P8 from all tiles containing it,
    average + L2-normalize.

    :param model: FastSAM model (frozen inference).
    :param dataset: SupervisedTileDataset.
    :param device: "cuda" or "cpu".
    :param source: "p4" (stride-16) or "p8" (stride-32).
    :param batch_size: batch size for feature extraction.
    :return: {class_id: prototype_tensor [feat_dim]}
    """
    model.model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_tiles, num_workers=0)

    # ── 收集每个类的特征 | Collect features per class ──
    class_features: dict[int, list[torch.Tensor]] = defaultdict(list)

    for batch in tqdm(loader, desc="Computing prototypes"):
        images = batch["image"]  # [B, 3, H, W]
        dominant_classes = batch["dominant_class"]

        feats = extract_features_batch(model, images, device, no_grad=True)
        src_feats = feats[source]  # [B, C, h, w]

        for i, cls_id in enumerate(dominant_classes):
            if cls_id == 0:
                continue  # BG tile, 跳过
            # spatial average → [C]
            vec = src_feats[i].mean(dim=(1, 2)).cpu()
            class_features[cls_id].append(vec)

    # ── 每个类: mean → L2-norm | Per class: mean → L2-norm ──
    prototypes: dict[int, torch.Tensor] = {}
    for cls_id, vecs in sorted(class_features.items()):
        stacked = torch.stack(vecs, dim=0)            # [N, C]
        proto = stacked.mean(dim=0)                   # [C]
        proto = F.normalize(proto, p=2, dim=0)        # L2-norm
        prototypes[cls_id] = proto

    print(f"  [Prototypes] Computed for {len(prototypes)} classes "
          f"(source={source}, dim={prototypes[list(prototypes.keys())[0]].shape[0]})")
    for cls_id in sorted(prototypes.keys()):
        n_tiles = len(class_features[cls_id])
        print(f"    Class {cls_id:>2d} ({CATEGORY_NAMES.get(cls_id, '?'):<18s}): "
              f"{n_tiles} tiles")

    return prototypes


# ═══════════════════════════════════════════════════════════════════
# 验证 | Validation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    model,
    decoder,
    val_dataset: SupervisedTileDataset,
    prototypes: dict[int, torch.Tensor] | None,
    device: str,
    decoder_type: str,
    proto_source: str = "p4",
    batch_size: int = 8,
) -> dict:
    """
    在验证集上计算 per-class IoU + mIoU.
    Compute per-class IoU + mIoU on validation set.
    """
    model.model.eval()
    decoder.eval()

    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_tiles, num_workers=0)

    # ── 累积 per-class intersection/union | Accumulate per-class intersection/union ──
    class_inter = defaultdict(float)
    class_union = defaultdict(float)

    for batch in tqdm(loader, desc="Validating", leave=False):
        images = batch["image"].to(device)          # [B, 3, H, W]
        gt_masks = batch["mask"].to(device)          # [B, H, W]
        dominant_classes = batch["dominant_class"]

        B, _, H, W = images.shape

        # ── 提取特征 (frozen) | Extract features (frozen) ──
        feats = extract_features_batch(model, images, device, no_grad=True)

        # ── Forward decoder ──
        if decoder_type == "pure":
            mask_pred = decoder(feats["p4"])  # [B, 1, H/4, W/4] or [B, H/4, W/4]
        elif decoder_type == "pure-p3p4":
            mask_pred = decoder(feats["p3"], feats["p4"])
        elif decoder_type in ("adaptive", "adaptive-p3p4"):
            # 每个 tile 使用其主导类 prototype | Each tile uses its dominant class prototype
            p4 = feats["p4"]          # [B, C, h, w]
            proto_masks = feats["proto"]  # [B, 32, H/4, W/4]

            # 构建 batch support_proto
            proto_vecs = []
            for cls_id in dominant_classes:
                if cls_id > 0 and cls_id in prototypes:
                    proto_vecs.append(prototypes[cls_id].to(device))
                else:
                    # BG tile: 使用零向量 | BG tile: use zero vector
                    proto_vecs.append(torch.zeros(p4.shape[1], device=device))
            support_proto = torch.stack(proto_vecs, dim=0)  # [B, C]

            if decoder_type == "adaptive":
                mask_pred = decoder(p4, proto_masks, support_proto)
            else:
                mask_pred = decoder(feats["p3"], p4, proto_masks, support_proto)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # ── Normalize to [B, 1, Hp, Wp] ──
        if mask_pred.dim() == 3:
            mask_pred = mask_pred.unsqueeze(1)  # [B, H, W] → [B, 1, H, W]

        # ── Upsample to GT resolution ──
        mask_pred = F.interpolate(
            mask_pred, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1)  # [B, H, W]

        # ── Sigmoid + threshold ──
        mask_bin = (mask_pred > 0.5).float()  # [B, H, W] (decoder 已返回 sigmoid 输出)

        # ── Per-class IoU (按主导类统计) | Per-class IoU (by dominant class) ──
        for b in range(B):
            cls_id = dominant_classes[b]
            if cls_id == 0:
                continue

            pred_b = mask_bin[b]
            gt_b = gt_masks[b]

            inter = (pred_b * gt_b).sum().item()
            union = (pred_b + gt_b).clamp(0, 1).sum().item()

            class_inter[cls_id] += inter
            class_union[cls_id] += union

    # ── 计算 mIoU | Compute mIoU ──
    per_class_iou = {}
    for cls_id in sorted(class_union.keys()):
        u = class_union[cls_id]
        if u > 0:
            per_class_iou[cls_id] = class_inter[cls_id] / u
        else:
            per_class_iou[cls_id] = 0.0

    valid_ious = [v for v in per_class_iou.values() if v > 0]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    return {
        "miou": miou,
        "per_class_iou": per_class_iou,
        "n_classes_evaluated": len(valid_ious),
    }


# ═══════════════════════════════════════════════════════════════════
# Main | Training Loop
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full Supervision Training — architecture ceiling measurement"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (默认 1; decoder 设计为 single-image, >1 不保证正确)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default="data/iSAID_instance_fewshot")
    parser.add_argument("--decoder", type=str, default="adaptive",
                        choices=["adaptive", "adaptive-p3p4", "pure", "pure-p3p4"],
                        help="Decoder type: adaptive (P4+proto), adaptive-p3p4 (P3P4+proto), "
                             "pure (P4 only, no proto), pure-p3p4 (P3P4, no proto)")
    parser.add_argument("--unfreeze-layers", type=int, default=12,
                        help="解冻 backbone 最后 N 层 | Unfreeze last N backbone layers")
    parser.add_argument("--lr-backbone", type=float, default=None,
                        help="Backbone LR (默认 lr/10) | Backbone learning rate")
    parser.add_argument("--prototype-source", type=str, default="p4",
                        choices=["p4", "p8"],
                        help="Prototype 特征来源 | Prototype feature source")
    parser.add_argument("--proto-update-freq", type=int, default=1,
                        help="每 N epoch 更新 prototype | Update prototypes every N epochs")
    parser.add_argument("--val-every", type=int, default=5,
                        help="每 N epoch 验证一次 | Validate every N epochs")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process)")
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold ID (0/1/2) for class split")
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    device = args.device

    data_root = Path(args.data_root)

    # ── 输出目录 | Output Dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        tag = f"{args.decoder}_uf{args.unfreeze_layers}"
        if args.prototype_source != "p4":
            tag += f"_proto{args.prototype_source}"
        args.output_dir = f"runs/train_supervised_full_{tag}_{ts}"
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    needs_prototype = args.decoder in ("adaptive", "adaptive-p3p4")

    print(f"{'=' * 60}")
    print(f"  Full Supervision Training (Ceiling Measurement)")
    print(f"  Data: {data_root}")
    print(f"  Decoder: {args.decoder} | Proto: {'YES' if needs_prototype else 'NO'}")
    print(f"  Unfreeze: {args.unfreeze_layers} layers | Epochs: {args.epochs}")
    print(f"  Batch: {args.batch_size} | LR: {args.lr} | Device: {device}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")

    # ── 1. Load datasets | 加载数据集 ──
    print(f"\n[1/5] Loading datasets (mode='all', fold={args.fold})...")

    train_ds = ISAIDInstanceFewShotDataset(
        root=str(data_root),
        split="train",
        fold=args.fold,
        mode="all",
    )
    val_ds = ISAIDInstanceFewShotDataset(
        root=str(data_root),
        split="val",
        fold=args.fold,
        mode="all",
    )

    # SCACS 协议: adaptive decoder → GT 仅包含 target class 实例
    # SCACS protocol: adaptive decoder → GT only contains target class instances
    # Pure decoder (无 class conditioning) → GT 包含全部实例
    # Pure decoder (no class conditioning) → GT contains all instances
    _target_cls_only = needs_prototype  # True for adaptive/adaptive-p3p4, False for pure/pure-p3p4
    train_dataset = SupervisedTileDataset(train_ds, target_class_only=_target_cls_only)
    val_dataset = SupervisedTileDataset(val_ds, target_class_only=_target_cls_only)

    print(f"  Train: {len(train_dataset)} tiles across {len(train_ds.get_visible_classes())} classes"
          f" (target_class_only={_target_cls_only})")
    print(f"  Val:   {len(val_dataset)} tiles across {len(val_ds.get_visible_classes())} classes"
          f" (target_class_only={_target_cls_only})")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_tiles, num_workers=args.num_workers,
        drop_last=True,
    )

    # ── 2. Load FastSAM | 加载 FastSAM ──
    print(f"\n[2/5] Loading FastSAM...")
    from ultralytics import FastSAM
    model_path = _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"
    model = FastSAM(str(model_path))
    model.model.cuda().eval()
    for p in model.model.parameters():
        p.requires_grad = False
    print(f"  Model loaded + frozen on {device}")

    # ── 2b. 部分解冻 Backbone | Partial Backbone Unfreezing ──
    backbone_trainable = False
    if args.unfreeze_layers > 0:
        backbone_trainable = True
        seq = model.model.model  # Sequential[23]
        n_total = len(seq)
        start_layer = max(0, n_total - args.unfreeze_layers)
        unfrozen_params = 0
        for i in range(start_layer, n_total):
            layer = seq[i]
            for p in layer.parameters():
                p.requires_grad = True
                unfrozen_params += p.numel()
        n_frozen_params = sum(p.numel() for p in model.model.parameters() if not p.requires_grad)
        print(f"  Backbone unfrozen: layers {start_layer}-{n_total-1}")
        print(f"  Trainable: {unfrozen_params/1e6:.2f}M / Frozen: {n_frozen_params/1e6:.2f}M")

    # ── 3. Build decoder | 构建 Decoder ──
    print(f"\n[3/5] Building Decoder ({args.decoder})...")

    # 自动检测通道数 | Auto-detect channel count
    _test_img = torch.zeros(1, 3, 896, 896, device=device)
    _test_feats = extract_features_batch(model, _test_img, device, no_grad=True)
    _p4_ch = _test_feats["p4"].shape[1]
    print(f"  P4 channels: {_p4_ch}")

    if args.decoder == "adaptive":
        decoder = AdaptiveSparseDecoder(in_channels=_p4_ch, proto_dim=32, hidden_dim=256, use_fdr=False)
        decoder_type = "adaptive"
    elif args.decoder == "adaptive-p3p4":
        _p3_ch = _test_feats["p3"].shape[1]
        print(f"  P3 channels: {_p3_ch}")
        decoder = AdaptiveDecoderP3P4(p3_channels=_p3_ch, p4_channels=_p4_ch,
                                       proto_dim=32, hidden_dim=256)
        decoder_type = "adaptive-p3p4"
    elif args.decoder == "pure":
        decoder = PureDecoder(in_channels=_p4_ch)
        decoder_type = "pure"
    elif args.decoder == "pure-p3p4":
        _p3_ch = _test_feats["p3"].shape[1]
        print(f"  P3 channels: {_p3_ch}")
        decoder = PureDecoderP3P4(p3_channels=_p3_ch, p4_channels=_p4_ch)
        decoder_type = "pure-p3p4"
    else:
        raise ValueError(f"Unknown decoder: {args.decoder}")

    decoder = decoder.to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder params: {n_params:,} ({n_params/1e6:.3f}M)")

    del _test_img, _test_feats
    torch.cuda.empty_cache()

    # ── 4. Pre-compute class prototypes (仅 adaptive decoder) ──
    prototypes: dict[int, torch.Tensor] | None = None
    if needs_prototype:
        print(f"\n[4/5] Pre-computing class prototypes (source={args.prototype_source})...")
        prototypes = compute_class_prototypes(
            model, train_dataset, device,
            source=args.prototype_source,
            batch_size=args.batch_size,
        )

    # ── 5. Optimizer + Scheduler ──
    print(f"\n[5/5] Setting up optimizer...")
    lr_backbone = args.lr_backbone if args.lr_backbone is not None else args.lr * 0.1

    param_groups = [{'params': decoder.parameters(), 'lr': args.lr}]
    if backbone_trainable:
        backbone_params = [p for p in model.model.parameters() if p.requires_grad]
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})
        print(f"  Decoder LR: {args.lr} | Backbone LR: {lr_backbone}")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ═══════════════════════════════════════════════════════════════
    # Training Loop | 训练循环
    # ═══════════════════════════════════════════════════════════════

    print(f"\n{'=' * 60}")
    print(f"  Training ({args.epochs} epochs, {len(train_loader)} batches/epoch)")
    print(f"{'=' * 60}\n")

    best_val_iou = 0.0
    train_log = []
    log_path = out_dir / "train_log.json"

    for epoch in range(args.epochs):
        # ── 定期更新 prototype | Periodic prototype update ──
        if needs_prototype and epoch > 0 and epoch % args.proto_update_freq == 0:
            prototypes = compute_class_prototypes(
                model, train_dataset, device,
                source=args.prototype_source,
                batch_size=args.batch_size,
            )

        # ── Training ──
        model.model.eval()  # 必须 eval mode (YOLOv8)
        decoder.train()

        epoch_losses = {"loss": [], "dice": [], "bce": []}
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch")

        for batch in pbar:
            images = batch["image"].to(device)              # [B, 3, H, W]
            gt_masks = batch["mask"].to(device)              # [B, H, W]
            dominant_classes = batch["dominant_class"]
            has_objects = batch["has_objects"]

            B, _, H_img, W_img = images.shape

            # ── 提取特征 | Extract features ──
            feats = extract_features_batch(model, images, device,
                                           no_grad=not backbone_trainable)

            # ── Forward decoder ──
            if decoder_type == "pure":
                mask_pred = decoder(feats["p4"])  # [B, 1, h, w] or [B, h, w]
            elif decoder_type == "pure-p3p4":
                mask_pred = decoder(feats["p3"], feats["p4"])
            elif decoder_type == "adaptive":
                p4 = feats["p4"]          # [B, C, h, w]
                proto_masks = feats["proto"]  # [B, 32, H/4, W/4]

                # 构建 batch support_proto | Build batch support_proto
                proto_vecs = []
                for cls_id in dominant_classes:
                    if cls_id > 0 and cls_id in prototypes:
                        proto_vecs.append(prototypes[cls_id].to(device))
                    else:
                        proto_vecs.append(torch.zeros(p4.shape[1], device=device))
                support_proto = torch.stack(proto_vecs, dim=0)  # [B, C]

                mask_pred = decoder(p4, proto_masks, support_proto)
            elif decoder_type == "adaptive-p3p4":
                p3 = feats["p3"]
                p4 = feats["p4"]
                proto_masks = feats["proto"]

                proto_vecs = []
                for cls_id in dominant_classes:
                    if cls_id > 0 and cls_id in prototypes:
                        proto_vecs.append(prototypes[cls_id].to(device))
                    else:
                        proto_vecs.append(torch.zeros(p4.shape[1], device=device))
                support_proto = torch.stack(proto_vecs, dim=0)

                mask_pred = decoder(p3, p4, proto_masks, support_proto)

            # ── Normalize → [B, 1, Hp, Wp] ──
            if mask_pred.dim() == 3:
                mask_pred = mask_pred.unsqueeze(1)

            # ── Upsample to GT resolution ──
            mask_pred = F.interpolate(
                mask_pred, size=(H_img, W_img), mode="bilinear", align_corners=False
            ).squeeze(1)  # [B, H, W]

            # ── Loss: Dice + BCE ──
            # 注意: decoder 返回 sigmoid 输出 (∈ [0,1]), 不是 logits
            # Note: decoder returns sigmoid output (∈ [0,1]), NOT logits
            mask_prob = mask_pred.clamp(1e-7, 1 - 1e-7)  # 数值稳定 | Numerical stability

            # Dice loss
            inter = (mask_prob * gt_masks).sum(dim=(1, 2))  # [B]
            union = mask_prob.sum(dim=(1, 2)) + gt_masks.sum(dim=(1, 2))  # [B]
            dice = (2.0 * inter + 1e-6) / (union + 1e-6)  # [B]
            d_loss = (1.0 - dice).mean()

            # BCE loss (输入已是概率) | BCE loss (input already probability)
            bce = F.binary_cross_entropy(mask_prob, gt_masks, reduction='mean')

            loss = d_loss + bce

            # ── Backward ──
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            all_params = list(decoder.parameters())
            if backbone_trainable:
                all_params += [p for p in model.model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)

            optimizer.step()

            # ── Log ──
            epoch_losses["loss"].append(loss.item())
            epoch_losses["dice"].append(d_loss.item())
            epoch_losses["bce"].append(bce.item())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{d_loss.item():.4f}",
                bce=f"{bce.item():.4f}",
            )

        scheduler.step()

        # ── Epoch summary ──
        avg_loss = np.mean(epoch_losses["loss"]) if epoch_losses["loss"] else 0
        avg_dice = np.mean(epoch_losses["dice"]) if epoch_losses["dice"] else 0
        avg_bce = np.mean(epoch_losses["bce"]) if epoch_losses["bce"] else 0
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, dice={avg_dice:.4f}, "
              f"bce={avg_bce:.4f}, lr={lr_now:.2e}")

        # ── Validation (每 val_every epoch) ──
        val_miou = 0.0
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_result = validate(
                model, decoder, val_dataset, prototypes,
                device, decoder_type, args.prototype_source,
                batch_size=args.batch_size,
            )
            val_miou = val_result["miou"]
            per_cls = val_result["per_class_iou"]
            print(f"  VAL epoch {epoch + 1}: mIoU={val_miou:.4f} "
                  f"(n_classes={val_result['n_classes_evaluated']})")
            # Print per-class
            cls_strs = []
            for cls_id in sorted(per_cls.keys()):
                n = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")
                cls_strs.append(f"{n}={per_cls[cls_id]:.3f}")
            print(f"    Per-class: {', '.join(cls_strs)}")

        # ── Save best ──
        if val_miou > best_val_iou:
            best_val_iou = val_miou
            ckpt = {
                "epoch": epoch + 1,
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_iou": best_val_iou,
                "decoder_type": decoder_type,
                "unfreeze_layers": args.unfreeze_layers,
            }
            if backbone_trainable:
                seq = model.model.model
                ckpt["backbone"] = {
                    str(i): seq[i].state_dict()
                    for i in range(max(0, len(seq) - args.unfreeze_layers), len(seq))
                }
            torch.save(ckpt, out_dir / "best_model.pt")
            print(f"  >> Best model saved (mIoU={best_val_iou:.4f})")

        # ── Save last ──
        ckpt_last = {
            "epoch": epoch + 1,
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_miou": val_miou,
            "decoder_type": decoder_type,
        }
        if backbone_trainable:
            seq = model.model.model
            ckpt_last["backbone"] = {
                str(i): seq[i].state_dict()
                for i in range(max(0, len(seq) - args.unfreeze_layers), len(seq))
            }
        torch.save(ckpt_last, out_dir / "last_model.pt")

        # ── Log entry ──
        log_entry = {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "dice": avg_dice,
            "bce": avg_bce,
            "val_miou": val_miou,
            "lr": lr_now,
        }
        if val_miou > 0 and val_result:
            log_entry["val_per_class_iou"] = {
                str(k): round(v, 4) for k, v in val_result["per_class_iou"].items()
            }
        train_log.append(log_entry)
        with open(log_path, "w") as f:
            json.dump(train_log, f, indent=2)

    # ── Final report ──
    print(f"\n{'=' * 60}")
    print(f"  Training Complete")
    print(f"  Best val mIoU: {best_val_iou:.4f}")
    print(f"  Output: {out_dir}")
    print(f"  Config: decoder={decoder_type}, uf={args.unfreeze_layers}, "
          f"proto_src={args.prototype_source}")
    print(f"{'=' * 60}")

    # ── 保存最终配置 | Save final config ──
    config = {
        "decoder": decoder_type,
        "unfreeze_layers": args.unfreeze_layers,
        "prototype_source": args.prototype_source,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lr_backbone": lr_backbone,
        "best_val_iou": best_val_iou,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
