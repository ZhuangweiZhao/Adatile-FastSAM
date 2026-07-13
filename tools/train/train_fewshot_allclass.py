#!/usr/bin/env python3
"""
全 15 类 K-shot Fine-tuning — ultralytics FastSAM + 多格式数据支持.
All-15-Class K-shot Fine-tuning — ultralytics FastSAM + multi-format data support.
====================================================================================

支持三种数据格式 / Supports three data formats:
    isaid5i:         iSAID-5i 256² crops (默认 / default)
    isaid_tiles:     prep_isaid_tiles.py output (旧 tile 格式 / old tile format)
    isaid_instance:  prep_isaid_instance.py output (新 COCO tile 格式 / new COCO tile format)

用法 | Usage::

    # iSAID-5i 格式 (默认)
    python tools/train/train_fewshot_allclass.py --k-shot 5 --epochs 50

    # 旧 tile 格式
    python tools/train/train_fewshot_allclass.py --k-shot 3 \
        --data-format isaid_tiles --data-root data/iSAID_tiles

    # 新 COCO tile 格式 (iSAID-few_tiles)
    python tools/train/train_fewshot_allclass.py --k-shot 3 \
        --data-format isaid_instance --data-root data/iSAID-few_tiles

    # K=1 shot, 快速验证
    python tools/train/train_fewshot_allclass.py --k-shot 1 --epochs 30 --lr 5e-4

输出 | Output:
    runs/train_fewshot_allclass_K5_*/
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
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from adatile.utils.seed import set_seed
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
from adatile.decoder.adaptive_decoder_p3p4 import AdaptiveDecoderP3P4
from adatile.decoder.pure_cnn_decoder import PureDecoder, PureDecoderP3P4
from adatile.decoder.dynamic_kernel_decoder import DynamicKernelDecoder
from adatile.decoder.center_affinity_decoder import CenterAffinityDecoder
from adatile.metrics.hungarian_matcher import (
    hungarian_match_instances, multi_instance_loss, mask_scores,
)
from adatile.sparse.spm import SparsePerceptionModule

# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# 数据加载 — 多格式支持 | Data Loading — Multi-format Support
# ═══════════════════════════════════════════════════════════════════

CATEGORY_NAMES = {
    1: "small_vehicle", 2: "large_vehicle", 3: "plane",
    4: "storage_tank", 5: "ship", 6: "harbor",
    7: "ground_track_field", 8: "soccer_ball_field", 9: "tennis_court",
    10: "swimming_pool", 11: "baseball_diamond", 12: "basketball_court",
    13: "bridge", 14: "helicopter", 15: "roundabout",
}

# ── 全局缓存: COCO 标注索引 (isaid_instance 格式) ──
# Global cache: COCO annotation index for isaid_instance format
_COCO_CACHE: dict = {}  # key: (data_root_str, split) → {"file_to_anns": {...}, "file_to_id": {...}}


# ═══════════════════════════════════════════════════════════════════
# 辅助: 从 tile stem 提取源图像 ID | Helper: extract source image ID
# ═══════════════════════════════════════════════════════════════════

def _extract_source_image(stem: str) -> str:
    """从 tile stem 提取源图像 ID | Extract source image ID from tile stem.

    P0019_t0001 → P0019   (isaid_instance / isaid_tiles)
    P0003_206_462_206_462 → P0003  (isaid-5i)
    """
    parts = stem.split("_")
    if len(parts) >= 2 and parts[1].startswith("t") and parts[1][1:].isdigit():
        return parts[0]  # P0019_t0001 format
    return parts[0]  # P0003_206_462_206_462 format


# ═══════════════════════════════════════════════════════════════════
# 类别索引 — 层级格式: cls_id → {source_img: [tile_stems]}
# Class Index — Hierarchical: cls_id → {source_img: [tile_stems]}
# ═══════════════════════════════════════════════════════════════════


def _resolve_paths(args) -> tuple:
    """
    解析数据根目录和格式 | Resolve data root and format.

    :return: (data_root: Path, data_format: str, data_split: str)
    """
    fmt = args.data_format

    # 默认 data-root | Default data-root
    if args.data_root is None:
        if fmt == "isaid_tiles":
            args.data_root = "data/iSAID_tiles"
        elif fmt == "isaid_instance":
            args.data_root = "data/iSAID_instance_fewshot"
        else:
            args.data_root = "data/iSAID-5i/iSAID"

    data_root = Path(args.data_root)

    # 训练/验证 split | Train/val split
    # isaid_tiles (旧格式): 仅在 val 上切了 tile，train 不可用
    # isaid_tiles (old format): tiles only on val, train not available
    # isaid_instance (新格式): 同时在 train/val 上切了 tile，各用各的
    # isaid_instance (new format): tiles on both train and val
    if fmt == "isaid_tiles":
        train_split = "val"   # 旧格式只有 val | Old format only has val
        val_split = "val"
    elif fmt == "isaid_instance":
        train_split = "train"  # 新格式有 train/val 分离 | New format has train/val separation
        val_split = "val"
    else:
        train_split = "train"  # isaid5i
        val_split = "train"

    return data_root, fmt, train_split, val_split


def _build_class_index_isaid5i(data_root: Path, split: str = "train") -> dict[int, dict[str, list[str]]]:
    """iSAID-5i 格式: 层级索引 cls_id → {source_img: [tile_stems]}"""
    root = data_root / split
    index: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for fold in range(3):
        list_name = "train_list" if split == "train" else "val_list"
        list_file = root / list_name / f"split{fold}_{split}.txt"
        if not list_file.exists():
            continue
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stem = line.rsplit("_instance_color_RGB.png_", 1)[0]
                cls_str = line.rsplit("_", 1)[-1]
                try:
                    cls_id = int(cls_str)
                except ValueError:
                    continue
                if 1 <= cls_id <= 15:
                    src = _extract_source_image(stem)
                    index[cls_id][src].append(stem)
    return {k: {src: sorted(set(tiles)) for src, tiles in v.items()} for k, v in index.items()}


def _build_class_index_tiles(data_root: Path, split: str) -> dict[int, dict[str, list[str]]]:
    """旧 tile 格式: 层级索引 cls_id → {source_img: [tile_stems]}"""
    meta_path = data_root / "metadata" / f"{split}.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Tile metadata not found: {meta_path}")
    with open(meta_path) as f:
        metadata = json.load(f)
    index: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for t in metadata:
        cdist = t.get("class_distribution", {})
        if not cdist:
            continue
        dom_cls = max(cdist, key=lambda k: cdist[k])
        cls_id = int(dom_cls)
        if 1 <= cls_id <= 15:
            stem = t["tile_name"].rsplit(".", 1)[0]
            src = _extract_source_image(stem)
            index[cls_id][src].append(stem)
    return {k: {src: sorted(set(tiles)) for src, tiles in v.items()} for k, v in index.items()}


def _build_class_index_instance(data_root: Path, split: str) -> dict[int, dict[str, list[str]]]:
    """新 COCO tile 格式: 层级索引 cls_id → {source_img: [tile_stems]}"""
    ann_file = data_root / "annotations" / f"instances_{split}.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation not found: {ann_file}")
    with open(ann_file) as f:
        coco = json.load(f)

    # file_name → stem mapping
    tile_stems = {img["id"]: Path(img["file_name"]).stem for img in coco["images"]}

    # Per-tile class distribution
    tile_classes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for ann in coco.get("annotations", []):
        cat_id = ann.get("category_id", 0)
        img_id = ann.get("image_id", 0)
        if 1 <= cat_id <= 15 and img_id in tile_stems:
            tile_classes[img_id][cat_id] += 1

    # 多类索引: 每个 tile 归属于它包含的所有类 (非仅主导类)
    # Multi-class index: each tile belongs to ALL classes it contains (not just dominant)
    # 之前的主导类索引导致稀有类 (helicopter/plane) 的 tile 被常见类 (ship/vehicle) "偷走",
    # val split 中 helicopter 从 65 source → 8 source, plane 从 14 → 5.
    # Previous dominant-class index caused rare-class tiles to be "stolen" by common classes.
    index: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for img_id, cls_counts in tile_classes.items():
        if not cls_counts:
            continue
        stem = tile_stems[img_id]
        src = _extract_source_image(stem)
        for cls_id in cls_counts.keys():  # 该 tile 中出现的所有类
            index[cls_id][src].append(stem)

    return {k: {src: sorted(set(tiles)) for src, tiles in v.items()} for k, v in index.items()}


def _load_coco_index(data_root: Path, split: str) -> dict:
    """缓存 COCO JSON 索引用于 mask 渲染 | Cache COCO JSON index for mask rendering."""
    cache_key = (str(data_root), split)
    if cache_key in _COCO_CACHE:
        return _COCO_CACHE[cache_key]

    ann_file = data_root / "annotations" / f"instances_{split}.json"
    if not ann_file.exists():
        _COCO_CACHE[cache_key] = {}
        return {}

    with open(ann_file) as f:
        coco = json.load(f)

    # file_name → annotations mapping
    file_to_anns = defaultdict(list)
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    for ann in coco.get("annotations", []):
        fname = img_id_to_file.get(ann["image_id"])
        if fname:
            file_to_anns[fname].append(ann)

    _COCO_CACHE[cache_key] = {"file_to_anns": dict(file_to_anns)}
    return _COCO_CACHE[cache_key]


def _render_instance_mask(anns: list, h: int, w: int,
                          target_class_id: int | None = None) -> np.ndarray:
    """从 COCO polygon 标注渲染二值 FG mask | Render binary FG mask from COCO polygons.

    :param anns: COCO 标注列表 | COCO annotation list.
    :param h: 画布高度 | Canvas height.
    :param w: 画布宽度 | Canvas width.
    :param target_class_id: 仅渲染该类别实例 (None=全部) | Only render instances of this class (None=all).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for ann in anns:
        cat_id = ann.get("category_id", 0)
        if cat_id < 1 or cat_id > 15:
            continue
        if target_class_id is not None and cat_id != target_class_id:
            continue  # 跳过非目标类 | Skip non-target classes
        seg = ann.get("segmentation", [])
        if not seg:
            bx, by, bw, bh = [int(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            mask[max(0, by):min(h, by + bh), max(0, bx):min(w, bx + bw)] = 1
            continue
        if isinstance(seg, list):
            if isinstance(seg[0], list):
                polys = seg
            elif isinstance(seg[0], (int, float)):
                polys = [seg]
            else:
                continue
            for poly in polys:
                if len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
                pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
                cv2.fillPoly(mask, [pts], 1)
    return mask


def load_tile_and_mask(stem: str, split: str = "val",
                        data_root: Path = None,
                        target_class_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """旧 tile 格式: 加载 tile 图像 + 语义 mask | Old tile format: load tile image + semantic mask.

    :param target_class_id: 仅保留该类别 (None=全部) | Only keep this class (None=all).
    """
    if data_root is None:
        data_root = Path("data/iSAID_tiles")
    img = np.array(Image.open(str(data_root / "images" / split / f"{stem}.png")).convert("RGB"))
    mask_path = data_root / "masks" / split / f"{stem}.png"
    if mask_path.exists():
        mask = np.array(Image.open(str(mask_path)))
        if target_class_id is not None:
            mask = (mask == target_class_id).astype(np.uint8)
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
    return img, mask


def load_instance_tile_and_mask(stem: str, split: str,
                                  data_root: Path,
                                  target_class_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """新 COCO tile 格式: 加载 tile 图像 + 从 COCO 渲染 mask | New COCO tile format: load image + render mask.

    :param target_class_id: 仅渲染该类别实例 (None=全部) | Only render instances of this class (None=all).
    """
    img = np.array(Image.open(str(data_root / "images" / split / f"{stem}.png")).convert("RGB"))
    H, W = img.shape[:2]

    # 从缓存加载标注 | Load annotations from cache
    coco_idx = _load_coco_index(data_root, split)
    file_to_anns = coco_idx.get("file_to_anns", {})
    anns = file_to_anns.get(f"{stem}.png", [])

    if anns:
        mask = _render_instance_mask(anns, H, W, target_class_id=target_class_id)
    else:
        mask = np.zeros((H, W), dtype=np.uint8)

    return img, mask


def load_image_and_mask(stem: str, split: str = "train",
                          data_root: Path = None) -> tuple[np.ndarray, np.ndarray]:
    """iSAID-5i 格式: 加载 256² 图像 + 语义 mask."""
    if data_root is None:
        data_root = Path("data/iSAID-5i/iSAID")
    root = data_root / split
    img = np.array(Image.open(str(root / "images" / f"{stem}.png")).convert("RGB"))
    mask = np.array(Image.open(str(root / "semantic_mask" / f"{stem}_instance_color_RGB.png")).convert("RGB"))
    return img, mask


def semantic_mask_to_binary(mask: np.ndarray, is_tile: bool = False) -> np.ndarray:
    """语义 mask → binary FG mask."""
    if is_tile:
        return (mask > 0).astype(np.float32)
    return (mask.max(axis=-1) > 0).astype(np.float32)


def load_instance_masks_from_coco(stem: str, split: str, data_root: Path,
                                  target_class_id: int,
                                  target_size: tuple[int, int] | None = None
                                  ) -> list[np.ndarray]:
    """从 COCO 标注加载逐实例二值 mask | Load per-instance binary masks from COCO annotations.

    用于 DynamicKernelDecoder 训练: 每个 GT 实例一个独立 mask。
    Used for DynamicKernelDecoder training: one mask per GT instance.

    :param stem: Tile stem (不含扩展名) | Tile stem (without extension).
    :param split: "train" or "val".
    :param data_root: 数据根目录 | Data root directory.
    :param target_class_id: 仅加载此类实例 | Only load instances of this class.
    :param target_size: (H, W) 目标分辨率 (None=保持原图) | Target resolution (None=keep original).
    :return: [mask_0, mask_1, ...] 每元素 [H, W] uint8 binary.
    """
    coco_idx = _load_coco_index(data_root, split)
    file_to_anns = coco_idx.get("file_to_anns", {})
    anns = file_to_anns.get(f"{stem}.png", [])

    # 加载图像获取尺寸 | Load image to get dimensions
    img_path = data_root / "images" / split / f"{stem}.png"
    if img_path.exists():
        img = np.array(Image.open(str(img_path)))
        H, W = img.shape[:2]
    else:
        H, W = 896, 896  # 默认 | default

    instances = []
    for ann in anns:
        cat_id = ann.get("category_id", 0)
        if cat_id < 1 or cat_id > 15:
            continue
        if cat_id != target_class_id:
            continue

        # 渲染单个实例 mask | Render single instance mask
        mask = np.zeros((H, W), dtype=np.uint8)
        seg = ann.get("segmentation", [])
        if not seg:
            bx, by, bw, bh = [int(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            mask[max(0, by):min(H, by + bh), max(0, bx):min(W, bx + bw)] = 1
        elif isinstance(seg, list):
            if isinstance(seg[0], list):
                polys = seg
            elif isinstance(seg[0], (int, float)):
                polys = [seg]
            else:
                continue
            for poly in polys:
                if len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                pts[:, :, 0] = np.clip(pts[:, :, 0], 0, W - 1)
                pts[:, :, 1] = np.clip(pts[:, :, 1], 0, H - 1)
                cv2.fillPoly(mask, [pts], 1)

        if mask.sum() > 0:
            if target_size is not None:
                mask = cv2.resize(mask, (target_size[1], target_size[0]),
                                  interpolation=cv2.INTER_NEAREST)
            instances.append(mask)

    return instances


# ═══════════════════════════════════════════════════════════════════
# 全图 Query 支持 | Full-Image Query Support
# ═══════════════════════════════════════════════════════════════════

_TILE_OFFSET_CACHE: dict = {}  # (data_root, split) → {stem: (x, y, w, h)}


def _get_tile_offsets(data_root: Path, split: str) -> dict:
    """获取所有 tile 在原图中的位置偏移 | Get tile offsets in full-image coordinates.

    :return: {stem: (orig_x, orig_y, width, height)}
    """
    cache_key = (str(data_root), split)
    if cache_key in _TILE_OFFSET_CACHE:
        return _TILE_OFFSET_CACHE[cache_key]

    ann_file = data_root / "annotations" / f"instances_{split}.json"
    offsets = {}
    if ann_file.exists():
        with open(ann_file) as f:
            coco = json.load(f)
        for img in coco["images"]:
            stem = Path(img["file_name"]).stem
            offsets[stem] = (
                img.get("orig_x", 0), img.get("orig_y", 0),
                img.get("width", 0), img.get("height", 0),
            )
    _TILE_OFFSET_CACHE[cache_key] = offsets
    return offsets


def build_full_image_gt(source_img: str, tile_stems: list[str],
                        split: str, data_root: Path,
                        target_class_id: int | None = None) -> tuple[np.ndarray, int, int, list[dict]]:
    """为一张源图构建全图 GT mask | Build full-image GT mask for a source image.

    将所有 tile 的 GT mask 按 orig_x/orig_y 拼回全图。
    Merge all tile GT masks into full-image coordinates.

    :param source_img: 源图 ID | Source image ID (e.g. "P0019")
    :param tile_stems: 该源图的所有 tile stem | All tile stems for this source
    :param split:      数据划分 | Data split
    :param data_root:  数据根目录 | Data root
    :param target_class_id: 仅渲染该类别实例 (None=全部) | Only render instances of this class (None=all).
    :return: (full_gt, H_full, W_full, tile_data)
        - full_gt:    [H_full, W_full] uint8 binary mask
        - H_full, W_full: 全图尺寸 | Full image dimensions
        - tile_data: [{stem, img, mask, orig_x, orig_y, w, h}, ...]
    """
    offsets = _get_tile_offsets(data_root, split)

    # 计算全图尺寸 | Compute full image dimensions
    max_x, max_y = 0, 0
    tile_data = []
    for stem in tile_stems:
        off = offsets.get(stem)
        if off is None:
            continue
        ox, oy, tw, th = off
        max_x = max(max_x, ox + tw)
        max_y = max(max_y, oy + th)

        img, mask = load_instance_tile_and_mask(stem, split, data_root, target_class_id=target_class_id)
        gt_bin = semantic_mask_to_binary(mask, is_tile=True)
        tile_data.append({
            "stem": stem, "img": img, "mask": gt_bin,
            "orig_x": ox, "orig_y": oy, "w": tw, "h": th,
        })

    if max_x == 0 or max_y == 0:
        return np.zeros((1, 1), dtype=np.uint8), 1, 1, tile_data

    # 拼合 GT | Merge GT masks
    full_gt = np.zeros((max_y, max_x), dtype=np.uint8)
    for td in tile_data:
        ox, oy, th, tw = td["orig_x"], td["orig_y"], td["h"], td["w"]
        gt_patch = td["mask"][:th, :tw]
        full_gt[oy:oy + th, ox:ox + tw] = np.maximum(
            full_gt[oy:oy + th, ox:ox + tw], gt_patch
        )

    return full_gt, max_y, max_x, tile_data


def merge_tile_predictions(predictions: list[dict], H_full: int, W_full: int) -> np.ndarray:
    """将 tile 级预测拼接为全图预测 | Merge tile-level predictions into full-image map.

    :param predictions: [{orig_x, orig_y, h, w, pred_bin}, ...]
    :param H_full, W_full: 全图尺寸 | Full image dimensions
    :return: [H_full, W_full] float32 prediction map
    """
    full_pred = np.zeros((H_full, W_full), dtype=np.float32)
    weight = np.zeros((H_full, W_full), dtype=np.float32)

    for p in predictions:
        ox, oy = p["orig_x"], p["orig_y"]
        th, tw = p["h"], p["w"]
        pred_patch = p["pred_bin"][:th, :tw]
        full_pred[oy:oy + th, ox:ox + tw] += pred_patch
        weight[oy:oy + th, ox:ox + tw] += 1.0

    # 重叠区域取平均 | Average in overlapping regions
    weight[weight == 0] = 1.0
    return full_pred / weight


# ═══════════════════════════════════════════════════════════════════
# Decoder | Lightweight Few-Shot Decoder
# ═══════════════════════════════════════════════════════════════════

class FewShotDecoder(nn.Module):
    """
    轻量少样本解码器 + Support Mask 空间模板.
    Lightweight Few-Shot Decoder + Support Mask Spatial Template.

    三路径 | Three Paths:
        Semantic:  Support Prototype [640] → MLP → 32 coeffs (类别语义)
        Template:  Support Mask avg → 64² template → spatial attention on P4 (形状先验)
        Combined:  P4 features × template_attention → refined → final mask
    """

    def __init__(self, feat_dim: int = 640, proto_dim: int = 32,
                 hidden_dim: int = 256, use_template: bool = True):
        super().__init__()
        self.use_template = use_template

        # Coefficient predictor (semantic): support prototype → 32 coeffs
        self.coeff_predictor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.InstanceNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.InstanceNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proto_dim),
        )

        # Mask template encoder: avg support mask → spatial attention
        # Input: [B, 1, 64, 64] avg support mask template
        # Output: [B, 1, H/16, W/16] spatial attention weights ∈ [0, 1]
        if use_template:
            self.template_encoder = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1, bias=False),
                nn.InstanceNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 3, padding=1, bias=False),
                nn.InstanceNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1, bias=False),
                nn.Sigmoid(),  # attention in [0, 1]
            )

        # P4 feature refinement
        self.feat_refine = nn.Sequential(
            nn.Conv2d(feat_dim, 256, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Mask head: 64 + 32 (proto coarse) → 1
        self.mask_head = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, p4: torch.Tensor, proto: torch.Tensor,
                support_proto: torch.Tensor,
                support_template: torch.Tensor = None) -> torch.Tensor:
        """
        :param p4: [B, 640, H/16, W/16] query P4 features
        :param proto: [B, 32, H/4, W/4] proto masks
        :param support_proto: [B, 640] support prototype (L2-normalized)
        :param support_template: [B, 1, 64, 64] avg support mask, or None
        :return: [B, 1, H, W] predicted mask logits
        """
        B, _, Hp, Wp = proto.shape
        _, _, H4, W4 = p4.shape

        # 1. Coefficient prediction
        coeffs = self.coeff_predictor(support_proto)  # [B, 32]

        # 2. Spatial template attention (Plan B core)
        if self.use_template and support_template is not None:
            # Resize template to P4 resolution
            tmpl = F.interpolate(support_template, size=(H4, W4),
                                 mode="bilinear", align_corners=False)  # [B, 1, H/16, W/16]
            attn = self.template_encoder(tmpl)  # [B, 1, H/16, W/16] in [0, 1]
            # Spatial gating: amplify attended regions up to 3×, suppress unattended to 0.5×
            p4 = p4 * (0.5 + 2.5 * attn)  # [B, 640, H/16, W/16]

        # 3. Coarse mask from proto
        proto_flat = proto.view(B, 32, -1)
        coarse_flat = torch.bmm(coeffs.unsqueeze(1), proto_flat)
        coarse = coarse_flat.view(B, 1, Hp, Wp)

        # 4. P4 refinement
        refined = self.feat_refine(p4)
        proto_down = F.interpolate(proto, size=(H4, W4), mode="bilinear",
                                   align_corners=False)
        fused = torch.cat([refined, proto_down], dim=1)

        # 5. Mask head
        logits_4 = self.mask_head(fused)
        logits = F.interpolate(logits_4, size=(Hp * 4, Wp * 4), mode="bilinear",
                               align_corners=False)

        return logits


# ═══════════════════════════════════════════════════════════════════
# Loss | Dice + BCE
# ═══════════════════════════════════════════════════════════════════

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss for binary masks."""
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return (1.0 - dice).mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Dice + BCE combined loss."""
    d_loss = dice_loss(pred, target)
    bce = F.binary_cross_entropy_with_logits(pred, target)
    total = d_loss + bce
    return total, {"dice": d_loss.item(), "bce": bce.item(), "total": total.item()}


# ═══════════════════════════════════════════════════════════════════
# Training | Episodic K-shot
# ═══════════════════════════════════════════════════════════════════

def extract_features(model, images: list[np.ndarray],
                     device: str = "cuda", no_grad: bool = True) -> list[dict]:
    """
    提取 FastSAM backbone 特征 (hook + 正确 forward).
    Extract FastSAM backbone features via hooks + proper forward.

    Uses SegmentationModel's save list [4,6,9,12,15,18,21] for Concat.
    Hooks capture P3@15 and P4@18, then proto from P3.

    :param model: FastSAM model (ultralytics).
    :param images: List of uint8 numpy arrays [H, W, 3].
    :param device: "cuda" or "cpu".
    :param no_grad: If True (default), use torch.no_grad() for frozen inference.
                    If False, allows gradients through unfrozen backbone layers.
    :return: [{p4: [1, C, H/16, W/16], proto: [1, 32, H/4, W/4], p8: [1, C, H/32, W/32]}, ...]
    """
    feats_list = []
    seg = model.model          # SegmentationModel (handles save/Concat)
    seq = seg.model            # Sequential[23]
    save_set = set(seg.save)   # {4, 6, 9, 12, 15, 18, 21}
    segment = seq[22]          # Segment head

    for img in images:
        if isinstance(img, np.ndarray):
            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            tensor = img
        tensor = tensor.to(device)

        # Register hooks — don't detach when training through backbone
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

        # Forward with correct Concat handling (matching YOLO's _predict_once)
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
            # x is now the detection output (not needed here)

        for h in handles:
            h.remove()

        p3 = hooked.get("p3")
        p4 = hooked.get("p4")
        p8 = hooked.get("p8")
        if p3 is None or p4 is None:
            raise RuntimeError(f"Hook failed: p3={p3 is not None}, p4={p4 is not None}")

        # Proto masks: inside grad context if backbone is trainable (Segment layer 22)
        # Proto masks need gradients if Segment head is unfrozen
        with ctx:
            proto = segment.proto(p3)  # [1, 32, H/4, W/4]

        feats_list.append({
            "p3": p3,       # [1, C, H/8, W/8] — 高分辨率细节 | High-res detail
            "p4": p4,       # [1, C, H/16, W/16]
            "proto": proto,  # [1, 32, H/4, W/4]
            "p8": p8,       # [1, C, H/32, W/32] — SPM input
        })
    return feats_list


def compute_support_mask_template(support_masks: list[np.ndarray],
                                   size: int = 64) -> torch.Tensor:
    """
    从 K 个 support mask 计算空间模板.
    Compute spatial template from K support masks.

    将所有 support mask resize 到 size×size, 取平均,
    生成 [1, 1, size, size] 的空间先验.
    Average all support masks resized to size² → spatial prior.

    :param support_masks: list of [H, W] binary FG masks.
    :param size: target template size.
    :return: [1, 1, size, size] float32 tensor.
    """
    templates = []
    for mask in support_masks:
        t = cv2.resize(mask.astype(np.float32), (size, size),
                       interpolation=cv2.INTER_LINEAR)
        templates.append(t)
    avg = np.mean(templates, axis=0)  # [size, size]
    return torch.tensor(avg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64]


def compute_support_bbox(support_masks: list[np.ndarray]) -> torch.Tensor:
    """
    从 K 个 support mask 计算归一化 bbox.
    取所有 support mask 的平均 bbox (归一化到 [0, 1]).

    :param support_masks: list of [H, W] binary FG masks (float32).
    :return: [1, 4] normalized bbox [x1, y1, x2, y2].
    """
    all_bboxes = []
    for mask in support_masks:
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)
        H, W = mask.shape
        if rows.any() and cols.any():
            y_idx = np.where(rows)[0]; x_idx = np.where(cols)[0]
            x1, y1 = float(x_idx[0]) / W, float(y_idx[0]) / H
            x2, y2 = float(x_idx[-1]) / W, float(y_idx[-1]) / H
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0  # full image fallback
        all_bboxes.append([x1, y1, x2, y2])

    avg_bbox = np.mean(all_bboxes, axis=0)  # [4]
    return torch.tensor(avg_bbox, dtype=torch.float32).unsqueeze(0)  # [1, 4]


def compute_support_prototype(support_feats: list[dict],
                              source: str = "p4") -> torch.Tensor:
    """
    从 K 个 support 特征计算 prototype.
    Compute prototype from K support features.

    :param support_feats: K 个 support 特征字典 | K support feature dicts.
    :param source: 特征来源 | Feature source — "p4" (stride-16, 语义-空间平衡)
                   or "p8" (stride-32, 全局语义).
    :return: [1, feat_dim] L2-normalized prototype.
    """
    vectors = []
    for sf in support_feats:
        v = sf[source].mean(dim=(2, 3))  # spatial avg → [1, feat_dim]
        v = F.normalize(v, p=2, dim=-1)
        vectors.append(v)
    proto = torch.stack(vectors).mean(dim=0)  # [1, feat_dim]
    return F.normalize(proto, p=2, dim=-1)


def _compute_gt_density(mask: np.ndarray, p8_h: int, p8_w: int) -> torch.Tensor:
    """将 GT mask 池化到 P8 分辨率作为 SPM 监督信号 | Pool GT mask to P8 resolution for SPM supervision.

    :param mask: [H, W] binary FG mask.
    :param p8_h, p8_w: P8 feature map spatial size.
    :return: [1, 1, p8_h, p8_w] FG density per cell ∈ [0, 1].
    """
    mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    density = F.adaptive_avg_pool2d(mask_t, (p8_h, p8_w))  # [1, 1, p8_h, p8_w]
    return density


def _normalize_mask_to_4d(mask: torch.Tensor) -> torch.Tensor:
    """将 decoder 输出统一为 [B, 1, H, W] 4D 格式 | Normalize decoder output to [B, 1, H, W]."""
    if mask.dim() == 2:
        return mask.unsqueeze(0).unsqueeze(0)   # [H,W] → [1,1,H,W]
    elif mask.dim() == 3:
        return mask.unsqueeze(1)                 # [B,H,W] → [B,1,H,W]
    elif mask.dim() == 4:
        return mask                              # already [B,1,H,W]
    else:
        raise ValueError(f"Unexpected mask dim: {mask.dim()}")


def train_episode(model, decoder, optimizer, class_id: int,
                  support_stems: list[str], query_stem: str,
                  split: str, device: str, data_format: str = "isaid5i",
                  data_root: Path = None, decoder_type: str = "baseline",
                  spm=None, backbone_trainable: bool = False,
                  proto_source: str = "p4") -> dict:
    """单次 episodic 训练步 | Single episodic training step."""
    is_tile = data_format in ("isaid_tiles", "isaid_instance")
    is_instance = (data_format == "isaid_instance")

    def _load(stem):
        if is_instance:
            return load_instance_tile_and_mask(stem, split, data_root, target_class_id=class_id)
        elif is_tile:
            return load_tile_and_mask(stem, split, data_root, target_class_id=class_id)
        else:
            return load_image_and_mask(stem, split, data_root)

    # Load support images + masks
    support_imgs, support_bmasks = [], []
    for stem in support_stems:
        img, mask = _load(stem)
        support_imgs.append(img)
        support_bmasks.append(semantic_mask_to_binary(mask, is_tile=is_tile))

    # Load query
    # SCACS协议: Pure decoder 无 class conditioning → GT 应包含全部实例
    # SCACS protocol: Pure decoder has no class conditioning → GT contains all instances
    # Adaptive decoder 有 prototype conditioning → GT 仅含 target class 实例
    # Adaptive decoder has prototype conditioning → GT only target class instances
    if decoder_type in ("pure", "pure-p3p4"):
        # Pure decoder: 无法区分类别, GT=全部可见实例 | Cannot distinguish classes, GT=all FG
        if is_instance:
            query_img, query_mask = load_instance_tile_and_mask(query_stem, split, data_root, target_class_id=None)
        elif is_tile:
            query_img, query_mask = load_tile_and_mask(query_stem, split, data_root, target_class_id=None)
        else:
            query_img, query_mask = load_image_and_mask(query_stem, split, data_root)
    else:
        # Adaptive decoder: 类条件, GT=仅 target class | Class-conditioned, GT=target class only
        query_img, query_mask = _load(query_stem)
    query_gt = semantic_mask_to_binary(query_mask, is_tile=is_tile)  # [H, W] float32

    # Extract features
    # Support: always frozen (no_grad=True) — support features are "inputs"
    # Query: allow gradients if backbone is trainable
    support_feats = extract_features(model, support_imgs, device, no_grad=True)
    query_feats = extract_features(model, [query_img], device, no_grad=not backbone_trainable)[0]

    # Support prototype (source: p4 or p8)
    support_proto = compute_support_prototype(support_feats, source=proto_source)  # [1, feat_dim]

    if decoder_type == "adaptive":
        # ── AdaptiveSparseDecoder: proto mask 线性组合 + P4 精炼 ──
        # No spatial template. Forward returns sigmoid mask (may be [H,W] or [1,H,W]).
        p4 = query_feats["p4"].to(device)            # [1, C, H/16, W/16]
        proto_masks = query_feats["proto"].to(device)  # [1, 32, H/4, W/4]
        mask_s4 = decoder(p4, proto_masks, support_proto)
        mask_s4 = _normalize_mask_to_4d(mask_s4)  # → [1, 1, H/4, W/4]

        # Upsample to GT resolution
        H_gt, W_gt = query_gt.shape
        mask_pred = F.interpolate(
            mask_s4, size=(H_gt, W_gt), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)  # [H_gt, W_gt]

        # Loss: BCE on sigmoid mask (use clamp for stability)
        gt_tensor = torch.from_numpy(query_gt).float().to(device)
        bce = F.binary_cross_entropy(mask_pred.clamp(1e-7, 1 - 1e-7), gt_tensor)
        # Dice on sigmoid mask
        inter = (mask_pred * gt_tensor).sum()
        union = mask_pred.sum() + gt_tensor.sum()
        dice = (2.0 * inter + 1e-6) / (union + 1e-6)
        d_loss = 1.0 - dice
        loss = bce + d_loss
        loss_dict = {"dice": d_loss.item(), "bce": bce.item(), "total": loss.item()}
    elif decoder_type == "adaptive-p3p4":
        # ── AdaptiveDecoderP3P4: P8→prototype + P3→boundary + P4→main ──
        p3 = query_feats["p3"].to(device)            # [1, C, H/8, W/8]
        p4 = query_feats["p4"].to(device)            # [1, C, H/16, W/16]
        proto_masks = query_feats["proto"].to(device)  # [1, 32, H/4, W/4]
        mask_s4 = decoder(p3, p4, proto_masks, support_proto)
        mask_s4 = _normalize_mask_to_4d(mask_s4)  # → [1, 1, H/4, W/4]

        # Upsample to GT resolution
        H_gt, W_gt = query_gt.shape
        mask_pred = F.interpolate(
            mask_s4, size=(H_gt, W_gt), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)  # [H_gt, W_gt]

        # Loss: BCE on sigmoid mask
        gt_tensor = torch.from_numpy(query_gt).float().to(device)
        bce = F.binary_cross_entropy(mask_pred.clamp(1e-7, 1 - 1e-7), gt_tensor)
        # Dice on sigmoid mask
        inter = (mask_pred * gt_tensor).sum()
        union = mask_pred.sum() + gt_tensor.sum()
        dice = (2.0 * inter + 1e-6) / (union + 1e-6)
        d_loss = 1.0 - dice
        loss = bce + d_loss
        loss_dict = {"dice": d_loss.item(), "bce": bce.item(), "total": loss.item()}
    elif decoder_type == "center_affinity":
        # ── CenterAffinityDecoder: Center Heatmap + Offset Field + Proto Mask ──
        p3 = query_feats["p3"].to(device)            # [1, C, H/8, W/8]
        p4 = query_feats["p4"].to(device)            # [1, C, H/16, W/16]
        proto_masks = query_feats["proto"].to(device)  # [1, 32, H/4, W/4]

        # Decoder forward: center heatmap + offset field + proto mask
        center_hm, offset_field, proto_mask = decoder(p3, p4, proto_masks, support_proto)
        # center_hm:    [H/8, W/8] ∈ [0, 1]
        # offset_field: [2, H/8, W/8] (dx, dy)
        # proto_mask:   [H/4, W/4] ∈ [0, 1]

        _, H_c, W_c = center_hm.shape  # H/8, W/8
        H_gt, W_gt = query_gt.shape

        # ── Common GT | 共享 GT ──
        gt_tensor = torch.from_numpy(query_gt).float().to(device)  # [H_gt, W_gt] merged

        # ═══════════════════════════════════════════════════════════════
        # Loss 1: Proto semantic FG (same as existing)
        # ═══════════════════════════════════════════════════════════════
        proto_up = F.interpolate(
            proto_mask.unsqueeze(0).unsqueeze(0), size=(H_gt, W_gt),
            mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)
        proto_bce = F.binary_cross_entropy(proto_up.clamp(1e-7, 1 - 1e-7), gt_tensor)
        proto_inter = (proto_up * gt_tensor).sum()
        proto_union = proto_up.sum() + gt_tensor.sum()
        proto_dice_val = (2.0 * proto_inter + 1e-6) / (proto_union + 1e-6)
        proto_loss = proto_bce + (1.0 - proto_dice_val)

        # ═══════════════════════════════════════════════════════════════
        # Loss 2: Center Heatmap — MSE vs Gaussian centers
        # ═══════════════════════════════════════════════════════════════
        center_gt = torch.zeros(H_c, W_c, device=device)
        gt_full_list = load_instance_masks_from_coco(
            query_stem, split, data_root, class_id, target_size=None,
        )

        for inst_mask in gt_full_list:
            # Compute centroid of each instance mask at stride-8 resolution
            inst = torch.from_numpy(inst_mask).float()
            H_i, W_i = inst.shape
            # Downsample mask to stride-8 for center computation
            inst_s8 = F.interpolate(
                inst.unsqueeze(0).unsqueeze(0), size=(H_c, W_c),
                mode="nearest",
            ).squeeze()
            if inst_s8.sum() < 1:
                continue
            # Centroid (mass center of binary mask)
            yy, xx = torch.meshgrid(
                torch.arange(H_c, device=device, dtype=torch.float32),
                torch.arange(W_c, device=device, dtype=torch.float32),
                indexing="ij",
            )
            mass = inst_s8.sum()
            cy = (yy * inst_s8).sum() / mass
            cx = (xx * inst_s8).sum() / mass

            # Gaussian kernel centered at (cx, cy), σ = 2.0 at stride-8
            sigma = 2.0
            gauss = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            center_gt = torch.maximum(center_gt, gauss)

        center_gt = center_gt.clamp(0, 1)
        center_loss = F.mse_loss(center_hm, center_gt)

        # ═══════════════════════════════════════════════════════════════
        # Loss 3: Offset Field — SmoothL1 on FG pixels
        # ═══════════════════════════════════════════════════════════════
        offset_gt = torch.zeros(2, H_c, W_c, device=device)  # [2, H/8, W/8]
        fg_mask_s8 = torch.zeros(H_c, W_c, device=device)    # which pixels are FG

        for inst_mask in gt_full_list:
            inst = torch.from_numpy(inst_mask).float()
            H_i, W_i = inst.shape
            inst_s8 = F.interpolate(
                inst.unsqueeze(0).unsqueeze(0), size=(H_c, W_c),
                mode="nearest",
            ).squeeze()
            inst_bin = (inst_s8 > 0.5).float()
            fg_mask_s8 = fg_mask_s8 + inst_bin
            if inst_bin.sum() < 1:
                continue

            # Centroid
            yy, xx = torch.meshgrid(
                torch.arange(H_c, device=device, dtype=torch.float32),
                torch.arange(W_c, device=device, dtype=torch.float32),
                indexing="ij",
            )
            mass = inst_bin.sum()
            cy = (yy * inst_bin).sum() / mass
            cx = (xx * inst_bin).sum() / mass

            # Offset: (center_x - x, center_y - y) for FG pixels
            # Normalize by sigma_offset
            sigma_off = 8.0
            off_dx = (cx - xx) / sigma_off
            off_dy = (cy - yy) / sigma_off

            # Only set for pixels in this instance
            inst_pixels = (inst_bin > 0.5)
            offset_gt[0, inst_pixels] = off_dx[inst_pixels]
            offset_gt[1, inst_pixels] = off_dy[inst_pixels]

        fg_mask_s8 = (fg_mask_s8 > 0.5).float()  # binary
        n_fg = fg_mask_s8.sum().clamp(min=1)

        # SmoothL1 on FG pixels only
        offset_diff = F.smooth_l1_loss(
            offset_field * fg_mask_s8.unsqueeze(0),
            offset_gt * fg_mask_s8.unsqueeze(0),
            reduction="none",
        )
        offset_loss = offset_diff.sum() / n_fg

        # ── Total loss ──
        loss = proto_loss + 1.0 * center_loss + 0.5 * offset_loss

        loss_dict = {
            "dice": proto_dice_val.item(), "bce": proto_bce.item(),
            "center_loss": center_loss.item(), "offset_loss": offset_loss.item(),
        }

        # IoU: proto mask semantic
        with torch.no_grad():
            pred_bin = (proto_up > 0.5).float()
            inter_iou = (pred_bin * gt_tensor).sum()
            union_iou = (pred_bin + gt_tensor).clamp(0, 1).sum()
            iou = (inter_iou / max(union_iou, 1)).item()

        if loss.dim() > 0:
            loss = loss.mean()
    elif decoder_type == "dynamic_kernel":
        # ── DynamicKernelDecoder: Prototype → N Kernels → N Instance Masks ──
        p3 = query_feats["p3"].to(device)            # [1, C, H/8, W/8]
        p4 = query_feats["p4"].to(device)            # [1, C, H/16, W/16]
        proto_masks = query_feats["proto"].to(device)  # [1, 32, H/4, W/4]

        # Decoder forward: N masks at stride-8 + proto_mask at stride-4
        masks_s8, proto_mask = decoder(p3, p4, proto_masks, support_proto)
        # masks_s8: [N_kernels, H/8, W/8] sigmoid ∈ [0,1]
        # proto_mask: [H/4, W/4]

        _, H_s8, W_s8 = masks_s8.shape
        H_gt, W_gt = query_gt.shape

        # ── Common GT | 共享 GT ──
        gt_tensor = torch.from_numpy(query_gt).float().to(device)  # [H_gt, W_gt] merged

        # ═══════════════════════════════════════════════════════════════
        # Loss 1: Proto auxiliary (训练 coeff_predictor + proto_norm)
        #          始终激活, 防止 proto branch 梯度枯竭
        # ═══════════════════════════════════════════════════════════════
        proto_up = F.interpolate(
            proto_mask.unsqueeze(0).unsqueeze(0), size=(H_gt, W_gt),
            mode='bilinear', align_corners=False,
        ).squeeze(0).squeeze(0)  # [H_gt, W_gt]
        proto_bce = F.binary_cross_entropy(proto_up.clamp(1e-7, 1 - 1e-7), gt_tensor)
        proto_inter = (proto_up * gt_tensor).sum()
        proto_union = proto_up.sum() + gt_tensor.sum()
        proto_dice_val = (2.0 * proto_inter + 1e-6) / (proto_union + 1e-6)
        proto_loss = proto_bce + (1.0 - proto_dice_val)

        # ═══════════════════════════════════════════════════════════════
        # Loss 2: Semantic warmup (训练 FPN + mask_feat + kernel_generator)
        #          max-pool over kernels = OR → BCE+Dice vs merged GT
        #          所有 kernel 合并后的前景应覆盖全部 GT
        #          Always active, ensures FPN/mask_feat/kernel_gen get gradient
        # ═══════════════════════════════════════════════════════════════
        kernel_merged = masks_s8.max(dim=0)[0]  # [H/8, W/8] — OR across N kernels
        kernel_up = F.interpolate(
            kernel_merged.unsqueeze(0).unsqueeze(0), size=(H_gt, W_gt),
            mode='bilinear', align_corners=False,
        ).squeeze(0).squeeze(0)  # [H_gt, W_gt]
        kernel_bce = F.binary_cross_entropy(kernel_up.clamp(1e-7, 1 - 1e-7), gt_tensor)
        kernel_inter = (kernel_up * gt_tensor).sum()
        kernel_union = kernel_up.sum() + gt_tensor.sum()
        kernel_dice_val = (2.0 * kernel_inter + 1e-6) / (kernel_union + 1e-6)
        kernel_sem_loss = kernel_bce + (1.0 - kernel_dice_val)

        # ═══════════════════════════════════════════════════════════════
        # Loss 3: Instance-level Hungarian matching (per-instance GT)
        #          初期 match 可能 = 0, 随着 semantic warmup 学会 FG 后逐步激活
        # ═══════════════════════════════════════════════════════════════
        gt_full_list = load_instance_masks_from_coco(
            query_stem, split, data_root, class_id, target_size=None,
        )

        if len(gt_full_list) == 0:
            inst_loss = masks_s8.mean() * 0.1
            loss_dict = {
                "dice": proto_dice_val.item(), "bce": proto_bce.item(),
                "kernel_dice": kernel_dice_val.item(),
                "kernel_cos_sim": 0.0,
                "n_matched": 0, "n_unmatched_pred": 0,
                "n_unmatched_gt": 0, "n_gt": 0,
            }
        else:
            gt_full = torch.from_numpy(np.stack(gt_full_list)).float().to(device)
            gt_s8 = F.interpolate(
                gt_full.unsqueeze(1).float(), size=(H_s8, W_s8),
                mode='nearest',
            ).squeeze(1)  # [M, H_s8, W_s8]
            masks_gt = F.interpolate(
                masks_s8.unsqueeze(0), size=(H_gt, W_gt),
                mode='bilinear', align_corners=False,
            ).squeeze(0)  # [N, H_gt, W_gt]

            matched, unmatched_pred, unmatched_gt = hungarian_match_instances(
                masks_s8, gt_s8, min_dice=0.05,
            )
            inst_loss, loss_dict = multi_instance_loss(
                masks_gt, gt_full, matched, unmatched_pred, unmatched_gt,
                dice_weight=1.0, bce_weight=1.0, empty_weight=0.5,
            )
            loss_dict["dice"] = proto_dice_val.item()
            loss_dict["bce"] = proto_bce.item()
            loss_dict["kernel_dice"] = kernel_dice_val.item()
            loss_dict["n_gt"] = len(gt_full_list)

        # ── Loss 4: Kernel diversity (prevents weight collapse) ──
        # 所有 kernel 从同一个 prototype 生成, 容易坍缩为相同权重.
        # All kernels share the same prototype input → prone to weight collapse.
        # Diversity loss pushes kernel weight vectors apart via cosine similarity penalty.
        kernels = getattr(decoder, "_last_kernels", None)
        if kernels is not None:
            div_loss = DynamicKernelDecoder.kernel_diversity_loss(kernels)
            loss_dict["kernel_cos_sim"] = div_loss.item()
        else:
            div_loss = 0.0

        # ── Total loss: proto auxiliary + kernel semantic + instance matching + diversity ──
        loss = proto_loss + kernel_sem_loss + inst_loss + 0.1 * div_loss

        # IoU: kernel semantic output (max-pool of all kernels)
        with torch.no_grad():
            pred_bin = (kernel_up > 0.5).float()
            inter_iou = (pred_bin * gt_tensor).sum()
            union_iou = (pred_bin + gt_tensor).clamp(0, 1).sum()
            iou = (inter_iou / max(union_iou, 1)).item()

        if loss.dim() > 0:
            loss = loss.mean()
    elif decoder_type in ("pure", "pure-p3p4"):
        # ── PureDecoder / PureDecoderP3P4: 无 prototype, 纯 query-driven ──
        p4 = query_feats["p4"].to(device)            # [1, C, H/16, W/16]
        if decoder_type == "pure-p3p4":
            p3 = query_feats["p3"].to(device)         # [1, C, H/8, W/8]
            mask_s4 = decoder(p3, p4)
        else:
            mask_s4 = decoder(p4)
        mask_s4 = _normalize_mask_to_4d(mask_s4)  # → [1, 1, H/4, W/4]

        # Upsample to GT resolution
        H_gt, W_gt = query_gt.shape
        mask_pred = F.interpolate(
            mask_s4, size=(H_gt, W_gt), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)  # [H_gt, W_gt]

        # Loss: Dice + BCE
        gt_tensor = torch.from_numpy(query_gt).float().to(device)
        bce = F.binary_cross_entropy(mask_pred.clamp(1e-7, 1 - 1e-7), gt_tensor)
        inter = (mask_pred * gt_tensor).sum()
        union = mask_pred.sum() + gt_tensor.sum()
        dice = (2.0 * inter + 1e-6) / (union + 1e-6)
        d_loss = 1.0 - dice
        loss = bce + d_loss
        loss_dict = {"dice": d_loss.item(), "bce": bce.item(), "total": loss.item()}
    else:
        # ── Baseline FewShotDecoder: coeffs + spatial template + P4 refine ──
        support_template = compute_support_mask_template(support_bmasks).to(device)  # [1, 1, 64, 64]

        p4 = query_feats["p4"].to(device)
        proto = query_feats["proto"].to(device)
        pred = decoder(p4, proto, support_proto, support_template)  # [1, 1, H, W] logits

        # Resize pred to match GT
        H_gt, W_gt = query_gt.shape
        pred = F.interpolate(pred, size=(H_gt, W_gt), mode="bilinear", align_corners=False)

        # Loss
        gt_tensor = torch.from_numpy(query_gt).unsqueeze(0).unsqueeze(0).float().to(device)
        loss, loss_dict = combined_loss(pred, gt_tensor)

    # ── SPM loss (可选) | Optional SPM loss ──
    spm_loss = None
    if spm is not None:
        p8 = query_feats.get("p8")
        if p8 is not None:
            p8 = p8.to(device)  # [1, C, H/32, W/32]
            importance = spm.importance_head(p8)  # [1, 1, H/32, W/32] raw logits
            # GT density: pool query mask to P8 resolution
            gt_density = _compute_gt_density(query_gt, importance.shape[2], importance.shape[3]).to(device)
            # BCE loss
            bce_spm = F.binary_cross_entropy_with_logits(importance, gt_density)
            # Budget loss: encourage ~40% tile selection
            imp_mean = torch.sigmoid(importance).mean()
            budget_target = 0.4
            budget = (imp_mean - budget_target) ** 2
            spm_loss = bce_spm + 0.1 * budget  # budget weight reduced
            loss = loss + 0.05 * spm_loss  # λ_spm = 0.05
            loss_dict["spm_bce"] = bce_spm.item()
            loss_dict["spm_budget"] = budget.item()
            loss_dict["spm_mean"] = imp_mean.item()

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # ── 前向统计 (机制诊断, 仅 collect_stats; 在 clip 前读 coeff 梯度) | forward stats (pre-clip) ──
    fwd_stats = {}
    if getattr(decoder, "collect_stats", False) and hasattr(decoder, "coeff_predictor"):
        cg_sq = sum(float(p.grad.detach().pow(2).sum().item())
                    for p in decoder.coeff_predictor.parameters() if p.grad is not None)
        fwd_stats = dict(getattr(decoder, "last_stats", {}))  # basis_l2/coeff_l2/pre_sigmoid/sat_frac
        fwd_stats["coeff_grad_norm"] = cg_sq ** 0.5

    all_params = list(decoder.parameters())
    if spm is not None:
        all_params += list(spm.parameters())
    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
    optimizer.step()

    # IoU
    with torch.no_grad():
        if decoder_type == "dynamic_kernel":
            # IoU already computed in training branch (proto mask semantic)
            pass  # iou variable already set
        elif decoder_type in ("adaptive", "adaptive-p3p4", "pure", "pure-p3p4"):
            pred_bin = (mask_pred > 0.5).float()
            inter = (pred_bin * gt_tensor).sum()
            union = (pred_bin + gt_tensor).clamp(0, 1).sum()
            iou = (inter / max(union, 1)).item()
        else:
            pred_bin = (torch.sigmoid(pred) > 0.5).float()
            inter = (pred_bin * gt_tensor).sum()
            union = (pred_bin + gt_tensor).clamp(0, 1).sum()
            iou = (inter / max(union, 1)).item()

    return {"loss": loss.item(), "iou": iou, **loss_dict, **fwd_stats}


# ═══════════════════════════════════════════════════════════════════
# Main | Training Loop
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="All-15-Class K-shot Fine-tuning")
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--episodes-per-epoch", type=int, default=200)
    parser.add_argument("--val-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None,
                        help="数据根目录 (默认根据格式自动选择) | Data root directory")
    parser.add_argument("--data-format", type=str, default="isaid5i",
                        choices=["isaid5i", "isaid_tiles", "isaid_instance"],
                        help="数据格式: isaid5i/iSAID-5i, isaid_tiles/旧tile, "
                             "isaid_instance/新COCO tile")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--decoder", type=str, default="baseline",
                        choices=["baseline", "adaptive", "adaptive-p3p4", "pure", "pure-p3p4",
                                 "dynamic_kernel", "center_affinity"],
                        help="Decoder 类型 | Decoder type: baseline (FewShotDecoder + template), "
                             "adaptive (AdaptiveSparseDecoder P4-only), "
                             "adaptive-p3p4 (P3+P4+P8), "
                             "pure (PureDecoder P4-only, no prototype), "
                             "pure-p3p4 (PureDecoderP3P4, no prototype), "
                             "dynamic_kernel (DynamicKernelDecoder: Prototype→Kernels→N masks), "
                             "center_affinity (CenterAffinityDecoder: Center+Offset→Instances)")
    parser.add_argument("--use-spm", action="store_true",
                        help="启用 SPM tile routing (P8 → Importance → Top-K)")
    parser.add_argument("--unfreeze-layers", type=int, default=0,
                        help="解冻 backbone 最后 N 层 (0=全冻结, 5=P4+P8+Segment, 8=完整FPN)")
    parser.add_argument("--lr-backbone", type=float, default=None,
                        help="Backbone 学习率 (默认 lr/10) | Backbone learning rate")
    parser.add_argument("--prototype-source", type=str, default="p4",
                        choices=["p4", "p8"],
                        help="Prototype 特征来源 | Prototype feature source: "
                             "p4 (stride-16, semantic-spatial balance) / "
                             "p8 (stride-32, global semantics)")
    parser.add_argument("--normalize-proto", type=str, default="none",
                        choices=["none", "l2", "layernorm", "scale"],
                        help="proto basis 归一化 (修复饱和; 仅 adaptive) | proto-basis normalization "
                             "(fixes saturation; adaptive decoder only): none/l2/layernorm/scale")
    parser.add_argument("--n-kernels", type=int, default=16,
                        help="DynamicKernelDecoder 每类核数 (max instances per class) | "
                             "Number of kernels per class (only for dynamic_kernel decoder)")
    parser.add_argument("--log-forward-stats", action="store_true",
                        help="记录逐 epoch 前向统计 (basis/coeff/pre-sigmoid/sat/grad; 仅 adaptive) | "
                             "log per-epoch forward stats (adaptive only)")
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    device = args.device

    # ── 解析数据格式 | Resolve data format ──
    data_root, data_format, train_split, val_split = _resolve_paths(args)
    is_tile = (data_format in ("isaid_tiles", "isaid_instance"))
    is_instance_tile = (data_format == "isaid_instance")

    fmt_labels = {"isaid5i": "iSAID-5i 256", "isaid_tiles": "TILES (old)",
                  "isaid_instance": "TILES COCO (new)"}
    fmt_label = fmt_labels.get(data_format, data_format)

    # ── 输出目录 | Output Dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        tag = args.decoder
        if args.unfreeze_layers > 0:
            tag += f"_uf{args.unfreeze_layers}"
        if args.use_spm:
            tag += "_spm"
        if args.prototype_source != "p4":
            tag += f"_proto{args.prototype_source}"
        if args.normalize_proto != "none":
            tag += f"_norm{args.normalize_proto}"
        args.output_dir = f"runs/train_fewshot_allcls_K{args.k_shot}_{tag}_{ts}"
    out_dir = Path(args.output_dir).resolve()  # 绝对路径 (ultralytics torch_save wrapper 需要)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  All-15-Class K-shot Fine-tuning")
    print(f"  Data: {data_root} (format: {fmt_label})")
    print(f"  K={args.k_shot} | Epochs={args.epochs} | LR={args.lr}")
    print(f"  Episodes/epoch={args.episodes_per_epoch}")
    print(f"  Device={device} | Output={out_dir}")
    print(f"{'=' * 60}")

    # ── 1. Build class index | 构建类别索引 ──
    print(f"\n[1/4] Building class index ({fmt_label})...")
    if data_format == "isaid_instance":
        train_index = _build_class_index_instance(data_root, train_split)
        val_index = _build_class_index_instance(data_root, val_split) \
            if val_split != train_split else train_index
    elif data_format == "isaid_tiles":
        train_index = _build_class_index_tiles(data_root, train_split)
        val_index = train_index  # 旧格式无独立 val | Old format: no separate val
    else:
        train_index = _build_class_index_isaid5i(data_root, train_split)
        val_index = train_index

    print(f"  [Train] split={train_split}:")
    for cls_id in sorted(train_index.keys()):
        src_to_tiles = train_index[cls_id]
        n_tiles = sum(len(tiles) for tiles in src_to_tiles.values())
        n_sources = len(src_to_tiles)
        print(f"    Class {cls_id:>2d} ({CATEGORY_NAMES.get(cls_id, '?'):<18s}): "
              f"{n_tiles:>5d} tiles from {n_sources:>3d} source images")
    total_tiles = sum(sum(len(tiles) for tiles in v.values()) for v in train_index.values())
    print(f"    Total: {total_tiles} tiles across {len(train_index)} classes")
    if val_index is not train_index:
        val_total = sum(sum(len(tiles) for tiles in v.values()) for v in val_index.values())
        print(f"  [Val] split={val_split}: {val_total} tiles")

    # ── 2. Load FastSAM | 加载 FastSAM ──
    print(f"\n[2/4] Loading FastSAM (ultralytics)...")
    from ultralytics import FastSAM
    model_path = _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"
    model = FastSAM(str(model_path))
    model.model.cuda().eval()
    # Freeze all parameters
    for p in model.model.parameters():
        p.requires_grad = False
    print(f"  Model frozen on {device}")

    # ── 部分解冻 Backbone | Partial Backbone Unfreezing ──
    backbone_trainable = False
    if args.unfreeze_layers > 0:
        backbone_trainable = True
        seq = model.model.model  # Sequential[23]
        n_total = len(seq)
        start_layer = max(0, n_total - args.unfreeze_layers)
        unfrozen_layers = []
        unfrozen_params = 0
        for i in range(start_layer, n_total):
            layer = seq[i]
            for p in layer.parameters():
                p.requires_grad = True
                unfrozen_params += p.numel()
            unfrozen_layers.append(f"[{i}] {type(layer).__name__}")
        n_frozen_params = sum(p.numel() for p in model.model.parameters() if not p.requires_grad)
        print(f"  Backbone unfrozen: last {args.unfreeze_layers} layers "
              f"(indices {start_layer}-{n_total-1})")
        for l in unfrozen_layers:
            print(f"    {l}")
        print(f"  Trainable: {unfrozen_params/1e6:.2f}M params / "
              f"Frozen: {n_frozen_params/1e6:.2f}M params")
        # 保持 eval mode (YOLOv8 需要), 但 requires_grad=True 的层可训练
        # Keep eval mode (required by YOLOv8), but layers with requires_grad=True are trainable

    # ── 3. Build decoder | 构建解码器 ──
    print(f"\n[3/4] Building Decoder (type={args.decoder})...")
    # 自动检测 P4 特征维度 | Auto-detect P4 feature dimension
    # 加载一张测试图提取特征 → 读取 P4 通道数
    # Load a test image to extract features → read P4 channel count
    _test_img = np.zeros((896, 896, 3), dtype=np.uint8)
    _test_feats = extract_features(model, [_test_img], device)
    _p4_channels = _test_feats[0]["p4"].shape[1]
    print(f"  Detected P4 channels: {_p4_channels}")
    if args.decoder == "adaptive":
        decoder = AdaptiveSparseDecoder(in_channels=_p4_channels, proto_dim=32, hidden_dim=256,
                                        use_fdr=False, normalize_proto=args.normalize_proto)
        decoder.collect_stats = args.log_forward_stats  # 前向统计仅 adaptive | forward-stats: adaptive only
        decoder_type = "adaptive"
    elif args.decoder == "adaptive-p3p4":
        _p3_channels = _test_feats[0]["p3"].shape[1]
        print(f"  Detected P3 channels: {_p3_channels}")
        decoder = AdaptiveDecoderP3P4(p3_channels=_p3_channels, p4_channels=_p4_channels,
                                       proto_dim=32, hidden_dim=256)
        decoder_type = "adaptive-p3p4"
    elif args.decoder == "pure":
        decoder = PureDecoder(in_channels=_p4_channels)
        decoder_type = "pure"
    elif args.decoder == "pure-p3p4":
        _p3_channels = _test_feats[0]["p3"].shape[1]
        print(f"  Detected P3 channels: {_p3_channels}")
        decoder = PureDecoderP3P4(p3_channels=_p3_channels, p4_channels=_p4_channels)
        decoder_type = "pure-p3p4"
    elif args.decoder == "dynamic_kernel":
        _p3_channels = _test_feats[0]["p3"].shape[1]
        print(f"  Detected P3 channels: {_p3_channels}")
        decoder = DynamicKernelDecoder(
            p3_channels=_p3_channels, p4_channels=_p4_channels,
            proto_dim=32, n_kernels=args.n_kernels, kernel_dim=256, fpn_dim=256,
            normalize_proto=args.normalize_proto,
        )
        decoder.collect_stats = args.log_forward_stats
        decoder_type = "dynamic_kernel"
        print(f"    n_kernels={args.n_kernels}")
    elif args.decoder == "center_affinity":
        _p3_channels = _test_feats[0]["p3"].shape[1]
        print(f"  Detected P3 channels: {_p3_channels}")
        decoder = CenterAffinityDecoder(
            p3_channels=_p3_channels, p4_channels=_p4_channels,
            proto_dim=32, fpn_dim=64,
            normalize_proto=args.normalize_proto,
        )
        decoder_type = "center_affinity"
    else:
        decoder = FewShotDecoder(feat_dim=_p4_channels, proto_dim=32, hidden_dim=256, use_template=True)
        decoder_type = "baseline"
    decoder = decoder.to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Decoder: {decoder_type}, params: {n_params:,} ({n_params/1e6:.2f}M)")

    # ── SPM (可选) | Optional SPM ──
    spm = None
    if args.use_spm:
        spm = SparsePerceptionModule(in_channels=_p4_channels, mid_channels=256).to(device)
        spm_params = sum(p.numel() for p in spm.parameters())
        print(f"  SPM: enabled, params: {spm_params:,} ({spm_params/1e6:.3f}M)")

    # ── Optimizer (decoder + optional SPM + optional backbone) ──
    lr_backbone = args.lr_backbone if args.lr_backbone is not None else args.lr * 0.1
    param_groups = [{'params': decoder.parameters(), 'lr': args.lr}]
    if spm is not None:
        param_groups.append({'params': spm.parameters(), 'lr': args.lr})
    if backbone_trainable:
        backbone_params = [p for p in model.model.parameters() if p.requires_grad]
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})
        print(f"  Backbone LR: {lr_backbone} (×0.1 of decoder LR)")
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if backbone_trainable and "backbone" in ckpt:
            # 恢复 backbone 权重 | Restore backbone weights
            seq = model.model.model
            for i, state in ckpt["backbone"].items():
                seq[int(i)].load_state_dict(state)
        start_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed from epoch {start_epoch}")

    # ── 4. Training loop | 训练循环 ──
    print(f"\n[4/4] Training...")

    # ── 数据统计 (论文 Experimental Setup / Table 1) | Dataset statistics ──
    total_sources = sum(len(v) for v in train_index.values())
    total_tiles_all = sum(sum(len(tiles) for tiles in v.values()) for v in train_index.values())
    print(f"  ╔══════════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  Dataset Statistics (Table 1 — paper-ready)                                 ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣")
    print(f"  ║  Classes:              {len(train_index):>2d}                                                ║")
    print(f"  ║  Source images:        {total_sources:>4d}  (avg {total_sources/len(train_index):.0f}/class)                                  ║")
    print(f"  ║  Tiles:                {total_tiles_all:>5d}  (avg {total_tiles_all/max(total_sources,1):.1f}/source)                               ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣")
    print(f"  ║  {'Class':<20s} {'Src':>4s} {'Tiles':>6s} {'Avg':>5s} {'Med':>5s} {'Min':>5s} {'Max':>5s} {'Kmax':>5s} ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣")
    for cls_id in sorted(train_index.keys()):
        src_to_tiles = train_index[cls_id]
        n_src = len(src_to_tiles)
        tile_counts = [len(t) for t in src_to_tiles.values()]
        n_tiles = sum(tile_counts)
        name = CATEGORY_NAMES.get(cls_id, '?')
        k_max = n_src - 1
        avg_t = n_tiles / max(n_src, 1)
        med_t = int(np.median(tile_counts))
        min_t = min(tile_counts)
        max_t = max(tile_counts)
        print(f"  ║  {name:<20s} {n_src:>4d} {n_tiles:>6d} {avg_t:>5.1f} {med_t:>5d} {min_t:>5d} {max_t:>5d} {k_max:>5d} ║")
    print(f"  ╠══════════════════════════════════════════════════════════════════════════════╣")
    print(f"  ║  Shot definition:     K = number of source images (not tiles)               ║")
    print(f"  ║  Support per episode: all tiles from K source images                        ║")
    print(f"  ║  Query (train):       1 random tile from a different source image           ║")
    print(f"  ║  Query (val):         full source image → all tiles → merge → full-image IoU║")
    print(f"  ║  Scene overlap:       0% (support ∩ query sources = ∅)                      ║")
    print(f"  ╚══════════════════════════════════════════════════════════════════════════════╝")
    # ── 预采样固定验证集 (Priority 5: reproducibility) | Pre-sample fixed validation episodes ──
    # 用独立 RNG 生成固定的 query sources，所有 K 值共享
    # Use independent RNG for fixed query sources, shared across all K values
    val_rng = random.Random(args.seed + 77777)  # 独立 RNG | Independent RNG
    val_query_rng = random.Random(args.seed + 88888)  # Query 独立于 support | Query independent of support

    fixed_val_episodes = {}  # cls_id → [{support_sources, query_source, query_tiles}, ...]
    max_val_k = args.k_shot  # 仅需当前 K 个 support source | Only need K support sources for current run
    # 之前 max(args.k_shot, 10) 导致 K=1 时要求 ≥11 个源图才能建验证 episode,
    # 大量 val split 中源图少的类被丢弃, n_classes 从 15 降到 11-12.
    # Old max(args.k_shot, 10) required ≥11 source images per class for val episodes,
    # dropping many classes with few sources in val split (n_classes dropped from 15 to 11-12).
    for cls_id, src_to_tiles in sorted(val_index.items()):
        sources = list(src_to_tiles.keys())
        if len(sources) < 2:
            continue
        # 每类最多预采样 per_class 个 query source | Pre-sample at most per_class query sources
        n_val_eps = min(args.val_episodes, len(sources) - 1)

        # Step 1: 固定 query source (独立 RNG) | Fix query sources with independent RNG
        query_sources_fixed = val_query_rng.sample(sources, min(n_val_eps, len(sources)))

        # Step 2: 为每个 query 采样 K 个 support (主 RNG) | Sample K supports per query (main RNG)
        eps_for_class = []
        for q_src in query_sources_fixed:
            support_candidates = [s for s in sources if s != q_src]
            if len(support_candidates) < max_val_k:
                continue
            sampled_supports = val_rng.sample(support_candidates, max_val_k)
            eps_for_class.append({
                "query_source": q_src,
                "support_sources": sampled_supports,  # 前 K 个用于当前 K-shot | First K used for current K
                "query_tiles": sorted(src_to_tiles[q_src]),
            })
        if eps_for_class:
            fixed_val_episodes[cls_id] = eps_for_class
            name = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")
            print(f"  [VAL-FIXED] Class {cls_id:>2d} ({name:<18s}): "
                  f"{len(eps_for_class)} fixed val episodes")

    # 构建扁平列表用于快速采样 | Build flat list for fast sampling
    _flat_val_eps = [
        (cls_id, ep)
        for cls_id, eps_list in fixed_val_episodes.items()
        for ep in eps_list
    ]
    # 使用独立 RNG shuffle 确保跨 epoch 一致性 | Shuffle with dedicated RNG for consistency
    val_sample_rng = random.Random(args.seed + 99999)
    val_sample_rng.shuffle(_flat_val_eps)
    print(f"  [VAL-FIXED] Total fixed val pool: {len(_flat_val_eps)} episodes across "
          f"{len(fixed_val_episodes)} classes")

    # 保存固定验证集到 JSON | Save fixed validation episodes to JSON
    val_eps_json = {
        "description": "Fixed validation episodes for reproducibility",
        "seed": args.seed,
        "k_shot": args.k_shot,
        "max_val_k": max_val_k,
        "data_root": str(data_root),
        "data_format": data_format,
        "train_split": train_split,
        "val_split": val_split,
        "protocol": {
            "shot_definition": "K = number of source images (not tiles)",
            "support": "all tiles from K source images",
            "query": "full source image → all tiles → merge → full-image IoU",
            "scene_overlap": "0% (support ∩ query sources = ∅)",
        },
        "episodes": {
            str(cls_id): [
                {
                    "query_source": ep["query_source"],
                    "support_sources": ep["support_sources"],
                    "query_tiles": ep["query_tiles"],
                    "n_query_tiles": len(ep["query_tiles"]),
                }
                for ep in eps_list
            ]
            for cls_id, eps_list in fixed_val_episodes.items()
        },
    }
    val_eps_path = out_dir / "fixed_val_episodes.json"
    with open(val_eps_path, "w", encoding="utf-8") as f:
        json.dump(val_eps_json, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] Fixed validation episodes → {val_eps_path}")
    print(f"  [NOTE]  Same query sources for all K values — cross-K comparison is fair.")

    best_val_iou = 0.0
    log_entries = []

    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        epoch_losses = {"loss": [], "iou": [], "dice": [], "bce": []}
        if spm is not None:
            epoch_losses["spm_mean"] = []
        if decoder_type == "dynamic_kernel":
            for _k in ("n_matched", "n_unmatched_pred", "n_unmatched_gt", "n_gt",
                       "kernel_dice", "kernel_cos_sim"):
                epoch_losses[_k] = []
        if decoder_type == "center_affinity":
            for _k in ("center_loss", "offset_loss"):
                epoch_losses[_k] = []
        if getattr(decoder, "collect_stats", False):  # 前向统计键 (须与 result 键一致) | forward-stat keys
            for _k in ("proto_basis_l2", "coeff_l2", "pre_sigmoid_absmax", "sat_frac", "coeff_grad_norm"):
                epoch_losses[_k] = []

        pbar = tqdm(range(args.episodes_per_epoch), desc=f"Epoch {epoch + 1}/{args.epochs}",
                    unit="ep")
        for ep_idx in pbar:
            # Sample a class (uniform)
            cls_id = random.choice(list(train_index.keys()))
            src_to_tiles = train_index[cls_id]  # {source_img: [tile_stems]}
            sources = list(src_to_tiles.keys())
            if len(sources) < args.k_shot + 1:
                continue  # 该类的源图像不够 | Not enough source images for this class

            # Sample K+1 different source images
            # Support = K 张源图的所有 tile (最多 max_support_tiles_per_source 个) | Support = tiles from K source images
            # Query   = 1 张 tile (来自第 K+1 张源图) | Query = 1 tile from (K+1)th source
            sampled_sources = random.sample(sources, args.k_shot + 1)
            support_stems = []
            for s in sampled_sources[:args.k_shot]:
                support_stems.extend(src_to_tiles[s])  # ALL tiles per source
            query_stem = random.choice(src_to_tiles[sampled_sources[args.k_shot]])

            # ── 诊断: 前 3 个 episode 打印采样详情 | Diagnose: print first 3 episodes ──
            if epoch == 0 and ep_idx < 3:
                name = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")
                print(f"\n  [DIAG] Episode {ep_idx}: class={name} (id={cls_id})")
                for si, s in enumerate(sampled_sources[:args.k_shot]):
                    tiles = src_to_tiles[s]
                    print(f"    Support src {si+1}: {s} → {len(tiles)} tiles: {tiles[:3]}{'...' if len(tiles)>3 else ''}")
                q_src = sampled_sources[args.k_shot]
                print(f"    Query   src:   {q_src} → tile: {query_stem}")
                src_set = set(sampled_sources[:args.k_shot])
                print(f"    Overlap check: query_src in support_srcs = {q_src in src_set} (should be False)")

            try:
                result = train_episode(model, decoder, optimizer, cls_id,
                                       support_stems, query_stem,
                                       train_split, device, data_format, data_root,
                                       decoder_type=decoder_type, spm=spm,
                                       backbone_trainable=backbone_trainable,
                                       proto_source=args.prototype_source)
                for k in epoch_losses:
                    epoch_losses[k].append(result[k])
                pbar.set_postfix(loss=f"{result['loss']:.4f}", iou=f"{result['iou']:.4f}")
            except Exception as e:
                pbar.set_postfix(err=str(e)[:30])
                continue

        # Epoch summary
        avg_loss = np.mean(epoch_losses["loss"]) if epoch_losses["loss"] else 0
        avg_iou = np.mean(epoch_losses["iou"]) if epoch_losses["iou"] else 0
        # 前向统计逐 epoch 均值 (formation 曲线) | per-epoch forward-stat means
        fwd_means = {}
        if getattr(decoder, "collect_stats", False):
            for _k in ("proto_basis_l2", "coeff_l2", "pre_sigmoid_absmax", "sat_frac", "coeff_grad_norm"):
                _v = epoch_losses.get(_k, [])
                fwd_means[_k] = float(np.mean(_v)) if _v else 0.0
        fwd_str = (f" | basis_l2={fwd_means['proto_basis_l2']:.2e} sat={fwd_means['sat_frac']:.3f} "
                   f"cgrad={fwd_means['coeff_grad_norm']:.2e}") if fwd_means else ""
        if spm is not None:
            avg_spm_mean = np.mean(epoch_losses.get("spm_mean", [])) if epoch_losses.get("spm_mean") else 0
            spm_str = f", spm_mean={avg_spm_mean:.3f}"
        else:
            spm_str = ""
        # Dynamic kernel: show matching stats + kernel semantic Dice
        dk_str = ""
        if decoder_type == "dynamic_kernel":
            matched_vals = epoch_losses.get("n_matched", [])
            gt_vals = epoch_losses.get("n_gt", [])
            kdice_vals = [v for v in epoch_losses.get("kernel_dice", []) if v is not None]
            parts = []
            if kdice_vals:
                parts.append(f"kdice={np.mean(kdice_vals):.3f}")
            if matched_vals and gt_vals:
                parts.append(f"match={np.mean(matched_vals):.1f}/{np.mean(gt_vals):.1f}")
            cos_vals = [v for v in epoch_losses.get("kernel_cos_sim", []) if v is not None and v > 0]
            if cos_vals:
                parts.append(f"cossim={np.mean(cos_vals):.3f}")
            if parts:
                dk_str = ", " + ", ".join(parts)
        if decoder_type == "center_affinity":
            c_losses = [v for v in epoch_losses.get("center_loss", []) if v is not None]
            o_losses = [v for v in epoch_losses.get("offset_loss", []) if v is not None]
            parts = []
            if c_losses:
                parts.append(f"closs={np.mean(c_losses):.3f}")
            if o_losses:
                parts.append(f"oloss={np.mean(o_losses):.3f}")
            if parts:
                dk_str = ", " + ", ".join(parts)
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, iou={avg_iou:.4f}, "
              f"lr={scheduler.get_last_lr()[0]:.2e}{spm_str}{dk_str}{fwd_str}")

        scheduler.step()

        # ── 验证: 使用固定预采样 episodes (可复现) | Validation: fixed pre-sampled episodes ──
        # SCACS 协议: per-class IoU → 15 类平均 → 真正 mIoU (非 episode 平均)
        # SCACS protocol: per-class IoU → 15-class mean → true mIoU (not episode average)
        decoder.eval()
        is_pure_decoder = decoder_type in ("pure", "pure-p3p4")
        val_class_inter = defaultdict(float)   # cls_id → intersection sum (adaptive decoder only)
        val_class_union = defaultdict(float)   # cls_id → union sum (adaptive decoder only)
        val_ious_flat = []                     # flat IoU list (pure decoder: no class conditioning)
        # 从固定池中取前 N 个 (循环偏移) | Take first N from fixed pool (cyclic offset)
        val_offset = (epoch * args.val_episodes) % max(len(_flat_val_eps), 1)
        val_batch = (_flat_val_eps[val_offset:val_offset + args.val_episodes] +
                     _flat_val_eps[:max(0, val_offset + args.val_episodes - len(_flat_val_eps))])
        with torch.no_grad():
            for cls_id, ep in val_batch[:args.val_episodes]:
                src_to_tiles = val_index[cls_id]  # {source_img: [tile_stems]}
                q_src = ep["query_source"]
                # 取前 K 个 support source (最多 max_support_tiles_per_source 个 tile) | Use first K support sources
                support_srcs = ep["support_sources"][:args.k_shot]
                val_support_stems = []
                for s in support_srcs:
                    val_support_stems.extend(src_to_tiles[s])  # ALL tiles per source
                try:
                    # ── Support: 所有 tile → prototype ──
                    support_imgs = []; support_bmasks_v = []
                    for s in val_support_stems:
                        if is_instance_tile:
                            simg, smask = load_instance_tile_and_mask(s, val_split, data_root, target_class_id=cls_id)
                        elif is_tile:
                            simg, smask = load_tile_and_mask(s, val_split, data_root, target_class_id=cls_id)
                        else:
                            simg, smask = load_image_and_mask(s, val_split, data_root)
                        support_imgs.append(simg)
                        support_bmasks_v.append(semantic_mask_to_binary(smask, is_tile=is_tile))
                    support_feats = extract_features(model, support_imgs, device)
                    support_proto = compute_support_prototype(support_feats,
                                                              source=args.prototype_source)

                    if decoder_type in ("adaptive", "adaptive-p3p4", "pure", "pure-p3p4",
                                         "dynamic_kernel", "center_affinity"):
                        # No template needed
                        support_tmpl = None
                    else:
                        support_tmpl = compute_support_mask_template(support_bmasks_v).to(device)

                    # ── Query: 整张源图 → 所有 tile → 预测 → 合并 → 全图 IoU ──
                    if is_instance_tile and q_src in src_to_tiles:
                        # Pure decoder: 无 class conditioning → GT=全部实例
                        # Pure decoder: no class conditioning → GT=all instances
                        # Adaptive decoder: 有 prototype conditioning → GT=仅 target class
                        # Adaptive decoder: has prototype conditioning → GT=target class only
                        _val_gt_cls = None if is_pure_decoder else cls_id
                        full_gt, H_full, W_full, tile_data = build_full_image_gt(
                            q_src, src_to_tiles[q_src], val_split, data_root, target_class_id=_val_gt_cls)
                        if H_full <= 1 and W_full <= 1:
                            continue

                        predictions = []
                        for td in tile_data:
                            q_feats = extract_features(model, [td["img"]], device)[0]
                            if decoder_type == "adaptive":
                                mask_s4 = decoder(q_feats["p4"], q_feats["proto"], support_proto)
                                mask_s4 = _normalize_mask_to_4d(mask_s4)
                                mask_tile = F.interpolate(
                                    mask_s4, size=(td["h"], td["w"]),
                                    mode="bilinear", align_corners=False
                                ).squeeze()
                                pred_bin_np = (mask_tile > 0.5).float().cpu().numpy()
                            elif decoder_type == "adaptive-p3p4":
                                mask_s4 = decoder(q_feats["p3"], q_feats["p4"],
                                                  q_feats["proto"], support_proto)
                                mask_s4 = _normalize_mask_to_4d(mask_s4)
                                mask_tile = F.interpolate(
                                    mask_s4, size=(td["h"], td["w"]),
                                    mode="bilinear", align_corners=False
                                ).squeeze()
                                pred_bin_np = (mask_tile > 0.5).float().cpu().numpy()
                            elif decoder_type == "pure":
                                mask_s4 = decoder(q_feats["p4"])
                                mask_s4 = _normalize_mask_to_4d(mask_s4)
                                mask_tile = F.interpolate(
                                    mask_s4, size=(td["h"], td["w"]),
                                    mode="bilinear", align_corners=False
                                ).squeeze()
                                pred_bin_np = (mask_tile > 0.5).float().cpu().numpy()
                            elif decoder_type == "pure-p3p4":
                                mask_s4 = decoder(q_feats["p3"], q_feats["p4"])
                                mask_s4 = _normalize_mask_to_4d(mask_s4)
                                mask_tile = F.interpolate(
                                    mask_s4, size=(td["h"], td["w"]),
                                    mode="bilinear", align_corners=False
                                ).squeeze()
                                pred_bin_np = (mask_tile > 0.5).float().cpu().numpy()
                            elif decoder_type == "dynamic_kernel":
                                # Use proto_mask output (semantic) for validation IoU
                                _, proto_mask = decoder(
                                    q_feats["p3"], q_feats["p4"],
                                    q_feats["proto"], support_proto,
                                )
                                mask_s4 = _normalize_mask_to_4d(proto_mask)
                                mask_tile = F.interpolate(
                                    mask_s4, size=(td["h"], td["w"]),
                                    mode="bilinear", align_corners=False
                                ).squeeze()
                                pred_bin_np = (mask_tile > 0.5).float().cpu().numpy()
                            elif decoder_type == "center_affinity":
                                # Use proto_mask output (semantic) for validation IoU
                                _, _, proto_mask = decoder(
                                    q_feats["p3"], q_feats["p4"],
                                    q_feats["proto"], support_proto,
                                )
                                mask_s4 = _normalize_mask_to_4d(proto_mask)
                                mask_tile = F.interpolate(
                                    mask_s4, size=(td["h"], td["w"]),
                                    mode="bilinear", align_corners=False
                                ).squeeze()
                                pred_bin_np = (mask_tile > 0.5).float().cpu().numpy()
                            else:
                                pred = decoder(q_feats["p4"], q_feats["proto"],
                                              support_proto, support_tmpl)
                                pred = F.interpolate(pred, size=(td["h"], td["w"]),
                                                     mode="bilinear", align_corners=False)
                                pred_bin_np = (torch.sigmoid(pred) > 0.5).float().squeeze().cpu().numpy()
                            predictions.append({
                                "orig_x": td["orig_x"], "orig_y": td["orig_y"],
                                "h": td["h"], "w": td["w"], "pred_bin": pred_bin_np,
                            })

                        full_pred = merge_tile_predictions(predictions, H_full, W_full)
                        full_pred_bin = (full_pred > 0.5).astype(np.float32)
                        gt_float = full_gt.astype(np.float32)
                        inter = float((full_pred_bin * gt_float).sum())
                        union = float((full_pred_bin + gt_float).clip(0, 1).sum())
                        iou_val = inter / max(union, 1)
                        # SCACS: adaptive→per-class 累积, pure→flat list (无 class conditioning)
                        # SCACS: adaptive→per-class accumulation, pure→flat list (no class conditioning)
                        if is_pure_decoder:
                            val_ious_flat.append(iou_val)
                        else:
                            val_class_inter[cls_id] += inter
                            val_class_union[cls_id] += union
                    else:
                        # 非 instance 格式 — 降级为单 tile query | Non-instance: fallback to single tile
                        val_query_stem = random.choice(src_to_tiles[q_src])
                        # Pure decoder: 无 class conditioning → GT=全部实例
                        # Pure decoder: no class conditioning → GT=all instances
                        _val_gt_cls = None if is_pure_decoder else cls_id
                        if is_instance_tile:
                            q_img, q_mask = load_instance_tile_and_mask(val_query_stem, val_split, data_root, target_class_id=_val_gt_cls)
                        elif is_tile:
                            q_img, q_mask = load_tile_and_mask(val_query_stem, val_split, data_root, target_class_id=_val_gt_cls)
                        else:
                            q_img, q_mask = load_image_and_mask(val_query_stem, val_split, data_root)
                        q_feats = extract_features(model, [q_img], device)[0]
                        q_gt = semantic_mask_to_binary(q_mask, is_tile=is_tile)
                        if decoder_type == "adaptive":
                            mask_s4 = decoder(q_feats["p4"], q_feats["proto"], support_proto)
                            mask_s4 = _normalize_mask_to_4d(mask_s4)
                            H_gt, W_gt = q_gt.shape
                            mask_tile = F.interpolate(
                                mask_s4, size=(H_gt, W_gt),
                                mode="bilinear", align_corners=False
                            ).squeeze()
                            gt_t = torch.from_numpy(q_gt).float().to(device)
                            pred_bin = (mask_tile > 0.5).float()
                            inter = float((pred_bin * gt_t).sum().item())
                            union = float((pred_bin + gt_t).clamp(0, 1).sum().item())
                            iou_val = inter / max(union, 1)
                        elif decoder_type == "adaptive-p3p4":
                            mask_s4 = decoder(q_feats["p3"], q_feats["p4"],
                                              q_feats["proto"], support_proto)
                            mask_s4 = _normalize_mask_to_4d(mask_s4)
                            H_gt, W_gt = q_gt.shape
                            mask_tile = F.interpolate(
                                mask_s4, size=(H_gt, W_gt),
                                mode="bilinear", align_corners=False
                            ).squeeze()
                            gt_t = torch.from_numpy(q_gt).float().to(device)
                            pred_bin = (mask_tile > 0.5).float()
                            inter = float((pred_bin * gt_t).sum().item())
                            union = float((pred_bin + gt_t).clamp(0, 1).sum().item())
                            iou_val = inter / max(union, 1)
                        elif decoder_type == "pure":
                            mask_s4 = decoder(q_feats["p4"])
                            mask_s4 = _normalize_mask_to_4d(mask_s4)
                            H_gt, W_gt = q_gt.shape
                            mask_tile = F.interpolate(
                                mask_s4, size=(H_gt, W_gt),
                                mode="bilinear", align_corners=False
                            ).squeeze()
                            gt_t = torch.from_numpy(q_gt).float().to(device)
                            pred_bin = (mask_tile > 0.5).float()
                            inter = float((pred_bin * gt_t).sum().item())
                            union = float((pred_bin + gt_t).clamp(0, 1).sum().item())
                            iou_val = inter / max(union, 1)
                        elif decoder_type == "pure-p3p4":
                            mask_s4 = decoder(q_feats["p3"], q_feats["p4"])
                            mask_s4 = _normalize_mask_to_4d(mask_s4)
                            H_gt, W_gt = q_gt.shape
                            mask_tile = F.interpolate(
                                mask_s4, size=(H_gt, W_gt),
                                mode="bilinear", align_corners=False
                            ).squeeze()
                            gt_t = torch.from_numpy(q_gt).float().to(device)
                            pred_bin = (mask_tile > 0.5).float()
                            inter = float((pred_bin * gt_t).sum().item())
                            union = float((pred_bin + gt_t).clamp(0, 1).sum().item())
                            iou_val = inter / max(union, 1)
                        elif decoder_type == "dynamic_kernel":
                            # Use proto_mask for semantic validation IoU
                            _, proto_mask = decoder(
                                q_feats["p3"], q_feats["p4"],
                                q_feats["proto"], support_proto,
                            )
                            mask_s4 = _normalize_mask_to_4d(proto_mask)
                            H_gt, W_gt = q_gt.shape
                            mask_tile = F.interpolate(
                                mask_s4, size=(H_gt, W_gt),
                                mode="bilinear", align_corners=False
                            ).squeeze()
                            gt_t = torch.from_numpy(q_gt).float().to(device)
                            pred_bin = (mask_tile > 0.5).float()
                            inter = float((pred_bin * gt_t).sum().item())
                            union = float((pred_bin + gt_t).clamp(0, 1).sum().item())
                            iou_val = inter / max(union, 1)
                        elif decoder_type == "center_affinity":
                            # Use proto_mask for semantic validation IoU
                            _, _, proto_mask = decoder(
                                q_feats["p3"], q_feats["p4"],
                                q_feats["proto"], support_proto,
                            )
                            mask_s4 = _normalize_mask_to_4d(proto_mask)
                            H_gt, W_gt = q_gt.shape
                            mask_tile = F.interpolate(
                                mask_s4, size=(H_gt, W_gt),
                                mode="bilinear", align_corners=False
                            ).squeeze()
                            gt_t = torch.from_numpy(q_gt).float().to(device)
                            pred_bin = (mask_tile > 0.5).float()
                            inter = float((pred_bin * gt_t).sum().item())
                            union = float((pred_bin + gt_t).clamp(0, 1).sum().item())
                            iou_val = inter / max(union, 1)
                        else:
                            pred = decoder(q_feats["p4"], q_feats["proto"], support_proto, support_tmpl)
                            H_gt, W_gt = q_gt.shape
                            pred = F.interpolate(pred, size=(H_gt, W_gt),
                                                 mode="bilinear", align_corners=False)
                            gt_t = torch.from_numpy(q_gt).unsqueeze(0).unsqueeze(0).float().to(device)
                            pred_bin = (torch.sigmoid(pred) > 0.5).float()
                            inter = float((pred_bin * gt_t).sum().item())
                            union = float((pred_bin + gt_t).clamp(0, 1).sum().item())
                            iou_val = inter / max(union, 1)
                        # SCACS: adaptive→per-class 累积, pure→flat list
                        # SCACS: adaptive→per-class accumulation, pure→flat list
                        if is_pure_decoder:
                            val_ious_flat.append(iou_val)
                        else:
                            val_class_inter[cls_id] += inter
                            val_class_union[cls_id] += union
                except Exception:
                    continue

        # ── SCACS: per-class IoU → 15 类平均 → 真正 mIoU ──
        # SCACS protocol: per-class IoU → 15-class mean → true mIoU
        # Pure decoder: 无 class conditioning → flat IoU 平均 (非 per-class)
        # Pure decoder: no class conditioning → flat IoU mean (not per-class)
        if is_pure_decoder:
            avg_val_iou = float(np.mean(val_ious_flat)) if val_ious_flat else 0.0
            n_eps_used = len(val_ious_flat)
            per_class_iou = {}
            print(f"  Val IoU: {avg_val_iou:.4f} (best={best_val_iou:.4f}, "
                  f"n_eps={n_eps_used}, no per-class breakdown for pure decoder)")
            log_entries.append({
                "epoch": epoch + 1, "train_loss": float(avg_loss),
                "train_iou": float(avg_iou), "val_iou": float(avg_val_iou),
                "val_n_eps": n_eps_used, "decoder_type": decoder_type,
                **fwd_means,
            })
        else:
            per_class_iou = {}
            for cls_id in sorted(set(list(val_class_inter.keys()) + list(val_class_union.keys()))):
                u = val_class_union[cls_id]
                if u > 0:
                    per_class_iou[cls_id] = val_class_inter[cls_id] / u
                else:
                    per_class_iou[cls_id] = 0.0

            valid_ious = [v for v in per_class_iou.values() if v > 0]
            avg_val_iou = np.mean(valid_ious) if valid_ious else 0.0
            n_classes_eval = len(valid_ious)
            n_eps_used = sum(1 for u in val_class_union.values() if u > 0)

            # 打印 per-class IoU 分解 | Print per-class IoU breakdown
            cls_short_str = ", ".join(
                f"{CATEGORY_NAMES.get(c, f'c{c}')}={per_class_iou[c]:.3f}"
                for c in sorted(per_class_iou.keys())
            )
            print(f"  Val mIoU: {avg_val_iou:.4f} (best={best_val_iou:.4f}, "
                  f"n_classes={n_classes_eval}/{len(per_class_iou)}, n_eps={n_eps_used})")
            if cls_short_str:
                print(f"    Per-class: {cls_short_str}")

            log_entries.append({
                "epoch": epoch + 1, "train_loss": float(avg_loss),
                "train_iou": float(avg_iou), "val_miou": float(avg_val_iou),
                "val_per_class_iou": {str(k): round(v, 4) for k, v in per_class_iou.items()},
                "val_n_classes": n_classes_eval,
                **fwd_means,
            })

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1, "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(), "val_iou": float(avg_val_iou),
            "k_shot": args.k_shot, "decoder_type": decoder_type,
            "unfreeze_layers": args.unfreeze_layers,
            "normalize_proto": args.normalize_proto,
        }
        if decoder_type == "dynamic_kernel":
            ckpt["n_kernels"] = args.n_kernels
        if backbone_trainable:
            # 保存解冻的 backbone 层权重 | Save unfrozen backbone layer weights
            seq = model.model.model
            n_total = len(seq)
            start = max(0, n_total - args.unfreeze_layers)
            ckpt["backbone"] = {str(i): seq[i].state_dict() for i in range(start, n_total)}
        if spm is not None:
            ckpt["spm"] = spm.state_dict()
        torch.save(ckpt, out_dir / "last_model.pt")

        if avg_val_iou > best_val_iou:
            best_val_iou = avg_val_iou
            torch.save(ckpt, out_dir / "best_model.pt")
            print(f"  [SAVED] best_model.pt (val_mIoU={best_val_iou:.4f})")

    # ── Save log | 保存日志 ──
    with open(out_dir / "train_log.json", "w", encoding="utf-8") as f:
        json.dump({"k_shot": args.k_shot, "best_val_miou": best_val_iou,
                   "entries": log_entries}, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  Training complete!")
    print(f"  Best val mIoU: {best_val_iou:.4f}")
    print(f"  Output: {out_dir}")
    print(f"  To evaluate: python tools/eval/eval_fastsam_prompted.py "
          f"--all-classes --per-class 5 --mode bbox --diagnose --device cuda")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
