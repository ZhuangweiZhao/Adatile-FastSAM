#!/usr/bin/env python3
"""
FastSAM Tile 可视化 — 全图推理 vs 切片拼接对比
FastSAM Tile Visualization — Full Image vs Tile Stitching Comparison.
=====================================================================

在 iSAID 遥感图像上对比两种推理策略：
Compare two inference strategies on iSAID aerial images:

1. **全图直接推理** (Full Image): resize → 1024² → FastSAM → 输出 mask
2. **切片推理+拼接** (Tile Stitching): 896² tiles → FastSAM per tile → 坐标映射拼接

目的 | Goal: 直观验证切片策略对航拍图像分割质量的提升。
        Visually verify tile strategy improves aerial image segmentation.

用法 | Usage::

    # 默认 (P0161, iSAID_tiles 格式)
    python tools/viz/viz_fastsam_tiles.py

    # 新 tile 格式 (iSAID-few_tiles, COCO JSON)
    python tools/viz/viz_fastsam_tiles.py --source P0161 \
        --data-root data/iSAID-few_tiles --full-image-dir data/iSAID-few/val/images

    # 指定图像 + 仅切片模式
    python tools/viz/viz_fastsam_tiles.py --source P0161 --mode tiles --device cuda

输出 | Output:
    runs/viz_fastsam_tiles_{src}_{timestamp}/
    ├── full_image_result.png      # 全图推理结果
    ├── tile_stitched_result.png   # 切片拼接结果
    ├── comparison.png             # 并排对比
    └── stats.json                 # 统计信息
"""

from __future__ import annotations

import sys, argparse, json, os
from pathlib import Path
from datetime import datetime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM"))

# Windows GBK 编码修复 | Windows GBK encoding fix
import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch

from adatile.utils.seed import set_seed

# ═══════════════════════════════════════════════════════════════════
# 常量 | Constants
# ═══════════════════════════════════════════════════════════════════

DEFAULT_SOURCE = "P0161"
FULL_IMAGE_DIR = "data/iSAID_processed/val/images"

# ── 数据源配置 | Data source configs ──
DATA_SOURCE_CONFIGS = {
    "iSAID_instance_fewshot": {
        "tile_dir": "data/iSAID_instance_fewshot/images/val",
        "anno_path": "data/iSAID_instance_fewshot/annotations/instances_val.json",
        "metadata_path": None,  # COCO JSON 本身含位置信息 | COCO JSON contains position info
        "tile_size": 896,
        "tile_stride": 512,
        "gt_type": "coco_instance",  # COCO 实例分割 | COCO instance segmentation
        "masks_full_dir": None,
    },
    "iSAID_tiles": {
        "tile_dir": "data/iSAID_tiles/images/val",
        "anno_path": None,  # 无 COCO JSON | No COCO JSON
        "metadata_path": "data/iSAID_tiles/metadata/val.json",
        "tile_size": 1024,
        "tile_stride": 1024,  # grid 无重叠 | Grid-based, no overlap
        "gt_type": "semantic_mask",  # 语义分割 mask | Semantic segmentation mask
        "masks_full_dir": "data/iSAID_tiles/masks_full",
    },
}
CATEGORY_NAMES = {
    1: "small_vehicle", 2: "large_vehicle", 3: "plane",
    4: "storage_tank", 5: "ship", 6: "harbor",
    7: "ground_track_field", 8: "soccer_ball_field", 9: "tennis_court",
    10: "swimming_pool", 11: "baseball_diamond", 12: "basketball_court",
    13: "bridge", 14: "helicopter", 15: "roundabout",
}

# 调色板 (cityscapes 风格, 用于可视化) | Color palette (cityscapes-style, for vis)
PALETTE = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
    (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
    (255, 0, 0), (0, 0, 142), (0, 0, 70),
]

# ═══════════════════════════════════════════════════════════════════
# 数据加载 | Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_tile_info(source: str, data_source: str = "iSAID_instance_fewshot") -> tuple[list[dict], dict]:
    """
    从指定数据源加载 tile 信息.
    Load tile info from specified data source.

    支持两种数据源 | Supports two data sources:
        - iSAID_instance_fewshot: COCO JSON with orig_x/orig_y per tile (896², 512 stride)
        - iSAID_tiles: metadata JSON with tile_idx (1024², grid-based, no overlap)

    :param source: 源图像名称 (如 P0161) | Source image name (e.g. P0161).
    :param data_source: "iSAID_instance_fewshot" or "iSAID_tiles".
    :return: (tiles_list, config_dict) — tiles 按位置排序, config 含 tile_size/stride.
        tiles sorted by position, config contains tile_size/stride.
    """
    cfg = DATA_SOURCE_CONFIGS[data_source]

    if data_source == "iSAID_tiles":
        return _load_tile_info_from_metadata(source, cfg), cfg
    else:
        return _load_tile_info_from_coco(source, cfg), cfg


def _load_tile_info_from_coco(source: str, cfg: dict) -> list[dict]:
    """从 COCO JSON 加载 tile 信息 (896², 含 orig_x/orig_y)."""
    with open(_PROJECT_ROOT / cfg["anno_path"]) as f:
        data = json.load(f)

    tiles = []
    tile_dir = _PROJECT_ROOT / cfg["tile_dir"]
    for img_info in data["images"]:
        name = img_info["file_name"]
        if name.startswith(f"{source}_t"):
            tile_path = tile_dir / name
            if tile_path.exists():
                tiles.append({
                    "name": name,
                    "orig_x": img_info["orig_x"],
                    "orig_y": img_info["orig_y"],
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "tile_path": str(tile_path),
                })

    tiles.sort(key=lambda t: (t["orig_y"], t["orig_x"]))
    return tiles


def _load_tile_info_from_metadata(source: str, cfg: dict) -> list[dict]:
    """
    从 metadata JSON 加载 tile 信息 (1024², grid 排列).
    Load tile info from metadata JSON (1024², grid layout).

    旧版 tile 命名: P0161_t000.png, 不含位置信息.
    根据 tile_idx 和全图尺寸反算 orig_x/orig_y (row-major grid).
    Old tile naming: P0161_t000.png, no position info in name.
    Derive orig_x/orig_y from tile_idx + full image size (row-major grid).
    """
    meta_path = _PROJECT_ROOT / cfg["metadata_path"]
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path) as f:
        metadata = json.load(f)

    # 获取全图尺寸 | Get full image size
    ts = cfg["tile_size"]
    full_img = load_full_image(source)
    if full_img is None:
        raise FileNotFoundError(f"Full image not found for {source}")
    H, W = full_img.shape[:2]

    # 计算 grid 布局 | Compute grid layout
    n_cols = (W + ts - 1) // ts  # ceil division
    # n_rows = (H + ts - 1) // ts

    tiles = []
    tile_dir = _PROJECT_ROOT / cfg["tile_dir"]
    for t in metadata:
        if t["img_id"] != source:
            continue
        tile_idx = t["tile_idx"]
        # Grid row-major: idx = row * n_cols + col
        row = tile_idx // n_cols
        col = tile_idx % n_cols
        orig_y = row * ts
        orig_x = col * ts

        name = t["tile_name"]
        tile_path = tile_dir / name
        if not tile_path.exists():
            continue
        # 边界 tile 可能小于 ts | Edge tiles may be smaller than ts
        th = min(ts, H - orig_y)
        tw = min(ts, W - orig_x)

        tiles.append({
            "name": name,
            "orig_x": orig_x,
            "orig_y": orig_y,
            "width": tw,
            "height": th,
            "tile_idx": tile_idx,
            "tile_path": str(tile_path),
        })

    tiles.sort(key=lambda t: (t["orig_y"], t["orig_x"]))
    return tiles


def load_full_image(source: str) -> np.ndarray | None:
    """加载全分辨率图像 | Load full-resolution image."""
    for split in ["val", "train"]:
        img_path = _PROJECT_ROOT / FULL_IMAGE_DIR.replace("val", split) / f"{source}.png"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def pad_to_multiple(img: np.ndarray, multiple: int = 32) -> np.ndarray:
    """
    填充图像到 multiple 的倍数 (FastSAM 要求).
    Pad image to multiple of `multiple` (FastSAM requirement).

    :param img: [H, W, C] RGB image.
    :param multiple: divisor (default 32 for FastSAM).
    :return: padded image.
    """
    H, W = img.shape[:2]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img


# ═══════════════════════════════════════════════════════════════════
# FastSAM 推理 | FastSAM Inference
# ═══════════════════════════════════════════════════════════════════

def load_fastsam_model(device: str = "cuda"):
    """
    加载 FastSAM 模型 | Load FastSAM model.

    :param device: "cuda" or "cpu".
    :return: FastSAM model instance.
    """
    from fastsam import FastSAM

    model_path = _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"FastSAM weights not found: {model_path}")

    model = FastSAM(str(model_path))
    # FastSAM 内部会处理 device，这里手动设置
    if torch.cuda.is_available() and device == "cuda":
        model.model.cuda()
    else:
        model.model.cpu()
    model.model.eval()
    return model


def predict_full_image(model, image: np.ndarray, imgsz: int = 1024,
                       conf: float = 0.25, device: str = "cuda") -> dict:
    """
    全图 FastSAM 推理 | Full-image FastSAM inference.

    FastSAM 要求输入为 imgsz² 的正方形图像。将原图 resize→推理→mask 映射回原图分辨率。
    FastSAM requires square input at imgsz². Resize→inference→remap masks to original resolution.

    :param model: FastSAM model.
    :param image: [H, W, C] RGB image in [0, 255].
    :param imgsz: 推理尺寸 (正方形) | Inference size (square).
    :param conf: 置信度阈值 | Confidence threshold.
    :param device: 设备 | Device.
    :return: {
        "masks": [N, H, W] binary masks at original resolution,
        "scores": [N] confidence scores (or None),
        "image_size": (H_orig, W_orig),
    }
    """
    H_orig, W_orig = image.shape[:2]

    # FastSAM 要求方形输入: resize 原图到 imgsz²
    # FastSAM requires square input: resize original image to imgsz²
    img_resized = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    # FastSAM everything mode
    results = model(
        source=img_resized,
        device=device,
        retina_masks=True,
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        verbose=False,
    )

    if results is None or len(results) == 0 or results[0].masks is None:
        return {"label_map": np.zeros((H_orig, W_orig), dtype=np.int32),
                "count": 0, "scores": np.array([]), "image_size": (H_orig, W_orig)}

    result = results[0]
    masks_data = result.masks.data  # [N, imgsz, imgsz]

    # 获取置信度 scores | Get confidence scores
    try:
        scores = result.masks.conf.cpu().numpy() if hasattr(result.masks, 'conf') else None
    except Exception:
        scores = None
    if scores is None:
        scores = np.ones(len(masks_data))

    # Resize masks back to original resolution → label map (内存高效)
    # 将 mask 从 imgsz² 映射回 (H_orig, W_orig)，写入 label map 而非 [N, H, W] 数组
    # Map masks from imgsz² to (H_orig, W_orig), write to label map (memory efficient)
    N = len(masks_data)
    label_map = np.zeros((H_orig, W_orig), dtype=np.int32)

    # 按 score 降序 (高分先写入) | Sort by score desc (high score writes first)
    order = np.argsort(scores)[::-1] if scores is not None else range(N)

    for label_id, idx in enumerate(order, start=1):
        mask_tensor = masks_data[idx].unsqueeze(0).unsqueeze(0).float()  # [1, 1, imgsz, imgsz]
        mask_resized = torch.nn.functional.interpolate(
            mask_tensor, size=(H_orig, W_orig), mode="bilinear", align_corners=False,
        ).squeeze(0).squeeze(0)  # [H_orig, W_orig]
        mask_bin = (mask_resized > 0.5).cpu().numpy()
        # 只写入未被占用的区域 | Only write to unoccupied pixels
        label_map[(mask_bin) & (label_map == 0)] = label_id

    return {
        "label_map": label_map,
        "count": N,
        "scores": np.array(scores),
        "image_size": (H_orig, W_orig),
    }


def predict_tiles(model, tiles: list[dict], device: str = "cuda",
                  conf: float = 0.25) -> list[dict]:
    """
    逐 tile FastSAM 推理 | Per-tile FastSAM inference.

    每个 tile 独立推理，mask 坐标映射到全图坐标系。
    Each tile: independent inference, masks mapped to full-image coordinates.

    :param model: FastSAM model.
    :param tiles: tile info list from load_tile_info().
    :param device: 设备 | Device.
    :param conf: 置信度阈值 | Confidence threshold.
    :return: [{tile_name, orig_x, orig_y, masks, scores, mask_coords}, ...]
        mask_coords are in full-image coordinate system.
    """
    tile_results = []

    for tile_info in tqdm(tiles, desc="Tiles", unit="tile"):
        img = cv2.imread(tile_info["tile_path"])
        if img is None:
            tile_results.append({**tile_info, "masks": [], "scores": []})
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        H_t, W_t = img.shape[:2]
        # Round to multiple of 32 (avoids FastSAM warning flood)
        imgsz_tile = ((max(H_t, W_t) + 31) // 32) * 32

        results = model(
            source=img,
            device=device,
            retina_masks=True,
            imgsz=imgsz_tile,
            conf=conf,
            iou=0.7,
            verbose=False,
        )

        if results is None or len(results) == 0 or results[0].masks is None:
            tile_results.append({
                **tile_info,
                "masks": np.zeros((0, tile_info["height"], tile_info["width"]), dtype=bool),
                "scores": np.array([]),
            })
            continue

        result = results[0]
        H_tile, W_tile = tile_info["height"], tile_info["width"]
        masks_data = result.masks.data  # [N, H_tile, W_tile]

        try:
            scores = result.masks.conf.cpu().numpy() if hasattr(result.masks, 'conf') else None
        except Exception:
            scores = None
        if scores is None:
            scores = np.ones(len(masks_data))

        mask_list = []
        for i in range(len(masks_data)):
            m = masks_data[i].cpu().numpy()
            # Resize if necessary (FastSAM might change resolution slightly)
            if m.shape != (H_tile, W_tile):
                m_tensor = torch.tensor(m).unsqueeze(0).unsqueeze(0).float()
                m = torch.nn.functional.interpolate(
                    m_tensor, size=(H_tile, W_tile), mode="bilinear",
                ).squeeze().cpu().numpy()
            mask_list.append(m > 0.5)

        tile_results.append({
            **tile_info,
            "masks": np.array(mask_list) if mask_list else np.zeros((0, H_tile, W_tile), dtype=bool),
            "scores": np.array(scores),
        })

    return tile_results


def stitch_tile_masks(tile_results: list[dict], full_size: tuple[int, int]) -> dict:
    """
    将 tile masks 拼接为全图 mask (内存高效版) | Memory-efficient tile mask stitching.

    策略 | Strategy:
        - 按 score 排序 → 顺序写入 label map → 占用标记防重叠
        - 返回 label map [H, W] int32 + mask count (而非 [N, H, W] bool 数组)
        - Sort by score → sequential write to label map → occupancy guard.
        - Returns label map [H, W] int32 + count (instead of [N, H, W] bool array).

    :param tile_results: results from predict_tiles().
    :param full_size: (H, W) full image size.
    :return: {"label_map": [H, W] int32 (0=BG, 1..N=masks), "count": int, "scores": [N]}.
    """
    H, W = full_size

    # 收集所有 mask reference (不展开到全图) | Collect mask references (no expansion)
    mask_refs = []  # list of (score, tile_idx, mask_idx)

    for ti, tr in enumerate(tile_results):
        for mi in range(len(tr["masks"])):
            score = tr["scores"][mi] if mi < len(tr["scores"]) else 1.0
            mask_refs.append((score, ti, mi))

    if not mask_refs:
        return {"label_map": np.zeros((H, W), dtype=np.int32), "count": 0, "scores": []}

    # 按 score 降序排列 | Sort by score descending
    mask_refs.sort(key=lambda x: -x[0])

    # 顺序写入: 高分 mask 先占位，低分 mask 无法覆盖已占区域
    # Sequential write: high-score masks claim first, low-score can't cover claimed pixels
    occupied = np.zeros((H, W), dtype=bool)
    label_map = np.zeros((H, W), dtype=np.int32)
    mask_scores = []
    label_id = 0

    for score, ti, mi in mask_refs:
        tr = tile_results[ti]
        mask = tr["masks"][mi]
        ox, oy = tr["orig_x"], tr["orig_y"]
        th, tw = tr["height"], tr["width"]

        # 有效区域裁剪 | Clip to valid region
        y1, y2 = max(oy, 0), min(oy + th, H)
        x1, x2 = max(ox, 0), min(ox + tw, W)
        my1 = y1 - oy
        my2 = y2 - oy
        mx1 = x1 - ox
        mx2 = x2 - ox

        mask_patch = mask[my1:my2, mx1:mx2]

        # 排除已占用区域 | Exclude already-occupied pixels
        target_region = occupied[y1:y2, x1:x2]
        unique = mask_patch & (~target_region)

        if unique.sum() > 16:  # 最小面积过滤 | Min area filter
            label_id += 1
            label_map[y1:y2, x1:x2][unique] = label_id
            occupied[y1:y2, x1:x2] |= mask_patch
            mask_scores.append(score)

    return {"label_map": label_map, "count": label_id, "scores": mask_scores}


# ═══════════════════════════════════════════════════════════════════
# 可视化 | Visualization
# ═══════════════════════════════════════════════════════════════════

def draw_masks(image: np.ndarray, masks=None, scores=None,
               label_map: np.ndarray | None = None,
               uniform_color: tuple | None = None,
               alpha: float = 0.5, max_masks: int = 500) -> np.ndarray:
    """
    在图像上绘制 mask 轮廓 | Draw mask contours on image.

    支持两种输入格式 | Supports two input formats:
        - masks: [N, H, W] binary mask array, OR
        - label_map: [H, W] int32 label map (0=BG, 1..N=masks)

    当 uniform_color 指定时，所有 mask 使用相同颜色 (用于 class-agnostic 预测)。
    否则循环使用 PALETTE (用于 per-category 标注)。
    When uniform_color is set, all masks use the same color (for class-agnostic predictions).
    Otherwise cycles through PALETTE (for per-category annotations).

    :param image: [H, W, C] RGB image in [0, 255].
    :param masks: [N, H, W] binary masks (optional if label_map provided).
    :param uniform_color: (R,G,B) single color for all masks, or None for palette cycling.
    :param label_map: [H, W] int32 label map (optional if masks provided).
    :param alpha: 透明度 | Transparency.
    :param max_masks: 最大绘制数量 | Max masks to draw.
    :return: [H, W, C] annotated image.
    """
    vis = image.copy().astype(np.float32)
    overlay = np.zeros_like(vis)
    contour_thick = 1 if uniform_color else 2

    if label_map is not None:
        N = label_map.max()
        if N == 0:
            return vis.astype(np.uint8)
        N = min(N, max_masks)
        for i in range(1, N + 1):
            mask_bin = (label_map == i).astype(np.uint8) * 255
            if mask_bin.sum() == 0:
                continue
            if uniform_color:
                color = uniform_color
            else:
                color = PALETTE[(i - 1) % len(PALETTE)]
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, thickness=contour_thick)
            if uniform_color:
                for c in contours:
                    cv2.fillPoly(overlay, [c], color)
    else:
        if masks is None or len(masks) == 0:
            return vis.astype(np.uint8)
        N = min(len(masks), max_masks)
        for i in range(N):
            color = uniform_color if uniform_color else PALETTE[i % len(PALETTE)]
            mask = masks[i].astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, thickness=contour_thick)
            if uniform_color:
                for c in contours:
                    cv2.fillPoly(overlay, [c], color)

    vis = cv2.addWeighted(vis, 1.0, overlay, alpha, 0.0)
    return vis.astype(np.uint8)


def load_gt_label_map(tiles: list[dict], full_size: tuple[int, int],
                      data_source: str = "iSAID_instance_fewshot") -> dict:
    """
    加载 GT 标注并生成全图 label map | Load GT annotations, create full-image label map.

    支持两种数据格式 | Supports two data formats:
        - iSAID_instance_fewshot: COCO JSON 实例分割 → per-instance label_map
        - iSAID_tiles: semantic mask PNG → per-class label_map

    :param tiles: tile info list from load_tile_info().
    :param full_size: (H, W) full image size.
    :param data_source: "iSAID_instance_fewshot" or "iSAID_tiles".
    :return: {"label_map": [H,W] int32, "cat_ids": [cat_id per label] (index 0=BG),
              "count": N, "gt_type": "instance"|"semantic"}
    """
    if data_source == "iSAID_tiles":
        return _load_gt_semantic_mask(tiles, full_size)
    else:
        return _load_gt_coco_instance(tiles, full_size)


def _load_gt_semantic_mask(tiles: list[dict], full_size: tuple[int, int]) -> dict:
    """
    从语义分割 mask 加载 GT (iSAID_tiles 格式).
    Load GT from semantic segmentation mask (iSAID_tiles format).

    直接读取 masks_full/{img_id}_mask.png，每个像素值 = class ID.
    Directly reads masks_full/{img_id}_mask.png, each pixel = class ID.
    """
    H, W = full_size
    cfg = DATA_SOURCE_CONFIGS["iSAID_tiles"]
    masks_full_dir = _PROJECT_ROOT / cfg["masks_full_dir"]

    # 从第一个 tile 推导源图像 ID | Derive source image ID from first tile
    if not tiles:
        return {"label_map": np.zeros((H, W), dtype=np.int32), "cat_ids": [], "count": 0,
                "gt_type": "semantic"}

    # tile name like "P0161_t000.png" → img_id = "P0161"
    img_id = tiles[0]["name"].rsplit("_t", 1)[0]
    mask_path = masks_full_dir / f"{img_id}_mask.png"
    if not mask_path.exists():
        print(f"  [WARN] Semantic mask not found: {mask_path}")
        return {"label_map": np.zeros((H, W), dtype=np.int32), "cat_ids": [], "count": 0,
                "gt_type": "semantic"}

    mask = np.array(Image.open(str(mask_path)))  # [H, W] uint8, values = class IDs
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # 收集出现过的类别 | Collect present classes
    unique_classes = sorted(c for c in np.unique(mask) if c > 0)

    return {
        "label_map": mask.astype(np.int32),  # 直接用 class ID 作为 label (0=BG)
        "cat_ids": [0] + unique_classes,  # idx 0 placeholder, rest = class IDs
        "count": len(unique_classes),
        "gt_type": "semantic",
        # 语义分割特有字段 | Semantic-specific fields
        "class_ids": unique_classes,
    }


def _load_gt_coco_instance(tiles: list[dict], full_size: tuple[int, int]) -> dict:
    """
    从 COCO JSON 加载实例级 GT | Load instance-level GT from COCO JSON.

    对每个 tile 的 GT 实例，将其 segmentation 多边形渲染到全图坐标。
    For each tile's GT instances, render segmentation polygons to full-image coords.
    """
    cfg = DATA_SOURCE_CONFIGS["iSAID_instance_fewshot"]
    with open(_PROJECT_ROOT / cfg["anno_path"]) as f:
        data = json.load(f)

    H, W = full_size

    # 构建 tile_id → tile_info 映射 | Build tile_id → tile_info map
    tile_id_to_info = {}
    tile_names = {t["name"] for t in tiles}
    for img_info in data["images"]:
        if img_info["file_name"] in tile_names:
            tile_id_to_info[img_info["id"]] = {
                "orig_x": img_info["orig_x"],
                "orig_y": img_info["orig_y"],
                "width": img_info["width"],
                "height": img_info["height"],
            }

    # 收集所有 GT 实例 | Collect all GT instances
    gt_instances = []
    for ann in data["annotations"]:
        if ann["image_id"] not in tile_id_to_info:
            continue
        ti = tile_id_to_info[ann["image_id"]]
        ox, oy = ti["orig_x"], ti["orig_y"]

        segs = ann["segmentation"]
        if not segs:
            continue
        full_segs = []
        for seg in segs:
            poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
            poly[:, 0] += ox
            poly[:, 1] += oy
            poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
            poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
            full_segs.append(poly.astype(np.int32))
        if full_segs:
            gt_instances.append({
                "category_id": ann["category_id"],
                "polygons": full_segs,
                "area": ann.get("area", 0),
                "ann_id": ann["id"],
            })

    if not gt_instances:
        return {"label_map": np.zeros((H, W), dtype=np.int32), "cat_ids": [], "count": 0,
                "gt_type": "instance"}

    # 确定性排序: area desc, ann_id asc
    gt_instances.sort(key=lambda x: (-x["area"], x["ann_id"]))

    label_map = np.zeros((H, W), dtype=np.int32)
    cat_ids = [0]

    for i, inst in enumerate(gt_instances):
        label_id = i + 1
        cat_ids.append(inst["category_id"])
        for poly in inst["polygons"]:
            cv2.fillPoly(label_map, [poly], label_id)

    return {"label_map": label_map, "cat_ids": cat_ids, "count": len(gt_instances),
            "gt_type": "instance"}


def draw_gt_by_category(image: np.ndarray, gt_data: dict,
                        alpha: float = 0.5, max_items: int = 500) -> np.ndarray:
    """
    按类别颜色绘制 GT | Draw GT color-coded by category.

    支持两种 GT 类型 | Supports two GT types:
        - instance: 每个 label_id 是独立实例，通过 cat_ids[label_id] 确定颜色
        - semantic: 每个像素值 = class ID，直接着色

    :param image: [H, W, C] RGB image.
    :param gt_data: dict from load_gt_label_map().
    :param alpha: 透明度 | Transparency.
    :param max_items: 最大绘制数 | Max items to draw.
    :return: annotated image.
    """
    gt_type = gt_data.get("gt_type", "instance")

    if gt_type == "semantic":
        return _draw_gt_semantic(image, gt_data, alpha)
    else:
        return _draw_gt_instance(image, gt_data, alpha, max_items)


def _draw_gt_semantic(image: np.ndarray, gt_data: dict, alpha: float) -> np.ndarray:
    """
    绘制语义分割 GT: 每个 class ID 对应 PALETTE 颜色.
    Draw semantic segmentation GT: each class ID maps to PALETTE color.
    """
    label_map = gt_data["label_map"]  # [H,W] int32, pixel value = class ID
    class_ids = gt_data.get("class_ids", [c for c in np.unique(label_map) if c > 0])

    vis = image.copy().astype(np.float32)
    overlay = np.zeros_like(vis)

    for cls_id in class_ids:
        mask_bin = (label_map == cls_id).astype(np.uint8) * 255
        if mask_bin.sum() == 0:
            continue
        color = PALETTE[(cls_id - 1) % len(PALETTE)]
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=1)
        for c in contours:
            cv2.fillPoly(overlay, [c], color)

    vis = cv2.addWeighted(vis, 1.0, overlay, alpha, 0.0)
    vis_u8 = vis.astype(np.uint8)

    # 类别图例 | Category legend
    legend_x, legend_y = 10, 30
    for cls_id in sorted(class_ids):
        color = PALETTE[(cls_id - 1) % len(PALETTE)]
        name = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")
        cv2.rectangle(vis_u8, (legend_x, legend_y),
                      (legend_x + 15, legend_y + 12), color, -1)
        cv2.putText(vis_u8, f"{cls_id}:{name}",
                    (legend_x + 20, legend_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        legend_y += 16

    return vis_u8


def _draw_gt_instance(image: np.ndarray, gt_data: dict, alpha: float,
                      max_instances: int) -> np.ndarray:
    """
    绘制实例分割 GT: 每个实例按类别着色.
    Draw instance segmentation GT: each instance colored by category.
    """
    label_map = gt_data["label_map"]
    cat_ids = gt_data["cat_ids"]

    vis = image.copy().astype(np.float32)
    overlay = np.zeros_like(vis)

    N = min(label_map.max(), max_instances)
    for i in range(1, N + 1):
        mask_bin = (label_map == i).astype(np.uint8) * 255
        if mask_bin.sum() == 0:
            continue
        cat_id = cat_ids[i] if i < len(cat_ids) else 0
        color = PALETTE[(cat_id - 1) % len(PALETTE)] if cat_id > 0 else (180, 180, 180)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, thickness=1)
        for c in contours:
            cv2.fillPoly(overlay, [c], color)

    vis = cv2.addWeighted(vis, 1.0, overlay, alpha, 0.0)
    vis_u8 = vis.astype(np.uint8)

    # 类别图例 | Category legend
    legend_x, legend_y = 10, 30
    seen_cats = set()
    for i in range(1, N + 1):
        cat_id = cat_ids[i] if i < len(cat_ids) else 0
        if cat_id > 0 and cat_id not in seen_cats:
            seen_cats.add(cat_id)
            color = PALETTE[(cat_id - 1) % len(PALETTE)]
            name = CATEGORY_NAMES.get(cat_id, f"cls{cat_id}")
            cv2.rectangle(vis_u8, (legend_x, legend_y),
                          (legend_x + 15, legend_y + 12), color, -1)
            cv2.putText(vis_u8, f"{cat_id}:{name}",
                        (legend_x + 20, legend_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            legend_y += 16

    return vis_u8


def create_comparison_figure(full_img: np.ndarray,
                             full_label_map: np.ndarray,
                             full_count: int,
                             tile_label_map: np.ndarray,
                             tile_count: int,
                             tile_results: list[dict],
                             gt_data: dict,
                             output_dir: Path):
    """
    创建对比图 | Create comparison figure.
    4-panel: 原图 | 全图推理 | 切片拼接 | GT标注
    4-panel: Original | Full-Image | Tile-Stitched | Ground Truth

    策略: 先在全分辨率上绘制 mask，再缩放面板 (避免 label_map 缩放溢出问题).
    Strategy: draw masks at full resolution first, then scale panels (avoids label_map overflow).
    """
    H, W = full_img.shape[:2]

    # ── 1. 在全分辨率上绘制 mask overlay | Draw mask overlays at full resolution ──
    # Full-Image FastSAM → 橙色 | Orange
    vis_full_f = draw_masks(full_img, None, None, label_map=full_label_map,
                            uniform_color=(255, 140, 0), alpha=0.4)
    # Tile-Stitched FastSAM → 青色 | Cyan
    vis_tile_f = draw_masks(full_img, None, None, label_map=tile_label_map,
                            uniform_color=(0, 200, 200), alpha=0.4)
    # GT → 类别调色板 | Per-category palette
    vis_gt_f = draw_gt_by_category(full_img, gt_data, alpha=0.5)

    # ── 2. 缩放到显示尺寸 | Scale to display size ──
    max_display = 1400  # per-panel max dimension (2×2 grid → ~2800×1500 total)
    scale = min(max_display / max(H, W), 1.0)
    if scale < 1.0:
        H_disp, W_disp = int(H * scale), int(W * scale)
        img_disp = cv2.resize(full_img, (W_disp, H_disp), interpolation=cv2.INTER_AREA)
        vis_full = cv2.resize(vis_full_f, (W_disp, H_disp), interpolation=cv2.INTER_AREA)
        vis_tile = cv2.resize(vis_tile_f, (W_disp, H_disp), interpolation=cv2.INTER_AREA)
        vis_gt = cv2.resize(vis_gt_f, (W_disp, H_disp), interpolation=cv2.INTER_AREA)
    else:
        H_disp, W_disp = H, W
        img_disp = full_img
        vis_full = vis_full_f
        vis_tile = vis_tile_f
        vis_gt = vis_gt_f

    # ── 3. 2×2 网格拼接 | 2×2 Grid ──
    title_h = 28
    gap = 4
    panel_w = W_disp
    panel_h = H_disp

    titles = ["(a) Original", "(b) Full-Image FastSAM",
              "(c) Tile-Stitched FastSAM", "(d) Ground Truth"]
    panels = [img_disp, vis_full, vis_tile, vis_gt]

    total_w = panel_w * 2 + gap
    total_h = (panel_h + title_h) * 2 + gap
    composite = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240

    positions = [(0, 0), (1, 0), (0, 1), (1, 1)]  # (col, row)
    for (col, row), title, panel in zip(positions, titles, panels):
        x = col * (panel_w + gap)
        y = row * (panel_h + title_h + gap)  # +gap between rows
        # 标题背景 | Title background
        cv2.rectangle(composite, (x, y), (x + panel_w, y + title_h), (50, 50, 50), -1)
        cv2.putText(composite, title, (x + 8, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        composite[y + title_h:y + title_h + panel_h, x:x + panel_w] = panel

    cv2.imwrite(str(output_dir / "comparison.png"),
                cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    print(f"  [OK] comparison.png ({total_w}x{total_h})")

    cv2.imwrite(str(output_dir / "full_image_result.png"),
                cv2.cvtColor(vis_full_f, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "tile_stitched_result.png"),
                cv2.cvtColor(vis_tile_f, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "ground_truth.png"),
                cv2.cvtColor(vis_gt_f, cv2.COLOR_RGB2BGR))
    print(f"  [OK] full_image_result.png, tile_stitched_result.png, ground_truth.png")


# ═══════════════════════════════════════════════════════════════════
# 主流程 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FastSAM Tile Visualization — Full Image vs Tile Stitching"
    )
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help=f"源图像名称 | Source image name (default: {DEFAULT_SOURCE})")
    parser.add_argument("--data-root", type=str, default=None,
                        help="数据根目录 (覆盖配置文件中的路径) | Data root directory")
    parser.add_argument("--full-image-dir", type=str, default=None,
                        help="全分辨率原图目录 (默认: data/iSAID_processed/val/images)")
    parser.add_argument("--data-source", type=str, default="iSAID_tiles",
                        choices=["iSAID_instance_fewshot", "iSAID_tiles"],
                        help="数据源 | Data source (default: iSAID_tiles)")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["full", "tiles", "both"],
                        help="推理模式: full (全图), tiles (切片), both (对比, 默认)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="推理设备 | Inference device")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="FastSAM 置信度阈值 | Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="全图推理尺寸 | Full-image inference size")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(42)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # ── 解析全图目录 | Resolve full image directory ──
    global FULL_IMAGE_DIR
    if args.full_image_dir:
        FULL_IMAGE_DIR = args.full_image_dir

    # ── 解析数据根目录 | Resolve data root ──
    data_source = args.data_source
    cfg = DATA_SOURCE_CONFIGS[data_source].copy()

    if args.data_root is not None:
        data_root = Path(args.data_root)
        # Auto-detect tile image directory
        for cand in ["images/val", "images/train", "images"]:
            if (data_root / cand).exists():
                cfg["tile_dir"] = str(data_root / cand)
                break

        # Auto-detect annotations (COCO JSON) or metadata
        for cand in ["annotations/instances_val.json", "annotations/instances_train.json"]:
            if (data_root / cand).exists():
                cfg["anno_path"] = str(data_root / cand)
                cfg["gt_type"] = "coco_instance"
                break
        else:
            # Fallback: old metadata format
            meta_cand = data_root / "metadata" / "val.json"
            if meta_cand.exists():
                cfg["metadata_path"] = str(meta_cand)

        # Auto-detect masks_full directory
        for cand in ["masks_full", "masks/val"]:
            if (data_root / cand).exists():
                cfg["masks_full_dir"] = str(data_root / cand)
                break

    DATA_SOURCE_CONFIGS[data_source] = cfg
    tile_size = cfg["tile_size"]
    tile_stride = cfg["tile_stride"]

    # ── 输出目录 | Output Dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        tag = "Tiles" if data_source == "iSAID_tiles" else "Inst"
        args.output_dir = f"runs/viz_fastsam_{tag}_{args.source}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"  FastSAM Tile Visualization")
    print(f"  Source: {args.source} | Mode: {args.mode}")
    print(f"  Data: {data_source} ({tile_size}² tiles)")
    print(f"  Device: {device} | Output: {out_dir}")
    print(f"{'='*60}")

    # ── 1. 加载数据 | Load Data ──
    print("\n[1/4] Loading data...")
    full_img = load_full_image(args.source)
    if full_img is None:
        print(f"ERROR: Full image {args.source} not found in {FULL_IMAGE_DIR}")
        sys.exit(1)
    print(f"  Full image: {full_img.shape[1]}×{full_img.shape[0]}")

    tiles, tile_cfg = load_tile_info(args.source, data_source)
    print(f"  Tiles: {len(tiles)} ({tile_size}², stride={tile_stride})")

    # ── 2. 加载模型 | Load Model ──
    print("\n[2/4] Loading FastSAM...")
    model = load_fastsam_model(device)
    print(f"  Model loaded on {device}")

    # ── 3. 推理 | Inference ──
    full_result = {"label_map": np.zeros(full_img.shape[:2], dtype=np.int32), "count": 0, "scores": []}
    tile_results = []
    tile_stitch = {"label_map": np.zeros(full_img.shape[:2], dtype=np.int32), "count": 0, "scores": []}
    gt_data = {"label_map": np.zeros(full_img.shape[:2], dtype=np.int32), "cat_ids": [], "count": 0,
               "gt_type": "semantic" if data_source == "iSAID_tiles" else "instance"}

    if args.mode in ("full", "both"):
        print(f"\n[3/4] Full-image inference (imgsz={args.imgsz})...")
        full_result = predict_full_image(model, full_img, imgsz=args.imgsz,
                                         conf=args.conf, device=device)
        print(f"  Found {full_result['count']} masks")

    if args.mode in ("tiles", "both"):
        print(f"\n[3/4] Tile inference ({len(tiles)} tiles x {tile_size}²)...")
        tile_results = predict_tiles(model, tiles, device=device, conf=args.conf)
        total_tile_masks = sum(len(tr["masks"]) for tr in tile_results)
        print(f"  Total tile masks: {total_tile_masks}")

        print("  Stitching tiles...")
        tile_stitch = stitch_tile_masks(tile_results, full_img.shape[:2])
        print(f"  After stitching: {tile_stitch['count']} unique masks")

    # ── 4. 可视化 + 统计 | Visualize + Stats ──
    print(f"\n[4/4] Visualizing...")

    if args.mode == "both":
        print("  Loading GT annotations...")
        gt_data = load_gt_label_map(tiles, full_img.shape[:2], data_source)
        if gt_data["gt_type"] == "semantic":
            class_ids = [int(c) for c in gt_data.get("class_ids", [])]
            print(f"  GT (semantic): {gt_data['count']} classes: {class_ids}")
        else:
            print(f"  GT (instance): {gt_data['count']} instances")

        create_comparison_figure(
            full_img,
            full_result["label_map"],
            full_result["count"],
            tile_stitch["label_map"],
            tile_stitch["count"],
            tile_results,
            gt_data,
            out_dir,
        )
    elif args.mode == "full":
        vis = draw_masks(full_img, None, None, label_map=full_result["label_map"],
                         uniform_color=(255, 140, 0))
        cv2.imwrite(str(out_dir / "full_image_result.png"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"  [OK] full_image_result.png")
    elif args.mode == "tiles":
        vis = draw_masks(full_img, None, None, label_map=tile_stitch["label_map"],
                         uniform_color=(0, 200, 200))
        cv2.imwrite(str(out_dir / "tile_stitched_result.png"),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"  [OK] tile_stitched_result.png")

    # ── 保存统计 | Save Stats ──
    stats = {
        "source": args.source,
        "data_source": data_source,
        "mode": args.mode,
        "image_size": list(full_img.shape[:2]),
        "tile_count": len(tiles),
        "tile_size": tile_size,
        "tile_stride": tile_stride,
        "full_image_masks": full_result["count"],
        "tile_total_masks": sum(len(tr["masks"]) for tr in tile_results),
        "stitched_masks": tile_stitch["count"],
        "gt_items": gt_data["count"],
        "gt_type": gt_data.get("gt_type", "instance"),
        "conf_threshold": args.conf,
        "imgsz_full": args.imgsz,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  [OK] stats.json")

    print(f"\n{'='*60}")
    print(f"  Done! Output: {out_dir}")
    print(f"  Full-image: {stats['full_image_masks']} masks")
    print(f"  Tile-stitched: {stats['stitched_masks']} masks (from {stats['tile_total_masks']} raw)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
