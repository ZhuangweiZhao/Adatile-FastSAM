#!/usr/bin/env python3
"""
GT-Prompted FastSAM 评估 — 用真值框/点提示 FastSAM，测试分割上限.
GT-Prompted FastSAM Evaluation — Use GT boxes/points as prompts, test segmentation ceiling.
==========================================================================================

核心问题 | Core Question:
    FastSAM zero-shot 的瓶颈是"找不到目标"(检测) 还是"画不准"(分割)？
    用 GT bbox/point 作为 prompt，跳过检测，直接衡量分割质量上限。
    Is FastSAM's bottleneck detection (can't find objects) or segmentation (can't draw well)?
    Use GT bbox/point as prompts to skip detection, directly measure segmentation ceiling.

三种模式 | Three Modes:
    1. bbox:   用 GT bounding box 提示 → "框给你了, 能画准吗？"
    2. point:  用 GT 中心点提示   → "点给你了, 能画准吗？"
    3. bbox+N: 用 GT bbox 扩大 N%  → "框松一点, 会不会更好？"

用法 | Usage::

    # iSAID-5i 格式 (默认) | iSAID-5i format (default)
    python tools/eval/eval_fastsam_prompted.py

    # iSAID_processed / iSAID-few 格式 (全图 + COCO JSON)
    # iSAID_processed / iSAID-few format (full images + COCO JSON)
    python tools/eval/eval_fastsam_prompted.py \
        --data-root data/iSAID-few --data-format isaid_processed

    # 更多样本 + bbox 扩展测试
    python tools/eval/eval_fastsam_prompted.py --num-samples 50 --bbox-expand 0,10,20

    # 仅 bbox 模式
    python tools/eval/eval_fastsam_prompted.py --mode bbox

输出 | Output:
    runs/eval_fastsam_prompted_{ts}/
    ├── stats.json
    └── examples.png      # 样例对比图
"""

from __future__ import annotations

import sys, argparse, json, random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

# ⚠️ tools.viz 会把 thirdLibrary/FastSAM 加入 sys.path（遮盖官方 ultralytics）
# 先导入 ultralytics 再导入 viz 模块
# ⚠️ tools.viz adds thirdLibrary/FastSAM to sys.path (shadows official ultralytics)
# Import ultralytics first, then viz modules

import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from adatile.utils.seed import set_seed

# ── 先加载 ultralytics FastSAM (在 tools.viz 污染 sys.path 之前) ──
# ── Load ultralytics FastSAM (before tools.viz pollutes sys.path) ──
_MODEL = None

# ── 导入 viz 模块的数据加载函数 ──
# Import data loading from viz module
from tools.viz.viz_isaid_5i_fastsam import (
    list_images, load_sample, extract_gt_instances, CATEGORY_NAMES,
)

# ── 清理 viz 模块对 sys.path 的污染 (thirdLibrary/FastSAM 遮盖官方 ultralytics) ──
# Clean up sys.path pollution from viz module
_thirdlib_path = str(_PROJECT_ROOT / "thirdLibrary" / "FastSAM")
sys.path = [p for p in sys.path if p != _thirdlib_path]

# ═══════════════════════════════════════════════════════════════════
# FastSAM 模型加载 (ultralytics 官方版本) | FastSAM Model Loading
# 必须在 tools.viz 污染 sys.path 之后调用 (from ultralytics 是延迟 import)
# ═══════════════════════════════════════════════════════════════════

def get_fastsam_model(device: str = "cuda"):
    """加载或获取已缓存的 FastSAM (ultralytics official version)."""
    global _MODEL
    if _MODEL is None:
        from ultralytics import FastSAM
        model_path = _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"FastSAM weights not found: {model_path}")
        _MODEL = FastSAM(str(model_path))
    return _MODEL


# ═══════════════════════════════════════════════════════════════════
# GT 几何提取 | GT Geometry Extraction
# ═══════════════════════════════════════════════════════════════════

def mask_to_bbox(mask: np.ndarray) -> tuple:
    """二值 mask → bounding box (x1, y1, x2, y2)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        h, w = mask.shape
        return (0, 0, w, h)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return (int(x1), int(y1), int(x2), int(y2))


def mask_to_center(mask: np.ndarray) -> tuple:
    """二值 mask → 中心点 | Binary mask → center point."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        h, w = mask.shape
        return (w // 2, h // 2)
    return (int(np.mean(xs)), int(np.mean(ys)))


def expand_bbox(bbox: tuple, ratio: float, img_w: int, img_h: int) -> tuple:
    """
    扩展/收缩 bbox | Expand/shrink bbox.

    ratio > 0 → expand (loosen box)
    ratio < 0 → shrink (tighten box)

    保持中心不变, clip 到图像边界.
    """
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = int(bw * ratio / 2), int(bh * ratio / 2)
    x1_new = max(0, x1 - pad_w)
    y1_new = max(0, y1 - pad_h)
    x2_new = min(img_w - 1, x2 + pad_w)
    y2_new = min(img_h - 1, y2 + pad_h)
    # 保证至少 1px | Ensure at least 1px
    if x2_new <= x1_new: x2_new = x1_new + 1
    if y2_new <= y1_new: y2_new = y1_new + 1
    return (x1_new, y1_new, x2_new, y2_new)


def bbox_to_mask(bbox: tuple, shape: tuple) -> np.ndarray:
    """Bbox → binary mask [H, W]."""
    x1, y1, x2, y2 = bbox
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    mask[max(0, y1):min(H, y2 + 1), max(0, x1):min(W, x2 + 1)] = True
    return mask


def mask_centroid(mask: np.ndarray) -> tuple:
    """Binary mask → centroid (cx, cy)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (mask.shape[1] // 2, mask.shape[0] // 2)
    return (float(np.mean(xs)), float(np.mean(ys)))


def instance_size_category(area: float) -> str:
    """
    按 COCO 标准分类实例大小 | Classify instance by COCO size.

    - Small:  area < 32²  (1024 px²)
    - Medium: 32² ≤ area < 96²
    - Large:  area ≥ 96²  (9216 px²)
    """
    if area < 32 * 32:
        return "small"
    elif area < 96 * 96:
        return "medium"
    else:
        return "large"


def _round_imgsz(H: int, W: int) -> int:
    """将最长边向上取整到 32 的倍数 (避免 FastSAM warning 刷屏)."""
    return ((max(H, W) + 31) // 32) * 32


# ═══════════════════════════════════════════════════════════════════
# Prompted 推理 | Prompted Inference
#
# ⚠️ 重要说明 | Important Note:
#   FastSAM bbox prompt 并非真正的 "GT Prompt → Decoder"。
#   FastSAM 内部仍然先生成 everything masks，再用 bbox 过滤已有 proposal。
#   因此本实验测的是 "Proposal 存在 + Prompt 能否正确筛选"，而非纯分割上限。
#   论文中必须明确这一限定。
#
#   FastSAM bbox prompt is NOT true "GT Prompt → Decoder".
#   FastSAM still generates everything masks first, then filters by bbox.
#   So this experiment measures "Proposal exists + can prompt filter it",
#   not the pure segmentation ceiling. Paper MUST state this limitation.
# ═══════════════════════════════════════════════════════════════════

def _resize_mask_to_shape(mask_tensor, H: int, W: int) -> np.ndarray:
    """将 torch mask tensor resize 到目标尺寸 → binary mask (仅在需要完整分辨率时调用)."""
    m = mask_tensor.cpu().numpy()
    if m.shape != (H, W):
        m = torch.nn.functional.interpolate(
            torch.tensor(m).unsqueeze(0).unsqueeze(0).float(),
            size=(H, W), mode="bilinear",
        ).squeeze().cpu().numpy()
    return m > 0.5


def run_everything(model, image: np.ndarray, H: int, W: int,
                    device: str = "cuda",
                    cache_imgsz: int = 1024) -> dict:
    """
    运行 FastSAM everything mode，缓存所有 mask (低分辨率).
    Run FastSAM everything once, cache masks at LOW resolution for speed.

    关键优化 | Key optimization:
        - 固定 imgsz=1024 (而非原图 3000+ px) → mask 在 1024² 而非 3808²
        - 每个 mask ~1MB (bool 1024²) 而非 ~10MB (bool 3808²)
        - 10× 内存节省, 10× 过滤加速
        - Use fixed imgsz=1024 instead of full resolution → masks at 1024²
        - Each mask ~1MB vs ~10MB → 10× memory saving, 10× filter speedup

    :return: {"masks": [(bool_ndarray, centroid_x, centroid_y), ...],
              "imgsz": int, "scale_x": float, "scale_y": float}
    """
    cache_h = cache_w = cache_imgsz
    scale_x = cache_w / W
    scale_y = cache_h / H

    try:
        results = model(
            source=image, device=device, retina_masks=True,
            imgsz=cache_imgsz, conf=0.001, iou=0.9, verbose=False,
        )
    except Exception:
        return {"masks": [], "imgsz": cache_imgsz, "scale_x": scale_x, "scale_y": scale_y}

    if results is None or len(results) == 0 or results[0].masks is None:
        return {"masks": [], "imgsz": cache_imgsz, "scale_x": scale_x, "scale_y": scale_y}

    masks_data = results[0].masks.data  # [N, h, w] float32, 可能 ≠ cache_imgsz (保持宽高比)
    cached = []
    for i in range(len(masks_data)):
        m_raw = masks_data[i].cpu().numpy()  # float32, variable shape
        # 统一 resize 到 cache_imgsz² 确保后续运算 shape 一致
        # Resize to uniform cache_imgsz² so all subsequent ops have consistent shapes
        if m_raw.shape != (cache_h, cache_w):
            m_t = torch.tensor(m_raw).unsqueeze(0).unsqueeze(0).float()
            m_t = torch.nn.functional.interpolate(
                m_t, size=(cache_h, cache_w), mode="bilinear",
            ).squeeze().cpu().numpy()
            m_bin = m_t > 0.5
        else:
            m_bin = m_raw > 0.5

        if m_bin.sum() > 4:  # 过滤微小碎片 | Filter tiny fragments
            cx, cy = mask_centroid(m_bin)
            cached.append((m_bin, cx, cy))

    return {"masks": cached, "imgsz": cache_imgsz, "scale_x": scale_x, "scale_y": scale_y}


def _get_fullres_mask(cached_entry, H: int, W: int) -> np.ndarray:
    """将缓存的低分辨率 mask 升采样到全分辨率 | Upsample cached low-res mask to full resolution."""
    m_bin, _, _ = cached_entry
    if m_bin.shape == (H, W):
        return m_bin
    m_tensor = torch.tensor(m_bin).unsqueeze(0).unsqueeze(0).float()
    m_full = torch.nn.functional.interpolate(
        m_tensor, size=(H, W), mode="bilinear",
    ).squeeze().cpu().numpy()
    return m_full > 0.5


def _filter_bbox_from_cache(cached: dict, bbox: tuple, H: int, W: int
                              ) -> tuple[np.ndarray | None, str]:
    """从缓存的 everything masks 中按 bbox IoU 筛选最佳 mask (在缓存分辨率上运算)."""
    masks = cached["masks"]
    if not masks:
        return None, "no_result"

    sx, sy = cached["scale_x"], cached["scale_y"]
    # 缩放 bbox 到缓存分辨率 | Scale bbox to cache resolution
    bx1 = int(bbox[0] * sx); by1 = int(bbox[1] * sy)
    bx2 = int(bbox[2] * sx); by2 = int(bbox[3] * sy)
    ch, cw = cached["imgsz"], cached["imgsz"]

    # 在缓存分辨率创建 bbox mask | Create bbox mask at cache resolution
    bbox_mask_c = np.zeros((ch, cw), dtype=bool)
    by1_c = max(0, by1); by2_c = min(ch, by2 + 1)
    bx1_c = max(0, bx1); bx2_c = min(cw, bx2 + 1)
    if by2_c > by1_c and bx2_c > bx1_c:
        bbox_mask_c[by1_c:by2_c, bx1_c:bx2_c] = True

    best_iou, best_entry = 0.0, None
    for entry in masks:
        m_bin, _, _ = entry
        inter = float((m_bin & bbox_mask_c).sum())
        union = float((m_bin | bbox_mask_c).sum())
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_entry = entry

    if best_entry is None or best_iou <= 0.0:
        return None, "no_overlap"

    return _get_fullres_mask(best_entry, H, W), ""


def _filter_point_from_cache(cached: dict, point: tuple, H: int, W: int
                               ) -> tuple[np.ndarray | None, str]:
    """从缓存中按点距离筛选 (在缓存分辨率上运算，质心已预计算)."""
    masks = cached["masks"]
    if not masks:
        return None, "no_result"

    sx, sy = cached["scale_x"], cached["scale_y"]
    px = int(point[0] * sx); py = int(point[1] * sy)
    ch, cw = cached["imgsz"], cached["imgsz"]

    containing, non_containing = [], []
    for entry in masks:
        m_bin, cx, cy = entry
        dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        if 0 <= py < ch and 0 <= px < cw and m_bin[py, px]:
            containing.append((entry, dist))
        else:
            non_containing.append((entry, dist))

    candidates = containing if containing else non_containing
    if not candidates:
        return None, "no_result"

    candidates.sort(key=lambda x: x[1])
    return _get_fullres_mask(candidates[0][0], H, W), ""


def _filter_multipoint_from_cache(cached: dict, points: list, H: int, W: int
                                    ) -> tuple[np.ndarray | None, str]:
    """从缓存中按多点命中数筛选 (在缓存分辨率上运算)."""
    masks = cached["masks"]
    if not masks:
        return None, "no_result"

    sx, sy = cached["scale_x"], cached["scale_y"]
    ch, cw = cached["imgsz"], cached["imgsz"]
    # 缩放采样点到缓存分辨率 | Scale sample points to cache resolution
    pts = [(int(px * sx), int(py * sy)) for px, py in points]

    best_entry, best_hits = None, -1
    for entry in masks:
        m_bin, _, _ = entry
        hits = sum(1 for px, py in pts
                   if 0 <= py < ch and 0 <= px < cw and m_bin[py, px])
        if hits > best_hits:
            best_hits = hits
            best_entry = entry

    if best_entry is None:
        return None, "no_result"
    return _get_fullres_mask(best_entry, H, W), ""


def _filter_posneg_from_cache(cached: dict, pos_points: list, H: int, W: int
                                ) -> tuple[np.ndarray | None, str]:
    """从缓存中按正点命中数筛选 (在缓存分辨率上运算)."""
    masks = cached["masks"]
    if not masks:
        return None, "no_result"

    sx, sy = cached["scale_x"], cached["scale_y"]
    ch, cw = cached["imgsz"], cached["imgsz"]
    pts = [(int(px * sx), int(py * sy)) for px, py in pos_points]

    best_entry, best_hits = None, -1
    for entry in masks:
        m_bin, _, _ = entry
        hits = sum(1 for px, py in pts
                   if 0 <= py < ch and 0 <= px < cw and m_bin[py, px])
        if hits > best_hits:
            best_hits = hits
            best_entry = entry

    if best_entry is None:
        return None, "no_result"
    return _get_fullres_mask(best_entry, H, W), ""


# ═══════════════════════════════════════════════════════════════════
# 评估指标 | Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_mask_metrics(pred: np.ndarray | None, gt: np.ndarray,
                         failure_reason: str = "") -> dict:
    """
    计算预测 mask vs GT 的完整指标.
    Compute full metrics: IoU, IoU@thresholds, Dice, Precision, Recall,
    Boundary IoU, Failure Rate.

    :param pred: [H, W] binary mask or None.
    :param gt: [H, W] binary mask.
    :param failure_reason: "" if OK, otherwise "no_result"/"no_overlap"/"exception".
    :return: metrics dict.
    """
    if pred is None:
        return {
            "iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0,
            "valid": False, "failure": True, "failure_reason": failure_reason or "none",
            "iou50": 0, "iou75": 0, "iou90": 0,
            "boundary_iou": 0.0, "boundary_f1": 0.0,
        }

    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())

    iou = inter / union if union > 0 else 0.0
    dice = 2.0 * inter / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    prec = inter / pred_sum if pred_sum > 0 else 0.0
    rec = inter / gt_sum if gt_sum > 0 else 0.0

    # IoU@Thresholds (binary flags for AP computation)
    iou50 = 1 if iou >= 0.50 else 0
    iou75 = 1 if iou >= 0.75 else 0
    iou90 = 1 if iou >= 0.90 else 0

    # Boundary IoU (mask contour within 2px of GT contour)
    biou, bf1 = _compute_boundary_iou(pred, gt, dilation=2)

    return {
        "iou": iou, "dice": dice, "precision": prec, "recall": rec,
        "valid": True, "failure": False, "failure_reason": "",
        "iou50": iou50, "iou75": iou75, "iou90": iou90,
        "boundary_iou": biou, "boundary_f1": bf1,
    }


def _compute_boundary_iou(pred: np.ndarray, gt: np.ndarray,
                          dilation: int = 2) -> tuple[float, float]:
    """
    Boundary IoU: 只计算 mask 边界 (±dilation px) 上的 IoU.
    Boundary IoU: IoU computed only on mask boundary (±dilation px).

    遥感图像中目标的边界是最难分割的部分 (尤其是 bridge/ship/harbor).
    In remote sensing, boundaries are the hardest part (esp. bridge/ship/harbor).

    :return: (boundary_iou, boundary_f1).
    """
    # 获取 GT 边界 | Get GT boundary
    gt_u8 = gt.astype(np.uint8)
    gt_erode = cv2.erode(gt_u8, np.ones((3, 3), np.uint8), iterations=1)
    gt_boundary = gt_u8 & (~gt_erode)  # 1px contour

    # Dilate to create boundary band | Create boundary band
    kernel = np.ones((dilation * 2 + 1, dilation * 2 + 1), np.uint8)
    gt_band = cv2.dilate(gt_boundary, kernel, iterations=1)

    # 只在边界带上计算 IoU | Compute IoU only on boundary band
    pred_band = pred & gt_band
    gt_band_only = gt & gt_band

    inter_b = float((pred_band & gt_band_only).sum())
    union_b = float((pred_band | gt_band_only).sum())

    biou = inter_b / union_b if union_b > 0 else 0.0
    bf1 = 2.0 * inter_b / (float(pred_band.sum()) + float(gt_band_only.sum())) \
        if (pred_band.sum() + gt_band_only.sum()) > 0 else 0.0

    return biou, bf1


# ═══════════════════════════════════════════════════════════════════
# COCO JSON 数据加载 (iSAID_processed / iSAID-few 格式)
# COCO JSON Data Loading (iSAID_processed / iSAID-few format)
# ═══════════════════════════════════════════════════════════════════

def _get_gt_mask(ann: dict, _label_map_unused, H: int, W: int) -> np.ndarray:
    """获取 GT mask — 兼容 iSAID-5i 直接 mask 和 COCO bbox-cropped 格式."""
    if "mask" in ann:
        return ann["mask"]  # iSAID-5i 格式: 已有全分辨率 mask
    # COCO 格式: bbox-cropped mask → 嵌入全分辨率
    cropped = ann.get("cropped")
    if cropped is not None:
        x1, y1 = ann["x1"], ann["y1"]
        x2, y2 = ann["x2"], ann["y2"]
        full = np.zeros((H, W), dtype=bool)
        full[y1:y2, x1:x2] = cropped
        return full
    return np.zeros((H, W), dtype=bool)


def _coco_list_images(data_root: Path, split: str) -> list[str]:
    """列出 COCO 格式数据集中的图片 stems | List image stems from COCO-format dataset."""
    img_dir = data_root / split / "images"
    if not img_dir.exists():
        return []
    return sorted([p.stem for p in img_dir.glob("*.png")])


def _coco_load_annotations(data_root: Path, split: str) -> dict:
    """
    加载 COCO JSON 并按 image_id 索引 | Load COCO JSON and index by image_id.

    :return: {image_id: {"file_name": str, "annotations": [...], "height": int, "width": int}}
    """
    ann_file = data_root / split / "annotations" / f"instances_{split}.json"
    if not ann_file.exists():
        return {}

    with open(ann_file) as f:
        coco = json.load(f)

    # 构建 image_id → image_info 映射 | Build image_id → image_info mapping
    id_to_image = {img["id"]: img for img in coco["images"]}
    # file_name → image_id | For stem lookup
    file_to_id = {img["file_name"]: img["id"] for img in coco["images"]}

    # 按 image_id 分组 annotations | Group annotations by image_id
    ann_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_by_image[ann["image_id"]].append(ann)

    return {
        "id_to_image": id_to_image,
        "file_to_id": file_to_id,
        "ann_by_image": ann_by_image,
    }


def _coco_load_sample(stem: str, data_root: Path, split: str,
                       coco_index: dict) -> dict:
    """
    加载 COCO 格式单张全图 + 从 COCO JSON 提取 GT 实例.
    Load a single full image + extract GT instances from COCO JSON.

    :return: {"image": [H,W,3] RGB, "semantic_mask": None, "instance_mask": None,
              "stem": str, "gt_instances": [{"mask": [H,W] bool, "class_id": int,
              "area": int}, ...]}
    """
    img_path = data_root / split / "images" / f"{stem}.png"
    image = np.array(Image.open(str(img_path)).convert("RGB"))
    H, W = image.shape[:2]

    # 从 COCO 索引提取该图的标注 | Extract annotations for this image
    file_to_id = coco_index.get("file_to_id", {})
    ann_by_image = coco_index.get("ann_by_image", {})

    # 尝试多种文件名格式匹配 | Try multiple filename formats
    img_id = None
    for fname, fid in file_to_id.items():
        if Path(fname).stem == stem:
            img_id = fid
            break
    if img_id is None:
        img_id = file_to_id.get(f"{stem}.png", None)

    anns = ann_by_image.get(img_id, []) if img_id is not None else []

    # ── 内存优化: bbox-cropped mask, 无覆盖冲突 ──
    # Memory optimization: bbox-cropped independent masks, no overwrite conflicts
    gt_annotations = []

    for ann in anns:
        cat_id = ann.get("category_id", 0)
        if cat_id < 1 or cat_id > 15:
            continue

        seg = ann.get("segmentation", [])
        if not seg:
            bx, by, bw, bh = [int(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            x1, y1 = max(0, bx), max(0, by)
            x2, y2 = min(W, bx + bw), min(H, by + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            # Bbox-cropped mask: 仅存储 bbox 区域 | Bbox-cropped mask: only store bbox region
            cropped = np.ones((y2 - y1, x2 - x1), dtype=bool)
            area = (y2 - y1) * (x2 - x1)
            gt_annotations.append({
                "cropped": cropped, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_id": cat_id, "area": area,
            })
        elif isinstance(seg, list):
            if isinstance(seg[0], list):
                polys = seg
            elif isinstance(seg[0], (int, float)):
                polys = [seg]
            else:
                continue

            # 计算 polygon 的 bbox，仅分配 bbox 大小的 mask
            # Compute polygon bbox, allocate mask only at bbox size
            all_pts = []
            for poly in polys:
                if len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                all_pts.append(pts)
            if not all_pts:
                continue

            all_pts = np.concatenate(all_pts, axis=0)
            x1 = int(np.clip(all_pts[:, 0].min(), 0, W - 1))
            y1 = int(np.clip(all_pts[:, 1].min(), 0, H - 1))
            x2 = int(np.clip(all_pts[:, 0].max(), 0, W - 1)) + 1
            y2 = int(np.clip(all_pts[:, 1].max(), 0, H - 1)) + 1
            if x2 <= x1 or y2 <= y1:
                continue

            # 在 bbox 裁剪区域内渲染 polygon (独立 mask, 无覆盖冲突)
            # Render polygons in cropped bbox (independent mask, no overwrite)
            ch, cw = y2 - y1, x2 - x1
            crop = np.zeros((ch, cw), dtype=np.uint8)
            for poly in polys:
                if len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                pts[:, :, 0] -= x1
                pts[:, :, 1] -= y1
                pts[:, :, 0] = np.clip(pts[:, :, 0], 0, cw - 1)
                pts[:, :, 1] = np.clip(pts[:, :, 1], 0, ch - 1)
                cv2.fillPoly(crop, [pts], 1)

            area = int(crop.sum())
            if area < 16:
                continue

            gt_annotations.append({
                "cropped": crop.astype(bool), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "class_id": cat_id, "area": area,
            })
        else:
            continue

    gt_annotations.sort(key=lambda x: -x["area"])

    return {
        "image": image,
        "semantic_mask": None,
        "instance_mask": None,
        "stem": stem,
        "gt_annotations": gt_annotations,
    }


def _coco_sample_all_classes(data_root: Path, split: str, per_class: int,
                               seed: int, stem_to_class: dict) -> list[str]:
    """从 COCO JSON 每类各采样 per_class 张图 | Sample per_class images from each class in COCO JSON."""
    ann_file = data_root / split / "annotations" / f"instances_{split}.json"
    if not ann_file.exists():
        print(f"  [ERROR] Annotation file not found: {ann_file}")
        return []

    with open(ann_file) as f:
        coco = json.load(f)

    # 构建 class_id → {stems} 映射 | Build class_id → {stems} mapping
    img_id_to_stem = {img["id"]: Path(img["file_name"]).stem for img in coco["images"]}
    stems_by_class = {c: set() for c in range(1, 16)}

    for ann in coco.get("annotations", []):
        cat_id = ann.get("category_id", 0)
        if 1 <= cat_id <= 15:
            stem = img_id_to_stem.get(ann["image_id"])
            if stem:
                stems_by_class[cat_id].add(stem)

    rng = random.Random(seed)
    selected = set()
    for cls_id in range(1, 16):
        stems = list(stems_by_class[cls_id])
        if len(stems) == 0:
            print(f"  [WARN] Class {cls_id} ({CATEGORY_NAMES.get(cls_id, '?')}): 0 images")
            continue
        n = min(per_class, len(stems))
        sampled = rng.sample(stems, n)
        for s in sampled:
            if stem_to_class is not None and s not in stem_to_class:
                stem_to_class[s] = cls_id
        selected.update(sampled)
        print(f"  Class {cls_id:>2d} ({CATEGORY_NAMES.get(cls_id, '?'):<18s}): "
              f"{len(stems):>5d} available -> {n} sampled")

    return sorted(selected)


# ═══════════════════════════════════════════════════════════════════
# 主流程 | Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GT-Prompted FastSAM Evaluation — Segmentation Ceiling Test"
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--data-root", type=str, default=None,
                        help="数据根目录 (默认: data/iSAID-5i/iSAID) | Data root directory")
    parser.add_argument("--data-format", type=str, default="isaid5i",
                        choices=["isaid5i", "isaid_processed"],
                        help="数据格式: isaid5i (默认) 或 isaid_processed (全图+COCO JSON)")
    parser.add_argument("--num-samples", type=int, default=30,
                        help="随机模式: 图像数量 | Random mode: num images")
    parser.add_argument("--all-classes", action="store_true",
                        help="全类别模式: 从 15 个类各采样 per-class 张图")
    parser.add_argument("--per-class", type=int, default=5,
                        help="全类别模式: 每类采样图像数 (default: 5)")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["bbox", "point", "bbox+point", "center-point",
                                 "multi-point", "pos-neg", "all"],
                        help="Prompt 模式 | Prompt mode (all = bbox+center+multi+posneg)")
    parser.add_argument("--num-points", type=int, default=5,
                        help="多点/正负点模式的采样点数 (default: 5)")
    parser.add_argument("--diagnose", action="store_true",
                        help="诊断模式: 分析 Proposal Failure vs Selection Failure")
    parser.add_argument("--bbox-expand", type=str, default="0",
                        help="Bbox 扩展比例 (逗号分隔) | e.g. '0,0.1,0.2'")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    device = args.device

    # ── 解析数据格式 | Resolve data format ──
    if args.data_root is None:
        if args.data_format == "isaid_processed":
            args.data_root = "data/iSAID_processed"
        else:
            args.data_root = "data/iSAID-5i/iSAID"
    data_root = Path(args.data_root)
    use_coco_format = (args.data_format == "isaid_processed")

    # ── 解析参数 | Parse args ──
    modes = []
    if args.mode == "all":
        modes = ["bbox", "center-point", "multi-point", "pos-neg"]
    elif args.mode == "bbox+point":
        modes = ["bbox", "center-point"]
    else:
        modes = [args.mode]
    bbox_expands = [float(x) for x in args.bbox_expand.split(",")]

    # ── 输出目录 | Output Dir ──
    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        tag = "allcls" if args.all_classes else "rand"
        args.output_dir = f"runs/eval_fastsam_prompted_{tag}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  GT-Prompted FastSAM Evaluation (ultralytics)")
    print(f"  Data: {data_root} (format: {args.data_format})")
    print(f"  Modes: {modes} | Bbox expands: {bbox_expands}")
    if args.all_classes:
        print(f"  All-Classes mode: {args.per_class} img/class x 15 classes")
    else:
        print(f"  Random mode: {args.num_samples} images")
    print(f"  Split: {args.split} | Device: {device}")
    print(f"{'=' * 60}")

    # ── 1. 采样 | Sample ──
    print(f"\n[1/3] Sampling images...")
    stem_to_class = {}  # stem → known class

    # Pre-load COCO index for isaid_processed format | Pre-load for COCO format
    coco_index = None
    if use_coco_format:
        coco_index = _coco_load_annotations(data_root, args.split)

    if args.all_classes:
        if use_coco_format:
            selected = _coco_sample_all_classes(
                data_root, args.split, args.per_class, args.seed, stem_to_class)
        else:
            selected = _sample_all_classes(args.split, args.per_class, args.seed, stem_to_class)
    else:
        if use_coco_format:
            all_imgs = _coco_list_images(data_root, args.split)
        else:
            all_imgs = list_images(args.split)
        random.shuffle(all_imgs)
        selected = sorted(all_imgs[:args.num_samples])
    print(f"  Selected: {len(selected)} images")

    # ── 2. 加载模型 | Load Model ──
    print(f"\n[2/3] Loading FastSAM...")
    model = get_fastsam_model(device)
    print(f"  Model loaded on {device}")

    # ── 3. 评估 | Evaluate ──
    print(f"\n[3/3] Running prompted inference...")

    # 聚合: {tag: {iou[], dice[], ..., iou50[], iou75[], iou90[],
    #              biou[], failures{}, per_class{}, per_size{}, best[], worst[]}}
    agg = {}
    for mode_tag in modes:
        if mode_tag == "bbox":
            for expand in bbox_expands:
                tag = _expand_tag(expand)
                agg[tag] = _init_aggregator()
        else:
            agg[mode_tag] = _init_aggregator()

    # 诊断数据 | Diagnosis data (everything proposals → GT matching)
    diag_proposals = []     # [(stem, cls_id, best_proposal_iou, best_proposal_rank,
                            #   n_intersecting, prompt_iou, prompt_rank), ...]
    diag_worst = []         # worst cases with full candidate + everything info

    total_gt = 0

    for stem in tqdm(selected, desc="Evaluating", unit="img"):
        if use_coco_format:
            sample = _coco_load_sample(stem, data_root, args.split, coco_index)
            gt_annotations = sample["gt_annotations"]
        else:
            sample = load_sample(stem, args.split)
            gt_annotations = extract_gt_instances(sample["semantic_mask"], sample["instance_mask"])
        image = sample["image"]
        H, W = image.shape[:2]

        known_class = stem_to_class.get(stem)

        # ── 关键优化: 每张图只跑 1 次 everything @ 1024²，全部分享缓存 ──
        # Key optimization: run everything ONCE @ 1024² per image
        cache = run_everything(model, image, H, W, device)
        everything_masks_cache = cache["masks"]  # [(bool_ndarray, cx, cy), ...]
        if not everything_masks_cache:
            for ann in gt_annotations:
                cls_id = known_class if known_class else ann.get("class_id", 0)
                gt_mask = _get_gt_mask(ann, None, H, W)
                area = float(gt_mask.sum())
                size_cat = instance_size_category(area)
                total_gt += 1
                m = compute_mask_metrics(None, gt_mask, "no_result")
                for tag in agg:
                    _add_metrics(agg, tag, m, cls_id, size_cat)
            continue

        for ann in gt_annotations:
            gt_mask = _get_gt_mask(ann, None, H, W)
            bbox = mask_to_bbox(gt_mask)
            center = mask_to_center(gt_mask)
            cls_id = known_class if known_class else ann["class_id"]
            area = float(gt_mask.sum())
            size_cat = instance_size_category(area)
            total_gt += 1

            # ── Bbox 模式 (从缓存筛选) | Bbox mode (filter from cache) ──
            if "bbox" in modes:
                for expand in bbox_expands:
                    eb = expand_bbox(bbox, expand, W, H) if expand != 0 else bbox
                    pred, fail_reason = _filter_bbox_from_cache(
                        cache, eb, H, W)
                    m = compute_mask_metrics(pred, gt_mask, fail_reason)
                    tag = _expand_tag(expand)
                    _add_metrics(agg, tag, m, cls_id, size_cat)

                    if tag == "bbox":
                        entry = {"stem": stem, "iou": m["iou"], "cls_id": cls_id,
                                 "bbox": bbox, "size_cat": size_cat, "failure": m["failure"],
                                 "failure_reason": m["failure_reason"]}
                        # 仅保留 top-3 完整数据用于可视化 | Only keep top-3 for viz
                        agg[tag]["best"].append(entry)
                        agg[tag]["worst"].append(entry)

                        # ── 诊断分析: Everything proposals vs GT ──
                        # 诊断需要全分辨率 → 按需升采样 | Diagnosis needs full res → upsample on demand
                        if args.diagnose:
                            bbox_mask = bbox_to_mask(bbox, (H, W))
                            # 只处理与 bbox 有重叠的缓存 mask | Only process masks overlapping bbox
                            sx, sy = cache["scale_x"], cache["scale_y"]
                            bx1 = int(bbox[0] * sx); by1 = int(bbox[1] * sy)
                            bx2 = int(bbox[2] * sx); by2 = int(bbox[3] * sy)
                            ch = cw = cache["imgsz"]

                            intersecting = []
                            for pi, entry in enumerate(everything_masks_cache):
                                m_cache, m_cx, m_cy = entry
                                # 快速跳过: 质心不在 bbox 附近的 | Quick skip: centroid not near bbox
                                if not (bx1 - 10 <= m_cx <= bx2 + 10 and by1 - 10 <= m_cy <= by2 + 10):
                                    continue
                                # 升采样到全分辨率做精确 IoU | Upsample to full res for exact IoU
                                m_full = _get_fullres_mask(entry, H, W)
                                inter = float((m_full & gt_mask).sum())
                                union = float((m_full | gt_mask).sum())
                                gt_iou = inter / union if union > 0 else 0.0
                                intersecting.append({
                                    "index": pi, "mask": m_full, "gt_iou": gt_iou,
                                })

                            intersecting.sort(key=lambda x: -x["gt_iou"])
                            best_gt_iou = intersecting[0]["gt_iou"] if intersecting else 0.0
                            best_proposal = intersecting[0]["mask"] if intersecting else None
                            prompt_iou = m["iou"]
                            prompt_rank = -1
                            if pred is not None and intersecting:
                                for pi, cand in enumerate(intersecting):
                                    if cand["mask"].shape == pred.shape:
                                        if (cand["mask"] & pred).sum() / max(
                                                (cand["mask"] | pred).sum(), 1) > 0.8:
                                            prompt_rank = pi
                                            break

                            diag_proposals.append({
                                "stem": stem, "cls_id": cls_id,
                                "n_everything": len(everything_masks_cache),
                                "n_intersecting": len(intersecting),
                                "best_proposal_iou": best_gt_iou,
                                "best_proposal_rank": (1 if best_gt_iou > 0 else -1),
                                "prompt_iou": prompt_iou,
                                "prompt_rank": prompt_rank,
                            })
                            diag_worst.append({
                                **entry,
                                "n_everything": len(everything_masks_cache),
                                "n_intersecting": len(intersecting),
                                "intersecting_top5": intersecting[:5],
                                "best_proposal_iou": best_gt_iou,
                                "best_proposal": best_proposal,
                            })

            # ── Center-Point 模式 (从缓存筛选) | Center Point mode (from cache) ──
            if "center-point" in modes:
                pred, fail_reason = _filter_point_from_cache(
                    cache, center, H, W)
                _add_metrics(agg, "center-point",
                             compute_mask_metrics(pred, gt_mask, fail_reason),
                             cls_id, size_cat)

            # ── Multi-Point 模式 (从缓存筛选) | Multi Point mode (from cache) ──
            if "multi-point" in modes:
                ys, xs = np.where(gt_mask)
                n_pts = min(args.num_points, len(ys))
                rng = np.random.RandomState(args.seed)
                idx = rng.choice(len(ys), max(1, n_pts), replace=False)
                mp_points = [[int(xs[i]), int(ys[i])] for i in idx]
                pred, fail_reason = _filter_multipoint_from_cache(
                    cache, mp_points, H, W)
                m = compute_mask_metrics(pred, gt_mask, fail_reason)
                _add_metrics(agg, "multi-point", m, cls_id, size_cat)
                if "multi-point" in agg:
                    entry = {"stem": stem, "iou": m["iou"], "cls_id": cls_id,
                             "bbox": bbox, "size_cat": size_cat, "failure": m["failure"],
                             "failure_reason": m["failure_reason"]}
                    agg["multi-point"]["best"].append(entry)
                    agg["multi-point"]["worst"].append(entry)

            # ── Pos+Neg Point 模式 (从缓存筛选) | Pos+Neg mode (from cache) ──
            if "pos-neg" in modes:
                ys_pos, xs_pos = np.where(gt_mask)
                n_pos = min(args.num_points, len(ys_pos))
                rng = np.random.RandomState(args.seed)
                idx_p = rng.choice(len(ys_pos), max(1, n_pos), replace=False)
                pos_points = [[int(xs_pos[i]), int(ys_pos[i])] for i in idx_p]
                pred, fail_reason = _filter_posneg_from_cache(
                    cache, pos_points, H, W)
                _add_metrics(agg, "pos-neg",
                             compute_mask_metrics(pred, gt_mask, fail_reason),
                             cls_id, size_cat)

    # ── 排序 best/worst | Sort best/worst ──
    for key in agg:
        if agg[key]["best"]:
            agg[key]["best"].sort(key=lambda x: -x["iou"])
            agg[key]["worst"].sort(key=lambda x: x["iou"])

    # ── 诊断分析 | Diagnosis Analysis ──
    diag_per_class = None
    if args.diagnose and diag_proposals:
        diag_worst.sort(key=lambda x: x["iou"])
        diag_per_class = _build_diag_per_class(diag_proposals)
        _print_diagnosis(diag_proposals, diag_per_class)

    # ── 汇总报告 | Summary Report ──
    _print_summary(agg, total_gt, diag_per_class)

    # ── 可视化 | Visualize ──
    print(f"\n  Creating visualizations...")
    if args.all_classes:
        _create_summary_chart(agg, out_dir)
    _create_examples_grid(agg, out_dir)
    if args.diagnose and diag_worst:
        _create_diagnosis_grid(diag_worst, out_dir)

    # ── 保存 | Save ──
    summary = _build_summary(agg, diag_per_class)
    stats = {
        "split": args.split, "num_samples": len(selected),
        "num_gt_instances": total_gt,
        "modes": modes, "bbox_expands": bbox_expands,
        "all_classes": args.all_classes,
        "note": ("FastSAM bbox prompt filters existing proposals, NOT true GT→Decoder. "
                 "See script docstring for details."),
        "summary": summary,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  [OK] stats.json → {out_dir / 'stats.json'}")
    print(f"\n{'=' * 60}")
    print(f"  Done! Output: {out_dir}")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════
# 聚合与报告辅助函数 | Aggregation & Reporting Helpers
# ═══════════════════════════════════════════════════════════════════

def _init_aggregator() -> dict:
    """初始化聚合器 | Initialize aggregator dict."""
    return {
        "iou": [], "dice": [], "prec": [], "rec": [],
        "iou50": [], "iou75": [], "iou90": [],
        "biou": [], "bf1": [],
        "valid": 0, "total": 0,
        "failures": {"no_result": 0, "no_overlap": 0, "exception": 0},
        "per_class": {}, "per_size": {},
        "best": [], "worst": [],
    }


def _expand_tag(expand: float) -> str:
    """将 expand ratio 转为 tag | Convert expand ratio to tag."""
    if expand == 0:
        return "bbox"
    if expand > 0:
        return f"bbox+{int(expand * 100)}"
    return f"bbox{int(expand * 100)}"  # negative: bbox-10, bbox-20


def _add_metrics(agg: dict, tag: str, metrics: dict, cls_id: int, size_cat: str):
    """将单实例指标加入聚合 | Add single-instance metrics to aggregate."""
    if tag not in agg:
        agg[tag] = _init_aggregator()
    a = agg[tag]
    a["total"] += 1

    # 失败统计 | Failure tracking
    if metrics["failure"]:
        reason = metrics["failure_reason"] or "none"
        a["failures"][reason] = a["failures"].get(reason, 0) + 1
        # 失败的也记录 IoU=0 用于公平统计 | Record IoU=0 for failed cases (fair comparison)
        a["iou"].append(0.0)
        a["dice"].append(0.0)
        a["prec"].append(0.0)
        a["rec"].append(0.0)
        a["iou50"].append(0)
        a["iou75"].append(0)
        a["iou90"].append(0)
        a["biou"].append(0.0)
        a["bf1"].append(0.0)
    else:
        a["valid"] += 1
        a["iou"].append(metrics["iou"])
        a["dice"].append(metrics["dice"])
        a["prec"].append(metrics["precision"])
        a["rec"].append(metrics["recall"])
        a["iou50"].append(metrics["iou50"])
        a["iou75"].append(metrics["iou75"])
        a["iou90"].append(metrics["iou90"])
        a["biou"].append(metrics["boundary_iou"])
        a["bf1"].append(metrics["boundary_f1"])

    # Per-class
    if cls_id not in a["per_class"]:
        a["per_class"][cls_id] = []
    a["per_class"][cls_id].append(metrics["iou"])

    # Per-size
    if size_cat not in a["per_size"]:
        a["per_size"][size_cat] = []
    a["per_size"][size_cat].append(metrics["iou"])


def _print_summary(agg: dict, total_gt: int, diag_per_class: dict = None):
    """打印完整汇总报告 | Print full summary report."""
    print(f"\n  Total GT instances: {total_gt}")

    # ── 主指标表 | Main metrics table ──
    print(f"\n  {'Mode':<12s} {'Total':>6s} {'Valid':>6s} {'Fail%':>6s} "
          f"{'IoU':>8s} {'Dice':>8s} {'IoU50':>7s} {'IoU75':>7s} {'IoU90':>7s} "
          f"{'B-IoU':>7s}")
    print(f"  {'-' * 80}")

    for key in sorted(agg.keys()):
        a = agg[key]
        total = a["total"]
        if total == 0:
            continue
        valid = a["valid"]
        fail_pct = (total - valid) / total * 100
        print(f"  {key:<12s} {total:>6d} {valid:>6d} {fail_pct:>5.1f}% "
              f"{np.mean(a['iou']):>8.4f} {np.mean(a['dice']):>8.4f} "
              f"{np.sum(a['iou50'])/total:>6.3f} {np.sum(a['iou75'])/total:>6.3f} "
              f"{np.sum(a['iou90'])/total:>6.3f} {np.mean(a['biou']):>7.4f}")

    # ── 失败分析 | Failure Analysis ──
    if "bbox" in agg:
        print(f"\n  ── Failure Analysis (bbox) ──")
        fails = agg["bbox"]["failures"]
        total_fail = sum(fails.values())
        if total_fail > 0:
            for reason, count in sorted(fails.items(), key=lambda x: -x[1]):
                if count > 0:
                    print(f"    {reason:<15s}: {count:>5d} ({count/agg['bbox']['total']*100:.1f}%)")

    # ── Per-Class IoU (bbox) ──
    if "bbox" in agg and "per_class" in agg["bbox"]:
        has_diag = diag_per_class is not None
        hdr = (f"\n  ── Per-Class IoU (bbox)"
               f"{' + Oracle Ceiling' if has_diag else ''} ──")
        print(hdr)
        pc = agg["bbox"]["per_class"]
        if has_diag:
            print(f"  {'Class':<20s} {'N':>5s} {'Prompt':>8s} {'Ceiling':>8s} "
                  f"{'Sel.Loss':>9s} {'Miss%':>7s}")
            print(f"  {'-' * 64}")
        else:
            print(f"  {'Class':<20s} {'N':>5s} {'IoU':>8s}")
            print(f"  {'-' * 36}")
        for cls_id in sorted(pc.keys()):
            vals = pc[cls_id]
            name = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")
            miou = np.mean(vals) if vals else 0.0
            if has_diag and cls_id in diag_per_class:
                dv = diag_per_class[cls_id]
                print(f"  {cls_id}:{name:<17s} {len(vals):>5d} "
                      f"{miou:>8.4f} {dv['mean_ceiling_iou']:>8.4f} "
                      f"{dv['selection_loss']:>+9.4f} {dv['missing_rate']:>7.1%}")
            else:
                print(f"  {cls_id}:{name:<17s} {len(vals):>5d} {miou:>8.4f}")

    # ── Per-Size IoU ──
    if "bbox" in agg and "per_size" in agg["bbox"]:
        print(f"\n  ── Per-Size IoU (bbox) ──")
        ps = agg["bbox"]["per_size"]
        for size_cat in ["small", "medium", "large"]:
            if size_cat in ps and ps[size_cat]:
                print(f"    {size_cat:<8s}: n={len(ps[size_cat]):>4d}  "
                      f"IoU={np.mean(ps[size_cat]):.4f}")

    # ── Ceiling Analysis ──
    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  Segmentation Ceiling Analysis                       ║")
    print(f"  ╠══════════════════════════════════════════════════════╣")
    if "bbox" in agg and agg["bbox"]["total"] > 0:
        a = agg["bbox"]
        miou = np.mean(a["iou"])
        mbiou = np.mean(a["biou"])
        ap50 = np.sum(a["iou50"]) / a["total"]
        ap75 = np.sum(a["iou75"]) / a["total"]
        fail_rate = sum(a["failures"].values()) / a["total"] * 100
        print(f"  ║  Bbox-Prompted (GT bbox → proposal filter):         ║")
        print(f"  ║    IoU = {miou:.4f}  B-IoU = {mbiou:.4f}               ║")
        print(f"  ║    AP@50 = {ap50:.4f}  AP@75 = {ap75:.4f}                   ║")
        print(f"  ║    Failure Rate = {fail_rate:.1f}%                        ║")
    print(f"  ║                                                        ║")
    print(f"  ║  ⚠️  NLP: FastSAM bbox prompt = filter proposals,     ║")
    print(f"  ║     NOT true GT→Decoder. 论文中必须明确说明。          ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")


def _build_summary(agg: dict, diag_per_class: dict = None) -> dict:
    """构建 JSON 兼容的汇总 | Build JSON-compatible summary."""
    summary = {}
    if diag_per_class:
        summary["_diagnosis_per_class"] = diag_per_class
    for key in sorted(agg.keys()):
        a = agg[key]
        total = a["total"]
        if total == 0:
            summary[key] = {"total": 0}
            continue
        entry = {
            "total": total, "valid": a["valid"],
            "failure_rate": round(sum(a["failures"].values()) / total, 4),
            "failures": a["failures"],
            "mean_iou": round(float(np.mean(a["iou"])), 4),
            "mean_dice": round(float(np.mean(a["dice"])), 4),
            "mean_precision": round(float(np.mean(a["prec"])), 4),
            "mean_recall": round(float(np.mean(a["rec"])), 4),
            "ap50": round(float(np.sum(a["iou50"])) / total, 4),
            "ap75": round(float(np.sum(a["iou75"])) / total, 4),
            "ap90": round(float(np.sum(a["iou90"])) / total, 4),
            "mean_boundary_iou": round(float(np.mean(a["biou"])), 4),
            "mean_boundary_f1": round(float(np.mean(a["bf1"])), 4),
        }
        # Per-class
        entry["per_class"] = {}
        for cls_id, vals in a["per_class"].items():
            if vals:
                entry["per_class"][str(cls_id)] = {
                    "name": CATEGORY_NAMES.get(cls_id, f"cls{cls_id}"),
                    "count": len(vals), "mean_iou": round(float(np.mean(vals)), 4),
                }
        # Per-size
        entry["per_size"] = {}
        for size_cat, vals in a["per_size"].items():
            if vals:
                entry["per_size"][size_cat] = {
                    "count": len(vals), "mean_iou": round(float(np.mean(vals)), 4),
                }
        summary[key] = entry
    return summary


def _sample_all_classes(split: str, per_class: int, seed: int,
                        stem_to_class: dict | None = None) -> list[str]:
    """从每个类别 (1-15) 各采样 per_class 张图像. 通过 train_list 确定类别."""
    root = _PROJECT_ROOT / "data/iSAID-5i/iSAID" / split
    stems_by_class = {c: set() for c in range(1, 16)}
    missing_classes = []

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
                    stems_by_class[cls_id].add(stem)

    rng = random.Random(seed)
    selected = set()
    for cls_id in range(1, 16):
        stems = list(stems_by_class[cls_id])
        if len(stems) == 0:
            missing_classes.append(cls_id)
            print(f"  [WARN] Class {cls_id} ({CATEGORY_NAMES.get(cls_id, '?')}): 0 images")
            continue
        n = min(per_class, len(stems))
        sampled = rng.sample(stems, n)
        for s in sampled:
            if stem_to_class is not None and s not in stem_to_class:
                stem_to_class[s] = cls_id
        selected.update(sampled)
        print(f"  Class {cls_id:>2d} ({CATEGORY_NAMES.get(cls_id, '?'):<18s}): "
              f"{len(stems):>5d} available → {n} sampled")

    if missing_classes:
        print(f"  [WARN] {len(missing_classes)} classes have no images: {missing_classes}")
    return sorted(selected)


def _create_summary_chart(agg: dict, out_dir: Path):
    """创建 per-class IoU 汇总图 (OpenCV 水平条形图)."""
    if "bbox" not in agg:
        return
    pc = agg["bbox"]["per_class"]
    if not pc:
        return

    items = []
    for cls_id in sorted(pc.keys()):
        vals = pc[cls_id]
        name = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")
        items.append((cls_id, name, len(vals), np.mean(vals)))

    bar_h, gap = 24, 3
    left_margin, right_margin, top_margin, bottom_margin = 160, 120, 40, 20
    chart_w = 600
    total_h = top_margin + (bar_h + gap) * len(items) + bottom_margin
    total_w = left_margin + chart_w + right_margin

    img = np.ones((total_h, total_w, 3), dtype=np.uint8) * 35
    cv2.putText(img, "Per-Class IoU (GT Bbox → FastSAM)", (left_margin, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    max_iou = max(iou for _, _, _, iou in items) if items else 1.0
    for idx, (cls_id, name, count, iou) in enumerate(items):
        y = top_margin + idx * (bar_h + gap)
        bar_w = int(iou / max(1.0, max_iou) * chart_w)
        label = f"{cls_id}:{name}"
        cv2.putText(img, label, (left_margin - 8, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
        r, g = int(220 * (1 - iou)), int(220 * iou)
        cv2.rectangle(img, (left_margin, y), (left_margin + bar_w, y + bar_h), (0, g, r), -1)
        cv2.putText(img, f"{iou:.4f}  (n={count})", (left_margin + bar_w + 6, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    cv2.imwrite(str(out_dir / "per_class_iou.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"  [OK] per_class_iou.png ({total_w}x{total_h})")


def _create_examples_grid(agg: dict, out_dir: Path):
    """创建最佳/最差示例网格: Original | GT | Pred | Info."""
    if "bbox" not in agg:
        return
    best = agg["bbox"]["best"][:3]
    worst = agg["bbox"]["worst"][:3]
    examples = best + worst
    # 检查是否有完整数据 | Check if entries have full data
    if examples and "image" not in examples[0]:
        print(f"  [SKIP] examples.png — entries don't contain full image data (memory-optimized mode)")
        return
    N = len(examples)
    if N == 0:
        return

    panel_size, cols, gap, title_h, margin, label_w = 256, 3, 3, 18, 15, 140
    total_w = margin * 2 + panel_size * cols + gap * (cols - 1) + label_w
    total_h = margin * 2 + (panel_size + title_h + 12) * N + gap * (N - 1)
    composite = np.ones((total_h, total_w, 3), dtype=np.uint8) * 40

    for ci, lbl in enumerate(["Original", "GT Mask", "Pred Mask"]):
        cv2.putText(composite, lbl, (margin + ci * (panel_size + gap), margin + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    for ri, entry in enumerate(examples):
        y0 = margin + title_h + gap + ri * (panel_size + title_h + 12 + gap)
        image, gt_mask, pred_mask = entry["image"], entry["gt_mask"], entry["pred_mask"]
        iou, cls_id = entry["iou"], entry["cls_id"]
        cls_name = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")
        fail_reason = entry.get("failure_reason", "")

        H, W = image.shape[:2]
        scale = min(panel_size / W, panel_size / H)
        nw, nh = int(W * scale), int(H * scale)
        ox, oy = (panel_size - nw) // 2, (panel_size - nh) // 2

        # Original
        p0 = np.ones((panel_size, panel_size, 3), dtype=np.uint8) * 30
        p0[oy:oy + nh, ox:ox + nw] = cv2.resize(image, (nw, nh))
        composite[y0:y0 + panel_size, margin:margin + panel_size] = p0

        # GT Mask
        p1 = np.ones((panel_size, panel_size, 3), dtype=np.uint8) * 30
        if gt_mask is not None:
            p1[oy:oy + nh, ox:ox + nw, 1] = cv2.resize(
                gt_mask.astype(np.uint8) * 255, (nw, nh))
        composite[y0:y0 + panel_size, margin + panel_size + gap:margin + panel_size * 2 + gap] = p1

        # Pred Mask
        p2 = np.ones((panel_size, panel_size, 3), dtype=np.uint8) * 30
        p2[oy:oy + nh, ox:ox + nw] = cv2.resize(image, (nw, nh))
        if pred_mask is not None:
            pr_s = cv2.resize(pred_mask.astype(np.uint8) * 255, (nw, nh))
            overlay = np.zeros_like(p2)
            overlay[oy:oy + nh, ox:ox + nw, 2] = pr_s
            contours, _ = cv2.findContours(pr_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay[oy:oy + nh, ox:ox + nw], contours, -1, (255, 140, 0), 1)
            p2 = cv2.addWeighted(p2, 0.7, overlay, 0.5, 0)
        composite[y0:y0 + panel_size, margin + (panel_size + gap) * 2:margin + panel_size * 3 + gap * 2] = p2

        # Info
        tag, tag_color = ("BEST", (0, 220, 100)) if ri < 3 else ("WORST", (100, 100, 255))
        lx = margin + panel_size * 3 + gap * 2 + 8
        cv2.putText(composite, f"[{tag}]", (lx, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, tag_color, 1)
        cv2.putText(composite, f"cls:{cls_id}:{cls_name}", (lx, y0 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(composite, f"IoU:{iou:.4f}", (lx, y0 + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1)
        if fail_reason:
            cv2.putText(composite, f"FAIL:{fail_reason}", (lx, y0 + 76),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 150, 255), 1)

    cv2.imwrite(str(out_dir / "examples.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    print(f"  [OK] examples.png ({total_w}x{total_h})")


def _build_diag_per_class(diag_proposals: list) -> dict:
    """按类别聚合诊断指标 | Aggregate diagnosis metrics per class."""
    pc = {}
    for d in diag_proposals:
        c = d["cls_id"]
        if c not in pc:
            pc[c] = {"ceiling_ious": [], "prompt_ious": [], "n": 0, "missing": 0}
        pc[c]["ceiling_ious"].append(d["best_proposal_iou"])
        pc[c]["prompt_ious"].append(d["prompt_iou"])
        pc[c]["n"] += 1
        if d["best_proposal_iou"] < 0.25:
            pc[c]["missing"] += 1

    result = {}
    for c, v in pc.items():
        result[c] = {
            "name": CATEGORY_NAMES.get(c, f"cls{c}"),
            "n": v["n"],
            "missing": v["missing"],
            "missing_rate": round(v["missing"] / v["n"], 4),
            "mean_ceiling_iou": round(float(np.mean(v["ceiling_ious"])), 4),
            "mean_prompt_iou": round(float(np.mean(v["prompt_ious"])), 4),
            "selection_loss": round(float(np.mean(np.array(v["ceiling_ious"]) -
                                                   np.array(v["prompt_ious"]))), 4),
        }
    return result


def _print_diagnosis(diag_proposals: list, diag_per_class: dict = None):
    """
    打印诊断报告: Everything Proposal 质量 + Prompt 筛选效果.
    Diagnosis: Everything proposal quality + Prompt filtering effect.

    三阶段分解 | Three-stage decomposition:
        Proposal Ceiling = max GT IoU among everything proposals
        Prompted Result  = IoU of the bbox-prompted mask
        Mask Gap         = Ceiling - Prompted (how much prompt loses vs best available)
    """
    n = len(diag_proposals)
    ceilings = [d["best_proposal_iou"] for d in diag_proposals]
    prompts = [d["prompt_iou"] for d in diag_proposals]
    n_intersecting = [d["n_intersecting"] for d in diag_proposals]
    n_everything = [d["n_everything"] for d in diag_proposals]

    ceilings_a = np.array(ceilings)
    prompts_a = np.array(prompts)
    gaps = ceilings_a - prompts_a  # positive = prompt loses info, negative = prompt better

    # Proposal quality tiers
    excellent = int((ceilings_a >= 0.75).sum())
    good = int(((ceilings_a >= 0.50) & (ceilings_a < 0.75)).sum())
    poor = int(((ceilings_a >= 0.25) & (ceilings_a < 0.50)).sum())
    missing = int((ceilings_a < 0.25).sum())

    print(f"\n  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  Failure Mode Diagnosis (Everything Proposals)          ║")
    print(f"  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Instances: {n:>5d} | Everything masks/img: {np.mean(n_everything):>5.1f}                ║")
    print(f"  ║  Intersecting with GT bbox: {np.mean(n_intersecting):>5.1f}/instance         ║")
    print(f"  ║                                                          ║")
    print(f"  ║  ── Proposal Quality (best everything mask vs GT) ──    ║")
    print(f"  ║  Excellent (≥0.75):  {excellent:>5d} ({excellent/n*100:>5.1f}%)                  ║")
    print(f"  ║  Good      (0.5-0.75):{good:>5d} ({good/n*100:>5.1f}%)                  ║")
    print(f"  ║  Poor      (0.25-0.5):{poor:>5d} ({poor/n*100:>5.1f}%)                  ║")
    print(f"  ║  Missing   (<0.25):   {missing:>5d} ({missing/n*100:>5.1f}%)                  ║")
    print(f"  ║                                                          ║")
    print(f"  ║  ── Prompt vs Ceiling ──                                ║")
    print(f"  ║  Mean Proposal Ceiling IoU: {np.mean(ceilings_a):.4f}                     ║")
    print(f"  ║  Mean Prompted IoU:         {np.mean(prompts_a):.4f}                     ║")
    print(f"  ║  Mean Gap (Ceiling−Prompt): {np.mean(gaps):+.4f}                     ║")
    print(f"  ║                                                          ║")
    # How often does prompt beat the best everything proposal?
    better = int((gaps < -0.01).sum())  # prompt > ceiling (unexpected but possible)
    equal = int((abs(gaps) <= 0.01).sum())  # prompt ≈ ceiling
    worse = int((gaps > 0.01).sum())  # prompt < ceiling (lost info)
    print(f"  ║  Prompt vs Best Proposal:                               ║")
    print(f"  ║    Prompt BETTER:  {better:>5d} ({better/n*100:>5.1f}%)                  ║")
    print(f"  ║    Prompt EQUAL:    {equal:>5d} ({equal/n*100:>5.1f}%)                  ║")
    print(f"  ║    Prompt WORSE:    {worse:>5d} ({worse/n*100:>5.1f}%)                  ║")
    print(f"  ║                                                          ║")
    print(f"  ║  ── Interpretation ──                                   ║")
    if missing / n > 0.3:
        print(f"  ║  Proposal Failure is MAJOR ({missing/n*100:.0f}%):       ║")
        print(f"  ║  → FastSAM generates no good proposal for these.        ║")
    if worse / n > 0.1:
        print(f"  ║  Selection Gap exists ({worse/n*100:.0f}%):              ║")
        print(f"  ║  → Prompt doesn't select the best available mask.       ║")
    print(f"  ║  → Remaining gap = genuine mask quality limit.          ║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    # Per-class diagnosis table
    if diag_per_class:
        print(f"\n  ── Per-Class Diagnosis (Everything Ceiling + Missing Rate) ──")
        print(f"  {'Class':<20s} {'N':>5s} {'Ceiling':>8s} {'Prompt':>8s} "
              f"{'Sel.Loss':>9s} {'Missing':>8s} {'Miss%':>7s}")
        print(f"  {'-' * 72}")
        for c in sorted(diag_per_class.keys()):
            v = diag_per_class[c]
            print(f"  {c}:{v['name']:<17s} {v['n']:>5d} "
                  f"{v['mean_ceiling_iou']:>8.4f} {v['mean_prompt_iou']:>8.4f} "
                  f"{v['selection_loss']:>+9.4f} {v['missing']:>8d} {v['missing_rate']:>7.1%}")


def _create_diagnosis_grid(worst_cases: list, out_dir: Path, top_n: int = 4):
    """
    诊断网格: GT | Prompted | Best Everything Proposal | Top-3 Intersecting.
    Diagnosis grid: GT | Prompted | Best Proposal | Top-3 intersecting.
    """
    cases = worst_cases[:top_n]
    N = len(cases)
    if N == 0:
        return
    # 检查是否有完整图像数据 | Check for full image data
    if "image" not in cases[0]:
        print(f"  [SKIP] diagnosis.png — entries don't contain full image data")
        return

    panel_size, max_show = 180, 3
    gap, title_h, margin = 3, 16, 12
    cols = 1 + 1 + 1 + max_show  # GT | Prompted | Best Prop | Top-3 Intersecting
    total_w = margin * 2 + panel_size * cols + gap * (cols - 1)
    total_h = margin * 2 + (panel_size + title_h + gap + 50) * N

    composite = np.ones((total_h, total_w, 3), dtype=np.uint8) * 35

    for ri, entry in enumerate(cases):
        y0 = margin + ri * (panel_size + title_h + gap + 50)
        image, gt_mask, pred = entry["image"], entry["gt_mask"], entry["pred_mask"]
        best_prop = entry.get("best_proposal")
        intersecting = entry.get("intersecting_top5", [])
        cls_id, iou, bpi = entry["cls_id"], entry["iou"], entry.get("best_proposal_iou", 0)
        cls_name = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")

        H, W = image.shape[:2]
        scale = min(panel_size / W, panel_size / H)
        nw, nh = int(W * scale), int(H * scale)
        ox, oy = (panel_size - nw) // 2, (panel_size - nh) // 2
        img_s = cv2.resize(image, (nw, nh))

        def _draw(msk, color=(255, 140, 0)):
            p = np.ones((panel_size, panel_size, 3), dtype=np.uint8) * 20
            p[oy:oy + nh, ox:ox + nw] = img_s
            if msk is not None:
                ms = cv2.resize(msk.astype(np.uint8) * 255, (nw, nh))
                ov = np.zeros_like(p)
                ov[oy:oy + nh, ox:ox + nw] = np.stack([ms * (c // 255) for c in color], axis=-1)
                p = cv2.addWeighted(p, 0.65, ov.astype(np.float32), 0.45, 0)
            return p

        # GT (green)
        composite[y0:y0 + panel_size, margin:margin + panel_size] = _draw(gt_mask, (0, 200, 80))

        # Prompted (orange)
        cx1 = margin + panel_size + gap
        composite[y0:y0 + panel_size, cx1:cx1 + panel_size] = _draw(pred, (255, 140, 0))

        # Best Proposal (blue)
        cx2 = margin + (panel_size + gap) * 2
        composite[y0:y0 + panel_size, cx2:cx2 + panel_size] = _draw(best_prop, (0, 160, 255))

        # Top intersecting
        for ci in range(max_show):
            cx = margin + (panel_size + gap) * (3 + ci)
            if ci < len(intersecting) and intersecting[ci] is not None:
                cand = intersecting[ci]
                composite[y0:y0 + panel_size, cx:cx + panel_size] = _draw(
                    cand["mask"], (100, 200, 255))
                gi = cand.get("gt_iou", 0)
                cv2.putText(composite, f"gI={gi:.3f}", (cx, y0 + panel_size + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (180, 180, 180), 1)

        # Info
        lx = margin + (panel_size + gap) * cols + 8
        cv2.putText(composite, f"#{ri + 1} cls:{cls_id}:{cls_name}",
                    (lx, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        cv2.putText(composite, f"Prompt IoU={iou:.4f}",
                    (lx, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 160, 60), 1)
        cv2.putText(composite, f"Best Proposal IoU={bpi:.4f} (Ceiling)",
                    (lx, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 180, 255), 1)
        gap_val = bpi - iou
        cv2.putText(composite, f"Gap={gap_val:+.4f} ({'ceiling reached' if gap_val <= 0.01 else 'proposal better'})",
                    (lx, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (200, 200, 200), 1)

    cv2.imwrite(str(out_dir / "diagnosis.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    print(f"  [OK] diagnosis.png ({total_w}x{total_h})")


if __name__ == "__main__":
    from tools._deprecated_guard import require_legacy_optin
    require_legacy_optin(__file__)  # DEPRECATED: pre-V3 protocol (see EVALUATION_PROTOCOL_V3.md §12.4)
    main()
