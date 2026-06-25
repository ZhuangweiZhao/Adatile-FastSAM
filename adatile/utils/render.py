"""
Category Mask Renderer (Instance→Dense). | 类别掩码渲染器（实例→密集）

Shared utility for rendering COCO-format polygon/bbox instance annotations
into a single-channel uint8 dense category-ID mask.
将 COCO 格式多边形/bbox 实例标注渲染为单通道 uint8 密集类别 ID 掩码。

This is an instance-segmentation-first project. The rendered output is a
per-pixel category map (merging all instances), used for decoder training
and evaluation. For per-instance binary masks, use the dataset in default mode.
本项目以实例分割为主。渲染输出为逐像素类别图（合并所有实例），用于解码器训练和评估。
逐实例二值掩码请使用数据集的默认模式。

Single canonical definition (2026-06-21).
Previously duplicated in prep_isaid_tiles.py and train_b04.py.
"""

from __future__ import annotations

import numpy as np


def render_category_mask(annotations: list, h: int, w: int) -> np.ndarray:
    """
    Render dense category-ID mask [H, W] uint8 (values 0-15) from COCO instance annotations.
    从 COCO 实例标注渲染密集类别 ID 掩码 [H, W] uint8（值 0-15）。

    Merges all instance polygons into a single per-pixel label map by category_id.
    Mapping is done by prep_isaid.py fix_annotations().
    将所有实例多边形按 category_id 合并到逐像素标签图中。ID 映射由 prep_isaid.py 完成。

    Uses cv2.fillPoly for speed; falls back to PIL if cv2 unavailable.
    """
    dense = np.zeros((h, w), dtype=np.uint8)
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False

    for ann in annotations:
        cat_id = ann.get("category_id", 0)
        if cat_id <= 0 or cat_id > 15:
            continue

        seg = ann.get("segmentation", [])
        if not seg:
            bbox = ann.get("bbox", [0, 0, 0, 0])
            x, y, bw, bh = bbox
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w, int(x + bw)), min(h, int(y + bh))
            if x2 > x1 and y2 > y1:
                dense[y1:y2, x1:x2] = cat_id
            continue
        if isinstance(seg, dict):
            continue

        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)

            if _has_cv2:
                cv2.fillPoly(dense, [pts], cat_id)
            else:
                from PIL import Image, ImageDraw
                mask_pil = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask_pil).polygon(
                    [(int(p[0]), int(p[1])) for p in pts.reshape(-1, 2)],
                    fill=cat_id,
                )
                dense = np.maximum(dense, np.array(mask_pil, dtype=np.uint8))
    return dense
