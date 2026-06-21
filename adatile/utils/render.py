"""
Semantic Mask Renderer.

Shared utility for rendering COCO-format polygon/bbox annotations
into a single-channel uint8 semantic mask (category IDs).

Single canonical definition (2026-06-21).
Previously duplicated in prep_isaid_tiles.py and train_b04.py.
"""

from __future__ import annotations

import numpy as np


def render_semantic_mask(annotations: list, h: int, w: int) -> np.ndarray:
    """
    Render semantic mask [H, W] uint8 (values 0-15) from COCO annotations.

    Uses ann["category_id"] directly. Mapping is done by
    prep_isaid.py fix_annotations().

    Uses cv2.fillPoly for speed; falls back to PIL if cv2 unavailable.
    """
    sem = np.zeros((h, w), dtype=np.uint8)
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
                sem[y1:y2, x1:x2] = cat_id
            continue
        if isinstance(seg, dict):
            continue

        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)

            if _has_cv2:
                cv2.fillPoly(sem, [pts], cat_id)
            else:
                from PIL import Image, ImageDraw
                mask_pil = Image.new("L", (w, h), 0)
                ImageDraw.Draw(mask_pil).polygon(
                    [(int(p[0]), int(p[1])) for p in pts.reshape(-1, 2)],
                    fill=cat_id,
                )
                sem = np.maximum(sem, np.array(mask_pil, dtype=np.uint8))
    return sem
