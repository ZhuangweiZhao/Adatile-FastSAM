"""
Decoder Prediction Visualizer v3 — 真实 Prototype + GT 对比 + 汇总大图 + 全图模式.
Decoder Prediction Visualizer v3 — Real Prototypes + GT Comparison + Full-Image Mode.

用法 | Usage:

    # Tile 模式 | Tile mode:
    python tools/diag/diag_pred_vis.py \
        --checkpoint runs/.../best_model.pt --decoder center_affinity \
        --mode tile --query-tile P0089_t0001 --split val \
        --data-root data/iSAID_instance_fewshot \
        --output vis_output/ --device cuda

    # 全图模式 | Full-image mode:
    python tools/diag/diag_pred_vis.py \
        --checkpoint runs/.../best_model.pt --decoder center_affinity \
        --mode full --image /path/to/P0089.png \
        --full-gt /path/to/instances_val.json \
        --output vis_output/ --device cuda

v3 改进 | v3 Improvements:
    1. 真实 prototype (从 support set 在线构建)
    2. GT 对比 (TP/FP/FN 叠加)
    3. 单张汇总大图
    4. 全图模式: 原图 → 切 tile → 逐 tile 推理 → 拼接 → 全图可视化
"""

from __future__ import annotations

import argparse, json, os, sys, random, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from adatile.utils.seed import set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ═══════════════════════════════════════════════════════════════════
# Constants | 常量
# ═══════════════════════════════════════════════════════════════════

ISAID_CLASSES = {
    1: "small_vehicle", 2: "large_vehicle", 3: "plane", 4: "storage_tank",
    5: "ship", 6: "harbor", 7: "ground_track_field", 8: "soccer_ball_field",
    9: "tennis_court", 10: "swimming_pool", 11: "baseball_diamond",
    12: "basketball_court", 13: "bridge", 14: "helicopter", 15: "roundabout",
}
CLASS_COLORS = {cid: plt.cm.tab20(i % 20) for i, cid in enumerate(ISAID_CLASSES.keys())}
_COCO_CACHE: dict = {}

# ═══════════════════════════════════════════════════════════════════
# Phase 1: Model Loading | 模型加载
# ═══════════════════════════════════════════════════════════════════

def _fastsam_weights_path():
    p = os.path.join(_PROJECT_ROOT, "thirdLibrary", "FastSAM", "weights", "FastSAM-x.pt")
    if os.path.exists(p):
        return p
    env_path = os.environ.get("FASTSAM_WEIGHTS", "")
    if env_path and os.path.exists(env_path):
        return env_path
    raise FileNotFoundError(f"FastSAM-x.pt not found at {p}.")


def _load_model_and_decoder(ckpt_path: str, unfreeze_layers: int, decoder_type: str, device: str):
    from ultralytics import FastSAM
    from tools.train.train_fewshot_allclass import extract_features

    model = FastSAM(str(_fastsam_weights_path()))
    model.model.to(device).eval()
    for p in model.model.parameters():
        p.requires_grad = False

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if unfreeze_layers > 0 and "backbone" in ckpt:
        seq = model.model.model
        for i_str, state in ckpt["backbone"].items():
            seq[int(i_str)].load_state_dict(state)
        print(f"  Restored {unfreeze_layers} backbone layers")

    test_img = np.zeros((896, 896, 3), dtype=np.uint8)
    test_feats = extract_features(model, [test_img], device)
    p3_channels = test_feats[0]["p3"].shape[1]
    p4_channels = test_feats[0]["p4"].shape[1]
    print(f"  P3={p3_channels}, P4={p4_channels}")

    normalize_proto = ckpt.get("normalize_proto", "none")
    if decoder_type == "dynamic_kernel":
        from adatile.decoder.dynamic_kernel_decoder import DynamicKernelDecoder
        n_kernels = ckpt.get("n_kernels", 16)
        decoder = DynamicKernelDecoder(
            p3_channels=p3_channels, p4_channels=p4_channels,
            proto_dim=32, n_kernels=n_kernels, kernel_dim=256, fpn_dim=256,
            normalize_proto=normalize_proto,
        ).to(device)
    elif decoder_type == "center_affinity":
        from adatile.decoder.center_affinity_decoder import CenterAffinityDecoder
        decoder = CenterAffinityDecoder(
            p3_channels=p3_channels, p4_channels=p4_channels,
            proto_dim=32, fpn_dim=64, normalize_proto=normalize_proto,
        ).to(device)
    elif decoder_type == "adaptive":
        from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
        decoder = AdaptiveSparseDecoder(in_channels=p4_channels, use_fdr=False,
                                        normalize_proto=normalize_proto).to(device)
    elif decoder_type == "adaptive-p3p4":
        from adatile.decoder.adaptive_decoder_p3p4 import AdaptiveDecoderP3P4
        decoder = AdaptiveDecoderP3P4(p3_channels=p3_channels, p4_channels=p4_channels,
                                      proto_dim=32, hidden_dim=256).to(device)
    else:
        raise ValueError(f"Unknown decoder: {decoder_type}")

    decoder.load_state_dict(ckpt["decoder"], strict=False)
    decoder.eval()
    print(f"  Loaded {decoder_type} decoder (epoch {ckpt.get('epoch', '?')})")
    return model, decoder, extract_features, ckpt


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Data Loading | 数据加载
# ═══════════════════════════════════════════════════════════════════

def _extract_source_image(stem: str) -> str:
    parts = stem.rsplit("_t", 1)
    return parts[0] if len(parts) == 2 else stem


def _load_coco_index(data_root: Path, split: str) -> dict:
    cache_key = (str(data_root), split)
    if cache_key in _COCO_CACHE:
        return _COCO_CACHE[cache_key]
    ann_file = data_root / "annotations" / f"instances_{split}.json"
    if not ann_file.exists():
        _COCO_CACHE[cache_key] = {}
        return {}
    with open(ann_file) as f:
        coco = json.load(f)
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    file_to_anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        fname = img_id_to_file.get(ann["image_id"])
        if fname:
            file_to_anns[fname].append(ann)
    _COCO_CACHE[cache_key] = {
        "file_to_anns": dict(file_to_anns), "images": coco["images"],
        "annotations": coco.get("annotations", []), "categories": coco.get("categories", []),
    }
    return _COCO_CACHE[cache_key]


def _build_class_index(data_root: Path, split: str) -> dict:
    coco_idx = _load_coco_index(data_root, split)
    if not coco_idx:
        return {}
    tile_stems = {img["id"]: Path(img["file_name"]).stem for img in coco_idx["images"]}
    tile_classes: dict = defaultdict(lambda: defaultdict(int))
    for ann in coco_idx["annotations"]:
        cat_id = ann.get("category_id", 0)
        img_id = ann.get("image_id", 0)
        if 1 <= cat_id <= 15 and img_id in tile_stems:
            tile_classes[img_id][cat_id] += 1
    index = defaultdict(lambda: defaultdict(list))
    for img_id, cls_counts in tile_classes.items():
        if not cls_counts:
            continue
        stem = tile_stems[img_id]
        src = _extract_source_image(stem)
        for cls_id in cls_counts.keys():
            index[cls_id][src].append(stem)
    return {k: {src: sorted(set(tiles)) for src, tiles in v.items()} for k, v in index.items()}


def load_gt_for_tile(stem: str, data_root: Path, split: str,
                     H: int = 896, W: int = 896) -> dict:
    coco_idx = _load_coco_index(data_root, split)
    file_to_anns = coco_idx.get("file_to_anns", {})
    anns = file_to_anns.get(f"{stem}.png", [])
    instances = []
    merged = np.zeros((H, W), dtype=bool)
    for ann in anns:
        cat_id = ann.get("category_id", 0)
        if cat_id < 1 or cat_id > 15:
            continue
        mask = _render_polygon_mask(ann, H, W)
        if mask.sum() > 0:
            instances.append({"category_id": cat_id, "mask": mask, "area": float(mask.sum())})
            merged = merged | mask
    return {"instances": instances, "class_ids": {i["category_id"] for i in instances}, "merged_mask": merged}


def _render_gt_mask_on_demand(gt_ann: dict, H: int, W: int) -> np.ndarray:
    """按需渲染单个 GT 实例 mask | Render a single GT instance mask on demand.

    避免预存 3304×7MB=23GB 的 mask 数组 | Avoids storing 23GB of pre-rendered masks.
    """
    return _render_polygon_mask({"segmentation": gt_ann["segmentation"],
                                  "bbox": gt_ann["bbox"]}, H, W)


def _render_polygon_mask(ann: dict, H: int, W: int) -> np.ndarray:
    """从 COCO annotation 渲染二值 mask | Render binary mask from COCO annotation."""
    seg = ann.get("segmentation", [])
    if not seg:
        bx, by, bw, bh = [int(v) for v in ann.get("bbox", [0, 0, 0, 0])]
        mask = np.zeros((H, W), dtype=bool)
        mask[max(0, by):min(H, by + bh), max(0, bx):min(W, bx + bw)] = True
        return mask
    if isinstance(seg, list):
        mask = np.zeros((H, W), dtype=np.uint8)
        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            if len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            pts[:, :, 0] = np.clip(pts[:, :, 0], 0, W - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1], 0, H - 1)
            cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)
    return np.zeros((H, W), dtype=bool)


def load_tile_image(stem: str, data_root: Path, split: str) -> np.ndarray:
    img_path = data_root / "images" / split / f"{stem}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Tile image not found: {img_path}")
    return np.array(Image.open(str(img_path)).convert("RGB"))


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Full-Image Tiling & Stitching | 全图切分 & 拼接
# ═══════════════════════════════════════════════════════════════════

def tile_full_image(image: np.ndarray, tile_size: int = 896, stride: int = 640
                    ) -> list[dict]:
    """将全图切分为 tile | Split full image into tiles.

    :return: list of {y0, x0, h, w, img: [H,W,3] uint8}
    """
    H, W = image.shape[:2]
    tiles = []

    for y0 in list(range(0, H - tile_size + 1, stride)) if H >= tile_size else [0]:
        for x0 in list(range(0, W - tile_size + 1, stride)) if W >= tile_size else [0]:
            _y0 = max(0, min(y0, H - tile_size)) if H - y0 < tile_size else y0
            _x0 = max(0, min(x0, W - tile_size)) if W - x0 < tile_size else x0

            tile_img = image[_y0:_y0 + tile_size, _x0:_x0 + tile_size]
            th, tw = tile_img.shape[:2]
            if th < tile_size or tw < tile_size:
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, tile_size - th, 0, tile_size - tw,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0),
                )
            tiles.append({"y0": _y0, "x0": _x0, "h": th, "w": tw, "img": tile_img})

    return tiles


def crop_gt_to_tile(full_annotations: list, y0: int, x0: int,
                    tile_size: int) -> list[dict]:
    """将全图 GT 标注裁剪到 tile 坐标系 | Crop full-image GT to tile coordinates.

    :return: list of {category_id, mask: bool[tile_size, tile_size], area}
    """
    tile_instances = []
    for ann in full_annotations:
        cat_id = ann.get("category_id", 0)
        if cat_id < 1 or cat_id > 15:
            continue
        bbox = ann.get("bbox", [0, 0, 0, 0])
        ax, ay, aw, ah = bbox
        if ax + aw <= x0 or ax >= x0 + tile_size or ay + ah <= y0 or ay >= y0 + tile_size:
            continue

        seg = ann.get("segmentation", [])
        if not seg:
            bx, by = max(0, ax - x0), max(0, ay - y0)
            bw, bh = min(aw, tile_size - bx), min(ah, tile_size - by)
            if bw <= 0 or bh <= 0:
                continue
            mask = np.zeros((tile_size, tile_size), dtype=bool)
            mask[by:by + bh, bx:bx + bw] = True
        elif isinstance(seg, list):
            polys = seg if isinstance(seg[0], list) else [seg]
            mask = np.zeros((tile_size, tile_size), dtype=np.uint8)
            for poly in polys:
                if len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                pts[:, :, 0] = pts[:, :, 0] - x0
                pts[:, :, 1] = pts[:, :, 1] - y0
                cv2.fillPoly(mask, [pts], 1)
            mask = mask.astype(bool)
        else:
            continue

        if mask.sum() > 0:
            tile_instances.append({"category_id": cat_id, "mask": mask, "area": float(mask.sum())})

    return tile_instances


def stitch_prob_maps(tile_results: list, full_H: int, full_W: int,
                     tile_size: int = 896, stride: int = 640) -> np.ndarray:
    """将 tile 级概率图拼接回全图 | Stitch tile probability maps to full image.

    重叠区域取平均值 | Overlap regions averaged.
    """
    acc = np.zeros((full_H, full_W), dtype=np.float64)
    weight = np.zeros((full_H, full_W), dtype=np.float64)

    for tr in tile_results:
        y0, x0, h, w = tr["y0"], tr["x0"], tr["h"], tr["w"]
        prob = tr["prob"]  # [tile_size, tile_size] or [h, w]
        if prob.shape != (h, w):
            prob = prob[:h, :w]

        # Linear ramp weight for smooth blending at edges
        wy = np.minimum(np.arange(h, dtype=np.float64), h - np.arange(h, dtype=np.float64) - 1)
        wx = np.minimum(np.arange(w, dtype=np.float64), w - np.arange(w, dtype=np.float64) - 1)
        wy = wy / max(wy.max(), 1.0)
        wx = wx / max(wx.max(), 1.0)
        w_map = np.outer(wy, wx)

        acc[y0:y0 + h, x0:x0 + w] += prob[:h, :w] * w_map
        weight[y0:y0 + h, x0:x0 + w] += w_map

    weight = np.maximum(weight, 1e-9)
    return (acc / weight).astype(np.float32)


def stitch_binary_masks(tile_results: list, full_H: int, full_W: int,
                        tile_size: int = 896, stride: int = 640) -> np.ndarray:
    """将 tile 级二值 mask 拼接回全图 (OR 逻辑) | Stitch binary masks (OR logic)."""
    full_mask = np.zeros((full_H, full_W), dtype=bool)
    for tr in tile_results:
        y0, x0, h, w = tr["y0"], tr["x0"], tr["h"], tr["w"]
        mask = tr.get("binary", tr.get("prob", np.zeros((h, w))) > 0.3)
        full_mask[y0:y0 + h, x0:x0 + w] |= mask[:h, :w]
    return full_mask


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Prototype Building | Prototype 构建
# ═══════════════════════════════════════════════════════════════════

def compute_support_prototype_vis(support_feats: list[dict], source: str = "p4") -> torch.Tensor:
    vectors = []
    for sf in support_feats:
        v = sf[source].mean(dim=(2, 3))
        v = F.normalize(v, p=2, dim=-1)
        vectors.append(v)
    proto = torch.stack(vectors).mean(dim=0)
    return F.normalize(proto, p=2, dim=-1)


def build_class_prototypes_vis(
    model, extract_features, class_index: dict, query_src: str,
    split: str, data_root: Path, k_shot: int,
    proto_source: str, device: str,
) -> dict:
    class_protos = {}
    for cls_id, src_to_tiles in class_index.items():
        support_pool = [(src, tile) for src, tiles in src_to_tiles.items()
                        if src != query_src for tile in tiles]
        if len(support_pool) < k_shot:
            continue
        rng = random.Random(42)
        chosen = rng.sample(support_pool, min(k_shot, len(support_pool)))
        support_imgs = [load_tile_image(tile_stem, data_root, split) for _, tile_stem in chosen]
        support_feats = extract_features(model, support_imgs, device)
        proto = compute_support_prototype_vis(support_feats, source=proto_source)
        class_protos[cls_id] = {"proto": proto.cpu().numpy()}
    return class_protos


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Visualization | 可视化
# ═══════════════════════════════════════════════════════════════════

def _draw_gt_overlay(ax, gt_data: dict):
    """在 ax 上绘制 GT 实例彩色轮廓."""
    for inst in gt_data["instances"]:
        mask = inst["mask"]
        rgba = list(CLASS_COLORS[inst["category_id"]])
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], color=rgba, linewidth=1.5)
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        overlay[mask > 0] = (*rgba[:3], 0.2)
        ax.imshow(overlay)


def _match_instances(pred_instances: list, gt_data: dict, iou_thr: float = 0.3):
    """贪心匹配 | Greedy matching. Returns (tp_count, fp_count, fn_count, matched_gt_indices)."""
    gt_matched = [False] * len(gt_data["instances"])
    tp = 0
    for pred in sorted(pred_instances, key=lambda x: x.get("score", 0), reverse=True):
        pm = pred["mask"]
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_data["instances"]):
            if gt_matched[j]:
                continue
            if pred.get("category_id", 1) != gt["category_id"]:
                continue
            inter = (pm & gt["mask"]).sum()
            union = (pm | gt["mask"]).sum()
            iou = inter / max(union, 1)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr:
            gt_matched[best_j] = True
            tp += 1
    return tp, len(pred_instances) - tp, len(gt_data["instances"]) - tp


def _draw_tp_fp_fn_overlay(ax, tp_mask, fp_mask, fn_mask, tp, fp, fn):
    """绘制 TP/FP/FN 三色叠加."""
    H, W = tp_mask.shape
    canvas = np.zeros((H, W, 4), dtype=np.float32)
    canvas[tp_mask] = [0, 1, 0, 0.5]   # 绿
    canvas[fp_mask] = [1, 0, 0, 0.5]   # 红
    canvas[fn_mask] = [0, 0, 1, 0.5]   # 蓝
    ax.imshow(canvas)
    ax.set_title(f"TP (绿) / FP (红) / FN (蓝)\nTP={tp}  FP={fp}  FN={fn}", fontsize=9)
    ax.legend(handles=[
        Patch(color='green', alpha=0.5, label=f'TP={tp}'),
        Patch(color='red', alpha=0.5, label=f'FP={fp}'),
        Patch(color='blue', alpha=0.5, label=f'FN={fn}'),
    ], loc='lower right', fontsize=7)


# ═══════════════════════════════════════════════════════════════════
# Full-Image Mode | 全图模式
# ═══════════════════════════════════════════════════════════════════

def run_full_image_mode(args, model, decoder, extract_features, ckpt, device):
    """全图模式: 原图 → 切 tile → 逐 tile 推理 → 拼接 → 可视化."""
    from tools.train.train_fewshot_allclass import extract_features as ef

    # ── Load full image + GT ──
    t0 = time.perf_counter()
    if not os.path.exists(args.image):
        print(f"  [FATAL] Image not found: {args.image}")
        sys.exit(1)
    full_img = np.array(Image.open(args.image).convert("RGB"))
    H_full, W_full = full_img.shape[:2]
    img_stem = os.path.splitext(os.path.basename(args.image))[0]
    print(f"\n  Full image: {img_stem} ({H_full}×{W_full})")

    # ── Load full-image GT ──
    # 关键: 不预渲染 mask (3304 实例 × 7MB = 23GB 内存爆炸)
    # Key: don't pre-render masks (3304 insts × 7MB each = 23GB RAM explosion)
    # 改为存 polygon/bbox → 匹配时按需渲染 | Store polygons → render on-demand for matching
    gt_anns_raw = []
    gt_full = {"instances": [], "class_ids": set(), "merged_mask": np.zeros((H_full, W_full), dtype=bool)}
    if args.full_gt and os.path.exists(args.full_gt):
        with open(args.full_gt) as f:
            full_coco = json.load(f)
        img_name = os.path.basename(args.image)
        img_id_to_file = {img["id"]: img["file_name"] for img in full_coco["images"]}
        file_to_id = {v: k for k, v in img_id_to_file.items()}
        target_id = file_to_id.get(img_name)
        if target_id is None:
            for fid, fname in img_id_to_file.items():
                if os.path.splitext(fname)[0] == img_stem:
                    target_id = fid
                    break

        if target_id is not None:
            for ann in full_coco.get("annotations", []):
                if ann.get("image_id") != target_id:
                    continue
                cat_id = ann.get("category_id", 0)
                if cat_id < 1 or cat_id > 15:
                    continue
                # 存 polygon/bbox 而非完整 mask | Store polygon/bbox, not full mask
                bbox = ann.get("bbox", [0, 0, 0, 0])
                area = ann.get("area", bbox[2] * bbox[3])
                gt_anns_raw.append({
                    "category_id": cat_id,
                    "segmentation": ann.get("segmentation", []),
                    "bbox": bbox, "area": float(area),
                })
                gt_full["class_ids"].add(cat_id)
            print(f"  Full GT: {len(gt_anns_raw)} instances in "
                  f"{len(gt_full['class_ids'])} classes (polygons only, masks rendered on-demand)")
        else:
            print(f"  [WARN] No GT found for {img_name} in {args.full_gt}")
    else:
        print(f"  [WARN] No full-image GT file (--full-gt). Showing predictions only.")
    print(f"  [TIMING] Image + GT loading: {time.perf_counter() - t0:.2f}s")

    # ── Tile the image ──
    t0 = time.perf_counter()
    tile_size = args.tile_size
    stride_val = args.stride
    tiles = tile_full_image(full_img, tile_size=tile_size, stride=stride_val)
    print(f"  Tiled into {len(tiles)} tiles ({tile_size}×{tile_size}, stride={stride_val})")
    print(f"  [TIMING] Tiling: {time.perf_counter() - t0:.2f}s")

    # ── Build prototypes (from tile-level index if available) ──
    t0 = time.perf_counter()
    class_protos = {}
    query_src = img_stem
    if args.data_root:
        data_root = Path(args.data_root)
        if data_root.exists():
            class_index = _build_class_index(data_root, args.split)
            print(f"  Class index: {len(class_index)} classes")
            class_protos = build_class_prototypes_vis(
                model, extract_features, class_index, query_src,
                args.split, data_root, args.k_shot, args.prototype_source, device,
            )
    if not class_protos:
        print("  [WARN] No class prototypes built. Using random prototypes (poor quality).")
        for cid in gt_full["class_ids"]:
            class_protos[cid] = {"proto": np.random.randn(640).astype(np.float32)}
        if not class_protos:
            class_protos[1] = {"proto": np.random.randn(640).astype(np.float32)}

    print(f"  Prototypes: {[ISAID_CLASSES.get(c, f'c{c}') for c in sorted(class_protos.keys())]}")
    print(f"  [TIMING] Prototype building: {time.perf_counter() - t0:.2f}s")

    # ── ⑤ Prototype-Query Similarity per class | Prototype-Query 相似度 ──
    print(f"\n  [⑤ Prototype-Query Cosine Similarity]")
    # Use first tile's features as query representation
    first_tile = tiles[0]["img"]
    pad_h, pad_w = (32 - first_tile.shape[0] % 32) % 32, (32 - first_tile.shape[1] % 32) % 32
    if pad_h or pad_w:
        first_tile_pad = np.pad(first_tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        first_tile_pad = first_tile
    with torch.no_grad():
        qf = extract_features(model, [first_tile_pad], device)[0]
    # mean over spatial dims (H,W) → [1, C] or [C]; handle both 3D and 4D
    q_feat = qf[args.prototype_source]
    if q_feat.dim() == 4:
        q_spatial = q_feat.mean(dim=(2, 3))  # [1, C]
    else:
        q_spatial = q_feat.mean(dim=(1, 2))  # [C]
    query_vec = F.normalize(q_spatial, p=2, dim=-1).cpu().numpy().squeeze()
    print(f"    {'Class':<22s} {'CosSim':>8s}")
    print(f"    {'-'*32}")
    proto_sims = {}
    for cls_id in sorted(class_protos.keys()):
        pv = np.asarray(class_protos[cls_id]["proto"]).squeeze()  # (C,)
        pv_norm = pv / (np.linalg.norm(pv) + 1e-8)
        sim = float(np.dot(pv_norm, query_vec))
        proto_sims[cls_id] = sim
        cname = ISAID_CLASSES.get(cls_id, f"c{cls_id}")
        bar = "█" * max(0, int(sim * 20)) + "░" * max(0, 20 - int(sim * 20))
        print(f"    {cname:<22s} {sim:>8.4f}  {bar}")

    # ── Per-tile inference (batched backbone forward) ──
    # 每 tile 推理（批量 backbone forward）
    print(f"\n  Running inference on {len(tiles)} tiles (batched backbone)...")
    all_tile_results = []
    t_infer_start = time.perf_counter()

    # Stack all tiles → single batch for backbone
    B = len(tiles)
    batch_imgs = []
    for ti in tiles:
        t_img = ti["img"]  # already tile_size×tile_size (896=28×32)
        batch_imgs.append(torch.from_numpy(t_img).permute(2, 0, 1).float().div_(255.0))
    batch_tensor = torch.stack(batch_imgs).to(device)  # [B, 3, 896, 896]

    with torch.no_grad():
        batched_feats = _extract_features_batched(model, batch_tensor, device, no_grad=True)

    # Per-tile decoder processing
    for ti, tile_info in enumerate(tiles):
        tile_img = tile_info["img"]
        y0, x0, h, w = tile_info["y0"], tile_info["x0"], tile_info["h"], tile_info["w"]
        # Extract single-tile features from batch
        feats = {
            "p3": batched_feats["p3"][ti],     # [C, H/8, W/8]
            "p4": batched_feats["p4"][ti],     # [C, H/16, W/16]
            "proto": batched_feats["proto"][ti],  # [32, H/4, W/4]
        }

        with torch.no_grad():
            if args.decoder == "center_affinity":
                # Class-agnostic center + offset
                first_pk = list(class_protos.values())[0]
                proto_vec = torch.from_numpy(first_pk["proto"]).float().to(device)
                center_hm, offset_field, _ = decoder(
                    feats["p3"].unsqueeze(0), feats["p4"].unsqueeze(0),
                    feats["proto"].unsqueeze(0), proto_vec,
                )
                center_np = F.interpolate(
                    center_hm.unsqueeze(0).unsqueeze(0) if center_hm.dim() == 2 else center_hm.unsqueeze(0),
                    size=(tile_size, tile_size), mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy()
                # Per-class proto mask → aggregate
                best_prob = np.zeros((tile_size, tile_size), dtype=np.float32)
                for cls_id, pk in class_protos.items():
                    proto_vec_c = torch.from_numpy(pk["proto"]).float().to(device)
                    proto_mask = decoder.forward_proto_only(
                        feats["proto"].unsqueeze(0), proto_vec_c,
                    )
                    pm = F.interpolate(
                        proto_mask.unsqueeze(0).unsqueeze(0),
                        size=(tile_size, tile_size), mode="bilinear", align_corners=False,
                    ).squeeze().cpu().numpy()
                    best_prob = np.maximum(best_prob, pm)
                tile_prob = best_prob
            elif args.decoder == "dynamic_kernel":
                best_prob = np.zeros((tile_size, tile_size), dtype=np.float32)
                for cls_id, pk in class_protos.items():
                    proto_vec = torch.from_numpy(pk["proto"]).float().to(device)
                    masks_s8, proto_mask = decoder(
                        feats["p3"].unsqueeze(0), feats["p4"].unsqueeze(0),
                        feats["proto"].unsqueeze(0), proto_vec,
                    )
                    kernel_max = masks_s8.max(dim=0)[0]
                    pm = F.interpolate(
                        kernel_max.unsqueeze(0).unsqueeze(0),
                        size=(tile_size, tile_size), mode="bilinear", align_corners=False,
                    ).squeeze().cpu().numpy()
                    best_prob = np.maximum(best_prob, pm)
                tile_prob = best_prob
            else:
                best_prob = np.zeros((tile_size, tile_size), dtype=np.float32)
                for cls_id, pk in class_protos.items():
                    proto_vec = torch.from_numpy(pk["proto"]).float().to(device)
                    if args.decoder == "adaptive":
                        out = decoder(feats["p4"].unsqueeze(0), feats["proto"].unsqueeze(0), proto_vec)
                    else:  # adaptive-p3p4
                        out = decoder(feats["p3"].unsqueeze(0), feats["p4"].unsqueeze(0),
                                      feats["proto"].unsqueeze(0), proto_vec)
                    if out.dim() == 2:
                        out = out.unsqueeze(0).unsqueeze(0)
                    elif out.dim() == 3:
                        out = out.unsqueeze(0)
                    pm = F.interpolate(
                        out, size=(tile_size, tile_size), mode="bilinear", align_corners=False,
                    ).squeeze().cpu().numpy()
                    best_prob = np.maximum(best_prob, pm)
                tile_prob = best_prob

        all_tile_results.append({
            "y0": y0, "x0": x0, "h": h, "w": w, "prob": tile_prob,
            "center": center_np if args.decoder == "center_affinity" else None,
        })

        if (ti + 1) % max(1, len(tiles) // 5) == 0:
            print(f"    {ti + 1}/{len(tiles)} tiles done")

    t_infer = time.perf_counter() - t_infer_start
    print(f"  [TIMING] Per-tile inference: {t_infer:.1f}s total, "
          f"{t_infer / len(tiles):.2f}s/tile ({len(tiles)} tiles, batched backbone)")

    # ── Stitch to full image ──
    t0 = time.perf_counter()
    print(f"\n  Stitching {len(all_tile_results)} tile predictions...")
    full_prob = stitch_prob_maps(all_tile_results, H_full, W_full, tile_size, stride_val)
    full_binary = full_prob > args.score_thr
    # Stitch center heatmap (center_affinity only)
    full_center = None
    if args.decoder == "center_affinity":
        center_results = [{"y0": tr["y0"], "x0": tr["x0"], "h": tr["h"], "w": tr["w"],
                           "prob": tr["center"]} for tr in all_tile_results]
        full_center = stitch_prob_maps(center_results, H_full, W_full, tile_size, stride_val)
    print(f"  [TIMING] Stitching: {time.perf_counter() - t0:.2f}s")

    # ── Generate instances from stitched prob map ──
    # 全图尺寸大, 用 connected_components 避免 watershed 卡死
    # Full-image scale: use connected_components to avoid watershed hang on large maps
    from adatile.metrics.instance_generation import generate_instances
    t0 = time.perf_counter()
    print(f"  Generating instances (method=connected_components, thr={args.score_thr})...")
    pred_instances = generate_instances(
        full_prob, method="connected_components",
        score_thr=args.score_thr, min_area=args.min_area,
    )
    print(f"  Generated {len(pred_instances)} instance predictions")
    print(f"  [TIMING] Instance generation: {time.perf_counter() - t0:.2f}s")

    # ── Compute metrics ──
    # 按需渲染 + bbox 预筛选: 避免 O(P×G) 次完整 mask 渲染
    # On-demand rendering + bbox pre-filter: avoids O(P×G) full mask renders
    t0 = time.perf_counter()
    tp, fp_count, fn_count = 0, 0, 0
    tp_mask = np.zeros((H_full, W_full), dtype=bool)
    fp_mask = np.zeros((H_full, W_full), dtype=bool)
    fn_mask = np.zeros((H_full, W_full), dtype=bool)

    if gt_anns_raw:
        n_gt = len(gt_anns_raw)
        gt_matched = [False] * n_gt

        for pred in sorted(pred_instances, key=lambda x: x.get("score", 0), reverse=True):
            pm = pred["mask"]
            # 全图模式预测是 class-agnostic (聚合 prob map), 匹配时不检查类别
            # Full-image preds are class-agnostic → no category check in matching
            # 获取 pred 的 bbox | Get prediction bbox
            pm_ys, pm_xs = np.where(pm)
            if len(pm_ys) == 0:
                fp_count += 1; fp_mask |= pm; continue
            p_by, p_bx = int(pm_ys.min()), int(pm_xs.min())
            p_ey, p_ex = int(pm_ys.max()) + 1, int(pm_xs.max()) + 1

            best_iou, best_j = 0.0, -1
            for j in range(n_gt):
                if gt_matched[j]:
                    continue
                gt_ann = gt_anns_raw[j]
                # (no category check — predictions are class-agnostic)
                # Bbox 预筛选: 不重叠 → IoU=0, 跳过昂贵的 mask 渲染
                # Bbox pre-filter: no overlap → IoU=0, skip expensive mask render
                gx, gy, gw, gh = [int(v) for v in gt_ann["bbox"]]
                if (p_ex <= gx or p_bx >= gx + gw or
                    p_ey <= gy or p_by >= gy + gh):
                    continue
                # ROI 渲染: 只在 pred+GT 交叠区域渲染, 比全图画布快 100-1000×
                # ROI render: only in pred+GT overlap region, 100-1000× faster
                rx = max(p_bx, gx); ry = max(p_by, gy)
                rw = min(p_ex, gx + gw) - rx
                rh = min(p_ey, gy + gh) - ry
                if rw <= 0 or rh <= 0:
                    continue
                gt_roi = np.zeros((rh, rw), dtype=np.uint8)
                seg = gt_ann.get("segmentation", [])
                if seg:
                    polys = seg if isinstance(seg[0], list) else [seg]
                    for poly in polys:
                        if len(poly) < 6:
                            continue
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                        pts[:, :, 0] = pts[:, :, 0] - rx
                        pts[:, :, 1] = pts[:, :, 1] - ry
                        cv2.fillPoly(gt_roi, [pts], 1)
                gt_roi = gt_roi.astype(bool)
                pred_roi = pm[ry:ry + rh, rx:rx + rw]
                inter = (pred_roi & gt_roi).sum()
                union = (pred_roi | gt_roi).sum()
                iou = inter / max(union, 1) if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= args.iou_thr:
                gt_matched[best_j] = True
                tp += 1
                tp_mask |= pm
            else:
                fp_count += 1
                fp_mask |= pm

        # FN: ROI 渲染 (小目标只渲染 bbox 区域, 快 1000×) | ROI render for speed
        for j in range(n_gt):
            if not gt_matched[j]:
                fn_count += 1
                gt_ann = gt_anns_raw[j]
                gx, gy, gw, gh = [int(v) for v in gt_ann["bbox"]]
                gx, gy = max(0, gx), max(0, gy)
                gw = min(gw, W_full - gx)
                gh = min(gh, H_full - gy)
                if gw <= 0 or gh <= 0:
                    continue
                # 在 ROI 内渲染, 比全图画布快 ~1000× | Render in ROI, ~1000× faster
                roi = np.zeros((gh, gw), dtype=np.uint8)
                seg = gt_ann.get("segmentation", [])
                if seg:
                    polys = seg if isinstance(seg[0], list) else [seg]
                    for poly in polys:
                        if len(poly) < 6:
                            continue
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                        pts[:, :, 0] = pts[:, :, 0] - gx
                        pts[:, :, 1] = pts[:, :, 1] - gy
                        cv2.fillPoly(roi, [pts], 1)
                fn_mask[gy:gy + gh, gx:gx + gw] |= roi.astype(bool)

        precision = tp / max(tp + fp_count, 1)
        recall = tp / max(tp + fn_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(f"  [TIMING] Metrics computation: {time.perf_counter() - t0:.2f}s")
        print(f"\n  Full-Image Metrics (IoU≥{args.iou_thr}):")
        print(f"    TP={tp}, FP={fp_count}, FN={fn_count}")
        print(f"    Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        # ── ① Per-class Recall | 按类别统计 Recall ──
        print(f"\n  [① Per-Class Recall]")
        n_per_class = defaultdict(int)
        n_matched_per_class = defaultdict(int)
        for j, gt_ann in enumerate(gt_anns_raw):
            n_per_class[gt_ann["category_id"]] += 1
            if gt_matched[j]:
                n_matched_per_class[gt_ann["category_id"]] += 1
        print(f"    {'Class':<22s} {'Recall':>7s}  {'Matched':>8s}  {'GT':>6s}")
        print(f"    {'-'*50}")
        for cls_id in sorted(gt_full["class_ids"]):
            n_gt_c = n_per_class.get(cls_id, 0)
            n_tp_c = n_matched_per_class.get(cls_id, 0)
            r_c = n_tp_c / max(n_gt_c, 1)
            cname = ISAID_CLASSES.get(cls_id, f"c{cls_id}")
            bar = "█" * int(r_c * 20) + "░" * (20 - int(r_c * 20))
            print(f"    {cname:<22s} {r_c:6.1%}  {n_tp_c:>8d}  {n_gt_c:>6d}  {bar}")

        # ── ② Area-bucket Recall | 按目标面积统计 Recall ──
        print(f"\n  [② Area-Bucket Recall]")
        area_buckets = [(0, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, 999999)]
        area_n = defaultdict(int)
        area_tp = defaultdict(int)
        for j, gt_ann in enumerate(gt_anns_raw):
            a = gt_ann["area"]
            for lo, hi in area_buckets:
                if lo <= a < hi:
                    area_n[(lo, hi)] += 1
                    if gt_matched[j]:
                        area_tp[(lo, hi)] += 1
                    break
        print(f"    {'Area Range':<16s} {'Recall':>7s}  {'Matched':>8s}  {'GT':>6s}")
        print(f"    {'-'*45}")
        for lo, hi in area_buckets:
            n_a = area_n.get((lo, hi), 0)
            tp_a = area_tp.get((lo, hi), 0)
            if n_a > 0:
                r_a = tp_a / n_a
                bar = "█" * int(r_a * 20) + "░" * (20 - int(r_a * 20))
                print(f"    [{lo:>4d}, {hi:>6d})  {r_a:6.1%}  {tp_a:>8d}  {n_a:>6d}  {bar}")

        # ── ③ Score Distribution | 预测分数分布 ──
        print(f"\n  [③ Prediction Score Distribution]")
        if pred_instances:
            scores = np.array([p.get("score", 0) for p in pred_instances])
            print(f"    n_pred={len(scores)}, min={scores.min():.4f}, max={scores.max():.4f}, "
                  f"mean={scores.mean():.4f}, std={scores.std():.4f}")
            # Histogram bins
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            print(f"    Score histogram:")
            for lo, hi in zip(bins[:-1], bins[1:]):
                n = ((scores >= lo) & (scores < hi)).sum()
                if n > 0:
                    bar = "█" * n
                    print(f"      [{lo:.1f}, {hi:.1f}): {n:>4d}  {bar}")

        # ── ④ Probability Map Statistics | 概率图统计 ──
        print(f"\n  [④ Probability Map Stats]")
        print(f"    Stitched prob map: min={full_prob.min():.6f}, max={full_prob.max():.4f}, "
              f"mean={full_prob.mean():.6f}, median={np.median(full_prob):.6f}")
        # Percentiles
        for pct in [50, 90, 95, 99, 99.9]:
            v = np.percentile(full_prob, pct)
            print(f"    P{pct:>4.1f}: {v:.6f}")
        # Fraction of pixels above common thresholds
        for thr in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            frac = (full_prob > thr).mean()
            print(f"    frac > {thr:.1f}: {frac:.4%} ({int(frac * full_prob.size):,} px)")

    # ── Generate full-image visualization ──
    t0 = time.perf_counter()
    out_path = os.path.join(args.output, f"{img_stem}_full_summary.png")
    _plot_full_image_summary(
        full_img, img_stem, gt_anns_raw, gt_full["class_ids"], pred_instances,
        full_prob, full_binary, full_center,
        tp, fp_count, fn_count, tp_mask, fp_mask, fn_mask,
        H_full, W_full, out_path,
    )
    print(f"\n  [SAVED] {out_path}")
    print(f"  [TIMING] Visualization rendering: {time.perf_counter() - t0:.2f}s")


def _plot_full_image_summary(full_img, stem, gt_anns_raw, gt_class_ids, pred_insts,
                             prob, binary, center_hm, tp, fp_count, fn,
                             tp_mask, fp_mask, fn_mask, H, W, out_path):
    """全图综合可视化: 2×3 grid | Full-image summary: 2×3 grid.

    GT 从 polygon 按需渲染, 不预存 mask | GT rendered from polygons on-demand.
    """
    n_gt = len(gt_anns_raw)

    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    fig.suptitle(f"Full-Image Prediction — {stem} ({H}×{W})", fontsize=14, fontweight="bold")

    # [0,0] Original + GT overlay (polygon contours, no full mask render)
    axes[0, 0].imshow(full_img)
    for gt_ann in gt_anns_raw:
        rgba = list(CLASS_COLORS[gt_ann["category_id"]])
        seg = gt_ann.get("segmentation", [])
        if seg:
            polys = seg if isinstance(seg[0], list) else [seg]
            for poly in polys:
                if len(poly) < 6:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                axes[0, 0].plot(pts[:, 0, 0], pts[:, 0, 1], color=rgba, linewidth=0.5, alpha=0.6)
    gt_classes = [ISAID_CLASSES.get(c, f"c{c}") for c in sorted(gt_class_ids)]
    axes[0, 0].set_title(f"Original + GT ({n_gt} insts)\n"
                         f"Classes: {', '.join(gt_classes[:6])}", fontsize=9)
    axes[0, 0].axis("off")

    # [0,1] Center Heatmap + GT Centers | 中心热力图 + GT 中心点
    axes[0, 1].imshow(full_img, alpha=0.4)
    if center_hm is not None:
        axes[0, 1].imshow(center_hm, cmap="hot", alpha=0.6, vmin=0, vmax=max(center_hm.max(), 0.01))
        c_max, c_mean = center_hm.max(), center_hm.mean()
    else:
        c_max, c_mean = 0, 0
    # Draw GT centers: color by class, size by area
    for gt_ann in gt_anns_raw:
        seg = gt_ann.get("segmentation", [])
        if not seg:
            continue
        polys = seg if isinstance(seg[0], list) else [seg]
        for poly in polys:
            if len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            cy, cx = pts[:, 0, 1].mean(), pts[:, 0, 0].mean()
            cat_id = gt_ann["category_id"]
            rgba = list(CLASS_COLORS[cat_id])
            sz = max(3, min(15, np.sqrt(gt_ann["area"]) / 8))
            axes[0, 1].plot(cx, cy, 'o', color=rgba, markersize=sz, markeredgewidth=0.5,
                           markerfacecolor='none', alpha=0.7)
    axes[0, 1].set_title(f"Center Heatmap + GT Centers\n"
                         f"HM max={c_max:.3f}, mean={c_mean:.4f}\n"
                         f"Circles = GT centers (color=class, size∝√area)", fontsize=9)
    axes[0, 1].axis("off")

    # [0,2] Predicted prob map
    axes[0, 2].imshow(prob, cmap="hot", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Stitched Prob Map\nmax={prob.max():.3f}, mean={prob.mean():.4f}", fontsize=9)
    axes[0, 2].axis("off")

    # [1,0] Binary prediction
    axes[1, 0].imshow(full_img, alpha=0.3)
    axes[1, 0].imshow(binary, cmap="gray", alpha=0.6)
    axes[1, 0].set_title(f"Binary >0.3\nFG area={binary.sum():,} px ({100 * binary.mean():.1f}%)", fontsize=9)
    axes[1, 0].axis("off")

    # [1,1] Predicted instances
    axes[1, 1].imshow(full_img, alpha=0.4)
    pred_label = np.zeros((H, W), dtype=np.int32)
    for idx, inst in enumerate(pred_insts, 1):
        if inst["mask"].shape == (H, W):
            pred_label[inst["mask"]] = idx
    axes[1, 1].imshow(pred_label, cmap="tab20", alpha=0.7, vmin=0, vmax=max(1, len(pred_insts)))
    axes[1, 1].set_title(f"Predicted Instances\n{len(pred_insts)} instances", fontsize=9)
    axes[1, 1].axis("off")

    # [1,2] TP/FP/FN or stats
    if tp_mask is not None and n_gt > 0:
        axes[1, 2].imshow(full_img, alpha=0.4)
        _draw_tp_fp_fn_overlay(axes[1, 2], tp_mask, fp_mask, fn_mask, tp, fp_count, fn)
    else:
        axes[1, 2].axis("off")
        axes[1, 2].text(0.5, 0.5, f"{len(pred_insts)} predicted instances\n"
                                  f"(no GT available for metrics)",
                        ha='center', va='center', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Tile Mode (kept from v2) | Tile 模式
# ═══════════════════════════════════════════════════════════════════

def run_tile_mode(args, model, decoder, extract_features, ckpt, device):
    """Tile 模式: 单 tile 可视化 (GT 来自 tile-level COCO)."""
    data_root = Path(args.data_root)
    print(f"\n  Loading GT for {args.query_tile}...")
    gt_data = load_gt_for_tile(args.query_tile, data_root, args.split)
    print(f"  GT: {len(gt_data['instances'])} instances in "
          f"{len(gt_data['class_ids'])} classes: "
          f"{[ISAID_CLASSES.get(c, f'c{c}') for c in sorted(gt_data['class_ids'])]}")

    tile_img = load_tile_image(args.query_tile, data_root, args.split)
    H, W = tile_img.shape[:2]
    print(f"  Tile: {H}×{W}")

    # Build prototypes
    class_index = _build_class_index(data_root, args.split)
    query_src = _extract_source_image(args.query_tile)
    class_protos = build_class_prototypes_vis(
        model, extract_features, class_index, query_src,
        args.split, data_root, args.k_shot, args.prototype_source, device,
    )
    print(f"  Prototypes: {[ISAID_CLASSES.get(c, f'c{c}') for c in sorted(class_protos.keys())]}")

    # Pad + extract features
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    img_pad = np.pad(tile_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect") if pad_h or pad_w else tile_img
    with torch.no_grad():
        feats = extract_features(model, [img_pad], device)[0]

    # ── Inference ──
    all_pred_instances = []
    best_proto = None

    for cls_id, pk in class_protos.items():
        proto_vec = torch.from_numpy(pk["proto"]).float().to(device)
        with torch.no_grad():
            if args.decoder == "center_affinity":
                proto_mask = decoder.forward_proto_only(
                    feats["proto"].unsqueeze(0) if feats["proto"].dim() == 3 else feats["proto"],
                    proto_vec,
                )
                pm = F.interpolate(
                    proto_mask.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy()
                if best_proto is None or pm.max() > (best_proto[1] if best_proto else -1):
                    best_proto = (pm, pm.max())
                fg = pm > 0.3
                from adatile.metrics.instance_generation import generate_instances
                insts = generate_instances(pm, method="watershed_distance",
                                          score_thr=0.3, min_area=16, min_distance=12)
                for it in insts:
                    it["category_id"] = cls_id
                all_pred_instances.extend(insts)
            elif args.decoder == "dynamic_kernel":
                masks_s8, proto_mask = decoder(
                    feats["p3"].unsqueeze(0) if feats["p3"].dim() == 3 else feats["p3"],
                    feats["p4"].unsqueeze(0) if feats["p4"].dim() == 3 else feats["p4"],
                    feats["proto"].unsqueeze(0) if feats["proto"].dim() == 3 else feats["proto"],
                    proto_vec,
                )
                km = masks_s8.max(dim=0)[0]
                pm = F.interpolate(
                    km.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy()
                if best_proto is None or pm.max() > (best_proto[1] if best_proto else -1):
                    best_proto = (pm, pm.max())
                from adatile.metrics.instance_generation import generate_instances
                insts = generate_instances(pm, method="watershed_distance",
                                          score_thr=0.3, min_area=16, min_distance=12)
                for it in insts:
                    it["category_id"] = cls_id
                all_pred_instances.extend(insts)
            else:
                if args.decoder == "adaptive":
                    out = decoder(feats["p4"], feats["proto"], proto_vec)
                else:
                    out = decoder(feats["p3"], feats["p4"], feats["proto"], proto_vec)
                if out.dim() == 2:
                    out = out.unsqueeze(0).unsqueeze(0)
                elif out.dim() == 3:
                    out = out.unsqueeze(0)
                pm = F.interpolate(
                    out, size=(H, W), mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy()
                if best_proto is None or pm.max() > (best_proto[1] if best_proto else -1):
                    best_proto = (pm, pm.max())
                from adatile.metrics.instance_generation import generate_instances
                insts = generate_instances(pm, method="watershed_distance",
                                          score_thr=0.3, min_area=16, min_distance=12)
                for it in insts:
                    it["category_id"] = cls_id
                all_pred_instances.extend(insts)

    prob_best = best_proto[0] if best_proto else np.zeros((H, W))

    # Compute TP/FP/FN
    tp, fp_count, fn_count = 0, 0, 0
    tp_mask = np.zeros((H, W), dtype=bool)
    fp_mask = np.zeros((H, W), dtype=bool)
    fn_mask = np.zeros((H, W), dtype=bool)
    if gt_data["instances"]:
        gt_matched = [False] * len(gt_data["instances"])
        for pred in sorted(all_pred_instances, key=lambda x: x.get("score", 0), reverse=True):
            pm = pred["mask"]
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_data["instances"]):
                if gt_matched[j]:
                    continue
                if pred.get("category_id", 1) != gt["category_id"]:
                    continue
                inter = (pm & gt["mask"]).sum()
                union = (pm | gt["mask"]).sum()
                iou = inter / max(union, 1)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= args.iou_thr:
                gt_matched[best_j] = True
                tp += 1
                tp_mask |= pm
            else:
                fp_count += 1
                fp_mask |= pm
        for j, gt in enumerate(gt_data["instances"]):
            if not gt_matched[j]:
                fn_count += 1
                fn_mask |= gt["mask"]

    # ── Generate figure ──
    out_path = os.path.join(args.output, f"{args.query_tile}_summary.png")
    _plot_tile_summary(
        tile_img, args.query_tile, gt_data, all_pred_instances, prob_best,
        tp, fp_count, fn_count, tp_mask, fp_mask, fn_mask, H, W, out_path,
    )
    print(f"  [SAVED] {out_path}")


def _plot_tile_summary(img, stem, gt, pred_insts, prob, tp, fp_count, fn,
                       tp_mask, fp_mask, fn_mask, H, W, out_path):
    """Tile 综合可视化: 2×3 grid."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Tile Prediction — {stem}", fontsize=14, fontweight="bold")

    # [0,0] Input + GT
    axes[0, 0].imshow(img)
    _draw_gt_overlay(axes[0, 0], gt)
    gt_classes = [ISAID_CLASSES.get(c, f"c{c}") for c in sorted(gt["class_ids"])]
    axes[0, 0].set_title(f"Input + GT ({len(gt['instances'])} insts)\n{', '.join(gt_classes)}",
                         fontsize=9)
    axes[0, 0].axis("off")

    # [0,1] GT Masks
    axes[0, 1].imshow(img, alpha=0.3)
    gt_label = np.zeros((H, W), dtype=np.int32)
    for idx, inst in enumerate(gt["instances"], 1):
        gt_label[inst["mask"]] = idx
    axes[0, 1].imshow(gt_label, cmap="tab20", alpha=0.7, vmin=0, vmax=max(1, len(gt["instances"])))
    axes[0, 1].set_title(f"GT Instances ({len(gt['instances'])})", fontsize=9)
    axes[0, 1].axis("off")

    # [0,2] Best prob map
    axes[0, 2].imshow(prob, cmap="hot", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Best Prob Map\nmax={prob.max():.3f}, mean={prob.mean():.4f}", fontsize=9)
    axes[0, 2].axis("off")

    # [1,0] Binary
    binary = prob > 0.3
    axes[1, 0].imshow(img, alpha=0.3)
    axes[1, 0].imshow(binary, cmap="gray", alpha=0.6)
    axes[1, 0].set_title(f"Binary >0.3\nFG={binary.sum():,} px", fontsize=9)
    axes[1, 0].axis("off")

    # [1,1] Pred instances
    axes[1, 1].imshow(img, alpha=0.4)
    pred_label = np.zeros((H, W), dtype=np.int32)
    for idx, inst in enumerate(pred_insts, 1):
        if inst["mask"].shape == (H, W):
            pred_label[inst["mask"]] = idx
    axes[1, 1].imshow(pred_label, cmap="tab20", alpha=0.7, vmin=0, vmax=max(1, len(pred_insts)))
    axes[1, 1].set_title(f"Pred Instances ({len(pred_insts)})", fontsize=9)
    axes[1, 1].axis("off")

    # [1,2] TP/FP/FN
    axes[1, 2].imshow(img, alpha=0.4)
    if gt["instances"]:
        _draw_tp_fp_fn_overlay(axes[1, 2], tp_mask, fp_mask, fn_mask, tp, fp_count, fn)
    else:
        axes[1, 2].text(0.5, 0.5, f"{len(pred_insts)} predictions\n(no GT)", ha='center',
                        va='center', fontsize=12, transform=axes[1, 2].transAxes)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# Batched Feature Extraction | 批量特征提取
# ═══════════════════════════════════════════════════════════════════

def _extract_features_batched(model, batch_tensor: "torch.Tensor",
                               device: str = "cuda", no_grad: bool = True) -> dict:
    """
    批量提取 FastSAM backbone 特征 (所有 tile 一次 forward).
    Batched feature extraction — all tiles in one forward pass.

    与 extract_features 不同, 此函数接受 [B, 3, H, W] 的 batch tensor,
    一次 forward 捕获 P3/P4/P8/Proto, 大幅提升 GPU 利用率.

    :param model: FastSAM model.
    :param batch_tensor: [B, 3, H, W] float32 tensor on device.
    :param device: "cuda" or "cpu".
    :param no_grad: If True, use torch.no_grad().
    :return: {p3: [B,C,H/8,W/8], p4: [B,C,H/16,W/16], p8: [B,C,H/32,W/32], proto: [B,32,H/4,W/4]}
    """
    seg = model.model          # SegmentationModel
    seq = seg.model            # Sequential[23]
    save_set = set(seg.save)   # {4, 6, 9, 12, 15, 18, 21}
    segment = seq[22]          # Segment head

    # Hooks
    hooked = {}

    def _hook(name):
        def _fn(m, inp, outp):
            hooked[name] = outp.detach() if no_grad else outp
        return _fn

    handles = [
        seq[15].register_forward_hook(_hook("p3")),
        seq[18].register_forward_hook(_hook("p4")),
        seq[21].register_forward_hook(_hook("p8")),
    ]

    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        x = batch_tensor
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

    # Proto masks from P3 (same ctx as forward)
    with ctx:
        proto = segment.proto(p3)  # [B, 32, H/4, W/4]

    return {"p3": p3, "p4": p4, "p8": p8, "proto": proto}


# Main | 主流程
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# Batch Mode | 批量模式
# ═══════════════════════════════════════════════════════════════════

def _process_one_image_metrics(args, model, decoder, extract_features, device,
                                img_path: str) -> dict | None:
    """处理单张全图, 返回指标字典 (不生成可视化) | Process one image, return metrics dict."""
    if not os.path.exists(img_path):
        return None
    stem = os.path.splitext(os.path.basename(img_path))[0]

    try:
        full_img = np.array(Image.open(img_path).convert("RGB"))
    except Exception:
        return None
    H_full, W_full = full_img.shape[:2]

    # Load GT
    gt_anns_raw = []
    gt_class_ids = set()
    if args.full_gt and os.path.exists(args.full_gt):
        with open(args.full_gt) as f:
            full_coco = json.load(f)
        img_name = os.path.basename(img_path)
        img_id_to_file = {img["id"]: img["file_name"] for img in full_coco["images"]}
        file_to_id = {v: k for k, v in img_id_to_file.items()}
        target_id = file_to_id.get(img_name)
        if target_id is None:
            for fid, fname in img_id_to_file.items():
                if os.path.splitext(fname)[0] == stem:
                    target_id = fid; break
        if target_id is not None:
            for ann in full_coco.get("annotations", []):
                if ann.get("image_id") != target_id:
                    continue
                cat_id = ann.get("category_id", 0)
                if cat_id < 1 or cat_id > 15:
                    continue
                bbox = ann.get("bbox", [0, 0, 0, 0])
                gt_anns_raw.append({
                    "category_id": cat_id,
                    "segmentation": ann.get("segmentation", []),
                    "bbox": bbox, "area": float(ann.get("area", bbox[2] * bbox[3])),
                })
                gt_class_ids.add(cat_id)

    # Tile
    tile_size = args.tile_size
    stride_val = args.stride
    tiles = tile_full_image(full_img, tile_size=tile_size, stride=stride_val)

    # Build prototypes (cached after first image)
    query_src = stem
    class_protos = getattr(_process_one_image_metrics, "_proto_cache", None)
    if class_protos is None:
        class_protos = {}
        if args.data_root:
            data_root = Path(args.data_root)
            if data_root.exists():
                class_index = _build_class_index(data_root, args.split)
                class_protos = build_class_prototypes_vis(
                    model, extract_features, class_index, query_src,
                    args.split, data_root, args.k_shot, args.prototype_source, device,
                )
        if not class_protos:
            for cid in gt_class_ids:
                class_protos[cid] = {"proto": np.random.randn(640).astype(np.float32)}
            if not class_protos:
                class_protos[1] = {"proto": np.random.randn(640).astype(np.float32)}
        _process_one_image_metrics._proto_cache = class_protos

    # ── Batched backbone forward: stack all tiles → single pass ──
    # 批量 backbone forward：所有 tile 堆叠 → 一次前向
    B = len(tiles)
    batch_imgs = []
    for ti in tiles:
        # tile_full_image() already pads to tile_size (896=28×32), no extra pad needed
        t_img = ti["img"]
        batch_imgs.append(torch.from_numpy(t_img).permute(2, 0, 1).float().div_(255.0))
    batch_tensor = torch.stack(batch_imgs).to(device)  # [B, 3, 896, 896]

    with torch.no_grad():
        batched_feats = _extract_features_batched(model, batch_tensor, device, no_grad=True)
    # batched_feats: {p3:[B,C,H/8,W/8], p4:[B,C,H/16,W/16], proto:[B,32,H/4,W/4]}

    # ── Per-tile decoder processing (lightweight, < 5ms per tile) ──
    # 逐 tile decoder 处理（轻量, < 5ms/tile）
    all_tile_results = []
    for i, tile_info in enumerate(tiles):
        y0, x0, h, w = tile_info["y0"], tile_info["x0"], tile_info["h"], tile_info["w"]
        # Extract single-tile features from batch
        feats = {
            "p3": batched_feats["p3"][i],     # [C, H/8, W/8]
            "p4": batched_feats["p4"][i],     # [C, H/16, W/16]
            "proto": batched_feats["proto"][i],  # [32, H/4, W/4]
        }

        with torch.no_grad():
            best_prob = np.zeros((tile_size, tile_size), dtype=np.float32)
            center_np = None
            if args.decoder == "center_affinity":
                first_pk = list(class_protos.values())[0]
                proto_vec = torch.from_numpy(first_pk["proto"]).float().to(device)
                center_hm, _, _ = decoder(
                    feats["p3"].unsqueeze(0), feats["p4"].unsqueeze(0),
                    feats["proto"].unsqueeze(0), proto_vec,
                )
                center_np = F.interpolate(
                    center_hm.unsqueeze(0).unsqueeze(0) if center_hm.dim() == 2 else center_hm.unsqueeze(0),
                    size=(tile_size, tile_size), mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy()
                for cls_id, pk in class_protos.items():
                    proto_vec_c = torch.from_numpy(pk["proto"]).float().to(device)
                    proto_mask = decoder.forward_proto_only(
                        feats["proto"].unsqueeze(0), proto_vec_c,
                    )
                    pm = F.interpolate(proto_mask.unsqueeze(0).unsqueeze(0),
                                      size=(tile_size, tile_size), mode="bilinear",
                                      align_corners=False).squeeze().cpu().numpy()
                    best_prob = np.maximum(best_prob, pm)
            else:
                for cls_id, pk in class_protos.items():
                    proto_vec = torch.from_numpy(pk["proto"]).float().to(device)
                    if args.decoder == "adaptive":
                        out = decoder(feats["p4"].unsqueeze(0), feats["proto"].unsqueeze(0), proto_vec)
                    else:
                        out = decoder(feats["p3"].unsqueeze(0), feats["p4"].unsqueeze(0),
                                      feats["proto"].unsqueeze(0), proto_vec)
                    if out.dim() == 2: out = out.unsqueeze(0).unsqueeze(0)
                    elif out.dim() == 3: out = out.unsqueeze(0)
                    pm = F.interpolate(out, size=(tile_size, tile_size),
                                      mode="bilinear", align_corners=False).squeeze().cpu().numpy()
                    best_prob = np.maximum(best_prob, pm)
            tile_prob = best_prob

        all_tile_results.append({
            "y0": y0, "x0": x0, "h": h, "w": w, "prob": tile_prob,
            "center": center_np,
        })

    # Stitch
    full_prob = stitch_prob_maps(all_tile_results, H_full, W_full, tile_size, stride_val)
    full_binary = full_prob > args.score_thr

    # Generate instances
    from adatile.metrics.instance_generation import generate_instances
    pred_instances = generate_instances(
        full_prob, method="connected_components",
        score_thr=args.score_thr, min_area=args.min_area,
    )

    # Match (class-agnostic)
    tp, fp_count, fn_count = 0, 0, 0
    gt_matched = [False] * len(gt_anns_raw) if gt_anns_raw else []
    if gt_anns_raw:
        for pred in sorted(pred_instances, key=lambda x: x.get("score", 0), reverse=True):
            pm = pred["mask"]
            pm_ys, pm_xs = np.where(pm)
            if len(pm_ys) == 0: fp_count += 1; continue
            p_by, p_bx = int(pm_ys.min()), int(pm_xs.min())
            p_ey, p_ex = int(pm_ys.max()) + 1, int(pm_xs.max()) + 1
            best_iou, best_j = 0.0, -1
            for j in range(len(gt_anns_raw)):
                if gt_matched[j]: continue
                gt_ann = gt_anns_raw[j]
                gx, gy, gw, gh = [int(v) for v in gt_ann["bbox"]]
                if p_ex <= gx or p_bx >= gx + gw or p_ey <= gy or p_by >= gy + gh:
                    continue
                rx, ry = max(p_bx, gx), max(p_by, gy)
                rw, rh = min(p_ex, gx + gw) - rx, min(p_ey, gy + gh) - ry
                if rw <= 0 or rh <= 0: continue
                gt_roi = np.zeros((rh, rw), dtype=np.uint8)
                seg = gt_ann.get("segmentation", [])
                if seg:
                    polys = seg if isinstance(seg[0], list) else [seg]
                    for poly in polys:
                        if len(poly) < 6: continue
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                        pts[:, :, 0] -= rx; pts[:, :, 1] -= ry
                        cv2.fillPoly(gt_roi, [pts], 1)
                gt_roi = gt_roi.astype(bool)
                pred_roi = pm[ry:ry + rh, rx:rx + rw]
                inter = (pred_roi & gt_roi).sum()
                union = (pred_roi | gt_roi).sum()
                iou = inter / max(union, 1) if union > 0 else 0.0
                if iou > best_iou: best_iou, best_j = iou, j
            if best_iou >= args.iou_thr:
                gt_matched[best_j] = True; tp += 1
            else:
                fp_count += 1
        fn_count = len(gt_anns_raw) - tp

    # Per-class
    n_per_class = defaultdict(int)
    n_matched_per_class = defaultdict(int)
    # Per-class + area-bucket stats
    n_per_class = defaultdict(int)
    n_matched_per_class = defaultdict(int)
    for j, gt_ann in enumerate(gt_anns_raw):
        n_per_class[gt_ann["category_id"]] += 1
        if gt_matched[j]:
            n_matched_per_class[gt_ann["category_id"]] += 1

    precision = tp / max(tp + fp_count, 1)
    recall = tp / max(tp + fn_count, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    area_buckets = _compute_area_bucket_stats(gt_anns_raw, gt_matched)

    return {
        "image": stem, "H": H_full, "W": W_full,
        "n_gt": len(gt_anns_raw), "n_pred": len(pred_instances),
        "n_classes_gt": len(gt_class_ids),
        "tp": tp, "fp": fp_count, "fn": fn_count,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "prob_max": float(full_prob.max()), "prob_mean": float(full_prob.mean()),
        "fg_frac": float((full_prob > 0.3).mean()),
        "per_class": {ISAID_CLASSES.get(k, f"c{k}"):
                      {"matched": n_matched_per_class.get(k, 0), "gt": v}
                      for k, v in n_per_class.items()},
        "area_buckets": area_buckets,
    }


# ═══════════════════════════════════════════════════════════════════
# Area-Bucket Recall | 按实例面积分桶 Recall
# ═══════════════════════════════════════════════════════════════════

AREA_BUCKETS = [
    (0, 32, "<32"),
    (32, 64, "32–64"),
    (64, 128, "64–128"),
    (128, 256, "128–256"),
    (256, 512, "256–512"),
    (512, float("inf"), ">512"),
]


def _compute_area_bucket_stats(gt_anns_raw: list[dict], gt_matched: list[bool]) -> dict:
    """按实例面积分桶统计 matched/total | Bucket matched/total by instance area."""
    buckets = {label: {"matched": 0, "total": 0} for _, _, label in AREA_BUCKETS}
    for j, gt_ann in enumerate(gt_anns_raw):
        area = gt_ann.get("area", 0)
        for lo, hi, label in AREA_BUCKETS:
            if lo <= area < hi:
                buckets[label]["total"] += 1
                if gt_matched[j]:
                    buckets[label]["matched"] += 1
                break
    return buckets


def _merge_area_bucket_stats(acc: dict, new_stats: dict) -> None:
    """合并面积分桶统计 | Merge area-bucket stats."""
    for label in acc:
        acc[label]["matched"] += new_stats[label]["matched"]
        acc[label]["total"] += new_stats[label]["total"]


# ═══════════════════════════════════════════════════════════════════
# Group Processing | 多图合并批量处理
# ═══════════════════════════════════════════════════════════════════

def _process_image_group(args, model, decoder, device, img_paths: list[str],
                         class_protos: dict) -> list[dict | None]:
    """一组多张图的所有 tile 堆成一个大 batch，一次 backbone forward | Group multiple images:
    all tiles → one batched backbone forward → per-image decoder + stitch + metrics."""

    # ── Step 1: Load all images, GT, and tile ──
    tile_size = args.tile_size
    stride_val = args.stride
    images_data = []  # [{path, H, W, gt_anns_raw, tiles:[...]}]
    all_tiles = []    # flat list of all tiles

    for img_path in img_paths:
        if not os.path.exists(img_path):
            images_data.append(None); continue
        try:
            full_img = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            images_data.append(None); continue
        H_full, W_full = full_img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]

        # Load GT
        gt_anns_raw = []
        gt_class_ids = set()
        if args.full_gt and os.path.exists(args.full_gt):
            full_coco = json.load(open(args.full_gt))
            img_id_to_file = {img["id"]: img["file_name"] for img in full_coco["images"]}
            file_to_id = {v: k for k, v in img_id_to_file.items()}
            target_id = file_to_id.get(os.path.basename(img_path))
            if target_id is None:
                for fid, fname in img_id_to_file.items():
                    if os.path.splitext(fname)[0] == stem:
                        target_id = fid; break
            if target_id is not None:
                for ann in full_coco.get("annotations", []):
                    if ann.get("image_id") != target_id: continue
                    cat_id = ann.get("category_id", 0)
                    if cat_id < 1 or cat_id > 15: continue
                    bbox = ann.get("bbox", [0, 0, 0, 0])
                    gt_anns_raw.append({
                        "category_id": cat_id,
                        "segmentation": ann.get("segmentation", []),
                        "bbox": bbox, "area": float(ann.get("area", bbox[2] * bbox[3])),
                    })
                    gt_class_ids.add(cat_id)

        tiles = tile_full_image(full_img, tile_size=tile_size, stride=stride_val)
        images_data.append({
            "path": img_path, "stem": stem, "H": H_full, "W": W_full,
            "gt_anns_raw": gt_anns_raw, "gt_class_ids": gt_class_ids,
            "tiles": tiles, "n_tiles": len(tiles),
        })
        all_tiles.extend(tiles)

    valid_images = [d for d in images_data if d is not None]
    if not valid_images:
        return [None] * len(img_paths)

    # ── Step 2: Batched backbone forward (sub-batch if too many tiles) ──
    # 批量 backbone forward (tile 过多时分 sub-batch 防 OOM)
    total_B = len(all_tiles)
    max_per_batch = getattr(args, "max_tiles_per_batch", 64)
    all_p3, all_p4, all_p8, all_proto = [], [], [], []

    for sub_start in range(0, total_B, max_per_batch):
        sub_end = min(sub_start + max_per_batch, total_B)
        sub_tiles = all_tiles[sub_start:sub_end]
        sub_imgs = [torch.from_numpy(t["img"]).permute(2, 0, 1).float().div_(255.0)
                     for t in sub_tiles]
        sub_batch = torch.stack(sub_imgs).to(device)

        with torch.no_grad():
            sub_feats = _extract_features_batched(model, sub_batch, device, no_grad=True)
        all_p3.append(sub_feats["p3"].cpu())
        all_p4.append(sub_feats["p4"].cpu())
        all_p8.append(sub_feats["p8"].cpu())
        all_proto.append(sub_feats["proto"].cpu())
        del sub_batch, sub_feats

    # Concatenate all sub-batches (keep on CPU for memory efficiency)
    batched_feats = {
        "p3": torch.cat(all_p3, dim=0),
        "p4": torch.cat(all_p4, dim=0),
        "p8": torch.cat(all_p8, dim=0),
        "proto": torch.cat(all_proto, dim=0),
    }
    del all_p3, all_p4, all_p8, all_proto

    # ── Step 3: Per-image decoder + stitch + metrics ──
    # Keep batched feats on CPU, move tile-by-tile to GPU for decoder
    results = []
    tile_offset = 0
    for img_data in images_data:
        if img_data is None:
            results.append(None)
            continue

        n_tiles = img_data["n_tiles"]
        # Slice this image's features from batch (on CPU for memory efficiency)
        img_feats_cpu = {
            "p3": batched_feats["p3"][tile_offset:tile_offset + n_tiles].cpu(),
            "p4": batched_feats["p4"][tile_offset:tile_offset + n_tiles].cpu(),
            "proto": batched_feats["proto"][tile_offset:tile_offset + n_tiles].cpu(),
        }

        # Per-tile decoder
        all_tile_results = []
        with torch.no_grad():
            for ti in range(n_tiles):
                tile_info = img_data["tiles"][ti]
                y0, x0, h, w = tile_info["y0"], tile_info["x0"], tile_info["h"], tile_info["w"]
                # Move single tile features to GPU
                feats = {
                    "p3": img_feats_cpu["p3"][ti].to(device),
                    "p4": img_feats_cpu["p4"][ti].to(device),
                    "proto": img_feats_cpu["proto"][ti].to(device),
                }

                best_prob = np.zeros((tile_size, tile_size), dtype=np.float32)
                center_np = None
                if args.decoder == "center_affinity":
                    first_pk = list(class_protos.values())[0]
                    proto_vec = torch.from_numpy(first_pk["proto"]).float().to(device)
                    center_hm, _, _ = decoder(
                        feats["p3"].unsqueeze(0), feats["p4"].unsqueeze(0),
                        feats["proto"].unsqueeze(0), proto_vec,
                    )
                    center_np = F.interpolate(
                        center_hm.unsqueeze(0).unsqueeze(0) if center_hm.dim() == 2 else center_hm.unsqueeze(0),
                        size=(tile_size, tile_size), mode="bilinear", align_corners=False,
                    ).squeeze().cpu().numpy()
                    for cls_id, pk in class_protos.items():
                        proto_vec_c = torch.from_numpy(pk["proto"]).float().to(device)
                        proto_mask = decoder.forward_proto_only(
                            feats["proto"].unsqueeze(0), proto_vec_c,
                        )
                        pm = F.interpolate(proto_mask.unsqueeze(0).unsqueeze(0),
                                          size=(tile_size, tile_size), mode="bilinear",
                                          align_corners=False).squeeze().cpu().numpy()
                        best_prob = np.maximum(best_prob, pm)
                else:
                    for cls_id, pk in class_protos.items():
                        proto_vec = torch.from_numpy(pk["proto"]).float().to(device)
                        if args.decoder == "adaptive":
                            out = decoder(feats["p4"].unsqueeze(0), feats["proto"].unsqueeze(0), proto_vec)
                        else:
                            out = decoder(feats["p3"].unsqueeze(0), feats["p4"].unsqueeze(0),
                                          feats["proto"].unsqueeze(0), proto_vec)
                        if out.dim() == 2: out = out.unsqueeze(0).unsqueeze(0)
                        elif out.dim() == 3: out = out.unsqueeze(0)
                        pm = F.interpolate(out, size=(tile_size, tile_size),
                                          mode="bilinear", align_corners=False).squeeze().cpu().numpy()
                        best_prob = np.maximum(best_prob, pm)

                all_tile_results.append({
                    "y0": y0, "x0": x0, "h": h, "w": w, "prob": best_prob,
                    "center": center_np,
                })

        tile_offset += n_tiles

        # Stitch
        full_prob = stitch_prob_maps(all_tile_results, img_data["H"], img_data["W"],
                                     tile_size, stride_val)

        # Generate instances + match
        from adatile.metrics.instance_generation import generate_instances
        pred_instances = generate_instances(
            full_prob, method="connected_components",
            score_thr=args.score_thr, min_area=args.min_area,
        )

        gt_anns_raw = img_data["gt_anns_raw"]
        tp, fp_count, fn_count = 0, 0, 0
        gt_matched = [False] * len(gt_anns_raw) if gt_anns_raw else []
        if gt_anns_raw:
            for pred in sorted(pred_instances, key=lambda x: x.get("score", 0), reverse=True):
                pm = pred["mask"]
                pm_ys, pm_xs = np.where(pm)
                if len(pm_ys) == 0: fp_count += 1; continue
                p_by, p_bx = int(pm_ys.min()), int(pm_xs.min())
                p_ey, p_ex = int(pm_ys.max()) + 1, int(pm_xs.max()) + 1
                best_iou, best_j = 0.0, -1
                for j in range(len(gt_anns_raw)):
                    if gt_matched[j]: continue
                    gt_ann = gt_anns_raw[j]
                    gx, gy, gw, gh = [int(v) for v in gt_ann["bbox"]]
                    if p_ex <= gx or p_bx >= gx + gw or p_ey <= gy or p_by >= gy + gh:
                        continue
                    rx, ry = max(p_bx, gx), max(p_by, gy)
                    rw, rh = min(p_ex, gx + gw) - rx, min(p_ey, gy + gh) - ry
                    if rw <= 0 or rh <= 0: continue
                    gt_roi = np.zeros((rh, rw), dtype=np.uint8)
                    seg = gt_ann.get("segmentation", [])
                    if seg:
                        polys = seg if isinstance(seg[0], list) else [seg]
                        for poly in polys:
                            if len(poly) < 6: continue
                            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                            pts[:, :, 0] -= rx; pts[:, :, 1] -= ry
                            cv2.fillPoly(gt_roi, [pts], 1)
                    gt_roi = gt_roi.astype(bool)
                    pred_roi = pm[ry:ry + rh, rx:rx + rw]
                    inter = (pred_roi & gt_roi).sum()
                    union = (pred_roi | gt_roi).sum()
                    iou = inter / max(union, 1) if union > 0 else 0.0
                    if iou > best_iou: best_iou, best_j = iou, j
                if best_iou >= args.iou_thr:
                    gt_matched[best_j] = True; tp += 1
                else:
                    fp_count += 1
            fn_count = len(gt_anns_raw) - tp

        precision = tp / max(tp + fp_count, 1)
        recall = tp / max(tp + fn_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        # Per-class + area-bucket stats
        n_per_class = defaultdict(int)
        n_matched_per_class = defaultdict(int)
        for j, gt_ann in enumerate(gt_anns_raw):
            n_per_class[gt_ann["category_id"]] += 1
            if gt_matched[j]:
                n_matched_per_class[gt_ann["category_id"]] += 1
        area_buckets = _compute_area_bucket_stats(gt_anns_raw, gt_matched)

        results.append({
            "image": img_data["stem"], "H": img_data["H"], "W": img_data["W"],
            "n_gt": len(gt_anns_raw), "n_pred": len(pred_instances),
            "n_classes_gt": len(img_data["gt_class_ids"]),
            "tp": tp, "fp": fp_count, "fn": fn_count,
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "prob_max": float(full_prob.max()), "prob_mean": float(full_prob.mean()),
            "fg_frac": float((full_prob > 0.3).mean()),
            "per_class": {ISAID_CLASSES.get(k, f"c{k}"):
                          {"matched": n_matched_per_class.get(k, 0), "gt": v}
                          for k, v in n_per_class.items()},
            "area_buckets": area_buckets,
        })
        # Free per-image features to limit CPU memory growth
        del img_feats_cpu

    del batched_feats
    return results


def run_batch_mode(args, model, decoder, extract_features, device):
    """批量处理整个目录 (多图 tile 合并 batched backbone) | Batch process all images in a directory
    with batched backbone across images."""
    import csv as _csv_module
    batch_dir = args.batch_dir
    if not os.path.isdir(batch_dir):
        print(f"  [FATAL] Not a directory: {batch_dir}")
        sys.exit(1)

    images = sorted([
        f for f in os.listdir(batch_dir)
        if f.lower().endswith(args.batch_ext.lower())
    ])
    print(f"\n  Found {len(images)} images in {batch_dir}")
    print(f"  Group size: {args.batch_group} images per backbone forward")
    print(f"  Processing...\n")

    # Build class prototypes once (from first image's data_root)
    class_protos = {}
    if args.data_root:
        data_root = Path(args.data_root)
        if data_root.exists():
            class_index = _build_class_index(data_root, args.split)
            class_protos = build_class_prototypes_vis(
                model, extract_features, class_index, images[0] if images else "",
                args.split, data_root, args.k_shot, args.prototype_source, device,
            )
    if not class_protos:
        # Load first image's GT to get class IDs
        first_path = os.path.join(batch_dir, images[0])
        first_img_name = os.path.basename(first_path)
        gt_class_ids = set()
        if args.full_gt and os.path.exists(args.full_gt):
            full_coco = json.load(open(args.full_gt))
            img_id_to_file = {img["id"]: img["file_name"] for img in full_coco["images"]}
            file_to_id = {v: k for k, v in img_id_to_file.items()}
            target_id = file_to_id.get(first_img_name)
            if target_id is not None:
                for ann in full_coco.get("annotations", []):
                    if ann.get("image_id") == target_id and 1 <= ann.get("category_id", 0) <= 15:
                        gt_class_ids.add(ann["category_id"])
        for cid in (gt_class_ids or range(1, 16)):
            class_protos[cid] = {"proto": np.random.randn(640).astype(np.float32)}
    print(f"  Prototypes: {[ISAID_CLASSES.get(c, f'c{c}') for c in sorted(class_protos.keys())]}")

    all_metrics = []
    overall_tp, overall_fp_count, overall_fn = 0, 0, 0
    overall_n_gt = 0
    group_size = args.batch_group
    t_total_start = time.perf_counter()

    for group_start in range(0, len(images), group_size):
        group_paths = [os.path.join(batch_dir, f) for f in images[group_start:group_start + group_size]]
        group_names = images[group_start:group_start + group_size]
        print(f"  ── Group [{group_start + 1}-{min(group_start + group_size, len(images))}/{len(images)}] "
              f"({len(group_paths)} images) ──", flush=True)

        t_group = time.perf_counter()
        group_results = _process_image_group(args, model, decoder, device, group_paths, class_protos)
        t_elapsed = time.perf_counter() - t_group

        for j, (fname, m) in enumerate(zip(group_names, group_results)):
            if m is None:
                print(f"    {fname}: SKIP")
                continue
            all_metrics.append(m)
            overall_tp += m["tp"]
            overall_fp_count += m["fp"]
            overall_fn += m["fn"]
            overall_n_gt += m["n_gt"]
            print(f"    {fname}: TP={m['tp']}/{m['n_gt']} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
        print(f"    [Group done in {t_elapsed:.1f}s, {t_elapsed / max(len(group_paths), 1):.1f}s/image]\n")

    # ── Summary ──
    overall_precision = overall_tp / max(overall_tp + overall_fp_count, 1)
    overall_recall = overall_tp / max(overall_tp + overall_fn, 1)
    overall_f1 = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-9)

    print(f"\n{'='*80}")
    print(f"  BATCH SUMMARY — {len(all_metrics)} images")
    print(f"{'='*80}")
    print(f"  Total GT:   {overall_n_gt}")
    print(f"  Overall TP: {overall_tp}")
    print(f"  Overall FP: {overall_fp_count}")
    print(f"  Overall FN: {overall_fn}")
    print(f"  Precision:  {overall_precision:.4f}")
    print(f"  Recall:     {overall_recall:.4f}")
    print(f"  F1:         {overall_f1:.4f}")

    # Per-class aggregation
    agg_per_class = defaultdict(lambda: {"matched": 0, "gt": 0})
    for m in all_metrics:
        for cls_name, pc in m["per_class"].items():
            agg_per_class[cls_name]["matched"] += pc["matched"]
            agg_per_class[cls_name]["gt"] += pc["gt"]
    print(f"\n  Per-Class Aggregate:")
    for cls_name in sorted(agg_per_class.keys()):
        pc = agg_per_class[cls_name]
        r = pc["matched"] / max(pc["gt"], 1)
        print(f"    {cls_name:<22s} R={r:.3f}  ({pc['matched']}/{pc['gt']})")

    # ── Area-Bucket Recall (按实例面积分桶) ──
    agg_buckets = {label: {"matched": 0, "total": 0} for _, _, label in AREA_BUCKETS}
    for m in all_metrics:
        if "area_buckets" in m:
            _merge_area_bucket_stats(agg_buckets, m["area_buckets"])
    print(f"\n  Area-Bucket Recall (按实例面积分桶 Recall):")
    print(f"    {'Bucket':>12s}  {'Recall':>8s}  {'Matched':>8s}  {'Total':>8s}  {'Histogram'}")
    print(f"    {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")
    max_total = max(b["total"] for b in agg_buckets.values()) or 1
    for _, _, label in AREA_BUCKETS:
        b = agg_buckets[label]
        r = b["matched"] / max(b["total"], 1)
        bar_len = int(b["total"] / max_total * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {label:>12s}  {r:>8.4f}  {b['matched']:>8d}  {b['total']:>8d}  {bar}")

    # Save CSV (with area-bucket columns)
    csv_path = os.path.join(args.output, "batch_metrics.csv")
    bucket_labels = [label for _, _, label in AREA_BUCKETS]
    with open(csv_path, "w", newline="") as f:
        writer = _csv_module.writer(f)
        writer.writerow(["image", "H", "W", "n_gt", "n_pred", "n_classes_gt",
                        "tp", "fp", "fn", "precision", "recall", "f1",
                        "prob_max", "prob_mean", "fg_frac"]
                       + [f"R_{b}" for b in bucket_labels]
                       + [f"gt_{b}" for b in bucket_labels])
        for m in all_metrics:
            row = [m["image"], m["H"], m["W"], m["n_gt"], m["n_pred"],
                   m["n_classes_gt"], m["tp"], m["fp"], m["fn"],
                   m["precision"], m["recall"], m["f1"],
                   m["prob_max"], m["prob_mean"], m["fg_frac"]]
            ab = m.get("area_buckets", {})
            for b in bucket_labels:
                bd = ab.get(b, {"matched": 0, "total": 1})
                row.append(round(bd["matched"] / max(bd["total"], 1), 4))
            for b in bucket_labels:
                bd = ab.get(b, {"matched": 0, "total": 0})
                row.append(bd["total"])
            writer.writerow(row)
    print(f"\n  [SAVED] Metrics CSV → {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Decoder Prediction Visualizer v3")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--decoder", type=str, required=True,
                        choices=["adaptive", "adaptive-p3p4", "dynamic_kernel", "center_affinity"])
    parser.add_argument("--mode", type=str, default="tile", choices=["tile", "full", "batch"],
                        help="tile=单tile可视化 | full=单张全图 | batch=批量处理整个目录")
    parser.add_argument("--batch-dir", type=str, help="批量模式下图片目录 | Image directory for batch mode")
    parser.add_argument("--batch-ext", type=str, default=".png", help="批量模式图片扩展名 | Image extension for batch")
    parser.add_argument("--batch-group", type=int, default=4,
                        help="每批处理的图片数 (tile 合并过 backbone) | Images per backbone forward group")
    parser.add_argument("--max-tiles-per-batch", type=int, default=64,
                        help="单次 backbone forward 最大 tile 数 (防 OOM) | Max tiles per forward pass")

    # Tile mode args
    parser.add_argument("--query-tile", type=str, help="Tile stem (e.g. P0089_t0001)")
    parser.add_argument("--data-root", type=str, default="data/iSAID_instance_fewshot",
                        help="Tile-level COCO 数据根目录")
    parser.add_argument("--split", type=str, default="val")

    # Full-image mode args
    parser.add_argument("--image", type=str, help="Full image path (full mode)")
    parser.add_argument("--full-gt", type=str, help="Full-image COCO JSON path (full mode)")
    parser.add_argument("--tile-size", type=int, default=896)
    parser.add_argument("--stride", type=int, default=640)

    # Common args
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--prototype-source", type=str, default="p4", choices=["p4", "p8"])
    parser.add_argument("--output", type=str, default="vis_output")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--min-area", type=int, default=64, help="全图模式最小实例面积")
    parser.add_argument("--min-distance", type=int, default=24, help="全图模式峰值最小间距")
    parser.add_argument("--iou-thr", type=float, default=0.3,
                        help="TP 匹配 IoU 阈值 | IoU threshold for TP matching")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(42)

    print("=" * 70)
    print(f"  Decoder Visualization v3 — {args.decoder} ({args.mode} mode)")
    print("=" * 70)

    t_total = time.perf_counter()

    # Load model
    t0 = time.perf_counter()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    unfreeze_layers = ckpt.get("unfreeze_layers", ckpt.get("config", {}).get("unfreeze_layers", 8))
    model, decoder, extract_features, ckpt_meta = _load_model_and_decoder(
        args.checkpoint, unfreeze_layers, args.decoder, device)
    t_load = time.perf_counter() - t0
    print(f"  [TIMING] Model loading: {t_load:.1f}s")

    if args.mode == "batch":
        if not args.batch_dir:
            print("  [FATAL] --batch-dir is required for batch mode")
            sys.exit(1)
        run_batch_mode(args, model, decoder, extract_features, device)
    elif args.mode == "full":
        if not args.image:
            print("  [FATAL] --image is required for full mode")
            sys.exit(1)
        run_full_image_mode(args, model, decoder, extract_features, ckpt, device)
    else:
        if not args.query_tile:
            print("  [FATAL] --query-tile is required for tile mode")
            sys.exit(1)
        if not args.data_root:
            print("  [FATAL] --data-root is required for tile mode")
            sys.exit(1)
        run_tile_mode(args, model, decoder, extract_features, ckpt, device)

    t_total = time.perf_counter() - t_total
    print(f"\n  [TIMING] Total wall time: {t_total:.1f}s")
    print(f"  Done → {args.output}/")


if __name__ == "__main__":
    main()
