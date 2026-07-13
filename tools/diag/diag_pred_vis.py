"""
Decoder Prediction Visualizer v3 — 真实 Prototype + GT 对比 + 汇总大图.
Decoder Prediction Visualizer v3 — Real Prototypes + GT Comparison + Summary Figure.

用法 | Usage:
    python tools/diag/diag_pred_vis.py \
        --checkpoint runs/.../best_model.pt \
        --decoder center_affinity \
        --data-root data/iSAID_instance_fewshot \
        --query-tile P0089_t0001 --split val \
        --output vis_output/ --device cuda

v3 改进 | v3 Improvements:
    1. 真实 prototype (从 support set 在线构建, 与 eval 一致)
       Real prototypes (built from support set, consistent with eval)
    2. GT 对比 (加载 COCO 标注, 叠加 TP/FP/FN)
       GT comparison (load COCO annotations, TP/FP/FN overlay)
    3. 单张汇总大图 (不再按 class 拆成多张)
       Single summary figure (no more per-class separate images)
"""

from __future__ import annotations

import argparse, json, os, sys, random
from collections import defaultdict
from pathlib import Path

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
import matplotlib.patches as mpatches
from matplotlib import cm


# ═══════════════════════════════════════════════════════════════════
# Constants | 常量
# ═══════════════════════════════════════════════════════════════════

# iSAID 15 类 | iSAID 15 classes
ISAID_CLASSES = {
    1:  "small_vehicle",
    2:  "large_vehicle",
    3:  "plane",
    4:  "storage_tank",
    5:  "ship",
    6:  "harbor",
    7:  "ground_track_field",
    8:  "soccer_ball_field",
    9:  "tennis_court",
    10: "swimming_pool",
    11: "baseball_diamond",
    12: "basketball_court",
    13: "bridge",
    14: "helicopter",
    15: "roundabout",
}

# 每类颜色 (Tab20 循环) | Per-class color (Tab20 cycle)
CLASS_COLORS = {cid: plt.cm.tab20(i % 20) for i, cid in enumerate(ISAID_CLASSES.keys())}

# COCO 索引缓存 | COCO index cache
_COCO_CACHE: dict = {}


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Model Loading | 模型加载
# ═══════════════════════════════════════════════════════════════════

def _fastsam_weights_path():
    """Locate FastSAM-x.pt."""
    p = os.path.join(_PROJECT_ROOT, "thirdLibrary", "FastSAM", "weights", "FastSAM-x.pt")
    if os.path.exists(p):
        return p
    env_path = os.environ.get("FASTSAM_WEIGHTS", "")
    if env_path and os.path.exists(env_path):
        return env_path
    raise FileNotFoundError(f"FastSAM-x.pt not found at {p}.")


def _load_model_and_decoder(ckpt_path: str, unfreeze_layers: int, decoder_type: str, device: str):
    """Load FastSAM + decoder from checkpoint. Returns (model, decoder, extract_features, ckpt_meta)."""
    from ultralytics import FastSAM
    from tools.train.train_fewshot_allclass import extract_features

    model = FastSAM(str(_fastsam_weights_path()))
    model.model.to(device).eval()
    for p in model.model.parameters():
        p.requires_grad = False

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Restore backbone if unfrozen
    if unfreeze_layers > 0 and "backbone" in ckpt:
        seq = model.model.model
        for i_str, state in ckpt["backbone"].items():
            seq[int(i_str)].load_state_dict(state)
        print(f"  Restored {unfreeze_layers} backbone layers")

    # Detect channels
    test_img = np.zeros((896, 896, 3), dtype=np.uint8)
    test_feats = extract_features(model, [test_img], device)
    p3_channels = test_feats[0]["p3"].shape[1]
    p4_channels = test_feats[0]["p4"].shape[1]
    print(f"  P3={p3_channels}, P4={p4_channels}")

    # Build decoder
    normalize_proto = ckpt.get("normalize_proto", "none")
    if decoder_type == "dynamic_kernel":
        from adatile.decoder.dynamic_kernel_decoder import DynamicKernelDecoder
        n_kernels = ckpt.get("n_kernels", 16)
        decoder = DynamicKernelDecoder(
            p3_channels=p3_channels, p4_channels=p4_channels,
            proto_dim=32, n_kernels=n_kernels, kernel_dim=256, fpn_dim=256,
            normalize_proto=normalize_proto,
        ).to(device)
        print(f"  decoder=dynamic_kernel n_kernels={n_kernels}")
    elif decoder_type == "center_affinity":
        from adatile.decoder.center_affinity_decoder import CenterAffinityDecoder
        decoder = CenterAffinityDecoder(
            p3_channels=p3_channels, p4_channels=p4_channels,
            proto_dim=32, fpn_dim=64, normalize_proto=normalize_proto,
        ).to(device)
        print(f"  decoder=center_affinity")
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
    """从 tile stem 提取源图名 | Extract source image name from tile stem.
    e.g. 'P0089_t0001' → 'P0089'
    """
    parts = stem.rsplit("_t", 1)
    return parts[0] if len(parts) == 2 else stem


def _load_coco_index(data_root: Path, split: str) -> dict:
    """缓存 COCO JSON 索引 | Cache COCO JSON index."""
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
        "file_to_anns": dict(file_to_anns),
        "img_id_to_file": img_id_to_file,
        "images": coco["images"],
        "annotations": coco.get("annotations", []),
    }
    return _COCO_CACHE[cache_key]


def _build_class_index(data_root: Path, split: str) -> dict[int, dict[str, list[str]]]:
    """构建层级索引 cls_id → {source_img: [tile_stems]}.
    Build hierarchical index for support prototype building.
    """
    coco_idx = _load_coco_index(data_root, split)
    if not coco_idx:
        return {}

    tile_stems = {img["id"]: Path(img["file_name"]).stem for img in coco_idx["images"]}
    tile_classes: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for ann in coco_idx["annotations"]:
        cat_id = ann.get("category_id", 0)
        img_id = ann.get("image_id", 0)
        if 1 <= cat_id <= 15 and img_id in tile_stems:
            tile_classes[img_id][cat_id] += 1

    index: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
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
    """加载单个 tile 的 GT 标注 | Load GT annotations for a single tile.

    :return: {
        "instances": list of {category_id, mask(bool [H,W]), area},
        "class_ids": set of category_ids present,
        "merged_mask": bool [H,W] — all instances merged,
    }
    """
    coco_idx = _load_coco_index(data_root, split)
    file_to_anns = coco_idx.get("file_to_anns", {})
    anns = file_to_anns.get(f"{stem}.png", [])

    instances = []
    merged = np.zeros((H, W), dtype=bool)
    for ann in anns:
        cat_id = ann.get("category_id", 0)
        if cat_id < 1 or cat_id > 15:
            continue
        seg = ann.get("segmentation", [])
        if not seg:
            bx, by, bw, bh = [int(v) for v in ann.get("bbox", [0, 0, 0, 0])]
            mask = np.zeros((H, W), dtype=bool)
            mask[max(0, by):min(H, by + bh), max(0, bx):min(W, bx + bw)] = True
        elif isinstance(seg, list):
            import cv2
            mask = np.zeros((H, W), dtype=np.uint8)
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
            mask = mask.astype(bool)
        else:
            continue

        if mask.sum() > 0:
            instances.append({"category_id": cat_id, "mask": mask, "area": float(mask.sum())})
            merged = merged | mask

    return {
        "instances": instances,
        "class_ids": {inst["category_id"] for inst in instances},
        "merged_mask": merged,
    }


def load_tile_image(stem: str, data_root: Path, split: str) -> np.ndarray:
    """加载 tile 图像 | Load tile image."""
    img_path = data_root / "images" / split / f"{stem}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Tile image not found: {img_path}")
    return np.array(Image.open(str(img_path)).convert("RGB"))


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Prototype Building | Prototype 构建
# ═══════════════════════════════════════════════════════════════════

def compute_support_prototype_vis(support_feats: list[dict], source: str = "p4") -> torch.Tensor:
    """从 K 个 support 特征计算 L2-normalized prototype."""
    vectors = []
    for sf in support_feats:
        v = sf[source].mean(dim=(2, 3))
        v = F.normalize(v, p=2, dim=-1)
        vectors.append(v)
    proto = torch.stack(vectors).mean(dim=0)
    return F.normalize(proto, p=2, dim=-1)


def build_class_prototypes_vis(
    model, extract_features, class_index: dict, query_stem: str,
    split: str, data_root: Path, k_shot: int,
    proto_source: str, device: str,
) -> dict[int, dict]:
    """为 query tile 中出现的所有类构建 prototype.
    Build prototypes for all classes present in the query tile.

    逻辑 | Logic:
        - 使用 k_shot 个 support tile (来自不同源图, 排除 query 源图)
        - Use k_shot support tiles (from different source images, excluding query source)
        - 每个 support tile 中只有 target class 的 mask 参与 prototype 计算
        - Only target class masks in each support tile contribute to prototype

    :return: {cls_id: {"proto": Tensor[1,C]}}
    """
    # 确定 query tile 的源图 | Determine query tile's source image
    query_src = _extract_source_image(query_stem)

    class_protos = {}
    for cls_id, src_to_tiles in class_index.items():
        # 排除 query 源图, 找 K 个 support tile | Exclude query source, find K support tiles
        support_pool = [
            (src, tile) for src, tiles in src_to_tiles.items()
            if src != query_src
            for tile in tiles
        ]

        if len(support_pool) < k_shot:
            print(f"  [WARN] Class {cls_id} ({ISAID_CLASSES.get(cls_id, '?')}): "
                  f"only {len(support_pool)} support tiles available (< K={k_shot}), skipped")
            continue

        # 随机选 K 个 support tile | Randomly select K support tiles
        rng = random.Random(42)
        chosen = rng.sample(support_pool, k_shot)

        support_imgs = []
        for src, tile_stem in chosen:
            img = load_tile_image(tile_stem, data_root, split)
            support_imgs.append(img)

        support_feats = extract_features(model, support_imgs, device)
        proto = compute_support_prototype_vis(support_feats, source=proto_source)
        class_protos[cls_id] = {"proto": proto.cpu().numpy()}

    return class_protos


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Center-Affinity Visualization | Center-Affinity 可视化
# ═══════════════════════════════════════════════════════════════════

def _draw_instance_contours(ax, instances: list, cmap_dict: dict = None, alpha: float = 0.6):
    """在 ax 上绘制实例轮廓 (彩色填充) | Draw instance contours on ax (colored fill)."""
    import cv2
    for inst in instances:
        mask = inst["mask"].astype(np.uint8)
        cat_id = inst.get("category_id", 1)
        color = cmap_dict.get(cat_id, (1, 1, 1)) if cmap_dict else (0, 1, 0)
        # Alpha blend
        overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
        overlay[mask > 0] = (*color[:3], alpha)
        ax.imshow(overlay)


def _draw_gt_overlay(ax, gt_data: dict):
    """在 ax 上绘制 GT 实例: 彩色填充 + 边缘轮廓."""
    import cv2
    for inst in gt_data["instances"]:
        mask = inst["mask"]
        cat_id = inst["category_id"]
        rgba = list(CLASS_COLORS[cat_id])
        # 轮廓 | Contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        for cnt in contours:
            ax.plot(cnt[:, 0, 0], cnt[:, 0, 1], color=rgba, linewidth=1.5)

        # 半透明填充 | Semi-transparent fill
        overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
        overlay[mask > 0] = (*rgba[:3], 0.25)
        ax.imshow(overlay)


def _draw_tp_fp_fn(ax, pred_instances: list, gt_data: dict, iou_thr: float = 0.3):
    """绘制 TP/FP/FN 叠加图: 绿=TP, 红=FP, 蓝=FN."""
    import cv2

    H, W = gt_data["merged_mask"].shape[:2] if gt_data["merged_mask"].sum() > 0 else (896, 896)
    if gt_data["merged_mask"].sum() > 0:
        H, W = gt_data["merged_mask"].shape
    else:
        H = W = 896

    tp_canvas = np.zeros((H, W, 3), dtype=np.float32)
    fp_canvas = np.zeros((H, W, 3), dtype=np.float32)
    fn_canvas = np.zeros((H, W, 3), dtype=np.float32)

    # 简单贪心匹配 | Simple greedy matching
    gt_matched = [False] * len(gt_data["instances"])
    for pred in sorted(pred_instances, key=lambda x: x["score"], reverse=True):
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
            tp_canvas[pm] = [0, 1, 0]  # 绿 | Green
        else:
            fp_canvas[pm] = [1, 0, 0]  # 红 | Red

    for j, gt in enumerate(gt_data["instances"]):
        if not gt_matched[j]:
            fn_canvas[gt["mask"]] = [0, 0, 1]  # 蓝 | Blue

    ax.imshow(tp_canvas, alpha=0.5)
    ax.imshow(fp_canvas, alpha=0.5)
    ax.imshow(fn_canvas, alpha=0.5)
    ax.set_title(f"TP (绿) / FP (红) / FN (蓝)\n"
                 f"TP={sum(gt_matched)}, FP={len(pred_instances) - sum(gt_matched)}, "
                 f"FN={len(gt_data['instances']) - sum(gt_matched)}",
                 fontsize=9)

    # 图例 | Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='green', alpha=0.5, label=f'TP ({sum(gt_matched)})'),
        Patch(color='red', alpha=0.5, label=f'FP ({len(pred_instances) - sum(gt_matched)})'),
        Patch(color='blue', alpha=0.5, label=f'FN ({len(gt_data["instances"]) - sum(gt_matched)})'),
    ], loc='lower right', fontsize=7)


def plot_center_affinity(
    image: np.ndarray,
    tile_stem: str,
    gt_data: dict,
    class_protos: dict,
    decoder,
    feats: dict,
    device: str,
    out_path: str,
):
    """Center-Affinity Decoder 综合可视化 | Center-Affinity comprehensive visualization.

    2×3 grid:
      Row 1: [Input+GT] [GT Instances] [Proto Mask vs GT FG]
      Row 2: [Center HM] [Offset Field] [TP/FP/FN]
    """
    H_img, W_img = image.shape[:2]

    # ── Run center-affinity inference ──
    all_pred_instances = []
    with torch.no_grad():
        # Class-agnostic center + offset (use first prototype)
        first_cls = next(iter(class_protos.keys()))
        proto_vec = torch.from_numpy(class_protos[first_cls]["proto"]).float().to(device)
        center_hm, offset_field, _ = decoder(
            feats["p3"].unsqueeze(0) if feats["p3"].dim() == 3 else feats["p3"],
            feats["p4"].unsqueeze(0) if feats["p4"].dim() == 3 else feats["p4"],
            feats["proto"].unsqueeze(0) if feats["proto"].dim() == 3 else feats["proto"],
            proto_vec,
        )
        center_hm = center_hm.cpu().numpy()          # [H/8, W/8]
        offset_field = offset_field.cpu().numpy()    # [2, H/8, W/8]

    # Upsample center+offset to tile resolution
    center_full = F.interpolate(
        torch.from_numpy(center_hm).unsqueeze(0).unsqueeze(0),
        size=(H_img, W_img), mode="bilinear", align_corners=False,
    ).squeeze().numpy()
    offset_full = F.interpolate(
        torch.from_numpy(offset_field).unsqueeze(0),
        size=(H_img, W_img), mode="bilinear", align_corners=False,
    ).squeeze().numpy()

    # Per-class proto_mask + grouping
    best_proto_mask = None
    for cls_id, pk in class_protos.items():
        with torch.no_grad():
            proto_vec = torch.from_numpy(pk["proto"]).float().to(device)
            proto_mask = decoder.forward_proto_only(
                feats["proto"].unsqueeze(0) if feats["proto"].dim() == 3 else feats["proto"],
                proto_vec,
            )
        proto_full = F.interpolate(
            proto_mask.unsqueeze(0).unsqueeze(0),
            size=(H_img, W_img), mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()

        fg_mask = proto_full > 0.3
        if best_proto_mask is None or fg_mask.sum() > (best_proto_mask[1] if best_proto_mask else 0):
            best_proto_mask = (proto_full, fg_mask.sum())

        # Group instances
        from adatile.metrics.instance_generation import generate_instances_center_affinity
        insts = generate_instances_center_affinity(
            center_full, offset_full, fg_mask,
            score_thr=0.2, min_area=16, min_distance=8, max_instances=100,
        )
        for it in insts:
            it["category_id"] = cls_id
        all_pred_instances.extend(insts)

    proto_best = best_proto_mask[0] if best_proto_mask else np.zeros((H_img, W_img))

    # ── Figure: 2 rows × 3 columns ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Center-Affinity Decoder — {tile_stem}", fontsize=14, fontweight="bold")

    # ═══════ Row 1 ═══════
    # [0,0] Input + GT overlay
    axes[0, 0].imshow(image)
    _draw_gt_overlay(axes[0, 0], gt_data)
    n_gt = len(gt_data["instances"])
    gt_classes = [ISAID_CLASSES.get(c, f"c{c}") for c in gt_data["class_ids"]]
    axes[0, 0].set_title(f"Input + GT ({n_gt} instances)\n{', '.join(gt_classes)}", fontsize=9)
    axes[0, 0].axis("off")

    # [0,1] GT Instance Masks (colored by instance)
    gt_canvas = np.zeros((H_img, W_img, 3), dtype=np.float32)
    gt_label = np.zeros((H_img, W_img), dtype=np.int32)
    for idx, inst in enumerate(gt_data["instances"], 1):
        gt_label[inst["mask"]] = idx
    axes[0, 1].imshow(image, alpha=0.3)
    axes[0, 1].imshow(gt_label, cmap="tab20", alpha=0.7, vmin=0, vmax=20)
    axes[0, 1].set_title(f"GT Instances ({n_gt})\nColored by instance ID", fontsize=9)
    axes[0, 1].axis("off")

    # [0,2] Proto Mask vs GT FG
    axes[0, 2].imshow(proto_best, cmap="hot", vmin=0, vmax=1)
    # GT contour on top
    for inst in gt_data["instances"]:
        import cv2
        contours, _ = cv2.findContours(
            inst["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        for cnt in contours:
            axes[0, 2].plot(cnt[:, 0, 0], cnt[:, 0, 1], 'g-', linewidth=0.8)
    axes[0, 2].set_title(f"Proto Mask (FG Prior)\n"
                         f"mean={proto_best.mean():.3f}, FG={best_proto_mask[1]}px",
                         fontsize=9)
    axes[0, 2].axis("off")

    # ═══════ Row 2 ═══════
    # [1,0] Center Heatmap + GT centers
    axes[1, 0].imshow(image, alpha=0.4)
    axes[1, 0].imshow(center_full, cmap="hot", alpha=0.6, vmin=0, vmax=1)
    # Mark GT centers
    for inst in gt_data["instances"]:
        ys, xs = np.where(inst["mask"])
        if len(ys) > 0:
            cy, cx = ys.mean(), xs.mean()
            axes[1, 0].plot(cx, cy, 'go', markersize=8, markerfacecolor='none', linewidth=1.5)
    axes[1, 0].set_title(f"Center Heatmap + GT Centers (○)\n"
                         f"max={center_full.max():.3f}, mean={center_full.mean():.3f}",
                         fontsize=9)
    axes[1, 0].axis("off")

    # [1,1] Offset Field (quiver, subsampled)
    step = 32  # subsample every 32 pixels
    H_q, W_q = H_img, W_img
    y_grid, x_grid = np.mgrid[step // 2:H_q:step, step // 2:W_q:step]
    dx = offset_full[0, y_grid, x_grid]
    dy = offset_full[1, y_grid, x_grid]
    # Normalize for visualization
    mag = np.sqrt(dx ** 2 + dy ** 2)
    mag_max = np.percentile(mag, 95) if mag.max() > 0 else 1.0
    dx_norm = dx / max(mag_max, 1e-6)
    dy_norm = dy / max(mag_max, 1e-6)

    axes[1, 1].imshow(image, alpha=0.3)
    # Color by direction (hue = angle, saturation = magnitude)
    angles = np.arctan2(dy, dx)  # [-pi, pi]
    colors = plt.cm.hsv((angles + np.pi) / (2 * np.pi))
    alpha = np.clip(mag / max(mag_max, 1e-6), 0.2, 1.0)

    #绘制偏移向量 | Draw offset vectors
    for i in range(len(y_grid.flat)):
        if mag.flat[i] > 0.1 * mag_max:
            axes[1, 1].arrow(
                x_grid.flat[i], y_grid.flat[i],
                dx_norm.flat[i] * step * 0.8, dy_norm.flat[i] * step * 0.8,
                head_width=4, head_length=4, fc=colors.flat[i], ec=colors.flat[i],
                alpha=float(alpha.flat[i]), linewidth=0.5,
            )

    # GT centers
    for inst in gt_data["instances"]:
        ys, xs = np.where(inst["mask"])
        if len(ys) > 0:
            axes[1, 1].plot(xs.mean(), ys.mean(), 'go', markersize=10, markerfacecolor='none', linewidth=2)

    axes[1, 1].set_title(f"Offset Vector Field (subsampled {step}px)\n"
                         f"Arrows → instance centers, ○ = GT centers",
                         fontsize=9)
    axes[1, 1].axis("off")
    axes[1, 1].set_xlim(0, W_img)
    axes[1, 1].set_ylim(H_img, 0)

    # [1,2] TP/FP/FN Overlay
    axes[1, 2].imshow(image, alpha=0.4)
    _draw_tp_fp_fn(axes[1, 2], all_pred_instances, gt_data, iou_thr=0.3)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")

    # ── Text summary ──
    print(f"\n  Summary for {tile_stem}:")
    print(f"    GT: {n_gt} instances in {len(gt_data['class_ids'])} classes")
    print(f"    Pred: {len(all_pred_instances)} instances")
    for cls_id in sorted(gt_data["class_ids"]):
        gt_n = sum(1 for i in gt_data["instances"] if i["category_id"] == cls_id)
        pred_n = sum(1 for i in all_pred_instances if i.get("category_id") == cls_id)
        cname = ISAID_CLASSES.get(cls_id, f"class_{cls_id}")
        print(f"    {cname} (id={cls_id}): GT={gt_n}, Pred={pred_n}")


# ═══════════════════════════════════════════════════════════════════
# Phase 4b: Semantic Decoder Visualization | 语义 Decoder 可视化
# ═══════════════════════════════════════════════════════════════════

def plot_semantic_decoder(
    image: np.ndarray,
    tile_stem: str,
    gt_data: dict,
    class_protos: dict,
    decoder,
    decoder_type: str,
    feats: dict,
    device: str,
    out_path: str,
):
    """Adaptive / DynamicKernel / Pure decoder 综合可视化."""
    H_img, W_img = image.shape[:2]

    # Per-class prob maps
    per_class_probs = {}
    all_pred_instances = []

    for cls_id, pk in class_protos.items():
        proto_vec = torch.from_numpy(pk["proto"]).float().to(device)
        with torch.no_grad():
            if decoder_type == "adaptive":
                prob = decoder(feats["p4"], feats["proto"], proto_vec)
            elif decoder_type == "adaptive-p3p4":
                prob = decoder(feats["p3"], feats["p4"], feats["proto"], proto_vec)
            elif decoder_type == "dynamic_kernel":
                masks_s8, proto_mask = decoder(
                    feats["p3"].unsqueeze(0) if feats["p3"].dim() == 3 else feats["p3"],
                    feats["p4"].unsqueeze(0) if feats["p4"].dim() == 3 else feats["p4"],
                    feats["proto"].unsqueeze(0) if feats["proto"].dim() == 3 else feats["proto"],
                    proto_vec,
                )
                # Use max-pool of kernels as semantic prob
                prob = masks_s8.max(dim=0)[0].unsqueeze(0).unsqueeze(0)
            else:
                prob = None

        if prob is not None:
            if prob.dim() == 2:
                prob = prob.unsqueeze(0).unsqueeze(0)
            elif prob.dim() == 3:
                prob = prob.unsqueeze(0)
            prob_full = F.interpolate(
                prob, size=(H_img, W_img), mode="bilinear", align_corners=False,
            ).squeeze().cpu().numpy()
            per_class_probs[cls_id] = prob_full

            # Generate instances
            from adatile.metrics.instance_generation import generate_instances
            insts = generate_instances(
                prob_full, method="watershed_distance",
                score_thr=0.3, min_area=16, min_distance=12,
            )
            for it in insts:
                it["category_id"] = cls_id
            all_pred_instances.extend(insts)

    # Best prob map for display
    best_prob = max(per_class_probs.values(), key=lambda p: p.max()) if per_class_probs else np.zeros((H_img, W_img))

    # ── Figure: 2 rows × 3 columns ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{decoder_type} Decoder — {tile_stem}", fontsize=14, fontweight="bold")

    # [0,0] Input + GT
    axes[0, 0].imshow(image)
    _draw_gt_overlay(axes[0, 0], gt_data)
    n_gt = len(gt_data["instances"])
    gt_classes = [ISAID_CLASSES.get(c, f"c{c}") for c in gt_data["class_ids"]]
    axes[0, 0].set_title(f"Input + GT ({n_gt} instances)\n{', '.join(gt_classes)}", fontsize=9)
    axes[0, 0].axis("off")

    # [0,1] GT Instances
    gt_label = np.zeros((H_img, W_img), dtype=np.int32)
    for idx, inst in enumerate(gt_data["instances"], 1):
        gt_label[inst["mask"]] = idx
    axes[0, 1].imshow(image, alpha=0.3)
    axes[0, 1].imshow(gt_label, cmap="tab20", alpha=0.7, vmin=0, vmax=20)
    axes[0, 1].set_title(f"GT Instances ({n_gt})", fontsize=9)
    axes[0, 1].axis("off")

    # [0,2] Best prob map
    axes[0, 2].imshow(best_prob, cmap="hot", vmin=0, vmax=1)
    best_cls = max(per_class_probs, key=lambda c: per_class_probs[c].max()) if per_class_probs else 0
    axes[0, 2].set_title(f"Best Prob Map (class {best_cls}: {ISAID_CLASSES.get(best_cls, '?')})\n"
                         f"max={best_prob.max():.3f}, mean={best_prob.mean():.3f}",
                         fontsize=9)
    axes[0, 2].axis("off")

    # [1,0] Binary threshold
    binary = best_prob > 0.3
    axes[1, 0].imshow(image, alpha=0.3)
    axes[1, 0].imshow(binary, cmap="gray", alpha=0.6)
    axes[1, 0].set_title(f"Binary >0.3\nFG area={binary.sum()} px", fontsize=9)
    axes[1, 0].axis("off")

    # [1,1] Per-class prob heatmaps (small multiples)
    n_cls = len(per_class_probs)
    if n_cls > 0:
        cls_ids_sorted = sorted(per_class_probs.keys())
        n_sub = min(n_cls, 6)
        sub_cols = min(3, n_sub)
        sub_rows = (n_sub + sub_cols - 1) // sub_cols
        # Use inset_axes for small multiples
        gs = axes[1, 1].get_gridspec()
        for i in range(n_sub):
            cls_id = cls_ids_sorted[i % n_cls]
            row = i // sub_cols
            col = i % sub_cols
            # Simple: just annotate on the axis
            cname = ISAID_CLASSES.get(cls_id, f"c{cls_id}")[:12]
            axes[1, 1].text(
                0.1 + 0.3 * col, 0.1 + 0.3 * row,
                f"{cname}\nmax={per_class_probs[cls_id].max():.2f}",
                fontsize=7, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            )
    axes[1, 1].set_title("Per-Class Prob Stats", fontsize=9)
    axes[1, 1].axis("off")

    # [1,2] TP/FP/FN
    axes[1, 2].imshow(image, alpha=0.4)
    _draw_tp_fp_fn(axes[1, 2], all_pred_instances, gt_data, iou_thr=0.3)
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Main | 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Decoder Prediction Visualizer v3")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练 checkpoint | Training checkpoint")
    parser.add_argument("--decoder", type=str, required=True,
                        choices=["adaptive", "adaptive-p3p4", "dynamic_kernel", "center_affinity"])
    parser.add_argument("--data-root", type=str, required=True,
                        help="数据根目录 | Data root (e.g. data/iSAID_instance_fewshot)")
    parser.add_argument("--query-tile", type=str, required=True,
                        help="要可视化的 tile stem | Tile stem to visualize (e.g. P0089_t0001)")
    parser.add_argument("--split", type=str, default="val",
                        help="数据分割 | Data split (train/val)")
    parser.add_argument("--k-shot", type=int, default=1,
                        help="Support set 大小 | K for support set")
    parser.add_argument("--prototype-source", type=str, default="p4", choices=["p4", "p8"],
                        help="Prototype 特征来源 | Prototype feature source")
    parser.add_argument("--output", type=str, default="vis_output")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--max-classes", type=int, default=8,
                        help="最多可视化几个类 | Max classes to visualize (top by proto norm)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(42)

    print("=" * 70)
    print(f"  Decoder Visualization v3 — {args.decoder}")
    print(f"  Query: {args.query_tile} ({args.split})")
    print("=" * 70)

    # ── Phase 1: Load model ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    unfreeze_layers = ckpt.get("unfreeze_layers", ckpt.get("config", {}).get("unfreeze_layers", 8))
    model, decoder, extract_features, ckpt_meta = _load_model_and_decoder(
        args.checkpoint, unfreeze_layers, args.decoder, device,
    )

    # ── Phase 2: Load GT ──
    data_root = Path(args.data_root)
    print(f"\n  Loading GT for {args.query_tile}...")
    gt_data = load_gt_for_tile(args.query_tile, data_root, args.split)
    print(f"  GT: {len(gt_data['instances'])} instances in "
          f"{len(gt_data['class_ids'])} classes: "
          f"{[ISAID_CLASSES.get(c, f'c{c}') for c in sorted(gt_data['class_ids'])]}")

    # ── Phase 3: Build prototypes ──
    print(f"\n  Building class index...")
    class_index = _build_class_index(data_root, args.split)
    print(f"  Found {len(class_index)} classes with index")

    print(f"  Building prototypes (K={args.k_shot}, exclude source of {args.query_tile})...")
    class_protos = build_class_prototypes_vis(
        model, extract_features, class_index, args.query_tile,
        args.split, data_root, args.k_shot,
        args.prototype_source, device,
    )
    print(f"  Built {len(class_protos)} class prototypes: "
          f"{[ISAID_CLASSES.get(c, f'c{c}') for c in sorted(class_protos.keys())]}")

    if not class_protos:
        print("  [FATAL] No class prototypes built. Check data_root and class index.")
        sys.exit(1)

    # ── Phase 4: Load tile image + extract features ──
    tile_img = load_tile_image(args.query_tile, data_root, args.split)
    H, W = tile_img.shape[:2]
    print(f"\n  Tile image: {H}×{W}")

    # Pad to multiple of 32
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(tile_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        img_padded = tile_img

    with torch.no_grad():
        feats = extract_features(model, [img_padded], device)[0]

    # ── Phase 5: Generate visualization ──
    stem = args.query_tile
    out_path = os.path.join(args.output, f"{stem}_summary.png")

    if args.decoder == "center_affinity":
        plot_center_affinity(
            tile_img, stem, gt_data, class_protos, decoder, feats, device, out_path,
        )
    else:
        plot_semantic_decoder(
            tile_img, stem, gt_data, class_protos, decoder, args.decoder, feats, device, out_path,
        )

    print(f"\n  Done → {out_path}")


if __name__ == "__main__":
    main()
