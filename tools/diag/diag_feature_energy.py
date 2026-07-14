#!/usr/bin/env python3
"""
Feature Energy Diagnostic — P3 vs P4 per-instance feature response.
特征能量诊断 — P3 vs P4 逐实例特征响应.

核心问题 | Core Question:
    P3 (stride-8) 是否对小目标保留了比 P4 (stride-16) 更强的特征响应？
    Does P3 preserve stronger feature responses for small objects than P4?

方法 | Method:
    对每个 GT 实例 mask 区域, 计算 P3 和 P4 特征的 L2 norm 均值,
    按实例面积分桶, 绘制 Energy(Area) 曲线。
    For each GT instance mask region, compute mean L2 norm of P3 and P4 features.
    Bucket by instance area, plot Energy(Area) curves.

证据链定位 | Evidence Chain Position:
    Step 0 (先于训练): 在 frozen backbone 上确认 P3 对小目标的特征优势。
    如果曲线显示 P3 在小面积桶显著高于 P4, 则 P3+P4 decoder 训练有望提升 small_vehicle。

用法 | Usage::

    # 1. 从已有 checkpoint 提取特征 (backbone 本身是 frozen 的, 不依赖 decoder)
    python tools/diag/diag_feature_energy.py \\
        --checkpoint runs/train_fewshot_allcls_K1_center_affinity_uf8_protop8_norml2_0713_2101/best_model.pt \\
        --data-root /root/autodl-tmp/iSAID_processed \\
        --split val --device cuda --max-images 50

    # 2. 仅输出统计 (不保存可视化)
    python tools/diag/diag_feature_energy.py \\
        --checkpoint <...> --device cuda --max-images 100 --no-plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── Torch imports (deferred) ──
import torch
import torch.nn.functional as F
from PIL import Image

# ── Add project root ──
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from adatile.utils.seed import set_seed
from adatile.logging import get_logger

logger = get_logger("diag_feature_energy")

# ═══════════════════════════════════════════════════════════════════
# Constants | 常量
# ═══════════════════════════════════════════════════════════════════

# 与 diag_pred_vis 一致的面积桶定义 | Area bucket definition (consistent with diag_pred_vis)
AREA_BUCKETS = [
    (0, 32, "<32"),
    (32, 64, "32-64"),
    (64, 128, "64-128"),
    (128, 256, "128-256"),
    (256, 512, "256-512"),
    (512, float("inf"), ">512"),
]

# iSAID 15 类映射 | iSAID 15-class mapping
ISAID_CLASSES = {
    1: "small_vehicle", 2: "large_vehicle", 3: "plane",
    4: "storage_tank", 5: "ship", 6: "harbor",
    7: "ground_track_field", 8: "soccer_ball_field", 9: "tennis_court",
    10: "swimming_pool", 11: "baseball_diamond", 12: "basketball_court",
    13: "bridge", 14: "helicopter", 15: "roundabout",
}

# ═══════════════════════════════════════════════════════════════════
# COCO loading | COCO 标注加载
# ═══════════════════════════════════════════════════════════════════

def _load_coco(data_root: Path, split: str) -> dict:
    """加载 COCO JSON 标注 | Load COCO JSON annotations."""
    ann_file = data_root / "annotations" / f"instances_{split}.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    with open(ann_file) as f:
        return json.load(f)


def _render_instance_mask(
    ann: dict, h: int, w: int, stride: int = 1
) -> np.ndarray | None:
    """
    从单个 COCO polygon 标注渲染二值实例 mask | Render binary instance mask from a single COCO polygon.

    :param ann: COCO annotation dict with 'segmentation' and/or 'bbox'.
    :param h: Feature map height (already strided).
    :param w: Feature map width (already strided).
    :param stride: Feature stride (8 for P3, 16 for P4, 32 for P8).
    :return: [h, w] bool mask, or None if cannot render.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    seg = ann.get("segmentation", [])
    bbox = ann.get("bbox", [0, 0, 0, 0])

    if seg and isinstance(seg, list):
        # Normalize polygon format
        if isinstance(seg[0], list):
            polys = seg
        elif isinstance(seg[0], (int, float)):
            polys = [seg]
        else:
            polys = []
        for poly in polys:
            if len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
            # Scale to feature map resolution
            pts[:, :, 0] = np.clip(pts[:, :, 0] / stride, 0, w - 1)
            pts[:, :, 1] = np.clip(pts[:, :, 1] / stride, 0, h - 1)
            pts = pts.astype(np.int32)
            import cv2
            cv2.fillPoly(mask, [pts], 1)
    else:
        # Fallback to bbox
        bx, by, bw, bh = [int(v / stride) for v in bbox]
        mask[max(0, by):min(h, by + bh), max(0, bx):min(w, bx + bw)] = 1

    if mask.sum() < 1:
        return None
    return mask.astype(bool)


# ═══════════════════════════════════════════════════════════════════
# Feature extraction | 特征提取
# ═══════════════════════════════════════════════════════════════════

def _extract_features_batched(
    model, batch_tensor: torch.Tensor, device: str = "cuda"
) -> dict:
    """
    批量提取 FastSAM backbone 特征 | Batched FastSAM backbone feature extraction.

    手动走 model.model (Sequential[23]), 通过 hook 捕获 P3@15, P4@18, P8@21.
    从 P3 生成 proto masks.

    :param model: FastSAM model.
    :param batch_tensor: [B, 3, H, W] float32 tensor, already on device.
    :return: {p3: [B,C,H/8,W/8], p4: [B,C,H/16,W/16], p8: [B,C,H/32,W/32], proto: [B,32,H/4,W/4]}
    """
    seg = model.model          # SegmentationModel
    seq = seg.model            # Sequential[23]
    save_set = set(seg.save)   # {4, 6, 9, 12, 15, 18, 21}
    segment = seq[22]          # Segment head

    hooked = {}

    def _hook(name):
        def _fn(m, inp, outp):
            hooked[name] = outp.detach()
        return _fn

    handles = [
        seq[15].register_forward_hook(_hook("p3")),
        seq[18].register_forward_hook(_hook("p4")),
        seq[21].register_forward_hook(_hook("p8")),
    ]

    with torch.no_grad():
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

    with torch.no_grad():
        proto = segment.proto(p3)  # [B, 32, H/4, W/4]

    return {"p3": p3, "p4": p4, "p8": p8, "proto": proto}


# ═══════════════════════════════════════════════════════════════════
# Core analysis | 核心分析
# ═══════════════════════════════════════════════════════════════════

def compute_instance_feature_energy(
    feats_p3: torch.Tensor,    # [C3, H3, W3]
    feats_p4: torch.Tensor,    # [C4, H4, W4]
    ann: dict,
    img_h: int,
    img_w: int,
) -> dict | None:
    """
    对单个 GT 实例计算 P3 和 P4 特征能量 | Compute P3/P4 feature energy for a single GT instance.

    Feature Energy = mean L2 norm of feature vectors within the instance mask region.
    特征能量 = 实例 mask 区域内特征向量的 L2 norm 均值.

    :return: {area, p3_energy, p4_energy, p3_n_pixels, p4_n_pixels, category_id} or None.
    """
    area = ann.get("area", 0)
    cat_id = ann.get("category_id", 0)

    if area < 1 or cat_id < 1 or cat_id > 15:
        return None

    # ── P3 mask (stride-8) ──
    H3, W3 = feats_p3.shape[1], feats_p3.shape[2]
    mask_p3 = _render_instance_mask(ann, H3, W3, stride=8)
    if mask_p3 is None or mask_p3.sum() < 1:
        return None

    # P3 energy: mean L2 norm over mask region
    feats_p3_masked = feats_p3[:, mask_p3]  # [C3, n_pixels]
    p3_l2 = feats_p3_masked.norm(dim=0)     # [n_pixels]
    p3_energy = float(p3_l2.mean().item())
    p3_n = int(mask_p3.sum())

    # ── P4 mask (stride-16) ──
    H4, W4 = feats_p4.shape[1], feats_p4.shape[2]
    mask_p4 = _render_instance_mask(ann, H4, W4, stride=16)

    if mask_p4 is None or mask_p4.sum() < 1:
        # P4 分辨率不足以渲染此实例的 mask | P4 resolution insufficient
        p4_energy = 0.0
        p4_n = 0
    else:
        feats_p4_masked = feats_p4[:, mask_p4]
        p4_l2 = feats_p4_masked.norm(dim=0)
        p4_energy = float(p4_l2.mean().item())
        p4_n = int(mask_p4.sum())

    return {
        "area": area,
        "p3_energy": p3_energy,
        "p4_energy": p4_energy,
        "p3_n_pixels": p3_n,
        "p4_n_pixels": p4_n,
        "category_id": cat_id,
    }


# ═══════════════════════════════════════════════════════════════════
# Aggregation | 聚合统计
# ═══════════════════════════════════════════════════════════════════

def aggregate_by_bucket(results: list[dict]) -> dict:
    """
    按面积桶聚合 | Aggregate results by area bucket.

    :return: {bucket_label: {p3_energies: [...], p4_energies: [...], counts: [...]}}
    """
    buckets = defaultdict(lambda: {"p3": [], "p4": [], "p3_n": [], "p4_n": [], "count": 0})

    for r in results:
        area = r["area"]
        for lo, hi, label in AREA_BUCKETS:
            if lo <= area < hi:
                buckets[label]["p3"].append(r["p3_energy"])
                buckets[label]["p4"].append(r["p4_energy"])
                buckets[label]["p3_n"].append(r["p3_n_pixels"])
                buckets[label]["p4_n"].append(r["p4_n_pixels"])
                buckets[label]["count"] += 1
                break

    return dict(buckets)


# ═══════════════════════════════════════════════════════════════════
# Main | 主流程
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Feature Energy Diagnostic — P3 vs P4 per-instance feature response",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to training checkpoint (.pt) for model architecture")
    parser.add_argument("--data-root", type=str,
                        default="/root/autodl-tmp/iSAID_processed",
                        help="Path to iSAID processed data")
    parser.add_argument("--split", type=str, default="val",
                        help="Data split (val/train)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Max images to process (0=all)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip visualization, print stats only")
    parser.add_argument("--output", type=str, default="vis_output",
                        help="Output directory")
    parser.add_argument("--max-tiles-per-batch", type=int, default=64,
                        help="Max tiles per backbone forward pass")
    parser.add_argument("--full-gt", type=str, default=None,
                        help="Full-image COCO JSON path (e.g., .../val/annotations/instances_val.json). "
                             "If not provided, uses {data_root}/annotations/instances_{split}.json")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ═══════════════════════════════════════════════════════════════
    # 1. Load model | 加载模型
    # ═══════════════════════════════════════════════════════════════
    print("=" * 65)
    print("  Feature Energy Diagnostic — P3 vs P4")
    print("=" * 65)

    set_seed(42)

    from ultralytics import FastSAM

    # 使用与训练脚本一致的本地权重路径 | Use same local weight path as training script
    model_path = _project_root / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"
    if not model_path.exists():
        # 回退: 尝试 ultralytics 自动下载 | Fallback: let ultralytics download
        model_path = "FastSAM-x.pt"

    print(f"  Loading FastSAM backbone (from {model_path})...")
    model = FastSAM(str(model_path))
    model = model.cuda() if args.device == "cuda" else model

    # Load checkpoint just for reference (features are frozen backbone, independent of decoder)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 2. Load COCO annotations | 加载 COCO 标注
    # ═══════════════════════════════════════════════════════════════
    print("  Loading COCO annotations...")
    if args.full_gt:
        gt_path = Path(args.full_gt)
        if not gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        with open(gt_path) as f:
            coco = json.load(f)
        # Infer split from path: .../val/annotations/instances_val.json → val
        _gt_split = gt_path.parent.parent.name if gt_path.parent.name == "annotations" else args.split
    else:
        ann_file = data_root / "annotations" / f"instances_{args.split}.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")
        with open(ann_file) as f:
            coco = json.load(f)
        _gt_split = args.split
    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    img_id_to_size = {img["id"]: (img["height"], img["width"]) for img in coco["images"]}
    file_to_anns = defaultdict(list)
    for ann in coco.get("annotations", []):
        fname = img_id_to_file.get(ann["image_id"])
        if fname:
            file_to_anns[fname].append(ann)

    # Find all images (try multiple path conventions)
    img_dir = data_root / _gt_split / "images"
    if not img_dir.exists():
        img_dir = data_root / "images" / _gt_split
    if not img_dir.exists():
        img_dir = data_root / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found (tried: "
                                f"{data_root / _gt_split / 'images'}, "
                                f"{data_root / 'images' / _gt_split}, "
                                f"{data_root / 'images'})")

    all_images = sorted([f for f in os.listdir(str(img_dir))
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if args.max_images > 0:
        all_images = all_images[:args.max_images]

    print(f"  Found {len(all_images)} images, {sum(len(file_to_anns.get(f, [])) for f in all_images)} GT instances")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 3. Process images — extract features, compute per-instance energy
    # ═══════════════════════════════════════════════════════════════
    print("  Processing images...")
    all_results = []
    device = args.device

    # Statistics for the "P4 under-resolution" counter
    p4_missing_count = 0  # instances where P4 mask has 0 pixels
    total_instances = 0

    t_start = time.time()

    for idx, img_name in enumerate(all_images):
        img_path = img_dir / img_name
        img = np.array(Image.open(str(img_path)).convert("RGB"))
        H_img, W_img = img.shape[:2]

        # Get annotations for this image
        anns = file_to_anns.get(img_name, [])
        if not anns:
            continue

        # ── Tile-based feature extraction ──
        from adatile.datasets.isaid_tiles import tile_full_image
        tiles_info = tile_full_image(img, tile_size=896, overlap=32)

        # Stack tiles for batched backbone forward
        all_tensors = []
        for ti in tiles_info:
            tile_img = ti["img"]
            t_tensor = torch.from_numpy(tile_img).permute(2, 0, 1).float().div_(255.0)
            all_tensors.append(t_tensor)

        # Sub-batch to avoid OOM
        all_p3, all_p4 = [], []
        max_per_batch = args.max_tiles_per_batch
        for sub_start in range(0, len(all_tensors), max_per_batch):
            sub_end = min(sub_start + max_per_batch, len(all_tensors))
            sub_batch = torch.stack(all_tensors[sub_start:sub_end]).to(device)
            sub_feats = _extract_features_batched(model, sub_batch, device)
            all_p3.append(sub_feats["p3"].cpu())
            all_p4.append(sub_feats["p4"].cpu())
            del sub_batch, sub_feats

        batched_p3 = torch.cat(all_p3, dim=0)  # [B, 960, H/8, W/8]
        batched_p4 = torch.cat(all_p4, dim=0)  # [B, 1280, H/16, W/16]

        # ── For each GT instance, find which tile it falls in ──
        for ann in anns:
            total_instances += 1
            cat_id = ann.get("category_id", 0)
            if cat_id < 1 or cat_id > 15:
                continue

            bbox = ann.get("bbox", [0, 0, 0, 0])
            bx, by, bw, bh = bbox
            cx = bx + bw / 2
            cy = by + bh / 2

            # Find which tile contains this instance center
            best_tile_idx = None
            best_overlap = 0
            for ti_idx, ti in enumerate(tiles_info):
                tx, ty = ti["x"], ti["y"]
                tw, th = ti["w"], ti["h"]
                # Check bbox-tile overlap
                ox = max(0, min(bx + bw, tx + tw) - max(bx, tx))
                oy = max(0, min(by + bh, ty + th) - max(by, ty))
                overlap = ox * oy
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_tile_idx = ti_idx

            if best_tile_idx is None or best_overlap < 1:
                continue

            ti = tiles_info[best_tile_idx]
            tx, ty = ti["x"], ti["y"]

            # Compute instance energy relative to tile features
            # Adjust ann coordinates to tile-local
            ann_local = dict(ann)
            if "bbox" in ann_local:
                ann_local["bbox"] = [
                    ann["bbox"][0] - tx,
                    ann["bbox"][1] - ty,
                    ann["bbox"][2],
                    ann["bbox"][3],
                ]
            if "segmentation" in ann_local and isinstance(ann_local["segmentation"], list):
                seg = ann_local["segmentation"]
                if isinstance(seg[0], list):
                    ann_local["segmentation"] = [
                        [v - tx if i % 2 == 0 else v - ty for i, v in enumerate(poly)]
                        for poly in seg
                    ]
                elif isinstance(seg[0], (int, float)):
                    ann_local["segmentation"] = [
                        v - tx if i % 2 == 0 else v - ty
                        for i, v in enumerate(seg)
                    ]

            feats_p3_tile = batched_p3[best_tile_idx]  # [960, H/8, W/8]
            feats_p4_tile = batched_p4[best_tile_idx]  # [1280, H/16, W/16]

            # Clip bbox to valid range
            ann_bbox = ann_local.get("bbox", [0, 0, 0, 0])
            if ann_bbox[0] < -896 or ann_bbox[1] < -896 or ann_bbox[0] > 1792 or ann_bbox[1] > 1792:
                continue

            result = compute_instance_feature_energy(
                feats_p3_tile, feats_p4_tile,
                ann_local,
                ti["h"], ti["w"],
            )

            if result is not None:
                if result["p4_n_pixels"] == 0:
                    p4_missing_count += 1
                all_results.append(result)

        # Cleanup
        del batched_p3, batched_p4, all_p3, all_p4, all_tensors

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_start
            print(f"    [{idx+1}/{len(all_images)}] {len(all_results)} instances, "
                  f"{elapsed:.1f}s, P4-missing: {p4_missing_count}/{total_instances}")

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Total instances: {total_instances}")
    print(f"  Valid results: {len(all_results)}")
    print(f"  P4 under-resolution (0 feature pixels): {p4_missing_count}/{total_instances} "
          f"({p4_missing_count/max(total_instances,1)*100:.1f}%)")
    print()

    # ═══════════════════════════════════════════════════════════════
    # 4. Aggregate by area bucket | 按面积桶聚合
    # ═══════════════════════════════════════════════════════════════
    buckets = aggregate_by_bucket(all_results)

    print("=" * 65)
    print("  FEATURE ENERGY BY AREA BUCKET")
    print("=" * 65)
    print(f"  {'Bucket':<12} {'Count':>6} {'P3 Energy':>10} {'P4 Energy':>10} "
          f"{'P3/P4':>8} {'P3 px':>7} {'P4 px':>7} {'P4 Lost%':>8}")
    print(f"  {'-'*70}")

    for lo, hi, label in AREA_BUCKETS:
        b = buckets.get(label, {"p3": [], "p4": [], "p3_n": [], "p4_n": [], "count": 0})
        if b["count"] == 0:
            print(f"  {label:<12} {'0':>6} {'N/A':>10} {'N/A':>10} {'N/A':>8}")
            continue
        p3_e = np.mean(b["p3"])
        p4_e = np.mean(b["p4"])
        p3_n = np.mean(b["p3_n"])
        p4_n = np.mean(b["p4_n"])
        ratio = p3_e / max(p4_e, 1e-8)
        p4_lost = max(0, (1 - p4_n / max(p3_n, 1))) * 100
        print(f"  {label:<12} {b['count']:>6} {p3_e:>10.4f} {p4_e:>10.4f} "
              f"{ratio:>8.2f} {p3_n:>7.1f} {p4_n:>7.1f} {p4_lost:>7.1f}%")

    print()
    print("  Key metrics | 关键指标:")
    print(f"    P3/P4 energy ratio at <32 px²: "
          f"{np.mean(buckets.get('<32', {}).get('p3', [0])) / max(np.mean(buckets.get('<32', {}).get('p4', [0])), 1e-8):.2f}")
    print(f"    P4 feature pixel loss at <32 px²: "
          f"{max(0, (1 - np.mean(buckets.get('<32', {}).get('p4_n', [1])) / max(np.mean(buckets.get('<32', {}).get('p3_n', [1])), 1)) * 100):.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 5. Per-class analysis | 按类别分析
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 65)
    print("  PER-CLASS FEATURE ENERGY")
    print("=" * 65)
    print(f"  {'Class':<22} {'Count':>6} {'P3 Energy':>10} {'P4 Energy':>10} "
          f"{'P3/P4':>8} {'Typical Size':>14}")
    print(f"  {'-'*75}")

    class_results = defaultdict(lambda: {"p3": [], "p4": []})
    for r in all_results:
        cid = r["category_id"]
        class_results[cid]["p3"].append(r["p3_energy"])
        class_results[cid]["p4"].append(r["p4_energy"])

    typical_sizes = {
        1: "8-20px", 2: "20-80px", 3: "30-100px", 4: "10-40px", 5: "15-80px",
        6: "50-300px", 7: "50-200px", 8: "80-200px", 9: "40-120px", 10: "20-80px",
        11: "50-150px", 12: "30-80px", 13: "20-200px", 14: "15-40px", 15: "20-80px",
    }

    for cid in sorted(class_results.keys(), key=lambda c: np.mean(class_results[c]["p3"]) / max(np.mean(class_results[c]["p4"]), 1e-8), reverse=True):
        cr = class_results[cid]
        name = ISAID_CLASSES.get(cid, f"class_{cid}")
        p3_e = np.mean(cr["p3"])
        p4_e = np.mean(cr["p4"])
        ratio = p3_e / max(p4_e, 1e-8)
        tsize = typical_sizes.get(cid, "?")
        print(f"  {name:<22} {len(cr['p3']):>6} {p3_e:>10.4f} {p4_e:>10.4f} "
              f"{ratio:>8.2f} {tsize:>14}")

    # ═══════════════════════════════════════════════════════════════
    # 6. Save results | 保存结果
    # ═══════════════════════════════════════════════════════════════
    output_file = output_dir / "feature_energy.json"
    with open(output_file, "w") as f:
        # Convert numpy values for JSON
        json_buckets = {}
        for label, b in buckets.items():
            json_buckets[label] = {
                "count": b["count"],
                "p3_energy_mean": float(np.mean(b["p3"])) if b["p3"] else 0,
                "p4_energy_mean": float(np.mean(b["p4"])) if b["p4"] else 0,
                "p3_energy_std": float(np.std(b["p3"])) if b["p3"] else 0,
                "p4_energy_std": float(np.std(b["p4"])) if b["p4"] else 0,
                "p3_n_pixels_mean": float(np.mean(b["p3_n"])) if b["p3_n"] else 0,
                "p4_n_pixels_mean": float(np.mean(b["p4_n"])) if b["p4_n"] else 0,
            }
        json.dump({
            "total_instances": total_instances,
            "valid_results": len(all_results),
            "p4_missing_count": p4_missing_count,
            "p4_missing_pct": p4_missing_count / max(total_instances, 1) * 100,
            "buckets": json_buckets,
        }, f, indent=2)

    print(f"\n  [SAVED] Results → {output_file}")
    print("  Done.")


if __name__ == "__main__":
    main()
