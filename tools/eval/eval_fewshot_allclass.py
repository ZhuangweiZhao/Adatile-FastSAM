#!/usr/bin/env python3
"""
评估全类 K-shot Fine-tuning 模型 + Zero-Shot baseline 对比.
Evaluate All-Class K-shot Fine-tuned Model vs Zero-Shot baseline.
================================================================

加载微调后的 FewShotDecoder，在同一批测试数据上同时评估:
Loads fine-tuned FewShotDecoder, evaluates on same test data:
    1. Fine-tuned: support prototype → decoder → mask
    2. Zero-Shot: GT bbox → FastSAM bbox prompt → mask

支持三种数据格式 / Supports three data formats:
    isaid5i / isaid_tiles / isaid_instance

用法 | Usage::

    # iSAID-5i 格式
    python tools/eval/eval_fewshot_allclass.py \
        --checkpoint runs/xxx/best_model.pt --k-shot 3

    # 新 COCO tile 格式 (iSAID-few_tiles)
    python tools/eval/eval_fewshot_allclass.py \
        --checkpoint runs/xxx/best_model.pt --k-shot 3 \
        --data-format isaid_instance --data-root data/iSAID-few_tiles

输出 | Output:
    Per-class: Fine-tuned IoU vs Zero-Shot IoU (same test samples)
"""

from __future__ import annotations

import sys, argparse, json, random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

from adatile.utils.seed import set_seed

from tools.train.train_fewshot_allclass import (
    _build_class_index_isaid5i, _build_class_index_tiles, _build_class_index_instance,
    _extract_source_image,
    load_image_and_mask, load_tile_and_mask, load_instance_tile_and_mask,
    semantic_mask_to_binary,
    extract_features, compute_support_prototype, compute_support_mask_template,
    build_full_image_gt, merge_tile_predictions,
    FewShotDecoder, CATEGORY_NAMES, _resolve_paths, _normalize_mask_to_4d,
)
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
from adatile.decoder.adaptive_decoder_p3p4 import AdaptiveDecoderP3P4
from adatile.sparse.spm import SparsePerceptionModule


# ═══════════════════════════════════════════════════════════════════
# Zero-Shot inference | FastSAM bbox prompt (inline to avoid import issues)
# ═══════════════════════════════════════════════════════════════════

def mask_to_bbox(mask: np.ndarray) -> tuple:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    h, w = mask.shape
    if not rows.any() or not cols.any():
        return (0, 0, max(w, 1), max(h, 1))
    y_idx = np.where(rows)[0]
    x_idx = np.where(cols)[0]
    y1, y2 = int(y_idx[0]), int(y_idx[-1])
    x1, x2 = int(x_idx[0]), int(x_idx[-1])
    # 保证最小 2px | Ensure min 2px
    if x2 <= x1: x2 = x1 + 2
    if y2 <= y1: y2 = y1 + 2
    # Clip to image bounds
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)
    return (x1, y1, x2, y2)


def zero_shot_bbox_iou(model, image: np.ndarray, gt_mask: np.ndarray,
                       device: str = "cuda") -> dict:
    """
    Zero-shot bbox-prompted FastSAM → mask IoU vs GT.
    对 query 图像使用 GT bbox 提示 FastSAM，返回与 GT 的 IoU.
    """
    H, W = image.shape[:2]
    bbox = mask_to_bbox(gt_mask)

    x1, y1, x2, y2 = bbox
    try:
        results = model(
            source=image, device=device, retina_masks=True,
            imgsz=max(H, W), conf=0.001, iou=0.9,
            bboxes=[[x1, y1, x2, y2]], verbose=False,
        )
    except Exception:
        return {"iou": 0.0, "valid": False, "n_masks": 0, "bbox": bbox}

    if results is None or len(results) == 0 or results[0].masks is None:
        return {"iou": 0.0, "valid": False, "n_masks": 0, "bbox": bbox}

    masks_data = results[0].masks.data
    if len(masks_data) == 0:
        return {"iou": 0.0, "valid": False, "n_masks": 0, "bbox": bbox}

    # Select best mask by GT IoU (Oracle selection — upper bound for zero-shot)
    best_iou = 0.0
    for i in range(len(masks_data)):
        m = masks_data[i].cpu().numpy()
        if m.shape != (H, W):
            m = F.interpolate(
                torch.tensor(m).unsqueeze(0).unsqueeze(0).float(),
                size=(H, W), mode="bilinear",
            ).squeeze().cpu().numpy()
        m_bin = (m > 0.5).astype(np.bool_)
        gt_bin = gt_mask.astype(np.bool_)
        inter = float((m_bin & gt_bin).sum())
        union = float((m_bin | gt_bin).sum())
        iou = inter / union if union > 0 else 0.0
        if not np.isnan(iou) and iou > best_iou:
            best_iou = iou

    return {"iou": float(best_iou), "valid": best_iou > 0, "n_masks": len(masks_data), "bbox": bbox}


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned vs zero-shot")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--per-class", type=int, default=20,
                        help="Test episodes per class (default: 20)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-root", type=str, default=None,
                        help="数据根目录 (默认根据格式自动选择) | Data root directory")
    parser.add_argument("--data-format", type=str, default="isaid5i",
                        choices=["isaid5i", "isaid_tiles", "isaid_instance"],
                        help="数据格式 / Data format")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--decoder", type=str, default="baseline",
                        choices=["baseline", "adaptive", "adaptive-p3p4"],
                        help="Decoder 类型 (需与训练时一致) | Decoder type (must match training)")
    parser.add_argument("--use-spm", action="store_true",
                        help="启用 SPM tile routing (需与训练时一致)")
    parser.add_argument("--spm-topk", type=float, default=0.4,
                        help="SPM 保留的 tile 比例 | Fraction of tiles kept by SPM")
    parser.add_argument("--spm-oracle", action="store_true",
                        help="Oracle mode: 用 GT FG ratio 代替 SPM (验证 Top-K 效率上限)")
    parser.add_argument("--prototype-source", type=str, default="p4",
                        choices=["p4", "p8"],
                        help="Prototype 特征来源 (需与训练时一致) | Prototype source (must match training)")
    parser.add_argument("--proto-ablation", type=str, default="none",
                        choices=["none", "zero", "random", "shuffle"],
                        help="Prototype 消融实验 | Prototype ablation: "
                             "none (正常) / zero (全零) / random (随机) / shuffle (跨类交换)")
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    device = args.device

    # ── 解析数据格式 | Resolve data format ──
    data_root, data_format, train_split, val_split = _resolve_paths(args)
    eval_split = val_split  # 评估用 val 集 | Eval uses val split
    is_instance_tile = (data_format == "isaid_instance")
    is_tile = (data_format in ("isaid_tiles", "isaid_instance"))

    fmt_labels = {"isaid5i": "iSAID-5i", "isaid_tiles": "TILES", "isaid_instance": "TILES-COCO"}
    fmt_label = fmt_labels.get(data_format, data_format)

    if args.output_dir is None:
        ts = datetime.now().strftime("%m%d_%H%M")
        ckpt_name = Path(args.checkpoint).parent.name
        args.output_dir = f"runs/eval_fewshot_{ckpt_name}_{ts}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  Fine-tuned vs Zero-Shot Comparison")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data: {data_root} (format: {fmt_label})")
    print(f"  Mode: {fmt_label} | K-shot: {args.k_shot} | Per-class: {args.per_class} | Seed: {args.seed}")
    if args.proto_ablation != "none":
        print(f"  Proto Ablation: {args.proto_ablation.upper()}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")

    # Load models
    print(f"\n[1/3] Loading models...")
    from ultralytics import FastSAM
    fastsam_path = _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"
    model = FastSAM(str(fastsam_path))
    model.model.cuda().eval()
    for p in model.model.parameters():
        p.requires_grad = False

    # ── 加载 checkpoint | Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device)

    # ── 恢复 backbone 权重 (如果 checkpoint 包含) | Restore backbone weights ──
    _unfreeze_layers = ckpt.get("unfreeze_layers", 0)
    if _unfreeze_layers > 0 and "backbone" in ckpt:
        seq = model.model.model  # Sequential[23]
        n_total = len(seq)
        start = max(0, n_total - _unfreeze_layers)
        for i_str, state in ckpt["backbone"].items():
            i = int(i_str)
            seq[i].load_state_dict(state)
        print(f"  Backbone: loaded {_unfreeze_layers} layers "
              f"(indices {start}-{n_total-1}) from checkpoint")
        # 不需要 requires_grad (eval only) | No requires_grad needed (eval only)
    elif _unfreeze_layers > 0:
        print(f"  [WARN] Checkpoint has unfreeze_layers={_unfreeze_layers} but no backbone weights")

    # Auto-detect P4 channels | 自动检测 P4 通道数
    _test_img = np.zeros((896, 896, 3), dtype=np.uint8)
    _test_feats = extract_features(model, [_test_img], device)
    _p4_channels = _test_feats[0]["p4"].shape[1]

    if args.decoder == "adaptive":
        decoder = AdaptiveSparseDecoder(in_channels=_p4_channels, use_fdr=False).to(device)
    elif args.decoder == "adaptive-p3p4":
        _p3_channels = _test_feats[0]["p3"].shape[1]
        print(f"  Detected P3 channels: {_p3_channels}")
        decoder = AdaptiveDecoderP3P4(p3_channels=_p3_channels, p4_channels=_p4_channels).to(device)
    else:
        decoder = FewShotDecoder(feat_dim=_p4_channels).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()

    # SPM (可选) | Optional SPM
    spm = None
    if args.use_spm:
        spm = SparsePerceptionModule(in_channels=_p4_channels, mid_channels=256).to(device)
        if "spm" in ckpt:
            spm.load_state_dict(ckpt["spm"])
            spm.eval()
        else:
            print("  [WARN] --use-spm set but checkpoint has no SPM weights")
            spm = None

    bb_label = f" + BB(uf{_unfreeze_layers})" if _unfreeze_layers > 0 else ""
    spm_label = " + SPM" if spm is not None else ""
    abl_label = f" [proto={args.proto_ablation}]" if args.proto_ablation != "none" else ""
    print(f"  FastSAM{bb_label} + {args.decoder} Decoder{spm_label} loaded "
          f"(epoch {ckpt.get('epoch', '?')}){abl_label}")

    # Build test index (fixed by seed)
    print(f"\n[2/3] Building test index ({fmt_label}, seed={args.seed})...")
    if data_format == "isaid_instance":
        class_index = _build_class_index_instance(data_root, eval_split)
    elif data_format == "isaid_tiles":
        class_index = _build_class_index_tiles(data_root, eval_split)
    else:
        class_index = _build_class_index_isaid5i(data_root, "train")
    rng = random.Random(args.seed)
    rng_query = random.Random(args.seed + 99999)  # 独立 RNG 固定 query | Separate RNG for fixed queries

    # Pre-sample test episodes per class: [(support_stems, query_stem), ...]
    # 关键修复 | Critical fix:
    #   - Query tile 选择独立于 K（不同 K 值用相同 queries）
    #   - Support tiles 必须来自不同的源图像（scene diversity）
    #   - Query selection is K-independent. Support from different source images.
    test_episodes = {}  # cls_id → [(support_stems, query_stem), ...]
    max_k = 10  # 预分配足够多的 support source | Reserve enough support sources for max K
    test_eps_for_json = {}  # 用于保存到 JSON | For JSON export
    for cls_id, src_to_tiles in list(class_index.items()):
        # src_to_tiles = {source_img: [tile_stems]}
        sources = list(src_to_tiles.keys())
        if len(sources) < max_k + 1:
            n_episodes = min(args.per_class, max(1, len(sources) - 1))
        else:
            n_episodes = args.per_class

        if len(sources) < 2:  # 至少需要 2 个源图像 (1 support + 1 query)
            print(f"  [SKIP] Class {cls_id}: only {len(sources)} source images")
            continue

        # Step 1: 用独立 RNG 固定 query source images + tiles | Fix queries K-independently
        query_sources = rng_query.sample(sources, min(n_episodes, len(sources)))
        query_tiles = [random.Random(args.seed + hash(s)).choice(src_to_tiles[s])
                       for s in query_sources]

        # Step 2: 用主 RNG 为每个 query 采样 K 个 support (来自不同源图)
        # Sample K supports from different source images per query with main RNG
        episodes = []
        eps_json_list = []
        for qi, (q_tile, q_src) in enumerate(zip(query_tiles, query_sources)):
            support_sources = [s for s in sources if s != q_src]
            if len(support_sources) < args.k_shot:
                continue
            sampled_srcs = rng.sample(support_sources, min(max_k, len(support_sources)))  # Reserve max_k supports
            supports = []
            for s in sampled_srcs[:args.k_shot]:
                supports.extend(src_to_tiles[s])  # ALL tiles per source image
            episodes.append((supports, q_tile))
            eps_json_list.append({
                "query_source": q_src,
                "query_tile": q_tile,
                "support_sources": sampled_srcs,  # All max_k supports (use first K)
            })
        test_episodes[cls_id] = episodes
        test_eps_for_json[str(cls_id)] = eps_json_list
        name = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")
        print(f"  Class {cls_id:>2d} ({name:<18s}): {len(episodes)} test episodes "
              f"(from {len(sources)} source images)")

    # ── 保存固定测试集到 JSON (Priority 5: reproducibility) ──
    test_eps_json_path = out_dir / "fixed_test_episodes.json"
    test_eps_config = {
        "description": "Fixed test episodes for reproducible evaluation",
        "seed": args.seed,
        "k_shot": args.k_shot,
        "max_k": max_k,
        "checkpoint": args.checkpoint,
        "data_root": str(data_root),
        "data_format": data_format,
        "eval_split": eval_split,
        "protocol": {
            "shot_definition": "K = number of source images (not tiles)",
            "support": "all tiles from K source images",
            "query": "full source image → all tiles → merge → full-image IoU",
            "scene_overlap": "0% (support ∩ query sources = ∅)",
            "query_independence": "Query tiles fixed by independent RNG (seed+99999) — cross-K fair",
        },
        "episodes": test_eps_for_json,
    }
    with open(test_eps_json_path, "w", encoding="utf-8") as f:
        json.dump(test_eps_config, f, indent=2, ensure_ascii=False)
    print(f"  [SAVED] Fixed test episodes → {test_eps_json_path}")
    print(f"  [NOTE]  Same query tiles for all K values — cross-K comparison is fair.")

    # Evaluate
    print(f"\n[3/3] Evaluating (fine-tuned + zero-shot on same data)...")
    results = {}

    for cls_id in sorted(test_episodes.keys()):
        episodes = test_episodes[cls_id]
        name = CATEGORY_NAMES.get(cls_id, f"cls{cls_id}")
        ft_ious, zs_ious = [], []
        src_to_tiles = class_index[cls_id]  # {source: [tiles]}

        # Determine query source for each episode by extracting from query_tile
        # (query_tile belongs to a specific source image)
        episode_sources = []
        for support_stems, query_tile in episodes:
            q_src = _extract_source_image(query_tile)
            episode_sources.append((support_stems, q_src, query_tile))

        for support_stems, query_src, query_tile in tqdm(episode_sources,
                                                          desc=f"  cls {cls_id:>2d} {name}",
                                                          unit="ep", leave=False):
            # ── Support: 所有 tile → prototype ──
            ft_iou = None
            zs_iou = None
            try:
                support_imgs, support_bmasks_v = [], []
                for s in support_stems:
                    if is_instance_tile:
                        simg, smask = load_instance_tile_and_mask(s, eval_split, data_root)
                    elif is_tile:
                        simg, smask = load_tile_and_mask(s, eval_split, data_root)
                    else:
                        simg, smask = load_image_and_mask(s, eval_split, data_root)
                    support_imgs.append(simg)
                    support_bmasks_v.append(semantic_mask_to_binary(smask, is_tile=is_tile))
                support_feats = extract_features(model, support_imgs, device)
                support_proto = compute_support_prototype(support_feats,
                                                          source=args.prototype_source)

                # ── Prototype Ablation (消融实验) ──
                if args.proto_ablation == "zero":
                    support_proto = torch.zeros_like(support_proto)
                elif args.proto_ablation == "random":
                    support_proto = F.normalize(
                        torch.randn_like(support_proto), p=2, dim=-1)
                elif args.proto_ablation == "shuffle":
                    # 随机归一化向量模拟"另一个类" | Random normalized vector = "other class"
                    shuffled = F.normalize(
                        torch.randn_like(support_proto), p=2, dim=-1)
                    support_proto = shuffled

                if args.decoder in ("adaptive", "adaptive-p3p4"):
                    support_tmpl = None
                else:
                    support_tmpl = compute_support_mask_template(support_bmasks_v).to(device)

                # ── FT Query: 整张源图 → 所有 tile → 预测 → 合并 → 全图 IoU ──
                if is_instance_tile and query_src in src_to_tiles:
                    full_gt, H_full, W_full, tile_data = build_full_image_gt(
                        query_src, src_to_tiles[query_src], eval_split, data_root)
                    if H_full > 1 and W_full > 1:
                        predictions = []
                        zs_tile_ious = []
                        n_skipped = 0

                        # SPM / Oracle 预扫描: 计算所有 tile 的重要性 | Pre-scan tile importance
                        tile_importances = []
                        if args.spm_oracle:
                            # Oracle: 用 GT FG ratio 排序 → 证明 Top-K 效率上限
                            # Uses GT FG ratio → proves upper bound of Top-K efficiency
                            for td in tile_data:
                                fg_ratio = td["mask"].mean()  # GT FG ratio
                                tile_importances.append(float(fg_ratio))
                            oracle_label = "Oracle"
                        elif spm is not None:
                            for td in tile_data:
                                q_feats = extract_features(model, [td["img"]], device)[0]
                                p8 = q_feats.get("p8")
                                if p8 is not None:
                                    imp = torch.sigmoid(spm.importance_head(p8.to(device))).max().item()
                                else:
                                    imp = 1.0  # fallback: keep all
                                tile_importances.append(imp)
                            oracle_label = "SPM"
                        else:
                            oracle_label = None
                        if oracle_label is not None:
                            # Top-K 选择 | Top-K selection
                            n_total = len(tile_importances)
                            n_select = max(1, int(n_total * args.spm_topk))
                            topk_idx = set(
                                sorted(range(n_total), key=lambda i: tile_importances[i], reverse=True)[:n_select]
                            )
                        else:
                            topk_idx = set(range(len(tile_data)))  # all tiles

                        for i, td in enumerate(tile_data):
                            if oracle_label is not None and i not in topk_idx:
                                # 跳过低重要性 tile → 预测为零 | Skip low-importance → predict all zero
                                pred_bin_np = np.zeros((td["h"], td["w"]), dtype=np.float32)
                                n_skipped += 1
                            else:
                                q_feats = extract_features(model, [td["img"]], device)[0]
                                with torch.no_grad():
                                    if args.decoder == "adaptive":
                                        mask_s4 = decoder(q_feats["p4"], q_feats["proto"], support_proto)
                                        mask_s4 = _normalize_mask_to_4d(mask_s4)
                                        pred = F.interpolate(mask_s4, size=(td["h"], td["w"]),
                                                             mode="bilinear", align_corners=False)
                                        pred_bin_np = (pred > 0.5).float().squeeze().cpu().numpy()
                                    elif args.decoder == "adaptive-p3p4":
                                        mask_s4 = decoder(q_feats["p3"], q_feats["p4"],
                                                          q_feats["proto"], support_proto)
                                        mask_s4 = _normalize_mask_to_4d(mask_s4)
                                        pred = F.interpolate(mask_s4, size=(td["h"], td["w"]),
                                                             mode="bilinear", align_corners=False)
                                        pred_bin_np = (pred > 0.5).float().squeeze().cpu().numpy()
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
                            # ZS per tile
                            zs = zero_shot_bbox_iou(model, td["img"], td["mask"], device)
                            zs_tile_ious.append(zs["iou"])

                        full_pred = merge_tile_predictions(predictions, H_full, W_full)
                        full_pred_bin = (full_pred > 0.5).astype(np.float32)
                        inter = (full_pred_bin * full_gt.astype(np.float32)).sum()
                        union = (full_pred_bin + full_gt.astype(np.float32)).clip(0, 1).sum()
                        ft_iou = float(inter / max(union, 1))
                        zs_iou = float(np.mean(zs_tile_ious)) if zs_tile_ious else 0.0
                else:
                    # Fallback: single tile query
                    q_img, q_mask = load_instance_tile_and_mask(query_tile, eval_split, data_root) \
                        if is_instance_tile else (
                        load_tile_and_mask(query_tile, eval_split, data_root) if is_tile
                        else load_image_and_mask(query_tile, eval_split, data_root))
                    q_feats = extract_features(model, [q_img], device)[0]
                    q_gt = semantic_mask_to_binary(q_mask, is_tile=is_tile)
                    with torch.no_grad():
                        if args.decoder == "adaptive":
                            mask_s4 = decoder(q_feats["p4"], q_feats["proto"], support_proto)
                            mask_s4 = _normalize_mask_to_4d(mask_s4)
                            H_gt, W_gt = q_gt.shape
                            pred = F.interpolate(mask_s4, size=(H_gt, W_gt),
                                                 mode="bilinear", align_corners=False)
                            pred_bin = (pred > 0.5).float()
                        elif args.decoder == "adaptive-p3p4":
                            mask_s4 = decoder(q_feats["p3"], q_feats["p4"],
                                              q_feats["proto"], support_proto)
                            mask_s4 = _normalize_mask_to_4d(mask_s4)
                            H_gt, W_gt = q_gt.shape
                            pred = F.interpolate(mask_s4, size=(H_gt, W_gt),
                                                 mode="bilinear", align_corners=False)
                            pred_bin = (pred > 0.5).float()
                        else:
                            pred = decoder(q_feats["p4"], q_feats["proto"], support_proto, support_tmpl)
                            H_gt, W_gt = q_gt.shape
                            pred = F.interpolate(pred, size=(H_gt, W_gt),
                                                 mode="bilinear", align_corners=False)
                            pred_bin = (torch.sigmoid(pred) > 0.5).float()
                        gt_t = torch.from_numpy(q_gt).unsqueeze(0).unsqueeze(0).float().to(device)
                        inter = (pred_bin * gt_t).sum()
                        union = (pred_bin + gt_t).clamp(0, 1).sum()
                        ft_iou = (inter / max(union, 1)).item()
                    zs = zero_shot_bbox_iou(model, q_img, q_gt, device)
                    zs_iou = zs["iou"]
            except Exception:
                continue

            ft_ious.append(ft_iou)
            if zs_iou is not None:
                zs_ious.append(zs_iou)

        if ft_ious:
            zs_mean = float(np.mean(zs_ious)) if zs_ious else 0.0
            delta = float(np.mean(ft_ious)) - zs_mean if zs_ious else float(np.mean(ft_ious))
            results[cls_id] = {
                "name": name,
                "n": len(ft_ious),
                "ft_mean": float(np.mean(ft_ious)),
                "ft_std": float(np.std(ft_ious)),
                "zs_mean": zs_mean,
                "zs_std": float(np.std(zs_ious)) if zs_ious else 0.0,
                "delta": delta,
                "zs_valid": len(zs_ious),
            }

    # Print comparison table
    print(f"\n  ── Fine-tuned (K={args.k_shot}) vs Zero-Shot ──")
    print(f"  {'Class':<22s} {'N':>4s} {'FT-IoU':>8s} {'ZS-IoU':>8s} "
          f"{'Delta':>8s} {'ZS-ok':>5s} {'Result'}")
    print(f"  {'-' * 76}")
    for cls_id in sorted(results.keys()):
        r = results[cls_id]
        zs_ok = r.get("zs_valid", 0)
        flag = "↑ FT better" if r["delta"] > 0.02 else ("↓ ZS better" if r["delta"] < -0.02 else "≈ tie")
        zs_str = f"{r['zs_mean']:>8.4f}" if zs_ok > 0 else "   FAIL"
        print(f"  {cls_id}:{r['name']:<19s} {r['n']:>4d} "
              f"{r['ft_mean']:>8.4f} {zs_str:>8s} "
              f"{r['delta']:>+8.4f} {zs_ok:>5d}  {flag}")

    ft_overall = np.mean([r["ft_mean"] for r in results.values()])
    zs_vals = [r["zs_mean"] for r in results.values() if r.get("zs_valid", 0) > 0]
    zs_overall = np.mean(zs_vals) if zs_vals else 0.0
    delta_overall = ft_overall - zs_overall
    print(f"\n  {'─' * 66}")
    print(f"  {'Overall':<22s} {'':>4s} "
          f"{ft_overall:>8.4f} {zs_overall:>8.4f} "
          f"{delta_overall:>+8.4f}  "
          f"{'FT wins' if delta_overall > 0 else 'ZS wins'}")
    if spm is not None or args.spm_oracle:
        spm_keep_pct = 100 * args.spm_topk
        mode = "Oracle (GT FG ratio)" if args.spm_oracle else "SPM"
        print(f"  {mode}: kept top {spm_keep_pct:.0f}% tiles (~{100-spm_keep_pct:.0f}% skipped)")

    # Save
    per_class_out = {}
    for cls_id, r in results.items():
        per_class_out[str(cls_id)] = {
            "name": r["name"], "n": r["n"],
            "ft_mean_iou": round(r["ft_mean"], 4),
            "zs_mean_iou": round(r["zs_mean"], 4),
            "delta": round(r["delta"], 4),
        }
    stats = {
        "k_shot": args.k_shot, "checkpoint": args.checkpoint,
        "seed": args.seed, "per_class": args.per_class,
        "ft_overall_mean_iou": round(ft_overall, 4),
        "zs_overall_mean_iou": round(zs_overall, 4),
        "delta_overall": round(delta_overall, 4),
        "per_class": per_class_out,
    }
    with open(out_dir / "comparison.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  [OK] comparison.json → {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
