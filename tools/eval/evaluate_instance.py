#!/usr/bin/env python3
"""
实例分割评估器 V3 | Instance Segmentation Evaluator V3.
========================================================

严格符合 Few-shot Remote Sensing Instance Segmentation 的评估协议:
Strict Evaluation Protocol for Few-shot Remote Sensing Instance Segmentation:

    1. 实例级评估 — 绝不把实例 merge 成 union mask 再算 IoU.
       Instance-level — NEVER merge instances into a union mask for IoU.
    2. 每个 GT 实例独立参与, 预测与 GT 一对一贪心匹配.
       Each GT instance participates independently; greedy one-to-one matching.
    3. COCO AP / AP50 / AP75 / APS / APM / APL — 官方 pycocotools 计算.
       COCO AP family via official pycocotools (no re-implementation).
    4. Instance mIoU — 每个 GT 取最大 IoU 预测再平均.
       Instance mIoU — per-GT max-IoU prediction, then averaged.
    5. Zero-shot 基线严格非 oracle — 只用 FastSAM 默认输出, 不看 GT 选 mask.
       Zero-shot baseline strictly non-oracle — FastSAM default output only.
    6. 输出 overall / per-class 指标 + 每类 GT/Pred/TP/FP/FN 便于调试.
       Outputs overall / per-class metrics + per-class GT/Pred/TP/FP/FN for debugging.

模型输出如何转成实例 | How model output becomes instances:
    decoder 输出的是"某类别的语义前景图"(该类所有实例合并), 不是实例.
    我们对前景图做 **连通域分解** (connected components), 每个连通块 = 一个预测实例,
    置信度 = 该块内前景概率均值. 这是纯模型输出, 无 oracle, 无 FastSAM proposal.
    The decoder outputs a class-conditioned *semantic* foreground map (all instances of
    the class merged), not instances. We split it via **connected components**: each
    component = one predicted instance, score = mean FG prob inside it. Model-only, no oracle.

评估单元 | Evaluation unit:
    按 tile (896²) — GT 的 COCO JSON 本身以 tile 为 image (image_id = tile), 无需坐标重映射.
    Per-tile (896²) — the COCO GT JSON is organized by tile (image_id = tile), no remap needed.

用法 | Usage::

    python tools/eval/evaluate_instance.py \
        --checkpoint runs/.../best_model.pt \
        --decoder adaptive --prototype-source p4 \
        --data-root data/iSAID_instance_fewshot --data-format isaid_instance \
        --k-shot 1 --per-class 20 --device cuda

不修改训练流程, 不改动旧评估脚本 eval_fewshot_allclass.py.
Does NOT touch the training flow or the legacy eval_fewshot_allclass.py.
"""

from __future__ import annotations

import sys
import json
import random
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

# ── 复用库设施 | Reuse library infrastructure ──
from adatile.utils.seed import set_seed
from adatile.logging import get_logger
from adatile.logging.backends import ConsoleBackend, FileBackend
from adatile.metrics.coco_eval import (
    COCOInstanceEvaluator,
    connected_components_to_instances,
    mask_to_bbox,
)
from adatile.metrics.instance_match import greedy_match, instance_miou
from adatile.metrics.instance_generation import generate_instances, InstanceMethod, generate_instances_center_affinity

# ── 复用训练脚本中的数据/特征/原型工具 | Reuse data/feature/prototype helpers ──
from tools.train.train_fewshot_allclass import (
    CATEGORY_NAMES,
    _resolve_paths,
    _extract_source_image,
    _build_class_index_instance,
    _build_class_index_tiles,
    load_instance_tile_and_mask,
    load_tile_and_mask,
    extract_features,
    compute_support_prototype,
    compute_support_mask_template,
    semantic_mask_to_binary,
    _normalize_mask_to_4d,
)
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
from adatile.decoder.adaptive_decoder_p3p4 import AdaptiveDecoderP3P4
from adatile.decoder.pure_cnn_decoder import PureDecoder, PureDecoderP3P4
from adatile.decoder.dynamic_kernel_decoder import DynamicKernelDecoder
from adatile.decoder.center_affinity_decoder import CenterAffinityDecoder

# 类别是否具备条件化能力 (需要 prototype) | Whether a decoder is class-conditioned
_CLASS_CONDITIONED = {"adaptive", "adaptive-p3p4", "baseline", "dynamic_kernel", "center_affinity"}


# ═══════════════════════════════════════════════════════════════════
# 确定性哈希 + 评估清单 | Deterministic hash + evaluation manifest
# ═══════════════════════════════════════════════════════════════════

def _det_hash(s: str) -> int:
    """确定性哈希 (跨 OS / Python 版本一致) | Deterministic hash (stable across OS/Python).

    禁用 Python 内置 hash(): 它对 str 带每进程随机盐 (PYTHONHASHSEED), 会使同一 seed 在
    不同运行/机器上选到不同 query tile, 破坏可复现性. md5 保证同名永远映射到同一整数.
    Avoids the built-in hash(), which is per-process salted for str (PYTHONHASHSEED) and would
    make the same seed pick different query tiles across runs/machines. md5 is stable.
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def load_manifest(path: Path) -> list[str]:
    """读取评估清单 → tile stem 列表 | Read evaluation manifest → list of tile stems.

    清单存图像文件名 (e.g. "P0003_t0001.png"); 返回去扩展名的 stem.
    The manifest stores image file names; returns their stems.
    """
    with open(path, encoding="utf-8") as f:
        names = json.load(f)
    return [Path(n).stem for n in names]


def save_manifest(path: Path, stems: list[str]) -> None:
    """保存评估清单 (排序后的图像文件名列表) | Save evaluation manifest (sorted file names).

    只写一次; 之后所有实验复用该文件, 保证评估集完全一致.
    Written once; all later experiments reuse it → identical evaluation set.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"{s}.png" for s in sorted(stems)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════
# 1. 模型 + Decoder 构建 | Model + Decoder construction
# ═══════════════════════════════════════════════════════════════════

def _fastsam_weights_path() -> Path:
    """FastSAM-x 预训练权重路径 | FastSAM-x pretrained weights path."""
    return _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"


def load_clean_fastsam(device: str):
    """加载一个权重未改动的 FastSAM (用于严格 zero-shot 基线).
    Load a FastSAM with UNMODIFIED weights (for the strict zero-shot baseline).

    关键 | Key point:
        微调后的 backbone 会被写回共享的 FastSAM 网络, 破坏其自带分割头,
        因此 zero-shot 必须用另一份原始权重的 FastSAM, 才是真正的"未适配"基线.
        The fine-tuned backbone is written back into the shared FastSAM network and
        breaks its native seg head; zero-shot therefore needs a separate ORIGINAL-weight
        FastSAM to be a genuine "un-adapted" baseline.
    """
    from ultralytics import FastSAM
    m = FastSAM(str(_fastsam_weights_path()))
    m.model.to(device).eval()
    for p in m.model.parameters():
        p.requires_grad = False
    return m


def build_model_and_decoder(args, device: str, logger):
    """加载 FastSAM backbone + 恢复 checkpoint 中的 backbone/decoder 权重.
    Load FastSAM backbone + restore backbone/decoder weights from checkpoint.

    与 eval_fewshot_allclass.py:200-263 等价 (无共享 helper, 故此处内联).
    Equivalent to eval_fewshot_allclass.py:200-263 (no shared helper exists).

    :return: (model, decoder, p4_channels)
    """
    from ultralytics import FastSAM

    fastsam_path = _fastsam_weights_path()
    model = FastSAM(str(fastsam_path))
    model.model.to(device).eval()
    for p in model.model.parameters():
        p.requires_grad = False

    ckpt = torch.load(args.checkpoint, map_location=device)

    # ── 恢复解冻的 backbone 层 | Restore unfrozen backbone layers ──
    unfreeze_layers = ckpt.get("unfreeze_layers", 0)
    if unfreeze_layers > 0 and "backbone" in ckpt:
        seq = model.model.model  # Sequential[23]
        for i_str, state in ckpt["backbone"].items():
            seq[int(i_str)].load_state_dict(state)
        logger.log_info("backbone", f"Restored {unfreeze_layers} backbone layers from checkpoint")
    elif unfreeze_layers > 0:
        logger.log_warn("backbone", f"unfreeze_layers={unfreeze_layers} but no backbone weights in ckpt")

    # ── 自动探测 P4/P3 通道 | Auto-detect P4/P3 channels ──
    test_img = np.zeros((896, 896, 3), dtype=np.uint8)
    test_feats = extract_features(model, [test_img], device)
    p4_channels = test_feats[0]["p4"].shape[1]

    # ── 按类型构建 decoder | Build decoder by type ──
    if args.decoder == "adaptive":
        # 复原训练时的 proto 归一化 (模型侧, 非评估协议; 旧 ckpt 无此键 → none) | restore proto-norm (model-side)
        normalize_proto = ckpt.get("normalize_proto", "none")
        decoder = AdaptiveSparseDecoder(in_channels=p4_channels, use_fdr=False,
                                        normalize_proto=normalize_proto).to(device)
    elif args.decoder == "adaptive-p3p4":
        p3_channels = test_feats[0]["p3"].shape[1]
        decoder = AdaptiveDecoderP3P4(p3_channels=p3_channels, p4_channels=p4_channels).to(device)
    elif args.decoder == "pure":
        decoder = PureDecoder(in_channels=p4_channels).to(device)
    elif args.decoder == "pure-p3p4":
        p3_channels = test_feats[0]["p3"].shape[1]
        decoder = PureDecoderP3P4(p3_channels=p3_channels, p4_channels=p4_channels).to(device)
    elif args.decoder == "dynamic_kernel":
        p3_channels = test_feats[0]["p3"].shape[1]
        normalize_proto = ckpt.get("normalize_proto", "none")
        n_kernels = ckpt.get("n_kernels", args.n_kernels)
        decoder = DynamicKernelDecoder(
            p3_channels=p3_channels, p4_channels=p4_channels,
            proto_dim=32, n_kernels=n_kernels, kernel_dim=256, fpn_dim=256,
            normalize_proto=normalize_proto,
        ).to(device)
        logger.log_info("decoder", f"  n_kernels={n_kernels} normalize_proto={normalize_proto}")
    elif args.decoder == "center_affinity":
        p3_channels = test_feats[0]["p3"].shape[1]
        normalize_proto = ckpt.get("normalize_proto", "none")
        decoder = CenterAffinityDecoder(
            p3_channels=p3_channels, p4_channels=p4_channels,
            proto_dim=32, fpn_dim=64,
            normalize_proto=normalize_proto,
        ).to(device)
        logger.log_info("decoder", f"  normalize_proto={normalize_proto} fpn_dim=64")
    else:  # baseline — FewShotDecoder 定义在训练脚本中 | defined in the training script
        from tools.train.train_fewshot_allclass import FewShotDecoder
        decoder = FewShotDecoder(feat_dim=p4_channels).to(device)

    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    logger.log_info("decoder", f"Loaded {args.decoder} decoder (epoch {ckpt.get('epoch', '?')})")
    return model, decoder, p4_channels


# ═══════════════════════════════════════════════════════════════════
# 2. Episode 采样 → 每类 prototype + query tile 集合
#    Episode sampling → per-class prototype + query tile set
# ═══════════════════════════════════════════════════════════════════

def _load_tile_img_mask(stem: str, split: str, data_root: Path,
                        is_instance: bool, target_class_id=None):
    """按数据格式加载 tile 图像 + 目标类 binary mask | Load tile image + target-class binary mask."""
    if is_instance:
        img, mask = load_instance_tile_and_mask(stem, split, data_root, target_class_id=target_class_id)
    else:
        img, mask = load_tile_and_mask(stem, split, data_root, target_class_id=target_class_id)
    return img, semantic_mask_to_binary(mask, is_tile=True)


def build_class_prototypes(model, decoder, args, class_index, split, data_root,
                           is_instance, device, logger, manifest_stems=None):
    """为每类构建 K-shot prototype; query 集合来自 manifest (若给定) 或确定性采样.
    Build a K-shot prototype per class; the query set comes from the manifest (if given)
    or from deterministic sampling. Support/query 源图始终 0% 场景重叠.

    :param manifest_stems: 固定评估集 (tile stem 集合); None → 采样, 由调用方保存清单.
        Fixed evaluation set (tile stems); None → sample (caller then saves the manifest).
    :return: (class_protos, query_stems)
        - class_protos: {cls_id: {"proto": Tensor[1,C], "support_tmpl": Tensor|None}}
        - query_stems: set[str] — 参与评估的全部 query tile stem | all evaluated query tiles.
    """
    rng = random.Random(args.seed)                      # 采样 support | supports
    rng_query = random.Random(args.seed + 99999)        # 独立采样 query (跨 K 一致) | queries
    class_protos = {}
    # 有 manifest → 评估集固定; 无 manifest → 边采样边累积 | fixed set vs sampled set
    query_stems: set[str] = set(manifest_stems) if manifest_stems is not None else set()

    for cls_id, src_to_tiles in class_index.items():
        sources = list(src_to_tiles.keys())
        if len(sources) < args.k_shot + 1:
            logger.log_warn("episode", f"Class {cls_id}: only {len(sources)} sources (<K+1), skipped")
            continue

        if manifest_stems is None:
            # ── 采样 query 源图 + tile (确定性哈希, 跨 K 固定) | sample query (deterministic) ──
            n_query = min(args.per_class, len(sources) - args.k_shot)
            query_sources = rng_query.sample(sources, n_query)
            for s in query_sources:
                q_tile = random.Random(args.seed + _det_hash(s)).choice(src_to_tiles[s])
                query_stems.add(q_tile)
        else:
            # ── query 源图从固定 manifest 推导 | derive query sources from the fixed manifest ──
            cls_tiles = {t for tiles in src_to_tiles.values() for t in tiles}
            query_sources = sorted({_extract_source_image(t) for t in (manifest_stems & cls_tiles)})

        # ── support 源图 (排除 query 源图, 0% 场景重叠 — 采样方法不变) | support (exclude query sources) ──
        query_src_set = set(query_sources)
        support_pool = [s for s in sources if s not in query_src_set]
        if len(support_pool) < args.k_shot:
            logger.log_warn("episode",
                            f"Class {cls_id}: no support left after excluding query sources, skipped")
            continue
        support_sources = rng.sample(support_pool, args.k_shot)
        support_stems = []
        for s in support_sources:
            support_stems.extend(src_to_tiles[s])

        # ── 提取 support 特征 → prototype | Extract support feats → prototype ──
        support_imgs, support_masks = [], []
        for stem in support_stems:
            img, m = _load_tile_img_mask(stem, split, data_root, is_instance, target_class_id=cls_id)
            support_imgs.append(img)
            support_masks.append(m)
        support_feats = extract_features(model, support_imgs, device)
        proto = compute_support_prototype(support_feats, source=args.prototype_source)

        # baseline decoder 需要空间模板 | baseline decoder needs a spatial template
        support_tmpl = None
        if args.decoder == "baseline":
            support_tmpl = compute_support_mask_template(support_masks).to(device)

        class_protos[cls_id] = {"proto": proto, "support_tmpl": support_tmpl}
        logger.log_info(
            "episode",
            f"Class {cls_id:>2d} ({CATEGORY_NAMES.get(cls_id, '?'):<18s}): "
            f"K={args.k_shot} support from {support_sources}",
        )

    return class_protos, query_stems


# ═══════════════════════════════════════════════════════════════════
# 3. Decoder 前向 → 前景概率图 | Decoder forward → foreground prob map
# ═══════════════════════════════════════════════════════════════════

def decoder_prob_map(decoder, feats, support_proto, support_tmpl,
                     decoder_type: str, tile_h: int, tile_w: int) -> np.ndarray:
    """运行 decoder 并返回 tile 尺寸的前景概率图 ∈ [0,1] (numpy [H,W]).
    Run the decoder and return a tile-sized foreground probability map in [0,1].

    不同 decoder 输出激活方式不同: baseline 输出 logits(需 sigmoid),
    其余输出已过 sigmoid 的概率图.
    Activation differs: baseline outputs logits (needs sigmoid); others output
    already-sigmoid probabilities.
    """
    with torch.no_grad():
        if decoder_type == "adaptive":
            out = decoder(feats["p4"], feats["proto"], support_proto)
            out = _normalize_mask_to_4d(out)
            prob = F.interpolate(out, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
        elif decoder_type == "adaptive-p3p4":
            out = decoder(feats["p3"], feats["p4"], feats["proto"], support_proto)
            out = _normalize_mask_to_4d(out)
            prob = F.interpolate(out, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
        elif decoder_type == "pure":
            out = decoder(feats["p4"])
            out = _normalize_mask_to_4d(out)
            prob = F.interpolate(out, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
        elif decoder_type == "pure-p3p4":
            out = decoder(feats["p3"], feats["p4"])
            out = _normalize_mask_to_4d(out)
            prob = F.interpolate(out, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
        else:  # baseline — 输出 logits | outputs logits
            out = decoder(feats["p4"], feats["proto"], support_proto, support_tmpl)
            prob = F.interpolate(out, size=(tile_h, tile_w), mode="bilinear", align_corners=False)
            prob = torch.sigmoid(prob)
    return prob.squeeze().float().cpu().numpy()


def prob_map_to_instances(prob_map: np.ndarray, category_id: int,
                          score_thr: float, min_area: int,
                          method: InstanceMethod = "connected_components",
                          min_distance: int = 12) -> list[dict]:
    """前景概率图 → 实例列表 (支持多种实例化方法).
    Foreground prob map → instance list (multiple generation methods).

    :param method: 实例化方法 | Instance generation method.
        connected_components — baseline
        watershed_distance    — distance transform + watershed (best for splitting touching objects)
        watershed_gradient    — probability gradient + watershed
    :param min_distance: 最近峰值间距 (watershed 系列参数) | Min distance between peaks.
    """
    instances_raw = generate_instances(
        prob_map, method=method,
        score_thr=score_thr, min_area=min_area,
        min_distance=min_distance,
    )
    # 添加 category_id | Attach category_id
    for inst in instances_raw:
        inst["category_id"] = category_id
    return instances_raw


# ═══════════════════════════════════════════════════════════════════
# 4. Zero-shot 非 oracle 实例 | Zero-shot non-oracle instances
# ═══════════════════════════════════════════════════════════════════

def zero_shot_tile_instances(model, img: np.ndarray, device: str) -> list[dict]:
    """FastSAM 默认输出的类无关实例 (严格非 oracle, 不看 GT).
    Class-agnostic instances from FastSAM default output (strictly non-oracle, never sees GT).

    只用模型默认结果: 不给 GT bbox 提示, 不按 GT IoU 选 mask.
    Model default output only: NO GT bbox prompt, NO GT-IoU mask selection.

    :return: list of {category_id=1, mask(bool [H,W]), score}
    """
    H, W = img.shape[:2]
    try:
        results = model(img, retina_masks=True, imgsz=max(H, W), verbose=False, device=device)
    except Exception:
        return []
    if not results or results[0].masks is None:
        return []

    masks_data = results[0].masks.data           # [N, mh, mw]
    confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else None

    instances = []
    for i in range(len(masks_data)):
        m = masks_data[i].cpu().numpy()
        if m.shape != (H, W):                     # 对齐到 tile 尺寸 | align to tile size
            m = cv2_resize_bool(m, H, W)
        m_bin = m > 0.5
        if m_bin.sum() == 0:
            continue
        score = float(confs[i]) if confs is not None and i < len(confs) else 1.0
        instances.append({"category_id": 1, "mask": m_bin, "score": score})
    return instances


def cv2_resize_bool(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """把 mask 双线性 resize 到 (h,w) 再二值化 | Bilinear-resize a mask to (h,w) then binarize."""
    import cv2
    r = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return r > 0.5


# ═══════════════════════════════════════════════════════════════════
# 5. GT 逐实例加载 | Per-instance GT loading (pycocotools)
# ═══════════════════════════════════════════════════════════════════

def load_gt_instances(coco, image_id: int) -> list[dict]:
    """从 COCO GT 读取该图的逐实例掩码 | Load per-instance GT masks for an image from COCO GT.

    :return: list of {category_id, mask(bool [H,W]), area}
    """
    ann_ids = coco.getAnnIds(imgIds=image_id, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    out = []
    for ann in anns:
        cat = ann.get("category_id", 0)
        if cat < 1 or cat > 15:
            continue
        m = coco.annToMask(ann).astype(bool)      # polygon → binary mask
        if m.sum() == 0:
            continue
        out.append({"category_id": cat, "mask": m, "area": float(ann.get("area", m.sum()))})
    return out


# ═══════════════════════════════════════════════════════════════════
# 6. 主流程 | Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Instance Segmentation Evaluator V3 (COCO AP + Instance mIoU, non-oracle)")
    p.add_argument("--checkpoint", type=str, required=True, help="训练 checkpoint | Training checkpoint")
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--per-class", type=int, default=20,
                   help="每类 query tile 数上限 | Max query tiles per class")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--data-format", type=str, default="isaid_instance",
                   choices=["isaid_instance", "isaid_tiles"],
                   help="仅支持 tile 格式 (COCO GT) | Tile formats with COCO GT only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--decoder", type=str, default="adaptive",
                   choices=["baseline", "adaptive", "adaptive-p3p4", "pure", "pure-p3p4",
                            "dynamic_kernel", "center_affinity"])
    p.add_argument("--n-kernels", type=int, default=16,
                   help="DynamicKernelDecoder: 每类动态核数 (需与训练一致) | "
                        "Kernels per class (must match training checkpoint)")
    p.add_argument("--prototype-source", type=str, default="p4", choices=["p4", "p8"])
    p.add_argument("--iou-thr", type=float, default=0.5,
                   help="Instance mIoU / TP-FP-FN 的匹配阈值 | Match threshold for mIoU/TP-FP-FN")
    p.add_argument("--score-thr", type=float, default=0.5,
                   help="前景二值化阈值 | Foreground binarization threshold")
    p.add_argument("--min-area", type=int, default=16,
                   help="连通域最小面积 | Min connected-component area")
    p.add_argument("--instance-method", type=str, default="connected_components",
                   choices=["connected_components", "watershed_distance", "watershed_gradient"],
                   help="实例生成方法 | Instance generation method. "
                        "watershed_distance: distance transform + watershed (best for touching objects)")
    p.add_argument("--min-distance", type=int, default=12,
                   help="Watershed 最近峰值间距 (像素) | Min distance between watershed peaks")
    p.add_argument("--no-zero-shot", action="store_true", help="跳过 zero-shot 基线 | Skip zero-shot baseline")
    p.add_argument("--manifest", type=str, default=None,
                   help="固定评估集清单路径 | Fixed evaluation-set manifest path. "
                        "存在则读取, 不存在则首次生成并保存 (之后所有实验复用). "
                        "Read if present; else generated & saved on first run, reused thereafter. "
                        "默认 <data_root>/evaluation_manifest_<split>.json.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)  # 复现性优先 | Reproducibility first (rule 5)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    data_root, data_format, _train_split, eval_split = _resolve_paths(args)
    data_root = Path(data_root)
    is_instance = (data_format == "isaid_instance")

    # ── 输出目录 | Output dir ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ckpt_name = Path(args.checkpoint).parent.name
        out_dir = _PROJECT_ROOT / "runs" / f"evalinst_{ckpt_name}_{datetime.now():%m%d_%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("evalinst")
    logger.add_backend(ConsoleBackend())
    logger.add_backend(FileBackend(str(out_dir / "eval.jsonl")))
    logger.log_info("config",
                    f"Instance Eval V3 | decoder={args.decoder} K={args.k_shot} "
                    f"proto={args.prototype_source} seed={args.seed} split={eval_split} "
                    f"iou_thr={args.iou_thr} score_thr={args.score_thr} → {out_dir}")

    # ── 模型 + decoder | Model + decoder ──
    model, decoder, _p4c = build_model_and_decoder(args, device, logger)

    # ── 类别索引 + prototype + query 集合 | Class index + prototypes + query set ──
    if is_instance:
        class_index = _build_class_index_instance(data_root, eval_split)
    else:
        class_index = _build_class_index_tiles(data_root, eval_split)

    # ── 评估清单: 固定评估集, 保证跨实验/跨运行完全一致 | Evaluation manifest (fixed eval set) ──
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = data_root / f"evaluation_manifest_{eval_split}.json"
    manifest_existed = manifest_path.exists()
    manifest_stems = set(load_manifest(manifest_path)) if manifest_existed else None
    if manifest_existed:
        logger.log_info("manifest", f"Loaded fixed eval set: {len(manifest_stems)} images <- {manifest_path}")

    class_protos, query_stems = build_class_prototypes(
        model, decoder, args, class_index, eval_split, data_root, is_instance, device, logger,
        manifest_stems=manifest_stems)

    # 首次运行 → 生成并保存清单; 之后所有实验复用 (禁止再次随机采样)
    # First run → generate & save; all later runs reuse it (no more random sampling)
    if not manifest_existed:
        save_manifest(manifest_path, sorted(query_stems))
        logger.log_info("manifest",
                        f"Generated & saved fixed eval set: {len(query_stems)} images -> {manifest_path}")

    is_conditioned = args.decoder in _CLASS_CONDITIONED
    logger.log_info("index", f"{len(class_protos)} classes with prototypes, "
                             f"{len(query_stems)} unique query tiles, conditioned={is_conditioned}")

    # ── COCO GT 评估器 | COCO GT evaluators ──
    gt_path = str(data_root / "annotations" / f"instances_{eval_split}.json")
    ft_eval = COCOInstanceEvaluator(gt_path, iouType="segm")
    coco = ft_eval.coco_gt
    stem_to_id = {Path(v["file_name"]).stem: k for k, v in coco.imgs.items()}
    zs_eval = None if args.no_zero_shot else COCOInstanceEvaluator(gt_path, iouType="segm")
    # zero-shot 用原始权重 FastSAM (与微调 backbone 分离) | zero-shot uses clean-weight FastSAM
    zs_model = None if args.no_zero_shot else load_clean_fastsam(device)

    # ── 逐 tile 预测 | Per-tile prediction ──
    # 类级累积 (Instance mIoU + TP/FP/FN) | Per-class accumulators
    per_class_gt_ious = defaultdict(list)   # cls → [per-GT max IoU, ...]
    per_class_counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_gt": 0, "n_pred": 0})
    evaluated_image_ids = []

    query_list = sorted(query_stems)

    # ── 评估协议摘要 (启动即打印, 确定性且固定) | Evaluation Protocol Summary ──
    eval_ids = [stem_to_id[s] for s in query_list if s in stem_to_id]
    n_gt_images = sum(1 for iid in eval_ids if len(coco.getAnnIds(imgIds=iid)) > 0)
    sampling_state = ("Disabled (loaded from manifest)" if manifest_existed
                      else "Disabled after first-gen (manifest saved & frozen)")
    logger.log_info(
        "protocol",
        "\n  ================ Evaluation Protocol V3 ================"
        f"\n    Query Images:    {len(query_list)}"
        f"\n    GT Images:       {n_gt_images}"
        f"\n    Manifest:        {manifest_path}"
        f"\n    Random Sampling: {sampling_state}"
        f"\n    Evaluation:      Deterministic"
        "\n  =======================================================")

    logger.log_info("eval", f"Evaluating {len(query_list)} query tiles ...")
    for qi, stem in enumerate(query_list):
        image_id = stem_to_id.get(stem)
        if image_id is None:
            continue  # 该 tile 不在 COCO GT 中 | tile not in COCO GT
        evaluated_image_ids.append(image_id)

        img, _ = _load_tile_img_mask(stem, eval_split, data_root, is_instance, target_class_id=None)
        H, W = img.shape[:2]
        feats = extract_features(model, [img], device)[0]

        # 逐类前景图 → 实例 | per-class FG map → instances
        pred_by_class = defaultdict(list)   # cls → [inst dict]
        if args.decoder == "dynamic_kernel":
            # DynamicKernelDecoder: Prototype → N Kernels → N Instance Masks
            # Each kernel output is a candidate instance (no CC decomposition needed)
            for cls_id, pk in class_protos.items():
                with torch.no_grad():
                    masks_s8, proto_mask = decoder(
                        feats["p3"], feats["p4"], feats["proto"], pk["proto"],
                    )
                # masks_s8: [N_kernels, H/8, W/8] sigmoid ∈ [0,1]
                # Upsample each kernel mask to tile resolution
                for k_idx in range(masks_s8.shape[0]):
                    kernel_mask = masks_s8[k_idx]  # [H/8, W/8]
                    mask_up = F.interpolate(
                        kernel_mask.unsqueeze(0).unsqueeze(0),
                        size=(H, W), mode="bilinear", align_corners=False,
                    ).squeeze()  # [H, W]
                    mask_np = mask_up.float().cpu().numpy()

                    # Score = mean prob within mask region (only for score_thr filtering)
                    binary = mask_np > args.score_thr
                    if binary.sum() < args.min_area:
                        continue
                    score = float(mask_np[binary].mean())

                    pred_by_class[cls_id].append({
                        "mask": binary,
                        "score": score,
                        "category_id": cls_id,
                    })
        elif args.decoder == "center_affinity":
            # CenterAffinityDecoder: Center Heatmap + Offset Field → Voronoi Grouping
            # Generate center+offset ONCE (class-agnostic), proto PER CLASS
            with torch.no_grad():
                # First forward (any prototype) to get class-agnostic center+offset
                # Use first class prototype for the initial forward
                first_pk = list(class_protos.values())[0]
                center_hm, offset_field, _ = decoder(
                    feats["p3"], feats["p4"], feats["proto"],
                    torch.from_numpy(first_pk["proto"]).float().to(device),
                )
                # center_hm:    [H/8, W/8] ∈ [0,1]
                # offset_field: [2, H/8, W/8]

            # Upsample center+offset to tile resolution
            center_full = F.interpolate(
                center_hm.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze().float().cpu().numpy()
            offset_full = F.interpolate(
                offset_field.unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze().float().cpu().numpy()

            for cls_id, pk in class_protos.items():
                # Class-conditioned proto_mask
                with torch.no_grad():
                    proto_mask = decoder.forward_proto_only(
                        feats["proto"],
                        torch.from_numpy(pk["proto"]).float().to(device),
                    )
                # proto_mask: [H/4, W/4] → upsample to tile resolution
                proto_full = F.interpolate(
                    proto_mask.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="bilinear", align_corners=False,
                ).squeeze().float().cpu().numpy()

                fg_mask = proto_full > args.score_thr

                # Center-affinity grouping
                insts = generate_instances_center_affinity(
                    center_full, offset_full, fg_mask,
                    score_thr=0.2,  # center peak threshold (lower than FG threshold)
                    min_area=args.min_area,
                    min_distance=8,  # at tile resolution (~64px at stride-8)
                    max_instances=100,
                )
                for it in insts:
                    it["category_id"] = cls_id
                pred_by_class[cls_id].extend(insts)
        elif is_conditioned:
            for cls_id, pk in class_protos.items():
                prob = decoder_prob_map(decoder, feats, pk["proto"], pk["support_tmpl"],
                                        args.decoder, H, W)
                insts = prob_map_to_instances(prob, cls_id, args.score_thr, args.min_area,
                                               method=args.instance_method, min_distance=args.min_distance)
                pred_by_class[cls_id].extend(insts)
        else:
            # pure decoder: 无类条件 → 单张前景图, 类无关 | class-agnostic single FG map
            prob = decoder_prob_map(decoder, feats, None, None, args.decoder, H, W)
            insts = prob_map_to_instances(prob, 1, args.score_thr, args.min_area,
                                          method=args.instance_method, min_distance=args.min_distance)
            pred_by_class[1].extend(insts)

        # 送入 COCO AP 评估器 | Feed COCO AP evaluator
        for cls_id, insts in pred_by_class.items():
            for it in insts:
                ft_eval.add_prediction(image_id, it["category_id"], it["mask"], it["score"])

        # GT 逐实例 | Per-instance GT
        gt_insts = load_gt_instances(coco, image_id)
        gt_by_class = defaultdict(list)
        for g in gt_insts:
            gt_by_class[g["category_id"]].append(g["mask"])

        # Instance mIoU + TP/FP/FN (类感知) | class-aware
        cls_universe = set(gt_by_class) | set(pred_by_class)
        for cls_id in cls_universe:
            gm = gt_by_class.get(cls_id, [])
            pm = [it["mask"] for it in pred_by_class.get(cls_id, [])]
            ps = [it["score"] for it in pred_by_class.get(cls_id, [])]
            # Instance mIoU: 只在有 GT 时计入 | only when GT exists
            if gm:
                per_gt, _ = instance_miou(pm, gm)
                per_class_gt_ious[cls_id].extend(per_gt)
            # 贪心一对一匹配计数 | greedy one-to-one counts
            mres = greedy_match(pm, ps, gm, iou_thr=args.iou_thr)
            c = per_class_counts[cls_id]
            c["tp"] += mres["tp"]; c["fp"] += mres["fp"]; c["fn"] += mres["fn"]
            c["n_gt"] += mres["n_gt"]; c["n_pred"] += mres["n_pred"]

        # Zero-shot (非 oracle, 原始权重 FastSAM) | Zero-shot (non-oracle, clean-weight FastSAM)
        if zs_eval is not None:
            for it in zero_shot_tile_instances(zs_model, img, device):
                zs_eval.add_prediction(image_id, it["category_id"], it["mask"], it["score"])

        if (qi + 1) % 20 == 0:
            logger.log_info("progress", f"{qi + 1}/{len(query_list)} tiles done")

    # ── COCO AP 计算 | COCO AP computation ──
    logger.log_info("coco", "Running COCO AP (fine-tuned) ...")
    ft_ap = ft_eval.evaluate(verbose=True, image_ids=evaluated_image_ids)
    ft_ap_agnostic = ft_eval.evaluate_class_agnostic(verbose=False, image_ids=evaluated_image_ids)
    ft_per_cat_ap = ft_eval.get_per_category_ap(image_ids=evaluated_image_ids) if is_conditioned else {}

    zs_ap_agnostic = None
    if zs_eval is not None:
        logger.log_info("coco", "Running COCO AP (zero-shot, class-agnostic) ...")
        zs_ap_agnostic = zs_eval.evaluate_class_agnostic(verbose=True, image_ids=evaluated_image_ids)

    # ── Instance mIoU 汇总 | Instance mIoU aggregation ──
    all_ious = [x for v in per_class_gt_ious.values() for x in v]
    overall_inst_miou = float(np.mean(all_ious)) if all_ious else 0.0
    per_class_inst_miou = {c: (float(np.mean(v)) if v else 0.0) for c, v in per_class_gt_ious.items()}
    class_mean_inst_miou = (float(np.mean(list(per_class_inst_miou.values())))
                            if per_class_inst_miou else 0.0)

    # ── 组装 per-class 输出 | Assemble per-class output ──
    per_class_out = {}
    for cls_id in sorted(set(per_class_counts) | set(per_class_inst_miou)):
        c = per_class_counts[cls_id]
        per_class_out[str(cls_id)] = {
            "name": CATEGORY_NAMES.get(cls_id, f"cls{cls_id}"),
            "n_gt": c["n_gt"], "n_pred": c["n_pred"],
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            "AP50": round(float(ft_per_cat_ap.get(cls_id, 0.0)), 4),
            "instance_miou": round(per_class_inst_miou.get(cls_id, 0.0), 4),
        }

    result = {
        "protocol": "instance_v3",
        "checkpoint": args.checkpoint,
        "decoder": args.decoder,
        "prototype_source": args.prototype_source,
        "k_shot": args.k_shot, "seed": args.seed,
        "eval_split": eval_split,
        "manifest": str(manifest_path),
        "n_query_tiles": len(evaluated_image_ids),
        "iou_thr": args.iou_thr, "score_thr": args.score_thr, "min_area": args.min_area,
        "class_conditioned": is_conditioned,
        "finetuned": {
            **{k: round(v, 4) for k, v in ft_ap.items() if k != "n_predictions"},
            "n_predictions": ft_ap["n_predictions"],
            "AP_class_agnostic": round(ft_ap_agnostic["AP"], 4),
            "AP50_class_agnostic": round(ft_ap_agnostic["AP50"], 4),
            "instance_miou_overall": round(overall_inst_miou, 4),
            "instance_miou_class_mean": round(class_mean_inst_miou, 4),
            "per_class": per_class_out,
        },
    }
    if zs_ap_agnostic is not None:
        result["zero_shot"] = {
            "AP_class_agnostic": round(zs_ap_agnostic["AP"], 4),
            "AP50_class_agnostic": round(zs_ap_agnostic["AP50"], 4),
            "AP75_class_agnostic": round(zs_ap_agnostic["AP75"], 4),
            "AR_max100": round(zs_ap_agnostic["AR_max100"], 4),
            "n_predictions": zs_ap_agnostic["n_predictions"],
            "note": "FastSAM default output, non-oracle, class-agnostic only",
        }

    # ── 写盘 + 日志标量 | Write to disk + log scalars ──
    with open(out_dir / "instance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    for k in ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large"]:
        logger.log_metric(f"ft_{k}", float(ft_ap[k]), tags=["instance", "finetuned"])
    logger.log_metric("ft_instance_miou", overall_inst_miou, tags=["instance", "finetuned"])
    if zs_ap_agnostic is not None:
        logger.log_metric("zs_AP_agnostic", float(zs_ap_agnostic["AP"]), tags=["instance", "zero-shot"])

    logger.log_info(
        "done",
        f"FT AP={ft_ap['AP']:.4f} AP50={ft_ap['AP50']:.4f} "
        f"AP_agnostic={ft_ap_agnostic['AP']:.4f} InstMIoU={overall_inst_miou:.4f}"
        + (f" | ZS AP_agnostic={zs_ap_agnostic['AP']:.4f}" if zs_ap_agnostic else "")
        + f" → {out_dir / 'instance_metrics.json'}",
    )


if __name__ == "__main__":
    main()
