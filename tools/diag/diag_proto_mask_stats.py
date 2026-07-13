#!/usr/bin/env python3
"""
Proto Mask 统计诊断 (Probe 1) | Proto Mask Statistics Diagnosis.
=================================================================

三层分析框架 — Layer 1→2 桥梁: 量化 L2 归一化对 proto_mask 统计属性的改变。
M-F-T Framework — Bridge from Mechanism to Function: Quantify how L2 normalization
changes the statistical properties of proto_mask.

核心问题 | Core Question:
    L2 恢复了 Mechanism (梯度), 但 Function (类别条件化) 仍死, Task 却涨了 +31%。
    proto_mask 到底发生了什么变化?

指标 | Metrics:
    - proto_mask_mean / std:           整体激活水平与空间变异性
    - proto_mask_entropy:              信息量 (二值化后)
    - proto_mask_sparsity:             激活区域占比 (>0.1 阈值)
    - proto_mask_fg_iou:               与 GT 前景的重叠 (IoU)
    - proto_mask_activation_histogram: 激活值分布 (50 bins, 0→1)
    - proto_basis_l2:                  proto basis 每通道 L2 范数 (检测幅值爆炸)
    - coeff_l2 / coeff_std:            系数幅值 (检测饱和转移到 coeff)

判断逻辑 | Decision Logic:
    - none: proto_mask ≈ 常数图 (std≈0, histogram 尖峰) → l2: 恢复空间结构 (std↑, histogram 分散)
      → 支持 H1 (净化融合)
    - none: proto_mask 已有空间结构 (IoU 与 GT 相关) → H1 部分错误
    - l2: proto_mask 与 none 分布几乎一样 → H1/H2 均存疑

用法 | Usage::

    python tools/diag/diag_proto_mask_stats.py \\
        --checkpoint-a runs/.../uf8_none/best_model.pt \\
        --checkpoint-b runs/.../uf8_l2/best_model.pt \\
        --label-a none --label-b l2 \\
        --device cuda

输出 | Output:
    runs/diag/proto_mask_stats/
    ├── proto_mask_histogram.png       # none vs l2 激活值分布对比
    ├── proto_mask_per_class.png       # 每类 proto_mask 统计对比
    ├── proto_basis_comparison.png     # proto basis 幅值对比
    └── proto_mask_stats.json          # 全部统计数据
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

import torch
import torch.nn.functional as F

from tools.train.train_fewshot_allclass import (
    _build_class_index_instance,
    _resolve_paths,
    _extract_source_image,
    load_instance_tile_and_mask,
    load_tile_and_mask,
    extract_features,
    compute_support_prototype,
    semantic_mask_to_binary,
    CATEGORY_NAMES,
)
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
from adatile.utils.seed import set_seed

OUT_DIR = _PROJECT_ROOT / "runs" / "diag" / "proto_mask_stats"


# ═══════════════════════════════════════════════════════════════════
# Captured Decoder Forward — 捕获所有中间张量
# Captured Decoder Forward — capture all intermediate tensors
# ═══════════════════════════════════════════════════════════════════

def captured_decoder_forward(
    decoder: AdaptiveSparseDecoder,
    p4_features: torch.Tensor,
    proto_masks: torch.Tensor,
    support_proto: torch.Tensor,
) -> dict:
    """
    运行 decoder 前向并捕获所有中间张量, 用于归因分析。
    Run decoder forward and capture ALL intermediate tensors for attribution.

    精确镜像 AdaptiveSparseDecoder.forward(), 跳过 FDR gating (本项目不使用 FDR)。
    Exactly mirrors AdaptiveSparseDecoder.forward(), skips FDR gating (not used).

    :return: dict with keys:
        proto_masks_norm  — [32, H/4, W/4] normalized proto basis
        coeffs            — [1, 32] predicted mask coefficients
        proto_mask        — [H/4, W/4] coarse proto mask (coeffs @ basis → sigmoid)
        query_refine      — [H/4, W/4] P4 refinement contribution (pre-sigmoid logit)
        final_logit       — [H/4, W/4] proto_mask + query_refine
        final_mask        — [H/4, W/4] sigmoid(final_logit)
    """
    # ── 输入标准化 (镜像 forward) | Input normalization ──
    if proto_masks.dim() == 4:
        proto_masks = proto_masks.squeeze(0)  # [32, H/4, W/4]
    if support_proto.dim() == 2:
        support_proto = support_proto.squeeze(0)  # [feat_dim]

    # ── Proto basis 归一化 | Proto basis normalization ──
    proto_masks_norm = decoder._normalize_proto(proto_masks)  # [32, H/4, W/4]

    # ── Step 1: Proto mask 生成 | Proto mask generation ──
    coeffs = decoder.coeff_predictor(support_proto.unsqueeze(0))  # [1, 32]
    proto_mask = decoder.coeff_predictor.generate_mask(
        coeffs, proto_masks_norm
    )  # [1, H/4, W/4] — 已过 sigmoid | already sigmoid

    # ── Step 2: P4 特征精炼 | P4 feature refinement ──
    feat_proj = decoder.feat_proj(p4_features)    # [1, 256, H/16, W/16]
    feat_refined = decoder.feat_refine(feat_proj)  # [1, 64, H/16, W/16]
    # NOTE: 跳过 FDR gating — 本项目 use_fdr=False | Skip FDR — not used

    # ── Step 3: 掩码头 | Mask head ──
    refined_logit = decoder.mask_head(feat_refined)  # [1, 1, H/16, W/16]
    refined_logit_up = F.interpolate(
        refined_logit, size=proto_mask.shape[1:],
        mode='bilinear', align_corners=False,
    )  # [1, 1, H/4, W/4]

    # ── Step 4: 加法融合 | Additive fusion ──
    query_refine = refined_logit_up.squeeze(1)  # [H/4, W/4]
    pm = proto_mask.squeeze(0)                   # [H/4, W/4]
    final_logit = query_refine + pm              # [H/4, W/4]
    final_mask = torch.sigmoid(final_logit)

    return {
        "proto_masks_norm": proto_masks_norm,
        "coeffs": coeffs,
        "proto_mask": pm,
        "query_refine": query_refine,
        "final_logit": final_logit,
        "final_mask": final_mask,
    }


# ═══════════════════════════════════════════════════════════════════
# Proto Mask 统计计算 | Proto Mask Statistics
# ═══════════════════════════════════════════════════════════════════

def compute_proto_mask_stats(
    proto_mask: torch.Tensor,       # [H/4, W/4] ∈ [0,1]
    gt_fg: np.ndarray | None,       # [H, W] binary GT foreground (None = no GT)
    proto_basis: torch.Tensor,      # [32, H/4, W/4] normalized basis
    coeffs: torch.Tensor,           # [1, 32]
) -> dict:
    """
    计算单张 proto_mask 的统计指标 | Compute stats for a single proto_mask.

    :return: dict of scalar stats (all Python floats).
    """
    pm = proto_mask.detach().float()

    # ── 基础统计 | Basic stats ──
    _mean = float(pm.mean().item())
    _std = float(pm.std().item())

    # ── 熵 (二值化后) | Entropy (binarized) ──
    pm_bin = (pm > 0.5).float()
    p_fg = float(pm_bin.mean().item() + 1e-8)
    p_bg = 1.0 - p_fg
    _entropy = float(-p_fg * np.log(max(p_fg, 1e-8)) - p_bg * np.log(max(p_bg, 1e-8)))

    # ── 稀疏度 (激活区域占比, 阈值 0.1) | Sparsity (active region fraction) ──
    _sparsity = float((pm > 0.1).float().mean().item())

    # ── 激活值直方图 | Activation histogram ──
    _hist = pm.flatten().cpu().numpy()  # 返回原始值, 调用方做 hist

    # ── 与 GT 前景的 IoU | IoU with GT foreground ──
    _fg_iou = None
    if gt_fg is not None:
        # 上采样 proto_mask 到 tile 分辨率 | Upsample proto_mask to tile resolution
        gt_h, gt_w = gt_fg.shape
        pm_up = F.interpolate(
            pm.unsqueeze(0).unsqueeze(0),
            size=(gt_h, gt_w), mode='bilinear', align_corners=False
        ).squeeze()
        pm_up_bin = (pm_up > 0.5).cpu().numpy()
        gt_t = gt_fg.astype(bool)
        inter = (pm_up_bin & gt_t).sum()
        union = (pm_up_bin | gt_t).sum()
        _fg_iou = float(inter / union) if union > 0 else 0.0

    # ── Proto basis 幅值 | Proto basis magnitude ──
    pf = proto_basis.reshape(proto_basis.shape[0], -1)  # [32, HW]
    _basis_l2 = float(pf.norm(dim=1).mean().item())       # 每 basis 平均 L2

    # ── Coeff 统计 | Coefficient stats ──
    c = coeffs.detach().squeeze(0)  # [32]
    _coeff_l2 = float(c.norm().item())
    _coeff_std = float(c.std().item())
    _coeff_mean = float(c.mean().item())

    return {
        "mean": _mean,
        "std": _std,
        "entropy": _entropy,
        "sparsity": _sparsity,
        "fg_iou": _fg_iou,
        "histogram_raw": _hist,
        "basis_l2": _basis_l2,
        "coeff_l2": _coeff_l2,
        "coeff_std": _coeff_std,
        "coeff_mean": _coeff_mean,
    }


# ═══════════════════════════════════════════════════════════════════
# 模型 + Decoder 构建 | Model + Decoder construction
# ═══════════════════════════════════════════════════════════════════

def _fastsam_weights_path() -> Path:
    return _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"


def build_model_and_decoder(args, device: str, checkpoint: str, label: str):
    """加载 FastSAM + decoder, 恢复 checkpoint 权重 | Load model + decoder, restore ckpt weights."""
    from ultralytics import FastSAM

    model = FastSAM(str(_fastsam_weights_path()))
    model.model.to(device).eval()
    for p in model.model.parameters():
        p.requires_grad = False

    ckpt = torch.load(checkpoint, map_location=device)

    # ── 恢复 backbone | Restore backbone ──
    unfreeze_layers = ckpt.get("unfreeze_layers", 0)
    if unfreeze_layers > 0 and "backbone" in ckpt:
        seq = model.model.model
        for i_str, state in ckpt["backbone"].items():
            seq[int(i_str)].load_state_dict(state)
        print(f"  [{label}] Restored {unfreeze_layers} backbone layers")

    # ── 自动探测 P4 通道数 | Auto-detect P4 channels ──
    test_img = np.zeros((896, 896, 3), dtype=np.uint8)
    test_feats = extract_features(model, [test_img], device)
    p4_channels = test_feats[0]["p4"].shape[1]
    print(f"  [{label}] Auto-detected P4 channels: {p4_channels}")

    # ── 构建 decoder | Build decoder ──
    normalize_proto = ckpt.get("normalize_proto", "none")
    decoder = AdaptiveSparseDecoder(
        in_channels=p4_channels, use_fdr=False, normalize_proto=normalize_proto
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    print(f"  [{label}] Decoder loaded (normalize_proto={normalize_proto}, epoch={ckpt.get('epoch','?')})")

    return model, decoder, normalize_proto


# ═══════════════════════════════════════════════════════════════════
# 确定性哈希 + Manifest | Deterministic hash + Manifest
# ═══════════════════════════════════════════════════════════════════

import hashlib

def _det_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def load_manifest(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        names = json.load(f)
    return [Path(n).stem for n in names]


def save_manifest(path: Path, stems: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"{s}.png" for s in sorted(stems)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2, ensure_ascii=False)


def _load_tile_img_mask(stem, split, data_root, is_instance, target_class_id=None):
    if is_instance:
        img, mask = load_instance_tile_and_mask(
            stem, split, data_root, target_class_id=target_class_id
        )
    else:
        img, mask = load_tile_and_mask(
            stem, split, data_root, target_class_id=target_class_id
        )
    return img, semantic_mask_to_binary(mask, is_tile=True)


# ═══════════════════════════════════════════════════════════════════
# 主流程 | Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Proto Mask Statistics Diagnosis (M-F-T Probe 1)")
    p.add_argument("--checkpoint-a", type=str, required=True,
                   help="Config A checkpoint (e.g. none)")
    p.add_argument("--checkpoint-b", type=str, required=True,
                   help="Config B checkpoint (e.g. l2)")
    p.add_argument("--label-a", type=str, default="none", help="Label for config A")
    p.add_argument("--label-b", type=str, default="l2", help="Label for config B")
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--per-class", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--data-format", type=str, default="isaid_instance")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None,
                   help="固定评估清单路径 | Fixed manifest path.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    data_root, data_format, _train_split, eval_split = _resolve_paths(args)
    data_root = Path(data_root)
    is_instance = (data_format == "isaid_instance")

    # ── 输出目录 | Output dir ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%m%d_%H%M")
        out_dir = OUT_DIR / f"{args.label_a}_vs_{args.label_b}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Proto Mask Stats Diagnosis: {args.label_a} vs {args.label_b}")
    print(f"Output: {out_dir}")

    # ── 加载两个 checkpoint | Load both checkpoints ──
    print("\n[1/5] Loading models...")
    model_a, decoder_a, norm_a = build_model_and_decoder(args, device, args.checkpoint_a, args.label_a)
    model_b, decoder_b, norm_b = build_model_and_decoder(args, device, args.checkpoint_b, args.label_b)

    # ── 类别索引 | Class index ──
    print("\n[2/5] Building class index...")
    if is_instance:
        class_index = _build_class_index_instance(data_root, eval_split)
    else:
        from tools.train.train_fewshot_allclass import _build_class_index_tiles
        class_index = _build_class_index_tiles(data_root, eval_split)
    print(f"  {len(class_index)} classes indexed")

    # ── 评估清单 | Evaluation manifest ──
    print("\n[3/5] Loading manifest & building prototypes...")
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = data_root / f"evaluation_manifest_{eval_split}.json"

    manifest_existed = manifest_path.exists()
    rng = random.Random(args.seed)
    rng_query = random.Random(args.seed + 99999)
    query_stems: set[str] = set()
    class_protos: dict = {}

    if manifest_existed:
        query_stems = set(load_manifest(manifest_path))
        print(f"  Loaded manifest: {len(query_stems)} query tiles")
    else:
        print(f"  Manifest not found, will generate from {args.per_class} tiles/class")

    # ── 构建 prototype (两个 checkpoint 共享同一个 support/query split) | Build prototypes ──
    #   关键: 同一套 tiles, 只是模型不同 → 隔离模型变量 | Same tiles, different models → isolate model var
    for cls_id, src_to_tiles in class_index.items():
        sources = list(src_to_tiles.keys())
        if len(sources) < args.k_shot + 1:
            continue

        if not manifest_existed:
            n_query = min(args.per_class, len(sources) - args.k_shot)
            query_sources = rng_query.sample(sources, n_query)
            for s in query_sources:
                q_tile = random.Random(args.seed + _det_hash(s)).choice(src_to_tiles[s])
                query_stems.add(q_tile)
        else:
            cls_tiles = {t for tiles in src_to_tiles.values() for t in tiles}
            query_sources = sorted({_extract_source_image(t) for t in (set(load_manifest(manifest_path)) & cls_tiles)})

        query_src_set = set(query_sources)
        support_pool = [s for s in sources if s not in query_src_set]
        if len(support_pool) < args.k_shot:
            continue
        support_sources = rng.sample(support_pool, args.k_shot)
        support_stems = [t for s in support_sources for t in src_to_tiles[s]]

        # 提取 support 特征 → prototype (使用 model_a, 另一个 model 分离计算)
        # Extract support features → prototype (use model_a; model_b computed separately)
        support_imgs, support_masks = [], []
        for stem in support_stems:
            img, m = _load_tile_img_mask(stem, eval_split, data_root, is_instance, target_class_id=cls_id)
            support_imgs.append(img)
            support_masks.append(m)
        support_feats = extract_features(model_a, support_imgs, device)
        class_protos[cls_id] = {
            "proto": compute_support_prototype(support_feats, source="p4"),
            "support_stems": support_stems,
        }

    if not manifest_existed:
        save_manifest(manifest_path, sorted(query_stems))
        print(f"  Generated & saved manifest: {len(query_stems)} tiles → {manifest_path}")

    query_list = sorted(query_stems)
    print(f"  {len(class_protos)} classes with prototypes, {len(query_list)} query tiles")

    # ── 主循环: 逐 tile 统计 | Main loop: per-tile statistics ──
    print(f"\n[4/5] Analyzing proto masks ({len(query_list)} tiles)...")

    # 累积统计数据 | Accumulated stats
    all_stats = {args.label_a: [], args.label_b: []}
    per_class_accum = {
        args.label_a: defaultdict(list),
        args.label_b: defaultdict(list),
    }

    models = {args.label_a: (model_a, decoder_a), args.label_b: (model_b, decoder_b)}

    for qi, stem in enumerate(tqdm(query_list, desc="  Tiles")):
        # 加载图像 + 特征提取 | Load image + extract features
        img, _ = _load_tile_img_mask(stem, eval_split, data_root, is_instance, target_class_id=None)
        H, W = img.shape[:2]

        # 两套模型独立提取特征 | Extract features separately for each model
        feats_a = extract_features(model_a, [img], device)[0]
        feats_b = extract_features(model_b, [img], device)[0]

        for label, (model, decoder), feats in [
            (args.label_a, models[args.label_a], feats_a),
            (args.label_b, models[args.label_b], feats_b),
        ]:
            for cls_id, pk in class_protos.items():
                try:
                    captured = captured_decoder_forward(
                        decoder, feats["p4"], feats["proto"], pk["proto"]
                    )
                except Exception as e:
                    continue  # 跳过异常 tile | skip bad tiles

                # 加载 GT 前景 (用于 IoU) | Load GT foreground for IoU
                gt_fg = None
                try:
                    _, gt_mask = _load_tile_img_mask(
                        stem, eval_split, data_root, is_instance, target_class_id=cls_id
                    )
                    gt_fg = gt_mask.astype(np.uint8)
                except Exception:
                    pass

                stats = compute_proto_mask_stats(
                    captured["proto_mask"],
                    gt_fg,
                    captured["proto_masks_norm"],
                    captured["coeffs"],
                )

                # 分离直方图 (不存入 JSON 列表) | Separate histogram from JSON
                hist_raw = stats.pop("histogram_raw")
                stats["histogram_bin_edges"] = None  # placeholder
                stats["histogram_counts"] = hist_raw.tolist() if len(hist_raw) < 1000 else \
                    np.histogram(hist_raw, bins=50, range=(0, 1))[0].tolist()
                stats["cls_id"] = cls_id
                stats["tile"] = stem

                all_stats[label].append(stats)
                per_class_accum[label][cls_id].append(stats)

        if (qi + 1) % 50 == 0:
            pass  # tqdm handles progress

    # ── 汇总统计 | Aggregate statistics ──
    print("\n[5/5] Aggregating & saving...")

    def aggregate(stats_list: list[dict]) -> dict:
        """汇总所有 tile 的统计 | Aggregate stats across tiles."""
        if not stats_list:
            return {}
        keys = ["mean", "std", "entropy", "sparsity", "fg_iou",
                "basis_l2", "coeff_l2", "coeff_std", "coeff_mean"]
        aggr = {}
        for k in keys:
            vals = [s[k] for s in stats_list if s.get(k) is not None]
            if vals:
                aggr[f"{k}_mean"] = float(np.mean(vals))
                aggr[f"{k}_std"] = float(np.std(vals))
                aggr[f"{k}_median"] = float(np.median(vals))
                aggr[f"{k}_q25"] = float(np.percentile(vals, 25))
                aggr[f"{k}_q75"] = float(np.percentile(vals, 75))
        # 汇总直方图 | Aggregate histogram
        all_hists = [s["histogram_counts"] for s in stats_list
                     if s.get("histogram_counts") and isinstance(s["histogram_counts"], list)]
        if all_hists:
            aggr["histogram_aggregated"] = np.stack(all_hists).mean(axis=0).tolist()
        aggr["n_tiles"] = len(stats_list)
        aggr["n_valid"] = sum(1 for s in stats_list if s.get("fg_iou") is not None)
        return aggr

    result = {
        "config": {
            "label_a": args.label_a, "label_b": args.label_b,
            "checkpoint_a": args.checkpoint_a, "checkpoint_b": args.checkpoint_b,
            "normalize_a": norm_a, "normalize_b": norm_b,
            "k_shot": args.k_shot, "seed": args.seed,
            "n_query_tiles": len(query_list), "n_classes": len(class_protos),
        },
        args.label_a: aggregate(all_stats[args.label_a]),
        args.label_b: aggregate(all_stats[args.label_b]),
        "per_class": {
            args.label_a: {
                str(cls): aggregate(vals)
                for cls, vals in per_class_accum[args.label_a].items()
            },
            args.label_b: {
                str(cls): aggregate(vals)
                for cls, vals in per_class_accum[args.label_b].items()
            },
        },
    }

    # ── 保存 JSON | Save JSON ──
    json_path = out_dir / "proto_mask_stats.json"
    # 清理直方图原始数据 (太大) | Clean histogram raw data (too large)
    for label in [args.label_a, args.label_b]:
        for s in all_stats[label]:
            s.pop("histogram_counts", None)
    result["_raw_stats"] = {
        args.label_a: all_stats[args.label_a],
        args.label_b: all_stats[args.label_b],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Stats saved: {json_path}")

    # ── 打印关键对比 | Print key comparison ──
    a_aggr = result[args.label_a]
    b_aggr = result[args.label_b]
    print(f"\n{'='*60}")
    print(f"  Proto Mask Stats: {args.label_a} vs {args.label_b}")
    print(f"{'='*60}")
    for k in ["mean_mean", "std_mean", "entropy_mean", "sparsity_mean",
              "fg_iou_mean", "basis_l2_mean", "coeff_l2_mean"]:
        va = a_aggr.get(k, "N/A")
        vb = b_aggr.get(k, "N/A")
        if isinstance(va, float) and isinstance(vb, float):
            delta = vb - va
            print(f"  {k:25s}: {va:10.6f} → {vb:10.6f}  (Δ={delta:+.6f})")
        else:
            print(f"  {k:25s}: {va} → {vb}")
    print(f"{'='*60}")

    # ── 生成图表 | Generate plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Proto Mask Statistics: {args.label_a} vs {args.label_b}",
            fontsize=14, fontweight="bold"
        )

        # (1) 激活值分布直方图 | Activation histogram
        ax = axes[0, 0]
        for label, color in [(args.label_a, "red"), (args.label_b, "blue")]:
            raw = [s["histogram_raw"] for s in all_stats[label]
                   if s.get("histogram_raw") is not None and len(s["histogram_raw"]) > 0]
            if raw:
                all_vals = np.concatenate([r[:10000] for r in raw])  # 采样防 OOM | subsample
                ax.hist(all_vals, bins=50, range=(0, 1), alpha=0.5, color=color,
                        label=label, density=True)
        ax.set_xlabel("proto_mask activation"); ax.set_ylabel("density")
        ax.set_title("Proto Mask Activation Distribution")
        ax.legend()

        # (2) Proto mask mean 分布 | Proto mask mean distribution
        ax = axes[0, 1]
        for label, color in [(args.label_a, "red"), (args.label_b, "blue")]:
            vals = [s["mean"] for s in all_stats[label]]
            ax.hist(vals, bins=30, alpha=0.5, color=color, label=label, density=True)
        ax.set_xlabel("proto_mask mean"); ax.set_title("Per-Tile Proto Mask Mean")
        ax.legend()

        # (3) Proto mask std 分布 | Proto mask std distribution
        ax = axes[0, 2]
        for label, color in [(args.label_a, "red"), (args.label_b, "blue")]:
            vals = [s["std"] for s in all_stats[label]]
            ax.hist(vals, bins=30, alpha=0.5, color=color, label=label, density=True)
        ax.set_xlabel("proto_mask std"); ax.set_title("Per-Tile Proto Mask Std (Spatial Variability)")
        ax.legend()

        # (4) FG IoU 分布 | FG IoU distribution
        ax = axes[1, 0]
        for label, color in [(args.label_a, "red"), (args.label_b, "blue")]:
            vals = [s["fg_iou"] for s in all_stats[label] if s.get("fg_iou") is not None]
            if vals:
                ax.hist(vals, bins=30, alpha=0.5, color=color, label=label, density=True)
        ax.set_xlabel("IoU with GT foreground"); ax.set_title("Proto Mask vs GT Foreground IoU")
        ax.legend()

        # (5) Proto basis L2 分布 | Proto basis L2
        ax = axes[1, 1]
        for label, color in [(args.label_a, "red"), (args.label_b, "blue")]:
            vals = [s["basis_l2"] for s in all_stats[label]]
            ax.hist(vals, bins=30, alpha=0.5, color=color, label=label, density=True)
        ax.set_xlabel("proto_basis L2 (per-channel mean)"); ax.set_title("Proto Basis Magnitude")
        ax.legend()

        # (6) Per-class FG IoU 对比 | Per-class FG IoU comparison
        ax = axes[1, 2]
        classes = sorted(set(
            list(per_class_accum[args.label_a].keys()) +
            list(per_class_accum[args.label_b].keys())
        ))
        x = np.arange(len(classes))
        width = 0.35
        for i, (label, color) in enumerate([(args.label_a, "red"), (args.label_b, "blue")]):
            ious = []
            for c in classes:
                vals = [s["fg_iou"] for s in per_class_accum[label].get(c, [])
                        if s.get("fg_iou") is not None]
                ious.append(np.mean(vals) if vals else 0)
            ax.bar(x + i * width, ious, width, color=color, alpha=0.7, label=label)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([CATEGORY_NAMES.get(c, str(c)) for c in classes],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean FG IoU"); ax.set_title("Per-Class Proto Mask FG IoU")
        ax.legend()

        plt.tight_layout()
        fig_path = out_dir / "proto_mask_stats.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved: {fig_path}")

    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
