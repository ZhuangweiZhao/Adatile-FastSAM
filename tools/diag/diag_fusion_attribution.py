#!/usr/bin/env python3
"""
Fusion 贡献比分析 (Probe 2) | Fusion Contribution Ratio Analysis.
==================================================================

三层分析框架 — 归因核心: 量化 proto_mask 与 query_refine 对 final_logits 的相对贡献。
M-F-T Framework — Attribution core: Quantify relative contribution of proto_mask vs query_refine.

核心问题 | Core Question:
    final_logits = query_refine + proto_mask
    Task 涨了 +31%, 是 proto_mask 的功劳还是 query_refine 的功劳?
    谁在主导输出?

指标 | Metrics:
    - proto_norm:     ||proto_mask||_2 (proto 支能量)
    - query_norm:     ||query_refine||_2 (query 支能量)
    - ratio:          ||proto|| / ||query|| (贡献比: >>1 = proto 主导, <<1 = query 主导)
    - cosine_sim:     cos(proto_mask, query_refine) (两支方向一致性: 协同还是对抗?)
    - final_variance:  final_logit 的空间方差 (越分散 → 越有区分力)

判断逻辑 | Decision Logic:
    - none: ratio >> 1 (proto 主导, 但 proto 是常数 → 输出退化)
      l2:   ratio 接近 1 (两支平衡, query 能参与) → 支持 H1 (净化融合)
    - l2:   ratio 仍然 >> 1 或 << 1 → H1 可能错误
    - l2:   ratio 与 AP 正相关 (proto 参与越多越好/越差) → 因果证据

用法 | Usage::

    python tools/diag/diag_fusion_attribution.py \\
        --checkpoint-a runs/.../uf8_none/best_model.pt \\
        --checkpoint-b runs/.../uf8_l2/best_model.pt \\
        --label-a none --label-b l2 \\
        --device cuda

输出 | Output:
    runs/diag/fusion_attribution/
    ├── fusion_ratio_histogram.png     # none vs l2 贡献比分布
    ├── fusion_scatter.png             # ratio vs activation 散点图
    ├── fusion_per_class.png           # 每类贡献比对比
    └── fusion_attribution.json        # 全部统计数据
"""

from __future__ import annotations

import sys, argparse, json, random, hashlib
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

OUT_DIR = _PROJECT_ROOT / "runs" / "diag" / "fusion_attribution"


# ═══════════════════════════════════════════════════════════════════
# Captured Decoder Forward (同 Probe 1 | Same as Probe 1)
# ═══════════════════════════════════════════════════════════════════

def captured_decoder_forward(
    decoder: AdaptiveSparseDecoder,
    p4_features: torch.Tensor,
    proto_masks: torch.Tensor,
    support_proto: torch.Tensor,
) -> dict:
    """
    运行 decoder 前向并捕获中间张量 | Run decoder forward & capture intermediates.

    精确镜像 AdaptiveSparseDecoder.forward() (跳过 FDR gating).
    Exactly mirrors AdaptiveSparseDecoder.forward() (skips FDR gating).

    :return: dict with proto_masks_norm, coeffs, proto_mask, query_refine, final_logit, final_mask.
    """
    if proto_masks.dim() == 4:
        proto_masks = proto_masks.squeeze(0)
    if support_proto.dim() == 2:
        support_proto = support_proto.squeeze(0)

    proto_masks_norm = decoder._normalize_proto(proto_masks)

    # Step 1: Proto mask
    coeffs = decoder.coeff_predictor(support_proto.unsqueeze(0))
    proto_mask = decoder.coeff_predictor.generate_mask(coeffs, proto_masks_norm)

    # Step 2-3: P4 refinement → mask head
    feat_proj = decoder.feat_proj(p4_features)
    feat_refined = decoder.feat_refine(feat_proj)
    refined_logit = decoder.mask_head(feat_refined)
    refined_logit_up = F.interpolate(
        refined_logit, size=proto_mask.shape[1:],
        mode='bilinear', align_corners=False,
    )

    # Step 4: Additive fusion
    query_refine = refined_logit_up.squeeze(1)
    pm = proto_mask.squeeze(0)
    final_logit = query_refine + pm
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
# Fusion 归因指标 | Fusion Attribution Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_fusion_attribution(
    proto_mask: torch.Tensor,       # [H/4, W/4]
    query_refine: torch.Tensor,     # [H/4, W/4]
    final_logit: torch.Tensor,      # [H/4, W/4]
    final_mask: torch.Tensor,       # [H/4, W/4]
) -> dict:
    """
    计算 proto_mask 与 query_refine 的贡献归因指标。
    Compute contribution attribution metrics between proto_mask and query_refine.

    :return: dict of scalar metrics.
    """
    pm = proto_mask.detach().float()
    qr = query_refine.detach().float()
    fl = final_logit.detach().float()
    fm = final_mask.detach().float()

    # ── L2 范数 (能量) | L2 norms (energy) ──
    proto_norm = float(pm.norm(p=2).item())
    query_norm = float(qr.norm(p=2).item())

    # ── 贡献比 | Contribution ratio ──
    #   ratio >> 1: proto 主导; ratio << 1: query 主导
    ratio = proto_norm / (query_norm + 1e-8)

    # ── 对数贡献比 (对称, 更适合统计) | Log ratio (symmetric, better for stats) ──
    log_ratio = float(np.log(max(ratio, 1e-8)))

    # ── Cosine 相似度 (方向一致性) | Cosine similarity (directional agreement) ──
    pm_flat = pm.reshape(-1)
    qr_flat = qr.reshape(-1)
    cos_sim = float(F.cosine_similarity(pm_flat.unsqueeze(0), qr_flat.unsqueeze(0)).item())

    # ── 归一化贡献比 (L1 归一化) | Normalized contribution (L1) ──
    pm_abs_sum = float(pm.abs().sum().item())
    qr_abs_sum = float(qr.abs().sum().item())
    total_abs = pm_abs_sum + qr_abs_sum + 1e-8
    proto_frac = pm_abs_sum / total_abs   # proto 贡献占比 ∈ [0, 1]
    query_frac = qr_abs_sum / total_abs

    # ── 空间方差 (区分力) | Spatial variance (discriminability) ──
    final_var = float(fl.var().item())
    proto_var = float(pm.var().item())
    query_var = float(qr.var().item())

    # ── Proto mask 与 final mask 的 Spearman 秩相关 | Rank correlation proto vs final ──
    #   度量 proto 对最终输出的"形状影响力" | Measures proto's "shape influence" on output
    try:
        from scipy.stats import spearmanr
        pm_sample = pm_flat[:5000].cpu().numpy()
        fm_sample = fm.reshape(-1)[:5000].cpu().numpy()
        spearman_r = float(spearmanr(pm_sample, fm_sample)[0])
    except (ImportError, ValueError):
        spearman_r = float(np.corrcoef(
            pm_flat[:5000].cpu().numpy(), fm.reshape(-1)[:5000].cpu().numpy()
        )[0, 1]) if pm_flat.shape[0] > 1 else 0.0

    # ── Proto mask 饱和分数 | Proto mask saturation fraction ──
    sat_frac = float(((pm < 1e-6) | (pm > 1 - 1e-6)).float().mean().item())

    return {
        "proto_norm": proto_norm,
        "query_norm": query_norm,
        "ratio": ratio,
        "log_ratio": log_ratio,
        "cosine_sim": cos_sim,
        "proto_frac": proto_frac,
        "query_frac": query_frac,
        "final_var": final_var,
        "proto_var": proto_var,
        "query_var": query_var,
        "spearman_r": spearman_r,
        "sat_frac": sat_frac,
    }


# ═══════════════════════════════════════════════════════════════════
# 模型加载 + Manifest (同 Probe 1 | Same as Probe 1)
# ═══════════════════════════════════════════════════════════════════

def _fastsam_weights_path() -> Path:
    return _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"


def build_model_and_decoder(args, device: str, checkpoint: str, label: str):
    from ultralytics import FastSAM

    model = FastSAM(str(_fastsam_weights_path()))
    model.model.to(device).eval()
    for p in model.model.parameters():
        p.requires_grad = False

    ckpt = torch.load(checkpoint, map_location=device)

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

    normalize_proto = ckpt.get("normalize_proto", "none")
    decoder = AdaptiveSparseDecoder(
        in_channels=p4_channels, use_fdr=False, normalize_proto=normalize_proto
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    print(f"  [{label}] Decoder loaded (normalize_proto={normalize_proto}, epoch={ckpt.get('epoch','?')})")

    return model, decoder, normalize_proto


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
        description="Fusion Contribution Ratio Analysis (M-F-T Probe 2)")
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
    p.add_argument("--manifest", type=str, default=None)
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

    print(f"Fusion Attribution Analysis: {args.label_a} vs {args.label_b}")
    print(f"Output: {out_dir}")

    # ── 加载模型 | Load models ──
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

    # ── Manifest + Prototypes | Manifest + Prototypes ──
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
            query_sources = sorted({
                _extract_source_image(t)
                for t in (set(load_manifest(manifest_path)) & cls_tiles)
            })

        query_src_set = set(query_sources)
        support_pool = [s for s in sources if s not in query_src_set]
        if len(support_pool) < args.k_shot:
            continue
        support_sources = rng.sample(support_pool, args.k_shot)
        support_stems = [t for s in support_sources for t in src_to_tiles[s]]

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

    # ── 主循环: 逐 tile 归因 | Main loop: per-tile attribution ──
    print(f"\n[4/5] Analyzing fusion contributions ({len(query_list)} tiles)...")

    all_attrib = {args.label_a: [], args.label_b: []}
    per_class_attrib = {
        args.label_a: defaultdict(list),
        args.label_b: defaultdict(list),
    }
    models = {args.label_a: (model_a, decoder_a), args.label_b: (model_b, decoder_b)}

    for qi, stem in enumerate(tqdm(query_list, desc="  Tiles")):
        img, _ = _load_tile_img_mask(stem, eval_split, data_root, is_instance, target_class_id=None)

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
                except Exception:
                    continue

                attrib = compute_fusion_attribution(
                    captured["proto_mask"],
                    captured["query_refine"],
                    captured["final_logit"],
                    captured["final_mask"],
                )
                attrib["cls_id"] = cls_id
                attrib["tile"] = stem
                all_attrib[label].append(attrib)
                per_class_attrib[label][cls_id].append(attrib)

    # ── 汇总 | Aggregate ──
    print("\n[5/5] Aggregating & saving...")

    def aggregate(attribs: list[dict]) -> dict:
        if not attribs:
            return {}
        keys = ["proto_norm", "query_norm", "ratio", "log_ratio", "cosine_sim",
                "proto_frac", "query_frac", "final_var", "proto_var", "query_var",
                "spearman_r", "sat_frac"]
        aggr = {}
        for k in keys:
            vals = [a[k] for a in attribs if a.get(k) is not None]
            if vals:
                aggr[f"{k}_mean"] = float(np.mean(vals))
                aggr[f"{k}_std"] = float(np.std(vals))
                aggr[f"{k}_median"] = float(np.median(vals))
                aggr[f"{k}_q25"] = float(np.percentile(vals, 25))
                aggr[f"{k}_q75"] = float(np.percentile(vals, 75))
        aggr["n_tiles"] = len(attribs)
        return aggr

    result = {
        "config": {
            "label_a": args.label_a, "label_b": args.label_b,
            "checkpoint_a": args.checkpoint_a, "checkpoint_b": args.checkpoint_b,
            "normalize_a": norm_a, "normalize_b": norm_b,
            "k_shot": args.k_shot, "seed": args.seed,
            "n_query_tiles": len(query_list), "n_classes": len(class_protos),
        },
        args.label_a: aggregate(all_attrib[args.label_a]),
        args.label_b: aggregate(all_attrib[args.label_b]),
        "per_class": {
            args.label_a: {
                str(cls): aggregate(vals)
                for cls, vals in per_class_attrib[args.label_a].items()
            },
            args.label_b: {
                str(cls): aggregate(vals)
                for cls, vals in per_class_attrib[args.label_b].items()
            },
        },
    }

    # ── 保存 JSON | Save JSON ──
    json_path = out_dir / "fusion_attribution.json"
    result["_raw_attrib"] = {
        args.label_a: all_attrib[args.label_a],
        args.label_b: all_attrib[args.label_b],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Stats saved: {json_path}")

    # ── 打印关键对比 | Print key comparison ──
    a_aggr = result[args.label_a]
    b_aggr = result[args.label_b]
    print(f"\n{'='*70}")
    print(f"  Fusion Attribution: {args.label_a} vs {args.label_b}")
    print(f"{'='*70}")
    print(f"  {'Metric':<28s} {'none':>14s} {'l2':>14s} {'Δ':>14s}")
    print(f"  {'-'*70}")
    for k, display in [
        ("ratio_mean", "proto/query ratio"),
        ("proto_frac_mean", "proto fraction"),
        ("query_frac_mean", "query fraction"),
        ("cosine_sim_mean", "cosine similarity"),
        ("spearman_r_mean", "Spearman r (proto→final)"),
        ("final_var_mean", "final logit variance"),
        ("sat_frac_mean", "proto sat fraction"),
    ]:
        va = a_aggr.get(k, None)
        vb = b_aggr.get(k, None)
        if isinstance(va, float) and isinstance(vb, float):
            delta = vb - va
            print(f"  {display:<28s} {va:14.6f} {vb:14.6f} {delta:+14.6f}")
        else:
            print(f"  {display:<28s} {va} {vb}")
    print(f"{'='*70}")

    # ── 解读提示 | Interpretation hints ──
    print(f"\n  Interpretation:")
    ratio_a = a_aggr.get("ratio_mean", 0)
    ratio_b = b_aggr.get("ratio_mean", 0)
    cos_a = a_aggr.get("cosine_sim_mean", 0)
    cos_b = b_aggr.get("cosine_sim_mean", 0)
    sat_a = a_aggr.get("sat_frac_mean", 0)
    sat_b = b_aggr.get("sat_frac_mean", 0)

    if ratio_a > 10:
        print(f"  [{args.label_a}] ratio={ratio_a:.1f}: proto DOMINATES (query suppressed)")
    elif ratio_a > 2:
        print(f"  [{args.label_a}] ratio={ratio_a:.1f}: proto leads")
    elif ratio_a > 0.5:
        print(f"  [{args.label_a}] ratio={ratio_a:.1f}: balanced contribution")
    else:
        print(f"  [{args.label_a}] ratio={ratio_a:.1f}: query dominates")

    if ratio_b > 10:
        print(f"  [{args.label_b}] ratio={ratio_b:.1f}: proto DOMINATES (query suppressed)")
    elif ratio_b > 2:
        print(f"  [{args.label_b}] ratio={ratio_b:.1f}: proto leads")
    elif ratio_b > 0.5:
        print(f"  [{args.label_b}] ratio={ratio_b:.1f}: balanced contribution")
    else:
        print(f"  [{args.label_b}] ratio={ratio_b:.1f}: query dominates")

    if sat_a > 0.5:
        print(f"  [{args.label_a}] sat_frac={sat_a:.3f}: proto_mask heavily saturated (dead zones)")
    if sat_b > 0.5:
        print(f"  [{args.label_b}] sat_frac={sat_b:.3f}: proto_mask heavily saturated (dead zones)")

    # ── 生成图表 | Generate plots ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Fusion Attribution: {args.label_a} vs {args.label_b}\n"
            f"final_logits = query_refine + proto_mask",
            fontsize=13, fontweight="bold"
        )

        # (1) 贡献比分布 | Ratio distribution
        ax = axes[0, 0]
        for label, color in [(args.label_a, "red"), (args.label_b, "blue")]:
            vals = [a["log_ratio"] for a in all_attrib[label]]
            ax.hist(vals, bins=40, alpha=0.5, color=color, label=label, density=True)
        ax.axvline(0, color="black", linestyle="--", alpha=0.5, label="equal (ratio=1)")
        ax.set_xlabel("log(proto/query ratio)"); ax.set_ylabel("density")
        ax.set_title("Proto vs Query Contribution Ratio")
        ax.legend()

        # (2) Cosine 相似度 | Cosine similarity
        ax = axes[0, 1]
        for label, color in [(args.label_a, "red"), (args.label_b, "blue")]:
            vals = [a["cosine_sim"] for a in all_attrib[label]]
            ax.hist(vals, bins=40, alpha=0.5, color=color, label=label, density=True)
        ax.set_xlabel("cos(proto_mask, query_refine)"); ax.set_title("Direction Agreement")
        ax.legend()

        # (3) Proto fraction 分布 | Proto fraction distribution
        ax = axes[0, 2]
        for label, color in [(args.label_a, "red"), (args.label_b, "blue")]:
            vals = [a["proto_frac"] for a in all_attrib[label]]
            ax.hist(vals, bins=40, alpha=0.5, color=color, label=label, density=True)
        ax.axvline(0.5, color="black", linestyle="--", alpha=0.5, label="equal (0.5)")
        ax.set_xlabel("proto fraction"); ax.set_title("Proto Absolute Contribution Fraction")
        ax.legend()

        # (4) 散点: ratio vs spearman_r | Scatter: ratio vs spearman_r
        ax = axes[1, 0]
        for label, color, marker in [(args.label_a, "red", "o"), (args.label_b, "blue", "x")]:
            xs = [a["log_ratio"] for a in all_attrib[label][:500]]
            ys = [a["spearman_r"] for a in all_attrib[label][:500]]
            ax.scatter(xs, ys, c=color, marker=marker, alpha=0.3, s=10, label=label)
        ax.set_xlabel("log(ratio)"); ax.set_ylabel("Spearman r (proto→final)")
        ax.set_title("Ratio vs Shape Influence (Spearman)")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.legend()

        # (5) Saturation fraction 分布 | Saturation fraction
        ax = axes[1, 1]
        for label, color in [(args.label_a, "red"), (args.label_b, "blue")]:
            vals = [a["sat_frac"] for a in all_attrib[label]]
            ax.hist(vals, bins=40, alpha=0.5, color=color, label=label, density=True)
        ax.set_xlabel("proto_mask sat fraction"); ax.set_title("Proto Mask Saturation")
        ax.legend()

        # (6) Per-class ratio 对比 | Per-class ratio comparison
        ax = axes[1, 2]
        classes = sorted(set(
            list(per_class_attrib[args.label_a].keys()) +
            list(per_class_attrib[args.label_b].keys())
        ))
        x = np.arange(len(classes))
        width = 0.35
        for i, (label, color) in enumerate([(args.label_a, "red"), (args.label_b, "blue")]):
            ratios = []
            for c in classes:
                vals = [a["ratio"] for a in per_class_attrib[label].get(c, [])]
                ratios.append(np.median(vals) if vals else 0)
            ax.bar(x + i * width, ratios, width, color=color, alpha=0.7, label=label)
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="equal")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([CATEGORY_NAMES.get(c, str(c)) for c in classes],
                           rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Median proto/query ratio"); ax.set_title("Per-Class Fusion Ratio")
        ax.legend()

        plt.tight_layout()
        fig_path = out_dir / "fusion_attribution.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure saved: {fig_path}")

    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
