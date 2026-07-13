#!/usr/bin/env python3
"""
Proto Weight Sweep — AP vs Proto Contribution Weight.
=======================================================

升级版 Probe 3: 不只看 OFF (w=0), 而是扫连续曲线 w ∈ {0, 0.25, 0.5, 1, 2}.

核心问题 | Core Question:
    final_logits = query_refine + w × proto_mask
    改变 w 会影响 AP 吗?

三种可能 | Three Possibilities:
    - 平坦 (w=0 ≈ w=1 ≈ w=2): Proto 推理时真死. 收益 100% 来自训练改善.
    - 单调上升 (AP ∝ w): Proto 虽只有 3%, 但对推理有用.
    - 倒 U (峰值在 w=0.5): Proto 有最优参与度.

用法 | Usage::

    python tools/diag/diag_proto_weight_sweep.py \
        --checkpoint runs/云服务器/runs(2)/best_model.pt \
        --device cuda

输出 | Output:
    runs/diag/proto_weight_sweep/
    ├── ap_vs_proto_weight.png     # AP-Proto Weight 曲线
    ├── per_class_heatmap.png      # 每类 AP 随 weight 变化
    └── proto_weight_sweep.json    # 完整数据
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
from adatile.metrics.coco_eval import (
    COCOInstanceEvaluator,
    connected_components_to_instances,
)
from adatile.utils.seed import set_seed

OUT_DIR = _PROJECT_ROOT / "runs" / "diag" / "proto_weight_sweep"

# ═══════════════════════════════════════════════════════════════════
# Weighted Decoder Forward | 加权 Decoder 前向
# ═══════════════════════════════════════════════════════════════════

def weighted_decoder_forward(
    decoder: AdaptiveSparseDecoder,
    p4_features: torch.Tensor,
    proto_masks: torch.Tensor,
    support_proto: torch.Tensor,
    proto_weight: float,
) -> torch.Tensor:
    """
    带 proto 权重的 decoder 前向 | Decoder forward with proto weight.

    唯一改动: final_logits = query_refine + proto_weight × proto_mask
    Only change: weighted proto contribution in additive fusion.

    :param proto_weight: 0 = proto OFF, 1 = default, 2 = double proto.
    :return: [H/4, W/4] final mask ∈ [0, 1].
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

    # Step 4: Weighted fusion ← ONLY CHANGE
    query_refine = refined_logit_up.squeeze(1).squeeze(0)  # [H/4, W/4]
    pm = proto_mask.squeeze(0)                               # [H/4, W/4]
    final_logit = query_refine + proto_weight * pm          # ← WEIGHT HERE
    final_mask = torch.sigmoid(final_logit)

    return final_mask


# ═══════════════════════════════════════════════════════════════════
# 模型加载 + Manifest (同 Probe 1/2)
# ═══════════════════════════════════════════════════════════════════

def _fastsam_weights_path() -> Path:
    return _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"


def build_model_and_decoder(device: str, checkpoint: str):
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
        print(f"  Restored {unfreeze_layers} backbone layers")

    test_img = np.zeros((896, 896, 3), dtype=np.uint8)
    test_feats = extract_features(model, [test_img], device)
    p4_channels = test_feats[0]["p4"].shape[1]
    print(f"  P4 channels: {p4_channels}")

    normalize_proto = ckpt.get("normalize_proto", "none")
    decoder = AdaptiveSparseDecoder(
        in_channels=p4_channels, use_fdr=False, normalize_proto=normalize_proto
    ).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    print(f"  Decoder: normalize_proto={normalize_proto}, epoch={ckpt.get('epoch','?')}")

    return model, decoder, normalize_proto


def _det_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def load_manifest(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        names = json.load(f)
    return [Path(n).stem for n in names]


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


def prob_map_to_instances(prob_map: np.ndarray, category_id: int,
                          score_thr: float = 0.5, min_area: int = 16) -> list[dict]:
    """前景概率图 → 实例列表."""
    binary = (prob_map > score_thr).astype(np.uint8)
    comps = connected_components_to_instances(binary, min_area=min_area)
    instances = []
    for comp in comps:
        score = float(prob_map[comp].mean())
        instances.append({"category_id": category_id, "mask": comp, "score": score})
    return instances


# ═══════════════════════════════════════════════════════════════════
# 主流程 | Main
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Proto Weight Sweep — AP vs Proto Contribution Weight")
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint to evaluate")
    p.add_argument("--weights", type=str, default="0,0.25,0.5,1,2",
                   help="Comma-separated proto weights (default: 0,0.25,0.5,1,2)")
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--per-class", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--data-format", type=str, default="isaid_instance")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=16)
    return p.parse_args()


def evaluate_at_weight(
    model, decoder, class_protos, query_stems, stem_to_id,
    coco_gt_path, data_root, eval_split, is_instance,
    device, proto_weight, score_thr, min_area,
) -> dict:
    """
    在指定 proto_weight 下运行完整评估 | Run full evaluation at given proto_weight.

    :return: {AP, AP50, AP75, AP_small, AP_medium, AP_large, AP_class_agnostic, instance_miou}
    """
    evaluator = COCOInstanceEvaluator(coco_gt_path, iouType="segm")
    evaluated_ids = []
    per_class_gt_ious = defaultdict(list)

    for stem in tqdm(query_stems, desc=f"  w={proto_weight:.2f}", leave=False):
        image_id = stem_to_id.get(stem)
        if image_id is None:
            continue
        evaluated_ids.append(image_id)

        img, _ = _load_tile_img_mask(stem, eval_split, data_root, is_instance, target_class_id=None)
        H, W = img.shape[:2]
        feats = extract_features(model, [img], device)[0]

        # 逐类前景图 → 实例 | per-class FG map → instances
        for cls_id, pk in class_protos.items():
            with torch.no_grad():
                final_mask = weighted_decoder_forward(
                    decoder, feats["p4"], feats["proto"], pk["proto"],
                    proto_weight=proto_weight,
                )
            # 上采样到 tile 分辨率 | Upsample to tile resolution
            # final_mask is [H/4, W/4] — add batch and channel dims
            prob = F.interpolate(
                final_mask.detach().unsqueeze(0).unsqueeze(0),  # [1, 1, H/4, W/4]
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze().detach().float().cpu().numpy()

            insts = prob_map_to_instances(prob, cls_id, score_thr, min_area)
            for it in insts:
                evaluator.add_prediction(image_id, it["category_id"], it["mask"], it["score"])

            # Instance mIoU accumulator
            gt_masks_for_cls = []  # Simplified — skip per-class mIoU for sweep
            # (Full mIoU computation is expensive and not needed for the sweep;
            #  we focus on AP which is the primary metric.)

    # COCO AP
    ap_result = evaluator.evaluate(verbose=False, image_ids=evaluated_ids)
    ap_agnostic = evaluator.evaluate_class_agnostic(verbose=False, image_ids=evaluated_ids)

    return {
        "proto_weight": proto_weight,
        "n_evaluated": len(evaluated_ids),
        "AP": round(float(ap_result["AP"]), 4),
        "AP50": round(float(ap_result["AP50"]), 4),
        "AP75": round(float(ap_result["AP75"]), 4),
        "AP_small": round(float(ap_result.get("AP_small", 0)), 4),
        "AP_medium": round(float(ap_result.get("AP_medium", 0)), 4),
        "AP_large": round(float(ap_result.get("AP_large", 0)), 4),
        "AP_class_agnostic": round(float(ap_agnostic["AP"]), 4),
        "AP50_class_agnostic": round(float(ap_agnostic["AP50"]), 4),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    data_root, data_format, _train_split, eval_split = _resolve_paths(args)
    data_root = Path(data_root)
    is_instance = (data_format == "isaid_instance")

    weights = [float(w.strip()) for w in args.weights.split(",")]
    print(f"Proto Weight Sweep: w ∈ {weights}")
    print(f"Checkpoint: {args.checkpoint}")

    # ── 输出目录 | Output dir ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%m%d_%H%M")
        out_dir = OUT_DIR / f"sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── 加载模型 | Load model ──
    print("\n[1/4] Loading model & decoder...")
    model, decoder, norm_proto = build_model_and_decoder(device, args.checkpoint)

    # ── 类别索引 | Class index ──
    print("\n[2/4] Building class index & manifest...")
    if is_instance:
        class_index = _build_class_index_instance(data_root, eval_split)
    else:
        from tools.train.train_fewshot_allclass import _build_class_index_tiles
        class_index = _build_class_index_tiles(data_root, eval_split)

    # ── Manifest + Prototypes | Manifest + Prototypes ──
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        manifest_path = data_root / f"evaluation_manifest_{eval_split}.json"

    if not manifest_path.exists():
        # 生成 manifest | Generate manifest
        rng = random.Random(args.seed)
        rng_query = random.Random(args.seed + 99999)
        query_stems_set = set()
        for cls_id, src_to_tiles in class_index.items():
            sources = list(src_to_tiles.keys())
            if len(sources) < args.k_shot + 1:
                continue
            n_query = min(args.per_class, len(sources) - args.k_shot)
            query_sources = rng_query.sample(sources, n_query)
            for s in query_sources:
                q_tile = random.Random(args.seed + _det_hash(s)).choice(src_to_tiles[s])
                query_stems_set.add(q_tile)
        query_stems_set = set(sorted(query_stems_set))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        names = sorted([f"{s}.png" for s in query_stems_set])
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(names, f, indent=2, ensure_ascii=False)
        print(f"  Generated manifest: {len(query_stems_set)} tiles → {manifest_path}")

    query_stems = sorted(load_manifest(manifest_path))
    print(f"  {len(query_stems)} query tiles from manifest")

    # ── 构建 prototypes (只用一次) | Build prototypes (once) ──
    rng = random.Random(args.seed)
    class_protos = {}
    for cls_id, src_to_tiles in class_index.items():
        sources = list(src_to_tiles.keys())
        if len(sources) < args.k_shot + 1:
            continue
        # Query sources from manifest
        cls_tiles = {t for tiles in src_to_tiles.values() for t in tiles}
        query_src = sorted({_extract_source_image(t) for t in (set(query_stems) & cls_tiles)})
        query_src_set = set(query_src)
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
        support_feats = extract_features(model, support_imgs, device)
        class_protos[cls_id] = {"proto": compute_support_prototype(support_feats, source="p4")}
    print(f"  {len(class_protos)} class prototypes built")

    # ── COCO GT setup | COCO GT ──
    coco_gt_path = str(data_root / "annotations" / f"instances_{eval_split}.json")
    from pycocotools.coco import COCO
    coco = COCO(coco_gt_path)
    stem_to_id = {Path(v["file_name"]).stem: k for k, v in coco.imgs.items()}

    # ── 主循环: 逐 weight 评估 | Main loop: evaluate at each weight ──
    print(f"\n[3/4] Sweeping {len(weights)} weights over {len(query_stems)} tiles...")
    results = []
    for w in weights:
        print(f"\n  Weight = {w:.2f}")
        r = evaluate_at_weight(
            model, decoder, class_protos, query_stems, stem_to_id,
            coco_gt_path, data_root, eval_split, is_instance,
            device, w, args.score_thr, args.min_area,
        )
        results.append(r)
        print(f"    AP={r['AP']:.4f}  AP50={r['AP50']:.4f}  AP_agnostic={r['AP_class_agnostic']:.4f}")

    # ── 保存 JSON | Save JSON ──
    print(f"\n[4/4] Saving results...")
    output = {
        "config": {
            "checkpoint": args.checkpoint,
            "normalize_proto": norm_proto,
            "weights": weights,
            "k_shot": args.k_shot,
            "seed": args.seed,
            "n_query_tiles": len(query_stems),
            "n_classes": len(class_protos),
            "score_thr": args.score_thr,
            "min_area": args.min_area,
        },
        "results": results,
    }
    json_path = out_dir / "proto_weight_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    # ── 打印汇总表 | Print summary table ──
    print(f"\n{'='*80}")
    print(f"  Proto Weight Sweep Results")
    print(f"{'='*80}")
    header = f"  {'Weight':<10s} {'AP':>8s} {'AP50':>8s} {'AP75':>8s} {'AP_agn':>8s} {'AP_s':>8s} {'AP_m':>8s} {'AP_l':>8s}"
    print(header)
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['proto_weight']:<10.2f} {r['AP']:>8.4f} {r['AP50']:>8.4f} "
              f"{r['AP75']:>8.4f} {r['AP_class_agnostic']:>8.4f} "
              f"{r['AP_small']:>8.4f} {r['AP_medium']:>8.4f} {r['AP_large']:>8.4f}")
    print(f"{'='*80}")

    # ── 判断 | Interpretation ──
    ap_0 = results[0]["AP"]
    ap_1 = next((r["AP"] for r in results if abs(r["proto_weight"] - 1.0) < 0.01), None)
    ap_max = max(r["AP"] for r in results)
    ap_min = min(r["AP"] for r in results)
    delta_max = ap_max - ap_min

    print(f"\n  Interpretation:")
    print(f"    AP(w=0) = {ap_0:.5f}  |  AP(w=1) = {ap_1:.5f}  |  max Δ = {delta_max:.5f}")
    if delta_max < 0.0005:
        print(f"    → FLAT curve: Proto truly dead at inference. 100% gain from training improvement.")
        print(f"      Strongest causal evidence for H4 (Optimization Interference).")
    elif ap_0 < ap_1 and delta_max > 0.001:
        print(f"    → MONOTONIC: Proto helps at inference despite 3% contribution.")
    elif ap_0 > ap_1:
        print(f"    → NEGATIVE: Proto hurts at inference (w=0 is best).")
    else:
        print(f"    → NON-MONOTONIC: Proto has optimal contribution level.")

    # ── 生成图表 | Generate plot ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Proto Weight Sweep: AP vs Proto Contribution\n{Path(args.checkpoint).parent.name}",
                     fontsize=12, fontweight="bold")

        ws = [r["proto_weight"] for r in results]

        # (1) AP main metrics
        ax = axes[0]
        for metric, color, marker in [
            ("AP", "#2196F3", "o-"),
            ("AP50", "#4CAF50", "s--"),
            ("AP75", "#FF9800", "^--"),
            ("AP_class_agnostic", "#9C27B0", "d:"),
        ]:
            vals = [r[metric] for r in results]
            ax.plot(ws, vals, marker, color=color, label=metric, markersize=8, linewidth=2)
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="default (w=1)")
        ax.set_xlabel("Proto Weight"); ax.set_ylabel("AP")
        ax.set_title("AP Metrics vs Proto Weight")
        ax.legend(); ax.grid(True, alpha=0.3)

        # (2) AP by size
        ax = axes[1]
        for metric, color, marker in [
            ("AP_small", "#E91E63", "o-"),
            ("AP_medium", "#00BCD4", "s--"),
            ("AP_large", "#FFC107", "^--"),
        ]:
            vals = [r[metric] for r in results]
            ax.plot(ws, vals, marker, color=color, label=metric, markersize=8, linewidth=2)
        ax.axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="default (w=1)")
        ax.set_xlabel("Proto Weight"); ax.set_ylabel("AP")
        ax.set_title("AP by Object Size vs Proto Weight")
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = out_dir / "ap_vs_proto_weight.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure: {fig_path}")

    except ImportError:
        print("  [WARN] matplotlib not available, skipping plot")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
