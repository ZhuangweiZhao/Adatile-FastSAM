#!/usr/bin/env python3
"""
Instance Generation Method Sweep — 对比不同实例化方法的 AP.
=============================================================

不重新训练, 纯推理比较: connected_components vs watershed_distance vs watershed_gradient.

用法 | Usage::

    python tools/diag/diag_instance_method_sweep.py \
        --checkpoint runs/云服务器/runs(2)/best_model.pt \
        --device cuda
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
    _build_class_index_instance, _resolve_paths, _extract_source_image,
    load_instance_tile_and_mask, load_tile_and_mask,
    extract_features, compute_support_prototype,
    semantic_mask_to_binary, CATEGORY_NAMES,
)
from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
from adatile.metrics.coco_eval import COCOInstanceEvaluator
from adatile.metrics.instance_generation import generate_instances
from adatile.utils.seed import set_seed

OUT_DIR = _PROJECT_ROOT / "runs" / "diag" / "instance_method_sweep"


def _fastsam_weights_path() -> Path:
    return _PROJECT_ROOT / "thirdLibrary" / "FastSAM" / "weights" / "FastSAM-x.pt"


def load_model(device, checkpoint):
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
    test_img = np.zeros((896, 896, 3), dtype=np.uint8)
    test_feats = extract_features(model, [test_img], device)
    p4c = test_feats[0]["p4"].shape[1]
    normalize_proto = ckpt.get("normalize_proto", "none")
    decoder = AdaptiveSparseDecoder(in_channels=p4c, use_fdr=False, normalize_proto=normalize_proto).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    return model, decoder, normalize_proto


def decoder_forward(decoder, p4, proto_masks, support_proto):
    """Standard decoder forward (no weight, for eval)."""
    if proto_masks.dim() == 4:
        proto_masks = proto_masks.squeeze(0)
    if support_proto.dim() == 2:
        support_proto = support_proto.squeeze(0)
    proto_masks_norm = decoder._normalize_proto(proto_masks)
    coeffs = decoder.coeff_predictor(support_proto.unsqueeze(0))
    proto_mask = decoder.coeff_predictor.generate_mask(coeffs, proto_masks_norm)
    feat_proj = decoder.feat_proj(p4)
    feat_refined = decoder.feat_refine(feat_proj)
    refined_logit = decoder.mask_head(feat_refined)
    refined_logit_up = F.interpolate(refined_logit, size=proto_mask.shape[1:], mode='bilinear', align_corners=False)
    query_refine = refined_logit_up.squeeze(1).squeeze(0)
    pm = proto_mask.squeeze(0)
    final_logit = query_refine + pm
    return torch.sigmoid(final_logit)


def evaluate_method(model, decoder, class_protos, query_stems, stem_to_id,
                    coco_gt_path, data_root, eval_split, is_instance,
                    device, method, score_thr, min_area, min_distance):
    """Run evaluation with a specific instance generation method."""
    evaluator = COCOInstanceEvaluator(coco_gt_path, iouType="segm")
    evaluated_ids = []

    for stem in tqdm(query_stems, desc=f"  {method}", leave=False):
        image_id = stem_to_id.get(stem)
        if image_id is None:
            continue
        evaluated_ids.append(image_id)

        if is_instance:
            img, _ = load_instance_tile_and_mask(stem, eval_split, data_root, target_class_id=None)
        else:
            img, _ = load_tile_and_mask(stem, eval_split, data_root, target_class_id=None)
        H, W = img.shape[:2]
        feats = extract_features(model, [img], device)[0]

        for cls_id, pk in class_protos.items():
            with torch.no_grad():
                final_mask = decoder_forward(decoder, feats["p4"], feats["proto"], pk["proto"])
            prob = F.interpolate(
                final_mask.unsqueeze(0).unsqueeze(0), size=(H, W),
                mode='bilinear', align_corners=False
            ).squeeze().detach().float().cpu().numpy()

            insts = generate_instances(
                prob, method=method, score_thr=score_thr,
                min_area=min_area, min_distance=min_distance,
            )
            for it in insts:
                evaluator.add_prediction(image_id, cls_id, it["mask"], it["score"])

    ap = evaluator.evaluate(verbose=False, image_ids=evaluated_ids)
    ap_ag = evaluator.evaluate_class_agnostic(verbose=False, image_ids=evaluated_ids)
    return {
        "method": method,
        "n_evaluated": len(evaluated_ids),
        "AP": round(float(ap["AP"]), 4),
        "AP50": round(float(ap["AP50"]), 4),
        "AP75": round(float(ap["AP75"]), 4),
        "AP_small": round(float(ap.get("AP_small", 0)), 4),
        "AP_medium": round(float(ap.get("AP_medium", 0)), 4),
        "AP_large": round(float(ap.get("AP_large", 0)), 4),
        "AP_class_agnostic": round(float(ap_ag["AP"]), 4),
    }


def main():
    p = argparse.ArgumentParser(description="Instance Generation Method Sweep")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--methods", type=str, default="connected_components,watershed_distance,watershed_gradient")
    p.add_argument("--k-shot", type=int, default=1)
    p.add_argument("--per-class", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--data-format", type=str, default="isaid_instance")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--min-area", type=int, default=16)
    p.add_argument("--min-distance", type=int, default=12)
    args = p.parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    data_root, data_format, _train_split, eval_split = _resolve_paths(args)
    data_root = Path(data_root)
    is_instance = (data_format == "isaid_instance")

    methods = [m.strip() for m in args.methods.split(",")]

    out_dir = OUT_DIR / datetime.now().strftime("%m%d_%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Instance Method Sweep: {methods}")
    print(f"Output: {out_dir}")

    # Load model
    print("\n[1/3] Loading model...")
    model, decoder, norm_proto = load_model(device, args.checkpoint)
    print(f"  normalize_proto={norm_proto}")

    # Build class index & prototypes
    print("\n[2/3] Building prototypes...")
    if is_instance:
        class_index = _build_class_index_instance(data_root, eval_split)
    else:
        from tools.train.train_fewshot_allclass import _build_class_index_tiles
        class_index = _build_class_index_tiles(data_root, eval_split)

    # Manifest
    manifest_path = data_root / f"evaluation_manifest_{eval_split}.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            query_stems = sorted([Path(n).stem for n in json.load(f)])
    else:
        rng = random.Random(args.seed)
        rng_query = random.Random(args.seed + 99999)
        query_stems_set = set()
        for cls_id, src_to_tiles in class_index.items():
            sources = list(src_to_tiles.keys())
            if len(sources) < args.k_shot + 1:
                continue
            nq = min(args.per_class, len(sources) - args.k_shot)
            for s in rng_query.sample(sources, nq):
                q_tile = random.Random(args.seed + int(hashlib.md5(s.encode()).hexdigest(), 16)).choice(src_to_tiles[s])
                query_stems_set.add(q_tile)
        query_stems = sorted(query_stems_set)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump([f"{s}.png" for s in query_stems], f, indent=2, ensure_ascii=False)

    print(f"  {len(query_stems)} query tiles")

    # Build prototypes
    rng = random.Random(args.seed)
    class_protos = {}
    for cls_id, src_to_tiles in class_index.items():
        sources = list(src_to_tiles.keys())
        if len(sources) < args.k_shot + 1:
            continue
        cls_tiles = {t for tiles in src_to_tiles.values() for t in tiles}
        query_src = sorted({_extract_source_image(t) for t in (set(query_stems) & cls_tiles)})
        support_pool = [s for s in sources if s not in set(query_src)]
        if len(support_pool) < args.k_shot:
            continue
        support_sources = rng.sample(support_pool, args.k_shot)
        support_stems = [t for s in support_sources for t in src_to_tiles[s]]
        support_imgs = []
        for stem in support_stems:
            if is_instance:
                img, _ = load_instance_tile_and_mask(stem, eval_split, data_root, target_class_id=cls_id)
            else:
                img, _ = load_tile_and_mask(stem, eval_split, data_root, target_class_id=cls_id)
            support_imgs.append(img)
        support_feats = extract_features(model, support_imgs, device)
        class_protos[cls_id] = {"proto": compute_support_prototype(support_feats, source="p4")}
    print(f"  {len(class_protos)} class prototypes")

    # COCO GT
    coco_gt_path = str(data_root / "annotations" / f"instances_{eval_split}.json")
    from pycocotools.coco import COCO
    coco = COCO(coco_gt_path)
    stem_to_id = {Path(v["file_name"]).stem: k for k, v in coco.imgs.items()}

    # Sweep methods
    print(f"\n[3/3] Sweeping {len(methods)} methods...")
    results = []
    for method in methods:
        print(f"\n  Method: {method}")
        r = evaluate_method(
            model, decoder, class_protos, query_stems, stem_to_id,
            coco_gt_path, data_root, eval_split, is_instance,
            device, method, args.score_thr, args.min_area, args.min_distance,
        )
        results.append(r)
        print(f"    AP={r['AP']:.4f}  AP50={r['AP50']:.4f}  AP75={r['AP75']:.4f}  "
              f"AP_s={r['AP_small']:.4f}  AP_m={r['AP_medium']:.4f}  AP_l={r['AP_large']:.4f}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"  Instance Generation Method Sweep")
    print(f"{'='*90}")
    print(f"  {'Method':<30s} {'AP':>8s} {'AP50':>8s} {'AP75':>8s} {'AP_s':>8s} {'AP_m':>8s} {'AP_l':>8s} {'AP_ag':>8s}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r['method']:<30s} {r['AP']:>8.4f} {r['AP50']:>8.4f} {r['AP75']:>8.4f} "
              f"{r['AP_small']:>8.4f} {r['AP_medium']:>8.4f} {r['AP_large']:>8.4f} "
              f"{r['AP_class_agnostic']:>8.4f}")

    # Best
    best = max(results, key=lambda r: r["AP"])
    base = results[0]
    print(f"\n  Best method: {best['method']} (AP={best['AP']:.4f})")
    print(f"  Δ vs baseline ({base['method']}): {best['AP'] - base['AP']:+.4f} AP")

    # Save
    json_path = out_dir / "instance_method_sweep.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: {json_path}")


if __name__ == "__main__":
    main()
