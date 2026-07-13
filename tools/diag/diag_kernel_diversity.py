"""
Kernel Diversity Diagnostic | 动态核多样性诊断.
================================================
诊断 DynamicKernelDecoder 的 16 个 kernel mask 输出:
- 各 kernel 是否产生不同的 mask?
- 有多少 kernel 是死核 (全零)?
- 各 kernel 的 mean/max 分布如何?
- Kernel 之间 pairwise IoU 是多少?

Diagnose: Are the N kernels producing diverse instance masks or redundant blobs?

用法 | Usage:
    python tools/diag/diag_kernel_diversity.py \
        --checkpoint runs/.../best_model.pt \
        --data-root data/iSAID_instance_fewshot \
        --device cuda --n-samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adatile.decoder.dynamic_kernel_decoder import DynamicKernelDecoder
from adatile.backbone.fastsam_backbone import FastSAMBackbone
from adatile.datasets.isaid_instance_fewshot import ISAIDInstanceFewShotDataset
from adatile.utils.seed import set_seed


def compute_pairwise_iou(masks: torch.Tensor) -> np.ndarray:
    """
    Compute pairwise IoU matrix for N binary masks.
    计算 N 个二值掩码的 pairwise IoU 矩阵.
    """
    N = masks.shape[0]
    iou_mat = np.zeros((N, N))
    masks_np = (masks > 0.5).float().cpu().numpy()
    for i in range(N):
        for j in range(i + 1, N):
            inter = (masks_np[i] * masks_np[j]).sum()
            union = (masks_np[i] + masks_np[j]).clip(0, 1).sum()
            iou_val = inter / max(union, 1)
            iou_mat[i, j] = iou_val
            iou_mat[j, i] = iou_val
    return iou_mat


def main():
    parser = argparse.ArgumentParser(description="Kernel Diversity Diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/iSAID_instance_fewshot")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of random tiles to sample | 随机采样的 tile 数")
    parser.add_argument("--score-thr", type=float, default=0.3,
                        help="Score threshold for prediction counting")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(42)

    print("=" * 70)
    print("  DynamicKernelDecoder — Kernel Diversity Diagnostic")
    print("=" * 70)

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", {})

    n_kernels = ckpt.get("n_kernels", ckpt_cfg.get("n_kernels", 16))
    normalize_proto = ckpt.get("normalize_proto", ckpt_cfg.get("normalize_proto", "none"))
    unfreeze_layers = ckpt.get("unfreeze_layers", ckpt_cfg.get("unfreeze_layers", 8))
    proto_source = ckpt.get("prototype_source", ckpt_cfg.get("prototype_source", "p8"))

    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  n_kernels={n_kernels}, normalize_proto={normalize_proto}")
    print(f"  proto_source={proto_source}, unfreeze_layers={unfreeze_layers}")

    # ── Load Backbone ──
    backbone = FastSAMBackbone(device=device)
    if unfreeze_layers > 0:
        backbone.set_trainable_layers(unfreeze_layers)
    backbone_state = ckpt.get("backbone_state", {})
    if backbone_state:
        backbone.load_state_dict(backbone_state, strict=False)
        print(f"  backbone = Restored {unfreeze_layers} backbone layers")

    # ── Decoder ──
    test_img = torch.zeros(1, 3, 896, 896).to(device)
    with torch.no_grad():
        test_feats, _ = backbone(test_img)
    p3_channels = test_feats["p3"].shape[1]
    p4_channels = test_feats["p4"].shape[1]
    print(f"  Detected P3 channels={p3_channels}, P4 channels={p4_channels}")

    decoder = DynamicKernelDecoder(
        p3_channels=p3_channels, p4_channels=p4_channels,
        n_kernels=n_kernels, normalize_proto=normalize_proto,
    ).to(device)

    decoder_state = ckpt.get("decoder_state", {})
    if decoder_state:
        decoder.load_state_dict(decoder_state, strict=False)
        decoder_epoch = ckpt.get("epoch", "?")
        print(f"  decoder = Loaded (epoch {decoder_epoch})")
    decoder.eval()

    # ── Load prototypes from checkpoint ──
    class_protos = ckpt.get("class_prototypes", {})
    if not class_protos:
        print("  [ERROR] No class_prototypes in checkpoint! Aborting.")
        sys.exit(1)

    class_names = ckpt.get("class_names", {})
    print(f"  class_protos = {len(class_protos)} classes loaded")
    for cid in sorted(int(k) for k in class_protos.keys()):
        name = class_names.get(str(cid), f"class_{cid}")
        proto_vec = class_protos[str(cid)]["proto"]
        proto_norm = float(np.linalg.norm(proto_vec))
        print(f"    Class {cid:2d} ({name:20s}): proto L2 norm = {proto_norm:.2f}")

    # ── Load dataset and sample tiles ──
    manifest_path = os.path.join(args.data_root, "evaluation_manifest_val.json")
    if not os.path.exists(manifest_path):
        print(f"  [ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    query_list = manifest.get("query_images", [])
    if not query_list:
        query_list = manifest.get("query_tiles", [])
    print(f"  manifest = {len(query_list)} query images/tiles")

    rng = np.random.RandomState(42)
    sampled = rng.choice(query_list, min(args.n_samples, len(query_list)), replace=False)

    # ── Statistics accumulators ──
    all_stats = defaultdict(list)

    for sample_idx, qinfo in enumerate(sampled):
        # qinfo can be dict or string
        if isinstance(qinfo, dict):
            tile_id = qinfo.get("tile_id", qinfo.get("file_name", str(qinfo)))
            image_path = qinfo.get("image_path", "")
        else:
            tile_id = str(qinfo)
            image_path = ""

        # Load image
        # Try to find the image file
        if image_path and os.path.exists(image_path):
            img_path = image_path
        else:
            # Search for tile file
            possible_paths = [
                os.path.join(args.data_root, "images", f"{tile_id}.png"),
                os.path.join(args.data_root, "images", f"{tile_id}.jpg"),
                os.path.join(args.data_root, "tiles", f"{tile_id}.png"),
                os.path.join(args.data_root, "tiles", f"{tile_id}.jpg"),
            ]
            img_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    img_path = p
                    break
            if img_path is None:
                print(f"  [SKIP] Sample {sample_idx}: cannot find image for {tile_id}")
                continue

        # Load image with FastSAM backbone
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        H, W = img_np.shape[:2]

        # Pad to 32x
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        import torchvision.transforms as T
        img_tensor = T.ToTensor()(img_np).unsqueeze(0).to(device)  # [1, 3, H, W]

        with torch.no_grad():
            feats, _ = backbone(img_tensor)

        print(f"\n{'─' * 70}")
        print(f"  Sample {sample_idx}: {tile_id} ({H}×{W})")
        print(f"{'─' * 70}")

        # ── For each class, run decoder and collect kernel stats ──
        for cls_id_str in sorted(class_protos.keys(), key=int):
            cls_id = int(cls_id_str)
            cls_name = class_names.get(cls_id_str, f"class_{cls_id}")
            pk = class_protos[cls_id_str]
            proto_vec = torch.from_numpy(pk["proto"]).float().to(device)

            with torch.no_grad():
                masks_s8, proto_mask = decoder(
                    feats["p3"], feats["p4"], feats["proto"], proto_vec,
                )

            # masks_s8: [N_kernels, H/8, W/8]
            N = masks_s8.shape[0]

            # Upsample to tile resolution
            masks_full = F.interpolate(
                masks_s8.unsqueeze(0), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(0)  # [N, H, W]

            # Per-kernel stats
            kernel_means = masks_full.view(N, -1).mean(dim=1).cpu().numpy()
            kernel_maxs = masks_full.view(N, -1).max(dim=1)[0].cpu().numpy()
            kernel_stds = masks_full.view(N, -1).std(dim=1).cpu().numpy()

            # Threshold and count
            binary_masks = masks_full > args.score_thr
            n_active_pixels = binary_masks.view(N, -1).sum(dim=1).cpu().numpy()
            n_active_kernels = (n_active_pixels > 0).sum()
            n_active_kernels_min16 = (n_active_pixels >= 16).sum()  # min_area filter

            # Merge (max across all kernels)
            merged = masks_full.max(dim=0)[0]  # [H, W]
            merged_mean = float(merged.mean().item())
            merged_max = float(merged.max().item())

            # Pairwise IoU
            iou_mat = compute_pairwise_iou(masks_full)
            iou_upper = iou_mat[np.triu_indices(N, k=1)]
            mean_pairwise_iou = float(np.mean(iou_upper)) if len(iou_upper) > 0 else 0.0

            # Kernel weight diversity
            # We can check if kernels produce different masks
            mask_diffs = []
            for i in range(min(N - 1, 15)):
                diff = (masks_full[i] - masks_full[i + 1]).abs().max().item()
                mask_diffs.append(diff)
            mean_diff = float(np.mean(mask_diffs)) if mask_diffs else 0.0

            all_stats["kernel_mean"].extend(kernel_means.tolist())
            all_stats["kernel_max"].extend(kernel_maxs.tolist())
            all_stats["n_active"].extend([n_active_kernels])
            all_stats["n_active_min16"].extend([n_active_kernels_min16])
            all_stats["merged_mean"].append(merged_mean)
            all_stats["pairwise_iou"].append(mean_pairwise_iou)
            all_stats["mean_diff"].append(mean_diff)

            # Print per-class (only if interesting: active kernels > 0 or first few samples)
            if sample_idx < 2 or n_active_kernels > 0:
                print(f"  Class {cls_id:2d} ({cls_name:20s}): "
                      f"active={n_active_kernels}/{N} (≥16px={n_active_kernels_min16}), "
                      f"merged_mean={merged_mean:.4f}, PW-IoU={mean_pairwise_iou:.3f}, "
                      f"mean_diff={mean_diff:.3f}, "
                      f"kernel_mean=[{kernel_means.min():.4f}, {kernel_means.max():.4f}]")

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print(f"  SUMMARY (across {len(sampled)} tiles × {len(class_protos)} classes)")
    print(f"{'═' * 70}")

    kernel_means = np.array(all_stats["kernel_mean"])
    kernel_maxs = np.array(all_stats["kernel_max"])
    n_active = np.array(all_stats["n_active"])
    n_active_min16 = np.array(all_stats["n_active_min16"])
    merged_means = np.array(all_stats["merged_mean"])
    pairwise_ious = np.array(all_stats["pairwise_iou"])
    mean_diffs = np.array(all_stats["mean_diff"])

    print(f"  Kernel means:        min={kernel_means.min():.4f}, median={np.median(kernel_means):.4f}, max={kernel_means.max():.4f}")
    print(f"  Kernel maxes:        min={kernel_maxs.min():.4f}, median={np.median(kernel_maxs):.4f}, max={kernel_maxs.max():.4f}")
    print(f"  Active kernels:      min={n_active.min()}, median={np.median(n_active):.0f}, max={n_active.max()} (out of {n_kernels})")
    print(f"  Active (≥16px):      min={n_active_min16.min()}, median={np.median(n_active_min16):.0f}, max={n_active_min16.max()}")
    print(f"  Merged mean:         median={np.median(merged_means):.4f} (FG coverage)")
    print(f"  Pairwise IoU:        median={np.median(pairwise_ious):.4f} (0=diverse, 1=identical)")
    print(f"  Mean kernel diff:    median={np.median(mean_diffs):.4f} (0=identical kernels)")

    # Diagnostic verdict
    print(f"\n  ── Diagnostic Verdict ──")
    if np.median(pairwise_ious) > 0.5:
        print(f"  ⚠️  HIGH REDUNDANCY: Median pairwise IoU = {np.median(pairwise_ious):.3f}")
        print(f"      Kernels are producing near-identical masks → NOT separating instances")
    else:
        print(f"  ✅ Kernel masks are spatially diverse (PW-IoU={np.median(pairwise_ious):.3f})")

    if np.median(n_active) < 2:
        print(f"  ⚠️  MOST KERNELS DEAD: Median active = {np.median(n_active):.0f}/{n_kernels}")
        print(f"      Most kernels produce all-zero output → gradient starvation pattern")
    else:
        print(f"  ✅ {np.median(n_active):.0f}/{n_kernels} kernels active per class")

    if np.median(merged_means) < 0.05:
        print(f"  ⚠️  LOW FG COVERAGE: Merged mean = {np.median(merged_means):.4f}")
        print(f"      Even max(kernels) has almost no foreground")
    else:
        print(f"  ✅ Merged FG coverage: mean={np.median(merged_means):.4f}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
