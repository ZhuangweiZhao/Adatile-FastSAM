"""
Decoder Prediction Visualizer | Decoder 预测可视化工具.
========================================================

对指定图片/目录运行模型推理，可视化 decoder 的多通道输出，
帮助诊断实例分离失败的具体位置。

Supports: adaptive, adaptive-p3p4, dynamic_kernel, center_affinity

用法 | Usage:
    # 单张图片
    python tools/diag/diag_pred_vis.py \
        --checkpoint runs/.../best_model.pt \
        --decoder center_affinity \
        --image data/iSAID_instance_fewshot/images/P0089.png \
        --output vis_output/ --device cuda

    # 整个目录 (自动取前 N 张)
    python tools/diag/diag_pred_vis.py \
        --checkpoint runs/.../best_model.pt \
        --decoder center_affinity \
        --image-dir data/iSAID_instance_fewshot/images/ \
        --n-images 10 \
        --output vis_output/ --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from adatile.utils.seed import set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _load_decoder(decoder_type: str, ckpt: dict, p3_channels: int, p4_channels: int, device):
    """Load decoder from checkpoint based on type."""
    normalize_proto = ckpt.get("normalize_proto", "none")

    if decoder_type == "adaptive":
        from adatile.decoder.adaptive_sparse_decoder import AdaptiveSparseDecoder
        d = AdaptiveSparseDecoder(in_channels=p4_channels, use_fdr=False,
                                  normalize_proto=normalize_proto).to(device)
    elif decoder_type == "adaptive-p3p4":
        from adatile.decoder.adaptive_decoder_p3p4 import AdaptiveDecoderP3P4
        d = AdaptiveDecoderP3P4(p3_channels=p3_channels, p4_channels=p4_channels,
                                proto_dim=32, hidden_dim=256).to(device)
    elif decoder_type == "dynamic_kernel":
        from adatile.decoder.dynamic_kernel_decoder import DynamicKernelDecoder
        n_kernels = ckpt.get("n_kernels", 16)
        d = DynamicKernelDecoder(
            p3_channels=p3_channels, p4_channels=p4_channels,
            proto_dim=32, n_kernels=n_kernels, kernel_dim=256, fpn_dim=256,
            normalize_proto=normalize_proto,
        ).to(device)
    elif decoder_type == "center_affinity":
        from adatile.decoder.center_affinity_decoder import CenterAffinityDecoder
        d = CenterAffinityDecoder(
            p3_channels=p3_channels, p4_channels=p4_channels,
            proto_dim=32, fpn_dim=64,
            normalize_proto=normalize_proto,
        ).to(device)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

    d.load_state_dict(ckpt["decoder"], strict=False)
    d.eval()
    print(f"  Loaded {decoder_type} decoder (epoch {ckpt.get('epoch', '?')})")
    return d


def pad_to_32(h: int, w: int) -> tuple[int, int]:
    """Pad dimensions to multiples of 32."""
    return (32 - h % 32) % 32, (32 - w % 32) % 32


def run_inference(image_np: np.ndarray, model, decoder, decoder_type: str,
                  class_protos: dict, device, class_names: dict):
    """
    Run inference on a single image. Returns dict of results keyed by class_id.

    For center_affinity: returns center_hm, offset_field, proto_mask, instances, raw_output
    For others: returns prob_map, instances
    """
    H, W = image_np.shape[:2]
    pad_h, pad_w = pad_to_32(H, W)

    # Pad and convert to tensor
    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        img_padded = image_np

    from tools.train.train_fewshot_allclass import extract_features

    with torch.no_grad():
        feats = extract_features(model, [img_padded], device)[0]

    results = {}

    if decoder_type == "center_affinity":
        # ── Center-Affinity specific ──
        # Generate center+offset ONCE (class-agnostic)
        first_pk = list(class_protos.values())[0]
        first_proto = torch.from_numpy(first_pk["proto"]).float().to(device)

        with torch.no_grad():
            center_hm, offset_field, _ = decoder(
                feats["p3"], feats["p4"], feats["proto"], first_proto,
            )

        # Upsample center+offset to tile resolution
        center_full = F.interpolate(
            center_hm.unsqueeze(0).unsqueeze(0),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()
        offset_full = F.interpolate(
            offset_field.unsqueeze(0),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(0).cpu().numpy()

        # Per-class proto + grouping
        for cls_id_str, pk in class_protos.items():
            cls_id = int(cls_id_str)
            proto_vec = torch.from_numpy(pk["proto"]).float().to(device)

            with torch.no_grad():
                proto_mask = decoder.forward_proto_only(feats["proto"], proto_vec)

            proto_full = F.interpolate(
                proto_mask.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze().cpu().numpy()

            fg_mask = proto_full > 0.3

            from adatile.metrics.instance_generation import generate_instances_center_affinity
            instances = generate_instances_center_affinity(
                center_full, offset_full, fg_mask,
                score_thr=0.2, min_area=16, min_distance=8, max_instances=100,
            )

            results[cls_id] = {
                "center": center_full,
                "offset": offset_full,
                "proto_mask": proto_full,
                "instances": instances,
                "fg_mask": fg_mask,
            }
    else:
        # ── Other decoders ──
        for cls_id_str, pk in class_protos.items():
            cls_id = int(cls_id_str)
            proto_vec = torch.from_numpy(pk["proto"]).float().to(device)

            with torch.no_grad():
                if decoder_type in ("adaptive", "adaptive-p3p4"):
                    if hasattr(decoder, "forward_proto_only"):
                        prob = decoder.forward_proto_only(feats["proto"], proto_vec)
                    else:
                        if decoder_type == "adaptive":
                            prob = decoder(feats["p4"], feats["proto"], proto_vec)
                        else:
                            prob = decoder(feats["p3"], feats["p4"], feats["proto"], proto_vec)
                elif decoder_type == "dynamic_kernel":
                    masks_s8, proto_mask = decoder(
                        feats["p3"], feats["p4"], feats["proto"], proto_vec,
                    )
                    prob = proto_mask  # Use proto branch for semantic
                else:
                    continue

            # Upsample to tile resolution
            if prob.dim() == 2:
                prob = prob.unsqueeze(0).unsqueeze(0)
            elif prob.dim() == 3:
                prob = prob.unsqueeze(0)
            prob_full = F.interpolate(
                prob, size=(H, W), mode="bilinear", align_corners=False,
            ).squeeze().cpu().numpy()

            # Generate instances via CC
            from adatile.metrics.instance_generation import generate_instances
            instances = generate_instances(
                prob_full, method="connected_components",
                score_thr=0.3, min_area=16, max_instances=100,
            )

            results[cls_id] = {
                "prob_map": prob_full,
                "instances": instances,
            }

    return results


def plot_center_affinity(image: np.ndarray, class_name: str, result: dict, out_path: str):
    """Plot center-affinity decoder outputs."""
    center = result["center"]
    offset = result["offset"]
    proto = result["proto_mask"]
    instances = result["instances"]
    fg_mask = result["fg_mask"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"Center-Affinity Decoder — {class_name}", fontsize=14, fontweight="bold")

    # (0,0): Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    # (0,1): Proto mask (FG prior)
    axes[0, 1].imshow(proto, cmap="hot", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Proto Mask (class-conditioned)\nmean={proto.mean():.3f}")
    axes[0, 1].axis("off")

    # (0,2): Center heatmap
    axes[0, 2].imshow(center, cmap="hot", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Center Heatmap\nmax={center.max():.3f}, peaks={len(instances)}")
    axes[0, 2].axis("off")

    # (1,0): Offset field (quiver)
    H, W = center.shape
    step = max(H // 40, 1)  # subsample for visibility
    y, x = np.mgrid[step//2:H:step, step//2:W:step]
    dx = offset[0, y, x]
    dy = offset[1, y, x]

    axes[1, 0].imshow(image)
    # Only show arrows on FG pixels
    fg_y, fg_x = y, x
    mask = fg_mask[y, x]
    axes[1, 0].quiver(fg_x[mask], fg_y[mask], dx[mask], dy[mask],
                      color="cyan", alpha=0.7, scale=50, width=0.003)
    axes[1, 0].set_title(f"Offset Field (FG pixels only)\nstep={step}")
    axes[1, 0].axis("off")

    # (1,1): FG mask overlay with detected centers
    axes[1, 1].imshow(image)
    # Show FG mask as green overlay
    fg_overlay = np.zeros((H, W, 4))
    fg_overlay[fg_mask, 1] = 0.5  # green
    fg_overlay[fg_mask, 3] = 0.3  # alpha
    axes[1, 1].imshow(fg_overlay)

    # Mark detected centers
    from scipy.ndimage import maximum_filter
    local_max = (center == maximum_filter(center, size=17))  # 8*2+1
    valid_peaks = local_max & (center > 0.2) & fg_mask
    py, px = np.where(valid_peaks)
    axes[1, 1].scatter(px, py, c="red", s=30, marker="x", linewidths=1.5)
    axes[1, 1].set_title(f"FG Mask + Centers\n{len(py)} peaks detected")
    axes[1, 1].axis("off")

    # (1,2): Instance masks
    axes[1, 2].imshow(image)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(instances), 1)))
    for i, inst in enumerate(instances[:20]):
        mask = inst["mask"]
        # Overlay with color
        color_mask = np.zeros((H, W, 4))
        color_mask[mask, :3] = colors[i][:3]
        color_mask[mask, 3] = 0.4
        axes[1, 2].imshow(color_mask)
        # Center of mass
        yy, xx = np.where(mask)
        if len(yy) > 0:
            axes[1, 2].text(xx.mean(), yy.mean(), str(i + 1),
                            fontsize=6, color="white", weight="bold",
                            ha="center", va="center",
                            bbox=dict(boxstyle="round", fc="black", alpha=0.6))

    axes[1, 2].set_title(f"Instance Masks ({len(instances)} detected)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_semantic(image: np.ndarray, class_name: str, result: dict, decoder_type: str, out_path: str):
    """Plot semantic/prob-map based decoder outputs."""
    prob_map = result["prob_map"]
    instances = result["instances"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{decoder_type} Decoder — {class_name}", fontsize=14, fontweight="bold")

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Prob map
    axes[1].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"Probability Map\nmean={prob_map.mean():.3f}")
    axes[1].axis("off")

    # Instances
    axes[2].imshow(image)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(instances), 1)))
    for i, inst in enumerate(instances[:20]):
        mask = inst["mask"]
        color_mask = np.zeros((*mask.shape, 4))
        color_mask[mask, :3] = colors[i][:3]
        color_mask[mask, 3] = 0.4
        axes[2].imshow(color_mask)
    axes[2].set_title(f"Instances ({len(instances)} via CC)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_dynamic_kernel(image: np.ndarray, class_name: str, result: dict, out_path: str):
    """Plot dynamic_kernel decoder outputs."""
    proto_mask = result.get("proto_mask", None)
    kernel_masks = result.get("kernel_masks", None)
    instances = result["instances"]

    n_kernels = len(kernel_masks) if kernel_masks is not None else 0
    n_cols = min(n_kernels + 2, 8)
    n_rows = (n_kernels + 2 + n_cols - 1) // n_cols
    n_rows = max(n_rows, 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f"DynamicKernel Decoder — {class_name}", fontsize=14, fontweight="bold")

    # Input image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    # Proto mask
    if proto_mask is not None:
        pm = proto_mask.squeeze().cpu().numpy() if hasattr(proto_mask, "cpu") else proto_mask
        axes[0, 1].imshow(pm, cmap="hot", vmin=0, vmax=1)
        axes[0, 1].set_title(f"Proto Mask\nmean={pm.mean():.3f}")
    axes[0, 1].axis("off")

    # Individual kernel masks
    for ki in range(min(n_kernels, n_rows * n_cols - 2)):
        r, c = divmod(ki + 2, n_cols)
        if r < n_rows and c < n_cols:
            km = kernel_masks[ki]
            km_np = km.squeeze().cpu().numpy() if hasattr(km, "cpu") else km
            axes[r, c].imshow(km_np, cmap="hot", vmin=0, vmax=1)
            axes[r, c].set_title(f"Kernel {ki}\nmax={km_np.max():.3f}")
            axes[r, c].axis("off")

    # Hide unused axes
    for idx in range(n_kernels + 2, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Decoder Prediction Visualizer")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--decoder", type=str, required=True,
                        choices=["baseline", "adaptive", "adaptive-p3p4", "dynamic_kernel", "center_affinity"])
    parser.add_argument("--image", type=str, default=None,
                        help="Single image path")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Image directory (process first N images)")
    parser.add_argument("--n-images", type=int, default=8,
                        help="Number of images from directory")
    parser.add_argument("--output", type=str, default="vis_output",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-root", type=str, default="data/iSAID_instance_fewshot",
                        help="Data root for COCO annotations")
    args = parser.parse_args()

    assert args.image or args.image_dir, "Must specify --image or --image-dir"

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(42)

    print("=" * 70)
    print(f"  Decoder Visualization — {args.decoder}")
    print("=" * 70)

    # ── Load checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_cfg = ckpt.get("config", {})
    unfreeze_layers = ckpt.get("unfreeze_layers", ckpt_cfg.get("unfreeze_layers", 8))
    print(f"  Checkpoint: epoch {ckpt.get('epoch', '?')}")

    # ── Load Backbone (same pattern as evaluate_instance.py) ──
    from ultralytics import FastSAM
    from tools.eval.evaluate_instance import _fastsam_weights_path
    from tools.train.train_fewshot_allclass import extract_features

    fastsam_path = _fastsam_weights_path()
    model = FastSAM(str(fastsam_path))
    model.model.to(device).eval()
    for p in model.model.parameters():
        p.requires_grad = False

    # Restore unfrozen backbone layers from checkpoint
    unfreeze_layers = ckpt.get("unfreeze_layers", 0)
    if unfreeze_layers > 0 and "backbone" in ckpt:
        seq = model.model.model  # Sequential[23]
        for i_str, state in ckpt["backbone"].items():
            seq[int(i_str)].load_state_dict(state)
        print(f"  Restored {unfreeze_layers} backbone layers from checkpoint")
    elif unfreeze_layers > 0:
        print(f"  [WARN] unfreeze_layers={unfreeze_layers} but no backbone weights in ckpt")

    # ── Detect channels ──
    test_img = np.zeros((896, 896, 3), dtype=np.uint8)
    test_feats = extract_features(model, [test_img], device)
    p3_channels = test_feats[0]["p3"].shape[1]
    p4_channels = test_feats[0]["p4"].shape[1]
    print(f"  P3 channels={p3_channels}, P4 channels={p4_channels}")

    # ── Load Decoder ──
    decoder = _load_decoder(args.decoder, ckpt, p3_channels, p4_channels, device)

    # ── Load class prototypes ──
    class_protos = ckpt.get("class_prototypes", {})
    class_names = ckpt.get("class_names", {})
    if not class_protos:
        print("  [ERROR] No class_prototypes in checkpoint!")
        sys.exit(1)
    print(f"  Loaded {len(class_protos)} class prototypes")

    # ── Get image list ──
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = sorted([
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ])[:args.n_images]

    print(f"  Processing {len(image_paths)} image(s)")

    # ── Per-image inference + visualization ──
    for img_idx, img_path in enumerate(image_paths):
        if not os.path.exists(img_path):
            print(f"  [SKIP] File not found: {img_path}")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        H, W = image.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n  [{img_idx + 1}/{len(image_paths)}] {stem} ({H}×{W})")

        results = run_inference(image, model, decoder, args.decoder,
                                class_protos, device, class_names)

        # ── Visualize per class ──
        # Filter to classes that have many tiles (likely to have objects)
        # Select top 5 classes by proto norm (more distinctive prototypes)
        proto_norms = []
        for cls_id_str, pk in class_protos.items():
            proto_norms.append((cls_id_str, float(np.linalg.norm(pk["proto"]))))
        proto_norms.sort(key=lambda x: -x[1])
        top_classes = [int(c) for c, _ in proto_norms[:5]]

        for cls_id in top_classes:
            cls_id_str = str(cls_id)
            if cls_id not in results:
                continue
            cls_name = class_names.get(cls_id_str, f"class_{cls_id}")
            result = results[cls_id]

            safe_name = cls_name.replace(" ", "_").replace("/", "_")
            out_path = os.path.join(args.output, f"{stem}_{safe_name}.png")

            if args.decoder == "center_affinity":
                plot_center_affinity(image, cls_name, result, out_path)
            elif args.decoder == "dynamic_kernel":
                plot_dynamic_kernel(image, cls_name, result, out_path)
            else:
                plot_semantic(image, cls_name, result, args.decoder, out_path)

            # Only visualize first 3 classes per image to avoid too many files
            if len(top_classes) > 3 and cls_id == top_classes[2]:
                break

    print(f"\n  Done. Output in: {args.output}/")
    print(f"  Total images: {len(image_paths)}")


if __name__ == "__main__":
    main()
