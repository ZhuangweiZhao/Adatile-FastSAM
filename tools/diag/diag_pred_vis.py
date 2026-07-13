"""
Decoder Prediction Visualizer | Decoder 预测可视化工具 (v2 — 复用 eval 基础设施).

用法 | Usage:
    python tools/diag/diag_pred_vis.py \
        --checkpoint runs/.../best_model.pt \
        --decoder dynamic_kernel \
        --image data/iSAID_instance_fewshot/images/P0089.png \
        --output vis_output/ --device cuda
"""

from __future__ import annotations

import argparse, json, os, sys, random
from collections import defaultdict

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


# ═══════════════════════════════════════════════════════════════════
# Lightweight prototype builder (no dependency on eval module internals)
# ═══════════════════════════════════════════════════════════════════

def _fastsam_weights_path():
    """Locate FastSAM-x.pt."""
    paths = [
        os.path.join(_PROJECT_ROOT, "weights", "FastSAM-x.pt"),
        os.path.join(_PROJECT_ROOT, "FastSAM-x.pt"),
        os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints", "FastSAM-x.pt"),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("FastSAM-x.pt not found. Download from https://github.com/CASIA-IVA-Lab/FastSAM")


def _load_model(ckpt_path: str, unfreeze_layers: int, decoder_type: str, device: str):
    """Load FastSAM + decoder from checkpoint. Returns (model, decoder, ckpt_meta)."""
    from ultralytics import FastSAM
    from tools.train.train_fewshot_allclass import extract_features

    model = FastSAM(str(_fastsam_weights_path()))
    model.model.to(device).eval()
    for p in model.model.parameters():
        p.requires_grad = False

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Restore backbone
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

    return model, decoder, extract_features


def _build_prototypes_from_checkpoint(model, decoder_type, ckpt_path: str, device: str):
    """
    Build per-class prototypes by re-running the support-set forward pass.
    Uses the same support sources stored in the checkpoint's fixed_val_episodes.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Check if prototypes were cached
    if "class_prototypes" in ckpt and ckpt["class_prototypes"]:
        print(f"  Loaded {len(ckpt['class_prototypes'])} cached prototypes")
        return ckpt["class_prototypes"], ckpt.get("class_names", {})

    # Try loading from fixed_val_episodes
    run_dir = os.path.dirname(ckpt_path)
    val_eps_path = os.path.join(run_dir, "fixed_val_episodes.json")
    if not os.path.exists(val_eps_path):
        # Try the run dir at the default location
        alt_paths = [os.path.join(os.path.dirname(ckpt_path), "fixed_val_episodes.json")]
        for p in alt_paths:
            if os.path.exists(p):
                val_eps_path = p
                break
        else:
            print("  [ERROR] No class_prototypes in ckpt and no fixed_val_episodes.json found")
            return {}, {}

    with open(val_eps_path) as f:
        val_eps = json.load(f)

    from tools.train.train_fewshot_allclass import extract_features, compute_support_prototype

    ckpt_cfg = ckpt.get("config", {})
    # Determine data root from run config or default
    data_root = ckpt_cfg.get("data_root", "data/iSAID_instance_fewshot")
    data_format = ckpt_cfg.get("data_format", "isaid_instance")
    split = "train"
    proto_source = ckpt_cfg.get("prototype_source", "p8")

    # Load tile helper
    if data_format in ("isaid_instance", "TILES COCO (new)"):
        from adatile.datasets.isaid_instance_fewshot import ISAIDInstanceFewShotDataset
        is_instance = True
    else:
        from adatile.datasets.isaid_tiles import FastISAIDTileDataset
        is_instance = False

    class_index = defaultdict(lambda: defaultdict(list))
    for ep in val_eps:
        cls_id = ep["class_id"]
        for stem in ep.get("support_tiles", ep.get("support_stems", [])):
            src = ep.get("support_sources", [None])[0] if "support_sources" in ep else "unknown"
            class_index[cls_id][src].append(stem)

    class_protos = {}
    class_names = ckpt.get("class_names", {})

    for cls_id, src_to_tiles in class_index.items():
        if not src_to_tiles:
            continue
        # Take K sources
        k_shot = ckpt_cfg.get("k_shot", 1)
        sources = list(src_to_tiles.keys())[:k_shot]
        support_stems = []
        for s in sources:
            support_stems.extend(src_to_tiles[s])

        support_imgs, support_masks = [], []
        for stem in support_stems[:50]:  # limit for speed
            try:
                if is_instance:
                    from tools.eval.evaluate_instance import _load_tile_img_mask
                    img, m = _load_tile_img_mask(stem, split, data_root, is_instance=True,
                                                  target_class_id=cls_id)
                else:
                    from tools.eval.evaluate_instance import _load_tile_img_mask
                    img, m = _load_tile_img_mask(stem, split, data_root, is_instance=False,
                                                  target_class_id=cls_id)
                support_imgs.append(img)
                support_masks.append(m)
            except Exception as e:
                continue

        if not support_imgs:
            continue

        support_feats = extract_features(model, support_imgs, device)
        proto = compute_support_prototype(support_feats, source=proto_source, masks=support_masks)
        class_protos[cls_id] = {"proto": proto.cpu().numpy()}
        cls_name = class_names.get(str(cls_id), f"class_{cls_id}")
        print(f"  Class {cls_id:>2d} ({cls_name:<20s}): {len(support_imgs)} support tiles")

    print(f"  Built {len(class_protos)} class prototypes")
    return class_protos, class_names


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_dynamic_kernel(image: np.ndarray, class_name: str, masks_s8, proto_mask, out_path: str):
    """Visualize DynamicKernelDecoder outputs: 16 kernel masks + proto."""
    H, W = image.shape[:2]
    N = masks_s8.shape[0]

    # Upsample all masks to image resolution
    masks_full = F.interpolate(
        masks_s8.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze(0).cpu().numpy()  # [N, H, W]

    proto_full = F.interpolate(
        proto_mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False,
    ).squeeze().cpu().numpy()  # [H, W]

    n_cols = 6
    n_rows = (N + 3 + n_cols - 1) // n_cols  # +3 for: input, proto, max-pool
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
    axes = axes.reshape(n_rows, n_cols)

    # Input
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image", fontsize=8)
    axes[0, 0].axis("off")

    # Proto mask
    axes[0, 1].imshow(proto_full, cmap="hot", vmin=0, vmax=1)
    axes[0, 1].set_title(f"Proto Mask\nmean={proto_full.mean():.3f}", fontsize=8)
    axes[0, 1].axis("off")

    # Max-pool of all kernels
    kernel_max = masks_full.max(axis=0)
    axes[0, 2].imshow(kernel_max, cmap="hot", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Max(kernels)\nmax={kernel_max.max():.3f}", fontsize=8)
    axes[0, 2].axis("off")

    # Binary threshold of max (what CC would see)
    binary_max = kernel_max > 0.3
    axes[0, 3].imshow(binary_max, cmap="gray")
    axes[0, 3].set_title(f"Max>0.3\narea={binary_max.sum()}", fontsize=8)
    axes[0, 3].axis("off")

    # Per-kernel IoU with previous (diversity check)
    pw_ious = []
    for i in range(1, N):
        bi = masks_full[i] > 0.5
        bj = masks_full[0] > 0.5
        inter = (bi & bj).sum()
        union = (bi | bj).sum()
        pw_ious.append(inter / max(union, 1))
    axes[0, 4].bar(range(1, N), pw_ious)
    axes[0, 4].set_title(f"Pairwise IoU vs K0\nmean={np.mean(pw_ious):.3f}", fontsize=8)
    axes[0, 4].set_xlabel("Kernel idx")
    axes[0, 4].set_ylabel("IoU")

    # Empty
    axes[0, 5].axis("off")

    # Individual kernel masks
    for ki in range(N):
        r, c = divmod(ki + 6, n_cols)  # start from row 1
        if r < n_rows and c < n_cols:
            km = masks_full[ki]
            axes[r, c].imshow(km, cmap="hot", vmin=0, vmax=1)
            bin_km = km > 0.3
            n_px = bin_km.sum()
            axes[r, c].set_title(f"K{ki}: max={km.max():.3f} px={n_px}", fontsize=7)
            axes[r, c].axis("off")

    # Hide remaining
    for idx in range(6 + N, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


def plot_semantic(image: np.ndarray, class_name: str, prob_map: np.ndarray, out_path: str):
    """Visualize semantic decoder output."""
    H, W = image.shape[:2]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"Semantic Decoder — {class_name}", fontsize=14, fontweight="bold")

    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title(f"Prob Map\nmean={prob_map.mean():.3f}")
    axes[1].axis("off")

    binary = prob_map > 0.3
    axes[2].imshow(binary, cmap="gray")
    axes[2].set_title(f"Binary >0.3\narea={binary.sum()}")
    axes[2].axis("off")

    # Contours
    import cv2
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    axes[3].imshow(overlay)
    axes[3].set_title(f"CC Contours ({len(contours)} blobs)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Decoder Prediction Visualizer v2")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--decoder", type=str, required=True,
                        choices=["adaptive", "adaptive-p3p4", "dynamic_kernel", "center_affinity"])
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="vis_output")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-thr", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(42)

    print("=" * 70)
    print(f"  Decoder Visualization v2 — {args.decoder}")
    print("=" * 70)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    unfreeze_layers = ckpt.get("unfreeze_layers", ckpt.get("config", {}).get("unfreeze_layers", 8))

    # Load model + decoder
    model, decoder, extract_features = _load_model(
        args.checkpoint, unfreeze_layers, args.decoder, device,
    )

    # Build / load prototypes
    class_protos = ckpt.get("class_prototypes", {})
    class_names = ckpt.get("class_names", {})

    if not class_protos:
        # Try building from fixed_val_episodes
        class_protos, class_names = _build_prototypes_from_checkpoint(
            model, args.decoder, args.checkpoint, device,
        )

    if not class_protos:
        # Fallback: use random prototype (still useful for kernel diversity check)
        print("  [WARN] No prototypes available — using random (masks will be random, but")
        print("         kernel diversity pattern is still visible)")
        # Detect proto dim from decoder
        proto_dim = 640  # P4 channels
        # Create 15 random class prototypes
        class_protos = {}
        class_names = {}
        for cid in range(1, 16):
            class_protos[str(cid)] = {"proto": np.random.randn(proto_dim).astype(np.float32)}
            class_names[str(cid)] = f"class_{cid}"

    # Load input image
    if not os.path.exists(args.image):
        print(f"  [FATAL] Image not found: {args.image}")
        sys.exit(1)
    image_np = np.array(Image.open(args.image).convert("RGB"))
    H, W = image_np.shape[:2]
    stem = os.path.splitext(os.path.basename(args.image))[0]
    print(f"\n  Image: {stem} ({H}×{W})")

    # Pad + extract features
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(image_np, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        img_padded = image_np

    with torch.no_grad():
        feats = extract_features(model, [img_padded], device)[0]

    # ── Per-class inference + visualization ──
    # Sort classes by proto norm (prioritize distinctive prototypes)
    proto_norms = []
    for cid, pk in class_protos.items():
        proto_norms.append((int(cid), float(np.linalg.norm(pk["proto"]))))
    proto_norms.sort(key=lambda x: -x[1])

    n_plotted = 0
    for cls_id, _ in proto_norms[:8]:  # top 8 classes
        cls_id_str = str(cls_id)
        if cls_id_str not in class_protos:
            continue
        pk = class_protos[cls_id_str]
        cls_name = class_names.get(cls_id_str, f"class_{cls_id}")
        safe_name = cls_name.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(args.output, f"{stem}_{cls_id:02d}_{safe_name}.png")

        proto_vec = torch.from_numpy(pk["proto"]).float().to(device)

        with torch.no_grad():
            if args.decoder == "dynamic_kernel":
                masks_s8, proto_mask = decoder(
                    feats["p3"], feats["p4"], feats["proto"], proto_vec,
                )
                plot_dynamic_kernel(image_np, cls_name, masks_s8, proto_mask, out_path)
            elif args.decoder == "center_affinity":
                center_hm, offset_field, proto_mask = decoder(
                    feats["p3"], feats["p4"], feats["proto"], proto_vec,
                )
                # TODO: full center_affinity plot (needs grouping)
                # For now, just show proto mask
                proto_full = F.interpolate(
                    proto_mask.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy()
                plot_semantic(image_np, cls_name, proto_full, out_path)
            else:
                # adaptive / adaptive-p3p4
                if args.decoder == "adaptive":
                    prob = decoder(feats["p4"], feats["proto"], proto_vec)
                else:
                    prob = decoder(feats["p3"], feats["p4"], feats["proto"], proto_vec)
                if prob.dim() == 2:
                    prob = prob.unsqueeze(0).unsqueeze(0)
                elif prob.dim() == 3:
                    prob = prob.unsqueeze(0)
                prob_full = F.interpolate(
                    prob, size=(H, W), mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy()
                plot_semantic(image_np, cls_name, prob_full, out_path)

        n_plotted += 1

    print(f"\n  Done. {n_plotted} visualizations → {args.output}/")


if __name__ == "__main__":
    main()
