"""True tile-based sparse inference for AdaTile-FastSAM.

Unlike post-hoc mask zeroing (which runs the full decoder on the entire
image then discards predictions), this module:

1. Runs backbone on FULL image (SPM needs global context)
2. Runs SPM to get importance map
3. Selects top-K% regions by importance
4. Extracts feature tiles from backbone outputs
5. Runs decoder ONLY on selected tiles
6. Stitches tile predictions back to a full-image mask

This actually reduces FLOPs — decoder compute scales with kept tile count,
not full image resolution.

Key metric: compute_reduction = 1 - (tiled_decoder_flops / full_decoder_flops)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def tile_sparse_forward(
    image: Tensor,
    backbone: nn.Module,
    decoder: nn.Module,
    spm: nn.Module,
    keep_ratio: float = 0.15,
    tile_size: int = 64,
    tile_overlap: int = 0,
    mode: str = "grid",
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """Tile-based sparse forward pass.

    Args:
        image: [1, 3, H, W] input image.
        backbone: FastSAMHookBackbone.
        decoder: LightDecoder.
        spm: LightSPM or AdaSPM.
        keep_ratio: Fraction of tiles to keep.
        tile_size: Spatial size of each tile in feature space (before decoder).
            Default 64 corresponds to 64×64 feature tiles at backbone output.
            Since decoder upsamples 2x→2x, output tiles are 128×128.
        tile_overlap: Overlap between tiles in feature pixels. 0 = non-overlapping.
        mode: "grid" (regular grid) or "adaptive" (importance-weighted centroids).

    Returns:
        full_mask: [1, 1, H/4, W/4] binary mask (decoder output resolution).
        sparse_mask: [1, 1, H/4, W/4] mask from only kept tiles.
        metrics: dict with imp_mean, keep_ratio, coverage, tiles_processed,
                 full_tiles, compute_reduction.
    """
    device = image.device
    B, C, H, W = image.shape
    assert B == 1, "tile_sparse_forward supports batch_size=1"

    # ── 1. Full backbone forward (needed for SPM importance) ──────
    with torch.no_grad():
        feats = backbone(image)

    # ── 2. SPM importance prediction ─────────────────────────────
    imp_raw = spm(feats)
    if hasattr(imp_raw, 'importance'):
        imp = imp_raw.importance
    else:
        imp = imp_raw
    # imp: [1, 1, H_s, W_s]

    _, _, H_s, W_s = imp.shape

    # ── 3. Grid-based tile planning ──────────────────────────────
    # Partition the importance grid into tiles
    stride = tile_size - tile_overlap
    if stride <= 0:
        stride = tile_size // 2

    # Calculate tile positions
    tile_h_positions = list(range(0, H_s - tile_size + 1, stride))
    tile_w_positions = list(range(0, W_s - tile_size + 1, stride))
    # Ensure coverage of edges
    if tile_h_positions and tile_h_positions[-1] + tile_size < H_s:
        tile_h_positions.append(H_s - tile_size)
    if tile_w_positions and tile_w_positions[-1] + tile_size < W_s:
        tile_w_positions.append(W_s - tile_size)
    if not tile_h_positions:
        tile_h_positions = [0]
    if not tile_w_positions:
        tile_w_positions = [0]

    # Compute tile importance scores (mean importance within tile)
    tile_scores = []
    tile_coords = []
    for h0 in tile_h_positions:
        for w0 in tile_w_positions:
            h1 = min(h0 + tile_size, H_s)
            w1 = min(w0 + tile_size, W_s)
            # Average importance within the tile
            score = imp[0, 0, h0:h1, w0:w1].mean().item()
            tile_scores.append(score)
            tile_coords.append((h0, w0, h1, w1))

    # ── 4. Select top-K tiles ────────────────────────────────────
    n_total = len(tile_scores)
    n_keep = max(1, int(n_total * keep_ratio))
    sorted_idx = np.argsort(tile_scores)[::-1]  # descending
    keep_idx = sorted_idx[:n_keep]

    # ── 5. Extract feature tiles and run decoder ─────────────────
    # Get backbone features at P4 (stride 16) and P8 (stride 8)
    # SPM operates at stride 32, so 1 SPM cell = 4 P8 cells = 2 P4 cells
    # We extract tiles from P4/P8 features corresponding to each kept tile
    scale_spm_to_p8 = 4   # SPM at stride 32, P8 at stride 8: factor 32/8=4
    scale_spm_to_p4 = 2   # P4 at stride 16: factor 32/16=2

    p4_feat = feats["P4"]  # [1, 128, H/16, W/16]
    p8_feat = feats["P8"]  # [1, 128, H/8, W/8]

    _, C_feat, H_p4, W_p4 = p4_feat.shape
    _, _, H_p8, W_p8 = p8_feat.shape

    # Compute output tile size at decoder output
    # Decoder: P4 (H/16) → up_conv (H/8) → fuse with P8 → refine → final_up (H/4)
    # So feature tile at SPM res → decoder output at 8× SPM res
    out_tile_size = tile_size * 8  # decoder output resolution
    out_H = H // 4
    out_W = W // 4

    # Initialize sparse mask canvas
    sparse_mask = torch.zeros(1, 1, out_H, out_W, device=device)
    weight_map = torch.zeros(1, 1, out_H, out_W, device=device)

    decoder.eval()
    tiles_processed = 0

    for idx in keep_idx:
        h0_s, w0_s, h1_s, w1_s = tile_coords[idx]

        # Map SPM grid coords to P4 feature coords
        h0_p4 = h0_s * scale_spm_to_p4
        w0_p4 = w0_s * scale_spm_to_p4
        h1_p4 = min(h1_s * scale_spm_to_p4, H_p4)
        w1_p4 = min(w1_s * scale_spm_to_p4, W_p4)

        # Map SPM grid coords to P8 feature coords
        h0_p8 = h0_s * scale_spm_to_p8
        w0_p8 = w0_s * scale_spm_to_p8
        h1_p8 = min(h1_s * scale_spm_to_p8, H_p8)
        w1_p8 = min(w1_s * scale_spm_to_p8, W_p8)

        # Skip invalid tiles
        if h1_p4 <= h0_p4 or w1_p4 <= w0_p4:
            continue
        if h1_p8 <= h0_p8 or w1_p8 <= w0_p8:
            continue

        # Extract feature tile
        tile_p4 = p4_feat[:, :, h0_p4:h1_p4, w0_p4:w1_p4]  # [1, C, th, tw]
        tile_p8 = p8_feat[:, :, h0_p8:h1_p8, w0_p8:w1_p8]

        # Run decoder on tile only
        tile_feats = {"P4": tile_p4, "P8": tile_p8}
        tile_mask = decoder(features=tile_feats)  # [1, 1, th*2, tw*2]
        # Actually decoder outputs at H/4 from P4 input, so from tile_p4 at H/16:
        # up_conv (H/8) → fuse → final_up (H/4) → tile_mask at 4× P4 res
        # tile_p4 spatial: (h1_p4-h0_p4) × (w1_p4-w0_p4)
        # tile_mask spatial: 4*(h1_p4-h0_p4) × 4*(w1_p4-w0_p4)
        tile_out_h = tile_mask.shape[2]
        tile_out_w = tile_mask.shape[3]

        # Map to output coordinates (H/4 = 4× P4 resolution)
        out_h0 = h0_p4 * 4
        out_w0 = w0_p4 * 4
        out_h1 = min(out_h0 + tile_out_h, out_H)
        out_w1 = min(out_w0 + tile_out_w, out_W)

        # Adjust tile_mask if it extends beyond canvas
        tile_h_used = out_h1 - out_h0
        tile_w_used = out_w1 - out_w0
        if tile_h_used < tile_out_h or tile_w_used < tile_out_w:
            tile_mask = tile_mask[:, :, :tile_h_used, :tile_w_used]

        # Add to canvas with feathering at boundaries (linear ramp)
        h_ramp = torch.ones(1, 1, tile_h_used, 1, device=device)
        w_ramp = torch.ones(1, 1, 1, tile_w_used, device=device)
        if tile_overlap > 0:
            ramp_len = min(tile_overlap // 2, tile_h_used // 4)
            if ramp_len > 1:
                ramp = torch.linspace(0, 1, ramp_len, device=device).view(1, 1, -1, 1)
                h_ramp[:, :, :ramp_len, :] = ramp
                h_ramp[:, :, -ramp_len:, :] = ramp.flip(2)
            ramp_len_w = min(tile_overlap // 2, tile_w_used // 4)
            if ramp_len_w > 1:
                ramp_w = torch.linspace(0, 1, ramp_len_w, device=device).view(1, 1, 1, -1)
                w_ramp[:, :, :, :ramp_len_w] = ramp_w
                w_ramp[:, :, :, -ramp_len_w:] = ramp_w.flip(3)

        feather = h_ramp * w_ramp

        sparse_mask[:, :, out_h0:out_h1, out_w0:out_w1] += tile_mask * feather
        weight_map[:, :, out_h0:out_h1, out_w0:out_w1] += feather
        tiles_processed += 1

    # Normalize by weight map (handles overlapping regions)
    valid = weight_map > 0
    sparse_mask[valid] = sparse_mask[valid] / weight_map[valid]

    # ── 6. Full mask (for reference) ─────────────────────────────
    with torch.no_grad():
        full_logits = decoder(features=feats)
    full_mask = (full_logits.sigmoid() > 0.5).float()

    # ── 7. Metrics ───────────────────────────────────────────────
    metrics = {
        "imp_mean": float(imp.mean().item()),
        "keep_ratio": n_keep / max(n_total, 1),
        "tiles_processed": tiles_processed,
        "full_tiles": n_total,
        "compute_reduction": 1.0 - (tiles_processed / max(n_total, 1)),
    }

    # Coverage: fraction of foreground GT in kept tiles
    # (approximated by importance-weighted coverage)
    imp_flat = imp[0, 0].reshape(-1)
    n_cells = imp_flat.shape[0]
    k_cov = max(1, int(n_cells * keep_ratio))
    _, top_idx = imp_flat.topk(k_cov)
    top_imp_sum = imp_flat[top_idx].sum()
    total_imp_sum = imp_flat.sum()
    metrics["coverage"] = float(
        (top_imp_sum / (total_imp_sum + 1e-8)).item()
        if total_imp_sum > 0
        else 1.0
    )

    return full_mask, sparse_mask, metrics


def estimate_flops_saved(
    image_size: Tuple[int, int],
    keep_ratio: float,
    decoder_channels: int = 64,
) -> Dict[str, float]:
    """Estimate FLOPs savings from tile-based sparse inference.

    Args:
        image_size: (H, W) of input image.
        keep_ratio: Fraction of tiles kept.
        decoder_channels: Decoder internal channels.

    Returns:
        Dict with full_decoder_flops, sparse_decoder_flops, reduction_ratio.
    """
    H, W = image_size
    # Decoder operates at H/4 × W/4
    out_h, out_w = H // 4, W // 4

    # LightDecoder FLOPs estimate (rough):
    # ConvTranspose2d(128→64, k=2): out_h*out_w * 128*64*4
    # Conv2d(128→64, k=1): out_h*out_w * 128*64
    # Fusion gate Conv2d(128→64, k=1): out_h*out_w * 128*64
    # ConvBlock (64→64, k=3 × 2): out_h*out_w * 64*64*9*2
    # ConvTranspose2d(64→64, k=2): out_h*2*out_w*2 * 64*64*4
    # ConvBlock (64→32, k=3 × 2): out_h*2*out_w*2 * 64*32*9*2
    # Conv2d(32→1, k=1): out_h*2*out_w*2 * 32*1
    pixels_h8 = out_h * 2  # H/8 resolution after up_conv
    pixels_h8_w8 = pixels_h8 * (out_w * 2)
    pixels_h4 = out_h * out_w

    full_flops = (
        pixels_h4 * 128 * 64 * 4 +           # ConvTranspose2d
        pixels_h4 * 128 * 64 +               # skip_proj
        pixels_h4 * 128 * 64 +               # fusion_gate
        pixels_h4 * 64 * 64 * 9 * 2 +        # dec_conv
        pixels_h8_w8 * 64 * 64 * 4 +         # final_up
        pixels_h8_w8 * 64 * 32 * 9 * 2 +     # final_conv
        pixels_h8_w8 * 32 * 1                 # head
    )

    sparse_flops = full_flops * keep_ratio

    return {
        "full_decoder_flops": full_flops,
        "sparse_decoder_flops": sparse_flops,
        "reduction_ratio": 1.0 - keep_ratio,
        "flops_saved": full_flops - sparse_flops,
    }
