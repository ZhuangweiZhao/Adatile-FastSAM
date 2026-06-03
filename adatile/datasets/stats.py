"""Dataset statistics computation utilities.

Computes tile-level and dataset-level statistics for AdaTile-FastSAM:
    - Tile entropy (Shannon, per-channel)
    - Sparse region ratio (background vs. foreground tile ratio)
    - Object density distribution (per-tile GT instance count)
    - Class distribution (per-category instance counts)
    - Scale distribution (tiny/small/medium/large object ratios)
    - Sparse efficiency statistics (token reduction opportunity)
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as scipy_entropy


# ── Tile-Level Statistics ────────────────────────────────────────────


def compute_tile_entropy(
    tile: np.ndarray,
    num_bins: int = 64,
    per_channel: bool = True,
) -> float:
    """Shannon entropy for an image tile.

    Args:
        tile: (H, W, 3) uint8 or float32 array.
        num_bins: Histogram bins.
        per_channel: Average over RGB channels.

    Returns:
        Mean entropy in bits.
    """
    if tile.max() <= 1.0:
        tile = (tile * 255).astype(np.uint8)
    tile = tile.astype(np.uint8)

    if per_channel and tile.ndim == 3:
        ents = []
        for c in range(tile.shape[2]):
            hist, _ = np.histogram(
                tile[:, :, c].ravel(), bins=num_bins, range=(0, 255)
            )
            hist = hist.astype(np.float64) / (hist.sum() + 1e-12)
            ents.append(scipy_entropy(hist + 1e-12))
        return float(np.mean(ents))
    else:
        gray = tile.mean(axis=2).astype(np.uint8) if tile.ndim == 3 else tile
        hist, _ = np.histogram(gray.ravel(), bins=num_bins, range=(0, 255))
        hist = hist.astype(np.float64) / (hist.sum() + 1e-12)
        return float(scipy_entropy(hist + 1e-12))


def compute_tile_entropy_batch(
    tiles: List[np.ndarray],
    num_bins: int = 64,
) -> np.ndarray:
    """Compute entropy for a batch of tiles.

    Returns:
        (N,) float array of entropy values.
    """
    return np.array([compute_tile_entropy(t, num_bins) for t in tiles])


def compute_sparse_ratio(
    masks: np.ndarray,
    tile_size: int = 384,
    stride: int = 192,
    bg_threshold: float = 0.95,
) -> Dict[str, float]:
    """Compute ratio of sparse (background-dominated) tiles.

    Args:
        masks: (N_inst, H, W) binary instance masks, OR (H, W) union mask.
        tile_size: Grid tile size.
        stride: Sliding stride.
        bg_threshold: Coverage below this ratio → sparse.

    Returns:
        Dict with: sparse_ratio, dense_ratio, num_tiles, num_sparse, num_dense.
    """
    if masks.ndim == 3:
        union = (masks.max(axis=0) > 0).astype(np.float32)
    else:
        union = masks.astype(np.float32)

    H, W = union.shape
    sparse_count = 0
    dense_count = 0

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)
            patch = union[y:y2, x:x2]
            coverage = patch.mean()
            if coverage < (1.0 - bg_threshold):
                sparse_count += 1
            else:
                dense_count += 1

    total = sparse_count + dense_count
    if total == 0:
        return {"sparse_ratio": 0.0, "dense_ratio": 1.0,
                "num_tiles": 0, "num_sparse": 0, "num_dense": 0}

    return {
        "sparse_ratio": sparse_count / total,
        "dense_ratio": dense_count / total,
        "num_tiles": total,
        "num_sparse": sparse_count,
        "num_dense": dense_count,
    }


# ── Object Scale Analysis ────────────────────────────────────────────


def compute_object_scale_distribution(
    annotations: List[Dict],
    image_area: int,
) -> Dict[str, float]:
    """Classify objects into scale buckets relative to image area.

    Buckets:
        tiny:   < 0.01% of image area
        small:  0.01% – 0.1%
        medium: 0.1% – 1%
        large:  1% – 10%
        huge:   > 10%

    Returns:
        Dict with ratios per scale bucket.
    """
    if not annotations:
        return {"tiny": 0.0, "small": 0.0, "medium": 0.0, "large": 0.0, "huge": 0.0}

    areas = []
    for ann in annotations:
        bbox = ann.get("bbox", [0, 0, 0, 0])
        areas.append(bbox[2] * bbox[3])

    areas = np.array(areas)
    rel_areas = areas / max(image_area, 1)
    n = len(areas)

    return {
        "tiny": float(np.sum(rel_areas < 1e-4) / n),
        "small": float(np.sum((rel_areas >= 1e-4) & (rel_areas < 1e-3)) / n),
        "medium": float(np.sum((rel_areas >= 1e-3) & (rel_areas < 1e-2)) / n),
        "large": float(np.sum((rel_areas >= 1e-2) & (rel_areas < 1e-1)) / n),
        "huge": float(np.sum(rel_areas >= 1e-1) / n),
    }


def recommend_tile_sizes(
    annotations: List[Dict],
    image_area: int,
    candidate_sizes: Optional[List[int]] = None,
) -> Dict[int, float]:
    """Recommend tile sizes based on object scale distribution.

    Larger objects → recommended larger tile sizes.
    Returns per-size "suitability" scores.

    Args:
        annotations: COCO annotations.
        image_area: Width × height.
        candidate_sizes: Tile sizes to score (default: [384, 768, 1536, 3072]).

    Returns:
        {tile_size: suitability_score} (higher = more suitable).
    """
    if candidate_sizes is None:
        candidate_sizes = [384, 768, 1536, 3072]

    scale_dist = compute_object_scale_distribution(annotations, image_area)

    # Heuristic: each size bucket maps to a tile size
    # tiny→384, small→768, medium→1536, large/huge→3072
    size_scores = defaultdict(float)
    size_scores[384] = scale_dist["tiny"] * 1.0 + scale_dist["small"] * 0.3
    size_scores[768] = scale_dist["small"] * 0.7 + scale_dist["medium"] * 0.3
    size_scores[1536] = scale_dist["medium"] * 0.7 + scale_dist["large"] * 0.5
    size_scores[3072] = scale_dist["large"] * 0.5 + scale_dist["huge"] * 1.0

    return {ts: size_scores.get(ts, 0.0) for ts in candidate_sizes}


# ── Sparse Efficiency Estimation ─────────────────────────────────────


def estimate_sparse_efficiency(
    dataset: Any,
    sample_indices: Optional[List[int]] = None,
    tile_sizes: Optional[List[int]] = None,
    stride_ratio: float = 0.5,
    density_threshold: float = 0.3,
    num_samples: int = 100,
) -> Dict[str, float]:
    """Estimate how many tiles can be skipped via sparse routing.

    Analyzes a sample of images to estimate:
        - Total tiles at each resolution
        - Density distribution
        - Expected skip ratio
        - Token reduction potential

    Args:
        dataset: BaseDataset instance.
        sample_indices: Specific indices (random if None).
        tile_sizes: Tile sizes to analyze.
        stride_ratio: Stride as fraction of tile size.
        density_threshold: Tiles below this density can be skipped.
        num_samples: Number of images to sample.

    Returns:
        Dict with: skip_ratio, token_reduction, avg_density, tile_distribution.
    """
    if tile_sizes is None:
        tile_sizes = [384, 768, 1536, 3072]

    # Sample indices
    if sample_indices is None:
        n = min(num_samples, len(dataset))
        sample_indices = np.random.choice(len(dataset), n, replace=False).tolist()

    total_tiles = 0
    skippable_tiles = 0
    density_values = []
    tile_size_counts = defaultdict(int)

    for idx in sample_indices:
        info = dataset.get_image_info(dataset.image_ids[idx])
        H, W = info.get("height", 1024), info.get("width", 1024)
        anns = dataset._load_annotations(idx)

        for ts in tile_sizes:
            stride = max(int(ts * stride_ratio), 128)
            if ts > H or ts > W:
                continue

            for y in range(0, H - ts + 1, stride):
                for x in range(0, W - ts + 1, stride):
                    total_tiles += 1
                    tile_size_counts[ts] += 1

                    # Estimate density: count overlapping instances
                    count = 0
                    for ann in anns:
                        bbox = ann.get("bbox", [0, 0, 0, 0])
                        bx, by, bw, bh = bbox
                        if (bx + bw > x and bx < x + ts and
                                by + bh > y and by < y + ts):
                            count += 1

                    density = count / (ts * ts)
                    density_values.append(density)
                    if density < density_threshold:
                        skippable_tiles += 1

    avg_density = float(np.mean(density_values)) if density_values else 0.0
    skip_ratio = skippable_tiles / max(total_tiles, 1)
    token_reduction = skip_ratio  # approximate: skipping = token reduction

    return {
        "skip_ratio": skip_ratio,
        "token_reduction": token_reduction,
        "avg_density": avg_density,
        "total_tiles_analyzed": total_tiles,
        "skippable_tiles": skippable_tiles,
        **{f"tile_{ts}_ratio": tile_size_counts[ts] / max(total_tiles, 1)
           for ts in tile_sizes},
    }


# ── Dataset-Level Reports ────────────────────────────────────────────


def generate_dataset_report(
    dataset: Any,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a comprehensive statistics report for a dataset.

    Computes and aggregates:
        - Image count and resolution distribution
        - Instance count and density
        - Object scale distribution
        - Class distribution
        - Sparse efficiency estimate
        - Tile size recommendations

    Args:
        dataset: BaseDataset instance.
        output_path: Optional path to save JSON report.

    Returns:
        Comprehensive statistics dict.
    """
    # Image stats
    image_stats = dataset.compute_image_stats()

    # Class distribution
    class_dist = dataset.compute_class_distribution()
    class_dist_norm = dataset.compute_class_distribution_normalized()

    # Object scale distribution (sample-based)
    sample_size = min(50, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False).tolist()
    scale_dists = []
    for idx in indices:
        anns = dataset._load_annotations(idx)
        area = dataset.get_image_area(idx)
        scale_dists.append(compute_object_scale_distribution(anns, area))

    avg_scale_dist = {
        k: float(np.mean([d[k] for d in scale_dists]))
        for k in scale_dists[0].keys()
    } if scale_dists else {}

    # Efficiency estimate
    efficiency = estimate_sparse_efficiency(dataset, num_samples=sample_size)

    report = {
        "dataset": dataset.__class__.__name__,
        "split": dataset.split,
        "generated_at": str(np.datetime64('now')),
        "image_stats": image_stats,
        "class_distribution": class_dist,
        "class_distribution_normalized": {str(k): v for k, v in class_dist_norm.items()},
        "average_object_scale_distribution": avg_scale_dist,
        "sparse_efficiency_estimate": efficiency,
        "num_classes": dataset.num_classes,
    }

    if output_path is not None:
        import json
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    return report
