"""Few-shot episode sampler for instance segmentation.

Implements N-way K-shot episodic sampling with:
    - Fixed split support (deterministic class-to-image mapping)
    - Class-balanced query selection
    - Sparse density-weighted sampling
    - Episode-level class and file tracking
    - Multi-split support (1-shot, 5-shot, 10-shot)
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from adatile.datasets.base import FewShotSplit


class FewShotEpisodicSampler(Sampler):
    """Episodic few-shot sampler for instance segmentation.

    Each episode yields:
        support_indices (N_way × K_shot) + query_indices (N_way × N_query)

    Yields indices in order: [support..., query...] so the collate
    function can separate them by position.

    Reference: Adapted from PANet (Wang et al., ICCV 2019) and
               HSNet (Min et al., ICCV 2021).

    Args:
        dataset: COCO-style dataset with image metadata.
        fewshot_split: Loaded FewShotSplit.
        n_way: Number of novel classes per episode.
        k_shot: Number of support samples per class.
        n_query: Number of query samples per class.
        num_episodes: Total episodes per epoch.
        seed: Random seed for reproducibility.
        balance_query: If True, sample equal number of queries per class.
        allow_support_as_query: If True, support images may also appear as queries.
    """

    def __init__(
        self,
        dataset: Dataset,
        fewshot_split: FewShotSplit,
        n_way: int = 3,
        k_shot: int = 5,
        n_query: int = 5,
        num_episodes: int = 1000,
        seed: int = 42,
        balance_query: bool = True,
        allow_support_as_query: bool = False,
    ):
        self.dataset = dataset
        self.split = fewshot_split
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.num_episodes = num_episodes
        self.balance_query = balance_query
        self.allow_support_as_query = allow_support_as_query

        # Index: filename → dataset index
        self._filename_to_idx: Dict[str, int] = {}
        self._class_to_filenames: Dict[int, List[str]] = defaultdict(list)

        # Pre-build indices
        self._build_index()
        self._build_class_index()

        random.seed(seed)
        np.random.seed(seed)

        # Episode logging (for debugging)
        self._episode_log: List[Dict] = []

    # ── Index Building ───────────────────────────────────────────

    def _build_index(self) -> None:
        """Map image filenames to dataset indices."""
        if self._filename_to_idx:
            return

        for idx in range(len(self.dataset)):
            if idx < len(self.dataset.image_ids):
                iid = self.dataset.image_ids[idx]
            else:
                continue
            info = self.dataset.get_image_info(iid)
            fname = info.get("file_name", str(iid))
            self._filename_to_idx[fname] = idx
            self._filename_to_idx[Path(fname).name] = idx

    def _build_class_index(self) -> None:
        """Map class IDs to all filenames containing that class."""
        if self._class_to_filenames:
            return

        for idx in range(len(self.dataset)):
            if idx < len(self.dataset.image_ids):
                iid = self.dataset.image_ids[idx]
            else:
                continue
            info = self.dataset.get_image_info(iid)
            fname = info.get("file_name", str(iid))

            # Get class IDs present in this image
            anns = self.dataset._load_annotations(idx)
            class_ids = set(ann.get("category_id") for ann in anns)
            for cid in class_ids:
                self._class_to_filenames[cid].append(fname)

    # ── Episode Sampling ─────────────────────────────────────────

    def _sample_episode(self) -> Tuple[List[int], List[int], Set[int]]:
        """Sample one N-way K-shot episode.

        Returns:
            support_indices: Dataset indices for support images.
            query_indices: Dataset indices for query images.
            sampled_classes: Class IDs used in this episode.
        """
        novel_classes = self.split.novel_classes
        n_way = min(self.n_way, len(novel_classes))
        sampled_classes = set(random.sample(novel_classes, n_way))

        support_files: Set[str] = set()
        support_indices: List[int] = []
        query_indices: List[int] = []

        for cls_id in sampled_classes:
            # Support: pick K images from the support set
            support_candidates = self.split.get_support_for_class(cls_id)[:self.k_shot]
            if len(support_candidates) < self.k_shot:
                # Not enough designated support files → sample from class index
                extra = [f for f in self._class_to_filenames.get(cls_id, [])
                         if f not in support_candidates]
                support_candidates += extra[:self.k_shot - len(support_candidates)]

            chosen_support = support_candidates[:self.k_shot]
            support_files.update(chosen_support)

            for fname in chosen_support:
                idx = self._filename_to_idx.get(fname)
                if idx is not None:
                    support_indices.append(idx)

            # Query: pick Q images NOT in the support set
            query_candidates = [
                f for f in self.split.query_images
                if f in self._class_to_filenames.get(cls_id, [])
                and (self.allow_support_as_query or f not in support_files)
            ]

            # Also include any image that has this class
            all_class_files = self._class_to_filenames.get(cls_id, [])
            extra_query = [
                f for f in all_class_files
                if f not in support_files
                and (self.allow_support_as_query or f not in support_files)
                and f not in query_candidates
            ]

            chosen_query = []
            for source in [query_candidates, extra_query]:
                needed = self.n_query - len(chosen_query)
                if needed <= 0:
                    break
                pool = [f for f in source if f not in chosen_query]
                if pool:
                    n_sample = min(needed, len(pool))
                    chosen_query.extend(random.sample(pool, n_sample))

            for fname in chosen_query:
                idx = self._filename_to_idx.get(fname)
                if idx is not None:
                    query_indices.append(idx)

        return support_indices, query_indices, sampled_classes

    # ── Iterator ─────────────────────────────────────────────────

    def __iter__(self) -> Iterator[List[int]]:
        for episode_idx in range(self.num_episodes):
            support_idx, query_idx, classes = self._sample_episode()

            # Log for debugging
            self._episode_log.append({
                "episode": episode_idx,
                "classes": sorted(classes),
                "num_support": len(support_idx),
                "num_query": len(query_idx),
            })

            # Return support first, then query (collate function separates)
            yield support_idx + query_idx

    def __len__(self) -> int:
        return self.num_episodes

    def get_episode_log(self) -> List[Dict]:
        """Get the episode sampling log for analysis."""
        return self._episode_log


# ── Sparse Sampler ──────────────────────────────────────────────────


class SparseSampler(Sampler):
    """Density-weighted sparse sampler.

    Samples images with probability proportional to their instance density.
    Used for curriculum learning: early epochs focus on dense images.
    As training progresses, reduce temperature to sample more uniformly.

    Args:
        dataset: Base dataset with density computation.
        density_dir: Path to pre-computed density .npy files.
        density_key: "gt" (from annotations) or "file" (from saved .npy).
        temperature: Softmax temperature. Lower → more bias toward dense.
        num_samples: Images per epoch.
        seed: Random seed.
    """

    def __init__(
        self,
        dataset: Dataset,
        density_dir: Optional[str] = None,
        density_key: str = "gt",
        temperature: float = 0.5,
        num_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.density_dir = Path(density_dir) if density_dir else None
        self.density_key = density_key
        self.temperature = temperature
        self.num_samples = num_samples or len(dataset)
        self._densities: Optional[np.ndarray] = None

        random.seed(seed)
        np.random.seed(seed)

    def compute_densities(self) -> np.ndarray:
        """Compute instance density for all images.

        Returns:
            (N,) float array of densities in [0, 1].
        """
        if self._densities is not None:
            return self._densities

        densities = np.zeros(len(self.dataset), dtype=np.float32)

        if self.density_key == "file" and self.density_dir is not None:
            for idx in range(len(self.dataset)):
                if idx < len(self.dataset.image_ids):
                    iid = self.dataset.image_ids[idx]
                else:
                    iid = idx
                density_path = self.density_dir / f"{iid}_density.npy"
                if density_path.exists():
                    densities[idx] = float(np.load(density_path).mean())
                else:
                    densities[idx] = 0.0
        else:
            # GT-based: instance count / image area
            for idx in range(len(self.dataset)):
                try:
                    densities[idx] = self.dataset.get_instance_density(idx)
                except Exception:
                    densities[idx] = 0.0

        # Ensure non-zero
        if densities.max() == 0:
            densities = np.ones_like(densities)

        self._densities = densities
        return densities

    def set_temperature(self, temp: float) -> None:
        """Update sampling temperature (e.g., for curriculum annealing)."""
        self.temperature = temp

    def __iter__(self) -> Iterator[int]:
        densities = self.compute_densities()
        probs = np.exp(densities / max(self.temperature, 1e-8))
        probs = probs / probs.sum()
        indices = np.random.choice(
            len(densities),
            size=self.num_samples,
            replace=True,
            p=probs,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def get_density_stats(self) -> Dict[str, float]:
        """Summary statistics of image densities."""
        d = self.compute_densities()
        return {
            "mean_density": float(np.mean(d)),
            "std_density": float(np.std(d)),
            "min_density": float(np.min(d)),
            "max_density": float(np.max(d)),
            "median_density": float(np.median(d)),
        }
