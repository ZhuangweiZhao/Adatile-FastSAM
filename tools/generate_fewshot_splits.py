#!/usr/bin/env python
"""Generate fixed few-shot splits for iSAID dataset.

Creates deterministic split files for 1-shot, 5-shot, and 10-shot
evaluation. Splits are generated once and committed to the repository
to ensure reproducibility (Reviewer Q3: multi-shot, Q10: leakage prevention).

Usage:
    python tools/generate_fewshot_splits.py --dataset datasets/iSAID --output datasets/iSAID/fewshot_splits --novel-classes 3 7 11 --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Generate few-shot splits for iSAID")
    parser.add_argument("--dataset", type=str, default="datasets/iSAID",
                        help="Path to dataset root")
    parser.add_argument("--output", type=str, default="datasets/iSAID/fewshot_splits",
                        help="Output directory for split files")
    parser.add_argument("--novel-classes", type=int, nargs="+", default=[3, 7, 11],
                        help="Novel class IDs for few-shot evaluation")
    parser.add_argument("--n-shots", type=int, nargs="+", default=[1, 5, 10],
                        help="K-shot values to generate splits for")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds for split generation")
    parser.add_argument("--n-query", type=int, default=20,
                        help="Number of query images per class")
    return parser.parse_args()


def discover_class_images(anno_path: str, novel_classes: List[int]) -> Dict[int, List[str]]:
    """Discover images containing each novel class from COCO annotations.

    Returns: {class_id: [image_filename, ...]}
    """
    if not os.path.exists(anno_path):
        print(f"WARNING: Annotation file not found: {anno_path}")
        print("Creating example split from metadata only. Replace with actual data.")
        return _create_example_split(novel_classes)

    with open(anno_path, "r") as f:
        data = json.load(f)

    # Build image_id → filename mapping
    img_id_to_name = {img["id"]: img["file_name"] for img in data.get("images", [])}

    # Build class_id → list of image filenames
    class_to_images: Dict[int, set] = defaultdict(set)
    for ann in data.get("annotations", []):
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        if cat_id in novel_classes and img_id in img_id_to_name:
            class_to_images[cat_id].add(img_id_to_name[img_id])

    return {k: sorted(list(v)) for k, v in class_to_images.items()}


def _create_example_split(novel_classes: List[int]) -> Dict[int, List[str]]:
    """Create example split with placeholder filenames (for testing)."""
    return {
        cid: [f"P{1000 + cid * 100 + i:04d}.png" for i in range(30)]
        for cid in novel_classes
    }


def create_split(
    class_to_images: Dict[int, List[str]],
    novel_classes: List[int],
    k_shot: int,
    n_query: int,
    seed: int,
) -> dict:
    """Create a single few-shot split.

    Args:
        class_to_images: {class_id: [all_image_filenames]}
        novel_classes: List of class IDs for few-shot evaluation
        k_shot: Number of support images per class
        n_query: Number of query images per class
        seed: Random seed for reproducibility

    Returns:
        Split dict with support_images, query_images, novel_classes.
    """
    rng = random.Random(seed)
    split: dict = {
        "novel_classes": novel_classes,
        "support_images": {},
        "query_images": [],
        "metadata": {
            "k_shot": k_shot,
            "n_query": n_query,
            "seed": seed,
        },
    }

    # For each novel class, select k_shot support + n_query query images
    # WITHOUT overlap to prevent support/query leakage (Reviewer Q10)
    for cid in novel_classes:
        all_images = class_to_images.get(cid, [])
        if len(all_images) < k_shot + n_query:
            print(f"  WARNING: Class {cid} has only {len(all_images)} images, "
                  f"need {k_shot + n_query}. Using all available.")
            k_shot = min(k_shot, len(all_images) // 2)
            n_query = min(n_query, len(all_images) - k_shot)

        shuffled = list(all_images)
        rng.shuffle(shuffled)

        support = shuffled[:k_shot]
        query = shuffled[k_shot:k_shot + n_query]

        split["support_images"][str(cid)] = support
        split["query_images"].extend(query)

    # Remove duplicates (an image could contain multiple novel classes)
    split["query_images"] = sorted(set(split["query_images"]))
    rng.shuffle(split["query_images"])

    return split


def main():
    args = parse_args()

    # Discover images per class
    anno_path = os.path.join(args.dataset, "annotations", "train",
                             "instances_train.json")
    class_to_images = discover_class_images(anno_path, args.novel_classes)

    print(f"Novel classes: {args.novel_classes}")
    for cid in args.novel_classes:
        print(f"  Class {cid}: {len(class_to_images.get(cid, []))} images")

    # Generate splits for all (k_shot, seed) combinations
    for k_shot in args.n_shots:
        shot_dir = Path(args.output) / f"{k_shot}shot"
        shot_dir.mkdir(parents=True, exist_ok=True)

        for seed_idx, seed in enumerate(args.seeds):
            split = create_split(
                class_to_images, args.novel_classes,
                k_shot=k_shot, n_query=args.n_query, seed=seed,
            )

            split_name = f"split{seed_idx}.json" if len(args.seeds) > 1 else "split0.json"
            split_path = shot_dir / split_name

            with open(split_path, "w") as f:
                json.dump(split, f, indent=2)

            total_support = sum(len(v) for v in split["support_images"].values())
            print(f"  {k_shot}-shot seed={seed}: "
                  f"{total_support} support, {len(split['query_images'])} query "
                  f"→ {split_path}")


if __name__ == "__main__":
    main()
