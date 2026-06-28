#!/usr/bin/env python3
"""
诊断原始 iSAID JSON 中 baseball_diamond 的真实 category_id
=============================================================
Diagnose the real category_id of baseball_diamond in the ORIGINAL iSAID JSONs.

用法 | Usage:
    python tools/diag/diag_original_categories.py --orig-root /root/autodl-pub/DOTA
"""

import json
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Diagnose original iSAID category IDs")
    p.add_argument("--orig-root", type=str, required=True,
                   help="Path to original DOTA/iSAID data (e.g. /root/autodl-pub/DOTA)")
    args = p.parse_args()

    orig_root = Path(args.orig_root)

    for split, rel_path in [
        ("train", "TrainData/train/Annotations/iSAID_train.json"),
        ("val", "ValidationData/val/Annotations/iSAID_val.json"),
    ]:
        orig_path = orig_root / rel_path
        if not orig_path.exists():
            print(f"[{split}] NOT FOUND: {orig_path}")
            continue

        with open(orig_path) as f:
            data = json.load(f)

        print(f"\n{'='*60}")
        print(f"  {split.upper()} — {orig_path}")
        print(f"{'='*60}")
        print(f"  {'orig_id':<10} {'name':<28} {'n_annotations':>14}")
        print(f"  {'-'*50}")

        total_anns = 0
        for cat in data.get("categories", []):
            n = sum(1 for a in data["annotations"] if a["category_id"] == cat["id"])
            total_anns += n
            marker = " ← BASEBALL_DIAMOND?" if "baseball" in cat["name"].lower() else ""
            print(f"  {cat['id']:<10} {cat['name']:<28} {n:>14}{marker}")

        print(f"  {'-'*50}")
        print(f"  Total annotations: {total_anns}")


if __name__ == "__main__":
    main()
