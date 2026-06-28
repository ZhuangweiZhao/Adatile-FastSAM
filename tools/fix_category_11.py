#!/usr/bin/env python3
"""
修复 COCO JSON 中 category_id=11 的名称: road -> baseball_diamond
=================================================================
Fix category_id=11 name in COCO JSONs: road -> baseball_diamond

用法 | Usage:
    python tools/fix_category_11.py --src-root /root/Adatile-FastSAM/data/iSAID_processed
"""

import json
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Fix category_id=11 name in iSAID COCO JSONs")
    p.add_argument("--src-root", type=str, required=True,
                   help="Path to iSAID_processed directory")
    args = p.parse_args()

    src_root = Path(args.src_root)

    for split in ["train", "val"]:
        path = src_root / split / "annotations" / f"instances_{split}.json"

        if not path.exists():
            print(f"[{split}] SKIP: {path} not found")
            continue

        with open(path) as f:
            data = json.load(f)

        # 修复 category 名称 | Fix category name
        fixed = False
        for cat in data["categories"]:
            if cat["id"] == 11:
                old_name = cat["name"]
                cat["name"] = "baseball_diamond"
                print(f"[{split}] Fix: {old_name} -> baseball_diamond")
                fixed = True

        if not fixed:
            print(f"[{split}] No category_id=11 found in categories")

        # 统计 cat_id=11 的标注数量 | Count annotations with cat_id=11
        n_11 = sum(1 for a in data["annotations"] if a["category_id"] == 11)
        print(f"[{split}] Annotations cat_id=11: {n_11}")

        # 写回 | Write back
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[{split}] Saved ✓")

    print("Done")


if __name__ == "__main__":
    main()
