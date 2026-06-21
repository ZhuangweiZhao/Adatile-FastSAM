#!/usr/bin/env python3
"""
单实例追踪: 原始 JSON → instances JSON → mask_full → tile mask
================================================================
Trace a single instance through the entire pipeline to verify label consistency.

追踪链路 | Trace chain:
    ① 原始 iSAID JSON: ann["category_id"] = ?
    ② prep_isaid.py fix_annotations → instances_{split}.json: ann["category_id"] = ?
    ③ prep_isaid_tiles.py render_semantic_mask → masks_full/*.png: pixel value = ?
    ④ prep_isaid_tiles.py step2 cut tiles → masks/{split}/*.png: pixel value = ?

用法 | Usage:
    python tools/diag_trace_single_instance.py --data-root /root/autodl-tmp
"""

import sys, json, argparse
from pathlib import Path
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

# 代码定义的类别 | Code-defined categories
ISAID_NAMES = {
    1: "small_vehicle", 2: "large_vehicle", 3: "plane",
    4: "storage_tank", 5: "ship", 6: "harbor",
    7: "ground_track_field", 8: "soccer_ball_field", 9: "tennis_court",
    10: "swimming_pool", 11: "road", 12: "basketball_court",
    13: "bridge", 14: "helicopter", 15: "roundabout",
}

# 保留旧映射用于对比 | Keep deprecated mapping for comparison
from adatile.utils.label_mapping import _DEPRECATED_ACTUAL_TO_CODE_ID as ACTUAL_TO_CODE_ID, build_mapping


def trace(args):
    """追踪 pipeline 各阶段的类别 ID | Trace category ID through pipeline stages.

    追踪链路 | Trace chain (4 stages):
        ① 原始 iSAID JSON → ② prep_isaid.py 修正 → ③ masks_full 全图掩码 → ④ tile 掩码
    """
    print("=" * 70)
    print("  SINGLE INSTANCE TRACE | 单实例类别 ID 追踪")
    print("=" * 70)

    # 数据目录 | Data directories
    src_root = Path(args.data_root) / "iSAID_processed"  # 预处理后的 COCO JSON
    tile_root = Path(args.data_root) / "iSAID_tiles"      # tile 数据集
    orig_root = Path(args.data_root) / "iSAID"            # 原始 iSAID 数据

    for split in ["train", "val"]:
        print(f"\n{'─'*70}")
        print(f"  SPLIT: {split}")
        print(f"{'─'*70}")

        # ── ① 原始 iSAID JSON | Original iSAID JSON ──
        if split == "train":
            orig_json = orig_root / "TrainData/train/Annotations/iSAID_train.json"
        else:
            orig_json = orig_root / "ValidationData/val/Annotations/iSAID_val.json"

        if not orig_json.exists():
            print(f"  ❌ Original JSON not found: {orig_json}")
            continue

        with open(orig_json) as f:
            orig_data = json.load(f)

        # categories in original JSON
        orig_cats = {c["id"]: c.get("name", "?") for c in orig_data.get("categories", [])}
        print(f"\n  ① Original JSON categories | 原始 JSON 类别:")
        for cid, cname in sorted(orig_cats.items()):
            print(f"     id={cid:2d} → {cname}")

        # ── ② instances_{split}.json (after fix_annotations) | 修正后 ──
        fixed_json = src_root / split / "annotations" / f"instances_{split}.json"
        if not fixed_json.exists():
            print(f"\n  ❌ Fixed JSON not found: {fixed_json}")
            print(f"     → prep_isaid.py may not have been run for {split}")
            continue

        with open(fixed_json) as f:
            fixed_data = json.load(f)

        fixed_cats = {c["id"]: c.get("name", "?") for c in fixed_data.get("categories", [])}
        print(f"\n  ② Fixed JSON categories (after prep_isaid.py) | 修正后类别:")
        for cid, cname in sorted(fixed_cats.items()):
            print(f"     id={cid:2d} → {cname}")

        # ── Pick one annotation to trace | 选一个标注追踪 ──
        anns = fixed_data.get("annotations", [])
        if not anns:
            print(f"\n  ⚠️  No annotations in fixed JSON!")
            continue

        # 选一个类别明确的标注 | Pick one annotation with clear category
        best_ann = None
        for ann in anns:
            cat_id = ann.get("category_id", 0)
            if cat_id in [3, 5, 14, 15]:  # plane, ship, helicopter, roundabout
                best_ann = ann
                break
        if best_ann is None:
            best_ann = anns[0]

        ann_cat = best_ann["category_id"]
        ann_img_id = best_ann["image_id"]

        # 查找对应的 image | Find corresponding image
        img_info = None
        for img in fixed_data.get("images", []):
            if img["id"] == ann_img_id:
                img_info = img
                break

        if img_info is None:
            print(f"  ❌ Cannot find image for image_id={ann_img_id}")
            continue

        img_name = img_info["file_name"]
        print(f"\n  ③ Tracing annotation | 追踪标注:")
        print(f"     image_id={ann_img_id}, file={img_name}")
        print(f"     category_id in fixed JSON = {ann_cat} ({ISAID_NAMES.get(ann_cat, 'UNKNOWN')})")

        # ── ③ masks_full/*.png | 全图掩码 ──
        mask_full_path = tile_root / "masks_full" / f"{Path(img_name).stem}_mask.png"
        if mask_full_path.exists():
            from PIL import Image
            mask_full = np.array(Image.open(mask_full_path))
            unique_full = sorted(np.unique(mask_full).tolist())
            print(f"\n  ③ masks_full pixel values | 全图掩码像素值:")
            print(f"     File: {mask_full_path}")
            print(f"     Unique values: {unique_full}")

            # 统计各值占比 | Value distribution
            for v in unique_full:
                if v == 0:
                    continue
                pct = (mask_full == v).sum() / mask_full.size * 100
                name = ISAID_NAMES.get(v, f"UNKNOWN_{v}")
                print(f"       value={v:2d} ({name:<20s}): {pct:.2f}%")
        else:
            print(f"\n  ③ masks_full: NOT FOUND ({mask_full_path})")
            print(f"     → Step 1 was likely not run, or --steps 2,3 was used")

        # ── ④ Tile masks | Tile 掩码 ──
        tile_mask_dir = tile_root / "masks" / split
        if not tile_mask_dir.exists():
            print(f"\n  ④ Tile masks dir NOT FOUND: {tile_mask_dir}")
            continue

        tile_files = sorted(tile_mask_dir.glob("*.png"))
        # 找属于该 image 的 tile | Find tiles belonging to this image
        img_stem = Path(img_name).stem
        img_tiles = [t for t in tile_files if t.stem.startswith(img_stem)]
        if not img_tiles:
            # 随便取几个 tile | Just sample random tiles
            import random
            random.seed(42)
            img_tiles = random.sample(tile_files, min(10, len(tile_files)))

        print(f"\n  ④ Tile masks | Tile 掩码 ({len(img_tiles)} tiles):")
        all_tile_values = set()
        for tile_path in img_tiles[:5]:  # 最多看 5 个
            tile_mask = np.array(Image.open(tile_path))
            unique_tile = sorted(np.unique(tile_mask).tolist())
            all_tile_values.update(unique_tile)
            fg_pct = (tile_mask > 0).sum() / tile_mask.size * 100
            print(f"     {tile_path.name}: unique={unique_tile}, FG={fg_pct:.1f}%")

        print(f"\n  Union of all tile values for {split}: {sorted(all_tile_values)}")

    # ── 跨 split 对比 | Cross-split comparison ──
    print(f"\n{'='*70}")
    print(f"  CROSS-SPLIT COMPARISON | 跨 Split 对比")
    print(f"{'='*70}")

    # 快速扫描 train vs val tile mask 的值 | Quick scan of train vs val tile mask values
    for split in ["train", "val"]:
        mask_dir = tile_root / "masks" / split
        if not mask_dir.exists():
            print(f"  {split}: NO MASKS")
            continue

        files = sorted(mask_dir.glob("*.png"))
        import random
        random.seed(42)
        sample_files = random.sample(files, min(50, len(files)))

        all_values = set()
        from PIL import Image
        for f in sample_files:
            m = np.array(Image.open(f))
            all_values.update(np.unique(m).tolist())

        print(f"  {split}: sampled {len(sample_files)}/{len(files)} tiles → unique values: {sorted(all_values)}")

    # ── 映射链路验证 | Mapping chain verification ──
    print(f"\n{'='*70}")
    print(f"  MAPPING CHAIN VERIFICATION | 映射链路验证")
    print(f"{'='*70}")

    # 选 plane (ISAID code id=3) 验证 | Verify with plane (code id=3)
    # In ACTUAL_TO_CODE_ID: 3→1, 4→3
    # So if original JSON has plane=?, after fix_annotations it becomes 3
    # Then render_semantic_mask maps it again: ACTUAL_TO_CODE_ID[3] = 1
    # But 1 in ISAID_CATEGORIES is small_vehicle!

    print(f"""
    If plane in ORIGINAL JSON has category_id = X:
      fix_annotations:  X → ACTUAL_TO_CODE_ID[X] = Y
      render_semantic_mask: Y → ACTUAL_TO_CODE_ID[Y] = Z
      Final pixel value = Z

    ISAID_CATEGORIES: plane = 3
    ACTUAL_TO_CODE_ID mapping:
      3 → 1  (so if original has plane=3, fix_annotations makes it 1, but that's small_vehicle!)
      4 → 3  (so if original has plane=4, fix_annotations makes it 3 = plane)

    DOUBLE mapping danger:
      If fix_annotations maps to 3 (correct plane):
        render_semantic_mask maps 3 → 1 (small_vehicle!) ← WRONG
      If fix_annotations maps to 1 (small_vehicle):
        render_semantic_mask maps 1 → 4 (storage_tank!) ← WRONG

    Only idempotent IDs survive: {2, 5, 11}
      2→2→2 (large_vehicle), 5→5→5 (ship), 11→11→11 (road)
""")

    print("  ✅ Run this script on the server to see the actual trace!")


def main():
    p = argparse.ArgumentParser(description="Trace single instance through pipeline")
    p.add_argument("--data-root", type=str, default="/root/autodl-tmp")
    args = p.parse_args()
    trace(args)


if __name__ == "__main__":
    main()
