#!/usr/bin/env python3
"""
修复标签映射 + 重建所有 tile | Fix label mapping + rebuild all tiles.
===============================================================

根因 | Root Cause:
    iSAID train 和 val 的原始 COCO JSON 使用不同的 category_id 编号，
    但 prep_isaid.py 用同一张 ACTUAL_TO_CODE_ID 映射表处理两者，
    导致 val 的类别被映射到错误的语义标签。

修复 | Fix:
    1. 为 train 和 val 分别建立正确的原始ID→ISAID_ID 映射
    2. 删除 prep_isaid_tiles.py 中 render_semantic_mask 的二次映射
    3. 重新生成所有 tile

用法 | Usage:
    # 先 dry-run 检查
    python tools/fix_label_mapping.py --data-root data --dry-run

    # 执行修复
    python tools/fix_label_mapping.py --data-root data
"""

import sys, json, argparse, shutil
from pathlib import Path
from collections import OrderedDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# ISAID_CATEGORIES: 目标编码 | Target encoding (代码定义的标准)
ISAID_CATEGORIES = OrderedDict([
    (1, "small_vehicle"), (2, "large_vehicle"), (3, "plane"),
    (4, "storage_tank"), (5, "ship"), (6, "harbor"),
    (7, "ground_track_field"), (8, "soccer_ball_field"), (9, "tennis_court"),
    (10, "swimming_pool"), (11, "road"), (12, "basketball_court"),
    (13, "bridge"), (14, "helicopter"), (15, "roundabout"),
])

# ISAID 名称 → 目标 ID | Name → target ID
_NAME_TO_TARGET = {name: tid for tid, name in ISAID_CATEGORIES.items()}


def build_mapping(original_categories):
    """
    根据原始 JSON 的 categories 列表构建映射表。
    Build mapping from original category_id → target ISAID ID.

    通过名称匹配 (case-insensitive + 常见别名)。
    Via name matching (case-insensitive + common aliases).
    """
    # 别名表 | Alias table (original names → ISAID standard names)
    ALIASES = {
        "small_vehicle": "small_vehicle",
        "small vehicle": "small_vehicle",
        "large_vehicle": "large_vehicle",
        "large vehicle": "large_vehicle",
        "plane": "plane",
        "storage_tank": "storage_tank",
        "storage tank": "storage_tank",
        "ship": "ship",
        "harbor": "harbor",
        "ground_track_field": "ground_track_field",
        "ground track field": "ground_track_field",
        "soccer_ball_field": "soccer_ball_field",
        "soccer ball field": "soccer_ball_field",
        "tennis_court": "tennis_court",
        "tennis court": "tennis_court",
        "swimming_pool": "swimming_pool",
        "swimming pool": "swimming_pool",
        "road": "road",
        "basketball_court": "basketball_court",
        "basketball court": "basketball_court",
        "baseball_diamond": "basketball_court",  # iSAID val uses this!
        "bridge": "bridge",
        "helicopter": "helicopter",
        "roundabout": "roundabout",
    }

    mapping = {}
    unmatched = []

    for cat in original_categories:
        orig_id = cat["id"]
        orig_name = cat["name"].lower().replace("_", " ").strip()
        # 先精确匹配 | Try exact match first
        std_name = ALIASES.get(orig_name) or ALIASES.get(cat["name"].strip())
        if std_name is None:
            unmatched.append((orig_id, cat["name"]))
            continue
        target_id = _NAME_TO_TARGET.get(std_name)
        if target_id is None:
            unmatched.append((orig_id, cat["name"]))
            continue
        mapping[orig_id] = target_id

    return mapping, unmatched


def fix_annotations_file(src_json, dst_json, mapping):
    """用正确的 mapping 修正一个 JSON 文件 | Fix one JSON with correct mapping."""
    with open(src_json) as f:
        data = json.load(f)

    fixed = 0
    for ann in data.get("annotations", []):
        old_id = ann.get("category_id", 0)
        if old_id in mapping:
            ann["category_id"] = mapping[old_id]
            fixed += 1

    # 写入标准 categories | Write standard categories
    data["categories"] = [
        {"id": tid, "name": name} for tid, name in ISAID_CATEGORIES.items()
    ]

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w") as f:
        json.dump(data, f)

    return fixed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    data_root = Path(args.data_root)
    isaid_orig = data_root / "iSAID"
    isaid_proc = data_root / "iSAID_processed"

    print("=" * 70)
    print("  FIX LABEL MAPPING | 修复标签映射")
    print("=" * 70)

    for split, orig_json_rel, proc_json_rel in [
        ("train", "TrainData/train/Annotations/iSAID_train.json", "train/annotations/instances_train.json"),
        ("val", "ValidationData/val/Annotations/iSAID_val.json", "val/annotations/instances_val.json"),
    ]:
        orig_json = isaid_orig / orig_json_rel
        proc_json = isaid_proc / proc_json_rel

        print(f"\n{'─'*70}")
        print(f"  SPLIT: {split}")
        print(f"  Original: {orig_json}")
        print(f"  Output:   {proc_json}")

        if not orig_json.exists():
            print(f"  ❌ Original JSON not found!")
            continue

        with open(orig_json) as f:
            orig_data = json.load(f)

        orig_cats = orig_data.get("categories", [])
        print(f"  Original categories: {[(c['id'], c['name']) for c in orig_cats]}")

        mapping, unmatched = build_mapping(orig_cats)
        print(f"  Built mapping: {mapping}")
        if unmatched:
            print(f"  ⚠️  Unmatched: {unmatched}")

        if args.dry_run:
            # 对比旧映射 | Compare with old mapping
            OLD_MAPPING = {1: 4, 2: 2, 3: 1, 4: 3, 5: 5, 6: 10, 7: 6, 8: 9,
                          9: 7, 10: 8, 11: 11, 12: 13, 13: 12, 14: 15, 15: 14}
            print(f"  OLD mapping: {OLD_MAPPING}")
            print(f"  NEW mapping: {mapping}")
            if mapping != OLD_MAPPING:
                print(f"  ✅ FIX NEEDED: mapping differs from old!")
            else:
                print(f"  ⚠️  Same as old mapping — no fix needed for {split}")
            continue

        # 备份旧文件 | Backup old file
        if proc_json.exists():
            backup = proc_json.with_suffix(".json.bak")
            shutil.copy2(proc_json, backup)
            print(f"  Backed up: {backup}")

        fixed = fix_annotations_file(orig_json, proc_json, mapping)
        print(f"  Fixed {fixed} annotations → {proc_json}")

    print(f"\n{'='*70}")
    if args.dry_run:
        print(f"  DRY-RUN complete. Run without --dry-run to apply fixes.")
        print(f"  After fixing JSONs, regenerate tiles:")
        print(f"    rm -rf data/iSAID_tiles/masks_full/ data/iSAID_tiles/masks/")
        print(f"    rm -rf data/iSAID_tiles/images/ data/iSAID_tiles/metadata/")
        print(f"    python tools/prep_isaid_tiles.py --steps 1,2,3 --splits train,val")
    else:
        print(f"  ✅ JSON files fixed. Next step:")
        print(f"    1. Fix render_semantic_mask in prep_isaid_tiles.py (remove ACTUAL_TO_CODE_ID)")
        print(f"    2. Regenerate all tiles")
        print(f"    3. Run diag_quick_check.py to verify consistency")


if __name__ == "__main__":
    main()
