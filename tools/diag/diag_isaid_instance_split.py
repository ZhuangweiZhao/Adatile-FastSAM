#!/usr/bin/env python3
"""
iSAID Instance Few-Shot Split 数据验证 | Data Validation Script.
=================================================================

验证 v3 Instance Few-Shot Split 的正确性 | Validates v3 Instance Few-Shot Split correctness.

检查项 | Checks:
    1. COCO JSON 格式正确性 (images/annotations/categories)
    2. Tile 尺寸一致性 (896×896)
    3. Per-class instance count + tile distribution
    4. Base/Novel fold 定义完整性
    5. 数据集类加载 + K-shot 采样

用法 | Usage::

    python tools/diag/diag_isaid_instance_split.py
    python tools/diag/diag_isaid_instance_split.py --root data/iSAID_instance_fewshot
"""

import sys, argparse, json
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

ISAID_CATEGORIES = {
    1: "small_vehicle", 2: "large_vehicle", 3: "plane",
    4: "storage_tank", 5: "ship", 6: "harbor",
    7: "ground_track_field", 8: "soccer_ball_field", 9: "tennis_court",
    10: "swimming_pool", 11: "baseball_diamond", 12: "basketball_court",
    13: "bridge", 14: "helicopter", 15: "roundabout",
}


def parse_args():
    p = argparse.ArgumentParser(description="iSAID Instance Few-Shot Split Validation")
    p.add_argument("--root", type=str, default="data/iSAID_instance_fewshot",
                   help="数据根目录 | Data root directory")
    return p.parse_args()


def check_coco_json(path: Path, expected_tile_size: int = 896) -> dict | None:
    """验证 COCO JSON 格式 | Validate COCO JSON format."""
    if not path.exists():
        print(f"  [FAIL] Not found: {path}")
        return None

    with open(path) as f:
        data = json.load(f)

    errors = []
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    # 检查 images | Check images
    if not images:
        errors.append("No images found")
    sizes_ok = 0
    sizes_bad = 0
    img_ids = set()
    for img in images:
        img_ids.add(img["id"])
        w, h = img.get("width", 0), img.get("height", 0)
        if w == expected_tile_size and h == expected_tile_size:
            sizes_ok += 1
        else:
            sizes_bad += 1
            if sizes_bad <= 3:
                errors.append(f"Unexpected tile size: {w}x{h} (id={img['id']})")

    # 检查 annotations | Check annotations
    if not annotations:
        errors.append("No annotations found")
    orphan_anns = 0
    ann_ids = set()
    cat_counts = defaultdict(int)
    for ann in annotations:
        ann_ids.add(ann["id"])
        if ann["image_id"] not in img_ids:
            orphan_anns += 1
        cat_counts[ann.get("category_id", 0)] += 1

    if orphan_anns > 0:
        errors.append(f"{orphan_anns} orphan annotations (image_id not in images)")

    # 检查 categories | Check categories
    if len(categories) != 15:
        errors.append(f"Expected 15 categories, got {len(categories)}")

    status = "[OK]" if not errors else "[WARN]"
    print(f"  {status} {path.name}: {len(images)} tiles, {len(annotations)} annotations, "
          f"{sizes_ok} @{expected_tile_size}x{expected_tile_size}" +
          (f", {sizes_bad} mismatched" if sizes_bad else ""))

    if orphan_anns:
        print(f"       orphan annotations: {orphan_anns}")

    if errors and not orphan_anns:
        for e in errors[:5]:
            print(f"       {e}")

    return {
        "n_tiles": len(images),
        "n_annotations": len(annotations),
        "sizes_ok": sizes_ok,
        "sizes_bad": sizes_bad,
        "per_class": dict(cat_counts),
        "errors": errors,
    }


def check_folds(folds_dir: Path) -> dict:
    """验证 Fold 定义 | Validate fold definitions."""
    print(f"\n--- Fold Definitions ---")

    all_ok = True
    for fold_id in range(3):
        fold_path = folds_dir / f"fold_{fold_id}.json"
        if not fold_path.exists():
            print(f"  [FAIL] Missing fold_{fold_id}.json")
            all_ok = False
            continue

        with open(fold_path) as f:
            fd = json.load(f)

        base = fd.get("base", [])
        novel = fd.get("novel", [])
        overlap = set(base) & set(novel)

        issues = []
        if len(base) != 10:
            issues.append(f"Expected 10 base classes, got {len(base)}")
        if len(novel) != 5:
            issues.append(f"Expected 5 novel classes, got {len(novel)}")
        if overlap:
            issues.append(f"Overlap between base and novel: {overlap}")

        all_ids = set(base) | set(novel)
        if all_ids != set(range(1, 16)):
            missing = set(range(1, 16)) - all_ids
            issues.append(f"Missing class IDs: {missing}")

        status = "[OK]" if not issues else "[FAIL]"
        print(f"  {status} Fold {fold_id}: Base={len(base)}, Novel={len(novel)}")
        for iss in issues:
            print(f"       {iss}")

        if not issues:
            for cid in base:
                name = ISAID_CATEGORIES.get(cid, "?")
                print(f"       Base:   {name} (id={cid})")
            for cid in novel:
                name = ISAID_CATEGORIES.get(cid, "?")
                print(f"       Novel:  {name} (id={cid})")

    return {"all_ok": all_ok}


def test_dataset(root: str, split: str, fold: int, mode: str) -> bool:
    """测试数据集加载 | Test dataset loading."""
    from adatile.datasets.isaid_instance_fewshot import (
        ISAIDInstanceFewShotDataset, sample_k_shot,
    )
    try:
        ds = ISAIDInstanceFewShotDataset(root=root, split=split, fold=fold, mode=mode)
        n = len(ds)
        if n == 0:
            print(f"  [WARN] {split}/{mode}: 0 tiles (empty dataset)")
            return True  # Not an error, just empty

        sample = ds[0]
        img_shape = list(sample["image"].shape)
        n_inst = len(sample["instances"])

        img_ok = img_shape == [3, 896, 896]
        status = "[OK]" if img_ok else "[WARN]"
        print(f"  {status} {split}/{mode}: {n} tiles, "
              f"sample[0] image={img_shape}, instances={n_inst}")

        if n_inst > 0:
            inst = sample["instances"][0]
            mask_shape = list(inst["mask"].shape)
            print(f"       instance: cat={inst['category_id']}, "
                  f"bbox={inst['bbox']}, mask={mask_shape}")

        # K-shot test
        if mode == "novel" and n > 0:
            k_indices = sample_k_shot(ds, k=5, seed=42)
            print(f"       K=5 shot: {len(k_indices)} tiles")

        return img_ok
    except Exception as e:
        print(f"  [FAIL] {split}/{mode}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    args = parse_args()
    root = Path(args.root)

    print("=" * 70)
    print("  iSAID Instance Few-Shot Split — Validation")
    print(f"  Root: {root}")
    print("=" * 70)

    if not root.exists():
        print(f"\n[FAIL] Root directory not found: {root}")
        print("Run: python tools/data/prep_isaid_instance.py")
        return

    all_ok = True

    # 1. COCO JSON 验证 | COCO JSON validation
    print("\n--- COCO JSON Validation ---")
    for split in ["train", "val"]:
        path = root / "annotations" / f"instances_{split}.json"
        result = check_coco_json(path, expected_tile_size=896)
        if result is None:
            all_ok = False
        elif result["errors"]:
            all_ok = False

    # 2. Fold 定义验证 | Fold definitions validation
    folds_dir = root / "folds"
    fold_result = check_folds(folds_dir)
    if not fold_result["all_ok"]:
        all_ok = False

    # 3. 数据集加载测试 | Dataset loading test
    print("\n--- Dataset Loading Test ---")
    for split in ["train", "val"]:
        anno_path = root / "annotations" / f"instances_{split}.json"
        if not anno_path.exists():
            continue
        for mode in ["base", "novel"]:
            ok = test_dataset(str(root), split, fold=0, mode=mode)
            if not ok:
                all_ok = False

    # 4. 统计摘要 | Statistics summary
    stats_path = root / "stats" / "class_distribution.json"
    if stats_path.exists():
        print("\n--- Statistics Summary ---")
        with open(stats_path) as f:
            stats = json.load(f)
        for split_name, split_stats in stats.get("splits", {}).items():
            n_tiles = split_stats.get("n_tiles", 0)
            n_anns = split_stats.get("n_annotations", 0)
            n_empty = split_stats.get("n_empty_tiles", 0)
            empty_pct = 100 * n_empty / max(1, n_tiles)
            print(f"  {split_name}: {n_tiles} tiles, {n_anns} annotations, "
                  f"{n_empty} empty ({empty_pct:.1f}%)")
    else:
        print(f"\n  [WARN] Statistics file not found: {stats_path}")

    # ── 最终判定 | Final verdict ──
    print(f"\n{'='*70}")
    if all_ok:
        print("  [OK] All validations passed!")
    else:
        print("  [WARN] Some validations had issues. Review above for details.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
