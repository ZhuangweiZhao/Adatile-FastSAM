#!/usr/bin/env python3
"""
iSAID-5i 数据集类别分布分析脚本 | iSAID-5i Class Distribution Analysis.
======================================================================

统计每个 Fold 下各类的 train/val tile 数量，与参考表对比。
Count train/val tile counts per class per fold, compare with reference.

参考来源 | Reference: iSAID-5i benchmark standard fold splits.
标准 Fold 划分 (ISAID5I_FOLDS):
    Fold 0: novel=[9(small_vehicle), 15(harbor), 11(swimming_pool), 5(basketball_court), 12(roundabout)]
    Fold 1: novel=[14(plane), 8(large_vehicle), 7(bridge), 6(ground_track_field), 4(tennis_court)]
    Fold 2: novel=[1(ship), 2(storage_tank), 10(helicopter), 13(soccer_ball_field), 3(baseball_diamond)]

用法 | Usage::
    python tools/diag/diag_isaid5i_class_stats.py
    python tools/diag/diag_isaid5i_class_stats.py --data-root data/iSAID-5i/iSAID
"""

from __future__ import annotations

import sys, argparse
from pathlib import Path
from collections import defaultdict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import cv2
from tqdm import tqdm

from adatile.utils.label_mapping import ISAID5I_CATEGORIES, ISAID5I_FOLDS

# ── 参考表 (论文标准值) | Reference Table (standard paper values) ──
# 格式: {class_id: (class_name, train_tiles, val_tiles)}
# 注意: 此为 iSAID-5i 256×256 tiles 的典型值，实际数量因数据集版本略有差异
REFERENCE_TABLE = {
    1:  ("ship",              4335, 1592),
    2:  ("storage_tank",      961,  280),
    3:  ("baseball_diamond",  542,  240),
    4:  ("tennis_court",      2592, 813),
    5:  ("basketball_court",  660,  183),
    6:  ("ground_track_field",1478, 547),
    7:  ("bridge",            246,  96),
    8:  ("large_vehicle",     3931, 1190),
    9:  ("small_vehicle",     5302, 1814),
    10: ("helicopter",        78,   10),
    11: ("swimming_pool",     243,  74),
    12: ("roundabout",        219,  54),
    13: ("soccer_ball_field", 2131, 738),
    14: ("plane",             1890, 1000),
    15: ("harbor",            3624, 1432),
}


def parse_args():
    p = argparse.ArgumentParser(description="iSAID-5i 类别分布分析 | Class Distribution Analysis")
    p.add_argument("--data-root", type=str,
                   default=str(_PROJECT_ROOT / "data" / "iSAID-5i" / "iSAID"),
                   help="iSAID-5i 数据根目录 | Data root")
    return p.parse_args()


def clean_tile_name(raw: str) -> str | None:
    """从 split 文件行提取干净的 tile 名 | Extract clean tile name."""
    raw = raw.strip()
    for suffix in ["_instance_color_RGB.png", "_instance_id_RGB.png", ".png"]:
        idx = raw.find(suffix)
        if idx > 0:
            return raw[:idx]
    return raw.rsplit(".", 1)[0] if "." in raw else raw


def get_mask_path(mask_dir: Path, tile_name: str) -> Path:
    """获取语义掩码路径 | Get semantic mask path."""
    p = mask_dir / f"{tile_name}_instance_color_RGB.png"
    if p.exists():
        return p
    p = mask_dir / f"{tile_name}.png"
    if p.exists():
        return p
    return mask_dir / f"{tile_name}_instance_color_RGB.png"


def count_class_tiles(data_root: Path, split: str, fold: int) -> dict[int, set]:
    """
    统计指定 split 下每类出现的 tile 集合。
    Count tile sets per class for a given split.

    :return: {class_id: {tile_name, ...}}
    """
    list_dir = data_root / split / f"{split}_list"
    list_file = list_dir / f"split{fold}_{split}.txt"

    if not list_file.exists():
        print(f"  WARNING: {list_file} not found!")
        return {}

    with open(list_file) as f:
        raw_names = [line.strip() for line in f if line.strip()]

    mask_dir = data_root / split / "semantic_png"
    cls_tiles = defaultdict(set)

    for raw in tqdm(raw_names, desc=f"  {split} fold={fold}", leave=False):
        clean = clean_tile_name(raw)
        if not clean:
            continue
        mask_path = get_mask_path(mask_dir, clean)
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        classes_in_tile = set(np.unique(mask).tolist()) - {0}
        for c in classes_in_tile:
            cls_tiles[int(c)].add(clean)

    return cls_tiles


def main():
    args = parse_args()
    data_root = Path(args.data_root)

    print("=" * 110)
    print("iSAID-5i Class Distribution Analysis | 类别分布分析")
    print("=" * 110)

    # ═══ 统计所有 Fold | Count all folds ═══
    all_counts = {}  # {fold: {split: {class_id: count}}}
    for fold in [0, 1, 2]:
        print(f"\n--- Fold {fold} ---")
        all_counts[fold] = {}
        for split in ["train", "val"]:
            cls_tiles = count_class_tiles(data_root, split, fold)
            all_counts[fold][split] = {c: len(tiles) for c, tiles in cls_tiles.items()}
            total_tiles_in_split = len(set.union(*cls_tiles.values())) if cls_tiles else 0
            print(f"  {split}: {total_tiles_in_split} tiles with FG, "
                  f"{len(cls_tiles)} classes present")

    # ═══ 与参考表对比 | Compare with reference ═══
    print(f"\n{'=' * 110}")
    print("Comparison with Reference Table | 与参考表对比")
    print(f"{'=' * 110}")

    # 对于每个 Fold，统计 Base 类的 train + val tile 数
    # Base 类 = 该 Fold 中不是 Novel 的类
    print(f"\n{'ID':>3s} {'Class':<22s} {'Ref-Tr':>7s} {'Ref-Val':>7s} "
          f"{'F0-Tr':>7s} {'F0-Val':>7s} {'F1-Tr':>7s} {'F1-Val':>7s} "
          f"{'F2-Tr':>7s} {'F2-Val':>7s} {'Note':>10s}")
    print("-" * 110)

    for cid in range(1, 16):
        ref_name, ref_train, ref_val = REFERENCE_TABLE.get(cid, ("?", 0, 0))
        name = ISAID5I_CATEGORIES.get(cid, ref_name)

        # 识别该类别在哪一个 Fold 是 Novel
        novel_fold = None
        for f in [0, 1, 2]:
            if cid in ISAID5I_FOLDS[f]["novel"]:
                novel_fold = f
                break

        # 获取各 Fold 的 tile 数
        f0_tr = all_counts[0]["train"].get(cid, 0)
        f0_vl = all_counts[0]["val"].get(cid, 0)
        f1_tr = all_counts[1]["train"].get(cid, 0)
        f1_vl = all_counts[1]["val"].get(cid, 0)
        f2_tr = all_counts[2]["train"].get(cid, 0)
        f2_vl = all_counts[2]["val"].get(cid, 0)

        # 判断是否匹配 (训练集: 选择非 Novel Fold 的值)
        # 参考表通常是在某特定 Fold 下的 train/test 数量
        # 我们用所有 Fold 的平均或首个非 Novel 的值来对比
        non_novel_train = []
        non_novel_val = []
        for f in [0, 1, 2]:
            if cid not in ISAID5I_FOLDS[f]["novel"]:
                non_novel_train.append(all_counts[f]["train"].get(cid, 0))
                non_novel_val.append(all_counts[f]["val"].get(cid, 0))

        avg_tr = int(np.mean(non_novel_train)) if non_novel_train else 0
        avg_vl = int(np.mean(non_novel_val)) if non_novel_val else 0

        # 检查参考值是否在实测范围附近 (±30%)
        tr_match = "OK" if ref_train > 0 and abs(avg_tr - ref_train) / ref_train < 0.30 else ""
        vl_match = "OK" if ref_val > 0 and abs(avg_vl - ref_val) / ref_val < 0.30 else ""
        note_str = novel_fold if novel_fold is not None else "?"

        print(f"{cid:>3d} {name:<22s} {ref_train:>7d} {ref_val:>7d} "
              f"{f0_tr:>7d} {f0_vl:>7d} {f1_tr:>7d} {f1_vl:>7d} "
              f"{f2_tr:>7d} {f2_vl:>7d} "
              f"Novel=F{note_str} {tr_match}{vl_match}")

    # ═══ 各 Fold Base/Novel 总结 | Per-Fold Base/Novel Summary ═══
    print(f"\n{'=' * 110}")
    print("Fold Structure Summary | Fold 结构摘要")
    print(f"{'=' * 110}")
    for fold in [0, 1, 2]:
        novel = ISAID5I_FOLDS[fold]["novel"]
        base = ISAID5I_FOLDS[fold]["base"]
        novel_names = [ISAID5I_CATEGORIES[c] for c in novel]
        base_names = [ISAID5I_CATEGORIES[c] for c in base]

        # 统计 Base 类总 tile 数
        tr_total = sum(all_counts[fold]["train"].get(c, 0) for c in base)
        vl_total = sum(all_counts[fold]["val"].get(c, 0) for c in base)
        print(f"\nFold {fold}:")
        print(f"  Base  ({len(base):>2d}): {', '.join(base_names)}")
        print(f"    Train tiles (Base classes): {tr_total}")
        print(f"    Val   tiles (Base classes): {vl_total}")
        print(f"  Novel ({len(novel):>2d}): {', '.join(novel_names)}")

    # ═══ 验证参考表的 Fold 归属 | Verify which fold the reference matches ═══
    print(f"\n{'=' * 110}")
    print("Fold Attribution Check | Fold 归属检查 (参考表可能是某篇论文的Fold 0)")
    print(f"{'=' * 110}")
    for test_fold in [0, 1, 2]:
        match_count = 0
        total_checked = 0
        for cid in range(1, 16):
            ref_name, ref_train, ref_val = REFERENCE_TABLE.get(cid, ("?", 0, 0))
            if ref_train == 0:
                continue
            total_checked += 1
            actual_tr = all_counts[test_fold]["train"].get(cid, 0)
            actual_vl = all_counts[test_fold]["val"].get(cid, 0)
            if abs(actual_tr - ref_train) / ref_train < 0.30:
                match_count += 1
        print(f"  Fold {test_fold}: {match_count}/{total_checked} classes match (+-30%) "
              f"-> {'MATCH' if match_count > total_checked * 0.7 else 'mismatch'}")

    print(f"\n{'=' * 110}")
    print("Done | 完成")
    print(f"{'=' * 110}")


if __name__ == "__main__":
    main()
