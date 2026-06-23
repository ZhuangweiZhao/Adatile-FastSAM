"""
类别 ID 映射工具 | Category ID Mapping Utilities.
==================================================

iSAID train 和 val 的原始 COCO JSON 使用不同的 category_id 编号。
此模块提供基于名称的自动映射，保证 per-split 正确性。

iSAID train and val original COCO JSONs use different category_id numbering.
This module provides name-based auto-mapping, ensuring per-split correctness.

关键规则 | Key Rules:
    1. 映射只做一次 (One mapping only)
       - prep_isaid.py fix_annotations() → instances_{split}.json
       - 之后所有代码直接使用 ann["category_id"]，不再二次映射
       - All downstream code uses ann["category_id"] directly, no double-mapping

    2. 目标编码为标准 ISAID_CATEGORIES (1-15)
       - Target encoding: standard ISAID_CATEGORIES (1-15)

用法 | Usage::
    from adatile.utils.label_mapping import build_mapping, get_category_id, ISAID_CATEGORIES

    # 构建 per-split 映射 | Build per-split mapping
    mapping = build_mapping(original_categories)
    code_id = mapping[original_id]

    # 或直接获取（如果已是标准 ID 则透传）| Or get directly (pass-through if already standard)
    code_id = get_category_id(ann["category_id"])
"""

from __future__ import annotations
from collections import OrderedDict
from typing import Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════
# 标准 ISAID 15 个类别 | Standard ISAID 15 Categories
# ═══════════════════════════════════════════════════════════════════

ISAID_CATEGORIES: OrderedDict[str, str] = OrderedDict([
    (1, "small_vehicle"),
    (2, "large_vehicle"),
    (3, "plane"),
    (4, "storage_tank"),
    (5, "ship"),
    (6, "harbor"),
    (7, "ground_track_field"),
    (8, "soccer_ball_field"),
    (9, "tennis_court"),
    (10, "swimming_pool"),
    (11, "road"),
    (12, "basketball_court"),
    (13, "bridge"),
    (14, "helicopter"),
    (15, "roundabout"),
])

# 名称 → ID | Name → ID
_NAME_TO_ID: Dict[str, int] = {name: tid for tid, name in ISAID_CATEGORIES.items()}
# ID → 名称 | ID → Name
_ID_TO_NAME: Dict[int, str] = dict(ISAID_CATEGORIES)
_ID_TO_NAME[0] = "background"

# 有效的类别 ID 范围 | Valid category ID range
VALID_CAT_IDS: set = set(ISAID_CATEGORIES.keys())


# ═══════════════════════════════════════════════════════════════════
# 原始名称 → 标准名称 别名表 | Original Name → Standard Name Alias Table
# ═══════════════════════════════════════════════════════════════════

_ORIGINAL_NAME_ALIASES: Dict[str, str] = {
    # 精确匹配 + 大小写变体 | Exact matches + case variants
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
    "baseball_diamond": "basketball_court",     # DOTA → iSAID: 棒球场→篮球场
    "bridge": "bridge",
    "helicopter": "helicopter",
    "roundabout": "roundabout",
}


# ═══════════════════════════════════════════════════════════════════
# 已废弃：旧的硬编码映射（仅对 train 有效，不要使用）
# Deprecated: old hardcoded mapping (train-only, DO NOT USE)
# ═══════════════════════════════════════════════════════════════════

_DEPRECATED_ACTUAL_TO_CODE_ID: Dict[int, int] = {
    1: 4, 2: 2, 3: 1, 4: 3, 5: 5, 6: 10, 7: 6, 8: 9,
    9: 7, 10: 8, 11: 11, 12: 13, 13: 12, 14: 15, 15: 14,
}
"""旧映射，仅对 train 原始编号有效。val 使用不同的原始编号，此表对 val 完全错误。
Old mapping, only valid for train original numbering. Completely wrong for val."""


# ═══════════════════════════════════════════════════════════════════
# 公共 API | Public API
# ═══════════════════════════════════════════════════════════════════

def build_mapping(
    original_categories: List[dict],
) -> Tuple[Dict[int, int], List[Tuple[int, str]]]:
    """
    根据原始 JSON 的 categories 列表构建 原始ID → ISAID_ID 映射。
    Build original_id → ISAID_ID mapping from original JSON categories list.

    通过名称匹配（不区分大小写 + 常见别名）。
    Via name matching (case-insensitive + common aliases).

    Example:
        >>> cats = [{"id": 1, "name": "plane"}, {"id": 2, "name": "ship"}]
        >>> mapping, unmatched = build_mapping(cats)
        >>> mapping
        {1: 3, 2: 5}   # plane→3, ship→5

    :param original_categories: COCO JSON 的 "categories" 列表。 List of category dicts from COCO JSON, each with "id" and "name".
    :type original_categories: List[dict]

    :return: (mapping, unmatched): mapping:   {original_id: isaid_id}  映射成功的条目。 unmatched: [(original_id, name)]    无法匹配的条目（名称不在别名表中）。
    :rtype: Tuple[Dict[int, int], List[Tuple[int, str]]]
    """
    mapping: Dict[int, int] = {}
    unmatched: List[Tuple[int, str]] = []

    for cat in original_categories:
        orig_id = cat["id"]
        orig_name = cat["name"].strip()

        # 规范化名称：下划线→空格，转小写 | Normalize: underscore→space, lowercase
        normalized = orig_name.replace("_", " ").lower().strip()

        # 查找别名 | Look up alias
        std_name = _ORIGINAL_NAME_ALIASES.get(normalized)
        if std_name is None:
            # 尝试原始名称 | Try original name as-is
            std_name = _ORIGINAL_NAME_ALIASES.get(orig_name.lower().strip())

        if std_name is None:
            unmatched.append((orig_id, orig_name))
            continue

        target_id = _NAME_TO_ID.get(std_name)
        if target_id is None:
            unmatched.append((orig_id, orig_name))
            continue

        mapping[orig_id] = target_id

    return mapping, unmatched


def get_category_id(category_id: int) -> int:
    """
    安全获取类别 ID（透传 + 验证）。| Safe category ID getter (pass-through + validation).

    如果 category_id 已在 0-15 范围内，直接返回（已经是标准 ISAID ID）。
    否则返回 0（背景/无效）。
    不再进行任何映射 — 映射只应在数据预处理阶段完成。

    If category_id is already in 0-15 range, return directly (already standard ISAID ID).
    Otherwise return 0 (background/invalid).
    No mapping is done — mapping should only happen during data preprocessing.

    :param category_id: 类别 ID（可能来自 annotation 或 mask）。 Category ID (from annotation or mask).
    :type category_id: int

    :return: int: 有效的类别 ID (0-15) | Valid category ID (0-15).
    :rtype: int
    """
    if category_id in VALID_CAT_IDS or category_id == 0:
        return category_id
    return 0


def is_valid_category(category_id: int) -> bool:
    """检查是否是有效的 ISAID 类别 ID (0-15)。| Check if valid ISAID category ID."""
    return category_id == 0 or category_id in VALID_CAT_IDS


def get_category_name(category_id: int) -> str:
    """获取类别名称 | Get category name."""
    return _ID_TO_NAME.get(category_id, f"unknown_{category_id}")
