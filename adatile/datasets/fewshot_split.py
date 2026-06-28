#!/usr/bin/env python3
"""
iSAID Few-Shot Instance Segmentation 3-Fold Split | 三折 Base/Novel 划分.

按照 PASCAL-5i 风格，将 15 类分为 3 个 Fold，每个 Fold 5 个 Novel 类，其余为 Base。
PASCAL-5i style: 15 classes → 3 folds, 5 Novel + rest Base per fold.

.. warning::
    **类别 ID 体系: 本模块使用 ``ISAID_CATEGORIES`` (标准 iSAID COCO ID)**

    与 ``adatile.utils.label_mapping.ISAID5I_FOLDS`` 使用不同的 ID 体系:

    ====================  ============================  ============================
    模块                    ID 体系                       small_vehicle 的 ID
    ====================  ============================  ============================
    ``fewshot_split.py``   ``ISAID_CATEGORIES``          1
    ``label_mapping.py``   ``ISAID5I_CATEGORIES``        9
    ====================  ============================  ============================

    **规则:**
    - ``--dataset fastsam`` (非标准 debug 路径) → ``ISAID_FEWSHOT_FOLDS`` + ``ISAID_CATEGORIES``
    - ``--dataset isaid5i`` (标准 FSS benchmark) → ``ISAID5I_FOLDS`` + ``ISAID5I_CATEGORIES``
    - **禁止混用**: 两套 ID 体系不可互换！相同类别名对应不同数字 ID。

    论文正式实验只用 ``ISAID5I_FOLDS`` (标准 iSAID-5i benchmark)。
    ``ISAID_FEWSHOT_FOLDS`` 仅用于早期快速验证 (``--dataset fastsam``)。

用法 | Usage::
    from adatile.datasets.fewshot_split import ISAID_FEWSHOT_FOLDS
"""

# ═══════════════════════════════════════════════════════════════════
# 3-Fold Split Definition | 三折划分定义
# ═══════════════════════════════════════════════════════════════════
# 设计原则:
#   1. 每 Fold Novel 类尽量包含不同类型的对象 (Vehicle/Infra/Object)
#   2. small_vehicle (252K instances, 71% of total) 单独放在 Fold 0
#      因其极端主导, 如果分散到多 Fold 会导致所有 Fold 的 Novel 数量失衡
#   3. 每个 Fold 5 个 Novel 类 (Fold 2 含 road 作为占位)
#
# 已知限制: Fold 0 的 Novel 实例数 (262K) 远大于 Fold 1 (49K) 和 Fold 2 (43K)
#   这是 iSAID 数据集本身的特性 (small_vehicle 占 71%)
#   论文将报告 per-fold + 平均, 并在附录中说明类别分布
#   PASCAL-5i 同样存在类别不平衡, 社区接受此做法
#
# 共 15 类, Fold0=5, Fold1=5, Fold2=5 (含 road)
#
# .. warning::
#    以下 ID 使用 ``ISAID_CATEGORIES`` 体系 (1=small_vehicle, 2=large_vehicle, ...)。
#    如需标准 iSAID-5i benchmark, 请使用 ``adatile.utils.label_mapping.ISAID5I_FOLDS``
#    (9=small_vehicle, 8=large_vehicle, ...)。

ISAID_FEWSHOT_FOLDS = {
    0: {  # Fold 0 — small_vehicle + 4 diverse classes
        "novel": [1, 6, 10, 12, 15],
        # small_vehicle (252K), harbor (6K), swimming_pool (2.4K),
        # basketball_court (1K), roundabout (0.4K)
        # Total Novel ≈ 262K instances
        "base":  [2, 3, 4, 5, 7, 8, 9, 13, 14],
    },
    1: {  # Fold 1 — large vehicles + sports + bridge
        "novel": [2, 3, 9, 13, 7],
        # large_vehicle (36K), plane (8K), tennis_court (2.5K),
        # bridge (2K), ground_track_field (0.4K)
        # Total Novel ≈ 49K instances
        "base":  [1, 4, 5, 6, 8, 10, 12, 14, 15],
    },
    2: {  # Fold 2 — ship + storage + tiny objects
        "novel": [5, 4, 14, 8, 11],
        # ship (36K), storage_tank (6K), helicopter (0.6K),
        # soccer_ball_field (0.4K), road (1)
        # Total Novel ≈ 43K instances
        "base":  [1, 2, 3, 6, 7, 9, 10, 12, 13, 15],
    },
}

# 方便访问 | Convenience access
FOLDS = ISAID_FEWSHOT_FOLDS


def get_novel_classes(fold: int) -> list[int]:
    """获取指定 Fold 的 Novel 类别 ID | Get novel class IDs for a fold."""
    return ISAID_FEWSHOT_FOLDS[fold]["novel"]


def get_base_classes(fold: int) -> list[int]:
    """获取指定 Fold 的 Base 类别 ID | Get base class IDs for a fold."""
    return ISAID_FEWSHOT_FOLDS[fold]["base"]


def get_fold_for_class(cls_id: int) -> int | None:
    """查询某个类别属于哪个 Fold 的 Novel 集 | Find which fold a class is novel in."""
    for fold_id, fold_data in ISAID_FEWSHOT_FOLDS.items():
        if cls_id in fold_data["novel"]:
            return fold_id
    return None
