"""
adatile.metrics — 评测指标 | Evaluation Metrics.
===================================================

四个核心评测指标：
- mIoU:  平均交并比 | Mean Intersection over Union
- Dice:  系数 | Dice coefficient
- FPS:   推理速度 | Frames Per Second
- Params: 模型参数统计 | Parameter counting
- COCO:  实例分割 COCO AP | Instance segmentation COCO AP

导出 | Exports:
    compute_miou()      — 多类别 mIoU 计算 | Multi-class mIoU computation
    compute_dice()      — Dice 系数（修复 v1 broadcast bug）| Dice coefficient
    FPSMeter            — FPS 测量器 | FPS meter class
    count_params()      — 参数计数 | Parameter counting
    format_param_count() — 参数格式化 | Parameter count formatting
    COCOInstanceEvaluator — COCO AP 评估器 | COCO AP Evaluator
    connected_components_to_instances — 连通分量分解 | CC decomposition
    instances_to_coco_predictions — 语义掩码转 COCO 预测 | Semantic mask to COCO preds
"""

from adatile.metrics.iou import compute_miou
from adatile.metrics.dice import compute_dice
from adatile.metrics.fps import FPSMeter
from adatile.metrics.params import count_params, format_param_count
from adatile.metrics.coco_eval import (
    COCOInstanceEvaluator,
    connected_components_to_instances,
    instances_to_coco_predictions,
    mask_to_bbox,
)
from adatile.metrics.instance_match import (
    pairwise_iou,
    greedy_match,
    instance_miou,
)

__all__ = [
    "compute_miou",
    "compute_dice",
    "FPSMeter",
    "count_params",
    "format_param_count",
    "COCOInstanceEvaluator",
    "connected_components_to_instances",
    "instances_to_coco_predictions",
    "mask_to_bbox",
    "pairwise_iou",
    "greedy_match",
    "instance_miou",
]
