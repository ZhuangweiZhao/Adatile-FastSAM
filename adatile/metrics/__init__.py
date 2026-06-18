"""
adatile.metrics — 评测指标 | Evaluation Metrics.
===================================================

四个核心评测指标：
- mIoU:  平均交并比 | Mean Intersection over Union
- Dice:  系数 | Dice coefficient
- FPS:   推理速度 | Frames Per Second
- Params: 模型参数统计 | Parameter counting

导出 | Exports:
    compute_miou()      — 多类别 mIoU 计算 | Multi-class mIoU computation
    compute_dice()      — Dice 系数（修复 v1 broadcast bug）| Dice coefficient
    FPSMeter            — FPS 测量器 | FPS meter class
    count_params()      — 参数计数 | Parameter counting
    format_param_count() — 参数格式化 | Parameter count formatting
"""

from adatile.metrics.iou import compute_miou
from adatile.metrics.dice import compute_dice
from adatile.metrics.fps import FPSMeter
from adatile.metrics.params import count_params, format_param_count

__all__ = [
    "compute_miou",
    "compute_dice",
    "FPSMeter",
    "count_params",
    "format_param_count",
]
