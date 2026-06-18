"""
参数计数 | Parameter counting utilities.
===========================================

统计模型的总参数、可训练参数、冻结参数，支持逐模块分解。
Count total, trainable, frozen parameters with per-module breakdown.
"""

from __future__ import annotations

import torch.nn as nn


def count_params(model: nn.Module) -> dict[str, int | dict[str, int]]:
    """
    统计模型参数 | Count model parameters.

    遍历模型的所有参数，分别统计：
    Traverses all model parameters, counting:

    - total:     总参数量 | Total parameters
    - trainable: 可训练参数（requires_grad=True）| Trainable parameters
    - frozen:    冻结参数（requires_grad=False）| Frozen parameters
    - per_module: 每个子模块的参数字典 | Per-submodule parameter dict

    Args:
        model: PyTorch 模型 | PyTorch model.

    Returns:
        dict with keys: "total", "trainable", "frozen", "per_module"
    """
    total = 0
    trainable = 0
    per_module: dict[str, int] = {}

    for name, param in model.named_parameters():
        n = param.numel()
        total += n

        if param.requires_grad:
            trainable += n

        # 提取顶级模块名 | Extract top-level module name
        module_name = name.split(".")[0]
        per_module[module_name] = per_module.get(module_name, 0) + n

    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "per_module": per_module,
    }


def format_param_count(n: int) -> str:
    """
    格式化参数数量为可读字符串 | Format parameter count to human-readable string.

    例 | Examples:
        1_200_000 → "1.20M"
        45_300    → "45.30K"
        999       → "999"
        0         → "0"

    Args:
        n: 参数数量 | Parameter count.

    Returns:
        str: 格式化后的字符串 | Formatted string.
    """
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return str(n)
