"""
随机种子工具 | Random Seed Utility.

统一所有实验脚本的随机种子设置。
Unified random seed setup for all experiment scripts.

用法 | Usage::
    from adatile.utils.seed import set_seed
    set_seed(42)
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    固定所有随机种子以保证可复现性。
    Fix all random seeds for reproducibility.

    包括 | Includes:
        - Python random
        - NumPy
        - PyTorch (CPU + CUDA)
        - cuDNN (deterministic mode)

    :param seed: 随机种子值 | Random seed value.
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 确定性 cuDNN (可能稍慢, 但保证可复现)
    # Deterministic cuDNN (slightly slower, but guarantees reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
