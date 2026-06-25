"""
Environment Info Utility — 环境信息收集工具.
==============================================

收集并保存当前运行环境的关键信息，用于实验可复现性。
Collects and saves key runtime environment info for experiment reproducibility.

Captures:
    - Python version
    - PyTorch version
    - CUDA version (if available)
    - GPU model name (if available)
    - OS / platform info
    - Key package versions

用法 | Usage::
    from adatile.utils.env import get_env_info
    info = get_env_info()
    # {"python": "3.13.11", "pytorch": "2.12.0", "cuda": "None", "gpu": "N/A", ...}
"""

from __future__ import annotations

import os
import sys
import platform
import json
from typing import Any


def get_env_info() -> dict[str, Any]:
    """
    收集当前运行环境的关键信息 | Collect key runtime environment information.

    Returns a dict with keys:
        - python: Python version string
        - pytorch: PyTorch version string
        - cuda: CUDA version (or "N/A" if not available)
        - gpu_name: GPU model name(s), comma-separated (or "N/A")
        - gpu_count: Number of GPUs available
        - os: Operating system description
        - hostname: Machine hostname
        - packages: dict of key package versions

    :returns: Environment info dict.
    :rtype: dict[str, Any]
    """
    info: dict[str, Any] = {}

    # ── Python 版本 | Python Version ────────────────────────────
    info["python"] = sys.version.split()[0]  # e.g. "3.13.11"
    info["python_full"] = sys.version        # full version string

    # ── PyTorch 版本 | PyTorch Version ──────────────────────────
    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda"] = torch.version.cuda if torch.version.cuda else "N/A"
        info["cudnn"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"

        # GPU 信息 | GPU Info
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            gpu_names = []
            for i in range(torch.cuda.device_count()):
                gpu_names.append(torch.cuda.get_device_name(i))
            info["gpu_name"] = ", ".join(gpu_names)
        else:
            info["gpu_count"] = 0
            info["gpu_name"] = "N/A (CPU-only)"
    except ImportError:
        info["pytorch"] = "N/A"
        info["cuda"] = "N/A"
        info["cudnn"] = "N/A"
        info["gpu_count"] = 0
        info["gpu_name"] = "N/A"

    # ── 操作系统 | Operating System ────────────────────────────
    info["os"] = platform.platform()
    info["hostname"] = platform.node()

    # ── 关键包版本 | Key Package Versions ──────────────────────
    pkg_versions: dict[str, str] = {}
    _key_packages = [
        "numpy", "cv2", "PIL", "yaml", "ultralytics",
        "shapely", "tifffile", "matplotlib",
    ]
    for pkg in _key_packages:
        try:
            mod = __import__(pkg)
            if hasattr(mod, "__version__"):
                pkg_versions[pkg] = str(mod.__version__)
            elif pkg == "cv2":
                pkg_versions["opencv"] = str(getattr(mod, "__version__", "N/A"))
        except ImportError:
            pkg_versions[pkg] = "N/A"
    info["packages"] = pkg_versions

    return info


def save_env_info(filepath: str | os.PathLike) -> dict[str, Any]:
    """
    收集环境信息并保存为 JSON 文件 | Collect environment info and save as JSON.

    :param filepath: 保存路径 | Save path (e.g. "runs/exp/env_info.json").
    :type filepath: str | os.PathLike

    :returns: Same dict as get_env_info().
    :rtype: dict[str, Any]
    """
    info = get_env_info()
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


def env_info_to_string(info: dict[str, Any] | None = None) -> str:
    """
    将环境信息格式化为人类可读的多行字符串 | Format env info as human-readable multi-line string.

    :param info: get_env_info() 返回的字典。None → 自动收集。
                 Dict from get_env_info(). None → auto-collect.
    :type info: dict[str, Any] | None

    :returns: Formatted string.
    :rtype: str
    """
    if info is None:
        info = get_env_info()

    lines = [
        f"Python:   {info.get('python', 'N/A')}",
        f"PyTorch:  {info.get('pytorch', 'N/A')}",
        f"CUDA:     {info.get('cuda', 'N/A')}",
        f"cuDNN:    {info.get('cudnn', 'N/A')}",
        f"GPU:      {info.get('gpu_name', 'N/A')} (×{info.get('gpu_count', 0)})",
        f"OS:       {info.get('os', 'N/A')}",
        f"Host:     {info.get('hostname', 'N/A')}",
    ]
    return "\n".join(lines)
