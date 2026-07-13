"""
实例生成方法 | Instance Generation Methods.
=============================================

从前景概率图生成实例掩码的多种方法。
Multiple methods for generating instance masks from foreground probability maps.

核心问题 | Core Problem:
    语义前景概率图 → 逐实例掩码 (实例分离)
    Semantic FG probability map → per-instance masks (instance separation)

    连通域方法 (baseline) 天然有 Recall Ceiling: 相邻同类别物体粘连成一个连通块
    → 1 TP + 1 FN → AP 上限被锁死。
    Connected components has an inherent Recall Ceiling: adjacent same-class objects
    merge into one component → 1 TP + 1 FN → AP ceiling is locked.

方法 | Methods:
    1. connected_components  — baseline, 最快但有 Recall Ceiling
    2. watershed_distance     — 距离变换 + 峰值检测 + 分水岭, 擅长分离粘连物体
    3. watershed_gradient     — 概率图梯度 + 分水岭, 利用概率边界的自然谷
    4. dbscan_spatial         — 空间 DBSCAN 聚类, 适合稀疏分布的小目标

用法 | Usage::

    from adatile.metrics.instance_generation import generate_instances

    instances = generate_instances(
        prob_map, method="watershed_distance",
        score_thr=0.3, min_area=16, min_distance=15,
    )
    # → list of {mask: bool[H,W], score: float}
"""

from __future__ import annotations

import numpy as np
from typing import Literal

# 支持的方法 | Supported methods
InstanceMethod = Literal[
    "connected_components",
    "watershed_distance",
    "watershed_gradient",
]


def generate_instances(
    prob_map: np.ndarray,
    method: InstanceMethod = "connected_components",
    score_thr: float = 0.5,
    min_area: int = 16,
    min_distance: int = 12,
    max_instances: int = 100,
) -> list[dict]:
    """
    从前景概率图生成实例列表 (统一入口) | Generate instance list from FG probability map (unified entry).

    :param prob_map: [H, W] float32/float64 ∈ [0, 1] 前景概率 | FG probability.
    :param method: 实例化方法 | Instance generation method.
    :param score_thr: binarization threshold (connected_components) or peak threshold.
    :param min_area: 最小实例面积 (像素) | Minimum instance area in pixels.
    :param min_distance: 最近峰值间距 (watershed 系列) | Minimum distance between peaks.
    :param max_instances: 最大实例数 | Maximum number of instances.
    :return: list of {mask: bool[H,W], score: float}, sorted by score descending.
    """
    prob_map = prob_map.astype(np.float32)

    if method == "connected_components":
        return _connected_components(prob_map, score_thr, min_area, max_instances)
    elif method == "watershed_distance":
        return _watershed_distance(prob_map, score_thr, min_area, min_distance, max_instances)
    elif method == "watershed_gradient":
        return _watershed_gradient(prob_map, score_thr, min_area, min_distance, max_instances)
    else:
        raise ValueError(f"Unknown instance method: {method}. "
                         f"Choose from: connected_components, watershed_distance, watershed_gradient")


# ═══════════════════════════════════════════════════════════════════
# 1. Connected Components (Baseline)
# ═══════════════════════════════════════════════════════════════════

def _connected_components(
    prob: np.ndarray,
    score_thr: float,
    min_area: int,
    max_instances: int,
) -> list[dict]:
    """
    连通域分解 | Connected component decomposition.
    二值化 → 8-连通域标记 → 每块 = 一个实例, 置信度 = 块内概率均值.
    """
    import cv2

    binary = (prob > score_thr).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    instances = []
    for i in range(1, min(num_labels, max_instances + 1)):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            mask = (labels == i)
            score = float(prob[mask].mean())
            instances.append({"mask": mask, "score": score})

    instances.sort(key=lambda x: x["score"], reverse=True)
    return instances[:max_instances]


# ═══════════════════════════════════════════════════════════════════
# 2. Watershed + Distance Transform
# ═══════════════════════════════════════════════════════════════════

def _watershed_distance(
    prob: np.ndarray,
    score_thr: float,
    min_area: int,
    min_distance: int,
    max_instances: int,
) -> list[dict]:
    """
    距离变换 + 分水岭分割 | Distance Transform + Watershed Segmentation.

    算法 | Algorithm:
        1. prob > score_thr → binary foreground mask
        2. cv2.distanceTransform → 每个前景像素到最近背景的距离
        3. 在距离变换图上找局部极大值 → 实例种子 (seeds)
        4. cv2.watershed(负距离图, seeds) → 实例边界沿距离谷地分割

    优势 | Advantage:
        - 自然分离粘连物体: 两物体接触处距离值最小 → watershed 在此分割
        - 物体中心种子最可靠: 距离极大值在物体几何中心
        - 不依赖概率图的精细边界, 只依赖"哪里是前景"
    """
    import cv2

    H, W = prob.shape

    # Step 1: 二值前景 | Binary foreground
    binary = (prob > score_thr).astype(np.uint8)

    # 如果没有前景像素, 返回空 | No FG pixels → empty
    if binary.sum() < min_area:
        return []

    # Step 2: 距离变换 | Distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)  # [H, W] float32

    # Step 3: 在距离变换图上检测局部极大值 | Find local maxima on distance map
    #   这些是实例的几何中心候选 | These are instance center candidates
    from scipy.ndimage import maximum_filter
    local_max = (dist == maximum_filter(dist, size=min_distance * 2 + 1))
    # 只保留前景内的极大值 | Only keep maxima inside FG
    local_max = local_max & (binary > 0)

    # 标记种子 | Label seeds
    seed_labels, n_seeds = _label_connected_components(local_max)

    if n_seeds == 0:
        # 退化为连通域 | Fallback to connected components
        return _connected_components(prob, score_thr, min_area, max_instances)

    if n_seeds == 1:
        # 只有一个种子 → 连通域即可 | Single seed → CC suffices
        return _connected_components(prob, score_thr, min_area, max_instances)

    # Step 4: 分水岭分割 | Watershed segmentation
    #   用负距离图: 物体中心 = 低值 (距离大 → 取负后小), 物体边缘/背景 = 高值
    #   Negative distance: centers = low, edges = high → watershed basins = instances
    markers = seed_labels.astype(np.int32)
    neg_dist = -dist
    # 归一化到 uint8 | Normalize to uint8
    neg_dist_norm = cv2.normalize(neg_dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 转 3 通道 (cv2.watershed 要求) | Convert to 3-channel (cv2.watershed requirement)
    neg_dist_rgb = cv2.cvtColor(neg_dist_norm, cv2.COLOR_GRAY2BGR)

    try:
        ws_labels = cv2.watershed(neg_dist_rgb, markers)
    except cv2.error:
        # watershed 失败 → 退化为连通域 | watershed failed → fallback
        return _connected_components(prob, score_thr, min_area, max_instances)

    # Step 5: 从 watershed 标签提取实例 | Extract instances from watershed labels
    instances = []
    for lbl in range(1, n_seeds + 1):
        mask = (ws_labels == lbl)
        area = mask.sum()
        if area >= min_area:
            score = float(prob[mask].mean())
            instances.append({"mask": mask, "score": score})

    # 如果没有有效的 watershed 实例, 退化为连通域 | No valid watershed → fallback
    if not instances:
        return _connected_components(prob, score_thr, min_area, max_instances)

    instances.sort(key=lambda x: x["score"], reverse=True)
    return instances[:max_instances]


# ═══════════════════════════════════════════════════════════════════
# 3. Watershed + Probability Gradient
# ═══════════════════════════════════════════════════════════════════

def _watershed_gradient(
    prob: np.ndarray,
    score_thr: float,
    min_area: int,
    min_distance: int,
    max_instances: int,
) -> list[dict]:
    """
    概率图梯度 + 分水岭 | Probability Map Gradient + Watershed.

    算法 | Algorithm:
        1. prob > score_thr → binary foreground
        2. 从概率图直接找峰值 (比距离变换更依赖概率质量) → seeds
        3. cv2.Sobel 计算概率图梯度幅值 → 物体边界处梯度大 = watershed "脊"
        4. cv2.watershed(gradient, seeds) → 实例边界沿概率梯度脊分割

    优势 | Advantage:
        - 利用概率图的软梯度信息, 不丢失概率边界细节
        - 对高概率内部 + 低概率边界的物体特别有效
        - 当概率图本身已有较好的实例边界暗示时, 比距离变换更精确
    """
    import cv2

    H, W = prob.shape

    # Step 1: 二值前景 | Binary foreground
    binary = (prob > score_thr).astype(np.uint8)
    if binary.sum() < min_area:
        return []

    # Step 2: 峰值检测 (概率图 + 距离门控) | Peak detection (prob + distance gating)
    from scipy.ndimage import maximum_filter

    # 用概率图直接找峰值 | Find peaks directly on probability map
    prob_peaks = (prob == maximum_filter(prob, size=min_distance * 2 + 1))
    # 只用前景内的峰值 + 概率足够高的 | Only FG peaks with high enough prob
    prob_peaks = prob_peaks & (binary > 0) & (prob > score_thr * 1.2)

    seed_labels, n_seeds = _label_connected_components(prob_peaks)

    if n_seeds <= 1:
        return _connected_components(prob, score_thr, min_area, max_instances)

    # Step 3: 概率图梯度幅值 | Probability gradient magnitude
    grad_x = cv2.Sobel(prob, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(prob, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Step 4: 分水岭 | Watershed on gradient
    markers = seed_labels.astype(np.int32)
    gradient_rgb = cv2.cvtColor(gradient_norm, cv2.COLOR_GRAY2BGR)

    try:
        ws_labels = cv2.watershed(gradient_rgb, markers)
    except cv2.error:
        return _connected_components(prob, score_thr, min_area, max_instances)

    # Step 5: 提取实例 | Extract instances
    instances = []
    for lbl in range(1, n_seeds + 1):
        mask = (ws_labels == lbl)
        area = mask.sum()
        if area >= min_area:
            score = float(prob[mask].mean())
            instances.append({"mask": mask, "score": score})

    if not instances:
        return _connected_components(prob, score_thr, min_area, max_instances)

    instances.sort(key=lambda x: x["score"], reverse=True)
    return instances[:max_instances]


# ═══════════════════════════════════════════════════════════════════
# 辅助函数 | Helpers
# ═══════════════════════════════════════════════════════════════════

def _label_connected_components(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """标记二值掩码中的连通分量 | Label connected components in a binary mask.

    :return: (labels [H,W] int32, n_seeds). labels 中 0=背景, 1..n=种子 | 0=bg, 1..n=seeds.
    """
    import cv2
    # 用 4-连通 (因为种子点通常是孤立的单像素或小团) | 4-connectivity (seeds are isolated)
    num_labels, labels = cv2.connectedComponents(binary_mask.astype(np.uint8), connectivity=4)
    # labels[0] = 背景, labels[1..] = 前景种子 | labels[0]=bg, labels[1..]=seeds
    return labels.astype(np.int32), num_labels - 1
