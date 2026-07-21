"""
缺陷 Prompt 提取器 | Defect Prompt Extractor.
===============================================

使用经典图像处理方法从工业表面图像中自动提取缺陷 Prompt。
无需训练 — 纯经典 CV + 可解释物理特征。
Classical CV methods extract defect prompts from industrial surface images.
Training-free — pure classical CV with interpretable physical features.

Prompt 来源 | Prompt Sources:
    1. FFT 高频响应 | FFT High-Frequency Response
       → 缺陷在频域表现为异常高频能量 → 高斯高通滤波 → 逆FFT → 热力图
       → Defects appear as anomalous high-freq energy → Gaussian HPF → iFFT → heatmap

    2. 梯度异常 | Gradient Anomaly
       → 缺陷边界产生强梯度响应 → Sobel 梯度幅值 → 热力图
       → Defect boundaries produce strong gradient → Sobel magnitude → heatmap

    3. 纹理异常 | Texture Anomaly
       → 缺陷破坏表面纹理一致性 → 局部标准差/LBP → 热力图
       → Defects disrupt surface texture → local std / LBP → heatmap

    4. 多源融合 | Multi-Source Fusion
       → 加权融合以上热力图 → 自适应阈值 → 连通分量 → Box Prompts
       → Weighted fusion of heatmaps → adaptive threshold → CC → box prompts

用法 | Usage::

    from adatile.prompt.defect_prompts import DefectPromptExtractor

    extractor = DefectPromptExtractor()
    boxes, heatmaps = extractor(image_np)  # boxes: [[x1,y1,x2,y2], ...]
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from scipy import ndimage


@dataclass
class DefectHeatmaps:
    """多源缺陷热力图 | Multi-source defect heatmaps."""
    fft: np.ndarray | None = None          # FFT高频响应 | FFT high-freq response
    gradient: np.ndarray | None = None     # 梯度幅值 | Gradient magnitude
    texture: np.ndarray | None = None      # 纹理异常 | Texture anomaly
    fused: np.ndarray | None = None        # 融合热力图 | Fused heatmap

    def as_dict(self) -> dict[str, np.ndarray]:
        d = {}
        if self.fft is not None: d["fft"] = self.fft
        if self.gradient is not None: d["gradient"] = self.gradient
        if self.texture is not None: d["texture"] = self.texture
        if self.fused is not None: d["fused"] = self.fused
        return d


class DefectPromptExtractor:
    """
    缺陷 Prompt 提取器 — 经典CV → 缺陷热力图 → Box/Point Prompts.
    Defect Prompt Extractor — classical CV → defect heatmaps → box/point prompts.

    完全无需训练, 所有参数物理可解释。
    Completely training-free, all parameters physically interpretable.

    Parameters
    ----------
    fft_cutoff : float
        FFT 高通滤波截止频率 (0.0~1.0, 相对于 Nyquist). 越小越敏感.
        FFT high-pass cutoff frequency (relative to Nyquist). Lower = more sensitive.
    gradient_kernel : int
        Sobel 梯度核大小 | Sobel kernel size (3 or 5).
    texture_window : int
        局部纹理窗口大小 | Local texture window size.
    fusion_weights : tuple[float, float, float]
        (w_fft, w_gradient, w_texture) 融合权重 | Fusion weights.
    nms_kernel : int
        NMS 核大小 (抑制相邻重复 prompt) | NMS kernel size (suppress adjacent duplicate prompts).
    min_box_area : int
        最小 box 面积 (像素) | Minimum box area in pixels.
    max_boxes : int
        每张图最大 box prompt 数量 | Maximum box prompts per image.
    """

    def __init__(
        self,
        fft_cutoff: float = 0.08,
        gradient_kernel: int = 3,
        texture_window: int = 7,
        fusion_weights: tuple[float, float, float] = (0.35, 0.35, 0.30),
        nms_kernel: int = 15,
        min_box_area: int = 16,
        max_boxes: int = 12,
    ):
        self.fft_cutoff = fft_cutoff
        self.gradient_kernel = gradient_kernel
        self.texture_window = texture_window
        self.fusion_weights = fusion_weights
        self.nms_kernel = nms_kernel
        self.min_box_area = min_box_area
        self.max_boxes = max_boxes

    # ═══════════════════════════════════════════════════════════════
    # Prompt Source 1: FFT 高频响应 | High-Frequency Response
    # ═══════════════════════════════════════════════════════════════

    def extract_fft(self, gray: np.ndarray) -> np.ndarray:
        """
        FFT 高频异常检测 | FFT high-frequency anomaly detection.

        工业表面 (金属、瓷砖) 的正常纹理集中在低频，
        缺陷 (裂纹、划痕) 表现为异常高频能量。
        Normal industrial surface textures concentrate in low frequencies;
        defects (cracks, scratches) appear as anomalous high-freq energy.

        :param gray: [H, W] 灰度图 | grayscale image, float32 [0,1].
        :return: [H, W] 高频异常热力图 | high-freq anomaly heatmap, [0,1].
        """
        H, W = gray.shape
        # FFT → 频谱 | FFT → spectrum
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        # 高斯高通滤波 | Gaussian high-pass filter
        crow, ccol = H // 2, W // 2
        y, x = np.ogrid[-crow:H - crow, -ccol:W - ccol]
        d2 = x * x + y * y
        # 截止频率 | cutoff frequency (relative to image size)
        cutoff_d = self.fft_cutoff * min(H, W)
        hpf = 1.0 - np.exp(-d2 / (2.0 * cutoff_d * cutoff_d))

        # 滤波 + 逆FFT | Filter + iFFT
        fshift_filtered = fshift * hpf
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        anomaly = np.abs(img_back)

        # 鲁棒归一化 (percentile clip 抑制噪声拖尾) | Robust normalize (percentile clip)
        if anomaly.max() > anomaly.min():
            p_low, p_high = np.percentile(anomaly, [2, 98])
            if p_high > p_low:
                anomaly = np.clip(anomaly, p_low, p_high)
                anomaly = (anomaly - p_low) / (p_high - p_low + 1e-8)
            else:
                anomaly = (anomaly - anomaly.min()) / (anomaly.max() - anomaly.min() + 1e-8)
        return anomaly.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════
    # Prompt Source 2: 梯度异常 | Gradient Anomaly
    # ═══════════════════════════════════════════════════════════════

    def extract_gradient(self, gray: np.ndarray) -> np.ndarray:
        """
        Sobel 梯度幅值 → 缺陷边缘响应 | Sobel gradient → defect edge response.

        缺陷与正常表面之间存在强度变化 → 梯度响应。
        Intensity changes between defect and normal surface → gradient response.

        :param gray: [H, W] 灰度图 | grayscale image, float32 [0,1].
        :return: [H, W] 梯度热力图 | gradient heatmap, [0,1].
        """
        # 转为 uint8 用于 Sobel | Convert to uint8 for Sobel
        gray_u8 = (gray * 255).astype(np.uint8)
        ksize = self.gradient_kernel
        gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=ksize)
        mag = np.sqrt(gx ** 2 + gy ** 2)

        # 鲁棒归一化 (percentile clip 抑制噪声拖尾) | Robust normalize (percentile clip)
        if mag.max() > mag.min():
            p_low, p_high = np.percentile(mag, [2, 98])
            if p_high > p_low:
                mag = np.clip(mag, p_low, p_high)
                mag = (mag - p_low) / (p_high - p_low + 1e-8)
            else:
                mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        return mag.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════
    # Prompt Source 3: 纹理异常 | Texture Anomaly
    # ═══════════════════════════════════════════════════════════════

    def extract_texture(self, gray: np.ndarray) -> np.ndarray:
        """
        局部标准差 → 纹理一致性异常 | Local std → texture consistency anomaly.

        正常表面纹理均匀 (低 std), 缺陷区域破坏均匀性 (高 std)。
        Normal surface has uniform texture (low std), defects disrupt it (high std).

        :param gray: [H, W] 灰度图 | grayscale image, float32 [0,1].
        :return: [H, W] 纹理异常热力图 | texture anomaly heatmap, [0,1].
        """
        k = self.texture_window
        # 局部均值 | Local mean
        local_mean = cv2.blur(gray, (k, k))
        # 局部方差 | Local variance
        local_sq_mean = cv2.blur(gray * gray, (k, k))
        local_var = np.maximum(local_sq_mean - local_mean * local_mean, 0)
        local_std = np.sqrt(local_var)

        # 鲁棒归一化 (percentile clip 抑制噪声拖尾) | Robust normalize (percentile clip)
        if local_std.max() > local_std.min():
            p_low, p_high = np.percentile(local_std, [2, 98])
            if p_high > p_low:
                local_std = np.clip(local_std, p_low, p_high)
                local_std = (local_std - p_low) / (p_high - p_low + 1e-8)
            else:
                local_std = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-8)
        return local_std.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════
    # 多源融合 + Prompt 生成 | Multi-Source Fusion + Prompt Generation
    # ═══════════════════════════════════════════════════════════════

    def fuse_heatmaps(self, heatmaps: DefectHeatmaps, weights: tuple = None) -> np.ndarray:
        """
        加权融合多源热力图 | Weighted fusion of multi-source heatmaps.

        :param heatmaps: 多源热力图 | multi-source heatmaps.
        :param weights: (w_fft, w_gradient, w_texture), None → 使用默认.
        :return: [H, W] 融合热力图 | fused heatmap, [0,1].
        """
        if weights is None:
            weights = self.fusion_weights
        w_fft, w_grad, w_tex = weights

        fused = np.zeros_like(heatmaps.fft)
        total_w = 0.0
        if heatmaps.fft is not None:
            fused += w_fft * heatmaps.fft; total_w += w_fft
        if heatmaps.gradient is not None:
            fused += w_grad * heatmaps.gradient; total_w += w_grad
        if heatmaps.texture is not None:
            fused += w_tex * heatmaps.texture; total_w += w_tex

        if total_w > 0:
            fused /= total_w

        # 高斯平滑减少噪声 | Gaussian smooth to reduce noise
        fused = cv2.GaussianBlur(fused, (5, 5), 1.0)

        if fused.max() > fused.min():
            fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
        return fused.astype(np.float32)

    def heatmap_to_boxes(
        self,
        heatmap: np.ndarray,
        nms_kernel: int | None = None,
        min_area: int | None = None,
        max_boxes: int | None = None,
        adaptive_thresh: bool = True,
    ) -> np.ndarray:
        """
        热力图 → Bounding Box Prompts | Heatmap → bounding box prompts.

        自适应阈值 → 连通分量 → 面积过滤 → NMS → Top-K.
        Adaptive threshold → connected components → area filter → NMS → Top-K.

        :param heatmap: [H, W] 融合热力图 | fused heatmap, [0,1].
        :param nms_kernel: NMS 核大小 | NMS kernel size.
        :param min_area: 最小 box 面积 | minimum box area.
        :param max_boxes: 最大 box 数量 | maximum box count.
        :param adaptive_thresh: 自适应阈值 (Otsu) vs 固定 0.5.
        :return: [N, 4] boxes in xyxy format, sorted by confidence descending.
        """
        if nms_kernel is None: nms_kernel = self.nms_kernel
        if min_area is None: min_area = self.min_box_area
        if max_boxes is None: max_boxes = self.max_boxes

        h, w = heatmap.shape

        # ── 阈值化 | Thresholding ──
        heat_u8 = (heatmap * 255).astype(np.uint8)
        if adaptive_thresh:
            thresh_val = cv2.threshold(heat_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
            # Otsu 可能阈值过高, 取 Otsu*0.6 作为下限 | Otsu may be too high, use 0.6× as floor
            thresh_val = max(thresh_val * 0.5, 15)
        else:
            thresh_val = 64  # ~0.25 in [0,255]
        _, binary = cv2.threshold(heat_u8, int(thresh_val), 255, cv2.THRESH_BINARY)

        # ── 形态学闭合 | Morphological close ──
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # ── 连通分量 | Connected components ──
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        boxes = []
        for i in range(1, num_labels):  # skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x1 = stats[i, cv2.CC_STAT_LEFT]
            y1 = stats[i, cv2.CC_STAT_TOP]
            x2 = x1 + stats[i, cv2.CC_STAT_WIDTH]
            y2 = y1 + stats[i, cv2.CC_STAT_HEIGHT]
            # 区域内热力图均值作为置信度 | Mean heatmap value in region as confidence
            region_heat = heatmap[y1:y2, x1:x2].mean()
            boxes.append([x1, y1, x2, y2, region_heat])

        if not boxes:
            return np.zeros((0, 4), dtype=np.float32)

        boxes = np.array(boxes)

        # ── NMS (基于热力图置信度) | NMS by heatmap confidence ──
        boxes = self._nms_boxes(boxes, nms_kernel)

        # ── Top-K ──
        boxes = boxes[:max_boxes]

        return boxes[:, :4].astype(np.float32)  # [N, 4] xyxy

    def _nms_boxes(self, boxes: np.ndarray, kernel: int) -> np.ndarray:
        """
        基于空间重叠的简单 NMS — 保留高置信度 box, 抑制低置信度重叠 box.
        Simple spatial NMS — keep high-confidence box, suppress overlapping lower ones.

        :param boxes: [N, 5] (x1, y1, x2, y2, score).
        :param kernel: 抑制半径 (像素) | suppression radius in pixels.
        :return: [M, 5] suppressed boxes.
        """
        if len(boxes) <= 1:
            return boxes

        # 按置信度降序 | Sort by confidence descending
        idx = np.argsort(boxes[:, 4])[::-1]
        boxes = boxes[idx]

        keep = []
        suppressed = np.zeros(len(boxes), dtype=bool)

        for i in range(len(boxes)):
            if suppressed[i]:
                continue
            keep.append(i)
            # 计算与其他 box 的中心距离 | Compute center distance to other boxes
            cx_i = (boxes[i, 0] + boxes[i, 2]) / 2
            cy_i = (boxes[i, 1] + boxes[i, 3]) / 2
            for j in range(i + 1, len(boxes)):
                if suppressed[j]:
                    continue
                cx_j = (boxes[j, 0] + boxes[j, 2]) / 2
                cy_j = (boxes[j, 1] + boxes[j, 3]) / 2
                dist = np.sqrt((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2)
                if dist < kernel:
                    suppressed[j] = True

        return boxes[keep]

    # ═══════════════════════════════════════════════════════════════
    # 主入口 | Main Entry
    # ═══════════════════════════════════════════════════════════════

    def __call__(
        self,
        image: np.ndarray,
        sources: tuple[str, ...] = ("fft", "gradient", "texture"),
        return_heatmaps: bool = False,
    ) -> tuple[np.ndarray, DefectHeatmaps | None]:
        """
        从单张工业图像提取缺陷 Prompt Boxes.
        Extract defect prompt boxes from a single industrial image.

        :param image: [H, W, 3] RGB 图像, float32 [0,1] 或 uint8 [0,255].
        :param sources: 使用的 prompt 源 | which prompt sources to use.
        :param return_heatmaps: 是否返回热力图 | whether to return heatmaps.
        :return:
            boxes:    [N, 4] xyxy format, or empty (0,4).
            heatmaps: DefectHeatmaps (if return_heatmaps=True), else None.
        """
        # ── 预处理: RGB → Gray | Preprocess: RGB → Gray ──
        if image.dtype == np.float32 and image.max() <= 1.0:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        heatmaps = DefectHeatmaps()

        # ── 逐源提取 | Extract each source ──
        if "fft" in sources:
            heatmaps.fft = self.extract_fft(gray)
        if "gradient" in sources:
            heatmaps.gradient = self.extract_gradient(gray)
        if "texture" in sources:
            heatmaps.texture = self.extract_texture(gray)

        # ── 融合 | Fusion ──
        heatmaps.fused = self.fuse_heatmaps(heatmaps)

        # ── 热力图 → Boxes | Heatmap → Boxes ──
        boxes = self.heatmap_to_boxes(heatmaps.fused)

        return boxes, heatmaps if return_heatmaps else None


# ═══════════════════════════════════════════════════════════════════
# 便捷函数 | Convenience Functions
# ═══════════════════════════════════════════════════════════════════

def extract_defect_boxes(
    image: np.ndarray,
    sources: tuple[str, ...] = ("fft", "gradient", "texture"),
    fft_cutoff: float = 0.08,
    max_boxes: int = 12,
) -> np.ndarray:
    """
    快速提取缺陷 Box Prompts (单行调用).
    Quick defect box prompt extraction (one-liner).

    :param image: [H, W, 3] RGB image.
    :param sources: prompt 源 | prompt sources.
    :param fft_cutoff: FFT 截止频率 | FFT cutoff.
    :param max_boxes: 最大 box 数 | max boxes.
    :return: [N, 4] boxes in xyxy format.
    """
    extractor = DefectPromptExtractor(
        fft_cutoff=fft_cutoff,
        max_boxes=max_boxes,
    )
    boxes, _ = extractor(image, sources=sources, return_heatmaps=False)
    return boxes
