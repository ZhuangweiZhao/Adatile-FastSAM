"""
FDE — Frequency-aware Defect Enhancement | 频率感知缺陷增强.
=============================================================

空间域高频增强 (replaces FFT version that failed to learn).
Spatial-domain high-frequency enhancement (replaces FFT version).

核心思想 | Core Idea:
    工业缺陷本质是高频纹理异常。avg_pool 提取低频 (平滑背景),
    x - avg_pool(x) = 高频残差 (纹理+边缘) → 可学习增强。
    Industrial defects are fundamentally high-frequency texture anomalies.
    avg_pool extracts low-freq (smooth BG), x - avg_pool(x) = high-freq
    residual (texture + edges) → learnable boost per channel.

为什么空间域替代 FFT | Why Spatial-domain Replaces FFT:
    FFT 版本的梯度路径: loss → decoder → iFFT → 幅度/相位 → 剖面插值 → profile
    路径太长, 经过复数域拆分后梯度信号稀薄, 剖面几乎不学习。
    FFT version gradient path: loss → decoder → iFFT → mag/phase → profile
    interp → profile. Too long — gradient vanishes after complex decomposition.

    空间域版本的梯度路径: loss → decoder → (x + α·high·weight) → x
    只有一次减法和乘法, 梯度直接流入 x, 权重和 α。
    Spatial version gradient path: loss → decoder → (x + α·high·weight) → x
    Single subtraction + multiplication, gradient flows directly.

实现 | Implementation:
    1. avg_pool(x, kernel_size=k) → 提取低频 | Extract low-freq
    2. high = x - low → 高频残差 | High-freq residual
    3. result = x + α × high × weight_ch → 可学习通道增强
    4. 多尺度: 小核 (3) = 细纹理, 中核 (7) = 中等纹理, 大核 (15) = 背景抑制
       Multi-scale: small kernel (3)=fine texture, mid (7)=coarse, large (15)=BG suppress

参数量 | Params: ~C × num_scales (e.g., 960 × 3 = 2.9K for P3)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqDefectEnhance(nn.Module):
    """
    空间域高频增强 | Spatial-domain High-Frequency Enhancement.

    多尺度 Difference-of-Pooling: 用不同 kernel_size 的 avg_pool
    提取不同尺度的低频分量，高频残差 = x - low，可学习增强。
    Multi-scale Difference-of-Pooling: avg_pool with different
    kernel_sizes extracts low-freq at different scales,
    high-freq residual = x - low, learnable channel-wise boost.

    Parameters
    ----------
    channels : int
        输入通道数 | Input channels.
    kernel_sizes : list[int]
        多尺度池化核大小 | Multi-scale pooling kernel sizes.
        小核提取细高频 (纹理), 大核抑制低频 (背景平滑).
        Small kernel = fine high-freq (texture), large kernel = BG suppression.
        Default: [3, 7, 15].
    alpha_init : float
        每个尺度的初始增强强度 | Initial enhancement strength per scale.
        小正数 → 初始接近恒等映射。
        Small positive → initially near-identity.
    """

    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int, ...] = (3, 7, 15),
        alpha_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)

        # ── 每个尺度的增强强度 (可学习) | Enhancement strength per scale (learnable) ──
        # 初始化为小正数 → 训练早期梯度小但非零
        # Init to small positive → small but non-zero gradient early in training
        self.alphas = nn.Parameter(
            torch.full((self.num_scales,), alpha_init)
        )  # [S]

        # ── 每个尺度、每个通道的增强权重 | Per-scale, per-channel enhancement weight ──
        # 初始化为 1.0 → 所有通道平等增强
        # Init to 1.0 → all channels boosted equally
        # Shape: [S, C, 1, 1]
        self.weights = nn.Parameter(
            torch.ones(self.num_scales, channels, 1, 1)
        )

        self._init_params_count()

    def _init_params_count(self) -> None:
        n = sum(p.numel() for p in self.parameters())
        from adatile.logging import get_logger
        get_logger("rectify.fde").log_info(
            "fde/init",
            f"FDE(spatial, ch={self.channels}, ks={list(self.kernel_sizes)}): "
            f"{n/1e3:.1f}K params",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        多尺度空间域高频增强 | Multi-scale spatial high-frequency enhancement.

        :param x: [B, C, H, W] 输入特征图 | Input feature map.
        :return: [B, C, H, W] 增强后特征 (同形状) | Enhanced features (same shape).

        梯度路径 | Gradient Path:
            result = x + Σ_s α_s × (x - pool_s(x)) × w_s
            ∂result/∂x = 1 + Σ_s α_s × w_s (在 pool 梯度之外)
            ∂result/∂w_s = α_s × (x - pool_s(x))
            ∂result/∂α_s = (x - pool_s(x)) × w_s
            — 全部直接可导, 无复数域变换.
        """
        result = x

        for s in range(self.num_scales):
            ks = self.kernel_sizes[s]

            # ── 1. avg_pool 提取低频 | Extract low-freq via avg_pool ──
            low = F.avg_pool2d(x, kernel_size=ks, stride=1, padding=ks // 2)

            # ── 2. 高频残差 | High-freq residual ──
            high = x - low  # [B, C, H, W]

            # ── 3. 可学习增强 | Learnable enhancement ──
            # 约束 alpha 在 [0, 2] 防止过度增强 | Clamp alpha to [0, 2]
            alpha = torch.clamp(self.alphas[s], 0.0, 2.0)
            result = result + alpha * high * self.weights[s]  # [B, C, H, W]

        return result

    def get_channel_weights_mean(self) -> torch.Tensor:
        """
        获取平均通道增强权重 (用于可解释性) | Get mean channel enhancement weights.

        :return: [C] 各通道在多尺度上的平均权重 | Mean weight per channel across scales.
        """
        return self.weights.mean(dim=0).squeeze()  # [C]

    def get_alphas(self) -> dict[int, float]:
        """
        获取各尺度的增强强度 | Get enhancement strength per scale.

        :return: {kernel_size: alpha_value}
        """
        return {ks: self.alphas[i].item()
                for i, ks in enumerate(self.kernel_sizes)}

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (f"FDE(spatial, ch={self.channels}, ks={list(self.kernel_sizes)}, "
                f"params={n/1e3:.1f}K)")
