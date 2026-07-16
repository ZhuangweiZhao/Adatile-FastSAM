"""
频域增强模块 | Frequency-Domain Enhancement Modules.
=======================================================

基于 DCT (离散余弦变换) 的即插即用频域模块, 用于 FastSAM 工业缺陷分割。
DCT-based plug-and-play frequency modules for FastSAM industrial defect segmentation.

动机 | Motivation:
    NEU_Seg 缺陷有明确的频域特征:
    - Scratch (划痕): 高频, 薄长边缘
    - Inclusion (夹杂物): 中频, 低对比度纹理
    - Patch (斑块): 低频, 大面积平滑区域
    普通卷积无法有效分离这些频段 → DCT 多频段分析.

    NEU_Seg defects have clear frequency signatures:
    - Scratch: high-frequency, thin edges
    - Inclusion: mid-frequency, low-contrast texture
    - Patch: low-frequency, large smooth regions
    Plain convolutions cannot effectively separate these bands → DCT multi-spectral analysis.

理论依据 | Theoretical Basis:
    FcaNet (ICCV 2021) 证明: 2D DCT 的多频段分量比 GAP 更能表达通道信息。
    FcaNet (ICCV 2021) proved: multi-spectral 2D DCT components capture richer
    channel information than GAP (which is just the DC component).

模块 | Modules:
    DCTSpectralAttention  — 多频段通道注意力 (替代 SE/GAP)
    FrequencyGuidedFusion  — 频谱能量驱动的多尺度融合权重
    dct_consistency_loss    — 频域一致性约束损失
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# 1. DCT 基向量预计算 | DCT Basis Pre-computation
# ═══════════════════════════════════════════════════════════════════

def _build_2d_dct_basis(
    height: int,
    width: int,
    n_freq: int = 16,
) -> torch.Tensor:
    """
    预计算 2D DCT 基向量 | Pre-compute 2D DCT basis vectors.

    生成 n_freq 个 2D DCT 基 (每个基是 H×W 的权重矩阵),
    对应从低频到高频的 DCT 分量。
    Generate n_freq 2D DCT bases (each is H×W weight matrix),
    corresponding to low-to-high frequency DCT components.

    参考 FcaNet: 按 zigzag 顺序选择 DCT 系数, 确保覆盖从
    DC (0,0) 到高频的多个频段。
    FcaNet reference: select DCT coefficients in zigzag order,
    ensuring coverage from DC (0,0) to high frequencies.

    :param height: 特征图高度 | Feature map height.
    :param width: 特征图宽度 | Feature map width.
    :param n_freq: DCT 频率分量数 | Number of DCT frequency components.
    :return: [n_freq, 1, height, width] DCT basis tensor.
    """
    basis = []
    # 生成 zigzag 顺序的 (u, v) 索引 | Generate (u, v) indices in zigzag order
    indices = []
    for s in range(height + width):
        for u in range(s + 1):
            v = s - u
            if u < height and v < width:
                indices.append((u, v))
            if len(indices) >= n_freq:
                break
        if len(indices) >= n_freq:
            break

    for u, v in indices[:n_freq]:
        # 2D DCT basis: B_{u,v}(h,w) = cos(πu(h+0.5)/H) * cos(πv(w+0.5)/W)
        h_idx = torch.arange(height, dtype=torch.float32).view(-1, 1)
        w_idx = torch.arange(width, dtype=torch.float32).view(1, -1)
        basis_uv = (
            torch.cos(math.pi * u * (h_idx + 0.5) / height) *
            torch.cos(math.pi * v * (w_idx + 0.5) / width)
        )
        basis.append(basis_uv)

    return torch.stack(basis).unsqueeze(1)  # [n_freq, 1, H, W]


# ═══════════════════════════════════════════════════════════════════
# 2. DCT 多频段通道注意力 | DCT Multi-Spectral Channel Attention
# ═══════════════════════════════════════════════════════════════════

class DCTSpectralAttention(nn.Module):
    """
    DCT 多频段通道注意力 (FcaNet 风格) | DCT Multi-Spectral Channel Attention.

    用多个 2D DCT 频率分量 (而非单频段 GAP) 压缩空间信息,
    使通道注意力能感知不同频段的特征。
    Use multiple 2D DCT frequency components (instead of single-band GAP)
    to compress spatial info, enabling channel attention across frequency bands.

    设计 | Design:
        1. 将通道分为 G 组, 每组用不同 DCT 基做空间压缩
        2. 拼接各组结果 → [B, C]
        3. FC → ReLU → FC → Sigmoid (标准 SE excitation)
        4. 通道注意力加权

        1. Split channels into G groups, each compressed with a different DCT basis
        2. Concat results → [B, C]
        3. FC → ReLU → FC → Sigmoid (standard SE excitation)
        4. Channel attention weighting

    对比 SE | vs SE:
        SE:  GAP (仅 DC 分量, 丢失所有高频信息)
        DCT: N 个 DCT 基 (覆盖 DC → 高频, 保留频域多样性)

    Parameters
    ----------
    channels : int
        输入通道数 | Input channels.
    reduction : int
        FC 压缩比 | FC reduction ratio.
    n_freq : int
        DCT 频率分量数 (建议 16) | Number of DCT frequency components.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        n_freq: int = 16,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.n_freq = n_freq
        mid = max(channels // reduction, 8)

        # DCT 基延迟初始化 (首次 forward 时根据 H,W 构建)
        # DCT basis initialized lazily (built on first forward based on H,W)
        self.dct_basis: torch.Tensor | None = None
        self._basis_h: int = 0
        self._basis_w: int = 0

        # SE-style excitation (共享)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )
        # 零初始化最后一层 (从 identity 开始)
        nn.init.zeros_(self.fc[-2].weight)

    def _ensure_basis(self, h: int, w: int, device: torch.device):
        """确保 DCT 基与当前特征图尺寸匹配 | Ensure DCT basis matches feature size."""
        if self.dct_basis is None or self._basis_h != h or self._basis_w != w:
            self.dct_basis = _build_2d_dct_basis(h, w, self.n_freq).to(device)
            self._basis_h = h
            self._basis_w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, C, H, W] 特征图 | Feature map.
        :return: [B, C, H, W] 注意力加权后的特征 | Attention-weighted features.
        """
        B, C, H, W = x.shape
        self._ensure_basis(H, W, x.device)

        # 将通道分为 n_freq 组 | Split channels into n_freq groups
        # 每组用对应的 DCT 基做空间压缩
        # Each group compressed with its corresponding DCT basis
        group_size = max(1, C // self.n_freq)

        freq_responses = []
        for i in range(self.n_freq):
            start = i * group_size
            end = start + group_size if i < self.n_freq - 1 else C
            if start >= C:
                break
            x_group = x[:, start:end, :, :]  # [B, g, H, W]
            # DCT 压缩: element-wise multiply with basis, then spatial sum
            # DCT compression: Σ(x_group * basis_uv)
            basis = self.dct_basis[i:i+1]  # [1, 1, H, W]
            response = (x_group * basis).sum(dim=[-2, -1])  # [B, g]
            freq_responses.append(response)

        # 拼接所有频段响应 | Concat all frequency responses
        freq_pooled = torch.cat(freq_responses, dim=1)  # [B, C]

        # 确保维度匹配 (处理分组余数) | Ensure dim match (handle remainder)
        if freq_pooled.shape[1] != C:
            freq_pooled = freq_pooled[:, :C]

        # SE excitation
        attn = self.fc(freq_pooled).view(B, C, 1, 1)  # [B, C, 1, 1]

        # 残差注意力 | Residual attention
        return x * (1.0 + attn)


# ═══════════════════════════════════════════════════════════════════
# 3. 多尺度频域注意力 | Multi-Scale Spectral Attention
# ═══════════════════════════════════════════════════════════════════

class MultiScaleSpectralAttention(nn.Module):
    """
    多尺度 DCT 频域注意力 (P2/P3/P4 各一个) | Multi-Scale DCT Spectral Attention.

    每个尺度有独立的 DCTSpectralAttention, 因为不同分辨率的频域特征不同:
    - P2 (H/4): 高频纹理细节 | High-frequency texture details
    - P3 (H/8): 中频结构 | Mid-frequency structures
    - P4 (H/16): 低频语义 | Low-frequency semantics

    Parameters
    ----------
    p2_channels, p3_channels, p4_channels : int
        各层通道数 | Per-layer channel counts.
    reduction : int
        FC 压缩比 | FC reduction ratio.
    n_freq : int
        DCT 频率分量数 | Number of DCT frequency components.
    """
    def __init__(
        self,
        p2_channels: int = 160,
        p3_channels: int = 960,
        p4_channels: int = 1280,
        reduction: int = 4,
        n_freq: int = 16,
    ) -> None:
        super().__init__()
        self.p2_attn = DCTSpectralAttention(p2_channels, reduction, n_freq)
        self.p3_attn = DCTSpectralAttention(p3_channels, reduction, n_freq)
        self.p4_attn = DCTSpectralAttention(p4_channels, reduction, n_freq)

    def forward(
        self,
        p2: torch.Tensor | None = None,
        p3: torch.Tensor | None = None,
        p4: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """对每个输入尺度分别应用频谱注意力."""
        result = {}
        if p2 is not None:
            result["p2"] = self.p2_attn(p2)
        if p3 is not None:
            result["p3"] = self.p3_attn(p3)
        if p4 is not None:
            result["p4"] = self.p4_attn(p4)
        return result

    def get_trainable_params(self) -> list[nn.Parameter]:
        params = []
        for attn in [self.p2_attn, self.p3_attn, self.p4_attn]:
            params.extend(attn.parameters())
        return params


# ═══════════════════════════════════════════════════════════════════
# 4. 频谱能量引导的多尺度融合 | Frequency-Energy Guided Multi-Scale Fusion
# ═══════════════════════════════════════════════════════════════════

class FrequencyGuidedFusion(nn.Module):
    """
    频谱能量驱动的动态融合权重 | Frequency-energy driven dynamic fusion weights.

    原理 | Principle:
        计算每个尺度的 DCT 频谱能量分布, 动态决定融合权重。
        Compute DCT spectral energy distribution per scale,
        dynamically determine fusion weights.

        - P2 高频能量高 → P2 权重大 (适合 Scratch/Inclusion)
        - P4 低频能量高 → P4 权重大 (适合 Patch)

    替代 BiFPN 的固定可学习权重, 使融合权重适应输入内容。
    Replaces BiFPN's fixed learnable weights with content-adaptive weights.

    实现 | Implementation:
        1. 对每个尺度的特征做 Patch-wise DCT
        2. 将 DCT 系数分为高/中/低三个频段
        3. 计算频段能量比 → 作为动态融合权重

        1. Patch-wise DCT on each scale's features
        2. Split DCT coefficients into H/M/L frequency bands
        3. Compute band energy ratio → dynamic fusion weights

    Parameters
    ----------
    mid_channels : int
        融合中间通道数 | Fusion mid channels.
    patch_size : int
        DCT patch 大小 | DCT patch size (建议 8).
    """

    def __init__(
        self,
        mid_channels: int = 128,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.patch_size = patch_size

        # 频段能量 → 融合权重 | Band energy → fusion weight
        # 输入: [B, 3, 3] (3 scales × 3 bands each)
        # 输出: [B, 3] (per-scale fusion weights)
        self.weight_net = nn.Sequential(
            nn.Linear(9, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1),  # 归一化权重 | normalized weights
        )
        # 初始化为均匀权重 | Init to uniform weights
        nn.init.zeros_(self.weight_net[-2].weight)
        nn.init.zeros_(self.weight_net[-2].bias)

    def _compute_band_energy(
        self,
        feat: torch.Tensor,  # [B, C, H, W]
    ) -> torch.Tensor:
        """
        计算 patch-wise DCT 频段能量 | Compute patch-wise DCT band energy.

        将特征划分为 patch, 对每个 patch 做 DCT, 统计高/中/低频能量比。
        Divide feature into patches, apply DCT per patch, compute H/M/L energy ratio.

        :param feat: [B, C, H, W].
        :return: [B, 3] high/mid/low band energy ratios.
        """
        B, C, H, W = feat.shape
        ps = self.patch_size

        # 确保 H, W 能被 patch_size 整除 | Ensure H,W divisible by patch_size
        if H % ps != 0 or W % ps != 0:
            feat = F.interpolate(
                feat, size=(H - H % ps, W - W % ps),
                mode="bilinear", align_corners=False,
            )
            B, C, H, W = feat.shape

        # Reshape to patches: [B, C, H/ps, ps, W/ps, ps] → [B*C*num_patches, ps, ps]
        feat_reshaped = feat.reshape(B, C, H // ps, ps, W // ps, ps)
        feat_reshaped = feat_reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
        patches = feat_reshaped.reshape(-1, ps, ps)  # [N_patches, ps, ps]

        # 2D DCT per patch (简单实现: 沿两个维度做 1D DCT)
        # 2D DCT per patch (simple: 1D DCT along both dims)
        # 使用轻量的频谱估计: FFT 的幅度
        # Use lightweight spectral estimation: FFT magnitude
        patches_fft = torch.fft.rfft2(patches.float(), norm='ortho')
        patches_amp = torch.abs(patches_fft)  # [N, ps, ps//2+1]

        # 频段划分 | Frequency band split
        # Low: 0 ~ ps/4
        # Mid: ps/4 ~ ps/2
        # High: ps/2 ~ ps
        half = ps // 2
        quart = ps // 4

        low_energy = patches_amp[:, :quart, :quart].mean(dim=[1, 2])
        mid_energy = patches_amp[:, quart:half, :half].mean(dim=[1, 2])
        # High: rest
        high_mask_h = torch.arange(half, ps)
        high_mask_w = torch.arange(ps // 2 + 1)
        high_energy = patches_amp[:, half:, :].mean(dim=[1, 2])

        # 聚合到 batch | Aggregate to batch
        n_patches_per_sample = patches.shape[0] // B
        band_energy = torch.stack([
            low_energy.reshape(B, n_patches_per_sample).mean(dim=1),
            mid_energy.reshape(B, n_patches_per_sample).mean(dim=1),
            high_energy.reshape(B, n_patches_per_sample).mean(dim=1),
        ], dim=1)  # [B, 3]

        # 归一化能量比 | Normalize energy ratios
        band_energy = band_energy / (band_energy.sum(dim=1, keepdim=True) + 1e-6)
        return band_energy

    def forward(
        self,
        f2: torch.Tensor,  # [B, C, H/4, W/4]
        f3: torch.Tensor,  # [B, C, H/8, W/8]
        f4: torch.Tensor,  # [B, C, H/16, W/16]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算频谱能量驱动的动态融合权重 | Compute spectral-energy driven dynamic fusion weights.

        :return: (w_p2, w_p3, w_p4) 归一化融合权重 | Normalized fusion weights.
            Each is [B, 1] broadcastable to feature dimensions.
        """
        # 计算每层的频段能量 | Compute band energy per scale
        e2 = self._compute_band_energy(f2)  # [B, 3]
        e3 = self._compute_band_energy(f3)  # [B, 3]
        e4 = self._compute_band_energy(f4)  # [B, 3]

        # 拼接: [B, 9] (3 scales × 3 bands)
        energy_feat = torch.cat([e2, e3, e4], dim=1)

        # 预测融合权重 | Predict fusion weights
        weights = self.weight_net(energy_feat)  # [B, 3]

        return (
            weights[:, 0:1],  # w_p2
            weights[:, 1:2],  # w_p3
            weights[:, 2:3],  # w_p4
        )


# ═══════════════════════════════════════════════════════════════════
# 5. DCT 频谱一致性损失 | DCT Spectral Consistency Loss
# ═══════════════════════════════════════════════════════════════════

def dct_spectral_loss(
    pred: torch.Tensor,      # [B, C, H, W] softmax probs
    target: torch.Tensor,    # [B, H, W] int64 class labels
    n_freq_bands: int = 4,
    ignore_bg: bool = True,
) -> torch.Tensor:
    """
    DCT 频谱一致性损失 | DCT Spectral Consistency Loss.

    比较预测 mask 和 GT mask 在 DCT 频域的差异。
    Compare predicted and GT masks in DCT frequency domain.

    对每个前景类别:
    1. 二值化 pred (argmax) 和 GT
    2. 计算 2D DCT 幅度谱
    3. 将频谱分为 n_freq_bands 个频段
    4. 每频段的 L1 差异加权平均

    Per foreground class:
    1. Binarize pred (argmax) and GT
    2. Compute 2D DCT amplitude spectrum
    3. Split spectrum into n_freq_bands bands
    4. Weighted L1 difference per band

    :param pred: [B, C, H, W] softmax probabilities.
    :param target: [B, H, W] int64 labels.
    :param n_freq_bands: 频段数 | Number of frequency bands.
    :param ignore_bg: 忽略背景类 | Ignore background class.
    :return: scalar spectral consistency loss.
    """
    B, C, H, W = pred.shape
    start_class = 1 if ignore_bg else 0
    pred_class = torch.argmax(pred, dim=1)  # [B, H, W]

    total_loss = torch.tensor(0.0, device=pred.device)
    n_classes = 0

    for c in range(start_class, C):
        pred_c = (pred_class == c).float()  # [B, H, W]
        gt_c = (target == c).float()

        # 跳过 batch 中该类都不存在的样本 | Skip if class absent in batch
        if gt_c.sum() == 0:
            continue

        # 2D FFT with log-amplitude normalization
        # Use log(1+amp) to compress dynamic range and stabilize gradients
        pred_fft = torch.fft.rfft2(pred_c, norm='ortho')
        gt_fft = torch.fft.rfft2(gt_c, norm='ortho')

        pred_amp = torch.log1p(torch.abs(pred_fft))  # [B, H, W//2+1]
        gt_amp = torch.log1p(torch.abs(gt_fft))

        # 按频率环带分组 | Group by frequency rings
        max_radius = min(H, W // 2 + 1)
        band_size = max(1, max_radius // n_freq_bands)

        band_loss = 0.0
        for band in range(n_freq_bands):
            r_start = band * band_size
            r_end = (band + 1) * band_size if band < n_freq_bands - 1 else max_radius

            # 创建频率环带 mask | Create frequency ring mask
            h_center, w_center = H // 2, (W // 2 + 1) // 2
            h_idx = torch.arange(H, device=pred.device).float()
            w_idx = torch.arange(W // 2 + 1, device=pred.device).float()
            h_dist = (h_idx.view(-1, 1) - h_center).abs()
            w_dist = (w_idx.view(1, -1) - w_center).abs()
            dist = torch.sqrt(h_dist ** 2 + w_dist ** 2)

            ring_mask = ((dist >= r_start) & (dist < r_end)).float()
            ring_area = ring_mask.sum() + 1e-6

            # 环带内平均 L2 差异 (归一化到单位面积) | Mean L2 diff within ring (normalized to unit area)
            pred_band = (pred_amp * ring_mask.unsqueeze(0)).sum(dim=[1, 2]) / ring_area
            gt_band = (gt_amp * ring_mask.unsqueeze(0)).sum(dim=[1, 2]) / ring_area
            band_loss += F.l1_loss(pred_band, gt_band)

        total_loss += band_loss / n_freq_bands
        n_classes += 1

    if n_classes == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return total_loss / n_classes


def spectral_combined_loss(
    pred: torch.Tensor,          # [B, C, H, W] softmax probs
    target: torch.Tensor,        # [B, H, W] int64 class labels
    ce_weight: torch.Tensor | None = None,
    ce_alpha: float = 0.45,
    dice_alpha: float = 0.45,
    spectral_alpha: float = 0.10,  # auxiliary loss, keep weight low
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    组合损失: CE + multi-Dice + DCT Spectral Consistency.
    Combined loss: CE + multi-Dice + DCT Spectral Consistency.
    """
    log_pred = torch.log(pred + 1e-7)
    ce = F.nll_loss(log_pred, target, weight=ce_weight)

    # Multi-class Dice (inline, avoid circular imports)
    C = pred.shape[1]
    dice_sum = 0.0
    count = 0
    for c in range(1, C):
        pred_c = pred[:, c, :, :]
        target_c = (target == c).float()
        if target_c.sum() > 0:
            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_sum += (2.0 * inter + 1e-6) / (union + 1e-6)
            count += 1
    dice = (1.0 - dice_sum / max(count, 1)) * torch.ones(1, device=pred.device)

    spectral = dct_spectral_loss(pred, target)

    total = ce_alpha * ce + dice_alpha * dice + spectral_alpha * spectral
    return total, {"ce": ce.item(), "dice": dice.item(), "spectral": spectral.item()}
