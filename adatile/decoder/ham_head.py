"""
LightHamHead — 层级注意力模块解码器 (SegNeXt).
================================================
LightHamHead: Hierarchical Attention Module Decoder for SegNeXt.

独立实现 — 不依赖 mmcv/mmseg 框架 | Standalone — no mmcv/mmseg dependency.
基于 Hamburger (NMF 矩阵分解) 的全局上下文聚合。
Hamburger-based (NMF matrix decomposition) global context aggregation.

SegNeXt (Guo et al., NeurIPS 2022):
    解码器: Concat(C2,C3,C4) → Squeeze → Hamburger(NMF) → Align → Classify.
    HamNet (Geng et al., ICLR 2022):
        "Is Attention Better Than Matrix Decomposition?"
        https://arxiv.org/abs/2109.04553

Usage::
    >>> from adatile.decoder.ham_head import LightHamHead
    >>> head = LightHamHead(in_channels=[64, 160, 256], num_classes=4)
    >>> output = head(feats)  # feats: list of [C2, C3, C4]
    >>> # output: [B, 4, H, W] (softmax logits at C2 resolution)
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger


# ═══════════════════════════════════════════════════════════════════
# 卷积+归一化+激活 组合 | Conv+Norm+Act Combo
# ═══════════════════════════════════════════════════════════════════

def conv_bn_act(in_ch: int, out_ch: int, kernel_size: int = 1,
                stride: int = 1, padding: int = 0, groups: int = 1,
                norm: callable | None = lambda ch: nn.BatchNorm2d(ch),
                act: type | None = nn.ReLU) -> nn.Sequential:
    """
    构建 Conv2d + Norm + Act 序列 | Build Conv2d + Norm + Act sequence.

    :param norm: 接受 channels 返回 nn.Module 的可调用对象。
                接收 channels 返回 nn.Module 的可调用对象。
                例: lambda ch: nn.GroupNorm(32, ch)。
    :param act: 激活函数类 (使用 inplace=True) | Activation class.
    """
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
    ]
    if norm is not None:
        layers.append(norm(out_ch))
    if act is not None:
        layers.append(act(inplace=True))
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════════
# Hamburger — NMF 矩阵分解模块 | NMF Matrix Decomposition Module
# ═══════════════════════════════════════════════════════════════════

class NMF2D(nn.Module):
    """
    2D Non-negative Matrix Factorization (NMF).
    将特征矩阵分解为字典基 × 系数，重建时压缩噪声。
    Decomposes feature matrix into dictionary bases × coefficients,
    compressing noise during reconstruction.

    MD_S: 子空间数 | Number of subspaces.
    MD_D: 每子空间维度 | Dimension per subspace.
    MD_R: 字典基数量 (rank) | Number of dictionary bases (rank).
    """

    def __init__(self, S: int = 1, D: int = 512, R: int = 64,
                 train_steps: int = 6, eval_steps: int = 7,
                 inv_t: int = 100, eta: float = 0.9,
                 rand_init: bool = True):
        super().__init__()
        self.S = S
        self.D = D
        self.R = R
        self.train_steps = train_steps
        self.eval_steps = eval_steps
        self.inv_t = inv_t
        self.eta = eta
        self.rand_init = rand_init

    def _build_bases(self, B: int, device: torch.device) -> torch.Tensor:
        """构建随机初始化的字典基 | Build randomly initialized dictionary bases."""
        bases = torch.rand((B * self.S, self.D, self.R), device=device)
        bases = F.normalize(bases, dim=1)
        return bases

    def local_step(self, x: torch.Tensor, bases: torch.Tensor,
                   coef: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        NMF multiplicative update step | NMF 乘性更新步。
        x: [B*S, D, N], bases: [B*S, D, R], coef: [B*S, N, R]
        """
        # Update coef | 更新系数
        numerator = torch.bmm(x.transpose(1, 2), bases)  # [B*S, N, R]
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))  # [B*S, N, R]
        coef = coef * numerator / (denominator + 1e-6)

        # Update bases | 更新基
        numerator = torch.bmm(x, coef)  # [B*S, D, R]
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))  # [B*S, D, R]
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x: torch.Tensor, bases: torch.Tensor,
                     coef: torch.Tensor) -> torch.Tensor:
        """计算最终系数 | Compute final coefficients."""
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef

    @torch.no_grad()
    def local_inference(self, x: torch.Tensor,
                        bases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """迭代推断: 从随机初始化出发, 通过 NMF 更新步收敛到稳定的基和系数。"""
        # 初始化系数 | Initialize coefficients via soft assignment
        coef = torch.bmm(x.transpose(1, 2), bases)  # [B*S, N, R]
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        NMF 前向: 分解→重建 | NMF forward: decompose → reconstruct.

        :param x: [B, C, H, W] 输入特征 | Input features.
        :return: [B, C, H, W] 重建特征 (压缩噪声后) | Reconstructed (denoised).
        """
        B, C, H, W = x.shape

        # (B, C, H, W) → (B * S, D, N)
        D = C // self.S
        N = H * W
        x_flat = x.view(B * self.S, D, N)

        # 构建字典基 | Build bases
        bases = self._build_bases(B, x.device)

        # NMF 分解 → 重建 | NMF decompose → reconstruct
        bases, coef = self.local_inference(x_flat, bases)
        coef = self.compute_coef(x_flat, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T → (B * S, D, N) → (B, C, H, W)
        x_out = torch.bmm(bases, coef.transpose(1, 2))
        x_out = x_out.view(B, C, H, W)

        return x_out


class Hamburger(nn.Module):
    """
    Hamburger 模块 — NMF 全局上下文聚合。
    Hamburger module: NMF-based global context aggregation.

    结构: Conv1×1 → ReLU → NMF2D → Conv1×1(BN/ReLU) → residual。
    Structure: Conv1×1 → ReLU → NMF2D → Conv1×1(Norm/Act) → residual.
    """

    def __init__(self, channels: int = 256, ham_kwargs: dict | None = None,
                 norm: callable = lambda ch: nn.BatchNorm2d(ch)):
        super().__init__()
        ham_kwargs = ham_kwargs or {}

        self.ham_in = conv_bn_act(channels, channels, norm=None, act=None)

        # NMF2D: 自动推导 D=channels, R=MD_R (通常 16)
        self.ham = NMF2D(
            S=1, D=channels,
            R=ham_kwargs.get("MD_R", 16),
            train_steps=ham_kwargs.get("TRAIN_STEPS", 6),
            eval_steps=ham_kwargs.get("EVAL_STEPS", 7),
            inv_t=ham_kwargs.get("INV_T", 100),
            eta=ham_kwargs.get("ETA", 0.9),
            rand_init=ham_kwargs.get("RAND_INIT", True),
        )

        self.ham_out = conv_bn_act(channels, channels, norm=norm, act=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        return F.relu(x + enjoy, inplace=True)


def _best_gn_groups(channels: int) -> int:
    """
    找到适合 GroupNorm 的最佳分组数 | Find the best number of groups for GroupNorm.
    优先 32 → 16 → 8 → 4，至少 1。| Prefer 32 → 16 → 8 → 4, minimum 1.
    """
    for g in [32, 16, 8, 4]:
        if channels % g == 0:
            return g
    return 1


# ═══════════════════════════════════════════════════════════════════
# LightHamHead — 层级注意力模块解码器
# ═══════════════════════════════════════════════════════════════════

class LightHamHead(nn.Module):
    """
    轻量层级注意力解码器 | Lightweight Hierarchical Attention Module Decoder.

    流程 | Pipeline:
        Concat(C2,C3,C4) → Squeeze(1×1) → Hamburger(NMF) → Align(1×1) → Classify(1×1).

    :param in_channels: 输入特征通道数 (通常 [64, 160, 256] for tiny) | Input channel list.
    :param num_classes: 输出类别数 (含 BG) | Number of output classes (including BG).
    :param channels: 对齐后通道数 | Channels after alignment. Default 256.
    :param ham_channels: Hamburger 内部通道数 | Hamburger internal channels. Default 256.
    :param ham_kwargs: NMF 参数 (MD_R 等) | NMF params (MD_R, etc.). Default MD_R=16.
    :param dropout_ratio: dropout 率 | Dropout ratio. Default 0.1.
    :param align_corners: F.interpolate align_corners. Default False.

    SegNeXt 使用 GroupNorm(32) 而非 BatchNorm:
        理由: NEU_Seg batch=1, BN 不稳定 → GN(32) 更适合。
        Reason: NEU_Seg batch=1, BN unstable → GN(32) more suitable.
    """

    def __init__(self, in_channels: Sequence[int] = (64, 160, 256),
                 num_classes: int = 4, channels: int = 256,
                 ham_channels: int = 256,
                 ham_kwargs: dict | None = None,
                 dropout_ratio: float = 0.1,
                 align_corners: bool = False):
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.ham_channels = ham_channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.logger = get_logger("ham_head")

        # ── 通道压缩 | Channel Squeeze ──
        # 将 C2+C3+C4 拼接后压缩到 ham_channels
        total_in = sum(self.in_channels)
        gn_for_ham = lambda ch: nn.GroupNorm(_best_gn_groups(ch), ch)
        self.squeeze = conv_bn_act(total_in, ham_channels, norm=gn_for_ham, act=nn.ReLU)

        # ── Hamburger (NMF 全局上下文) ──
        ham_kwargs = ham_kwargs or {}
        self.hamburger = Hamburger(channels=ham_channels, ham_kwargs=ham_kwargs,
                                   norm=gn_for_ham)

        # ── 通道对齐 | Channel Align ──
        gn_for_align = lambda ch: nn.GroupNorm(_best_gn_groups(ch), ch)
        self.align = conv_bn_act(ham_channels, channels, norm=gn_for_align, act=nn.ReLU)

        # ── 分类头 | Classification Head ──
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.cls_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        self._log_init()

    def _log_init(self) -> None:
        """记录初始化信息 | Log init info."""
        total_in = sum(self.in_channels)
        params = sum(p.numel() for p in self.parameters())
        self.logger.log_info(
            "ham_head/init",
            f"LightHamHead: in={self.in_channels} (sum={total_in}) → "
            f"ham={self.ham_channels} → align={self.channels} → cls={self.num_classes}, "
            f"params={params / 1e3:.1f}K",
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        :param inputs: 3 个特征图 [C2, C3, C4] | 3 feature maps.
            - C2: [B, 64, H/8, W/8]
            - C3: [B, 160, H/16, W/16]
            - C4: [B, 256, H/32, W/32]
        :return: [B, num_classes, H/8, W/8] 分割 logits | Segmentation logits.
        """
        # ── 多尺度特征对齐 | Multi-scale feature alignment ──
        # 将 C3, C4 上采样到 C2 的尺寸 → concat
        target_size = inputs[0].shape[2:]
        aligned = [inputs[0]]  # C2, reference resolution
        for feat in inputs[1:]:
            aligned.append(F.interpolate(
                feat, size=target_size, mode="bilinear",
                align_corners=self.align_corners,
            ))

        # ── 拼接 → Squeeze → Hamburger → Align → Classify ──
        x = torch.cat(aligned, dim=1)  # [B, sum(C2,C3,C4), H/8, W/8]
        x = self.squeeze(x)            # [B, ham_channels, H/8, W/8]
        x = self.hamburger(x)          # NMF 全局上下文 | Global context via NMF
        x = self.align(x)              # [B, channels, H/8, W/8]

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.cls_seg(x)            # [B, num_classes, H/8, W/8]
        return x
