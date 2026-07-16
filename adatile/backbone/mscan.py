"""
MSCAN — Multi-Scale Convolutional Attention Network (SegNeXt).
==============================================================
MSCAN: 多尺度卷积注意力编码器 | Multi-Scale Convolutional Attention Encoder.

独立实现 — 不依赖 mmcv/mmseg 框架 | Standalone — no mmcv/mmseg dependency.
适配 NEU_Seg 200×200 小图工业缺陷分割 | Adapted for 200×200 industrial defect.

SegNeXt (Guo et al., NeurIPS 2022):
    "Rethinking Convolutional Attention Design for Semantic Segmentation"
    https://arxiv.org/abs/2209.08575

架构 | Architecture:
    4-stage hierarchical encoder with strip convolutions (1×7, 7×1, 1×11, 11×1, 1×21, 21×1)
    as spatial attention gating units — replacing heavy self-attention with lightweight conv.

Usage::
    >>> from adatile.backbone.mscan import MSCAN
    >>> backbone = MSCAN(model_size="tiny")
    >>> feats = backbone(image)  # [C1, C2, C3, C4]
    >>> # C1: [B, 32, H/4, W/4], C2: [B, 64, H/8, W/8], C3: [B, 160, H/16, W/16], C4: [B, 256, H/32, W/32]
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════
# 模型规格 | Model Specifications
# ═══════════════════════════════════════════════════════════════════

SEGNEXT_CONFIGS = {
    "tiny":  dict(embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2],    mlp_ratios=[8, 8, 4, 4]),
    "small": dict(embed_dims=[64, 128, 320, 512], depths=[2, 2, 4, 2],   mlp_ratios=[8, 8, 4, 4]),
    "base":  dict(embed_dims=[64, 128, 320, 512], depths=[3, 3, 12, 3],  mlp_ratios=[8, 8, 4, 4]),
    "large": dict(embed_dims=[64, 128, 320, 512], depths=[3, 5, 27, 3],  mlp_ratios=[8, 8, 4, 4]),
}
"""预定义模型规格 | Pre-defined model specs."""


# ═══════════════════════════════════════════════════════════════════
# 基础组件 | Basic Components
# ═══════════════════════════════════════════════════════════════════

class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) — 随机丢弃整个残差分支。
    DropPath: randomly drop entire residual branches during training.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class DWConv(nn.Module):
    """深度可分离卷积 | Depthwise convolution (3×3, groups=dim)."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv(x)


class Mlp(nn.Module):
    """
    MLP with depthwise conv | 带深度卷积的 MLP。
    结构: Conv1x1 → DWConv3x3 → GELU → Drop → Conv1x1 → Drop。
    """

    def __init__(self, in_features: int, hidden_features: int | None = None,
                 out_features: int | None = None, act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionModule(nn.Module):
    """
    多尺度条带卷积注意力 | Multi-scale strip convolution attention.

    使用 5×5 depthwise + 三组条带卷积 (1×7/7×1, 1×11/11×1, 1×21/21×1)
    作为空间门控单元。Multi-scale strips capture long-range context efficiently.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)
        return attn * u  # 门控 | Gating


class SpatialAttention(nn.Module):
    """
    空间注意力块 | Spatial Attention Block。
    结构: Conv1×1 → GELU → AttentionModule → Conv1×1 → residual。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class MSCANBlock(nn.Module):
    """
    MSCAN 基本块 | MSCAN Basic Block。
    Norm → SpatialAttention (residual) → Norm → MLP (residual), with LayerScale.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0,
                 drop_path: float = 0.0, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        # LayerScale (CaiT-style) — 可学习的缩放因子 | Learnable scaling factor
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: [B, N, C] 序列格式 | sequence format."""
        B, N, C = x.shape
        # [B, N, C] → [B, C, H, W]  (2D 空间格式用于卷积注意力)
        x_2d = x.permute(0, 2, 1).view(B, C, H, W)
        # LayerScale: (C,) → (C, 1, 1)
        scale_1 = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
        scale_2 = self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
        # Spatial Attention
        x_2d = x_2d + self.drop_path(scale_1 * self.attn(self.norm1(x_2d)))
        # MLP
        x_2d = x_2d + self.drop_path(scale_2 * self.mlp(self.norm2(x_2d)))
        # 回到序列格式 | Back to sequence format
        x = x_2d.view(B, C, N).permute(0, 2, 1)
        return x


# ═══════════════════════════════════════════════════════════════════
# Patch Embedding | 图像分块嵌入
# ═══════════════════════════════════════════════════════════════════

class StemConv(nn.Module):
    """
    第一阶段的卷积下采样 (stride=4) | Stage 1 conv downsampling (stride=4).
    两个 3×3 Conv (s=2) + BN + GELU。
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] → [B, N, C]
        return x, H, W


class OverlapPatchEmbed(nn.Module):
    """
    重叠 Patch Embedding | Overlapping Patch Embedding.
    使用 stride 卷积实现重叠分块，保留局部连续性。
    Uses strided conv for overlapping patches to preserve local continuity.
    """

    def __init__(self, patch_size: int = 3, stride: int = 2,
                 in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)  # [B, C, H, W] → [B, N, C]
        return x, H, W


# ═══════════════════════════════════════════════════════════════════
# MSCAN Backbone | 多尺度卷积注意力编码器
# ═══════════════════════════════════════════════════════════════════

class MSCAN(nn.Module):
    """
    MSCAN — Multi-Scale Convolutional Attention Network.
    多尺度卷积注意力编码器 | Multi-Scale Convolutional Attention Encoder.

    4 阶段层级结构，使用条带卷积注意力代替自注意力。
    4-stage hierarchy with strip convolution attention replacing self-attention.

    :param model_size: "tiny" | "small" | "base" | "large"
    :param in_chans: 输入通道数 | Input channels (default 3 for RGB).
    :param drop_rate: dropout 率 | Dropout rate.
    :param drop_path_rate: DropPath 率 (stochastic depth) | DropPath rate.
    :param pretrained: 预训练权重路径 | Pretrained checkpoint path (optional).

    Returns (forward):
        List of 4 feature maps: [C1, C2, C3, C4]
        - Tiny: C1[32, H/4], C2[64, H/8], C3[160, H/16], C4[256, H/32]
        - Small/Base/Large: C1[64, H/4], C2[128, H/8], C3[320, H/16], C4[512, H/32]
    """

    def __init__(self, model_size: str = "tiny", in_chans: int = 3,
                 drop_rate: float = 0.0, drop_path_rate: float = 0.1,
                 pretrained: str | None = None):
        super().__init__()

        if model_size not in SEGNEXT_CONFIGS:
            raise ValueError(f"Unknown model_size: {model_size!r}. "
                             f"Options: {list(SEGNEXT_CONFIGS.keys())}")

        cfg = SEGNEXT_CONFIGS[model_size]
        self.embed_dims = cfg["embed_dims"]
        self.depths = cfg["depths"]
        self.mlp_ratios = cfg["mlp_ratios"]
        self.num_stages = 4

        # Stochastic depth decay rule | 随机深度衰减规则
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0

        for i in range(self.num_stages):
            if i == 0:
                patch_embed = StemConv(in_chans, self.embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_chans=in_chans if i == 0 else self.embed_dims[i - 1],
                    embed_dim=self.embed_dims[i],
                )

            block = nn.ModuleList([
                MSCANBlock(
                    dim=self.embed_dims[i],
                    mlp_ratio=self.mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                )
                for j in range(self.depths[i])
            ])
            norm = nn.LayerNorm(self.embed_dims[i])
            cur += self.depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # 初始化权重 | Initialize weights
        self.apply(self._init_weights)

        # 加载预训练权重 | Load pretrained weights
        self._pretrained = pretrained
        if pretrained is not None:
            self._load_pretrained(pretrained)

    def _init_weights(self, m: nn.Module) -> None:
        """标准初始化 | Standard weight initialization."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _load_pretrained(self, checkpoint_path: str) -> None:
        """
        加载官方 MSCAN 预训练权重 | Load official MSCAN pretrained weights.

        mmseg checkpoint 格式: state_dict 在 "state_dict" key 下，
        backbone 参数前缀为 "backbone." — 需要剥离。
        mmseg checkpoint format: state_dict under "state_dict" key,
        backbone params prefixed with "backbone." — need to strip.
        """
        # weights_only=False: 官方 checkpoint 含 argparse.Namespace 等非张量对象
        # (torch>=2.6 默认 weights_only=True 会拒绝加载, 该权重来源可信)
        # Official ckpt contains argparse.Namespace; trusted source, so opt out
        # of the torch>=2.6 weights_only default.
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)

        # 剥离 "backbone." 前缀 | Strip "backbone." prefix
        new_state = {}
        for k, v in state.items():
            if k.startswith("backbone."):
                new_state[k[len("backbone."):]] = v
            else:
                new_state[k] = v

        missing, unexpected = self.load_state_dict(new_state, strict=False)
        if missing:
            print(f"[MSCAN] Missing keys: {len(missing)} "
                  f"(first 3: {missing[:3]})")
        if unexpected:
            print(f"[MSCAN] Unexpected keys: {len(unexpected)} "
                  f"(first 3: {unexpected[:3]})")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        前向传播 | Forward pass.

        :param x: [B, 3, H, W] 输入图像 | Input image.
        :return: [C1, C2, C3, C4] 四阶段特征图 | Four stage feature maps.
        """
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


# ═══════════════════════════════════════════════════════════════════
# 工厂函数 | Factory Function
# ═══════════════════════════════════════════════════════════════════

def build_mscan(model_size: str = "tiny", pretrained: str | None = None,
                **kwargs) -> MSCAN:
    """
    构建 MSCAN backbone | Build MSCAN backbone.

    :param model_size: "tiny" | "small" | "base" | "large"
    :param pretrained: 预训练权重路径 | Pretrained checkpoint path.
    :param kwargs: 传给 MSCAN 的额外参数 | Extra args for MSCAN.
    :return: MSCAN instance.
    """
    return MSCAN(model_size=model_size, pretrained=pretrained, **kwargs)
