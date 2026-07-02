"""
LightDecoder — 轻量解码器 | Lightweight Segmentation Decoder.
===============================================================

P4 特征 → 渐进上采样 + 多层 Conv → 密集分割掩码。
P4 features → gradual upsampling + multi-layer Conv → dense segmentation mask.

支持两种模式 | Two modes:
    Binary (num_classes=1):  1280→64→64→32→32→1, ~716K params
    Multi-class (num_classes>1): 1280→256→128→64→32→C, ~716K params

用法 | Usage::
    # Binary mode (pre-experiments, MassBuildings)
    decoder = LightDecoder(in_channels=1280, num_classes=1)

    # Multi-class (iSAID 15-class dense label prediction)
    decoder = LightDecoder(in_channels=1280, num_classes=16)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.logging import get_logger


class LightDecoder(nn.Module):
    """
    轻量解码器 | Lightweight Decoder.

    从 P4 特征（stride-16）渐进上采样到原始分辨率。
    Gradually upsamples P4 features (stride-16) to original resolution.

    num_classes=1:  binary mode, ~716K params
    num_classes>1:  multi-class mode, ~716K params (wider early layers)
    """

    def __init__(self, in_channels: int = 1280, num_classes: int = 1):
        super().__init__()
        self.logger = get_logger("decoder.light")
        self.num_classes = num_classes
        self._is_binary = (num_classes == 1)

        if self._is_binary:
            # Binary mode: thinner, deeper | 二值掩码模式
            self.stage1 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.stage3 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.stage4 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Conv2d(32, 1, kernel_size=1, bias=True)
        else:
            # Multi-class mode: wider Stage1, fewer upsamples | 多类别密集标签模式
            self.stage1 = nn.Sequential(
                nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.stage2 = nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.stage3 = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.stage4 = None  # multi-class mode has no stage4
            self.head = nn.Conv2d(32, num_classes, kernel_size=1, bias=True)

        # 参数统计 | Parameter stats
        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.log_info(
            "light_decoder/init",
            f"LightDecoder: mode={'binary' if self._is_binary else 'multi'}, "
            f"num_classes={num_classes}, "
            f"params={n_total:,} (trainable={n_trainable:,})",
        )

    def forward(self, features: dict[str, torch.Tensor],
                target_size: tuple[int, int] | None = None) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        :param features: {"p4": [B, 1280, H/16, W/16]}
        :type features: dict[str, torch.Tensor]

        :param target_size: (H, W) 目标尺寸。None → 返回 stride-2 (binary) 或 stride-4 (multi) 的 logit。 Target size. None → return logits at output stride.
        :type target_size: tuple[int, int] | None

        :return: logit [B, C, *, *] — raw logits (before sigmoid/softmax). Binary mode: C=1, for BCEWithLogitsLoss. Multi-class mode: C=num_classes, for CrossEntropyLoss.
        :rtype: torch.Tensor
        """
        x = features["p4"]  # [B, in_channels, H/16, W/16]

        # Stage 1: compress at stride-16 | 压缩特征
        x = self.stage1(x)  # [B, ch, H/16, W/16]

        # Stage 2: H/16 → H/8
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2(x)  # [B, ch, H/8, W/8]

        # Stage 3: H/8 → H/4
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage3(x)  # [B, ch, H/4, W/4]

        # Stage 4 (binary only): H/4 → H/2
        if self.stage4 is not None:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.stage4(x)  # [B, 32, H/2, W/2]

        # Final upsample to target
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        # Head: 投影到输出通道 | Project to output channels
        logit = self.head(x)  # [B, C, *, *]

        return logit  # raw logit, NO activation

    def predict(self, features: dict[str, torch.Tensor],
                target_size: tuple[int, int]) -> torch.Tensor:
        """
        预测分割掩码 | Predict segmentation mask.

        :param features: backbone 输出 | Backbone output.
        :type features: dict[str, torch.Tensor]

        :param target_size: (H, W) 目标尺寸 | Target size.
        :type target_size: tuple[int, int]

        :return: Binary mode: [B, 1, H, W] float mask. Multi-class mode: [B, H, W] int64 class indices.
        :rtype: torch.Tensor
        """
        logit = self.forward(features, target_size=target_size)
        if self._is_binary:
            return (torch.sigmoid(logit) > 0.5).float()
        else:
            return logit.argmax(dim=1)


class LightDecoderP3P4(nn.Module):
    """
    轻量解码器 (P3+P4 多尺度融合) | Lightweight Decoder with P3+P4 Multi-Scale Fusion.

    P3 (stride-8, 960ch) + P4 (stride-16, 1280ch) → 投影→融合→渐进上采样→分割掩码。
    P3 (stride-8, 960ch) + P4 (stride-16, 1280ch) → project→fuse→gradual upsample→mask.

    相比纯 P4 版本，P3 提供更高分辨率特征 (stride-8 vs stride-16)，
    有利于小目标和细长结构 (bridge, helicopter, plane) 的分割。
    Compared to P4-only version, P3 provides higher resolution features,
    benefiting small objects and thin structures (bridge, helicopter, plane).

    参数量 | Params: ~274K (比纯 P4 的 716K 更轻量 | lighter than P4-only 716K).
    原因: P3+P4 投影层替代了 P4-only 的厚重 stage1 (1280→256→128 Conv).
    Reason: P3+P4 projectors replace P4-only heavy stage1 (1280→256→128 Conv).

    用法 | Usage::
        decoder = LightDecoderP3P4(p3_channels=960, p4_channels=1280, num_classes=16)
        feats = backbone(img)  # {"p3": [B,960,H/8,W/8], "p4": [B,1280,H/16,W/16]}
        logit = decoder(feats, target_size=(256, 256))
    """

    def __init__(
        self,
        p3_channels: int = 960,
        p4_channels: int = 1280,
        num_classes: int = 16,
        fuse_dim: int = 64,
    ):
        super().__init__()
        self.logger = get_logger("decoder.p3p4")
        self.num_classes = num_classes

        # ── P4 投影 (stride-16 → fuse_dim) | P4 projection ──
        self.p4_proj = nn.Sequential(
            nn.Conv2d(p4_channels, fuse_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True),
        )

        # ── P3 投影 (stride-8 → fuse_dim) | P3 projection ──
        self.p3_proj = nn.Sequential(
            nn.Conv2d(p3_channels, fuse_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True),
        )

        # ── 融合层 (concat(P3, P4_up) → fuse_dim) | Fusion layer ──
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_dim * 2, fuse_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True),
        )

        # ── 上采样阶段 (与 LightDecoder 一致) | Upsampling stages (same as LightDecoder) ──
        self.stage2 = nn.Sequential(
            nn.Conv2d(fuse_dim, fuse_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fuse_dim),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(fuse_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, num_classes, kernel_size=1, bias=True)

        # ── 参数统计 | Parameter stats ──
        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.log_info(
            "light_decoder_p3p4/init",
            f"LightDecoderP3P4: p3_ch={p3_channels}, p4_ch={p4_channels}, "
            f"fuse_dim={fuse_dim}, num_classes={num_classes}, "
            f"params={n_total:,} (trainable={n_trainable:,})",
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        前向传播 | Forward pass.

        :param features: {"p3": [B, 960, H/8, W/8], "p4": [B, 1280, H/16, W/16]}
        :param target_size: (H, W) 目标尺寸。None → stride-2 的 logit。
        :return: logit [B, num_classes, *, *] — raw logits (for CrossEntropyLoss).
        """
        p3 = features["p3"]  # [B, C3, H/8, W/8]
        p4 = features["p4"]  # [B, C4, H/16, W/16]

        # ── 投影到共同维度 | Project to common dim ──
        f3 = self.p3_proj(p3)  # [B, fuse_dim, H/8, W/8]
        f4 = self.p4_proj(p4)  # [B, fuse_dim, H/16, W/16]

        # ── P4 上采样到 P3 分辨率 | Upsample P4 to P3 resolution ──
        f4_up = F.interpolate(f4, size=f3.shape[2:], mode="bilinear", align_corners=False)

        # ── 融合 | Fuse ──
        fused = torch.cat([f3, f4_up], dim=1)  # [B, 2*fuse_dim, H/8, W/8]
        x = self.fuse(fused)                    # [B, fuse_dim, H/8, W/8]

        # ── 渐进上采样 | Gradual upsampling ──
        # H/8 → H/4
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2(x)  # [B, fuse_dim, H/4, W/4]

        # H/4 → H/2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage3(x)  # [B, 32, H/2, W/2]

        # ── 最终上采样到目标尺寸 | Final upsample to target ──
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        # ── Head: 投影到输出通道 | Project to output channels ──
        logit = self.head(x)  # [B, num_classes, *, *]
        return logit  # raw logit, NO activation

    def predict(
        self,
        features: dict[str, torch.Tensor],
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        预测分割掩码 | Predict segmentation mask.

        :return: [B, H, W] int64 class indices.
        """
        logit = self.forward(features, target_size=target_size)
        return logit.argmax(dim=1)
