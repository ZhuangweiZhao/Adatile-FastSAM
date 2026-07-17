"""
Prompt-Conditioned Decoder | Prompt 条件解码器.
================================================

基于 PureDecoderP3P4，仅增加 DefectPrompt Generator 对 P4 的全局条件注入。
Based on PureDecoderP3P4 — the ONLY change is global prompt conditioning on P4.

消融对照 | Ablation:
    PureDecoderP3P4(p3, p4)          → baseline, no prompt
    PromptDecoderP3P4(p2, p3, p4)    → +DefectPrompt, same decoder structure

Δ = PromptDecoder − PureDecoderP3P4 → prompt 的纯贡献
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.prompt.defect_prompt import DefectPromptGenerator, prompt_global_conditioning


class PromptDecoderP3P4(nn.Module):
    """
    Prompt 条件的 P3+P4 解码器 | Prompt-Conditioned P3+P4 Decoder.

    与 PureDecoderP3P4 唯一区别:
        P4 特征在进入 proj 前拼接全局 prompt 向量 (concat conditioning)。
    Only difference from PureDecoderP3P4:
        P4 features receive global prompt concatenation before proj.

    P3 路径完全不变 → 保证公平消融。
    P3 path completely unchanged → ensures fair ablation.

    Parameters
    ----------
    p2_channels : int
        P2 特征通道数 | P2 feature channels (FastSAM-x: 160).
    p3_channels : int
        P3 特征通道数 | P3 feature channels (FastSAM-x: 960).
    p4_channels : int
        P4 特征通道数 | P4 feature channels (FastSAM-x: 1280).
    out_channels : int
        输出类别数 | Number of output classes (4 for NEU-Seg).
    prompt_dim : int
        Prompt token 维度 | Prompt token dimension.
    num_prompts : int
        Prompt 数量 K | Number of prompts K.
    """

    def __init__(
        self,
        p2_channels: int = 160,
        p3_channels: int = 960,
        p4_channels: int = 1280,
        out_channels: int = 4,
        prompt_dim: int = 256,
        num_prompts: int = 5,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.num_prompts = num_prompts

        # ── DefectPrompt Generator (~30K) ──
        self.prompt_gen = DefectPromptGenerator(
            in_channels=p2_channels,
            prompt_dim=prompt_dim,
            num_prompts=num_prompts,
        )

        # ── P4 路径 (主力，接收 prompt 条件) | P4 Path (workhorse, receives prompt) ──
        # p4_channels + prompt_dim → projected
        self.p4_proj = nn.Sequential(
            nn.Conv2d(p4_channels + prompt_dim, 256, kernel_size=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True),
        )
        self.p4_refine = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )
        self.p4_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

        # ── P3 路径 (边界细化，不加 prompt) | P3 Path (boundary, no prompt) ──
        self.p3_proj = nn.Sequential(
            nn.Conv2d(p3_channels, 128, kernel_size=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )
        self.p3_refine = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
        )
        self.p3_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

        # ── 双路融合 | Two-Way Fusion ──
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming 初始化 | Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, p2_features: torch.Tensor, p3_features: torch.Tensor,
                p4_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 | Forward pass.

        :param p2_features: [B, C2, H/4, W/4] P2 特征，用于 prompt 生成。
        :param p3_features: [B, C3, H/8, W/8] P3 特征。
        :param p4_features: [B, C4, H/16, W/16] P4 特征。
        :return:
            pred: [C, H/4, W/4] softmax probabilities (batch squeezed for consistency).
            heatmaps: [K, H/4, W/4] attention heatmaps (for visualization).
        """
        # ── 生成 defect prompt | Generate defect prompts ──
        prompt_emb, heatmaps = self.prompt_gen(p2_features)  # [B, K, D], [B, K, H2, W2]

        # ── 全局条件注入 P4 | Global conditioning on P4 ──
        p4_cond = prompt_global_conditioning(prompt_emb, p4_features, mode="concat")

        # ── P4 路径: H/16 → H/4 | P4 Path: H/16 → H/4 ──
        p4_x = self.p4_proj(p4_cond)          # [B, 256, H/16, W/16]
        p4_x = self.p4_refine(p4_x)            # [B, 64, H/16, W/16]
        p4_logit = self.p4_head(p4_x)          # [B, out, H/16, W/16]

        # ── P3 路径: H/8 → H/4 (不变 | unchanged) ──
        p3_x = self.p3_proj(p3_features)       # [B, 128, H/8, W/8]
        p3_x = self.p3_refine(p3_x)             # [B, 64, H/8, W/8]
        p3_logit = self.p3_head(p3_x)           # [B, out, H/8, W/8]

        # ── 统一上采样到 H/4 | Unified upsample to H/4 ──
        H_out = p3_logit.shape[2] * 2
        W_out = p3_logit.shape[3] * 2

        p4_up = F.interpolate(p4_logit, size=(H_out, W_out),
                              mode="bilinear", align_corners=False)
        p3_up = F.interpolate(p3_logit, size=(H_out, W_out),
                              mode="bilinear", align_corners=False)

        # ── 融合 | Fusion ──
        fused = torch.cat([p4_up, p3_up], dim=1)  # [B, 2*out, H/4, W/4]
        pred = self.fusion(fused)                  # [B, out, H/4, W/4]

        # 返回 [C, H/4, W/4] (squeeze batch to match PureDecoderP3P4 API)
        # Return [C, H/4, W/4] (squeeze batch to match PureDecoderP3P4 API)
        pred_prob = pred.softmax(dim=1)
        heatmaps_out = heatmaps

        if pred_prob.shape[0] == 1:
            pred_prob = pred_prob.squeeze(0)
            heatmaps_out = heatmaps_out.squeeze(0)

        return pred_prob, heatmaps_out
