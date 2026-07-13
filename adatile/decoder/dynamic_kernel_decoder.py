"""
Prototype-Conditioned Dynamic Kernel Decoder | 原型条件动态核解码器.
=====================================================================

基于 CondInst / BlendMask 思想，将 Prototype 直接映射为动态卷积核权重，
在共享特征图上生成多个实例掩码。

Core idea: Prototype → N dynamic conv kernels → applied to shared feature map → N instance masks.
This replaces the single-channel semantic output with multi-instance output,
directly addressing the "blob problem" (adjacent same-class objects merging into one CC).

设计动机 | Design Motivation:
    Current decoder: P4 → Conv → single probability map → CC → instances
    Problem: Adjacent same-class objects → 1 merged blob → 1 TP + 1 FN (Recall Ceiling)
    Fix: Decoder outputs N separate masks directly, Prototype controls instance generation

架构总览 | Architecture Overview::

    Support Set → P8 MAP → Prototype [feat_dim]
                                   │
                          ┌────────▼────────┐
                          │  Kernel Generator │  MLP: feat_dim → N × kernel_dim
                          │  (per-class)      │  Generates N dynamic conv kernels
                          └────────┬────────┘
                                   │ kernels [N, kernel_dim, 1, 1]
                                   │
    Query Image → Backbone ──┐     │
    ├── P3 [960, H/8, W/8]   │     │
    └── P4 [640, H/16, W/16] │     │
              │               │     │
         ┌────▼────┐          │     │
         │   FPN   │  P3→lateral→upsample ─┐
         │  Fusion │  P4→lateral───────────┤
         └────┬────┘          concat→fuse  │
              │ [fpn_dim, H/8, W/8]        │
         ┌────▼────┐                       │
         │  Mask   │ Conv→Norm→ReLU blocks │
         │ Feature │ Shared feature map    │
         └────┬────┘                       │
              │ [kernel_dim, H/8, W/8]     │
              └────────────┬───────────────┘
                           │
                    Dynamic Conv (1×1):
                    mask_i = Σ_c kernel_i[c] × feature[c]
                           │
                           ▼
                    N 个实例掩码 [N, H/8, W/8]
                           │
                           ▼
                    Sigmoid → Upsample → Per-Instance Masks

训练 | Training:
    Hungarian Matching: pred_masks ↔ GT instance masks
    Loss: BCE + Dice per matched pair
    Unmatched predictions pulled toward zero

推理 | Inference:
    Score threshold (mask mean) → filter → NMS if needed → instance masks

与现有 Decoder 对比 | Comparison with Existing Decoders:
    AdaptiveSparseDecoder:  P4 → 1 mask (semantic)
    AdaptiveDecoderP3P4:    P3+P4 → 1 mask (semantic, multi-scale)
    DynamicKernelDecoder:   P3+P4+Proto → N masks (instance-level)

用法 | Usage::

    from adatile.decoder.dynamic_kernel_decoder import DynamicKernelDecoder

    decoder = DynamicKernelDecoder(p3_channels=960, p4_channels=640, n_kernels=16)
    masks = decoder(p3_features, p4_features, proto_masks, support_proto)
    # → [N_kernels, H/8, W/8] instance masks (sigmoid, ∈ [0, 1])
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.sparse.coefficient_predictor import ProtoCoeffPredictor


class DynamicKernelDecoder(nn.Module):
    """
    Prototype-conditioned dynamic kernel instance decoder.
    原型条件动态核实例分割解码器。

    Key innovation: Prototype → Kernels → Instances.
    Unlike AdaptiveSparseDecoder (prototype → coefficients → 1 mask via proto basis),
    this decoder generates multiple instance masks directly via dynamic convolution.

    Parameters
    ----------
    p3_channels : int
        FastSAM P3 特征通道数 | P3 feature channels (default 960).
    p4_channels : int
        FastSAM P4 特征通道数 | P4 feature channels (default 640).
    proto_dim : int
        Proto mask 数量 | Number of proto masks (default 32, kept for backward compat).
    n_kernels : int
        每类生成的动态核数量 (= 每类最大实例数) | Kernels per class (= max instances per class).
    kernel_dim : int
        动态核维度 (特征图通道数) | Kernel dimension (feature map channels).
    fpn_dim : int
        FPN 输出通道数 | FPN output channels.
    normalize_proto : str
        Proto basis 归一化模式 | Proto basis normalization mode.
    """

    def __init__(
        self,
        p3_channels: int = 960,
        p4_channels: int = 640,
        proto_dim: int = 32,
        n_kernels: int = 16,
        kernel_dim: int = 256,
        fpn_dim: int = 256,
        normalize_proto: str = "none",
    ):
        super().__init__()

        self.p3_channels = p3_channels
        self.p4_channels = p4_channels
        self.proto_dim = proto_dim
        self.n_kernels = n_kernels
        self.kernel_dim = kernel_dim
        self.fpn_dim = fpn_dim
        self.normalize_proto = normalize_proto

        # ── 诊断: forward 时收集前向统计 (默认关闭) | optional forward-stat collection ──
        self.collect_stats = False
        self.last_stats: dict = {}

        # ═══════════════════════════════════════════════════════════════
        # 1. FPN 多尺度融合 | Multi-Scale Feature Fusion
        # ═══════════════════════════════════════════════════════════════
        # P3 (stride 8, high-res) + P4 (stride 16, semantic) → fused [fpn_dim, H/8, W/8]
        # Lateral connections: project each level to fpn_dim
        # Top-down: upsample P4 to P3 resolution, then fuse
        self.lateral_p4 = nn.Sequential(
            nn.Conv2d(p4_channels, fpn_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(fpn_dim, affine=True),
            nn.ReLU(inplace=True),
        )
        self.lateral_p3 = nn.Sequential(
            nn.Conv2d(p3_channels, fpn_dim, kernel_size=1, bias=False),
            nn.InstanceNorm2d(fpn_dim, affine=True),
            nn.ReLU(inplace=True),
        )
        # Post-fusion smoothing
        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * 2, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(fpn_dim, affine=True),
            nn.ReLU(inplace=True),
        )

        # ═══════════════════════════════════════════════════════════════
        # 2. Mask Feature (共享特征图) | Shared Mask Feature Map
        # ═══════════════════════════════════════════════════════════════
        # 动态核将在此特征图上做 1×1 卷积生成实例掩码
        # Dynamic kernels are applied on this feature map via 1×1 conv
        self.mask_feat = nn.Sequential(
            nn.Conv2d(fpn_dim, kernel_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(kernel_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_dim, kernel_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(kernel_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_dim, kernel_dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(kernel_dim, affine=True),
            nn.ReLU(inplace=True),
        )

        # ═══════════════════════════════════════════════════════════════
        # 3. Kernel Generator (动态核生成器) | Dynamic Kernel Generator
        # ═══════════════════════════════════════════════════════════════
        # Prototype [feat_dim] → MLP → [N_kernels × kernel_dim]
        # 每个 kernel 是一个 1×1 conv 权重, 应用于 mask_feat 生成一个实例掩码
        # Each kernel is a 1×1 conv weight, applied on mask_feat to produce one instance mask
        self.kernel_generator = nn.Sequential(
            nn.Linear(p4_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_kernels * kernel_dim),
        )

        # ═══════════════════════════════════════════════════════════════
        # 4. Proto Coefficient Predictor (保留, 用于 proto mask 生成)
        #    Proto Coeff Predictor (retained for proto mask generation)
        # ═══════════════════════════════════════════════════════════════
        # 保留 proto 支路作为辅助信号 (提供全局形状先验)
        # Keep proto branch as auxiliary signal (global shape prior)
        self.coeff_predictor = ProtoCoeffPredictor(
            proto_dim=proto_dim,
            feat_dim=p4_channels,
            hidden_dim=256,
        )

        # ── Proto basis 归一化 | Proto basis normalization ──
        self.proto_norm = (
            nn.InstanceNorm2d(proto_dim, affine=True)
            if normalize_proto == "layernorm"
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重 (跳过 ProtoCoeffPredictor, 其自行初始化)
        Initialize weights (skip ProtoCoeffPredictor, self-initializing).
        """
        for name, module in self.named_modules():
            if "coeff_predictor" in name:
                continue
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _normalize_proto(self, proto_masks: torch.Tensor) -> torch.Tensor:
        """归一化 proto basis 幅值 | Normalize proto basis magnitude.

        与 AdaptiveSparseDecoder._normalize_proto() 相同逻辑。
        Same logic as AdaptiveSparseDecoder._normalize_proto().
        """
        mode = self.normalize_proto
        if mode == "none":
            return proto_masks
        C, H, W = proto_masks.shape
        if mode == "l2":
            flat = F.normalize(proto_masks.reshape(C, -1), p=2, dim=1)
            return flat.view(C, H, W)
        if mode == "scale":
            return proto_masks / (float(H * W) ** 0.5)
        if mode == "layernorm":
            return self.proto_norm(proto_masks.unsqueeze(0)).squeeze(0)
        raise ValueError(f"unknown normalize_proto: {mode}")

    def _build_fpn_features(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
    ) -> torch.Tensor:
        """
        FPN 多尺度融合 | FPN multi-scale fusion.

        P3 (H/8) + P4 (H/16) → upsample P4 to H/8 → concat with P3 → fuse.

        :param p3: [B, 960, H/8, W/8]
        :param p4: [B, 640, H/16, W/16]
        :return: [B, fpn_dim, H/8, W/8]
        """
        # Lateral projections
        lat_p4 = self.lateral_p4(p4)  # [B, fpn_dim, H/16, W/16]
        lat_p3 = self.lateral_p3(p3)  # [B, fpn_dim, H/8, W/8]

        # Top-down: upsample P4 to P3 resolution
        p4_up = F.interpolate(
            lat_p4, size=lat_p3.shape[2:],
            mode="bilinear", align_corners=False,
        )  # [B, fpn_dim, H/8, W/8]

        # Concat + fuse
        fused = torch.cat([lat_p3, p4_up], dim=1)  # [B, 2*fpn_dim, H/8, W/8]
        return self.fpn_fuse(fused)  # [B, fpn_dim, H/8, W/8]

    def _generate_kernels(self, support_proto: torch.Tensor) -> torch.Tensor:
        """
        从 prototype 生成动态核 | Generate dynamic kernels from prototype.

        :param support_proto: [feat_dim] 或 [1, feat_dim] prototype.
        :return: [N, kernel_dim] 动态核权重 | dynamic kernel weights.
        """
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)
        kernels_flat = self.kernel_generator(support_proto)  # [N * kernel_dim]
        kernels = kernels_flat.view(self.n_kernels, self.kernel_dim)  # [N, kernel_dim]
        return kernels

    def forward(
        self,
        p3_features: torch.Tensor,
        p4_features: torch.Tensor,
        proto_masks: torch.Tensor,
        support_proto: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播 | Forward pass.

        :param p3_features: [B, 960, H/8, W/8] FastSAM P3 特征 | FastSAM P3 features.
        :param p4_features: [B, 640, H/16, W/16] FastSAM P4 特征 | FastSAM P4 features.
        :param proto_masks: [proto_dim, H/4, W/4] 或 [1, proto_dim, H/4, W/4] Proto basis masks.
        :param support_proto: [feat_dim] 或 [1, feat_dim] Support prototype (P8 MAP, L2-normalized).
        :return: (masks, proto_mask)
            - masks: [N_kernels, H/8, W/8] instance masks (sigmoid, ∈ [0, 1])
            - proto_mask: [H/4, W/4] proto-coefficient mask (auxiliary, for loss/comparison)
        """
        # ── 输入标准化 | Input normalization ──
        if proto_masks.dim() == 4:
            proto_masks = proto_masks.squeeze(0)
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)

        # ═══════════════════════════════════════════════════════════════
        # Step 1: Proto mask (auxiliary branch, kept for backward compat)
        # ═══════════════════════════════════════════════════════════════
        proto_masks_norm = self._normalize_proto(proto_masks)
        coeffs = self.coeff_predictor(support_proto.unsqueeze(0))  # [1, proto_dim]
        proto_mask = self.coeff_predictor.generate_mask(
            coeffs, proto_masks_norm
        ).squeeze(0)  # [H/4, W/4]

        # ═══════════════════════════════════════════════════════════════
        # Step 2: FPN 多尺度融合 | Multi-scale feature fusion
        # ═══════════════════════════════════════════════════════════════
        fpn_feat = self._build_fpn_features(p3_features, p4_features)
        # → [B, fpn_dim, H/8, W/8]

        # ═══════════════════════════════════════════════════════════════
        # Step 3: 共享特征图 | Shared mask feature map
        # ═══════════════════════════════════════════════════════════════
        mask_feat = self.mask_feat(fpn_feat)  # [B, kernel_dim, H/8, W/8]

        # ═══════════════════════════════════════════════════════════════
        # Step 4: 生成动态核 | Generate dynamic kernels
        # ═══════════════════════════════════════════════════════════════
        kernels = self._generate_kernels(support_proto)  # [N, kernel_dim]

        # ═══════════════════════════════════════════════════════════════
        # Step 5: 动态卷积 → N 个实例掩码 | Dynamic conv → N instance masks
        # ═══════════════════════════════════════════════════════════════
        # kernels: [N, kernel_dim] → [N, kernel_dim, 1, 1]
        # mask_feat: [B, kernel_dim, H/8, W/8]
        # conv2d → [B, N, H/8, W/8]
        kernel_weights = kernels.view(self.n_kernels, self.kernel_dim, 1, 1)
        masks_logit = F.conv2d(mask_feat, kernel_weights)  # [B, N, H/8, W/8]
        masks = torch.sigmoid(masks_logit)  # [B, N, H/8, W/8]

        # ── 可选: 收集前向统计 | Optional: collect forward stats ──
        if self.collect_stats:
            with torch.no_grad():
                k_norm = kernels.norm(dim=1)  # [N]
                self.last_stats = {
                    "kernel_norm_mean": float(k_norm.mean().item()),
                    "kernel_norm_std": float(k_norm.std().item()),
                    "mask_mean_per_kernel": float(masks.mean(dim=(2, 3)).mean().item()),
                    "mask_max_per_kernel": float(masks.max(dim=3)[0].max(dim=2)[0].mean().item()),
                    "proto_mask_mean": float(proto_mask.mean().item()),
                }

        # Squeeze batch dim for backward compatibility with existing training loop
        masks = masks.squeeze(0)  # [N_kernels, H/8, W/8]
        return masks, proto_mask

    def forward_with_proto_only(
        self,
        proto_masks: torch.Tensor,
        support_proto: torch.Tensor,
    ) -> torch.Tensor:
        """仅使用 proto mask 生成 (消融用) | Proto-only mask generation (for ablation)."""
        if proto_masks.dim() == 4:
            proto_masks = proto_masks.squeeze(0)
        if support_proto.dim() == 2:
            support_proto = support_proto.squeeze(0)

        coeffs = self.coeff_predictor(support_proto.unsqueeze(0))
        return self.coeff_predictor.generate_mask(coeffs, proto_masks).squeeze(0)

    def get_submodule_params(self) -> dict[str, int]:
        """获取各子模块参数量 | Get parameter count per submodule."""
        def _count(m):
            return sum(p.numel() for p in m.parameters())

        return {
            "fpn": _count(self.lateral_p4) + _count(self.lateral_p3) + _count(self.fpn_fuse),
            "mask_feat": _count(self.mask_feat),
            "kernel_generator": _count(self.kernel_generator),
            "coeff_predictor": _count(self.coeff_predictor),
            "total": _count(self),
        }

    def __repr__(self) -> str:
        params = self.get_submodule_params()
        return (
            f"DynamicKernelDecoder(p3_ch={self.p3_channels}, p4_ch={self.p4_channels}, "
            f"n_kernels={self.n_kernels}, kernel_dim={self.kernel_dim}, "
            f"total_params={params['total'] / 1e3:.1f}K)"
        )
