"""
CAT-SAM-A Style Feature Adaptation for FastSAM Few-Shot.
==========================================================
CAT-SAM-A 风格特征适配 — 将 CAT-SAM-A 的 PromptGenerator 机制适配到 FastSAM 少样本分割.

CAT-SAM-A 核心思想 | Core Ideas (from cat-sam/cat_sam/models/encoders.py):
    1. FFT 高频特征作为 handcrafted 条件信号 | FFT high-freq as handcrafted conditioning
    2. Learnable embeddings 从 patch features 投影 | Learnable embeddings from patch features
    3. 多源特征 → MLP → 空间 prompt → 注入编码器 | Multi-source → MLP → spatial prompt → inject

FastSAM 适配 | FastSAM Adaptation:
    ViT blocks (per-layer injection) → YOLOv8 neck (P3/P4 injection)
    Self-conditioning (CAT-SAM-A on same image) → Cross-conditioning (support → query)

Architecture | 架构:
    Support Images → FFT (high-freq) ────────────────→ handcrafted_feat ─┐
    Support Images → FastSAM → FG tokens ──→ proj ──→ embedding_feat ──┤
                                                                         ├─→ PromptGenerator → spatial prompts
    Query P3/P4 ────────────────────────────────────────────────────────→ prompts + features → conditioned features
                                                                         │
    Conditioned P3/P4 → Decoder → mask

Key Modules | 关键模块:
    - FFTPromptExtractor: 从 support image 提取 FFT 高频特征 | Extract FFT high-freq from support
    - CrossPromptGenerator: 多源特征 → 空间 prompt | Multi-source → spatial prompt
    - CATSAMAFewShotDecoder: 完整的 CAT-SAM-A 风格 few-shot decoder

Paper Reference | 论文引用:
    CAT-SAM: Conditional Tuning for Segment Anything Model
    https://github.com/.../cat-sam

用法 | Usage::
    python tools/train/train_fewshot.py \
        --dataset isaid5i --fold 0 --shot 5 \
        --decoder catsama --adapter catsama \
        --epochs 40 --amp --device cuda
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════
# FFT Feature Extractor | FFT 特征提取器
# ═══════════════════════════════════════════════════════════════════
# 复刻 CAT-SAM-A 的 handcrafted feature: FFT 高通滤波提取高频空间结构
# Replicates CAT-SAM-A handcrafted feature: FFT high-pass for spatial structure


class FFTPromptExtractor(nn.Module):
    """
    FFT-based high-frequency feature extractor for spatial structure.
    基于 FFT 的高频空间结构特征提取器.

    复刻 CAT-SAM-A PromptGenerator.init_handcrafted() + fft():
    - 对输入图像做 2D FFT → 高通滤波 (保留高频, 去除低频) → IFFT
    - 将高频空间结构投影到 prompt 空间
    - 输出与 query 特征图相同空间分辨率的 spatial prompt

    Replicates CAT-SAM-A:
    - 2D FFT → high-pass filter (keep high-freq, remove low-freq) → IFFT
    - Project high-freq spatial structure to prompt space
    - Output spatial prompt at same resolution as query feature maps

    Parameters | 参数:
        freq_rate: FFT 高通滤波保留比例 (越低保留越少低频, 越高保留越多高频).
                   CAT-SAM-A default: 0.25.
        prompt_dim: 输出 prompt 通道数 | Output prompt channels.
    """

    def __init__(self, freq_rate: float = 0.25, prompt_dim: int = 256):
        super().__init__()
        self.freq_rate = freq_rate
        self.prompt_dim = prompt_dim

        # Patch embed 风格的投影: 将原始图像投影到 prompt 空间
        # Patch-embed style projection: raw image → prompt space
        # 使用 stride=1 conv 保持空间分辨率 | stride=1 to preserve spatial resolution
        self.proj = nn.Sequential(
            nn.Conv2d(3, prompt_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(prompt_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(prompt_dim // 2, prompt_dim, kernel_size=1, bias=False),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def fft_highpass(self, x: torch.Tensor) -> torch.Tensor:
        """
        FFT 高通滤波: 保留高频, 去除低频中心 | FFT high-pass: keep high-freq, remove low-freq center.

        复刻 CAT-SAM-A 的 fft() 方法 | Replicates CAT-SAM-A fft() method.

        :param x: [B, 3, H, W] 输入图像 | Input image.
        :return: [B, 3, H, W] 高频特征图 | High-frequency feature map.
        """
        B, C, H, W = x.shape
        # 创建高通 mask: 中心区域置零, 保留边缘高频 | Create high-pass mask: zero center
        mask = torch.zeros((H, W), device=x.device)
        line = int((H * W * self.freq_rate) ** 0.5 // 2)
        h_center, w_center = H // 2, W // 2
        mask[h_center - line:h_center + line, w_center - line:w_center + line] = 1

        # 2D FFT → shift → high-pass → inverse shift → IFFT
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        fft = fft * (1 - mask[None, None, :, :])  # high-pass filter
        fr, fi = fft.real, fft.imag
        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        return torch.abs(inv)

    def forward(self, x: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """
        :param x: [B, 3, H_img, W_img] 输入图像 | Input image.
        :param target_size: (H_target, W_target) 目标空间分辨率 | Target spatial resolution.
        :return: [B, prompt_dim, H_target, W_target] 高频空间 prompt | High-freq spatial prompt.
        """
        # FFT 高通滤波 + 投影到 prompt 空间
        # FFT high-pass + project to prompt space
        hf = self.fft_highpass(x)  # [B, 3, H, W]
        prompt = self.proj(hf)     # [B, prompt_dim, H, W]
        # 缩放到目标分辨率 | Resize to target resolution
        if prompt.shape[2:] != target_size:
            prompt = F.interpolate(prompt, size=target_size, mode="bilinear", align_corners=False)
        return prompt


# ═══════════════════════════════════════════════════════════════════
# Cross Prompt Generator | 交叉 Prompt 生成器
# ═══════════════════════════════════════════════════════════════════
# CAT-SAM-A 的 get_prompt() 机制, 适配为 support → query 交叉条件
# CAT-SAM-A get_prompt() mechanism, adapted for support → query cross-conditioning


class CrossPromptGenerator(nn.Module):
    """
    Support-to-Query cross prompt generator (CAT-SAM-A style).
    Support→Query 交叉 prompt 生成器 (CAT-SAM-A 风格).

    将 support FG tokens + support FFT features 融合生成空间 prompt,
    注入到 query 特征图中, 实现 support-conditioned query encoding.

    Fuses support FG tokens + support FFT features → spatial prompts,
    injects into query feature maps for support-conditioned encoding.

    Replicates CAT-SAM-A PromptGenerator.get_prompt():
        joint_feature = handcrafted_feature(FFT) + embedding_feature(FG tokens)
        prompt = lightweight_mlp(joint_feature)
        prompt = shared_mlp(prompt)

    Parameters | 参数:
        feat_dim: support token 特征维度 | Support token feature dim (P4=1280).
        prompt_dim: prompt 通道数 | Prompt channels (default 256).
        hidden_ratio: MLP 瓶颈比例 | MLP bottleneck ratio (0.25 = CAT-SAM-A default).
    """

    def __init__(self, feat_dim: int = 1280, prompt_dim: int = 256,
                 hidden_ratio: float = 0.25):
        super().__init__()
        self.feat_dim = feat_dim
        self.prompt_dim = prompt_dim
        hidden_dim = int(prompt_dim * hidden_ratio)

        # Support FG token → embedding feature (复刻 embedding_generator)
        # Support FG token → embedding feature (replicates embedding_generator)
        self.token_proj = nn.Sequential(
            nn.Linear(feat_dim, prompt_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prompt_dim, prompt_dim),
        )

        # FFT handcrafted → align to prompt_dim (已由 FFTPromptExtractor 处理)
        # FFT handcrafted → aligned to prompt_dim (handled by FFTPromptExtractor)

        # Lightweight MLP: prompt_dim → prompt_dim (复刻 lightweight_mlp)
        # Lightweight MLP: prompt_dim → prompt_dim (replicates lightweight_mlp)
        self.lightweight_mlp = nn.Sequential(
            nn.Conv2d(prompt_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, prompt_dim, kernel_size=1, bias=False),
        )

        # Shared MLP: prompt_dim → feat_dim (复刻 shared_mlp, 输出回原特征空间)
        # Shared MLP: prompt_dim → feat_dim (replicates shared_mlp, output to feat space)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(prompt_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feat_dim, kernel_size=1, bias=False),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        query_feat: torch.Tensor,
        handcrafted_feat: torch.Tensor,
        support_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        生成 support-conditioned 空间 prompt 并注入 query 特征.
        Generate support-conditioned spatial prompt and inject into query features.

        :param query_feat: [B, C_q, H, W] query 特征图 (P3 或 P4) | Query feature map.
        :param handcrafted_feat: [B, prompt_dim, H, W] FFT handcrafted 特征 | FFT features.
        :param support_tokens: [N, C_q] support FG tokens (raw, before token_proj).
        :return: [B, C_q, H, W] conditioned query features | 条件化后的 query 特征.
        """
        B, C_q, H, W = query_feat.shape

        # ── 1. Support FG tokens → embedding feature | FG token → 嵌入特征 ──
        # 对 K 个 token 取平均 → 全局 support 表征 (cat-sam 风格)
        # Average over K tokens → global support representation (cat-sam style)
        token_embed = self.token_proj(support_tokens).mean(dim=0)  # [prompt_dim]
        embedding_feat = token_embed[None, :, None, None].expand(B, -1, H, W)  # [B, prompt_dim, H, W]

        # ── 2. Joint feature: handcrafted(FFT) + embedding(FG tokens) ──
        # 复刻 CAT-SAM-A: joint_feature += handcrafted_feature + embedding_feature
        joint = handcrafted_feat + embedding_feat  # [B, prompt_dim, H, W]

        # ── 3. Lightweight MLP → per-position prompt ──
        prompt = self.lightweight_mlp(joint)  # [B, prompt_dim, H, W]

        # ── 4. Shared MLP → project back to feature space → residual add ──
        delta = self.shared_mlp(prompt)  # [B, C_q, H, W]
        return query_feat + delta

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════
# CAT-SAM-A Style Few-Shot Decoder | CAT-SAM-A 风格少样本解码器
# ═══════════════════════════════════════════════════════════════════


class CATSAMAFewShotDecoder(nn.Module):
    """
    CAT-SAM-A style few-shot decoder for FastSAM.
    CAT-SAM-A 风格 FastSAM 少样本解码器.

    核心流程 | Core Pipeline:
        1. Support images → FFT (high-freq spatial structure)
        2. Support images → FastSAM → FG pixel tokens
        3. CrossPromptGenerator: FFT + FG tokens → spatial prompts → condition query features
        4. Conditioned P3+P4 → CNN decoder → mask

    Key insight | 关键洞察:
        CAT-SAM-A uses self-conditioning (image's own FFT → own features).
        We adapt this to CROSS-conditioning (support FFT+FG → query features).
        This creates a support-conditioned query feature space without learned attention.

    Parameters | 参数:
        feat_dim_p3: P3 通道数 | P3 channels (960).
        feat_dim_p4: P4 通道数 | P4 channels (1280).
        prompt_dim: prompt 内部维度 | Prompt internal dim (256).
        hidden_dim: decoder CNN 隐藏维 | Decoder CNN hidden dim (128).
        max_tokens: support FG token 最大采样数 | Max support FG token count (128).
        use_fft: 是否使用 FFT handcrafted 特征 | Whether to use FFT features.
    """

    def __init__(
        self,
        feat_dim_p3: int = 960,
        feat_dim_p4: int = 1280,
        prompt_dim: int = 256,
        hidden_dim: int = 128,
        max_tokens: int = 128,
        use_fft: bool = True,
    ):
        super().__init__()
        self.feat_dim_p3 = feat_dim_p3
        self.feat_dim_p4 = feat_dim_p4
        self.prompt_dim = prompt_dim
        self.max_tokens = max_tokens
        self.use_fft = use_fft

        # ── FFT Extractor (per scale) | FFT 提取器（每尺度）──
        if use_fft:
            self.fft_extractor = FFTPromptExtractor(freq_rate=0.25, prompt_dim=prompt_dim)

        # ── Cross Prompt Generators (per scale) | 交叉 Prompt 生成器（每尺度）──
        self.prompt_gen_p4 = CrossPromptGenerator(
            feat_dim=feat_dim_p4, prompt_dim=prompt_dim, hidden_ratio=0.25,
        )
        self.prompt_gen_p3 = CrossPromptGenerator(
            feat_dim=feat_dim_p3, prompt_dim=prompt_dim, hidden_ratio=0.25,
        )

        # ── P3/P4 压缩投影 | P3/P4 compression bottleneck ──
        # 将高维特征压缩到低维 bottleneck, 再 concatenate, 大幅减少 decoder 参数量
        # Compress high-dim features to low-dim bottleneck before concat
        self.compress_p3 = nn.Sequential(
            nn.Conv2d(feat_dim_p3, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )
        self.compress_p4 = nn.Sequential(
            nn.Conv2d(feat_dim_p4, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )

        # ── Decoder CNN | 解码器 CNN ──
        # Input: compressed P3(128) + compressed P4↑(128) = 256 channels → mask
        in_ch = hidden_dim * 2
        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 4), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 4, 1, kernel_size=1, bias=True),
        )

        self._log_params()

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        pg_p4 = self.prompt_gen_p4.num_params
        pg_p3 = self.prompt_gen_p3.num_params
        fft_n = sum(p.numel() for p in self.fft_extractor.parameters()) if self.use_fft else 0
        print(f"[CATSAMAFewShotDecoder] Total: {total:,} params "
              f"(PG-P4: {pg_p4:,}, PG-P3: {pg_p3:,}, FFT: {fft_n:,})")

    @property
    def num_prototypes(self):
        """Compat with prototype-based evaluation code."""
        return 1

    def _sample_tokens(self, tokens_raw: torch.Tensor) -> torch.Tensor:
        """Random sample up to max_tokens from support FG pixels."""
        N = tokens_raw.shape[0]
        if N <= self.max_tokens:
            return tokens_raw
        indices = torch.randperm(N, device=tokens_raw.device)[:self.max_tokens]
        return tokens_raw[indices]

    def forward(
        self,
        query_p3: torch.Tensor,
        query_p4: torch.Tensor,
        support_tokens_raw: torch.Tensor,
        target_size: tuple[int, int] | None = None,
        support_imgs: torch.Tensor | None = None,
        return_attn: bool = False,
        return_fused: bool = False,
    ) -> torch.Tensor:
        """
        :param query_p3: [B, 960, H3, W3] query P3 features.
        :param query_p4: [B, 1280, H4, W4] query P4 features.
        :param support_tokens_raw: [N, 1280] all support FG pixel vectors.
        :param target_size: (H, W) output mask size.
        :param support_imgs: [K, 3, H_s, W_s] support images (for FFT, optional).
        :return: [B, 1, H, W] mask logit.
        """
        B = query_p3.shape[0]
        H3, W3 = query_p3.shape[2:]

        # ── 0. Sample support tokens | 采样 support token ──
        sampled_tokens = self._sample_tokens(support_tokens_raw)

        # ── 1. FFT handcrafted features (if enabled) | FFT 手工特征 ──
        if self.use_fft and support_imgs is not None:
            # 对 support images 做 FFT 高通滤波
            # FFT high-pass on support images
            hf_p3 = self.fft_extractor(support_imgs, target_size=(H3, W3))
            H4, W4 = query_p4.shape[2:]
            hf_p4 = self.fft_extractor(support_imgs, target_size=(H4, W4))
        else:
            # Fallback: zero handcrafted features (仅用 FG token embedding)
            hf_p3 = torch.zeros(B, self.prompt_dim, H3, W3, device=query_p3.device)
            H4, W4 = query_p4.shape[2:]
            hf_p4 = torch.zeros(B, self.prompt_dim, H4, W4, device=query_p4.device)

        # ── 2. Cross Prompt Generation | 交叉 Prompt 生成 ──
        cond_p4 = self.prompt_gen_p4(query_p4, hf_p4, sampled_tokens)
        cond_p3 = self.prompt_gen_p3(query_p3, hf_p3, sampled_tokens)

        # ── 3. Multi-scale fusion | 多尺度融合 ──
        # P4 → upsample to P3 resolution → compress channels
        p4_up = F.interpolate(cond_p4, size=(H3, W3), mode="bilinear", align_corners=False)
        p4_compressed = self.compress_p4(p4_up)   # [B, hidden_dim, H3, W3]
        p3_compressed = self.compress_p3(cond_p3)  # [B, hidden_dim, H3, W3]
        fused = torch.cat([p3_compressed, p4_compressed], dim=1)  # [B, 2*hidden_dim, H3, W3]

        # ── 4. Decoder CNN | 解码器 CNN ──
        x = fused
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decoder[0:3](x)   # conv1
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decoder[3:6](x)   # conv2
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decoder[6:9](x)   # conv3
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.decoder[9:](x)    # 1×1 → mask

        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        if return_fused:
            return x, fused
        return x
