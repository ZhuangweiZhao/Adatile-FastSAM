#!/usr/bin/env python3
"""
ProtoModule — 可学习原型模块 (从 E007 恢复)
ProtoModule — Learnable Prototype Module (recovered from E007).
==============================================================

P4 → Embedding → Cosine Sim w/ Learnable Prototypes → Seg Logit.

结构约束 | Structural constraint:
    分割必须通过 N 个固定原型向量的 cosine similarity 完成。
    Segmentation must go through cosine similarity to N fixed prototypes.
    这迫使网络学习有意义的原型，而非任意线性组合。
    This forces the network to learn meaningful prototypes, not arbitrary linear combos.

用法 | Usage::

    from adatile.decoder.proto_module import ProtoModule

    proto = ProtoModule(in_channels=1280, embed_dim=128, n_protos=12)
    embedding, sim_maps, logit = proto(p4, temperature=0.1)
    hard_assign = proto.get_hard_assignment(p4)

注意 | Note:
    此文件从 git 历史恢复 (commit 3422193^)，原路径 tools/eval_e007_proto_module.py
    移至 adatile/decoder/ 作为正式库模块，供 viz_paper_a_p6.py 等使用。
    Recovered from git history, moved to adatile/decoder/ as a permanent library module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from adatile.metrics import format_param_count


class ProtoModule(nn.Module):
    """
    原型模块 | Prototype Module.

    Architecture:
        P4 [B, 1280, H/16, W/16]
             │
        1×1 Conv(1280 → embed_dim) + ReLU
             │
        Embedding [B, embed_dim, H/16, W/16]
             │
        Cosine Similarity with N learnable Prototype vectors
             │
        Similarity Maps [B, N, H/16, W/16]
             │
        1×1 Conv(N → 1) → Segmentation Logit (BCE training signal)
    """

    def __init__(self, in_channels: int = 1280, embed_dim: int = 128, n_protos: int = 8):
        """
        :param in_channels: P4 特征通道数 | P4 feature channels (FastSAM = 1280).
        :param embed_dim:   嵌入空间维度 | Embedding space dimension.
        :param n_protos:    原型数量 | Number of prototype vectors.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.n_protos = n_protos

        # 特征投影 | Feature projection
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # 可学习原型向量 | Learnable prototype vectors
        self.prototypes = nn.Parameter(torch.randn(n_protos, embed_dim) * 0.1)

        # 分割头 | Segmentation head — 1×1 linear combination of proto responses
        self.head = nn.Conv2d(n_protos, 1, kernel_size=1, bias=True)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"  ProtoModule: {format_param_count(n_params)} ({n_params:,})")
        print(f"    project={sum(p.numel() for p in self.project.parameters()):,}")
        print(f"    prototypes={self.prototypes.numel():,}")
        print(f"    head={sum(p.numel() for p in self.head.parameters()):,}")

    def forward(self, p4: torch.Tensor, temperature: float = 0.1
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        :param p4:          P4 feature map [B, 1280, H/16, W/16].
        :param temperature: Softmax temperature (lower = sharper assignment).

        :return: (embedding, sim_maps, logit) tuple:
            - embedding: [B, D, H, W]   低维嵌入 | Low-dim embedding
            - sim_maps:  [B, N, H, W]   Proto 相似度图 | Proto similarity maps
            - logit:     [B, 1, H, W]   分割 logit | Segmentation logit
        """
        # 投影 | Project P4 → embedding
        embedding = self.project(p4)  # [B, D, H, W]

        # L2 normalize for cosine similarity
        emb_norm = F.normalize(embedding, dim=1, p=2)          # [B, D, H, W]
        proto_norm = F.normalize(self.prototypes, dim=1, p=2)  # [N, D]

        # Cosine similarity: einsum [B,D,H,W] × [N,D] → [B,N,H,W]
        sim_maps = torch.einsum("bdhw,nd->bnhw", emb_norm, proto_norm)

        # Temperature scaling
        sim_maps = sim_maps / temperature

        # Segmentation logit: linear combination of proto responses
        logit = self.head(sim_maps)  # [B, 1, H, W]

        return embedding, sim_maps, logit

    def get_soft_assignment(self, p4: torch.Tensor, temperature: float = 0.1
                            ) -> torch.Tensor:
        """
        获取每个像素的软分配 (softmax over prototypes).
        Get soft assignment per pixel (softmax over prototypes).

        :return: [B, N, H, W] softmax probabilities.
        """
        _, sim_maps, _ = self.forward(p4, temperature)
        return F.softmax(sim_maps, dim=1)

    def get_hard_assignment(self, p4: torch.Tensor) -> torch.Tensor:
        """
        获取每个像素的硬分配 (argmax over prototypes).
        Get hard assignment per pixel (argmax over prototypes).

        :return: [B, H, W] prototype index per pixel (0 to N-1).
        """
        _, sim_maps, _ = self.forward(p4, temperature=0.01)
        return sim_maps.argmax(dim=1)  # [B, H, W]
