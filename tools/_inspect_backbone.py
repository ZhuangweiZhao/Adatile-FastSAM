#!/usr/bin/env python3
"""
检查 FastSAM Backbone 层结构与参数量 | Inspect FastSAM Backbone Layer Structure & Params
============================================================================================

功能 | Features:
    - 打印每层类型、参数数、维度信息 | Print per-layer type, param count, dimension info
    - 识别 C2f 模块详情 (cv1/cv2/m 子结构) | Identify C2f module details (cv1/cv2/m substructure)
    - 统计总参数量 | Total parameter count

用途 | Purpose:
    了解 FastSAM 模型结构，确定 hook 层位置 (P4/P8) 和特征维度。
    Understand FastSAM model structure, determine hook layer positions (P4/P8) and feature dimensions.

用法 | Usage::
    python tools/_inspect_backbone.py
"""

import sys
sys.path.insert(0, "thirdLibrary/FastSAM")
from fastsam import FastSAM

# 加载模型 | Load model
model = FastSAM("thirdLibrary/FastSAM/weights/FastSAM-x.pt")
seq = model.model.model

# ═══ 第一遍: 层结构总览 | Pass 1: Layer Structure Overview ═══
print("Layer structure (idx, type, param count):")
print("=" * 70)
total = 0
for i, layer in enumerate(seq):
    n = sum(p.numel() for p in layer.parameters())
    total += n
    name = type(layer).__name__
    print(f"  [{i:2d}] {name:<25s} params={n:>10,}  ({n/1e6:.2f}M)")
print(f"  Total: {total:>10,} ({total/1e6:.2f}M)")

# ═══ 第二遍: C2f 模块详情 (特征提取核心模块) | Pass 2: C2f Block Details (core feature extraction module) ═══
print()
print("C2f blocks detail:")
for i, layer in enumerate(seq):
    if "C2f" in type(layer).__name__:
        n = sum(p.numel() for p in layer.parameters())
        print(f"  [{i:2d}] {type(layer).__name__} params={n:>10,} ({n/1e6:.2f}M):")
        # cv1: 1×1 降维卷积 | cv1: 1×1 channel reduction conv
        if hasattr(layer, "cv1"):
            print(f"        cv1: {type(layer.cv1).__name__}  "
                  f"ch_in={layer.cv1.conv.in_channels}, ch_out={layer.cv1.conv.out_channels}")
        # cv2: 1×1 输出投影卷积 | cv2: 1×1 output projection conv
        if hasattr(layer, "cv2"):
            print(f"        cv2: {type(layer.cv2).__name__}  "
                  f"ch_in={layer.cv2.conv.in_channels}, ch_out={layer.cv2.conv.out_channels}")
        # m: 瓶颈层序列 (Bottleneck sequence) | m: bottleneck sequence
        if hasattr(layer, "m"):
            m = layer.m
            if hasattr(m, "__len__"):
                print(f"        m: nn.Sequential with {len(m)} bottlenecks")
                for j, b in enumerate(m):
                    bn = sum(p.numel() for p in b.parameters())
                    print(f"          [{j}] {type(b).__name__} params={bn:>10,}")
            else:
                print(f"        m: {type(m).__name__}")
