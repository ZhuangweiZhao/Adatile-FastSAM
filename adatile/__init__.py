"""
AdaTile-FastSAM v2.0
=====================
自适应稀疏 FastSAM，用于小样本高分辨率实例分割。
Adaptive Sparse FastSAM for Few-Shot High-Resolution Instance Segmentation.

两个核心创新 | Two innovations:
1. Ada-SPM — 密度监督的稀疏感知模块，学习重要性图 → Top-K 瓦片选择
   Density-supervised sparse perception module: learns importance maps → Top-K tile selection
2. 解耦稀疏训练 — 解码器始终接收全量特征；SPM 通过 GT 驱动的损失函数并行训练
   Decoupled Sparse Training — decoder always receives full features; SPM trained via GT-driven losses in parallel
"""

__version__ = "2.0.0.dev0"
