# D-01: ProtoOnly Few-Shot 实例分割训练

**日期**: 2026-07-03
**脚本**: `tools/train/train_instance_fewshot.py`
**数据**: iSAID-5i Fold 0, Novel classes, tile mode (256² tiles)
**设备**: 云服务器 (RTX 5090)

## 实验配置 | Experiment Config

| 参数 | 值 |
|------|-----|
| Decoder | **ProtoOnlyDecoder** (~427K params, 纯 MLP) |
| Backbone | FastSAM (frozen) |
| FDR | 加载但未使用（ProtoOnly 不接收 FDR） |
| K-shot | 5 |
| Episodes | 2000 |
| 训练类别 | Novel: [small_vehicle(9), harbor(15), swimming_pool(11), basketball_court(5), roundabout(12)] |
| 训练 tile 数 | 9,090 (train split) |
| 评估 tile 数 | 3,108 (val split) |
| LR | 1e-4 (AdamW) |

## 结果 | Results

### 核心指标

| 指标 | 最优值 | Episode | 最终值 |
|------|--------|---------|--------|
| **mIoU** | **0.2354** | 1500 | 0.1902 |
| **AP50** | **0.0000** | 全程 | 0.0000 |

### 训练曲线

```
Episode    10: loss=0.522  focal=0.066  dice=0.978  pred_mean=0.367
Episode   100: loss=0.414  focal=0.117  dice=0.711  pred_mean=0.364
Episode   300: loss=0.380  focal=0.220  dice=0.540  pred_mean=0.391  mIoU=0.150 ← 首次跃升
Episode   500: loss=0.486  focal=0.018  dice=0.954  pred_mean=0.152  mIoU=0.168
Episode   900: loss=0.284  focal=0.089  dice=0.479  pred_mean=0.449  mIoU=0.139
Episode  1100: loss=0.485  focal=0.006  dice=0.964  pred_mean=0.122  mIoU=0.186 ← 持续上升
Episode  1500: loss=0.484  focal=0.093  dice=0.875  pred_mean=0.157  mIoU=0.235 ← ★ BEST
Episode  2000: loss=0.482  focal=0.010  dice=0.955  pred_mean=0.080  mIoU=0.190
```

### Loss 分解趋势

| 阶段 | Focal Loss | Dice Loss | 现象 |
|------|-----------|-----------|------|
| 前 100 ep | 0.07→0.12 | 0.98→0.71 | 缓慢学习 FG |
| 100-500 ep | 0.22→0.02 | 0.54→0.95 | **Focal 坍塌: 模型学会"全预测 BG"** |
| 500-1500 ep | 0.01-0.15 | 0.44-0.99 | Dice 振荡, mIoU 缓慢上升→0.235 |
| 1500-2000 ep | 0.01 | 0.96 | mIoU 回退至 0.190, 可能过拟合 |

### Per-Class 分析（从 loss 模式推断）

| 类 | 典型 loss | pred_mean | 状态 |
|----|-----------|-----------|------|
| **cls5** basketball_court | focal=0.07-0.28, dice=0.38-0.83 | 0.35-0.50 | ★ 唯一有实质学习的类 |
| **cls15** harbor | focal=0.03-0.12, dice=0.44-0.95 | 0.17-0.33 | 有信号但不稳定 |
| **cls11** swimming_pool | focal=0.01-0.15, dice=0.57-0.99 | 0.06-0.37 | 时而学习时而崩塌 |
| **cls9** small_vehicle | focal=0.01-0.10, dice=0.65-0.99 | 0.08-0.28 | **小目标，几乎无学习** |
| **cls12** roundabout | focal=0.01-0.19, dice=0.77-0.99 | 0.06-0.24 | **仅 14 train tiles, 数据极度不足** |

## 关键发现 | Key Findings

### 1. "预测全背景" 坍塌 (Focal Collapse)

ProtoOnly 存在严重的 **Focal-Dice 不对称**:
- Focal loss 从 0.22 骤降至 0.01 → 模型学会了 "pred≈0 for BG pixels"，Focal 轻松达标
- Dice loss 从 0.54 反弹至 0.96 → 模型几乎不预测 FG，Dice 无法优化
- pred_mean 从 0.39 降至 0.08 → 模型倾向于输出越来越低的概率

**根因**: 32-d MLP 系数预测器从 1280-d 单一向量中无法提取足够的空间信息来区分 FG/BG。
当 FG 像素占比 <5% 时，predict-all-background 是 Focal loss 的最优解。

### 2. Self-Support Autoencoder 测试暴露架构瓶颈

当前评估使用 self-support (tile 自己的 GT mask → prototype → 预测自己)，等价于 autoencoder。
**连 autoencoder 都只能达到 mIoU=0.235**，说明:

> 瓶颈在 ProtoCoeffPredictor 的**表达能力**，不是 few-shot 的 sample efficiency。
> 32 个 SA-1B 预训练 proto 基函数无法有效表达航拍场景的物体形状。

### 3. 三类失败模式

| 失败模式 | 典型类 | 原因 |
|----------|--------|------|
| **基函数失配** | small_vehicle, roundabout | SA-1B 的自然物体 proto ≠ 航拍的矩形/圆形 |
| **数据极度不足** | roundabout (14 tiles) | K=5 shot 几乎覆盖全部数据，无多样性 |
| **尺度错位** | small_vehicle (小目标) | Stride 4 proto 上的小目标 = ~2-4px，无有效信号 |

### 4. 与 Zero-Shot Baseline 的对比

| 指标 | Zero-Shot (FastSAM) | ProtoOnly K=5 | 提升 |
|------|---------------------|---------------|------|
| **语义 mIoU** | ~0.01 (估计) | **0.235** | ~23× |
| **COCO AP50** | 0.023 | **0.000** | ❌ 退步 |
| 任务性质 | 实例分割 (per-instance mask) | 语义分割 (per-class blob) | — |

> ProtoOnly 学到的是 per-class 语义信号，不是 per-instance 实例分割。
> COCO AP=0 意味着连通分量分解无法从预测的 blob 中提取有意义的单个实例。

### 5. 对论文叙事的意义

```
Zero-Shot AP=0.015 → "SA-1B doesn't work on aerial imagery"
ProtoOnly mIoU=0.235, AP=0 → "Naive prototype matching is insufficient for instance segmentation"
AdaptiveSparse (next) → "P4 refinement + FDR attention restores instance-level precision"
```

ProtoOnly 的失败**不是无用的**——它精确地定位了瓶颈:
1. 需要 P4 特征精炼（补空间细节）
2. 需要 FDR 密度引导（聚焦高密度区域）
3. 需要更强的 mask head（不只是 32-d 线性组合）

## 后续 | Next

- **D-02**: AdaptiveSparseDecoder (P4 refinement) @ K=5 → 预期 mIoU + AP 双重提升
- **D-03**: FDR 消融 @ K=5 → 密度引导的加速/质量 tradeoff
- **诊断**: 检查 self-support 下 per-class 的 IoU 分布（了解哪些类最需要 refinement）

## 文件 | Files

- 训练脚本: `tools/train/train_instance_fewshot.py`
- Decoder: `adatile/decoder/adaptive_sparse_decoder.py` → `ProtoOnlyDecoder`
- 数据集: `adatile/datasets/isaid_instance.py` → `ISAIDInstanceDataset`
- 日志: `runs/ifewshot_test_F0/train.jsonl`
- 最佳模型: `runs/ifewshot_test_F0/best_model.pt`
