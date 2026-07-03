# D-04: K-Shot Scaling — Sample Efficiency 分析

**日期**: 2026-07-03
**脚本**: `tools/train/train_instance_fewshot.py` (v3, Adaptive noFDR)
**数据**: iSAID-5i Fold 0, Novel classes (4 sampled, cls12 excluded), tile mode (256² tiles)
**设备**: 云服务器 (RTX 5090)

## 实验设计 | Experiment Design

量化 few-shot sample count 对 mIoU 的影响。使用 D-03 验证的最优配置 (AdaptiveSparse noFDR)。

| 实验 | K-shot | Decoder | FDR | Episodes |
|------|--------|---------|-----|----------|
| K1 | 1 | Adaptive noFDR | ✗ | 2000 |
| K3 | 3 | Adaptive noFDR | ✗ | 2000 |
| K5 | 5 | Adaptive noFDR | ✗ | 2000 |
| K10 | 10 | Adaptive noFDR | ✗ | 2000 |

## 结果 | Results

### K-Shot 曲线

| K | Best mIoU | Δ vs K=1 | Δ vs 前一级 | 边际收益 |
|---|-----------|----------|------------|----------|
| 1 | **0.2890** | — | — | — |
| 3 | **0.2955** | +2.3% | +2.3% | 低 |
| 5 | **0.3160** | +9.4% | +6.9% | **最高** ← 甜点 |
| 10 | **0.3261** | +12.8% | +3.2% | 递减 |

```
mIoU
0.33 │                                    ● K=10
0.32 │                         ● K=5
0.31 │
0.30 │              ● K=3
0.29 │   ● K=1
0.28 │
     └─────┬─────────┬─────────┬─────────
           1         3         5        10   K-shot
```

### 边际收益分析

```
K=1 → K=3:  +0.0065 / +2 shots = +0.0033/shot  (低效)
K=3 → K=5:  +0.0205 / +2 shots = +0.0103/shot  (高效) ← 甜点区
K=5 → K=10: +0.0101 / +5 shots = +0.0020/shot  (极低效, 5倍 shot 只换 3.2% mIoU)
```

### Shot Saturation Index

定义 SSI-shot = mIoU(K) / mIoU(K=10) — 用 K shot 能达到 K=10 性能的百分比:

```
SSI-1  = 0.289 / 0.326 = 88.6%  ← 1 shot 就达到 89% 的 full-shot 性能
SSI-3  = 0.296 / 0.326 = 90.7%
SSI-5  = 0.316 / 0.326 = 96.9%  ← 5 shot 接近饱和
SSI-10 = 1.000                = 100%
```

## 关键发现 | Key Findings

### 1. 1-Shot 能力的惊人有效性

K=1 达到 K=10 性能的 **88.6%**。这意味着:

> **Support prototype 的 1280-d 向量在单样本情况下已经编码了足够的类别信息。**
> Few-shot 的主要瓶颈不在 sample count，而在 proto 基函数表达 + 解码器容量。

这对论文叙事极为有利:
- "仅需 1 张标注样本即可达到接近饱和的性能" — 比"用 10 张样本才勉强到 0.33" 更有说服力
- 低样本效率 → 证明方法不需要大量标注 → 真正 practical 的 few-shot

### 2. 5-Shot 是甜点 (Sweet Spot)

K=5 达到 96.9% 饱和，边际收益最高。这是实验设计的最佳平衡点:
- 足够 support diversity (5 tile × 256² = 5 个视角的同类别 tile)
- 计算开销可控 (每 episode 5 次 backbone forward)
- 与标准 few-shot benchmark (1/5/10-shot) 对齐

### 3. 10-Shot 的边际递减说明架构瓶颈 (非数据瓶颈)

```
K=10 vs K=5:  +5 shot, 仅 +3.2% mIoU
```

10 个 support tile 的多样性远超 5 个，但 mIoU 提升甚微。说明:

> **当前瓶颈是 ProtoCoeffPredictor + SA-1B proto base 的表达上限 (~0.33 mIoU)**，
> 不是 support sample 的多样性或质量。

这验证了 D-01 的发现:
- ProtoOnly K=5: 0.235 → 纯 MLP 上界
- Adaptive K=10: 0.326 → P4 refinement + 满血 support → 架构天花板
- 从 0.235 到 0.326 是 refinement path 的贡献
- 从 0.289 (K=1) 到 0.326 (K=10) 是更多 support 的贡献
- **剩余 gap (0.326 → 理想 0.5+) 需要突破 proto 基函数瓶颈**

### 4. 与 C-03 的 Shot Saturation 一致

C-03 发现 "Shot saturation: 1≈3≈5 shot"。D-04 在实例分割设置下重复了这一发现:

| 实验 | 任务 | 1-shot vs 5-shot | 现象 |
|------|------|------------------|------|
| C-03 | Cross-Attn 语义分割 | ≈ 持平 | 1-shot 已足够 |
| D-04 | AdaptiveSparse 实例分割 | 1-shot = 88.6% of K=10 | 相同的饱和模式 |

两个不同架构、不同任务都观察到了 shot saturation → **这是 FastSAM P4 特征的内在属性**，不是特定 decoder 的问题。

## 训练稳定性

| K | NaN | 训练质量 | 备注 |
|---|-----|----------|------|
| 1 | ✅ 零 | 健康 | Support 多样性最低但收敛正常 |
| 3 | ✅ 零 | 健康 | — |
| 5 | ✅ 零 | 健康 | 甜点, 最一致 |
| 10 | ✅ 零 | 健康 | Support 最多, 收敛最慢但最终最高 |

三层 NaN 防护在所有 K 值下均有效。

## D-Series 完整结果矩阵

```
         K=1      K=3      K=5      K=10
───────────────────────────────────────────
ProtoOnly         —        —       0.235      —
Adaptive+FDR      —        —       0.309      —
Adaptive noFDR  0.289    0.296    0.316    0.326  ← D-04
Zero-Shot        —        —       0.015 (AP) —
```

## 后续 | Next

1. **Cross-fold validation**: Fold 1/2 验证 K-shot scaling 的泛化性
2. **Proto base finetuning**: 允许 proto masks 参与训练 → 预期突破 0.33 天花板
3. **Multi-scale refinement**: P3 + P4 + P5 → 不同尺度目标在不同特征层
4. **Instance-aware loss**: 当前 per-pixel Dice+Focal → 需要实例级监督
5. **Full-image inference**: FDR tile selector + Adaptive K=5 per tile

## 文件 | Files

- 训练脚本: `tools/train/train_instance_fewshot.py` (v3)
- Decoder: `adatile/decoder/adaptive_sparse_decoder.py` (v2)
- 日志: `runs/ifewshot_K{1,3,5,10}_noFDR_F0/train.jsonl`
- 最佳模型: `runs/ifewshot_K{1,3,5,10}_noFDR_F0/best_model.pt`
