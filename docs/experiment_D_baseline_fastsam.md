# D-00: FastSAM Zero-Shot 实例分割 Baseline

**日期**: 2026-07-03
**脚本**: `tools/instance/eval_baseline_instance.py`
**数据**: iSAID val (458 张全尺寸航拍图像)
**设备**: 云服务器 (RTX 5090)

## 实验目的 | Purpose

建立 FastSAM 在 iSAID 上的 zero-shot 实例分割 COCO AP 参考基线。
这是 few-shot fine-tuning 的**上界参考**：如果 FastSAM 全模型（SA-1B 预训练）能达到 AP=X，
few-shot adaptation 的目标就是尽量接近 X，并在此过程中引入稀疏计算（FDR）的加速收益。

## 推理配置 | Inference Config

| 参数 | 值 |
|------|-----|
| 模型 | FastSAM-x (YOLOv8-S backbone, SA-1B 预训练) |
| 置信度阈值 | 0.25 |
| NMS IoU 阈值 | 0.7 |
| 推理尺寸 | 1024px |
| 评估模式 | Class-Agnostic (GT 和 pred 全部映射为 cls=1) |

## 结果 | Results

### COCO 指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **AP** (IoU=0.50:0.95) | **0.0147** | 极低 — FastSAM 在航拍场景几乎无法产生精确掩码 |
| **AP50** | **0.0233** | 即使宽松的 IoU=0.5 也很低 |
| **AP75** | **0.0126** | 严格 IoU 更差 |
| AP_small | 0.010 | 小目标 (<32²) 完全失败 |
| AP_medium | 0.052 | 中等目标 (32²-96²) 勉强有一点 |
| AP_large | 0.127 | 大目标 (≥96²) 相对最好但仍很差 |
| AR_max1 | 0.001 | 每图 1 个预测→几乎无召回 |
| AR_max10 | 0.005 | 每图 10 个预测→仍极低 |
| AR_max100 | 0.026 | 每图 100 个预测→整体召回仅 2.6% |
| AR_small @100 | 0.009 | 小目标召回率 <1% |
| AR_medium @100 | 0.175 | 中等目标召回 17.5% |
| AR_large @100 | 0.372 | 大目标召回 37.2% ← 唯一有意义的信号 |

### 关键发现

#### 1. 目标尺度分层效应明显

```
AR_large  (≥96²)  : 37.2%  ← FastSAM 能"看到"大目标
AR_medium (32²-96²): 17.5%  ← 中等目标显著下降
AR_small  (<32²)   :  0.9%  ← 小目标几乎完全丢失
```

这是一个约 **41× 的尺度差距**（large vs small）。
iSAID 航拍图像中绝大多数目标是小目标（small_vehicle, ship 等），
这解释了为什么整体 AP 如此之低。

#### 2. "看到但掩不精" — 召回 vs 精度的鸿沟

```
AR_large@100 = 0.372    → 37.2% 的大目标至少有一个预测 IoU>0.5
AP_large     = 0.127    → 但只有 12.7% 的预测 mask 质量达标
AP (all IoU) = 0.015    → 严格 IoU 0.50:0.95 平均后几乎为 0
```

这意味着 FastSAM 在 iSAID 上**能定位物体，但无法产生精确的实例掩码**。
IoU 在 0.5-0.95 的高区间几乎全部失败。

#### 3. 与 C-01 历史结果的关系

C-01 曾报告 `mR@50≈41.5%`。这是不同的指标和设置：
- C-01 的 mR@50 是 per-class mean Recall at IoU=0.5，且是在 256² tiles 上评估
- D-00 的 COCO AP 是 IoU=0.50:0.95 的平均，且在全尺寸图像 (1024px resize) 上评估
- 全尺寸 resize→小目标缩小→FastSAM 更难检测

**C-01 的 41.5% recall 到 D-00 的 1.5% AP 的差距，说明了两个关键问题：**
1. AP 比 AR 严格得多（IoU 平均 vs 单阈值）
2. 全图 resize 对小目标是毁灭性的（这也是为什么我们需要 tile-based approach）

#### 4. 为什么这么低？

| 因素 | 影响 |
|------|------|
| **领域偏移**: SA-1B (自然图像) → iSAID (航拍) | FastSAM 从未见过航拍视角、密集排列的物体 |
| **尺度失配**: SA-1B 物体大，iSAID 物体小 | 1024px resize 后小目标 < 10px |
| **Proto 基函数**: 32 个 proto mask 在自然场景学到的是"物体形状基" | 航拍物体是矩形/小圆点，与自然物体形状完全不同 |
| **Class-agnostic 预训练**: SA-1B 没有类别概念 | 无法利用类别先验过滤 |
| **密集场景**: iSAID 一张图可能有 1000+ 实例 | FastSAM 的 NMS 和检测头不是为这种密度设计的 |

## 战略意义 | Strategic Implications

### 这个 0.015 对论文意味着什么

1. **Zero-shot 的惨败 = 我们的叙事起点**
   - SA-1B 的 "segment anything" 在航拍场景并不成立
   - 这正好证明了 **domain-specific fine-tuning 的必要性** — 我们的核心论点

2. **Few-shot 的上升空间巨大**
   - 从 0.015 → 0.10 就有 6.7× 提升 → 展示 few-shot 的有效性
   - 从 0.015 → 0.20 就有 13× 提升 → 可以宣称 "few samples unlock aerial segmentation"

3. **Tile 策略的必要性被验证**
   - AR_large=37.2% vs AR_small=0.9% → 小目标完全不可见
   - 全图 resize 对小目标是**致命**的 → Tile 是解决此问题的唯一途径
   - 这与 C-03 的结论一致: **Tile is not an optimization — it's a necessity**

4. **FDR 的加速收益在上界很低的情况下更有意义**
   - 如果 zero-shot AP 本身只有 0.015，frozen backbone 的上界也有限
   - FDR 可以减少计算量而不损失太多精度（因为精度本来就不高）
   - 核心卖点变成：**用更少的计算达到相同（甚至更好）的精度**

### 调整后的叙事逻辑

```
不是: "Few-shot 让我们接近 SA-1B 的 segment-anything 能力"
而是: "SA-1B 预训练在航拍场景失效 → Few-shot domain adaptation 修复了这个问题
       → 同时我们的稀疏计算(FDR)减少了 60% 的计算量"
```

## 后续实验 | Next Steps

1. **D-01**: K-Shot Scaling — 验证 few-shot 能否从 0.015 提升到有意义的水平
2. **D-02**: Decoder 架构对比 — ProtoOnly vs AdaptiveSparse
3. **D-03**: FDR 消融 — 验证稀疏计算在 low-AP 场景下的收益
4. **诊断**: 检查 tile-based 推理的 zero-shot AP（256²/512² tiles），验证"尺度是主要 bottleneck"

## 备注 | Notes

- 这是 class-agnostic 评估（所有 GT 合并为 cls=1）。per-class AP 会因为类别匹配问题更差。
- FastSAM 的原生 COCO AP（在 MS-COCO 自然图像上）约 40%+ — iSAID 的 1.5% 反映了 ~27× 的领域差距。
- 这个结果**不是失败** — 它是论文 Introduction 中 "domain gap motivates our method" 的核心证据。
