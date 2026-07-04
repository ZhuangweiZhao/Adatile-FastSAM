# V3-01 Zero-Shot Baseline 完整分析 | Full Analysis

> **日期**: 2026-07-04
> **实验 ID**: V3-01
> **运行目录**: `runs/v3_01_zero_shot_val_0704_1202/`
> **状态**: ✅ 完成

---

## 1. 实验配置 | Experiment Config

| 参数 | 值 |
|------|-----|
| 模型 | FastSAM-x (SA-1B 预训练) |
| 数据 | v3 Instance Few-Shot Split (896² tiles) |
| Split | val (8,331 tiles, 225,619 GT instances) |
| 置信度阈值 | 0.25 |
| NMS IoU | 0.7 |
| 设备 | CUDA (单卡) |
| 评估协议 | Class-Agnostic COCO AP (segm) |
| 运行时间 | ~87 分钟 (推理 8,331 tiles + COCOeval) |

---

## 2. 核心结果 | Core Results

```
============================================================
  V3-01 Zero-Shot Baseline — Results
  Split: val, Tiles: 8331
  Predictions: 445,027
============================================================
  AP           (IoU=0.50:0.95): 0.0182    ← 主指标
  AP50         (IoU=0.50):      0.0426
  AP75         (IoU=0.75):      0.0126
  AP_small     (<32²):          0.0199
  AP_medium    (32²-96²):       0.0674    ← 最佳
  AP_large     (>=96²):         0.0204
  AR_max1:                      0.0033
  AR_max10:                     0.0240
  AR_max100:                    0.0953    ← 召回率不到 10%
============================================================
```

---

## 3. 与 D-00 (v2 基线) 对比 | Comparison with D-00

| 指标 | D-00 (v2, 全图 resize→1024²) | V3-01 (v3, 896² tile) | Δ |
|------|------------------------------|------------------------|-----|
| **AP** | 0.0147 | **0.0182** | **+23.8%** |
| **AP50** | 0.0233 | **0.0426** | **+82.8%** |
| **AP75** | 0.0126 | 0.0126 | 0% |
| AP_small | ~0.009 (est.) | **0.0199** | **+2.2×** |
| AP_medium | ~0.017 (est.) | **0.0674** | **+3.9×** |
| AR_large | 0.372 | — | 不可比¹ |

> ¹ V3-01 的 AR per area 被 COCOeval API 限制未正确输出 (均为 0.0)。

### 关键结论 | Key Takeaways

1. **Tile 策略有显著提升但不是根本解决方案**: AP 从 0.0147 → 0.0182 (+23.8%)，验证了 896² tile 保留了更多空间细节。但量级仍在 ~2% AP，领域差距是根本性的。

2. **AP50 翻倍 (0.023→0.043)**: 宽松匹配 (IoU≥0.5) 下提升 82.8%，说明 tile 推理让 FastSAM 能找到更多目标，但无法生成精确 mask。

3. **Medium 目标受益最大 (AP=0.0674)**: 32²-96² 像素大小的目标最容易分割，与 SA-1B 训练分布一致。

4. **领域差距确认**: 3 个数量级的差距 (AP 1.8% vs COCO 典型 ~40%)。SA-1B 自然图像 → 航拍遥感的 zero-shot 迁移几乎完全失败。

---

## 4. 预测分析 | Prediction Analysis

### 4.1 预测量

| 统计量 | 值 |
|--------|-----|
| 总预测 masks | 445,027 |
| 有预测的 tiles | 8,235 / 8,331 (98.8%) |
| 无预测的 tiles | 96 (1.2%) |
| 每 tile 平均 masks | 54.0 |
| GT instances | 225,619 |
| Pred/GT 比 | 1.97× |

### 4.2 面积分布

| 类别 | 数量 | 占比 |
|------|------|------|
| Small (<32² px) | 192,952 | 43.4% |
| Medium (32²–96² px) | 205,364 | 46.1% |
| Large (≥96² px) | 46,711 | 10.5% |

面积均值 13,132 px²，中位数应该在 medium 区间。FastSAM 倾向于生成中等大小的 mask。

### 4.3 Score 限制 | Score Limitation ⚠

**所有 445,027 个预测的 score 均为 1.0。** FastSAM 的 `Masks` 对象不提供 `conf` 属性（与标准 Ultralytics YOLO 不同）。这对 AP 的影响：

- pycocotools 按 score 降序排列 predictions，同分数时顺序不确定
- 精确率-召回率曲线在 score=1.0 处只有一个数据点
- 无法通过降低阈值来 trade-off precision vs recall
- **实际 AP 可能被低估** — 如果有真实置信度，高精度 mask 可排前面提升 PR 曲线

**后续改进方案**:
1. 用 mask area 作为伪 score (`score = area / max_area`)
2. 用 mask 的 IoU 稳定性做 self-consistency score (多次推理)
3. 用 detection head 的 objectness 作为 proxy（需修改 FastSAM 推理代码）

---

## 5. 与 v3 论文叙事的关系 | Narrative Fit

### 5.1 第一段: 问题陈述

```
SA-1B 预训练的 FastSAM 在航拍图像上 zero-shot AP 仅 1.8%。
即使使用高分辨率 tile 推理 (896²) 保留空间细节，
性能仍比典型 COCO 场景 (~40% AP) 低 22 倍。
这揭示了 SA-1B ↔ 遥感之间的根本性领域差距。
```

### 5.2 作为 Baseline

| 定位 | 说明 |
|------|------|
| **下界** | 不做任何微调，FastSAM 能到达的极限 |
| **对比目标** | Few-shot fine-tuning 应在此基础上有数量级提升 (目标 10-20×) |
| **效率参考** | 全 tile 推理 = 8,331 forward passes → SPM 稀疏路由后的加速比 |

### 5.3 后续实验链条

```
V3-01 (本实验): AP = 0.018  ← Zero-shot 下界
    ↓
V3-02~04: 空间稀疏性验证 (B-series 复用)
    ↓
V3-05~07: Few-shot fine-tuning → 目标 AP >> 0.018 (10×+)
    ↓
V3-09~11: SPM 效率 → FLOPs 减少 + AP 保持
```

---

## 6. 运行命令 | Run Commands

### 本地
```bash
# 快速测试 (100 tiles)
python tools/eval/eval_zero_shot.py --max-tiles 100

# 全量 (≈1.5h)
python tools/eval/eval_zero_shot.py --device cuda
```

### 云服务器
```bash
nohup python tools/eval/eval_zero_shot.py \
    --data-root /root/autodl-tmp/iSAID_instance_fewshot \
    --split val --device cuda \
    > /root/autodl-tmp/v3_01_zero_shot.log 2>&1 &
```

---

## 7. 文件清单 | Artifacts

| 文件 | 大小 | 说明 |
|------|------|------|
| `predictions.json` | 153 MB | 445,027 COCO-format predictions |
| `gt_agnostic.json` | 104 MB | Class-agnostic GT |
| `results.json` | 753 B | 汇总结果 |
| `eval.jsonl` | 2.9 KB | 结构化日志 |

---

*Next: V3-02 — Spatial Sparsity validation (复用 B-00 tile size sensitivity 实验)*
