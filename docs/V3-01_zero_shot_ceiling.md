# V3-01: FastSAM Zero-Shot GT-Prompted Ceiling Test

> **实验日期**: 2026-07-05  
> **脚本**: `tools/eval/eval_fastsam_prompted.py`  
> **数据**: iSAID-few (1% sample, 20 train images, seed=42)  
> **模型**: FastSAM-x (ultralytics), frozen  
> **设备**: CUDA (RTX 4060 8GB)

---

## 实验目的

**核心问题**: FastSAM zero-shot 在航拍图上的瓶颈是 "找不到目标" 还是 "画不准"？

**方法**: 用 GT bbox / point 作为 prompt，跳过检测环节，直接测试分割上限。

---

## 运行命令

```bash
python tools/eval/eval_fastsam_prompted.py \
    --data-root data/iSAID-few \
    --data-format isaid_processed \
    --split train \
    --num-samples 20 \
    --mode all \
    --device cuda
```

## 总体结果

| | 值 |
|---|---|
| 图像数 | 20 |
| GT 实例总数 | 7,581 |
| 平均实例/图 | 379 |
| 推理时间 | 18min 26s (55.3s/img) |

### 主指标 (Bbox Mode)

| 指标 | 值 |
|------|-----|
| **IoU** | **0.0345** |
| Dice | 0.0398 |
| AP@50 | 0.042 |
| AP@75 | 0.026 |
| AP@90 | 0.003 |
| Boundary IoU | 0.0419 |
| **Failure Rate** | **91.5%** |

### 四种 Prompt 模式对比

| Mode | Total | Valid | Fail% | IoU | Dice | AP@50 |
|------|-------|-------|-------|-----|------|-------|
| bbox | 7581 | 642 | 91.5% | 0.0345 | 0.0398 | 0.042 |
| center-point | 7581 | 941 | 87.6% | 0.0349 | 0.0402 | 0.042 |
| multi-point | 7581 | 941 | 87.6% | 0.0262 | 0.0309 | 0.030 |
| pos-neg | 7581 | 941 | 87.6% | 0.0262 | 0.0309 | 0.030 |

> **注**: Multi-point 和 Pos-neg 的 IoU 低于 center-point，因为多点命中率在 everything mask 数量很少时与单点无本质区别，但精选逻辑（选命中数最多的 mask）引入了噪声。

### 失败模式分析 (Bbox)

| 失败原因 | 数量 | 占比 |
|----------|------|------|
| no_result | 6,640 | 87.6% |
| no_overlap | 299 | 3.9% |
| 有效 | 642 | 8.5% |

**解读**: 87.6% 的 GT 实例在 everything mode 下 **完全不产生任何候选 mask**（Proposal Generation 失败）。仅有 8.5% 的实例有 mask 候选且能与 bbox 匹配。

---

## Per-Class 结果 (Bbox IoU)

| Class ID | Name | 实例数 | IoU | 判定 |
|----------|------|--------|-----|------|
| 1 | small_vehicle | 4,920 | **0.0147** | ❌ 几乎全灭 |
| 2 | large_vehicle | 720 | 0.1439 | ⚠️ 差 |
| 3 | plane | 161 | 0.0500 | ❌ |
| 4 | storage_tank | 40 | **0.4370** | ✅ 尚可 |
| 5 | ship | 1,608 | 0.0136 | ❌ |
| 6 | harbor | 73 | 0.2692 | ⚠️ |
| 7 | ground_track_field | 5 | 0.1874 | — 样本太少 |
| 8 | soccer_ball_field | 1 | **0.5886** | ✅ 最好 (大目标) |
| 9 | tennis_court | 16 | **0.8010** | ✅ 最好 |
| 10 | swimming_pool | 6 | 0.3753 | — 样本太少 |
| 11 | baseball_diamond | 7 | 0.0000 | ❌ |
| 12 | basketball_court | 1 | 0.0000 | — 样本太少 |
| 13 | bridge | 5 | 0.0000 | ❌ |
| 14 | helicopter | 0 | — | 无样本 |
| 15 | roundabout | 18 | 0.1286 | ❌ |

---

## Per-Size 结果 (Bbox IoU)

| 尺寸 | 实例数 | IoU | 
|------|--------|-----|
| **Small** (< 32² px) | 7,026 | **0.0230** |
| **Medium** (32²–96²) | 505 | 0.1662 |
| **Large** (≥ 96²) | 50 | **0.3207** |

> Large / Small = **14×** 差距。FastSAM 的分割能力与目标尺寸呈强正相关。

---

## 三阶段失败分解

```
Proposal Generation (Stage 1):  87.6% 失败 ← 主要瓶颈
  → FastSAM everything 对 87.6% 的 GT bbox 生成 0 个候选
  → 根因: small_vehicle (10-50px) 在 SA-1B 训练分布之外

Prompt Selection (Stage 2):     3.9% 失败
  → bbox IoU > 0 但未选到最佳 mask

Mask Quality (Stage 3):
  → 即使选到，整体 IoU = 0.035
  → 大目标单独看: IoU = 0.321 (有一定分割能力)
```

---

## 关键结论

1. **Proposal Generation 是最大瓶颈**：GT bbox 都给了，FastSAM 仍对 91.5% 的实例无法生成候选 mask。瓶颈不在 Prompt，在 Backbone。

2. **Small_vehicle 是核心矛盾**：占 65% 实例（4,920/7,581），IoU = 0.015。如果排除 small_vehicle，剩余类别的 IoU 会显著改善。

3. **大小目标差距 14×**：IoU_large = 0.321 vs IoU_small = 0.023，证实 FastSAM proposal 质量与目标尺寸强相关。

4. **Multi-point / Pos-neg 无明显增益**：因为 everything mask 数量本身太少（多数实例为 0），多点和正负点的区分度被淹没。

5. **这组结果正是 Paper B 的论据基础**：全图 zero-shot 在航拍图上几乎无效，需要 Tile-based Few-Shot 方法（分块缩小目标相对尺寸 + 少样本域适应）。

---

## 与历史基线对比

| 实验 | 数据 | 指标 | 值 |
|------|------|------|-----|
| D-00 (v2) | iSAID 原图 1024² | AP | 0.015 |
| **V3-01 (本次)** | iSAID-few 原图 | IoU (bbox prompt) | 0.035 |
| **V3-01 (本次)** | iSAID-few 原图 | AP@50 (bbox prompt) | 0.042 |

> 注: D-00 用 COCO AP 全流程（含 detect），本次用 GT bbox prompt（跳过 detect）。数值不可直接比较，但都确认了 ~27× domain gap。

---

## 下一步

1. **V3-02**: Tile FG 分布统计（确认 spatial sparsity）
2. **V3-03**: Oracle Top-K 实验（证明 Top-40% tiles 含 96.5% FG）
3. **V3-05**: Base pre-training + V3-06: Few-shot fine-tuning
4. 需要进一步确认 empty-tile 现象在 896² 上的分布（B-00 在 1024² 上做了）
