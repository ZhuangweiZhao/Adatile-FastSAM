# D-03: FDR 消融 — AdaptiveSparse w/ vs w/o FDR Gate

**日期**: 2026-07-03
**脚本**: `tools/train/train_instance_fewshot.py` (v3: 三层 NaN 防护)
**数据**: iSAID-5i Fold 0, Novel classes (4 classes, cls12 excluded), tile mode (256² tiles)
**设备**: 云服务器 (RTX 5090)

## 实验设计 | Experiment Design

量化 FDR (Foreground Density Router) gate 对 AdaptiveSparseDecoder 的独立贡献。

| 实验 | FDR gate | Params | 变量 |
|------|----------|--------|------|
| D-03 (noFDR) | ✗ | 1,142K | 纯 P4 refinement |
| D-02 (+FDR) | ✓ | 1,161K | P4 refinement + FDR density gate |

其他条件完全一致：K=5, 2000 episodes, 4 Novel classes (cls12 excluded), same seed, same LR.

## 结果 | Results

### 核心指标

| 指标 | Adaptive noFDR | Adaptive +FDR | Δ |
|------|---------------|---------------|----|
| **Best mIoU** | **0.3160** (Ep 1600) | 0.3085 (Ep 1700) | **-2.4%** |
| **AP50** | 0.0000 | 0.0000 | — |
| Params | 1,142,561 | 1,161,378 | +18,817 |
| NaN | ✅ 零 | ✅ 零 | — |

### 训练曲线

```
Episode    noFDR    +FDR      Δ
   100     0.156    0.149    -4.5%  (FDR 拖慢)
   500     0.170    0.143   -15.9%  (FDR 严重拖慢)
   900     0.272    0.227   -16.5%  (noFDR 率先突破 0.25)
  1100     0.274    0.238   -13.1%
  1500     0.315    0.303    -3.8%  (差距缩小)
  1600     0.316    0.303    -4.1%  ← noFDR BEST
  1700     0.308    0.309    +0.3%  ← +FDR BEST (唯一反超)
  2000     0.283    0.261    -7.8%
```

### Loss 分解对比 @ Ep 900

| Loss | noFDR | +FDR | 现象 |
|------|-------|------|------|
| Focal | 0.020 | 0.007 | noFDR 保持适度的 FG 误判惩罚 |
| Dice | 0.910 | 0.983 | +FDR 几乎无 FG 预测 |
| pred_mean | 0.243 | 0.206 | noFDR 更积极地预测 FG |

## 关键发现 | Key Findings

### 1. 未训练的 FDR = 随机噪声 = 有害

```
Adaptive noFDR:  0.316  ← WINNER
Adaptive +FDR:   0.309  ← -2.2%  (FDR 在干扰而非帮助)
ProtoOnly:       0.235  ← baseline
```

**根因链**:
```
FDR (untrained, frozen, random weights)
  → DensityHead 输出 ≈ random [0,1] map
    → F.interpolate → fdr_gate input: [feat_refined | random_noise]
      → fdr_gate produces semi-random gate [0,1]
        → feat_refined = feat_refined * random_gate
          → 随机遮罩 64-d 精炼特征 → 破坏空间一致性
```

这是**架构层面的设计缺陷**，不是训练超参数问题。

### 2. FDR 的最佳角色是 Tile Selector，不是 Pixel Gate

| FDR 角色 | 粒度 | 验证状态 | 当前瓶颈 |
|----------|------|----------|----------|
| **Tile Selector** (B-series) | 粗 | ✅ 已验证 (r=0.889, SSI=2.41×) | 需要大图推理 pipeline |
| **Pixel Gate** (D-series) | 细 | ❌ 有害 (-2.2%) | 需要预训练 FDR |
| **当前用法** | 细 | ❌ 有害 | 未训练的随机噪声 |

B-series 验证的 FDR 能力是**从大图中选出高密度 tile**（粗粒度），而不是**在 tile 内部做 pixel-level density modulation**（细粒度）。这两个角色需要完全不同的训练策略。

### 3. 256² Tile = 已被隐式选择的"高密度区域"

当前实验在 256² tile 上做 few-shot。这些 tile 本身就是从全图中切出的高密度区域（FG>5% 过滤）。在"已选择的高密度区域"内部再做 per-pixel density gating 是**冗余且有害**的。

正确的 FDR 用法：
```
大图推理 (4000×4000):
  Step 1: FDR → tile importance → Top-K% tiles (稀疏选择，FDR 的战场)
  Step 2: AdaptiveSparse noFDR → per-tile mask (P4 refinement，noFDR 的战场)
  Step 3: Merge tiles → full-image instance mask
```

### 4. P4 Refinement 才是 mIoU 的核心驱动力

Δ(ProtoOnly → Adaptive noFDR) = +0.081 (+34.5%)

这 0.081 的提升完全来自 P4 特征精炼路径：
```
P4 [1280, 16, 16] → feat_proj [256, 16, 16] → feat_refine [64, 16, 16] → mask_head → per-pixel logit
```

**没有** FDR gate 的干扰，这个路径的梯度流更干净、收敛更快。

## 全局对比矩阵

```
D-00: Zero-Shot            AP=0.015  (上界)
D-01: ProtoOnly             mIoU=0.235  (纯 MLP 下界)
D-02: Adaptive +FDR         mIoU=0.309  (+31.5%)
D-03: Adaptive noFDR        mIoU=0.316  (+34.5%)  ← 当前最优
```

## 设计修正 | Design Revision

### 修正前 (D-02)
```python
class AdaptiveSparseDecoder:
    def forward(p4, proto_masks, support_proto, fdr_map):
        ...
        feat_refined = feat_refined * fdr_gate(fdr_map)  # ← 有害: 未经训练的 gate
        ...
```

### 修正后 (D-03, 推荐)
```python
class AdaptiveSparseDecoder:
    use_fdr = False  # 在 tile-level few-shot 场景禁用 per-pixel FDR gate

    def forward(p4, proto_masks, support_proto, fdr_map=None):
        ...
        # FDR gate 仅在 fdr_map 有意义时启用（需要预训练 FDR）
        if self.use_fdr and fdr_map is not None:
            feat_refined = feat_refined * self.fdr_gate(fdr_map)
        ...
```

### FDR 正确使用场景
```python
# 大图推理 | Full-image inference
fdr_map = fdr_router(feats["p8"])          # [1, 1, H/32, W/32]
tile_mask = fdr.select_tiles(fdr_map, k=0.4)  # Top-40% tiles
selected_regions = apply_tile_mask(image, tile_mask)

for tile in selected_regions:
    mask = adaptive_decoder NoFDR(tile_p4, proto, prototype)  # noFDR per tile
```

## 后续 | Next

1. **D-04 K-shot scaling**: K=1/3/5/10 @ noFDR (当前最优配置)
2. **FDR 预训练**: 在 iSAID train tiles 上用 fg_ratio 监督训练 FDR
3. **大图推理 pipeline**: FDR tile selector + Adaptive noFDR per tile
4. **B-series 对齐**: 验证训练后的 FDR 在 full-image 上的 tile selection 效果

## 文件 | Files

- 训练脚本: `tools/train/train_instance_fewshot.py` (v3)
- Decoder: `adatile/decoder/adaptive_sparse_decoder.py` (v2: InstanceNorm2d)
- FDR: `adatile/sparse/spatial_router.py` → `ForegroundDensityRouter`
- 日志: `runs/ifewshot_adaptive_noFDR_v2_F0/train.jsonl`
- 最佳模型: `runs/ifewshot_adaptive_noFDR_v2_F0/best_model.pt`
