# AdaTile-FastSAM v2 — 项目全面状态文档

> **更新日期**: 2026-06-28 | **分支**: `paper-b` | **阶段**: Paper B — Few-Shot Token Routing 验证

---

## 目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [模块详解](#3-模块详解)
4. [数据管线](#4-数据管线)
5. [训练协议与 Decoder](#5-训练协议与-decoder)
6. [实验体系](#6-实验体系)
7. [评估指标](#7-评估指标)
8. [当前状态与已知结果](#8-当前状态与已知结果)
9. [已知问题与教训](#9-已知问题与教训)
10. [开发路线图](#10-开发路线图)

---

## 1. 项目概述

**AdaTile-FastSAM**: 面向高分辨率遥感图像少样本实例分割的自适应稀疏 FastSAM 框架。

### 双论文策略 (2026-06-21)

| | Paper A (Proto Sparsity) | Paper B (Dual/Spatial Sparsity) |
|---|---|---|
| **分支** | `main` (已归档) | `paper-b` (活跃) |
| **核心创新** | Learned Sparse Proto Routing | Token-Level Dynamic Routing + Foreground Density Router |
| **证据链** | E007→E011-U 完成 | B-00→B-03 关闭, B-04→B-09 + C 系列推进中 |
| **目标** | ICIP/CCIG 短篇 | CVPR/ICCV/ECCV 顶刊 |

### 核心创新

1. **Ada-SPM** (Paper A): 密度监督的稀疏感知模块 — 学习 importance maps → Top-K tile 选择
2. **Foreground Density Router / FDR** (Paper B): 75K params, Pareto 最优空间路由器 — 学习 objectness/density, 不学边缘或类别语义
3. **Decoupled Sparse Training**: Decoder 始终接收全量特征; SPM/Router 通过 GT 驱动 loss 并行训练

---

## 2. 系统架构

### 2.1 推理数据流 (FSS Pipeline)

```
Support Images [K, 3, H, W]              Query Image [1, 3, H, W]
        │                                         │
        ▼                                         ▼
  FastSAM Backbone (frozen)              FastSAM Backbone (frozen)
  ├─ P3 [K, 960, H/8, W/8]              ├─ P3 [1, 960, H/8, W/8]
  ├─ P4 [K, 1280, H/16, W/16]           ├─ P4 [1, 1280, H/16, W/16]
  └─ P8 [K, 1280, H/32, W/32]           └─ P8 [1, 1280, H/32, W/32]
        │                                         │
        ▼                                         │
  FG-Masked Average Pooling                       │
  ────────────────────────                        │
  Prototype [C] or [K_proto, C]                   │
        │                                         │
        ▼                                         ▼
  ┌─────────────────────────────────────────────────┐
  │              Few-Shot Decoder                    │
  │  ├─ FiLMFewShotDecoder (P4 only, ~1.1M)         │
  │  ├─ P3P4FiLMFusionDecoder (P3+P4, ~1.8M) ★     │
  │  ├─ ProtoRefineDecoder (~716K)                  │
  │  └─ CATFewShotDecoder (Cross-Attention)         │
  └─────────────────────────────────────────────────┘
                         │
                         ▼
                   Binary Mask [1, 1, H, W]
```

### 2.2 SPM/Token Routing (待验证)

```
Query P4 [1, 1280, H/16, W/16]
        │
        ▼
  FDR (Foreground Density Router, ~75K)
  DensityHead: Conv1×1→DWConv3×3→Conv1×1→Sigmoid
        │
        ▼
  Importance Map [1, 1, H/16, W/16]
        │
        ▼
  Top-K Token Selection → Binary Mask
        │
        ▼
  Masked P4 × Decoder → Mask
```

---

## 3. 模块详解

### 3.1 Backbone: FastSAMBackbone (`adatile/backbone/fastsam_backbone.py`)

| 特性 | 值 |
|------|-----|
| **基础模型** | YOLOv8-x (FastSAM-x.pt) |
| **运行模式** | **强制 eval mode** — `model.train()` 会崩溃 YOLOv8 Detect head |
| **梯度控制** | `requires_grad` 控制, 不调用 `.train()` |
| **输出** | `{"p3": [B,960,H/8,W/8], "p4": [B,1280,H/16,W/16], "p8": [B,1280,H/32,W/32]}` |
| **特征提取** | Hook-based — 自动探测 stride≈8/16/32 的层 |
| **Detect Head** | `_forward_features()` 跳过 Detect/Segment 层, 省 ~30-40% 显存+时间 |

#### LoRA 支持

```python
backbone.apply_lora(rank=4)  # +17,920 trainable params
# P3: Conv(960→4→960), init: kaiming + zeros (identity start)
# P4: Conv(1280→4→1280), 同上
# 应用方式: feature = feature + lora(feature)
```

### 3.2 Decoders (`adatile/decoder/` + `tools/instance/eval_c04_full_fewshot.py`)

| Decoder | Params | 输入 | 用途 |
|---------|--------|------|------|
| `LinearProbe` | ~1.3K | P4 | 验证 backbone 特征质量 |
| `FusionProbe` | ~10K | P4+P8 | 双分支融合探测 |
| `LightDecoder` | ~716K | P4 | B-04 二值/多类密集分割 |
| `InstanceDecoder` | ~716K | P4 | U-Net 风格实例分割 |
| `FiLMFewShotDecoder` | ~1.1M | P4 + Proto | ★ FSS, Proto→FiLM (γ, β) 调制 |
| `P3P4FiLMFusionDecoder` | ~1.8M | P3+P4+Proto | ★ FSS, Proto→Gate(α) + FiLM + Utilization |
| `ProtoRefineDecoder` | ~716K | P4 + Proto | Proto→CosineSim→Refine CNN |
| `CATFewShotDecoder` | ~1.2M | P4 + Proto | Cross-Attention tile decoder (来自 C-03) |
| `ContrastiveProtoDecoder` | ~1.3M | P4 + Proto | 对比学习投影 + 原型匹配 (实验性) |

#### P3P4FiLMFusionDecoder 架构 (当前主力)

```
P3 [B, 960, H/8, W/8]                P4 [B, 1280, H/16, W/16]
      │                                       │
      ▼                                       ▼
  proj_p3 → [B, 256, H/8, W/8]     proj_p4 → [B, 256, H/8, W/8] (上采样)
      │                                       │
      └────────────┬──────────────────────────┘
                   ▼
          Proto → Gate MLP → α ∈ [0,1]^256
          fused = α·P3 + (1-α)·P4
                   │
                   ▼
          Utilization Module → util_map × fused
                   │
                   ▼
          Proto → FiLM MLP → γ, β → γ·fused + β
                   │
                   ▼
          Upsample ×3 → Conv1×1 → Mask
```

### 3.3 Sparse Router (`adatile/sparse/spatial_router.py`)

| 组件 | Params | 用途 |
|------|--------|------|
| `DensityHead` | ~75K (in=576, mid=128) | 前景密度预测: 1×1→DW3×3→1×1→Sigmoid |
| `EdgeHead` | ~in×64 | 边缘感知 (消融专用, Sobel 初始化) |
| `ForegroundDensityRouter (FDR)` | ~75K | **Paper B 主线**: 密度驱动, 类别无关 |
| `DualStreamRouter` | ~75K + EdgeHead | 消融 R3: Density+Edge 融合 |
| `TinyCNNRouter` | ~20K | 极轻量下界: RGB→4×Conv(stride=2)→1×1 |

**B-03 消融结论**:
- R0 (MV3 full): 1.48M, Spearman r=0.884 (上界)
- R2 (FDR only): 75K, Spearman r=0.846 (**主线, 20× 压缩, 仅 -4.3%**)
- R3 (+Edge): +0.009 Δr, 边缘几乎无贡献

**设计原则**:
1. 密度驱动: 监督信号 = fg_ratio, 非边缘/类别
2. 类别无关: 学习 "哪里有目标", 不学 "什么目标"
3. 极致轻量: 相对分割 decoder 可忽略不计

### 3.4 Loss Functions (`adatile/losses/seg_losses.py`)

| Loss | 配置 | 用途 |
|------|------|------|
| `FocalLoss` | γ=5.0, ignore_index=255 | 极端类别不平衡 (遥感) |
| `DiceLoss` | per-FG-class, smooth=1e-8 | 小目标分割 |
| `CombinedLoss` | α·Focal + (1-α)·Dice | B-04 默认 |
| `focal_dice_loss` (C-04) | BCE+pos_weight + Dice + FG presence | FSS 训练 (在 `eval_c04_full_fewshot.py`) |

### 3.5 Metrics (`adatile/metrics/`)

| 函数 | 用途 |
|------|------|
| `compute_miou(pred, target)` | 平均 IoU, 支持 label/one-hot, 处理 {0,255} |
| `compute_dice(pred, target)` | Dice 系数, 修复 V1 broadcast bug |
| `FPSMeter` | GPU/CPU FPS 计时器 (CUDA events + warmup) |
| `count_params(model)` | 参数统计: total/trainable/frozen/per-module |
| `binary_iou(pred, gt)` | 单张二值 IoU (C-04 FSS 评估用) |

### 3.6 Logging (`adatile/logging/`)

架构: `LogRecord → Logger → [ConsoleBackend, FileBackend(JSONL), WandbBackend]`

- **ConsoleBackend**: ANSI 彩色终端输出
- **FileBackend**: 非阻塞 JSONL (daemon thread + queue), `buffer_size=1, flush_interval=1.0` — 崩溃安全
- **LogContext**: `contextvars` 实现, 线程安全
- **MetricTracker**: 滑动窗口 mean/EMA/min/max/std

---

## 4. 数据管线

### 4.1 数据预处理流程

```
iSAID 原始 COCO JSON                    Cityscapes
      │                                     │
prep_isaid.py (修正标注 ID)          prep_cityscapes.py
      │                                     │
prep_isaid_tiles.py ─────────────────────────┘
  ├── Step 1: render_category_mask() → masks_full/
  ├── Step 2: 切 tile → images/ + masks/
  └── Step 3: 生成 metadata JSON
      │
      ▼
  data/iSAID_tiles/ (1024×1024 tiles, legacy format)
```

```
iSAID 原始大图 (data/iSAID_processed)
      │
prep_isaid5i_multisize.py
  ├── 全图 → dense mask → 多尺寸滑动窗口切 tile
  ├── 支持 256/384/512/640/768/896 六种尺寸
  └── 输出: data/iSAID5i_tiles/tile_{size}/
      │
      ▼
  PreCutTileAdapter → FewShotEpisodeDataset → 训练
```

### 4.2 支持的数据集

| 数据集 | 类数 | 类型 | SSI (Top40% FG) | FSS 适用 |
|--------|------|------|-----------------|----------|
| **iSAID (COCO)** | 15 | 实例分割 (遥感) | >70 ✅ | ✅ C 系列 |
| **iSAID Tiles 896** | 15 | 预切 tile | >70 ✅ | ✅ 当前主力 |
| **iSAID-5i Benchmark** | 15 | 官方 256×256 tiles | — | ⚠️ 256px FastSAM 不兼容 |
| **LoveDA** | 7 | 密集地物分类 | <50 ❌ | ✅ B-08 |
| **Vaihingen** | 6 | 密集地物分类 | <50 ❌ | ✅ B-08 |
| **NWPU-VHR-10** | 10 | bbox 弱标注 | >70 ✅ | ✅ B-09 |
| **Massachusetts Buildings** | 2 | 二值建筑分割 | — | ✅ E 系列 |

### 4.3 iSAID-5i 官方 Fold 划分

| Fold | Novel (5类) | Base (10类) |
|------|-------------|-------------|
| **0** | small_vehicle, harbor, swimming_pool, basketball_court, roundabout | ship, storage_tank, baseball_diamond, tennis_court, ground_track_field, bridge, large_vehicle, helicopter, soccer_ball_field, plane |
| **1** | plane, large_vehicle, bridge, ground_track_field, tennis_court | ship, storage_tank, baseball_diamond, basketball_court, small_vehicle, helicopter, swimming_pool, roundabout, soccer_ball_field, harbor |
| **2** | ship, storage_tank, helicopter, soccer_ball_field, baseball_diamond | tennis_court, basketball_court, ground_track_field, bridge, large_vehicle, small_vehicle, swimming_pool, roundabout, plane, harbor |

### 4.4 类别映射

**ISAID 标准 ID** (iSAID COCO 处理后的标准映射): 1=small_vehicle, 2=large_vehicle, 3=plane, 4=storage_tank, 5=ship, 6=harbor, 7=ground_track_field, 8=soccer_ball_field, 9=tennis_court, 10=swimming_pool, 11=road, 12=basketball_court, 13=bridge, 14=helicopter, 15=roundabout

**ISAID5I 官方 ID** (iSAID-5i benchmark): 1=ship, 2=storage_tank, 3=baseball_diamond, 4=tennis_court, 5=basketball_court, 6=ground_track_field, 7=bridge, 8=large_vehicle, 9=small_vehicle, 10=helicopter, 11=swimming_pool, 12=roundabout, 13=soccer_ball_field, 14=plane, 15=harbor

> **关键**: 两个系统的 ID 编号不同! `--dataset isaid5i` 使用 ISAID5I 官方 ID, `--dataset fastsam` 使用 ISAID 标准 ID。

---

## 5. 训练协议与 Decoder

### 5.1 协议对比

| | `--dataset fastsam` | `--dataset isaid5i` |
|---|---|---|
| **Split** | 自定义 AdaTile 3-fold | 官方 iSAID-5i 3-fold |
| **Tile 尺寸** | 896 (可配) | 256 (官方) / 896 (自定义) |
| **Meta-training** | 仅 Novel 类 | Base 类→meta-train, Novel 类→meta-test |
| **论文可比性** | 不可比 | ✅ 可直接对标 PANet/PFENet/HSNet |

### 5.2 训练入口

```bash
# 标准 FSS 训练
python tools/train/train_fewshot.py \
    --dataset isaid5i \
    --src-root data/iSAID_processed \
    --tile-root data/iSAID5i_tiles/tile_896 \
    --fold 0 --shot 1 --epochs 60 \
    --decoder film --feature-level p3p4 \
    --device cuda
```

**关键参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--shot` | 1 | K-shot (1/3/5) |
| `--epochs` | 60 | 训练轮数 |
| `--decoder` | film/p3p4film/protorefine/crossattn | Decoder 架构 |
| `--feature-level` | p4/p3p4 | 特征层级 |
| `--num-prototypes` | 1 | Multi-Prototype K 值 |
| `--lora-rank` | 0 | LoRA 秩 (0=禁用) |
| `--val-batch-size` | 2 | 验证批大小 |
| `--ema-decay` | 0.997 | EMA 衰减率 |
| `--early-stop-patience` | 15 | 早停耐心 |

### 5.3 训练流程

```
Episodic Training (Meta-training on Base classes)
  │
  ├── Per epoch: 200 episodes
  │   ├── 随机选 Base class
  │   ├── 采样 K support + 1 query (均从 train_ds)
  │   ├── Support → Backbone → FG-masked Pooling → Prototype
  │   ├── Query → Backbone → Decoder(Query, Prototype) → Mask
  │   └── focal_dice_loss → backward
  │
  ├── Per epoch: Validation on Novel classes
  │   ├── 5 Novel classes × 30 episodes
  │   ├── Batched backbone forward (特征缓存 CPU)
  │   └── Per-class binary IoU
  │
  └── Checkpoint: best val mIoU → decoder_{type}_{shot}shot_best.pt
```

---

## 6. 实验体系

### 6.1 Paper A — Proto Sparsity (已归档 `main`)

| 实验 | 内容 | 关键结论 |
|------|------|----------|
| E007 | Proto vs Embedding 公平对比 | Proto 机制成立: Δparams=0.55%, ΔDice=+0.016 |
| E007-B | Proto 语义证明 | Silhouette Score 4.75× 提升 |
| E008 | SPM 稀疏性验证 | A: sim magnitude ≠ importance; B: 50% Proto = 100% Acc; C: 稀疏可学习 |
| E009 | Learned vs Fixed Router | SPM Router 超越 fixed |w*sim| |
| E009-D | Proto 使用率分析 | N≥4 时 Dice 饱和 |
| E010 | iSAID 多类 Proto vs Embedding | Head-to-head on 15-class |
| E011 | SPM on iSAID | SPM 修复 Proto Collapse: BG protos 12→6 |
| E011-T | Tile 尺寸消融 | 256-2048, sweet spot 分析 |
| E011-U | Proto 容量扫描 | N=2-64, 非容量瓶颈 |

### 6.2 Paper B — Spatial Sparsity (活跃 `paper-b`)

| 实验 | 内容 | 关键结论 | 状态 |
|------|------|----------|------|
| **B-00** | Tile 尺寸敏感性 | 所有尺度都存在空间稀疏性; 1024px 最优 | ✅ |
| **B-01** | Oracle Top-K | Top40% tiles → 96.5% FG, IDG=2.41× | ✅ |
| **B-02** | 可学习性 | Spearman r=0.889 (MV3), 重要性可学习 | ✅ |
| **B-02.5** | 泛化性 | 类别无关 (holdout r=0.651), 跨数据集 | ✅ |
| **B-03** | Router 架构 | FDR 75K ≈ R0 1.48M (Δr=-0.038) | ✅ |
| **B-04** | 端到端集成 | Decoder verified (val_fg5≈0.47), FDR 训练 | ✅ |
| **B-05** | Oracle Importance | fg_ratio vs IoU vs Contribution 多指标 | ✅ |
| **B-06** | 贡献粒度 | 不同粒度体现不同层面的贡献不均衡 | ✅ |
| **B-07** | GT Contribution | Contribution(tile_i) = ΔmIoU(remove tile_i) | ✅ |
| **B-08** | FastSAM FSS (密集) | LoveDA/Vaihingen few-shot | ✅ |
| **B-09** | FastSAM FSS (实例) | NWPU-VHR-10 bbox weak masks | ✅ |
| **C-01** | FastSAM Zero-Shot | mR@50≈41.5% | ✅ |
| **C-02A** | Proto Matching | 1-shot mIoU=0.31% — P4≠proto 直接匹配失败 | ✅ |
| **C-02B** | Decoder Few-Shot | Proto+RefineCNN (~0.5%), 训练崩溃 | ✅ |
| **C-03** | ★ Cross-Attn + Tile | 1-shot mIoU=32.7% (118× lift) | ✅ |
| **C-04** | ★ Full 15-Class FSS | 当前活跃实验 | 🔄 |

### 6.3 关键发现

- **256×256 + FastSAM = 灾难**: P4 stride=16 → 仅 16×16=256 cells。5 种 decoder 全部 E1 peak 后崩溃
- **896×896 = 必要**: 56×56=3136 cells, 训练稳定
- **C-03 关键洞察**: Tile 不是优化 — 对 FastSAM few-shot 它是必需品

---

## 7. 评估指标

### 7.1 FSS 核心指标

| 指标 | 计算方式 | 用途 |
|------|----------|------|
| **Binary IoU** | `(pred ∩ GT) / (pred ∪ GT)` per episode | 每 episode 评估 |
| **Per-class mIoU** | mean IoU across all episodes for a class | 类别级性能 |
| **Novel mIoU** | mean of per-class mIoU over Novel classes | ★ 主要对标指标 |
| **Base val mIoU** | Meta-training 验证指标 | Checkpoint 选择 |
| **SES** | 1-shot / 5-shot mIoU ratio | 样本效率 |

### 7.2 Token Routing 指标 (待验证)

| 指标 | 定义 |
|------|------|
| **FG Recall** | Top-K 选中的 token 中 GT FG token 的占比 |
| **FG Precision** | Top-K 选中的 token 中实际是 FG 的比例 |
| **Relative Retention** | mIoU(K) / mIoU(100) × 100% |
| **Oracle→SPM gap** | (Oracle - SPM) / Oracle × 100% (越小越好) |
| **SPM→Random gap** | (SPM - Random) / Oracle × 100% (越大越好) |

### 7.3 Paper B 专项指标

| 指标 | 定义 |
|------|------|
| **SSI** (Spatial Sparsity Index) | Oracle Top40% FG retention. >70 → Router 适用 |
| **IDG** (IDEA Density Gain) | FG retention / kept fraction |
| **Spearman r** | Prediction vs GT fg_ratio 排序相关性 |

---

## 8. 当前状态与已知结果

### 8.1 系统状态

| 组件 | 状态 | 备注 |
|------|------|------|
| FastSAM Backbone (frozen, eval-only) | ✅ 就绪 | `_forward_features` 跳过 Detect head |
| LoRA 支持 | ✅ 就绪 | Feature-level, +17,920 params |
| FSS 训练 (iSAID-5i + 896×896) | 🔄 进行中 | 服务器 6GB OOM → 修复中 |
| Token Routing 验证 (Oracle/Random/SPM) | ⏸️ 等待 Decoder 训练完成 | 脚本就绪: `eval_fdr_token_fss.py` |
| FDR 模块 | ✅ 就绪 | 75K params, Spearman r=0.846 (B-03) |
| Multi-Prototype | ✅ 就绪 | [K, C] matrix + max-similarity decoder |
| Paper A 归档 | ✅ 完成 | E007→E011-U on `main` |

### 8.2 已知基线

| 实验 | 配置 | Novel mIoU |
|------|------|------------|
| 896×896 + FiLM P3P4 (之前) | Fold 0, 1-shot | ~40% (预期) |
| 896×896 + FiLM P3P4 | Fold 0, 1-shot | 🔄 等待新训练 |
| iSAID-5i SOTA 目标 | PANet/PFENet/HSNet | 37-40% |

### 8.3 下一步阻塞点

**Decoder 需要先训练出来才能跑 Oracle/Random/SPM 路由验证。** 6GB 本地卡不够 896×896 — 解决方案:
- 服务器 (>12GB GPU) 跑 896×896
- 或本地 512×512 (32×32=1024 cells, 够用)

---

## 9. 已知问题与教训

### V1 核心教训 (必须遵守)

1. **YOLOv8 eval mode**: `model.train()` 崩溃 Detect head → 保持 eval mode + `requires_grad` 控制
2. **Decoder-SPM 解耦**: Decoder 始终接收全量特征; SPM 并行训练
3. **Budget loss**: `(imp > 0.5).float().mean()` 梯度为零 → 用 `(imp.mean − target)²`
4. **SPM 三件套**: GT density focal + Top-K BCE + budget loss → 缺一不可
5. **Episodic baseline**: Baseline 也必须 episodic training → 公平对比
6. **Dice broadcast bug**: `unsqueeze(0)` + batch>1 → `[1,B,H,W]` 广播爆炸

### 当前已知问题

| 问题 | 影响 | 状态 |
|------|------|------|
| 256×256 + FastSAM 结构失配 | 仅 16×16 cells, 所有 decoder E1 崩溃 | ✅ 已确认, 用 ≥512px |
| 双标签映射 bug (train/val ID 不一致) | val mIoU ~ 0 | ✅ 已修复 (per-split mapping) |
| 6GB 显存 + 896×896 OOM | 本地无法训练 | ⚠️ CPU 特征缓存已修复, 仍需测试 |
| `_predict_once` 跑全模型 | 多占显存+时间 | ✅ `_forward_features` 跳过 head |

---

## 10. 开发路线图

### 已完成

```
✅ Phase 0: FSS 基线 (896×896 + FiLM P3P4)
✅ FastSAM backbone + hook-based feature extraction
✅ LoRA 支持 (Feature-Level)
✅ Multi-Prototype ([K, C] + max-similarity)
✅ iSAID-5i 标准协议 (official folds + 自定义 tiles)
✅ SPM/FDR 模块 (75K, Spearman r=0.846)
✅ Oracle/Random/SPM 评估脚本 (eval_fdr_token_fss.py)
✅ Paper A 归档 (E 系列全部完成)
✅ Paper B 理论链 B-00→B-03
```

### 当前: Token Routing 验证

```
🔄 Step 1: 训练 Decoder baseline (服务器 896×896)
⬜ Step 2: Oracle Routing → 验证 Token 冗余上界
⬜ Step 3: Random Routing → 建立下界
⬜ Step 4: SPM/FDR Routing → 验证方法有效性
⬜ Step 5: 梯度验证 (Oracle > SPM ≫ Random?)
```

### 后续: Dynamic Token Routing 完整系统

```
⬜ Phase 3: Prototype-guided Routing (joint score = α·SPM + β·Semantic)
⬜ Phase 4: Adaptive Budget Controller (per-image budget)
⬜ Phase 5: 完整消融 + 跨数据集 + 效率分析
```

---

## 附录

### A. 常用命令速查

```bash
# 开发安装
pip install -e ".[dev,viz]"

# 测试
pytest tests/ -v --cov=adatile --cov-report=term-missing

# Lint/Format
ruff check adatile/ && black adatile/ tests/

# 数据预处理
python tools/data/prep_isaid5i_multisize.py \
    --src-root data/iSAID_processed --dst-root data/iSAID5i_tiles \
    --sizes 896 --stride-ratio 0.57

# FSS 训练
python tools/train/train_fewshot.py \
    --dataset isaid5i --src-root data/iSAID_processed \
    --tile-root data/iSAID5i_tiles/tile_896 \
    --fold 0 --shot 1 --epochs 60 \
    --decoder film --feature-level p3p4 --device cuda

# Token Routing 验证
python tools/paper_b/eval_fdr_token_fss.py \
    --tile-root data/iSAID5i_tiles/tile_896 \
    --fold 0 --shot 1 \
    --decoder-ckpt runs/.../decoder_p3p4film_1shot_best.pt \
    --device cuda
```

### B. 项目文件索引

```
adatile/                        # 核心库
├── backbone/                   # FastSAMBackbone + LoRA
├── config/                     # ExperimentConfig + Recorder
├── datasets/                   # 8 种数据集 + FewShotEpisodeDataset
├── decoder/                    # LightDecoder, InstanceDecoder, probes
├── logging/                    # 结构化日志 (Console/File/Wandb)
├── losses/                     # FocalLoss, DiceLoss, CombinedLoss
├── metrics/                    # mIoU, Dice, FPS, Params
├── sparse/                     # FDR, DensityHead, TinyCNNRouter
└── utils/                      # seed, label_mapping, prototype, render

tools/
├── train/                      # 训练入口 (train_fewshot.py, train_b04.py)
├── paper_a/                    # Paper A 实验 (E 系列, 11 个脚本)
├── paper_b/                    # Paper B 实验 (B 系列, ~15 个脚本)
├── instance/                   # C 系列 Few-Shot 实例分割
├── data/                       # 数据预处理
├── diag/                       # 诊断工具
└── viz/                        # 可视化
```

### C. 测试覆盖

**~143 测试**: logging(30) + config(16) + backbone(14) + metrics(26) + losses(18) + datasets(12) + mass_buildings(20) + integration(7)
