# 🔬 AdaTile-FastSAM 最终代码评审报告

**评审人视角**: CVPR/ICCV 论文审稿人 + 资深深度学习架构师  
**评审日期**: 2026-06-12  
**项目**: 高分辨率实例分割 · 稀疏计算 · FastSAM 改进 · Ada-SPM · Decoupled Training  

---

## 第一部分：项目结构评审

### 1.1 当前目录结构

```
adatile/                           # 核心库 (57个.py文件)
├── adaptation/                    # CAT 域自适应模块 (cat_adapter.py)
├── backbone/                      # FastSAMHookBackbone + FPN + Timm/ResNet
│   ├── fastsam_hook.py            #    钩子式特征提取 (支持2/4级输出)
│   ├── fpn.py                     #    FPN 多尺度融合 (从 sparse/ 移入)
│   └── base.py                    #    TimmBackbone, ResNet50Backbone
├── config/                        # Dataclass 配置系统 (11个子配置)
├── core/                          # ABC 抽象接口 (SparseImportancePredictor等)
├── datasets/                      # 数据加载
│   ├── universal.py               #    自动检测布局 (bsseg/loveda/flat)
│   ├── coco.py, bsseg.py          #    COCO / BSDSeg 数据集
│   ├── collate.py                 #    批次整理 (coco/tile/fewshot/pad)
│   ├── cache/tile_cache.py        #    Tile 预计算缓存
│   ├── loaders/                   #    DynamicTile, FewShot 加载器
│   └── samplers/                  #    FewShot 采样器
├── decoder/                       # LightDecoder (UNet-like, P4+P8融合)
├── engine/                        # 训练引擎
│   ├── builder.py                 #    build_components / build_backbone / build_spm
│   ├── experiment_runner.py       #    ExperimentRunner (3种训练模式)
│   ├── trainer.py                 #    Trainer (旧版hook-based, 含AMP/DDP)
│   └── hooks.py                   #    Training hooks
├── evaluation/                    # 评估
│   ├── metrics.py                 #    COCOEvaluator, FewShotEvaluator, SparseEfficiency
│   └── sparse_eval.py             #    sparse_eval + split_support_query
├── inference/                     # 推理
│   └── tile_inference.py          #    tile_sparse_forward (真实FLOPs节省)
├── logging/                       # RunLogger, ExperimentLogger, ExperimentHook
├── losses/                        # 损失函数
│   ├── seg_loss.py                #    BinarySegLoss, MultiClassSegLoss, SegLoss
│   ├── spm_loss.py                #    DensityLoss, TopKLoss
│   ├── budget_loss.py             #    FixedBudget, LearnableBudget, EntropyBudget
│   └── unified.py                 #    UnifiedLoss (组合上述三个)
├── modeling/                      # [已清空] 旧版pipeline已删除
├── profiling/                     # 性能分析 (export, pipeline_profiler, stats, timer)
├── registry/                      # 注册表模式
├── sparse/                        # 稀疏感知
│   ├── light_spm.py               #    LightSPM (3层conv, 86行)
│   ├── ada_spm.py                 #    AdaSPM-Full (FPN+Transformer, 843行)
│   └── base.py                    #    UniformImportance baseline
└── utils/                         # 工具 (early_stop, checkpoint, oom_guard, ...)

tools/                             # 实验脚本 (12个.py)
├── train_as_fastsam.py            # 统一训练入口 (Stage A/B/C, 277行)
├── exp_fewshot.py                 # Few-shot 消融
├── ablation_spm_supervision.py    # SPM 监督方式消融 (Density vs Top-K)
├── ablation_spm_architecture.py   # SPM 架构消融 (Light vs Lite vs Full) [NEW]
├── ablation_tile_ratio.py         # Tile 保留率扫参
├── ablation_domain_shift.py       # 域迁移消融 (Urban/Rural)
├── ablation_sparse_inference.py   # 稀疏推理消融 (Full vs PostHoc vs Tile) [NEW]
├── build_isaid_dota.py            # 数据集构建
├── convert_isaid_binary.py        # 数据转换
├── convert_loveda.py              # 数据转换
└── preprocess/                    # 预处理

legacy/                            # 旧版模块 (参考资料)
├── ada_spm.py                     # 旧版 AdaSPM (已在 adatile/sparse/ 恢复)
├── routing/                       # DTRv2Router + attention
├── tokenizer/                     # DynamicTileTokenizer + TilePlanner
└── segmentation_base.py           # AdaTileFastSAMPipeline 参考
```

### 1.2 评审意见

#### 优点 ✅

1. **模块粒度合理** — 14 个顶级子包各司其职。`losses/` 4文件拆分（seg/spm/budget/unified）和 `engine/` 4文件拆分（builder/runner/trainer/hooks）是优秀的模块化设计。

2. **Import 依赖图清洁** — 所有实验脚本从 `adatile.*` 导入，零循环依赖。`from tools.train_as_fastsam import` 的引用已全部消除（仅剩自身docstring引用）。

3. **新老代码隔离彻底** — `legacy/` 目录明确隔离旧版 pipeline（routing/tokenizer/segmentation_base）；`modeling/` 包已清空旧版 builder。

4. **适配层已剥离** — `adaptation/` 独立包管理 CAT 域自适应模块，与核心 SPM 逻辑解耦。

5. **FPN 归属正确** — `backbone/fpn.py` 作为特征提取的延续放在 backbone/ 下，架构语义正确。

6. **Inference 模块独立** — `inference/tile_inference.py` 实现了真正的 tile-based 稀疏推理，是独立的推理逻辑。

#### 缺点 ⚠️

1. **`sparse/ada_spm.py` 与 `legacy/ada_spm.py` 内容几乎相同** — 两处维护同一份代码。`legacy/` 中的副本应标记为"与 active 版本不同步"的只读引用。

2. **`core/base.py` 定义的 ABC 接口与当前 pipeline 不匹配** — `SparseImportancePredictor` 抽象类的 `compute_loss` 方法是 MSE-based，但当前使用的 UnifiedLoss 不走这个接口。LightSPM 甚至没有继承 `SparseImportancePredictor`。

3. **`engine/trainer.py` (旧版 Trainer) 未被使用** — 400行的完整训练器（含 AMP/DDP/Hook系统），但所有实验都走 `ExperimentRunner` 或手动循环。

4. **`experiment_runner.py` 未被充分使用** — `train_as_fastsam.py` 和 `exp_fewshot.py` 仍保留手动训练循环（仅用 Runner 做 opt/sch/stopper 初始化）。

5. **`tools/` 中仍有参数解析重复** — 7个实验脚本各有独立的 `parse_args()`，`--dataset`、`--image-size`、`--max-steps`、`--lr`、`--unfreeze-layers` 等参数定义重复了7次。

### 1.3 改进建议

| 优先级 | 建议 | 影响 |
|--------|------|------|
| 低 | `legacy/ada_spm.py` 加注释标记"只读引用，与 active 版本不同步" | 减少混淆 |
| 低 | 提取公共 argparse 参数到 `tools/_common_args.py` | 消除7处重复 |
| 低 | 将 `train_as_fastsam.py` 的循环迁移到 `ExperimentRunner.run_epoch_based()` | 消除最后一处手动循环 |
| 低 | 将 `exp_fewshot.py` 的循环迁移到 `EpisodicRunner.run()` | 消除 episodic 手动循环 |

---

## 第二部分：创新点实现评审

### 2.1 创新点 1: Ada-SPM (Adaptive Sparse Perception Module)

#### 对应代码

| 实现 | 文件 | 行数 | 架构 |
|------|------|------|------|
| **LightSPM** | `adatile/sparse/light_spm.py` | 86行 | `Conv→BN→ReLU ×2, Conv→Sigmoid`, 输入P8单级特征 |
| **AdaSPM-Full** | `adatile/sparse/ada_spm.py` | 843行 | `FPN(4级) → SpatialTransformer(窗口注意力) → DensityHead + GranularityHead` |

#### 分析

**LightSPM** (`light_spm.py:37-85`):
```python
self.conv = nn.Sequential(
    Conv2d(in, hidden, 3), BN, ReLU,
    Conv2d(hidden, hidden, 3), BN, ReLU,
    Conv2d(hidden, 1, 1), Sigmoid()
)
# 仅使用 P8 特征, 输出 H/32×W/32 重要性图
x = features["P8"]
imp = self.conv(x)
imp = F.interpolate(imp, size=(H//32, W//32), mode="area")
```

**AdaSPM-Full** (`ada_spm.py:371-467`):
```python
# 1. FPN 融合4级特征 (P3/P4/P5/P8)
fused, pyramid = self.fusion(fp32_features)
# 2. 窗口自注意力细化
fused = self.transformer(fused)
# 3. Density + Granularity 双头预测
density = self.density_head(fused)
granularity_soft, granularity_hard = self.granularity_head(fused)
# 4. 组合重要性
importance = density * (0.5 + 0.5 * granularity_weight)
```

#### 评审意见

| 维度 | 评分 | 说明 |
|------|------|------|
| 论文思想实现 (LightSPM) | ★★☆☆☆ | 声称"自适应空间分区"，但3层conv只是单尺度重要性预测器 |
| 论文思想实现 (AdaSPM-Full) | ★★★★☆ | FPN+Transformer+Density+Granularity 实现了"自适应分区"的全部要素 |
| 消融完备性 | ★★★★☆ | `ablation_spm_architecture.py` 可对比两种实现——这是审稿人会认可的 |
| 冗余计算 | ★★★☆☆ | LightSPM 丢弃了P4特征——这是刻意简化，但需要消融数据支撑 |

**核心判断**: LightSPM 的简洁性是优势而非劣势——如果消融实验证明3层conv的Dice与AdaSPM-Full差距<1%，这就是强有力的"simple yet effective"故事。但论文必须诚实地将其定位为"Lightweight Importance Predictor"而非"Adaptive Spatial Partition Module"——后者指的是 AdaSPM-Full 的设计理念。

**FP32 强制问题** (`ada_spm.py:383`):
```python
with torch.cuda.amp.autocast(enabled=False):
    return self._forward_impl(features, return_aux)
```
AdaSPM 全程禁用 AMP——这在FPN+Transformer的组合中表明数值稳定性问题。多处NaN guard (`_compute_importance:497-544`) 进一步证实了这一点。审稿人会问："如果FP32是必须的，FLOPs优势还成立吗？"

### 2.2 创新点 2: Top-K Ranking Supervision

#### 对应代码: `adatile/losses/spm_loss.py:36-56`

```python
class TopKLoss(nn.Module):
    def forward(self, importance, gt_binary, keep_ratio):
        gt_d_flat = gt_binary.view(B, -1)
        k = max(1, int(N * keep_ratio))
        topk_vals = gt_d_flat.topk(k, dim=1).values[:, -1:]  # 每图自适应阈值
        gt_spm = (gt_d_flat >= topk_vals).float()             # 二值标签
        # Focal BCE
        bce = F.binary_cross_entropy(imp_c, gt_spm, reduction="none")
        pt = torch.where(gt_spm > 0.5, imp_c, 1 - imp_c)
        loss = ((1 - pt) ** 2 * bce).mean()
```

#### 评审意见

| 维度 | 评分 | 说明 |
|------|------|------|
| Ranking vs Classification | ★★☆☆☆ | **本质是 Threshold Classification**，不是 Ranking Learning |
| 创新性 | ★★★☆☆ | Per-image adaptive threshold 是有效的方法贡献 |
| 与 DensityLoss 区分度 | ★★★☆☆ | 区别在于"相对排序"vs"绝对回归"，需要更清晰的对比 |

**关键判断**: 
- 真正的 Ranking Learning 应该优化排序指标（NDCG/MRR）或使用 pairwise/listwise loss
- 当前实现：将 GT density 最高的 top-K% 标记为1，其余为0，做Focal BCE → **二元分类**
- 创新在于 `topk_vals = gt_d_flat.topk(k, dim=1).values[:, -1:]` — 每张图自适应阈值
- **建议定位**: "Adaptive Per-Image Threshold for Importance Binary Classification"
- 论文应诚实对比：Global Threshold vs Per-Image Threshold（这是真正的消融），而非声称是"Ranking Learning"

### 2.3 创新点 3: Sparse Routing / Sparse Inference

#### 对应代码

| 实现 | 文件 | 机制 |
|------|------|------|
| **Post-hoc Masking** | `evaluation/sparse_eval.py:144-178` | 全图forward → top-K区域 → 其他mask置零 |
| **Tile-Based Inference** | `inference/tile_inference.py:31-220` | 全图backbone → top-K tiles → 仅tile区域decoder → 缝合 |

#### 评审意见

| 维度 | Post-hoc Masking | Tile-Based Inference |
|------|-----------------|---------------------|
| 是否减少FLOPs | ❌ 零节省 | ✅ decoder FLOPs ~(1-keep_ratio) 节省 |
| Compute Allocation | ❌ 只是mask后处理 | △ 做了空间选择，但backbone仍全图 |
| 是否为 Routing | ❌ 不是routing | △ 是top-K selection，不是dynamic routing |

**关键判断**:
- **Post-hoc masking 不是创新** — 只是评测指标计算方式的变化
- **Tile-Based Inference 是有意义的工程贡献** — `tile_inference.py` 真正减少了decoder计算：
  ```python
  # 核心逻辑 (tile_inference.py:144-196)
  for idx in keep_idx:  # 只处理top-K tiles
      tile_p4 = p4_feat[:, :, h0:h1, w0:w1]   # 提取特征tile
      tile_p8 = p8_feat[:, :, h0:h1, w0:w1]
      tile_mask = decoder(features={"P4": tile_p4, "P8": tile_p8})  # 仅tile forward
      # Feathering缝合
      sparse_mask[:, :, out_h0:out_h1, out_w0:out_w1] += tile_mask * feather
  ```
- 但这仍然不是"Routing"——没有动态路由决策、没有token级处理路径选择
- **建议定位**: "Sparse Inference via Importance-Guided Tile Selection"（而非 "Sparse Routing"）

### 2.4 创新点 4: Decoupled Training

#### 对应代码: `tools/train_as_fastsam.py:96-100, 166-194`

```python
# Stage A: Backbone + Decoder (--use-spm False)
# Stage B: Backbone + Decoder + SPM (--use-spm True, --use-planner False)
# Stage C: Backbone + Decoder + SPM + Planner (--use-spm True --use-planner True)

# 训练时 Decoder 始终接收完整特征
feats = backbone(img)               # 全图特征
lgs = decoder(features=feats)       # Decoder用完整特征
imp = spm(feats) if spm else None   # SPM并行预测
loss, metrics = loss_fn(lgs, gt, imp)  # 联合loss (Seg + SPM + Budget)
```

#### 评审意见

| 维度 | 评分 | 说明 |
|------|------|------|
| 设计合理性 | ★★★★☆ | Decoder不收SPM梯度影响——设计正确 |
| 实现完成度 | ★★★☆☆ | SPM loss的梯度传回Backbone（未detach） |
| 是否存在重复训练 | ★★★☆☆ | Stage A→B从头训练Backbone，未利用预训练权重 |
| 是否存在训练目标冲突 | ★★★☆☆ | Seg loss + SPM loss 可能冲突（都通过Backbone回传） |

**关键判断**:
- **Decoupled Training 是正确的设计选择** — 这是项目最强的创新点
- **但"解耦"不彻底**: `train_as_fastsam.py:184-194` 中 Seg loss 和 SPM loss 共享 Backbone 梯度路径——理想的解耦应该是 SPM loss 对 Backbone detach
- **缺少渐进式训练**: Stage B 应加载 Stage A 的 Backbone+Decoder 权重，只初始化 SPM
- **建议**: 添加 `--resume-from` 参数，消融 "From Scratch" vs "Progressive" 的 Dice 差异

---

## 第三部分：代码质量评审

### 3.1 God File 分析

| 文件 | 行数 | 承担职责 | 状态 |
|------|------|----------|------|
| `tools/train_as_fastsam.py` | 277 | CLI + 训练循环 (2种eval分支) + checkpoint + 最终eval | **合理** — 277行已可接受 |
| `adatile/inference/tile_inference.py` | 294 | tile_sparse_forward + estimate_flops_saved | **合理** — 单一主题 |
| `adatile/sparse/ada_spm.py` | 843 | 4个SPM变体 + 2个预测头 + Transformer | **边界** — 可拆分为 heads.py + transformer.py |
| `adatile/engine/experiment_runner.py` | 487 | Base + 2子类 + 3训练模式 + 回调系统 | **合理** — 相关功能集中 |

**初评问题已修复**: UnifiedLoss、build_components、sparse_eval 已从 train_as_fastsam.py 提取到独立模块。当前无 God File（>500行且职责混杂）存在。

### 3.2 重复代码分析

| 重复内容 | 当前状态 | 涉及文件 |
|----------|----------|----------|
| 模型构建 | ✅ 已消除 — 全部使用 `build_components()` | 所有 tools/*.py |
| 参数收集 | ✅ 已消除 — 全部使用 `collect_params()` | 所有 tools/*.py |
| 损失构建 | ✅ 已消除 — 全部从 `adatile.losses` 导入 `UnifiedLoss` | 所有 tools/*.py |
| 优化器/调度器 | ✅ 已消除 — 由 `ExperimentRunner.setup()` 管理 | 所有 tools/*.py |
| **参数解析** | ⚠️ 仍重复 — `--dataset`, `--image-size`, `--max-steps`, `--lr`, `--unfreeze-layers` 在7个脚本中各自定义 | 所有 tools/*.py |
| **训练循环** | △ 部分消除 — `ablation_tile_ratio.py` 完全使用Runner, `exp_fewshot.py`/`train_as_fastsam.py` 仍手动 | train_as_fastsam.py, exp_fewshot.py |

### 3.3 废弃代码分析

| 文件 | 状态 | 建议 |
|------|------|------|
| `legacy/routing/` | 旧版DTRv2Router | **保留** — 未来路由对比实验的参考 |
| `legacy/tokenizer/` | 旧版DynamicTileTokenizer | **保留** — TilePlanner 可能被 Phase 3 未来使用 |
| `legacy/segmentation_base.py` | 旧版AdaTileFastSAMPipeline | **保留** — 定义了旧版pipeline接口 |
| `legacy/ada_spm.py` | 与 `adatile/sparse/ada_spm.py` 重复 | **保留但标记** — 加注释"只读引用" |
| `engine/trainer.py` | 旧版Trainer (Hook/AMP/DDP) | **保留** — 可能用于大规模分布式训练 |
| `core/base.py:188-483` | ABC接口（与当前pipeline不匹配） | **保留但标记** — 是接口文档参考 |
| ~~`modeling/adatile_fastsam.py`~~ | ~~旧版pipeline builder~~ | ✅ **已删除** |
| ~~`segmentation/`~~ | ~~空壳目录~~ | ✅ **已删除** |
| ~~`legacy/decoder_base.py`~~ | ~~被LightDecoder替代~~ | ✅ **已删除** |
| ~~`legacy/fastsam_backbone.py`~~ | ~~被HookBackbone替代~~ | ✅ **已删除** |
| ~~`sparse/fpn_fusion.py`~~ | ~~已移至backbone/fpn.py~~ | ✅ **已删除** |
| ~~`sparse/cat_adapter.py`~~ | ~~已移至adaptation/~~ | ✅ **已删除** |

### 3.4 Import 清洁度

```
from tools.train_as_fastsam import  →  0处 (仅自身docstring)
try/except ImportError for deleted files →  0处
旧路径 sparse.fpn_fusion →  0处
旧路径 sparse.cat_adapter →  0处
旧路径 modeling.adatile_fastsam →  0处
```

---

## 第四部分：性能优化评审

### 4.1 GPU 显存估算

| 模块 | 显存占比 | 依据 |
|------|----------|------|
| **FastSAM-x Backbone** | ~55% | YOLOv8-x ~232M 参数, 冻结但前向激活保留 (hook提取) |
| **中间特征 (P4/P8)** | ~10% | [B,128,H/16,W/16] + [B,128,H/8,W/8] |
| **LightDecoder** | ~10% | ConvTranspose + ConvBlock 中间激活 |
| **LightSPM** | ~3% | 3层小型conv, H/8→H/32 |
| **AdaSPM-Full** (如果用) | ~12% | FPN + Transformer + 双头的中间激活 |
| **优化器状态** | ~8% | AdamW 的 m/v 缓存 (unfrozen params only) |
| **损失计算** | ~2% | GT插值 + BCE/Dice |
| **其他** | ~10% | 输入图像 + 其他张量 |

**瓶颈**: Backbone 占主导。但 `torch.no_grad()` 包裹 Backbone forward (`fastsam_hook.py:145-146`)——冻结层不保留梯度。Unfrozen 层（默认2层）是主要的梯度内存消耗。

### 4.2 FLOPs 分析

| 场景 | Decoder FLOPs | 说明 |
|------|--------------|------|
| Full inference | 100% | 标准流程 |
| Post-hoc masking | 100% | **零节省** — 只是mask后处理 |
| Tile-based (kr=0.50) | ~50% | **~50%节省** — 真实减少 |
| Tile-based (kr=0.25) | ~25% | **~75%节省** |
| Tile-based (kr=0.15) | ~15% | **~85%节省** |
| Tile-based (kr=0.05) | ~5% | **~95%节省** |

`estimate_flops_saved()` (`tile_inference.py:223-294`) 提供了数学估算，与 `tile_sparse_forward()` 实际行为匹配。

**"Sparse Computation" 设计目标评估**:
- ✅ Decoder: tile-based模式实现了真实FLOPs减少
- ❌ Backbone: 仍然全图forward（SPM需要全局上下文——设计上不可避免）
- ❌ SPM: 轻量但非稀疏

### 4.3 推理速度阻塞点

| 阻塞点 | 位置 | 严重程度 | 说明 |
|--------|------|----------|------|
| **全图 Backbone Forward** | `fastsam_hook.py:145-146` | 🔴 高 | 不可避免，SPM需要全局上下文 |
| **Hook 注册/移除开销** | `fastsam_hook.py:140-149` | 🟡 中 | 每次 forward 都 register + remove |
| **32倍填充** | `fastsam_hook.py:123-127` | 🟡 中 | 非标准分辨率增加无效计算 |
| **两次插值** (LightSPM) | `light_spm.py:82-83` | 🟢 低 | Conv输出H/8→插值到H/32 |
| **FP32 强制** (AdaSPM-Full) | `ada_spm.py:383` | 🟡 中 | 禁用AMP，无法享受FP16加速 |
| **Tile stitching** | `tile_inference.py:175-196` | 🟢 低 | Feathering开销与tile数成正比 |

---

## 第五部分：论文发表性评审

### 5.1 目标期刊适配度

| 期刊/会议 | 难度 | 适配度 | 说明 |
|-----------|------|--------|------|
| **ISPRS Journal** | 中 | ★★★★☆ | 遥感应用 + 方法创新，最匹配 |
| **TGRS** | 中 | ★★★★☆ | 遥感图像分析 |
| **Pattern Recognition** | 中高 | ★★★☆☆ | 需要强调通用性 |
| **CVPR** | 极高 | ★★☆☆☆ | 创新深度不足 |
| **ICCV** | 极高 | ★★☆☆☆ | 同上 |
| **WACV** | 中 | ★★★☆☆ | 可接受的发表平台 |
| **NeurIPS** | 极高 | ★☆☆☆☆ | 理论深度不够 |

### 5.2 每个创新点评级

| 模块 | 评分 | 理由 |
|------|------|------|
| **Decoupled Training** 策略 | ★★★★☆ | 方法论正确，设计巧妙。最强贡献点 |
| **LightSPM** (3层conv) | ★★★☆☆ | 简单有效——如果消融证明与Full差距<1% |
| **AdaSPM-Full** (FPN+Transformer) | ★★★★☆ | 完整的自适应分区实现——但已退居二线 |
| **Top-K BCE Loss** | ★★★☆☆ | Per-image adaptive threshold 有效，但非 Ranking |
| **Tile-Based Sparse Inference** | ★★★☆☆ | 真实FLOPs节省——从零到有的突破 |
| **UnifiedLoss** 组合 | ★★☆☆☆ | Engineering composition, 非学术创新 |
| **Budget Loss** (可导) | ★★★☆☆ | 解决梯度断裂问题的 engineering fix |
| **CAT Adapter** | ★★★☆☆ | 借鉴 CAT-SAM，差异化不足 |
| **UniversalDataset** | ★★☆☆☆ | 工程便利，非学术创新 |
| **ExperimentRunner** | ★★☆☆☆ | 工程基础设施 |

### 5.3 创新价值排序

1. **Decoupled Sparse Training** — 训练策略创新 ★★★★☆
2. **Ada-SPM + Per-Image Adaptive Threshold** — 空间重要性学习 ★★★☆☆
3. **Tile-Based Sparse Inference** — 真实计算节省 ★★★☆☆
4. **LightSPM Architecture** — 极致轻量化 ★★★☆☆

### 5.4 建议删除/弱化的部分

| 内容 | 建议 | 原因 |
|------|------|------|
| "Sparse Routing" 术语 | 弱化为 "Sparse Inference via Importance-Guided Tile Selection" | 当前无路由——只是 top-K selection |
| "Top-K Ranking Supervision" 术语 | 弱化为 "Per-Image Adaptive Threshold for Importance Binarization" | 不是排序学习——是自适应阈值二元分类 |
| GranularityHead (tile-size预测) | 如果无实验结果则删除 | Legacy AdaSPM 有实现但当前实验不使用 |
| CAT Adapter 作为主要贡献 | 降级为 Related Work discussion | 借鉴 CAT-SAM，非原创 |

---

## 第六部分：推荐架构（当前状态已接近最优）

### 6.1 当前目录 vs 推荐

当前目录与推荐已高度一致。以下是仅有的建议调整：

| 当前 → 推荐 | 优先级 |
|-------------|--------|
| 保持现状 — 架构已合理 | — |
| `legacy/ada_spm.py` 加注释 "Reference copy — see adatile/sparse/ada_spm.py for active version" | 低 |
| 提取公共 argparse 到 `tools/_common_args.py` | 低 |

### 6.2 文件状态清单

**保留 (核心活跃)**: 57个adatile/文件 + 12个tools/文件 — 项目架构健康。

**保留 (参考资料)**: 
- `legacy/routing/`, `legacy/tokenizer/`, `legacy/segmentation_base.py` — 旧版pipeline参考
- `engine/trainer.py` — 未来分布式训练可能需要
- `core/base.py` — 接口文档价值

**已删除 (清理完毕)**: 6个文件 — `modeling/adatile_fastsam.py`, `segmentation/`, `legacy/decoder_base.py`, `legacy/fastsam_backbone.py`, `sparse/fpn_fusion.py`, `sparse/cat_adapter.py`

**新增 (本轮重构)**: 7个模块 — `losses/unified.py`, `engine/builder.py`, `engine/experiment_runner.py`, `evaluation/sparse_eval.py`, `inference/tile_inference.py`, `ablation_spm_architecture.py`, `ablation_sparse_inference.py`

---

## 第七部分：最终结论

### 7.1 综合评分

| 维度 | 分数 | 说明 |
|------|------|------|
| **项目成熟度** | **72%** | 核心pipeline稳定，多轮重构后架构健康 |
| **论文完成度** | **55%** | 实验脚本齐备但缺少SOTA对比和系统消融数据 |
| **工程质量** | **75%** | God File消除、import清洁、模块化完善 |
| **创新性** | **50%** | Decoupled Training是亮点，其余属于incremental |
| **发表潜力** (ISPRS/TGRS) | **62%** | 投稿ISPRS/TGRS机会良好，需补充SOTA和消融数据 |

### 7.2 最大优点

1. **工程架构经过验证的模块化设计** — 14个子包职责清晰，import依赖图清洁，零循环依赖

2. **Decoupled Sparse Training 策略** — 方法论正确：Decoder始终接收完整特征，SPM通过GT-driven loss并行训练。这是可以站得住脚的学术贡献

3. **消融实验覆盖完整** — 7个实验脚本覆盖SPM监督方式、SPM架构、Tile保留率、域迁移、稀疏推理模式——实验矩阵设计专业

4. **Tile-Based Sparse Inference** — `tile_inference.py` 实现了从"零FLOPs节省"到"真实~85% decoder FLOPs节省"的质变

### 7.3 最大缺点

1. **缺少 SOTA 对比实验** — 未与 Mask2Former、SAM-adapter、SegFormer 等 baseline 在相同数据集上对比。这是审稿人会最关注的问题

2. **核心创新表述与实际实现存在差距**:
   - "Adaptive Spatial Partition" → 实际是3层conv重要性预测器
   - "Sparse Routing" → 实际是top-K tile selection
   - "Top-K Ranking Supervision" → 实际是自适应阈值二元分类
   
   论文需要用诚实的术语重新定位这些创新点

3. **AdaSPM-Full 的数值稳定性未根治** — FP32强制 + 多处NaN guard 表明存在未解决的工程问题

4. **缺少端到端的分布式训练/评测运行记录** — 所有脚本只有 quick mode 验证，缺少 full run 的实验结果数据

### 7.4 下一步最值得投入的方向

1. **运行所有消融实验，产出论文数据** (优先级: 🔴 最高)
   - `ablation_spm_architecture.py`: Light vs Lite vs Full 的 Dice/Coverage/FLOPs/#Params 对比
   - `ablation_sparse_inference.py`: FLOPs vs Dice trade-off 曲线
   - `ablation_domain_shift.py`: Urban↔Rural 域迁移结果

2. **补充 SOTA 对比** (优先级: 🔴 最高)
   - 在 iSAID / LoveDA 上对比 Mask2Former, SAM-adapter, SegFormer
   - 指标: mIoU, Dice, FLOPs, #Params, 推理时间

3. **修复 Decoupled Training 的不彻底问题** (优先级: 🟡 中)
   - Stage B: 加载 Stage A 权重，SPM loss 对 Backbone detach
   - 消融: From Scratch vs Progressive 的 Dice 差异

4. **论文撰写时务实的术语调整** (优先级: 🟡 中)
   - "Ada-SPM" → "Lightweight Importance Predictor with Per-Image Adaptive Threshold"
   - "Sparse Routing" → "Sparse Inference via Importance-Guided Tile Selection"
   - "Top-K Ranking" → "Per-Image Adaptive Threshold Binarization"

### 7.5 最应该停止投入的方向

1. **CAT 模块** (`adaptation/cat_adapter.py`) — 344行代码与当前实验无关，先验证基础pipeline
2. **旧版 pipeline 恢复** — `legacy/routing/` 和 `legacy/tokenizer/` 的 DTRv2Router 和 DynamicTileTokenizer 不值得恢复
3. **更多实验脚本** — 现有7个已足够，在产出数据之前不要再新增
4. **过度复杂的 ExperimentRunner 扩展** — 当前的手动循环 + Runner setup 的混合模式是务实的80/20方案，不需要100%迁移

---

*评审完成。本报告基于对全部代码文件的逐行阅读和两轮重构后的最终状态。*
