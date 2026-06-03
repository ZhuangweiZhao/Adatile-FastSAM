# AdaTile-FastSAM: 自适应稀疏瓦片化的高分辨率小样本实例分割

> **Ada**ptive Sparse **Tile**-based **Fast** **S**egment **A**nything **M**odel

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 11.8+](https://img.shields.io/badge/cuda-11.8+-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 📖 目录

- [项目概述](#项目概述)
- [核心动机](#核心动机)
- [创新亮点](#创新亮点)
- [整体架构](#整体架构)
- [技术路线](#技术路线)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [训练与评测](#训练与评测)
- [基准测试](#基准测试)
- [数据集](#数据集)
- [实验结果](#实验结果)
- [引用](#引用)

---

## 项目概述

**AdaTile-FastSAM** 是一个面向**高分辨率遥感/航拍图像**的**小样本实例分割**框架。核心思想是 **"用计算换分辨率"**：通过一个可学习的稀疏重要性预测器（Ada-SPM）动态决定 _在哪里_ 以及 _以何种粒度_ 处理图像，而非均匀地在整张高分辨率图像上密集计算。

本框架将以下四项技术创新有机整合为一个端到端的可训练系统：

| 模块 | 角色 | 创新性 |
|------|------|--------|
| **Ada-SPM** | 学习「哪里值得看」 | 🌟 核心创新 |
| **Dynamic Tile Planner** | 将重要性图转化为具体瓦片方案 | 🌟 核心创新 |
| **DTR-v2 Router** | 可微的 4 级 token 路由 + 计算预算控制 | 🌟 核心创新 |
| **Prototype Memory** | 掩码平均池化原型 + 余弦相似度检索 | 🌟 核心创新 |

---

## 核心动机

### 问题：高分辨率图像 × 小样本 × 有限算力

1. **高分辨率**：遥感图像动辄 4000×4000 甚至更大，全图密集处理导致 OOM 或推理延迟不可接受
2. **小样本**：遥感标注成本极高（每张图数百个实例，需专家标注），每类仅有 1-5 个标注样本
3. **算力受限**：研究者通常只有消费级 GPU（6-12 GB 显存），无法处理高分辨率全图

### 现有方案的局限

| 方案 | 问题 |
|------|------|
| 整图下采样 | 丢失小目标（航拍图中的车辆、船只） |
| 均匀滑窗 | 80%+ 窗口是无效背景区域，计算浪费 |
| 固定多尺度金字塔 | 无法自适应图像内容稀疏性 |
| SAM / FastSAM | 需要大量 prompt，不支持端到端小样本 |

### 我们的方案

> **只在高信息密度区域使用细粒度瓦片，在背景区域使用粗粒度瓦片或直接跳过**

这一思路受 Google Gemini 1.5 的「全局缩略图 + 局部高分辨率瓦片」架构启发，但做了两点关键改造：

1. **可学习的稀疏性**：瓦片的位置和粒度不是固定的，而是由 Ada-SPM 从 backbone 特征中端到端学习
2. **可微的 token 路由**：DTR-v2 使用 Gumbel-Softmax + 预算控制器实现完全可微的 token 级路由，保证梯度流通

---

## 创新亮点

### 1. Ada-SPM（自适应空间划分模块）

- **输入**：backbone 多尺度特征（通过 FPN 融合）
- **输出**：`SparsePrediction(importance, density, granularity)`
- **架构**：FPN 融合 → SpatialTransformerRefine（窗口自注意力）→ DensityHead（Conv→Sigmoid）+ GranularityHead（Conv→Gumbel-Softmax）
- **变体**：`AdaSPM` / `AdaSPMLite` / `AdaSPMFull` / `DensityOnlySPM`

```
Backbone Features → FPN Fusion → Window Self-Attn → ┬→ Density Head → [B,1,H,W]
                                                       └→ Granularity Head → [B,K,H,W]
```

### 2. Dynamic Tile Planner（动态瓦片规划器）

将 Ada-SPM 输出的连续重要性图转化为**具体可执行的瓦片提取方案**：

- **三种跳过模式**：
  - `threshold`：重要性 < 阈值则跳过，边界区域用粗粒度瓦片
  - `hard`：严格阈值跳过（无边界区）
  - `topk`：保留 Top-K 重要区域
- **四叉树自适应分解**：高信息密度区域递归细分为更小瓦片（`plan_quadtree()`）
- **可微对齐损失**：`planning_alignment_loss` 将规划器的离散决策与 Ada-SPM 的连续预测桥接，保证端到端梯度

### 3. DTR-v2 Router（可微 4 级 token 路由）

每个瓦片 token 被动态分配到 4 个处理级别：

| Level | 处理方式 | 计算量 | 适用场景 |
|-------|---------|--------|---------|
| 0 (Skip) | 直接丢弃 | 0 | 纯背景、无信息区域 |
| 1 (Window) | 窗口局部注意力 | 低 | 低密度区域 |
| 2 (Block-Sparse) | 块稀疏注意力 | 中 | 中等密度区域 |
| 3 (Full) | 全注意力（FlashAttention-2） | 高 | 高密度、关键区域 |

**关键设计**：
- 使用 **Gumbel-Softmax** + **Straight-Through 梯度估计**实现完全可微
- **BudgetController** 通过可微 logit 偏置控制各级别计算预算（_非后验分配_）
- **PrototypeRouter** 将小样本原型相似度注入路由 logit，引导相关 token 进入高处理级别

### 4. Prototype-guided Few-Shot（原型引导小样本）

- **原型计算**：对支持集进行掩码平均池化 → 每类一个 `[C]` 向量
- **检索机制**：查询 token 与原型余弦相似度 → 注入路由和 decoder
- **无需微调**：支持任意 N-way K-shot 配置（1/5/10-shot）

---

## 整体架构

### 端到端 Pipeline（Gemini 启发式 Global+Local）

```
┌─────────────────────────────────────────────────────┐
│                  原始高分辨率图像                      │
│                 [B, 3, 4000×4000]                    │
└──────────────┬──────────────────┬───────────────────┘
               │                  │
         缩略图 512²          原始分辨率
               │                  │
    ┌──────────▼──────────┐       │
    │   timm Backbone      │       │
    │  (ResNet50/Swin-T)   │       │
    └──────────┬──────────┘       │
               │                  │
    ┌──────────▼──────────┐       │
    │    FPN Fusion        │       │
    │  (torchvision FPN)   │       │
    └──────────┬──────────┘       │
               │                  │
    ┌──────────▼──────────┐       │
    │     Ada-SPM    🌟    │       │
    │  importance + density │      │
    │  + granularity        │      │
    └──────────┬──────────┘       │
               │                  │
         上采样到原图尺度          │
               │                  │
    ┌──────────▼──────────────────▼──────┐
    │     Dynamic Tile Planner 🌟         │
    │   瓦片规划：位置、大小、跳过比例       │
    └──────────┬─────────────────────────┘
               │
    ┌──────────▼─────────────────────────┐
    │     Token Generator + PE            │
    │   F.grid_sample 提取原始分辨率瓦片    │
    └──────────┬─────────────────────────┘
               │
    ┌──────────▼─────────────────────────┐
    │       DTR-v2 Router 🌟              │
    │  Level 0/1/2/3 可微路由 + 预算控制    │
    │  FlashAttention-2 dispatch          │
    └──────────┬─────────────────────────┘
               │
    ┌──────────▼─────────────────────────┐
    │   FastSAM-style Decoder             │
    │  Proto + Coeff Head + batched_nms   │
    └──────────┬─────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Prototype Guidance  │ (小样本模式)
    │   🌟 掩码平均池化原型  │
    └──────────────────────┘
               │
    ┌──────────▼──────────┐
    │  SegmentationOutput   │
    │  masks + scores + boxes│
    └──────────────────────┘
```

### 关键设计决策

| 决策 | 原因 |
|------|------|
| Backbone 仅处理 512² 缩略图 | 4000² 全图 backbone 在 6GB GPU 上 OOM |
| 瓦片从**原始分辨率**图像采样 | 保留细节，不损失小目标 |
| Decoder 输出**瓦片内裁剪 mask** | `O(Σ h_i·w_i)` 而非 `O(N_inst · H · W)` |
| FPN 自实现（~80 行） | torchvision FPN 与此 PyTorch 版本不兼容 |
| Attention 使用 SDPA [B,H,N,D] | PyTorch 2.0+ 自动 dispatch 到 FlashAttention-2 |
| 混合精度 FP16 + NaN 防护 | 6GB GPU 必需；关键路径用 `autocast(enabled=False)` 包裹 |

---

## 技术路线

### 可信第三方基础设施

框架建立在经过社区验证的成熟库之上——只有创新模块是自定义的：

| 组件 | 使用的库 | 文件位置 |
|------|---------|---------|
| Backbone | `timm` (ResNet50, Swin-T, ViT) | `adatile/backbone/base.py` |
| FPN | 自实现 (兼容性原因) | `adatile/sparse/fpn_fusion.py` |
| Full Attention | `F.scaled_dot_product_attention` (FlashAttention-2) | `adatile/routing/attention.py` |
| Window/Block-Sparse Attn | SDPA + 手工 mask | `adatile/routing/attention.py` |
| Decoder | FastSAM 架构 (Proto + Coeff) | `adatile/decoder/base.py` |
| NMS | `torchvision.ops.batched_nms` | `adatile/decoder/base.py` |
| Profiling | `torch.profiler` + `fvcore` | `adatile/profiling/` |
| Loss | Dice + Focal + MSE (自实现) | `adatile/segmentation/base.py` |

### 六核心抽象接口

所有组件继承自 `adatile/core/base.py` 中定义的 ABC：

```
SparseImportancePredictor  →  预测重要性/密度/粒度
DynamicTileTokenizer       →  图像 → 瓦片 + tokens
BaseRouter                 →  tokens → 路由输出
PrototypeMemory            →  支持集 → 类别原型
GlobalContextBranch        →  全局场景上下文
SegmentationDecoder        →  瓦片特征 → 实例 mask
```

### 类型化数据结构

```python
@dataclass(frozen=True)
class SparsePrediction:
    importance: Tensor          # [B, 1, H, W] ∈ [0,1]
    density: Tensor             # [B, 1, H, W] ∈ [0,1]
    granularity_soft: Tensor    # [B, K, H, W] 可选
    granularity_hard: Tensor    # [B, 1, H, W] 可选

@dataclass
class RoutingOutput:
    routed_tokens: Tensor       # [N_active, C]
    assignments: Tensor         # [N_active, 1]  ∈ {1,2,3}
    routing_weights: Tensor     # [N_active, 1]
    skipped_mask: Tensor        # [N_total] bool
    aux_loss: Tensor | None

@dataclass
class SegmentationOutput:
    masks: Tensor               # [N_inst, H, W]
    scores: Tensor              # [N_inst]
    boxes: Tensor | None        # [N_inst, 4]
    classes: Tensor | None      # [N_inst]
```

### 注册器系统

10 个注册器实现解耦和可扩展性：`BACKBONE`, `SPARSE`, `TOKENIZER`, `ROUTER`, `DECODER`, `PROTOTYPE`, `SEGMENTATION`, `DATASET`, `TRANSFORM`, `LOSS`

```python
@ROUTER.register()
class DTRv2Router(BaseRouter):
    ...

router = build_router("DTRv2Router", embed_dim=256)
```

### 配置系统

纯 dataclass、不可变友好、支持多层嵌套和 CLI 覆盖：

```bash
python tools/train.py --config configs.isaid.get_isaid_config \
    -o train.epochs=100 data.batch_size=4 sparse.importance_threshold=0.15
```

**[注意]** 该项目尚有不完善之处（部分模块需要优化和补充），欢迎大家在遵守 MIT 许可证的前提下根据自身需求自行修改和使用。由于代码处于快速迭代阶段，建议在使用前仔细检查各模块的兼容性和正确性。</parameter>
