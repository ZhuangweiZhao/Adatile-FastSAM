# AdaTile-FastSAM 代码学习指南

> 一份按「依赖关系从底层到顶层、从数据到模型、从训练到评测」组织的系统性代码阅读路线图。

---

## 目录

- [阅读路线总览](#阅读路线总览)
- [第一阶段：基础设施层](#第一阶段基础设施层)
- [第二阶段：核心数据结构与抽象接口](#第二阶段核心数据结构与抽象接口)
- [第三阶段：模块注册与配置系统](#第三阶段模块注册与配置系统)
- [第四阶段：Backbone 与 FPN](#第四阶段backbone-与-fpn)
- [第五阶段：稀疏预测模块](#第五阶段稀疏预测模块)
- [第六阶段：动态瓦片化](#第六阶段动态瓦片化)
- [第七阶段：Token 路由与稀疏注意力](#第七阶段token-路由与稀疏注意力)
- [第八阶段：分割解码器](#第八阶段分割解码器)
- [第九阶段：原型记忆](#第九阶段原型记忆)
- [第十阶段：完整 Pipeline 组装](#第十阶段完整-pipeline-组装)
- [第十一阶段：模型构建与入口](#第十一阶段模型构建与入口)
- [第十二阶段：数据集与数据加载](#第十二阶段数据集与数据加载)
- [第十三阶段：训练引擎](#第十三阶段训练引擎)
- [第十四阶段：评测与性能分析](#第十四阶段评测与性能分析)
- [第十五阶段：工具与配置](#第十五阶段工具与配置)
- [第十六阶段：测试代码](#第十六阶段测试代码)
- [全局数据流全景图](#全局数据流全景图)
- [模块依赖关系图](#模块依赖关系图)
- [关键梯度流路径](#关键梯度流路径)

---

## 阅读路线总览

整个项目约 **60 个 Python 源文件**，按功能分为 16 个阶段。建议按以下顺序阅读：

```
Phase 1-3:  基础设施  (utils/, core/, config/, registry/)     ← 零依赖，先读
Phase 4:    特征提取  (backbone/, sparse/fpn_fusion.py)
Phase 5-9:  核心创新  (sparse/, tokenizer/, routing/, decoder/, prototype/)
Phase 10-11: 组装     (segmentation/, modeling/)
Phase 12:   数据      (datasets/)
Phase 13:   训练      (engine/)
Phase 14:   评测      (evaluation/, profiling/)
Phase 15:   入口      (tools/, benchmark.py, configs/)
Phase 16:   测试      (tests/)
```

---

## 第一阶段：基础设施层

> **先读这些** — 不依赖任何项目内部模块，是整个项目的基础。

### 1.1 `adatile/utils/logging.py`

**作用**：日志、进度条、运行平均值。

**关键类/函数**：
| 符号 | 角色 |
|------|------|
| `setup_logger(name, level)` | 配置 rich 彩色日志 |
| `get_logger()` | 获取模块级 logger |
| `AverageMeter(name, fmt)` | 指数移动平均追踪器，支持 `update(val, n)` 和 `reset()` |
| `ProgressMeter(meters, prefix)` | 批量打印多个 AverageMeter 的格式化输出 |

**阅读要点**：`AverageMeter` 使用 `val * n / count` 加权更新，是训练过程中追踪 loss 的基础组件。

---

### 1.2 `adatile/utils/checkpoint.py`

**作用**：模型检查点的保存、加载、管理。

**关键类/函数**：
| 符号 | 角色 |
|------|------|
| `save_checkpoint(model, optimizer, step, epoch, cfg, path)` | 保存完整训练状态到 `.pth` |
| `load_checkpoint(path, model, optimizer, device)` | 恢复训练状态，返回 `(step, epoch, metrics)` |
| `CheckpointManager(checkpoint_dir, max_to_keep)` | 管理检查点生命周期，自动清理旧文件，支持 `save_best()` |

**阅读要点**：`CheckpointManager` 保存的字典结构为 `{"model": ..., "optimizer": ..., "step": ..., "epoch": ..., "cfg": ...}`。

---

### 1.3 `adatile/utils/distributed.py`

**作用**：分布式训练支持（DDP）。

**关键函数**：
| 函数 | 角色 |
|------|------|
| `is_distributed()` | 检查是否在 DDP 环境中 |
| `get_rank()` / `get_world_size()` | 获取当前进程排名和总进程数 |
| `init_distributed(backend)` | 初始化 NCCL 进程组 |
| `reduce_tensor(tensor)` | 跨所有进程求和 |
| `gather_tensor(tensor)` | 收集所有进程的 tensor |
| `synchronize()` | 等待所有进程到达 barrier |

---

### 1.4 `adatile/utils/memory_logger.py`

**作用**：GPU 显存追踪。

**关键类**：
| 符号 | 角色 |
|------|------|
| `MemorySnapshot(allocated_mb, reserved_mb, peak_mb, stage, step)` | 单次显存快照 |
| `MemoryLogger(warn_threshold_mb, log_dir)` | 每阶段追踪显存变化，支持 CSV 导出、阈值警告 |

**阅读要点**：`MemoryLogger.log(stage_name)` 调用 `torch.cuda.memory_allocated()` 等 API 记录当前显存状态。

---

### 1.5 `adatile/utils/oom_guard.py`

**作用**：预测性 OOM 检测。

**关键类**：
| 符号 | 角色 |
|------|------|
| `OOMGuard(warn_threshold_mb, critical_threshold_mb)` | 在分配前检查张量大小是否安全 |
| `CrashReport` | 训练崩溃时自动保存诊断信息 |
| `check_tensor(shape, name, dtype_bytes)` → bool | 核心预测函数 |
| `check_mask_allocation(n_inst, h, w)` | 特化检查全图 mask 分配风险 |
| `wrap_forward(fn, stage_name, *args)` | Try-catch 包装器，OOM 时保存 crash report |

**阅读要点**：`check_tensor` 使用 `total_gpu_mb * 0.9` 作为硬上限，`critical_threshold_mb`（默认 4000MB）作为单张量上限。

---

### 1.6 `adatile/utils/diagnostics.py`

**作用**：统一的训练诊断数据收集。

**关键类**：
| 符号 | 角色 |
|------|------|
| `TileStats` | 瓦片数量、大小分布、跳过比例 |
| `TokenStats` | Token 总数、有效 token 数、压缩率 |
| `RouterStats` | 路由级别分布、平均权重 |
| `DecoderStats` | 实例数、原型形状、平均分数 |
| `LatencyStats` | 各阶段耗时 |
| `DiagnosticsCollector` | 聚合所有统计，写入 CSV 和 TensorBoard |

---

## 第二阶段：核心数据结构与抽象接口

> ⭐ **最重要的文件** — 定义了整个项目的类型系统和模块契约。

### 2.1 `adatile/core/base.py`

**文件职责**：定义 4 个类型化数据结构 + 7 个抽象基类。所有模块必须遵守这些接口。

#### 数据结构

```python
@dataclass
class TileInfo:            # 单个瓦片元数据
    tile_id: str           # 唯一标识
    image_id: str          # 所属图像
    x1, y1, x2, y2: int    # 边界框坐标
    tile_size: int         # 瓦片边长
    object_density: float  # 物体密度 (用于优先级排序)

@dataclass(frozen=True)
class SparsePrediction:    # Ada-SPM 输出 (冻结 = 不可变)
    importance: Tensor     # [B, 1, H, W] 组合重要性 ∈ [0,1]
    density: Tensor        # [B, 1, H, W] 原始密度
    granularity_soft: Optional[Tensor]  # [B, K, H, W] 软分类
    granularity_hard: Optional[Tensor]  # [B, 1, H, W] 硬分类 (long)
    # __post_init__ 中执行严格验证

@dataclass
class RoutingOutput:       # 路由器输出
    routed_tokens: Tensor  # [N_active, C] 处理后的 token
    assignments: Tensor    # [N_active, 1] ∈ {1,2,3} (窗口/块稀疏/全注意力)
    routing_weights: Tensor # [N_active, 1] 路由概率
    skipped_mask: Tensor   # [N_total] 布尔掩码 (True=被丢弃)
    aux_loss: Optional[Tensor] # 辅助负载均衡损失

@dataclass
class SegmentationOutput:  # 解码器输出
    masks: Tensor          # [N_inst, max_h, max_w] 裁剪级别 mask
    scores: Tensor         # [N_inst] 置信度
    boxes: Optional[Tensor] # [N_inst, 4]
    classes: Optional[Tensor]
```

#### 关键设计细节

1. **`SparsePrediction` 是 frozen dataclass**：一旦创建不可修改，`__post_init__` 中执行 8 项验证（4D 检查、channel=1 检查、dtype 检查等）
2. **`SegmentationOutput.masks` 是裁剪级别的**：_永远不是_ `[N_inst, H_full, W_full]`。每个实例的 mask 只覆盖其所在瓦片的 ROI 区域。`max_h` 和 `max_w` 最多 256。这是避免 OOM 的关键
3. **`RoutingOutput.skipped_mask`**：Level-0 token（被完全丢弃）标记为 True

#### 抽象基类

| ABC | 核心方法 | 具体实现 |
|-----|---------|---------|
| `SparseImportancePredictor` | `forward(features) -> SparsePrediction` | `AdaSPM`, `AdaSPMLite`, `AdaSPMFull` |
| `DynamicTileTokenizer` | `forward(image, features, importance, granularity) -> (List[TileInfo], Tensor)` | `DynamicTileTokenizerImpl`, `UniformTileTokenizer` |
| `BaseRouter` | `forward(tokens, metadata) -> RoutingOutput` | `DTRv2Router`, `UniformRouter`, `IdentityRouter` |
| `PrototypeMemory` | `forward(support_features, support_masks) -> Dict[int,Tensor]` | `MaskedAveragePrototype` |
| `GlobalContextBranch` | `forward(image, features) -> (Tensor, Dict)` | `GlobalThumbnailBranch` |
| `SegmentationDecoder` | `forward(tile_features, tile_infos, ...) -> SegmentationOutput` | `FastSAMDecoder` |
| `LossFunction` | `forward(predictions, targets, **kwargs) -> Tensor` | `DiceLoss`, `FocalLoss`, `SegmentationLoss` |

**阅读要点**：这是理解整个项目的「地图」。每个模块都可以通过其 ABC 接口来理解——不需要先看实现细节。所有 `forward()` 的输入输出类型在 ABC 中已明确。

---

## 第三阶段：模块注册与配置系统

### 3.1 `adatile/registry/registry.py`

**职责**：提供装饰器方式的模块注册 + 工厂构建。

```python
class Registry(Generic[T]):
    def register(self, name=None):  # 返回装饰器
        """@ROUTER.register() 或 @ROUTER.register("MyRouter")"""

    def build(self, name, **kwargs):  # 工厂方法
        """根据名称字符串实例化已注册类"""

    def get(self, name):  # 按名查找
    def list(self):       # 列出所有注册名
```

10 个模块级注册表单例：
```python
BACKBONE    SPARSE    TOKENIZER    ROUTER    DECODER
PROTOTYPE   SEGMENTATION   DATASET    TRANSFORM   LOSS
```

**阅读要点**：这是实现「可插拔」架构的关键。每个具体类用 `@ROUTER.register()` 装饰后，即可通过 `build_router("DTRv2Router")` 按字符串名实例化。

---

### 3.2 `adatile/config/config.py`

**职责**：纯 dataclass 配置系统，支持 YAML/JSON/CLI 覆盖。

#### 配置层次结构

```
Config (顶层)
├── BackboneConfig    (name, embed_dim, output_scales, pretrained, ...)
├── SparseConfig       (name, importance_threshold, density_loss_weight, ...)
├── TokenizerConfig    (name, max_tokens_per_image, skip_mode, tile_sizes, ...)
├── RouterConfig       (name, embed_dim, max_full_ratio, aux_loss_weight, ...)
├── PrototypeConfig    (name, prototype_dim, temperature, ...)
├── DecoderConfig      (name, mask_dim, num_mask_tokens, ...)
├── DataConfig         (name, root_dir, batch_size, fewshot_split, n_shot, ...)
├── TrainConfig        (lr, max_steps, mixed_precision, log_interval, ...)
├── EvalConfig         (metrics, iou_thresholds, max_dets, ...)
```

#### 关键方法

| 方法 | 用途 |
|------|------|
| `Config.from_yaml(path)` | 从 YAML 文件加载 |
| `Config.from_dict(d)` | 从嵌套字典构建 |
| `cfg.to_yaml()` / `cfg.to_dict()` | 序列化 |
| `cfg.clone()` | 深拷贝 |

#### CLI 覆盖语法

```bash
# 点号分隔的嵌套键 = 新值
python tools/train.py -o train.epochs=100 data.batch_size=4 sparse.importance_threshold=0.15
```

实现原理：将 `"train.epochs=100"` 解析为 `{"train": {"epochs": 100}}`，递归合并到 `Config`。

**阅读要点**：所有默认值定义在各子 config 的 dataclass 字段默认值中。查看一个配置参数的含义，直接看对应的 `*Config` 类即可。

---

## 第四阶段：Backbone 与 FPN

### 4.1 `adatile/backbone/base.py`

**职责**：基于 `timm` 的多尺度特征提取。

```
输入: [B, 3, 512, 512] (缩略图)
  ↓
timm.create_model(name, features_only=True, out_indices=(1,2,3,4))
  ↓
输出: {"p2": [B,C,H/4,W/4], "p3": [B,C,H/8,W/8],
       "p4": [B,C,H/16,W/16], "p5": [B,C,H/32,W/32]}
```

**两个注册类**：

| 类 | 差异 |
|----|------|
| `TimmBackbone` | 通用类，可指定任意 timm 模型名 |
| `ResNet50Backbone` | 继承 `TimmBackbone(name="resnet50")`，增加 `freeze_stages` 兼容 |

**关键设计**：`out_indices=(1,2,3,4)` 跳过 ResNet 的 stem（stride=2+2=4），从 stride=4 开始输出。`features_only=True` 使 timm 返回多尺度特征而非最终分类 logit。

**工厂函数别名**：`build_backbone("fastsam_vit_b")` → `ResNet50Backbone`，`build_backbone("resnet50")` → `ResNet50Backbone`。

---

### 4.2 `adatile/sparse/fpn_fusion.py`

**职责**：将 backbone 的不同通道维度统一到相同维度，融合多尺度信息。

**为什么自实现**：torchvision FPN 的 `get_result_from_inner_blocks` 与当前 PyTorch 版本不兼容。

```
输入: {"p2": [B,256,H/4,W/4], "p3": [B,512,H/8,W/8], ...}
         ↓
    Lateral 1x1 Conv (每个尺度 → out_dim)
         ↓
    Smooth 3x3 Conv
         ↓
    Top-Down Pathway: p5 → upsample → add p4 → upsample → add p3 → ...
         ↓
输出: (fused [B,out_dim,H/32,W/32], pyramid_list)
```

**两个类**：
| 类 | out_dim | 用途 |
|-----|---------|------|
| `MultiScaleFPNFusion` | 256 | AdaSPM / AdaSPMFull |
| `LightweightFPNFusion` | 128 | AdaSPMLite (低延迟) |

**关键细节**：使用延迟初始化 (`_ensure_built`)，在首次 forward 时根据实际输入通道数创建 Conv2d。这解决了 AMP 下 dtype 不匹配的问题。

---

## 第五阶段：稀疏预测模块

> ⭐ **核心创新 #1：Ada-SPM**

### 5.1 `adatile/sparse/ada_spm.py`

**职责**：从 backbone 多尺度特征预测「哪里值得处理」。

#### 完整前向流程

```
Backbone Features → FPN Fusion → [B,256,H/32,W/32]
    │
    ├─→ SpatialTransformerRefine (可选, 窗口自注意力)
    │       │  8×8 窗口内的多头自注意力
    │       │  深度可分离 FFN
    │       ↓
    ├─→ DensityHead                    ├─→ GranularityHead
    │   Conv3x3→Conv3x3→Conv1x1→Sigmoid│   Conv3x3→Conv3x3→Conv1x1
    │   ↓                              │   ↓
    │   density [B,1,H,W]             │   logits [B,K,H,W]
    │                                  │   ↓ Gumbel-Softmax
    │                                  │   granularity_soft [B,K,H,W]
    │                                  │   ↓ argmax
    │                                  │   granularity_hard [B,1,H,W]
    │                                  │
    └─→ _compute_importance ──────────┘
        importance = density * (0.5 + 0.5 * Σ(tile_size_weight * bias * granularity))
```

#### 子模块详解

**`SpatialTransformerRefine`**：
- 将特征图分割为 8×8 窗口
- 在每个窗口内执行 Self-Attention
- 使用深度可分离卷积作为 FFN
- **关键**：计算量为 `O(N * W²)` 而非全局注意力的 `O(N²)`

**`DensityHead`**：
- 最简单的部分：Conv → Sigmoid
- 输出 `[B, 1, H, W]`，每个值在 [0, 1] 之间
- 监督信号：GT 实例密度图

**`GranularityHead`**：
- 输出 `[B, K, H, W]`，K 为瓦片尺寸类别数
- 训练时通过 Gumbel-Softmax 产生软分配（保持可微）
- 推理时 argmax 产生硬分配
- `tile_importance_bias`: 可学习参数 `[1, K, 1, 1]`，初始化为 `linspace(0.2, 0.8, K)`，自动学习每种瓦片尺寸的重要性贡献

**`_compute_importance(density, granularity_soft)`**：
```python
# 核心公式
importance = density * (0.5 + 0.5 * Σ_k(weight[k] * bias[k] * granularity[k]))
# weight[k]: 小瓦片权重大（= 更精细处理），大瓦片权重小
```

**4 个注册变体**：

| 变体 | fusion_dim | hidden_dim | Transformer | 用途 |
|------|-----------|------------|-------------|------|
| `AdaSPM` | 256 | 128 | ✓ | 基础版 |
| `AdaSPMLite` | 128 | 64 | ✗ | 低计算量 |
| `AdaSPMFull` | 256 | 256 | ✓ | 高精度 |
| `DensityOnlySPM` | 256 | - | ✗ | 消融实验 (无粒度) |

#### Loss 方法

| 方法 | 公式 | 作用 |
|------|------|------|
| `compute_density_loss(pred, target)` | MSE(pred_importance, target_density) | 监督密度预测 |
| `compute_entropy_loss(granularity)` | -Σ p*log(p) | 鼓励确定的瓦片尺寸选择 |
| `compute_sparsity_loss(importance, target)` | L1(mean(imp) - target_sparsity) | 控制总体稀疏性 |

---

### 5.2 `adatile/sparse/base.py`

**职责**：消融实验用的 baseline。

- `UniformImportance`：输出全 1 的重要性图（= 所有区域同等重要）
- 重新导出 `AdaSPM`, `AdaSPMLite` 等

---

### 5.3 `adatile/sparse/analysis.py`

**职责**：稀疏预测的可视化和分析工具。

| 类/函数 | 用途 |
|---------|------|
| `FLOPsCounter` | 基于 hook 的手动 FLOPs 追踪 |
| `SparsityTracker` | 重要性/密度/跳过比例的滚动窗口统计 |
| `RoutingAnalysisHook` | 捕获重要性分布、瓦片尺寸直方图 |
| `GradientFlowMonitor` | 跟踪各子模块的梯度范数 |
| `render_importance_heatmap()` | 重要性热力图 |
| `render_granularity_map()` | 粒度分配可视化 |

---

## 第六阶段：动态瓦片化

> ⭐ **核心创新 #2：Dynamic Tile Planner + Token Generator**

### 6.1 `adatile/tokenizer/tile_planner.py`

**职责**：将连续的重要性图转化为「具体从哪里提取多大瓦片」的执行方案。

#### 数据结构

```python
@dataclass
class TileSpec:    # 单个规划的瓦片
    x1, y1, x2, y2: int     # 坐标
    tile_size: int           # 边长 (如 384, 768, 1536)
    stride: int              # 与相邻瓦片的重叠步长
    importance: float        # 该位置的 Ada-SPM 重要性
    priority: float          # 预算排序优先级 (= importance * size_weight)
    density: float           # 物体密度
    scale_level: int         # 粒度层级 (0=最细, K-1=最粗)

@dataclass
class PlannerStats:  # 效率统计
    total_cells: int          # 图像网格单元总数
    skipped_cells: int        # 完全跳过的单元
    borderline_cells: int     # 边界单元 (粗瓦片)
    cells_with_tiles: int     # 有瓦片的单元
    skip_ratio: float         # 跳过比例
    estimated_flops_saved: float
    estimated_memory_saved_bytes: float

@dataclass
class TilePlan:     # 完整分配方案
    specs: List[TileSpec]
    image_size: Tuple[int,int]
    total_tiles: int
    token_budget_used: int
    token_budget_max: int
    planner_stats: PlannerStats
```

#### 核心类 `TilePlanner`

**`plan(importance, image_size, granularity_hard, ...) -> TilePlan`**：

```
输入: importance [1,1,H_s,W_s], 原图尺寸 (H,W)
  │
  ├─ 1. 网格划分: 将原图分为 N×M 个 cell (按 stride 滑动)
  │
  ├─ 2. 逐 cell 分配:
  │      for each cell (i,j):
  │          imp = importance[0,0,i,j]
  │          if skip_mode == "threshold":
  │              if imp < 0.5 * threshold → 跳过 (背景)
  │              elif imp < threshold       → 粗瓦片 (borderline)
  │              else                       → 细瓦片 (granularity_hard 决定尺寸)
  │          elif skip_mode == "hard":
  │              if imp < threshold         → 跳过
  │              else                       → 细瓦片
  │          elif skip_mode == "topk":
  │              按重要性排序，保留 top-K%
  │
  ├─ 3. 预算限制: 按 priority 排序，保留前 max_tokens_per_image 个
  │
  └─ 输出: TilePlan
```

**三种跳过模式对比**：
| 模式 | 低于阈值的单元 | 边界单元 | 适用场景 |
|------|-------------|---------|---------|
| `threshold` | 跳过 | 粗瓦片 | **默认推荐**（不会丢失全部信息） |
| `hard` | 跳过 | 跳过 | 激进稀疏 |
| `topk` | 按比例 | 按比例 | 固定计算预算 |

**`plan_quadtree()`**：四叉树递归分解替代方案。
```
全图 → 4 个子区域 → 高密度的子区域再分 4 份 → ... (递归直到 tile_size 达到最小)
```

**`compute_planning_alignment_loss(importance, plan) -> Tensor`**：

这是 **关键的梯度桥接**：
```python
# 1. 从 plan.specs 构建空间覆盖图 (离散决策已 detach)
# 2. 计算 BCE(importance, coverage_map)
# 3. BCE 的梯度通过 importance 流回 Ada-SPM
# 4. 用 autocast(enabled=False) 包裹以避免 AMP NaN
```

**`set_calibrated_costs(flops_per_tile, mem_per_tile)`**：接受 benchmark 的实测数据替代分析估算。

---

### 6.2 `adatile/tokenizer/token_generator.py`

**职责**：根据 `TilePlan` 从原始图像提取瓦片并生成 token 嵌入。

#### 子模块

**`PatchEmbed`**：
```
输入: [N, 3, H_tile, W_tile] (N 个瓦片)
  ↓ Conv3x3 s2 → BN → ReLU
  ↓ Conv3x3 → BN → ReLU
  ↓ Conv3x3 s2 → BN → ReLU
  ↓ AdaptiveAvgPool2d(1)
输出: [N, embed_dim]
```

**`PosEmbed2D`**：
```
输入: 每个瓦片的 (cx, cy) 归一化中心坐标 + scale
  ↓ 32 频段正弦编码 (多频 sin/cos)
  ↓ MLP 投影
输出: [N, embed_dim] 位置编码
```

**`TokenGenerator.forward()`**：
```
输入: 原始图像 [B,3,H,W], List[TileSpec]
  │
  ├─ 1. _extract_tiles: F.affine_grid + F.grid_sample
  │      将每个瓦片坐标映射到 [-1,1] 仿射变换参数
  │      从原始图像采样 (保留原始分辨率细节!)
  │      分块处理: 每次最多 16 个瓦片 (避免 OOM)
  │
  ├─ 2. PatchEmbed: 每个瓦片 → embed_dim 向量
  │
  └─ 3. + PosEmbed2D: 加入空间位置信息
  ↓
输出: (tokens [N, embed_dim], centers [N,2], scale_ids [N])
```

**关键设计**：`F.grid_sample` 从原始分辨率图像采样，而非缩略图。这保证了 backbone 只处理 512² 缩略图，但瓦片保留了原始分辨率的所有细节。

---

### 6.3 `adatile/tokenizer/base.py`

**职责**：组合 `TilePlanner + TokenGenerator` 为完整的 `DynamicTileTokenizer`。

- `DynamicTileTokenizerImpl`：自适应瓦片化 + 全局上下文分支 (可选)
- `UniformTileTokenizer`：固定网格 baseline (消融用)

---

### 6.4 `adatile/tokenizer/global_branch.py`

**职责**：全局缩略图上下文 + 瓦片间融合。

- `GlobalThumbnailBranch`：处理 512² 缩略图获取全局场景嵌入，通过交叉注意力注入瓦片特征
- `TileMerger`：重叠区域软融合、边界抑制、类别感知 NMS

---

### 6.5 `adatile/tokenizer/analysis.py`

**职责**：瓦片化分析工具。

- `TokenBudgetTracker`：预算监控 + 溢出检测
- `render_tile_layout()`：瓦片布局可视化
- `render_scale_histogram()`：瓦片尺寸分布直方图

---

## 第七阶段：Token 路由与稀疏注意力

> ⭐ **核心创新 #3：DTR-v2 Router**

### 7.1 `adatile/routing/router.py`

**职责**：对每个瓦片 token 决定分配到哪个处理级别，并执行对应注意力计算。

#### 完整路由 Pipeline

```
输入: tokens [N_tiles, C], metadata {"importance": [N], "prototypes": {id: [C]}}
  │
  ├─ 1. RoutingHead: MLP(256→128→64→4)
  │      token → 4 类 logits [skip, window, block_sparse, full]
  │      + 可学习 level_bias [4]
  │
  ├─ 2. PrototypeRouter (可选): 如果有 prototypes
  │      sim = cosine(token, prototype)
  │      features = [max_sim, mean_sim, entropy_sim]
  │      MLP(features) → bias [4] 加到 logits
  │      (已知类别的 token 被引导到高处理级别)
  │
  ├─ 3. BudgetController:
  │      可微的 logit 偏置 (不是后验硬分配!)
  │      ├─ skip_bias: 如果 skip 比例 > max_skip_ratio → 降低 skip logit
  │      ├─ full_bias: 如果 full 比例 > max_full_ratio → 降低 full logit
  │      ├─ linear_bias: 如果 linear 比例 < min_linear_ratio → 提高 window logit
  │      └─ 通过 bias_strength 控制约束力度
  │
  ├─ 4. Gumbel-Softmax:
  │      训练时: probs = softmax((logits + gumbel_noise) / temperature)
  │              硬 one_hot = STE(probs)  # Straight-Through Estimator
  │      推理时: argmax (确定性)
  │
  ├─ 5. Token Sparsification:
  │      过滤 assignments == 0 的 token (skip)
  │      保留 assignments ∈ {1,2,3} 的 token
  │
  ├─ 6. MultiLevelAttention:
  │      ├─ Level-1 tokens → LinearAttention (窗口局部)
  │      ├─ Level-2 tokens → LowRankAttention (块稀疏)
  │      └─ Level-3 tokens → FullAttention (FlashAttention-2)
  │
  │      ⭐ routed_tokens = attn_output * routing_weights.unsqueeze(-1)
  │         (乘权重保证梯度通过 STE 反向传播到 RoutingHead!)
  │
  ├─ 7. ConfidenceEstimator: MLP(embed+4→64→1)
  │      预测每个 token 的路由置信度
  │
  └─ 8. Loss: aux_loss = entropy_loss + load_balance_loss
         (鼓励确定分配 + 均匀利用各级别)
```

#### 关键子模块

| 子模块 | 输入 | 输出 | 关键细节 |
|--------|------|------|---------|
| `RoutingHead` | tokens [N,C] | logits [N,4], probs [N,4] | 可学习 `level_bias` |
| `BudgetController` | probs [N,4] | biased_probs, hard_onehot, assignments, weights | **可微偏置**而非后验过滤 |
| `PrototypeRouter` | tokens [N,C], prototypes Dict | bias [N,4] | 余弦相似度 → MLP → bias |
| `ConfidenceEstimator` | concat(tokens, probs) [N,C+4] | confidence [N,1] | 路由不确定度估计 |

#### 4 个路由级别

| Level | 名称 | 注意力类型 | 计算复杂度 | 典型占比 |
|-------|------|----------|-----------|---------|
| 0 | Skip | 无 (丢弃) | 0 | 40-70% |
| 1 | Window | 窗口局部注意力 | O(N*W*d) | 15-30% |
| 2 | Block-Sparse | 块稀疏注意力 | O(N*r*d) | 5-15% |
| 3 | Full | 全注意力 (FlashAttention-2) | O(N²*d) | 5-15% |

#### 消融 Baseline

| 类 | 行为 | 用途 |
|----|------|------|
| `UniformRouter` | 所有 token → 同一级别 (1/2/3) | 证明自适应路由的必要性 |
| `IdentityRouter` | 直通 (assignments=0, weights=1.0) | "无路由"对照组 |

**阅读要点**：`BudgetController` 是最精妙的部分——通过在 softmax 之前偏置 logit 来实现预算控制，保持了可微性。如果使用后验硬分配（如"满了就不分配"），梯度链路就断了。

---

### 7.2 `adatile/routing/attention.py`

**职责**：三种注意力后端的具体实现。

| 后端 | 掩码方式 | SDPA 格式 | 用途 |
|------|---------|----------|------|
| `LinearAttention` | 块对角掩码 `[1,1,N,N]` | 窗口大小 W=32，token 只关注同窗口内 | Level-1 |
| `LowRankAttention` | 块稀疏掩码 | token 关注同块 + 相邻块 | Level-2 |
| `FullAttention` | 无掩码 | 标准 MHA，PyTorch 自动 dispatch 到 FlashAttention-2 | Level-3 |

**`MultiLevelAttention`**：调度器。

```python
def forward(self, tokens, level_assignments):
    # 1. 按 level 分组
    l1_mask = level_assignments == 1
    l2_mask = level_assignments == 2
    l3_mask = level_assignments == 3

    # 2. 各组独立处理
    out = torch.zeros_like(tokens)
    if l1_mask.any():
        out[l1_mask] = self.linear_attn(tokens[l1_mask])
    if l2_mask.any():
        out[l2_mask] = self.lowrank_attn(tokens[l2_mask])
    if l3_mask.any():
        out[l3_mask] = self.full_attn(tokens[l3_mask])

    # 3. AMP 兼容: 转换回原始 dtype
    for idx in range(len(out)):
        out[idx] = out[idx].to(dtype=tokens.dtype)
    return out
```

**AMP 兼容性关键点**：`F.scaled_dot_product_attention` 可能返回 float32（即使输入是 float16），所以需要 `.to(dtype=out.dtype)`。

**`_reshape_for_attention(x)`** 将 `[N, D]` 变为 `[1, H, N, d]`（SDPA 期望的 4D 格式）。

---

## 第八阶段：分割解码器

### 8.1 `adatile/decoder/base.py`

**职责**：将处理后的瓦片 token 解码为实例分割 mask。

#### ⚠️ 核心设计约束

> **永远不在全图分辨率创建 mask！**

全图 mask `[N_inst, H_img, W_img]` 对 4000² 图像即使只有 100 个实例也需要 `100 × 4000 × 4000 × 4 bytes = 6.4 GB`。

**解决方案**：每实例只在所属瓦片的 ROI 内创建 mask，pad 到 `max_crop=256` 统一批处理。

#### 子模块

**`TileProtoModule`**：
```
输入: tile_token [C] + grid 展开
  ↓ Conv stack
输出: proto [32, H_roi, W_roi]  (瓦片内的 mask 原型)
```

**`FastSAMDecoder`** (继承 `SegmentationDecoder`)：

```
forward(tile_features [N_tiles, C], tile_infos, prototypes, skipped_indices):

  for each active tile:
    1. 将 token 展开为空间网格
    2. TileProtoModule → proto [32, H_tile, W_tile]
    3. Detection head → (score [H_tile*W_tile], bbox_offset [4])
    4. 选取 top-K 高置信度候选位置
    5. CoeffHead → coefficients [K, 32]
    6. mask = sigmoid(Σ coeff[k] * proto[k]) → [K, H_tile, W_tile]
    7. 将 mask 放置到全图坐标位置 (仅 ROI 区域, 非全图)

  后处理:
    8. batched_nms(bboxes, scores) → 去重
    9. masks pad 到 [N_kept, max_h, max_w] 其中 max_h ≤ 256
    
输出: SegmentationOutput(masks=[N_inst, max_h, max_w], scores=[N_inst], boxes=[N_inst,4])
```

**内存节省**：`100 × 256 × 256 × 4 bytes = 25 MB` (vs 6.4 GB 全图方案)。

---

## 第九阶段：原型记忆

### 9.1 `adatile/prototype/base.py`

**职责**：小样本分割的类别原型计算和检索。

**`MaskedAveragePrototype`**：

```
forward(support_features [B_s,C,H,W], support_masks [B_s,H,W], class_ids):
  ┌─ 1. 将 support_features 调整到与 mask 相同分辨率
  │     (F.interpolate 上采样)
  │
  ├─ 2. 掩码平均池化:
  │     prototype[c] = Σ(feature * mask) / Σ(mask)
  │     对每个类别 c 独立计算
  │
  └─ 3. 返回 {class_id: prototype [C]} (或 [K,C] 多原型)

retrieve(query_features, prototypes, temperature):
  ┌─ 1. 将 prototypes 堆叠为 [N_classes, C] 矩阵
  │
  ├─ 2. 余弦相似度: cos(query, proto) / temperature
  │
  └─ 3. 返回 [B_q, N_classes, H, W] 相似度图
```

**使用位置**：
- 路由器中 `PrototypeRouter` 使用原型相似度偏置路由 logit
- 解码器中用作原型引导的 mask 解码

---

## 第十阶段：完整 Pipeline 组装

> ⭐ **最重要的文件** — 将所有模块串联为端到端系统。

### 10.1 `adatile/segmentation/base.py`

**职责**：`AdaTileFastSAMPipeline` 实现完整的端到端推理流程，以及 loss 函数。

#### Loss 函数

```python
@LOSS.register()
class DiceLoss:     # Dice = 1 - 2*|A∩B|/(|A|+|B|)
    def forward(pred, target):
        return 1 - (2*intersection+smooth) / (sum+smooth)

@LOSS.register()
class FocalLoss:    # Focal = α*(1-p_t)^γ * BCE
    def forward(pred, target):
        return α*(1-exp(-BCE))^γ * BCE

@LOSS.register()
class SegmentationLoss:  # Composite = 5*Dice + 1*Focal + 1*IoU_MSE
    def forward(pred_masks, target_masks, pred_iou, target_iou):
        return {"loss_mask": total, "loss_dice": ..., "loss_focal": ...}
```

#### `AdaTileFastSAMPipeline` 前向流程

**标准模式 `forward_standard(image)`**：

```python
B, C, H_orig, W_orig = image.shape  # 如 [1, 3, 4000, 4000]

# ═══ 第一步：全局缩略图 ═══
thumbnail = F.interpolate(image, size=(512, 512))
features = self.backbone(thumbnail)  # backbone 只看到 512²!

# ═══ 第二步：Ada-SPM ═══
spm_output = self.sparse_predictor(features)
importance_thumb = spm_output.importance      # [1, 1, 16, 16]
granularity_hard_thumb = spm_output.granularity_hard

# ═══ 第三步：上采样到原图尺度 ═══
importance = F.interpolate(importance_thumb, size=(H_s, W_s))
granularity_hard = F.interpolate(granularity_hard_thumb, size=(H_s, W_s))

# 释放缩略图相关张量
del thumbnail, features, spm_output
torch.cuda.empty_cache()

# ═══ 第四步：从原始图像提取瓦片 ═══
tile_infos, tile_tokens = self.tokenizer(
    image,             # ← 原始分辨率图像
    features=None,
    importance=importance,
    granularity_hard=granularity_hard,
)

# ═══ 第五步：Token 路由 ═══
imp_for_router = self._extract_token_importance(tile_infos, importance, ...)
route_decision = self.router(tile_tokens, {"importance": imp_for_router})

# ═══ 第六步：解码 ═══
skipped_indices = route_decision.skipped_mask.nonzero()
output = self.decoder(
    route_decision.routed_tokens,
    tile_infos,
    image_size=(H_orig, W_orig),
    skipped_indices=skipped_indices,
)

# ═══ 第七步：Loss ═══
planning_loss = self.tokenizer.planner.compute_planning_alignment_loss(
    importance[0], plan
)

return output, {"importance": importance, "planning_alignment_loss": planning_loss, ...}
```

**小样本模式 `forward_fewshot(support, support_masks, query)`**：

```
Support 分支:
  support_thumb → backbone → features
    → PrototypeMemory → {class_id: prototype_vector}

Query 分支:
  query_thumb → backbone → Ada-SPM → importance
    → TilePlanner → TokenGenerator → tokens

联合处理:
  router(tokens, metadata={"importance": ..., "prototypes": {...}})
    → PrototypeRouter 用原型相似度偏置路由

  decoder(routed_tokens, prototypes={...})
    → 原型引导 mask 解码
```

---

## 第十一阶段：模型构建与入口

### 11.1 `adatile/modeling/adatile_fastsam.py`

**职责**：模型构建工厂。

```python
class AdaTileFastSAM(nn.Module):
    def __init__(self, cfg: Config):
        self.pipeline = self._build_pipeline(cfg)
    
    def _build_pipeline(cfg):
        backbone    = build_backbone(cfg.backbone.name, **kwargs)
        sparse      = build_sparse(cfg.sparse.name, **kwargs)
        tokenizer   = build_tokenizer(cfg.tokenizer.name, **kwargs)
        router      = build_router(cfg.router.name, **kwargs)
        decoder     = build_decoder(cfg.decoder.name, **kwargs)
        prototype   = build_prototype(cfg.prototype.name, **kwargs)
        # 传入 AdaTileFastSAMPipeline
        return AdaTileFastSAMPipeline(backbone, sparse, tokenizer,
                                       router, decoder, prototype)

def build_adatile_fastsam(cfg) -> AdaTileFastSAM:
    """主要入口：配置 → 模型"""
```

**阅读要点**：`_build_pipeline` 展示了注册器系统的完整用法——每个模块通过 `build_*(cfg.name, **cfg_kwargs)` 创建，完全由配置驱动。

---

## 第十二阶段：数据集与数据加载

### 12.1 `adatile/datasets/base.py`

**职责**：数据集抽象基类 + Few-Shot Split。

| 类 | 作用 |
|----|------|
| `BaseDataset(Dataset)` | 多边形/RLE → mask 转换、密度图生成、类别分布、图像统计 |
| `FewShotSplit` | 管理固定 split JSON 文件，支持 1/5/10-shot 配置 |

**密度图生成**：GT 实例 → 在 stride 网格中统计实例中心 → `[H/stride, W/stride]` 数组 → 归一化到 [0, 1] → 用作 Ada-SPM 的监督信号。

---

### 12.2 `adatile/datasets/coco.py`

**三个注册数据集**：

| 类 | 数据集 | 特点 |
|----|--------|------|
| `CocoDataset` | COCO | 通用 |
| `ISAIDDataset` | iSAID 航拍 | 高分辨率 + 小目标，懒加载索引 |
| `LoveDADataset` | LoveDA 遥感 | 语义 mask 支持 |

**关键修复**：`__len__` 触发懒加载（访问 `coco_data` 属性触发 COCO JSON 解析），避免 `num_samples=0`。

---

### 12.3 其他数据集文件

| 文件 | 关键内容 |
|------|---------|
| `manager.py` | `CacheManager`：并联瓦片预计算、密度图生成、元数据聚合 |
| `transforms.py` | `build_train_pipeline()`：基于 Albumentations 的数据增强 |
| `collate.py` | `fewshot_collate()`：支持/查询双分支批处理 |
| `samplers/fewshot_sampler.py` | `FewShotEpisodicSampler`：N-way K-shot 场景采样 |
| `stats.py` | `compute_tile_entropy()`, `estimate_sparse_efficiency()` |
| `metadata.py` | `MetadataManager`：JSON 持久化 + 可读摘要 |
| `cache/tile_cache.py` | `TileCache`：多分辨率磁盘缓存 |
| `loaders/dynamic_tile.py` | `DynamicTileDataLoader`：瓦片感知批处理 |
| `loaders/fewshot.py` | `FewShotDataLoader`：场景式小样本加载 |

---

## 第十三阶段：训练引擎

### 13.1 `adatile/engine/trainer.py`

**职责**：完整的训练循环（Detectron2 风格的 Hook 系统）。

#### Trainer 状态管理

```python
class Trainer:
    # 核心状态
    self.global_step: int          # 全局步数
    self.current_epoch: int        # 当前 epoch
    self.max_steps: int            # 最大步数
    self.max_epochs: int           # 最大 epoch
    
    # 训练特性
    self.use_amp: bool             # 是否混合精度
    self.amp_dtype: torch.dtype    # fp16 或 bf16
    self.scaler: GradScaler        # fp16 梯度缩放器
    self.gradient_accumulation_steps: int  # 梯度累积
    
    # 组件
    self.model: nn.Module          # 可能被 DDP 包装
    self.optimizer: Optimizer      # AdamW
    self.hooks: List[HookBase]     # 生命周期回调
```

#### 训练循环伪代码

```python
def train():
    _call_hooks("before_train")
    for epoch in range(max_epochs):
        _call_hooks("before_epoch")
        for batch in train_loader:
            _call_hooks("before_step")
            
            # 前向 + 反向
            batch = _to_device(batch)
            loss_dict = _forward_backward(batch)
            
            # 梯度累积
            if (batch_idx+1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # 更新 meters
            for key, meter in meters:
                meter.update(loss_dict[key])
            
            global_step += 1
            _call_hooks("after_step")
        _call_hooks("after_epoch")
    _call_hooks("after_train")
```

#### `_forward_backward` 细节

```python
def _forward_backward(batch):
    with autocast(dtype=amp_dtype):  # 混合精度上下文
        output, aux = model(images, support_images, support_masks)
        _collect_diagnostics(output, aux)  # 收集诊断数据
        
        if loss_fn:
            loss_dict = loss_fn(output, batch, aux)
        else:
            # 默认: planning_alignment_loss + 正则化
            loss = aux.get("planning_alignment_loss", torch.tensor(0.0))
            loss = loss + output.scores.mean() * 0.001
    
    scaler.scale(loss).backward()  # 缩放反向传播
    # ... reduce across GPUs if distributed
```

#### `_collect_diagnostics` 数据流

```
pipeline aux ─┬→ planner_stats  → TileStats    (瓦片数/跳过率/大小分布)
              ├→ routed_tokens  → TokenStats   (总数/有效/压缩率)
              ├→ routing_weights → RouterStats  (级别分布/平均权重)
              ├→ output.masks   → DecoderStats (实例数/原型形状)
              └→ output.masks   → OOMGuard     (显存检查)
                                      ↓
                                 DiagnosticsHook → CSV + TensorBoard
```

---

### 13.2 `adatile/engine/hooks.py`

**职责**：训练生命周期中的可插拔回调。

| Hook | 触发时机 | 行为 |
|------|---------|------|
| `LRSchedulerHook` | after_step | `scheduler.step()` |
| `LoggingHook` | after_step (每 N 步) | 打印 loss / lr |
| `CheckpointHook` | after_step (每 N 步) | 保存模型 |
| `EvalHook` | after_step (每 N 步) | 运行验证 + 追踪最佳指标 |
| `TensorBoardHook` | after_step | 写入 SummaryWriter |
| `DiagnosticsHook` | before_step / after_step | 集成所有诊断工具 |

**Hook 基类**：
```python
class HookBase(ABC):
    def before_train(self): pass
    def after_train(self): pass
    def before_epoch(self): pass
    def after_epoch(self): pass
    def before_step(self): pass
    def after_step(self): pass
    def before_eval(self): pass
    def after_eval(self): pass
```

---

## 第十四阶段：评测与性能分析

### 14.1 `adatile/evaluation/metrics.py`

| 类 | 用途 | 关键指标 |
|----|------|---------|
| `COCOEvaluator` | 标准 COCO 评测 | bbox AP, mask AP (通过 pycocotools) |
| `FewShotEvaluator` | 小样本评测 | 逐类 mIoU, FB-IoU, 跨场景聚合 |
| `SparseEfficiencyMetrics` | 效率评测 | 跳过比例, token 压缩率, 级别利用率 |

---

### 14.2 `adatile/profiling/`

| 文件 | 关键类 | 用途 |
|------|--------|------|
| `timer.py` | `CUDATimer`, `StageTimer` | CUDA event 微秒级 GPU 计时 |
| `stats.py` | `BenchmarkResult`, `CompareResult` | 结构化性能数据，支持 `speedup_vs()` |
| `pipeline_profiler.py` | `PipelineProfiler` | 使用 `torch.profiler` + `fvcore` 的内核级性能分析 |
| `export.py` | `export_csv/json/chrome_trace()`, `plot_results()` | 导出和可视化 |

---

## 第十五阶段：工具与配置

### 15.1 入口脚本

| 文件 | 用途 |
|------|------|
| `tools/train.py` | **训练入口**。解析 CLI → 构建模型 → 创建 Trainer → 注册 Hooks → `trainer.train()` |
| `tools/eval.py` | 评测入口 |
| `benchmark.py` | **性能基准入口**。4 种配置对比、分辨率扫描、FPS 测量 |
| `tools/setup_datasets.py` | 数据集目录结构创建 |
| `tools/generate_fewshot_splits.py` | Few-shot split 生成器 |
| `tools/experiments/baselines.py` | 第三方方法对比 (FastSAM/SAM/Mask2Former) |
| `tools/experiments/run_fewshot.py` | 多种子训练 + Wilcoxon 检验 + Bootstrap CI |

### 15.2 配置文件

| 文件 | 场景 |
|------|------|
| `configs/default.py` | COCO 基础配置 (1024², bs=8) |
| `configs/isaid.py` | iSAID 航拍 (2048², bs=4) |
| `configs/dota.py` | DOTA-v2.0 (2048², 18 类) |
| `configs/high_res.py` | 分辨率扫描 (1024/2048/4096) |
| `configs/fewshot/one_shot.py` | 1-shot (threshold=0.15, 128 tokens) |
| `configs/fewshot/five_shot.py` | 5-shot |
| `configs/fewshot/ten_shot.py` | 10-shot |

---

## 第十六阶段：测试代码

### 16.1 测试文件

| 文件 | 测试内容 | 关键验证点 |
|------|---------|-----------|
| `tests/test_sparse_prediction.py` | `SparsePrediction` 构造、验证、梯度 | Shape/dtype 契约、Ada-SPM 梯度流 |
| `tests/test_routing.py` | Router 输出格式、DTRv2 梯度 | STE 梯度通过 `routing_weights` |
| `tests/test_tile_planner.py` | 三种 skip 模式、四叉树、预算限制 | Skip 语义正确性 |
| `tests/test_e2e_pipeline.py` | 端到端 forward、梯度完整性 | 全梯度链路检查 |

**阅读要点**：测试是理解模块接口的最佳文档。`test_e2e_pipeline.py` 验证所有组件在组装后的兼容性。

---

## 全局数据流全景图

```
                         ┌──────────────────┐
                         │  原始高分辨率图像   │
                         │ [B, 3, 4000, 4000]│
                         └────────┬─────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │ 缩略图 (512²)│              │ 原始分辨率
                    ▼             │              │
          ┌─────────────┐        │              │
          │  timm        │        │              │
          │  Backbone    │        │              │
          └──────┬──────┘        │              │
                 │               │              │
          ┌──────▼──────┐        │              │
          │  FPN Fusion  │        │              │
          └──────┬──────┘        │              │
                 │               │              │
          ┌──────▼──────────────────────┐       │
          │      Ada-SPM 🌟             │       │
          │  importance + density       │       │
          │  + granularity              │       │
          └──────┬──────────────────────┘       │
                 │                              │
                 │ 上采样到原图尺度              │
                 ▼                              │
          ┌─────────────────────────────────────▼──┐
          │     Dynamic Tile Planner 🌟             │
          │  TilePlan: specs, skip_ratio, stats     │
          └─────────────────────────────────────┬──┘
                                                 │
          ┌──────────────────────────────────────▼──┐
          │     Token Generator + PosEmbed2D        │
          │  F.grid_sample → PatchEmbed → +Pos      │
          └──────────────────────────────────────┬──┘
                                                 │
                              tokens [N, C]       │
                                                 │
          ┌──────────────────────────────────────▼──┐
          │        DTR-v2 Router 🌟                 │
          │  RoutingHead → Budget → Gumbel → SDPA  │
          │  routing_weights * attn_output          │
          └──────────────────────────────────────┬──┘
                                                 │
                      routed_tokens [M, C]        │
                                                 │
          ┌──────────────────────────────────────▼──┐
          │     FastSAM Decoder                     │
          │  Per-tile: proto → coeff → crop mask    │
          │  batched_nms → SegmentationOutput       │
          └──────────────────────────────────────┬──┘
                                                 │
                              ┌──────────────────▼──┐
                              │  SegmentationOutput  │
                              │  masks [N, max_h,    │
                              │          max_w]       │
                              │  scores [N]           │
                              └──────────────────────┘
```

---

## 模块依赖关系图

```
adatile/core/base.py ──────────── 根 (数据结构 + ABC)
  ↑
  ├── adatile/sparse/  ──────── Ada-SPM (依赖 core)
  │     ├── fpn_fusion.py      自实现 FPN
  │     └── ada_spm.py         核心稀疏预测
  │
  ├── adatile/tokenizer/ ────── 瓦片化 (依赖 core)
  │     ├── tile_planner.py    瓦片规划
  │     └── token_generator.py 特征提取
  │
  ├── adatile/routing/ ──────── 路由 (依赖 core)
  │     ├── router.py          DTR-v2
  │     └── attention.py       注意力后端
  │
  ├── adatile/decoder/ ──────── 解码器 (依赖 core)
  ├── adatile/prototype/ ────── 原型 (依赖 core)
  │
  └── adatile/segmentation/ ─── 组装 (依赖所有以上)
        │
        └── adatile/modeling/ ── 构建入口
              │
              └── tools/train.py ── 训练入口

adatile/registry/ ───────────── 横向 (注册所有模块)
adatile/config/   ───────────── 横向 (配置所有模块)
adatile/engine/   ───────────── 训练 (依赖 segmentation + config)
adatile/datasets/ ───────────── 数据 (依赖 core)
adatile/evaluation/ ─────────── 评测 (依赖 core)
adatile/profiling/ ──────────── 分析 (独立)
adatile/utils/    ───────────── 工具 (独立)
```

---

## 关键梯度流路径

### 梯度路径 1：Ada-SPM（主要路径）

```
planning_alignment_loss
  = BCE(importance, planner_coverage.detach())
       │
       ▼ 梯度流入 importance
  F.interpolate (双线性, 可微)
       │
       ▼ 梯度流入 importance_thumb
  _compute_importance(density, granularity)
       │
       ├──→ DensityHead 参数  (grad 流经 sigmoid)
       └──→ GranularityHead 参数 (grad 流经 gumbel_softmax STE)
```

### 梯度路径 2：DTR-v2 Router

```
decoder loss
  │
  ▼ routed_tokens = attn_output * routing_weights.unsqueeze(-1)
  │                  ↑ 乘权重使梯度流经 STE
  ▼ routing_weights
  │   = gumbel_softmax(logits, hard=True) 中的 "soft" 部分
  ▼ logits
  │   = RoutingHead(tokens) + BudgetController.bias + PrototypeRouter.bias
  ▼
  ├──→ RoutingHead MLP 参数
  ├──→ BudgetController bias_strength (弱)
  └──→ PrototypeRouter MLP 参数
```

### 梯度路径 3：解码器

```
Dice + Focal loss (mask 监督)
  │
  ▼ segmentation_output.masks
  │   = sigmoid(Σ coeff[k] * proto[k])
  ▼
  ├──→ CoeffHead 参数
  ├──→ TileProtoModule 参数
  └──→ routed_tokens (回传到路由器, 见路径 2)
```

### 梯度阻断处

| 位置 | 操作 | 原因 |
|------|------|------|
| `TilePlanner.plan()` | 离散瓦片分配 `.detach()` | 规划是离散的、不可微 |
| `Gumbel-Softmax argmax` | 硬 one-hot | 前向硬分配、反向 STE 用软概率 |
| `TokenSparsifier` | `assignments != 0` 过滤 | 离散 token 保留/丢弃 |

**设计原则**：所有「不可微」的离散决策处都有一个平行的「可微」路径（planning_alignment_loss 桥接 Planner, STE 桥接 Router）。

---

## 建议的阅读顺序（按文件）

### 第一天：理解框架

1. `adatile/core/base.py` — 全部（数据结构 + ABC）
2. `adatile/registry/registry.py` — 注册器模式
3. `adatile/config/config.py` — 配置系统
4. `adatile/utils/` 目录 — 先 `logging.py`，再其他

### 第二天：核心创新

5. `adatile/sparse/ada_spm.py` — Ada-SPM 完整实现
6. `adatile/tokenizer/tile_planner.py` — 瓦片规划
7. `adatile/tokenizer/token_generator.py` — Token 生成
8. `adatile/routing/router.py` — DTR-v2 路由
9. `adatile/routing/attention.py` — 注意力后端

### 第三天：Pipeline 组装与训练

10. `adatile/decoder/base.py` — FastSAM 解码器
11. `adatile/prototype/base.py` — 原型记忆
12. `adatile/segmentation/base.py` — **完整 Pipeline**
13. `adatile/modeling/adatile_fastsam.py` — 模型构建
14. `adatile/engine/trainer.py` + `hooks.py` — 训练循环

### 第四天：数据、评测、工具

15. `adatile/datasets/base.py` + `coco.py` — 数据集
16. `adatile/evaluation/metrics.py` — 评测
17. `tools/train.py` — 训练入口
18. `benchmark.py` — 性能基准
19. `configs/` — 配置文件

### 第五天：测试与调试

20. `tests/test_e2e_pipeline.py` — 端到端测试理解
21. `tests/test_routing.py` — 路由测试
22. `adatile/profiling/` — 性能分析工具
23. `adatile/visualization/` — 可视化工具

---

## 代码量统计

| 模块 | 文件数 | 核心代码行数 | 复杂度 |
|------|--------|------------|--------|
| `core/` | 1 | ~480 | ⭐⭐ |
| `config/` | 1 | ~250 | ⭐⭐ |
| `sparse/` | 3 | ~550 | ⭐⭐⭐⭐ (Ada-SPM) |
| `tokenizer/` | 5 | ~1200 | ⭐⭐⭐⭐⭐ (最复杂) |
| `routing/` | 3 | ~650 | ⭐⭐⭐⭐ (DTR-v2) |
| `decoder/` | 1 | ~350 | ⭐⭐⭐ |
| `prototype/` | 1 | ~80 | ⭐ |
| `segmentation/` | 1 | ~450 | ⭐⭐⭐⭐⭐ (核心组装) |
| `engine/` | 2 | ~450 | ⭐⭐⭐ |
| `datasets/` | 9 | ~800 | ⭐⭐⭐ |
| 其他 | 15+ | ~1500 | ⭐⭐ |

---

> **提示**：配合 `pytest tests/ -v` 运行测试可以验证你对每个模块的理解。每个测试文件都是以「这个模块应该做什么」的视角编写的。
