# eval_c04_full_fewshot.py 代码讲解

> **文件路径:** `tools/instance/eval_c04_full_fewshot.py`  
> **实验名称:** C-04 Full-Category Few-Shot Instance Segmentation on iSAID  
> **核心问题:** FastSAM + Decoder 的 Few-Shot 上限到底在哪里？

---

## 一、整体思路

```
支持图像 (Support)                    查询图像 (Query)
 K张有标注的小图                        1张没见过的图
      |                                      |
      v                                      v
  FastSAM (冻结)                         FastSAM (冻结)
      |                                      |
      v P4特征                              v P4特征
  按mask取平均                    +------------------+
      |                          |   解码器 Decoder  |
      v  prototype (1个向量)      |  (可训练的轻量网络) |
  +----------+                   |                  |
  | 类别原型  | ---------------> |  proto + query   |
  | (1280维) |    输入条件        |  -> 预测mask      |
  +----------+                   +------------------+
                                      |
                                      v
                                 预测的二值mask
                                     vs
                                 真实mask (GT)
                                      |
                                      v
                                  计算loss
                                 反向传播只更新Decoder
```

**核心思想：** 类别的"原型向量"（prototype）编码了"这个类长什么样"。Decoder 学习如何根据原型在查询图上找到同类目标。

---

## 二、代码结构（从顶到底）

### 第 1 层：四个 Decoder 变体（第 105-256 行）

这是四种不同的解码器，区别在于**如何用 prototype 调制 query 特征**：

| Decoder | 参数量 | 核心机制 |
|---------|--------|----------|
| `ProtoRefineDecoder` (baseline) | ~1万 | proto 与 query 做余弦相似度 -> 一个小CNN精修 -> mask |
| `FiLMFewShotDecoder` (film) | ~110万 | 用 proto 生成缩放/偏移参数(gamma,beta) 调制 query 特征 -> 逐级上采样 -> mask |
| `CATFewShotDecoder` (crossattn) | ~110万 | 从 C-03 导入：proto 与 query 做交叉注意力 -> 逐级上采样 -> mask |
| `ContrastiveProtoDecoder` (contrastive) | ~130万 | proto 和 query 各自投影到对比学习空间 -> 余弦相似度 -> CNN精修 -> mask |

**类比理解：**

- **baseline** = 拿一张类别照片(query)，和参考照(proto)逐像素比相似度
- **film** = 参考照告诉你"应该偏红一点、亮一点"，按此调整查询照片
- **crossattn** = 参考照的每个细节都和查询照的每个位置交互匹配
- **contrastive** = 双方先转换到同一个"语义空间"再比较

### 第 2 层：Episode 训练（第 314-368 行）

```python
def train_episode(decoder, backbone, support_idxs, query_idx, ...)
```

一个 **episode** = 一次完整的前向+反向传播：

```
1. 随机选一个类别（如 "small_vehicle"）
2. 从训练集抽 K 张该类图片作为 Support
3. 从验证集抽 1 张该类图片作为 Query
4. Support 图 -> FastSAM -> P4特征 -> 按GT mask取平均 -> prototype向量
5. Query 图 -> FastSAM -> P4特征 + prototype -> Decoder -> 预测mask
6. 计算 Focal + Dice Loss -> 反向传播
```

**关键设计：** FastSAM 在 `torch.no_grad()` 中，梯度只流过 Decoder。

### 第 3 层：主训练循环（第 514-684 行）

```python
def train_and_evaluate(decoder, backbone, ...)
```

```
每个 epoch:
  |-- 采样 200 个 episode 训练（类别按稀有度加权采样）
  |-- 每类 30 个 episode 做验证
  |-- 保存每类最佳 checkpoint
  |-- 记录 metrics 到 JSONL 文件

训练结束后:
  |-- 加载全局最佳 checkpoint
  |-- 200 个 episode 全量评估
```

### 第 4 层：评估函数（第 414-508 行）

```python
def evaluate_full(decoder, backbone, ...)
```

```
200 个随机 episode:
  |-- 每个episode:
      |-- 随机选类 -> 抽support -> 算proto -> query预测 -> 算IoU
      |-- 按类别 + 按分组(vehicle/infra/object) 汇总
```

### 第 5 层：Main 入口（第 739-942 行）

```
解析参数
  |
  v
加载 iSAID 数据集 (train/val)
  |
  v
加载冻结的 FastSAM backbone
  |
  v
对所有 (decoder类型 x K-shot) 组合:
  |-- 构建 decoder
  |-- 训练 (30 epochs x 200 episodes)
  |-- 评估
  |
  v
打印对比表格 + 保存 JSON 结果
```

---

## 三、关键技术细节

### 3.1 类别不平衡处理（第 542-557 行）

```python
class_weights[c] = total_images / count    # 图片少的类权重大
class_probs[c] = weight / weight_sum       # 归一化为采样概率
```

iSAID 中 `road` 类有几千张图，`helicopter` 只有十几张。不加权训练的话模型会忽略稀有类。

### 3.2 损失函数（第 297-308 行）

```python
loss = 0.5 x Focal Loss + 0.5 x Dice Loss
```

- **Focal Loss (gamma=5):** 自动降低简单样本的权重，让模型专注于难样本
- **Dice Loss:** 直接优化 IoU（交并比），缓解类别不平衡

### 3.3 Warmup + Cosine 学习率（第 528-537 行）

```
lr: 0.1xlr -> lr -> 0.01xlr
    |-- warmup --|---- cosine decay ----|
    前3个epoch              后27个epoch
```

前 3 个 epoch 逐步升温，避免初期梯度爆炸；之后余弦下降，平稳收敛。

### 3.4 逐类最佳检查点（第 629-633 行）

```python
for cls_id, val_iou in per_cls_val.items():
    if val_iou > best_per_cls_iou[cls_id]:
        best_per_cls_state[cls_id] = decoder.state_dict()  # 保存每类最优
```

每个类别独立追踪最佳 val IoU 并保存对应的 decoder 权重。最终可以用**不同 epoch** 的 decoder 来做每类的推理——这对稀有类特别有用。

### 3.5 SES 指标（第 895-905 行）

```python
ses = miou_1shot / miou_5shot
```

**Shot Efficiency Score**：1-shot 能保留 5-shot 性能的百分比。越接近 1.0 说明模型越样本高效。

---

## 四、数据流总结

```
输入: data/iSAID_processed/
  |-- train/  (训练集图像 + 实例标注)
  |-- val/    (验证集图像 + 实例标注)
       |
       v ISAIDInstanceDataset (从 eval_c02a 复用)
       |   .load_image(idx)        -> 3x1024x1024 tensor
       |   .render_class_mask(idx, class_id) -> HxW binary mask
       |   .class_to_images(class_id) -> 包含该类所有图片的索引列表
       |
       v
  实验控制: parse_args() -> shots, decoder_types, epochs...
       |
       v
  训练/评估 -> 输出到 runs/c04_full_fewshot/
       |-- c04.jsonl               (实验配置日志)
       |-- decoder_baseline_1shot_best.pt          (全局最佳权重)
       |-- decoder_baseline_1shot_best_c1_small_vehicle.pt  (逐类最佳)
       |-- decoder_baseline_1shot_metrics.jsonl    (逐epoch指标)
       |-- c04_results.json        (最终汇总结果)
```

---

## 五、一句话总结

> 冻结 FastSAM 提取特征，用少量标注样本做一个**类别原型向量**，训练一个**轻量解码器**学会根据原型在新图上找到同类目标。四种解码器变体 (baseline / FiLM / CrossAttn / Contrastive) 在 15 类 iSAID 上对比 1/3/5-shot 性能上限。

---

## 六、已知问题（第二轮评审）

| 级别 | 问题 | 位置 |
|------|------|------|
| (RED) Critical | rng 在 training/validation 之间共享状态 -> 实验可复现性受损 | L582 |
| (YELLOW) Medium | `ProtoRefineDecoder` / `NonParametricMatcher` 中 prototype 未做 L2 normalization，不是真正的余弦相似度 | L127-128, L697-698 |
| (YELLOW) Medium | `FiLMFewShotDecoder` 空 prototype 时静默跳过 FiLM 调制 | L181 |
| (GREEN) Minor | `__init__` 中 `print()` 未改为 logger | L124, L175, L240 |
| (GREEN) Minor | log 消息中 `\\n` 字面量意图不明确 | L521-523, L818, L858 |
