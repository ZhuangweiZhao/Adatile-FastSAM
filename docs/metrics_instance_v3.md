# 实例分割评估协议 V3 | Instance Segmentation Evaluation Protocol V3

> 定义 `tools/eval/evaluate_instance.py` 使用的全部度量与匹配规则。
> Defines every metric and matching rule used by `tools/eval/evaluate_instance.py`.
>
> 背景 | Background: 历史脚本 `eval_fewshot_allclass.py` 采用**语义**评估 (把一类的所有实例合并成
> union mask 再算 IoU), 且 zero-shot 用 **oracle** 选 mask, 与实例级标准**不可比**、数值偏高
> (见 `docs/protocol_reconciliation.md`, zero-shot 下限 0.238 vs 实例级 0.015)。
> 本协议是严格**实例级**、**非 oracle** 的重建。

---

## 0. 为什么禁止 union-mask IoU | Why union-mask IoU is forbidden

实例分割的对象是**单个实例**。若把一类的 N 个实例合并成一张前景图再与 GT 前景图算 IoU,
则:①无法区分"检出 3 个中的 1 个"与"检出全部 3 个";②相邻/粘连实例被当成一个;
③与 COCO 标准不可比。因此本协议**任何环节都不做 union 合并**,每个实例独立参与。

Instance segmentation is about *individual* instances. Merging a class's N instances into one
foreground map and IoU-ing against the GT foreground cannot distinguish "1 of 3 detected" from
"all 3 detected", collapses touching instances, and is incomparable to COCO. So this protocol
**never merges**; every instance participates independently.

---

## 1. 模型输出 → 实例 | Model output → instances

模型的 decoder 输出的是**某类别的语义前景概率图** (该类所有实例合并), 不是实例。
转换方式 (用户确认): **连通域分解 (connected components)**。

The decoder outputs a **class-conditioned semantic foreground probability map** (all instances of
the class merged), not instances. Conversion (user-approved): **connected components**.

- 阈值化 | Threshold: `binary = prob_map > score_thr` (默认 `score_thr=0.5`)。
- 连通域 | CC: `connected_components_to_instances(binary, min_area)` (默认 `min_area=16`)。
  每个连通块 = 一个预测实例 | each component = one predicted instance.
- 置信度 | Score: 连通块内前景概率均值(**当前默认;属模型侧,可更换为 max / mask-score**)|
  mean FG probability inside the component (current default; **model-side, not frozen**).
- **纯模型输出**: 无 oracle, 无 FastSAM proposal, 不看 GT。
- **边界 | Boundary**: 实例化与打分属**模型侧**,非协议冻结面;协议只固定"如何度量",不固定"模型如何产出预测"
  (见 `EVALUATION_PROTOCOL_V3.md` §4 / §12.1)。

> 局限 | Limitation: 同类相邻/粘连实例会并成一个连通块 (欠分割)。这是语义 decoder 的真实能力
> 上界, 诚实反映, 不做修饰。Touching same-class instances merge into one component
> (under-segmentation) — an honest reflection of the semantic decoder's ceiling.

---

## 2. 评估单元 | Evaluation unit

**按 tile (896²)**。GT 的 COCO JSON (`instances_{split}.json`) 本身以 tile 为 image
(`image_id = tile`, 标注为 tile-local polygon), 故无需坐标重映射, COCO 原生。
Per-tile: the COCO GT is organized by tile, so no coordinate remapping is needed.

评估的图像集 = 各类采样 query tile 的并集; AP 分母限定为该集合 (`image_ids` 参数)。
Evaluated image set = union of per-class sampled query tiles; AP denominator restricted to it.

---

## 3. COCO AP 家族 | COCO AP family (official pycocotools)

直接调用官方 `pycocotools.cocoeval.COCOeval(iouType="segm")` (经 `COCOInstanceEvaluator` 封装),
**不重复实现**。GT 逐实例来自 `COCO.annToMask` (polygon → 掩码)。
Uses official pycocotools via `COCOInstanceEvaluator`; **no re-implementation**.

报告 | Reported: `AP` (IoU=0.50:0.95), `AP50`, `AP75`, `AP_small`, `AP_medium`, `AP_large`,
以及 `AR_max1/10/100`。COCO 的匹配由 COCOeval 内部完成 (按 score 排序、每个 GT 至多匹配一个预测、
按 IoU 阈值扫描) — 这是一对一匹配, 满足需求 #3。

- **class-aware AP** (主指标): 每类预测只与该类 GT 竞争。**这是微调模型的头条指标。**
- **class-agnostic AP** (`AP_class_agnostic`): 所有类别重映射为 1 类, 评估"分不分得出物体"。
  > 注意 | Caveat: 若 decoder 的 prototype 通路失效 (见 [[prototype-functional-death]]),
  > 各类前景图近乎相同 → 每个前景区域产生 15× 重复预测 → class-agnostic AP 被 FP 拉低,
  > 不代表分割质量。此时以 class-aware AP 为准。

---

## 4. Instance mIoU (需求 #5 定义) | Instance mIoU (requirement #5)

对**每个 GT 实例**, 在所有预测中取最大 IoU; 再对所有 GT 求平均。绝不 union。
For **each GT instance**, take the max IoU over all predictions; average over all GT. Never union.

```
InstanceMIoU = mean_over_GT( max_over_Pred IoU(gt_i, pred_j) )
```

实现 | Impl: `adatile/metrics/instance_match.py :: instance_miou`。

- `instance_miou_overall`: 所有类的 GT 实例汇到一起求均值 (被高频类如 small_vehicle 主导)。
  Pooled over all GT instances (dominated by high-frequency classes).
- `instance_miou_class_mean`: 先算每类均值再对类求均值 (类均衡)。
  Per-class mean, then averaged over classes (class-balanced).
- 有 GT 无预测 → 该 GT 记 0; 无 GT 的图 → 跳过 (记 NaN, 不计入)。
  GT but no prediction → 0 for that GT; image with no GT → skipped.

> 与匹配的区别 | Difference from matching: 此定义**不**强制一对一 (两个 GT 可命中同一预测),
> 是"分割覆盖度"度量; TP/FP/FN 才用一对一贪心匹配 (§5)。刻意分离。
> This definition does NOT enforce one-to-one (a "coverage" measure); TP/FP/FN use the
> one-to-one greedy match (§5). Intentionally separate.

---

## 5. 贪心一对一匹配 → TP/FP/FN | Greedy one-to-one matching → TP/FP/FN

用于每类调试计数 (需求 #6)。实现 | Impl: `instance_match.py :: greedy_match`。

规则 | Rule (默认 `iou_thr=0.5`):
1. 预测按 score 降序 | predictions sorted by descending score;
2. 每个预测在**尚未被认领**的 GT 中选 IoU 最大且 ≥ 阈值者认领 | each claims the highest-IoU
   *still-unmatched* GT with IoU ≥ thr;
3. **一个预测 ↔ 至多一个 GT, 一个 GT 至多被认领一次** (需求 #3)。

定义 | Definitions:
- **TP** = 成功认领 GT 的预测数 | predictions that claimed a GT.
- **FP** = 未认领任何 GT 的预测数 | predictions that claimed nothing (`n_pred − TP`)。
- **FN** = 未被认领的 GT 数 | GT left unclaimed (`n_gt − TP`)。
- 不变式 | Invariants: `TP + FN == n_gt`, `TP + FP == n_pred` (代码与单测均校验)。

---

## 6. Zero-shot 基线 (严格非 oracle) | Zero-shot baseline (strictly non-oracle)

需求 #7: **禁止 oracle, 禁止用 GT IoU 选 mask, 只用模型默认输出。**
Requirement #7: no oracle, no GT-IoU mask selection, model default output only.

实现 | Impl:
- 用**原始权重的 FastSAM** (与微调 backbone 分离, 见 `load_clean_fastsam`) —— 微调 backbone 会
  破坏 FastSAM 自带分割头, 那不是真正的"未适配"基线。
  Uses a **clean-weight FastSAM** separate from the fine-tuned backbone.
- 调用 `model(img, retina_masks=True)` 的默认输出 (不给 GT bbox 提示, 不设 oracle 阈值),
  取 `results[0].masks.data` + `boxes.conf` 作为类无关实例。
  Default output only; masks + confidences as class-agnostic instances.
- 仅报 **class-agnostic AP** (FastSAM 不产生类别标签)。
  Class-agnostic AP only (FastSAM has no class labels).

> 健全性 | Sanity: zero-shot AP 应落在论文所述 ~0.015 量级 (SA-1B→航拍 27× 领域差距),
> 与旧 oracle 协议的 0.238 形成对照。dry-run 实测 ZS AP≈0.019, 符合。

---

## 7. 输出 schema | Output schema (`instance_metrics.json`)

```
{ protocol:"instance_v3", checkpoint, decoder, prototype_source, k_shot, seed,
  eval_split, n_query_tiles, iou_thr, score_thr, min_area, class_conditioned,
  finetuned: {
    AP, AP50, AP75, AP_small, AP_medium, AP_large, AR_max1/10/100, n_predictions,
    AP_class_agnostic, AP50_class_agnostic,
    instance_miou_overall, instance_miou_class_mean,
    per_class: { "<cid>": {name, n_gt, n_pred, tp, fp, fn, AP50, instance_miou} }
  },
  zero_shot: { AP_class_agnostic, AP50_class_agnostic, AP75_class_agnostic,
               AR_max100, n_predictions, note }
}
```

---

## 8. 复现 | Reproducibility

- `set_seed(args.seed)` 最先调用。
- **确定性哈希 (FAIL-7a 修复)**: query tile 选择用 `_det_hash`(md5),**禁用** Python 内置 `hash()`
  (后者对 str 带每进程随机盐 PYTHONHASHSEED)。同一 `source_name` 在任意 OS / Python 版本恒得同值。
- **评估清单 (FAIL-7b 修复)**: `--manifest <path>`(默认 `<data_root>/evaluation_manifest_<split>.json`)。
  存在则读取、不存在则**首次生成并冻结**;之后所有实验/所有 seed **复用同一固定评估集**,不再随机采样。
  清单存图像文件名列表(`["P0003_t0001.png", ...]`)。**建议首次用较大 `--per-class`(如 9999)生成**,
  得到全面代表性的冻结集。
- Support/Prototype 仍逐 seed 计算(多 seed 实验的正确行为),但 support 源图始终排除 manifest query 源图
  → 0% 场景重叠不变。
- 启动打印 `Evaluation Protocol V3` 摘要(Query/GT Images、Manifest、Random Sampling=Disabled、
  Evaluation=Deterministic)。
- 观测标量经 `adatile.logging` 落盘(`eval.jsonl`);完整指标写 `instance_metrics.json`(含 `manifest` 路径)。

---

## 9. 相关 | See also

- 代码 | Code: `tools/eval/evaluate_instance.py`, `adatile/metrics/instance_match.py`,
  `adatile/metrics/coco_eval.py`。
- 协议对账 | Reconciliation: `docs/protocol_reconciliation.md` (为何旧数字不可比)。
- 单测 | Tests: `tests/test_instance_match.py` (14 项)。
