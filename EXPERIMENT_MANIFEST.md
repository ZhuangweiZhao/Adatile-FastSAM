# EXPERIMENT_MANIFEST.md — Paper Experiment Registry | 论文实验注册清单

> **状态 | Status: 唯一权威追溯入口 (single source of truth).**
> 本文件把**每个论文实验编号**绑定到:Protocol · Evaluation Manifest · Seed · Checkpoint · Commit。
> 目的:杜绝"**同一实验编号,先后对应不同 Protocol**"的事故(见 [`docs/protocol_reconciliation.md`])。
>
> Binds every paper experiment ID to its protocol, manifest, seeds, checkpoint, and commit — so a
> Table row can always be traced back to exactly one reproducible run.

---

## 0. 铁律 | Registration Rules

- **R1 · 编号不可变,且绑定唯一 Protocol。** 一个 ID 只对应一套 Protocol。若用不同 Protocol 重评,
  **必须换新后缀**(例:`A-8` 历史口径 → V3 重评记为 **`A-8/V3`**),**绝不复用旧编号覆盖**。
  An experiment ID is immutable and bound to exactly one protocol; re-evaluation under a different
  protocol gets a new suffix.
- **R2 · 进论文表 = 行内必须齐全。** 任一数字进入论文/报告表格,其注册行**必须**同时给出
  `Protocol=V3` + 冻结 `Manifest` + `Checkpoint` + `Commit`。缺一不可上报。
  No number enters a paper table unless its row shows Protocol=V3 + a frozen Manifest + Checkpoint + Commit.
- **R3 · 历史口径结果不得上报。** 历史协议(oracle / union-mask,见 §Legacy)数字仅供内部参考,
  一律标注 `NOT-for-paper`,进论文前必须在 V3 下重跑。
- **R4 · 每次登记引用 Commit。** 用产生该结果时的 `git rev-parse --short HEAD`。

---

## 1. 固定评估常量 | Fixed Evaluation Constants (V3)

| 项 | 值 |
|---|---|
| **Protocol** | Evaluation Protocol V3 (frozen) — [`EVALUATION_PROTOCOL_V3.md`](EVALUATION_PROTOCOL_V3.md) |
| **Evaluator (唯一入口)** | `tools/eval/evaluate_instance.py` |
| **Metrics 库** | `adatile/metrics/{instance_match,coco_eval}.py` |
| **Evaluation Manifest** | `data/iSAID_instance_fewshot/evaluation_manifest_val.json`(首次生成即冻结,之后只读复用) |
| **Seeds (标准三种)** | `42`, `123`, `456` |
| **上报指标** | `AP / AP50 / AP75 / AP_small / AP_medium / AP_large`;`Instance mIoU (overall + class_mean)`;per-class `TP/FP/FN` |
| **结果产物** | 每次运行写 `instance_metrics.json`(含 `manifest` 路径 + `seed` + `checkpoint`) |
| **守卫** | `tests/test_protocol_frozen.py`(冻结)+ `tests/test_protocol_audit.py`(审计)+ `.githooks/pre-commit` |

> Manifest 建议:首次用大 `--per-class`(如 9999)生成权威冻结集,之后全实验/全 seed 复用同一份。

---

## 2. 论文实验注册表 | Paper Experiment Registry (V3)

> **当前状态**:Evaluation Protocol V3 于 2026-07-12 冻结。除下方"验证 dry-run"外,**尚无**任何实验在
> 冻结 Manifest 下完成 V3 上报评估 → 各行 `V3 Status = ⏳ pending`。**先跑 §4 步骤,再回填本表。**

图例 | Legend: `✅ done` = 已在冻结 Manifest 下产出 `instance_metrics.json`;`⏳ pending` = 待 V3 重评;`—` = 不适用。

### 2.1 Phase 1–3 (Zero-shot → Few-shot)

| ID | Config | Decoder / Proto | K | Seeds | Manifest | Checkpoint | Commit | V3 Status |
|---|---|---|---|---|---|---|---|---|
| V3-01 | Zero-shot baseline (clean FastSAM) | — | 0 | 42 | `evaluation_manifest_val.json` | (原始权重) | `TBD` | ⏳ pending |
| V3-05 | Base pre-train (10 类) | adaptive / p4 | — | 42 | `…val.json` | `TBD` | `TBD` | ⏳ pending |
| V3-06 | K-shot scaling | adaptive / p4 | 1/3/5/10 | 42,123,456 | `…val.json` | `TBD` | `TBD` | ⏳ pending |
| V3-07 | Decoder ablation (ProtoOnly vs +Refine) | protoonly · adaptive | 5 | 42,123,456 | `…val.json` | `TBD` | `TBD` | ⏳ pending |
| V3-08 | Per-class breakdown | adaptive / p4 | 5 | 42 | `…val.json` | `TBD` | `TBD` | ⏳ pending |

### 2.2 Phase 4–5 (Sparse Routing · Cross-Dataset)

| ID | Config | Note | Seeds | Manifest | Checkpoint | Commit | V3 Status |
|---|---|---|---|---|---|---|---|
| V3-09 | SPM tile routing @ full-image | efficiency | 42 | `…val.json` | `TBD` | `TBD` | ⏳ pending |
| V3-10 | Top-K% sweep (FLOPs vs AP Pareto) | efficiency | 42 | `…val.json` | `TBD` | `TBD` | ⏳ pending |
| V3-11 | FPS + GPU Memory | efficiency | 42 | — | `TBD` | `TBD` | ⏳ pending |
| V3-12 | NWPU-VHR-10 few-shot transfer | cross-dataset | 42,123,456 | `evaluation_manifest_nwpu.json` | `TBD` | `TBD` | ⏳ pending |

---

## 3. 验证 dry-run (非注册,不得上报) | Validation dry-runs (NOT registry rows)

仅用于确认 V3 评估器行为正确,**未使用冻结 Manifest,不进任何论文表**。

| 时间 | Checkpoint | K | FT AP | Instance mIoU | ZS AP (class-agnostic) | 用途 |
|---|---|---|---|---|---|---|
| 2026-07-12 | uf8 (Phase-A baseline) | 1 | 0.024 | 0.056 | 0.019 | 验证 evaluator 端到端;ZS≈0.019 对上立论 ~0.015 量级 |

> 说明:此为 evaluator 联调 dry-run,**非** V3 正式结果。正式行须走 §4 并登记于 §2。

---

## 4. 如何登记一次论文实验 | How to Register a Paper Experiment

```bash
# 1) (首次) 生成并冻结权威评估集
python tools/eval/evaluate_instance.py --checkpoint <ckpt> --data-root data/iSAID_instance_fewshot \
    --data-format isaid_instance --per-class 9999 --k-shot 1 --seed 42     # 生成 evaluation_manifest_val.json

# 2) 复用同一冻结 Manifest 跑每个 seed
for s in 42 123 456; do
  python tools/eval/evaluate_instance.py --checkpoint <ckpt> --decoder adaptive --prototype-source p4 \
      --data-root data/iSAID_instance_fewshot --data-format isaid_instance --k-shot 5 --seed $s \
      --manifest data/iSAID_instance_fewshot/evaluation_manifest_val.json --output-dir runs/V3-07_s$s
done

# 3) 记录 commit,回填本表对应行
git rev-parse --short HEAD
```

登记时把该行 `Checkpoint / Commit / Manifest / Seeds` 填齐,`V3 Status` 改 `✅ done`,并在论文表脚注引用本行 ID。

---

## 5. Legacy Results (历史协议 · NOT-for-paper)

以下历史实验(A / B / C / D / E / F / H 系列)在**历史评估口径**下产出(ImageShot-oracle 或 auto-eval 语义并集),
**与 V3 不可比,禁止进论文表**。完整数值与对账见 [`docs/protocol_reconciliation.md`](docs/protocol_reconciliation.md)。
如需上报,必须在 V3 下重跑并按 R1 以 `*/V3` 新后缀登记于 §2。

| 系列 | 内容 | 历史协议 | 处置 |
|---|---|---|---|
| A-series (A-0…A-23) | 解冻曲线 (uf0…uf23) | ImageShot oracle | 需 V3 重评 → `A-*/V3` |
| B-series | K-shot scaling | ImageShot oracle | 需 V3 重评 |
| C-series | Decoder / Proto 消融 | ImageShot oracle | 需 V3 重评 |
| D-series | ProtoOnly / Adaptive / FDR | auto-eval 语义并集 | 需 V3 重评 |
| E / F-series | 纯 CNN / P3P4 | ImageShot oracle | 需 V3 重评 |

---

## See also | 相关

- 唯一官方协议 | Protocol: [`EVALUATION_PROTOCOL_V3.md`](EVALUATION_PROTOCOL_V3.md)
- 论文向验证 | Validation: [`docs/appendix_A_protocol_validation.md`](docs/appendix_A_protocol_validation.md)
- 历史对账 | Reconciliation: [`docs/protocol_reconciliation.md`](docs/protocol_reconciliation.md)
- 指标定义 | Metrics: [`docs/metrics_instance_v3.md`](docs/metrics_instance_v3.md)
