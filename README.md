# AdaTile-FastSAM

**High-Resolution Few-Shot Instance Segmentation with Adaptive Sparse Computation.**

> FastSAM fails on aerial imagery (AP≈0.015, ~27× domain gap). We enable it via few-shot
> fine-tuning with adaptive sparse computation — competitive performance while processing only
> ~40% of image tiles.

- 项目总览 | Project overview: [`CLAUDE.md`](CLAUDE.md) · [`RESEARCH_MAP.md`](RESEARCH_MAP.md) · [`docs/PROJECT_MASTER.md`](docs/PROJECT_MASTER.md)
- 活跃开发分支 | Active branch: `paper-b` (v3). `main` = Paper A archive.

---

## Documentation

### Evaluation (协议体系 · 已冻结 | frozen protocol suite)

- **[Evaluation Protocol V3](EVALUATION_PROTOCOL_V3.md)** — 唯一官方评估协议(开发者向)。
  Official, frozen evaluation protocol used for **all** experiments reported in this repository.
  Defines what is / isn't part of the protocol, plus the change-control process.
- **[Appendix A – Protocol Validation](docs/appendix_A_protocol_validation.md)** — 论文向补充材料。
  Explains **why** Protocol V3 is consistent with standard COCO instance-segmentation evaluation
  (V3 ↔ COCO correspondence, Model-vs-Protocol boundary). Drop-in for Supplementary Material.
- **[Metric Definitions](docs/metrics_instance_v3.md)** — AP 家族 / Instance mIoU / 一对一匹配 / TP·FP·FN 的形式化定义。
  Formal definitions of AP, Instance mIoU, matching strategy, and evaluation statistics.
- **[Historical Protocol Reconciliation](docs/protocol_reconciliation.md)** — 历史口径 → V3 的迁移与排除说明。
  Documents the transition from historical evaluation protocols to V3 and why historical
  results are **not** used in the paper.

### Data & Experiments

- **[Dataset Protocol](docs/isaid_instance_fewshot_spec.md)** — iSAID Instance Few-Shot Split(896², COCO 格式)。
- **[Experiment Registry](EXPERIMENT_MANIFEST.md)** — 论文实验注册清单:每个实验编号 → Protocol / Manifest /
  Seed / Checkpoint / Commit。**投稿追溯的唯一入口。** Paper experiment registry (single source of truth
  binding each experiment ID to its protocol, manifest, seeds, checkpoint, and commit).
- **[v3 Restructure Plan](docs/V3_RESTRUCTURE_PLAN.md)** — v3 重构策略与实验线。

---

## Evaluation Quickstart

评估**唯一入口**是官方 V3 评估器;历史评估器已隔离并在运行时阻断(见协议 §12.4)。
The only sanctioned way to produce reportable numbers is the official V3 evaluator.

```bash
# 官方 V3 实例分割评估 | Official V3 instance-seg evaluation
python tools/eval/evaluate_instance.py \
    --checkpoint <best_model.pt> --decoder adaptive --prototype-source p4 \
    --data-root data/iSAID_instance_fewshot --data-format isaid_instance \
    --k-shot 1 --seed 42            # writes instance_metrics.json

# 协议守卫 (冻结 + 审计) | Protocol guards (freeze + audit)
pytest tests/test_protocol_frozen.py tests/test_protocol_audit.py tests/test_instance_match.py -q

# 启用提交钩子 (每个克隆一次) | Enable the commit hook (once per clone)
git config core.hooksPath .githooks
```

---

## Development

```bash
pip install -e ".[dev,viz]"          # dev install
pytest tests/ -v                      # tests
ruff check adatile/ && black adatile/ tests/   # lint / format
```

开发规则见 [`CLAUDE.md`](CLAUDE.md):日志先行、中英双语注释、模块逐一评审、`set_seed()` 可复现。
