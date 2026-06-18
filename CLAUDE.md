# CLAUDE.md — AdaTile-FastSAM v2

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AdaTile-FastSAM**: Adaptive Sparse FastSAM for Few-Shot High-Resolution Instance Segmentation.

**Two innovations (from v1, carried forward):**
1. **Ada-SPM** — density-supervised sparse perception module: learns importance maps → Top-K tile selection
2. **Decoupled Sparse Training** — decoder always receives full features; SPM trained via GT-driven losses in parallel

**v2 Status (2026-06-16):** Complete rewrite from scratch. All v1 code deleted. Building module by module with review.

## Development Rules

### 1. Logging First (日志先行)

**ALL new code MUST route observable values through `adatile.logging`. No bare `print()`.**

```python
from adatile.logging import get_logger
logger = get_logger("module_name")

# Quantitative values
logger.log_metric("iou", 0.85, step=step, tags=["few-shot"])
logger.log_loss("seg", 0.3, step=step)

# Progress / phase markers
logger.log_info("phase", "Stage B complete", step=step)
```

See [[logging-first-rule]] for full details.

### 2. Bilingual Comments (中英文注释)

**ALL code MUST have clear Chinese + English bilingual comments.** Every file, class, function, and non-obvious logic block.

```python
# 计算 Top-K BCE loss：只对 importance 最高的 K% 像素计算 loss
# Top-K BCE loss: compute loss only on top-K% highest importance pixels
loss_spm = topk_bce_loss(importance, density_map, k=keep_ratio)
```

See [[bilingual-comments]] for full details.

### 3. Test-Driven

Each module: write tests first, then implementation. Tests must verify logging output.

### 4. Review Each Module

Complete one module → review → approve → next module. Do NOT batch multiple modules.

## Project Structure

```
AdaTile-FastSAM/
├── pyproject.toml          # Package metadata & tool config
├── adatile/
│   ├── __init__.py          # v2.0.0.dev0
│   ├── logging/             # ✅ Structured logging system (30 tests)
│   │   ├── record.py        #   LogRecord, LogLevel
│   │   ├── base.py          #   LogBackend (ABC)
│   │   ├── backends.py      #   Console, File (JSONL), Wandb
│   │   ├── tracker.py       #   MetricTracker (running stats)
│   │   ├── context.py       #   LogContext (phase/scope stack)
│   │   └── registry.py      #   get_logger(), global registry
│   ├── backbone/            # ⬜ Feature extraction
│   ├── decoder/             # ⬜ Segmentation decoder
│   ├── sparse/              # ⬜ Ada-SPM
│   ├── losses/              # ⬜ Loss functions
│   ├── datasets/            # ⬜ Data loading
│   ├── utils/               # ⬜ Utilities
│   └── config/              # ⬜ Configuration
├── tools/                   # ⬜ Training & experiment scripts
├── tests/                   # test_logging.py ✅
└── configs/                 # ⬜ YAML configs
```

✅ = Complete | ⬜ = TODO

## Common Commands

```bash
# Install (dev mode)
pip install -e ".[dev,viz]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_logging.py -v

# Lint
ruff check adatile/
mypy adatile/
```

## Key Lessons from v1 (must follow)

1. **YOLOv8 eval mode**: `model.train()` crashes YOLOv8 detect head. Keep eval mode + `requires_grad` control.
2. **Decoder-SPM decoupled**: Decoder always receives full features. SPM trained in parallel.
3. **Budget loss differentiable**: `(imp > 0.5).float().mean()` has zero gradient → use `(imp.mean − target)²`.
4. **SPM three pillars**: GT density focal + Top-K BCE + budget loss. Missing any → importance collapse.
5. **Episodic training**: Baseline MUST also use episodic training for fair comparison.
6. **Dice GT broadcast**: `unsqueeze(0)` with batch>1 → `[1,B,H,W]` broadcast explosion.

## Architecture (from v1, for reference)

```
FastSAM-x (hook P4, P8) ──→ LightDecoder ──→ mask [B,K,H/4,W/4]
                    │
                    └──→ LightSPM ──→ importance [B,1,H/32,W/32]
                              │
                    L_spm (Top-K BCE) + L_budget (imp.mean − target)²
```

### Stage → Component Mapping (v1 reference)

| Stage | Backbone | Decoder | SPM | Loss |
|-------|----------|---------|-----|------|
| A | FastSAMHook | LightDecoder | ✗ | SegLoss only |
| B | FastSAMHook | LightDecoder | LightSPM | SegLoss + TopKLoss + Budget |
| C | FastSAMHook | LightDecoder | LightSPM + Planner | + sparse inference |

### Loss Structure (v1 reference)

```
UnifiedLoss
  ├── SegLoss      → BinarySegLoss | MultiClassSegLoss
  ├── TopKLoss     → per-image top-K ranking
  ├── DensityLoss  → absolute density regression
  ├── FixedBudget  → (imp.mean - target)²
  └── LearnableBudget → auto-discover optimal target + λ
```

L_total = L_seg + λ_spm × L_spm + λ_budget × L_budget + λ_reg × σ(w)

Default: λ_spm=1.0, budget_mode="ratio", λ_budget=5.0, keep_ratio=0.15
