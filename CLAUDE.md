# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AS-FastSAM**: Adaptive Sparse FastSAM for Few-Shot High-Resolution Instance Segmentation.

**Two innovations:**
1. **Ada-SPM** — density-supervised sparse perception module: learns importance maps → Top-K tile selection
2. **Decoupled Sparse Training** — decoder always receives full features; SPM trained via GT-driven losses in parallel

## Common Commands

```bash
# Install
pip install -e ".[dev,viz]"

# ── Unified Trainer (Stage A/B/C) ─────────────────────────
python tools/train_as_fastsam.py                        # Stage A: Backbone + Decoder
python tools/train_as_fastsam.py --use-spm              # Stage B: + Ada-SPM
python tools/train_as_fastsam.py --use-spm --use-planner # Stage C: + Sparse Inference

# ── Experiments ──────────────────────────────────────────
python tools/ablation_spm_supervision.py    # SPM supervision: Density vs Top-K, λ sweep, keep_ratio sweep
python tools/exp_fewshot.py                 # Few-shot: Baseline vs SPM
python tools/ablation_domain_shift.py       # Domain shift (Urban/Rural)
python tools/ablation_tile_ratio.py         # Keep ratio sweep

# ── Dataset Building ─────────────────────────────────────
python tools/build_isaid_dota.py            # Build iSAID+DOTA combined dataset

# ── Tests & Linting ──────────────────────────────────────
pytest tests/ -v
mypy adatile/
ruff check adatile/
```

## Architecture

### Core Pipeline

```
FastSAM-x (hook P4, P8) ──→ LightDecoder ──→ mask [B,K,H/4,W/4]
                    │
                    └──→ LightSPM ──→ importance [B,1,H/32,W/32]
                              │
                    L_spm (Top-K BCE) + L_budget (imp.mean − target)²
```

### Module Map

| Module | Role | Files |
|--------|------|-------|
| `adatile/backbone/` | Feature extraction | `fastsam_hook.py` (active), `base.py` |
| `adatile/decoder/` | Segmentation decoder | `light_decoder.py` |
| `adatile/sparse/` | Importance prediction | `light_spm.py` (active), `fpn_fusion.py` |
| **`adatile/losses/`** | Loss functions | `seg_loss.py`, `spm_loss.py`, `budget_loss.py` |
| `adatile/datasets/` | Data loading | `universal.py` (auto-detect) |
| `adatile/logging/` | Experiment tracking | `run_logger.py` |
| `tools/` | Scripts | `train_as_fastsam.py` (unified entry) |
| `legacy/` | Old pipeline | FPN + transformer + tokenizer + router |

### Loss Structure

```
UnifiedLoss (composition)
  ├── SegLoss      → BinarySegLoss | MultiClassSegLoss
  ├── TopKLoss     → per-image top-K ranking (Ada-SPM core)
  ├── DensityLoss  → absolute density regression (baseline)
  ├── FixedBudget  → (imp.mean - target)²   (manual)
  └── LearnableBudget → auto-discover optimal target + λ
```

L_total = L_seg + λ_spm × L_spm(top-K) + λ_budget × (imp.mean − σ(w))² + λ_reg × σ(w)

Default: λ_spm=1.0, budget_mode="ratio" with λ_budget=5.0, keep_ratio=0.15

## Stage → Component Mapping

| Stage | Backbone | Decoder | SPM | Loss |
|-------|----------|---------|-----|------|
| A | FastSAMHook | LightDecoder | ✗ | SegLoss only |
| B | FastSAMHook | LightDecoder | LightSPM | SegLoss + TopKLoss + Budget |
| C | FastSAMHook | LightDecoder | LightSPM + Planner | + sparse inference |

## Critical Lessons

### 1. YOLOv8 Segment Head Cannot Be Fine-Tuned
Use hook-based feature extraction + custom decoder.

### 2. Importance + Decoder Must Be Decoupled
Decoder always receives full features during training. SPM trained in parallel via GT-driven losses.

### 3. Budget Loss Must Be Fully Differentiable
`(imp > 0.5).float().mean()` has zero gradient. Fix: `(imp.mean − target)²`.

### 4. Multi-Class Mask Detection
{0, 255} masks need `n_unique ≤ 2 → binary` check, not just `max_val ≤ 1`.
