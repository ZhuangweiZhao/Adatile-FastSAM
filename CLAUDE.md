# CLAUDE.md — AdaTile-FastSAM v3

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AdaTile-FastSAM**: High-Resolution Few-Shot Instance Segmentation with Adaptive Sparse Computation.

**v3 Restructuring (2026-07-03):**
- **Paper A + B merged → Single Paper.** All resources focused on one publication.
- **Task unified**: High-Resolution Few-Shot Instance Segmentation (not FSS, not Sparse-only).
- **Protocol changed**: Custom Instance Few-Shot Split (弃用 iSAID-5i FSS protocol).
- **Module unified**: Ada-SPM + FDR → **SPM (Sparse Perception Module)**.
- **Evaluation switched**: mIoU → **COCO AP** as primary metric.

**Core innovation — Adaptive Sparse Computation:**
1. **SPM (Sparse Perception Module)** — Importance Head + Dynamic Tile Router. Unified from Ada-SPM + FDR. Predicts tile importance → Top-K selection.
2. **Adaptive Decoder** — ProtoCoeffPredictor (support prototype → 32 mask coefficients) + P4 Feature Refinement (spatial detail compensation) + Mask Head.
3. **Few-Shot Fine-tuning Protocol** — Base pre-training → K-shot Novel fine-tune → direct inference (not Episode-based FSS).

## One-Sentence Summary

> **FastSAM fails on aerial imagery (AP=0.015, 27× domain gap). We enable it via Few-Shot Fine-tuning with Adaptive Sparse Computation — competitive performance while processing only ~40% of image tiles.**

## Git Branches

```
main      → Paper A archive (E-series experiments, historical reference)
paper-b   → Active development (v3: all new experiments)
```

## Paper Narrative (v3)

```
1. Introduction
   SA-1B fails on aerial imagery (AP=0.015, 27× domain gap)
   → Few-shot domain adaptation + Sparse computation

2. Method (3 modules)
   2.1 SPM (Sparse Perception Module)
       ImportanceHead → tile importance map → TileRouter → Top-K tiles
   2.2 Adaptive Decoder
       ProtoCoeffPredictor → 32 coeffs + P4 Refinement → Mask
   2.3 Few-Shot Fine-tuning Protocol
       Base pre-train → K-shot fine-tune → direct inference

3. Experiments
   3.1 Zero-Shot Baseline (AP=0.015)
   3.2 Spatial Sparsity (60% empty, Top-40% → 96.5% FG)
   3.3 Few-Shot Scaling (K=1/3/5/10, SSI-1=88.6%)
   3.4 SPM Efficiency (FLOPs vs AP Pareto frontier)
   3.5 Decoder Ablation (ProtoOnly vs +Refinement, Δ=+34.5%)
   3.6 Cross-Dataset (NWPU, category-agnostic)
```

## Development Rules

### 1. Logging First (日志先行)

**ALL new code MUST route observable values through `adatile.logging`. No bare `print()`.**

```python
from adatile.logging import get_logger
logger = get_logger("module_name")
logger.log_metric("iou", 0.85, step=step, tags=["few-shot"])
logger.log_info("phase", "Stage B complete", step=step)
```

**FileBackend is crash-safe**: `buffer_size=1`, `flush_interval=1.0` — every record flushed to disk immediately.

### 2. Bilingual Comments (中英文注释)

Every file, class, function, and non-obvious logic block must have Chinese + English bilingual comments.

### 3. Test-Covered

Core library modules should have tests verifying shape, value range, and edge cases.
Current coverage: logging (16), metrics (13), losses (7), backbone (pending), decoder (4), spatial_router (5).
Experiment scripts (tools/) are validated via dry-run, not unit tests.

### 4. Review Each Module

Complete one module → review → approve → next module. Do NOT batch multiple modules.

### 5. Reproducibility

All experiment scripts must call `set_seed()` from `adatile.utils.seed`. This sets Python/Random, NumPy, PyTorch, and cuDNN deterministic mode.

## Project Structure

```
adatile/
├── logging/         ✅ Structured logging (Console, File/JSONL, Wandb backends)
├── backbone/        ✅ FastSAMBackbone (hook P3/P4/P8, Proto extraction, eval-mode enforced)
├── config/          ✅ ExperimentConfig + ExperimentRecorder + generate_exp_id()
├── metrics/         ✅ compute_miou, compute_dice, FPSMeter, count_params, COCOInstanceEvaluator
├── decoder/
│   ├── light_decoder.py         ✅ LightDecoder (716K, P4-only)
│   ├── light_decoder_p3p4.py    ✅ LightDecoderP3P4 (274K, multi-scale)
│   └── adaptive_decoder.py      ✅ AdaptiveDecoder + ProtoOnlyDecoder (v2: InstanceNorm2d)
├── sparse/
│   ├── spm.py                   🔄 SparsePerceptionModule (renamed from spatial_router.py)
│   │   ├── ImportanceHead       ← DensityHead (重命名)
│   │   └── TileRouter           ← select_tiles()
│   └── coefficient_predictor.py ✅ ProtoCoeffPredictor (3-layer MLP, ~427K)
├── datasets/
│   ├── isaid_tiles.py              ✅ FastISAIDTileDataset (1024² tiles)
│   ├── isaid_tile_wrapper.py       ✅ Full-image→tile wrapper (bbox overlap, LRU P4 cache)
│   ├── isaid_instance_fewshot.py   🔄 ISAIDInstanceFewShotDataset (NEW, 896², COCO format)
│   ├── p4_cache.py                 ✅ P4 precompute cache (GPU/CPU/fp16)
│   ├── nwpu.py                     ✅ NWPU-VHR-10 bbox-based weak masks (10-class)
│   └── loveda_tiles.py             ✅ LoveDA land-cover tiles
├── losses/           ✅ FocalLoss (eps=1e-4, γ=5.0), DiceLoss, CombinedLoss
└── utils/
    ├── seed.py             ✅ Unified set_seed() with cuDNN deterministic
    ├── label_mapping.py    ✅ Per-split category ID mapping for iSAID
    ├── render.py           ✅ Shared render_category_mask() (canonical def)
    └── prototype.py        ✅ compute_fg_prototype() (canonical)

tools/
├── data/                            # Data preprocessing
│   ├── prep_isaid.py                iSAID COCO -> category-id masks (Step 0)
│   ├── prep_isaid_tiles.py          Full pipeline: render mask -> cut tiles -> metadata
│   └── prep_isaid_instance.py       🔄 NEW: iSAID Instance Few-Shot Split
├── train/                           # Training entry points (v3)
│   ├── train_base.py                🔄 NEW: Base pre-training on 10 classes
│   ├── train_fewshot.py             🔄 NEW: Novel K-shot fine-tune
│   └── train_supervised.py          ✅ A-Series: full supervision (archived reference)
├── eval/                            # Evaluation (v3)
│   ├── eval_zero_shot.py            🔄 NEW: Zero-shot COCO AP baseline
│   └── eval_fewshot.py              🔄 NEW: Few-shot COCO AP evaluation
├── archive/                         # v2 archived scripts (reference only)
│   ├── train_instance_fewshot.py    D-Series training script (v3 NaN fixes preserved)
│   ├── eval_baseline_instance.py    D-00 zero-shot baseline
│   ├── eval_c03_catsam_fewshot.py   C-03 Cross-Attention FSS
│   └── eval_c04_full_fewshot.py     C-04 Full 15-class FSS
├── paper_a/                         # Paper A archive (main branch, historical)
├── paper_b/                         # B-Series experiments (reusable in v3)
│   ├── eval_b00_tile_size_sensitivity.py   → V3-02
│   ├── eval_b01_oracle_topk.py             → V3-03
│   ├── eval_b02_learnability.py            → V3-04
│   ├── eval_b03_router_architecture.py     → SPM ablation
│   └── ...
├── diag/                            # Diagnostics
│   ├── diag_b04_overfit.py
│   ├── diag_class_stats.py
│   ├── diag_check_labels.py
│   └── diag_trace_labels.py
└── viz/                             # Visualization
    ├── viz_paper_a_p6.py
    └── viz_paper_a_router.py
```

## Key Lessons from v1/v2 (MUST follow)

1. **YOLOv8 eval mode**: `model.train()` crashes YOLOv8 detect head. Keep eval mode + `requires_grad` control.
2. **Decoder-SPM decoupled**: Decoder always receives full features. SPM trained in parallel.
3. **Budget loss differentiable**: `(imp > 0.5).float().mean()` has zero gradient → use `(imp.mean − target)²`.
4. **SPM three pillars**: GT density focal + Top-K BCE + budget loss. Missing any → importance collapse.
5. **Episodic baseline**: Baseline MUST also use episodic training for fair comparison (not applicable in v3 few-shot fine-tuning).
6. **Dice GT broadcast**: `unsqueeze(0)` with batch>1 → `[1,B,H,W]` broadcast explosion.
7. **Focal for remote sensing**: eps=1e-4 (not 1e-8, prevents gradient explosion), γ=5.0 (extreme FG/BG imbalance).
8. **BatchNorm→InstanceNorm**: bs=1 training requires InstanceNorm2d(affine=True) throughout decoder.
9. **min_tiles filter**: Exclude classes with <30 tiles from training (prevents rare-class NaN).
10. **FDR is tile-selector, NOT pixel-gate**: Per-pixel FDR gating on already-selected tiles is harmful (-2.2% mIoU).

## v3 Architecture

```
High-Res Image (H×W, up to 4000×4000)
      │
      ▼
┌─────────────────────────────────────────┐
│  FastSAM Backbone (frozen)              │
│  ├── P3 [H/8, W/8, 960]                 │
│  ├── P4 [H/16, W/16, 1280]   ← decoder  │
│  ├── P8 [H/32, W/32, 1280]   ← SPM      │
│  └── Proto [H/4, W/4, 32]               │
└─────────────────────────────────────────┘
      │              │              │
      ▼              ▼              ▼
┌──────────┐  ┌───────────┐  ┌────────────────┐
│   SPM    │  │   Proto   │  │  Support Set    │
│ P8→Imp.  │  │  Masks    │  │  K-shot/class   │
│ →Tiles   │  │ [32,H,W]  │  │  → Prototype    │
└────┬─────┘  └─────┬─────┘  └───────┬────────┘
     │              │                 │
     │   K tiles    │   crop to tile  │
     └──────┬───────┘                 │
            ▼                         ▼
   ┌─────────────────────────────────────┐
   │  Adaptive Decoder                   │
   │  ├── ProtoCoeffPredictor (427K)     │
   │  ├── Feature Refinement (P4, 714K)  │
   │  └── Mask Head                      │
   └─────────────────────────────────────┘
            │
            ▼
   Instance Masks + Confidence Scores
            │
            ▼
   COCO AP Evaluation (AP/AP50/AP75/APS)
```

## v3 Experiment Line

```
Phase 1: Zero-Shot Baseline
  V3-01: FastSAM on iSAID Instance Split → AP=0.015

Phase 2: Spatial Sparsity (reuse B-00→B-03)
  V3-02: Tile FG distribution → 60% empty @ 1024px
  V3-03: Oracle Top-K → SSI, Top-40% → 96.5% FG
  V3-04: SPM learnability → Spearman r=0.889

Phase 3: Few-Shot Fine-tuning
  V3-05: Base pre-training (10 classes, full data)
  V3-06: K-shot scaling (K=1/3/5/10) → SSI-1=88.6%
  V3-07: Decoder ablation (ProtoOnly vs +Refinement)
  V3-08: Per-class breakdown

Phase 4: Sparse Routing Efficiency
  V3-09: SPM tile routing @ full-image
  V3-10: Top-K% sweep → FLOPs vs AP Pareto
  V3-11: FPS + GPU Memory measurement

Phase 5: Cross-Dataset
  V3-12: NWPU-VHR-10 few-shot transfer
```

## Module Naming Convention (v3)

```
SPM (Sparse Perception Module)    ← was FDR + Ada-SPM
  └── ImportanceHead              ← was DensityHead
  └── TileRouter                  ← was select_tiles()

AdaptiveDecoder                   ← was AdaptiveSparseDecoder
  └── ProtoCoeffPredictor         (unchanged)
  └── FeatureRefinement           ← was feat_proj + feat_refine
  └── MaskHead                    ← was mask_head
```

## Data Protocol (v3)

```
✅ iSAID Instance Few-Shot Split (NEW, 896² tiles)
   Format: COCO JSON (instance segmentation)
   Split:  Base (10 classes) → pre-train
           Novel (5 classes)  → K-shot fine-tune
           3-Fold cross-validation

❌ iSAID-5i FSS protocol (DEPRECATED for v3)
   Kept for historical reference only.
```

## Evaluation (v3)

```
Primary:
  COCO AP, AP50, AP75, AP_small, AP_medium, AP_large

Efficiency:
  FLOPs, FPS, GPU Memory (peak)

Supplementary:
  mIoU (semantic level), AR, per-class IoU breakdown
```

## Known Issues & Workarounds

### FastSAM thirdLibrary PyTorch 2.x compatibility

`thirdLibrary/FastSAM/ultralytics/nn/modules/conv.py:297` — `torch.cat(x, self.d)` fails on PyTorch ≥2.0. Fixed to:
```python
if isinstance(x, torch.Tensor):
    return x
return torch.cat(x, dim=self.d)
```

### Non-square images cause FastSAM dimension mismatch

FastSAM requires input dimensions to be multiples of 32. Always pad images:
```python
pad_h = (32 - H % 32) % 32
pad_w = (32 - W % 32) % 32
```

### FastSAM CUDA OOM on large images

Full-size iSAID images (4000×4000+) cause OOM on GPUs < 12GB. Use `--max-image-size 2048` or `--device cpu`.

### Focal NaN on rare classes

Three-layer protection:
1. `min_tiles=30` filter in EpisodeSampler (exclude <30 tile classes)
2. `focal_loss(eps=1e-4)` — caps gradient from 1e8 to 1e4
3. `focal_loss(gamma=5.0)` — stronger BG suppression
4. NaN skip + `optimizer.zero_grad()` safety net

### FileBackend data loss on crash

Fixed: `buffer_size=1`, `flush_interval=1.0` globally in `adatile/logging/backends.py`. Every record immediately written.

## Common Commands

```bash
# Dev install (full toolchain)
pip install -e ".[dev,viz]"

# Tests
pytest tests/ -v
pytest tests/ -v --cov=adatile --cov-report=term-missing

# Lint / Format
ruff check adatile/
black adatile/ tests/

# Data preprocessing (new v3 pipeline)
python tools/data/prep_isaid.py                         # Step 0: fix COCO JSON
python tools/data/prep_isaid_instance.py                # 🔄 NEW: Instance Few-Shot Split

# v3 Training
python tools/train/train_base.py --epochs 50 --batch-size 8    # 🔄 NEW: Base pre-training
python tools/train/train_fewshot.py --k-shot 5 --fold 0        # 🔄 NEW: Few-shot fine-tune

# v3 Evaluation
python tools/eval/eval_zero_shot.py --split val                 # 🔄 NEW
python tools/eval/eval_fewshot.py --checkpoint best_model.pt    # 🔄 NEW

# B-Series (reusable in v3)
python tools/paper_b/eval_b00_tile_size_sensitivity.py
python tools/paper_b/eval_b01_oracle_topk.py
python tools/paper_b/eval_b02_learnability.py
```

## Supplementary Docs

- **`docs/V3_RESTRUCTURE_PLAN.md`** — v3 restructuring plan (strategy, module rename, data protocol, experiment line)
- **`docs/PROJECT_MASTER.md`** — Full project overview (all A/B/C/D series + engineering)
- **`docs/D_series_master.md`** — D-Series complete archive (5 experiments + NaN engineering)
- **`RESEARCH_MAP.md`** — Research map with evidence chains and architecture diagrams
- **`docs/c04_code_explanation.md`** — Line-by-line walkthrough of C-04 full 15-class experiment

## Persistent Memory

Project memory stored at `C:\Users\20871\.claude\projects\E--A-postgraduate-stude-AdaTile-FastSAM\memory\`. Index at `MEMORY.md`.
