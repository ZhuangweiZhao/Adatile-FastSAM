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
   3.1 Zero-Shot Baseline (AP=0.0182 @ 896² tiles)
   3.2 Spatial Sparsity (60% empty, Top-40% → 96.5% FG)
   3.3 Few-Shot Scaling (K=1/3/5/10, SSI-1=88.6%)
   3.4 SPM Efficiency (FLOPs vs AP Pareto frontier)
   3.5 Decoder Ablation (ProtoOnly vs +Refinement, Δ=+34.5%)
   3.6 Prototype Mechanism-Function-Task Decoupling (NEW)
       → L2 restores gradient (Mech ✓) but NOT class-conditioning (Func ✗)
       → Yet Task improves (+31% AP) → attribution probes needed
   3.7 Cross-Dataset (NWPU, category-agnostic)

## Mechanism-Function-Task Analysis Framework

**Three layers must be verified independently — one does NOT imply another:**

```
Mechanism (梯度流/数值稳定性)
    ↓
Function (模块是否承担设计的语义角色)
    ↓
Task (下游指标: AP, IoU, etc.)
```

**Current state (l2 vs none, uf8 backbone):**
- **Mechanism ✓**: L2 restores gradient flow (basis≈1, sat≈0, coeff_grad≈4e-3)
- **Function ✗**: Prototype still dead (coeff_cos=1.0, Normal≈Zero)
- **Task ✓**: AP +31%, class-agnostic 9×

**Key insight**: Mechanism recovery ≠ Function recovery ≠ Performance gain. The T/F/T combination is the discovery.

**Attribution probes** (see `tools/diag/`):
- Probe 1 (`diag_proto_mask_stats.py`): Proto mask statistics (mean/std/entropy/FG IoU)
- Probe 2 (`diag_fusion_attribution.py`): Fusion contribution ratio (||proto||/||query||)
- Probe 3 (TBD): Proto OFF ablation — causal cut to isolate gain source

## Evaluation Protocol V3 (Frozen)

**`EVALUATION_PROTOCOL_V3.md` is the single official evaluation protocol.** All paper results MUST come from the V3 evaluator. Key facts:

- **Official evaluator**: `tools/eval/evaluate_instance.py` — the ONLY sanctioned entry point
- **Protocol guards**: `tests/test_protocol_frozen.py` (freeze) + `tests/test_protocol_audit.py` (audit) + `tests/test_instance_match.py` (matching)
- **Commit hook**: `.githooks/pre-commit` — blocks any commit that changes frozen metric/matching definitions
- **Experiment registry**: `EXPERIMENT_MANIFEST.md` — binds every paper experiment ID → Protocol → Manifest → Seed → Checkpoint → Commit (single source of truth)

```bash
# Enable protocol freeze guard (once per clone)
git config core.hooksPath .githooks

# Verify protocol integrity
pytest tests/test_protocol_frozen.py tests/test_protocol_audit.py tests/test_instance_match.py -q
```

**Protocol boundary**: The protocol defines evaluation semantics (AP, Instance mIoU, one-to-one matching, no union masks). Model-side choices (decoder design, score aggregation, connected-component threshold) are outside the protocol and can be freely improved.

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
├── metrics/
│   ├── instance_match.py       ✅ Instance-level matching + Instance mIoU (pure numpy)
│   └── coco_eval.py            ✅ COCO AP official wrapper (ONLY file calling pycocotools.COCOeval)
├── decoder/
│   ├── adaptive_sparse_decoder.py  ✅ AdaptiveSparseDecoder + ProtoOnlyDecoder (v3 main)
│   ├── adaptive_decoder_p3p4.py    ✅ AdaptiveDecoderP3P4 (multi-scale P3+P4)
│   ├── pure_cnn_decoder.py         ✅ PureDecoder + PureDecoderP3P4 (no-prototype baseline)
│   ├── conditioned_decoder.py      ✅ ConditionedDecoder (support-template conditioned)
│   ├── light_decoder.py            ✅ LightDecoder + LightDecoderP3/P3P4 (v2 reference)
│   ├── proto_module.py             ✅ ProtoModule (standalone proto mask generation)
│   ├── instance_decoder.py         ✅ InstanceDecoder (instance-level head)
│   ├── linear_probe.py             ✅ LinearProbe (1×1 Conv, E002)
│   └── fusion_probe.py             ✅ FusionProbe (P4+P8, E003)
├── adapter/           ✅ ConvAdapter + MultiScaleAdapter (CAT-SAM-style channel attention)
├── prompt/            ✅ GenericPrompt + PrototypePrompt + Fusion (class-conditional prompt tokens)
├── sparse/
│   ├── spm.py                   ✅ SparsePerceptionModule + ImportanceHead + TileRouter
│   ├── spatial_router.py        ✅ SpatialRouter (v2 reference, kept for ablation)
│   └── coefficient_predictor.py ✅ ProtoCoeffPredictor (3-layer MLP, ~427K)
├── datasets/
│   ├── isaid_tiles.py              ✅ FastISAIDTileDataset (1024² tiles)
│   ├── isaid_tile_wrapper.py       ✅ Full-image→tile wrapper (bbox overlap, LRU P4 cache)
│   ├── isaid_instance_fewshot.py   ✅ ISAIDInstanceFewShotDataset (v3, 896², COCO format)
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
│   └── prep_isaid_instance.py       ✅ iSAID Instance Few-Shot Split
├── train/                           # Training entry points (v3)
│   ├── train_fewshot_allclass.py    ✅ V3: All-15-class K-shot fine-tuning (main training script)
│   ├── train_fewshot_finetune.py    ✅ V3: K-shot fine-tune variant
│   ├── train_base.py                ✅ V3-05: Base pre-training on 10 classes
│   ├── train_fewshot.py             ✅ V3-06: Novel K-shot fine-tune (fixed support set)
│   ├── train_supervised_full.py     ✅ V3: Full supervision (all 15 classes)
│   ├── train_supervised.py          ✅ A-Series: full supervision (archived reference)
│   ├── train_catsam_style.py        CAT-SAM-style prompt training
│   └── train_instance_fewshot.py    D-Series archive (v3 NaN fixes preserved)
├── eval/                            # Evaluation (v3)
│   ├── evaluate_instance.py         ✅ OFFICIAL V3 evaluator (ONLY sanctioned entry point)
│   ├── eval_fewshot_allclass.py     ✅ V3: All-class few-shot eval
│   ├── eval_novel_fewshot.py        ✅ V3: Novel-class few-shot eval
│   ├── eval_zero_shot.py            ✅ V3-01: Zero-shot COCO AP baseline
│   ├── eval_fastsam_prompted.py     ✅ FastSAM prompted eval
│   ├── export_results_csv.py        ✅ Export eval results to CSV
│   └── run_ablation.py              ✅ Ablation sweep runner
├── instance/                        # v2 archived eval scripts (reference only)
│   ├── eval_baseline_instance.py    D-00 zero-shot baseline
│   ├── eval_c02a_fastsam_fewshot.py C-02a FastSAM few-shot
│   ├── eval_c03_catsam_fewshot.py   C-03 Cross-Attention FSS
│   └── eval_c04_full_fewshot.py     C-04 Full 15-class FSS
├── diag/                            # Diagnostics
│   ├── diag_proto_mask_stats.py     🆕 Probe 1: Proto mask statistics (l2 vs none)
│   ├── diag_fusion_attribution.py   🆕 Probe 2: Fusion contribution ratio analysis
│   ├── diag_prototype_analysis.py   ✅ Proto collapse & diversity diagnosis
│   ├── diag_gradient_starvation.py  ✅ Gradient starvation diagnosis
│   ├── diag_object_size.py          ✅ Object size distribution analysis
│   ├── diag_feature_vis.py          ✅ Feature space visualization
│   ├── diag_feature_space.py        ✅ Feature space analysis
│   ├── diag_attention_sparse.py     ✅ Sparse attention quality
│   ├── diag_dense_analysis.py       ✅ Dense matching analysis
│   ├── diag_causal_dense_softmax.py ✅ Causal dense softmax
│   └── ...                          (25+ diagnostic scripts total)
├── paper_a/                         # Paper A archive (main branch, historical)
├── paper_b/                         # B-Series experiments (reusable in v3)
│   ├── eval_b00_tile_size_sensitivity.py   → V3-02
│   ├── eval_b01_oracle_topk.py             → V3-03
│   ├── eval_b02_learnability.py            → V3-04
│   ├── eval_b03_router_architecture.py     → SPM ablation
│   └── ...
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
  V3-01: FastSAM on iSAID Instance Split → AP=0.0182

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

### Proto-basis magnitude explosion (saturation death)

Unfrozen fine-tuning (uf≥5) causes FastSAM proto basis norm to explode (523 → 3.4e7 → 1e11).
This pushes `coeffs@proto` into the sigmoid dead zone → zero gradient → CoeffPredictor dies.

**Workaround**: `--normalize-proto l2` constrains per-basis L2 norm to 1, restoring gradient flow.
**Caveat**: Saturation can "transfer" from basis to coefficients (coeff_l2 → 19183).
LayerNorm variant (`--normalize-proto layernorm`) constrains both basis and coeffs simultaneously.
See `AdaptiveSparseDecoder._normalize_proto()`.

## Common Commands

```bash
# One-time setup
pip install -e ".[dev,viz]"                              # dev install
git config core.hooksPath .githooks                      # enable protocol freeze guard

# Tests
pytest tests/ -v
pytest tests/ -v --cov=adatile --cov-report=term-missing
pytest tests/test_protocol_frozen.py tests/test_protocol_audit.py tests/test_instance_match.py -q

# Lint / Format
ruff check adatile/
black adatile/ tests/

# Data preprocessing
python tools/data/prep_isaid.py                         # Step 0: fix COCO JSON
python tools/data/prep_isaid_instance.py                # Instance Few-Shot Split

# v3 Training (main entry points)
python tools/train/train_fewshot_allclass.py --k-shot 1 --epochs 50 --unfreeze-layers 8
python tools/train/train_base.py --epochs 50 --batch-size 8
python tools/train/train_supervised_full.py --epochs 50

# v3 Evaluation (OFFICIAL — use evaluate_instance.py for ALL paper numbers)
python tools/eval/evaluate_instance.py \
    --checkpoint best_model.pt --decoder adaptive --prototype-source p4 \
    --data-root data/iSAID_instance_fewshot --data-format isaid_instance \
    --k-shot 1 --seed 42            # writes instance_metrics.json

# Zero-shot baseline
python tools/eval/eval_zero_shot.py --split val --data-format isaid_instance

# Protocol guards
pytest tests/test_protocol_frozen.py tests/test_protocol_audit.py tests/test_instance_match.py -q

# Diagnostics (Probe 1 & 2 — M-F-T attribution)
python tools/diag/diag_proto_mask_stats.py --checkpoint-a <none.pt> --checkpoint-b <l2.pt> --device cuda
python tools/diag/diag_fusion_attribution.py --checkpoint-a <none.pt> --checkpoint-b <l2.pt> --device cuda

# B-Series (reusable in v3)
python tools/paper_b/eval_b00_tile_size_sensitivity.py
python tools/paper_b/eval_b01_oracle_topk.py
python tools/paper_b/eval_b02_learnability.py
```

## Supplementary Docs

- **`EVALUATION_PROTOCOL_V3.md`** — ⚠️ Official frozen evaluation protocol (single source of truth for all paper results)
- **`EXPERIMENT_MANIFEST.md`** — Paper experiment registry: each ID → Protocol → Manifest → Seed → Checkpoint → Commit
- **`docs/V3_RESTRUCTURE_PLAN.md`** — v3 restructuring plan (strategy, module rename, data protocol, experiment line)
- **`docs/PROJECT_MASTER.md`** — Full project overview (all A/B/C/D series + engineering)
- **`docs/D_series_master.md`** — D-Series complete archive (5 experiments + NaN engineering)
- **`RESEARCH_MAP.md`** — Research map with evidence chains and architecture diagrams
- **`docs/c04_code_explanation.md`** — Line-by-line walkthrough of C-04 full 15-class experiment
- **`docs/appendix_A_protocol_validation.md`** — Protocol V3 ↔ COCO consistency proof (Supplementary Material)
- **`docs/protocol_reconciliation.md`** — Historical protocol → V3 migration (why historical results are NOT used in paper)

## Persistent Memory

Project memory stored at `C:\Users\20871\.claude\projects\E--A-postgraduate-stude-AdaTile-FastSAM\memory\`. Index at `MEMORY.md`.
