# CLAUDE.md — AdaTile-FastSAM v2

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AdaTile-FastSAM**: Adaptive Sparse FastSAM for Few-Shot High-Resolution Instance Segmentation.

**Two-Paper Strategy (2026-06-21):**
- **Paper A** (Proto Sparsity / Learned Sparse Proto Routing): Archived on `main`. Evidence chain E007→E011-U complete. ICIP/CCIG target.
- **Paper B** (Dual Sparsity / Spatial Sparsity / AdaTile): Active on `paper-b`. Theory chain B-00→B-03 CLOSED. B-04 Decoder verified. B-05->B-09 few-shot + cross-dataset expansion in progress.

**Core innovations:**
1. **Ada-SPM** — density-supervised sparse perception module: learns importance maps → Top-K tile selection (Paper A)
2. **Foreground Density Router (FDR)** — 75K params, Pareto optimal spatial router: learns objectness/density, not edges or class semantics (Paper B)
3. **Decoupled Sparse Training** — decoder always receives full features; SPM/Router trained via GT-driven losses in parallel

## Git Branches

```
main      → Paper A archive (all E-series experiments)
paper-b   → Paper B active development (B-series + C-series few-shot experiments)
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
├── backbone/        ✅ FastSAMBackbone (hook P4/P8, eval-mode enforced, freeze control)
├── config/          ✅ ExperimentConfig + ExperimentRecorder + generate_exp_id()
├── metrics/         ✅ compute_miou, compute_dice, FPSMeter, count_params
├── decoder/         ✅ LightDecoder, InstanceDecoder, LinearProbe, FusionProbe
├── datasets/
│   ├── base.py                 ✅ BaseSegDataset
│   ├── mass_buildings.py       ✅ MassBuildings tile dataset
│   ├── isaid.py                ✅ iSAID COCO dataset (full-image)
│   ├── isaid_tiles.py          ✅ FastISAIDTileDataset (pre-cut 1024x1024 tiles)
│   ├── isaid_tile_wrapper.py   ✅ Full-image→tile wrapper (bbox overlap, LRU P4 cache)
│   ├── p4_cache.py             ✅ P4 precompute cache (GPU/CPU/fp16/persistent/build_fast)
│   ├── loveda_tiles.py         ✅ LoveDA land-cover tiles (7-class dense, SSI<50)
│   ├── nwpu.py                 ✅ NWPU-VHR-10 bbox-based weak masks (10-class)
│   └── vaihingen_tiles.py      ✅ Vaihingen tiles (6-class dense land-cover)
├── sparse/
│   └── spatial_router.py    ✅ ForegroundDensityRouter, DensityHead, EdgeHead, TinyCNNRouter
├── losses/           ✅ FocalLoss, DiceLoss, CombinedLoss
└── utils/
    ├── seed.py             ✅ Unified set_seed() with cuDNN deterministic
    ├── label_mapping.py    ✅ Per-split category ID mapping for iSAID
    ├── render.py           ✅ Shared render_category_mask() (canonical def)
    └── prototype.py        ✅ compute_fg_prototype() (canonical, 4 files→1)

tools/
├── data/                            # Data preprocessing
│   ├── prep_isaid.py                iSAID COCO -> category-id masks (Step 0)
│   ├── prep_isaid_tiles.py          Full pipeline: render mask -> cut tiles -> metadata
│   ├── prep_cityscapes.py           Cityscapes -> tile format
│   └── fix_labels.py                Repair tool: fix category ID mapping in instances JSON
├── train/                           # Training entry points
│   ├── train_isaid_mc.py            iSAID multi-class training entry point
│   ├── train_b04.py                 B-04 end-to-end: FDR + Decoder training
│   └── fdr_predictor.py             FDR wrapper (frozen MV3 + DensityHead)
├── paper_a/                         # Paper A experiments (main branch)
│   ├── eval_e007b_proto_vs_embedding.py   Proto vs Embedding fair comparison
│   ├── eval_e008_spm_sparsity.py          SPM sparsity validation (A/B/C)
│   ├── eval_e009_spm_router.py            Learned vs Fixed Router
│   ├── eval_e009d_proto_usage.py          Effective Proto count analysis
│   ├── eval_e009_verify.py                Router verification
│   ├── eval_e010_isaid_mc.py              iSAID multi-class Proto vs Embedding
│   ├── eval_e011_spm_isaid.py             SPM on iSAID
│   ├── eval_e011t_tile_ablation.py        Tile size ablation (256-2048)
│   └── eval_e011u_proto_capacity.py       Proto count scanning (2-64)
├── paper_b/                         # Paper B experiments (paper-b branch)
│   ├── eval_b00_tile_size_sensitivity.py   Spatial Sparsity: 7 tile sizes
│   ├── eval_b01_oracle_topk.py             Oracle Top-K: FG retention, SSI definition
│   ├── eval_b01_spatial_baseline.py        Tile foreground distribution analysis
│   ├── eval_b02_learnability.py            Learnability: MV3 predict importance (r=0.889)
│   ├── eval_b02_5_generalization.py        Generalization: category-agnostic, cross-dataset
│   ├── eval_b03_router_architecture.py     FDR vs Edge ablation: R0/R1/R2/R3
│   ├── eval_b05_oracle_importance.py       Oracle tile importance: multi-metric analysis
│   ├── eval_b05_5_tile_size.py             Tile size x importance interaction
│   ├── eval_b06_contribution_granularity.py  Contribution imbalance vs tile granularity
│   ├── eval_b06_decoder_diag.py            Decoder diagnostics: per-class IoU
│   ├── eval_b07_contribution_gt.py          GT contribution analysis
│   ├── eval_b07_fdr_data_efficiency.py      FDR sample efficiency (1%-100% data)
│   ├── eval_b08_fastsam_fewshot.py          FastSAM-FSS: few-shot on dense land-cover (LoveDA/Vaihingen)
│   ├── eval_b09_nwpu_fewshot.py             FastSAM-FSS on NWPU-VHR-10 (bbox weak masks)
│   └── eval_paper_b_pipeline.py             Unified Paper B pipeline (multi-dataset)
├── instance/                        # C-series few-shot instance segmentation
│   ├── eval_fastsam_zero_shot.py          FastSAM zero-shot baseline (mR@50≈41.5%)
│   ├── run_c01_sweep.py                   Hyperparameter sweep runner (max_det×conf)
│   ├── eval_c02a_fastsam_fewshot.py       Proto Matching baseline (1-shot mIoU=0.31%, proves P4≠proto)
│   ├── eval_c02b_decoder_fewshot.py       Proto + Refine CNN (1-shot≈0.5%, training collapse)
│   ├── eval_c03_catsam_fewshot.py         ★ Cross-Attn + Tile (1-shot mIoU=32.7%, 118× lift; ACTIVE)
│   ├── eval_c04_full_fewshot.py           ★ Full 15-class few-shot instance seg (Phase D; ACTIVE)
├── diag/                            # Diagnostics
│   ├── diag_b04_tiles.py               Tile dataset: mask values, fg_ratio
│   ├── diag_b04_overfit.py             Overfit test (20 tiles x 100 epoch)
│   ├── diag_b04_exp12.py               Exp1 (FG>5% multi-class) + Exp2 (binary)
│   ├── diag_class_stats.py             COCO GT stats + tile stats + anomaly detection
│   ├── diag_check_labels.py            Quick train/val label space consistency check
│   ├── diag_trace_labels.py            Single-instance mapping chain trace
│   └── test_loader.py                  Dataset loader validation
└── viz/                             # Visualization
    ├── viz_paper_a_p6.py               P6 feature visualization for Paper A
    └── viz_paper_a_router.py           Router behavior visualization for Paper A
```

## Key Lessons from v1 (MUST follow)

1. **YOLOv8 eval mode**: `model.train()` crashes YOLOv8 detect head. Keep eval mode + `requires_grad` control.
2. **Decoder-SPM decoupled**: Decoder always receives full features. SPM trained in parallel.
3. **Budget loss differentiable**: `(imp > 0.5).float().mean()` has zero gradient → use `(imp.mean − target)²`.
4. **SPM three pillars**: GT density focal + Top-K BCE + budget loss. Missing any → importance collapse.
5. **Episodic training**: Baseline MUST also use episodic training for fair comparison.
6. **Dice GT broadcast**: `unsqueeze(0)` with batch>1 → `[1,B,H,W]` broadcast explosion.

## Paper B Architecture

```
Paper B evidence chain:

B-00: Tile Size Sensitivity       -> Spatial Sparsity EXISTS. All scales -- 60% empty at 1024px.
B-01: Oracle Top-K                 -> Upper bound: Top40% tiles -> 96.5% FG, IDG=2.41x. Defines SSI.
B-02: Learnability                 -> Importance IS LEARNABLE: Spearman r=0.889 (MV3 backbone).
B-02.5: Generalization             -> Category-AGNOSTIC (holdout r=0.651), cross-dataset possible.
B-03: Router Architecture          -> FDR 75K ~= R0 1.48M (Dr=-0.038). Edge != Importance (+0.009 only).
B-04: End-to-End Integration       -> Decoder verified (val_fg5~0.47, E13). FDR training complete.
B-05: Oracle Importance            -> What defines "important" tiles? Multi-metric analysis.
B-06: Contribution Granularity     -> Contribution imbalance across tile sizes + decoder diagnostics.
B-07: FDR Data Efficiency          -> 1%-100% data scaling, FDR-SES score.
B-08: FastSAM Few-Shot (Dense Land-Cover)  -> LoveDA/Vaihingen: FastSAM + Prototype + Decoder, K=1/3/5.
B-09: FastSAM Few-Shot (Instance)  -> NWPU-VHR-10: bbox weak masks, K-shot prototype matching.
C-01->C-04: Full Few-Shot Pipeline -> iSAID 15-class instance seg, CAT-SAM baseline, Phase D sweep.
```

**C-series key results (see `RESEARCH_MAP.md` for full details):**
- **C-01 FastSAM Zero-Shot**: mR@50≈41.5% (recall upper bound), bottleneck = max_det, not conf
- **C-02A Proto Matching**: 1-shot mIoU=0.31% — proves P4×Proto matching mechanism fails, NOT that P4 lacks info
- **C-03 Cross-Attention + Tile**: ★ 1-shot mIoU=32.7% (118× lift over C-02A). Shot saturation: 1≈3≈5 shot. Bottleneck: small_vehicle (11%), not few-shot count
- **C-04 Full 15-Class**: Running. Tests whether C-03's 3-class findings hold on all iSAID classes

**C-03 critical insight**: Full-image resize→P4 stride=16→small objects disappear→model learns "predict all background." **Tile is not an optimization — it's a necessity for FastSAM few-shot.**

**Active development files** (2026-06-24): `tools/instance/eval_c03_catsam_fewshot.py` and `tools/instance/eval_c04_full_fewshot.py` are actively being edited. New modules: `adatile/datasets/isaid_tile_wrapper.py` (full-image→tile with bbox overlap + LRU cache), `adatile/datasets/p4_cache.py` (P4 precompute), `adatile/utils/prototype.py` (canonical `compute_fg_prototype`).

**Paper B Laws (from B-00):**
1. **Spatial Sparsity**: All scales are sparse — even 2048×2048 has 49.9% empty tiles
2. **Foreground Concentration**: Top 17-48% tiles capture 95% FG (monotonic with tile size)
3. **Scale-Sparsity Trade-off**: Larger tile → lower sparsity, higher FG capture needed

**Paper B Core Hypothesis (post B-04 Decoder):**
FastSAM P4 carries sufficient semantic information (val_fg5≈0.47). The question is now:
> Can FDR reduce compute (Top-K% tiles) while preserving this ~0.47 mIoU?

**Key scientific value**: Per-class analysis of dynamic selection impact, especially on rare/long-tail classes (helicopter, bridge, pool). If K=40% drops overall mIoU by 1% but helicopter by 50%, this becomes a compelling analysis point about dynamic compute vs. long-tail fairness.

**Spatial Sparsity Index (SSI):**
- SSI = Oracle Top40% FG retention. Pre-experiment, zero-cost criterion.
- SSI > 70 → Router applicable (object-centric: iSAID, DOTA, xView)
- SSI < 50 → Router meaningless (land-cover: LoveDA, Potsdam)

**Foreground Density Router (FDR) — Paper B core module:**
```
Image → Frozen MV3 backbone → Feature Map → DensityHead (75K) → Importance Map → Top-K tiles
```
- Supervised by: `fg_ratio` (foreground density per tile) — NOT edges, NOT class labels
- Learns: objectness / instance density, category-agnostic
- `adatile/sparse/spatial_router.py` — `ForegroundDensityRouter`, `DensityHead`, `EdgeHead` (ablation only), `TinyCNNRouter` (lower-bound)

**B-04 LightDecoder (for binary/multi-class dense segmentation):**
```
P4 [B,1280,H/16,W/16] → Conv(1280→64) → Upsample×2 → Conv(64→64) → Upsample×2
                      → Conv(64→32) → Upsample×2 → Conv(32→32) → Upsample → Conv(32→1)
```
~800K params. See `adatile/decoder/light_decoder.py`.

**Critical B-04 findings (revised 2026-06-21):**
- **Double-mapping bug (ROOT CAUSE of val≈0.001)**: `prep_isaid.py` fixed annotations to standard ISAID IDs, but `render_category_mask()` applied `ACTUAL_TO_CODE_ID` a second time, permuting val class IDs into wrong label space. Train/val used different original numbering → single hardcoded table couldn't work for both. **Fixed**: per-split `build_mapping()` via name matching. After fix: E1 val_fg5=0.345, E13=0.472 — normal training curve.
- **FG>5% filter (real but secondary)**: FG>1% filter keeps 34% BG-dominated tiles → noise dilutes foreground signal. FG>5% → 12% meaningful tiles. True contribution: ~0.05-0.10 mIoU improvement, NOT the 0.001→0.801 jump.
- **Focal γ=5.0 + Dice**: For extreme class imbalance in remote sensing.
- **Rare class oversampling**: basketball/pool/helicopter ×5. Note: pre-fix class counts were corrupted by double-mapping (e.g., pool appeared as 24 tiles, actually 189 after fix). True rare classes post-fix: helicopter=14 tiles, pool=189, basketball=189.
- **Current Decoder capability (2026-06-21)**: train=0.757, val_fg5=0.472 (E13). 716K params, frozen FastSAM P4 only, single-scale. Hard ceiling ~0.50-0.55 due to frozen backbone limitation. Per-class weak spots: bridge=0.0, helicopter=0.09, pool=0.17 — genuine data scarcity, not bugs.

## Config & Tooling

- **pyproject.toml** is the single source of truth for: dependencies, pytest/ruff/mypy/black config, build system.
- **Line length**: 100 (ruff + black both configured).
- **Python**: ≥3.10.
- Dev install: `pip install -e ".[dev,viz]"` (includes pytest, ruff, mypy, black, wandb, matplotlib).
- Package metadata: `readme = "CLAUDE.md"` — this file IS the package description.
- CLI entry points (planned): `adatile-train`, `adatile-eval` (see `[project.scripts]`).

## Supplementary Docs

- **`RESEARCH_MAP.md`** — Comprehensive, up-to-date research map with C-series results, architecture diagrams, roadmap, and lessons learned. More detailed than this file.
- **`docs/c04_code_explanation.md`** — Line-by-line walkthrough of the C-04 full 15-class experiment script.

## Data Pipeline

```
iSAID COCO JSON                    Cityscapes
      │                                │
prep_isaid.py (fix annotations)  prep_cityscapes.py
      │                                │
prep_isaid_tiles.py ───────────────────┘
  ├── Step 1: render_category_mask() → masks_full/
  ├── Step 2: cut 1024×1024 tiles → images/ + masks/
  └── Step 3: metadata JSON → metadata/{split}.json
      │
FastISAIDTileDataset(root_dir, split, dense_labels=bool)
  → {"image": [3,1024,1024], "mask": [1024,1024], "image_id": str}
```

## Label Mapping (Critical)

**Mapping only happens ONCE in `prep_isaid.py`.** All downstream code uses `ann["category_id"]` directly.

Shared module: `adatile/utils/label_mapping.py` — `build_mapping()`, `ISAID_CATEGORIES`, `get_category_id()`. See module docstring for details.

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

### Tile preprocessing: all masks zero

If `prep_isaid_tiles.py --steps 2,3` skips Step 1, tile masks are all `unique=[0]`. Always run `--steps 1,2,3` or ensure `masks_full/` already exists.

### Decoder FG-mIoU stuck near 0 (train=0.71, val≈0.001)

**Root cause: Double category ID mapping.** `render_category_mask()` applied `ACTUAL_TO_CODE_ID` on already-mapped annotations, permuting val class IDs. Train/val used different label spaces → model correctly learned train classes but val labels were gibberish. **Fix**: per-split `build_mapping()` in `prep_isaid.py`, remove second mapping in all `render_category_mask()` calls. After fix: E1 val_fg5=0.345.

**Contributing factor**: FG>1% filter kept 34% BG-dominated tiles as noise. FG>5% filter → 12% meaningful tiles. Diagnosis: `tools/diag/diag_b04_exp12.py`, `tools/diag/diag_trace_labels.py`.

### Category label mismatch between train/val

iSAID train and val use different original category_id numbering. `prep_isaid.py` now uses per-split name-based mapping (`adatile/utils/label_mapping.py`). Diagnosis: `tools/diag/diag_trace_labels.py`.

### FileBackend data loss on crash

Fixed: `buffer_size=1`, `flush_interval=1.0` globally in `adatile/logging/backends.py`. Every record immediately written.

## Common Commands

```bash
# Dev install (full toolchain)
pip install -e ".[dev,viz]"

# Tests — all
pytest tests/ -v

# Tests — single file
pytest tests/test_logging.py -v

# Tests — single test function
pytest tests/test_logging.py::test_file_backend -v

# Tests — with coverage
pytest tests/ -v --cov=adatile --cov-report=term-missing

# Lint
ruff check adatile/

# Type check (optional)
mypy adatile/ --ignore-missing-imports

# Format
black adatile/ tests/

# Data preprocessing (full pipeline from scratch)
python tools/data/prep_isaid.py                         # Step 0: fix COCO JSON annotations
python tools/data/prep_isaid_tiles.py \                 # Steps 1-3: render masks → cut tiles → metadata
    --src-root data/iSAID_processed \
    --dst-root data/iSAID_tiles \
    --steps 1,2,3 --splits train,val

# Label validation
python tools/diag/diag_check_labels.py --tile-root data/iSAID_tiles
python tools/diag/diag_trace_labels.py --data-root data

# Dataset diagnostics
python tools/diag/diag_class_stats.py --isaid-root data/iSAID_processed --tile-root data/iSAID_tiles
python tools/diag/diag_b04_tiles.py --tile-root data/iSAID_tiles

# Overfit test (verify decoder can learn)
python tools/diag/diag_b04_overfit.py --tile-root data/iSAID_tiles

# Paper B experiments
python tools/paper_b/eval_b00_tile_size_sensitivity.py
python tools/paper_b/eval_b01_oracle_topk.py
python tools/paper_b/eval_b02_learnability.py

# B-04 end-to-end (local test)
python tools/train/train_b04.py --decoder-epochs 10 --fdr-epochs 5 --batch-size 4

# B-04 full run (cloud server, RTX 5090)
nohup python tools/train/train_b04.py \
    --src-root /root/autodl-tmp/iSAID_processed \
    --tile-root /root/autodl-tmp/iSAID_tiles \
    --decoder-epochs 50 --fdr-epochs 20 --batch-size 8 \
    > /root/autodl-tmp/b04.log 2>&1 &

# C-series few-shot experiments (current focus)
python tools/instance/eval_c03_catsam_fewshot.py    # Cross-Attention + Tile, 3-class
python tools/instance/eval_c04_full_fewshot.py      # Full 15-class
```

## Persistent Memory

Project memory stored at `C:\Users\20871\.claude\projects\E--A-postgraduate-stude-AdaTile-FastSAM\memory\`. Key files:
- `two-paper-strategy.md` — Paper A/B split rationale and publication targets
- `paper-b-evidence-chain.md` — Paper B complete theory chain: B-00→B-03 finalized
- `spatial-sparsity-index.md` — SSI definition, criterion values, dataset applicability
- `paper-a-final.md` — Paper A archive with file index and completion status
- `publication-strategy.md` — Journal selection, reviewer attack points, scoring
- `paper-positioning.md` — Related work analysis, overlap, differentiation
- Various v1 lessons (decoder-gradient, dice-broadcast, importance-collapse, etc.)
