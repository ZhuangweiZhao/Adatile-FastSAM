# AGENTS.md â€?AdaTile-FastSAM v2

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**AdaTile-FastSAM**: Adaptive Sparse FastSAM for Few-Shot High-Resolution Instance Segmentation.

**Two-Paper Strategy (2026-06-21):**
- **Paper A** (Proto Sparsity / Learned Sparse Proto Routing): Archived on `main`. Evidence chain E007â†’E011-U complete. ICIP/CCIG target.
- **Paper B** (Dual Sparsity / Spatial Sparsity / AdaTile): Active on `paper-b`. Theory chain B-00â†’B-03 CLOSED. B-04 end-to-end integration in progress.

**Core innovations:**
1. **Ada-SPM** â€?density-supervised sparse perception module: learns importance maps â†?Top-K tile selection (Paper A)
2. **Foreground Density Router (FDR)** â€?75K params, Pareto optimal spatial router: learns objectness/density, not edges or class semantics (Paper B)
3. **Decoupled Sparse Training** â€?decoder always receives full features; SPM/Router trained via GT-driven losses in parallel

## Git Branches

```
main      â†?Paper A archive (all E-series experiments)
paper-b   â†?Paper B active development (B-series experiments, B-04 in progress)
```

## Development Rules

### 1. Logging First (æ—¥å¿—å…ˆè¡Œ)

**ALL new code MUST route observable values through `adatile.logging`. No bare `print()`.**

```python
from adatile.logging import get_logger
logger = get_logger("module_name")
logger.log_metric("iou", 0.85, step=step, tags=["few-shot"])
logger.log_info("phase", "Stage B complete", step=step)
```

**FileBackend is crash-safe**: `buffer_size=1`, `flush_interval=1.0` â€?every record flushed to disk immediately.

### 2. Bilingual Comments (ä¸­è‹±æ–‡æ³¨é‡?

Every file, class, function, and non-obvious logic block must have Chinese + English bilingual comments.

### 3. Test-Covered

Each module: tests required before merge. Tests must verify logging output. Coverage is deliberately uneven: core library modules are well-tested; experiment/tools scripts are not.

### 4. Review Each Module

Complete one module â†?review â†?approve â†?next module. Do NOT batch multiple modules.

### 5. Reproducibility

All experiment scripts must call `set_seed()` from `adatile.utils.seed`. This sets Python/Random, NumPy, PyTorch, and cuDNN deterministic mode.

## Project Structure

```
adatile/
â”œâ”€â”€ logging/         âœ?Structured logging (Console, File/JSONL, Wandb backends)
â”œâ”€â”€ backbone/        âœ?FastSAMBackbone (hook P4/P8, eval-mode enforced, freeze control)
â”œâ”€â”€ config/          âœ?ExperimentConfig + ExperimentRecorder + generate_exp_id()
â”œâ”€â”€ metrics/         âœ?compute_miou, compute_dice, FPSMeter, count_params
â”œâ”€â”€ decoder/         âœ?LinearProbe, FusionProbe, LightDecoder
â”œâ”€â”€ datasets/
â”?  â”œâ”€â”€ base.py              âœ?BaseSegDataset
â”?  â”œâ”€â”€ mass_buildings.py    âœ?MassBuildings tile dataset
â”?  â”œâ”€â”€ isaid.py             âœ?iSAID COCO dataset (full-image)
â”?  â””â”€â”€ isaid_tiles.py       âœ?FastISAIDTileDataset (pre-cut 1024Ã—1024 tiles)
â”œâ”€â”€ sparse/
â”?  â””â”€â”€ spatial_router.py    âœ?ForegroundDensityRouter, DensityHead, EdgeHead, TinyCNNRouter
â”œâ”€â”€ losses/           â¬?Loss functions (skeleton only)
â””â”€â”€ utils/
    â””â”€â”€ seed.py       âœ?Unified set_seed() with cuDNN deterministic

tools/
â”œâ”€â”€ data/                            # Data preprocessing
â”?  â”œâ”€â”€ prep_isaid.py                iSAID COCO â†?category-id masks (Step 0)
â”?  â”œâ”€â”€ prep_isaid_tiles.py          Full pipeline: render mask â†?cut tiles â†?metadata
â”?  â”œâ”€â”€ prep_cityscapes.py           Cityscapes â†?tile format
â”?  â””â”€â”€ fix_labels.py                Repair tool: fix category ID mapping in instances JSON
â”?â”œâ”€â”€ train/                           # Training entry points
â”?  â”œâ”€â”€ train_isaid_mc.py            iSAID multi-class training entry point
â”?  â””â”€â”€ train_b04.py                 B-04 end-to-end: FDR + Decoder training
â”?â”œâ”€â”€ paper_a/                         # Paper A experiments (main branch)
â”?  â”œâ”€â”€ eval_e007b_proto_vs_embedding.py   Proto vs Embedding fair comparison
â”?  â”œâ”€â”€ eval_e008_spm_sparsity.py          SPM sparsity validation (A/B/C)
â”?  â”œâ”€â”€ eval_e009_spm_router.py            Learned vs Fixed Router
â”?  â”œâ”€â”€ eval_e009d_proto_usage.py          Effective Proto count analysis
â”?  â”œâ”€â”€ eval_e009_verify.py                Router verification
â”?  â”œâ”€â”€ eval_e010_isaid_mc.py              iSAID multi-class Proto vs Embedding
â”?  â”œâ”€â”€ eval_e011_spm_isaid.py             SPM on iSAID
â”?  â”œâ”€â”€ eval_e011t_tile_ablation.py        Tile size ablation (256-2048)
â”?  â””â”€â”€ eval_e011u_proto_capacity.py       Proto count scanning (2-64)
â”?â”œâ”€â”€ paper_b/                         # Paper B experiments (paper-b branch)
â”?  â”œâ”€â”€ eval_b00_tile_size_sensitivity.py   Spatial Sparsity: 7 tile sizes, empty/meaningful/FG-capture
â”?  â”œâ”€â”€ eval_b01_oracle_topk.py             Oracle Top-K: FG retention upper bound, SSI definition
â”?  â”œâ”€â”€ eval_b01_spatial_baseline.py        Tile foreground distribution analysis
â”?  â”œâ”€â”€ eval_b02_learnability.py            Learnability: can MV3 predict tile importance? (r=0.889)
â”?  â”œâ”€â”€ eval_b02_5_generalization.py        Generalization: category-agnostic? cross-dataset? (3 exps)
â”?  â””â”€â”€ eval_b03_router_architecture.py     FDR vs Edge ablation: R0/R1/R2/R3, Densityâ‰ Edge proof
â”?â”œâ”€â”€ diag/                            # Diagnostics (paper-b branch)
â”?  â”œâ”€â”€ diag_b04_tiles.py               Tile dataset: mask values, fg_ratio, class distribution
â”?  â”œâ”€â”€ diag_b04_overfit.py             Overfit test (20 tiles Ã— 100 epoch) + 5-panel visualization
â”?  â”œâ”€â”€ diag_b04_exp12.py               Exp1 (FG>5% multi-class) + Exp2 (binary FG/BG)
â”?  â”œâ”€â”€ diag_class_stats.py             COCO GT stats + tile stats + cross-validation + anomaly detection
â”?  â”œâ”€â”€ diag_check_labels.py            Quick train/val label space consistency check
â”?  â”œâ”€â”€ diag_trace_labels.py            Single-instance mapping chain trace (JSONâ†’maskâ†’tile)
â”?  â””â”€â”€ test_loader.py                  Dataset loader validation
â”?â””â”€â”€ viz/                             # Visualization
    â”œâ”€â”€ viz_paper_a_p6.py               P6 feature visualization for Paper A
    â””â”€â”€ viz_paper_a_router.py           Router behavior visualization for Paper A
```

## Key Lessons from v1 (MUST follow)

1. **YOLOv8 eval mode**: `model.train()` crashes YOLOv8 detect head. Keep eval mode + `requires_grad` control.
2. **Decoder-SPM decoupled**: Decoder always receives full features. SPM trained in parallel.
3. **Budget loss differentiable**: `(imp > 0.5).float().mean()` has zero gradient â†?use `(imp.mean âˆ?target)Â²`.
4. **SPM three pillars**: GT density focal + Top-K BCE + budget loss. Missing any â†?importance collapse.
5. **Episodic training**: Baseline MUST also use episodic training for fair comparison.
6. **Dice GT broadcast**: `unsqueeze(0)` with batch>1 â†?`[1,B,H,W]` broadcast explosion.

## Paper B Architecture

```
Paper B evidence chain (COMPLETE):

B-00: Tile Size Sensitivity       â†?Spatial Sparsity EXISTS. All scales â€?60% empty at 1024px.
B-01: Oracle Top-K                 â†?Upper bound: Top40% tiles â†?96.5% FG, IDG=2.41Ã—. Defines SSI.
B-02: Learnability                 â†?Importance IS LEARNABLE: Spearman r=0.889 (MV3 backbone).
B-02.5: Generalization             â†?Category-AGNOSTIC (holdout r=0.651), cross-dataset possible.
B-03: Router Architecture          â†?FDR 75K â‰?R0 1.48M (Î”r=âˆ?.038). Edge â‰?Importance (+0.009 only).
B-04: End-to-End Integration       â†?Decoder verified (val_fg5â‰?.47, E13). FDR training + dynamic selection eval in progress.
```

**Paper B Laws (from B-00):**
1. **Spatial Sparsity**: All scales are sparse â€?even 2048Ã—2048 has 49.9% empty tiles
2. **Foreground Concentration**: Top 17-48% tiles capture 95% FG (monotonic with tile size)
3. **Scale-Sparsity Trade-off**: Larger tile â†?lower sparsity, higher FG capture needed

**Paper B Core Hypothesis (post B-04 Decoder):**
FastSAM P4 carries sufficient semantic information (val_fg5â‰?.47). The question is now:
> Can FDR reduce compute (Top-K% tiles) while preserving this ~0.47 mIoU?

**Key scientific value**: Per-class analysis of dynamic selection impact, especially on rare/long-tail classes (helicopter, bridge, pool). If K=40% drops overall mIoU by 1% but helicopter by 50%, this becomes a compelling analysis point about dynamic compute vs. long-tail fairness.

**Spatial Sparsity Index (SSI):**
- SSI = Oracle Top40% FG retention. Pre-experiment, zero-cost criterion.
- SSI > 70 â†?Router applicable (object-centric: iSAID, DOTA, xView)
- SSI < 50 â†?Router meaningless (land-cover: LoveDA, Potsdam)

**Foreground Density Router (FDR) â€?Paper B core module:**
```
Image â†?Frozen MV3 backbone â†?Feature Map â†?DensityHead (75K) â†?Importance Map â†?Top-K tiles
```
- Supervised by: `fg_ratio` (foreground density per tile) â€?NOT edges, NOT class labels
- Learns: objectness / instance density, category-agnostic
- `adatile/sparse/spatial_router.py` â€?`ForegroundDensityRouter`, `DensityHead`, `EdgeHead` (ablation only), `TinyCNNRouter` (lower-bound)

**B-04 LightDecoder (for binary segmentation):**
```
P4 [B,1280,H/16,W/16] â†?Conv(1280â†?4) â†?UpsampleÃ—2 â†?Conv(64â†?4) â†?UpsampleÃ—2
                      â†?Conv(64â†?2) â†?UpsampleÃ—2 â†?Conv(32â†?2) â†?Upsample â†?Conv(32â†?)
```
~800K params. See `adatile/decoder/light_decoder.py`.

**Critical B-04 findings (revised 2026-06-21):**
- **Double-mapping bug (ROOT CAUSE of valâ‰?.001)**: `prep_isaid.py` fixed annotations to standard ISAID IDs, but `render_semantic_mask()` applied `ACTUAL_TO_CODE_ID` a second time, permuting val class IDs into wrong semantic space. Train/val used different original numbering â†?single hardcoded table couldn't work for both. **Fixed**: per-split `build_mapping()` via name matching. After fix: E1 val_fg5=0.345, E13=0.472 â€?normal training curve.
- **FG>5% filter (real but secondary)**: FG>1% filter keeps 34% BG-dominated tiles â†?noise dilutes foreground signal. FG>5% â†?12% meaningful tiles. True contribution: ~0.05-0.10 mIoU improvement, NOT the 0.001â†?.801 jump.
- **Focal Î³=5.0 + Dice**: For extreme class imbalance in remote sensing.
- **Rare class oversampling**: basketball/pool/helicopter Ã—5. Note: pre-fix class counts were corrupted by double-mapping (e.g., pool appeared as 24 tiles, actually 189 after fix). True rare classes post-fix: helicopter=14 tiles, pool=189, basketball=189.
- **Current Decoder capability (2026-06-21)**: train=0.757, val_fg5=0.472 (E13). 716K params, frozen FastSAM P4 only, single-scale. Hard ceiling ~0.50-0.55 due to frozen backbone limitation. Per-class weak spots: bridge=0.0, helicopter=0.09, pool=0.17 â€?genuine data scarcity, not bugs.

## Data Pipeline

```
iSAID COCO JSON                    Cityscapes
      â”?                               â”?prep_isaid.py (fix annotations)  prep_cityscapes.py
      â”?                               â”?prep_isaid_tiles.py â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”?  â”œâ”€â”€ Step 1: render_semantic_mask() â†?masks_full/
  â”œâ”€â”€ Step 2: cut 1024Ã—1024 tiles â†?images/ + masks/
  â””â”€â”€ Step 3: metadata JSON â†?metadata/{split}.json
      â”?FastISAIDTileDataset(root_dir, split, semantic=bool)
  â†?{"image": [3,1024,1024], "mask": [1024,1024], "image_id": str}
```

## Label Mapping (Critical)

**Mapping only happens ONCE in `prep_isaid.py`.** All downstream code uses `ann["category_id"]` directly.

Shared module: `adatile/utils/label_mapping.py` â€?`build_mapping()`, `ISAID_CATEGORIES`, `get_category_id()`. See module docstring for details.

## Known Issues & Workarounds

### FastSAM thirdLibrary PyTorch 2.x compatibility

`thirdLibrary/FastSAM/ultralytics/nn/modules/conv.py:297` â€?`torch.cat(x, self.d)` fails on PyTorch â‰?.0. Fixed to:
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

Full-size iSAID images (4000Ã—4000+) cause OOM on GPUs < 12GB. Use `--max-image-size 2048` or `--device cpu`.

### Tile preprocessing: all masks zero

If `prep_isaid_tiles.py --steps 2,3` skips Step 1, tile masks are all `unique=[0]`. Always run `--steps 1,2,3` or ensure `masks_full/` already exists.

### Decoder FG-mIoU stuck near 0 (train=0.71, valâ‰?.001)

**Root cause: Double category ID mapping.** `render_semantic_mask()` applied `ACTUAL_TO_CODE_ID` on already-mapped annotations, permuting val class IDs. Train/val used different semantic spaces â†?model correctly learned train classes but val labels were gibberish. **Fix**: per-split `build_mapping()` in `prep_isaid.py`, remove second mapping in all `render_semantic_mask()` calls. After fix: E1 val_fg5=0.345.

**Contributing factor**: FG>1% filter kept 34% BG-dominated tiles as noise. FG>5% filter â†?12% meaningful tiles. Diagnosis: `tools/diag/diag_b04_exp12.py`, `tools/diag/diag_trace_labels.py`.

### Category label mismatch between train/val

iSAID train and val use different original category_id numbering. `prep_isaid.py` now uses per-split name-based mapping (`adatile/utils/label_mapping.py`). Diagnosis: `tools/diag/diag_trace_labels.py`.

### FileBackend data loss on crash

Fixed: `buffer_size=1`, `flush_interval=1.0` globally in `adatile/logging/backends.py`. Every record immediately written.

## Common Commands

```bash
# Install
pip install -e .

# Tests
pytest tests/ -v

# Lint
ruff check adatile/

# Data preprocessing (full pipeline from scratch)
python tools/data/prep_isaid.py                         # Step 0: fix COCO JSON annotations
python tools/data/prep_isaid_tiles.py \                 # Steps 1-3: render masks â†?cut tiles â†?metadata
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
```

## Persistent Memory

Project memory stored at `C:\Users\20871\.Codex\projects\E--A-postgraduate-stude-AdaTile-FastSAM\memory\`. Key files:
- `two-paper-strategy.md` â€?Paper A/B split rationale and publication targets
- `paper-b-evidence-chain.md` â€?Paper B complete theory chain: B-00â†’B-03 finalized
- `spatial-sparsity-index.md` â€?SSI definition, criterion values, dataset applicability
- `paper-a-final.md` â€?Paper A archive with file index and completion status
- `publication-strategy.md` â€?Journal selection, reviewer attack points, scoring
- `paper-positioning.md` â€?Related work analysis, overlap, differentiation
- Various v1 lessons (decoder-gradient, dice-broadcast, importance-collapse, etc.)
