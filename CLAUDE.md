# CLAUDE.md — AdaTile-FastSAM v2

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AdaTile-FastSAM**: Adaptive Sparse FastSAM for Few-Shot High-Resolution Instance Segmentation.

**Two-Paper Strategy (2026-06-19):**
- **Paper A** (Proto Sparsity / Learned Sparse Proto Routing): Archived on `main`. Evidence chain E007→E011-U complete. ICIP/CCIG target.
- **Paper B** (Dual Sparsity / Spatial Sparsity / AdaTile): Active on `paper-b`. Observation → Oracle → Learnability → Method pipeline.

**Core innovations:**
1. **Ada-SPM** — density-supervised sparse perception module: learns importance maps → Top-K tile selection
2. **Decoupled Sparse Training** — decoder always receives full features; SPM trained via GT-driven losses in parallel

## Git Branches

```
main      → Paper A archive (all E-series experiments)
paper-b   → Paper B active development (B-series experiments)
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

### 2. Bilingual Comments (中英文注释)

Every file, class, function, and non-obvious logic block must have Chinese + English bilingual comments.

### 3. Test-Driven

Each module: write tests first, then implementation. Tests must verify logging output.

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
├── decoder/         ✅ LinearProbe, FusionProbe, LightDecoder
├── datasets/
│   ├── base.py              ✅ BaseSegDataset
│   ├── mass_buildings.py    ✅ MassBuildings tile dataset
│   ├── isaid.py             ✅ iSAID COCO dataset (full-image)
│   └── isaid_tiles.py       ✅ FastISAIDTileDataset (pre-cut 1024×1024 tiles)
├── sparse/           ⬜ Ada-SPM
├── losses/           ⬜ Loss functions
└── utils/
    └── seed.py       ✅ Unified set_seed() with cuDNN deterministic

tools/
├── prep_isaid.py            iSAID COCO → category-id masks (Step 0)
├── prep_isaid_tiles.py      Full pipeline: render mask → cut tiles → metadata
├── prep_cityscapes.py       Cityscapes → tile format
├── train_isaid_multiclass.py iSAID multi-class training entry point
├── test_isaid_loader.py     Dataset loader validation
│
├── Paper A experiments (main branch):
│   eval_e007b_proto_vs_embedding.py   Proto vs Embedding fair comparison
│   eval_e008_spm_sparsity.py          SPM sparsity validation (A/B/C)
│   eval_e009_spm_router.py            Learned vs Fixed Router
│   eval_e009d_proto_usage.py          Effective Proto count analysis
│   eval_e009_verify.py                Router verification
│   eval_e010_isaid_mc.py              iSAID multi-class Proto vs Embedding
│   eval_e011_spm_isaid.py             SPM on iSAID
│   eval_e011t_tile_ablation.py        Tile size ablation (256-2048)
│   eval_e011u_proto_capacity.py       Proto count scanning (2-64)
│
└── Paper B experiments (paper-b branch):
    eval_b00_tile_size_sensitivity.py   Spatial Sparsity: tile size vs empty ratio
    eval_b01_oracle_topk.py             Oracle Top-K: FG retention upper bound
    eval_b01_spatial_baseline.py        Tile foreground distribution analysis
    eval_b02_learnability.py            Learnability: can MobileNetV3 predict tile importance?
```

## Key Lessons from v1 (MUST follow)

1. **YOLOv8 eval mode**: `model.train()` crashes YOLOv8 detect head. Keep eval mode + `requires_grad` control.
2. **Decoder-SPM decoupled**: Decoder always receives full features. SPM trained in parallel.
3. **Budget loss differentiable**: `(imp > 0.5).float().mean()` has zero gradient → use `(imp.mean − target)²`.
4. **SPM three pillars**: GT density focal + Top-K BCE + budget loss. Missing any → importance collapse.
5. **Episodic training**: Baseline MUST also use episodic training for fair comparison.
6. **Dice GT broadcast**: `unsqueeze(0)` with batch>1 → `[1,B,H,W]` broadcast explosion.

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

## Paper B Architecture

```
Paper B evidence chain:

B-00: Tile Size Sensitivity    → Spatial Sparsity exists (60% tiles empty at 1024px)
B-01: Oracle Top-K              → Upper bound (Top40% tiles → 96.5% FG retained, IDG=2.41x)
B-02: Learnability Study        → Can model learn tile importance?
        ↓ (if LEARNABLE)
B-03: DTR Network               → Full Dual-Tile-Router implementation
```

**Paper B Laws (from B-00):**
1. **Spatial Sparsity**: All scales are sparse — even 2048×2048 has 49.9% empty tiles
2. **Foreground Concentration**: Top 17-48% tiles capture 95% FG (monotonic with tile size)
3. **Scale-Sparsity Trade-off**: Larger tile → lower sparsity, higher FG capture needed

## Data Pipeline

```
iSAID COCO JSON                    Cityscapes
      │                                │
prep_isaid.py (render masks)    prep_cityscapes.py
      │                                │
prep_isaid_tiles.py ───────────────────┘
  ├── Step 1: render_semantic_mask() → masks_full/
  ├── Step 2: cut 1024×1024 tiles → images/ + masks/
  └── Step 3: metadata JSON → metadata/{split}.json
      │
FastISAIDTileDataset(root_dir, split, semantic=bool)
  → {"image": [3,1024,1024], "mask": [1024,1024], "image_id": str}
```

## Common Commands

```bash
# Install
pip install -e .

# Tests
pytest tests/ -v

# Lint
ruff check adatile/
```

## Persistent Memory

Project memory stored at `C:\Users\20871\.claude\projects\E--A-postgraduate-stude-AdaTile-FastSAM\memory\`. Key files:
- `two-paper-strategy.md` — Paper A/B split rationale
- `paper-a-final.md` — Paper A archive with file index and completion status
- Various v1 lessons and bug records (decoder-gradient, dice-broadcast, importance-collapse, etc.)
