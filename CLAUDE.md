# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdaTile-FastSAM: Adaptive Sparse Tiling for High-Resolution Few-Shot Instance Segmentation. A PyTorch research framework that trades compute for resolution by learning *where* to process — a sparse importance predictor (Ada-SPM) + adaptive tile tokenizer + 4-level token router (DTR-v2) + prototype-guided few-shot decoder.

## Common Commands

```bash
# Install (editable)
pip install -e .[dev,viz]

# Run tests
pytest tests/ -v
pytest tests/test_routing.py -v                    # single test file
pytest tests/test_sparse_prediction.py -v -k "test_adspm_importance"  # single test

# Benchmarking
python benchmark.py --quick                        # quick single-config (adaptive_tiling)
python benchmark.py --compare                      # compare all 4 configs
python benchmark.py --full                         # compare + chrome trace + plots
python benchmark.py --config hard_sparse --iters 20

# Real FLOPs counting (requires fvcore)
python -c "from adatile.modeling import build_adatile_fastsam; from adatile.config import Config; from fvcore.nn import FlopCountAnalysis; m = build_adatile_fastsam(Config()); print(FlopCountAnalysis(m, torch.randn(1,3,1024,1024)).total())"

# Training
python tools/train.py --config configs.default.get_default_config
python tools/train.py --config configs.isaid.get_isaid_config -o train.epochs=100 data.batch_size=4
python tools/train.py --config configs/isaid.yaml --resume checkpoints/isaid/

# Type checking & linting
mypy adatile/
ruff check adatile/
```

## Architecture

### Trusted Third-Party Foundation

The framework is built on verified external libraries — only the novel sparse tiling components are custom:

| Component | Library | File |
|-----------|---------|------|
| Backbone | `timm` | `adatile/backbone/base.py` |
| FPN | `torchvision.ops.FeaturePyramidNetwork` | `adatile/sparse/fpn_fusion.py` (thin wrapper) |
| Full Attention | `F.scaled_dot_product_attention` (FlashAttention-2) | `adatile/routing/attention.py` |
| Efficient Attention | windowed / block-sparse via SDPA | `adatile/routing/attention.py` |
| Decoder | FastSAM-inspired (Proto + coefficient head) | `adatile/decoder/base.py` |
| NMS | `torchvision.ops.nms` / `batched_nms` | `adatile/decoder/base.py` |
| Profiling | `torch.profiler` + `fvcore` | `adatile/profiling/pipeline_profiler.py` |

### Custom Innovation (the only custom code)

| Module | What it does |
|--------|-------------|
| **Ada-SPM** | Predicts per-region importance + tile granularity from backbone features |
| **Dynamic Tile Planner** | Converts importance map → concrete tile plan (3 skip modes + quadtree) |
| **DTR-v2 Router** | Differentiable 4-level token routing with budget control |
| **Prototype Memory** | Few-shot: masked-average-pooling prototypes + cosine similarity retrieval |

### Pipeline (top to bottom)

```
timm Backbone (e.g. resnet50, swin_tiny) → multi-scale features
    ↓
torchvision FPN (unify channel dims)
    ↓
Ada-SPM → SparsePrediction(importance, density, granularity)  [CUSTOM]
    ↓
Dynamic Tile Planner → TilePlan(specs, skip_ratio, stats)      [CUSTOM]
    ↓
Token Generator → tile patches + 2D positional encoding
    ↓
DTR-v2 Router → RoutingOutput(routed, assignments, skip_mask)  [CUSTOM]
    ↓
FastSAM-style Decoder → SegmentationOutput(masks, scores)
    ↓
Prototype Guidance (few-shot mode)                              [CUSTOM]
```

Profiling: `torch.profiler.profile()` for kernel timing, `fvcore.nn.FlopCountAnalysis` for verified FLOPs.

### Six Core ABCs (`adatile/core/base.py`)

All components inherit from these abstract interfaces, which define the contract. Key typed data structures defined here:
- `SparsePrediction` (frozen dataclass, validated in `__post_init__`): importance [B,1,H,W], density [B,1,H,W], optional granularity
- `RoutingOutput`: routed_tokens, assignments, routing_weights, skipped_mask, aux_loss
- `SegmentationOutput`: masks, scores, boxes, classes

### Registry System (`adatile/registry/registry.py`)

12 registries: `BACKBONE`, `SPARSE`, `TOKENIZER`, `ROUTER`, `DECODER`, `PROTOTYPE`, `SEGMENTATION`, `DATASET`, `TRANSFORM`, `TRAINER`, `LOSS`, `METRIC`. Components register via decorator (`@ROUTER.register()`) and are built via `build_*()` factory functions (e.g., `build_router("DTRv2Router", embed_dim=256)`).

### Config System (`adatile/config/config.py`)

Dataclass-based, immutable-friendly. `Config` nests 9 sub-configs (BackboneConfig, SparseConfig, TokenizerConfig, RouterConfig, PrototypeConfig, DecoderConfig, DataConfig, TrainConfig, EvalConfig). Supports `from_dict()`, `from_yaml()`, `from_json()`, `to_dict()`, `to_yaml()`. Dot-notation overrides at CLI (`-o train.epochs=100`).

### Ada-SPM (`adatile/sparse/ada_spm.py`)

The core innovation. Input: multi-scale backbone features. Output: `SparsePrediction`. Architecture: FPN fusion → optional SpatialTransformerRefine (windowed self-attention) → DensityHead (conv → sigmoid) + GranularityHead (conv → Gumbel-Softmax over K tile-size categories). Variants: `AdaSPM`, `AdaSPMLite` (no transformer), `AdaSPMFull` (larger dims), `DensityOnlySPM` (ablation, no granularity).

### Tile Planner (`adatile/tokenizer/tile_planner.py`)

Converts importance map → concrete tile extraction plan. Three skip modes:
- `"threshold"`: skip if importance < 0.5×threshold, borderline → coarse tiles (size 3072)
- `"hard"`: skip if importance < threshold (no borderline tiles)
- `"topk"`: keep top-K×100% cells by importance, independent of threshold

Also supports `plan_quadtree()` for recursive region splitting. Outputs `PlannerStats` with real sparsity statistics (skipped_cells, flops/memory saved).

### DTR-v2 Router (`adatile/routing/router.py`)

4-level routing: 0=skip, 1=windowed_attn, 2=block_sparse_attn, 3=full_attn. Pipeline: RoutingHead (MLP → 4-class logits) → optional PrototypeRouter bias → BudgetController (differentiable logit biasing, NOT post-hoc assignment) → Gumbel-Softmax (straight-through gradient) → token sparsification → MultiLevelAttention. Attention backends use `F.scaled_dot_product_attention` (FlashAttention-2 dispatch) with window/block-sparse masks. Other routers: `UniformRouter` (all to same level, ablation), `IdentityRouter` (pass-through, baseline).

### Training (`adatile/engine/trainer.py`, `tools/train.py`)

Hook-based training loop (inspired by Detectron2). Hooks: `LRSchedulerHook`, `LoggingHook`, `CheckpointHook`, `EvalHook`, `TensorBoardHook`. Supports mixed precision (fp16/bf16), gradient accumulation, distributed training (DDP), and checkpoint resume. Config loaded from YAML files or Python module functions.

### Benchmark Suite (`benchmark.py`, `adatile/profiling/`)

4 pre-defined configurations compared head-to-head: `fixed_tiling` (baseline, uniform tiles, identity router), `adaptive_tiling` (dynamic tiles, DTRv2), `no_sparsity` (dynamic tiles, identity router), `sparse_routing` (hard skip, DTRv2). Profiling uses `torch.profiler.profile()` for kernel-level GPU timing and `fvcore.nn.FlopCountAnalysis` for verified FLOPs counting. Results exportable as CSV, JSON, Chrome trace, and comparison plots.

### Dataset Architecture

COCO-style annotations with few-shot split support. Key features:
- Density map generation: GT instances → [H/stride, W/stride] float32 array for Ada-SPM supervision
- `FewShotSplit`: fixed split format (`novel_classes`, `support_images`, `query_images`)
- `BaseDataset`: abstract with polygon/RLE → mask conversion, class distribution stats
- Pre-computed tile cache, prototype cache, and density maps (see memory: [[dataset-architecture]])

### Config files

- `configs/default.py` — base COCO config (1024², batch_size=8)
- `configs/isaid.py` — iSAID aerial imagery (2048², batch_size=4, longer training)
- `configs/fewshot/one_shot.py` — 1-shot experimental config

### Key module layout

| Module | Purpose |
|--------|---------|
| `adatile/core/` | ABCs + typed data structures |
| `adatile/config/` | Dataclass config system |
| `adatile/registry/` | Decorator-based module registry |
| `adatile/sparse/` | Ada-SPM + torchvision FPN wrapper |
| `adatile/tokenizer/` | Dynamic tile planner + token generator + budget tracker |
| `adatile/routing/` | DTR-v2 router + SDPA-backed multi-level attention |
| `adatile/backbone/` | timm-backed backbones (`TimmBackbone`, `ResNet50Backbone`) |
| `adatile/decoder/` | FastSAM-inspired decoder (Proto + coefficient + torchvision NMS) |
| `adatile/prototype/` | Masked-average-pooling prototype memory |
| `adatile/segmentation/` | Full pipeline + loss functions |
| `adatile/modeling/` | `AdaTileFastSAM` wrapper + `build_adatile_fastsam()` entry point |
| `adatile/datasets/` | COCO/iSAID/LoveDA datasets, tile cache, few-shot sampler |
| `adatile/engine/` | Trainer + hooks |
| `adatile/evaluation/` | COCO evaluator + few-shot metrics + sparse efficiency |
| `adatile/profiling/` | torch.profiler-based pipeline profiler + fvcore FLOPs |
| `adatile/utils/` | Logging, checkpoint, distributed utils |
| `adatile/visualization/` | Sparsity + rendering visualizations |
| `tools/` | Entry scripts: train.py, eval.py, build_tile_cache.py |
