"""Default training configuration for AdaTile-FastSAM.

Base config with reasonable defaults for COCO-style instance segmentation.
Override in dataset-specific configs.
"""

from adatile.config import Config, get_default_config as _base


def get_default_config() -> Config:
    """Return the default configuration for AdaTile-FastSAM.

    Suitable for COCO 2017 instance segmentation.
    """
    cfg = _base()

    # ── Experiment ───────────────────────────────────────────────
    cfg.experiment_name = "adatile_fastsam_default"
    cfg.output_dir = "output/default"

    # ── Backbone: FastSAM ViT-det ────────────────────────────────
    cfg.backbone.name = "fastsam_vit_b"
    cfg.backbone.pretrained = True
    cfg.backbone.embed_dim = 768
    cfg.backbone.depth = 12
    cfg.backbone.num_heads = 12
    cfg.backbone.patch_size = 16
    cfg.backbone.output_scales = [4, 8, 16, 32]

    # ── Ada-SPM ──────────────────────────────────────────────────
    cfg.sparse.name = "ada_spm"
    cfg.sparse.num_scales = 4
    cfg.sparse.importance_threshold = 0.5
    cfg.sparse.density_loss_weight = 1.0
    cfg.sparse.sparse_loss_weight = 0.1

    # ── Tile Tokenizer ───────────────────────────────────────────
    cfg.tokenizer.name = "dynamic_tile"
    cfg.tokenizer.tile_sizes = [384, 768, 1536]
    cfg.tokenizer.stride_ratios = [0.5, 0.5, 0.5]
    cfg.tokenizer.max_tokens_per_image = 4096
    cfg.tokenizer.skip_mode = "threshold"
    cfg.tokenizer.hard_skip_multiplier = 1.0

    # ── Router ───────────────────────────────────────────────────
    cfg.router.name = "DTRv2Router"
    cfg.router.max_full_ratio = 0.25
    cfg.router.max_skip_ratio = 0.15
    cfg.router.min_linear_ratio = 0.20

    # ── Decoder ──────────────────────────────────────────────────
    cfg.decoder.name = "fastsam_decoder"
    cfg.decoder.mask_dim = 256
    cfg.decoder.num_mask_tokens = 4
    cfg.decoder.iou_prediction = True

    # ── Prototype (few-shot only) ────────────────────────────────
    cfg.prototype.name = "masked_avg"
    cfg.prototype.prototype_dim = 256
    cfg.prototype.temperature = 0.1

    # ── Data ─────────────────────────────────────────────────────
    cfg.data.name = "coco"
    cfg.data.root_dir = "datasets/COCO"
    cfg.data.img_size = (1024, 1024)
    cfg.data.batch_size = 8
    cfg.data.num_workers = 4

    # ── Training ─────────────────────────────────────────────────
    cfg.train.epochs = 50
    cfg.train.lr = 1e-4
    cfg.train.weight_decay = 0.05
    cfg.train.max_steps = 50000
    cfg.train.warmup_steps = 500
    cfg.train.mixed_precision = "fp16"
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.max_grad_norm = 1.0
    cfg.train.log_interval = 50
    cfg.train.eval_interval = 1000
    cfg.train.save_interval = 5000
    cfg.train.checkpoint_dir = "checkpoints/default"
    cfg.train.seed = 42

    return cfg
