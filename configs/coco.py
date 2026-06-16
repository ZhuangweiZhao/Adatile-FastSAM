"""COCO 2017 training config for AdaTile-FastSAM.

COCO 2017 directory layout:
    datasets/COCO/
    ├── images/{train2017,val2017}/
    └── annotations/instances_{train2017,val2017}.json

Usage:
    python tools/train.py --config configs.coco.get_coco_config
"""

from adatile.config import Config


def get_coco_config() -> Config:
    """COCO 2017 instance segmentation config."""
    cfg = Config()

    cfg.experiment_name = "adatile_coco"
    cfg.output_dir = "output/coco"

    # ── Data: COCO 2017 ────────────────────────────────────────
    cfg.data.name = "coco"
    cfg.data.root_dir = "datasets/COCO"
    cfg.data.split_train = "train2017"
    cfg.data.split_val = "val2017"
    cfg.data.img_size = (1024, 1024)
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False

    # ── Sparse ─────────────────────────────────────────────────
    cfg.sparse.importance_threshold = 0.15

    # ── Tokenizer ──────────────────────────────────────────────
    cfg.tokenizer.max_tokens_per_image = 64
    cfg.tokenizer.skip_mode = "threshold"

    # ── Router ─────────────────────────────────────────────────
    cfg.router.name = "DTRv2Router"

    # ── Training ───────────────────────────────────────────────
    cfg.train.epochs = 1
    cfg.train.max_steps = 500
    cfg.train.lr = 1e-4
    cfg.train.mixed_precision = "fp16"
    cfg.train.checkpoint_dir = "checkpoints/coco"
    cfg.train.log_interval = 10

    return cfg
