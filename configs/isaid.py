"""iSAID-optimized configuration for high-resolution aerial instance segmentation."""

from adatile.config import Config
from configs.default import get_default_config


def get_isaid_config() -> Config:
    """Return configuration tuned for iSAID.

    Key differences from default:
        - Higher input resolution (2048 vs 1024)
        - Larger tile sizes (up to 2048 for coarse tiles)
        - Lower batch size (GPU memory)
        - Longer training (more epochs, higher max_steps)
        - Density-aware sparse sampling
    """
    cfg = get_default_config()

    cfg.experiment_name = "adatile_isaid"
    cfg.output_dir = "output/isaid"

    # ── Data ─────────────────────────────────────────────────────
    cfg.data.name = "isaid"
    cfg.data.root_dir = "datasets/iSAID"
    cfg.data.img_size = (2048, 2048)
    cfg.data.max_size = 4096
    cfg.data.batch_size = 4
    cfg.data.num_workers = 8

    # ── Tokenizer (high-res tiles) ───────────────────────────────
    cfg.tokenizer.tile_sizes = [384, 768, 1536, 2048]
    cfg.tokenizer.stride_ratios = [0.5, 0.5, 0.5, 0.5]
    cfg.tokenizer.max_tokens_per_image = 8192

    # ── Sparse ───────────────────────────────────────────────────
    cfg.sparse.min_tile_ratio = 0.05
    cfg.sparse.importance_threshold = 0.4

    # ── Training ─────────────────────────────────────────────────
    cfg.train.epochs = 100
    cfg.train.lr = 5e-5
    cfg.train.weight_decay = 0.1
    cfg.train.max_steps = 100000
    cfg.train.warmup_steps = 1000
    cfg.train.checkpoint_dir = "checkpoints/isaid"
    cfg.train.log_interval = 20
    cfg.train.eval_interval = 2000
    cfg.train.save_interval = 10000

    # ── Evaluation ───────────────────────────────────────────────
    cfg.eval.iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    return cfg
