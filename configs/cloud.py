"""Cloud GPU training config — optimized for A100 / RTX 4090.

BF16 mode for numerical stability (no NaN issues),
larger batch size and token budget.
"""

from adatile.config import Config
from configs.fewshot.one_shot import get_1shot_config


def get_cloud_config() -> Config:
    """Cloud-optimized 1-shot config.

    Key differences from local 6GB GPU:
        - BF16 instead of FP16 (no overflow → no NaN guards needed)
        - batch_size 4-8 (faster throughput)
        - max_tokens 256 (more tiles → better coverage)
        - num_workers 4 (faster data loading)
    """
    cfg = get_1shot_config()

    cfg.experiment_name = "adatile_cloud_1shot"
    cfg.output_dir = "output/cloud_1shot"

    # ── Precision: BF16 is stable on Ampere+ ──────────────────
    cfg.train.mixed_precision = "bf16"

    # ── Larger batch ──────────────────────────────────────────
    cfg.data.batch_size = 4
    cfg.data.num_workers = 4
    cfg.data.pin_memory = True

    # ── More tokens → better coverage ─────────────────────────
    cfg.tokenizer.max_tokens_per_image = 256

    # ── Sparsity ──────────────────────────────────────────────
    cfg.sparse.importance_threshold = 0.15

    # ── Router ────────────────────────────────────────────────
    cfg.router.max_full_ratio = 0.20
    cfg.router.max_skip_ratio = 0.60

    # ── Training ──────────────────────────────────────────────
    cfg.train.epochs = 200
    cfg.train.lr = 1e-4
    cfg.train.max_steps = 200000
    cfg.train.checkpoint_dir = "checkpoints/cloud_1shot"

    return cfg
