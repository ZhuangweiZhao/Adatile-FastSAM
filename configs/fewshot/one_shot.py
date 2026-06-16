"""1-shot configuration for AdaTile-FastSAM few-shot instance segmentation."""

from adatile.config import Config
from configs.default import get_default_config


def get_1shot_config() -> Config:
    """Return 1-shot few-shot segmentation configuration.

    Key differences:
        - Few-shot split loading enabled
        - Prototype memory active
        - Episodic training with N-way K-shot sampling
        - Support/query dual forward
    """
    cfg = get_default_config()

    cfg.experiment_name = "adatile_fewshot_1shot"
    cfg.output_dir = "output/fewshot_1shot"

    # ── Data ─────────────────────────────────────────────────────
    cfg.data.name = "isaid"
    cfg.data.root_dir = "datasets/iSAID"
    cfg.data.img_size = (2048, 2048)
    cfg.data.num_workers = 0   # single-process loading avoids CUDA pin_memory race
    cfg.data.pin_memory = False

    # ── Sparse — lower threshold to avoid skipping everything ─────
    cfg.sparse.importance_threshold = 0.15   # low enough that typical importance (0.3-0.5)
                                               # produces tiles; was 0.5 default → Tiles=0

    # ── Tokenizer (GPU-friendly for 6GB) ─────────────────────────
    cfg.tokenizer.max_tokens_per_image = 64  # 64 for 6GB GPU; use 128+ on 12GB+
    cfg.tokenizer.skip_mode = "threshold"     # keeps borderline cells as coarse tiles

    # ── Few-Shot Split ───────────────────────────────────────────
    cfg.data.fewshot_split = "datasets/iSAID/fewshot_splits/1shot/split0.json"
    cfg.data.n_shot = 1
    cfg.data.n_way = 3
    cfg.data.batch_size = 1  # One episode per batch

    # ── Prototype ────────────────────────────────────────────────
    cfg.prototype.name = "masked_avg"
    cfg.prototype.prototype_dim = 256
    cfg.prototype.temperature = 0.1
    cfg.prototype.num_prototypes_per_class = 1

    # ── Router (prototype-guided) ────────────────────────────────
    cfg.router.name = "DTRv2Router"

    # ── Training ─────────────────────────────────────────────────
    cfg.train.epochs = 200
    cfg.train.lr = 1e-4
    cfg.train.max_steps = 200000
    cfg.train.warmup_steps = 2000
    cfg.train.eval_interval = 5000
    cfg.train.checkpoint_dir = "checkpoints/fewshot_1shot"

    # ── Evaluation ───────────────────────────────────────────────
    cfg.eval.metrics = ["fewshot_iou", "fewshot_fb_iou"]

    return cfg
