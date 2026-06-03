"""10-shot few-shot segmentation configuration for iSAID.

Addresses Reviewer Q3: 10-shot evaluation with multi-seed support.
"""
from adatile.config import Config
from configs.default import get_default_config


def get_10shot_config(seed: int = 42) -> Config:
    cfg = get_default_config()

    cfg.experiment_name = "adatile_10shot_seed{}".format(seed)
    cfg.output_dir = "output/fewshot/10shot/seed{}".format(seed)

    cfg.data.name = "isaid"
    cfg.data.root_dir = "datasets/iSAID"
    cfg.data.img_size = (2048, 2048)
    cfg.data.batch_size = 1
    cfg.data.n_shot = 10
    cfg.data.n_way = 3
    cfg.data.fewshot_split = "datasets/iSAID/fewshot_splits/10shot/split0.json"

    cfg.sparse.name = "ada_spm"
    cfg.sparse.num_scales = 4
    cfg.sparse.importance_threshold = 0.4

    cfg.tokenizer.name = "dynamic_tile"
    cfg.tokenizer.tile_sizes = [384, 768, 1536, 2048]
    cfg.tokenizer.max_tokens_per_image = 8192

    cfg.router.name = "DTRv2Router"

    cfg.prototype.name = "masked_avg"
    cfg.prototype.prototype_dim = 256
    cfg.prototype.temperature = 0.1
    cfg.prototype.num_prototypes_per_class = 10

    cfg.decoder.name = "fastsam_decoder"

    cfg.train.epochs = 200
    cfg.train.lr = 1e-4
    cfg.train.max_steps = 200000
    cfg.train.warmup_steps = 2000
    cfg.train.seed = seed
    cfg.train.eval_interval = 5000
    cfg.train.save_interval = 10000
    cfg.train.checkpoint_dir = "checkpoints/fewshot/10shot/seed{}".format(seed)

    cfg.eval.metrics = ["fewshot_iou", "fewshot_fb_iou", "coco_mask_ap"]

    return cfg
