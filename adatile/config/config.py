"""Dataclass-style configuration system with YAML/JSON support.

Design principles:
- Dataclass-based: type-checked, IDE-friendly, no string-key lookups.
- Immutable by default: use `frozen=True` to prevent accidental mutation.
- Composable: top-level Config nests sub-configs for each module.
- Serializable: `to_dict()` / `from_dict()` for checkpointing and YAML round-trips.
"""

from __future__ import annotations

import json
import yaml
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ── Helpers ──────────────────────────────────────────────────────────

def _dict_to_dataclass(data: dict, cls: type) -> Any:
    """Recursively convert dict to dataclass instance."""
    if not is_dataclass(cls):
        return data
    kwargs = {}
    for f in fields(cls):
        key = f.name
        if key not in data:
            continue
        value = data[key]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[key] = _dict_to_dataclass(value, f.type)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass to dict."""
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = _dataclass_to_dict(value)
        return result
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


# ── Sub-Configs ──────────────────────────────────────────────────────

@dataclass
class BackboneConfig:
    """Backbone network configuration."""

    name: str = "fastsam_vit_b"
    pretrained: bool = True
    pretrained_path: Optional[str] = None
    freeze_stages: List[int] = field(default_factory=list)
    output_scales: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    patch_size: int = 16
    drop_path_rate: float = 0.1


@dataclass
class SparseConfig:
    """Sparse importance prediction (Ada-SPM) configuration."""

    name: str = "ada_spm"
    num_scales: int = 4
    importance_threshold: float = 0.5
    min_tile_ratio: float = 0.1
    max_tile_ratio: float = 0.8
    density_loss_weight: float = 1.0
    sparse_loss_weight: float = 0.1
    # Future: learnable threshold, scale-conditioned importance


@dataclass
class TokenizerConfig:
    """Dynamic tile tokenizer configuration."""

    name: str = "dynamic_tile"
    tile_sizes: List[int] = field(default_factory=lambda: [384, 768, 1536])
    stride_ratios: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    max_tokens_per_image: int = 4096
    cache_tiles: bool = True
    cache_dir: Optional[str] = None
    skip_mode: str = "threshold"
    hard_skip_multiplier: float = 1.0


@dataclass
class RouterConfig:
    """Dynamic token router (DTR-v2) configuration."""

    name: str = "DTRv2Router"
    embed_dim: int = 256
    max_full_ratio: float = 0.25
    max_skip_ratio: float = 0.15
    min_linear_ratio: float = 0.20
    aux_loss_weight: float = 0.01


@dataclass
class PrototypeConfig:
    """Prototype memory configuration."""

    name: str = "masked_avg"
    prototype_dim: int = 256
    num_prototypes_per_class: int = 1
    temperature: float = 0.1
    cache_path: Optional[str] = None
    # Optional: learnable prototype aggregation (PANet-style)


@dataclass
class DecoderConfig:
    """Segmentation decoder configuration."""

    name: str = "fastsam_decoder"
    mask_dim: int = 256
    iou_prediction: bool = True
    num_mask_tokens: int = 4
    # Optional: Multi-scale mask refinement


@dataclass
class DataConfig:
    """Dataset and data loading configuration."""

    name: str = "coco"
    root_dir: str = "datasets"
    image_dir: str = "images"
    anno_dir: str = "annotations"
    tile_dir: str = "tiles"
    density_dir: str = "density_maps"
    proto_cache_dir: str = "prototype_cache"

    # Image
    img_size: Tuple[int, int] = (1024, 1024)
    max_size: int = 4096

    # Augmentation
    train_pipeline: List[str] = field(
        default_factory=lambda: [
            "random_resize",
            "random_flip",
            "random_brightness_contrast",
        ]
    )
    val_pipeline: List[str] = field(default_factory=lambda: ["resize"])

    # Dataloader
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True

    # Dataset split names (COCO uses "train2017", iSAID uses "train")
    split_train: str = "train"
    split_val: str = "val"

    # Few-shot
    fewshot_split: Optional[str] = None  # e.g. "datasets/iSAID/fewshot_splits/5shot"
    n_shot: int = 5
    n_way: int = 3


@dataclass
class TrainConfig:
    """Training configuration."""

    # Optimization
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.999)

    # Schedule
    scheduler: str = "cosine"
    warmup_steps: int = 500
    max_steps: int = 50000
    warmup_ratio: float = 0.01

    # Training
    epochs: int = 50
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"

    # Distributed
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1

    # Logging
    log_interval: int = 50
    eval_interval: int = 1000
    save_interval: int = 5000
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "adatile-fastsam"

    # Misc
    seed: int = 42
    deterministic: bool = True
    resume_from: Optional[str] = None


@dataclass
class CATConfig:
    """CAT (Conditional Adaptive Tuning) configuration.

    Controls the CAT-SAM-inspired prompt bridge and adapter modules.
    Set enabled=False to use standard AdaTile without CAT modules.
    """

    enabled: bool = False
    token_dim: int = 256
    spm_dim: int = 128
    adapter_reduction: int = 4
    spatial_size: int = 32


@dataclass
class LossConfig:
    """Loss function configuration.

    Controls mask supervision and auxiliary loss weights.
    Set any auxiliary weight to 0.0 to skip its computation entirely.
    """

    # Mask supervision (Focal + Dice)
    mask_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_weight: float = 5.0
    focal_weight: float = 1.0

    # Auxiliary losses — set to 0.0 to skip computation
    density_weight: float = 0.5
    sparsity_weight: float = 0.002
    routing_weight: float = 0.1

    # Matching
    match_iou_threshold: float = 0.3


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    metrics: List[str] = field(
        default_factory=lambda: ["coco_bbox", "coco_mask", "fewshot_iou"]
    )
    iou_thresholds: List[float] = field(
        default_factory=lambda: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    max_dets: int = 100
    eval_mask: bool = True
    save_predictions: bool = False
    pred_dir: Optional[str] = None


# ── Top-Level Config ─────────────────────────────────────────────────

@dataclass
class Config:
    """Top-level configuration composing all sub-configs.

    Usage:
        cfg = Config()
        cfg.backbone.name = "vit_l"

        cfg = Config.from_yaml("configs/isaid.py")
        cfg = Config.from_dict({"backbone": {"name": "resnet50"}})
    """

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    sparse: SparseConfig = field(default_factory=SparseConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    cat: CATConfig = field(default_factory=CATConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Experiment metadata
    experiment_name: str = "adatile_fastsam"
    output_dir: str = "output"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a plain dict."""
        return _dataclass_to_dict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def clone(self) -> Config:
        """Return a deep copy of this config."""
        return deepcopy(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        """Create Config from a dict (partial updates allowed)."""
        default = get_default_config().to_dict()
        _deep_update(default, data)
        return _dict_to_dataclass(default, cls)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> Config:
        """Create Config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> Config:
        """Create Config from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update nested dictionaries."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def get_default_config() -> Config:
    """Return the default configuration."""
    return Config()
