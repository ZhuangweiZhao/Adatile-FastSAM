"""Hook-based FastSAM feature extractor for Stage A.

Extracts intermediate backbone features (P4, P8) via forward hooks.
Does NOT use YOLOv8 detection head — avoids the 84%-positive problem.

Reference: CS-FastSAM project approach.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List


class FastSAMHookBackbone(nn.Module):
    """Extract FastSAM intermediate features via forward hooks.

    Loads YOLOv8-seg checkpoint, extracts features at specific
    backbone layers. Detection head and Segment head are NOT used.

    Args:
        weight_file: Path to FastSAM-x.pt or yolov8s-seg.pt.
        out_channels: Unified projection channels for all outputs.
        unfreeze_layers: Number of backbone layers to unfreeze (0=all frozen).
        image_size: Expected input size (resizes if needed).
        output_levels: Dict mapping output name → layer index.
            Default: {"P8": 4, "P4": 6} (2 levels for LightSPM).
            For AdaSPM: {"P8": 4, "P4": 6, "P5": 8, "P3": 2} (4 levels for FPN).
    """

    # Pre-defined level configs
    LEVEL_CONFIGS = {
        "default": {"P8": 4, "P4": 6},           # 2 levels (LightSPM)
        "fpn2": {"P8": 4, "P4": 6},              # explicit 2-level
        "fpn4": {"P3": 2, "P4": 6, "P5": 8, "P8": 4},  # 4 levels (AdaSPM-Full)
    }

    def __init__(
        self,
        weight_file: str = "FastSAM-x.pt",
        out_channels: int = 128,
        unfreeze_layers: int = 0,
        image_size: Optional[int] = None,
        output_levels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.image_size = image_size

        # Load checkpoint
        ckpt = self._load_ckpt(weight_file)
        self.model = ckpt["model"].float()

        # Determine output layers
        children = list(self.model.model.children())
        n_children = len(children)

        # Resolve output_levels from config name or dict
        if output_levels is None:
            output_levels = self.LEVEL_CONFIGS["default"]
        elif isinstance(output_levels, str):
            output_levels = self.LEVEL_CONFIGS.get(
                output_levels, self.LEVEL_CONFIGS["default"]
            )

        self.output_layers = {}
        for name, idx in output_levels.items():
            if idx < n_children:
                self.output_layers[name] = children[idx]
            else:
                import logging
                _log = logging.getLogger("adatile.backbone")
                _log.warning(
                    "Layer index %d ('%s') out of range (0..%d). Skipping.",
                    idx, name, n_children - 1,
                )

        if not self.output_layers:
            raise ValueError(
                f"No valid output layers found. n_children={n_children}, "
                f"requested={output_levels}"
            )

        # Channel projections (lazy — built on first forward)
        self.projections = nn.ModuleDict()
        self._projections_built = False

        # Freeze / unfreeze
        self._apply_freeze(children, unfreeze_layers)

    @staticmethod
    def _load_ckpt(path):
        import sys, importlib
        compat = {
            "ultralytics.yolo.engine.trainer": "ultralytics.engine.trainer",
            "ultralytics.yolo.utils": "ultralytics.utils",
            "ultralytics.yolo.utils.checks": "ultralytics.utils.checks",
            "ultralytics.yolo.utils.loss": "ultralytics.utils.loss",
            "ultralytics.yolo.utils.metrics": "ultralytics.utils.metrics",
            "ultralytics.yolo.utils.tal": "ultralytics.utils.tal",
            "ultralytics.yolo.data": "ultralytics.data",
            "ultralytics.yolo.data.build": "ultralytics.data.build",
            "ultralytics.yolo.data.utils": "ultralytics.data.utils",
            "ultralytics.yolo.cfg": "ultralytics.cfg",
            "ultralytics.yolo.nn.tasks": "ultralytics.nn.tasks",
            "ultralytics.yolo.v8.segment": "ultralytics.models.yolo.segment",
            "ultralytics.yolo.v8.segment.val": "ultralytics.models.yolo.segment.val",
            "ultralytics.yolo.v8.segment.train": "ultralytics.models.yolo.segment.train",
            "ultralytics.yolo.v8.segment.predict": "ultralytics.models.yolo.segment.predict",
            "ultralytics.yolo": "ultralytics",
        }
        for old, new in compat.items():
            if old not in sys.modules:
                try:
                    sys.modules[old] = importlib.import_module(new)
                except ImportError:
                    pass
        return torch.load(path, map_location="cpu", weights_only=False)

    def _apply_freeze(self, children: List[nn.Module], unfreeze_layers: int):
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        for i in range(min(unfreeze_layers, len(children))):
            for p in children[i].parameters():
                p.requires_grad = True
            children[i].train()

    def _ensure_projections(self, sample_channels: Dict[str, int], device, dtype):
        if self._projections_built:
            return
        for name, ch in sample_channels.items():
            self.projections[name] = nn.Conv2d(ch, self.out_channels, 1, bias=False).to(device=device, dtype=dtype)
        self._projections_built = True

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: [B, 3, H, W] in [0, 1] range.

        Returns:
            {"P4": [B, C, H/16, W/16], "P8": [B, C, H/8, W/8]}
        """
        _, _, H, W = image.shape
        # Resize only if image_size is set and doesn't match
        if self.image_size is not None:
            target_h, target_w = (self.image_size if isinstance(self.image_size, tuple)
                                  else (self.image_size, self.image_size))
            if H != target_h or W != target_w:
                image = nn.functional.interpolate(
                    image, size=(target_h, target_w),
                    mode="bilinear", align_corners=False,
                )

        # Pad to multiple of 32 (YOLOv8 requirement for Concat layers)
        _, _, H, W = image.shape
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        if pad_h > 0 or pad_w > 0:
            image = nn.functional.pad(image, (0, pad_w, 0, pad_h), mode="reflect")

        x = image * 255.0  # YOLO expects [0, 255]

        # Hook-based extraction
        outputs = {}

        def make_hook(name):
            def hook(_, __, outp):
                if isinstance(outp, torch.Tensor):
                    outputs[name] = outp
            return hook

        hooks = []
        for name, layer in self.output_layers.items():
            h = layer.register_forward_hook(make_hook(name))
            hooks.append(h)

        with torch.no_grad():
            self.model(x)

        for h in hooks:
            h.remove()

        # Project to unified channels
        self._ensure_projections(
            {k: v.shape[1] for k, v in outputs.items()},
            device=outputs[list(outputs.keys())[0]].device,
            dtype=outputs[list(outputs.keys())[0]].dtype,
        )

        result = {}
        for name in self.output_layers:
            if name in outputs:
                feat = self.projections[name](outputs[name])
                # Put BN/trainable layers in correct mode
                if self.projections[name].training:
                    result[name] = feat
                else:
                    result[name] = feat

        return result
