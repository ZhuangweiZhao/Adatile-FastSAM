"""Segmentation decoder module.

FastSAM-inspired mask decoder with tile merging support.
"""

from adatile.registry import DECODER

# Import implementations to trigger @DECODER.register() decorators
from adatile.decoder.base import (
    FastSAMDecoder,
    SegmentationDecoder,
)


def build_decoder(name: str, **kwargs):
    """Factory: instantiate a registered decoder by name.

    Available names:
        - "FastSAMDecoder" — FastSAM-inspired mask decoder with torchvision ops
        - "fastsam_decoder" — alias for FastSAMDecoder
    """
    _aliases = {
        "fastsam_decoder": "FastSAMDecoder",
    }
    name = _aliases.get(name, name)
    return DECODER.build(name, **kwargs)


__all__ = [
    "DECODER",
    "build_decoder",
    "FastSAMDecoder",
    "SegmentationDecoder",
]
