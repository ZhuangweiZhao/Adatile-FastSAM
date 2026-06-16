"""Segmentation decoders.

Active (Stage A/B/C):
    - LightDecoder: simple conv decoder for binary segmentation

Legacy (train.py compatibility):
    - DifferentiableDecoder: complex tile-level decoder
"""

from adatile.registry import DECODER
from adatile.decoder.light_decoder import LightDecoder

# Legacy (moved to legacy/)
DifferentiableDecoder = None
TileProtoModule = None


def build_decoder(name: str, **kwargs):
    """Factory: instantiate a registered decoder by name."""
    _aliases = {"fastsam_decoder": "DifferentiableDecoder"}
    name = _aliases.get(name, name)
    return DECODER.build(name, **kwargs)


__all__ = [
    "LightDecoder",
    "DifferentiableDecoder",
    "TileProtoModule",
    "build_decoder",
]
