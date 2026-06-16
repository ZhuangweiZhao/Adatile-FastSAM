"""Sparse inference with true tile-based decoder forward.

Unlike the post-hoc mask zeroing in sparse_eval (which runs the full
decoder on the entire image then discards low-importance predictions),
this module extracts feature tiles and only runs the decoder on
high-importance regions — actually reducing FLOPs.
"""

from adatile.inference.tile_inference import (
    tile_sparse_forward,
    estimate_flops_saved,
)

__all__ = ["tile_sparse_forward", "estimate_flops_saved"]
