"""Prototype memory module.

Stores and retrieves class prototypes for few-shot segmentation.
Supports prototype-guided routing.
"""

from adatile.registry import PROTOTYPE

# Import implementations to trigger @PROTOTYPE.register() decorators
from adatile.prototype.base import (
    MaskedAveragePrototype,
    PrototypeMemory,
)


def build_prototype(name: str, **kwargs):
    """Factory: instantiate a registered prototype module by name.

    Available names:
        - "MaskedAveragePrototype" — classic masked-average pooling
        - "masked_avg" — alias for MaskedAveragePrototype
    """
    _aliases = {
        "masked_avg": "MaskedAveragePrototype",
    }
    name = _aliases.get(name, name)
    return PROTOTYPE.build(name, **kwargs)


__all__ = [
    "PROTOTYPE",
    "build_prototype",
    "MaskedAveragePrototype",
    "PrototypeMemory",
]
