"""Dynamic token routing subsystem (DTR-v2).

Provides DTRv2Router, UniformRouter, IdentityRouter,
and attention backends (Linear, LowRank, Full, MultiLevel).
"""

from adatile.registry import ROUTER

from adatile.routing.router import (
    DTRv2Router, UniformRouter, IdentityRouter,
    RoutingHead, BudgetController, PrototypeRouter, ConfidenceEstimator,
)
from adatile.routing.attention import (
    LinearAttention, LowRankAttention, FullAttention,
    MultiLevelAttention, SparseAttentionBase,
)
# Re-export core data structures for convenience
from adatile.core import RoutingOutput


def build_router(name: str, **kwargs):
    """Factory: instantiate a registered router by name."""
    return ROUTER.build(name, **kwargs)


__all__ = [
    "DTRv2Router", "UniformRouter", "IdentityRouter",
    "RoutingHead", "BudgetController", "PrototypeRouter", "ConfidenceEstimator",
    "LinearAttention", "LowRankAttention", "FullAttention",
    "MultiLevelAttention", "SparseAttentionBase",
    "RoutingOutput",
    "build_router",
]
