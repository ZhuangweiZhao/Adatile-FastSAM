"""
边界细化模块 | Boundary Refinement Module.
============================================

FastSAM produces coarse masks (SAM prior); BoundaryRefiner recovers fine edges.
"""

from adatile.refine.boundary_refiner import BoundaryRefiner, boundary_aware_loss
