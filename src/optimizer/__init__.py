"""Layout optimizer for furniture placement.

Provides constraint-based optimization using SciPy SLSQP to place
furniture against walls while avoiding collisions and maintaining
clear pathways from doors.
"""

from src.optimizer.specs import FurnitureSpec, PlacementDecision
from src.optimizer.layout import (
    LayoutOptimizer,
    optimize_layout,
)

__all__ = [
    "FurnitureSpec",
    "LayoutOptimizer",
    "PlacementDecision",
    "optimize_layout",
]
