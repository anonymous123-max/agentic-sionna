"""Furniture specification and placement decision dataclasses.

Defines the input specification for furniture placement (FurnitureSpec)
and the output decision record (PlacementDecision) used by the layout optimizer.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from src.models.room import BoundingBox


@dataclass
class FurnitureSpec:
    """Specification for furniture to place (before optimization).

    Attributes:
        category: Furniture category (e.g., "bed", "sofa", "desk")
        model_id: 3D-FUTURE model UUID
        model_path: Path to the model file (OBJ)
        dimensions: Bounding box dimensions from OBJ
        preferred_wall: Optional wall constraint ("north", "south", "east", "west")
        orientation_offset: Rotation offset to normalize model facing direction.
            Computed from catalog.get_orientation_offset() to handle inconsistent
            3D-FUTURE model orientations within categories.
    """

    category: str
    model_id: str
    model_path: str
    dimensions: BoundingBox
    preferred_wall: Literal["north", "south", "east", "west"] | None = None
    orientation_offset: float = 0.0


@dataclass
class PlacementDecision:
    """Captured optimization decision for explanation.

    Contains the constraint costs for a single furniture item's placement,
    enabling generation of human-readable placement rationales.

    Attributes:
        furniture_id: Unique ID of the furniture item
        category: Furniture category (e.g., "bed", "desk")
        position: Tuple of (x, y, theta) coordinates
        constraint_costs: Dict of constraint name to cost value
        dominant_constraint: Name of constraint with highest cost
    """

    furniture_id: str
    category: str
    position: tuple[float, float, float]  # x, y, theta
    constraint_costs: dict[str, float]  # {wall_affinity, collision, pathway, in_room}
    dominant_constraint: str  # Name of constraint with highest cost
