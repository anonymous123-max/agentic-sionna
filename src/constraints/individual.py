"""Single-furniture constraint cost functions.

Cost functions that evaluate individual furniture items against the room.
Based on FlairGPT Individual.py constraint implementations.

Each function returns 0 when the constraint is satisfied and increases
with deviation from the desired state.

Polygon support:
- Functions accept optional polygon/wall_segments parameters
- When None, existing rectangular logic runs unchanged (backward compat)
- When provided, Shapely-based polygon containment/distance is used
"""

import math
from typing import List, Literal, Optional

import numpy as np
from shapely import Point, LineString
from shapely import Polygon as ShapelyPolygon

from src.models.room import FurnitureItem, WallSegment
from src.utils.geometry import corners


def wall_affinity_cost(
    furniture: FurnitureItem,
    room_width: float,
    room_length: float,
    side: Literal["back", "front", "left", "right"] = "back",
    wall_segments: Optional[List[WallSegment]] = None,
) -> float:
    """Cost function for furniture-to-wall proximity.

    Returns 0 when furniture side touches a wall, increases quadratically
    with distance from nearest wall.

    Port of FlairGPT ind_next_to_wall() from Individual.py lines 63-113.

    Args:
        furniture: FurnitureItem with position and dimensions
        room_width: Room width in meters (X dimension)
        room_length: Room length in meters (Y dimension)
        side: Which side should touch wall ("back", "front", "left", "right")
              - back: faces away from furniture's facing direction
              - front: faces furniture's facing direction
              - left/right: perpendicular to facing direction
        wall_segments: Optional wall segments for polygon rooms.
            When None, uses rectangular wall distance (backward compat).

    Returns:
        Cost value (0 = against wall, increases with gap)
    """
    x = furniture.position.x
    y = furniture.position.y
    theta = furniture.position.theta
    w = furniture.dimensions.width
    l = furniture.dimensions.depth  # noqa: E741

    # Get all corners: TL, TR, BR, BL order
    cs = np.array(corners(x, y, theta, w, l))

    # Determine which corners belong to which side
    side_corners = {
        "back": [0, 1],   # TL, TR
        "front": [3, 2],  # BL, BR
        "left": [0, 3],   # TL, BL
        "right": [1, 2],  # TR, BR
    }
    corner_indices = side_corners[side]

    if wall_segments is not None:
        # Polygon branch: compute distance from side corners to nearest wall segment
        min_distance = float("inf")
        for ci in corner_indices:
            pt = Point(cs[ci])
            for seg in wall_segments:
                line = LineString([seg.start, seg.end])
                dist = pt.distance(line)
                if dist < min_distance:
                    min_distance = dist
        min_distance = max(0.0, min_distance)
        return 2.0 * min_distance ** 2

    # Rectangular branch (original code, unchanged)
    dist_north = room_length - cs[:, 1]
    dist_east = room_width - cs[:, 0]
    dist_south = cs[:, 1]
    dist_west = cs[:, 0]

    wall_distances = np.stack([dist_north, dist_east, dist_south, dist_west], axis=1)
    side_wall_distances = wall_distances[corner_indices, :]
    min_distance = np.min(side_wall_distances)
    min_distance = max(0.0, min_distance)

    return 2.0 * min_distance ** 2


def in_room_cost(
    furniture: FurnitureItem,
    room_width: float,
    room_length: float,
    room_polygon: Optional[ShapelyPolygon] = None,
) -> float:
    """Penalty for furniture extending outside room bounds.

    Returns 0 when fully inside, increases with overlap amount.

    Args:
        furniture: FurnitureItem with position and dimensions
        room_width: Room width in meters (X dimension)
        room_length: Room length in meters (Y dimension)
        room_polygon: Optional Shapely Polygon for non-rectangular rooms.
            When None, uses rectangular bounds check (backward compat).

    Returns:
        Cost value (0 = inside, increases with outside amount)
    """
    x = furniture.position.x
    y = furniture.position.y
    theta = furniture.position.theta
    w = furniture.dimensions.width
    l = furniture.dimensions.depth  # noqa: E741

    # Get all corners: TL, TR, BR, BL order
    cs = np.array(corners(x, y, theta, w, l))

    if room_polygon is not None:
        # Polygon branch: distance-based cost for corners outside polygon
        total_cost = 0.0
        for corner in cs:
            pt = Point(corner[0], corner[1])
            if not room_polygon.contains(pt):
                dist = pt.distance(room_polygon)
                total_cost += dist ** 2
        return total_cost

    # Rectangular branch (original code, unchanged)
    total_cost = 0.0

    for corner in cs:
        cx, cy = corner

        # Check X bounds violation
        if cx < 0:
            total_cost += cx ** 2
        elif cx > room_width:
            total_cost += (cx - room_width) ** 2

        # Check Y bounds violation
        if cy < 0:
            total_cost += cy ** 2
        elif cy > room_length:
            total_cost += (cy - room_length) ** 2

    return total_cost
