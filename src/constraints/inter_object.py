"""Multi-furniture constraint cost functions.

Cost functions that evaluate relationships between multiple furniture items
or between furniture and room features (doors, windows).

Based on FlairGPT InterObject.py constraint implementations.

Polygon support:
- pathway_cost accepts optional wall_segments for polygon door clearance
- total_cost passes polygon params through to individual constraints
"""

from typing import List, Optional

from shapely import Polygon, box
from shapely import Polygon as ShapelyPolygon

from src.models.room import Door, FurnitureItem, WallSegment
from src.constraints.individual import in_room_cost, wall_affinity_cost
from src.utils.geometry import furniture_polygon


def collision_cost(furniture_list: List[FurnitureItem]) -> float:
    """Cost for furniture-furniture overlaps.

    Returns 0 when no collisions, increases with overlap area.

    Args:
        furniture_list: List of FurnitureItem objects

    Returns:
        Cost proportional to total intersection area (weighted by 10x)

    Implementation:
        1. For each pair of furniture items
        2. Create Shapely polygons using furniture_polygon()
        3. If polygons intersect (not just touch), add intersection.area to cost
        4. Return sum * 10 (high penalty for collisions)
    """
    if len(furniture_list) < 2:
        return 0.0

    total_area = 0.0

    # Create all polygons once
    polygons = [furniture_polygon(item) for item in furniture_list]

    # Check all pairs
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            poly1 = polygons[i]
            poly2 = polygons[j]

            # Check for intersection (not just touching)
            if poly1.intersects(poly2) and not poly1.touches(poly2):
                intersection = poly1.intersection(poly2)
                total_area += intersection.area

    # High penalty multiplier for collisions
    return 10.0 * total_area


def pathway_cost(
    furniture_list: List[FurnitureItem],
    doors: List[Door],
    room_width: float,
    room_length: float,
    min_clearance: float = 0.6,
    wall_segments: Optional[List[WallSegment]] = None,
) -> float:
    """Cost for blocked pathways from doors.

    Returns 0 when clear paths exist from all doors.

    Args:
        furniture_list: List of FurnitureItem objects
        doors: List of Door objects in the room
        room_width: Room width in meters (X dimension)
        room_length: Room length in meters (Y dimension)
        min_clearance: Minimum clearance distance from door (meters)
        wall_segments: Optional wall segments for polygon rooms.
            When None, uses rectangular clearance zones (backward compat).

    Returns:
        Cost proportional to intersection with clearance zones
    """
    if not doors or not furniture_list:
        return 0.0

    total_cost = 0.0

    # Create furniture polygons once
    furniture_polygons = [furniture_polygon(item) for item in furniture_list]

    for door in doors:
        # Choose clearance zone method based on polygon vs rectangular
        if wall_segments is not None and door.wall_segment_index is not None:
            clearance_zone = _create_door_clearance_zone_polygon(
                door, wall_segments, min_clearance
            )
        else:
            clearance_zone = _create_door_clearance_zone(
                door, room_width, room_length, min_clearance
            )

        if clearance_zone is None:
            continue

        # Check intersection with all furniture
        for poly in furniture_polygons:
            if poly.intersects(clearance_zone) and not poly.touches(clearance_zone):
                intersection = poly.intersection(clearance_zone)
                total_cost += intersection.area

    # Weight for pathway cost
    return 3.0 * total_cost


def _create_door_clearance_zone(
    door: Door,
    room_width: float,
    room_length: float,
    min_clearance: float,
) -> Polygon | None:
    """Create a clearance zone polygon for a door.

    The clearance zone extends from the door into the room.

    Args:
        door: Door object with wall and position
        room_width: Room width (X dimension)
        room_length: Room length (Y dimension)
        min_clearance: How far into room to extend clearance

    Returns:
        Shapely Polygon representing the clearance zone, or None if invalid
    """
    # Door position is distance along wall from origin corner
    # For south/north walls: position is X coordinate
    # For east/west walls: position is Y coordinate

    door_center = door.position + door.width / 2
    half_width = door.width / 2

    if door.wall == "south":
        # Door on south wall (y=0), clearance extends north into room
        x_min = max(0, door_center - half_width)
        x_max = min(room_width, door_center + half_width)
        y_min = 0
        y_max = min_clearance
        return box(x_min, y_min, x_max, y_max)

    elif door.wall == "north":
        # Door on north wall (y=room_length), clearance extends south into room
        x_min = max(0, door_center - half_width)
        x_max = min(room_width, door_center + half_width)
        y_min = room_length - min_clearance
        y_max = room_length
        return box(x_min, y_min, x_max, y_max)

    elif door.wall == "west":
        # Door on west wall (x=0), clearance extends east into room
        x_min = 0
        x_max = min_clearance
        y_min = max(0, door_center - half_width)
        y_max = min(room_length, door_center + half_width)
        return box(x_min, y_min, x_max, y_max)

    elif door.wall == "east":
        # Door on east wall (x=room_width), clearance extends west into room
        x_min = room_width - min_clearance
        x_max = room_width
        y_min = max(0, door_center - half_width)
        y_max = min(room_length, door_center + half_width)
        return box(x_min, y_min, x_max, y_max)

    return None


def _create_door_clearance_zone_polygon(
    door: Door,
    wall_segments: List[WallSegment],
    min_clearance: float,
) -> Polygon | None:
    """Create clearance zone for a door on a polygon wall segment.

    Builds a rectangle perpendicular to the wall segment, extending
    inward by min_clearance from the door center.

    Args:
        door: Door with wall_segment_index
        wall_segments: List of WallSegment from polygon room
        min_clearance: How far into room to extend clearance

    Returns:
        Shapely Polygon or None if invalid
    """
    if door.wall_segment_index is None or door.wall_segment_index >= len(wall_segments):
        return None

    seg = wall_segments[door.wall_segment_index]
    dx, dy = seg.direction
    nx, ny = seg.inward_normal

    # Door center along the segment
    door_center_along = door.position + door.width / 2
    half_w = door.width / 2

    # Points along wall at door edges
    p1_x = seg.start[0] + dx * (door_center_along - half_w)
    p1_y = seg.start[1] + dy * (door_center_along - half_w)
    p2_x = seg.start[0] + dx * (door_center_along + half_w)
    p2_y = seg.start[1] + dy * (door_center_along + half_w)

    # Extend inward by clearance
    p3_x = p2_x + nx * min_clearance
    p3_y = p2_y + ny * min_clearance
    p4_x = p1_x + nx * min_clearance
    p4_y = p1_y + ny * min_clearance

    return Polygon([(p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y), (p4_x, p4_y)])


def total_cost(
    furniture_list: List[FurnitureItem],
    doors: List[Door],
    room_width: float,
    room_length: float,
    room_polygon: Optional[ShapelyPolygon] = None,
    wall_segments: Optional[List[WallSegment]] = None,
) -> float:
    """Combined cost function for optimizer.

    Weights:
        - wall_affinity: 2 (from wall_affinity_cost)
        - collision: 10 (from collision_cost)
        - pathway: 3 (from pathway_cost)
        - in_room: 5 (multiplied here)

    Args:
        furniture_list: List of FurnitureItem objects
        doors: List of Door objects in the room
        room_width: Room width in meters
        room_length: Room length in meters
        room_polygon: Optional Shapely Polygon for non-rectangular rooms
        wall_segments: Optional wall segments for polygon rooms

    Returns:
        Total weighted cost for the furniture arrangement
    """
    cost = 0.0

    # Individual costs (wall affinity for each piece)
    for furniture in furniture_list:
        cost += wall_affinity_cost(
            furniture, room_width, room_length, wall_segments=wall_segments
        )
        cost += 5.0 * in_room_cost(
            furniture, room_width, room_length, room_polygon=room_polygon
        )

    # Inter-object costs
    cost += collision_cost(furniture_list)
    cost += pathway_cost(
        furniture_list, doors, room_width, room_length,
        wall_segments=wall_segments,
    )

    return cost
