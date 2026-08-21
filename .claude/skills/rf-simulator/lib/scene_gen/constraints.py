"""Constraint cost functions for furniture placement.

Cost functions return 0 when satisfied and increase with deviation. Higher-
weight terms (collision=10, in_room=5, pathway=3, wall_affinity=2)
dominate when violated; the optimizer (`optimizer.py`) minimizes the
sum returned by `total_cost()`.

Top-level utility for end-of-pipeline validation: `validate_scene(scene)`
returns a list of human-readable violation strings (empty = clean).
"""
from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
from shapely import LineString, Point, Polygon, box
from shapely import Polygon as ShapelyPolygon

from .geometry import corners, furniture_polygon
from .models import (
    Door,
    FurnitureItem,
    Scene,
    WallSegment,
)

__all__ = [
    "wall_affinity_cost",
    "in_room_cost",
    "collision_cost",
    "pathway_cost",
    "total_cost",
    "validate_scene",
]


# ---------------------------------------------------------------------------
# Individual constraints
# ---------------------------------------------------------------------------

def wall_affinity_cost(
    furniture: FurnitureItem,
    room_width: float,
    room_length: float,
    side: Literal["back", "front", "left", "right"] = "back",
    wall_segments: Optional[List[WallSegment]] = None,
) -> float:
    """0 when furniture side touches a wall; quadratic with gap distance."""
    x = furniture.position.x
    y = furniture.position.y
    theta = furniture.position.theta
    w = furniture.dimensions.width
    l = furniture.dimensions.depth  # noqa: E741

    cs = np.array(corners(x, y, theta, w, l))

    side_corners = {
        "back": [0, 1],
        "front": [3, 2],
        "left": [0, 3],
        "right": [1, 2],
    }
    corner_indices = side_corners[side]

    if wall_segments is not None:
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

    dist_north = room_length - cs[:, 1]
    dist_east = room_width - cs[:, 0]
    dist_south = cs[:, 1]
    dist_west = cs[:, 0]

    wall_distances = np.stack(
        [dist_north, dist_east, dist_south, dist_west], axis=1
    )
    side_wall_distances = wall_distances[corner_indices, :]
    min_distance = float(np.min(side_wall_distances))
    min_distance = max(0.0, min_distance)
    return 2.0 * min_distance ** 2


def in_room_cost(
    furniture: FurnitureItem,
    room_width: float,
    room_length: float,
    room_polygon: Optional[ShapelyPolygon] = None,
) -> float:
    """0 when fully inside room; squared corner-overshoot otherwise."""
    x = furniture.position.x
    y = furniture.position.y
    theta = furniture.position.theta
    w = furniture.dimensions.width
    l = furniture.dimensions.depth  # noqa: E741

    cs = np.array(corners(x, y, theta, w, l))

    if room_polygon is not None:
        cost = 0.0
        for corner in cs:
            pt = Point(corner[0], corner[1])
            if not room_polygon.contains(pt):
                dist = pt.distance(room_polygon)
                cost += dist ** 2
        return cost

    cost = 0.0
    for corner in cs:
        cx, cy = corner
        if cx < 0:
            cost += cx ** 2
        elif cx > room_width:
            cost += (cx - room_width) ** 2
        if cy < 0:
            cost += cy ** 2
        elif cy > room_length:
            cost += (cy - room_length) ** 2
    return cost


# ---------------------------------------------------------------------------
# Inter-object constraints
# ---------------------------------------------------------------------------

def collision_cost(furniture_list: List[FurnitureItem]) -> float:
    """0 when no overlaps; otherwise 10x sum of pairwise intersection areas."""
    if len(furniture_list) < 2:
        return 0.0

    total_area = 0.0
    polygons = [furniture_polygon(item) for item in furniture_list]

    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            poly1 = polygons[i]
            poly2 = polygons[j]
            if poly1.intersects(poly2) and not poly1.touches(poly2):
                intersection = poly1.intersection(poly2)
                total_area += intersection.area
    return 10.0 * total_area


def pathway_cost(
    furniture_list: List[FurnitureItem],
    doors: List[Door],
    room_width: float,
    room_length: float,
    min_clearance: float = 0.6,
    wall_segments: Optional[List[WallSegment]] = None,
) -> float:
    """0 when door clearance zones are unobstructed."""
    if not doors or not furniture_list:
        return 0.0

    cost = 0.0
    furniture_polygons = [furniture_polygon(item) for item in furniture_list]

    for door in doors:
        if wall_segments is not None and door.wall_segment_index is not None:
            zone = _create_door_clearance_zone_polygon(
                door, wall_segments, min_clearance
            )
        else:
            zone = _create_door_clearance_zone(
                door, room_width, room_length, min_clearance
            )

        if zone is None:
            continue

        for poly in furniture_polygons:
            if poly.intersects(zone) and not poly.touches(zone):
                intersection = poly.intersection(zone)
                cost += intersection.area
    return 3.0 * cost


def _create_door_clearance_zone(
    door: Door,
    room_width: float,
    room_length: float,
    min_clearance: float,
) -> Polygon | None:
    door_center = door.position + door.width / 2
    half_width = door.width / 2

    if door.wall == "south":
        return box(
            max(0, door_center - half_width), 0,
            min(room_width, door_center + half_width), min_clearance,
        )
    if door.wall == "north":
        return box(
            max(0, door_center - half_width), room_length - min_clearance,
            min(room_width, door_center + half_width), room_length,
        )
    if door.wall == "west":
        return box(
            0, max(0, door_center - half_width),
            min_clearance, min(room_length, door_center + half_width),
        )
    if door.wall == "east":
        return box(
            room_width - min_clearance, max(0, door_center - half_width),
            room_width, min(room_length, door_center + half_width),
        )
    return None


def _create_door_clearance_zone_polygon(
    door: Door,
    wall_segments: List[WallSegment],
    min_clearance: float,
) -> Polygon | None:
    if (
        door.wall_segment_index is None
        or door.wall_segment_index >= len(wall_segments)
    ):
        return None

    seg = wall_segments[door.wall_segment_index]
    dx, dy = seg.direction
    nx, ny = seg.inward_normal

    door_center_along = door.position + door.width / 2
    half_w = door.width / 2

    p1_x = seg.start[0] + dx * (door_center_along - half_w)
    p1_y = seg.start[1] + dy * (door_center_along - half_w)
    p2_x = seg.start[0] + dx * (door_center_along + half_w)
    p2_y = seg.start[1] + dy * (door_center_along + half_w)
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
    """Weighted sum used by the optimizer."""
    cost = 0.0
    for furniture in furniture_list:
        cost += wall_affinity_cost(
            furniture, room_width, room_length, wall_segments=wall_segments
        )
        cost += 5.0 * in_room_cost(
            furniture, room_width, room_length, room_polygon=room_polygon
        )
    cost += collision_cost(furniture_list)
    cost += pathway_cost(
        furniture_list, doors, room_width, room_length,
        wall_segments=wall_segments,
    )
    return cost


# ---------------------------------------------------------------------------
# Top-level scene validator (NEW — wraps cost functions for boolean check)
# ---------------------------------------------------------------------------

def validate_scene(scene: Scene) -> List[str]:
    """Return a list of human-readable violations (empty = clean).

    Checks:
    - Indoor: pairwise furniture overlap, in-bounds containment
    - Indoor: TX inside room and z within room height
    - Outdoor: TX inside scene width/length
    """
    violations: List[str] = []

    if scene.room is not None:
        room = scene.room
        # Pairwise overlap (the high-confidence collision check)
        items = list(room.furniture)
        polys = [furniture_polygon(it) for it in items]
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if polys[i].intersects(polys[j]) and not polys[i].touches(polys[j]):
                    violations.append(
                        f"furniture overlap: {items[i].id} <-> {items[j].id}"
                    )
        # In-bounds
        room_poly = room.shapely_polygon
        for it in items:
            poly = furniture_polygon(it)
            if not room_poly.contains(poly):
                violations.append(
                    f"furniture out of bounds: {it.id}"
                )
        # TX inside room
        for tx in scene.transmitters:
            x, y, z = tx.position
            pt = Point(x, y)
            if not room_poly.contains(pt):
                violations.append(
                    f"transmitter outside room: {tx.id}"
                )
            if z <= 0 or z > room.height:
                violations.append(
                    f"transmitter z={z:.2f} outside room height [0, {room.height}]: {tx.id}"
                )

    if scene.outdoor is not None:
        out = scene.outdoor
        for tx in scene.transmitters:
            x, y, _ = tx.position
            if not (0 <= x <= out.width and 0 <= y <= out.length):
                violations.append(
                    f"transmitter outside outdoor scene: {tx.id}"
                )

    return violations
