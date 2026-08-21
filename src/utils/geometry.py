"""Geometry utilities for rotated rectangle calculations.

Ported from FlairGPT Scene_Synthesis/Class_Structures.py

Coordinate system:
- Origin at SW corner of room
- X increases east, Y increases north
- theta=0 means furniture faces positive Y (north)
- width is perpendicular to facing direction (left-right when facing north)
- depth (length) is in facing direction (front-back when facing north)

Corner naming convention (when theta=0, facing north):
- TL (Top-Left): northwest corner
- TR (Top-Right): northeast corner
- BR (Bottom-Right): southeast corner
- BL (Bottom-Left): southwest corner
"""

import math
from typing import List, Tuple

from shapely import Polygon

from src.models.room import FurnitureItem


def TR(x: float, y: float, theta: float, w: float, l: float) -> Tuple[float, float]:
    """Calculate top-right corner of rotated rectangle.

    Args:
        x: Center x coordinate
        y: Center y coordinate
        theta: Rotation angle in radians (0 = facing north)
        w: Width (perpendicular to facing direction)
        l: Length/depth (in facing direction)

    Returns:
        (x, y) coordinates of top-right corner
    """
    return (
        x + w / 2 * math.cos(theta) + l / 2 * math.sin(theta),
        y + w / 2 * math.sin(theta) - l / 2 * math.cos(theta),
    )


def TL(x: float, y: float, theta: float, w: float, l: float) -> Tuple[float, float]:
    """Calculate top-left corner of rotated rectangle.

    Args:
        x: Center x coordinate
        y: Center y coordinate
        theta: Rotation angle in radians (0 = facing north)
        w: Width (perpendicular to facing direction)
        l: Length/depth (in facing direction)

    Returns:
        (x, y) coordinates of top-left corner
    """
    return (
        x - w / 2 * math.cos(theta) + l / 2 * math.sin(theta),
        y - w / 2 * math.sin(theta) - l / 2 * math.cos(theta),
    )


def BR(x: float, y: float, theta: float, w: float, l: float) -> Tuple[float, float]:
    """Calculate bottom-right corner of rotated rectangle.

    Args:
        x: Center x coordinate
        y: Center y coordinate
        theta: Rotation angle in radians (0 = facing north)
        w: Width (perpendicular to facing direction)
        l: Length/depth (in facing direction)

    Returns:
        (x, y) coordinates of bottom-right corner
    """
    return (
        x + w / 2 * math.cos(theta) - l / 2 * math.sin(theta),
        y + w / 2 * math.sin(theta) + l / 2 * math.cos(theta),
    )


def BL(x: float, y: float, theta: float, w: float, l: float) -> Tuple[float, float]:
    """Calculate bottom-left corner of rotated rectangle.

    Args:
        x: Center x coordinate
        y: Center y coordinate
        theta: Rotation angle in radians (0 = facing north)
        w: Width (perpendicular to facing direction)
        l: Length/depth (in facing direction)

    Returns:
        (x, y) coordinates of bottom-left corner
    """
    return (
        x - w / 2 * math.cos(theta) - l / 2 * math.sin(theta),
        y - w / 2 * math.sin(theta) + l / 2 * math.cos(theta),
    )


def corners(
    x: float, y: float, theta: float, w: float, l: float
) -> List[Tuple[float, float]]:
    """Calculate all four corners of a rotated rectangle.

    Returns corners in counterclockwise order: TL, TR, BR, BL.
    This order is suitable for creating Shapely polygons.

    Args:
        x: Center x coordinate
        y: Center y coordinate
        theta: Rotation angle in radians (0 = facing north)
        w: Width (perpendicular to facing direction)
        l: Length/depth (in facing direction)

    Returns:
        List of (x, y) tuples for TL, TR, BR, BL corners
    """
    return [
        TL(x, y, theta, w, l),
        TR(x, y, theta, w, l),
        BR(x, y, theta, w, l),
        BL(x, y, theta, w, l),
    ]


def furniture_polygon(item: FurnitureItem) -> Polygon:
    """Create a Shapely Polygon from a furniture item's position and dimensions.

    Args:
        item: FurnitureItem with position and dimensions

    Returns:
        Shapely Polygon representing the furniture footprint
    """
    pts = corners(
        item.position.x,
        item.position.y,
        item.position.theta,
        item.dimensions.width,
        item.dimensions.depth,
    )
    return Polygon(pts)


def polygons_intersect(poly1: Polygon, poly2: Polygon) -> bool:
    """Check if two polygons overlap (excluding touching edges).

    Two polygons that only share an edge (touch) are not considered
    as intersecting, which is the desired behavior for furniture
    collision detection.

    Args:
        poly1: First Shapely Polygon
        poly2: Second Shapely Polygon

    Returns:
        True if polygons overlap (not just touch), False otherwise
    """
    return poly1.intersects(poly2) and not poly1.touches(poly2)
