"""Geometry utilities for rotated rectangles + simple AABB helpers.

Plain-coordinate helpers (`aabb_overlap`, `in_bounds`, `rotate_2d`) documented
in `references/core-patterns.md`.

Coordinate convention:
- Origin at SW corner; X east, Y north
- theta=0 means facing positive Y (radians)
- width perpendicular to facing direction; depth in facing direction

Corner naming when theta=0 (facing north):
- TL = NW, TR = NE, BR = SE, BL = SW
"""
from __future__ import annotations

import math
from typing import Any, List, Tuple

from shapely import Polygon

from .models import FurnitureItem


# ------------------------------------------------------------------
# Rotated-rectangle corners
# ------------------------------------------------------------------

def TR(x: float, y: float, theta: float, w: float, l: float) -> Tuple[float, float]:
    return (
        x + w / 2 * math.cos(theta) + l / 2 * math.sin(theta),
        y + w / 2 * math.sin(theta) - l / 2 * math.cos(theta),
    )


def TL(x: float, y: float, theta: float, w: float, l: float) -> Tuple[float, float]:
    return (
        x - w / 2 * math.cos(theta) + l / 2 * math.sin(theta),
        y - w / 2 * math.sin(theta) - l / 2 * math.cos(theta),
    )


def BR(x: float, y: float, theta: float, w: float, l: float) -> Tuple[float, float]:
    return (
        x + w / 2 * math.cos(theta) - l / 2 * math.sin(theta),
        y + w / 2 * math.sin(theta) + l / 2 * math.cos(theta),
    )


def BL(x: float, y: float, theta: float, w: float, l: float) -> Tuple[float, float]:
    return (
        x - w / 2 * math.cos(theta) - l / 2 * math.sin(theta),
        y - w / 2 * math.sin(theta) + l / 2 * math.cos(theta),
    )


def corners(
    x: float, y: float, theta: float, w: float, l: float
) -> List[Tuple[float, float]]:
    """Four corners of a rotated rectangle in CCW order: TL, TR, BR, BL."""
    return [
        TL(x, y, theta, w, l),
        TR(x, y, theta, w, l),
        BR(x, y, theta, w, l),
        BL(x, y, theta, w, l),
    ]


def furniture_polygon(item: FurnitureItem) -> Polygon:
    """Shapely Polygon for a furniture item's footprint."""
    pts = corners(
        item.position.x,
        item.position.y,
        item.position.theta,
        item.dimensions.width,
        item.dimensions.depth,
    )
    return Polygon(pts)


def polygons_intersect(poly1: Polygon, poly2: Polygon) -> bool:
    """True if polygons overlap (touching edges count as no-overlap)."""
    return bool(poly1.intersects(poly2) and not poly1.touches(poly2))


# ------------------------------------------------------------------
# Plain-coordinate AABB helpers (used by core-patterns.md snippets)
# ------------------------------------------------------------------

def aabb_overlap(
    a: Any, b: Any, margin: float = 0.0
) -> bool:
    """AABB collision test for two axis-aligned rectangles.

    Each operand may be either a FurnitureItem (uses position/dimensions)
    or a dict-like with keys: x, y, w, h (or w, depth/d).
    """
    ax, ay, aw, ah = _aabb(a)
    bx, by, bw, bh = _aabb(b)
    return not (
        ax + aw / 2 + margin <= bx - bw / 2
        or bx + bw / 2 + margin <= ax - aw / 2
        or ay + ah / 2 + margin <= by - bh / 2
        or by + bh / 2 + margin <= ay - ah / 2
    )


def _aabb(obj: Any) -> Tuple[float, float, float, float]:
    if isinstance(obj, FurnitureItem):
        return (
            obj.position.x,
            obj.position.y,
            obj.dimensions.width,
            obj.dimensions.depth,
        )
    # dict-like
    h = obj.get("h", obj.get("depth", obj.get("d", 0.0)))
    return (obj["x"], obj["y"], obj["w"], h)


def in_bounds(
    x: float,
    y: float,
    w: float,
    d: float,
    room_w: float,
    room_l: float,
    margin: float = 0.05,
) -> bool:
    """True if an axis-aligned rect of (w, d) at center (x, y) is inside room."""
    return (
        x - w / 2 >= margin
        and x + w / 2 <= room_w - margin
        and y - d / 2 >= margin
        and y + d / 2 <= room_l - margin
    )


def rotate_2d(x: float, y: float, theta_deg: float) -> Tuple[float, float]:
    """Rotate a point (x, y) by theta degrees around the origin (CCW)."""
    theta = math.radians(theta_deg)
    c, s = math.cos(theta), math.sin(theta)
    return (x * c - y * s, x * s + y * c)
