"""PNG floor plan exporter — 2D top-down view.

- SW-corner origin (X east, Y north matching everything else)
- orientation_offset is added to position.theta when rotating each item
- Color-coded by furniture category
- Walls drawn as thick lines

Lazy-imports matplotlib so the rest of lib/scene_gen is usable on
matplotlib-free environments.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Scene

# Color palette by category (lowercase substring match)
_CATEGORY_COLORS: dict[str, str] = {
    "bed": "#8B5A3C",
    "sofa": "#6B7280",
    "armchair": "#7C8089",
    "table": "#A0522D",
    "desk": "#A0522D",
    "chair": "#5C4033",
    "cabinet": "#704214",
    "wardrobe": "#704214",
    "shelf": "#704214",
    "tv": "#1F2937",
    "lamp": "#FCD34D",
    "nightstand": "#8B6F47",
}
_DEFAULT_COLOR = "#9CA3AF"
_WALL_COLOR = "#1F2937"


def _color_for(category: str) -> str:
    cat = category.lower()
    for key, val in _CATEGORY_COLORS.items():
        if key in cat:
            return val
    return _DEFAULT_COLOR


def export_png(
    scene: "Scene",
    output_path: str | Path,
    *,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 150,
) -> Path:
    """Write a 2D floor-plan PNG of the scene's room.

    Returns the path written. Raises if matplotlib is missing.
    """
    try:
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "PNG export requires matplotlib. pip install matplotlib"
        ) from e

    if scene.room is None:
        raise ValueError("export_png requires scene.room to be set")
    room = scene.room

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Room polygon
    poly = room.shapely_polygon
    xs, ys = poly.exterior.xy
    ax.plot(list(xs), list(ys), color=_WALL_COLOR, linewidth=3.0, zorder=2)
    ax.fill(list(xs), list(ys), color="#F9FAFB", alpha=1.0, zorder=1)

    # Doors
    for door in room.doors:
        xs_d, ys_d = _door_segment(door, room.width, room.length)
        ax.plot(xs_d, ys_d, color="#16A34A", linewidth=4.0, zorder=3)

    # Furniture — orientation = position.theta + orientation_offset (the rule)
    for item in room.furniture:
        final_theta_rad = item.position.theta + item.orientation_offset
        final_theta_deg = math.degrees(final_theta_rad)
        rect = patches.Rectangle(
            (
                item.position.x - item.dimensions.width / 2,
                item.position.y - item.dimensions.depth / 2,
            ),
            item.dimensions.width,
            item.dimensions.depth,
            angle=final_theta_deg,
            rotation_point="center",
            color=_color_for(item.category),
            ec=_WALL_COLOR,
            lw=0.8,
            zorder=4,
        )
        ax.add_patch(rect)
        ax.text(
            item.position.x, item.position.y, item.category,
            ha="center", va="center", fontsize=6.5, color="white", zorder=5,
        )

    # Transmitters
    for tx in scene.transmitters:
        x, y, _ = tx.position
        ax.plot(x, y, marker="*", markersize=14, color="#DC2626", zorder=6)
        ax.text(x + 0.2, y + 0.2, tx.id, fontsize=7, color="#DC2626", zorder=6)

    ax.set_xlim(-0.5, room.width + 0.5)
    ax.set_ylim(-0.5, room.length + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X (east, m)")
    ax.set_ylabel("Y (north, m)")
    ax.set_title(f"Floor plan {room.width:.1f}×{room.length:.1f} m")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _door_segment(door, room_w: float, room_l: float):
    p, w = door.position, door.width
    if door.wall == "south":
        return [p, p + w], [0, 0]
    if door.wall == "north":
        return [p, p + w], [room_l, room_l]
    if door.wall == "west":
        return [0, 0], [p, p + w]
    return [room_w, room_w], [p, p + w]
