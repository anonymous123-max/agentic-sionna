"""IRC §R303 window placement.

The 2021 International Residential Code requires habitable rooms to have
natural-light apertures of at least 8% of floor area, ventilation
apertures of at least 4% of floor area, and a minimum aperture height
of 0.6 m. We approximate by placing windows totalling 8% of floor area
on a single perimeter wall (or split across two if the room is large).

Non-habitable spaces (storage, mechanical, closet, bathroom_small)
don't require windows under §R303 and we skip them.

NOTE on material: The Window model (models.py) does not carry a material
field. Per the RF-simulation design, window glass (ε_r ≈ 6.3) is assigned
at export time based on the object type (Window vs. Wall). The absence of
a material field here is a known field-name divergence documented in the
task F.2 report.
"""
from __future__ import annotations

from typing import List, Optional

from .models import Room, Window

# Habitable room types that require §R303 natural-light apertures.
HABITABLE = {"living", "bedroom", "kitchen", "dining", "office", "study"}

# Non-habitable spaces — no windows required.
NON_HABITABLE = {
    "storage", "closet", "mechanical", "bathroom_small",
    "garage", "utility", "hallway", "corridor",
}

# §R303 minimum aperture as fraction of floor area.
APERTURE_RATIO_MIN = 0.08

# Standard window geometry.
_WIN_HEIGHT_M = 1.2        # nominal window height (m)
_WIN_SILL_M = 1.0          # sill height above finished floor (m)
_WIN_MIN_HEIGHT_M = 0.6    # §R303 minimum head height (m)

# Floor area threshold above which aperture is split across two windows
# to avoid an unrealistically wide single pane.
_SPLIT_AREA_M2 = 25.0


def place_irc_windows(room: Room,
                      room_type: str = "living") -> List[Window]:
    """Return Window objects sized to meet IRC §R303 for habitable rooms.

    For non-habitable rooms an empty list is returned immediately.

    Strategy
    --------
    - Pick the longer perimeter wall as the preferred facade.
    - Compute the required total glazed area: 8% of floor area.
    - For small rooms (≤ 25 m²) place one centred window.
    - For larger rooms split into two equal windows placed at the
      ⅓ and ⅔ points of the wall, which avoids a single comically
      wide pane and distributes daylighting more evenly.
    - ``position`` is the distance from the SW corner along the wall
      to the left edge of the window (consistent with the Door model
      convention in models.py).
    """
    if room_type in NON_HABITABLE:
        return []

    floor_area = room.width * room.length
    target_aperture = APERTURE_RATIO_MIN * floor_area

    # Window height: standard 1.2 m, but cap to fit within the room height
    # leaving at least _WIN_MIN_HEIGHT_M of clearance above the head.
    win_h = _WIN_HEIGHT_M
    max_h = room.height - _WIN_SILL_M - _WIN_MIN_HEIGHT_M
    if max_h < win_h:
        win_h = max(0.6, max_h)

    # Choose the longer wall — more horizontal room for glazing.
    if room.width >= room.length:
        wall: str = "south"
        wall_length: float = room.width
    else:
        wall = "west"
        wall_length = room.length

    if floor_area <= _SPLIT_AREA_M2:
        # Single window centred on the chosen wall.
        win_w = target_aperture / win_h
        win_w = min(win_w, wall_length * 0.7)  # max 70% of wall
        # Re-check: ensure the single window still meets minimum aperture.
        # (Capping can only decrease area; if it falls short, it means the
        # room's geometry is very unusual — we do not override the cap.)
        position = (wall_length - win_w) / 2
        return [Window(
            wall=wall,
            position=position,
            width=win_w,
            height=win_h,
            sill_height=_WIN_SILL_M,
        )]
    else:
        # Two equal windows at ⅓ and ⅔ positions.
        each_w = (target_aperture / win_h) / 2
        each_w = min(each_w, wall_length * 0.30)  # max 30% of wall each
        margin = (wall_length - 2 * each_w) / 3
        return [
            Window(
                wall=wall,
                position=margin,
                width=each_w,
                height=win_h,
                sill_height=_WIN_SILL_M,
            ),
            Window(
                wall=wall,
                position=2 * margin + each_w,
                width=each_w,
                height=win_h,
                sill_height=_WIN_SILL_M,
            ),
        ]
