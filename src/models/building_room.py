"""Convert Building footprint to polygon Room for indoor analysis.

Bridges outdoor buildings and indoor rooms by converting a Building's
polygon footprint into a Room with polygon floor boundary and auto-placed door.
"""

import math
from typing import List, Optional

from src.models.outdoor import Building
from src.models.room import Door, Room, WallSegment


def building_to_room(
    building: Building,
    height: Optional[float] = None,
    doors: Optional[List[Door]] = None,
) -> Room:
    """Convert a Building footprint into a polygon Room.

    Steps:
    1. Get building footprint polygon
    2. Normalize to origin (translate so min corner = (0,0))
    3. Create Room.from_polygon with appropriate height
    4. Auto-place door on longest south-facing segment if no doors given

    Args:
        building: Building with polygon footprint
        height: Room ceiling height (default: min(building.height, 3.0))
        doors: Optional list of doors. If None, auto-places one door.

    Returns:
        Room with floor_polygon matching building footprint
    """
    poly = building.as_polygon()
    minx, miny, maxx, maxy = poly.bounds

    # Normalize footprint to origin
    coords = list(poly.exterior.coords)[:-1]  # Drop closing point
    normalized = [(x - minx, y - miny) for x, y in coords]

    room_height = height or min(building.height, 3.0)

    # Create room from polygon to get wall segments
    room_no_doors = Room.from_polygon(normalized, height=room_height)

    if doors is None:
        door = auto_place_door_on_polygon(room_no_doors)
        doors = [door]

    return Room.from_polygon(
        normalized,
        height=room_height,
        doors=doors,
    )


def auto_place_door_on_polygon(
    room: Room,
    preferred_cardinal: str = "south",
) -> Door:
    """Auto-place a door on the best wall segment of a polygon room.

    Finds the wall segment closest to the preferred cardinal direction,
    preferring longer segments. Places a standard door at the center.

    Args:
        room: Room with wall_segments
        preferred_cardinal: Preferred wall direction ("south", "north", etc.)

    Returns:
        Door placed on the chosen wall segment
    """
    segments = room.wall_segments
    if not segments:
        # Fallback for degenerate cases
        return Door(wall="south", position=0.0, width=0.9)

    door_width = 0.9  # Standard door width

    # Score each segment: prefer matching cardinal + longer segments
    best_idx = 0
    best_score = -1.0
    for i, seg in enumerate(segments):
        score = seg.length  # Base score = length
        if seg.cardinal_direction == preferred_cardinal:
            score *= 3.0  # Strong preference for matching cardinal
        if score > best_score and seg.length >= door_width:
            best_score = score
            best_idx = i

    chosen = segments[best_idx]
    # Place door at center of segment
    door_pos = (chosen.length - door_width) / 2
    door_pos = max(0.0, door_pos)

    return Door(
        wall=chosen.cardinal_direction,
        position=door_pos,
        width=door_width,
        wall_segment_index=best_idx,
    )
