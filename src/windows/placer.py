"""WindowPlacer for automatic window placement.

Provides automatic window generation based on room type and IRC building codes.
Bedrooms get egress-compliant windows; living rooms get glazing-ratio windows.
"""

from typing import List, Literal, Optional

from src.models.room import Door, Room, Window
from src.windows.building_codes import (
    EGRESS_MAX_SILL_HEIGHT_IN,
    EGRESS_MIN_OPENING_AREA_SQFT,
    EGRESS_MIN_OPENING_HEIGHT_IN,
    EGRESS_MIN_OPENING_WIDTH_IN,
    M2_TO_SQFT,
    M_TO_IN,
)


# Default exterior walls for window placement (south preferred for solar gain)
DEFAULT_EXTERIOR_WALLS = ["south", "east", "west", "north"]

# Minimum gap from corners and doors (meters)
MIN_GAP = 0.3

# Standard window heights (meters)
EGRESS_WINDOW_HEIGHT = 1.0  # ~39" > 24" required
LIVING_WINDOW_HEIGHT = 1.2  # Standard living room window

# Sill heights (meters)
EGRESS_SILL_HEIGHT = 0.6   # ~24" < 44" max, low for egress
LIVING_SILL_HEIGHT = 0.9   # Standard sill


WallName = Literal["north", "south", "east", "west"]


class WindowPlacer:
    """Automatic window placement based on room type and building codes.

    Places windows on rooms according to IRC building code requirements:
    - Bedrooms: Egress-compliant windows (min 5.7 sqft, max 44" sill)
    - Living rooms: 8-10% glazing ratio of floor area

    Windows avoid walls with doors when alternatives exist.

    Example:
        >>> from src.models.room import Room, Door
        >>> room = Room(width=4.0, length=5.0, doors=[Door(wall="south", position=1.5, width=0.9)])
        >>> placer = WindowPlacer()
        >>> room_with_window = placer.place_for_bedroom(room)
        >>> print(f"Window on {room_with_window.windows[0].wall} wall")
    """

    def __init__(
        self,
        glazing_ratio: float = 0.08,
        exterior_walls: Optional[List[str]] = None,
    ):
        """Initialize WindowPlacer.

        Args:
            glazing_ratio: Target glazing ratio as fraction of floor area.
                Default 0.08 (8%) per IRC R303.1.
            exterior_walls: Preferred walls for window placement, in order
                of preference. Default ["south", "east", "west", "north"].
        """
        self.glazing_ratio = glazing_ratio
        self.exterior_walls = exterior_walls or DEFAULT_EXTERIOR_WALLS.copy()

    def place_for_bedroom(self, room: Room) -> Room:
        """Place egress-compliant windows for bedroom.

        Creates a window meeting IRC R310 egress requirements:
        - Minimum 5.7 sqft net clear opening
        - Minimum 24" opening height
        - Minimum 20" opening width
        - Maximum 44" sill height from floor

        Args:
            room: Room to add window to

        Returns:
            New Room instance with egress window added
        """
        # Calculate target glazing area
        floor_area = room.width * room.length
        target_glazing = floor_area * self.glazing_ratio

        # Start with egress-compliant dimensions
        window_height = EGRESS_WINDOW_HEIGHT  # ~39" > 24" required
        sill_height = EGRESS_SILL_HEIGHT      # ~24" < 44" max

        # Calculate minimum width for egress compliance
        # 5.7 sqft = 0.53 m^2, need width >= 0.53 / height
        min_egress_area_m2 = EGRESS_MIN_OPENING_AREA_SQFT / M2_TO_SQFT
        min_width_for_area = min_egress_area_m2 / window_height

        # Also need minimum 20" width
        min_width_for_code = EGRESS_MIN_OPENING_WIDTH_IN / M_TO_IN

        # Use larger of: egress minimum, code minimum, or target glazing
        target_width = target_glazing / window_height
        window_width = max(min_width_for_area, min_width_for_code, target_width)

        # Add small margin to ensure compliance
        window_width += 0.05

        # Select wall (prefer exterior, avoid doors)
        wall = self._select_wall(room, self.exterior_walls)

        # Find position on wall (centered, avoiding doors)
        position = self._find_position(room, wall, window_width)

        new_window = Window(
            wall=wall,
            position=position,
            width=window_width,
            height=window_height,
            sill_height=sill_height,
        )

        return room.model_copy(update={"windows": [*room.windows, new_window]})

    def place_for_living(self, room: Room) -> Room:
        """Place windows for living room (8-10% glazing ratio).

        Creates windows totaling approximately 8% of floor area in glazing.
        May place multiple windows if single window would be too wide.

        Args:
            room: Room to add windows to

        Returns:
            New Room instance with living room windows added
        """
        floor_area = room.width * room.length
        target_glazing = floor_area * self.glazing_ratio

        window_height = LIVING_WINDOW_HEIGHT
        sill_height = LIVING_SILL_HEIGHT

        # Calculate total width needed
        total_width = target_glazing / window_height

        # Determine number of windows (max ~2m per window)
        max_window_width = 2.0
        num_windows = max(1, int(total_width / max_window_width + 0.5))
        individual_width = total_width / num_windows

        # Place windows on available walls
        new_windows = []
        walls_used = set()

        for _ in range(num_windows):
            # Select next available wall
            available_walls = [
                w for w in self.exterior_walls if w not in walls_used
            ]
            if not available_walls:
                # All walls used, allow reuse
                available_walls = self.exterior_walls.copy()

            wall = self._select_wall(room, available_walls)
            walls_used.add(wall)

            position = self._find_position(room, wall, individual_width)

            new_windows.append(Window(
                wall=wall,
                position=position,
                width=individual_width,
                height=window_height,
                sill_height=sill_height,
            ))

        return room.model_copy(update={"windows": [*room.windows, *new_windows]})

    def _select_wall(
        self,
        room: Room,
        preferred_walls: List[str],
    ) -> WallName:
        """Select best wall for window placement.

        Prefers walls without doors. Falls back to first preferred wall
        if all have doors.

        Args:
            room: Room to analyze
            preferred_walls: Walls in order of preference

        Returns:
            Selected wall name
        """
        door_walls = {door.wall for door in room.doors}

        # Try preferred walls without doors first
        for wall in preferred_walls:
            if wall not in door_walls:
                return wall  # type: ignore

        # All preferred walls have doors, use first preferred
        return preferred_walls[0]  # type: ignore

    def _find_position(
        self,
        room: Room,
        wall: WallName,
        window_width: float,
    ) -> float:
        """Find position for window on wall, avoiding doors.

        Centers window on available wall space, maintaining minimum
        gap from corners and doors.

        Args:
            room: Room containing doors
            wall: Wall to place window on
            window_width: Width of window to place

        Returns:
            Position along wall (distance from origin corner in meters)
        """
        # Get wall length based on wall orientation
        if wall in ("north", "south"):
            wall_length = room.width
        else:  # east, west
            wall_length = room.length

        # Get doors on this wall
        doors_on_wall = [d for d in room.doors if d.wall == wall]

        if not doors_on_wall:
            # No doors - center window on wall
            return (wall_length - window_width) / 2

        # Find available segments (wall sections not occupied by doors)
        # Start with full wall
        segments = [(MIN_GAP, wall_length - MIN_GAP)]

        for door in doors_on_wall:
            door_start = door.position - MIN_GAP
            door_end = door.position + door.width + MIN_GAP

            new_segments = []
            for seg_start, seg_end in segments:
                # Check if door overlaps this segment
                if door_end <= seg_start or door_start >= seg_end:
                    # No overlap, keep segment
                    new_segments.append((seg_start, seg_end))
                else:
                    # Split segment around door
                    if seg_start < door_start:
                        new_segments.append((seg_start, door_start))
                    if door_end < seg_end:
                        new_segments.append((door_end, seg_end))

            segments = new_segments

        # Find largest segment that fits window
        best_segment = None
        best_length = 0

        for seg_start, seg_end in segments:
            seg_length = seg_end - seg_start
            if seg_length >= window_width and seg_length > best_length:
                best_segment = (seg_start, seg_end)
                best_length = seg_length

        if best_segment:
            # Center window in segment
            seg_start, seg_end = best_segment
            return seg_start + (seg_end - seg_start - window_width) / 2

        # Fallback: center on wall (window may overlap with door)
        return (wall_length - window_width) / 2
