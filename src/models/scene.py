"""Unified scene container for indoor, outdoor, or combined scenes.

The Scene model is the top-level container passed to exporters. It can hold:
- Room only (indoor scene) - existing v1.0 functionality
- OutdoorScene only (outdoor scene) - new v1.1 outdoor generation
- Both Room and OutdoorScene (combined) - future combined scenarios

Coordinate system:
- All components use origin at SW corner
- X increases east, Y increases north
- Positions in meters
"""

from typing import Dict, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from src.models.room import Room
from src.models.outdoor import OutdoorScene


class Scene(BaseModel):
    """Unified container for indoor, outdoor, or combined scenes.

    At least one of room or outdoor must be present.
    When both are present, represents a combined indoor/outdoor scenario
    (e.g., room inside a building in an outdoor scene).

    Attributes:
        room: Optional indoor room with furniture
        outdoor: Optional outdoor scene with buildings/roads/trees
        building_rooms: Mapping of building IDs to interior Room objects.
            Populated when user drills into a building from outdoor view.
    """

    model_config = ConfigDict(frozen=True)

    room: Optional[Room] = None
    outdoor: Optional[OutdoorScene] = None
    building_rooms: Dict[str, Room] = {}

    @model_validator(mode="after")
    def validate_has_content(self) -> "Scene":
        """Ensure scene has at least one component."""
        if self.room is None and self.outdoor is None:
            raise ValueError("Scene must contain room, outdoor, or both")
        return self

    @property
    def is_indoor_only(self) -> bool:
        """True if scene contains only a room."""
        return self.room is not None and self.outdoor is None

    @property
    def is_outdoor_only(self) -> bool:
        """True if scene contains only outdoor elements."""
        return self.outdoor is not None and self.room is None

    @property
    def is_combined(self) -> bool:
        """True if scene contains both indoor and outdoor."""
        return self.room is not None and self.outdoor is not None
