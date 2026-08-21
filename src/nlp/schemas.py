"""Pydantic schemas for NLP extraction.

Intermediate data models for LLM structured output extraction.
These models are designed to capture natural language room descriptions
before they are converted to Room objects.

All fields include descriptions to guide LLM extraction.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class RawDoor(BaseModel):
    """Door specification from natural language.

    Attributes:
        wall: Which wall the door is on
        position_along_wall: Distance from start of wall in meters
        width: Door width in meters
    """

    wall: Literal["north", "south", "east", "west"] = Field(
        description="Which wall the door is located on (north, south, east, or west)"
    )
    position_along_wall: float = Field(
        default=1.0,
        description="Distance in meters from the start of the wall to the door center. "
        "For south/north walls, this is distance from west corner. "
        "For east/west walls, this is distance from south corner.",
    )
    width: float = Field(
        default=0.9,
        description="Width of the door opening in meters. Standard doors are 0.8-0.9m.",
    )


class RawWindow(BaseModel):
    """Window specification from natural language.

    Attributes:
        wall: Which wall the window is on
        position_along_wall: Distance from start of wall in meters
        width: Window width in meters
        height: Window height in meters
        sill_height: Height of window sill from floor in meters
    """

    wall: Literal["north", "south", "east", "west"] = Field(
        description="Which wall the window is located on"
    )
    position_along_wall: float = Field(
        default=1.5,
        description="Distance in meters from the start of the wall to the window center",
    )
    width: float = Field(
        default=1.2,
        description="Width of the window in meters",
    )
    height: float = Field(
        default=1.2,
        description="Height of the window in meters",
    )
    sill_height: float = Field(
        default=0.9,
        description="Height of the window sill from floor in meters",
    )


class RawFurnitureRequest(BaseModel):
    """Furniture request from natural language.

    Attributes:
        category: Type of furniture (e.g., bed, sofa, desk, chair)
        quantity: Number of this furniture type to place
        style: Optional style preference (e.g., modern, minimalist)
        preferred_wall: Optional wall placement preference
    """

    category: str = Field(
        description="Type of furniture. Use standard category names: "
        "bed, sofa, desk, chair, table, wardrobe, nightstand, "
        "bookshelf, cabinet, dresser, coffee table, dining table. "
        "For colloquial terms: couch->sofa, settee->sofa, "
        "closet->wardrobe, bedside table->nightstand."
    )
    quantity: int = Field(
        default=1,
        description="Number of this furniture item to place. "
        "Use explicit numbers from description or infer (e.g., 'pair of nightstands' = 2).",
    )
    style: Optional[str] = Field(
        default=None,
        description="Style preference if mentioned (e.g., modern, minimalist, rustic, industrial)",
    )
    preferred_wall: Optional[Literal["north", "south", "east", "west"]] = Field(
        default=None,
        description="Which wall to place furniture against, if specified",
    )


class RawRoomSpec(BaseModel):
    """Room specification from natural language.

    Attributes:
        width_meters: Room width (east-west dimension) in meters
        length_meters: Room length (north-south dimension) in meters
        height_meters: Ceiling height in meters
        room_type: Type of room for automatic window placement
    """

    width_meters: float = Field(
        description="Room width in meters (east-west dimension, X axis). "
        "Extract from descriptions like '4 meter by 5 meter' (4 is width), "
        "'4x5 meter' (4 is width), or '5m wide' (5 is width)."
    )
    length_meters: float = Field(
        description="Room length in meters (north-south dimension, Y axis). "
        "Extract from descriptions like '4 meter by 5 meter' (5 is length), "
        "'4x5 meter' (5 is length), or '6m long' (6 is length)."
    )
    height_meters: float = Field(
        default=2.7,
        description="Ceiling height in meters. Default is 2.7m if not specified.",
    )
    room_type: Optional[Literal["bedroom", "living_room", "office", "other"]] = Field(
        default=None,
        description="Type of room if mentioned or inferable. "
        "bedroom -> requires egress-compliant window; "
        "living_room (or family room) -> requires glazing ratio window; "
        "office/other -> no automatic windows. "
        "Infer from context: '4x5 bedroom' is bedroom, 'living room with sofa' is living_room.",
    )


class ParsedRoomDescription(BaseModel):
    """Complete parsed room description from natural language.

    This is the top-level model extracted by the LLM.
    Contains room dimensions, doors, windows, and furniture requests.

    Coordinate system:
    - Origin at SW corner of room
    - X axis points east (width direction)
    - Y axis points north (length direction)
    - Walls named by compass direction they face
    """

    room: RawRoomSpec = Field(
        description="Room dimensions extracted from description"
    )
    doors: List[RawDoor] = Field(
        default_factory=list,
        description="List of doors mentioned in the description. "
        "Default to south wall if wall not specified.",
    )
    windows: List[RawWindow] = Field(
        default_factory=list,
        description="List of windows mentioned in the description",
    )
    furniture: List[RawFurnitureRequest] = Field(
        default_factory=list,
        description="List of furniture items to place in the room",
    )
