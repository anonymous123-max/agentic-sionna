"""Data models for Sionna Skill.

This module provides the authoritative data models for room layouts
and outdoor scenes. All exports (floor plan, XML, GLTF) derive from
these models.

Models:
- Room: Indoor room with furniture, doors, windows
- OutdoorScene: Outdoor environment with buildings, roads, trees
- Scene: Unified container for indoor, outdoor, or combined scenes
"""

from src.models.room import (
    Position,
    BoundingBox,
    Door,
    Window,
    FurnitureItem,
    Room,
)
from src.models.outdoor import (
    Building,
    Road,
    Tree,
    GroundPlane,
    OutdoorScene,
)
from src.models.scene import Scene

__all__ = [
    # Room models
    "Position",
    "BoundingBox",
    "Door",
    "Window",
    "FurnitureItem",
    "Room",
    # Outdoor models
    "Building",
    "Road",
    "Tree",
    "GroundPlane",
    "OutdoorScene",
    # Unified container
    "Scene",
]
