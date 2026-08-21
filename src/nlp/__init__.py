"""Natural Language Processing module for room description parsing.

Provides the RoomParser class for converting natural language room
descriptions into structured Room objects with optimized furniture layout.

Also provides ImageParser for extracting room structure from floor plan
images, and PlacementExplainer for generating human-readable explanations
of furniture placement decisions.
"""

from src.nlp.parser import RoomParser
from src.nlp.image_parser import ImageParser, ImageExtractedRoom
from src.nlp.explainer import PlacementExplainer, PlacementExplanations
from src.nlp.schemas import (
    ParsedRoomDescription,
    RawDoor,
    RawFurnitureRequest,
    RawRoomSpec,
    RawWindow,
)

__all__ = [
    "RoomParser",
    "ImageParser",
    "ImageExtractedRoom",
    "PlacementExplainer",
    "PlacementExplanations",
    "ParsedRoomDescription",
    "RawDoor",
    "RawFurnitureRequest",
    "RawRoomSpec",
    "RawWindow",
]
