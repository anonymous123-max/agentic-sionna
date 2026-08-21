"""Scene generation library — public API surface for room/floor-plan generation.

Pairs with the Sionna coding side of this skill. Import from inside generated agent code:

    import sys
    sys.path.insert(0, "$RF_SKILL_DIR")  # set by harness
    from lib.scene_gen import Scene, Room, place_furniture, export_all
"""
from .models import (
    Scene, Room, OutdoorScene,
    Position, BoundingBox, FurnitureItem, WallSegment, Door, Window,
    Building, Road, Tree, GroundPlane,
    Transmitter, Receiver,
)
from .geometry import aabb_overlap, in_bounds, rotate_2d
from .constraints import validate_scene
from .optimizer import place_furniture, place_tx
from .windows import place_irc_windows
from .exporters import export_png, export_xml, export_gltf, export_all

__all__ = [
    # Top-level
    "Scene",
    # Indoor
    "Room", "Position", "BoundingBox", "FurnitureItem",
    "WallSegment", "Door", "Window",
    # Outdoor
    "OutdoorScene", "Building", "Road", "Tree", "GroundPlane",
    # Sionna additions
    "Transmitter", "Receiver",
    # Geometry / constraints / optimizer
    "aabb_overlap", "in_bounds", "rotate_2d", "validate_scene",
    "place_furniture", "place_tx", "place_irc_windows",
    # Exporters
    "export_png", "export_xml", "export_gltf", "export_all",
]
