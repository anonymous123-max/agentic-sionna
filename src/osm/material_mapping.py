"""OSM material tag to ITU material mapping.

Converts OpenStreetMap building:material and surface tags to ITU material
Literal types matching the Building.material and Road.material fields in
src/models/outdoor.py.

Building.material: Literal["concrete", "brick", "glass", "metal"]
Road.material: Literal["concrete", "wet_ground", "very_dry_ground"]
"""

import math
from typing import Optional

# OSM building:material tag -> ITU material matching Building.material Literal
BUILDING_MATERIALS: dict[str, str] = {
    "brick": "brick",
    "concrete": "concrete",
    "glass": "glass",
    "metal": "metal",
    "steel": "metal",
    "stone": "concrete",      # conservative RF fallback
    "wood": "concrete",       # conservative RF fallback
    "plaster": "concrete",    # conservative RF fallback
}

# OSM surface tag -> ITU material matching Road.material Literal
ROAD_MATERIALS: dict[str, str] = {
    "asphalt": "concrete",
    "concrete": "concrete",
    "paved": "concrete",
    "gravel": "very_dry_ground",
    "unpaved": "very_dry_ground",
    "ground": "wet_ground",
    "grass": "wet_ground",
}


def map_building_material(
    osm_material: Optional[str], default: str = "concrete"
) -> str:
    """Map an OSM building:material tag to an ITU material Literal.

    Args:
        osm_material: Value of the OSM building:material tag, or None
            if the tag is missing.
        default: Fallback material for missing or unknown tags.

    Returns:
        One of "concrete", "brick", "glass", "metal" matching
        Building.material Literal type.
    """
    if osm_material is None:
        return default
    if not isinstance(osm_material, str):
        # Handle NaN from pandas and other non-string values
        try:
            if math.isnan(osm_material):
                return default
        except (TypeError, ValueError):
            pass
        return default
    return BUILDING_MATERIALS.get(osm_material.lower(), default)


def map_road_material(
    osm_surface: Optional[str], default: str = "concrete"
) -> str:
    """Map an OSM surface tag to an ITU road material Literal.

    Args:
        osm_surface: Value of the OSM surface tag, or None
            if the tag is missing.
        default: Fallback material for missing or unknown tags.

    Returns:
        One of "concrete", "wet_ground", "very_dry_ground" matching
        Road.material Literal type.
    """
    if osm_surface is None:
        return default
    if not isinstance(osm_surface, str):
        # Handle NaN from pandas and other non-string values
        try:
            if math.isnan(osm_surface):
                return default
        except (TypeError, ValueError):
            pass
        return default
    return ROAD_MATERIALS.get(osm_surface.lower(), default)
