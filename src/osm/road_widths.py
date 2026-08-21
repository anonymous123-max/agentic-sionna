"""Road width lookup by OSM highway type.

Maps OpenStreetMap highway classification tags to default road widths in meters.
Widths are compatible with the Road.width field in src/models/outdoor.py.

Reference: https://wiki.openstreetmap.org/wiki/Key:highway
"""

from typing import Union

# OSM highway type -> default road width in meters
ROAD_WIDTHS: dict[str, float] = {
    "motorway": 12.0,
    "trunk": 11.0,
    "primary": 10.0,
    "secondary": 8.0,
    "tertiary": 6.0,
    "residential": 5.0,
    "living_street": 4.0,
    "service": 3.5,
    "footway": 2.0,
    "cycleway": 2.0,
    "path": 1.5,
}


def get_road_width(
    highway_type: Union[str, list[str]], default: float = 5.0
) -> float:
    """Look up road width for an OSM highway type.

    Args:
        highway_type: OSM highway tag value. May be a string or a list
            (OSM sometimes returns multiple types like ["primary", "secondary"]).
            When a list, the first element is used.
        default: Width to return for unknown highway types.

    Returns:
        Road width in meters (float), compatible with Road.width field.
    """
    if isinstance(highway_type, list):
        highway_type = highway_type[0] if highway_type else ""
    return ROAD_WIDTHS.get(highway_type, default)
