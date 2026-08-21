"""OpenStreetMap integration for real-world outdoor scenes.

Provides lookup tables, material mappings, configuration, fetcher, and
converter for downloading real-world building/road/tree data from
OpenStreetMap via OSMnx and converting to OutdoorScene models.

Usage:
    from src.osm import fetch_outdoor_scene, OSMConfig

    scene = fetch_outdoor_scene("Times Square, NYC")

    config = OSMConfig(radius=150, default_building_height=12.0)
    scene = fetch_outdoor_scene("Shibuya, Tokyo", config=config)

    scene = fetch_outdoor_scene_bbox(
        north=40.759, south=40.755, east=-73.983, west=-73.988
    )
"""

from src.osm.config import OSMConfig
from src.osm.converter import fetch_outdoor_scene, fetch_outdoor_scene_bbox
from src.osm.material_mapping import (
    BUILDING_MATERIALS,
    ROAD_MATERIALS,
    map_building_material,
    map_road_material,
)
from src.osm.road_widths import ROAD_WIDTHS, get_road_width

__all__ = [
    "OSMConfig",
    "fetch_outdoor_scene",
    "fetch_outdoor_scene_bbox",
    "get_road_width",
    "ROAD_WIDTHS",
    "map_building_material",
    "map_road_material",
    "BUILDING_MATERIALS",
    "ROAD_MATERIALS",
]
