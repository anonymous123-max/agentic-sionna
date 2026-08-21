"""OSM GeoDataFrame to OutdoorScene conversion.

Transforms projected UTM GeoDataFrames from fetcher.py into OutdoorScene
model instances. Handles coordinate centering, height parsing, and geometry
translation so all scene coordinates are in local meters with origin at the
south-west corner.

Public API:
    fetch_outdoor_scene(place, config) -> OutdoorScene
    fetch_outdoor_scene_bbox(north, south, east, west, config) -> OutdoorScene

Internal converters:
    convert_building, convert_road, convert_tree, convert_to_outdoor_scene
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import geopandas as gpd
from shapely.affinity import translate
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

from src.models.outdoor import (
    Building,
    GroundPlane,
    OutdoorScene,
    Road,
    Tree,
)
from src.osm.config import OSMConfig
from src.osm.fetcher import fetch_buildings, fetch_roads, fetch_trees
from src.osm.material_mapping import map_building_material, map_road_material
from src.osm.road_widths import get_road_width


def _parse_height(value) -> Optional[float]:
    """Parse an OSM height value to float meters.

    Handles common formats:
    - "45" -> 45.0
    - "45 m" -> 45.0
    - "45m" -> 45.0
    - "150'" -> 45.72 (feet to meters)
    - None / NaN -> None

    Args:
        value: Raw OSM height tag value.

    Returns:
        Height in meters as float, or None if unparseable.
    """
    if value is None:
        return None

    # Handle NaN (pandas)
    try:
        if math.isnan(value):
            return None
    except (TypeError, ValueError):
        pass

    s = str(value).strip()
    if not s:
        return None

    # Handle feet notation
    if s.endswith("'"):
        try:
            return float(s[:-1]) * 0.3048
        except ValueError:
            return None

    # Remove common unit suffixes
    for suffix in (" m", "m", " meters", " metre"):
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)].strip()
            break

    try:
        result = float(s)
        if result <= 0:
            return None
        return result
    except ValueError:
        return None


def convert_building(
    geometry, row, index: int, config: OSMConfig
) -> Union[Building, List[Building]]:
    """Convert an OSM building geometry and attributes to Building model(s).

    Height is determined by 3-tier fallback:
    1. OSM height tag (parsed from string)
    2. building:levels * config.meters_per_level
    3. config.default_building_height

    Args:
        geometry: Shapely Polygon or MultiPolygon (already translated).
        row: GeoDataFrame row with OSM tags.
        index: Row index for generating unique IDs.
        config: OSMConfig with height defaults.

    Returns:
        Single Building for Polygon, list of Buildings for MultiPolygon.
    """
    # 3-tier height fallback
    height = _parse_height(row.get("height"))
    if height is None:
        levels = _parse_height(row.get("building:levels"))
        if levels is not None:
            height = levels * config.meters_per_level
    if height is None:
        height = config.default_building_height

    # Material
    material = map_building_material(row.get("building:material"))

    # Name from OSM tags (try name, then addr:housename, then building type)
    import pandas as pd
    raw_name = row.get("name")
    if pd.isna(raw_name) if isinstance(raw_name, float) else not raw_name:
        raw_name = row.get("addr:housename")
    if pd.isna(raw_name) if isinstance(raw_name, float) else not raw_name:
        raw_name = None
    name = str(raw_name).strip() if raw_name else None

    # Optional simplification
    if config.simplify_tolerance:
        geometry = geometry.simplify(config.simplify_tolerance)

    if isinstance(geometry, MultiPolygon):
        buildings = []
        for j, poly in enumerate(geometry.geoms):
            footprint = list(poly.exterior.coords[:-1])  # drop closing duplicate
            if len(footprint) >= 3:
                buildings.append(
                    Building(
                        id=f"bldg_{index}_{j}",
                        footprint=footprint,
                        height=height,
                        material=material,
                        name=name,
                    )
                )
        return buildings
    else:
        # Single Polygon
        footprint = list(geometry.exterior.coords[:-1])  # drop closing duplicate
        return Building(
            id=f"bldg_{index}",
            footprint=footprint,
            height=height,
            material=material,
            name=name,
        )


def convert_road(geometry, row, index: int, config: OSMConfig) -> Optional[Road]:
    """Convert an OSM road geometry and attributes to Road model.

    Width is derived from highway type classification via road_widths module.

    Args:
        geometry: Shapely LineString or MultiLineString (already translated).
        row: GeoDataFrame row with OSM tags.
        index: Row index for generating unique ID.
        config: OSMConfig (unused currently, reserved for future options).

    Returns:
        Road instance, or None if geometry is invalid.
    """
    # Extract highway type
    highway_type = row.get("highway", "residential")
    width = get_road_width(highway_type)

    # Material
    material = map_road_material(row.get("surface"))

    # Extract centerline coordinates
    if isinstance(geometry, MultiLineString):
        # Use first line of MultiLineString
        if len(geometry.geoms) == 0:
            return None
        centerline = list(geometry.geoms[0].coords)
    elif isinstance(geometry, LineString):
        centerline = list(geometry.coords)
    else:
        return None

    if len(centerline) < 2:
        return None

    # Convert 3D coords to 2D if needed (OSMnx sometimes adds Z)
    centerline = [(x, y) for x, y, *_ in centerline]

    return Road(
        id=f"road_{index}",
        centerline=centerline,
        width=width,
        material=material,
    )


def convert_tree(geometry, row, index: int, config: OSMConfig) -> Optional[Tree]:
    """Convert an OSM tree point and attributes to Tree model.

    Args:
        geometry: Shapely Point (already translated).
        row: GeoDataFrame row with OSM tags.
        index: Row index for generating unique ID.
        config: OSMConfig (unused currently, reserved for future options).

    Returns:
        Tree instance, or None if geometry is not a Point.
    """
    if not isinstance(geometry, Point):
        return None

    position = (geometry.x, geometry.y)

    # Height with fallback
    height = _parse_height(row.get("height"))
    if height is None:
        height = 8.0

    # Crown radius from diameter_crown
    crown_radius = _parse_height(row.get("diameter_crown"))
    if crown_radius is not None:
        crown_radius = crown_radius / 2.0
    else:
        crown_radius = 3.0

    # Species from leaf_type
    leaf_type = row.get("leaf_type")
    species = "conifer" if leaf_type == "needleleaved" else "deciduous"

    return Tree(
        id=f"tree_{index}",
        position=position,
        height=height,
        crown_radius=crown_radius,
        species=species,
    )


def convert_to_outdoor_scene(
    buildings_gdf: Optional[gpd.GeoDataFrame],
    roads_gdf: Optional[gpd.GeoDataFrame],
    trees_gdf: Optional[gpd.GeoDataFrame],
    config: OSMConfig,
) -> OutdoorScene:
    """Convert OSM GeoDataFrames to an OutdoorScene.

    Computes combined bounds from all non-empty GeoDataFrames, translates
    all geometries so the SW corner is at (0, 0), then converts each
    feature to the corresponding model type.

    Args:
        buildings_gdf: Projected building polygons (may be None or empty).
        roads_gdf: Projected road linestrings (may be None or empty).
        trees_gdf: Projected tree points (may be None or empty).
        config: OSMConfig with height defaults and options.

    Returns:
        OutdoorScene with buildings, roads, trees with origin at SW corner.
    """
    # Collect non-empty GeoDataFrames for bounds computation
    gdfs = []
    for gdf in [buildings_gdf, roads_gdf, trees_gdf]:
        if gdf is not None and not gdf.empty:
            gdfs.append(gdf)

    if not gdfs:
        warnings.warn(
            "No OSM features found. Returning minimal empty scene.", stacklevel=2
        )
        return OutdoorScene(
            width=100.0,
            length=100.0,
            ground=GroundPlane(width=100.0, length=100.0, material="wet_ground"),
        )

    # Compute combined bounds
    all_bounds = [gdf.total_bounds for gdf in gdfs]  # [minx, miny, maxx, maxy]
    minx = min(b[0] for b in all_bounds)
    miny = min(b[1] for b in all_bounds)
    maxx = max(b[2] for b in all_bounds)
    maxy = max(b[3] for b in all_bounds)

    # Scene dimensions with padding
    padding = 10.0
    width = (maxx - minx) + padding
    length = (maxy - miny) + padding

    # Origin at SW corner of bounds (with half-padding offset so scene
    # coordinates run from (0, 0) to (width, length), matching the
    # OutdoorScene coordinate convention documented in outdoor.py)
    origin_x = minx - padding / 2
    origin_y = miny - padding / 2

    # Translate all geometries so SW corner is at (0, 0)
    for gdf in gdfs:
        gdf.geometry = gdf.geometry.apply(
            lambda geom: translate(geom, xoff=-origin_x, yoff=-origin_y)
        )

    # Convert buildings
    buildings: List[Building] = []
    if buildings_gdf is not None and not buildings_gdf.empty:
        for i, (_, row) in enumerate(buildings_gdf.iterrows()):
            result = convert_building(row.geometry, row, i, config)
            if isinstance(result, list):
                buildings.extend(result)
            else:
                buildings.append(result)

    # Convert roads
    roads: List[Road] = []
    if roads_gdf is not None and not roads_gdf.empty:
        for i, (_, row) in enumerate(roads_gdf.iterrows()):
            road = convert_road(row.geometry, row, i, config)
            if road is not None:
                roads.append(road)

    # Convert trees
    trees: List[Tree] = []
    if trees_gdf is not None and not trees_gdf.empty:
        for i, (_, row) in enumerate(trees_gdf.iterrows()):
            tree = convert_tree(row.geometry, row, i, config)
            if tree is not None:
                trees.append(tree)

    # Ground plane
    ground = GroundPlane(width=width, length=length, material="wet_ground")

    return OutdoorScene(
        width=width,
        length=length,
        ground=ground,
        buildings=buildings,
        roads=roads,
        trees=trees,
    )


def fetch_outdoor_scene(
    place: str, config: Optional[OSMConfig] = None
) -> OutdoorScene:
    """Fetch a real-world location from OpenStreetMap and convert to OutdoorScene.

    Downloads buildings, roads, and trees (based on config flags) for the
    specified place name, then converts to an OutdoorScene with coordinates
    in local meters and origin at the SW corner.

    Args:
        place: Place name to geocode (e.g. "Times Square, NYC").
        config: Fetch configuration. Uses defaults if None.

    Returns:
        OutdoorScene with real-world buildings, roads, and trees.

    Raises:
        ImportError: If osmnx is not installed.
    """
    if config is None:
        config = OSMConfig()

    buildings_gdf = None
    roads_gdf = None
    trees_gdf = None

    if config.include_buildings:
        buildings_gdf = fetch_buildings(place=place)

    if config.include_roads:
        roads_gdf = fetch_roads(place=place, network_type=config.road_network_type)

    if config.include_trees:
        trees_gdf = fetch_trees(place=place)

    return convert_to_outdoor_scene(buildings_gdf, roads_gdf, trees_gdf, config)


def fetch_outdoor_scene_bbox(
    north: float,
    south: float,
    east: float,
    west: float,
    config: Optional[OSMConfig] = None,
) -> OutdoorScene:
    """Fetch a bounding box area from OpenStreetMap and convert to OutdoorScene.

    Downloads buildings, roads, and trees (based on config flags) for the
    specified bounding box, then converts to an OutdoorScene with coordinates
    in local meters and origin at the SW corner.

    Args:
        north: Northern latitude in decimal degrees.
        south: Southern latitude in decimal degrees.
        east: Eastern longitude in decimal degrees.
        west: Western longitude in decimal degrees.
        config: Fetch configuration. Uses defaults if None.

    Returns:
        OutdoorScene with real-world buildings, roads, and trees.

    Raises:
        ImportError: If osmnx is not installed.
    """
    if config is None:
        config = OSMConfig()

    # osmnx 2.0 expects bbox as (left, bottom, right, top) = (west, south, east, north)
    bbox = (west, south, east, north)

    buildings_gdf = None
    roads_gdf = None
    trees_gdf = None

    if config.include_buildings:
        buildings_gdf = fetch_buildings(bbox=bbox)

    if config.include_roads:
        roads_gdf = fetch_roads(bbox=bbox, network_type=config.road_network_type)

    if config.include_trees:
        trees_gdf = fetch_trees(bbox=bbox)

    return convert_to_outdoor_scene(buildings_gdf, roads_gdf, trees_gdf, config)
