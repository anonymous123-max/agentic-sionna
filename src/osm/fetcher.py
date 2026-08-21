"""OSMnx wrapper for downloading buildings, roads, and trees from OpenStreetMap.

Provides three fetch functions that return projected UTM GeoDataFrames:
- fetch_buildings: Building footprint polygons
- fetch_roads: Road network edges as linestrings
- fetch_trees: Individual tree point locations

OSMnx is lazily imported so the module can be imported without osmnx installed.
A helpful ImportError is raised when fetch functions are actually called.
"""

import warnings
from typing import Optional, Tuple

import geopandas as gpd


def _import_osmnx():
    """Lazy import of osmnx with helpful error message."""
    try:
        import osmnx as ox

        return ox
    except ImportError:
        raise ImportError(
            "osmnx is required for OpenStreetMap integration. "
            "Install it with: pip install 'sionna-skill[osm]'"
        )


def _validate_place_or_bbox(
    place: Optional[str], bbox: Optional[Tuple[float, float, float, float]]
) -> None:
    """Validate that exactly one of place or bbox is provided."""
    if place is not None and bbox is not None:
        raise ValueError("Provide either 'place' or 'bbox', not both")
    if place is None and bbox is None:
        raise ValueError("Provide either 'place' or 'bbox'")


def fetch_buildings(
    place: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> gpd.GeoDataFrame:
    """Fetch building footprints from OpenStreetMap.

    Downloads building polygons and projects them to UTM coordinates (meters).
    Filters out non-polygon geometries (Points, Lines) that sometimes appear
    in OSM building data.

    Args:
        place: Place name to geocode (e.g. "Times Square, NYC").
        bbox: Bounding box as (north, south, east, west) in decimal degrees.

    Returns:
        GeoDataFrame with building polygons in projected UTM coordinates.

    Raises:
        ValueError: If neither or both of place/bbox are provided.
        ImportError: If osmnx is not installed.
    """
    _validate_place_or_bbox(place, bbox)
    ox = _import_osmnx()

    try:
        if place is not None:
            gdf = ox.features_from_place(place, tags={"building": True})
        else:
            gdf = ox.features_from_bbox(bbox=bbox, tags={"building": True})
    except ox._errors.InsufficientResponseError:
        warnings.warn(
            f"No building data found for the specified area. "
            f"The area may lack OSM building coverage.",
            stacklevel=2,
        )
        return gpd.GeoDataFrame()

    if gdf.empty:
        warnings.warn("No buildings found in the specified area.", stacklevel=2)
        return gdf

    # Filter to Polygon/MultiPolygon only (skip Points/Lines)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    if gdf.empty:
        warnings.warn(
            "Buildings found but none have polygon geometries.", stacklevel=2
        )
        return gdf

    # Project to UTM (meters)
    gdf = ox.projection.project_gdf(gdf)
    return gdf


def fetch_roads(
    place: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    network_type: str = "drive",
) -> gpd.GeoDataFrame:
    """Fetch road network from OpenStreetMap.

    Downloads the road graph and converts to edge GeoDataFrame in UTM
    coordinates (meters).

    Args:
        place: Place name to geocode (e.g. "Shibuya, Tokyo").
        bbox: Bounding box as (north, south, east, west) in decimal degrees.
        network_type: OSMnx network type: "all", "drive", "walk", "bike".

    Returns:
        GeoDataFrame with road linestrings in projected UTM coordinates.

    Raises:
        ValueError: If neither or both of place/bbox are provided.
        ImportError: If osmnx is not installed.
    """
    _validate_place_or_bbox(place, bbox)
    ox = _import_osmnx()

    try:
        if place is not None:
            G = ox.graph_from_place(place, network_type=network_type)
        else:
            G = ox.graph_from_bbox(bbox=bbox, network_type=network_type)
    except ox._errors.InsufficientResponseError:
        warnings.warn(
            f"No road data found for the specified area. "
            f"The area may lack OSM road coverage.",
            stacklevel=2,
        )
        return gpd.GeoDataFrame()

    # Convert graph to edges GeoDataFrame
    edges = ox.graph_to_gdfs(G, nodes=False)

    if edges.empty:
        warnings.warn("No roads found in the specified area.", stacklevel=2)
        return edges

    # Project to UTM (meters)
    edges = ox.projection.project_gdf(edges)
    return edges


def fetch_trees(
    place: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> gpd.GeoDataFrame:
    """Fetch tree locations from OpenStreetMap.

    Downloads individual tree point locations and projects to UTM coordinates
    (meters). Many areas lack tree data, so an empty result is common.

    Args:
        place: Place name to geocode (e.g. "Central Park, New York").
        bbox: Bounding box as (north, south, east, west) in decimal degrees.

    Returns:
        GeoDataFrame with tree points in projected UTM coordinates.
        May be empty if the area lacks tree data.

    Raises:
        ValueError: If neither or both of place/bbox are provided.
        ImportError: If osmnx is not installed.
    """
    _validate_place_or_bbox(place, bbox)
    ox = _import_osmnx()

    try:
        if place is not None:
            gdf = ox.features_from_place(place, tags={"natural": "tree"})
        else:
            gdf = ox.features_from_bbox(bbox=bbox, tags={"natural": "tree"})
    except ox._errors.InsufficientResponseError:
        warnings.warn(
            f"No tree data found for the specified area. "
            f"Many areas lack individual tree data in OSM.",
            stacklevel=2,
        )
        return gpd.GeoDataFrame()

    if gdf.empty:
        warnings.warn(
            "No trees found in the specified area. "
            "Many areas lack individual tree data in OSM.",
            stacklevel=2,
        )
        return gdf

    # Filter to Point geometries only
    gdf = gdf[gdf.geometry.type == "Point"].copy()

    if gdf.empty:
        warnings.warn(
            "Tree features found but none are point geometries.", stacklevel=2
        )
        return gdf

    # Project to UTM (meters)
    gdf = ox.projection.project_gdf(gdf)
    return gdf
