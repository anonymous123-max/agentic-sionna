"""OSMConfig model for OpenStreetMap fetch configuration.

Provides a frozen Pydantic model with validated parameters for controlling
OSM data fetching: search radius, feature toggles, height defaults, and
network type selection.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class OSMConfig(BaseModel):
    """Configuration for OSM data fetching.

    All fields have sensible defaults for typical urban scenes.
    The model is frozen (immutable) following project convention.

    Attributes:
        radius: Search radius in meters around center point for place-name
            fetches. Must be positive.
        include_buildings: Whether to fetch building footprints.
        include_roads: Whether to fetch road network.
        include_trees: Whether to fetch tree/vegetation data.
        default_building_height: Fallback height in meters when OSM has no
            height data. Must be positive.
        meters_per_level: Height per building level for building:levels to
            height conversion. Must be positive.
        road_network_type: OSMnx network type parameter for road fetching.
        simplify_tolerance: Optional polygon simplification tolerance in
            meters. None means no simplification. Useful for large areas.
    """

    model_config = ConfigDict(frozen=True)

    radius: float = 200.0
    include_buildings: bool = True
    include_roads: bool = True
    include_trees: bool = True
    default_building_height: float = 10.0
    meters_per_level: float = 3.0
    road_network_type: Literal["all", "drive", "walk", "bike"] = "drive"
    simplify_tolerance: Optional[float] = None

    @field_validator("radius")
    @classmethod
    def validate_radius(cls, v: float) -> float:
        """Ensure radius is positive."""
        if v <= 0:
            raise ValueError("Radius must be positive")
        return v

    @field_validator("default_building_height")
    @classmethod
    def validate_default_building_height(cls, v: float) -> float:
        """Ensure default building height is positive."""
        if v <= 0:
            raise ValueError("Default building height must be positive")
        return v

    @field_validator("meters_per_level")
    @classmethod
    def validate_meters_per_level(cls, v: float) -> float:
        """Ensure meters per level is positive."""
        if v <= 0:
            raise ValueError("Meters per level must be positive")
        return v
