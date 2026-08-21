"""Outdoor scene data models.

Single source of truth for outdoor scene elements. All models are frozen (immutable)
to ensure consistency across exports.

Coordinate system:
- Origin at SW corner of scene
- X increases east, Y increases north
- All positions in meters
"""

from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator
from shapely import LineString, Polygon
from shapely.geometry import CAP_STYLE
from shapely.validation import explain_validity


class Building(BaseModel):
    """Building with polygon footprint extruded to height.

    Footprint is stored as coordinate list for serializability.
    Use as_polygon() to get Shapely Polygon for geometry operations.

    Attributes:
        id: Unique identifier
        footprint: Polygon vertices as (x, y) tuples, CCW order
        height: Building height in meters
        material: ITU material for walls (concrete, brick, glass, metal)
        name: Human-readable name from OSM (e.g. "Athens National Library")
    """

    model_config = ConfigDict(frozen=True)

    id: str
    footprint: List[Tuple[float, float]]
    height: float
    material: Literal["concrete", "brick", "glass", "metal"] = "concrete"
    name: Optional[str] = None

    @field_validator("footprint")
    @classmethod
    def validate_footprint(
        cls, v: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Validate footprint forms valid polygon."""
        if len(v) < 3:
            raise ValueError("Footprint requires at least 3 vertices")
        # Create polygon to validate geometry
        poly = Polygon(v)
        if not poly.is_valid:
            raise ValueError(f"Invalid footprint geometry: {explain_validity(poly)}")
        return v

    @field_validator("height")
    @classmethod
    def validate_height(cls, v: float) -> float:
        """Ensure height is positive."""
        if v <= 0:
            raise ValueError("Height must be positive")
        return v

    def as_polygon(self) -> Polygon:
        """Create Shapely Polygon from footprint."""
        return Polygon(self.footprint)

    @property
    def area(self) -> float:
        """Building footprint area in square meters."""
        return self.as_polygon().area

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Bounding box as (minx, miny, maxx, maxy)."""
        return self.as_polygon().bounds


class Road(BaseModel):
    """Road segment defined by centerline and width.

    Attributes:
        id: Unique identifier
        centerline: Path as list of (x, y) points
        width: Total road width in meters
        material: ITU material (concrete, wet_ground, very_dry_ground)
    """

    model_config = ConfigDict(frozen=True)

    id: str
    centerline: List[Tuple[float, float]]
    width: float
    material: Literal["concrete", "wet_ground", "very_dry_ground"] = "concrete"

    @field_validator("centerline")
    @classmethod
    def validate_centerline(
        cls, v: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Ensure centerline has at least 2 points."""
        if len(v) < 2:
            raise ValueError("Centerline requires at least 2 points")
        return v

    @field_validator("width")
    @classmethod
    def validate_width(cls, v: float) -> float:
        """Ensure width is positive."""
        if v <= 0:
            raise ValueError("Width must be positive")
        return v

    def as_polygon(self) -> Polygon:
        """Create road polygon by buffering centerline."""
        line = LineString(self.centerline)
        # Square ends for road segments
        return line.buffer(self.width / 2, cap_style=CAP_STYLE.flat)


class Tree(BaseModel):
    """Tree with trunk and crown geometry.

    Species affects crown shape:
    - deciduous: spherical/ellipsoid crown
    - conifer: conical crown

    Attributes:
        id: Unique identifier
        position: (x, y) center of trunk base
        height: Total tree height in meters
        trunk_radius: Trunk radius in meters
        crown_radius: Crown radius in meters (widest point)
        crown_height: Height of crown portion in meters
        species: Tree type (affects crown shape for rendering)
    """

    model_config = ConfigDict(frozen=True)

    id: str
    position: Tuple[float, float]
    height: float
    trunk_radius: float = 0.15  # ~30cm diameter trunk
    crown_radius: float = 2.0  # 4m diameter crown
    crown_height: float = 3.0  # Crown is top 3m of tree
    species: Literal["deciduous", "conifer"] = "deciduous"

    @field_validator("height", "trunk_radius", "crown_radius", "crown_height")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Ensure dimensions are positive."""
        if v <= 0:
            raise ValueError("Dimension must be positive")
        return v

    @property
    def trunk_height(self) -> float:
        """Height of trunk (total minus crown)."""
        return max(0, self.height - self.crown_height)


class GroundPlane(BaseModel):
    """Ground surface for outdoor scene.

    Attributes:
        width: X dimension in meters
        length: Y dimension in meters
        material: ITU material (concrete, wet_ground, medium_dry_ground, very_dry_ground)
    """

    model_config = ConfigDict(frozen=True)

    width: float
    length: float
    material: Literal[
        "concrete", "wet_ground", "medium_dry_ground", "very_dry_ground"
    ] = "wet_ground"

    @field_validator("width", "length")
    @classmethod
    def validate_dimensions(cls, v: float) -> float:
        """Ensure dimensions are positive."""
        if v <= 0:
            raise ValueError("Dimension must be positive")
        return v


class OutdoorScene(BaseModel):
    """Container for all outdoor scene elements.

    Coordinate system matches Room:
    - Origin at SW corner
    - X increases east, Y increases north
    - All positions in meters

    Attributes:
        width: Scene width (X) in meters
        length: Scene length (Y) in meters
        ground: Ground plane geometry and material
        buildings: List of Building objects
        roads: List of Road objects
        trees: List of Tree objects
    """

    model_config = ConfigDict(frozen=True)

    width: float
    length: float
    ground: GroundPlane
    buildings: List[Building] = []
    roads: List[Road] = []
    trees: List[Tree] = []

    @field_validator("width", "length")
    @classmethod
    def validate_dimensions(cls, v: float) -> float:
        """Ensure scene dimensions are positive."""
        if v <= 0:
            raise ValueError("Scene dimension must be positive")
        return v
