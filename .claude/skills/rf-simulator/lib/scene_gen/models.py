"""Pydantic v2 data models for scene generation.

Three model groups in one file:
- Indoor: Room, Door, Window, Position, BoundingBox, FurnitureItem, WallSegment
- Outdoor: OutdoorScene, Building, Road, Tree, GroundPlane
- Container: Scene

Coordinate system across all models:
- Origin at SW corner
- X increases east, Y increases north
- All positions in meters
- theta=0 means facing positive Y (north); stored in radians, normalized
  to [0, 2pi)

All models are frozen (immutable). Mutate via .model_copy(update={...}).
"""
from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from shapely import LineString
from shapely import Polygon as ShapelyPolygon
from shapely.geometry import CAP_STYLE
from shapely.validation import explain_validity


# ------------------------------------------------------------------
# Indoor primitives
# ------------------------------------------------------------------

class Position(BaseModel):
    """Position with x, y coordinates and rotation theta.

    theta is normalized to [0, 2*pi) at construction time. theta=0 means
    facing north (+Y).
    """
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    theta: float

    @field_validator("theta")
    @classmethod
    def normalize_theta(cls, v: float) -> float:
        two_pi = 2 * math.pi
        normalized = v % two_pi
        if normalized < 0:
            normalized += two_pi
        return normalized


class WallSegment(BaseModel):
    """A single wall edge of a polygon room (CCW orientation)."""
    model_config = ConfigDict(frozen=True)

    start: Tuple[float, float]
    end: Tuple[float, float]

    @property
    def length(self) -> float:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx * dx + dy * dy)

    @property
    def midpoint(self) -> Tuple[float, float]:
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2,
        )

    @property
    def direction(self) -> Tuple[float, float]:
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        seg_len = self.length
        if seg_len < 1e-9:
            return (1.0, 0.0)
        return (dx / seg_len, dy / seg_len)

    @property
    def inward_normal(self) -> Tuple[float, float]:
        """Unit inward normal — for CCW polygon, rotate direction 90 CW."""
        dx, dy = self.direction
        return (dy, -dx)

    @property
    def cardinal_direction(self) -> Literal["north", "south", "east", "west"]:
        nx, ny = self.inward_normal
        if abs(ny) >= abs(nx):
            return "north" if ny > 0 else "south"
        return "east" if nx > 0 else "west"

    @property
    def facing_angle(self) -> float:
        nx, ny = self.inward_normal
        return math.atan2(ny, nx)

    @property
    def angle(self) -> float:
        dx, dy = self.direction
        return math.atan2(dy, dx)


class BoundingBox(BaseModel):
    """Furniture dimensions in meters: width perp to facing, depth in facing dir."""
    model_config = ConfigDict(frozen=True)

    width: float
    depth: float
    height: float

    @field_validator("width", "depth", "height")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Dimension must be positive")
        return v


class Door(BaseModel):
    """Door in a room wall."""
    model_config = ConfigDict(frozen=True)

    wall: Literal["north", "south", "east", "west"]
    position: float
    width: float
    wall_segment_index: Optional[int] = None


class Window(BaseModel):
    """Window in a room wall."""
    model_config = ConfigDict(frozen=True)

    wall: Literal["north", "south", "east", "west"]
    position: float
    width: float
    height: float
    sill_height: float
    wall_segment_index: Optional[int] = None


class FurnitureItem(BaseModel):
    """Single piece of furniture with fixed position.

    orientation_offset compensates for inconsistent 3D-FUTURE model
    orientations. ALL exporters MUST add orientation_offset to position.theta.
    """
    model_config = ConfigDict(frozen=True)

    id: str
    category: str
    model_id: str
    model_path: str
    model_file: str = ""
    position: Position
    dimensions: BoundingBox
    orientation_offset: float = 0.0

    def get_mesh_path(self) -> str:
        if self.model_file:
            return self.model_file
        if self.model_path:
            return f"{self.model_path}/raw_model.obj"
        return ""


# ------------------------------------------------------------------
# Indoor container
# ------------------------------------------------------------------

class Room(BaseModel):
    """Room layout — single source of truth for indoor scenes.

    Rectangular: floor_polygon=None, uses width/length.
    Polygon: floor_polygon is CCW vertex list (origin-normalized);
    width/length must equal the polygon bbox.
    """
    model_config = ConfigDict(frozen=True)

    width: float
    length: float
    height: float = 2.7
    floor_polygon: Optional[List[Tuple[float, float]]] = None
    doors: List[Door] = []
    windows: List[Window] = []
    furniture: List[FurnitureItem] = []

    @field_validator("width", "length", "height")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Room dimension must be positive")
        return v

    @field_validator("floor_polygon")
    @classmethod
    def validate_polygon(
        cls, v: Optional[List[Tuple[float, float]]]
    ) -> Optional[List[Tuple[float, float]]]:
        if v is None:
            return v
        if len(v) < 3:
            raise ValueError("Floor polygon requires at least 3 vertices")
        poly = ShapelyPolygon(v)
        if not poly.is_valid:
            raise ValueError(f"Invalid floor polygon: {explain_validity(poly)}")
        return v

    @model_validator(mode="after")
    def validate_polygon_bbox(self) -> "Room":
        if self.floor_polygon is not None:
            poly = ShapelyPolygon(self.floor_polygon)
            minx, miny, maxx, maxy = poly.bounds
            bbox_w = maxx - minx
            bbox_l = maxy - miny
            if abs(self.width - bbox_w) > 0.01 or abs(self.length - bbox_l) > 0.01:
                raise ValueError(
                    f"width/length ({self.width:.2f}, {self.length:.2f}) must match "
                    f"polygon bbox ({bbox_w:.2f}, {bbox_l:.2f})"
                )
        return self

    @property
    def is_rectangular(self) -> bool:
        return self.floor_polygon is None

    @property
    def shapely_polygon(self) -> ShapelyPolygon:
        if self.floor_polygon is not None:
            return ShapelyPolygon(self.floor_polygon)
        return ShapelyPolygon([
            (0, 0), (self.width, 0),
            (self.width, self.length), (0, self.length),
        ])

    @property
    def wall_segments(self) -> List[WallSegment]:
        coords = list(self.shapely_polygon.exterior.coords)
        segments: List[WallSegment] = []
        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i + 1]
            segments.append(WallSegment(start=(a[0], a[1]), end=(b[0], b[1])))
        return segments

    @classmethod
    def from_polygon(
        cls,
        polygon: List[Tuple[float, float]],
        height: float = 2.7,
        doors: Optional[List[Door]] = None,
        windows: Optional[List[Window]] = None,
        furniture: Optional[List[FurnitureItem]] = None,
    ) -> "Room":
        poly = ShapelyPolygon(polygon)
        minx, miny, maxx, maxy = poly.bounds
        normalized = [(x - minx, y - miny) for x, y in polygon]
        return cls(
            width=maxx - minx,
            length=maxy - miny,
            height=height,
            floor_polygon=normalized,
            doors=doors or [],
            windows=windows or [],
            furniture=furniture or [],
        )


# ------------------------------------------------------------------
# Outdoor primitives + container
# ------------------------------------------------------------------

class Building(BaseModel):
    """Building with polygon footprint extruded to height."""
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
        if len(v) < 3:
            raise ValueError("Footprint requires at least 3 vertices")
        poly = ShapelyPolygon(v)
        if not poly.is_valid:
            raise ValueError(f"Invalid footprint geometry: {explain_validity(poly)}")
        return v

    @field_validator("height")
    @classmethod
    def validate_height(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Height must be positive")
        return v

    def as_polygon(self) -> ShapelyPolygon:
        return ShapelyPolygon(self.footprint)

    @property
    def area(self) -> float:
        return self.as_polygon().area

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.as_polygon().bounds


class Road(BaseModel):
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
        if len(v) < 2:
            raise ValueError("Centerline requires at least 2 points")
        return v

    @field_validator("width")
    @classmethod
    def validate_width(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Width must be positive")
        return v

    def as_polygon(self) -> ShapelyPolygon:
        line = LineString(self.centerline)
        return line.buffer(self.width / 2, cap_style=CAP_STYLE.flat)


class Tree(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    position: Tuple[float, float]
    height: float
    trunk_radius: float = 0.15
    crown_radius: float = 2.0
    crown_height: float = 3.0
    species: Literal["deciduous", "conifer"] = "deciduous"

    @field_validator("height", "trunk_radius", "crown_radius", "crown_height")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Dimension must be positive")
        return v

    @property
    def trunk_height(self) -> float:
        return max(0, self.height - self.crown_height)


class GroundPlane(BaseModel):
    model_config = ConfigDict(frozen=True)

    width: float
    length: float
    material: Literal[
        "concrete", "wet_ground", "medium_dry_ground", "very_dry_ground"
    ] = "wet_ground"

    @field_validator("width", "length")
    @classmethod
    def validate_dimensions(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Dimension must be positive")
        return v


class OutdoorScene(BaseModel):
    """Container for all outdoor scene elements (matches Room coord system)."""
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
        if v <= 0:
            raise ValueError("Scene dimension must be positive")
        return v


# ------------------------------------------------------------------
# Sionna-coding additions (TX/RX) — not in original sionna_skill but
# required for full Sionna RT round-trip. Keep simple; the scene still
# carries a Room or OutdoorScene and adds optional TX/RX lists.
# ------------------------------------------------------------------

class Transmitter(BaseModel):
    """Sionna RT transmitter."""
    model_config = ConfigDict(frozen=True)

    id: str
    name: str = ""
    position: Tuple[float, float, float]
    power_dbm: float = 20.0
    frequency_hz: float = 3.5e9
    orientation_deg: float = 0.0
    pattern: Literal["iso", "tr38901", "dipole"] = "iso"
    polarization: Literal["V", "H", "VH"] = "V"


class Receiver(BaseModel):
    """Sionna RT receiver — single point or grid."""
    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["point", "grid"] = "point"
    position: Optional[Tuple[float, float, float]] = None
    height: float = 1.5
    resolution: float = 0.5
    bounds: Optional[List[Tuple[float, float]]] = None


# ------------------------------------------------------------------
# Top-level Scene container
# ------------------------------------------------------------------

class Scene(BaseModel):
    """Unified container for indoor, outdoor, or combined scenes.

    At least one of `room` or `outdoor` must be present. When both are
    present, represents an indoor room embedded in an outdoor world (the
    paper's "combined" case).

    `transmitters` and `receivers` are optional lists for Sionna RT
    round-trip; they are agnostic to whether the scene is indoor or outdoor.
    """
    model_config = ConfigDict(frozen=True)

    room: Optional[Room] = None
    outdoor: Optional[OutdoorScene] = None
    building_rooms: Dict[str, Room] = {}
    transmitters: List[Transmitter] = []
    receivers: List[Receiver] = []
    frequency_hz: float = 3.5e9

    @model_validator(mode="after")
    def validate_has_content(self) -> "Scene":
        if self.room is None and self.outdoor is None:
            raise ValueError("Scene must contain room, outdoor, or both")
        return self

    @property
    def is_indoor_only(self) -> bool:
        return self.room is not None and self.outdoor is None

    @property
    def is_outdoor_only(self) -> bool:
        return self.outdoor is not None and self.room is None

    @property
    def is_combined(self) -> bool:
        return self.room is not None and self.outdoor is not None
