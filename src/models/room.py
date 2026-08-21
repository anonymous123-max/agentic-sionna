"""Room and furniture data models.

Single source of truth for all scene exports. All models are frozen (immutable)
to ensure consistency across floor plan, XML, and GLTF outputs.

Coordinate system:
- Origin at SW corner of room
- X increases east, Y increases north
- theta=0 means furniture faces positive Y (north)
- width is perpendicular to facing direction
- depth (length) is in facing direction

Polygon rooms:
- floor_polygon defines non-rectangular room boundaries (CCW vertices)
- WallSegment represents a single edge of the polygon
- When floor_polygon is None, room is rectangular (backward compatible)
"""

import math
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from shapely import Polygon as ShapelyPolygon, Point, LineString


class Position(BaseModel):
    """Position with x, y coordinates and rotation theta.

    Attributes:
        x: X coordinate in meters from room origin (SW corner)
        y: Y coordinate in meters from room origin
        theta: Rotation in radians, normalized to [0, 2*pi).
               theta=0 means facing north (+Y direction).
    """
    model_config = ConfigDict(frozen=True)

    x: float
    y: float
    theta: float

    @field_validator("theta")
    @classmethod
    def normalize_theta(cls, v: float) -> float:
        """Normalize theta to [0, 2*pi) range."""
        two_pi = 2 * math.pi
        # Handle negative and large values
        normalized = v % two_pi
        # Ensure non-negative (Python % can return negative for negative inputs)
        if normalized < 0:
            normalized += two_pi
        return normalized


class WallSegment(BaseModel):
    """A single wall edge of a polygon room.

    Defined by start and end points (CCW order). Provides computed
    properties for wall direction, inward normal, and cardinal direction.

    Attributes:
        start: (x, y) start point of wall segment
        end: (x, y) end point of wall segment
    """
    model_config = ConfigDict(frozen=True)

    start: Tuple[float, float]
    end: Tuple[float, float]

    @property
    def length(self) -> float:
        """Length of this wall segment in meters."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx * dx + dy * dy)

    @property
    def midpoint(self) -> Tuple[float, float]:
        """Midpoint of the wall segment."""
        return (
            (self.start[0] + self.end[0]) / 2,
            (self.start[1] + self.end[1]) / 2,
        )

    @property
    def direction(self) -> Tuple[float, float]:
        """Unit direction vector from start to end."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        seg_len = self.length
        if seg_len < 1e-9:
            return (1.0, 0.0)
        return (dx / seg_len, dy / seg_len)

    @property
    def inward_normal(self) -> Tuple[float, float]:
        """Unit inward normal (for CCW polygon, rotate direction 90 CW)."""
        dx, dy = self.direction
        return (dy, -dx)

    @property
    def cardinal_direction(self) -> Literal["north", "south", "east", "west"]:
        """Closest cardinal direction based on inward normal.

        Useful for backward-compatible wall references.
        """
        nx, ny = self.inward_normal
        # Determine which cardinal direction the inward normal most closely faces
        if abs(ny) >= abs(nx):
            return "north" if ny > 0 else "south"
        else:
            return "east" if nx > 0 else "west"

    @property
    def facing_angle(self) -> float:
        """Angle of inward normal in radians for furniture theta matching."""
        nx, ny = self.inward_normal
        return math.atan2(ny, nx)

    @property
    def angle(self) -> float:
        """Angle of wall direction vector in radians (for XML rotation)."""
        dx, dy = self.direction
        return math.atan2(dy, dx)


class BoundingBox(BaseModel):
    """Furniture dimensions in meters.

    Attributes:
        width: Dimension perpendicular to facing direction
        depth: Dimension in facing direction (length)
        height: Vertical dimension
    """
    model_config = ConfigDict(frozen=True)

    width: float
    depth: float
    height: float

    @field_validator("width", "depth", "height")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        """Ensure all dimensions are positive."""
        if v <= 0:
            raise ValueError("Dimension must be positive")
        return v


class Door(BaseModel):
    """Door in a room wall.

    Attributes:
        wall: Which wall the door is on (cardinal direction for rectangular rooms)
        position: Distance along wall from origin corner (meters)
        width: Width of door opening (meters)
        wall_segment_index: Index into Room.wall_segments for polygon rooms.
            None for rectangular rooms (uses cardinal `wall` instead).
    """
    model_config = ConfigDict(frozen=True)

    wall: Literal["north", "south", "east", "west"]
    position: float
    width: float
    wall_segment_index: Optional[int] = None


class Window(BaseModel):
    """Window in a room wall.

    Attributes:
        wall: Which wall the window is on (cardinal direction for rectangular rooms)
        position: Distance along wall from origin corner (meters)
        width: Width of window (meters)
        height: Height of window (meters)
        sill_height: Height of window sill from floor (meters)
        wall_segment_index: Index into Room.wall_segments for polygon rooms.
            None for rectangular rooms (uses cardinal `wall` instead).
    """
    model_config = ConfigDict(frozen=True)

    wall: Literal["north", "south", "east", "west"]
    position: float
    width: float
    height: float
    sill_height: float
    wall_segment_index: Optional[int] = None


class FurnitureItem(BaseModel):
    """Single piece of furniture with fixed position.

    Attributes:
        id: Unique identifier for this furniture instance
        category: Furniture category (e.g., "bed", "sofa", "desk")
        model_id: 3D-FUTURE model UUID
        model_path: Path to the model file
        position: Position and rotation in room coordinates
        dimensions: Bounding box dimensions
        orientation_offset: Rotation offset to normalize model facing direction.
            Applied by exporters to correct inconsistent 3D-FUTURE orientations.
    """
    model_config = ConfigDict(frozen=True)

    id: str
    category: str
    model_id: str
    model_path: str
    position: Position
    dimensions: BoundingBox
    orientation_offset: float = 0.0


class Room(BaseModel):
    """Single source of truth for room layout.

    All exports (floor plan image, Sionna XML, GLTF) derive from this model.
    The Room is frozen after optimization to ensure all exports are consistent.

    Supports both rectangular and polygon rooms:
    - Rectangular: floor_polygon is None, uses width/length
    - Polygon: floor_polygon is CCW vertex list, width/length match bbox

    Attributes:
        width: Room width in meters (X dimension, east-west)
        length: Room length in meters (Y dimension, north-south)
        height: Room height in meters (Z dimension, floor to ceiling)
        floor_polygon: Optional CCW vertex list for non-rectangular rooms.
            None means rectangular. When set, width/length must match bbox.
        doors: List of doors in the room
        windows: List of windows in the room
        furniture: List of furniture items with optimized positions
    """
    model_config = ConfigDict(frozen=True)

    width: float
    length: float
    height: float = 3.0
    floor_polygon: Optional[List[Tuple[float, float]]] = None
    doors: List[Door] = []
    windows: List[Window] = []
    furniture: List[FurnitureItem] = []

    @field_validator("width", "length", "height")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        """Ensure room dimensions are positive."""
        if v <= 0:
            raise ValueError("Room dimension must be positive")
        return v

    @field_validator("floor_polygon")
    @classmethod
    def validate_polygon(
        cls, v: Optional[List[Tuple[float, float]]]
    ) -> Optional[List[Tuple[float, float]]]:
        """Validate floor polygon if provided."""
        if v is None:
            return v
        if len(v) < 3:
            raise ValueError("Floor polygon requires at least 3 vertices")
        poly = ShapelyPolygon(v)
        if not poly.is_valid:
            from shapely.validation import explain_validity
            raise ValueError(f"Invalid floor polygon: {explain_validity(poly)}")
        return v

    @model_validator(mode="after")
    def validate_polygon_bbox(self) -> "Room":
        """Ensure width/length match polygon bounding box when polygon is set."""
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
        """True if room has no custom polygon (standard rectangular room)."""
        return self.floor_polygon is None

    @property
    def shapely_polygon(self) -> ShapelyPolygon:
        """Shapely Polygon for this room (rectangle or custom polygon)."""
        if self.floor_polygon is not None:
            return ShapelyPolygon(self.floor_polygon)
        return ShapelyPolygon([
            (0, 0), (self.width, 0),
            (self.width, self.length), (0, self.length),
        ])

    @property
    def wall_segments(self) -> List[WallSegment]:
        """Derive WallSegment list from polygon edges.

        For rectangular rooms, returns 4 segments matching cardinal walls.
        For polygon rooms, returns one segment per edge.
        """
        coords = list(self.shapely_polygon.exterior.coords)
        # coords has closing point == first point; skip it
        segments = []
        for i in range(len(coords) - 1):
            segments.append(WallSegment(start=coords[i], end=coords[i + 1]))
        return segments

    @classmethod
    def from_polygon(
        cls,
        polygon: List[Tuple[float, float]],
        height: float = 3.0,
        doors: Optional[List[Door]] = None,
        windows: Optional[List[Window]] = None,
        furniture: Optional[List[FurnitureItem]] = None,
    ) -> "Room":
        """Create a Room from a polygon, auto-computing width/length from bbox.

        Args:
            polygon: CCW vertex list for floor boundary
            height: Room height in meters
            doors: Optional door list
            windows: Optional window list
            furniture: Optional furniture list

        Returns:
            Room with floor_polygon set and width/length matching bbox
        """
        poly = ShapelyPolygon(polygon)
        minx, miny, maxx, maxy = poly.bounds
        # Normalize polygon to origin
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
