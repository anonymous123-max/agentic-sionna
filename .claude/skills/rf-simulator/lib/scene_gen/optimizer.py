"""Layout optimizer using SciPy SLSQP.

Public entry points:
- `LayoutOptimizer` — class for power users (multi-stage, fixed obstacles)
- `optimize_layout(...)` — one-shot call returning a Room
- `place_furniture(scene, request, ...)` — friendly Scene-in/Scene-out
- `place_tx(scene, ...)` — TX placement helper

The cost weights (collision=10, in_room=5, pathway=3, wall_affinity=2)
match FlairGPT-derived defaults; do not lower without re-running the
benchmark suite.
"""
from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, minimize
from shapely import Polygon as ShapelyPolygon

from .constraints import (
    collision_cost,
    in_room_cost,
    pathway_cost,
    total_cost,
    wall_affinity_cost,
)
from .models import (
    BoundingBox,
    Door,
    FurnitureItem,
    Position,
    Room,
    Scene,
    Transmitter,
    WallSegment,
)

__all__ = [
    "FurnitureSpec",
    "PlacementDecision",
    "LayoutOptimizer",
    "optimize_layout",
    "place_furniture",
    "place_tx",
]

logger = logging.getLogger(__name__)


@dataclass
class FurnitureSpec:
    """Pre-optimization furniture specification."""
    category: str
    model_id: str
    model_path: str
    dimensions: BoundingBox
    model_file: str = ""
    preferred_wall: Literal["north", "south", "east", "west"] | None = None
    orientation_offset: float = 0.0


@dataclass
class PlacementDecision:
    furniture_id: str
    category: str
    position: tuple[float, float, float]
    constraint_costs: dict[str, float]
    dominant_constraint: str


class LayoutOptimizer:
    """Constraint-based furniture layout optimizer (SciPy SLSQP)."""

    WEIGHT_WALL_AFFINITY = 2.0
    WEIGHT_COLLISION = 10.0
    WEIGHT_PATHWAY = 3.0
    WEIGHT_IN_ROOM = 5.0

    def __init__(
        self,
        room_width: float,
        room_length: float,
        doors: List[Door] | None = None,
        seed: int = 42,
        room_polygon: Optional[ShapelyPolygon] = None,
        wall_segments: Optional[List[WallSegment]] = None,
    ) -> None:
        self.room_width = room_width
        self.room_length = room_length
        self.doors = doors or []
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._fixed_obstacles: List[FurnitureItem] = []
        self.room_polygon = room_polygon
        self.wall_segments = wall_segments

    def _filter_oversized(
        self, specs: List[FurnitureSpec]
    ) -> tuple[List[FurnitureSpec], List[str]]:
        valid: List[FurnitureSpec] = []
        skipped: List[str] = []
        for spec in specs:
            w = spec.dimensions.width
            d = spec.dimensions.depth
            margin = max(w, d) / 2 + 0.1
            if margin >= self.room_width / 2 or margin >= self.room_length / 2:
                warnings.warn(
                    f"Furniture '{spec.category}' ({w:.1f}x{d:.1f}m) too large "
                    f"for room ({self.room_width:.1f}x{self.room_length:.1f}m); "
                    "skipping",
                    stacklevel=3,
                )
                skipped.append(f"{spec.category}_{spec.model_id[:8]}")
            else:
                valid.append(spec)
        return valid, skipped

    def optimize(
        self,
        furniture_specs: List[FurnitureSpec],
        max_iterations: int = 500,
    ) -> List[FurnitureItem]:
        if not furniture_specs:
            return []
        valid_specs, skipped = self._filter_oversized(furniture_specs)
        if skipped:
            logger.info("Skipped %d oversized items: %s", len(skipped), skipped)
        if not valid_specs:
            return []

        x0 = self._initialize_positions(valid_specs)
        bounds = self._get_bounds(valid_specs)
        result = minimize(
            fun=self._cost_function,
            x0=x0,
            args=(valid_specs,),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": max_iterations, "disp": False},
        )
        return self._positions_to_furniture(result.x, valid_specs)

    def optimize_with_obstacles(
        self,
        furniture_specs: List[FurnitureSpec],
        fixed_obstacles: List[FurnitureItem],
        max_iterations: int = 500,
    ) -> List[FurnitureItem]:
        if not furniture_specs:
            return []
        valid_specs, skipped = self._filter_oversized(furniture_specs)
        if skipped:
            logger.info("Skipped %d oversized items: %s", len(skipped), skipped)
        if not valid_specs:
            return []

        self._fixed_obstacles = fixed_obstacles
        try:
            x0 = self._initialize_positions(valid_specs)
            bounds = self._get_bounds(valid_specs)
            result = minimize(
                fun=self._cost_function,
                x0=x0,
                args=(valid_specs,),
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": max_iterations, "disp": False},
            )
            return self._positions_to_furniture(result.x, valid_specs)
        finally:
            self._fixed_obstacles = []

    def _initialize_positions(self, specs: List[FurnitureSpec]) -> np.ndarray:
        positions: List[float] = []
        if self.wall_segments is not None:
            return self._initialize_positions_polygon(specs)

        for i, spec in enumerate(specs):
            w = spec.dimensions.width
            d = spec.dimensions.depth
            margin = max(w, d) / 2 + 0.1
            wall = (
                spec.preferred_wall
                or ["south", "north", "east", "west"][i % 4]
            )
            if wall == "south":
                x = self.rng.uniform(margin, self.room_width - margin)
                y = margin + d / 2 + self.rng.uniform(0, 0.3)
                theta = 0.0
            elif wall == "north":
                x = self.rng.uniform(margin, self.room_width - margin)
                y = self.room_length - margin - d / 2 - self.rng.uniform(0, 0.3)
                theta = math.pi
            elif wall == "west":
                x = margin + d / 2 + self.rng.uniform(0, 0.3)
                y = self.rng.uniform(margin, self.room_length - margin)
                theta = math.pi / 2
            else:  # east
                x = self.room_width - margin - d / 2 - self.rng.uniform(0, 0.3)
                y = self.rng.uniform(margin, self.room_length - margin)
                theta = 3 * math.pi / 2
            positions.extend([x, y, theta])
        return np.array(positions)

    def _initialize_positions_polygon(
        self, specs: List[FurnitureSpec]
    ) -> np.ndarray:
        positions: List[float] = []
        segments = self.wall_segments
        assert segments is not None

        for i, spec in enumerate(specs):
            d = spec.dimensions.depth
            if spec.preferred_wall:
                matching = [
                    (j, s) for j, s in enumerate(segments)
                    if s.cardinal_direction == spec.preferred_wall
                ]
                if matching:
                    j, seg = max(matching, key=lambda x: x[1].length)
                else:
                    j = i % len(segments); seg = segments[j]
            else:
                j = i % len(segments); seg = segments[j]

            mx, my = seg.midpoint
            dx, dy = seg.direction
            nx, ny = seg.inward_normal
            jitter = self.rng.uniform(-seg.length * 0.3, seg.length * 0.3)
            inward_offset = d / 2 + 0.15

            x = mx + dx * jitter + nx * inward_offset
            y = my + dy * jitter + ny * inward_offset
            theta = math.atan2(ny, nx)
            if theta < 0:
                theta += 2 * math.pi
            positions.extend([x, y, theta])
        return np.array(positions)

    def _get_bounds(self, specs: List[FurnitureSpec]) -> Bounds:
        lb: List[float] = []
        ub: List[float] = []

        if self.room_polygon is not None:
            minx, miny, maxx, maxy = self.room_polygon.bounds
        else:
            minx, miny = 0.0, 0.0
            maxx, maxy = self.room_width, self.room_length

        for spec in specs:
            w = spec.dimensions.width
            d = spec.dimensions.depth
            margin = max(w, d) / 2
            lb.append(minx + margin); ub.append(maxx - margin)
            lb.append(miny + margin); ub.append(maxy - margin)
            lb.append(0.0); ub.append(2 * math.pi)
        return Bounds(np.array(lb), np.array(ub))

    def _cost_function(
        self, positions: np.ndarray, specs: List[FurnitureSpec]
    ) -> float:
        furniture_list = self._positions_to_furniture(positions, specs)
        total = 0.0
        for item in furniture_list:
            total += wall_affinity_cost(
                item, self.room_width, self.room_length,
                wall_segments=self.wall_segments,
            )
            total += self.WEIGHT_IN_ROOM * in_room_cost(
                item, self.room_width, self.room_length,
                room_polygon=self.room_polygon,
            )
        all_furniture = furniture_list + list(self._fixed_obstacles)
        total += collision_cost(all_furniture)
        total += pathway_cost(
            all_furniture, self.doors, self.room_width, self.room_length,
            wall_segments=self.wall_segments,
        )
        return total

    def _positions_to_furniture(
        self, positions: np.ndarray, specs: List[FurnitureSpec]
    ) -> List[FurnitureItem]:
        furniture: List[FurnitureItem] = []
        for i, spec in enumerate(specs):
            idx = i * 3
            x = float(positions[idx])
            y = float(positions[idx + 1])
            theta = float(positions[idx + 2])
            furniture.append(FurnitureItem(
                id=f"{spec.category}_{i}_{spec.model_id[:8] or 'inline'}",
                category=spec.category,
                model_id=spec.model_id,
                model_path=spec.model_path,
                model_file=spec.model_file,
                position=Position(x=x, y=y, theta=theta),
                dimensions=spec.dimensions,
                orientation_offset=spec.orientation_offset,
            ))
        return furniture

    def get_initial_cost(self, specs: List[FurnitureSpec]) -> float:
        self.rng = np.random.default_rng(self._seed)
        x0 = self._initialize_positions(specs)
        return self._cost_function(x0, specs)


def optimize_layout(
    room_width: float,
    room_length: float,
    doors: List[Door] | None = None,
    furniture_specs: List[FurnitureSpec] | None = None,
    seed: int = 42,
    max_iterations: int = 500,
    floor_polygon: Optional[List[Tuple[float, float]]] = None,
) -> Room:
    """Optimize and return a complete Room. Multi-restart on concave polygons."""
    room_polygon: Optional[ShapelyPolygon] = None
    wall_segments: Optional[List[WallSegment]] = None
    if floor_polygon is not None:
        room_polygon = ShapelyPolygon(floor_polygon)
        coords = list(room_polygon.exterior.coords)
        wall_segments = []
        for i in range(len(coords) - 1):
            a, b = coords[i], coords[i + 1]
            wall_segments.append(
                WallSegment(start=(a[0], a[1]), end=(b[0], b[1]))
            )
        is_concave = (room_polygon.convex_hull.area / room_polygon.area) > 1.05
    else:
        is_concave = False

    def _run_once(run_seed: int) -> Tuple[Room, float]:
        opt = LayoutOptimizer(
            room_width=room_width,
            room_length=room_length,
            doors=doors or [],
            seed=run_seed,
            room_polygon=room_polygon,
            wall_segments=wall_segments,
        )
        furniture = opt.optimize(
            furniture_specs=furniture_specs or [],
            max_iterations=max_iterations,
        )
        room = Room(
            width=room_width,
            length=room_length,
            floor_polygon=floor_polygon,
            doors=doors or [],
            furniture=furniture,
        )
        cost = total_cost(
            furniture, doors or [], room_width, room_length,
            room_polygon=room_polygon, wall_segments=wall_segments,
        )
        return room, cost

    if is_concave:
        best_room, best_cost = _run_once(seed)
        for restart in range(1, 3):
            room, cost = _run_once(seed + restart * 1000)
            if cost < best_cost:
                best_room, best_cost = room, cost
        return best_room

    room, _ = _run_once(seed)
    return room


# ---------------------------------------------------------------------------
# Friendly wrappers for the agent's Scene-centric API
# ---------------------------------------------------------------------------

def place_furniture(
    scene: Scene,
    request: dict[str, Any] | List[dict[str, Any]],
    *,
    seed: int = 42,
    max_iterations: int = 500,
) -> Scene:
    """Add furniture to scene.room, returning a new frozen Scene.

    `request` is either a single spec dict or a list of them. Spec dicts
    accept: type/category, count (default 1), dims=(w,d,h) or
    {width,depth,height}, model_id (optional), preferred_wall (optional).

    Example:
        scene = place_furniture(scene, {"type":"desk","count":2})
        scene = place_furniture(scene, [
            {"type":"bed", "dims":(2.0,1.6,0.5)},
            {"type":"nightstand", "preferred_wall":"east"},
        ])
    """
    if scene.room is None:
        raise ValueError("place_furniture requires scene.room to be set")
    requests = [request] if isinstance(request, dict) else list(request)

    specs: List[FurnitureSpec] = []
    for r in requests:
        cat = r.get("type") or r.get("category")
        if not cat:
            raise ValueError(f"request missing type/category: {r}")
        count = int(r.get("count", 1))
        dims = r.get("dims") or r.get("dimensions")
        if isinstance(dims, dict):
            w, d, h = dims["width"], dims["depth"], dims["height"]
        elif isinstance(dims, (list, tuple)) and len(dims) == 3:
            w, d, h = dims
        else:
            w, d, h = 1.0, 0.6, 0.75  # generic fallback
        model_id = r.get("model_id", "")
        preferred_wall = r.get("preferred_wall")
        for _ in range(count):
            specs.append(FurnitureSpec(
                category=cat,
                model_id=model_id,
                model_path=r.get("model_path", ""),
                model_file=r.get("model_file", ""),
                dimensions=BoundingBox(width=w, depth=d, height=h),
                preferred_wall=preferred_wall,
                orientation_offset=r.get("orientation_offset", 0.0),
            ))

    optimized_room = optimize_layout(
        room_width=scene.room.width,
        room_length=scene.room.length,
        doors=scene.room.doors,
        furniture_specs=(list(scene.room.furniture) and []) + specs,  # see note
        seed=seed,
        max_iterations=max_iterations,
        floor_polygon=scene.room.floor_polygon,
    )
    # Keep existing furniture as fixed obstacles when adding new items.
    # We re-optimize only the new specs but treat existing as fixed obstacles
    # for collision/pathway terms. This is the multi-turn refinement path.
    if scene.room.furniture:
        opt = LayoutOptimizer(
            room_width=scene.room.width,
            room_length=scene.room.length,
            doors=scene.room.doors,
            seed=seed,
        )
        new_items = opt.optimize_with_obstacles(
            furniture_specs=specs,
            fixed_obstacles=list(scene.room.furniture),
            max_iterations=max_iterations,
        )
        merged_room = scene.room.model_copy(update={
            "furniture": list(scene.room.furniture) + new_items,
        })
    else:
        merged_room = optimized_room.model_copy(update={
            "doors": scene.room.doors,
            "windows": scene.room.windows,
            "height": scene.room.height,
        })

    return scene.model_copy(update={"room": merged_room})


def place_tx(
    scene: Scene,
    *,
    position: Optional[Tuple[float, float, float]] = None,
    height: float = 2.5,
    power_dbm: float = 20.0,
    name: str = "AP",
    tx_id: str | None = None,
) -> Scene:
    """Add a transmitter to the scene. Defaults to room center, h=2.5m."""
    if position is None:
        if scene.room is not None:
            position = (scene.room.width / 2, scene.room.length / 2, height)
        elif scene.outdoor is not None:
            position = (scene.outdoor.width / 2, scene.outdoor.length / 2, height)
        else:
            raise ValueError("place_tx requires scene.room or scene.outdoor")
    tx = Transmitter(
        id=tx_id or f"tx_{len(scene.transmitters) + 1}",
        name=name,
        position=position,
        power_dbm=power_dbm,
        frequency_hz=scene.frequency_hz,
    )
    return scene.model_copy(update={
        "transmitters": list(scene.transmitters) + [tx],
    })
