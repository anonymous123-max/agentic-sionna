"""Layout optimizer using SciPy SLSQP.

Constraint-based furniture layout optimization that places furniture
against walls, avoids collisions, and maintains clear pathways.

Usage:
    room = optimize_layout(
        room_width=5.0, room_length=4.0,
        doors=[Door(wall="south", position=2.0, width=0.9)],
        furniture_specs=[
            FurnitureSpec(category="bed", model_id="abc", ...),
        ],
        seed=42
    )
"""

import logging
import math
import uuid
import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, Bounds
from shapely import Polygon as ShapelyPolygon

from src.models.room import (
    BoundingBox,
    Door,
    FurnitureItem,
    Position,
    Room,
    WallSegment,
)
from src.constraints.individual import wall_affinity_cost, in_room_cost
from src.constraints.inter_object import collision_cost, pathway_cost

# Import from extracted module
from src.optimizer.specs import FurnitureSpec, PlacementDecision

# Re-export for backward compatibility
__all__ = [
    "FurnitureSpec",
    "LayoutOptimizer",
    "PlacementDecision",
    "optimize_layout",
]

logger = logging.getLogger(__name__)


class LayoutOptimizer:
    """Constraint-based furniture layout optimizer using SciPy SLSQP.

    Optimizes (x, y, theta) for each furniture item to minimize:
    - Wall distance (furniture should be against walls)
    - Collisions (no overlapping furniture)
    - Pathway blocking (doors must have clearance)
    - Out-of-bounds (furniture must be inside room)

    Attributes:
        room_width: Room width in meters (X dimension)
        room_length: Room length in meters (Y dimension)
        doors: List of doors in the room
        rng: NumPy random generator for reproducibility
    """

    # Cost function weights (from FlairGPT research)
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
        """Initialize optimizer with room parameters.

        Args:
            room_width: Room width in meters (X dimension)
            room_length: Room length in meters (Y dimension)
            doors: List of Door objects
            seed: Random seed for reproducibility
            room_polygon: Optional Shapely Polygon for non-rectangular rooms
            wall_segments: Optional wall segments for polygon rooms
        """
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
        """Filter out furniture too large for the room.

        A furniture item is too large when its margin (max(w, d) / 2 + 0.1)
        would exceed half the room width or half the room length, making it
        impossible to place without clipping walls.

        Args:
            specs: List of FurnitureSpec objects

        Returns:
            Tuple of (valid_specs, skipped_names)
        """
        valid: List[FurnitureSpec] = []
        skipped: List[str] = []

        for spec in specs:
            w = spec.dimensions.width
            d = spec.dimensions.depth
            margin = max(w, d) / 2 + 0.1

            if margin >= self.room_width / 2 or margin >= self.room_length / 2:
                warnings.warn(
                    f"Furniture '{spec.category}' ({w:.1f}x{d:.1f}m) too large "
                    f"for room ({self.room_width:.1f}x{self.room_length:.1f}m), "
                    f"skipping",
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
        """Optimize furniture positions.

        Args:
            furniture_specs: List of furniture to place
            max_iterations: SLSQP iteration limit

        Returns:
            List of FurnitureItem with optimized positions
        """
        if not furniture_specs:
            return []

        # Filter oversized furniture before optimization
        valid_specs, skipped = self._filter_oversized(furniture_specs)
        if skipped:
            logger.info("Skipped %d oversized items: %s", len(skipped), skipped)
        if not valid_specs:
            return []

        # Initialize positions near walls with seeded RNG
        x0 = self._initialize_positions(valid_specs)

        # Get bounds for optimizer
        bounds = self._get_bounds(valid_specs)

        # Run SLSQP optimization
        result = minimize(
            fun=self._cost_function,
            x0=x0,
            args=(valid_specs,),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": max_iterations, "disp": False},
        )

        # Convert optimized positions to FurnitureItem list
        return self._positions_to_furniture(result.x, valid_specs)

    def optimize_with_obstacles(
        self,
        furniture_specs: List[FurnitureSpec],
        fixed_obstacles: List[FurnitureItem],
        max_iterations: int = 500,
    ) -> List[FurnitureItem]:
        """Optimize with some furniture fixed as obstacles.

        Used for multi-turn refinement where some furniture positions are
        locked and should not move. Locked furniture acts as collision
        obstacles for the items being optimized.

        Args:
            furniture_specs: Furniture to optimize (unlocked)
            fixed_obstacles: Furniture that cannot move (locked)
            max_iterations: SLSQP iteration limit

        Returns:
            List of newly optimized FurnitureItem (does NOT include fixed_obstacles)
        """
        if not furniture_specs:
            return []

        # Filter oversized furniture before optimization
        valid_specs, skipped = self._filter_oversized(furniture_specs)
        if skipped:
            logger.info("Skipped %d oversized items: %s", len(skipped), skipped)
        if not valid_specs:
            return []

        # Store fixed obstacles for cost function access
        self._fixed_obstacles = fixed_obstacles

        try:
            # Initialize positions near walls with seeded RNG
            x0 = self._initialize_positions(valid_specs)

            # Get bounds for optimizer
            bounds = self._get_bounds(valid_specs)

            # Run SLSQP optimization
            result = minimize(
                fun=self._cost_function,
                x0=x0,
                args=(valid_specs,),
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": max_iterations, "disp": False},
            )

            # Convert optimized positions to FurnitureItem list
            return self._positions_to_furniture(result.x, valid_specs)
        finally:
            # Clear fixed obstacles after optimization completes
            self._fixed_obstacles = []

    def _initialize_positions(
        self, specs: List[FurnitureSpec]
    ) -> np.ndarray:
        """Generate initial positions using seeded RNG.

        Places furniture near walls with random jitter to help convergence.
        For polygon rooms, places along wall segments instead of cardinal walls.

        Args:
            specs: List of FurnitureSpec objects

        Returns:
            Flat array: [x0, y0, theta0, x1, y1, theta1, ...]
        """
        positions = []

        if self.wall_segments is not None:
            return self._initialize_positions_polygon(specs)

        for i, spec in enumerate(specs):
            w = spec.dimensions.width
            d = spec.dimensions.depth
            margin = max(w, d) / 2 + 0.1  # Small buffer from walls

            # Choose a wall to start near
            if spec.preferred_wall:
                wall = spec.preferred_wall
            else:
                # Distribute furniture around different walls
                walls = ["south", "north", "east", "west"]
                wall = walls[i % 4]

            # Generate position near the chosen wall
            if wall == "south":
                x = self.rng.uniform(margin, self.room_width - margin)
                y = margin + d / 2 + self.rng.uniform(0, 0.3)
                theta = 0  # Face north
            elif wall == "north":
                x = self.rng.uniform(margin, self.room_width - margin)
                y = self.room_length - margin - d / 2 - self.rng.uniform(0, 0.3)
                theta = math.pi  # Face south
            elif wall == "west":
                x = margin + d / 2 + self.rng.uniform(0, 0.3)
                y = self.rng.uniform(margin, self.room_length - margin)
                theta = math.pi / 2  # Face east
            else:  # east
                x = self.room_width - margin - d / 2 - self.rng.uniform(0, 0.3)
                y = self.rng.uniform(margin, self.room_length - margin)
                theta = 3 * math.pi / 2  # Face west

            positions.extend([x, y, theta])

        return np.array(positions)

    def _initialize_positions_polygon(
        self, specs: List[FurnitureSpec]
    ) -> np.ndarray:
        """Generate initial positions for polygon rooms.

        Places furniture along wall segments, distributing across segments.

        Args:
            specs: List of FurnitureSpec objects

        Returns:
            Flat array: [x0, y0, theta0, x1, y1, theta1, ...]
        """
        positions = []
        segments = self.wall_segments

        for i, spec in enumerate(specs):
            d = spec.dimensions.depth

            # Find best segment: prefer matching cardinal, then longest
            if spec.preferred_wall:
                matching = [
                    (j, s) for j, s in enumerate(segments)
                    if s.cardinal_direction == spec.preferred_wall
                ]
                if matching:
                    # Pick longest matching segment
                    j, seg = max(matching, key=lambda x: x[1].length)
                else:
                    # Distribute across all segments
                    j = i % len(segments)
                    seg = segments[j]
            else:
                j = i % len(segments)
                seg = segments[j]

            # Place at segment midpoint with jitter along segment
            mx, my = seg.midpoint
            dx, dy = seg.direction
            nx, ny = seg.inward_normal
            jitter = self.rng.uniform(-seg.length * 0.3, seg.length * 0.3)
            inward_offset = d / 2 + 0.15

            x = mx + dx * jitter + nx * inward_offset
            y = my + dy * jitter + ny * inward_offset

            # Set theta so furniture back faces the wall
            # Furniture faces the inward normal direction
            theta = math.atan2(ny, nx)
            # Normalize to [0, 2pi)
            if theta < 0:
                theta += 2 * math.pi

            positions.extend([x, y, theta])

        return np.array(positions)

    def _get_bounds(self, specs: List[FurnitureSpec]) -> Bounds:
        """Compute position bounds for each furniture item.

        For polygon rooms, uses polygon bounding box.
        For rectangular rooms, uses room dimensions.

        Args:
            specs: List of FurnitureSpec objects

        Returns:
            Bounds object for scipy.optimize.minimize
        """
        lb = []  # Lower bounds
        ub = []  # Upper bounds

        # Use polygon bounds if available, else room dimensions
        if self.room_polygon is not None:
            minx, miny, maxx, maxy = self.room_polygon.bounds
        else:
            minx, miny = 0.0, 0.0
            maxx, maxy = self.room_width, self.room_length

        for spec in specs:
            w = spec.dimensions.width
            d = spec.dimensions.depth
            margin = max(w, d) / 2

            lb.append(minx + margin)
            ub.append(maxx - margin)

            lb.append(miny + margin)
            ub.append(maxy - margin)

            lb.append(0.0)
            ub.append(2 * math.pi)

        return Bounds(lb, ub)

    def _cost_function(
        self,
        positions: np.ndarray,
        specs: List[FurnitureSpec],
    ) -> float:
        """Total cost combining all constraints.

        Weights (from FlairGPT research):
        - wall_affinity: 2.0 (furniture should be against walls)
        - collision: 10.0 (no overlaps - highest priority)
        - pathway: 3.0 (doors need clearance)
        - in_room: 5.0 (must be inside room)

        Args:
            positions: Flat array [x0, y0, theta0, x1, y1, theta1, ...]
            specs: List of FurnitureSpec objects

        Returns:
            Total weighted cost
        """
        # Create temporary FurnitureItem objects for constraint evaluation
        furniture_list = self._positions_to_furniture(positions, specs)

        total = 0.0

        # Individual constraints for each piece
        for item in furniture_list:
            # Wall affinity cost (already has 2x weight inside)
            total += wall_affinity_cost(
                item, self.room_width, self.room_length,
                wall_segments=self.wall_segments,
            )
            # In-room cost with weight
            total += self.WEIGHT_IN_ROOM * in_room_cost(
                item, self.room_width, self.room_length,
                room_polygon=self.room_polygon,
            )

        # Combine optimized furniture with fixed obstacles for collision/pathway
        # Fixed obstacles are included in collision detection but not optimized
        all_furniture = furniture_list + list(self._fixed_obstacles)

        # Inter-object constraints (already have weights inside)
        total += collision_cost(all_furniture)
        total += pathway_cost(
            all_furniture, self.doors, self.room_width, self.room_length,
            wall_segments=self.wall_segments,
        )

        return total

    def _positions_to_furniture(
        self,
        positions: np.ndarray,
        specs: List[FurnitureSpec],
    ) -> List[FurnitureItem]:
        """Convert flat position array to FurnitureItem list.

        Args:
            positions: Flat array [x0, y0, theta0, x1, y1, theta1, ...]
            specs: List of FurnitureSpec objects

        Returns:
            List of FurnitureItem objects
        """
        furniture = []

        for i, spec in enumerate(specs):
            idx = i * 3
            x = float(positions[idx])
            y = float(positions[idx + 1])
            theta = float(positions[idx + 2])

            item = FurnitureItem(
                id=f"{spec.category}_{i}_{spec.model_id[:8]}",
                category=spec.category,
                model_id=spec.model_id,
                model_path=spec.model_path,
                position=Position(x=x, y=y, theta=theta),
                dimensions=spec.dimensions,
                orientation_offset=spec.orientation_offset,
            )
            furniture.append(item)

        return furniture

    def get_initial_cost(self, specs: List[FurnitureSpec]) -> float:
        """Calculate cost at initial positions (for testing).

        Args:
            specs: List of FurnitureSpec objects

        Returns:
            Cost at initial positions
        """
        # Reset RNG to get same initial positions
        self.rng = np.random.default_rng(self._seed)
        x0 = self._initialize_positions(specs)
        return self._cost_function(x0, specs)

    def get_placement_decisions(
        self, furniture: List[FurnitureItem]
    ) -> List[PlacementDecision]:
        """Calculate individual constraint costs for explanation.

        Analyzes each furniture item's placement and computes the cost
        contribution from each constraint. This data can be used to
        generate human-readable placement rationales.

        Args:
            furniture: List of FurnitureItem objects with optimized positions

        Returns:
            List of PlacementDecision objects with per-item constraint costs
        """
        decisions = []

        for item in furniture:
            # Calculate individual constraint costs
            wall_cost = wall_affinity_cost(item, self.room_width, self.room_length)
            room_cost = self.WEIGHT_IN_ROOM * in_room_cost(
                item, self.room_width, self.room_length
            )

            costs = {
                "wall_affinity": wall_cost,
                "in_room": room_cost,
            }

            # Find dominant (highest) non-zero cost
            if any(costs.values()):
                dominant = max(costs.items(), key=lambda x: x[1])[0]
            else:
                dominant = "none"

            decisions.append(
                PlacementDecision(
                    furniture_id=item.id,
                    category=item.category,
                    position=(item.position.x, item.position.y, item.position.theta),
                    constraint_costs=costs,
                    dominant_constraint=dominant,
                )
            )

        return decisions


def optimize_layout(
    room_width: float,
    room_length: float,
    doors: List[Door] | None = None,
    furniture_specs: List[FurnitureSpec] | None = None,
    seed: int = 42,
    max_iterations: int = 500,
    floor_polygon: Optional[List[Tuple[float, float]]] = None,
) -> Room:
    """Convenience function: optimize and return complete Room.

    Usage:
        room = optimize_layout(
            room_width=5.0, room_length=4.0,
            doors=[Door(wall="south", position=2.0, width=0.9)],
            furniture_specs=[
                FurnitureSpec(category="bed", model_id="abc", ...),
                FurnitureSpec(category="desk", model_id="def", ...),
            ],
            seed=42
        )

    For polygon rooms:
        room = optimize_layout(
            room_width=4.0, room_length=5.0,
            floor_polygon=[(0,0),(4,0),(4,3),(2,3),(2,5),(0,5)],
            ...
        )

    Args:
        room_width: Room width in meters (X dimension)
        room_length: Room length in meters (Y dimension)
        doors: List of Door objects
        furniture_specs: List of FurnitureSpec objects
        seed: Random seed for reproducibility
        max_iterations: SLSQP iteration limit
        floor_polygon: Optional CCW vertex list for non-rectangular rooms

    Returns:
        Complete Room object with optimized furniture positions
    """
    # Set up polygon params if polygon room
    room_polygon = None
    wall_segments = None
    if floor_polygon is not None:
        room_polygon = ShapelyPolygon(floor_polygon)
        # Build wall segments from polygon edges
        coords = list(room_polygon.exterior.coords)
        wall_segments = []
        for i in range(len(coords) - 1):
            wall_segments.append(WallSegment(start=coords[i], end=coords[i + 1]))

        # Detect concavity for multi-restart
        convexity_ratio = room_polygon.convex_hull.area / room_polygon.area
        is_concave = convexity_ratio > 1.05
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
        # Compute final cost for comparison
        from src.constraints.inter_object import total_cost
        cost = total_cost(
            furniture, doors or [], room_width, room_length,
            room_polygon=room_polygon, wall_segments=wall_segments,
        )
        return room, cost

    if is_concave:
        # Multiple restarts for concave rooms
        best_room, best_cost = _run_once(seed)
        for restart in range(1, 3):
            room, cost = _run_once(seed + restart * 1000)
            if cost < best_cost:
                best_room, best_cost = room, cost
        return best_room

    room, _ = _run_once(seed)
    return room
