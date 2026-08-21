"""PNG floor plan exporter.

Renders room layouts as PNG images using matplotlib's 3D plotting with
orthographic top-down camera for realistic floor plan visualization.
Works on macOS without requiring OpenGL/pyglet.

Supports:
- Indoor scenes (Room) - existing v1.0 functionality
- Outdoor scenes (OutdoorScene) - v1.1 outdoor generation
- Combined scenes (both) - future functionality
"""

import logging
import math
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

from src.models.room import Room, FurnitureItem
from src.models.outdoor import OutdoorScene, Building, Road, Tree
from src.models.scene import Scene

# Import from extracted module
from src.exporters.png_colors import (
    BUILDING_COLOR,
    CATEGORY_COLORS,
    DEFAULT_COLOR,
    FLOOR_COLOR,
    GROUND_COLORS,
    ROAD_COLOR,
    TREE_COLORS,
    WALL_COLOR,
    WINDOW_COLOR,
    WINDOW_EDGE_COLOR,
)

logger = logging.getLogger(__name__)


class PNGExporter:
    """Export room layouts as PNG images using matplotlib 3D rendering.

    Creates floor plan visualizations by rendering the scene from above
    with an orthographic-like view. Shows actual furniture mesh geometry
    using matplotlib's Poly3DCollection. Works on macOS without OpenGL.
    """

    def __init__(self, resolution: tuple = (1024, 1024), dpi: int = 150):
        """Initialize PNG exporter.

        Args:
            resolution: Output image resolution as (width, height)
            dpi: DPI for matplotlib figure
        """
        self.resolution = resolution
        self.dpi = dpi

    def export(self, scene_or_room: Union[Scene, Room]) -> bytes:
        """Export scene or room to PNG bytes using matplotlib 3D rendering.

        Args:
            scene_or_room: Scene or Room model to export (backward compatible)

        Returns:
            PNG image data as bytes
        """
        # Normalize input to Scene for unified handling
        if isinstance(scene_or_room, Room):
            scene = Scene(room=scene_or_room)
        else:
            scene = scene_or_room

        # Route to appropriate renderer
        if scene.is_indoor_only:
            return self._export_room(scene.room)
        elif scene.is_outdoor_only:
            return self._export_outdoor(scene.outdoor)
        else:
            return self._export_combined(scene)

    def _export_room(self, room: Room) -> bytes:
        """Export indoor room to PNG bytes.

        Args:
            room: Room model to export

        Returns:
            PNG image data as bytes
        """
        # Create figure with 3D axes
        fig = plt.figure(figsize=(self.resolution[0]/self.dpi,
                                   self.resolution[1]/self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Draw floor
        self._draw_floor(ax, room)

        # Draw walls with door gaps
        self._draw_walls(ax, room)

        # Draw windows on walls
        self._draw_windows(ax, room)

        # Draw furniture meshes
        for item in room.furniture:
            self._draw_furniture(ax, item)

        # Set up top-down orthographic-like view
        # azim=-90 with elev=90 gives: +X on right, +Y at top (standard floor plan)
        ax.view_init(elev=90, azim=-90)

        # Set axis limits with padding (positive range, no mirroring)
        padding = 0.3
        ax.set_xlim(-padding, room.width + padding)
        ax.set_ylim(-padding, room.length + padding)
        ax.set_zlim(0, max(room.width, room.length))

        # Make it look more orthographic by setting equal aspect
        ax.set_box_aspect([room.width, room.length, max(room.width, room.length)])

        # Hide axes for cleaner look
        ax.set_axis_off()
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Adjust layout
        plt.tight_layout(pad=0)

        # Save to bytes
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=self.dpi,
                    bbox_inches='tight', pad_inches=0.1,
                    facecolor='white', edgecolor='none')
        plt.close(fig)

        buffer.seek(0)
        return buffer.read()

    def _export_outdoor(self, outdoor: OutdoorScene) -> bytes:
        """Export outdoor scene to PNG bytes.

        Args:
            outdoor: OutdoorScene model to export

        Returns:
            PNG image data as bytes
        """
        # Create figure with 3D axes
        fig = plt.figure(figsize=(self.resolution[0]/self.dpi,
                                   self.resolution[1]/self.dpi), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Draw ground plane
        self._draw_ground(ax, outdoor)

        # Draw roads (under buildings)
        for road in outdoor.roads:
            self._draw_road(ax, road)

        # Draw buildings
        for building in outdoor.buildings:
            self._draw_building(ax, building)

        # Draw trees (on top)
        for tree in outdoor.trees:
            self._draw_tree(ax, tree)

        # Set up top-down orthographic-like view
        ax.view_init(elev=90, azim=-90)

        # Set axis limits with padding
        padding = 1.0
        ax.set_xlim(-padding, outdoor.width + padding)
        ax.set_ylim(-padding, outdoor.length + padding)

        # Z limit based on tallest element
        max_height = 1.0  # minimum
        if outdoor.buildings:
            max_height = max(max_height, max(b.height for b in outdoor.buildings))
        if outdoor.trees:
            max_height = max(max_height, max(t.height for t in outdoor.trees))
        ax.set_zlim(0, max_height * 1.5)

        # Make it look more orthographic by setting equal aspect
        ax.set_box_aspect([outdoor.width, outdoor.length, max_height])

        # Hide axes for cleaner look
        ax.set_axis_off()
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Adjust layout
        plt.tight_layout(pad=0)

        # Save to bytes
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=self.dpi,
                    bbox_inches='tight', pad_inches=0.1,
                    facecolor='white', edgecolor='none')
        plt.close(fig)

        buffer.seek(0)
        return buffer.read()

    def _export_combined(self, scene: Scene) -> bytes:
        """Export combined indoor/outdoor scene to PNG bytes.

        Args:
            scene: Scene with both room and outdoor

        Returns:
            PNG image data as bytes

        Note:
            Stub implementation - currently renders outdoor only.
            Future: overlay room on building footprint.
        """
        # For now, prioritize outdoor if both exist
        if scene.outdoor is not None:
            return self._export_outdoor(scene.outdoor)
        return self._export_room(scene.room)

    def _draw_ground(self, ax: Axes3D, outdoor: OutdoorScene) -> None:
        """Draw ground plane as a rectangle.

        Args:
            ax: Matplotlib 3D axes
            outdoor: OutdoorScene for dimensions and material
        """
        ground = outdoor.ground
        color = GROUND_COLORS.get(ground.material, GROUND_COLORS["wet_ground"])

        verts = [
            [(0, 0, 0), (ground.width, 0, 0),
             (ground.width, ground.length, 0), (0, ground.length, 0)]
        ]
        floor = Poly3DCollection(verts, alpha=1.0)
        floor.set_facecolor(color[:3])
        floor.set_edgecolor((0.5, 0.5, 0.5))
        floor.set_linewidth(0.5)
        ax.add_collection3d(floor)

    def _draw_building(self, ax: Axes3D, building: Building) -> None:
        """Draw building footprint as filled polygon.

        Args:
            ax: Matplotlib 3D axes
            building: Building to draw
        """
        polygon = building.as_polygon()
        coords = list(polygon.exterior.coords)

        # Draw footprint at ground level (visible from top-down)
        verts = [[(x, y, 0.01) for x, y in coords]]  # Slightly above ground
        poly = Poly3DCollection(verts, alpha=1.0)
        poly.set_facecolor(BUILDING_COLOR[:3])
        poly.set_edgecolor((0.2, 0.2, 0.2))
        poly.set_linewidth(1.0)
        ax.add_collection3d(poly)

    def _draw_road(self, ax: Axes3D, road: Road) -> None:
        """Draw road as dark gray polygon.

        Args:
            ax: Matplotlib 3D axes
            road: Road to draw
        """
        polygon = road.as_polygon()
        coords = list(polygon.exterior.coords)

        # Draw road at ground level
        verts = [[(x, y, 0.005) for x, y in coords]]  # Just above ground
        poly = Poly3DCollection(verts, alpha=1.0)
        poly.set_facecolor(ROAD_COLOR[:3])
        poly.set_edgecolor(ROAD_COLOR[:3])  # Same as fill for roads
        poly.set_linewidth(0.5)
        ax.add_collection3d(poly)

    def _draw_tree(self, ax: Axes3D, tree: Tree) -> None:
        """Draw tree as circle at position representing crown projection.

        Args:
            ax: Matplotlib 3D axes
            tree: Tree to draw
        """
        color = TREE_COLORS.get(tree.species, TREE_COLORS["deciduous"])
        x, y = tree.position

        # Draw circle for crown projection (approximated as polygon)
        num_segments = 24
        angles = np.linspace(0, 2 * np.pi, num_segments + 1)
        circle_x = x + tree.crown_radius * np.cos(angles)
        circle_y = y + tree.crown_radius * np.sin(angles)

        # Create polygon vertices
        verts = [[(cx, cy, 0.02) for cx, cy in zip(circle_x, circle_y)]]
        poly = Poly3DCollection(verts, alpha=color[3])
        poly.set_facecolor(color[:3])
        poly.set_edgecolor((0.1, 0.3, 0.1))  # Darker green edge
        poly.set_linewidth(0.5)
        ax.add_collection3d(poly)

    def _draw_floor(self, ax: Axes3D, room: Room) -> None:
        """Draw floor as a rectangle or polygon.

        Args:
            ax: Matplotlib 3D axes
            room: Room model for dimensions
        """
        if room.is_rectangular:
            coords = [(0, 0), (room.width, 0),
                       (room.width, room.length), (0, room.length)]
        else:
            coords = list(room.shapely_polygon.exterior.coords)

        verts = [[(x, y, 0) for x, y in coords]]
        floor = Poly3DCollection(verts, alpha=1.0)
        floor.set_facecolor(FLOOR_COLOR[:3])
        floor.set_edgecolor((0.7, 0.7, 0.7))
        floor.set_linewidth(0.5)
        ax.add_collection3d(floor)

    def _draw_walls(self, ax: Axes3D, room: Room) -> None:
        """Draw walls with door gaps.

        Supports both rectangular and polygon rooms.

        Args:
            ax: Matplotlib 3D axes
            room: Room model for dimensions and doors
        """
        wall_height = 0.15  # Low walls for visibility

        if not room.is_rectangular:
            # Polygon room: draw each wall segment
            for seg in room.wall_segments:
                self._draw_wall_segment(
                    ax, seg.start[0], seg.start[1],
                    seg.end[0], seg.end[1], wall_height
                )
            return

        # Rectangular room: original code with door gaps
        doors_by_wall = {"north": [], "south": [], "east": [], "west": []}
        for door in room.doors:
            doors_by_wall[door.wall].append((door.position, door.width))

        # South wall (y=0)
        for start, end in self._get_wall_segments(room.width, doors_by_wall["south"]):
            self._draw_wall_segment(ax, start, 0, end, 0, wall_height)

        # North wall (y=length)
        for start, end in self._get_wall_segments(room.width, doors_by_wall["north"]):
            self._draw_wall_segment(ax, start, room.length, end, room.length, wall_height)

        # West wall (x=0)
        for start, end in self._get_wall_segments(room.length, doors_by_wall["west"]):
            self._draw_wall_segment(ax, 0, start, 0, end, wall_height)

        # East wall (x=width)
        for start, end in self._get_wall_segments(room.length, doors_by_wall["east"]):
            self._draw_wall_segment(ax, room.width, start, room.width, end, wall_height)

    def _draw_wall_segment(self, ax: Axes3D, x1: float, y1: float,
                           x2: float, y2: float, height: float) -> None:
        """Draw a single wall segment.

        Args:
            ax: Matplotlib 3D axes
            x1, y1: Start point
            x2, y2: End point
            height: Wall height
        """
        # Wall as a vertical rectangle
        verts = [
            [(x1, y1, 0), (x2, y2, 0), (x2, y2, height), (x1, y1, height)]
        ]
        wall = Poly3DCollection(verts, alpha=1.0)
        wall.set_facecolor(WALL_COLOR[:3])
        wall.set_edgecolor((0.3, 0.3, 0.3))
        wall.set_linewidth(1)
        ax.add_collection3d(wall)

    def _get_wall_segments(self, wall_length: float, doors: list) -> list:
        """Get wall segments with door gaps.

        Args:
            wall_length: Total wall length
            doors: List of (position, width) tuples for doors

        Returns:
            List of (start, end) tuples for wall segments
        """
        if not doors:
            return [(0, wall_length)]

        sorted_doors = sorted(doors, key=lambda d: d[0])
        segments = []
        current_pos = 0

        for door_pos, door_width in sorted_doors:
            if door_pos > current_pos:
                segments.append((current_pos, door_pos))
            current_pos = door_pos + door_width

        if current_pos < wall_length:
            segments.append((current_pos, wall_length))

        return segments

    def _draw_windows(self, ax: Axes3D, room: Room) -> None:
        """Draw windows as light blue rectangles on walls.

        Args:
            ax: Matplotlib 3D axes
            room: Room model for dimensions and windows
        """
        for window in room.windows:
            self._draw_window(ax, room, window)

    def _draw_window(self, ax: Axes3D, room: Room, window) -> None:
        """Draw a single window on its wall.

        Args:
            ax: Matplotlib 3D axes
            room: Room model for dimensions
            window: Window model with wall, position, width, height, sill_height
        """
        # Window dimensions
        pos = window.position
        w = window.width
        z_bottom = window.sill_height
        z_top = window.sill_height + window.height

        # Calculate 3D rectangle vertices based on wall
        if window.wall == "south":
            # South wall (y=0): x varies, y=0 fixed
            verts = [
                [(pos, 0, z_bottom), (pos + w, 0, z_bottom),
                 (pos + w, 0, z_top), (pos, 0, z_top)]
            ]
        elif window.wall == "north":
            # North wall (y=length): x varies, y=length fixed
            verts = [
                [(pos, room.length, z_bottom), (pos + w, room.length, z_bottom),
                 (pos + w, room.length, z_top), (pos, room.length, z_top)]
            ]
        elif window.wall == "east":
            # East wall (x=width): y varies, x=width fixed
            verts = [
                [(room.width, pos, z_bottom), (room.width, pos + w, z_bottom),
                 (room.width, pos + w, z_top), (room.width, pos, z_top)]
            ]
        elif window.wall == "west":
            # West wall (x=0): y varies, x=0 fixed
            verts = [
                [(0, pos, z_bottom), (0, pos + w, z_bottom),
                 (0, pos + w, z_top), (0, pos, z_top)]
            ]
        else:
            return  # Unknown wall, skip

        # Create window polygon with light blue color
        window_poly = Poly3DCollection(verts, alpha=WINDOW_COLOR[3])
        window_poly.set_facecolor(WINDOW_COLOR[:3])
        window_poly.set_edgecolor(WINDOW_EDGE_COLOR)
        window_poly.set_linewidth(1.5)
        ax.add_collection3d(window_poly)

    def _draw_furniture(self, ax: Axes3D, item: FurnitureItem) -> None:
        """Draw furniture mesh from OBJ file.

        Args:
            ax: Matplotlib 3D axes
            item: FurnitureItem with position and model_path
        """
        obj_path = Path(item.model_path) / "raw_model.obj"
        color = CATEGORY_COLORS.get(item.category, DEFAULT_COLOR)

        try:
            mesh = trimesh.load(str(obj_path), force="mesh")

            # Handle Scene (multiple objects in OBJ)
            if isinstance(mesh, trimesh.Scene):
                meshes = list(mesh.geometry.values())
                if not meshes:
                    raise ValueError("Empty mesh")
                mesh = trimesh.util.concatenate(meshes)

            # 3D-FUTURE OBJ models use Y-up coordinates
            # Convert to matplotlib's Z-up for rendering
            # Use proper -90 degree X rotation (matches XML exporter)
            rot_x_neg90 = trimesh.transformations.rotation_matrix(
                -math.pi / 2, [1, 0, 0]
            )
            mesh.apply_transform(rot_x_neg90)

            # Center mesh horizontally, keep floor at Z=0
            centroid = mesh.centroid
            z_min = mesh.bounds[0][2]  # Floor level after swap
            mesh.apply_translation([-centroid[0], -centroid[1], -z_min])

            # Apply rotation around Z axis (up in matplotlib coords)
            effective_theta = item.position.theta + item.orientation_offset
            rotation = trimesh.transformations.rotation_matrix(
                effective_theta, [0, 0, 1]
            )
            mesh.apply_transform(rotation)

            # Translate to room position
            translation = trimesh.transformations.translation_matrix(
                [item.position.x, item.position.y, 0]
            )
            mesh.apply_transform(translation)

            # Simplify mesh for faster rendering (decimate if too many faces)
            max_faces = 5000
            if len(mesh.faces) > max_faces:
                mesh = mesh.simplify_quadric_decimation(face_count=max_faces)

            # Draw mesh faces
            vertices = mesh.vertices
            faces = mesh.faces

            # Create polygon collection
            poly3d = [[vertices[idx] for idx in face] for face in faces]
            collection = Poly3DCollection(poly3d, alpha=color[3])
            collection.set_facecolor(color[:3])
            collection.set_edgecolor((0.2, 0.2, 0.2))
            collection.set_linewidth(0.1)
            ax.add_collection3d(collection)

        except Exception as e:
            logger.warning("Failed to render OBJ for '%s': %s. Drawing placeholder box.", item.id, e)
            self._draw_box(ax, item, color)

    def _draw_box(self, ax: Axes3D, item: FurnitureItem, color: tuple) -> None:
        """Draw a simple box for furniture (fallback).

        Args:
            ax: Matplotlib 3D axes
            item: FurnitureItem with position and dimensions
            color: RGBA color tuple
        """
        w = item.dimensions.width
        d = item.dimensions.depth
        h = item.dimensions.height
        x = item.position.x
        y = item.position.y
        theta = item.position.theta + item.orientation_offset

        # Box corners (centered at origin)
        corners = np.array([
            [-w/2, -d/2, 0], [w/2, -d/2, 0], [w/2, d/2, 0], [-w/2, d/2, 0],
            [-w/2, -d/2, h], [w/2, -d/2, h], [w/2, d/2, h], [-w/2, d/2, h]
        ])

        # Rotate around Z
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
        corners = corners @ rot.T

        # Translate
        corners += [x, y, 0]

        # Define faces (6 faces of a box)
        faces = [
            [corners[0], corners[1], corners[2], corners[3]],  # bottom
            [corners[4], corners[5], corners[6], corners[7]],  # top
            [corners[0], corners[1], corners[5], corners[4]],  # front
            [corners[2], corners[3], corners[7], corners[6]],  # back
            [corners[0], corners[3], corners[7], corners[4]],  # left
            [corners[1], corners[2], corners[6], corners[5]],  # right
        ]

        collection = Poly3DCollection(faces, alpha=color[3])
        collection.set_facecolor(color[:3])
        collection.set_edgecolor((0.2, 0.2, 0.2))
        collection.set_linewidth(0.5)
        ax.add_collection3d(collection)

    def export_file(self, scene_or_room: Union[Scene, Room], path: Path) -> None:
        """Export scene or room to a PNG file.

        Args:
            scene_or_room: Scene or Room model to export
            path: Output file path
        """
        data = self.export(scene_or_room)
        Path(path).write_bytes(data)
