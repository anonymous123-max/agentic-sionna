"""GLTF 3D scene exporter.

Exports Room and Scene models to GLTF/GLB format for 3D visualization.
Uses trimesh for scene composition and mesh loading.

Coordinate system mapping:
- Room/Scene: X east, Y north, Z up (right-handed, Z-up)
- GLTF: X right, Y up, Z forward (right-handed, Y-up)
- Transform: Room X -> GLTF X, Room Y -> GLTF Z, Room Z -> GLTF Y
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import trimesh
import trimesh.creation
import trimesh.transformations
from trimesh.visual.material import PBRMaterial

from src.models.outdoor import Building, GroundPlane, OutdoorScene, Road, Tree
from src.models.room import Door, Room, Window
from src.models.scene import Scene

# Import from extracted module
from src.exporters.gltf_materials import (
    FURNITURE_CATEGORY_MATERIALS,
    MATERIAL_COLORS,
    _consolidate_materials,
    _material_cache,
    apply_material_color,
    clear_material_cache,
    get_color_for_category,
    get_color_for_material,
    get_pbr_material,
)

logger = logging.getLogger(__name__)


class GLTFExporter:
    """Export room layouts as GLTF/GLB 3D scenes.

    Creates 3D scenes with:
    - Optional room geometry (floor and walls as boxes)
    - Furniture items loaded from OBJ files
    - Correct coordinate system transformation
    """

    def __init__(self, include_room_geometry: bool = True):
        """Initialize GLTF exporter.

        Args:
            include_room_geometry: Whether to include floor and walls (default True)
        """
        self.include_room_geometry = include_room_geometry

    def export(self, scene_or_room: Union[Scene, Room]) -> bytes:
        """Export scene or room to GLB (binary GLTF) bytes.

        Args:
            scene_or_room: Scene or Room model to export (Room for backward compatibility)

        Returns:
            GLB binary data as bytes
        """
        # Clear material cache for clean material names on each export
        clear_material_cache()

        # Normalize input to Scene for unified handling
        if isinstance(scene_or_room, Room):
            scene = Scene(room=scene_or_room)
        else:
            scene = scene_or_room

        if scene.is_indoor_only:
            return self._export_room(scene.room)
        elif scene.is_outdoor_only:
            return self._export_outdoor(scene.outdoor)
        else:
            return self._export_combined(scene)

    def _export_room(self, room: Room) -> bytes:
        """Export indoor room to GLB.

        Args:
            room: Room model to export

        Returns:
            GLB binary data as bytes
        """
        tm_scene = trimesh.Scene()

        if self.include_room_geometry:
            self._add_room_geometry(tm_scene, room)

        for item in room.furniture:
            self._add_furniture(tm_scene, item)

        # Handle empty scene - trimesh requires at least one geometry
        if len(tm_scene.geometry) == 0:
            # Add a tiny invisible marker at origin
            marker = trimesh.creation.box(extents=[0.001, 0.001, 0.001])
            tm_scene.add_geometry(marker, node_name='_origin_marker')

        # Consolidate materials to avoid duplicates like 'wood.001'
        _consolidate_materials(tm_scene)

        # Export as GLB (binary GLTF)
        return tm_scene.export(file_type='glb')

    def _export_outdoor(self, outdoor: OutdoorScene) -> bytes:
        """Export outdoor scene to GLB.

        Args:
            outdoor: OutdoorScene model to export

        Returns:
            GLB binary data as bytes
        """
        tm_scene = trimesh.Scene()

        # Add ground plane
        self._add_ground_mesh(tm_scene, outdoor.ground, outdoor.width, outdoor.length)

        # Add buildings
        for building in outdoor.buildings:
            self._add_building_mesh(tm_scene, building)

        # Add roads
        for road in outdoor.roads:
            self._add_road_mesh(tm_scene, road)

        # Add trees
        for tree in outdoor.trees:
            self._add_tree_mesh(tm_scene, tree)

        # Handle empty scene - trimesh requires at least one geometry
        if len(tm_scene.geometry) == 0:
            marker = trimesh.creation.box(extents=[0.001, 0.001, 0.001])
            tm_scene.add_geometry(marker, node_name='_origin_marker')

        # Consolidate materials to avoid duplicates like 'wood.001'
        _consolidate_materials(tm_scene)

        return tm_scene.export(file_type='glb')

    def _export_combined(self, scene: Scene) -> bytes:
        """Export combined indoor/outdoor scene to GLB.

        For now, routes to outdoor export. Combined scenes will be
        implemented in a future phase.

        Args:
            scene: Scene with both room and outdoor

        Returns:
            GLB binary data as bytes
        """
        # TODO: Implement combined scene export with room placed in outdoor
        # For now, just export the outdoor portion
        return self._export_outdoor(scene.outdoor)

    def _zup_to_yup(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Convert Z-up mesh to Y-up for GLTF.

        Performs coordinate axis swap: (x, y, z) -> (x, z, y)
        - X stays X (east)
        - Z becomes Y (up in GLTF)
        - Y becomes Z (north/forward in GLTF)

        Args:
            mesh: Mesh in Z-up coordinate system

        Returns:
            Mesh transformed to Y-up (GLTF standard)
        """
        # Axis swap matrix: x->x, z->y, y->z
        # This is NOT a rotation but a coordinate permutation
        swap_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],  # new Y = old Z
            [0, 1, 0, 0],  # new Z = old Y
            [0, 0, 0, 1]
        ], dtype=float)
        mesh.apply_transform(swap_matrix)
        return mesh

    def _add_ground_mesh(
        self, scene: trimesh.Scene, ground: GroundPlane, width: float, length: float
    ) -> None:
        """Add ground plane mesh to scene.

        GLTF is Y-up: ground is XZ plane, thin in Y.

        Args:
            scene: trimesh Scene to add geometry to
            ground: GroundPlane model
            width: Scene width (X dimension)
            length: Scene length (Z dimension in GLTF)
        """
        ground_thickness = 0.01
        ground_mesh = trimesh.creation.box(
            extents=[width, ground_thickness, length]
        )
        # Position at center with top surface at Y=0
        transform = trimesh.transformations.translation_matrix(
            [width / 2, -ground_thickness / 2, length / 2]
        )
        ground_mesh.apply_transform(transform)
        # Apply PBR material based on ground material (ITU name)
        apply_material_color(ground_mesh, ground.material)
        scene.add_geometry(ground_mesh, node_name='ground')

    def _add_building_mesh(self, scene: trimesh.Scene, building: Building) -> None:
        """Add building mesh to scene.

        Creates extruded polygon from building footprint.

        Args:
            scene: trimesh Scene to add geometry to
            building: Building model with footprint and height
        """
        polygon = building.as_polygon()
        # Extrude polygon in Z direction (Z-up coords)
        mesh = trimesh.creation.extrude_polygon(polygon, height=building.height)
        # Convert Z-up to Y-up for GLTF
        self._zup_to_yup(mesh)
        # Apply PBR material based on building material (ITU name)
        apply_material_color(mesh, building.material)
        scene.add_geometry(mesh, node_name=building.id)

    def _add_road_mesh(self, scene: trimesh.Scene, road: Road) -> None:
        """Add road mesh to scene.

        Creates thin extruded polygon from road footprint.

        Args:
            scene: trimesh Scene to add geometry to
            road: Road model with centerline and width
        """
        polygon = road.as_polygon()
        road_thickness = 0.02
        # Extrude polygon (Z-up coords)
        mesh = trimesh.creation.extrude_polygon(polygon, height=road_thickness)
        # Convert Z-up to Y-up for GLTF
        self._zup_to_yup(mesh)
        # Translate slightly up to avoid z-fighting with ground
        translate = trimesh.transformations.translation_matrix([0, 0.01, 0])
        mesh.apply_transform(translate)
        # Apply PBR material based on road material (ITU name)
        apply_material_color(mesh, road.material)
        scene.add_geometry(mesh, node_name=road.id)

    def _add_tree_mesh(self, scene: trimesh.Scene, tree: Tree) -> None:
        """Add tree mesh to scene.

        Creates low-poly tree with cylinder trunk and crown as separate meshes.
        Crown shape depends on species:
        - deciduous: icosphere
        - conifer: cone

        Target: ~100 polygons per tree for ray tracer performance.

        Args:
            scene: trimesh Scene to add geometry to
            tree: Tree model with position, dimensions, species
        """
        x, y = tree.position

        # Create trunk cylinder (low poly - sections=8 -> ~32 faces)
        trunk = trimesh.creation.cylinder(
            radius=tree.trunk_radius,
            height=tree.trunk_height,
            sections=8
        )
        # Position trunk: base at Z=0, then convert to Y-up and translate
        trunk.apply_translation([0, 0, tree.trunk_height / 2])
        self._zup_to_yup(trunk)
        trunk.apply_transform(trimesh.transformations.translation_matrix([x, 0, y]))
        # Apply PBR material - trees use 'wood' ITU material
        apply_material_color(trunk, 'wood')
        scene.add_geometry(trunk, node_name=f'{tree.id}_trunk')

        # Create crown based on species
        if tree.species == "conifer":
            # Cone crown (~16 faces)
            crown = trimesh.creation.cone(
                radius=tree.crown_radius,
                height=tree.crown_height,
                sections=8
            )
            crown_z = tree.trunk_height + tree.crown_height / 2
        else:
            # Deciduous: icosphere crown (~20 faces at subdivisions=1)
            crown = trimesh.creation.icosphere(
                radius=tree.crown_radius,
                subdivisions=1
            )
            crown_z = tree.trunk_height

        # Position crown at top of trunk, convert to Y-up, translate
        crown.apply_translation([0, 0, crown_z])
        self._zup_to_yup(crown)
        crown.apply_transform(trimesh.transformations.translation_matrix([x, 0, y]))
        # Apply PBR material - trees use 'wood' ITU material
        apply_material_color(crown, 'wood')
        scene.add_geometry(crown, node_name=f'{tree.id}_crown')

    def export_file(self, scene_or_room: Union[Scene, Room], path: Path) -> None:
        """Export scene or room to a GLB file.

        Args:
            scene_or_room: Scene or Room model to export
            path: Output file path
        """
        data = self.export(scene_or_room)
        path.write_bytes(data)

    def _add_room_geometry(self, scene: trimesh.Scene, room: Room) -> None:
        """Add floor and wall geometry to scene.

        Supports both rectangular and polygon rooms.
        Walls are split into segments to create door openings.

        Args:
            scene: trimesh Scene to add geometry to
            room: Room model for dimensions
        """
        floor_thickness = 0.02

        if not room.is_rectangular:
            # Polygon floor: extrude polygon shape
            floor_mesh = trimesh.creation.extrude_polygon(
                room.shapely_polygon, height=floor_thickness
            )
            self._zup_to_yup(floor_mesh)
            floor_mesh.apply_transform(
                trimesh.transformations.translation_matrix([0, -floor_thickness, 0])
            )
            apply_material_color(floor_mesh, 'wood')
            scene.add_geometry(floor_mesh, node_name='floor')

            # Polygon walls: extrude each wall segment as a thin wall
            wall_thickness = 0.1
            for i, seg in enumerate(room.wall_segments):
                line = trimesh.path.entities.Line([0, 1])
                path = trimesh.path.Path2D(
                    entities=[line],
                    vertices=np.array([seg.start, seg.end]),
                )
                wall_poly = path.buffer_paths(wall_thickness / 2)
                if wall_poly and len(wall_poly) > 0:
                    try:
                        wall_mesh = trimesh.creation.extrude_polygon(
                            wall_poly[0], height=room.height
                        )
                        self._zup_to_yup(wall_mesh)
                        apply_material_color(wall_mesh, 'plasterboard')
                        scene.add_geometry(wall_mesh, node_name=f'wall_seg_{i}')
                    except Exception:
                        pass  # Skip degenerate wall segments
            return

        # Rectangular floor (original code)
        floor = trimesh.creation.box(
            extents=[room.width, floor_thickness, room.length]
        )
        floor_transform = trimesh.transformations.translation_matrix(
            [room.width / 2, -floor_thickness / 2, room.length / 2]
        )
        floor.apply_transform(floor_transform)
        apply_material_color(floor, 'wood')
        scene.add_geometry(floor, node_name='floor')

        # Wall parameters
        wall_thickness = 0.1
        door_height = 2.1  # Standard door height

        # Group doors and windows by wall
        doors_by_wall = {'north': [], 'south': [], 'east': [], 'west': []}
        for door in room.doors:
            doors_by_wall[door.wall].append(door)

        windows_by_wall = {'north': [], 'south': [], 'east': [], 'west': []}
        for window in room.windows:
            windows_by_wall[window.wall].append(window)

        # Add walls with door and window cutouts
        self._add_wall_with_openings(
            scene, 'south', room.width, room.height, wall_thickness,
            doors_by_wall['south'], windows_by_wall['south'], door_height,
            # South wall: GLTF z = -thickness/2, wall runs along X
            lambda cx, w: [cx, room.height / 2, -wall_thickness / 2],
            is_x_wall=True
        )

        self._add_wall_with_openings(
            scene, 'north', room.width, room.height, wall_thickness,
            doors_by_wall['north'], windows_by_wall['north'], door_height,
            # North wall: GLTF z = length + thickness/2
            lambda cx, w: [cx, room.height / 2, room.length + wall_thickness / 2],
            is_x_wall=True
        )

        self._add_wall_with_openings(
            scene, 'west', room.length, room.height, wall_thickness,
            doors_by_wall['west'], windows_by_wall['west'], door_height,
            # West wall: GLTF x = -thickness/2, wall runs along Z
            lambda cz, w: [-wall_thickness / 2, room.height / 2, cz],
            is_x_wall=False
        )

        self._add_wall_with_openings(
            scene, 'east', room.length, room.height, wall_thickness,
            doors_by_wall['east'], windows_by_wall['east'], door_height,
            # East wall: GLTF x = width + thickness/2
            lambda cz, w: [room.width + wall_thickness / 2, room.height / 2, cz],
            is_x_wall=False
        )

    def _add_wall_with_openings(
        self,
        scene: trimesh.Scene,
        wall_name: str,
        wall_length: float,
        wall_height: float,
        wall_thickness: float,
        doors: List[Door],
        windows: List[Window],
        door_height: float,
        position_fn,
        is_x_wall: bool
    ) -> None:
        """Add a wall with door and window openings as separate segments.

        Creates wall geometry by segmenting around openings:
        - Doors: full-height openings from floor (just header above)
        - Windows: openings with sill below and header above

        Args:
            scene: trimesh Scene to add geometry to
            wall_name: Name prefix for wall segments
            wall_length: Total length of the wall
            wall_height: Height of the wall
            wall_thickness: Thickness of the wall
            doors: List of Door objects on this wall
            windows: List of Window objects on this wall
            door_height: Height of door openings
            position_fn: Function(center, width) -> [x, y, z] for segment position
            is_x_wall: True if wall runs along X axis (south/north), False for Z axis
        """
        if not doors and not windows:
            # No openings - create single wall segment
            self._add_wall_segment(
                scene, f'wall_{wall_name}', 0.0, wall_length,
                0.0, wall_height, wall_thickness, position_fn, is_x_wall
            )
            return

        # Create unified list of openings with type marker
        # Each opening: (position, width, bottom_z, top_z, type, index)
        openings: List[Tuple[float, float, float, float, str, int]] = []

        for i, door in enumerate(doors):
            openings.append((door.position, door.width, 0.0, door_height, 'door', i))

        for i, window in enumerate(windows):
            window_top = window.sill_height + window.height
            openings.append((window.position, window.width, window.sill_height, window_top, 'window', i))

        # Sort openings by position along wall
        openings.sort(key=lambda o: o[0])

        # Create wall segments
        segment_idx = 0
        current_pos = 0.0

        for position, width, bottom_z, top_z, opening_type, opening_idx in openings:
            opening_start = position
            opening_end = position + width

            # Full-height wall segment before this opening (if any)
            if opening_start > current_pos:
                self._add_wall_segment(
                    scene, f'wall_{wall_name}_{segment_idx}',
                    current_pos, opening_start,
                    0.0, wall_height, wall_thickness, position_fn, is_x_wall
                )
                segment_idx += 1

            if opening_type == 'door':
                # Door: just header above
                if wall_height > top_z:
                    self._add_wall_segment(
                        scene, f'wall_{wall_name}_header_{segment_idx}',
                        opening_start, opening_end,
                        top_z, wall_height, wall_thickness, position_fn, is_x_wall
                    )
            else:
                # Window: sill below and header above
                if bottom_z > 0:
                    self._add_wall_segment(
                        scene, f'wall_{wall_name}_sill_{segment_idx}',
                        opening_start, opening_end,
                        0.0, bottom_z, wall_thickness, position_fn, is_x_wall
                    )
                if top_z < wall_height:
                    self._add_wall_segment(
                        scene, f'wall_{wall_name}_header_{segment_idx}',
                        opening_start, opening_end,
                        top_z, wall_height, wall_thickness, position_fn, is_x_wall
                    )

            current_pos = opening_end

        # Full-height wall segment after last opening (if any)
        if current_pos < wall_length:
            self._add_wall_segment(
                scene, f'wall_{wall_name}_{segment_idx}',
                current_pos, wall_length,
                0.0, wall_height, wall_thickness, position_fn, is_x_wall
            )

    def _add_wall_segment(
        self,
        scene: trimesh.Scene,
        name: str,
        start_pos: float,
        end_pos: float,
        bottom_z: float,
        top_z: float,
        thickness: float,
        position_fn,
        is_x_wall: bool
    ) -> None:
        """Add a single wall box segment.

        Args:
            scene: trimesh Scene to add geometry to
            name: Unique name for this segment
            start_pos: Start position along wall
            end_pos: End position along wall
            bottom_z: Bottom of segment (in room Z coords, becomes GLTF Y)
            top_z: Top of segment
            thickness: Wall thickness
            position_fn: Function(center, width) -> [x, y, z] for segment position
            is_x_wall: True if wall runs along X axis
        """
        seg_length = end_pos - start_pos
        seg_height = top_z - bottom_z

        if seg_length <= 0 or seg_height <= 0:
            return  # Skip degenerate segments

        seg_center_pos = start_pos + seg_length / 2
        seg_center_z = bottom_z + seg_height / 2

        if is_x_wall:
            extents = [seg_length, seg_height, thickness]
        else:
            extents = [thickness, seg_height, seg_length]

        wall_seg = trimesh.creation.box(extents=extents)
        pos = position_fn(seg_center_pos, seg_length)
        pos[1] = seg_center_z  # Adjust Y (GLTF up) for vertical position
        transform = trimesh.transformations.translation_matrix(pos)
        wall_seg.apply_transform(transform)
        # Apply PBR material - walls use 'plasterboard' ITU material
        apply_material_color(wall_seg, 'plasterboard')
        scene.add_geometry(wall_seg, node_name=name)

    def _add_furniture(self, scene: trimesh.Scene, item) -> None:
        """Add furniture item to scene.

        Loads the OBJ model and applies position/rotation transform.
        If the OBJ file cannot be loaded, creates a placeholder box using
        the furniture's dimensions.

        The transform is stored in the scene graph node (not baked into mesh)
        to enable position extraction by validator.

        Args:
            scene: trimesh Scene to add geometry to
            item: FurnitureItem with position and model_path
        """
        obj_path = f"{item.model_path}/raw_model.obj"
        mesh = None

        try:
            # Load OBJ file - may return Trimesh or Scene
            loaded = trimesh.load(obj_path, force='mesh')

            # Handle case where load returns a Scene (multiple objects in OBJ)
            if isinstance(loaded, trimesh.Scene):
                # Concatenate all meshes into one
                meshes = list(loaded.geometry.values())
                if meshes:
                    mesh = trimesh.util.concatenate(meshes)
            else:
                mesh = loaded

            # 3D-FUTURE OBJ models use Y-up coordinates (same as GLTF)
            # Center mesh at origin with floor at Y=0 (local transform only)
            centroid = mesh.centroid
            y_min = mesh.bounds[0][1]
            mesh.apply_translation([-centroid[0], -y_min, -centroid[2]])

        except Exception as e:
            logger.warning("Failed to load OBJ for '%s': %s. Using placeholder box.", item.id, e)

        if mesh is None:
            # Create placeholder box using furniture dimensions
            # In GLTF coords: width -> X, height -> Y, depth -> Z
            mesh = trimesh.creation.box(
                extents=[
                    item.dimensions.width,
                    item.dimensions.height,
                    item.dimensions.depth
                ]
            )

        # Apply ITU material to all furniture (both OBJ models and placeholders)
        # OBJ materials like "Material_6.001" don't work with Sionna or display in Blender
        # Use category-based ITU material for proper RF simulation and visualization
        category = getattr(item, 'category', '') or ''
        cat_lower = category.lower()

        # Map category to ITU material
        material_name = 'wood'  # Default
        for key, mat in FURNITURE_CATEGORY_MATERIALS.items():
            if key in cat_lower or cat_lower in key:
                material_name = mat
                break

        apply_material_color(mesh, material_name)

        # Build node transform: Room coords -> GLTF coords
        effective_theta = item.position.theta + item.orientation_offset
        rotation = trimesh.transformations.rotation_matrix(
            effective_theta,
            [0, 1, 0]  # GLTF Y axis (up)
        )

        # Room Y -> GLTF +Z (north is forward in GLTF view)
        # Position furniture with bottom at floor (Y = height/2 for placeholder boxes,
        # Y = 0 for meshes since we already moved floor to Y=0)
        y_offset = item.dimensions.height / 2 if mesh.bounds[0][1] < -0.001 else 0
        translation = trimesh.transformations.translation_matrix(
            [item.position.x, y_offset, item.position.y]
        )

        # Combined transform: rotation first, then translation
        transform = translation @ rotation

        scene.add_geometry(mesh, node_name=item.id, transform=transform)
