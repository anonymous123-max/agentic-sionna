"""Sionna XML (Mitsuba 3.0) scene exporter.

Exports Room and OutdoorScene models to Mitsuba 3.0 XML format for Sionna ray tracing.
Uses ITU-R P.2040-3 radio materials for realistic wireless propagation modeling.

Coordinate system:
- Mitsuba uses right-handed coords with Z-up by default
- Room/Scene coordinates map directly: X -> XML X, Y -> XML Y
- Theta rotation around Z axis

Reference: Sionna RT documentation, Mitsuba 3 scene format
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Set, Union

from src.models.outdoor import Building, GroundPlane, OutdoorScene, Road, Tree
from src.models.room import Room
from src.models.scene import Scene

# Import from extracted module
from src.exporters.materials import (
    CATEGORY_MATERIAL_MAP,
    ITU_MATERIALS,
    get_material_for_category,
)

# Re-export for backward compatibility
__all__ = [
    "CATEGORY_MATERIAL_MAP",
    "ITU_MATERIALS",
    "XMLExporter",
    "get_material_for_category",
]


class XMLExporter:
    """Export room layouts as Sionna-compatible Mitsuba 3.0 XML.

    Creates XML scene files with:
    - ITU radio materials for walls, floor, and furniture
    - Floor and wall geometry as rectangles
    - Furniture as external OBJ file references
    """

    def __init__(
        self,
        wall_material: str = 'concrete',
        floor_material: str = 'concrete',
        furniture_material: str = 'wood',
        ground_material: str = 'wet_ground',
    ):
        """Initialize XML exporter with material settings.

        Args:
            wall_material: ITU material for walls (default 'concrete')
            floor_material: ITU material for floor (default 'concrete')
            furniture_material: ITU material for furniture (default 'wood').
                Deprecated: Furniture materials are now assigned per-item based on
                category via CATEGORY_MATERIAL_MAP. This parameter is kept for
                backward compatibility but is no longer used.
            ground_material: ITU material for outdoor ground (default 'wet_ground')

        Raises:
            ValueError: If any material is not a valid ITU material name
        """
        self._validate_material(wall_material, 'wall_material')
        self._validate_material(floor_material, 'floor_material')
        self._validate_material(furniture_material, 'furniture_material')
        self._validate_material(ground_material, 'ground_material')

        self.wall_material = wall_material
        self.floor_material = floor_material
        self.furniture_material = furniture_material
        self.ground_material = ground_material

    def _validate_material(self, material: str, param_name: str) -> None:
        """Validate that material is a valid ITU material name.

        Args:
            material: Material name to validate
            param_name: Parameter name for error message

        Raises:
            ValueError: If material is not in ITU_MATERIALS
        """
        if material not in ITU_MATERIALS:
            raise ValueError(
                f"Invalid ITU material for {param_name}: '{material}'. "
                f"Valid materials: {sorted(ITU_MATERIALS)}"
            )

    def export(self, scene_or_room: Union[Scene, Room]) -> bytes:
        """Export scene or room to Mitsuba 3.0 XML bytes.

        Args:
            scene_or_room: Scene or Room model to export

        Returns:
            XML scene data as UTF-8 encoded bytes with XML declaration
        """
        xml_str = self.export_str(scene_or_room)
        return xml_str.encode('utf-8')

    def export_str(self, scene_or_room: Union[Scene, Room]) -> str:
        """Export scene or room to Mitsuba 3.0 XML string.

        Args:
            scene_or_room: Scene or Room model to export

        Returns:
            XML scene as string with XML declaration
        """
        # Normalize input to Scene
        if isinstance(scene_or_room, Room):
            scene = Scene(room=scene_or_room)
        else:
            scene = scene_or_room

        if scene.is_indoor_only:
            return self._export_room_str(scene.room)
        elif scene.is_outdoor_only:
            return self._export_outdoor_str(scene.outdoor)
        else:
            return self._export_combined_str(scene)

    def export_file(self, scene_or_room: Union[Scene, Room], path: Path) -> None:
        """Export scene or room to a Mitsuba 3.0 XML file.

        Args:
            scene_or_room: Scene or Room model to export
            path: Output file path
        """
        data = self.export(scene_or_room)
        path.write_bytes(data)

    def _export_room_str(self, room: Room) -> str:
        """Export indoor room to Mitsuba 3.0 XML string.

        Supports both rectangular and polygon rooms. Polygon rooms
        emit per-segment wall shapes instead of 4 cardinal walls.

        Args:
            room: Room model to export

        Returns:
            XML scene as string with XML declaration
        """
        # Create root scene element
        scene = ET.Element('scene', version='3.0.0')

        # Add integrator (required by Mitsuba)
        ET.SubElement(scene, 'integrator', type='path')

        # Define BSDF materials
        self._add_material(scene, 'wall_mat', self.wall_material)
        self._add_material(scene, 'floor_mat', self.floor_material)
        self._add_material(scene, 'ceiling_mat', 'ceiling_board')
        self._add_material(scene, 'glass_mat', 'glass')

        # Collect unique furniture materials needed based on categories
        furniture_materials = {
            get_material_for_category(item.category)
            for item in room.furniture
        }
        for material in furniture_materials:
            mat_id = f"{material}_mat"
            # Only add if not already defined (avoid duplicates with wall_mat etc.)
            if mat_id not in ['wall_mat', 'floor_mat', 'glass_mat']:
                self._add_material(scene, mat_id, material)

        # Add floor (uses bbox rectangle for both rectangular and polygon rooms)
        self._add_floor(scene, room)

        # Add walls - polygon rooms use per-segment walls
        if room.is_rectangular:
            self._add_walls(scene, room)
        else:
            self._add_polygon_walls(scene, room)

        # Add ceiling (uses bbox rectangle for both)
        self._add_ceiling(scene, room)

        # Add windows
        self._add_windows(scene, room)

        # Add furniture
        for item in room.furniture:
            self._add_furniture(scene, item)

        # Generate XML string with declaration
        xml_str = ET.tostring(scene, encoding='unicode')
        return f'<?xml version="1.0"?>\n{xml_str}'

    def _export_outdoor_str(self, outdoor: OutdoorScene) -> str:
        """Export outdoor scene to Mitsuba 3.0 XML string.

        Args:
            outdoor: OutdoorScene model to export

        Returns:
            XML scene as string with XML declaration
        """
        scene = ET.Element('scene', version='3.0.0')
        ET.SubElement(scene, 'integrator', type='path')

        # Collect all unique materials needed from outdoor elements
        materials_needed: Set[str] = {'wood'}  # Always need wood for trees
        materials_needed.add(outdoor.ground.material)
        for building in outdoor.buildings:
            materials_needed.add(building.material)
        for road in outdoor.roads:
            materials_needed.add(road.material)

        # Define all needed materials
        for material in materials_needed:
            self._add_material(scene, f'{material}_mat', material)

        # Add ground plane
        self._add_ground_plane(scene, outdoor)

        # Add buildings
        for building in outdoor.buildings:
            self._add_outdoor_building(scene, building)

        # Add roads
        for road in outdoor.roads:
            self._add_road(scene, road)

        # Add trees
        for tree in outdoor.trees:
            self._add_outdoor_tree(scene, tree)

        return f'<?xml version="1.0"?>\n{ET.tostring(scene, encoding="unicode")}'

    def _export_combined_str(self, scene: Scene) -> str:
        """Export combined indoor/outdoor scene to Mitsuba 3.0 XML string.

        For now, routes to outdoor export. Future: merge room into outdoor context.

        Args:
            scene: Scene with both room and outdoor components

        Returns:
            XML scene as string with XML declaration
        """
        # TODO: Properly merge indoor room into outdoor scene context
        return self._export_outdoor_str(scene.outdoor)

    def _add_material(self, scene: ET.Element, mat_id: str, material: str) -> None:
        """Add an ITU radio material BSDF definition.

        Args:
            scene: Root scene element
            mat_id: Material ID for referencing
            material: ITU material name
        """
        bsdf = ET.SubElement(scene, 'bsdf', type='itu-radio-material', id=mat_id)
        # Sionna 1.2.1+ expects 'type' property for ITU material selection
        ET.SubElement(bsdf, 'string', name='type', value=material)

    def _add_floor(self, scene: ET.Element, room: Room) -> None:
        """Add floor rectangle to scene.

        Floor is positioned at Z=0, centered in the room.

        Args:
            scene: Root scene element
            room: Room model for dimensions
        """
        floor = ET.SubElement(scene, 'shape', type='rectangle', id='floor')

        # Transform: scale to room size, translate to room center
        transform = ET.SubElement(floor, 'transform', name='to_world')
        # Mitsuba rectangle is 2x2, so scale by half room dimensions
        ET.SubElement(
            transform, 'scale',
            x=str(room.width / 2),
            y=str(room.length / 2),
            z='1'
        )
        ET.SubElement(
            transform, 'translate',
            x=str(room.width / 2),
            y=str(room.length / 2),
            z='0'
        )

        # Reference floor material
        ET.SubElement(floor, 'ref', id='floor_mat')

    def _add_ceiling(self, scene: ET.Element, room: Room) -> None:
        """Add ceiling rectangle to scene.

        Ceiling is positioned at Z=height, centered in the room.
        Uses ceiling_board material for accurate RF simulation.

        Args:
            scene: Root scene element
            room: Room model for dimensions
        """
        ceiling = ET.SubElement(scene, 'shape', type='rectangle', id='ceiling')

        # Transform: scale to room size, translate to room center at height
        transform = ET.SubElement(ceiling, 'transform', name='to_world')
        # Mitsuba rectangle is 2x2, so scale by half room dimensions
        ET.SubElement(
            transform, 'scale',
            x=str(room.width / 2),
            y=str(room.length / 2),
            z='1'
        )
        # Rotate 180 degrees around X to face downward
        ET.SubElement(transform, 'rotate', x='1', angle='180')
        ET.SubElement(
            transform, 'translate',
            x=str(room.width / 2),
            y=str(room.length / 2),
            z=str(room.height)
        )

        # Reference ceiling material
        ET.SubElement(ceiling, 'ref', id='ceiling_mat')

    def _add_walls(self, scene: ET.Element, room: Room) -> None:
        """Add four wall rectangles to scene.

        Walls are vertical rectangles positioned at room edges.

        Args:
            scene: Root scene element
            room: Room model for dimensions
        """
        # South wall: at y=0, facing north (+Y)
        self._add_wall(
            scene, 'wall_south',
            center_x=room.width / 2,
            center_y=0,
            center_z=room.height / 2,
            half_width=room.width / 2,
            half_height=room.height / 2,
            rotation_angle=90,  # Rotate to vertical, facing +Y
            rotation_axis='x'
        )

        # North wall: at y=length, facing south (-Y)
        self._add_wall(
            scene, 'wall_north',
            center_x=room.width / 2,
            center_y=room.length,
            center_z=room.height / 2,
            half_width=room.width / 2,
            half_height=room.height / 2,
            rotation_angle=-90,  # Rotate to vertical, facing -Y
            rotation_axis='x'
        )

        # West wall: at x=0, facing east (+X)
        self._add_wall(
            scene, 'wall_west',
            center_x=0,
            center_y=room.length / 2,
            center_z=room.height / 2,
            half_width=room.length / 2,
            half_height=room.height / 2,
            rotation_angle=90,  # Rotate to vertical, facing +X
            rotation_axis='y'
        )

        # East wall: at x=width, facing west (-X)
        self._add_wall(
            scene, 'wall_east',
            center_x=room.width,
            center_y=room.length / 2,
            center_z=room.height / 2,
            half_width=room.length / 2,
            half_height=room.height / 2,
            rotation_angle=-90,  # Rotate to vertical, facing -X
            rotation_axis='y'
        )

    def _add_wall(
        self,
        scene: ET.Element,
        wall_id: str,
        center_x: float,
        center_y: float,
        center_z: float,
        half_width: float,
        half_height: float,
        rotation_angle: float,
        rotation_axis: str,
    ) -> None:
        """Add a single wall rectangle to scene.

        Args:
            scene: Root scene element
            wall_id: Unique ID for the wall shape
            center_x, center_y, center_z: Wall center position
            half_width, half_height: Half dimensions for scale
            rotation_angle: Rotation angle in degrees
            rotation_axis: Axis to rotate around ('x' or 'y')
        """
        wall = ET.SubElement(scene, 'shape', type='rectangle', id=wall_id)

        transform = ET.SubElement(wall, 'transform', name='to_world')

        # Scale first (Mitsuba rectangle is 2x2)
        ET.SubElement(
            transform, 'scale',
            x=str(half_width),
            y=str(half_height),
            z='1'
        )

        # Rotate to vertical position
        rotate_attrs = {'angle': str(rotation_angle)}
        rotate_attrs[rotation_axis] = '1'
        ET.SubElement(transform, 'rotate', **rotate_attrs)

        # Translate to final position
        ET.SubElement(
            transform, 'translate',
            x=str(center_x),
            y=str(center_y),
            z=str(center_z)
        )

        # Reference wall material
        ET.SubElement(wall, 'ref', id='wall_mat')

    def _add_windows(self, scene: ET.Element, room: Room) -> None:
        """Add window rectangles to scene with glass material.

        Windows are vertical rectangles positioned on walls.
        Uses ITU 'glass' material for accurate RF propagation modeling.

        Args:
            scene: Root scene element
            room: Room model containing window definitions
        """
        for window in room.windows:
            self._add_window(scene, window, room)

    def _add_window(self, scene: ET.Element, window, room: Room) -> None:
        """Add a single window rectangle to scene.

        Args:
            scene: Root scene element
            window: Window model with wall, position, width, height, sill_height
            room: Room model for wall positions
        """
        # Create unique window ID
        window_id = f"window_{window.wall}_{int(window.position * 100)}"
        shape = ET.SubElement(scene, 'shape', type='rectangle', id=window_id)

        transform = ET.SubElement(shape, 'transform', name='to_world')

        # Calculate window center position and rotation based on wall
        # Mitsuba rectangle is 2x2, so scale by half dimensions
        half_width = window.width / 2
        half_height = window.height / 2
        center_z = window.sill_height + half_height

        if window.wall == "south":
            # South wall at y=0, facing north (+Y)
            center_x = window.position + half_width
            center_y = 0
            ET.SubElement(transform, 'scale', x=str(half_width), y=str(half_height), z='1')
            ET.SubElement(transform, 'rotate', x='1', angle='90')
            ET.SubElement(transform, 'translate', x=str(center_x), y=str(center_y), z=str(center_z))

        elif window.wall == "north":
            # North wall at y=length, facing south (-Y)
            center_x = window.position + half_width
            center_y = room.length
            ET.SubElement(transform, 'scale', x=str(half_width), y=str(half_height), z='1')
            ET.SubElement(transform, 'rotate', x='1', angle='-90')
            ET.SubElement(transform, 'translate', x=str(center_x), y=str(center_y), z=str(center_z))

        elif window.wall == "west":
            # West wall at x=0, facing east (+X)
            center_x = 0
            center_y = window.position + half_width
            ET.SubElement(transform, 'scale', x=str(half_width), y=str(half_height), z='1')
            ET.SubElement(transform, 'rotate', y='1', angle='90')
            ET.SubElement(transform, 'translate', x=str(center_x), y=str(center_y), z=str(center_z))

        elif window.wall == "east":
            # East wall at x=width, facing west (-X)
            center_x = room.width
            center_y = window.position + half_width
            ET.SubElement(transform, 'scale', x=str(half_width), y=str(half_height), z='1')
            ET.SubElement(transform, 'rotate', y='1', angle='-90')
            ET.SubElement(transform, 'translate', x=str(center_x), y=str(center_y), z=str(center_z))

        # Reference glass material for RF propagation
        ET.SubElement(shape, 'ref', id='glass_mat')

    def _add_furniture(self, scene: ET.Element, item) -> None:
        """Add furniture item as OBJ shape reference or cube fallback.

        3D-FUTURE models use Y-up coordinates but Sionna uses Z-up.
        A -90 degree rotation around X-axis converts Y-up to Z-up.
        If model_path is empty, falls back to cube geometry.

        Args:
            scene: Root scene element
            item: FurnitureItem with position and model_path
        """
        if not item.model_path:
            self._add_furniture_box(scene, item)
            return

        shape = ET.SubElement(scene, 'shape', type='obj', id=item.id)

        # Reference OBJ file
        obj_path = f"{item.model_path}/raw_model.obj"
        ET.SubElement(shape, 'string', name='filename', value=obj_path)

        # Transform: convert Y-up to Z-up, rotate around Z, then translate
        # Mitsuba applies transforms in order: first listed = first applied
        transform = ET.SubElement(shape, 'transform', name='to_world')

        # 1. Convert Y-up (3D-FUTURE) to Z-up (Sionna): rotate -90 around X
        ET.SubElement(transform, 'rotate', x='1', angle='-90')

        # 2. Rotate by effective theta (theta + orientation_offset) around Z axis
        effective_theta = item.position.theta + item.orientation_offset
        angle_degrees = math.degrees(effective_theta)
        ET.SubElement(transform, 'rotate', z='1', angle=str(angle_degrees))

        # 3. Translate to position (furniture sits on floor at z=0)
        ET.SubElement(
            transform, 'translate',
            x=str(item.position.x),
            y=str(item.position.y),
            z='0'
        )

        # Reference material based on furniture category
        material = get_material_for_category(item.category)
        mat_id = f"{material}_mat"
        ET.SubElement(shape, 'ref', id=mat_id)

    def _add_furniture_box(self, scene: ET.Element, item) -> None:
        """Add furniture as cube geometry (no OBJ file needed).

        Used for dashboard-created furniture that has no 3D model file.
        Mitsuba cube extends from (-1,-1,-1) to (1,1,1), so scale by
        half-dimensions.

        Args:
            scene: Root scene element
            item: FurnitureItem with position and dimensions
        """
        shape = ET.SubElement(scene, 'shape', type='cube', id=item.id)

        transform = ET.SubElement(shape, 'transform', name='to_world')

        # Scale by half dimensions (Mitsuba cube is 2x2x2)
        ET.SubElement(
            transform, 'scale',
            x=str(item.dimensions.width / 2),
            y=str(item.dimensions.depth / 2),
            z=str(item.dimensions.height / 2),
        )

        # Rotate around Z by theta
        effective_theta = item.position.theta + item.orientation_offset
        angle_degrees = math.degrees(effective_theta)
        ET.SubElement(transform, 'rotate', z='1', angle=str(angle_degrees))

        # Translate to position (center at height/2 so bottom sits on floor)
        ET.SubElement(
            transform, 'translate',
            x=str(item.position.x),
            y=str(item.position.y),
            z=str(item.dimensions.height / 2),
        )

        # Reference material based on furniture category
        material = get_material_for_category(item.category)
        mat_id = f"{material}_mat"
        ET.SubElement(shape, 'ref', id=mat_id)

    # =========================================================================
    # Polygon room wall helpers
    # =========================================================================

    def _add_polygon_walls(self, scene: ET.Element, room: Room) -> None:
        """Add wall segments for polygon rooms.

        Each polygon edge becomes a separate wall rectangle, positioned
        and rotated to match the edge direction.

        Args:
            scene: Root scene element
            room: Room model with polygon floor
        """
        for i, seg in enumerate(room.wall_segments):
            self._add_polygon_wall_segment(
                scene, f'wall_seg_{i}', seg, room.height
            )

    def _add_polygon_wall_segment(
        self,
        scene: ET.Element,
        wall_id: str,
        seg: "WallSegment",
        height: float,
    ) -> None:
        """Add a single polygon wall segment as a Mitsuba rectangle.

        The rectangle is scaled to segment length x room height,
        rotated to match the segment angle, and translated to the
        segment midpoint at half height.

        Args:
            scene: Root scene element
            wall_id: Unique ID for this wall shape
            seg: WallSegment with start/end points
            height: Wall height in meters
        """
        from src.models.room import WallSegment

        wall = ET.SubElement(scene, 'shape', type='rectangle', id=wall_id)
        transform = ET.SubElement(wall, 'transform', name='to_world')

        # Scale: half segment length in X, half height in Y (Mitsuba rect is 2x2)
        half_len = seg.length / 2
        half_h = height / 2
        ET.SubElement(
            transform, 'scale',
            x=str(half_len),
            y=str(half_h),
            z='1'
        )

        # Rotate: first stand up (90 around X), then rotate around Z for direction
        angle_deg = math.degrees(seg.angle)
        ET.SubElement(transform, 'rotate', x='1', angle='90')
        ET.SubElement(transform, 'rotate', z='1', angle=str(angle_deg))

        # Translate to segment midpoint at half height
        mx, my = seg.midpoint
        ET.SubElement(
            transform, 'translate',
            x=str(mx),
            y=str(my),
            z=str(half_h)
        )

        ET.SubElement(wall, 'ref', id='wall_mat')

    # =========================================================================
    # Outdoor scene geometry helpers
    # =========================================================================

    def _add_ground_plane(self, scene: ET.Element, outdoor: OutdoorScene) -> None:
        """Add ground plane rectangle to outdoor scene.

        Ground is positioned at Z=0, covering scene dimensions.

        Args:
            scene: Root scene element
            outdoor: OutdoorScene for dimensions
        """
        ground = ET.SubElement(scene, 'shape', type='rectangle', id='ground')

        transform = ET.SubElement(ground, 'transform', name='to_world')
        # Mitsuba rectangle is 2x2, scale by half dimensions
        ET.SubElement(
            transform, 'scale',
            x=str(outdoor.width / 2),
            y=str(outdoor.length / 2),
            z='1'
        )
        ET.SubElement(
            transform, 'translate',
            x=str(outdoor.width / 2),
            y=str(outdoor.length / 2),
            z='0'
        )

        ET.SubElement(ground, 'ref', id=f'{outdoor.ground.material}_mat')

    def _add_outdoor_building(self, scene: ET.Element, building: Building) -> None:
        """Add building as cube shape to outdoor scene.

        Mitsuba cube extends from (-1,-1,-1) to (1,1,1), so scale by half-dimensions.

        Args:
            scene: Root scene element
            building: Building model with footprint and height
        """
        shape = ET.SubElement(scene, 'shape', type='cube', id=f'building_{building.id}')

        # Get bounding box from polygon
        minx, miny, maxx, maxy = building.as_polygon().bounds
        width = maxx - minx
        depth = maxy - miny
        height = building.height

        # Calculate center position
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2
        center_z = height / 2

        transform = ET.SubElement(shape, 'transform', name='to_world')
        # Scale by half dimensions (cube is 2x2x2)
        ET.SubElement(
            transform, 'scale',
            x=str(width / 2),
            y=str(depth / 2),
            z=str(height / 2)
        )
        ET.SubElement(
            transform, 'translate',
            x=str(center_x),
            y=str(center_y),
            z=str(center_z)
        )

        # Use building's material
        ET.SubElement(shape, 'ref', id=f'{building.material}_mat')

    def _add_road(self, scene: ET.Element, road: Road) -> None:
        """Add road as rectangle shape to outdoor scene.

        Road is placed at slight Z offset (0.01) to avoid z-fighting with ground.

        Args:
            scene: Root scene element
            road: Road model with centerline and width
        """
        shape = ET.SubElement(scene, 'shape', type='rectangle', id=f'road_{road.id}')

        # Get bounding box from road polygon
        poly = road.as_polygon()
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        depth = maxy - miny

        # Calculate center position
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2

        transform = ET.SubElement(shape, 'transform', name='to_world')
        ET.SubElement(
            transform, 'scale',
            x=str(width / 2),
            y=str(depth / 2),
            z='1'
        )
        ET.SubElement(
            transform, 'translate',
            x=str(center_x),
            y=str(center_y),
            z='0.01'  # Slight offset to avoid z-fighting
        )

        # Use road's material
        ET.SubElement(shape, 'ref', id=f'{road.material}_mat')

    def _add_outdoor_tree(self, scene: ET.Element, tree: Tree) -> None:
        """Add tree as cylinder (trunk) + sphere (crown) to outdoor scene.

        Mitsuba primitives are centered at origin, so translate to position.

        Args:
            scene: Root scene element
            tree: Tree model with position and dimensions
        """
        x, y = tree.position
        trunk_height = tree.trunk_height

        # Add trunk as cylinder
        trunk = ET.SubElement(scene, 'shape', type='cylinder', id=f'trunk_{tree.id}')
        trunk_transform = ET.SubElement(trunk, 'transform', name='to_world')
        # Mitsuba cylinder has unit height along Z, scale and translate
        ET.SubElement(
            trunk_transform, 'scale',
            x=str(tree.trunk_radius),
            y=str(tree.trunk_radius),
            z=str(trunk_height / 2)
        )
        ET.SubElement(
            trunk_transform, 'translate',
            x=str(x),
            y=str(y),
            z=str(trunk_height / 2)
        )
        ET.SubElement(trunk, 'ref', id='wood_mat')

        # Add crown as sphere
        crown = ET.SubElement(scene, 'shape', type='sphere', id=f'crown_{tree.id}')
        crown_transform = ET.SubElement(crown, 'transform', name='to_world')
        # Mitsuba sphere is unit radius, scale and translate
        ET.SubElement(
            crown_transform, 'scale',
            x=str(tree.crown_radius),
            y=str(tree.crown_radius),
            z=str(tree.crown_radius)
        )
        # Crown sits on top of trunk
        crown_center_z = trunk_height + tree.crown_height / 2
        ET.SubElement(
            crown_transform, 'translate',
            x=str(x),
            y=str(y),
            z=str(crown_center_z)
        )
        ET.SubElement(crown, 'ref', id='wood_mat')
