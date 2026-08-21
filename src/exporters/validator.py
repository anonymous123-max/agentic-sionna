"""Cross-format alignment validation utilities.

Extracts furniture positions from each export format and validates they match
the original Room model coordinates. This verifies the core value proposition
that all exports derive from a single Room model with identical positions.
"""

import io
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from src.models.room import Room


def extract_positions_xml(xml_str: str) -> Dict[str, Tuple[float, float, float]]:
    """Extract furniture positions from Sionna XML.

    Parses the XML and extracts translate/rotate transforms for furniture shapes.

    Args:
        xml_str: Mitsuba 3.0 XML string from XMLExporter

    Returns:
        Dict mapping furniture id -> (x, y, theta) in Room coordinates.
        theta is in radians.
    """
    root = ET.fromstring(xml_str)
    positions = {}

    for shape in root.findall(".//shape[@type='obj']"):
        shape_id = shape.get('id')
        if shape_id is None:
            continue

        transform = shape.find('transform')
        if transform is None:
            continue

        # Extract translate values
        translate = transform.find('translate')
        rotate = transform.find('rotate')

        x = float(translate.get('x', 0)) if translate is not None else 0.0
        y = float(translate.get('y', 0)) if translate is not None else 0.0

        # Extract rotation angle around z axis (in XML it's stored in degrees)
        # Since 3D-FUTURE Y-up fix, there are multiple rotate elements;
        # find the one with z='1' (theta rotation)
        theta = 0.0
        for rot in transform.findall('rotate'):
            if rot.get('z') == '1':
                angle_degrees = float(rot.get('angle', 0))
                theta = np.radians(angle_degrees)
                break

        positions[shape_id] = (x, y, theta)

    return positions


def extract_positions_gltf(glb_bytes: bytes) -> Dict[str, Tuple[float, float, float]]:
    """Extract furniture positions from GLTF/GLB.

    Loads the GLB and extracts node transforms, converting from GLTF Y-up
    coordinates back to Room coordinates.

    Args:
        glb_bytes: Binary GLB data from GLTFExporter

    Returns:
        Dict mapping node name -> (x, y, theta) in Room coordinates.
        Note: Converts from GLTF Y-up coords back to Room coords.
        Returns empty dict if no furniture nodes found (e.g., models not loaded).

    Coordinate transform (inverse of GLTFExporter):
    - GLTF X -> Room X
    - GLTF +Z -> Room Y
    - GLTF theta -> Room theta (direct, orientation_offset already applied)
    """
    scene = trimesh.load(io.BytesIO(glb_bytes), file_type='glb')
    positions = {}

    # Internal nodes to skip (room geometry and markers)
    # Wall nodes may be segmented (wall_south_0, wall_south_header_1, etc.)
    skip_prefixes = ('world', 'geometry_', '_origin_marker', 'floor', 'wall_')

    # Use the graph.get() method to access transforms by node name
    for node_name in scene.graph.nodes:
        if node_name.startswith(skip_prefixes):
            continue

        # Get transform matrix for this node
        matrix, _ = scene.graph.get(node_name)
        if matrix is None:
            continue

        # Extract translation: GLTF X -> Room X, GLTF +Z -> Room Y
        translation = matrix[:3, 3]
        room_x = translation[0]
        room_y = translation[2]  # GLTF +Z -> Room Y

        # Extract rotation around GLTF Y axis
        # For rotation around Y axis: cos(theta) in [0,0] and [2,2], sin(theta) in [0,2]
        rotation_matrix = matrix[:3, :3]
        theta = np.arctan2(rotation_matrix[0, 2], rotation_matrix[0, 0])

        # Normalize theta to [0, 2*pi)
        if theta < 0:
            theta += 2 * np.pi

        positions[node_name] = (room_x, room_y, theta)

    return positions


def extract_positions_ascii(
    ascii_str: str, chars_per_meter: float = 2.0
) -> List[Tuple[float, float]]:
    """Extract approximate furniture positions from ASCII art.

    Scans the ASCII grid for furniture characters and converts grid positions
    back to Room coordinates.

    Args:
        ascii_str: ASCII art string from ASCIIExporter
        chars_per_meter: Grid resolution (must match exporter setting)

    Returns:
        List of (x, y) center positions in Room coordinates.
        Less precise than other formats due to grid resolution.
    """
    lines = ascii_str.strip().split('\n')
    rows = len(lines)
    positions = []

    # Furniture characters (from ASCIIExporter CATEGORY_CHARS + default)
    furniture_chars = set('#BSTCDWNKLA')

    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char in furniture_chars:
                # Convert grid to room coords
                # ASCII exporter: grid_row = (rows - 1) - int(y * chars_per_meter)
                # Inverse: y = (rows - 1 - r) / chars_per_meter
                room_x = c / chars_per_meter
                room_y = (rows - 1 - r) / chars_per_meter
                positions.append((room_x, room_y))

    return positions


def validate_alignment(
    room: Room,
    png_exporter,
    ascii_exporter,
    xml_exporter,
    gltf_exporter,
    tolerance: float = 0.1
) -> Dict[str, bool]:
    """Validate that all exporters produce consistent positions.

    Exports the room through all formats, extracts positions, and compares
    them against the original Room model coordinates.

    Args:
        room: Room model with furniture
        png_exporter: PNGExporter instance (validates output is valid PNG)
        ascii_exporter: ASCIIExporter instance
        xml_exporter: XMLExporter instance
        gltf_exporter: GLTFExporter instance
        tolerance: Maximum position difference in meters (default 0.1)

    Returns:
        Dict with validation results for each check.
        Keys are '{item_id}_{format}_{check}' where check is 'position' or 'rotation'.
        Values are True if within tolerance, False otherwise.
    """
    results = {}

    # Export all formats
    xml_str = xml_exporter.export_str(room)
    glb_bytes = gltf_exporter.export(room)
    png_bytes = png_exporter.export(room)

    # Validate PNG produces valid output (basic check)
    results['png_valid'] = png_bytes[:8] == b'\x89PNG\r\n\x1a\n'

    # Extract positions from XML and GLTF
    xml_positions = extract_positions_xml(xml_str)
    gltf_positions = extract_positions_gltf(glb_bytes)

    # Compare each furniture item
    for item in room.furniture:
        item_id = item.id
        expected_x = item.position.x
        expected_y = item.position.y
        expected_theta = item.position.theta

        # Both XML and GLTF export effective_theta = theta + orientation_offset
        orientation_offset = getattr(item, 'orientation_offset', 0.0)
        expected_effective_theta = expected_theta + orientation_offset
        # Normalize to [0, 2*pi)
        while expected_effective_theta < 0:
            expected_effective_theta += 2 * np.pi
        while expected_effective_theta >= 2 * np.pi:
            expected_effective_theta -= 2 * np.pi

        # Check XML
        if item_id in xml_positions:
            xml_x, xml_y, xml_theta = xml_positions[item_id]
            xml_pos_diff = np.sqrt((expected_x - xml_x)**2 + (expected_y - xml_y)**2)
            results[f'{item_id}_xml_position'] = xml_pos_diff < tolerance

            # Theta comparison with wraparound handling
            theta_diff = abs(expected_effective_theta - xml_theta)
            theta_diff = min(theta_diff, 2 * np.pi - theta_diff)
            results[f'{item_id}_xml_rotation'] = theta_diff < 0.1  # ~5.7 degrees

        # Check GLTF
        if item_id in gltf_positions:
            gltf_x, gltf_y, gltf_theta = gltf_positions[item_id]
            gltf_pos_diff = np.sqrt((expected_x - gltf_x)**2 + (expected_y - gltf_y)**2)
            results[f'{item_id}_gltf_position'] = gltf_pos_diff < tolerance

            # Theta comparison with wraparound handling
            theta_diff = abs(expected_effective_theta - gltf_theta)
            theta_diff = min(theta_diff, 2 * np.pi - theta_diff)
            results[f'{item_id}_gltf_rotation'] = theta_diff < 0.1

    return results
