"""Scene loading and TX/RX configuration for Sionna RT.

Loads exported Sionna XML scenes and configures transmitters/receivers
with antenna arrays for wireless simulation.

Coordinate system matches Room model:
- Origin at SW corner
- X increases east, Y increases north, Z up
- Positions in meters
"""

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.room import FurnitureItem

try:
    import sionna.rt as rt
    from sionna.rt import Transmitter, Receiver, PlanarArray

    HAS_SIONNA = True
except ImportError:
    HAS_SIONNA = False
    rt = None
    Transmitter = None
    Receiver = None
    PlanarArray = None


def _check_sionna() -> None:
    """Raise ImportError if sionna is not installed."""
    if not HAS_SIONNA:
        raise ImportError(
            "sionna is required for wireless simulation. "
            "Install with: pip install sionna>=1.2.1"
        )


@dataclass(frozen=True)
class SceneConfig:
    """Configuration for scene loading and antenna arrays.

    Attributes:
        frequency: Carrier frequency in Hz (default 60 GHz mmWave)
                   Note: ITU materials support up to ~100 GHz. For THz (>100 GHz),
                   use custom materials or the theoretical THz coverage model.
        tx_power_dbm: Transmitter power in dBm (default 30 dBm = 1W for mmWave/THz)
        tx_antenna_pattern: TX antenna pattern ("iso", "dipole", "tr38901")
        rx_antenna_pattern: RX antenna pattern
        tx_polarization: TX antenna polarization ("V", "H", "VH", "cross")
        rx_polarization: RX antenna polarization
        tx_array_rows: Number of TX antenna rows (default 8 for mmWave MIMO)
        tx_array_cols: Number of TX antenna columns (default 8 for mmWave MIMO)
        rx_array_rows: Number of RX antenna rows (default 1)
        rx_array_cols: Number of RX antenna columns (default 1)
    """

    frequency: float = 60e9  # 60 GHz mmWave (highest ITU-supported)
    tx_power_dbm: float = 30.0  # 1W for mmWave/THz
    tx_antenna_pattern: str = "tr38901"  # 3GPP pattern for beamforming
    rx_antenna_pattern: str = "iso"
    tx_polarization: str = "cross"  # Cross-polarized for MIMO
    rx_polarization: str = "V"
    tx_array_rows: int = 8  # 8x8 = 64 element massive MIMO
    tx_array_cols: int = 8
    rx_array_rows: int = 1
    rx_array_cols: int = 1


def load_scene(
    xml_path: Path,
    config: Optional[SceneConfig] = None,
) -> "rt.Scene":
    """Load a Sionna RT scene from exported XML.

    Args:
        xml_path: Path to Mitsuba 3.0 XML file (from XMLExporter)
        config: Scene configuration (uses defaults if None)

    Returns:
        Configured Sionna RT Scene object

    Raises:
        ImportError: If sionna is not installed
        FileNotFoundError: If xml_path does not exist
    """
    _check_sionna()

    if config is None:
        config = SceneConfig()

    # Load scene with merge_shapes for performance
    scene = rt.load_scene(str(xml_path), merge_shapes=True)

    # Set carrier frequency
    scene.frequency = config.frequency

    # Configure TX antenna array (massive MIMO for mmWave/THz)
    scene.tx_array = PlanarArray(
        num_rows=config.tx_array_rows,
        num_cols=config.tx_array_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=config.tx_antenna_pattern,
        polarization=config.tx_polarization,
    )

    # Configure RX antenna array
    scene.rx_array = PlanarArray(
        num_rows=config.rx_array_rows,
        num_cols=config.rx_array_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=config.rx_antenna_pattern,
        polarization=config.rx_polarization,
    )

    return scene


def configure_transmitter(
    scene: "rt.Scene",
    position: List[float],
    power_dbm: Optional[float] = None,
    orientation: Optional[List[float]] = None,
    name: str = "tx",
) -> None:
    """Add a transmitter to the scene at specified position.

    Args:
        scene: Sionna RT Scene object
        position: [x, y, z] position in meters (room coordinates)
        power_dbm: Transmit power in dBm (optional override)
        orientation: [yaw, pitch, roll] in radians (optional)
        name: Transmitter name (default "tx")

    Raises:
        ImportError: If sionna is not installed
    """
    _check_sionna()

    # Create transmitter at position
    tx = Transmitter(name=name, position=position)

    # Set power if specified
    if power_dbm is not None:
        tx.power_dbm = power_dbm

    # Set orientation if specified
    if orientation is not None:
        tx.orientation = orientation

    # Add to scene
    scene.add(tx)


def configure_receivers(
    scene: "rt.Scene",
    positions: List[List[float]],
    names: Optional[List[str]] = None,
) -> None:
    """Add multiple receivers to the scene at specified positions.

    Args:
        scene: Sionna RT Scene object
        positions: List of [x, y, z] positions in meters
        names: Optional list of receiver names (default "rx_0", "rx_1", ...)

    Raises:
        ImportError: If sionna is not installed
        ValueError: If names provided but length doesn't match positions
    """
    _check_sionna()

    if names is not None and len(names) != len(positions):
        raise ValueError(
            f"Number of names ({len(names)}) must match "
            f"number of positions ({len(positions)})"
        )

    for i, pos in enumerate(positions):
        name = names[i] if names is not None else f"rx_{i}"
        rx = Receiver(name=name, position=pos)
        scene.add(rx)


def load_furniture_into_scene(
    scene: "rt.Scene",
    furniture_items: List["FurnitureItem"],
) -> int:
    """Add furniture meshes to ray tracing scene as obstructions.

    Loads OBJ files from each furniture item's model_path and adds them
    to the scene using scene.edit(). Materials are assigned based on
    furniture category using materials already defined in the scene.

    IMPORTANT: This uses scene.edit(), not scene.add(). The scene.add()
    method only accepts Transmitter/Receiver/RadioMaterial objects, not
    geometry. For SceneObject geometry, scene.edit(add=[...]) is required.

    Note: 3D-FUTURE models use Y-up coordinates but Sionna uses Z-up.
    A -90 degree rotation around X-axis is applied to convert Y-up to Z-up.

    Args:
        scene: Sionna RT Scene object (already loaded from XML with materials)
        furniture_items: List of FurnitureItem from Room model

    Returns:
        Number of furniture items successfully added

    Raises:
        ImportError: If sionna is not installed
        FileNotFoundError: If furniture OBJ file not found and fail_on_missing=True

    Example:
        >>> from src.wireless import load_scene, load_furniture_into_scene
        >>> from src.models.room import Room
        >>> scene = load_scene(xml_path)
        >>> count = load_furniture_into_scene(scene, room.furniture)
        >>> print(f"Added {count} furniture items as obstructions")
    """
    _check_sionna()

    from src.exporters.xml import get_material_for_category

    # Map ITU material names to scene material keys
    # Scene materials are named like "wood_mat" -> need to find actual material
    available_materials = scene.radio_materials

    def get_scene_material(category: str):
        """Get radio material from scene based on furniture category."""
        itu_name = get_material_for_category(category)  # e.g., "wood", "metal", "glass"
        mat_key = f"{itu_name}_mat"

        # Try exact match first
        if mat_key in available_materials:
            return available_materials[mat_key]

        # Try without _mat suffix
        if itu_name in available_materials:
            return available_materials[itu_name]

        # Fallback to concrete (always available from walls)
        if "concrete" in available_materials:
            return available_materials["concrete"]
        if "wall_mat" in available_materials:
            return available_materials["wall_mat"]

        # Use first available material
        return next(iter(available_materials.values()))

    furniture_objects = []
    positions = []  # Store positions to set after adding

    for item in furniture_items:
        # Get OBJ path
        obj_path = Path(item.model_path) / "raw_model.obj"
        if not obj_path.exists():
            warnings.warn(f"Furniture OBJ not found: {obj_path}")
            continue

        try:
            # Get material for this furniture category
            material = get_scene_material(item.category)

            # Create SceneObject with fname (Sionna 1.2.1+ API)
            scene_obj = rt.SceneObject(
                fname=str(obj_path),
                name=item.id,
                radio_material=material
            )
            furniture_objects.append(scene_obj)

            # Store position/orientation to apply after adding to scene
            # 3D-FUTURE models use Y-up, Sionna uses Z-up
            # Need to apply -90 degree rotation around X-axis to convert
            # Sionna orientation is [yaw, pitch, roll] where:
            #   - yaw (rotation around Z) = theta + orientation_offset
            #   - pitch (rotation around Y) = 0
            #   - roll (rotation around X) = -90 degrees for Y-up to Z-up
            yaw_degrees = math.degrees(item.position.theta + item.orientation_offset)
            positions.append({
                'obj': scene_obj,
                'position': [item.position.x, item.position.y, 0.0],
                'orientation': [yaw_degrees, 0, -90]  # [yaw, pitch, roll] - roll converts Y-up to Z-up
            })

        except Exception as e:
            warnings.warn(f"Failed to create SceneObject for {item.id}: {e}")
            continue

    # Add all furniture in single edit() call (more efficient)
    if furniture_objects:
        scene.edit(add=furniture_objects)

        # Now set positions and orientations (must be done after adding to scene)
        for pos_info in positions:
            try:
                pos_info['obj'].position = pos_info['position']
                pos_info['obj'].orientation = pos_info['orientation']
            except Exception as e:
                warnings.warn(f"Failed to set position for furniture: {e}")

    return len(furniture_objects)
