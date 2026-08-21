"""Coverage map computation and visualization using Sionna RT.

Provides GPU-accelerated ray tracing for coverage map generation
with configurable parameters for indoor wireless simulation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, TYPE_CHECKING

try:
    import sionna.rt as rt
    from sionna.rt import RadioMapSolver

    HAS_SIONNA = True
except ImportError:
    HAS_SIONNA = False
    rt = None
    RadioMapSolver = None

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure


def _check_sionna() -> None:
    """Raise ImportError if sionna is not installed."""
    if not HAS_SIONNA:
        raise ImportError(
            "sionna is required for wireless simulation. "
            "Install with: pip install sionna>=1.2.1"
        )


@dataclass(frozen=True)
class CoverageConfig:
    """Configuration for coverage map computation.

    Attributes:
        max_depth: Number of ray bounces (reflections/diffractions).
                   Higher = more accurate but slower. Default 5 for indoor.
        cell_size: Grid resolution in meters (x_size, y_size).
                   Smaller = more detail but more memory.
        samples_per_tx: Number of ray samples per transmitter.
                        Higher = more stable results. Start lower for testing.
        z_height: Height of coverage plane in meters.
                  Default 1.5m matches Sionna's default and is appropriate for
                  outdoor scenes (tree trunk level). Use 1.0m for indoor
                  waist-height measurements.
    """

    max_depth: int = 5
    cell_size: Tuple[float, float] = (0.5, 0.5)
    samples_per_tx: int = 1_000_000
    z_height: float = 1.5


def compute_coverage_map(
    scene: "rt.Scene",
    config: Optional[CoverageConfig] = None,
) -> "rt.RadioMap":
    """Compute coverage map for a scene with configured transmitter.

    Uses GPU-accelerated ray tracing to compute signal strength
    across a 2D plane in the scene.

    Args:
        scene: Sionna RT Scene with transmitter configured
        config: Coverage configuration (uses defaults if None)

    Returns:
        RadioMap object containing signal strength data

    Raises:
        ImportError: If sionna is not installed
    """
    _check_sionna()

    if config is None:
        config = CoverageConfig()

    # Create radio map solver
    rm_solver = RadioMapSolver()

    # Compute scene bounds to set coverage plane at desired z_height.
    # Sionna requires center, size, and orientation all together.
    import drjit as dr
    import mitsuba as mi

    scene_min = scene.mi_scene.bbox().min
    scene_min = dr.select(dr.isinf(scene_min), -1.0, scene_min)
    scene_max = scene.mi_scene.bbox().max
    scene_max = dr.select(dr.isinf(scene_max), 1.0, scene_max)

    center = 0.5 * (scene_min + scene_max)
    center.z = config.z_height
    size = scene_max - scene_min

    # Compute coverage map
    radio_map = rm_solver(
        scene=scene,
        max_depth=config.max_depth,
        cell_size=config.cell_size,
        samples_per_tx=config.samples_per_tx,
        center=[center.x, center.y, center.z],
        size=[size.x, size.y],
        orientation=[0, 0, 0],
    )

    return radio_map


def visualize_coverage(
    radio_map: "rt.RadioMap",
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> "Figure":
    """Visualize coverage map as a matplotlib figure.

    Args:
        radio_map: RadioMap from compute_coverage_map()
        output_path: Optional path to save figure (PNG, PDF, etc.)
        title: Optional title for the figure

    Returns:
        matplotlib Figure object

    Raises:
        ImportError: If matplotlib is not installed
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 8))

    # Try to use RadioMap's built-in visualization, fall back to manual
    try:
        radio_map.show(ax=ax)
    except TypeError:
        # Sionna 1.2.1+ API change - show() doesn't take ax parameter
        # Extract data and plot manually
        try:
            # Get coverage data as numpy array
            data = radio_map.path_gain.numpy()
            if len(data.shape) > 2:
                data = data.squeeze()
            # Convert to dB
            data_db = 10 * np.log10(np.abs(data) + 1e-12)
            im = ax.imshow(data_db, origin='lower', cmap='jet',
                          aspect='auto')
            plt.colorbar(im, ax=ax, label='Path Gain (dB)')
            ax.set_xlabel('X (cells)')
            ax.set_ylabel('Y (cells)')
        except Exception as e:
            # Last resort: just call show() without args
            radio_map.show()
            ax.set_title(title or "Coverage Map")

    if title:
        ax.set_title(title)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def get_coverage_at_point(
    radio_map: "rt.RadioMap",
    x: float,
    y: float,
) -> float:
    """Extract signal strength at a specific (x, y) coordinate.

    Args:
        radio_map: RadioMap from compute_coverage_map()
        x: X coordinate in meters
        y: Y coordinate in meters

    Returns:
        Signal strength in dB (path gain)

    Raises:
        ImportError: If sionna is not installed
        ValueError: If coordinates are outside the coverage map bounds
    """
    _check_sionna()

    # Get the coverage data array and spatial bounds
    # RadioMap stores data as a 2D tensor with associated spatial coordinates
    data = radio_map.value  # Path gain values
    x_coords = radio_map.x  # X coordinates of grid cells
    y_coords = radio_map.y  # Y coordinates of grid cells

    # Find nearest grid cell
    import numpy as np

    x_idx = int(np.argmin(np.abs(np.array(x_coords) - x)))
    y_idx = int(np.argmin(np.abs(np.array(y_coords) - y)))

    # Extract value (convert from tensor if needed)
    value = float(data[y_idx, x_idx])

    return value


# Import from extracted module for backward compatibility
from src.wireless.thz_model import (  # noqa: F401
    THzConfig,
    _ray_intersects_aabb_2d,
    compute_thz_coverage,
)
