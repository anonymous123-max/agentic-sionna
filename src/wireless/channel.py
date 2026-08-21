"""OFDM resource grid and MIMO antenna array configuration for Sionna.

Provides configuration dataclasses and factory functions for:
- OFDM multi-carrier modulation (5G NR compatible)
- MIMO antenna arrays (planar arrays with configurable geometry)
- Path computation for channel impulse response

The OFDM configuration is used for link-level simulation (PHY layer),
while MIMO arrays affect ray tracing results through spatial diversity.
"""

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

try:
    import sionna.rt as rt
    from sionna.rt import PlanarArray, PathSolver

    HAS_SIONNA = True
except ImportError:
    HAS_SIONNA = False
    rt = None
    PlanarArray = None
    PathSolver = None

try:
    from sionna.phy.ofdm import ResourceGrid

    HAS_SIONNA_PHY = True
except ImportError:
    HAS_SIONNA_PHY = False
    ResourceGrid = None

if TYPE_CHECKING:
    import sionna.rt as rt
    from sionna.rt import PlanarArray, Paths
    from sionna.phy import ResourceGrid


def _check_sionna() -> None:
    """Raise ImportError if sionna is not installed."""
    if not HAS_SIONNA:
        raise ImportError(
            "sionna is required for wireless simulation. "
            "Install with: pip install sionna>=1.2.1"
        )


def _check_sionna_phy() -> None:
    """Raise ImportError if sionna.phy is not available."""
    if not HAS_SIONNA_PHY:
        raise ImportError(
            "sionna.phy is required for OFDM configuration. "
            "Install with: pip install sionna>=1.2.1"
        )


@dataclass(frozen=True)
class OFDMConfig:
    """Configuration for OFDM resource grid.

    Default values follow 5G NR specifications for typical indoor scenarios.

    Attributes:
        num_ofdm_symbols: Number of OFDM symbols per slot (default 14 for 5G NR)
        fft_size: FFT size for OFDM (default 128)
        subcarrier_spacing: Subcarrier spacing in Hz (default 30e3 for 5G NR)
        num_tx: Number of transmit antennas (default 1)
        num_streams_per_tx: Number of spatial streams per TX (default 1)
        cyclic_prefix_length: Cyclic prefix length in samples (default 20)
        pilot_pattern: Pilot pattern type ("kronecker", "empty", default "kronecker")
        pilot_ofdm_symbol_indices: OFDM symbol indices for pilots (default [2, 11])
    """

    num_ofdm_symbols: int = 14
    fft_size: int = 128
    subcarrier_spacing: float = 30e3
    num_tx: int = 1
    num_streams_per_tx: int = 1
    cyclic_prefix_length: int = 20
    pilot_pattern: str = "kronecker"
    pilot_ofdm_symbol_indices: tuple = (2, 11)


@dataclass(frozen=True)
class MIMOConfig:
    """Configuration for MIMO antenna array.

    Creates a planar (rectangular) array of antenna elements.
    Total elements = num_rows * num_cols.

    Attributes:
        num_rows: Number of antenna rows (default 1)
        num_cols: Number of antenna columns (default 1)
        vertical_spacing: Vertical spacing in wavelengths (default 0.5)
        horizontal_spacing: Horizontal spacing in wavelengths (default 0.5)
        pattern: Antenna radiation pattern ("iso", "dipole", "tr38901")
                 Default "tr38901" for 3GPP compliance
        polarization: Antenna polarization ("V", "H", "VH", "cross")
                      Default "V" for vertical polarization
    """

    num_rows: int = 1
    num_cols: int = 1
    vertical_spacing: float = 0.5
    horizontal_spacing: float = 0.5
    pattern: str = "tr38901"
    polarization: str = "V"

    @property
    def num_elements(self) -> int:
        """Total number of antenna elements."""
        return self.num_rows * self.num_cols


def create_ofdm_resource_grid(
    config: Optional[OFDMConfig] = None,
) -> "ResourceGrid":
    """Create an OFDM resource grid with the specified configuration.

    The resource grid defines the time-frequency structure for OFDM
    modulation, including subcarrier allocation and pilot patterns.

    Args:
        config: OFDM configuration (uses 5G NR defaults if None)

    Returns:
        Configured Sionna ResourceGrid object

    Raises:
        ImportError: If sionna.phy is not installed
    """
    _check_sionna_phy()

    if config is None:
        config = OFDMConfig()

    rg = ResourceGrid(
        num_ofdm_symbols=config.num_ofdm_symbols,
        fft_size=config.fft_size,
        subcarrier_spacing=config.subcarrier_spacing,
        num_tx=config.num_tx,
        num_streams_per_tx=config.num_streams_per_tx,
        cyclic_prefix_length=config.cyclic_prefix_length,
        pilot_pattern=config.pilot_pattern,
        pilot_ofdm_symbol_indices=list(config.pilot_ofdm_symbol_indices),
    )

    return rg


def create_mimo_array(
    config: Optional[MIMOConfig] = None,
) -> "PlanarArray":
    """Create a MIMO planar antenna array with the specified configuration.

    Planar arrays are commonly used in massive MIMO systems for beamforming.
    The array geometry affects spatial resolution and capacity.

    Args:
        config: MIMO configuration (uses 1x1 defaults if None)

    Returns:
        Configured Sionna PlanarArray object

    Raises:
        ImportError: If sionna is not installed
    """
    _check_sionna()

    if config is None:
        config = MIMOConfig()

    array = PlanarArray(
        num_rows=config.num_rows,
        num_cols=config.num_cols,
        vertical_spacing=config.vertical_spacing,
        horizontal_spacing=config.horizontal_spacing,
        pattern=config.pattern,
        polarization=config.polarization,
    )

    return array


def apply_mimo_to_scene(
    scene: "rt.Scene",
    tx_config: MIMOConfig,
    rx_config: Optional[MIMOConfig] = None,
) -> None:
    """Apply MIMO antenna array configurations to a scene.

    Updates the scene's TX and RX arrays with the specified configurations.
    This replaces any simple arrays set during scene loading.

    Args:
        scene: Sionna RT Scene object
        tx_config: MIMO configuration for transmitter(s)
        rx_config: MIMO configuration for receiver(s). If None, uses tx_config.

    Raises:
        ImportError: If sionna is not installed
    """
    _check_sionna()

    if rx_config is None:
        rx_config = tx_config

    # Create and apply TX array
    scene.tx_array = PlanarArray(
        num_rows=tx_config.num_rows,
        num_cols=tx_config.num_cols,
        vertical_spacing=tx_config.vertical_spacing,
        horizontal_spacing=tx_config.horizontal_spacing,
        pattern=tx_config.pattern,
        polarization=tx_config.polarization,
    )

    # Create and apply RX array
    scene.rx_array = PlanarArray(
        num_rows=rx_config.num_rows,
        num_cols=rx_config.num_cols,
        vertical_spacing=rx_config.vertical_spacing,
        horizontal_spacing=rx_config.horizontal_spacing,
        pattern=rx_config.pattern,
        polarization=rx_config.polarization,
    )


def compute_paths(
    scene: "rt.Scene",
    max_depth: int = 5,
) -> "Paths":
    """Compute propagation paths between all TX/RX pairs in the scene.

    Uses GPU-accelerated ray tracing to find paths including:
    - Line-of-sight (LOS) paths
    - Reflected paths (up to max_depth reflections)
    - Diffracted paths

    The returned Paths object contains channel impulse response data:
    - tau: Path delays (propagation time)
    - a: Complex path gains (amplitude and phase)
    - theta_t, phi_t: Departure angles (elevation, azimuth)
    - theta_r, phi_r: Arrival angles (elevation, azimuth)

    Args:
        scene: Sionna RT Scene with configured TX/RX
        max_depth: Maximum number of ray bounces (default 5 for indoor)

    Returns:
        Paths object with channel impulse response data

    Raises:
        ImportError: If sionna is not installed
        ValueError: If scene has no transmitters or receivers
    """
    _check_sionna()

    # Validate scene has TX and RX
    if not scene.transmitters:
        raise ValueError("Scene has no transmitters. Use configure_transmitter() first.")
    if not scene.receivers:
        raise ValueError("Scene has no receivers. Use configure_receivers() first.")

    # Use PathSolver directly (Sionna 1.2.1+ API)
    solver = rt.PathSolver()
    paths = solver(scene, max_depth=max_depth)

    return paths
