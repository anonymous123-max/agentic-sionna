"""Wireless simulation module using Sionna RT.

This module provides wireless signal propagation simulation including:
- Scene loading from exported Sionna XML files
- Transmitter/receiver configuration with antenna arrays
- Coverage map computation using GPU-accelerated ray tracing
- OFDM resource grid configuration for link-level simulation
- MIMO antenna array configuration for spatial multiplexing
- Path computation for channel impulse response
- Channel coding analysis with LDPC and Polar codes

Requires sionna>=1.2.1 for full functionality.
"""

__version__ = "0.1.0"

from src.wireless.scene import (
    SceneConfig,
    load_scene,
    configure_transmitter,
    configure_receivers,
    load_furniture_into_scene,
)
from src.wireless.ray_tracing import (
    CoverageConfig,
    compute_coverage_map,
    visualize_coverage,
    get_coverage_at_point,
)
from src.wireless.thz_model import (
    THzConfig,
    compute_thz_coverage,
)
from src.wireless.channel import (
    OFDMConfig,
    MIMOConfig,
    create_ofdm_resource_grid,
    create_mimo_array,
    apply_mimo_to_scene,
    compute_paths,
)
from src.wireless.coding import (
    CodecConfig,
    LDPCCodec,
    PolarCodec,
    simulate_awgn_ber,
)
from src.wireless.ber_analysis import (
    compute_ber_curve,
    compare_codecs,
    plot_ber_curves,
)
from src.wireless.optimization import (
    OptimizationConfig,
    OptimizationResult,
    optimize_tx_orientation,
)
from src.wireless.tx_placement import (
    optimize_tx_position,
    optimize_for_target_coverage,
)
from src.wireless.gpu import (
    setup_rtx5090_environment,
    check_cuda_version,
    verify_gpu_capability,
    get_gpu_info,
)

__all__ = [
    # Scene configuration
    "SceneConfig",
    "load_scene",
    "configure_transmitter",
    "configure_receivers",
    "load_furniture_into_scene",
    # Coverage map
    "CoverageConfig",
    "compute_coverage_map",
    "visualize_coverage",
    "get_coverage_at_point",
    # THz coverage (>100 GHz)
    "THzConfig",
    "compute_thz_coverage",
    # OFDM/MIMO configuration
    "OFDMConfig",
    "MIMOConfig",
    "create_ofdm_resource_grid",
    "create_mimo_array",
    "apply_mimo_to_scene",
    "compute_paths",
    # Channel coding
    "CodecConfig",
    "LDPCCodec",
    "PolarCodec",
    "simulate_awgn_ber",
    "compute_ber_curve",
    "compare_codecs",
    "plot_ber_curves",
    # Optimization
    "OptimizationConfig",
    "OptimizationResult",
    "optimize_tx_orientation",
    "optimize_tx_position",
    "optimize_for_target_coverage",
    # GPU compatibility
    "setup_rtx5090_environment",
    "check_cuda_version",
    "verify_gpu_capability",
    "get_gpu_info",
]
