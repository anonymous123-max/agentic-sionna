"""BER curve computation and codec comparison utilities.

Provides functions for computing BER vs SNR curves for channel codecs
and plotting comparison charts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from src.wireless.coding import LDPCCodec, PolarCodec

# Use module-level reference for late binding so mock.patch on
# src.wireless.coding.simulate_awgn_ber works correctly in tests
import src.wireless.coding as _coding_module


def compute_ber_curve(
    codec: Union[LDPCCodec, PolarCodec],
    snr_range: List[float],
    num_codewords: int = 1000,
) -> Dict[float, float]:
    """Compute BER vs SNR curve for a codec over AWGN channel.

    Args:
        codec: LDPC or Polar codec instance
        snr_range: List of SNR values in dB to simulate
        num_codewords: Number of codewords per SNR point

    Returns:
        Dictionary mapping SNR (dB) -> BER
    """
    results = {}

    for snr_db in snr_range:
        ber = _coding_module.simulate_awgn_ber(codec, snr_db, num_codewords)
        results[snr_db] = ber

    return results


def compare_codecs(
    codecs: Dict[str, Union[LDPCCodec, PolarCodec]],
    snr_range: List[float],
    num_codewords: int = 1000,
) -> Dict[str, Dict[float, float]]:
    """Compare BER performance of multiple codecs.

    Args:
        codecs: Dictionary mapping codec name -> codec instance
        snr_range: List of SNR values in dB to simulate
        num_codewords: Number of codewords per SNR point

    Returns:
        Nested dictionary: codec_name -> (SNR -> BER)

    Example:
        >>> ldpc = LDPCCodec()
        >>> polar = PolarCodec()
        >>> results = compare_codecs(
        ...     {'LDPC': ldpc, 'Polar': polar},
        ...     snr_range=[-2, 0, 2, 4],
        ...     num_codewords=500
        ... )
    """
    results = {}

    for name, codec in codecs.items():
        # Look up through coding module for mock.patch compatibility
        results[name] = _coding_module.compute_ber_curve(codec, snr_range, num_codewords)

    return results


def plot_ber_curves(
    results: Dict[str, Dict[float, float]],
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
) -> "Figure":
    """Plot BER curves for multiple codecs.

    Creates a semilogy plot with BER on y-axis and SNR on x-axis.

    Args:
        results: Nested dictionary from compare_codecs or compute_ber_curve
        output_path: Optional path to save the figure
        title: Optional plot title

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Define markers and colors for different codecs
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, (name, curve) in enumerate(results.items()):
        snrs = sorted(curve.keys())
        bers = [curve[s] for s in snrs]

        # Replace zeros with small value for log scale
        bers_plot = [max(b, 1e-10) for b in bers]

        ax.semilogy(
            snrs,
            bers_plot,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            label=name,
            linewidth=2,
            markersize=8,
        )

    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax.set_title(title or "BER vs SNR Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.set_ylim(1e-6, 1)

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
