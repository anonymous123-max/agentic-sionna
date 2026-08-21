"""Channel coding analysis with LDPC and Polar codes for BER curve generation.

Provides:
- LDPCCodec: 5G LDPC encoder/decoder wrapper
- PolarCodec: 5G Polar encoder/decoder with SCL decoding
- BER curve computation for AWGN channel
- Codec comparison utilities for performance analysis

Requires sionna>=1.2.1 for full functionality.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import tensorflow as tf
    from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
    from sionna.phy.fec.polar import Polar5GEncoder, Polar5GDecoder

    HAS_SIONNA = True
except ImportError:
    HAS_SIONNA = False
    tf = None
    LDPC5GEncoder = None
    LDPC5GDecoder = None
    Polar5GEncoder = None
    Polar5GDecoder = None


def _check_sionna() -> None:
    """Raise ImportError if sionna is not installed."""
    if not HAS_SIONNA:
        raise ImportError(
            "sionna is required for channel coding. "
            "Install with: pip install sionna>=1.2.1"
        )


@dataclass(frozen=True)
class CodecConfig:
    """Configuration for channel coding encoder/decoder.

    Attributes:
        k: Number of information bits (default 1024)
        n: Number of codeword bits (default 2048, implies rate ~1/2)
        num_iter: Number of decoder iterations (default 20 for LDPC, 1 for Polar)
    """

    k: int = 1024
    n: int = 2048
    num_iter: int = 20

    @property
    def rate(self) -> float:
        """Code rate = k/n."""
        return self.k / self.n


class LDPCCodec:
    """5G LDPC encoder/decoder wrapper.

    Wraps Sionna's LDPC5GEncoder and LDPC5GDecoder for easy BER analysis.

    Example:
        >>> codec = LDPCCodec()
        >>> print(f"Rate: {codec.rate:.2f}")
        Rate: 0.50
    """

    def __init__(self, config: Optional[CodecConfig] = None):
        """Initialize LDPC codec with given configuration.

        Args:
            config: Codec configuration. Uses defaults if None.
        """
        _check_sionna()

        self.config = config or CodecConfig()
        self.encoder = LDPC5GEncoder(k=self.config.k, n=self.config.n)
        self.decoder = LDPC5GDecoder(
            encoder=self.encoder, num_iter=self.config.num_iter
        )

    @property
    def k(self) -> int:
        """Number of information bits."""
        return self.config.k

    @property
    def n(self) -> int:
        """Number of codeword bits."""
        return self.config.n

    @property
    def rate(self) -> float:
        """Code rate = k/n."""
        return self.k / self.n

    def encode(self, bits: "tf.Tensor") -> "tf.Tensor":
        """Encode information bits to codewords.

        Args:
            bits: Information bits tensor of shape [batch, k]

        Returns:
            Codewords tensor of shape [batch, n]
        """
        return self.encoder(bits)

    def decode(self, llr: "tf.Tensor") -> "tf.Tensor":
        """Decode LLRs to information bits.

        Args:
            llr: Log-likelihood ratios tensor of shape [batch, n]

        Returns:
            Decoded bits tensor of shape [batch, k]
        """
        return self.decoder(llr)


class PolarCodec:
    """5G Polar encoder/decoder wrapper with SCL decoding.

    Wraps Sionna's Polar5GEncoder and Polar5GDecoder with SCL (Successive
    Cancellation List) decoding for improved accuracy.

    Example:
        >>> codec = PolarCodec()
        >>> print(f"Rate: {codec.rate:.2f}")
        Rate: 0.50
    """

    def __init__(self, config: Optional[CodecConfig] = None):
        """Initialize Polar codec with given configuration.

        Args:
            config: Codec configuration. Uses defaults if None.
                    Note: Polar uses num_iter=1 as list_size provides accuracy.
        """
        _check_sionna()

        # Polar defaults: smaller block size, list decoding instead of iterations
        self.config = config or CodecConfig(k=512, n=1024, num_iter=1)
        self.encoder = Polar5GEncoder(k=self.config.k, n=self.config.n)
        # SCL decoder with list_size=8 for good accuracy
        # Note: Sionna 1.2.1+ uses enc_polar parameter name
        self.decoder = Polar5GDecoder(
            enc_polar=self.encoder, dec_type="SCL", list_size=8
        )

    @property
    def k(self) -> int:
        """Number of information bits."""
        return self.config.k

    @property
    def n(self) -> int:
        """Number of codeword bits."""
        return self.config.n

    @property
    def rate(self) -> float:
        """Code rate = k/n."""
        return self.k / self.n

    def encode(self, bits: "tf.Tensor") -> "tf.Tensor":
        """Encode information bits to codewords.

        Args:
            bits: Information bits tensor of shape [batch, k]

        Returns:
            Codewords tensor of shape [batch, n]
        """
        return self.encoder(bits)

    def decode(self, llr: "tf.Tensor") -> "tf.Tensor":
        """Decode LLRs to information bits.

        Args:
            llr: Log-likelihood ratios tensor of shape [batch, n]

        Returns:
            Decoded bits tensor of shape [batch, k]
        """
        return self.decoder(llr)


def simulate_awgn_ber(
    codec: Union[LDPCCodec, PolarCodec],
    snr_db: float,
    num_codewords: int = 1000,
) -> float:
    """Simulate BER over AWGN channel at given SNR.

    Performs:
    1. Generate random information bits
    2. Encode to codewords
    3. BPSK modulation (0->-1, 1->+1)
    4. Add AWGN noise based on SNR
    5. Compute LLRs
    6. Decode
    7. Compute BER

    Args:
        codec: LDPC or Polar codec instance
        snr_db: Signal-to-noise ratio in dB
        num_codewords: Number of codewords to simulate (batch size)

    Returns:
        Bit error rate (BER) as float between 0 and 1
    """
    _check_sionna()

    k = codec.k
    n = codec.n

    # Generate random information bits [num_codewords, k]
    # Sionna FEC expects float32 bits (0.0 or 1.0)
    bits = tf.cast(tf.random.uniform([num_codewords, k], 0, 2, dtype=tf.int32), tf.float32)

    # Encode to codewords [num_codewords, n]
    codewords = codec.encode(bits)

    # BPSK modulation: 0 -> -1, 1 -> +1
    x = 2.0 * tf.cast(codewords, tf.float32) - 1.0

    # AWGN channel
    # SNR = signal_power / noise_power
    # For BPSK with unit power: noise_var = 1/snr_linear
    snr_linear = 10 ** (snr_db / 10.0)
    noise_var = 1.0 / snr_linear
    noise_std = tf.sqrt(noise_var)
    noise = tf.random.normal(tf.shape(x), stddev=noise_std)
    y = x + noise

    # Compute LLRs: llr = 2*y/noise_var for BPSK
    llr = 2.0 * y / noise_var

    # Decode
    decoded = codec.decode(llr)

    # Handle soft output if needed (convert to hard decision)
    if decoded.dtype == tf.float32:
        decoded = tf.cast(decoded > 0.5, tf.float32)
    else:
        decoded = tf.cast(decoded, tf.float32)

    # Compute BER (both bits and decoded should be float32 now)
    errors = tf.reduce_sum(tf.cast(tf.not_equal(bits, decoded), tf.float32))
    total_bits = num_codewords * k
    ber = float(errors) / float(total_bits)

    return ber


# Import from extracted module for backward compatibility
from src.wireless.ber_analysis import (  # noqa: F401
    compute_ber_curve,
    compare_codecs,
    plot_ber_curves,
)
