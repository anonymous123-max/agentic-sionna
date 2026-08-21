"""Pre-compute N4 PHY reference oracles.

For each of the 4 N4 tasks we produce a reference curve (Eb/N0_dB → metric):
  1. ber_qpsk_awgn:         analytical  0.5 * erfc(sqrt(Eb/N0_lin))
  2. ber_16qam_awgn:        approximate analytical for square 16-QAM Gray-coded
  3. bler_ldpc_qpsk_awgn:   empirical Monte Carlo via Sionna 2.0 PHY chain
  4. throughput_ldpc_qpsk:  derived: code_rate * log2(M) * (1 - BLER)

The first two are analytical (verifier can recompute them on demand), but we
also save .npy files so the verifier path is uniform across all 4. The two
LDPC references require a Sionna MC run, so we pre-compute and freeze them.

Output: benchmark/oracles/n4_phy/{task}.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "oracles" / "n4_phy"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ber_qpsk_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    """Analytical uncoded QPSK BER over AWGN."""
    from scipy.special import erfc
    ebn0_lin = 10 ** (np.asarray(ebn0_db) / 10.0)
    return 0.5 * erfc(np.sqrt(ebn0_lin))


def ber_mqam_awgn(ebn0_db: np.ndarray, M: int) -> np.ndarray:
    """Approximate uncoded square M-QAM Gray-coded BER over AWGN.

    BER ≈ (4/log2(M)) * (1 - 1/sqrt(M)) * Q(sqrt(3 log2(M) Eb/N0 / (M-1)))
        ≈ (2/log2(M)) * (1 - 1/sqrt(M)) * erfc(sqrt(...) / sqrt(2))
    """
    from scipy.special import erfc
    ebn0_lin = 10 ** (np.asarray(ebn0_db) / 10.0)
    log2M = np.log2(M)
    arg = np.sqrt(3 * log2M * ebn0_lin / (M - 1))
    # Q(x) = 0.5 erfc(x/sqrt(2))
    Q = 0.5 * erfc(arg / np.sqrt(2))
    return (4 / log2M) * (1 - 1 / np.sqrt(M)) * Q


def run_sionna_ldpc_qpsk_bler(ebn0_db: np.ndarray,
                              k: int = 1024,
                              n: int = 2048,
                              batch_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Empirical LDPC QPSK BLER + throughput via Sionna 2.0 PHY MC.

    Returns (bler_array, throughput_array) over the same Eb/N0 grid.
    """
    import torch
    from sionna.phy.fec import LDPC5GEncoder, LDPC5GDecoder
    from sionna.phy.mapping import Mapper, Demapper
    from sionna.phy.channel import AWGN
    from sionna.phy.utils import ebnodb2no

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bps = 2  # QPSK
    code_rate = k / n

    encoder = LDPC5GEncoder(k=k, n=n)
    decoder = LDPC5GDecoder(encoder=encoder, num_iter=20)
    mapper = Mapper(constellation_type="qam", num_bits_per_symbol=bps)
    demapper = Demapper(demapping_method="app", constellation_type="qam",
                        num_bits_per_symbol=bps)
    channel = AWGN()

    bler = []
    tput = []
    for snr in ebn0_db:
        no = ebnodb2no(float(snr), num_bits_per_symbol=bps, coderate=code_rate)
        bits = torch.randint(0, 2, (batch_size, 1, 1, k),
                             dtype=torch.float32, device=device)
        coded = encoder(bits)
        symbols = mapper(coded)
        noisy = channel(symbols, no)
        llr = demapper(noisy, no)
        decoded = decoder(llr)
        bit_err = torch.ne(bits, decoded)
        codeword_fail = torch.any(bit_err.flatten(start_dim=-1), dim=-1).float().mean()
        bler_val = float(codeword_fail)
        bler_val = max(bler_val, 1e-7)
        bler.append(bler_val)
        tput.append(code_rate * bps * (1.0 - bler_val))
    return np.array(bler), np.array(tput)


def main() -> int:
    import sionna
    print(f"sionna {sionna.__version__}  out: {OUT_DIR}")

    # Task 1: BER QPSK AWGN, Eb/N0 = [0, 2, 4, 6, 8]
    ebn0_1 = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    ber_1 = ber_qpsk_awgn(ebn0_1)
    (OUT_DIR / "ber_qpsk_awgn.json").write_text(json.dumps({
        "task_id":         "N4_ber_qpsk_awgn",
        "metric_type":     "ber",
        "modulation":      "QPSK",
        "codec":           "uncoded",
        "channel":         "AWGN",
        "eb_n0_db":        ebn0_1.tolist(),
        "metric_values":   ber_1.tolist(),
        "source":          "analytical 0.5*erfc(sqrt(Eb/N0))",
    }, indent=2))
    print(f"  BER QPSK   {dict(zip(ebn0_1, np.round(ber_1, 6).tolist()))}")

    # Task 2: BER 16-QAM AWGN, Eb/N0 = [4, 6, 8, 10, 12]
    ebn0_2 = np.array([4.0, 6.0, 8.0, 10.0, 12.0])
    ber_2 = ber_mqam_awgn(ebn0_2, M=16)
    (OUT_DIR / "ber_16qam_awgn.json").write_text(json.dumps({
        "task_id":         "N4_ber_16qam_awgn",
        "metric_type":     "ber",
        "modulation":      "16-QAM",
        "codec":           "uncoded",
        "channel":         "AWGN",
        "eb_n0_db":        ebn0_2.tolist(),
        "metric_values":   ber_2.tolist(),
        "source":          "analytical Gray-coded square 16-QAM",
    }, indent=2))
    print(f"  BER 16-QAM {dict(zip(ebn0_2, np.round(ber_2, 6).tolist()))}")

    # Task 3: BLER LDPC QPSK AWGN, Eb/N0 = [0.5, 1.0, 1.5, 2.0, 2.5]
    ebn0_3 = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    t0 = time.time()
    bler_3, tput_3 = run_sionna_ldpc_qpsk_bler(ebn0_3, k=1024, n=2048,
                                                batch_size=5000)  # 5k for stable
    print(f"  BLER LDPC took {time.time()-t0:.1f}s")
    (OUT_DIR / "bler_ldpc_qpsk_awgn.json").write_text(json.dumps({
        "task_id":         "N4_bler_ldpc_qpsk_awgn",
        "metric_type":     "bler",
        "modulation":      "QPSK",
        "codec":           "ldpc",
        "code_rate":       0.5,
        "channel":         "AWGN",
        "eb_n0_db":        ebn0_3.tolist(),
        "metric_values":   bler_3.tolist(),
        "source":          "Sionna 2.0 LDPC5GEncoder/Decoder, k=1024, n=2048, batch=5000 codewords",
    }, indent=2))
    print(f"  BLER LDPC  {dict(zip(ebn0_3, np.round(bler_3, 6).tolist()))}")

    # Task 4: Throughput LDPC QPSK AWGN, same Eb/N0 — derive from same BLER above
    (OUT_DIR / "throughput_ldpc_qpsk_awgn.json").write_text(json.dumps({
        "task_id":         "N4_throughput_ldpc_qpsk_awgn",
        "metric_type":     "throughput",
        "modulation":      "QPSK",
        "codec":           "ldpc",
        "code_rate":       0.5,
        "channel":         "AWGN",
        "eb_n0_db":        ebn0_3.tolist(),
        "metric_values":   tput_3.tolist(),
        "source":          "derived from BLER above: (k/n) * log2(M) * (1 - BLER)",
    }, indent=2))
    print(f"  Tput LDPC  {dict(zip(ebn0_3, np.round(tput_3, 4).tolist()))}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
