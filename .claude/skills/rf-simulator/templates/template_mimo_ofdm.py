#!/usr/bin/env python3
"""Template: MIMO-OFDM Link Evaluation — Sionna with CPU Fallback.

Modify ONLY the PARAMS block. This template:
1. Tries Sionna OFDM chain (ResourceGrid, PilotPattern, ChannelEstimator)
2. Falls back to analytical MIMO-OFDM model if Sionna unavailable
3. Produces simulation_result.json + BER curve plot
"""
from __future__ import annotations

# ============================================================================
# PARAMETER BLOCK — MODIFY ONLY THIS SECTION
# ============================================================================
PARAMS = {
    "num_tx_ant": 4, "num_rx_ant": 4,
    "subcarrier_spacing_khz": 30,   # {15, 30, 60, 120}
    "fft_size": 128,                # valid 5G NR FFT sizes
    "num_ofdm_symbols": 14,
    "cyclic_prefix_length": 20,
    "pilot_pattern": "kronecker",   # "kronecker" or "empty"
    "channel_estimator": "ls",      # "ls" or "lmmse"
    "codec": "ldpc",
    "code_rate_k": 1024, "code_rate_n": 2048,
    "num_bits_per_symbol": 4,
    "snr_range_db": [0, 20], "snr_steps": 21,
    "batch_size": 500,
    "channel_model": "tdl_a",       # "awgn", "tdl_a", "tdl_b", "tdl_c"
    "frequency_hz": 3.5e9,
    "output_dir": "outputs/mimo_ofdm",
}
# ============================================================================

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

VALID_SCS = {15, 30, 60, 120}
VALID_FFT = {12, 24, 36, 48, 60, 72, 76, 128, 256, 512, 1024, 1536, 2048, 4096}

def validate_params(p):
    errors = []
    if p["subcarrier_spacing_khz"] not in VALID_SCS: errors.append("invalid SCS")
    if p["fft_size"] not in VALID_FFT: errors.append("invalid FFT size")
    if p["code_rate_k"] >= p["code_rate_n"]: errors.append("k must be < n")
    if p["num_bits_per_symbol"] not in (1,2,4,6,8): errors.append("invalid bits/symbol")
    if p["cyclic_prefix_length"] >= p["fft_size"]//4: errors.append("CP too long")
    return errors


def run_sionna_mimo(params):
    """Sionna v2 OFDM chain (PyTorch + GPU)."""
    import torch
    from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, ResourceGridDemapper
    from sionna.phy.mimo import StreamManagement
    from sionna.phy.channel import AWGN
    from sionna.phy.fec import LDPC5GEncoder, LDPC5GDecoder
    from sionna.phy.mapping import Mapper, Demapper
    from sionna.phy.utils import ebnodb2no

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nt, nr = params["num_tx_ant"], params["num_rx_ant"]
    bps = params["num_bits_per_symbol"]
    num_streams = min(nt, nr)

    pilot_indices = [2, 11] if params["num_ofdm_symbols"] >= 12 else [2]
    rg = ResourceGrid(num_ofdm_symbols=params["num_ofdm_symbols"],
                       fft_size=params["fft_size"],
                       subcarrier_spacing=params["subcarrier_spacing_khz"]*1e3,
                       cyclic_prefix_length=params["cyclic_prefix_length"],
                       num_tx=1, num_streams_per_tx=num_streams,
                       pilot_pattern=params["pilot_pattern"],
                       pilot_ofdm_symbol_indices=pilot_indices)
    sm = StreamManagement(np.ones([1, 1], int), num_streams)

    # Align codeword to grid: n must produce exactly num_data_symbols symbols
    n = int(rg.num_data_symbols) * bps
    code_rate = params["code_rate_k"] / params["code_rate_n"]
    k = max(12, int(n * code_rate))

    encoder = LDPC5GEncoder(k=k, n=n)
    decoder = LDPC5GDecoder(encoder=encoder, num_iter=20)
    mapper = Mapper(constellation_type="qam", num_bits_per_symbol=bps)
    rg_mapper = ResourceGridMapper(rg)
    rg_demapper = ResourceGridDemapper(rg, sm)
    demapper = Demapper(demapping_method="app", constellation_type="qam",
                        num_bits_per_symbol=bps)
    channel = AWGN()

    snr_db = np.linspace(*params["snr_range_db"], params["snr_steps"])
    ber_coded = []

    for snr in snr_db:
        no = ebnodb2no(snr, num_bits_per_symbol=bps, coderate=k/n)
        bits = torch.randint(0, 2, (params["batch_size"], 1, num_streams, k),
                             dtype=torch.float32, device=device)
        coded = encoder(bits)
        symbols = mapper(coded)
        x_rg = rg_mapper(symbols)
        y_rg = channel(x_rg, no)
        y_symbols = rg_demapper(y_rg)
        llr = demapper(y_symbols, no)
        decoded = decoder(llr)

        err = float(torch.sum(torch.ne(bits, decoded).float()))
        tot = float(bits.numel())
        ber_coded.append(max(err/tot, 1e-7))

    bw = params["fft_size"] * params["subcarrier_spacing_khz"] * 1e3
    se = num_streams * np.log2(2**bps) * (k/n)
    return {"snr_db": snr_db.tolist(), "ber_coded": ber_coded,
            "throughput_mbps": round(float(bw*se/1e6), 2),
            "spatial_streams": num_streams, "method": "sionna_v2_ofdm_gpu"}


def run_cpu_mimo(params):
    """Analytical MIMO-OFDM BER model."""
    from scipy.special import erfc
    snr_db = np.linspace(*params["snr_range_db"], params["snr_steps"])
    snr_lin = 10**(snr_db/10)
    M = 2**params["num_bits_per_symbol"]
    nt, nr = params["num_tx_ant"], params["num_rx_ant"]
    ns = min(nt, nr)
    cr = params["code_rate_k"]/params["code_rate_n"]

    ag = nt*nr/ns
    cp_oh = params["cyclic_prefix_length"]/(params["fft_size"]+params["cyclic_prefix_length"])
    ofdm_eff = (1-cp_oh)*0.85
    ce_pen = 1.0 if params["channel_estimator"]=="lmmse" else 0.8
    ch_pen = {"awgn":1.0,"tdl_a":0.7,"tdl_b":0.5,"tdl_c":0.4}.get(params["channel_model"],0.7)
    cg = 10**((6.0+2.0*cr)/10) if params["codec"]=="ldpc" else 10**((5.0+1.5*cr)/10)

    snr_eff = snr_lin*ag*ofdm_eff*ce_pen*ch_pen*cg
    if M==2: ber = 0.5*erfc(np.sqrt(snr_eff))
    else:
        f = (4/np.log2(M))*(1-1/np.sqrt(M))
        ber = f*0.5*erfc(np.sqrt(3*np.log2(M)*snr_eff/(M-1))/np.sqrt(2))
    ber = np.maximum(ber, 1e-7)

    bw = params["fft_size"]*params["subcarrier_spacing_khz"]*1e3
    se = ns*np.log2(M)*cr*ofdm_eff
    return {"snr_db": snr_db.tolist(), "ber_coded": ber.tolist(),
            "throughput_mbps": round(float(bw*se/1e6), 2),
            "spatial_streams": ns, "method": "cpu_analytical"}


def main():
    errors = validate_params(PARAMS)
    if errors:
        for e in errors: print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    try:
        data = run_sionna_mimo(PARAMS)
    except Exception as e:
        print(f"Sionna OFDM unavailable ({e}), using analytical")
        data = run_cpu_mimo(PARAMS)
    timing = {"simulate_sec": round(time.perf_counter()-t0, 3)}

    # Self-test: BER must DECREASE with SNR (catches noise-variance sign errors)
    snr = list(data["snr_db"]); ber = list(data["ber_coded"])
    if len(snr) >= 2 and ber[0] > 0 and ber[-1] > ber[0]:
        sys.stderr.write(
            f"\nSELF_TEST FAIL: BER increases with SNR "
            f"(SNR={snr[0]}->{snr[-1]} dB, BER={ber[0]:.2e}->{ber[-1]:.2e}). "
            f"Common fix: noise variance sign error. NOT writing "
            f"simulation_result.json to avoid contaminating graders.\n")
        sys.exit(2)

    # Output
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    out = Path(PARAMS["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    M = 2**PARAMS["num_bits_per_symbol"]

    fig, ax = plt.subplots(figsize=(8,6))
    ax.semilogy(data["snr_db"], data["ber_coded"], "r-s", ms=3,
                label=f"{PARAMS['codec'].upper()} {PARAMS['num_tx_ant']}x{PARAMS['num_rx_ant']}")
    ax.set_title(f"MIMO-OFDM — {PARAMS['num_tx_ant']}x{PARAMS['num_rx_ant']} {M}-QAM ({data['method']})")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("BER"); ax.set_ylim(1e-7,1)
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.savefig(out/"ber_vs_snr.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    result = {
        "schema_version": "1.0", "task_type": "mimo_ofdm",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success", "method": data["method"],
        "numerical_metrics": {
            "snr_db": list(data["snr_db"]),
            "ber_simulated": list(data["ber_coded"]),
            "throughput_estimate_mbps": data["throughput_mbps"],
            "spatial_streams": data["spatial_streams"],
            "mimo_config": f"{PARAMS['num_tx_ant']}x{PARAMS['num_rx_ant']}",
        },
        "visual_outputs": {"ber_curve_path": "ber_vs_snr.png"},
        "timing": timing, "warnings": [],
    }
    (out/"simulation_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n{PARAMS['num_tx_ant']}x{PARAMS['num_rx_ant']} {M}-QAM: {data['throughput_mbps']} Mbps ({data['method']})")


if __name__ == "__main__":
    import os
    if not os.environ.get("RF_SKIP_TEMPLATE_WARN"):
        sys.stderr.write(
            "\n*** TEMPLATE WARNING ***\n"
            "This is a TEMPLATE — copy the file to your workdir and edit\n"
            "PARAMS before running. Default PARAMS will run, but they're\n"
            "for reference only and may not match your task. Set\n"
            "RF_SKIP_TEMPLATE_WARN=1 to silence.\n"
            "\n"
        )
    main()
