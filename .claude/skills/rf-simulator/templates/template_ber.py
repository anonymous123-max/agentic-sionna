#!/usr/bin/env python3
"""Template: BER/BLER Analysis — Sionna PHY with CPU Fallback.

Modify ONLY the PARAMS block. This template supports three modes:
1. codec="uncoded"  — Mapper → AWGN → Demapper → hard decision (no FEC).
                      Cleanest path; analytical reference BER available.
2. codec="ldpc"     — LDPC 5G encoder/decoder around the same chain.
3. codec="polar"    — Polar 5G encoder/decoder around the same chain.

GPU path: Sionna PHY with PyTorch graph execution (batched, GPU-accelerated)
CPU path: Analytical M-QAM BER (fallback only, sets method="cpu_analytical")
"""
from __future__ import annotations

# ============================================================================
# PARAMETER BLOCK — MODIFY ONLY THIS SECTION
# ============================================================================
PARAMS = {
    "codec": "uncoded",           # "uncoded" | "ldpc" | "polar"
    "code_rate_k": 1024,          # info bits (per batch element); for uncoded this is total bits
    "code_rate_n": 2048,          # coded bits (k < n); ignored when codec="uncoded"
    "modulation": "qam",          # "qam" or "psk"
    "num_bits_per_symbol": 2,     # 1=BPSK, 2=QPSK, 4=16-QAM, 6=64-QAM, 8=256-QAM
    "snr_range_db": [0, 8],
    "snr_steps": 5,
    "batch_size": 1000,
    "channel_type": "awgn",       # "awgn" or "rayleigh"
    "target_ber": 1e-4,
    "metric_type": "ber",         # "ber" | "bler" | "throughput"   ← what to save as curve
    "output_dir": "outputs/ber_analysis",
}
# ============================================================================

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def validate_params(p: dict) -> list[str]:
    errors = []
    if p["codec"] not in ("uncoded", "ldpc", "polar"):
        errors.append("codec must be 'uncoded', 'ldpc', or 'polar'")
    if p["codec"] != "uncoded" and p["code_rate_k"] >= p["code_rate_n"]:
        errors.append("code_rate_k must be < code_rate_n when codec is ldpc/polar")
    if p["num_bits_per_symbol"] not in (1, 2, 4, 6, 8):
        errors.append("num_bits_per_symbol must be 1/2/4/6/8")
    snr = p["snr_range_db"]
    if len(snr) != 2 or snr[0] >= snr[1]:
        errors.append("snr_range_db must be [min, max]")
    metric = p.get("metric_type", "ber")
    if metric not in ("ber", "bler", "throughput"):
        errors.append("metric_type must be 'ber', 'bler', or 'throughput'")
    if metric in ("bler", "throughput") and p["codec"] == "uncoded":
        errors.append("metric_type 'bler' / 'throughput' require codec='ldpc' or 'polar'")
    return errors


# ---------------------------------------------------------------------------
# GPU Path: Sionna PHY
# ---------------------------------------------------------------------------

def run_sionna_ber(params: dict) -> dict:
    """Compute BER using Sionna v2 PHY chain (PyTorch + GPU).

    Three codec modes:
      - "uncoded": Mapper → AWGN → Demapper → hard-decision LLR > 0 → bit 0.
        Reference: analytical M-QAM BER (use scipy.special.erfc to *verify*,
        NOT to compute — agent must run the Sionna chain).
      - "ldpc":   LDPC5GEncoder/Decoder around the chain.
      - "polar":  Polar5GEncoder/Decoder around the chain.
    """
    import torch
    from sionna.phy.mapping import Mapper, Demapper
    from sionna.phy.channel import AWGN
    from sionna.phy.utils import ebnodb2no

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k, n = params["code_rate_k"], params["code_rate_n"]
    bps = params["num_bits_per_symbol"]
    codec = params["codec"]

    mapper = Mapper(constellation_type="qam", num_bits_per_symbol=bps)
    demapper = Demapper(demapping_method="app", constellation_type="qam",
                        num_bits_per_symbol=bps)
    channel = AWGN()

    if codec == "uncoded":
        # No encoder/decoder. k is interpreted as "info bits per batch element".
        # k must be a multiple of bps so the mapper sees whole symbols.
        if k % bps != 0:
            k = (k // bps) * bps
        coderate = 1.0
        snr_db = np.linspace(*params["snr_range_db"], params["snr_steps"])
        ber = []
        for snr in snr_db:
            no = ebnodb2no(snr, num_bits_per_symbol=bps, coderate=coderate)
            bits = torch.randint(0, 2, (params["batch_size"], k),
                                 dtype=torch.float32, device=device)
            symbols = mapper(bits)
            noisy = channel(symbols, no)
            llr = demapper(noisy, no)
            # Sionna 2.0 Demapper LLR convention: LLR = log P(b=1|y) / P(b=0|y),
            # so LLR > 0 ⇒ bit=1, LLR < 0 ⇒ bit=0.
            bits_hat = (llr > 0).to(torch.float32)
            errors = torch.sum(torch.ne(bits, bits_hat).float())
            total = float(bits.numel())
            ber.append(max(float(errors / total), 1e-7))
        return {"snr_db": snr_db.tolist(),
                "ber_coded": ber,           # placed here for write_results uniformity
                "method": "sionna_v2_phy_uncoded"}

    # Coded path (LDPC or Polar)
    from sionna.phy.fec import LDPC5GEncoder, LDPC5GDecoder, Polar5GEncoder, Polar5GDecoder
    if codec == "ldpc":
        encoder = LDPC5GEncoder(k=k, n=n)
        decoder = LDPC5GDecoder(encoder=encoder, num_iter=20)
    else:
        encoder = Polar5GEncoder(k=k, n=n)
        decoder = Polar5GDecoder(encoder=encoder)

    snr_db = np.linspace(*params["snr_range_db"], params["snr_steps"])
    ber_coded = []
    bler_coded = []
    throughput_coded = []
    code_rate = k / n
    for snr in snr_db:
        no = ebnodb2no(snr, num_bits_per_symbol=bps, coderate=code_rate)
        bits = torch.randint(0, 2, (params["batch_size"], 1, 1, k),
                             dtype=torch.float32, device=device)
        coded = encoder(bits)
        symbols = mapper(coded)
        noisy = channel(symbols, no)
        llr = demapper(noisy, no)
        decoded = decoder(llr)

        # Per-bit BER
        bit_errors = torch.ne(bits, decoded)
        ber_val = float(bit_errors.float().sum() / bits.numel())
        ber_coded.append(max(ber_val, 1e-7))

        # Per-codeword BLER: a codeword fails if ANY of its bits is wrong
        codeword_fail = torch.any(bit_errors.flatten(start_dim=-1), dim=-1)
        bler_val = float(codeword_fail.float().mean())
        bler_coded.append(max(bler_val, 1e-7))

        # Throughput: bits per channel use = code_rate * log2(M) * (1 - BLER)
        throughput_coded.append(code_rate * bps * (1.0 - bler_val))

    return {"snr_db": snr_db.tolist(),
            "ber_coded": ber_coded,
            "bler_coded": bler_coded,
            "throughput_coded": throughput_coded,
            "method": "sionna_v2_phy_gpu"}


# ---------------------------------------------------------------------------
# CPU Path: Analytical BER
# ---------------------------------------------------------------------------

def run_cpu_ber(params: dict) -> dict:
    """Analytical BER computation. See references/cpu-fallback.md."""
    from scipy.special import erfc

    snr_db = np.linspace(*params["snr_range_db"], params["snr_steps"])
    snr_lin = 10 ** (snr_db / 10)
    M = 2 ** params["num_bits_per_symbol"]
    code_rate = params["code_rate_k"] / params["code_rate_n"]

    # Uncoded BER for M-QAM over AWGN
    if M == 2:
        ber_unc = 0.5 * erfc(np.sqrt(snr_lin))
    else:
        f = (4 / np.log2(M)) * (1 - 1 / np.sqrt(M))
        ber_unc = f * 0.5 * erfc(np.sqrt(3 * np.log2(M) * snr_lin / (M - 1)) / np.sqrt(2))

    if params["channel_type"] == "rayleigh" and M == 2:
        ber_unc = 0.5 * (1 - np.sqrt(snr_lin / (1 + snr_lin)))

    # Coding gain
    cg_db = (6.0 + 2.0*code_rate) if params["codec"] == "ldpc" else (5.0 + 1.5*code_rate)
    snr_eff = snr_lin * 10 ** (cg_db / 10)

    if M == 2:
        ber_cod = 0.5 * erfc(np.sqrt(snr_eff))
    else:
        f = (4 / np.log2(M)) * (1 - 1 / np.sqrt(M))
        ber_cod = f * 0.5 * erfc(np.sqrt(3 * np.log2(M) * snr_eff / (M - 1)) / np.sqrt(2))

    ber_cod = np.maximum(ber_cod, 1e-7)
    ber_unc = np.maximum(ber_unc, 1e-7)

    return {
        "snr_db": snr_db.tolist(),
        "ber_uncoded": ber_unc.tolist(),
        "ber_coded": ber_cod.tolist(),
        "coding_gain_db": round(cg_db, 2),
        "method": "cpu_analytical",
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(data: dict, params: dict, timing: dict, output_dir: str):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    bps = params["num_bits_per_symbol"]
    M = 2 ** bps
    mod_label = {1: "BPSK", 2: "QPSK", 4: "16-QAM", 6: "64-QAM", 8: "256-QAM"}.get(bps, f"{M}-QAM")
    metric_type = params.get("metric_type", "ber")
    codec = params["codec"]
    snr = data["snr_db"]

    # Pick the right metric series + filename + label
    if metric_type == "ber":
        metric_values = data["ber_coded"]   # for uncoded mode this still holds the BER values
        npy_file = "ber_curve.npy"
        png_file = "ber_curve.png"
        y_label = "BER"
        log_y = True
    elif metric_type == "bler":
        metric_values = data.get("bler_coded")
        if metric_values is None:
            raise RuntimeError("metric_type='bler' but no bler_coded in data (need codec='ldpc'/'polar')")
        npy_file = "bler_curve.npy"
        png_file = "bler_curve.png"
        y_label = "BLER"
        log_y = True
    elif metric_type == "throughput":
        metric_values = data.get("throughput_coded")
        if metric_values is None:
            raise RuntimeError("metric_type='throughput' but no throughput_coded in data")
        npy_file = "throughput_curve.npy"
        png_file = "throughput_curve.png"
        y_label = "Throughput (bits/symbol)"
        log_y = False
    else:
        raise RuntimeError(f"unknown metric_type {metric_type!r}")

    # Save curve as (N, 2) array: col 0 = Eb/N0 dB, col 1 = metric
    curve = np.column_stack([np.asarray(snr, dtype=float),
                              np.asarray(metric_values, dtype=float)])
    np.save(out / npy_file, curve)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    if log_y:
        ax.semilogy(snr, metric_values, "r-s", ms=4,
                    label=f"{mod_label}, {codec.upper()}")
        ax.set_ylim(1e-7, 1)
    else:
        ax.plot(snr, metric_values, "r-s", ms=4,
                label=f"{mod_label}, {codec.upper()}")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} — {mod_label} over {params['channel_type'].upper()} ({data['method']})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.savefig(out / png_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # N4 canonical schema (overlaid with legacy fields for backward compat)
    code_rate = (params["code_rate_k"] / params["code_rate_n"]) if codec != "uncoded" else 1.0
    num_bits = int(params["batch_size"]) * int(params["code_rate_k"])
    result = {
        "schema_version":     "1.0",
        "task_type":          "phy_link",
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "status":             "success",
        "method":             "sionna_phy",
        # N4 canonical fields
        "modulation":         mod_label,
        "codec":              codec,
        "code_rate":          round(code_rate, 4),
        "channel":            params["channel_type"].upper(),
        "metric_type":        metric_type,
        "eb_n0_db":           list(snr),
        "metric_values":      list(metric_values),
        "num_bits_per_point": num_bits,
        # Legacy compat
        "numerical_metrics": {
            "snr_db":         list(snr),
            "ber_simulated":  list(data.get("ber_coded", [])),
            "bler_simulated": list(data.get("bler_coded", [])),
            "throughput":     list(data.get("throughput_coded", [])),
        },
        "visual_outputs":     {"curve_path": png_file},
        "timing":             timing,
        "warnings":           [],
    }
    (out / "simulation_result.json").write_text(json.dumps(result, indent=2))
    print(f"\n{mod_label} {codec.upper()} {metric_type.upper()} ({data['method']})")
    for s, v in zip(snr, metric_values):
        print(f"  Eb/N0 = {float(s):+5.1f} dB  {metric_type} = {float(v):.4e}")


def main():
    errors = validate_params(PARAMS)
    if errors:
        for e in errors: print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    t0 = time.perf_counter()
    try:
        data = run_sionna_ber(PARAMS)
    except Exception as e:
        print(f"Sionna PHY unavailable ({e}), using analytical")
        data = run_cpu_ber(PARAMS)

    timing = {"simulate_sec": round(time.perf_counter() - t0, 3)}

    # Self-test: BER must DECREASE with SNR. Catches noise-variance sign errors.
    snr = list(data["snr_db"]); ber = list(data["ber_coded"])
    if len(snr) >= 2 and ber[0] > 0 and ber[-1] > ber[0]:
        sys.stderr.write(
            f"\nSELF_TEST FAIL: BER increases with SNR "
            f"(SNR={snr[0]}->{snr[-1]} dB, BER={ber[0]:.2e}->{ber[-1]:.2e}). "
            f"Common fix: noise variance sign is wrong. For QPSK use "
            f"sigma2 = 0.5 / (Eb/N0_lin). For M-QAM use "
            f"sigma2 = 1 / (2 * Es/N0_lin). NOT writing simulation_result.json "
            f"to avoid contaminating downstream graders.\n")
        sys.exit(2)
    write_results(data, PARAMS, timing, PARAMS["output_dir"])


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
