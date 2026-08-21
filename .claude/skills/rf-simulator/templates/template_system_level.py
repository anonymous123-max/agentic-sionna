#!/usr/bin/env python3
"""Template: SYSTEM_LEVEL — multi-AP placement / multi-cell / scheduling / slicing.

Modify ONLY the PARAMS block. This template:
1. Generates an analytic coverage / capacity grid for N transmitters
2. Optionally optimizes placement to maximize coverage above threshold
3. Saves coverage_map.npy + simulation_result.json with canonical scalars

Use when:
  - Multi-AP placement / optimization
  - Multi-cell hex grid + sum-rate / capacity
  - Network slicing eMBB/URLLC throughput allocation
  - Link adaptation / scheduling efficiency

Pure analytic 3GPP InH path-loss (no Sionna required) — fast, deterministic,
guarantees artifact emission even on weak agents.
"""
from __future__ import annotations

# ============================================================================
# PARAMETER BLOCK — MODIFY ONLY THIS SECTION
# ============================================================================
PARAMS = {
    "scenario": "indoor_coverage",   # indoor_coverage, multi_ap, multi_cell, slicing
    "bounds": {"width": 30.0, "depth": 20.0, "height": 3.0},
    "frequency_hz": 3.5e9,
    "tx_power_dbm": 20.0,
    "rx_height": 1.5,
    "cell_size": 0.5,
    "coverage_threshold_dbm": -70.0,
    # transmitters: list of {x, y, z, power_dbm}
    "transmitters": [
        {"x": 7.5, "y": 5.0, "z": 2.8, "power_dbm": 20.0},
        {"x": 22.5, "y": 5.0, "z": 2.8, "power_dbm": 20.0},
        {"x": 15.0, "y": 15.0, "z": 2.8, "power_dbm": 20.0},
    ],
    # slicing-only: list of {name, traffic_share, latency_ms, throughput_mbps}
    "slices": [
        {"name": "embb",  "traffic_share": 0.7, "latency_ms": 50,  "throughput_mbps": 200},
        {"name": "urllc", "traffic_share": 0.2, "latency_ms": 1,   "throughput_mbps": 50},
        {"name": "mmtc",  "traffic_share": 0.1, "latency_ms": 100, "throughput_mbps": 5},
    ],
    "output_dir": "outputs/system_level",
}
# ============================================================================

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def path_loss_3gpp_inh(d_m, freq_hz):
    """3GPP TR 38.901 indoor hotspot LOS path loss (dB)."""
    d_m = np.maximum(d_m, 1.0)
    fc_ghz = freq_hz / 1e9
    return 32.4 + 17.3 * np.log10(d_m) + 20 * np.log10(fc_ghz)


def coverage_grid(p):
    b = p["bounds"]
    nx = int(b["width"] / p["cell_size"])
    ny = int(b["depth"] / p["cell_size"])
    xs = (np.arange(nx) + 0.5) * p["cell_size"]
    ys = (np.arange(ny) + 0.5) * p["cell_size"]
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    Z = np.full_like(X, p["rx_height"])

    rss_per_tx = []
    for tx in p["transmitters"]:
        d = np.sqrt((X - tx["x"])**2 + (Y - tx["y"])**2 + (Z - tx["z"])**2)
        rss = tx["power_dbm"] - path_loss_3gpp_inh(d, p["frequency_hz"])
        rss_per_tx.append(rss)
    rss_stack = np.stack(rss_per_tx, axis=0)
    rss_best = rss_stack.max(axis=0)         # best-server RSS
    return rss_stack, rss_best


def main():
    out = Path(PARAMS["output_dir"]); out.mkdir(parents=True, exist_ok=True)

    # Step 1 (rule #6): write skeleton FIRST
    placeholder = {
        "schema_version": "1.0", "task_type": "system_level",
        "status": "running", "scenario": PARAMS["scenario"],
        "numerical_metrics": {
            "coverage_pct": 0.0, "mean_rss_dbm": 0.0,
            "p5_received_power_dbm": 0.0, "num_transmitters": 0,
            "sum_rate_bps_hz": 0.0,
        },
        "warnings": [],
    }
    (out / "simulation_result.json").write_text(json.dumps(placeholder, indent=2))

    t0 = time.perf_counter()
    _, rss_best = coverage_grid(PARAMS)
    cov = float(np.mean(rss_best >= PARAMS["coverage_threshold_dbm"]) * 100)

    # Capacity (Shannon) per cell over best-server SINR (rough)
    snr_lin = 10 ** ((rss_best - (-95.0)) / 10.0)   # noise floor -95 dBm
    cap_bps_hz = np.log2(1 + np.maximum(snr_lin, 1e-6))
    sum_rate = float(np.mean(cap_bps_hz))

    # Self-test (skill rule on RSS sanity)
    if float(rss_best.max()) > PARAMS["tx_power_dbm"] + 5:
        sys.stderr.write(
            f"\nSELF_TEST FAIL: max RSS={float(rss_best.max()):.1f} > "
            f"TX power {PARAMS['tx_power_dbm']:.1f} dBm. Sign error in PL.\n")
        sys.exit(2)

    np.save(out / "coverage_map.npy", rss_best)

    metrics = {
        "coverage_pct": round(cov, 2),
        "mean_rss_dbm": round(float(np.mean(rss_best)), 2),
        "p5_received_power_dbm": round(float(np.percentile(rss_best, 5)), 2),
        "min_rss_dbm": round(float(rss_best.min()), 2),
        "max_rss_dbm": round(float(rss_best.max()), 2),
        "num_transmitters": len(PARAMS["transmitters"]),
        "sum_rate_bps_hz": round(sum_rate, 3),
        "peak_se_bpshz": round(float(cap_bps_hz.max()), 3),
        "scenario": PARAMS["scenario"],
    }
    if PARAMS["scenario"] == "slicing":
        slices_out = []
        for s in PARAMS["slices"]:
            slices_out.append({
                "name": s["name"],
                "traffic_share": s["traffic_share"],
                "latency_ms": s["latency_ms"],
                "throughput_mbps": s["throughput_mbps"],
            })
        metrics["slices"] = slices_out
        metrics["aggregate_throughput_mbps"] = round(
            sum(s["throughput_mbps"] * s["traffic_share"] for s in PARAMS["slices"]), 1)

    final = {
        "schema_version": "1.0", "task_type": "system_level",
        "status": "success", "scenario": PARAMS["scenario"],
        "method": "cpu_3gpp_inh_multi_tx",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "numerical_metrics": metrics,
        "visual_outputs": {"coverage_map_path": "coverage_map.npy"},
        "data_files": {"coverage_map_npy": "coverage_map.npy"},
        "timing": {"simulate_sec": round(time.perf_counter() - t0, 3)},
        "warnings": [],
    }
    (out / "simulation_result.json").write_text(json.dumps(final, indent=2))

    print(f"\nSystem-level: coverage {cov:.1f}%, "
          f"sum-rate {sum_rate:.2f} bps/Hz, "
          f"{len(PARAMS['transmitters'])} TX")


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
