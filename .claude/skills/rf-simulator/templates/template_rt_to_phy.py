#!/usr/bin/env python3
"""Template: Hybrid RT-to-PHY — Site-Specific Link Budget.

Modify ONLY the PARAMS block. This template:
1. Computes per-location RSS via RT or analytical model
2. Converts RSS → SNR → per-location BER and throughput
3. Maps achievable MCS at each receiver location

GPU path: Sionna RT → CIR → OFDM channel → PHY processing
CPU path: 3GPP InH RSS → SNR → analytical BER per cell
"""
from __future__ import annotations

# ============================================================================
# PARAMETER BLOCK — MODIFY ONLY THIS SECTION
# ============================================================================
PARAMS = {
    "scene_path": "scene_state.json",
    "frequency_hz": 3.5e9,
    "tx_position": [5.0, 4.0, 2.8],
    "tx_power_dbm": 20.0,
    "tx_antenna": "iso",
    "rx_height": 1.5,
    "cell_size": 0.5,
    "coverage_threshold_dbm": -70.0,
    "num_tx_ant": 2, "num_rx_ant": 2,
    "subcarrier_spacing_khz": 30,
    "fft_size": 128,
    "num_ofdm_symbols": 14,
    "cyclic_prefix_length": 20,
    "codec": "ldpc",
    "code_rate_k": 1024, "code_rate_n": 2048,
    "num_bits_per_symbol": 4,
    "channel_estimator": "ls",
    "max_depth": 5,
    "output_dir": "outputs/rt_to_phy",
}
# ============================================================================

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


def validate_params(p):
    errors = []
    if not Path(p["scene_path"]).exists(): errors.append("scene_path not found")
    if p["code_rate_k"] >= p["code_rate_n"]: errors.append("k must be < n")
    if p["num_bits_per_symbol"] not in (1,2,4,6,8): errors.append("invalid bits/symbol")
    return errors


def run_sionna_rt_phy(params):
    """Sionna RT → CIR → per-location SNR + BER."""
    import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import sionna.rt as rt

    # Load scene (reuse template_rt_coverage export logic)
    state = json.loads(Path(params["scene_path"]).read_text())
    bounds = state["scene"]["bounds"]
    w, l = bounds["width"], bounds["depth"]

    xml_dir = Path(params["scene_path"]).parent / "sionna"
    xml_path = xml_dir / "scene.xml"
    if not xml_path.exists():
        # Import inline to avoid dependency when not needed
        from template_rt_coverage import export_scene_to_ply_xml
        export_scene_to_ply_xml(state, xml_dir)

    scene = rt.load_scene(str(xml_path))
    scene.frequency = params["frequency_hz"]
    scene.tx_array = rt.PlanarArray(num_rows=params["num_tx_ant"], num_cols=1,
                                     vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern=params["tx_antenna"], polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=params["num_rx_ant"], num_cols=1,
                                     vertical_spacing=0.5, horizontal_spacing=0.5,
                                     pattern="iso", polarization="V")
    tx = rt.Transmitter("tx", position=params["tx_position"],
                         power_dbm=params["tx_power_dbm"])
    scene.add(tx)

    # RadioMap for per-cell RSS
    rm = rt.RadioMapSolver()
    radio_map = rm(scene, cell_size=[params["cell_size"]]*2,
                    samples_per_tx=1_000_000, max_depth=params["max_depth"],
                    center=[w/2, l/2, params["rx_height"]],
                    orientation=[0,0,0], size=[w, l])

    pg = np.array(radio_map.path_gain)
    if pg.ndim == 3: pg = pg[0]
    rss = params["tx_power_dbm"] + 10*np.log10(np.where(pg > 0, pg, np.nan))
    return rss, "sionna_rt_phy"


def run_cpu_rt_phy(params):
    """CPU analytical RSS (reuse 3GPP InH from coverage template)."""
    from template_rt_coverage import run_cpu_analytical
    rss, method = run_cpu_analytical(params["scene_path"], params)
    return rss, "cpu_analytical_phy"


def compute_phy_metrics(rss, params):
    """Convert RSS map to per-cell SNR, BER, throughput, MCS."""
    from scipy.special import erfc

    noise_floor = -95.0
    snr_db = rss - noise_floor
    snr_lin = np.maximum(10**(snr_db/10), 1e-10)

    M = 2**params["num_bits_per_symbol"]
    nt, nr = params["num_tx_ant"], params["num_rx_ant"]
    ns = min(nt, nr)
    cr = params["code_rate_k"]/params["code_rate_n"]
    ag = nt*nr/ns
    cp_oh = params["cyclic_prefix_length"]/(params["fft_size"]+params["cyclic_prefix_length"])
    ofdm_eff = (1-cp_oh)*0.85
    ce_pen = 1.0 if params["channel_estimator"]=="lmmse" else 0.8
    cg = 10**((6.0+2.0*cr)/10) if params["codec"]=="ldpc" else 10**((5.0+1.5*cr)/10)

    snr_eff = snr_lin * ag * ofdm_eff * ce_pen * cg
    if M == 2: ber = 0.5*erfc(np.sqrt(snr_eff))
    else:
        f = (4/np.log2(M))*(1-1/np.sqrt(M))
        ber = f*0.5*erfc(np.sqrt(3*np.log2(M)*snr_eff/(M-1))/np.sqrt(2))
    ber = np.maximum(ber, 1e-7)

    bw = params["fft_size"]*params["subcarrier_spacing_khz"]*1e3
    capacity = bw*ns*np.log2(1+snr_lin*ag*ofdm_eff)*ofdm_eff/1e6
    throughput = np.minimum(capacity, bw*ns*np.log2(M)*cr*ofdm_eff/1e6)

    mcs = np.zeros_like(snr_db, dtype=int)
    mcs[snr_db >= 20] = 27; mcs[(snr_db>=13)&(snr_db<20)] = 20
    mcs[(snr_db>=6)&(snr_db<13)] = 11; mcs[(snr_db>=0)&(snr_db<6)] = 4

    return {"rss": rss, "snr_db": snr_db, "ber": ber,
            "throughput_mbps": throughput, "mcs": mcs}


def main():
    errors = validate_params(PARAMS)
    if errors:
        for e in errors: print(f"ERROR: {e}", file=sys.stderr); sys.exit(1)

    t0 = time.perf_counter()
    try:
        rss, method = run_sionna_rt_phy(PARAMS)
    except Exception as e:
        print(f"Sionna RT unavailable ({e}), using CPU analytical")
        rss, method = run_cpu_rt_phy(PARAMS)
    timing = {"simulate_sec": round(time.perf_counter()-t0, 3)}

    phy = compute_phy_metrics(rss, PARAMS)
    out = Path(PARAMS["output_dir"]); out.mkdir(parents=True, exist_ok=True)

    # Save data
    for name in ["rss", "snr_db", "ber", "throughput_mbps"]:
        np.save(out/f"{name}.npy", phy[name])

    # Plots
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    state = json.loads(Path(PARAMS["scene_path"]).read_text())
    w, l = state["scene"]["bounds"]["width"], state["scene"]["bounds"]["depth"]
    tx = PARAMS["tx_position"]

    for name, data, cmap, label in [
        ("throughput_map", phy["throughput_mbps"], "YlOrRd", "Mbps"),
        ("ber_map", np.log10(np.maximum(phy["ber"], 1e-7)), "RdYlGn_r", "log₁₀(BER)"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, origin="lower", extent=(0,w,0,l), cmap=cmap, aspect="equal")
        ax.plot(tx[0], tx[1], "w^", ms=12, markeredgecolor="black", markeredgewidth=1.5)
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        fig.savefig(out/f"{name}.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # Coverage
    valid = np.isfinite(rss)
    rss_v = rss[valid]
    cov = float(np.sum(valid & (rss >= PARAMS["coverage_threshold_dbm"]))/max(np.sum(valid),1)*100)
    tp_v = phy["throughput_mbps"][np.isfinite(phy["throughput_mbps"])]

    # Self-test: RSS in physical range; throughput non-negative
    if rss_v.size > 0:
        if float(np.max(rss_v)) > PARAMS["tx_power_dbm"] + 5:
            sys.stderr.write(
                f"\nSELF_TEST FAIL: max RSS {float(np.max(rss_v)):.1f} dBm > "
                f"TX power {PARAMS['tx_power_dbm']:.1f} dBm. Path-loss sign error.\n")
            sys.exit(2)
    if tp_v.size > 0 and float(np.min(tp_v)) < 0:
        sys.stderr.write(f"\nSELF_TEST FAIL: negative throughput.\n"); sys.exit(2)

    result = {
        "schema_version": "1.0", "task_type": "rt_to_phy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "success", "method": method,
        "numerical_metrics": {
            "coverage_pct": round(cov, 2),
            "mean_throughput_mbps": round(float(np.mean(tp_v)), 2) if tp_v.size else 0,
            "mean_ber": f"{float(np.mean(phy['ber'][valid])):.2e}" if valid.any() else None,
            "p5_snr_db": round(float(np.percentile(phy["snr_db"][valid], 5)), 2) if valid.any() else None,
            "mimo_config": f"{PARAMS['num_tx_ant']}x{PARAMS['num_rx_ant']}",
        },
        "visual_outputs": {"throughput_map_path": "throughput_map.png", "ber_map_path": "ber_map.png"},
        "data_files": {k: f"{k}.npy" for k in ["rss","snr_db","ber","throughput_mbps"]},
        "timing": timing, "warnings": [],
    }
    (out/"simulation_result.json").write_text(json.dumps(result, indent=2))
    print(f"\nCoverage: {cov:.1f}%, Mean throughput: {result['numerical_metrics']['mean_throughput_mbps']} Mbps ({method})")

    if method.startswith("sionna"): import os; os._exit(0)

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
