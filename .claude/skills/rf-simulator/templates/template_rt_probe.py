#!/usr/bin/env python3
"""Template: Sionna RT Point-to-Point Link Probe.

For a single (TX, RX) pair on a Sionna built-in scene, compute path-level
link metrics via PathSolver.

Modify ONLY the PARAMS block. Outputs simulation_result.json with the
required link characteristics.
"""
from __future__ import annotations

# ============================================================================
# PARAMETER BLOCK — MODIFY ONLY THIS SECTION
# ============================================================================
PARAMS = {
    "scene_name":      "box_two_screens",   # one of sionna.rt.scene.*
    "tx_position":     [0.0, 0.0, 4.5],     # meters (x, y, z)
    "rx_position":     [3.0, 2.0, 1.5],     # meters
    "frequency_hz":    5e9,
    "tx_power_dbm":    20.0,                 # 100 mW
    "max_depth":       3,                    # path solver depth (reflections)
    "samples_per_src": 1_000_000,            # MC samples per source
    "output_dir":      ".",
}
# ============================================================================

import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    p = PARAMS
    out = Path(p["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    import sionna
    import sionna.rt as rt

    scene = rt.load_scene(getattr(rt.scene, p["scene_name"]))
    scene.frequency = p["frequency_hz"]
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                    pattern="iso", polarization="V")
    scene.add(rt.Transmitter(name="tx", position=list(p["tx_position"]),
                             power_dbm=p["tx_power_dbm"]))
    scene.add(rt.Receiver(name="rx", position=list(p["rx_position"])))

    t0 = time.time()
    paths = rt.PathSolver()(scene,
                            max_depth=p["max_depth"],
                            samples_per_src=p["samples_per_src"])
    elapsed = time.time() - t0

    # paths.a is (real, imag) tuple; shape (num_rx, n_rx_ant, num_tx, n_tx_ant, num_paths)
    a_re, a_im = paths.a
    a_re = np.array(a_re).squeeze()
    a_im = np.array(a_im).squeeze()
    a_mag2 = a_re ** 2 + a_im ** 2
    tau = np.array(paths.tau).squeeze()
    valid = np.array(paths.valid).squeeze().astype(bool)

    a_mag2 = a_mag2[valid]
    tau = tau[valid]
    total_power = float(a_mag2.sum())
    n_paths = int(valid.sum())
    if total_power > 0:
        path_gain_db = 10 * np.log10(total_power)
        mean_delay_s = float(np.sum(a_mag2 * tau) / total_power)
        rms_delay_s = float(
            np.sqrt(np.sum(a_mag2 * (tau - mean_delay_s) ** 2) / total_power))
    else:
        path_gain_db = -200.0
        mean_delay_s = rms_delay_s = 0.0

    result = {
        "scene_name":       p["scene_name"],
        "tx_position":      list(p["tx_position"]),
        "rx_position":      list(p["rx_position"]),
        "frequency_hz":     p["frequency_hz"],
        "tx_power_dbm":     p["tx_power_dbm"],
        "path_gain_db":     round(float(path_gain_db), 2),
        "num_paths":        n_paths,
        "mean_delay_ns":    round(mean_delay_s * 1e9, 3),
        "delay_spread_ns":  round(rms_delay_s * 1e9, 3),
        "method":           "sionna_rt",
        "solver_elapsed_s": round(elapsed, 2),
        "sionna_version":   sionna.__version__,
    }
    (out / "simulation_result.json").write_text(json.dumps(result, indent=2))
    print(f"PathSolver on {p['scene_name']}: "
          f"path_gain={result['path_gain_db']:.2f} dB, "
          f"#paths={result['num_paths']}, "
          f"DS={result['delay_spread_ns']:.2f} ns")


if __name__ == "__main__":
    import os
    if not os.environ.get("RF_SKIP_TEMPLATE_WARN"):
        sys.stderr.write(
            "\n*** TEMPLATE WARNING ***\n"
            "This is a TEMPLATE — copy the file to your workdir and edit\n"
            "PARAMS before running. Set RF_SKIP_TEMPLATE_WARN=1 to silence.\n\n"
        )
    main()
