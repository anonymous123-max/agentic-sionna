"""Pre-compute N1.probe reference oracles for 4 (scene, TX, RX) configurations.

For each configuration we run Sionna RT's PathSolver and extract:
  - path_gain_db    = 10 log10(sum |a_i|^2)
  - num_paths       = number of valid paths
  - mean_delay_ns   = first moment of |a|^2 over tau
  - delay_spread_ns = sqrt(second central moment) — RMS delay spread

These references are the Layer C ground truth for the N1.probe verifier.

Run with:
  $RF_SIONNA_PY benchmark/_smoke_rt/run_n1_probe_reference.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import sionna
import sionna.rt as rt

CONFIGS = [
    {"name": "box_two_screens",      "tx": [0, 0, 4.5], "rx": [3, 2, 1.5],
     "max_depth": 3, "samples_per_src": 1_000_000, "kind": "indoor"},
    {"name": "box_one_screen",       "tx": [0, 0, 4.5], "rx": [3, 2, 1.5],
     "max_depth": 3, "samples_per_src": 1_000_000, "kind": "indoor"},
    {"name": "simple_street_canyon", "tx": [0, 0, 15],  "rx": [5, 0, 8],
     "max_depth": 3, "samples_per_src": 1_000_000, "kind": "outdoor"},
    {"name": "etoile",               "tx": [0, 0, 30],  "rx": [5, 0, 25],
     "max_depth": 3, "samples_per_src": 1_000_000, "kind": "outdoor"},
]
FREQ_HZ = 5e9
TX_POWER_DBM = 20.0   # 100 mW

OUT_DIR = Path(__file__).resolve().parents[1] / "oracles" / "n1_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_link_metrics(scene_name: str, tx_pos, rx_pos,
                         max_depth: int, samples_per_src: int) -> dict:
    s = rt.load_scene(getattr(rt.scene, scene_name))
    s.frequency = FREQ_HZ
    s.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                pattern="iso", polarization="V")
    s.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                pattern="iso", polarization="V")
    s.add(rt.Transmitter(name="tx", position=list(tx_pos),
                         power_dbm=TX_POWER_DBM))
    s.add(rt.Receiver(name="rx", position=list(rx_pos)))

    t0 = time.time()
    paths = rt.PathSolver()(s, max_depth=max_depth,
                            samples_per_src=samples_per_src)
    elapsed = time.time() - t0

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

    return {
        "scene_name":       scene_name,
        "tx_position":      list(tx_pos),
        "rx_position":      list(rx_pos),
        "frequency_hz":     FREQ_HZ,
        "tx_power_dbm":     TX_POWER_DBM,
        "max_depth":        max_depth,
        "samples_per_src":  samples_per_src,
        "tx_rx_distance_m": float(np.linalg.norm(np.array(tx_pos) - np.array(rx_pos))),
        "path_gain_db":     round(float(path_gain_db), 2),
        "num_paths":        n_paths,
        "mean_delay_ns":    round(mean_delay_s * 1e9, 3),
        "delay_spread_ns":  round(rms_delay_s * 1e9, 3),
        "method":           "sionna_rt",
        "solver_elapsed_s": round(elapsed, 2),
        "sionna_version":   sionna.__version__,
    }


def main() -> int:
    print(f"sionna {sionna.__version__}  oracle out: {OUT_DIR}")
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']}  ({cfg['kind']}) ===")
        m = compute_link_metrics(
            cfg["name"], cfg["tx"], cfg["rx"],
            cfg["max_depth"], cfg["samples_per_src"])
        out = OUT_DIR / f"{cfg['name']}.json"
        out.write_text(json.dumps(m, indent=2))
        print(f"  TX{cfg['tx']}  RX{cfg['rx']}  d={m['tx_rx_distance_m']:.2f}m")
        print(f"  path_gain={m['path_gain_db']:.2f} dB  #paths={m['num_paths']}  "
              f"mean_delay={m['mean_delay_ns']:.2f} ns  "
              f"DS={m['delay_spread_ns']:.2f} ns  "
              f"({m['solver_elapsed_s']:.1f}s)")
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
