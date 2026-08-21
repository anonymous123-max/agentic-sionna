"""Pre-compute P2 Pareto-frontier reference oracles.

P2: For one TX -> one UE, sweep (azimuth, TX_power_dBm) over a 5x3 grid
of 15 configurations. Per config compute throughput via Sionna PHY chain
(LDPC QPSK 1/2 BLER curve, interpolated) and convert TX power to linear
mW. The two-objective Pareto frontier is the subset of non-dominated
points in (throughput, tx_power_mw) where the agent wants to MAXIMISE
throughput and MINIMISE tx_power_mw.

Skill story: high power + good aim => high throughput but high energy;
low power + good aim => moderate throughput at minimal energy; poor aim
at any power is dominated. The Pareto frontier is the rate-vs-energy
trade-off curve.

We reuse the BLER curve produced by run_p1_optimize_reference.py
(benchmark/oracles/p1_optimize/_bler_curve.json) so the throughput
interpolation is identical across P1 and P2.

Output: benchmark/oracles/p2_pareto/{scene}/oracle.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import sionna
import sionna.rt as rt


FREQ_HZ = 5e9
NOISE_DBM = -85.0
AZIMUTH_GRID = [-90.0, -45.0, 0.0, 45.0, 90.0]   # 5 azimuths (matches P1)
# downtilt is fixed per-scene (from P1 best dt); power is the 2nd sweep axis

CONFIGS = [
    {"name": "box_one_screen",       "kind": "indoor",
     "tx_pos": [-2.0, 0.0, 4.5], "ue_pos": [2.0, 0.0, 1.5],
     "downtilt_fixed_deg": 45.0,                  # P1 best dt
     "power_grid_dbm":     [-35.0, -31.0, -28.0], # waterfall span at best aim
     "max_depth": 3, "samples_per_src": 500_000},
    {"name": "box_two_screens",      "kind": "indoor",
     "tx_pos": [-2.0, 0.0, 4.5], "ue_pos": [2.0, 0.0, 1.5],
     "downtilt_fixed_deg": 0.0,                   # P1 best dt (best aim is ±90 az)
     "power_grid_dbm":     [-13.0, -9.0, -6.0],
     "max_depth": 3, "samples_per_src": 500_000},
    {"name": "simple_street_canyon", "kind": "outdoor",
     "tx_pos": [-15.0, 0.0, 15.0], "ue_pos": [15.0, 0.0, 1.5],
     "downtilt_fixed_deg": 0.0,
     "power_grid_dbm":     [-18.0, -14.0, -11.0],
     "max_depth": 3, "samples_per_src": 500_000},
    {"name": "etoile",               "kind": "outdoor",
     "tx_pos": [-50.0, 0.0, 30.0], "ue_pos": [50.0, 0.0, 1.5],
     "downtilt_fixed_deg": 0.0,
     "power_grid_dbm":     [-6.0, -3.0, 1.0],
     "max_depth": 3, "samples_per_src": 500_000},
]

OUT_ROOT = Path(__file__).resolve().parents[1] / "oracles" / "p2_pareto"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
BLER_CURVE_PATH = (Path(__file__).resolve().parents[1] / "oracles"
                    / "p1_optimize" / "_bler_curve.json")
PHY_K, PHY_N, PHY_BPS = 1024, 2048, 2
PHY_CODE_RATE = PHY_K / PHY_N


def load_bler_curve() -> dict:
    if not BLER_CURVE_PATH.exists():
        raise FileNotFoundError(
            f"Expected P1 BLER curve at {BLER_CURVE_PATH} - run "
            f"run_p1_optimize_reference.py first.")
    return json.loads(BLER_CURVE_PATH.read_text())


def interp_bler(ebn0_db: float, curve: dict) -> float:
    return float(np.clip(np.interp(ebn0_db, curve["ebn0_db"], curve["bler"]),
                         0.0, 1.0))


def solve_path_gain_db(scene_name: str, tx_pos, ue_pos,
                       az_deg: float, dt_deg: float,
                       max_depth: int, samples_per_src: int) -> tuple[float, int]:
    s = rt.load_scene(getattr(rt.scene, scene_name))
    s.frequency = FREQ_HZ
    s.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                pattern="tr38901", polarization="V")
    s.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                pattern="iso", polarization="V")
    tx = rt.Transmitter(name="tx", position=list(tx_pos), power_dbm=0.0)
    s.add(tx)
    s.add(rt.Receiver(name="rx", position=list(ue_pos)))
    tx.orientation = [math.radians(az_deg), -math.radians(dt_deg), 0.0]

    paths = rt.PathSolver()(s, max_depth=max_depth,
                            samples_per_src=samples_per_src)
    a_re, a_im = paths.a
    a_re = np.array(a_re).squeeze()
    a_im = np.array(a_im).squeeze()
    valid = np.array(paths.valid).squeeze().astype(bool)
    a_mag2 = a_re ** 2 + a_im ** 2
    a_mag2 = a_mag2[valid]
    n_paths = int(valid.sum())
    total = float(a_mag2.sum())
    if total > 0:
        return round(10.0 * math.log10(total), 3), n_paths
    return -200.0, 0


def pareto_indices(rows: list[dict],
                   max_keys: tuple[str, ...] = ("throughput",),
                   min_keys: tuple[str, ...] = ("tx_power_mw",)) -> list[int]:
    pareto = []
    for i, ri in enumerate(rows):
        dominated = False
        for j, rj in enumerate(rows):
            if i == j:
                continue
            better_eq = (all(rj[k] >= ri[k] - 1e-9 for k in max_keys) and
                         all(rj[k] <= ri[k] + 1e-9 for k in min_keys))
            strictly_better = (any(rj[k] > ri[k] + 1e-9 for k in max_keys) or
                                any(rj[k] < ri[k] - 1e-9 for k in min_keys))
            if better_eq and strictly_better:
                dominated = True
                break
        if not dominated:
            pareto.append(i)
    return pareto


def hypervolume_2d(points: list[tuple[float, float]],
                   ref_max: tuple[float, float]) -> float:
    """Hypervolume in 2D for (max throughput, min tx_power_mw).

    For a Pareto set in (max-throughput, min-power), the dominated region wrt
    reference point (0_throughput, ref_max_power) is the union of rectangles
       [0, throughput_i] x [tx_power_mw_i, ref_max_power].

    True Pareto points satisfy: when sorted by ascending throughput, power
    is also ascending. The HV is therefore
       HV = sum_i (thr_i - thr_{i-1}) * (ref_max_power - pwr_i)
    where thr_0 = 0 and we sort by ascending throughput.
    """
    _, ref_max_power = ref_max
    if not points:
        return 0.0
    pts = sorted(points, key=lambda p: p[0])    # ascending throughput
    hv = 0.0
    prev_thr = 0.0
    for thr, pwr in pts:
        if thr <= prev_thr:
            continue
        width  = thr - prev_thr
        height = max(0.0, ref_max_power - pwr)
        hv += width * height
        prev_thr = thr
    return hv


def compute_oracle(cfg: dict, bler_curve: dict) -> dict:
    name = cfg["name"]
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_fixed = cfg["downtilt_fixed_deg"]
    power_grid = cfg["power_grid_dbm"]

    print(f"  TX{cfg['tx_pos']} UE{cfg['ue_pos']} dt={dt_fixed} powers={power_grid}")
    t_rt = 0.0
    rows = []
    # Cache path_gain per azimuth — same for all powers (path gain is independent of TX power)
    pg_cache = {}
    for az in AZIMUTH_GRID:
        if az not in pg_cache:
            t0 = time.time()
            pg_db, n_paths = solve_path_gain_db(
                name, cfg["tx_pos"], cfg["ue_pos"], az, dt_fixed,
                cfg["max_depth"], cfg["samples_per_src"])
            t_rt += time.time() - t0
            pg_cache[az] = (pg_db, n_paths)
        pg_db, n_paths = pg_cache[az]
        for p_dbm in power_grid:
            snr_db = p_dbm + pg_db - NOISE_DBM
            ebn0_db = snr_db   # rate*bps=1
            bler = interp_bler(ebn0_db, bler_curve)
            thr = PHY_CODE_RATE * PHY_BPS * (1.0 - bler)
            tx_power_mw = round(10.0 ** (p_dbm / 10.0), 5)
            rows.append({
                "az_deg":       round(az, 2),
                "tx_power_dbm": round(p_dbm, 2),
                "tx_power_mw":  tx_power_mw,
                "path_gain_db": round(pg_db, 3),
                "num_paths":    n_paths,
                "snr_db":       round(snr_db, 3),
                "ebn0_db":      round(ebn0_db, 3),
                "bler":         round(bler, 5),
                "throughput":   round(thr, 5),
            })

    par_idx = pareto_indices(rows)
    pareto_pts = [(rows[i]["throughput"], rows[i]["tx_power_mw"]) for i in par_idx]

    # Reference hypervolume for verifier: use (0, max_power_in_grid_mw) as ref
    ref_max_power = max(rows[i]["tx_power_mw"] for i in range(len(rows)))
    hv = hypervolume_2d(pareto_pts, ref_max=(0.0, ref_max_power))

    oracle = {
        "task_family":          "P2_pareto",
        "scene_name":           name,
        "kind":                 cfg["kind"],
        "tx_position":          cfg["tx_pos"],
        "ue_position":          cfg["ue_pos"],
        "frequency_hz":         FREQ_HZ,
        "noise_dbm":            NOISE_DBM,
        "downtilt_fixed_deg":   dt_fixed,
        "azimuth_deg_grid":     AZIMUTH_GRID,
        "tx_power_dbm_grid":    power_grid,
        "antenna":              {"num_rows": 1, "num_cols": 1,
                                  "pattern": "tr38901", "polarization": "V"},
        "phy":                  {"codec": "ldpc", "k": PHY_K, "n": PHY_N,
                                  "modulation": "QPSK",
                                  "code_rate": PHY_CODE_RATE,
                                  "throughput_model": "code_rate * bps * (1 - BLER)"},
        "pareto_objectives":    {"maximize": ["throughput"],
                                  "minimize": ["tx_power_mw"]},
        "sweep_table":          rows,
        "pareto_set":           par_idx,
        "pareto_hypervolume":   round(hv, 6),
        "hv_ref_point":         {"throughput": 0.0,
                                  "tx_power_mw": ref_max_power},
        "elapsed_s":            {"rt": round(t_rt, 1)},
        "method":               "sionna_rt + sionna_phy",
        "sionna_version":       sionna.__version__,
    }
    (out_dir / "oracle.json").write_text(json.dumps(oracle, indent=2))
    return oracle


def main() -> int:
    print(f"sionna {sionna.__version__}  oracle root: {OUT_ROOT}")
    print(f"  loading BLER curve from {BLER_CURVE_PATH}")
    curve = load_bler_curve()

    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']}  ({cfg['kind']}) ===")
        m = compute_oracle(cfg, curve)
        n_pareto = len(m["pareto_set"])
        print(f"  Pareto: {n_pareto}/15 non-dominated  HV={m['pareto_hypervolume']:.3f}")
        for idx in m["pareto_set"]:
            r = m["sweep_table"][idx]
            print(f"    [{idx:2d}] az={r['az_deg']:+5.0f} P={r['tx_power_dbm']:+5.1f}dBm "
                  f"({r['tx_power_mw']:8.4f} mW)  tput={r['throughput']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
