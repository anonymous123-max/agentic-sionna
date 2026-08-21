"""Pre-compute P1 optimization reference oracles.

P1: AP azimuth/downtilt optimization for single-UE throughput.
For each of 4 scenes we run a 5x3 sweep (5 azimuths x 3 downtilts = 15 configs)
of a 1x1 tr38901 directional element at the AP, with the UE at a fixed position.

Per config:
  1. PathSolver -> path_gain_db (includes antenna pattern, depends on orientation)
  2. Eb/N0_dB = TX_power_dbm + path_gain_db - noise_dbm   (rate*bps=1 for QPSK 1/2)
  3. BLER via interpolation of a precomputed LDPC5G+QPSK BLER curve
  4. throughput = code_rate * bps * (1 - BLER) = 1 - BLER  bits/symbol

We frame the AP as a "low-power node" so TX_power is set per-scene s.t. the
best config lands near 3 dB Eb/N0 (top of LDPC QPSK 1/2 waterfall) and the
worst lands below 0 dB. Spread is meaningful across the sweep.

Output: benchmark/oracles/p1_optimize/{scene}/oracle.json
        benchmark/oracles/p1_optimize/_bler_curve.json    (shared)
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
AZIMUTH_GRID = [-90.0, -45.0, 0.0, 45.0, 90.0]   # widened so off-boresight dips
DOWNTILT_GRID = [0.0, 45.0, 90.0]                # widened similarly
BLER_GRID_EBNO_DB = list(np.arange(-5.0, 10.5, 0.5))  # 31 points
PHY_K, PHY_N = 1024, 2048
PHY_CODE_RATE = PHY_K / PHY_N
PHY_BPS = 2
TARGET_BEST_EBNO_DB = 4.0   # tune TX power so best config lands here

CONFIGS = [
    {"name": "box_two_screens",      "kind": "indoor",
     "tx_pos": [-2.0, 0.0, 4.5], "ue_pos": [2.0, 0.0, 1.5],
     "max_depth": 3, "samples_per_src": 500_000},
    {"name": "box_one_screen",       "kind": "indoor",
     "tx_pos": [-2.0, 0.0, 4.5], "ue_pos": [2.0, 0.0, 1.5],
     "max_depth": 3, "samples_per_src": 500_000},
    {"name": "simple_street_canyon", "kind": "outdoor",
     "tx_pos": [-15.0, 0.0, 15.0], "ue_pos": [15.0, 0.0, 1.5],
     "max_depth": 3, "samples_per_src": 500_000},
    {"name": "etoile",               "kind": "outdoor",
     "tx_pos": [-50.0, 0.0, 30.0], "ue_pos": [50.0, 0.0, 1.5],
     "max_depth": 3, "samples_per_src": 500_000},
]

OUT_ROOT = Path(__file__).resolve().parents[1] / "oracles" / "p1_optimize"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def precompute_bler_curve(batch_size: int = 500) -> dict:
    """Compute LDPC5G QPSK 1/2 BLER curve over Eb/N0_dB grid (once)."""
    import torch
    from sionna.phy.fec import LDPC5GEncoder, LDPC5GDecoder
    from sionna.phy.mapping import Mapper, Demapper
    from sionna.phy.channel import AWGN
    from sionna.phy.utils import ebnodb2no

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = LDPC5GEncoder(k=PHY_K, n=PHY_N)
    dec = LDPC5GDecoder(encoder=enc, num_iter=20)
    mapper = Mapper(constellation_type="qam", num_bits_per_symbol=PHY_BPS)
    demapper = Demapper(demapping_method="app", constellation_type="qam",
                        num_bits_per_symbol=PHY_BPS)
    channel = AWGN()

    blers = []
    t0 = time.time()
    for ebn0 in BLER_GRID_EBNO_DB:
        no = ebnodb2no(float(ebn0), num_bits_per_symbol=PHY_BPS,
                       coderate=PHY_CODE_RATE)
        bits = torch.randint(0, 2, (batch_size, 1, 1, PHY_K),
                             dtype=torch.float32, device=device)
        coded = enc(bits)
        symbols = mapper(coded)
        noisy = channel(symbols, no)
        llr = demapper(noisy, no)
        decoded = dec(llr)
        bit_err = torch.ne(bits, decoded)
        cw_fail = torch.any(bit_err.flatten(start_dim=-1), dim=-1).float().mean()
        blers.append(round(float(cw_fail), 5))
    elapsed = time.time() - t0
    print(f"  BLER curve: 31 pts in {elapsed:.1f}s")
    for db, bl in zip(BLER_GRID_EBNO_DB[::4], blers[::4]):
        print(f"    Eb/N0 {db:+5.1f} dB -> BLER {bl:.4f}")
    return {
        "ebn0_db": BLER_GRID_EBNO_DB,
        "bler":    blers,
        "k":       PHY_K,
        "n":       PHY_N,
        "code_rate": PHY_CODE_RATE,
        "modulation": "QPSK",
        "batch_size": batch_size,
    }


def interp_bler(ebn0_db: float, curve: dict) -> float:
    return float(np.clip(np.interp(ebn0_db, curve["ebn0_db"], curve["bler"]),
                         0.0, 1.0))


def solve_path_gain_db(scene_name: str, tx_pos, ue_pos,
                       az_deg: float, dt_deg: float,
                       max_depth: int, samples_per_src: int) -> tuple[float, int]:
    """Return (path_gain_db, num_paths) for one orientation."""
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


def compute_oracle(cfg: dict, bler_curve: dict) -> dict:
    name = cfg["name"]
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: all RT solves at TX_power=0 so we get raw path_gain_db
    print(f"  TX{cfg['tx_pos']} UE{cfg['ue_pos']}  raw path-gain sweep:")
    t_rt = 0.0
    raw_rows = []
    for az in AZIMUTH_GRID:
        for dt in DOWNTILT_GRID:
            t0 = time.time()
            pg_db, n_paths = solve_path_gain_db(
                name, cfg["tx_pos"], cfg["ue_pos"], az, dt,
                cfg["max_depth"], cfg["samples_per_src"])
            t_rt += time.time() - t0
            raw_rows.append((az, dt, pg_db, n_paths))
    max_pg = max(r[2] for r in raw_rows)
    min_pg = min(r[2] for r in raw_rows)
    # tune TX power so best path-gain -> Eb/N0 = TARGET_BEST_EBNO_DB
    # Eb/N0_dB = TX + pg - noise   (rate*bps=1)
    tx_power = round(TARGET_BEST_EBNO_DB + NOISE_DBM - max_pg, 1)
    print(f"    pg range [{min_pg:.2f}, {max_pg:.2f}] dB, spread {max_pg-min_pg:.2f} dB"
          f"  -> TX_power={tx_power} dBm")

    # Pass 2: compute SNR, Eb/N0, BLER, throughput per config
    sweep_rows = []
    for az, dt, pg_db, n_paths in raw_rows:
        snr_db = tx_power + pg_db - NOISE_DBM
        ebn0_db = snr_db   # rate*bps=1
        bler = interp_bler(ebn0_db, bler_curve)
        thr = PHY_CODE_RATE * PHY_BPS * (1.0 - bler)
        sweep_rows.append({
            "az_deg":       round(az, 2),
            "dt_deg":       round(dt, 2),
            "path_gain_db": round(pg_db, 3),
            "num_paths":    n_paths,
            "snr_db":       round(snr_db, 3),
            "ebn0_db":      round(ebn0_db, 3),
            "bler":         round(bler, 5),
            "throughput":   round(thr, 5),
        })

    best = max(sweep_rows, key=lambda r: r["throughput"])
    worst = min(sweep_rows, key=lambda r: r["throughput"])

    oracle = {
        "task_family":      "P1_optimize",
        "scene_name":       name,
        "kind":             cfg["kind"],
        "tx_position":      cfg["tx_pos"],
        "ue_position":      cfg["ue_pos"],
        "tx_power_dbm":     tx_power,
        "noise_dbm":        NOISE_DBM,
        "frequency_hz":     FREQ_HZ,
        "max_depth":        cfg["max_depth"],
        "samples_per_src":  cfg["samples_per_src"],
        "azimuth_deg_grid": AZIMUTH_GRID,
        "downtilt_deg_grid": DOWNTILT_GRID,
        "antenna":          {"num_rows": 1, "num_cols": 1,
                              "pattern": "tr38901", "polarization": "V"},
        "phy":              {"codec": "ldpc", "k": PHY_K, "n": PHY_N,
                              "modulation": "QPSK", "code_rate": PHY_CODE_RATE,
                              "throughput_model": "code_rate * bps * (1 - BLER)"},
        "sweep_table":      sweep_rows,
        "best":             {"az_deg":      best["az_deg"],
                              "dt_deg":      best["dt_deg"],
                              "throughput":  best["throughput"],
                              "snr_db":      best["snr_db"]},
        "worst":            {"az_deg":      worst["az_deg"],
                              "dt_deg":      worst["dt_deg"],
                              "throughput":  worst["throughput"],
                              "snr_db":      worst["snr_db"]},
        "elapsed_s":        {"rt": round(t_rt, 1)},
        "method":           "sionna_rt + sionna_phy",
        "sionna_version":   sionna.__version__,
    }
    (out_dir / "oracle.json").write_text(json.dumps(oracle, indent=2))
    return oracle


def main() -> int:
    print(f"sionna {sionna.__version__}  oracle root: {OUT_ROOT}")
    print("\n=== Precompute LDPC QPSK 1/2 BLER curve ===")
    curve = precompute_bler_curve()
    (OUT_ROOT / "_bler_curve.json").write_text(json.dumps(curve, indent=2))

    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']}  ({cfg['kind']}) ===")
        m = compute_oracle(cfg, curve)
        b = m["best"]; w = m["worst"]
        spread = b["throughput"] - w["throughput"]
        print(f"  best=(az={b['az_deg']:+.0f},dt={b['dt_deg']:.0f}) "
              f"snr={b['snr_db']:+.1f} dB tput={b['throughput']:.3f}  "
              f"worst=(az={w['az_deg']:+.0f},dt={w['dt_deg']:.0f}) "
              f"snr={w['snr_db']:+.1f} dB tput={w['throughput']:.3f}  "
              f"spread={spread:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
