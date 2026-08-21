"""Pre-compute S1-S4 system-level reference oracles.

All four tasks share a single scene (simple_street_canyon) and the same
2-AP / 4-UE (or 1-AP / 4-UE) deployment style, so per-task path gains are
re-used across the suite.

Definitions:
  S1: 2 APs + 4 UEs. Each UE associates with best-RSS AP. Compute
      SINR_user, rate_user = log2(1 + SINR), sum throughput, Jain's index.
  S2: 1 AP + 4 UEs. Proportional-fair scheduler over T=10 time slots, with
      a single best-rate UE chosen per slot (single carrier, full bandwidth).
      Report per-user throughput, scheduled-UE sequence, Jain's index.
  S3: 2 APs + 2 UEs. Each AP sweeps az in {-45, 0, +45} deg => 3x3 = 9
      joint beam configs. Pick the config maximising sum_rate.
  S4: 1 AP + 4 UEs. 5 RB-allocation strategies (equal / max-rate / max-min /
      proportional-fair / weighted). For each: (sum_rate, jain_fairness).
      Report Pareto frontier in (max sum_rate, max fairness).

Output: benchmark/oracles/s1_s4/{s1,s2,s3,s4}.json
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

import sionna
import sionna.rt as rt


SCENE = "simple_street_canyon"
FREQ_HZ = 5e9
NOISE_DBM = -85.0
TX_POWER_DBM_DEFAULT = 0.0   # outdoor scene; calibrated so SINR lands in 0-25 dB

# S1 / S2 layout
AP_POSITIONS_2 = [[-15.0, 0.0, 15.0], [+15.0, 0.0, 15.0]]
AP_POSITIONS_1 = [[0.0,   0.0, 15.0]]
UE_POSITIONS_4 = [[-20.0, 0.0, 1.5],
                  [ -5.0, 0.0, 1.5],
                  [ +5.0, 0.0, 1.5],
                  [+20.0, 0.0, 1.5]]
# S3 layout: 2 APs + 2 UEs (each AP has one UE in its bore-sight half)
S3_AP_POSITIONS = [[-15.0, 0.0, 15.0], [+15.0, 0.0, 15.0]]
S3_UE_POSITIONS = [[-10.0, 0.0, 1.5], [+10.0, 0.0, 1.5]]
S3_AZ_GRID = [-45.0, 0.0, 45.0]

OUT_DIR = Path(__file__).resolve().parents[1] / "oracles" / "s1_s4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def path_gain_db(scene_name: str, tx_pos, ue_pos,
                 az_deg: float = 0.0, dt_deg: float = 0.0,
                 antenna_pattern: str = "iso",
                 max_depth: int = 3, samples_per_src: int = 500_000) -> float:
    s = rt.load_scene(getattr(rt.scene, scene_name))
    s.frequency = FREQ_HZ
    s.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                pattern=antenna_pattern, polarization="V")
    s.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                pattern="iso", polarization="V")
    tx = rt.Transmitter(name="tx", position=list(tx_pos), power_dbm=0.0)
    s.add(tx)
    s.add(rt.Receiver(name="rx", position=list(ue_pos)))
    if antenna_pattern == "tr38901":
        tx.orientation = [math.radians(az_deg), -math.radians(dt_deg), 0.0]
    p = rt.PathSolver()(s, max_depth=max_depth, samples_per_src=samples_per_src)
    a_re, a_im = p.a
    a_re = np.array(a_re).squeeze()
    a_im = np.array(a_im).squeeze()
    valid = np.array(p.valid).squeeze().astype(bool)
    a_mag2 = (a_re ** 2 + a_im ** 2)[valid]
    total = float(a_mag2.sum())
    return round(10.0 * math.log10(total), 3) if total > 0 else -200.0


def jain_fairness(rates: Iterable[float]) -> float:
    r = np.asarray(list(rates), dtype=float)
    if r.size == 0 or r.sum() <= 0:
        return 0.0
    return float((r.sum() ** 2) / (r.size * (r ** 2).sum()))


# ----------------------------------------------------------------------------
# S1: Fixed deployment -> per-user rates, sum throughput, Jain's fairness
# ----------------------------------------------------------------------------
def oracle_s1(tx_power_dbm: float = TX_POWER_DBM_DEFAULT) -> dict:
    print(f"  [S1] 2 APs x 4 UEs, isotropic antenna, TX={tx_power_dbm} dBm")
    pg = np.zeros((len(AP_POSITIONS_2), len(UE_POSITIONS_4)))
    for i, ap in enumerate(AP_POSITIONS_2):
        for j, ue in enumerate(UE_POSITIONS_4):
            pg[i, j] = path_gain_db(SCENE, ap, ue)
    # Convert: received power per (AP, UE) in dBm
    rx_dbm = tx_power_dbm + pg
    # Each UE associates with strongest AP
    serving_ap = np.argmax(rx_dbm, axis=0)
    sinr_db = np.zeros(len(UE_POSITIONS_4))
    rates = np.zeros(len(UE_POSITIONS_4))
    rx_lin = 10 ** (rx_dbm / 10.0)
    noise_lin = 10 ** (NOISE_DBM / 10.0)
    for j in range(len(UE_POSITIONS_4)):
        sig = rx_lin[serving_ap[j], j]
        interf = sum(rx_lin[i, j] for i in range(len(AP_POSITIONS_2))
                     if i != serving_ap[j])
        sinr = sig / (interf + noise_lin)
        sinr_db[j] = 10 * math.log10(max(sinr, 1e-12))
        rates[j] = math.log2(1.0 + sinr)
    return {
        "task_family":     "S1_fixed_deployment",
        "scene_name":      SCENE,
        "ap_positions":    AP_POSITIONS_2,
        "ue_positions":    UE_POSITIONS_4,
        "tx_power_dbm":    tx_power_dbm,
        "noise_dbm":       NOISE_DBM,
        "frequency_hz":    FREQ_HZ,
        "path_gain_db":    pg.round(3).tolist(),
        "serving_ap":      serving_ap.tolist(),
        "sinr_db":         [round(x, 2) for x in sinr_db],
        "per_user_rate_bps_hz": [round(x, 4) for x in rates],
        "sum_throughput_bps_hz": round(float(rates.sum()), 4),
        "mean_throughput_bps_hz": round(float(rates.mean()), 4),
        "fairness_index":  round(jain_fairness(rates), 4),
        "method":          "sionna_rt",
        "sionna_version":  sionna.__version__,
    }


# ----------------------------------------------------------------------------
# S2: 1 AP + 4 UEs, proportional-fair scheduler over T time slots
# ----------------------------------------------------------------------------
def oracle_s2(tx_power_dbm: float = TX_POWER_DBM_DEFAULT,
              T: int = 10, alpha: float = 0.1) -> dict:
    print(f"  [S2] 1 AP x 4 UEs, PF scheduler, T={T} slots, alpha={alpha}")
    pg = np.array([path_gain_db(SCENE, AP_POSITIONS_1[0], ue)
                   for ue in UE_POSITIONS_4])
    rx_dbm = tx_power_dbm + pg
    sinr_lin = 10 ** ((rx_dbm - NOISE_DBM) / 10.0)
    inst_rate = np.log2(1.0 + sinr_lin)  # bps/Hz per UE if served alone
    # PF: schedule UE = argmax(inst_rate / avg_rate); avg_rate updates EMA.
    avg = np.ones_like(inst_rate) * 1e-6
    scheduled = []
    per_user_throughput = np.zeros(len(UE_POSITIONS_4))
    for t in range(T):
        metric = inst_rate / avg
        ue = int(np.argmax(metric))
        scheduled.append(ue)
        per_user_throughput[ue] += inst_rate[ue]
        # EMA update for all UEs (served UE += rate, others += 0)
        served_rate = np.where(np.arange(len(avg)) == ue, inst_rate[ue], 0.0)
        avg = (1 - alpha) * avg + alpha * served_rate
    per_user_throughput /= T   # average per slot
    return {
        "task_family":     "S2_fixed_scheduler",
        "scene_name":      SCENE,
        "ap_position":     AP_POSITIONS_1[0],
        "ue_positions":    UE_POSITIONS_4,
        "tx_power_dbm":    tx_power_dbm,
        "noise_dbm":       NOISE_DBM,
        "frequency_hz":    FREQ_HZ,
        "scheduler":       "proportional_fair",
        "alpha":           alpha,
        "num_slots":       T,
        "path_gain_db":    [round(x, 3) for x in pg],
        "sinr_db":         [round(10*math.log10(max(s,1e-12)), 2) for s in sinr_lin],
        "instantaneous_rate_bps_hz": [round(x, 4) for x in inst_rate],
        "scheduled_ue":    scheduled,
        "per_user_throughput_bps_hz": [round(x, 4) for x in per_user_throughput],
        "sum_throughput_bps_hz": round(float(per_user_throughput.sum()), 4),
        "fairness_index":  round(jain_fairness(per_user_throughput), 4),
        "method":          "sionna_rt",
        "sionna_version":  sionna.__version__,
    }


# ----------------------------------------------------------------------------
# S3: Joint beam optimisation over 3x3 azimuth grid -> max sum rate
# ----------------------------------------------------------------------------
def oracle_s3(tx_power_dbm: float = +5.0) -> dict:
    print(f"  [S3] 2 APs x 2 UEs, joint az sweep 3x3, TX={tx_power_dbm} dBm")
    rows = []
    pg_cache = {}
    for az1 in S3_AZ_GRID:
        for az2 in S3_AZ_GRID:
            # path gains: 2x2 (AP, UE) for this (az1, az2)
            pg = np.zeros((2, 2))
            for i, ap in enumerate(S3_AP_POSITIONS):
                az = az1 if i == 0 else az2
                for j, ue in enumerate(S3_UE_POSITIONS):
                    key = (i, j, az)
                    if key not in pg_cache:
                        pg_cache[key] = path_gain_db(
                            SCENE, ap, ue, az_deg=az, antenna_pattern="tr38901")
                    pg[i, j] = pg_cache[key]
            rx_lin = 10 ** ((tx_power_dbm + pg) / 10.0)
            noise_lin = 10 ** (NOISE_DBM / 10.0)
            # Assume AP_i serves UE_i (matched pair); other AP -> interference.
            sinr1 = rx_lin[0, 0] / (rx_lin[1, 0] + noise_lin)
            sinr2 = rx_lin[1, 1] / (rx_lin[0, 1] + noise_lin)
            r1, r2 = math.log2(1 + sinr1), math.log2(1 + sinr2)
            rows.append({
                "az1_deg":    az1,
                "az2_deg":    az2,
                "path_gain_db": pg.round(3).tolist(),
                "sinr_db_ue1": round(10*math.log10(max(sinr1, 1e-12)), 2),
                "sinr_db_ue2": round(10*math.log10(max(sinr2, 1e-12)), 2),
                "rate_ue1":   round(r1, 4),
                "rate_ue2":   round(r2, 4),
                "sum_rate":   round(r1 + r2, 4),
            })
    best = max(rows, key=lambda r: r["sum_rate"])
    return {
        "task_family":     "S3_joint_beamforming",
        "scene_name":      SCENE,
        "ap_positions":    S3_AP_POSITIONS,
        "ue_positions":    S3_UE_POSITIONS,
        "tx_power_dbm":    tx_power_dbm,
        "noise_dbm":       NOISE_DBM,
        "frequency_hz":    FREQ_HZ,
        "antenna_pattern": "tr38901",
        "az_grid_deg":     S3_AZ_GRID,
        "sweep_table":     rows,
        "best":            {"az1_deg": best["az1_deg"],
                              "az2_deg": best["az2_deg"],
                              "sum_rate": best["sum_rate"]},
        "method":          "sionna_rt",
        "sionna_version":  sionna.__version__,
    }


# ----------------------------------------------------------------------------
# S4: RB allocation Pareto in (sum_rate, fairness)
# ----------------------------------------------------------------------------
def oracle_s4(tx_power_dbm: float = TX_POWER_DBM_DEFAULT,
              total_rb: int = 50) -> dict:
    print(f"  [S4] 1 AP x 4 UEs, RB allocation Pareto, total_RB={total_rb}")
    pg = np.array([path_gain_db(SCENE, AP_POSITIONS_1[0], ue)
                   for ue in UE_POSITIONS_4])
    rx_dbm = tx_power_dbm + pg
    sinr_lin = 10 ** ((rx_dbm - NOISE_DBM) / 10.0)
    se = np.log2(1.0 + sinr_lin)   # bps/Hz per UE if all bandwidth assigned

    def rate_from_alloc(alloc):
        # Each UE i gets alloc[i] / total_rb fraction of bandwidth
        return (np.asarray(alloc) / total_rb) * se

    # Build 5 named allocations
    sorted_idx = np.argsort(-se)   # best SNR first
    n = len(UE_POSITIONS_4)
    allocs = {}
    # equal
    base, rem = divmod(total_rb, n)
    eq = np.full(n, base); eq[:rem] += 1
    allocs["equal"] = eq.tolist()
    # max-rate: all RBs to best-SNR UE
    mr = np.zeros(n, dtype=int); mr[sorted_idx[0]] = total_rb
    allocs["max_rate"] = mr.tolist()
    # max-min: balance so all UE rates equal (alloc_i ∝ 1/se_i)
    weights = 1.0 / se
    weights /= weights.sum()
    mm = np.maximum(np.round(weights * total_rb).astype(int), 1)
    mm[np.argmax(mm)] -= int(mm.sum() - total_rb)   # rebalance to exact total
    allocs["max_min"] = mm.tolist()
    # proportional fair: alloc_i ∝ se_i (gives more to better channels)
    pf_w = se / se.sum()
    pf = np.maximum(np.round(pf_w * total_rb).astype(int), 1)
    pf[np.argmax(pf)] -= int(pf.sum() - total_rb)
    allocs["proportional_fair"] = pf.tolist()
    # weighted: 25/15/7/3 in sorted-SE order
    w_template = [25, 15, 7, 3]
    w_alloc = np.zeros(n, dtype=int)
    for k, idx in enumerate(sorted_idx):
        w_alloc[idx] = w_template[k] if k < n else 0
    if w_alloc.sum() != total_rb:
        w_alloc[sorted_idx[0]] += int(total_rb - w_alloc.sum())
    allocs["weighted"] = w_alloc.tolist()

    rows = []
    for name, a in allocs.items():
        rates = rate_from_alloc(a)
        rows.append({
            "strategy":           name,
            "allocation":         list(a),
            "per_user_rate_bps_hz": [round(x, 4) for x in rates],
            "sum_rate":           round(float(rates.sum()), 4),
            "fairness":           round(jain_fairness(rates), 4),
        })

    # Pareto frontier in (max sum_rate, max fairness)
    par_idx = []
    for i, ri in enumerate(rows):
        dominated = False
        for j, rj in enumerate(rows):
            if i == j: continue
            ge = (rj["sum_rate"] >= ri["sum_rate"] - 1e-9 and
                  rj["fairness"] >= ri["fairness"] - 1e-9)
            sb = (rj["sum_rate"] > ri["sum_rate"] + 1e-9 or
                  rj["fairness"] > ri["fairness"] + 1e-9)
            if ge and sb:
                dominated = True; break
        if not dominated:
            par_idx.append(i)

    return {
        "task_family":     "S4_rb_pareto",
        "scene_name":      SCENE,
        "ap_position":     AP_POSITIONS_1[0],
        "ue_positions":    UE_POSITIONS_4,
        "tx_power_dbm":    tx_power_dbm,
        "noise_dbm":       NOISE_DBM,
        "total_rb":        total_rb,
        "path_gain_db":    [round(x, 3) for x in pg],
        "per_user_se_bps_hz": [round(x, 4) for x in se],
        "sweep_table":     rows,
        "pareto_set":      par_idx,
        "method":          "sionna_rt",
        "sionna_version":  sionna.__version__,
    }


def main() -> int:
    print(f"sionna {sionna.__version__}  oracle root: {OUT_DIR}")
    t0 = time.time()
    for name, fn in (("s1", oracle_s1), ("s2", oracle_s2),
                     ("s3", oracle_s3), ("s4", oracle_s4)):
        print(f"\n=== {name.upper()} ===")
        out = fn()
        (OUT_DIR / f"{name}.json").write_text(json.dumps(out, indent=2))
        # short summary
        if name == "s1":
            print(f"  serving_ap={out['serving_ap']}  sum={out['sum_throughput_bps_hz']:.2f}"
                  f"  fair={out['fairness_index']:.3f}")
        elif name == "s2":
            print(f"  scheduled={out['scheduled_ue']}  sum={out['sum_throughput_bps_hz']:.2f}"
                  f"  fair={out['fairness_index']:.3f}")
        elif name == "s3":
            b = out['best']
            print(f"  best=(az1={b['az1_deg']:+.0f}, az2={b['az2_deg']:+.0f})"
                  f"  sum_rate={b['sum_rate']:.2f}")
        elif name == "s4":
            print(f"  pareto_set size={len(out['pareto_set'])}  "
                  f"  strategies: {[r['strategy'] for r in out['sweep_table']]}")
    print(f"\nTotal oracle time: {time.time()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
