"""Pre-compute S1-S4 oracles for all 4 built-in scenes.

Each scene gets its own (AP, UE) topology + TX-power calibration so the
SINR/rate values fall in a meaningful range. Output is per-scene
sub-directory: benchmark/oracles/s1_s4/{scene}/{s_id}.json
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

# Per-scene topology + TX-power calibration.
# Keys: ap_2, ap_1, ue_4, ap_pair (S3), ue_pair (S3),
#       tx_dbm_s1/s2/s4, tx_dbm_s3 (S3 uses tr38901 directional).
SCENES = {
    "box_one_screen": {
        "ap_2":     [[-2.0, 0.0, 4.5], [+2.0, 0.0, 4.5]],
        "ap_1":     [0.0, 0.0, 4.5],
        "ue_4":     [[-2.0, -1.5, 1.5], [-2.0, +1.5, 1.5],
                     [+2.0, -1.5, 1.5], [+2.0, +1.5, 1.5]],
        "ap_pair":  [[-2.0, 0.0, 4.5], [+2.0, 0.0, 4.5]],
        "ue_pair":  [[-2.0, -2.0, 1.5], [+2.0, +2.0, 1.5]],
        "tx_dbm_s1": -25.0,
        "tx_dbm_s3": -15.0,
    },
    "box_two_screens": {
        "ap_2":     [[-2.0, 0.0, 4.5], [+2.0, 0.0, 4.5]],
        "ap_1":     [0.0, 0.0, 4.5],
        "ue_4":     [[-2.0, -1.5, 1.5], [-2.0, +1.5, 1.5],
                     [+2.0, -1.5, 1.5], [+2.0, +1.5, 1.5]],
        "ap_pair":  [[-2.0, 0.0, 4.5], [+2.0, 0.0, 4.5]],
        "ue_pair":  [[-2.0, -2.0, 1.5], [+2.0, +2.0, 1.5]],
        "tx_dbm_s1": -25.0,
        "tx_dbm_s3": -15.0,
    },
    "simple_street_canyon": {
        "ap_2":     [[-15.0, 0.0, 15.0], [+15.0, 0.0, 15.0]],
        "ap_1":     [0.0, 0.0, 15.0],
        "ue_4":     [[-20.0, 0.0, 1.5], [-5.0, 0.0, 1.5],
                     [+5.0, 0.0, 1.5], [+20.0, 0.0, 1.5]],
        "ap_pair":  [[-15.0, 0.0, 15.0], [+15.0, 0.0, 15.0]],
        "ue_pair":  [[-10.0, 0.0, 1.5], [+10.0, 0.0, 1.5]],
        "tx_dbm_s1": 0.0,
        "tx_dbm_s3": +5.0,
    },
    "etoile": {
        "ap_2":     [[-50.0, 0.0, 30.0], [+50.0, 0.0, 30.0]],
        "ap_1":     [0.0, 0.0, 30.0],
        "ue_4":     [[-70.0, 0.0, 1.5], [-20.0, 0.0, 1.5],
                     [+20.0, 0.0, 1.5], [+70.0, 0.0, 1.5]],
        "ap_pair":  [[-30.0, 0.0, 30.0], [+30.0, 0.0, 30.0]],
        "ue_pair":  [[-15.0, 0.0, 1.5], [+15.0, 0.0, 1.5]],
        "tx_dbm_s1": +10.0,
        "tx_dbm_s3": +15.0,
    },
}

S3_AZ_GRID = [-45.0, 0.0, 45.0]
OUT_ROOT = Path(__file__).resolve().parents[1] / "oracles" / "s1_s4"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def path_gain_db(scene_name, tx_pos, ue_pos,
                 az_deg=0.0, dt_deg=0.0, pattern="iso",
                 max_depth=3, samples_per_src=500_000):
    s = rt.load_scene(getattr(rt.scene, scene_name))
    s.frequency = FREQ_HZ
    s.tx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                pattern=pattern, polarization="V")
    s.rx_array = rt.PlanarArray(num_rows=1, num_cols=1,
                                pattern="iso", polarization="V")
    tx = rt.Transmitter(name="tx", position=list(tx_pos), power_dbm=0.0)
    s.add(tx)
    s.add(rt.Receiver(name="rx", position=list(ue_pos)))
    if pattern == "tr38901":
        tx.orientation = [math.radians(az_deg), -math.radians(dt_deg), 0.0]
    p = rt.PathSolver()(s, max_depth=max_depth, samples_per_src=samples_per_src)
    a_re, a_im = p.a
    a_re = np.array(a_re).squeeze()
    a_im = np.array(a_im).squeeze()
    valid = np.array(p.valid).squeeze().astype(bool)
    a_mag2 = (a_re ** 2 + a_im ** 2)[valid]
    total = float(a_mag2.sum())
    return round(10.0 * math.log10(total), 3) if total > 0 else -200.0


def jain(rates):
    r = np.asarray(rates, float)
    if r.size == 0 or r.sum() <= 0:
        return 0.0
    return float((r.sum() ** 2) / (r.size * (r ** 2).sum()))


def s1(scene, cfg):
    aps, ues = cfg["ap_2"], cfg["ue_4"]
    tx_power = cfg["tx_dbm_s1"]
    pg = np.zeros((len(aps), len(ues)))
    for i, ap in enumerate(aps):
        for j, ue in enumerate(ues):
            pg[i, j] = path_gain_db(scene, ap, ue)
    rx_dbm = tx_power + pg
    serving_ap = np.argmax(rx_dbm, axis=0)
    sinr_db, rates = [], []
    rx_lin = 10 ** (rx_dbm / 10.0)
    noise_lin = 10 ** (NOISE_DBM / 10.0)
    for j in range(len(ues)):
        sig = rx_lin[serving_ap[j], j]
        interf = sum(rx_lin[i, j] for i in range(len(aps)) if i != serving_ap[j])
        sinr = sig / (interf + noise_lin)
        sinr_db.append(round(10 * math.log10(max(sinr, 1e-12)), 2))
        rates.append(math.log2(1 + sinr))
    return {
        "task_family":     "S1_fixed_deployment",
        "scene_name":      scene,
        "ap_positions":    aps,
        "ue_positions":    ues,
        "tx_power_dbm":    tx_power, "noise_dbm": NOISE_DBM, "frequency_hz": FREQ_HZ,
        "path_gain_db":    pg.round(3).tolist(),
        "serving_ap":      serving_ap.tolist(),
        "sinr_db":         sinr_db,
        "per_user_rate_bps_hz": [round(r, 4) for r in rates],
        "sum_throughput_bps_hz": round(float(sum(rates)), 4),
        "mean_throughput_bps_hz": round(float(sum(rates)/len(rates)), 4),
        "fairness_index":  round(jain(rates), 4),
        "method":          "sionna_rt",
        "sionna_version":  sionna.__version__,
    }


def s2(scene, cfg, T=10, alpha=0.1):
    ues = cfg["ue_4"]
    tx_power = cfg["tx_dbm_s1"]
    ap = cfg["ap_1"]
    pg = np.array([path_gain_db(scene, ap, ue) for ue in ues])
    sinr_lin = 10 ** ((tx_power + pg - NOISE_DBM) / 10.0)
    inst = np.log2(1 + sinr_lin)
    avg = np.ones_like(inst) * 1e-6
    sched = []; per_user = np.zeros(len(ues))
    for _ in range(T):
        ue = int(np.argmax(inst / avg))
        sched.append(ue); per_user[ue] += inst[ue]
        served = np.where(np.arange(len(avg)) == ue, inst[ue], 0.0)
        avg = (1 - alpha) * avg + alpha * served
    per_user /= T
    return {
        "task_family":     "S2_fixed_scheduler",
        "scene_name":      scene, "ap_position": ap, "ue_positions": ues,
        "tx_power_dbm":    tx_power, "noise_dbm": NOISE_DBM, "frequency_hz": FREQ_HZ,
        "scheduler":       "proportional_fair", "alpha": alpha, "num_slots": T,
        "path_gain_db":    [round(x, 3) for x in pg],
        "sinr_db":         [round(10*math.log10(max(s,1e-12)),2) for s in sinr_lin],
        "instantaneous_rate_bps_hz": [round(x, 4) for x in inst],
        "scheduled_ue":    sched,
        "per_user_throughput_bps_hz": [round(x, 4) for x in per_user],
        "sum_throughput_bps_hz": round(float(per_user.sum()), 4),
        "fairness_index":  round(jain(per_user), 4),
        "method":          "sionna_rt",
        "sionna_version":  sionna.__version__,
    }


def s3(scene, cfg):
    aps, ues = cfg["ap_pair"], cfg["ue_pair"]
    tx_power = cfg["tx_dbm_s3"]
    rows = []
    cache = {}
    for az1 in S3_AZ_GRID:
        for az2 in S3_AZ_GRID:
            pg = np.zeros((2, 2))
            for i, ap in enumerate(aps):
                az = az1 if i == 0 else az2
                for j, ue in enumerate(ues):
                    key = (i, j, az)
                    if key not in cache:
                        cache[key] = path_gain_db(scene, ap, ue,
                                                  az_deg=az, pattern="tr38901")
                    pg[i, j] = cache[key]
            rx_lin = 10 ** ((tx_power + pg) / 10.0)
            noise_lin = 10 ** (NOISE_DBM / 10.0)
            sinr1 = rx_lin[0, 0] / (rx_lin[1, 0] + noise_lin)
            sinr2 = rx_lin[1, 1] / (rx_lin[0, 1] + noise_lin)
            r1, r2 = math.log2(1 + sinr1), math.log2(1 + sinr2)
            rows.append({"az1_deg": az1, "az2_deg": az2,
                         "path_gain_db": pg.round(3).tolist(),
                         "sinr_db_ue1": round(10*math.log10(max(sinr1,1e-12)), 2),
                         "sinr_db_ue2": round(10*math.log10(max(sinr2,1e-12)), 2),
                         "rate_ue1": round(r1, 4), "rate_ue2": round(r2, 4),
                         "sum_rate": round(r1 + r2, 4)})
    best = max(rows, key=lambda r: r["sum_rate"])
    return {
        "task_family":     "S3_joint_beamforming",
        "scene_name":      scene, "ap_positions": aps, "ue_positions": ues,
        "tx_power_dbm":    tx_power, "noise_dbm": NOISE_DBM, "frequency_hz": FREQ_HZ,
        "antenna_pattern": "tr38901", "az_grid_deg": S3_AZ_GRID,
        "sweep_table":     rows,
        "best":            {"az1_deg": best["az1_deg"], "az2_deg": best["az2_deg"],
                              "sum_rate": best["sum_rate"]},
        "method":          "sionna_rt",
        "sionna_version":  sionna.__version__,
    }


def s4(scene, cfg, total_rb=50):
    ues = cfg["ue_4"]
    ap = cfg["ap_1"]
    tx_power = cfg["tx_dbm_s1"]
    pg = np.array([path_gain_db(scene, ap, ue) for ue in ues])
    sinr_lin = 10 ** ((tx_power + pg - NOISE_DBM) / 10.0)
    se = np.log2(1 + sinr_lin)
    sorted_idx = np.argsort(-se)
    n = len(ues)

    base, rem = divmod(total_rb, n)
    eq = np.full(n, base); eq[:rem] += 1
    allocs = {}
    allocs["equal"] = eq.tolist()
    mr = np.zeros(n, dtype=int); mr[sorted_idx[0]] = total_rb
    allocs["max_rate"] = mr.tolist()
    w_mm = 1.0 / se; w_mm /= w_mm.sum()
    mm = np.maximum(np.round(w_mm * total_rb).astype(int), 1)
    mm[np.argmax(mm)] -= int(mm.sum() - total_rb)
    allocs["max_min"] = mm.tolist()
    w_pf = se / se.sum()
    pf = np.maximum(np.round(w_pf * total_rb).astype(int), 1)
    pf[np.argmax(pf)] -= int(pf.sum() - total_rb)
    allocs["proportional_fair"] = pf.tolist()
    w_template = [25, 15, 7, 3]
    w_alloc = np.zeros(n, dtype=int)
    for k, idx in enumerate(sorted_idx):
        w_alloc[idx] = w_template[k] if k < n else 0
    if w_alloc.sum() != total_rb:
        w_alloc[sorted_idx[0]] += int(total_rb - w_alloc.sum())
    allocs["weighted"] = w_alloc.tolist()

    rows = []
    for name, a in allocs.items():
        rates = (np.asarray(a) / total_rb) * se
        rows.append({"strategy": name, "allocation": list(a),
                     "per_user_rate_bps_hz": [round(x, 4) for x in rates],
                     "sum_rate": round(float(rates.sum()), 4),
                     "fairness": round(jain(rates), 4)})
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
        "scene_name":      scene, "ap_position": ap, "ue_positions": ues,
        "tx_power_dbm":    tx_power, "noise_dbm": NOISE_DBM, "total_rb": total_rb,
        "path_gain_db":    [round(x, 3) for x in pg],
        "per_user_se_bps_hz": [round(x, 4) for x in se],
        "sweep_table":     rows, "pareto_set": par_idx,
        "method":          "sionna_rt",
        "sionna_version":  sionna.__version__,
    }


def main() -> int:
    print(f"sionna {sionna.__version__}  oracle root: {OUT_ROOT}")
    t0 = time.time()
    for scene, cfg in SCENES.items():
        print(f"\n=== {scene} ===")
        scene_dir = OUT_ROOT / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        for name, fn in (("s1", s1), ("s2", s2), ("s3", s3), ("s4", s4)):
            t1 = time.time()
            out = fn(scene, cfg)
            (scene_dir / f"{name}.json").write_text(json.dumps(out, indent=2))
            dt = time.time() - t1
            if name == "s1":
                print(f"  [s1] sum={out['sum_throughput_bps_hz']:.2f}  fair={out['fairness_index']:.3f}  ({dt:.1f}s)")
            elif name == "s2":
                print(f"  [s2] sum={out['sum_throughput_bps_hz']:.2f}  fair={out['fairness_index']:.3f}  ({dt:.1f}s)")
            elif name == "s3":
                b = out['best']
                print(f"  [s3] best=({b['az1_deg']:+.0f},{b['az2_deg']:+.0f})  sum={b['sum_rate']:.2f}  ({dt:.1f}s)")
            elif name == "s4":
                pts = [(r['sum_rate'], r['fairness']) for r in out['sweep_table']]
                print(f"  [s4] pareto={len(out['pareto_set'])}/{len(out['sweep_table'])}  "
                      f"rates={[r['sum_rate'] for r in out['sweep_table']]}  ({dt:.1f}s)")
    print(f"\nTotal oracle time: {time.time()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
