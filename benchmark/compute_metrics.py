"""compute_metrics.py — reviewer-requested quantitative metrics.

For every trial under benchmark/results/**/simulation_result.json, compute
the 5 continuous metrics (path-gain MAE, RSS grid MAE, SINR error,
BER log-domain error, throughput relative error) against the oracle.

Only trials with a matching oracle are scored; the rest are silently
skipped (they land in the "no oracle available" bucket).

Output: benchmark/metrics_per_trial.csv  (one row per trial)
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results"
ORACLES = ROOT / "benchmark" / "oracles"
TASK_MAP = ROOT / "scratchpad" / "task_oracle_map.json"


# ─────────────────────── helpers ────────────────────────

def _load_json(p: Path | str) -> dict | None:
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _first(d: dict, *keys, default=None):
    """Return the first key present in d (recursive one level)."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    nm = d.get("numerical_metrics") if isinstance(d.get("numerical_metrics"), dict) else None
    if nm:
        for k in keys:
            if k in nm and nm[k] is not None:
                return nm[k]
    return default


def _as_num(x) -> float | None:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _as_list(x) -> list | None:
    if isinstance(x, list):
        return x
    return None


def _log_ber(v):
    """Log10 with floor at 1e-8 so BER=0 doesn't blow up."""
    if v is None:
        return None
    return math.log10(max(float(v), 1e-8))


# ─────────────────── per-family scorers ─────────────────
# Each returns a dict with any of:
#   path_gain_mae_db, rss_grid_mae_db, sinr_err_db,
#   ber_log_err, bler_log_err, throughput_re_pct
# Missing keys just aren't reported for that trial.

def score_p1(agent: dict, oracle: dict) -> dict:
    out: dict[str, float] = {}
    # Path-gain MAE across the 15-config sweep grid.
    a_sw = agent.get("sweep_table") or []
    o_sw = oracle.get("sweep_table") or []
    if a_sw and o_sw:
        # Key by (round(az,1), round(dt,1))
        o_map = {(round(float(r.get("az_deg", 1e9)), 1),
                  round(float(r.get("dt_deg", 1e9)), 1)):
                 _as_num(r.get("path_gain_db")) for r in o_sw}
        diffs = []
        for r in a_sw:
            k = (round(float(r.get("az_deg", 1e9)), 1),
                 round(float(r.get("dt_deg", 1e9)), 1))
            apg = _as_num(r.get("path_gain_db"))
            rpg = o_map.get(k)
            if apg is None or rpg is None:
                continue
            if apg <= -180 or rpg <= -180:  # "no path" sentinel
                continue
            diffs.append(abs(apg - rpg))
        if diffs:
            out["path_gain_mae_db"] = float(np.mean(diffs))

    # Throughput at the agent's reported optimum vs oracle's optimum.
    a_best = agent.get("best") or {}
    o_best = oracle.get("best") or {}
    a_t = _as_num(a_best.get("throughput"))
    o_t = _as_num(o_best.get("throughput"))
    if a_t is not None and o_t is not None and o_t > 1e-6:
        out["throughput_re_pct"] = 100.0 * abs(a_t - o_t) / o_t
    return out


def score_p2(agent: dict, oracle: dict) -> dict:
    out: dict[str, float] = {}
    a_sw = agent.get("sweep_table") or []
    o_sw = oracle.get("sweep_table") or []
    if a_sw and o_sw:
        # Key by (az, tx_power)
        def key(r):
            return (round(float(r.get("az_deg", 1e9)), 1),
                    round(float(r.get("tx_power_dbm", 1e9)), 1))
        o_map = {key(r): _as_num(r.get("path_gain_db")) for r in o_sw}
        diffs = []
        for r in a_sw:
            apg = _as_num(r.get("path_gain_db"))
            rpg = o_map.get(key(r))
            if apg is None or rpg is None: continue
            if apg <= -180 or rpg <= -180: continue
            diffs.append(abs(apg - rpg))
        if diffs:
            out["path_gain_mae_db"] = float(np.mean(diffs))

    # Throughput RE = |max(agent frontier) - max(oracle frontier)| / oracle_max
    def max_thr(sw):
        vals = [_as_num(r.get("throughput")) for r in sw]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    a_max = max_thr(a_sw)
    o_max = max_thr(o_sw)
    if a_max is not None and o_max is not None and o_max > 1e-6:
        out["throughput_re_pct"] = 100.0 * abs(a_max - o_max) / o_max
    return out


def score_n1_probe(agent: dict, oracle_json: dict, oracle_npy: Path | None,
                    workdir: Path) -> dict:
    """N1 = single-AP coverage probe.

    Grid MAE if agent produced coverage_*.npy AND we have oracle .npy.
    Fallback to scalar rss_dbm_mean comparison if grid unavailable.
    """
    out: dict[str, float] = {}
    # Grid MAE — look for any agent coverage_*.npy in the workdir
    if oracle_npy and oracle_npy.exists():
        try:
            ref_grid = np.load(oracle_npy)
            # Candidate agent grids
            cands = list(workdir.glob("coverage*.npy")) + \
                    list(workdir.glob("*coverage*.npy"))
            # Prefer file whose shape matches oracle
            for f in cands:
                try:
                    g = np.load(f)
                    if g.shape != ref_grid.shape:
                        continue
                    both = np.isfinite(g) & np.isfinite(ref_grid)
                    if both.sum() >= 100:
                        mae = float(np.abs(g[both] - ref_grid[both]).mean())
                        out["rss_grid_mae_db"] = mae
                        break
                except Exception:
                    continue
        except Exception:
            pass

    # Scalar fallback — rss_dbm_mean
    if "rss_grid_mae_db" not in out:
        a_mean = _first(agent, "rss_dbm_mean", "mean_rss_dbm")
        o_mean = _first(oracle_json, "rss_dbm_mean", "mean_rss_dbm")
        if a_mean is not None and o_mean is not None:
            out["rss_grid_mae_db"] = abs(float(a_mean) - float(o_mean))
    return out


def score_n2(agent: dict, oracle_json: dict, oracle_npy: Path | None,
             workdir: Path) -> dict:
    """N2 = coverage edit (before/after). Grid MAE on the AFTER grid."""
    out: dict[str, float] = {}
    if oracle_npy and oracle_npy.exists():
        try:
            ref = np.load(oracle_npy)
            for f in workdir.glob("*.npy"):
                try:
                    g = np.load(f)
                    if g.shape != ref.shape: continue
                    both = np.isfinite(g) & np.isfinite(ref)
                    if both.sum() >= 100:
                        out["rss_grid_mae_db"] = float(
                            np.abs(g[both] - ref[both]).mean())
                        break
                except Exception:
                    continue
        except Exception:
            pass
    if "rss_grid_mae_db" not in out:
        # scalar: after.rss_dbm_mean vs oracle rss_dbm_mean
        a_after = agent.get("after") or {}
        a_mean = _first(a_after, "rss_dbm_mean") or \
                 _first(agent, "rss_dbm_mean_after", "rss_dbm_mean")
        o_mean = _first(oracle_json, "rss_dbm_mean")
        if a_mean is not None and o_mean is not None:
            out["rss_grid_mae_db"] = abs(float(a_mean) - float(o_mean))
    return out


def score_n3(agent: dict, oracle_dir: Path, workdir: Path) -> dict:
    """N3 = multi-AP. Oracle is a DIR with best_server.npy + coverage_ap_*.npy."""
    out: dict[str, float] = {}
    ref_bs = oracle_dir / "best_server.npy"
    if ref_bs.exists():
        try:
            ref = np.load(ref_bs)
            for name in ("coverage_best_server.npy", "best_server.npy",
                          "coverage_serving.npy"):
                f = workdir / name
                if not f.exists(): continue
                try:
                    g = np.load(f)
                    if g.shape != ref.shape: continue
                    both = np.isfinite(g) & np.isfinite(ref)
                    if both.sum() >= 100:
                        out["rss_grid_mae_db"] = float(
                            np.abs(g[both] - ref[both]).mean())
                        break
                except Exception:
                    continue
        except Exception:
            pass
    if "rss_grid_mae_db" not in out:
        # Scalar: best_server_rss_dbm_mean
        a_mean = _first(agent, "best_server_rss_dbm_mean") or \
                 (agent.get("best_server_metrics") or {}).get("rss_dbm_mean")
        # Oracle metadata
        meta = _load_json(oracle_dir / "metadata.json") or {}
        o_mean = _first(meta, "best_server_rss_dbm_mean", "rss_dbm_mean")
        if a_mean is not None and o_mean is not None:
            out["rss_grid_mae_db"] = abs(float(a_mean) - float(o_mean))
    return out


def score_n4(agent: dict, oracle: dict) -> dict:
    """N4 = PHY link. Route by oracle.metric_type."""
    out: dict[str, float] = {}
    o_metric = (oracle.get("metric_type") or "").lower()
    o_ebn0 = oracle.get("eb_n0_db") or oracle.get("ebn0_db")
    o_vals = oracle.get("metric_values")
    a_ebn0 = _first(agent, "eb_n0_db", "ebn0_db")

    # Pick the agent field that matches the oracle metric type.
    if o_metric == "bler":
        a_vals = _first(agent, "bler_simulated", "bler")
    elif o_metric == "throughput":
        a_vals = _first(agent, "throughput", "throughput_simulated")
    else:  # default: BER
        a_vals = _first(agent, "ber_simulated", "ber")

    if a_ebn0 and o_ebn0 and o_vals and isinstance(a_vals, list) \
            and isinstance(a_ebn0, list):
        # Align by nearest-EbN0 (tolerance 0.5 dB) and score.
        errs_log = []      # log10-abs for BER/BLER
        errs_rel = []      # relative-abs for throughput
        for i, e_o in enumerate(o_ebn0):
            if i >= len(o_vals): break
            for j, e_a in enumerate(a_ebn0):
                if j >= len(a_vals): break
                if abs(float(e_a) - float(e_o)) <= 0.5:
                    av, rv = _as_num(a_vals[j]), _as_num(o_vals[i])
                    if av is None or rv is None: break
                    if o_metric == "throughput":
                        if rv > 1e-6:
                            errs_rel.append(100.0 * abs(av - rv) / rv)
                    else:
                        la, lr = _log_ber(av), _log_ber(rv)
                        if la is not None and lr is not None:
                            errs_log.append(abs(la - lr))
                    break
        if errs_log:
            key = "bler_log_err" if o_metric == "bler" else "ber_log_err"
            out[key] = float(np.mean(errs_log))
        if errs_rel:
            out["throughput_re_pct"] = float(np.mean(errs_rel))
    return out


def score_s1(agent: dict, oracle: dict) -> dict:
    """S1 = fixed deployment. SINR per user + sum throughput."""
    out: dict[str, float] = {}
    a_sinr = _as_list(_first(agent, "sinr_db"))
    o_sinr = _as_list(_first(oracle, "sinr_db"))
    if a_sinr and o_sinr and len(a_sinr) == len(o_sinr):
        diffs = [abs(float(a) - float(r)) for a, r in zip(a_sinr, o_sinr)
                 if _as_num(a) is not None and _as_num(r) is not None]
        if diffs:
            out["sinr_err_db"] = float(np.mean(diffs))
    a_sum = _first(agent, "sum_throughput_bps_hz", "sum_rate_bps_hz")
    o_sum = _first(oracle, "sum_throughput_bps_hz", "sum_rate_bps_hz")
    if a_sum is not None and o_sum is not None and float(o_sum) > 1e-6:
        out["throughput_re_pct"] = 100.0 * abs(float(a_sum) - float(o_sum)) / float(o_sum)
    return out


def score_s2(agent: dict, oracle: dict) -> dict:
    """S2 = PF scheduler. Same signals as S1 + path-gain."""
    out = score_s1(agent, oracle)
    # Path-gain MAE (per-UE, list of 4)
    a_pg = _as_list(_first(agent, "path_gain_db"))
    o_pg = _as_list(_first(oracle, "path_gain_db"))
    if a_pg and o_pg and len(a_pg) == len(o_pg):
        diffs = [abs(float(a) - float(r)) for a, r in zip(a_pg, o_pg)
                 if _as_num(a) is not None and _as_num(r) is not None]
        if diffs:
            out["path_gain_mae_db"] = float(np.mean(diffs))
    return out


def score_s3(agent: dict, oracle: dict) -> dict:
    """S3 = joint beamforming sweep."""
    out: dict[str, float] = {}
    a_sw = agent.get("sweep_table") or []
    o_sw = oracle.get("sweep_table") or []
    if a_sw and o_sw:
        def key(r):
            return (round(float(r.get("az1_deg", 1e9)), 1),
                    round(float(r.get("az2_deg", 1e9)), 1))
        # Oracle SINR keys: sinr_db_ue1, sinr_db_ue2 (per-config)
        # Agent  SINR keys: sinr_ue0_db, sinr_ue1_db (per-config)
        def o_sinr_pair(r):
            u1 = _as_num(r.get("sinr_db_ue1"))
            u2 = _as_num(r.get("sinr_db_ue2"))
            return [x for x in (u1, u2) if x is not None]

        def a_sinr_pair(r):
            u0 = _as_num(r.get("sinr_ue0_db"))
            u1 = _as_num(r.get("sinr_ue1_db"))
            return [x for x in (u0, u1) if x is not None]

        o_sinr_map = {key(r): o_sinr_pair(r) for r in o_sw}
        sinr_diffs = []
        for r in a_sw:
            k = key(r)
            os_ = o_sinr_map.get(k) or []
            as_ = a_sinr_pair(r)
            if len(os_) == len(as_) and os_:
                sinr_diffs.extend([abs(a - r_) for a, r_ in zip(as_, os_)])
        if sinr_diffs:
            out["sinr_err_db"] = float(np.mean(sinr_diffs))

    # Throughput RE via best.sum_rate
    a_best = agent.get("best") or {}
    o_best = oracle.get("best") or {}
    a_sr = _as_num(a_best.get("sum_rate"))
    o_sr = _as_num(o_best.get("sum_rate"))
    if a_sr is not None and o_sr is not None and o_sr > 1e-6:
        out["throughput_re_pct"] = 100.0 * abs(a_sr - o_sr) / o_sr
    return out


def score_s4(agent: dict, oracle: dict) -> dict:
    """S4 = RB Pareto."""
    out: dict[str, float] = {}
    a_pg = _as_list(_first(agent, "path_gain_db"))
    o_pg = _as_list(_first(oracle, "path_gain_db"))
    if a_pg and o_pg and len(a_pg) == len(o_pg):
        diffs = [abs(float(a) - float(r)) for a, r in zip(a_pg, o_pg)
                 if _as_num(a) is not None and _as_num(r) is not None]
        if diffs:
            out["path_gain_mae_db"] = float(np.mean(diffs))
    # Throughput RE = best sum_rate in agent sweep_table vs oracle
    def best_sr(sw):
        vals = [_as_num(r.get("sum_rate")) for r in (sw or [])]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    a_sr = best_sr(agent.get("sweep_table"))
    o_sr = best_sr(oracle.get("sweep_table"))
    if a_sr is not None and o_sr is not None and o_sr > 1e-6:
        out["throughput_re_pct"] = 100.0 * abs(a_sr - o_sr) / o_sr
    return out


# ─────────────────── dispatcher ─────────────────────────

def score_trial(task_id: str, oracle_path: str | None, agent: dict,
                workdir: Path) -> dict[str, float]:
    """Route to the right family scorer based on task_id prefix."""
    if not task_id:
        return {}
    prefix = task_id.split("_")[0]
    op = Path(oracle_path) if oracle_path else None
    if op and not op.is_absolute():
        op = ROOT / op

    try:
        if prefix == "P1" and op and op.exists():
            oracle = _load_json(op) or {}
            return score_p1(agent, oracle)
        if prefix == "P2" and op and op.exists():
            oracle = _load_json(op) or {}
            return score_p2(agent, oracle)
        if prefix == "N4" and op and op.exists():
            oracle = _load_json(op) or {}
            return score_n4(agent, oracle)
        if prefix == "N1":
            # Task source's oracle_path may be .npy; we also want the .json sibling
            npy = op if (op and op.suffix == ".npy") else None
            j_path = (op.with_suffix(".json") if op else None)
            if not (j_path and j_path.exists()):
                # try derive from task_id
                scene = task_id.replace("N1_", "").replace("probe_", "")
                j_path = ORACLES / "n1" / f"{scene}.json"
                npy = ORACLES / "n1" / f"{scene}.npy"
            j = _load_json(j_path) or {}
            return score_n1_probe(agent, j, npy, workdir)
        if prefix == "N2":
            # oracle_path may be missing; derive from task_id
            # tasks like N2_freq_box_one_screen or N2_edit_box_one_screen
            scene_and_suf = task_id.replace("N2_", "")
            if "freq" in scene_and_suf:
                scene = scene_and_suf.replace("freq_", "")
                cand_j = ORACLES / "n2_freq" / f"{scene}_2ghz.json"
                cand_npy = ORACLES / "n2_freq" / f"{scene}_2ghz.npy"
            else:
                # try _edit
                scene = scene_and_suf.replace("edit_", "")
                # Pick any *_after file for the scene
                cand_j = None; cand_npy = None
                for f in (ORACLES / "n2_edit").glob(f"{scene}_*_after.json"):
                    cand_j = f
                    cand_npy = f.with_suffix(".npy")
                    break
            j = _load_json(cand_j) if cand_j and cand_j.exists() else {}
            return score_n2(agent, j or {}, cand_npy, workdir)
        if prefix == "N3":
            scene = task_id.replace("N3_2ap_", "")
            odir = ORACLES / "n3_multi_ap" / scene
            if odir.is_dir():
                return score_n3(agent, odir, workdir)
        if prefix == "S1":
            oracle = _load_json(op) if op and op.exists() else None
            if oracle is None:
                scene = task_id.replace("S1_", "")
                oracle = _load_json(ORACLES / "s1_s4" / scene / "s1.json") or {}
            return score_s1(agent, oracle)
        if prefix == "S2":
            oracle = _load_json(op) if op and op.exists() else None
            if oracle is None:
                # task ids like S2_fixed_scheduler use the top-level oracle
                oracle = _load_json(ORACLES / "s1_s4" / "s2.json") or \
                          _load_json(ORACLES / "s1_s4" / "box_one_screen" / "s2.json") or {}
            return score_s2(agent, oracle)
        if prefix == "S3":
            oracle = _load_json(op) if op and op.exists() else None
            if oracle is None:
                oracle = _load_json(ORACLES / "s1_s4" / "s3.json") or \
                          _load_json(ORACLES / "s1_s4" / "box_one_screen" / "s3.json") or {}
            return score_s3(agent, oracle)
        if prefix == "S4":
            oracle = _load_json(op) if op and op.exists() else None
            if oracle is None:
                oracle = _load_json(ORACLES / "s1_s4" / "s4.json") or \
                          _load_json(ORACLES / "s1_s4" / "box_one_screen" / "s4.json") or {}
            return score_s4(agent, oracle)
    except Exception as e:
        return {"_error": str(e)[:80]}
    return {}


# ─────────────────── main scan ──────────────────────────

# ─────────────────── layer classification ───────────────
# Route each verifier sub-check into one of the 3 layers.
# Layer 1 = artifact presence + schema check
# Layer 2 = executable, used-sionna, geometry sanity
# Layer 3 = oracle numerical match (the source of our continuous distances)

_L2_NAMES = {"sionna_rt_used", "sionna_phy_used", "sionna_loadable",
             "collision_free", "in_bounds", "scene_nontrivial"}


def _classify_check(name: str) -> str:
    """Return 'L1', 'L2', or 'L3' for a check name."""
    if not name:
        return "L?"
    if name.startswith("artifact:"):
        return "L1"
    if name.startswith("threshold:") or name.startswith("range:") or \
       name.startswith("monotonic:"):
        return "L3"
    if name.endswith("_oracle"):
        return "L3"
    if name in _L2_NAMES:
        return "L2"
    if name.startswith("code_contains:"):
        return "L2"
    return "L?"


def _layer_pass(checks: list) -> dict:
    """Compute per-layer pass rate for a single trial.

    A layer PASSES if every sub-check assigned to it passes. Missing
    layers (no sub-checks) are None (not counted in the mean).
    """
    by_layer: dict[str, list[bool]] = {"L1": [], "L2": [], "L3": []}
    for c in checks or []:
        lay = _classify_check(c.get("name", ""))
        if lay in by_layer:
            by_layer[lay].append(bool(c.get("passed")))
    out: dict[str, bool | None] = {}
    for lay, vals in by_layer.items():
        out[f"{lay}_pass"] = (all(vals) if vals else None)
    return out


def infer_trial_meta(sim_path: Path) -> dict:
    """Extract study/condition/task_id/trial_id/pass/layer-pass from the trial dir."""
    # path pattern: benchmark/results/<study>/<condition>/<task_id>/t<n>/simulation_result.json
    parts = sim_path.parts
    try:
        idx = parts.index("results")
        study = parts[idx + 1]
        condition = parts[idx + 2]
        task_id = parts[idx + 3]
        trial_dir = parts[idx + 4]
    except Exception:
        return {}
    trial_id = re.sub(r"^t", "", trial_dir)

    result_path = sim_path.parent / "result.json"
    res = _load_json(result_path) or {}
    passed = res.get("verification", {}).get("passed", None)
    checks = res.get("verification", {}).get("checks", [])
    layers = _layer_pass(checks)
    condition_meta = res.get("condition", condition)
    return {
        "study": study,
        "condition": condition_meta,
        "task_id": task_id,
        "tier": task_id.split("_")[0],
        "trial_id": trial_id,
        "passed": passed,
        "L1_pass": layers["L1_pass"],
        "L2_pass": layers["L2_pass"],
        "L3_pass": layers["L3_pass"],
        "workdir": sim_path.parent,
    }


def main():
    task_map = json.loads(TASK_MAP.read_text()) if TASK_MAP.exists() else {}
    print(f"loaded task_id → oracle_path map: {len(task_map)} entries",
          file=sys.stderr)

    sim_files = sorted(glob(str(RESULTS / "**" / "simulation_result.json"),
                              recursive=True))
    print(f"scanning {len(sim_files)} simulation_result.json files",
          file=sys.stderr)

    rows = []
    stats = {"total": 0, "with_metrics": 0, "no_oracle": 0, "error": 0}
    for sp in sim_files:
        sp = Path(sp)
        stats["total"] += 1
        meta = infer_trial_meta(sp)
        if not meta:
            continue
        agent = _load_json(sp)
        if not agent:
            continue
        # skip harness pre-shipped skeleton
        if agent.get("status") == "placeholder_pre_shipped_by_harness":
            continue

        oracle_path = task_map.get(meta["task_id"])
        metrics = score_trial(meta["task_id"], oracle_path, agent, meta["workdir"])
        if not metrics:
            stats["no_oracle"] += 1
        elif "_error" in metrics:
            stats["error"] += 1
        else:
            stats["with_metrics"] += 1

        rows.append({
            "study": meta["study"],
            "condition": meta["condition"],
            "tier": meta["tier"],
            "task_id": meta["task_id"],
            "trial_id": meta["trial_id"],
            "passed": meta["passed"],
            "L1_pass": meta["L1_pass"],
            "L2_pass": meta["L2_pass"],
            "L3_pass": meta["L3_pass"],
            "path_gain_mae_db": metrics.get("path_gain_mae_db"),
            "rss_grid_mae_db": metrics.get("rss_grid_mae_db"),
            "sinr_err_db": metrics.get("sinr_err_db"),
            "ber_log_err": metrics.get("ber_log_err"),
            "bler_log_err": metrics.get("bler_log_err"),
            "throughput_re_pct": metrics.get("throughput_re_pct"),
            "error": metrics.get("_error"),
        })

    out_csv = ROOT / "benchmark" / "metrics_per_trial.csv"
    if rows:
        keys = list(rows[0].keys())
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"wrote {out_csv}  rows={len(rows)}", file=sys.stderr)
    print(f"stats: {stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
