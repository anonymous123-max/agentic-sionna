"""Task-agnostic plausibility checks for Sionna RF simulations.

Extracted from benchmark/verifier.py so the skill's verify_output.py has
no benchmark/ import dependency. The benchmark continues to call this
module too (Task A.5 will make benchmark/verifier.py re-export from here)
— single source of truth, two consumers.

Public surface:
    CheckResult                     dataclass for one named check
    load_sim_result(output_dir)     parse simulation_result.json (or None)
    load_all_code(output_dir)       concatenate every .py the agent wrote
    extract_scalar(sim, metric, v)  best-effort scalar lookup with aliases
    extract_array(sim, metric)      best-effort array lookup
    check_plausibility(output_dir)  BER/coverage/NMSE/RSS/training reality checks

Anything that needs a task spec (metric_threshold, composite, scene
collision/bounds checks) stays in benchmark/verifier.py.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Output parsing helpers
# ─────────────────────────────────────────────────────────────

def load_sim_result(output_dir: Path) -> dict | None:
    """Return the parsed simulation_result.json if present."""
    p = output_dir / "simulation_result.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def load_all_code(output_dir: Path) -> str:
    """Concatenate agent-produced text (.py files plus scene_state.json content)
    for code_contains checks.

    Including scene_state.json content lets grep find furniture/material
    tokens that an agent placed directly in JSON (e.g., via Write rather
    than emitting a .py file). The actual geometric verification still
    parses scene_state.json structurally; this grep is only for capability
    tokens declared in the task spec.
    """
    parts = []
    for p in sorted(output_dir.rglob("*.py")):
        try:
            parts.append(p.read_text(errors="replace"))
        except Exception:
            pass
    for p in sorted(output_dir.rglob("scene_state.json")):
        try:
            parts.append(p.read_text(errors="replace"))
        except Exception:
            pass
    return "\n".join(parts)


def load_bash_commands(output_dir: Path) -> str:
    """Concatenate every Bash tool call's `command` from stdout.txt JSONL.

    Captures Python heredocs (`python -c "..."`) that agents emit instead
    of writing .py files. Used by check_code_contains as a fallback when
    the workdir has no .py files (or as augmentation when it does).
    """
    out = []
    sout = output_dir / "stdout.txt"
    if not sout.exists():
        return ""
    for line in sout.read_text(errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") != "assistant":
            continue
        for c in ev.get("message", {}).get("content", []):
            if c.get("type") == "tool_use" and c.get("name") == "Bash":
                cmd = c.get("input", {}).get("command")
                if isinstance(cmd, str):
                    out.append(cmd)
    return "\n".join(out)


def _find_snr_at_ber(snr: list[float], ber: list[float],
                     target_ber: float) -> float | None:
    """Linear interpolation to find SNR where BER crosses target_ber.

    Assumes ber is monotonically decreasing with SNR. Uses log-space
    interpolation on BER since BER varies over orders of magnitude."""
    import math
    if len(snr) != len(ber) or len(snr) < 2:
        return None
    # Find bracketing pair where ber[i] >= target >= ber[i+1]
    for i in range(len(ber) - 1):
        b0, b1 = ber[i], ber[i+1]
        if b0 <= 0 or b1 <= 0:
            continue
        if b0 >= target_ber >= b1:
            # Log-space interp
            lg0, lg1, lgT = math.log(b0), math.log(b1), math.log(target_ber)
            t = (lgT - lg0) / (lg1 - lg0) if lg1 != lg0 else 0
            return snr[i] + t * (snr[i+1] - snr[i])
    return None


def _find_scalar_anywhere(d: dict, keys: list[str]) -> float | None:
    """Recursively search a nested dict for any key in `keys` with a numeric value.
    Returns the first match found (depth-first)."""
    if not isinstance(d, dict):
        return None
    for k, v in d.items():
        if k in keys and isinstance(v, (int, float)):
            return float(v)
    for v in d.values():
        if isinstance(v, dict):
            found = _find_scalar_anywhere(v, keys)
            if found is not None:
                return found
    return None


def _find_array_anywhere(d: dict, keys: list[str],
                         substring_match: bool = False) -> list[float] | None:
    """Recursively search a nested dict for any key in `keys` (or containing
    any of `keys` as substring if `substring_match=True`) whose value is a
    list of numbers."""
    if not isinstance(d, dict):
        return None
    for k, v in d.items():
        match = (k in keys) or (substring_match and any(s in k.lower() for s in keys))
        if match and isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return [float(x) for x in v]
    for v in d.values():
        if isinstance(v, dict):
            found = _find_array_anywhere(v, keys, substring_match)
            if found is not None:
                return found
    return None


# Alias table: verifier metric -> possible agent output field names
_METRIC_ALIASES: dict[str, list[str]] = {
    "bler_waterfall_range": ["waterfall_ebn0_db", "waterfall_db",
                              "waterfall_snr_db", "waterfall",
                              "bler_polar", "bler_ldpc", "bler_soft",
                              "bler", "block_error_rate"],
    "doppler_hz": ["max_doppler_hz", "doppler_hz", "doppler_frequency_hz",
                    "fd_hz", "doppler_max_hz"],
    "noise_power_dbm": ["noise_power_dbm", "thermal_noise_dbm",
                         "noise_floor_dbm", "noise_power",
                         "total_noise_power_dbm", "receiver_noise_dbm",
                         "n0_dbm"],
    "ber_at_snr": ["ber_at_target", "ber_at_snr", "ber_at_10db",
                    "ber_at_15db", "ber_at_target_snr", "ber_at_ebn0"],
    "nmse_db": ["nmse_db", "nmse", "channel_nmse_db", "csi_nmse_db",
                 "reconstruction_nmse_db"],
    "path_loss_range": ["path_loss_db", "path_loss", "pl_db", "mean_pl_db",
                        "mean_path_loss_db"],
    "coverage_pct": ["coverage_pct", "coverage_percent", "coverage",
                      "coverage_percentage"],
    "ris_gain_db": ["ris_gain_db", "gain_db", "ris_gain"],
    "min_errors_per_point": ["min_errors", "errors_per_point",
                              "min_errors_per_point"],
    "snr_at_ber_1e4_db": ["snr_for_ber_1e4_db", "snr_at_ber_1e4",
                           "ebn0_at_ber_1e4_db", "ebn0_at_target_ber_db"],
    "peak_se": ["peak_se", "peak_spectral_efficiency",
                 "peak_se_bpshz", "spectral_efficiency_bpshz"],
    "ber_gap_to_classical": ["ber_gap_to_classical",
                              "ber_gap_db_vs_classical",
                              "neural_vs_classical_gap_db"],
    "topology_cells": ["num_cells", "n_cells", "topology_cells",
                        "total_cells", "num_bs"],
    "sum_rate_bps_hz": ["sum_rate_bps_hz", "sum_rate", "system_throughput",
                         "total_throughput_bps_hz", "throughput_bps_hz",
                         "eMBB_throughput", "URLLC_BLER", "throughput",
                         "BPSK", "QPSK", "16-QAM", "64-QAM",
                         "modulation_throughput"],
    "fairness_index": ["fairness_index", "jain_fairness", "fairness",
                        "jains_index", "jain_index"],
    "localization_rmse_m": ["localization_rmse_m", "rmse_m", "rmse_meters",
                              "localization_error_m", "position_rmse"],
}


def _get_ber_arrays(sim: dict) -> tuple[list[float], list[float], list[float]] | None:
    """Extract (snr, ber_simulated, ber_theoretical) tuple from any of the
    known schemas the agent might produce."""
    if not isinstance(sim, dict):
        return None
    nm = sim.get("numerical_metrics")
    if isinstance(nm, dict):
        snr = (nm.get("ebn0_db") or nm.get("snr_db") or nm.get("ebno_db")
               or nm.get("ebn0_range_db") or nm.get("snr_range_db")
               or nm.get("snr") or nm.get("snrs"))
        sim_ber = (nm.get("ber_simulated") or nm.get("ber")
                   or nm.get("ber_agent") or nm.get("bler")
                   or nm.get("ber_results") or nm.get("ber_sim")
                   or nm.get("ber_values"))
        th_ber = (nm.get("ber_theoretical") or nm.get("ber_theory")
                  or nm.get("ber_analytical"))
        if isinstance(snr, list) and isinstance(sim_ber, list):
            return ([float(x) for x in snr],
                    [float(x) for x in sim_ber],
                    [float(x) for x in th_ber] if isinstance(th_ber, list) else [])
    # Fallback: ber_curves[] list-of-dicts
    curves = sim.get("ber_curves")
    if isinstance(curves, list) and curves:
        c = curves[0]
        snr = c.get("snr_db") or c.get("ebno_db")
        ber = c.get("ber")
        if isinstance(snr, list) and isinstance(ber, list):
            return ([float(x) for x in snr],
                    [float(x) for x in ber], [])
    # Fallback: ber_curves dict-of-dicts (one curve per key)
    if isinstance(curves, dict) and curves:
        first = next(iter(curves.values()))
        if isinstance(first, dict):
            snr = first.get("ebno_db") or first.get("snr_db")
            ber = first.get("ber") or first.get("bler")
            if isinstance(snr, list) and isinstance(ber, list):
                return ([float(x) for x in snr],
                        [float(x) for x in ber], [])
    # Fallback: top-level ALL-CAPS modulation keys (Gemma4 emits this often:
    # `{"ebno_db":[...], "BPSK":[...], "QPSK":[...]}`). Treat the SNR axis
    # as ebno_db/snr_db at the top level, and the first list-of-numbers
    # value at an ALL-CAPS or modulation-named key as the ber array.
    snr_top = (sim.get("ebno_db") or sim.get("snr_db")
               or sim.get("ebn0_db") or sim.get("ebno"))
    if isinstance(snr_top, list):
        for k, v in sim.items():
            if not isinstance(v, list) or not v:
                continue
            if not all(isinstance(x, (int, float)) for x in v):
                continue
            kupper = k.upper().replace("-", "").replace("_", "")
            mod_tokens = ("BPSK", "QPSK", "QAM", "PSK", "PAM",
                          "BER", "BLER")
            if any(t in kupper for t in mod_tokens):
                return ([float(x) for x in snr_top],
                        [float(x) for x in v], [])
    return None


def extract_scalar(sim: dict | None, metric: str,
                   verifier: dict | None = None) -> float | None:
    """Best-effort scalar extraction from simulation_result.json.

    For derived metrics (ber_gap_db, nmse_db, doppler_hz) this function
    will compute from raw arrays when possible."""
    if sim is None:
        return None
    verifier = verifier or {}
    # Try EXACT metric name first — only fall back to aliases if absent.
    # Otherwise an alias on a different scalar can shadow the field the
    # task explicitly named (e.g. snr_at_ber_1e4_db being shadowed by
    # ebn0_at_target_ber_db when both are present).
    found = _find_scalar_anywhere(sim, [metric])
    if found is None:
        aliases = _METRIC_ALIASES.get(metric, [])
        if aliases:
            found = _find_scalar_anywhere(sim, aliases)
    if found is not None:
        return found
    # Derived: BER gap to theoretical at a target BER
    if metric == "ber_gap_db":
        arrs = _get_ber_arrays(sim)
        if arrs is None:
            return None
        snr, sim_ber, th_ber = arrs
        target = (verifier.get("at_ber") or 1e-3)
        snr_sim = _find_snr_at_ber(snr, sim_ber, target)
        snr_th = _find_snr_at_ber(snr, th_ber, target) if th_ber else None
        if snr_sim is None or snr_th is None:
            # Require the agent to produce both simulated and theoretical
            # curves — saves the verifier from needing scipy.
            return None
        return abs(snr_sim - snr_th)
    # Derived: BER at a specific SNR (linear interp)
    if metric == "ber_at_snr":
        arrs = _get_ber_arrays(sim)
        if arrs is None:
            return None
        snr, sim_ber, _ = arrs
        target_snr = verifier.get("at_snr") or verifier.get("snr_db") or 10.0
        # Linear interp in log(BER)
        import math
        for i in range(len(snr) - 1):
            if snr[i] <= target_snr <= snr[i+1]:
                b0, b1 = sim_ber[i], sim_ber[i+1]
                if b0 <= 0 or b1 <= 0:
                    return b0 or b1
                lg0, lg1 = math.log(b0), math.log(b1)
                t = (target_snr - snr[i]) / (snr[i+1] - snr[i])
                return math.exp(lg0 + t * (lg1 - lg0))
        return None
    return None


def extract_array(sim: dict | None, metric: str) -> list[float] | None:
    """Extract an array of values for monotone checks."""
    if sim is None:
        return None
    if metric in ("ber_vs_snr", "ber", "bler"):
        arrs = _get_ber_arrays(sim)
        if arrs is not None:
            return arrs[1]  # simulated BER
    if metric in sim and isinstance(sim[metric], list):
        try:
            return [float(x) for x in sim[metric]]
        except Exception:
            return None
    nm = sim.get("numerical_metrics")
    if isinstance(nm, dict) and metric in nm and isinstance(nm[metric], list):
        try:
            return [float(x) for x in nm[metric]]
        except Exception:
            return None
    # Fallback: recursive substring match on metric keywords (e.g.
    # "bler_monotone_decreasing" -> any key with "bler" in it).
    metric_lc = metric.lower()
    substrings = [metric_lc]
    for token in ["bler", "ber", "nmse", "snr", "power", "throughput"]:
        if token in metric_lc:
            substrings.append(token)
    return _find_array_anywhere(sim, substrings, substring_match=True)


# ─────────────────────────────────────────────────────────────
# Check types
# ─────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


# ─────────────────────────────────────────────────────────────
# Plausibility checks (reward-hacking defense)
# ─────────────────────────────────────────────────────────────

def _check_scene_nontrivial(output_dir: Path) -> "CheckResult | None":
    """Reject scene_state.json files that are structurally empty.

    A scene with no rooms AND no top-level furniture AND no per-room
    furniture is the pre-shipped placeholder skeleton — verifier-leak
    guard against agents that emit zero LLM tokens and let the harness
    skeleton vacuously pass collision_free / in_bounds checks.

    Returns None when no scene_state.json exists (lets other checks
    decide), CheckResult(passed=True) when scene has content,
    CheckResult(passed=False) when the scene is empty.
    """
    p = output_dir / "scene_state.json"
    if not p.exists():
        return None
    try:
        scene = json.loads(p.read_text())
    except Exception:
        return CheckResult(name="scene_nontrivial", passed=False,
                           detail="scene_state.json malformed")
    rooms = scene.get("rooms") or []
    top_furn = scene.get("furniture") or []
    has_room_furniture = any(
        (r.get("furniture") or []) for r in rooms if isinstance(r, dict))
    if not rooms and not top_furn and not has_room_furniture:
        return CheckResult(name="scene_nontrivial", passed=False,
                           detail="scene has no rooms or furniture "
                                  "(likely pre-ship skeleton untouched)")
    return CheckResult(name="scene_nontrivial", passed=True,
                       detail=f"rooms={len(rooms)} top_furniture={len(top_furn)}")


def check_plausibility(output_dir: Path) -> list[CheckResult]:
    """Run a fixed set of reality checks on the agent's simulation_result.json
    and code. A reward-hacking agent that hard-codes impossible outputs will
    fail these; a genuine simulation will pass. All checks RETURN A FAILING
    check if they detect an implausibility — they never silently pass.

    These run on every task in addition to the task-specific verifier, and
    failing ANY plausibility check marks the trial as failed regardless of
    the task-specific check.
    """
    out: list[CheckResult] = []
    sim = load_sim_result(output_dir)
    code = load_all_code(output_dir)

    # (Removed the "must import sionna" check — legitimate analytical
    # solutions using 3GPP TR 38.901 path-loss formulas don't need
    # Sionna and were being false-flagged as reward hacking.)

    # 2. BER physical-range check.
    #    What's truly impossible: BER << 1 at the LOWEST SNR point of the
    #    sweep — even a perfect coded system can't beat random at the
    #    fundamental noise floor. The previous broader rule rejected
    #    legitimate LDPC waterfalls (BER drops 0.5 → 1e-7 in 2 dB is
    #    textbook coding gain), so we now only flag the lowest-SNR sample.
    arrs = _get_ber_arrays(sim) if sim else None
    if arrs is not None:
        snr, ber_sim, _ = arrs
        if snr and ber_sim:
            # Pair SNR with BER, find the minimum-SNR sample. At the
            # lowest test SNR the system MUST be in the noise — BER < 0.05
            # there is implausible.
            paired = sorted(zip(snr, ber_sim), key=lambda x: x[0])
            s_min, b_min = paired[0]
            if b_min < 0.05 and s_min < 0:  # only flag at NEGATIVE SNR
                out.append(CheckResult(
                    name="plausibility:ber_at_lowest_snr",
                    passed=False,
                    detail=f"BER={b_min:.2e} at SNR={s_min}dB "
                            "(lowest in sweep) — implausible at negative SNR"))
        # All-zero or all-identical BER across many points is suspicious
        if ber_sim and len(ber_sim) >= 4 and len(set(ber_sim)) == 1:
            out.append(CheckResult(
                name="plausibility:ber_not_constant",
                passed=False,
                detail=f"BER is constant at {ber_sim[0]} across "
                        f"{len(ber_sim)} SNR points — no real simulation"))

    # 3. Coverage percent must be in [0, 100].
    coverage = extract_scalar(sim, "coverage_pct",
                               {"metric": "coverage_pct"}) if sim else None
    if coverage is not None and (coverage < 0 or coverage > 100):
        out.append(CheckResult(
            name="plausibility:coverage_pct_range",
            passed=False,
            detail=f"coverage_pct={coverage} out of [0,100]"))

    # 4. NMSE suspiciously low (perfect reconstruction suggests cheating)
    nmse = extract_scalar(sim, "nmse_db",
                           {"metric": "nmse_db"}) if sim else None
    if nmse is not None and nmse < -40:
        out.append(CheckResult(
            name="plausibility:nmse_floor",
            passed=False,
            detail=f"NMSE={nmse} dB — below -40 dB is typically "
                    "impossible; inspect for hard-coded output"))

    # 5. RSS cannot exceed TX power in dBm (physically impossible)
    rss = extract_scalar(sim, "rss_dbm", {"metric": "rss_dbm"}) if sim else None
    tx_pow = extract_scalar(sim, "tx_power_dbm",
                             {"metric": "tx_power_dbm"}) if sim else None
    if rss is not None and tx_pow is not None and rss > tx_pow + 1:
        out.append(CheckResult(
            name="plausibility:rss_vs_tx_power",
            passed=False,
            detail=f"RSS={rss} > TX power {tx_pow} dBm — "
                    "energy conservation violated"))

    # 6. Training-task checks — catches agents that claim "trained NMSE = -15 dB"
    #    without actually training. Only fires if the result claims a neural
    #    component was trained (task_type + training section present).
    training = sim.get("training") if isinstance(sim, dict) else None
    claims_training = (
        (sim and sim.get("task_type") == "neural_component")
        or isinstance(training, dict)
        or "torch.optim" in code or "Adam(" in code
        or "loss.backward" in code or "keras.optimizers" in code)
    if claims_training:
        loss_hist = None
        if isinstance(training, dict):
            loss_hist = training.get("loss_history") or training.get("losses")
        if loss_hist is None and isinstance(sim, dict):
            nm = sim.get("numerical_metrics") or {}
            loss_hist = nm.get("loss_history") or nm.get("training_loss")

        # Evidence that training actually ran (any of these is sufficient
        # independent corroboration, beyond a populated loss_history):
        ckpt = output_dir / "model_checkpoint.pt"
        has_real_ckpt = ckpt.exists() and ckpt.stat().st_size >= 1024
        has_training_png = (output_dir / "training_loss.png").exists() \
            or any(output_dir.rglob("training_*.png"))
        other_training_evidence = has_real_ckpt or has_training_png

        # 6a: loss_history must exist with ≥3 points — UNLESS there's
        # other corroborating evidence the agent actually trained. This
        # avoids penalizing agents who trained correctly but didn't
        # populate the JSON field in our expected format.
        loss_hist_valid = isinstance(loss_hist, list) and len(loss_hist) >= 3
        if not loss_hist_valid and not other_training_evidence:
            out.append(CheckResult(
                name="plausibility:training_loss_history",
                passed=False,
                detail="training claimed but no loss_history (≥3 points) "
                        "AND no checkpoint/plot evidence. Fake training."))
        elif loss_hist_valid:
            assert isinstance(loss_hist, list)  # narrows type for Pyright
            # 6b: if loss_history IS provided, it should decrease
            try:
                start, end = float(loss_hist[0]), float(loss_hist[-1])
                if end >= start * 0.99:
                    out.append(CheckResult(
                        name="plausibility:training_loss_decreasing",
                        passed=False,
                        detail=f"loss did not decrease: "
                                f"start={start:.3g} end={end:.3g}"))
            except (ValueError, TypeError):
                pass
        # 6c: if a model_checkpoint.pt is present, it must be nonempty
        if ckpt.exists() and ckpt.stat().st_size < 1024:
            out.append(CheckResult(
                name="plausibility:model_checkpoint_nonempty",
                passed=False,
                detail=f"model_checkpoint.pt is {ckpt.stat().st_size} bytes "
                        "— too small to hold real weights"))

    # 7. Scene non-trivial: reject pre-shipped empty skeleton.
    nontrivial = _check_scene_nontrivial(output_dir)
    if nontrivial is not None:
        out.append(nontrivial)

    return out


# ─── Tier-5 domain checks ──────────────────────────────────────────
def check_tier5_domain(capability: str, output_dir: Path) -> list[CheckResult]:
    """Domain-specific physics checks keyed off task capability.

    Returns [] for unknown capabilities (skill-side: opt-in;
    benchmark-side: only invoked for known tier-5 capabilities).

    Each capability registers a `_check_<name>(sim, code, output_dir)`
    function that returns list[CheckResult]. Subsequent tasks (C.2-C.9)
    add more entries to this dispatch table.
    """
    sim = load_sim_result(output_dir)
    code = load_all_code(output_dir)
    dispatch = {
        "channel_charting": _check_channel_charting,
        "isac_tradeoff_curve": _check_isac_tradeoff,
        "otfs_waveform": _check_otfs,
        "semantic_communication": _check_semantic,
        "semantic_metric": _check_semantic,
        "thz_channel": _check_thz,
        "near_field_beamforming": _check_nearfield,
        "federated_learning": _check_federated,
        "star_ris": _check_star_ris,
        "star-ris": _check_star_ris,
        "channel_prediction": _check_channel_prediction,
        # C.2-C.9 add capabilities here.
    }
    fn = dispatch.get(capability)
    if fn is None:
        return []
    return fn(sim, code, output_dir)


def _check_channel_charting(sim, code, output_dir):
    """Pearson r between chart coordinates and ground-truth positions >= 0.7
    over >=100 CSI samples. Fails on either threshold."""
    out: list[CheckResult] = []
    if sim is None:
        return [CheckResult(name="channel_charting:no_result",
                            passed=False, detail="simulation_result.json missing")]
    r = extract_scalar(sim, "pearson_r", {"metric": "pearson_r"})
    n = extract_scalar(sim, "n_points", {"metric": "n_points"})
    if r is None:
        out.append(CheckResult(name="channel_charting:pearson_r_present",
                               passed=False,
                               detail="numerical_metrics.pearson_r missing"))
    else:
        out.append(CheckResult(name="channel_charting:pearson_r_threshold",
                               passed=r >= 0.7,
                               detail=f"r={r:.3f} (target >=0.7)"))
    if n is None:
        out.append(CheckResult(name="channel_charting:n_points_present",
                               passed=False,
                               detail="numerical_metrics.n_points missing"))
    else:
        out.append(CheckResult(name="channel_charting:n_points_threshold",
                               passed=n >= 100,
                               detail=f"n={int(n)} (target >=100)"))
    return out


# ── C.2 ───────────────────────────────────────────────────────────────────────
def _check_isac_tradeoff(sim, code, output_dir):
    """ISAC tradeoff curves require >=20 channel realizations and a real
    tradeoff (comm rate decreases as sensing RMSE improves)."""
    out: list[CheckResult] = []
    if sim is None:
        return [CheckResult(name="isac:no_result", passed=False,
                            detail="simulation_result.json missing")]
    nr = extract_scalar(sim, "n_realizations", {"metric": "n_realizations"})
    out.append(CheckResult(
        name="isac:n_realizations",
        passed=(nr is not None and nr >= 20),
        detail=f"n_realizations={nr} (target >=20)"))
    comm = extract_array(sim, "comm_rate_bps_hz")
    sense = extract_array(sim, "sensing_rmse_m")
    if not comm or not sense or len(comm) != len(sense) or len(comm) < 3:
        out.append(CheckResult(name="isac:tradeoff_arrays",
                               passed=False,
                               detail=f"comm={len(comm or [])} "
                                       f"sense={len(sense or [])}"))
        return out
    paired = sorted(zip(sense, comm), reverse=True)
    comm_sorted = [c for _, c in paired]
    monotone = all(comm_sorted[i] >= comm_sorted[i + 1]
                   for i in range(len(comm_sorted) - 1))
    out.append(CheckResult(
        name="isac:tradeoff_monotone",
        passed=monotone,
        detail=f"comm rate {'decreases' if monotone else 'does not decrease'} "
                f"as sensing RMSE improves"))
    return out


# ── C.3 ───────────────────────────────────────────────────────────────────────
def _check_otfs(sim, code, output_dir):
    """OTFS code must reference 2D delay-Doppler grid."""
    code_lc = code.lower()
    has_2d = ("delay_doppler" in code_lc
              or ("delay" in code_lc and "doppler" in code_lc
                  and "grid" in code_lc))
    return [CheckResult(
        name="otfs:delay_doppler_grid",
        passed=has_2d,
        detail="delay-Doppler grid term not found in code"
                if not has_2d else "2D pilot pattern present")]


# ── C.4 ───────────────────────────────────────────────────────────────────────
def _check_semantic(sim, code, output_dir):
    """Semantic comm reports classification accuracy in [0,1]."""
    if sim is None:
        return [CheckResult(name="semantic:no_result", passed=False,
                            detail="simulation_result.json missing")]
    acc = extract_scalar(sim, "classification_accuracy",
                        {"metric": "classification_accuracy"})
    if acc is None:
        acc = extract_scalar(sim, "accuracy", {"metric": "accuracy"})
    if acc is None:
        acc = extract_scalar(sim, "top1_accuracy",
                              {"metric": "top1_accuracy"})
    return [CheckResult(
        name="semantic:accuracy_reported",
        passed=(acc is not None and 0.0 <= acc <= 1.0),
        detail=(f"accuracy={acc}" if acc is not None
                else "no classification_accuracy field"))]


# ── C.5 ───────────────────────────────────────────────────────────────────────
def _check_thz(sim, code, output_dir):
    code_lc = code.lower()
    has_absorption = (
        "absorption" in code_lc
        or "p.676" in code_lc or "p676" in code_lc
        or "molecular_loss" in code_lc)
    return [CheckResult(
        name="thz:absorption_applied",
        passed=has_absorption,
        detail="no molecular absorption term — Friis-only at THz"
                if not has_absorption else "absorption present")]


# ── C.6 ───────────────────────────────────────────────────────────────────────
def _check_nearfield(sim, code, output_dir):
    import re as _re
    code_lc = code.lower()
    has_rayleigh = (
        "rayleigh_distance" in code_lc
        or _re.search(r"2\s*\*\s*d\s*\*?\*?\s*2\s*/\s*(lambda|wavelength)",
                      code_lc) is not None
        or _re.search(r"2\s*\*\s*aperture\s*\*?\*?\s*2\s*/", code_lc)
        is not None)
    has_spherical = ("spherical" in code_lc and "steer" in code_lc)
    out = [CheckResult(
        name="nearfield:rayleigh_distance",
        passed=has_rayleigh,
        detail="no Rayleigh-distance computation found"
                if not has_rayleigh else "Rayleigh present")]
    out.append(CheckResult(
        name="nearfield:spherical_steering",
        passed=has_spherical,
        detail="no spherical-wave steering term found"
                if not has_spherical else "spherical steering present"))
    return out


# ── C.7 ───────────────────────────────────────────────────────────────────────
def _check_federated(sim, code, output_dir):
    if sim is None:
        return [CheckResult(name="federated:no_result", passed=False,
                            detail="simulation_result.json missing")]
    sizes = extract_array(sim, "client_dataset_sizes")
    if not sizes:
        return [CheckResult(name="federated:client_sizes_present",
                            passed=False,
                            detail="numerical_metrics.client_dataset_sizes missing")]
    has_variance = len(set(sizes)) > 1
    return [CheckResult(
        name="federated:client_size_variance",
        passed=has_variance,
        detail=f"sizes={sizes} {'vary' if has_variance else 'are all equal'}")]


# ── C.8 ───────────────────────────────────────────────────────────────────────
def _check_star_ris(sim, code, output_dir):
    if sim is None:
        return [CheckResult(name="star_ris:no_result", passed=False,
                            detail="simulation_result.json missing")]
    t = extract_array(sim, "t_coeffs_abs2")
    r = extract_array(sim, "r_coeffs_abs2")
    if not t or not r or len(t) != len(r):
        return [CheckResult(name="star_ris:t_and_r_arrays",
                            passed=False,
                            detail=f"t={len(t or [])} r={len(r or [])}")]
    bad = [i for i in range(len(t)) if abs((t[i] + r[i]) - 1.0) > 0.05]
    return [CheckResult(
        name="star_ris:energy_conservation",
        passed=len(bad) == 0,
        detail=f"{len(bad)}/{len(t)} elements violate |t|^2+|r|^2=1")]


# ── C.9 ───────────────────────────────────────────────────────────────────────
def _check_channel_prediction(sim, code, output_dir):
    if sim is None:
        return [CheckResult(name="channel_pred:no_result", passed=False,
                            detail="simulation_result.json missing")]
    nmse = extract_array(sim, "nmse_db_at_horizon")
    if not nmse or len(nmse) < 3:
        return [CheckResult(name="channel_pred:nmse_array",
                            passed=False,
                            detail=f"nmse_db_at_horizon has {len(nmse or [])} pts")]
    increasing = all(nmse[i] <= nmse[i + 1] + 0.5
                     for i in range(len(nmse) - 1))
    strictly_grew = nmse[-1] > nmse[0] + 1.0
    return [CheckResult(
        name="channel_pred:nmse_increases_with_horizon",
        passed=increasing and strictly_grew,
        detail=f"NMSE went {nmse[0]:.1f} -> {nmse[-1]:.1f} dB")]
