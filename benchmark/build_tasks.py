"""Merge benchmark/tasks/_sources/capability_grid.json (60) +
benchmark/tasks/_sources/tutorial_variants.json (99) into
benchmark/tasks/tasks.json, deduplicated and normalized to one schema.

Unified task schema:
{
  "id": "U001",                       # sequential unified ID
  "origin": "suite" | "corpus",
  "origin_id": "T01" | "A01",
  "tier": "T1_phy" | ... | "T0_scene_gen" | "T6_anchor",
  "capability": "ber_awgn" | "ber_coded" | ...,
  "difficulty": "easy" | "medium" | "hard",
  "name": "...",
  "prompt": "...",
  "distractor": "...",                # optional
  "scene_path": null | "benchmark/scenes/...",
  "verifier": {
    "type": "metric_threshold" | "metric_range" | "metric_monotone"
          | "count" | "code_contains" | "value_exact"
          | "execution_ok" | "file_exists" | "composite",
    ...type-specific params...
  },
  "assertions": ["human-readable ...", ...],
  "required_artifacts": ["simulation_result.json", ...]
}
"""
from __future__ import annotations
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUITE_PATH = ROOT / "benchmark/tasks/_sources/capability_grid.json"
CORPUS_PATH = ROOT / "benchmark/tasks/_sources/tutorial_variants.json"
OUT_PATH = ROOT / "benchmark/tasks/tasks.json"

# Tier mapping for suite tasks: numeric tier -> semantic id
SUITE_TIER_MAP = {
    1: "T1_phy_link_level",
    2: "T2_ray_tracing",
    3: "T3_ml_neural",
    4: "T4_system_level",
    5: "T5_emerging",
}

# Corpus category -> tier
CORPUS_TIER_MAP = {
    "A_ber_analysis": "T1_phy_link_level",
    "B_indoor_coverage": "T4_system_level",  # multi-AP coverage = system-level
    "C_mimo_ofdm": "T1_phy_link_level",
    "D_scene_generation": "T0_scene_gen",
    "E_optimization": "T4_system_level",
    "F_rt_to_phy": "T2_ray_tracing",
    "G_anchor_sionna_tutorials": "T6_anchor",
}

# Corpus categories to KEEP (unique capabilities not in suite).
# A and C are fully covered by suite T01-T18.
KEEP_CORPUS = {"B_indoor_coverage", "D_scene_generation",
               "E_optimization", "F_rt_to_phy",
               "G_anchor_sionna_tutorials"}


def infer_capability(category: str, task: dict) -> str:
    """Fine-grained capability tag derived from category + keywords."""
    prompt = (task.get("prompt") or "").lower()
    if category == "A_ber_analysis":
        if "polar" in prompt: return "ber_polar"
        if "ldpc" in prompt: return "ber_ldpc"
        return "ber_awgn"
    if category == "B_indoor_coverage":
        return "indoor_coverage_map"
    if category == "C_mimo_ofdm":
        return "mimo_ofdm"
    if category == "D_scene_generation":
        if "outdoor" in prompt or "osm" in prompt: return "scene_outdoor"
        return "scene_indoor"
    if category == "E_optimization":
        if "ris" in prompt: return "opt_ris"
        if "multi" in prompt or "multiple" in prompt: return "opt_multi_ap"
        return "opt_ap_placement"
    if category == "F_rt_to_phy":
        return "rt_to_phy"
    if category == "G_anchor_sionna_tutorials":
        return "anchor_tutorial"
    return "unknown"


def suite_verifier_to_unified(v: dict) -> dict:
    """Convert suite verifier spec to unified dispatcher schema.

    The suite uses ~50 unique metric names; we classify each into one of
    the 9 dispatcher types so the verifier can execute them uniformly.
    """
    metric = v.get("metric", "")
    # Numeric threshold checks (metric <= threshold or gap <= threshold)
    if metric in {"ber_gap_db", "ber_at_snr", "nmse_db",
                  "ber_gap_to_classical", "min_errors_per_point",
                  "noise_power_dbm"}:
        return {
            "type": "metric_threshold",
            "metric": metric,
            "threshold": v.get("threshold", v.get("max_gap_db", v.get("expected"))),
            "tolerance": v.get("tolerance", v.get("tolerance_pct")),
            "at_ber": v.get("at_ber"),
            "at_snr": v.get("at_snr"),
            "direction": "<=" if metric != "min_errors_per_point" else ">=",
        }
    # Range checks
    if metric in {"bler_waterfall_range", "path_loss_range", "doppler_hz"}:
        if "range_db" in v:
            lo, hi = v["range_db"]
        elif "min_db" in v and "max_db" in v:
            lo, hi = v["min_db"], v["max_db"]
        elif "expected" in v and "tolerance_pct" in v:
            exp = v["expected"]
            tol = exp * v["tolerance_pct"] / 100
            lo, hi = exp - tol, exp + tol
        else:
            lo, hi = None, None
        return {"type": "metric_range", "metric": metric, "min": lo, "max": hi}
    # Count checks
    if metric in {"curve_count"} or "expected" in v and isinstance(v.get("expected"), int):
        return {"type": "count", "metric": metric, "expected": v["expected"]}
    # Monotonic / ordering checks
    if metric in {"monotonic_bler_vs_rate", "bler_monotone_decreasing",
                  "bler_monotone_with_power", "power_increases",
                  "bler_decreasing_per_round"}:
        return {
            "type": "metric_monotone",
            "metric": metric,
            "direction": "increasing" if "increases" in metric else "decreasing",
            "min_points": v.get("min_iterations_monotone",
                                v.get("rounds", 3)),
        }
    # Code-contains checks (model selection, API usage)
    if metric in {"correct_model_selection", "code_runs",
                  "code_runs_v2", "cir_shape_valid",
                  "permittivity_valid", "norm_constrained",
                  "new_channel_per_item", "power_normalized",
                  "learnable_params_present",
                  "eval_metric_is_accuracy", "update_interval_correct",
                  "absorption_applied", "rayleigh_distance_correct",
                  "snr_formula_correct", "star_better_than_conventional"}:
        return {"type": "code_contains", "metric": metric, "spec": v}
    # Value-exact checks (within tolerance)
    if "expected" in v and "tolerance" in v:
        return {"type": "value_exact", "metric": metric,
                "expected": v["expected"], "tolerance": v["tolerance"]}
    # Pattern-based classification for the long tail of metric names.
    m_lower = metric.lower()
    # Scalar thresholds: names ending in *_db / *_hz / *_m, or containing
    # rmse/nmse/gain — numeric comparisons where the spec carries the target.
    scalar_target = (v.get("threshold") or v.get("min_gain")
                     or v.get("pearson_min") or v.get("cap_dbm")
                     or v.get("target_bler") or v.get("expected"))
    if (m_lower.endswith(("_db", "_hz", "_m", "_dbm", "_mbps"))
            or "rmse" in m_lower or "nmse" in m_lower or "nve" in m_lower
            or "gain" in m_lower or "sinr" in m_lower):
        if scalar_target is not None:
            # "Lower is better" metrics (errors, validation loss, RMSE, NVE):
            # NVE = ratio of agent BLER to perfect-CSI BLER, so smaller=closer
            # to oracle. Same for NMSE, RMSE, BLER, generic *error* names.
            lower_is_better = (
                "nmse" in m_lower or "rmse" in m_lower
                or "bler" in m_lower or "error" in m_lower
                or m_lower == "nve" or "nve_" in m_lower
                or m_lower.endswith("_loss"))
            direction = "<=" if lower_is_better else ">="
            return {"type": "metric_threshold", "metric": metric,
                    "threshold": scalar_target, "direction": direction}
    # Count-adjacent names: "two_", "three_", "four_" with a number in spec
    if any(m_lower.startswith(p) for p in ("two_", "three_", "four_", "five_")):
        count_expected = {"two": 2, "three": 3, "four": 4, "five": 5}
        for word, n in count_expected.items():
            if m_lower.startswith(word + "_"):
                return {"type": "count", "metric": metric, "expected": n}
    # Monotonicity names
    if any(w in m_lower for w in ("monotone", "increases", "decreases",
                                    "_increasing", "_decreasing")):
        direction = "increasing" if ("increases" in m_lower
                                      or "_increasing" in m_lower) else "decreasing"
        return {"type": "metric_monotone", "metric": metric,
                "direction": direction, "min_points": 3}
    # Comparative / boolean ("X_better_than_Y", "uses_X", "foo_correct")
    # Require the agent's code to mention the metric concept (tokenized).
    # This is weaker than a numeric check but still forces the agent to
    # address the right comparison, not just produce any output.
    return {"type": "code_contains", "metric": metric, "spec": v}


def corpus_to_unified_verifier(task: dict) -> dict:
    """Build a unified verifier spec from corpus task's validation + ground_truth."""
    val = task.get("validation", {}) or {}
    gt = task.get("ground_truth", {}) or {}
    checks = []
    # Artifact existence checks from validation.must_produce_*
    for k, required in val.items():
        if not required:
            continue
        if k.startswith("must_produce_") or k.startswith("must_create_")\
                or k.startswith("must_have_") or k == "must_report_coverage_pct"\
                or k == "must_report_throughput" or k == "must_report_tx_position":
            checks.append({"type": "file_exists", "key": k})
        elif k == "must_be_collision_free":
            checks.append({"type": "code_contains", "metric": "collision_free_check"})
        elif k == "must_be_in_bounds":
            checks.append({"type": "code_contains", "metric": "in_bounds_check"})
        elif k == "ber_must_decrease_with_snr":
            checks.append({"type": "metric_monotone",
                           "metric": "ber_vs_snr", "direction": "decreasing",
                           "min_points": 3})
        elif k == "coverage_in_range":
            rng = gt.get("coverage_range_pct")
            if rng:
                checks.append({"type": "metric_range",
                               "metric": "coverage_pct",
                               "min": rng[0], "max": rng[1]})
        elif k == "coverage_must_meet_target":
            tgt = gt.get("target_pct")
            if tgt:
                checks.append({"type": "metric_threshold",
                               "metric": "coverage_pct",
                               "threshold": tgt, "direction": ">="})
    # Ground-truth tolerances
    if "snr_tolerance_db" in val and "snr_for_ber_1e4_db" in gt:
        checks.append({"type": "value_exact",
                       "metric": "snr_at_ber_1e4_db",
                       "expected": gt["snr_for_ber_1e4_db"],
                       "tolerance": val["snr_tolerance_db"]})
    # Multi-check: wrap as composite
    if len(checks) == 0:
        return {"type": "execution_ok"}
    if len(checks) == 1:
        return checks[0]
    return {"type": "composite", "subchecks": checks}


CORPUS_DISTRACTORS = {
    "A_ber_analysis": (
        "Using Es/N0 instead of Eb/N0 on the x-axis shifts the curve by "
        "log2(M) dB for M-ary modulation — silent but wrong result."),
    "B_indoor_coverage": (
        "Placing TX outside the room bounds, or forgetting to set "
        "scene.frequency before the solver, produces all-zero or "
        "wrong-permittivity coverage maps."),
    "C_mimo_ofdm": (
        "Using CDL in a multi-user/multi-TX scenario, or forgetting to "
        "configure StreamManagement for the chosen num_streams_per_tx, "
        "silently corrupts spatial multiplexing."),
    "D_scene_generation": (
        "Placing furniture that overlaps walls or other furniture produces "
        "physically impossible scenes that break downstream simulation. "
        "Forgetting rf_materials makes propagation use default vacuum."),
    "E_optimization": (
        "Using scipy.optimize on a coverage landscape that is non-smooth "
        "and mostly flat fails immediately — the agent must sweep "
        "candidate positions, not gradient-descend a path-loss blob."),
    "F_rt_to_phy": (
        "Skipping scene.frequency before ray tracing, or calling "
        "paths.cir() without normalize_delays=True, produces CIR that "
        "doesn't match the OFDM cyclic prefix length and creates a BER "
        "error floor from inter-symbol interference."),
    "G_anchor_sionna_tutorials": (
        "Deviating from the canonical Sionna tutorial parameters (k, n, "
        "num_bits_per_symbol, channel_model) makes the agent's output "
        "incomparable to the tutorial's published baseline."),
}


def build_assertions_from_corpus(task: dict) -> list[str]:
    """Generate human-readable assertions from corpus validation/ground_truth."""
    val = task.get("validation", {}) or {}
    gt = task.get("ground_truth", {}) or {}
    out = []
    for k, v in val.items():
        if v is True:
            out.append(f"{k.replace('_', ' ')}")
        elif isinstance(v, (int, float)):
            out.append(f"{k.replace('_', ' ')} = {v}")
    for k, v in gt.items():
        out.append(f"expect {k}={v}")
    return out or ["executes and produces expected artifacts"]


def required_artifacts_from_corpus(task: dict, category: str) -> list[str]:
    val = task.get("validation", {}) or {}
    out = []
    mapping = {
        "must_produce_ber_curve": "simulation_result.json",
        "must_produce_heatmap": "coverage_map.npy",
        "must_produce_ber_map": "ber_map.npy",
        "must_produce_throughput_map": "throughput_map.npy",
        "must_create_scene_state": "scene_state.json",
        "must_report_coverage_pct": "simulation_result.json",
        "must_report_throughput": "simulation_result.json",
        "must_report_tx_position": "tx_position.json",
    }
    for k, art in mapping.items():
        if val.get(k) and art not in out:
            out.append(art)
    if not out:
        if category == "D_scene_generation":
            out = ["scene_state.json"]
        else:
            out = ["simulation_result.json"]
    return out


def suite_required_artifacts(task: dict) -> list[str]:
    """Suite tasks don't specify artifacts explicitly; infer from tier."""
    tier = task.get("tier")
    if tier == 1:
        return ["simulation_result.json"]
    if tier == 2:
        return ["cir.npy", "radio_map.npy"] if "map" in task["name"].lower() else ["cir.npy"]
    if tier == 3:
        return ["model_checkpoint.pt", "simulation_result.json"]
    if tier == 4:
        return ["simulation_result.json"]
    if tier == 5:
        return ["simulation_result.json"]
    return ["simulation_result.json"]


def main():
    suite = json.loads(SUITE_PATH.read_text())
    corpus = json.loads(CORPUS_PATH.read_text())

    unified = []
    uid = 1

    # --- Suite tasks (all 60, keep as authoritative for overlapping capabilities)
    for t in suite["tasks"]:
        tier_id = SUITE_TIER_MAP.get(t["tier"], f"T{t['tier']}_unknown")
        cap = t.get("name", "").split(":")[0].strip().lower().replace(" ", "_")
        unified.append({
            "id": f"U{uid:03d}",
            "origin": "suite",
            "origin_id": t["id"],
            "tier": tier_id,
            "capability": cap,
            "difficulty": t["difficulty"],
            "name": t["name"],
            "prompt": t["prompt"],
            "distractor": t.get("distractor"),
            "scene_path": None,
            "verifier": suite_verifier_to_unified(t.get("verifier", {})),
            "assertions": t.get("assertions", []),
            "required_artifacts": suite_required_artifacts(t),
        })
        uid += 1

    # --- Corpus tasks (keep only categories whose capabilities aren't covered by suite)
    for cid, cdata in corpus["categories"].items():
        if cid not in KEEP_CORPUS:
            continue
        tier_id = CORPUS_TIER_MAP[cid]
        for t in cdata.get("tasks", []):
            cap = infer_capability(cid, t)
            unified.append({
                "id": f"U{uid:03d}",
                "origin": "corpus",
                "origin_id": t["id"],
                "tier": tier_id,
                "capability": cap,
                "difficulty": t.get("difficulty", "medium"),
                "name": f"{cid}: {t['prompt'][:60]}",
                "prompt": t["prompt"],
                "distractor": CORPUS_DISTRACTORS.get(cid),
                "scene_path": t.get("scene_path"),
                "verifier": corpus_to_unified_verifier(t),
                "assertions": build_assertions_from_corpus(t),
                "required_artifacts": required_artifacts_from_corpus(t, cid),
            })
            uid += 1

    # ── 60/40 train/test split, stratified by tier, fixed seed for reproducibility
    # The test split is NEVER iterated against; evaluated once at the end.
    by_tier: dict[str, list[dict]] = defaultdict(list)
    for t in unified:
        by_tier[t["tier"]].append(t)
    rng = random.Random(20260417)
    for tier, group in by_tier.items():
        rng.shuffle(group)
        n_test = max(1, int(round(0.40 * len(group))))
        for i, t in enumerate(group):
            t["split"] = "test" if i < n_test else "train"

    # Summary
    by_tier_counts: dict[str, int] = {}
    by_origin: dict[str, int] = {}
    by_vtype: dict[str, int] = {}
    by_split: dict[str, int] = {"train": 0, "test": 0}
    for t in unified:
        by_tier_counts[t["tier"]] = by_tier_counts.get(t["tier"], 0) + 1
        by_origin[t["origin"]] = by_origin.get(t["origin"], 0) + 1
        vt = t["verifier"]["type"]
        by_vtype[vt] = by_vtype.get(vt, 0) + 1
        by_split[t["split"]] = by_split.get(t["split"], 0) + 1
    by_tier = by_tier_counts

    manifest = {
        "version": "1.1",
        "description": "Active benchmark task set merging _sources/capability_grid.json (60) + _sources/tutorial_variants.json kept categories. Includes stratified 60/40 train/test split (seed=20260417) — test tasks must not be used for skill iteration.",
        "total_tasks": len(unified),
        "split_policy": "stratified-by-tier, seed=20260417",
        "by_tier": by_tier,
        "by_origin": by_origin,
        "by_verifier_type": by_vtype,
        "by_split": by_split,
        "tasks": unified,
    }
    OUT_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT_PATH} with {len(unified)} tasks")
    print(f"by tier: {by_tier}")
    print(f"by origin: {by_origin}")
    print(f"by verifier type: {by_vtype}")


if __name__ == "__main__":
    main()
