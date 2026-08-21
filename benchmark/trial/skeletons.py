"""Pre-ship canonical-artifact skeletons. The harness writes these
into the trial workdir BEFORE the agent runs so a crashed/empty agent
still leaves parseable artifacts for the verifier."""
from __future__ import annotations
import json
from pathlib import Path


# Pre-ship a placeholder simulation_result.json
# with canonical field NAMES (null/empty values) before the agent runs.
# When the agent crashes mid-run or end_turns empty, the verifier still
# finds an artifact with the right structure — partial-credit floor +
# guarantees the canonical field names are present even if the agent
# overwrites with descriptive variants. The agent's own write fully
# replaces this file (json.dump with mode 'w'), so this never masks
# real output.
_SKELETON_BASE = {
    "schema_version": "1.0",
    "status": "placeholder_pre_shipped_by_harness",
    "numerical_metrics": {},
}

_SKELETON_BY_TIER = {
    "T0_scene_gen": {
        **_SKELETON_BASE,
        "rooms": [], "furniture": [],
    },
    "T1_phy_link_level": {
        **_SKELETON_BASE,
        "numerical_metrics": {
            "snr_db": [], "ebn0_db": [],
            "ber_simulated": [], "ber_theoretical": [], "bler_simulated": [],
            "ber_at_snr_10db": None, "ber_at_snr_15db": None,
            "ber_at_target": None, "target_ber": None,
            "ebn0_at_target_ber_db": None,
            "ber_gap_db": None, "ber_gap_to_classical": None,
            "waterfall_ebn0_db": None, "shannon_limit_db": None,
            "coding_gain_db": None,
        },
    },
    "T2_ray_tracing": {
        **_SKELETON_BASE,
        "numerical_metrics": {
            "coverage_pct": None, "target_coverage_pct": None,
            "mean_rss_dbm": None, "min_rss_dbm": None, "max_rss_dbm": None,
            "p5_received_power_dbm": None, "path_loss_db": None,
            "path_loss_range_db": [],
            "ris_gain_db": None, "sum_rate_bps_hz": None,
            "peak_se_bpshz": None, "total_paths": None,
        },
    },
    "T3_ml_neural": {
        # Intentionally no `task_type` and no `training` dict — both would
        # trip the plausibility "claims_training" check on empty runs.
        # The agent adds them when actually training.
        **_SKELETON_BASE,
        "numerical_metrics": {
            "snr_db": [], "ber_simulated": [], "ber_classical": [],
            "nmse_db": None, "ber_gap_to_classical": None,
        },
    },
    "T4_system_level": {
        **_SKELETON_BASE,
        "numerical_metrics": {
            "coverage_pct": None, "sum_rate_bps_hz": None,
            "peak_se_bpshz": None, "topology_cells": None,
            "num_users": None, "fairness_index": None,
        },
    },
    "T5_emerging": {
        **_SKELETON_BASE,
        "numerical_metrics": {
            "snr_db": [], "ber_simulated": [],
            "localization_rmse_m": None, "doppler_hz": None,
        },
    },
    "T6_anchor": {
        **_SKELETON_BASE,
        "numerical_metrics": {
            "snr_db": [], "ber_simulated": [],
            "ber_at_snr_10db": None, "coding_gain_db": None,
        },
    },
}


_SCENE_SKELETON = {
    "schema_version": "1.0",
    "status": "placeholder_pre_shipped_by_harness",
    "rooms": [],
    "furniture": [],
    "numerical_metrics": {
        "num_rooms": 0, "num_walls": 0, "num_furniture": 0,
        "num_transmitters": 0, "scene_area_m2": 0.0,
    },
}


def pre_ship_skeleton(task: dict, workdir: Path) -> None:
    """Write canonical-field placeholders to workdir BEFORE the agent runs,
    so a crashed/empty agent still leaves parseable artifacts for the
    verifier. The agent's own json.dump fully overwrites these files when
    it produces real output."""
    required = set(task.get("required_artifacts") or [])
    tier = task.get("tier", "")

    # simulation_result.json
    if "simulation_result.json" in required \
            or not required:  # default required when unspecified
        skel = _SKELETON_BY_TIER.get(tier,
                                      _SKELETON_BY_TIER["T1_phy_link_level"])
        target = workdir / "simulation_result.json"
        if not target.exists():
            target.write_text(json.dumps(skel, indent=2))

    # scene_state.json (T0 tasks). If the task specifies a scene_path,
    # copy that source scene as the starting point (used by scene_edit
    # tasks); otherwise write the canonical placeholder.
    if "scene_state.json" in required:
        target = workdir / "scene_state.json"
        if not target.exists():
            scene_path = task.get("scene_path")
            if scene_path:
                # Resolve relative to repo root (parents[2] = repo from
                # benchmark/trial/skeletons.py).
                root = Path(__file__).resolve().parents[2]
                src = (root / scene_path) if not Path(scene_path).is_absolute() \
                    else Path(scene_path)
                if src.exists():
                    target.write_text(src.read_text())
                else:
                    target.write_text(json.dumps(_SCENE_SKELETON, indent=2))
            else:
                target.write_text(json.dumps(_SCENE_SKELETON, indent=2))
