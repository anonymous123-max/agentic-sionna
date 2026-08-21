"""check_code_contains routes every known metric to its handler;
unknown metrics fall through to the generic-token check."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmark.verifier import check_code_contains, _CODE_CONTAINS_HANDLERS


def _task(metric: str, **extra) -> dict:
    return {"verifier": {"metric": metric, **extra},
            "required_artifacts": []}


def test_v2_namespace_passes_clean_code(tmp_path):
    (tmp_path / "x.py").write_text("import sionna.phy")
    r = check_code_contains(_task("code_runs_v2"), tmp_path)
    assert r.passed


def test_v2_namespace_fails_legacy(tmp_path):
    (tmp_path / "x.py").write_text("import sionna.channel.tdl")
    r = check_code_contains(_task("code_runs_v2"), tmp_path)
    assert not r.passed


def test_norm_constrained_passes(tmp_path):
    (tmp_path / "x.py").write_text("torch.linalg.norm(x)")
    r = check_code_contains(_task("norm_constrained"), tmp_path)
    assert r.passed


def test_unknown_metric_falls_back_to_token_check(tmp_path):
    long_body = "x = 1\n" * 60  # >200 chars to satisfy length guard
    (tmp_path / "x.py").write_text(f"# uses mmwave channel\n{long_body}")
    r = check_code_contains(_task("uses_mmwave_channel"), tmp_path)
    assert r.passed  # generic-token finds 'mmwave' (synonym) and 'channel'


def test_handlers_dict_covers_known_metrics():
    expected = {"code_runs_v2", "norm_constrained", "power_normalized",
                "learnable_params_present", "eval_metric_is_accuracy",
                "absorption_applied", "rayleigh_distance_correct",
                "correct_model_selection"}
    assert expected <= set(_CODE_CONTAINS_HANDLERS.keys())


def test_collision_free_routes_to_scene_helper(tmp_path):
    """metric=collision_free_check goes to _check_scene_collision_free,
    which reads scene_state.json — not to the registry."""
    (tmp_path / "scene_state.json").write_text(json.dumps({
        "rooms": [{"room_type": "office", "dimensions": [10, 10],
                   "furniture": []}]
    }))
    r = check_code_contains(_task("collision_free_check"), tmp_path)
    assert r.name == "collision_free"
