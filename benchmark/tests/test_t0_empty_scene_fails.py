"""An empty scene_state.json (no rooms, no furniture) must NOT pass
verification. Closes the cycle6 U079 leak where Gemma scored 1.00 with
turns=0 because the pre-shipped skeleton vacuously satisfied
collision_free + in_bounds."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))

from benchmark.verifier import verify


def test_empty_scene_fails(tmp_path: Path):
    # Pre-ship skeleton (what trial.py writes before the agent runs)
    (tmp_path / "scene_state.json").write_text(json.dumps({
        "schema_version": "1.0",
        "status": "placeholder_pre_shipped_by_harness",
        "rooms": [],
        "furniture": [],
        "numerical_metrics": {"num_rooms": 0, "num_walls": 0, "num_furniture": 0},
    }))
    # Minimal T0 task spec
    task = {
        "id": "TEST_T0",
        "tier": "T0_scene_gen",
        "capability": "scene_gen",
        "required_artifacts": ["scene_state.json"],
        "verifier": {"type": "composite", "subchecks": [
            {"type": "code_contains", "metric": "collision_free_check"},
            {"type": "code_contains", "metric": "in_bounds_check"},
        ]},
    }
    result = verify(task, tmp_path, exec_success=True)
    assert not result.passed, (
        f"Empty scene incorrectly passed verification "
        f"(score={result.score}, "
        f"checks={[c.name for c in result.checks if c.passed]})"
    )


def test_artifact_check_rejects_pre_shipped_simulation_placeholder(tmp_path: Path):
    """Pre-shipped simulation_result.json (status="placeholder_pre_shipped_by_harness")
    must NOT pass the artifact:simulation_result.json existence check. Closes
    the false-positive where an agent that never wrote anything still
    received credit because the harness skeleton was on disk."""
    from benchmark.verifier import _check_file_exists
    (tmp_path / "simulation_result.json").write_text(json.dumps({
        "schema_version": "1.0",
        "status": "placeholder_pre_shipped_by_harness",
        "numerical_metrics": {},
    }))
    task = {
        "id": "TEST_PLACEHOLDER",
        "required_artifacts": ["simulation_result.json"],
    }
    checks = _check_file_exists(task, tmp_path)
    assert len(checks) == 1
    assert checks[0].name == "artifact:simulation_result.json"
    assert not checks[0].passed, (
        f"Pre-shipped placeholder must not satisfy artifact check; "
        f"got {checks[0]}")
    assert "placeholder" in checks[0].detail.lower()

    # Sanity: a real (non-placeholder) JSON does pass the artifact check.
    (tmp_path / "simulation_result.json").write_text(json.dumps({
        "schema_version": "1.0",
        "numerical_metrics": {"snr_db": [0, 5, 10],
                              "ber_simulated": [0.1, 0.01, 1e-4]},
    }))
    checks = _check_file_exists(task, tmp_path)
    assert checks[0].passed, f"Real output should pass; got {checks[0]}"


def test_nonempty_scene_does_not_trip_plausibility(tmp_path: Path):
    """Sanity: a real (small) scene with one room one furniture doesn't
    fail the new plausibility guard."""
    (tmp_path / "scene_state.json").write_text(json.dumps({
        "schema_version": "1.0",
        "rooms": [{
            "room_type": "bedroom",
            "dimensions": [4.0, 3.0, 2.5],
            "furniture": [{"id": "bed", "position": [2.0, 1.5],
                           "dimensions": [1.5, 2.0], "theta": 0}]
        }],
    }))
    (tmp_path / "simulation_result.json").write_text("{}")
    task = {
        "id": "TEST_T0",
        "tier": "T0_scene_gen",
        "capability": "scene_gen",
        "required_artifacts": ["scene_state.json"],
        "verifier": {"type": "code_contains", "metric": "collision_free_check"},
    }
    result = verify(task, tmp_path, exec_success=True)
    notes = " ".join(result.notes)
    assert "plausibility check failed" not in notes, (
        f"Non-empty scene tripped plausibility: {result.notes}")
