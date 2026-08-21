"""Verifier check: action_plan.json validates against the published
schema with the closed cause taxonomy + typed action vocabulary."""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))


def _minimal_valid_plan():
    return {
        "coverage_current": 75.0, "coverage_target": 90.0,
        "blind_spots": [{"location": [3.0, 4.0], "area_m2": 1.5,
                          "cause": "wall_occlusion"}],
        "actions": [{"type": "reposition", "ap_id": "AP1",
                     "delta": [0.0, 1.5, 0.0], "expected_gain_pp": 4.0}],
        "confidence": 0.7, "stop_recommended": False,
    }


def test_passes_with_valid_plan(tmp_path):
    from verifier import check_action_plan_schema
    (tmp_path / "action_plan.json").write_text(json.dumps(_minimal_valid_plan()))
    task = {"verifier": {"metric": "action_plan_schema"}}
    assert check_action_plan_schema(task, tmp_path).passed


def test_fails_when_missing_required_key(tmp_path):
    from verifier import check_action_plan_schema
    plan = _minimal_valid_plan()
    del plan["confidence"]
    (tmp_path / "action_plan.json").write_text(json.dumps(plan))
    r = check_action_plan_schema({"verifier": {}}, tmp_path)
    assert not r.passed
    assert "confidence" in r.detail


def test_fails_with_invalid_cause(tmp_path):
    from verifier import check_action_plan_schema
    plan = _minimal_valid_plan()
    plan["blind_spots"][0]["cause"] = "alien_interference"
    (tmp_path / "action_plan.json").write_text(json.dumps(plan))
    r = check_action_plan_schema({"verifier": {}}, tmp_path)
    assert not r.passed
    assert "alien_interference" in r.detail or "cause" in r.detail.lower()


def test_fails_with_invalid_action_type(tmp_path):
    from verifier import check_action_plan_schema
    plan = _minimal_valid_plan()
    plan["actions"][0]["type"] = "teleport"
    (tmp_path / "action_plan.json").write_text(json.dumps(plan))
    r = check_action_plan_schema({"verifier": {}}, tmp_path)
    assert not r.passed


def test_fails_when_file_missing(tmp_path):
    from verifier import check_action_plan_schema
    r = check_action_plan_schema({"verifier": {}}, tmp_path)
    assert not r.passed
    assert "missing" in r.detail.lower()
