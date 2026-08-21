"""Verifier check: planning_state.json shows iterative convergence per
iterative-planning-protocol.md. Used by Phase T iterative tasks."""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))


def _state(history, mode="MACRO", stop=False):
    return {
        "iteration": len(history),
        "mode": mode,
        "deployment": {"APs": [{"id": "AP1", "x": 5, "y": 5, "z": 2.5,
                                  "power_dbm": 20}]},
        "history": history,
        "stop": stop,
    }


def test_passes_with_monotone_improvement(tmp_path):
    from verifier import check_iterative_convergence
    (tmp_path / "planning_state.json").write_text(json.dumps(_state([
        {"iter": 1, "coverage": 0.65, "action_taken": "init"},
        {"iter": 2, "coverage": 0.78, "action_taken": "reposition"},
        {"iter": 3, "coverage": 0.92, "action_taken": "reposition"},
    ], stop=True)))
    task = {"verifier": {"metric": "iterative_convergence",
                          "min_iterations": 2, "min_improvement": 0.10}}
    assert check_iterative_convergence(task, tmp_path).passed


def test_fails_when_no_improvement(tmp_path):
    from verifier import check_iterative_convergence
    (tmp_path / "planning_state.json").write_text(json.dumps(_state([
        {"iter": 1, "coverage": 0.65},
        {"iter": 2, "coverage": 0.66},
        {"iter": 3, "coverage": 0.65},
    ])))
    task = {"verifier": {"metric": "iterative_convergence",
                          "min_iterations": 2, "min_improvement": 0.10}}
    r = check_iterative_convergence(task, tmp_path)
    assert not r.passed
    assert "improvement" in r.detail.lower()


def test_fails_when_too_few_iterations(tmp_path):
    from verifier import check_iterative_convergence
    (tmp_path / "planning_state.json").write_text(json.dumps(_state([
        {"iter": 1, "coverage": 0.95},
    ])))
    task = {"verifier": {"metric": "iterative_convergence",
                          "min_iterations": 2, "min_improvement": 0.10}}
    r = check_iterative_convergence(task, tmp_path)
    assert not r.passed
    assert "iter" in r.detail.lower()


def test_fails_when_file_missing(tmp_path):
    from verifier import check_iterative_convergence
    task = {"verifier": {"metric": "iterative_convergence"}}
    r = check_iterative_convergence(task, tmp_path)
    assert not r.passed
