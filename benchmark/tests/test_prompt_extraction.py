"""Verify prompt blobs are loaded from benchmark/prompts/ files (not
hardcoded in trial.py) and that build_prompt composes them correctly."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))

import trial


def test_prompts_dir_exists():
    assert (Path(trial.__file__).parent / "prompts" / "skill_hint_full.txt").exists()
    assert (Path(trial.__file__).parent / "prompts" / "skill_hint_minimal.txt").exists()
    assert (Path(trial.__file__).parent / "prompts" / "task_tail.txt").exists()


def test_build_prompt_with_skill_includes_lookup():
    task = {"prompt": "x", "required_artifacts": ["simulation_result.json"]}
    out = trial.build_prompt(task, "with_skill")
    assert "$RF_SKILL_DIR/scripts/lookup.py" in out  # skill hint present
    assert "debug-and-retry rule" in out             # tail present


def test_build_prompt_no_skill_excludes_skill_hint():
    task = {"prompt": "x", "required_artifacts": ["simulation_result.json"]}
    out = trial.build_prompt(task, "no_skill")
    assert "$RF_SKILL_DIR" not in out
    assert "debug-and-retry rule" in out  # tail still present


def test_build_prompt_substitutes_artifacts():
    task = {"prompt": "x", "required_artifacts": ["scene_state.json"]}
    out = trial.build_prompt(task, "with_skill")
    assert "['scene_state.json']" in out
