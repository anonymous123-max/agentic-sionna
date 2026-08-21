"""trial.py env exports — confirm FURNITURE_CATALOG_PATH is set when the
catalog exists, and not set otherwise."""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))


def test_catalog_path_set_when_dir_exists(tmp_path, monkeypatch):
    fake_catalog = tmp_path / "3D-FUTURE-model"
    fake_catalog.mkdir()
    (fake_catalog / "model_info.json").write_text("[]")
    # Make Path.home() return our tmp_path so the candidate path matches.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    import importlib
    import trial
    importlib.reload(trial)
    env = trial._build_invoke_env({"id": "X"}, "with_skill")
    assert env.get("FURNITURE_CATALOG_PATH") == str(fake_catalog)


def test_catalog_path_unset_when_dir_missing(tmp_path, monkeypatch):
    # tmp_path is empty — no 3D-FUTURE-model subdir
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    import importlib
    import trial
    importlib.reload(trial)
    env = trial._build_invoke_env({"id": "X"}, "with_skill")
    assert "FURNITURE_CATALOG_PATH" not in env
