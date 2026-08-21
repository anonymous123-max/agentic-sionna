"""Per-model skill_hint verbosity. Default full (with LOOKUP); minimal
strips LOOKUP for small models that get confused by the extra option."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))

import trial


def test_full_includes_lookup(monkeypatch):
    monkeypatch.setenv("RF_SKILL_HINT_LEVEL", "full")
    trial._load_template.cache_clear()
    out = trial.build_prompt({"prompt": "x"}, "with_skill")
    assert "lookup.py" in out


def test_minimal_excludes_lookup(monkeypatch):
    monkeypatch.setenv("RF_SKILL_HINT_LEVEL", "minimal")
    trial._load_template.cache_clear()
    out = trial.build_prompt({"prompt": "x"}, "with_skill")
    assert "lookup.py" not in out
    assert "run_ber_analytical.py" in out  # RUNNABLE section still there


def test_default_is_full(monkeypatch):
    monkeypatch.delenv("RF_SKILL_HINT_LEVEL", raising=False)
    trial._load_template.cache_clear()
    out = trial.build_prompt({"prompt": "x"}, "with_skill")
    assert "lookup.py" in out


def test_invalid_level_falls_back_to_full(monkeypatch):
    monkeypatch.setenv("RF_SKILL_HINT_LEVEL", "garbage")
    trial._load_template.cache_clear()
    out = trial.build_prompt({"prompt": "x"}, "with_skill")
    assert "lookup.py" in out  # fallback works
