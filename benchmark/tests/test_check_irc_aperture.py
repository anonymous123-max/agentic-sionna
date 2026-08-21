"""Verifier check: scene_state.json's habitable rooms meet IRC §R303 8%
window aperture rule. Used by Phase Q's IRC-compliance tasks."""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmark"))


def _write_scene(tmp_path, room, windows):
    scene = {
        "rooms": [{
            "room_type": room.get("type", "living"),
            "dimensions": [room["w"], room["l"]],
            "windows": windows,
        }],
    }
    (tmp_path / "scene_state.json").write_text(json.dumps(scene))


def test_passes_when_aperture_at_minimum(tmp_path):
    from verifier import check_irc_aperture
    # 6×5 = 30 m² floor, need ≥2.4 m² aperture; 2 windows of 1.2×1.0 = 2.4 m²
    _write_scene(tmp_path, {"type": "living", "w": 6.0, "l": 5.0}, [
        {"width": 1.2, "height": 1.0, "wall": "south"},
        {"width": 1.2, "height": 1.0, "wall": "south"},
    ])
    task = {"verifier": {"metric": "irc_aperture", "min_ratio": 0.08}}
    result = check_irc_aperture(task, tmp_path)
    assert result.passed, result.detail


def test_fails_when_aperture_below_minimum(tmp_path):
    from verifier import check_irc_aperture
    _write_scene(tmp_path, {"type": "living", "w": 6.0, "l": 5.0}, [
        {"width": 0.5, "height": 1.0, "wall": "south"},
    ])
    task = {"verifier": {"metric": "irc_aperture", "min_ratio": 0.08}}
    result = check_irc_aperture(task, tmp_path)
    assert not result.passed
    assert "below" in result.detail.lower() or "<" in result.detail


def test_passes_for_non_habitable_rooms(tmp_path):
    from verifier import check_irc_aperture
    _write_scene(tmp_path, {"type": "storage", "w": 2.0, "l": 2.0}, [])
    task = {"verifier": {"metric": "irc_aperture", "min_ratio": 0.08}}
    result = check_irc_aperture(task, tmp_path)
    assert result.passed


def test_fails_when_windows_not_on_perimeter(tmp_path):
    from verifier import check_irc_aperture
    _write_scene(tmp_path, {"type": "living", "w": 6.0, "l": 5.0}, [
        {"width": 1.2, "height": 1.0, "wall": "interior"},
        {"width": 1.2, "height": 1.0, "wall": "south"},
    ])
    task = {"verifier": {"metric": "irc_aperture", "min_ratio": 0.08}}
    result = check_irc_aperture(task, tmp_path)
    assert not result.passed
    assert "perimeter" in result.detail.lower()
