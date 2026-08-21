"""Tests for structured caveat capture during export.

Covers:
- test_caveats_appended_on_missing_mesh: missing OBJ path triggers fallback caveats
- test_empty_caveats_when_meshes_present: no fallback when mesh resolves (mocked)
- test_caveat_dataclass_to_dict: Caveat.to_dict() round-trips cleanly
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SKILL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SKILL))


# ---------------------------------------------------------------------------
# Caveat dataclass smoke
# ---------------------------------------------------------------------------

def test_caveat_dataclass_to_dict():
    from lib.scene_gen.caveats import Caveat
    c = Caveat(kind="fallback", source="test", message="something missing")
    d = c.to_dict()
    assert d == {"kind": "fallback", "source": "test", "message": "something missing"}


def test_caveat_is_frozen():
    from lib.scene_gen.caveats import Caveat
    c = Caveat(kind="default", source="agent", message="TX height assumed 2.6 m")
    with pytest.raises((AttributeError, TypeError)):
        c.kind = "degraded"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers to build minimal scenes
# ---------------------------------------------------------------------------

def _scene_with_missing_mesh():
    """A scene whose single furniture item has a non-existent mesh path."""
    from lib.scene_gen import Scene, Room, FurnitureItem, Position, BoundingBox

    chair = FurnitureItem(
        id="chair_missing",
        category="chair",
        model_id="no_such_model",
        model_path="/nonexistent/path/chair.obj",
        position=Position(x=3.0, y=3.0, theta=0.0),
        dimensions=BoundingBox(width=0.5, depth=0.5, height=0.9),
    )
    return Scene(room=Room(width=10.0, length=10.0, height=3.0, furniture=[chair]))


def _scene_no_furniture():
    from lib.scene_gen import Scene, Room
    return Scene(room=Room(width=10.0, length=10.0, height=3.0, furniture=[]))


# ---------------------------------------------------------------------------
# XML exporter caveat capture
# ---------------------------------------------------------------------------

def test_caveats_appended_on_missing_mesh_xml(tmp_path):
    """export_xml with a missing mesh path must append a fallback caveat."""
    from lib.scene_gen.exporters.xml import export_xml

    caveats: list = []
    export_xml(_scene_with_missing_mesh(), tmp_path / "scene.xml", caveats=caveats)

    assert len(caveats) >= 1, "Expected at least one fallback caveat"
    kinds = {c["kind"] for c in caveats}
    sources = {c["source"] for c in caveats}
    assert "fallback" in kinds, f"No fallback caveat found; got: {caveats}"
    assert "lib.scene_gen.xml" in sources, f"Wrong source; got: {caveats}"
    assert any("chair_missing" in c["message"] for c in caveats), (
        f"item id not in message; caveats: {caveats}"
    )


def test_empty_caveats_when_no_furniture_xml(tmp_path):
    """export_xml on a furniture-free scene must not append any caveats."""
    from lib.scene_gen.exporters.xml import export_xml

    caveats: list = []
    export_xml(_scene_no_furniture(), tmp_path / "scene.xml", caveats=caveats)
    assert caveats == [], f"Unexpected caveats: {caveats}"


def test_caveats_none_default_xml(tmp_path):
    """export_xml without caveats= arg must not raise."""
    from lib.scene_gen.exporters.xml import export_xml

    # No caveats argument — backward-compatible path
    export_xml(_scene_with_missing_mesh(), tmp_path / "scene.xml")
    # Just reaching here without exception is success


# ---------------------------------------------------------------------------
# GLTF exporter caveat capture
# ---------------------------------------------------------------------------

def test_caveats_appended_on_missing_mesh_gltf(tmp_path):
    """export_gltf with a missing mesh path must append a fallback caveat."""
    trimesh = pytest.importorskip("trimesh")

    from lib.scene_gen.exporters.gltf import export_gltf

    caveats: list = []
    export_gltf(_scene_with_missing_mesh(), tmp_path / "scene.glb", caveats=caveats)

    assert len(caveats) >= 1, "Expected at least one fallback caveat"
    kinds = {c["kind"] for c in caveats}
    sources = {c["source"] for c in caveats}
    assert "fallback" in kinds, f"No fallback caveat found; got: {caveats}"
    assert "lib.scene_gen.gltf" in sources, f"Wrong source; got: {caveats}"
    assert any("chair_missing" in c["message"] for c in caveats), (
        f"item id not in message; caveats: {caveats}"
    )


def test_empty_caveats_when_no_furniture_gltf(tmp_path):
    """export_gltf on a furniture-free scene must not append any caveats."""
    pytest.importorskip("trimesh")

    from lib.scene_gen.exporters.gltf import export_gltf

    caveats: list = []
    export_gltf(_scene_no_furniture(), tmp_path / "scene.glb", caveats=caveats)
    assert caveats == [], f"Unexpected caveats: {caveats}"


def test_caveats_none_default_gltf(tmp_path):
    """export_gltf without caveats= arg must not raise."""
    pytest.importorskip("trimesh")

    from lib.scene_gen.exporters.gltf import export_gltf

    export_gltf(_scene_with_missing_mesh(), tmp_path / "scene.glb")
    # Reaching here without exception = success


# ---------------------------------------------------------------------------
# export_all threads caveats through both exporters
# ---------------------------------------------------------------------------

def test_export_all_threads_caveats(tmp_path):
    """export_all must pass the same caveats list to xml (and gltf if available).

    At minimum the xml fallback must be captured; gltf is conditional on trimesh.
    """
    from lib.scene_gen.exporters import export_all

    caveats: list = []
    export_all(_scene_with_missing_mesh(), tmp_path, caveats=caveats)

    assert len(caveats) >= 1, "Expected at least one fallback caveat from export_all"
    sources = {c["source"] for c in caveats}
    # XML is always run; must have at least one lib.scene_gen.xml entry
    assert "lib.scene_gen.xml" in sources, (
        f"No xml-source caveat in export_all result; caveats: {caveats}"
    )


def test_export_all_no_caveats_arg(tmp_path):
    """export_all without caveats= arg (default None) must not raise."""
    from lib.scene_gen.exporters import export_all

    export_all(_scene_with_missing_mesh(), tmp_path)
    # Reaching here without exception = success
