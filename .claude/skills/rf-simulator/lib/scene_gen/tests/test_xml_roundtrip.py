"""Mitsuba XML → Sionna RT round-trip.

Tests in this module cover:
- test_exported_xml_uses_v2_radio_material_bsdf: BSDF type spelling (no deps)
- test_xml_loads_through_sionna_rt: Full round-trip (skipped if sionna.rt missing)

Skipped if sionna.rt isn't installed (CI without GPU). When it does run,
it loads the smallest possible exported scene and asserts no exception.
The point is to catch silent breakage between the XML exporter and
Sionna RT's loader.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

SKILL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SKILL))


def _minimal_scene():
    from lib.scene_gen import Scene, Room
    return Scene(room=Room(width=10.0, length=10.0, height=3.0,
                           furniture=[]))


def test_exported_xml_uses_v2_radio_material_bsdf(tmp_path):
    """BSDF block must use the type string Sionna RT 2.0 actually registers."""
    from lib.scene_gen.exporters.xml import export_xml
    xml_path = tmp_path / "scene.xml"
    caveats: list = []
    export_xml(_minimal_scene(), xml_path, caveats=caveats)
    assert caveats == [], f"Unexpected caveats on furniture-free scene: {caveats}"
    text = xml_path.read_text()
    # Sionna RT 2.0 expects radio-material; v0.x/v1.x used itu-radio-material.
    assert 'type="radio-material"' in text, \
        f"BSDF type wrong; head of XML:\n{text[:600]}"
    assert 'type="itu-radio-material"' not in text, \
        "legacy v0/v1 BSDF type still present"


def test_xml_loads_through_sionna_rt(tmp_path):
    """Exported XML for an empty 10x10x3 m room must load via load_scene()."""
    sionna_rt = pytest.importorskip("sionna.rt")
    from lib.scene_gen.exporters.xml import export_xml
    xml_path = tmp_path / "scene.xml"
    export_xml(_minimal_scene(), xml_path)
    try:
        scene = sionna_rt.load_scene(str(xml_path))
    except Exception as e:
        pytest.fail(f"sionna.rt.load_scene() rejected exported XML: {e!r}\n"
                    f"--- XML head ---\n{xml_path.read_text()[:1000]}")
    assert scene is not None


def test_wall_normals_face_inward(tmp_path):
    """For a 10x10 room, each wall's normal must point toward room center.

    Parses the four wall transforms and applies each rotate to the default
    Mitsuba rectangle normal [0, 0, 1]. The dot product with the vector
    from wall center to room center (5, 5, 1.5) must be positive.
    """
    import xml.etree.ElementTree as ET
    import math
    from lib.scene_gen.exporters.xml import export_xml
    xml_path = tmp_path / "scene.xml"
    export_xml(_minimal_scene(), xml_path)
    tree = ET.parse(xml_path)
    rectangles = [s for s in tree.getroot().findall("shape")
                  if s.get("type") == "rectangle"]
    # Skip floor (centered at z=0); keep only walls (centered at z=h/2)
    wall_shapes = []
    for s in rectangles:
        translate = s.find(".//translate")
        if translate is not None and float(translate.get("z", "0")) > 0:
            wall_shapes.append(s)
    assert len(wall_shapes) == 4, (
        f"expected 4 walls, got {len(wall_shapes)}")

    center = (5.0, 5.0, 1.5)
    for s in wall_shapes:
        translate = s.find(".//translate")
        wx = float(translate.get("x"))
        wy = float(translate.get("y"))
        toward = (center[0] - wx, center[1] - wy, 0.0)
        # Apply each <rotate> in document order to base normal [0,0,1]
        n = [0.0, 0.0, 1.0]
        for rot in s.findall(".//rotate"):
            ang = math.radians(float(rot.get("angle")))
            ax = rot.get("x"); ay = rot.get("y"); az = rot.get("z")
            c, sn = math.cos(ang), math.sin(ang)
            if ax == "1":
                n = [n[0], c * n[1] - sn * n[2], sn * n[1] + c * n[2]]
            elif ay == "1":
                n = [c * n[0] + sn * n[2], n[1], -sn * n[0] + c * n[2]]
            elif az == "1":
                n = [c * n[0] - sn * n[1], sn * n[0] + c * n[1], n[2]]
        dot = n[0] * toward[0] + n[1] * toward[1]
        assert dot > 0, (
            f"wall at ({wx},{wy}) normal {n} not facing inward "
            f"(toward={toward}, dot={dot})")


def test_furniture_without_mesh_uses_cube_fallback(tmp_path):
    """Furniture with no resolvable mesh path must fall back to <shape type='cube'>,
    not <shape type='obj' filename='placeholder.obj'>."""
    from lib.scene_gen import Scene, Room, FurnitureItem, Position, BoundingBox
    from lib.scene_gen.exporters.xml import export_xml
    chair = FurnitureItem(
        id="c1", category="chair",
        model_id="", model_path="",
        position=Position(x=2.0, y=3.0, theta=0.0),
        dimensions=BoundingBox(width=0.5, depth=0.5, height=0.9),
    )
    scene = Scene(room=Room(width=10.0, length=10.0, height=3.0,
                            furniture=[chair]))
    xml_path = tmp_path / "scene.xml"
    export_xml(scene, xml_path)
    text = xml_path.read_text()
    # No reference to a non-existent placeholder OBJ
    assert "placeholder.obj" not in text, (
        f"placeholder.obj still emitted; XML head:\n{text[:800]}")
    # A cube shape exists for the chair
    assert 'type="cube"' in text, (
        f"no cube fallback found; XML head:\n{text[:800]}")


def test_xml_contains_sensor_element(tmp_path):
    """Mitsuba's validator requires at least one <sensor>; Sionna RT ignores it."""
    from lib.scene_gen.exporters.xml import export_xml
    xml_path = tmp_path / "scene.xml"
    export_xml(_minimal_scene(), xml_path)
    text = xml_path.read_text()
    assert "<sensor" in text, f"no sensor element in XML; head:\n{text[:600]}"
