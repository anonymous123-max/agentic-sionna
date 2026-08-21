"""Mitsuba 3.0 XML exporter for Sionna RT.

Invariants:
- Mitsuba `version="3.0.0"`
- ITU radio material BSDFs (`itu-radio-material`)
- Z-up coordinate system (X east, Y north, Z up) — matches Sionna
- Furniture: applies `position.theta + orientation_offset` (the rule)
- Furniture meshes: rotate +90° around X to lift Y-up models to Z-up
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree.ElementTree import Element, SubElement, tostring

from .materials import resolve_material

if TYPE_CHECKING:
    from ..models import Scene


def export_xml(scene: "Scene", output_path: str | Path,
               caveats: list | None = None) -> Path:
    """Write a Mitsuba 3.0 XML for Sionna RT loading."""
    if scene.room is None and scene.outdoor is None:
        raise ValueError("export_xml requires scene.room or scene.outdoor")

    root = Element("scene", version="3.0.0")
    SubElement(root, "integrator", type="path")

    # Mitsuba requires at least one <sensor>; Sionna RT places its own
    # TX/RX cameras programmatically and ignores this stub, but the XML
    # validator rejects scenes without one.
    sensor = SubElement(root, "sensor", type="perspective")
    SubElement(sensor, "float", name="fov", value="45")
    sensor_t = SubElement(sensor, "transform", name="to_world")
    SubElement(sensor_t, "lookat",
               origin="0,0,5", target="0,0,0", up="0,1,0")
    sampler = SubElement(sensor, "sampler", type="independent")
    SubElement(sampler, "integer", name="sample_count", value="1")
    film = SubElement(sensor, "film", type="hdrfilm")
    SubElement(film, "integer", name="width", value="64")
    SubElement(film, "integer", name="height", value="64")

    # Materials BSDF dictionary (one per unique material we'll cite)
    used_materials: set[str] = set()

    if scene.room is not None:
        room = scene.room
        floor_mat = "concrete"
        used_materials.add(floor_mat)
        _add_floor(root, room, floor_mat)
        wall_mat = "concrete"
        used_materials.add(wall_mat)
        _add_walls(root, room, wall_mat)
        # Furniture
        for item in room.furniture:
            mat = resolve_material(_furniture_material(item))
            used_materials.add(mat)
            _add_furniture(root, item, mat, caveats=caveats)

    if scene.outdoor is not None:
        out = scene.outdoor
        used_materials.add(out.ground.material)
        _add_ground(root, out, out.ground.material)
        for b in out.buildings:
            mat = resolve_material(b.material)
            used_materials.add(mat)
            _add_building(root, b, mat)

    # BSDF block (insert at top)
    bsdf_block = []
    for mat in sorted(used_materials):
        b = Element("bsdf", type="radio-material", id=f"mat-itu_{mat}")
        SubElement(b, "string", name="material_name", value=f"itu_{mat}")
        bsdf_block.append(b)
    for i, b in enumerate(bsdf_block):
        root.insert(1 + i, b)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(tostring(root, encoding="utf-8"))
    return output


def _furniture_material(item) -> str:
    """Pick ITU material for a furniture item."""
    from .materials import get_material_for_category
    return get_material_for_category(item.category)


def _add_floor(root: Element, room, mat: str) -> None:
    s = SubElement(root, "shape", type="rectangle")
    SubElement(s, "ref", id=f"mat-itu_{mat}")
    t = SubElement(s, "transform", name="to_world")
    SubElement(t, "scale", x=str(room.width / 2), y=str(room.length / 2), z="1")
    SubElement(t, "translate",
               x=str(room.width / 2), y=str(room.length / 2), z="0")


def _add_walls(root: Element, room, mat: str) -> None:
    h = room.height
    walls = [
        # name, length, theta_deg around z, translate
        # theta values derived so that after rotate-X(90°) then rotate-Z(theta),
        # each wall's normal [0,0,1] ends up pointing toward the room interior:
        #   south (y=0):  needs +Y → theta=180°
        #   north (y=L):  needs -Y → theta=0°
        #   west  (x=0):  needs +X → theta=90°   (unchanged)
        #   east  (x=W):  needs -X → theta=270°  (unchanged)
        ("south", room.width,  180, (room.width / 2, 0,            h / 2)),
        ("north", room.width,    0, (room.width / 2, room.length,  h / 2)),
        ("west",  room.length,  90, (0,              room.length / 2, h / 2)),
        ("east",  room.length, 270, (room.width,     room.length / 2, h / 2)),
    ]
    for name, length, theta_deg, (tx, ty, tz) in walls:
        s = SubElement(root, "shape", type="rectangle")
        SubElement(s, "ref", id=f"mat-itu_{mat}")
        t = SubElement(s, "transform", name="to_world")
        SubElement(t, "scale", x=str(length / 2), y="1", z=str(h / 2))
        # rotate-X(90°) lifts the rectangle to vertical; rotate-Z(theta)
        # then spins it in the XY plane to aim the normal inward.
        # After X(90°), the base normal [0,0,1] → [0,-1,0]. Z(theta) then
        # maps that to [sin(theta), -cos(theta), 0], so each wall's theta
        # is chosen to produce the correct inward-pointing direction.
        SubElement(t, "rotate", x="1", angle="90")
        SubElement(t, "rotate", z="1", angle=str(theta_deg))
        SubElement(t, "translate", x=str(tx), y=str(ty), z=str(tz))


def _add_furniture(root: Element, item, mat: str,
                   caveats: list | None = None) -> None:
    mesh_path = item.get_mesh_path()
    final_theta_rad = item.position.theta + item.orientation_offset
    final_theta_deg = math.degrees(final_theta_rad)

    if mesh_path and Path(mesh_path).exists():
        s = SubElement(root, "shape", type="obj")
        SubElement(s, "string", name="filename", value=str(mesh_path))
        SubElement(s, "ref", id=f"mat-itu_{mat}")
        t = SubElement(s, "transform", name="to_world")
        # 3D-FUTURE / GLB models are Y-up; lift to Z-up
        SubElement(t, "rotate", x="1", angle="90")
        SubElement(t, "rotate", z="1", angle=f"{final_theta_deg:.4f}")
        SubElement(t, "translate",
                   x=f"{item.position.x:.4f}",
                   y=f"{item.position.y:.4f}", z="0")
    else:
        # AABB cube fallback: w/d/h from dimensions, centered on position.
        # Mitsuba <cube> spans [-1,1]³ before transform — scale half-extents
        # to match the bounding box, then translate so the cube sits on z=0
        # with center at z=h/2.
        if caveats is not None:
            from ..caveats import Caveat
            caveats.append(Caveat(
                kind="fallback",
                source="lib.scene_gen.xml",
                message=f"OBJ mesh missing for {item.id}; emitted AABB cube",
            ).to_dict())
        bb = item.dimensions
        s = SubElement(root, "shape", type="cube")
        SubElement(s, "ref", id=f"mat-itu_{mat}")
        t = SubElement(s, "transform", name="to_world")
        SubElement(t, "scale",
                   x=f"{bb.width / 2:.4f}",
                   y=f"{bb.depth / 2:.4f}",
                   z=f"{bb.height / 2:.4f}")
        SubElement(t, "rotate", z="1", angle=f"{final_theta_deg:.4f}")
        SubElement(t, "translate",
                   x=f"{item.position.x:.4f}",
                   y=f"{item.position.y:.4f}",
                   z=f"{bb.height / 2:.4f}")


def _add_ground(root: Element, out, mat: str) -> None:
    s = SubElement(root, "shape", type="rectangle")
    SubElement(s, "ref", id=f"mat-itu_{mat}")
    t = SubElement(s, "transform", name="to_world")
    SubElement(t, "scale", x=str(out.width / 2), y=str(out.length / 2), z="1")
    SubElement(t, "translate",
               x=str(out.width / 2), y=str(out.length / 2), z="0")


def _add_building(root: Element, building, mat: str) -> None:
    """Extrude the footprint to a box approximation (skipping CGAL precision)."""
    minx, miny, maxx, maxy = building.bounds
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    sx, sy = (maxx - minx) / 2, (maxy - miny) / 2
    h = building.height
    s = SubElement(root, "shape", type="cube")
    SubElement(s, "ref", id=f"mat-itu_{mat}")
    t = SubElement(s, "transform", name="to_world")
    SubElement(t, "scale", x=f"{sx:.4f}", y=f"{sy:.4f}", z=f"{h / 2:.4f}")
    SubElement(t, "translate", x=f"{cx:.4f}", y=f"{cy:.4f}", z=f"{h / 2:.4f}")
