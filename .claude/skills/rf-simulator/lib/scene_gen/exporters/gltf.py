"""GLB/GLTF exporter for browser/Blender viewing.

- GLB = Y-up (do NOT pre-rotate furniture; the XML exporter handles Z-up)
- Wall node names start with "wall" so the viewer can apply translucent material
- Furniture: `final_theta = position.theta + orientation_offset` (the rule)
- Falls back to box geometry when an OBJ mesh path is missing
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Scene


def export_gltf(scene: "Scene", output_path: str | Path,
                caveats: list | None = None) -> Path:
    """Write a GLB file. Returns the path written. Lazy-imports trimesh."""
    try:
        import numpy as np
        import trimesh
    except ImportError as e:
        raise ImportError(
            "GLTF export requires trimesh + numpy. pip install trimesh numpy"
        ) from e

    if scene.room is None and scene.outdoor is None:
        raise ValueError("export_gltf requires scene.room or scene.outdoor")

    glb_scene = trimesh.Scene()

    if scene.room is not None:
        room = scene.room

        # Floor (Y-up: thickness along Y)
        floor = trimesh.creation.box(extents=[room.width, 0.02, room.length])
        try:
            floor.visual.face_colors = [40, 40, 50, 255]
        except Exception:
            pass
        floor.apply_translation([room.width / 2, -0.01, room.length / 2])
        glb_scene.add_geometry(floor, node_name="floor")

        # Walls (4 cardinal walls; node names start with "wall")
        h = room.height
        walls = [
            ("wall_south", [room.width, h, 0.15], [room.width / 2, h / 2, 0]),
            ("wall_north", [room.width, h, 0.15], [room.width / 2, h / 2, room.length]),
            ("wall_west",  [0.15, h, room.length], [0, h / 2, room.length / 2]),
            ("wall_east",  [0.15, h, room.length], [room.width, h / 2, room.length / 2]),
        ]
        for name, extents, translate in walls:
            w_mesh = trimesh.creation.box(extents=extents)
            try:
                w_mesh.visual.face_colors = [80, 80, 100, 80]
            except Exception:
                pass
            w_mesh.apply_translation(translate)
            glb_scene.add_geometry(w_mesh, node_name=name)

        # Furniture
        for item in room.furniture:
            mesh_path = item.get_mesh_path()
            if mesh_path and Path(mesh_path).exists():
                try:
                    mesh = trimesh.load(mesh_path, force="mesh")
                except Exception as exc:
                    warnings.warn(
                        f"Failed to load mesh for {item.id}: {exc}; "
                        "falling back to box",
                        stacklevel=2,
                    )
                    mesh = _fallback_box(trimesh, item)
                    if caveats is not None:
                        from ..caveats import Caveat
                        caveats.append(Caveat(
                            kind="fallback",
                            source="lib.scene_gen.gltf",
                            message=f"OBJ mesh load failed for {item.id} ({exc}); emitted AABB cube",
                        ).to_dict())
            else:
                if mesh_path:
                    warnings.warn(
                        f"Mesh missing for {item.id}: {mesh_path}; falling back to box",
                        stacklevel=2,
                    )
                if caveats is not None:
                    from ..caveats import Caveat
                    caveats.append(Caveat(
                        kind="fallback",
                        source="lib.scene_gen.gltf",
                        message=f"OBJ mesh missing for {item.id}; emitted AABB cube",
                    ).to_dict())
                mesh = _fallback_box(trimesh, item)

            # Rotate around Y by final_theta (theta_position + orientation_offset)
            final_theta_rad = item.position.theta + item.orientation_offset
            R = trimesh.transformations.rotation_matrix(final_theta_rad, [0, 1, 0])
            mesh.apply_transform(R)
            mesh.apply_translation([item.position.x, 0, item.position.y])
            glb_scene.add_geometry(mesh, node_name=item.id)

    if scene.outdoor is not None:
        out = scene.outdoor
        ground = trimesh.creation.box(extents=[out.width, 0.02, out.length])
        try:
            ground.visual.face_colors = [70, 70, 60, 255]
        except Exception:
            pass
        ground.apply_translation([out.width / 2, -0.01, out.length / 2])
        glb_scene.add_geometry(ground, node_name="ground")

        for b in out.buildings:
            minx, miny, maxx, maxy = b.bounds
            extents = [maxx - minx, b.height, maxy - miny]
            box = trimesh.creation.box(extents=extents)
            box.apply_translation([
                (minx + maxx) / 2, b.height / 2, (miny + maxy) / 2,
            ])
            glb_scene.add_geometry(box, node_name=f"building_{b.id}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    glb_scene.export(str(output), file_type="glb")
    return output


def _fallback_box(trimesh_mod, item):
    """Cardboard-box stand-in when an OBJ mesh isn't available."""
    box = trimesh_mod.creation.box(extents=[
        item.dimensions.width,
        item.dimensions.height,
        item.dimensions.depth,
    ])
    try:
        box.visual.face_colors = [180, 180, 180, 255]
    except Exception:
        pass
    return box
