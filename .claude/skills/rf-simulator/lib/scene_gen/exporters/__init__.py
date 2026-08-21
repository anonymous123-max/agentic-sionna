"""Exporters: PNG / XML / GLTF / materials / validator."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .gltf import export_gltf
from .materials import (
    CATEGORY_MATERIAL_MAP,
    ITU_MATERIALS,
    get_material_for_category,
    resolve_material,
)
from .png import export_png
from .validator import validate_export
from .xml import export_xml

if TYPE_CHECKING:
    from ..models import Scene

__all__ = [
    "export_png", "export_xml", "export_gltf",
    "get_material_for_category", "resolve_material",
    "ITU_MATERIALS", "CATEGORY_MATERIAL_MAP",
    "validate_export",
    "export_all",
]


def export_all(
    scene: "Scene",
    output_dir: str | Path,
    caveats: list | None = None,
) -> dict[str, Path]:
    """Write all three formats to ``output_dir``. Returns the path map.

    PNG export is best-effort: skipped silently if matplotlib is missing.
    GLB export is best-effort: skipped silently if trimesh is missing.

    If ``caveats`` is a list, each exporter appends a structured Caveat dict
    for every fallback it applies (missing mesh → AABB cube, etc.). Callers
    that don't pass ``caveats`` get the old behaviour unchanged.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # XML always works (no heavy deps)
    paths["xml"] = export_xml(scene, out / "scene.xml", caveats=caveats)

    try:
        paths["png"] = export_png(scene, out / "scene.png")
    except ImportError:
        pass

    try:
        paths["gltf"] = export_gltf(scene, out / "scene.glb", caveats=caveats)
    except ImportError:
        pass

    return paths
