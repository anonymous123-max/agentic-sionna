"""Round-trip validator: ensure exported files contain expected items.

Returns a list of warnings rather than raising — exports should not fail
validation, only flag suspicious round-trip mismatches.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List

if TYPE_CHECKING:
    from ..models import Scene


def validate_export(
    scene: "Scene",
    paths: dict[str, str | Path],
    caveats: list | None = None,
) -> List[str]:
    """Confirm written files exist and contain expected counts.

    Each entry in `paths` is a {format_name: file_path} mapping. Validation
    is best-effort: missing files yield warnings; XML/GLB are sniff-checked
    only (parsing libraries are NOT required to be installed).

    If ``caveats`` is a list, round-trip count mismatches append a structured
    Caveat entry (kind="fallback") in addition to the plain-string issue.
    """
    issues: List[str] = []

    for fmt, p in paths.items():
        path = Path(p)
        if not path.exists():
            issues.append(f"{fmt}: file not written ({path})")
            continue
        if path.stat().st_size < 64:
            issues.append(f"{fmt}: suspiciously small ({path.stat().st_size}B)")

    expected_furniture_count = (
        len(scene.room.furniture) if scene.room is not None else 0
    )
    expected_buildings = (
        len(scene.outdoor.buildings) if scene.outdoor is not None else 0
    )

    # XML sniff: count <shape ... type="obj" ...> as furniture
    if "xml" in paths:
        xml_path = Path(paths["xml"])
        if xml_path.exists():
            text = xml_path.read_text(errors="ignore")
            obj_count = text.count('type="obj"')
            if obj_count != expected_furniture_count:
                msg = (
                    f"xml: furniture count mismatch — expected "
                    f"{expected_furniture_count}, found {obj_count}"
                )
                issues.append(msg)
                if caveats is not None:
                    from ..caveats import Caveat
                    caveats.append(Caveat(
                        kind="fallback",
                        source="lib.scene_gen.validator",
                        message=f"Round-trip count mismatch: {msg}",
                    ).to_dict())
            cube_count = text.count('type="cube"')
            if cube_count != expected_buildings:
                msg = (
                    f"xml: building count mismatch — expected "
                    f"{expected_buildings}, found {cube_count}"
                )
                issues.append(msg)
                if caveats is not None:
                    from ..caveats import Caveat
                    caveats.append(Caveat(
                        kind="fallback",
                        source="lib.scene_gen.validator",
                        message=f"Round-trip count mismatch: {msg}",
                    ).to_dict())

    return issues


def confirm_paths_present(paths: Iterable[str | Path]) -> List[str]:
    """Quick check that all paths exist."""
    missing: List[str] = []
    for p in paths:
        if not Path(p).exists():
            missing.append(str(p))
    return missing
