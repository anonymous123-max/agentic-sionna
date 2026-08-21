"""Generate the 3 floorplan PNG fixtures used by U149-U153 tasks.

Run from the repo root:
    python3 benchmark/scenes/floorplans/_generate.py

The script creates:
    benchmark/scenes/floorplans/office/floor.png      + scene_state.json (ground truth)
    benchmark/scenes/floorplans/apartment/floor.png   + scene_state.json (ground truth)
    benchmark/scenes/floorplans/warehouse/floor.png   + scene_state.json (ground truth)

NOTE: scene_state.json is ground truth for human verification — it is NOT shown to agents.
The agent only receives the PNG path via the task prompt.

SCENE NOTES (the actual model uses a single Room per Scene, not a list):
- office:    8x6 m, modelled as a single 8x6 room representing a 2-zone office
- apartment: 10x8 m, single open-plan room representing the combined footprint
- warehouse: 20x15 m, single large storage room
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / ".claude/skills/rf-simulator"))

from lib.scene_gen import Room, Scene, export_png
from lib.scene_gen.exporters import export_xml


def office_scene() -> Scene:
    """8x6 m office — represented as a single open-plan room."""
    room = Room(width=8.0, length=6.0, height=2.5)
    return Scene(room=room)


def apartment_scene() -> Scene:
    """10x8 m apartment footprint — full floor as single room."""
    room = Room(width=10.0, length=8.0, height=2.5)
    return Scene(room=room)


def warehouse_scene() -> Scene:
    """20x15 m single-bay warehouse."""
    room = Room(width=20.0, length=15.0, height=4.0)
    return Scene(room=room)


def save_scene_state(scene: Scene, path: Path) -> None:
    """Write a compact scene_state.json for ground-truth reference."""
    room = scene.room
    state = {
        "room": {
            "width": room.width,
            "length": room.length,
            "height": room.height,
            "furniture": [],
        },
        "transmitters": [],
    }
    path.write_text(json.dumps(state, indent=2))


def main() -> None:
    out_base = REPO / "benchmark/scenes/floorplans"

    configs = [
        ("office", office_scene, "8x6 m two-zone office"),
        ("apartment", apartment_scene, "10x8 m apartment footprint"),
        ("warehouse", warehouse_scene, "20x15 m single-bay warehouse"),
    ]

    for name, scene_fn, desc in configs:
        out_dir = out_base / name
        out_dir.mkdir(parents=True, exist_ok=True)

        scene = scene_fn()

        # PNG — the fixture the agent sees
        png_path = export_png(scene, out_dir / "floor.png")
        print(f"  {name}: PNG  -> {png_path} ({png_path.stat().st_size // 1024} KB)")

        # scene_state.json — ground truth, NOT given to agent
        json_path = out_dir / "scene_state.json"
        save_scene_state(scene, json_path)
        print(f"  {name}: JSON -> {json_path}")

        print(f"         ({desc})")


if __name__ == "__main__":
    main()
