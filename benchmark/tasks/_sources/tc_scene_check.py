"""Pre-flight check: just generate scene_state.json for each of the 20 TC scenes.

No RF simulation — verify the 20 scene designs are buildable.
Output: tc_scene_check.json (20 tasks, one per scene).
"""
from __future__ import annotations
import json
from pathlib import Path

# Reuse the scene definitions from tc_chained_gen.py
import sys
sys.path.insert(0, str(Path(__file__).parent))
from tc_chained_gen import SCENES, scene_to_phrase, scene_furniture_token

OUT = Path(__file__).parent / "tc_scene_check.json"

TASKS = []
for scene in SCENES:
    sid, diff, name, dims, furn, walls, rtype = scene
    scene_phrase = scene_to_phrase(scene)
    furn_tok = scene_furniture_token(scene)
    id = f"TCsc_{sid}"
    prompt = (f"Generate {scene_phrase}. Write a valid scene_state.json with "
              f"rooms, walls, furniture, materials. No simulation needed for this task.")
    distractor = (f"Wrong: putting furniture outside the room bounds or overlapping. "
                  f"Right: place each item at a unique position inside the room and check AABB.")
    task = {
        "id": id,
        "origin": "scene_check",
        "origin_id": id,
        "tier": "TC_scene_check",
        "capability": "scene_only",
        "difficulty": diff,
        "split": "train",
        "name": f"Scene-only check: {sid} ({name})",
        "prompt": prompt,
        "distractor": distractor,
        "scene_path": None,
        "required_artifacts": ["scene_state.json"],
        "assertions": [
            "must create scene state",
            "must be collision free",
            "must be in bounds",
            "must be loadable by Sionna RT",
            f"expect room dims {dims[0]} m × {dims[1]} m",
            f"expect expected_furniture={furn}",
        ],
        "verifier": {
            "type": "composite",
            "subchecks": [
                {"key": "must_create_scene_state", "type": "file_exists"},
                {"metric": "collision_free_check",  "type": "code_contains"},
                {"metric": "in_bounds_check",       "type": "code_contains"},
                {"metric": "sionna_loadable_check", "type": "code_contains"},
                {"metric": furn_tok,                "type": "code_contains"},
            ]
        }
    }
    TASKS.append(task)

out_doc = {
    "version": "1.0",
    "tier": "TC_scene_check",
    "design": "20 scene-only sanity-check tasks (no RF simulation)",
    "count": len(TASKS),
    "tasks": TASKS,
}
OUT.write_text(json.dumps(out_doc, indent=2))
print(f"Wrote {len(TASKS)} scene-only tasks -> {OUT}")
