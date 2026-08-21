# Scene Generation Library (`lib/scene_gen/`)

Importable Python utilities for room-and-furniture scene generation.
Use these instead of re-deriving collision, exporters, or layout per task.
The pure-`numpy`/`scipy` core has no Sionna dependency, so it works in
environments without GPU or Sionna RT.

## Quick start

```python
import sys
sys.path.insert(0, "$RF_SKILL_DIR")  # set by harness
from lib.scene_gen import (
    Scene, Room, place_furniture, place_tx, export_all, validate_scene,
)

room = Room(width=10, length=8, height=2.7)
scene = Scene(room=room, frequency_hz=3.5e9)
scene = place_furniture(scene, [
    {"type": "desk", "dims": (1.4, 0.7, 0.75)},
    {"type": "office chair", "dims": (0.55, 0.55, 0.95)},
])
scene = place_tx(scene)         # AP at room center, h=2.5m
print(validate_scene(scene))    # [] = clean
paths = export_all(scene, "out/")  # {'xml':..., 'png':..., 'gltf':...}
```

## Module map

| Module | Purpose | When to use |
|---|---|---|
| `models` | Frozen Pydantic v2 Scene/Room/FurnitureItem/Position/Door/Window/Building/Road/Tree/GroundPlane/OutdoorScene/Transmitter/Receiver | Always — single source of truth |
| `geometry` | rotated-rectangle corners, AABB overlap, in-bounds, rotate-2D, Shapely `furniture_polygon` | Custom placement loops |
| `constraints` | wall_affinity / in_room / collision / pathway costs + `validate_scene(scene)` | Pre-export sanity check |
| `optimizer` | `LayoutOptimizer`, `optimize_layout`, friendly `place_furniture`, `place_tx` | Auto-layout from a request dict |
| `exporters/png` | 2D top-down floor plan (matplotlib) | Documentation, paper figures |
| `exporters/xml` | Mitsuba 3.0 XML for Sionna RT | Required for Sionna RT load_scene |
| `exporters/gltf` | GLB for browsers / Blender (trimesh) | Interactive 3D viewer |
| `exporters/materials` | ITU radio material map by furniture category | Resolved internally by exporters |
| `exporters/validator` | Round-trip count check (`validate_export(scene, paths)`) | Final pre-submit gate |

## Critical invariants (do NOT violate)

- **orientation_offset rule:** every exporter MUST add `orientation_offset` to `position.theta` when computing the final rotation. PNG, XML, and GLTF all comply. If you write a custom exporter, copy the same rule.
- **Coordinate system:** SW origin, X east, Y north, Z up, theta=0 = facing north.
- **GLB = Y-up; XML = Z-up.** The GLTF exporter does NOT pre-rotate furniture (browser scene root is Y-up). The XML exporter rotates +90° around X to lift Y-up models into Sionna's Z-up world.
- **Frozen models:** mutate via `scene.model_copy(update={...})`, never direct attribute set. Pydantic v2 raises `ValidationError` on direct mutation.
- **NaN guard for 3D-FUTURE catalog:** `(m.get("category") or "")` — null categories are common.
- **Margin tolerances:** `optimizer` puts furniture against walls; the validator flags items extending past the polygon by more than ~0.05m. If validator flags edge cases at the wall, that's expected — call `model_copy(update={'position': Position(x=..., y=..., theta=...)})` to nudge.

## When NOT to use the library

- Task asks for a flat `scene_state.json` fixture (verifier just checks JSON
  field presence, no collision logic) — use `templates/template_scene.py`
  instead, which writes the canonical schema directly.
- Task is purely Sionna PHY (BER, OFDM, MIMO, neural receivers) — there's
  no scene to generate; use the relevant `template_*.py`.

## Optional dependencies

- `pip install matplotlib` — required for PNG export (skipped silently if missing in `export_all`)
- `pip install trimesh` — required for GLB export (skipped silently)
- `pip install osmnx pyproj` — required only if calling `lib.scene_gen` outdoor utilities (lazy-imported)

The XML exporter depends only on stdlib + the model objects, so it always works.

## IRC window placement

For habitable rooms (living, bedroom, kitchen, dining, office, study),
place IRC §R303-compliant windows automatically. Aperture meets ≥8% of
floor area.

```python
from lib.scene_gen import Room, place_irc_windows

room = Room(width=6.0, length=5.0, height=2.7, furniture=[])
windows = place_irc_windows(room, room_type="living")
# Returns a list of Window objects with width/height/wall/position/sill_height.
# Total aperture is ≥ 8% of room.width * room.length.
```

For non-habitable rooms (storage, mechanical, closet, bathroom_small,
garage, utility) `place_irc_windows()` returns `[]`.

The Window model itself doesn't carry a material field — glass material
(ε_r ≈ 6.3) is assigned at XML export time by the exporter based on
object type.

## Mesh resolution

**Mesh resolution:** `FurnitureItem.get_mesh_path()` returns `model_file` if set, else `{model_path}/raw_model.obj`. To use the 3D-FUTURE catalog, set `model_path = $FURNITURE_CATALOG_PATH/<model_id>`. The benchmark harness exports `$FURNITURE_CATALOG_PATH` automatically when the catalog is on disk.

## See also

- `core-patterns.md` — the principle-level snippets (kept for cases where a custom approach is preferable)
- `export-formats.md` — the orientation-offset rule explained at length
- `scene-state-schema.md` — the canonical JSON schema used by the harness
- `data-sources.md` — 3D-FUTURE catalog discovery

## Implementation files

- Exporters: `lib/scene_gen/exporters/png.py`, `lib/scene_gen/exporters/xml.py`, `lib/scene_gen/exporters/gltf.py`, `lib/scene_gen/exporters/materials.py`, `lib/scene_gen/exporters/validator.py`
- Window placement: `lib/scene_gen/windows.py`
- Layout optimizer: `lib/scene_gen/optimizer.py`
- Pydantic models: `lib/scene_gen/models.py`
- Constraints (cost functions): `lib/scene_gen/constraints.py`
- Geometry helpers: `lib/scene_gen/geometry.py`
