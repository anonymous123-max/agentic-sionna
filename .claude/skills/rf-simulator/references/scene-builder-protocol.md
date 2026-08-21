# Scene Builder Protocol

Use this when the user describes a room (and optionally furniture) in natural language and expects a complete RF-ready scene as output. The deliverables are a populated `scene_state.json`, a 2-D floor-plan PNG, a GLTF mesh (Y-up), and a Mitsuba XML scene description (Z-up) — all spatially consistent.

The protocol is sequential. Do not skip ahead; each stage feeds the next with measurements the optimizer needs.

## Contents

1. [Intent parsing](#intent-parsing)
2. [Model-first 3D-FUTURE retrieval](#model-first-3d-future-retrieval)
3. [Constrained spatial optimization](#constrained-spatial-optimization)
4. [Architectural annotation](#architectural-annotation)
5. [Multi-format export and coordinate reconciliation](#multi-format-export-and-coordinate-reconciliation)

---

## Intent parsing

Parse the user prompt into a structured room spec plus a furniture list. Ask one clarifying question only when an essential field is missing; do not guess.

Essential room fields:

- `room_dims_m` — length, width, height in meters.
- `room_type` — office, residential, warehouse, etc. Drives default material assignments.
- `style_preferences` — modern / industrial / minimalist / traditional. Drives 3D-FUTURE selection.

Essential furniture fields per item:

- `category` — sofa, desk, bookshelf, etc. (controlled vocabulary).
- `count` — integer.
- `qualitative_size` or `target_dims_m` — "tall bookshelf" vs explicit 1.8×0.4×2.1 m.
- `constraints` — wall-affinity ("against north wall"), pairing ("near desk"), exclusions.

If `room_dims_m` is absent and no defaults apply, ask. Do not silently pick 10×8×3 m for a request that depends on dimensions.

---

## Model-first 3D-FUTURE retrieval

Critical ordering: retrieve the 3D model **before** running spatial optimization. The optimizer's collision and clearance constraints depend on the actual axis-aligned bounding box (AABB), and a 1.8 m vs 2.1 m bookshelf changes line-of-sight and therefore RF coverage.

For each furniture item:

1. Query 3D-FUTURE by `category`. Filter candidates by:
   - Stylistic match against `style_preferences`.
   - Dimensional match against `target_dims_m` (or qualitative-size mapping).
2. Select the best-matching model.
3. Measure the AABB via Trimesh on the selected mesh; record `aabb_xyz_m`.
4. Only after AABB is known, hand the item to the spatial optimizer.

If 3D-FUTURE has no acceptable match for a category, fall back to a primitive box with the qualitative or default dimensions, and flag the item with `provenance: "primitive_fallback"` in the scene state so the user knows the visual fidelity is reduced.

---

## Constrained spatial optimization

Layout is solved as a constrained minimization. The LLM picks which objectives apply to each item; a deterministic solver does the search.

- Solver: **SLSQP** (`scipy.optimize.minimize(method="SLSQP")`).
- Iteration cap: **500** SLSQP iterations.
- Variables: per-furniture `(x, y, theta)`. Heights `z` are taken from the AABB and the floor plane.
- Objective: weighted sum of penalty terms.

Standard weights:

| Term | Weight | Definition |
|---|---|---|
| Wall affinity | `2.0` | Distance from designated wall (when the item has a wall-affinity constraint, e.g. bookshelf against north wall). |
| Collision avoidance | `10.0` | Shapely polygon intersection between this item's 2-D footprint and every other item's footprint; weight is intentionally an order of magnitude above the others so overlaps are essentially infeasible. |
| Pathway clearance | `3.0` | Penalty for blocking the door-to-window or door-to-door minimum-width corridor (≥0.9 m clear). |
| In-room containment | `5.0` | Penalty for any vertex of the footprint exiting `room_dims_m`. |

These weights are the calibrated defaults; do not change them without recording a justification in `scene_state.optimizer.notes`.

Each item's 2-D footprint is the AABB projection to the floor plane after rotation by `theta`. Use Shapely for the polygon intersection tests.

---

## Architectural annotation

After furniture is placed, annotate openings and structural elements. Use the user's description when explicit; otherwise apply code-minimum defaults.

- **Windows.** Each habitable room must have an aperture meeting IRC 2021 minima for the room type (typically ≥5.7 ft² / ≥0.53 m² for bedrooms; smaller for other rooms). If the user did not specify window placement, place one on an exterior wall with the longest unobstructed run.
- **Doors.** Place per user description. Default width 0.9 m, height 2.1 m.
- **Partitions.** Half-height or full-height internal divisions when the user asks for them; assign `itu_plasterboard` by default.
- **Structural pillars.** Place when the user describes them; assign `itu_concrete` and treat as collision obstacles in the layout pass (re-run optimization if pillar collides with existing furniture).

Record each annotation under `scene_state.architecture` with explicit position, dimensions, and material.

---

## Multi-format export and coordinate reconciliation

Emit all three viewer/simulator artifacts plus the floor-plan PNG. They must share the same world frame up to documented up-axis conventions.

| Artifact | Purpose | Up axis | Notes |
|---|---|---|---|
| `floor_plan.png` | 2-D top-down view for review and documentation | n/a (XY only) | Color-coded by furniture category; APs marked if present. |
| `scene.gltf` | Web / Blender 3-D viewer | **Y-up** | Convert Z-up internal coords by swapping Y↔Z for vertices and rotations. |
| `scene.xml` | Sionna RT Mitsuba scene | **Z-up** | Native Mitsuba convention; no conversion needed from internal Z-up. |
| `scene_state.json` | Authoritative state for downstream tools | Z-up | Single source of truth; exporters read from here. |

**Spatial-consistency tolerance.** Before returning, verify that every furniture item's `(x, y, z)` agrees across `scene_state.json`, the GLTF mesh transform, and the Mitsuba XML `<transform>` block to within **0.1 m**. Apply `orientation_offset` consistently (see `export-formats.md`). Any item whose three positions disagree by more than 0.1 m blocks export; re-run reconciliation rather than shipping a mismatched scene.

## Related

- [scene-state-schema.md](scene-state-schema.md) — exact JSON schema for `scene_state.json`.
- [sionna-materials.md](sionna-materials.md) — ITU material catalog used during annotation.
- [export-formats.md](export-formats.md) — format matrix, GLTF / Mitsuba XML / PNG generators, `orientation_offset` rule.
- [scene-gen-library.md](scene-gen-library.md) — `lib/scene_gen` Python API that implements this protocol.
