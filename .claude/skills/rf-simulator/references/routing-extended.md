# Extended Routing Table

The 8 quick rows in `SKILL.md` cover the common cases. Use this file when
the task type doesn't match those rows.

| Task Type | Keywords | Template / Pattern | Reference (essential) | (if needed) |
|---|---|---|---|---|
| `CHANNEL_MODEL` | CDL, TDL, UMi, UMa, RMa, fading, Doppler | — (write custom) | `channel-models.md` | — |
| `DIFF_OPTIMIZE` | gradient, differentiable, learn material | — (write custom) | `differentiable-optimization.md` | — |
| `RIS` | RIS, reconfigurable surface, phase shift | — (write custom) | `differentiable-optimization.md` | `sionna-diffraction-ris.md` |
| `OUTDOOR` | OSM, outdoor, city block, downtown | osmnx + `template_rt_coverage.py` | `data-sources.md` | — |
| `CALIBRATE` | "I measured", ground truth, calibrate | Calibration pattern | `physics-validation.md` | — |
| `SCENE_EDIT` | edit scene, move object, change material | — | `sionna-scene-editing.md` | `scene-state-schema.md` |
| `RESEARCH` | RF-3DGS, neural radiance, gaussian splatting | Discussion + scaffold | `rf-3dgs-reference.md` | — |
| `COMPOSITE` | combines above | Scene first, then simulation | per sub-task | — |

**Optional-on-demand references (load ONLY if explicitly needed):**

| Need | File |
|---|---|
| 3D viewer in output | `references/viewer-spec.md` |
| Sionna unavailable / no GPU | `references/cpu-fallback.md` |
| 3GPP path-loss formulas | `references/3gpp-models.md` |
| Physical constants | `references/static-knowledge.md` |
| Sionna API changed since v2.0 | `agents/rf-researcher.md` |
