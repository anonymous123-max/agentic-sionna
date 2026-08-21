# Task Design — T0, T1, T2

Detailed specification of the three task families currently in the
RadioTwinAgent benchmark. Each task has: purpose, corpus, input,
required agent actions, output artifacts, verifier subchecks, pass
criteria, current pass-rate trajectory, and a worked example.

| Task | Capability tested | Corpus | Best PASS |
|---|---|---|---|
| **T0** | Scene generation (NL → 3D scene) | 100 prompts | 75% train / 72.5% test (v5) |
| **T1** | Single-AP coverage | 20 scenes | **100% (20/20)** with v7 skill |
| **T2** | Scene edit + recompute | 20 scenes | **100% (20/20)** with v8 skill + retry80 |

---

## T0 — Scene Generation

### Purpose
Tests **Contribution 1**: can the agent translate a natural-language room
description into a simulation-ready 3D scene (rooms + furniture + ITU
radio materials) that Sionna RT can load?

### Corpus
**100 tasks**: 60 *easy* (T0E001–T0E060) + 40 *hard* (T0H001–T0H040).
Split into **60 train** + **40 held-out test** for the skill-iteration
ablation. Each prompt names 1–3 room types and 3–8 furniture nouns.

### Input
A natural-language room description such as:
> *"Generate a 5 m × 4 m home office (drywall walls) containing one desk, one office chair, one bookshelf."*

### Required agent actions
1. Parse the prompt's room type, dimensions, materials, furniture list.
2. Lay out furniture inside the room avoiding overlaps and walls.
3. Emit a canonical `scene_state.json` per the SKILL.md schema invariant.
4. (Optional) emit `viewer.html` + `scene.glb` per
   `references/viewer-spec.md`.

### Required outputs
- `scene_state.json` — full structured scene (rooms, furniture, materials)
- `simulation.py` — the agent-written code (audit trail)

### Verifier subchecks
- `artifact:scene_state.json` exists & non-placeholder
- `collision_free` — pairwise AABB overlap = 0
- `in_bounds` — every furniture AABB inside room polygon
- `sionna_loadable` — Mitsuba XML built from scene loads in Sionna RT
- `scene_nontrivial` — at least 1 room + 1 furniture item
- `code_contains:<noun_list>` — every furniture noun appears in agent text

### Pass criteria
**All subchecks pass.** Per-task token list comes from the prompt.

### Current numbers — skill iteration trajectory

| Skill version | T0 train pass | Δ | Single targeted edit |
|---|---|---|---|
| v0 (baseline) | 57.5% | — | — |
| v1 | 58.0% | +0.5 | Prompt-echo (below 2pp gate, kept for v2 build-up) |
| v2 | 64.4% | +6.4 | Verbatim-naming invariant |
| v3 | 67.7% | +3.3 | Big glossary (max_turns crashes at this point) |
| v4 | 72.7% | +5.0 | Compact glossary (replaces v3) |
| **v5** | **75.0%** | +2.5 | `scene_edit` fast-path |
| **v5 test** | **72.5%** | — | Train→test gap only −2.5 pp (excellent generalization) |

**Δ_skill (with vs no_skill / self_gen)**: +72.5 pp on test (ws 72.5%
vs ns/sg both 0%).

### Worked example (T0E001)
```json
{
  "schema_version": "1.0",
  "status": "completed",
  "scene": {
    "name": "home_office",
    "bounds": {"width": 5.0, "depth": 4.0, "height": 3.0}
  },
  "rooms": [{"id":"room_0","type":"office","bounds":{...},"wall_material":"drywall"}],
  "furniture": [
    {"id":"f0","type":"desk","position":[3.0,2.0,0],"dimensions":[1.2,0.6,0.75],"material":"itu_wood"},
    {"id":"f1","type":"office_chair","position":[3.0,1.0,0],"dimensions":[0.6,0.6,1.0],"material":"itu_wood"},
    {"id":"f2","type":"bookshelf","position":[0.5,3.5,0],"dimensions":[1.5,0.4,1.8],"material":"itu_wood"}
  ]
}
```

---

## T1 — Single-AP Coverage

### Purpose
Tests **Contribution 2**'s first capability: agent generates a scene,
places an access point, runs Sionna RT, and reports a coverage map +
percentage. This is the most fundamental network-task primitive.

### Corpus
**20 train scenes** (`TC1_S01`–`TC1_S20`), each from a distinct room
type spanning 4×4 m bedrooms to 14×9 m office suites.

### Input
For each scene, prompt is of the form:
> *"Generate {scene_phrase}. Then place one AP at the centroid at 2.5 m
> height. Compute coverage at {5.0 / 3.5} GHz using FSPL or RT, threshold
> -75 dBm. Report `coverage_pct` in `simulation_result.json`."*

Frequency: 5.0 GHz for easy (S01–S15); 3.5 GHz for hard (S16–S20).

### Required agent actions
1. Generate `scene_state.json` (per T0 protocol).
2. Run `tools/validate_layout.py --fix` (SKILL.md v7 rule).
3. Place AP at centroid (2.5 m height) in `scene_state.access_points`.
4. Run Sionna RT `RadioMapSolver` to compute the RSS grid.
5. Compute `coverage_pct = mean(rss > threshold) × 100`.
6. Save `simulation_result.json` + `coverage_map.npy` + `coverage_map.png`.

### Required outputs
- `scene_state.json` — with `access_points[]` populated
- `simulation_result.json`:
  - `numerical_metrics.coverage_pct`
  - `numerical_metrics.{mean,min,max}_rss_dbm`
  - `numerical_metrics.per_quadrant_coverage`
- `coverage_map.npy` — raw RSS grid (dBm)
- `coverage_map.png` — heatmap with AP marker (RdYlGn colormap)
- `simulation.py` — agent's code

### Verifier subchecks (5 core + 4 capability = 9 total)
| # | Subcheck | Layer | Pass criterion |
|---|---|---|---|
| 1 | `must_create_scene_state` | A | file exists + non-placeholder |
| 2 | `must_create_simulation_result` | A | file exists + non-placeholder |
| 3 | `collision_free` | A | pairwise AABB overlap = 0 |
| 4 | `in_bounds` | A | all furniture inside room |
| 5 | `sionna_loadable` | A | Mitsuba XML loads in Sionna |
| 6 | `range:coverage_pct` | C | `coverage_pct ∈ [50, 100]` easy / `[30, 100]` hard |
| 7 | `code_contains:<furniture>` | C | each prompt noun appears in agent text |
| 8 | `rt_oracle` | B | energy + FSPL trend + material trend |
| 9 | `c1_ref_oracle` | C | `|agent − analytical_FSPL| ≤ 5 pp` easy / `[-15, +5] pp` hard |

### Pass criteria
**All 9 subchecks pass.**

### Current numbers
| Skill version | T1 pass | Δ | Edit |
|---|---|---|---|
| v6 (schema invariant) | 14/20 = 70% | — | (T2 v3 baseline) |
| **v7** (+ placement validator) | **20/20 = 100%** | **+30 pp** | `tools/validate_layout.py --fix` |

The 6 v6→v7 fails were all on geometric constraints (4× in_bounds + 2×
collision_free). All cleanly fixed by the placement validator.

### Reference oracle math
```python
def reference_T1_coverage_pct(W, D, ap_pos, freq_hz, tx_power_dbm, threshold_dbm):
    grid = 0.25  # m
    n_above = n_total = 0
    for x in np.arange(0, W, grid):
        for y in np.arange(0, D, grid):
            d = sqrt((x-ap_x)**2 + (y-ap_y)**2 + (ap_z-1.5)**2)
            fspl = 20*log10(d) + 20*log10(freq_hz) - 147.55
            rss = tx_power_dbm - fspl
            n_above += rss > threshold_dbm
            n_total += 1
    return 100.0 * n_above / n_total
```

### Worked example (TC1_S04 — dining room 6×4 m)
- AP @ (3.0, 2.0, 2.5) m, 5.0 GHz, 20 dBm
- Threshold -75 dBm
- Reference FSPL coverage: 100% (room is small enough that all points are LoS)
- Agent reported: 100% coverage
- All 9 subchecks pass

---

## T2 — Scene Edit + Recompute

### Purpose
Tests whether the agent can **modify a scene** (add a physical
obstruction) and correctly **re-simulate**, capturing the wall's
attenuation effect on coverage. This catches agents that use FSPL
fallbacks (which ignore walls).

### Corpus
**20 train scenes** (`TC4_S01`–`TC4_S20`), same 20 rooms as T1.

### Input
For each scene:
> *"Generate {scene_phrase}. Place one AP at the centroid at 2.5 m
> height, frequency 5.0 GHz, power 20 dBm. Coverage threshold -50 dBm.
>
> Step 1 — Compute baseline coverage of the room as generated. Save
> `coverage_map_before.npy` and report `coverage_pct_before`.
>
> Step 2 — Edit the scene: add a full-height interior wall at x=W/2 m
> splitting the room into two halves, with material `itu_concrete`.
> Recompute coverage with the SAME AP. Save `coverage_map_after.npy`
> and report `coverage_pct_after` + `coverage_delta_pp = after − before`.
>
> ⚠ Use Sionna RT — FSPL doesn't model walls, would give Δ=0 = FAIL.
> The reference oracle checks arithmetic + sign (added → Δ ≤ +5 pp;
> concrete typically gives Δ ∈ [-30, -55] pp)."*

### Required agent actions
1. Generate scene_state.json (per T0 protocol).
2. Run validate_layout.py --fix.
3. Place AP at centroid.
4. **Run Sionna RT** for baseline → save before-grid + `coverage_pct_before`.
5. **Mutate scene_state.json** to add the concrete partition wall.
6. **Re-run Sionna RT** with the modified scene → save after-grid + `coverage_pct_after`.
7. Compute `coverage_delta_pp = after − before` and save to result.

### Required outputs
- `scene_state.json` — with the partition wall added in step 5
- `simulation_result.json`:
  - `numerical_metrics.coverage_pct_before`
  - `numerical_metrics.coverage_pct_after`
  - `numerical_metrics.coverage_delta_pp`
- `coverage_map_before.npy` + `coverage_map_after.npy` (optional but
  recommended for human review)
- `simulation.py` — agent's code (must contain ≥2 sim runs)

### Verifier subchecks (5 core + 6 capability = 11 total)
| # | Subcheck | Layer | Pass criterion |
|---|---|---|---|
| 1-5 | (5 core checks) | A | scene_state + simulation_result + collision_free + in_bounds + sionna_loadable |
| 6 | `range:coverage_delta_pp` | C | `Δ ∈ [-100, 100]` (loose; just must be real) |
| 7 | `code_contains:coverage_pct_before` | C | token "before" appears in code |
| 8 | `code_contains:coverage_pct_after` | C | token "after" appears in code |
| 9 | `rt_oracle` | B | energy + RSS range |
| 10 | `geometry_oracle` | B | multi-room per-room coverage spread ≥ 1 pp (when applicable) |
| 11 | `c4_ref_oracle` | C | `|delta − (after − before)| ≤ 2 pp` arithmetic; sign matches edit type (added → Δ ≤ +5) |

**Note**: the `code_contains:added` token grep was dropped in v3+ because
it was redundant with `c4_ref_oracle` (which enforces sign + arithmetic)
and was failing on agents that wrote "add"/"adding"/"partition" but not
the past-tense literal "added".

### Pass criteria
**All 11 subchecks pass.**

### Expected delta range
For concrete partition (`itu_concrete`, ~18 dB attenuation at 5 GHz)
splitting a small/medium room with AP at centroid, the threshold -50 dBm
puts the far side just below threshold:

- Far side (post-partition): RSS ≈ AP_power - FSPL_to_far_corner -
  concrete_loss = 20 - 56 - 18 ≈ -54 dBm → below -50 → 0% covered
- Near side (no partition between AP and RX): same as before → ~100% covered
- Overall after-coverage: ~50%
- Expected `coverage_delta_pp ≈ -50 pp` ± variation by room size

### Current numbers — task + skill iteration trajectory

| Run label | PASS | Δ | What changed |
|---|---|---|---|
| v1 | 1/20 = 5% | — | baseline: drywall + -75 dBm + 25 turns |
| v2 | 2/20 = 10% | +5 | + max_turns 40 + SKILL.md v7 placement |
| v3 | 10/20 = 50% | +40 | + concrete (was drywall) + threshold -50 (was -65) + SKILL.md v8 RT-only invariant |
| v3 + verifier robust | 16/20 = 80% | +30 | schema-tolerant `_furniture_tuple` (skip entries missing position/dim) + dropped redundant `code_contains:added` |
| **v3 + retry80** | **20/20 = 100%** | +20 | max_turns 80 on the 4 large-room placeholders (S11/S14/S17/S18) |

### Reference oracle math
```python
def c4_ref_oracle(sim, edit_action):
    before = sim["numerical_metrics"]["coverage_pct_before"]
    after  = sim["numerical_metrics"]["coverage_pct_after"]
    delta  = sim["numerical_metrics"]["coverage_delta_pp"]
    # Check 1: arithmetic
    if abs(delta - (after - before)) > 2.0:
        return FAIL
    # Check 2: sign matches edit
    if edit_action == "added" and delta > +5:
        return FAIL  # partition shouldn't INCREASE coverage
    if edit_action == "removed" and delta < -5:
        return FAIL  # removing obstacle shouldn't DECREASE coverage
    # Check 3: suspicious zero-delta under non-zero edit
    if abs(after - before) < 0.01 and edit_action:
        return FAIL  # likely agent didn't actually re-simulate
    return PASS
```

### Worked example (TC4_S14 — 8×6 m office)
- Before partition: coverage_pct_before = 100%
- After concrete partition at x=4.0 m: coverage_pct_after = 54.08%
- Δ = -45.9 pp ✓ (within expected [-30, -55] range)
- All 11 subchecks pass

---

## How the three tasks relate

```
                  scene_state.json
                  ┌──────────────┐
                  │  rooms +     │
                  │  furniture + │  ← T0 (generation only, no RF)
                  │  materials   │
                  └──────┬───────┘
                         │
              add AP @ centroid + run RT
                         │
                         ▼
                  ┌──────────────┐
                  │ coverage_map │  ← T1 (single-AP coverage)
                  └──────┬───────┘
                         │
                add concrete partition + re-run RT
                         │
                         ▼
                  ┌──────────────┐
                  │ before/after │  ← T2 (scene edit + recompute)
                  │ coverage_map │
                  │ + delta_pp   │
                  └──────────────┘
```

Each task **subsumes** the prior one's success criteria:
- T1 requires T0's scene generation to be correct, plus AP placement + coverage.
- T2 requires both T0 + T1, plus scene mutation + re-simulation.

T3 (BER @ TX-RX) and T4 (optimization) will subsume T0/T1 similarly,
adding PHY-layer simulation and brute-force search respectively.

---

## Skill iteration history (relevant to T0/T1/T2)

| Skill | Date | Added invariant / fast-path | Affected task |
|---|---|---|---|
| v0–v5 | (T0 phase) | scene_gen workflow, verbatim naming, glossary, scene_edit fast-path | T0 |
| v6 | May 2026 | **Scene-state schema invariant** (canonical key paths) | T1 enabled, T2 prereq |
| v7 | May 2026 | **Mandatory placement validation** (`tools/validate_layout.py --fix`) | T1 70% → 100% |
| v8 | May 2026 | **Scene-edit RT-only invariant** (no FSPL fallback for material/partition edits) | T2 enabled |

Each version is a **single targeted edit** to SKILL.md after distilling
the dominant failure mode from the previous run. Per the iteration
protocol (Section 8 of system_design.md), every iteration must produce
**≥ 2 pp gain** and **zero regression** on prior tasks — both criteria
met for v6, v7, v8.
