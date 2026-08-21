# C1–C7 Per-Capability Verifier Reference

Precise specification of how each TC capability is graded. Every TC task is
a composite verifier — pass requires **all subchecks to pass**. Each section
below lists the exact subchecks, the pass criterion, the underlying check
function, and a concrete pass/fail example.

Common ground first:

- **Tier**: all TC tasks have `tier: TC_chained`
- **Required artifacts**: every TC task requires both `scene_state.json` AND `simulation_result.json` to exist and be non-placeholder
- **Trial pass criterion**: `pass_strict = verification.passed AND (exec_success OR score == 1.0)`. The verifier itself flags a check as failed; trial pass requires ALL checks pass (`verification.passed = (score == 1.0)`)
- **Lookup**: subcheck dispatch lives in `benchmark/verifier.py:check_code_contains` (magic metric names) and `benchmark/verifier.py:run_checks` (verifier type dispatcher)

---

## Three-Layer Verifier Structure

Every TC subcheck answers one of three orthogonal questions about the
agent's output. Naming this structure explicitly clarifies what each
subcheck contributes, prevents redundancy, and supports the paper's claim
that the verifier covers *scene → simulation → task* end-to-end.

| Layer | Question | Example subchecks |
|---|---|---|
| **A. Scene Validity** | Can this 3D scene be a wireless-simulation input? | `must_create_scene_state`, `collision_free_check`, `in_bounds_check`, `sionna_loadable_check` |
| **B. Network Plausibility** | Are the reported communication metrics physically credible? | `rt_oracle_check`, `phy_oracle_check`, `sys_oracle_check`, `geometry_oracle_check` |
| **C. Task Completion + Reference Correctness** | Did the agent actually complete the capability? Within tolerance of ground truth? | `coverage_pct` range, `fairness_index` threshold, `c1_ref_oracle_check` (analytical FSPL match), capability-specific tokens |

Every C1–C7 composite verifier mixes subchecks from all three layers:
- 5 core checks ≈ Layer A (4 of 5) + plausibility (1)
- Each capability's oracle subcheck = Layer B
- Each capability's metric/range subchecks + reference-oracle = Layer C

When reading the per-capability tables below, the **Type / Function** column
tells you the layer:
- `_check_scene_*` / `file_exists` → Layer A
- `_check_*_oracle` (rt/phy/sys/geometry) → Layer B
- `metric_range` / `metric_threshold` / `c1_ref_oracle_check` / generic token grep → Layer C

---

## Five core checks (every C1–C7 task)

These run on every TC task regardless of capability:

| # | Subcheck `metric` / `key` | Type | Function | Pass criterion |
|---|---|---|---|---|
| 1 | `must_create_scene_state` | file_exists | `_check_file_exists` | `output_dir/scene_state.json` exists; JSON parses; `status != "placeholder_pre_shipped_by_harness"`; if status is placeholder, `numerical_metrics` must have at least one non-null value |
| 2 | `must_create_simulation_result` | file_exists | `_check_file_exists` | Same as #1 for `simulation_result.json` |
| 3 | `collision_free_check` | code_contains (magic) | `_check_scene_collision_free` | Parse `scene_state.json`, treat each furniture as axis-aligned bounding box centered at `position[0:2]` with width/depth `dimensions[0:2]`. Run pairwise AABB-overlap. Pass iff `overlaps == 0` |
| 4 | `in_bounds_check` | code_contains (magic) | `_check_scene_in_bounds` | For each room, every furniture AABB must satisfy `0 ≤ x±w/2 ≤ room_width` AND `0 ≤ y±d/2 ≤ room_depth`. Supports both `room.dimensions[]` and `room.bounds{width,depth}` schemas |
| 5 | `sionna_loadable_check` | code_contains (magic) | `_check_sionna_loadable` | Build minimal Mitsuba 3.0 XML from `scene_state.json` (floor cubes + ITU material BSDFs). Call `sionna.rt.load_scene(xml)`. Pass iff loader does not raise. Lazy-imports sionna; if not installed, returns pass with "skipped" |

Plus a **plausibility band** (`check_plausibility`) runs unconditionally on
all tasks (BER ∈ [0,1], RSS ≤ TX power, NMSE in band) — short-circuits the
overall verdict if any physical impossibility is detected.

---

## C1 — single_ap_coverage

**Task**: generate a scene, place one AP at the centroid at 2.5 m height,
compute coverage at a target frequency/threshold, report `coverage_pct`.

**Total subchecks**: 5 core + 4 capability = **9**

| # | Subcheck | Type | Layer | Pass criterion |
|---|---|---|---|---|
| 1-2 | `must_create_scene_state`, `must_create_simulation_result` | file_exists | A | both artifacts present + non-placeholder |
| 3 | `collision_free_check` | code_contains (magic) | A | pairwise AABB overlap = 0 |
| 4 | `in_bounds_check` | code_contains (magic) | A | all furniture within room bounds |
| 5 | `sionna_loadable_check` | code_contains (magic) | A | Mitsuba XML built from scene loads via `sionna.rt.load_scene` |
| 6 | `coverage_pct` range | metric_range | C | `coverage_pct ∈ [50, 100]` easy / `[30, 100]` hard |
| 7 | `<furniture_token>` | code_contains | C | per-scene prompt nouns appear in agent text |
| 8 | `rt_oracle_check` | code_contains (magic) | B | energy conservation + RSS physical range + FSPL trend + material trend |
| 9 | **`c1_ref_oracle_check`** *(new, May 2026)* | code_contains (magic) | C | **`_check_c1_reference_oracle` — build analytical FSPL grid (0.25 m, rx height 1.5 m) from `scene.bounds` + `access_points[0].position` + `frequency_hz` + `power_dbm`. Compute reference `coverage_pct` vs threshold. Pass iff `|agent − reference| ≤ 5 pp` (easy) / `−15 pp ≤ Δ ≤ +5 pp` (hard, asymmetric: agent can be lower due to walls/multipath but never exceed FSPL ceiling)** |

Subcheck #9 is the **reference oracle** — it answers "did the agent
compute the *correct* number?" rather than "is it in a plausible band?".
This catches reward-hacking attempts where the agent fabricates a
plausible-but-fictitious coverage value without running a simulation.

**Example pass** (TC1_S03 with_skill, 4×4 bedroom):

```
✓ artifact:scene_state.json         status="completed"
✓ artifact:simulation_result.json   coverage_pct=82.5
✓ collision_free                    boxes=4 overlaps=0
✓ in_bounds                         total=4 out_of_bounds=0
✓ sionna_loadable                   Sionna RT loaded
✓ range:coverage_pct                82.5 ∈ [50, 100]
✓ code_contains:double_bed_..._dresser  all tokens present
✓ rt_oracle_check                   max_rss=-22 ≤ 25; range OK
✓ c1_ref_oracle                     agent=82.5 ref=83.1 Δ=-0.6pp ∈ ±5pp (easy)
                                    ──────────────
                                    9/9 → pass_strict = ✓
```

**Failure mode separation** (C1 train v6, 20 ws trials):

| Mode | Count | Layer | Interpretation |
|---|---|---|---|
| Numerical correctness (ref_oracle) | 20/20 pass | C | agent's coverage_pct match analytical FSPL to ±5 pp |
| Collision-free placement | 18/20 pass | A | 2 trials placed overlapping furniture |
| In-bounds placement | 16/20 pass | A | 4 trials placed furniture extending past room walls |

This separation is itself a result: **the C1 skill knows the physics
(numerical layer perfect) but the placement constraint (geometric layer)
is the remaining gap.** This points the next iteration at a placement
validator, not at a coverage formula.

---

## C2 — multi_ap_optimization

**Task**: place 2 (easy) or 3 (hard) APs, maximize minimum-RSS across the
floor, report `ap_positions[]`, `min_rss_dbm`, `coverage_pct`.

**Total subchecks**: 5 core + 4 capability = **9**

| # | Subcheck | Type | Pass criterion |
|---|---|---|---|
| 1-5 | (5 core checks) | — | see above |
| 6 | `min_rss_dbm` | metric_threshold | Read `numerical_metrics.min_rss_dbm`. Pass iff `min_rss_dbm ≥ -85 dBm` |
| 7 | `two_aps` or `three_aps` | code_contains | Tokens `[two, aps]` or `[three, aps]` must appear |
| 8 | `rt_oracle_check` | code_contains (magic) | See C1 #8 |
| 9 | `geometry_oracle_check` | code_contains (magic) | `_check_geometry_oracle` — for multi-room scenes with `per_room_coverage_pct`, the spread `max - min ≥ 1 pp` (walls must matter; agent didn't model coverage as if walls were transparent) |

**Capability-specific failure modes**:
- `min_rss_dbm < -85` → coverage is too sparse; suggests AP placement is bad
- Geometry oracle fails → agent reported identical coverage across rooms (wall effect ignored)

---

## C3 — material_frequency

**Task**: compute coverage at TWO frequencies (e.g., 2.4 vs 5 GHz, or 5 vs
28 GHz), report `coverage_pct_<f1>_ghz`, `coverage_pct_<f2>_ghz`, and
`coverage_diff_pp = coverage(low) − coverage(high)`.

**Total subchecks**: 5 core + 4 capability = **9**

| # | Subcheck | Type | Layer | Pass criterion |
|---|---|---|---|---|
| 1-5 | (5 core checks) | — | A + plausibility | see above |
| 6 | `coverage_diff_pp` | metric_range | C | Pass iff `∈ [-50, 100]` (allows some inversion; mostly expects positive) |
| 7 | `<f1>_ghz` (e.g., `24_ghz`) | code_contains | C | Token list `["ghz"]` after filter; lower freq mentioned in code/JSON |
| 8 | `<f2>_ghz` (e.g., `60_ghz`) | code_contains | C | Same for higher frequency |
| 9 | `rt_oracle_check` | code_contains (magic) | B | See C1 #8. The FSPL trend will catch `coverage(low) < coverage(high) − 5pp` (FSPL violation) |

**Capability-specific failure modes**:
- `rt_oracle_check` fails on FSPL trend → agent reported high-freq coverage > low-freq (violates free-space path loss principle)

**Planned strengthening (priority 1)**: replace grep-style `<f>_ghz` tokens
with a `c3_ref_oracle_check` that requires the agent emit canonical fields
in `simulation_result.json`:

```json
"simulation_config": {"frequencies_ghz": [2.4, 5.0]},
"numerical_metrics": {
  "coverage_pct_low_freq":  78.2,
  "coverage_pct_high_freq": 55.1,
  "coverage_diff_pp":       23.1
}
```

The verifier then re-computes (a) two analytical FSPL coverage values at the
declared frequencies on the same scene, (b) the analytical diff. Pass iff
`|agent − reference| ≤ 5pp` per frequency AND `coverage_diff_pp ≈ low − high`
to within 2pp arithmetic tolerance. This makes C3 honest — agents can no
longer get credit just by saying "ghz" twice.

---

## C4 — scene_edit_recompute

**Task**: apply one edit to the scene (remove furniture, add partition,
change material, etc.), recompute coverage at the same AP, report
`coverage_pct_before`, `coverage_pct_after`, `coverage_delta_pp`.

**Total subchecks**: 5 core + 6 capability = **11**

| # | Subcheck | Type | Layer | Pass criterion |
|---|---|---|---|---|
| 1-5 | (5 core checks) | — | A + plausibility | see above |
| 6 | `coverage_delta_pp` | metric_range | C | `∈ [-100, 100]` (very loose; must be a real number, not null) |
| 7 | `coverage_pct_before` | code_contains | C | Token `[coverage, before]` must appear |
| 8 | `coverage_pct_after` | code_contains | C | Token `[coverage, after]` must appear |
| 9 | `<action>` (added/removed/changed) | code_contains | C | The edit verb (per task) appears in code |
| 10 | `rt_oracle_check` | code_contains (magic) | B | See C1 #8 |
| 11 | `geometry_oracle_check` | code_contains (magic) | B | See C2 #9 |

**Capability-specific failure modes**:
- Missing `coverage_pct_before` / `coverage_pct_after` → agent didn't run two simulations (just reported baseline once)
- Geometry oracle fails on multi-room edits → agent didn't model the partition's effect

**Planned strengthening (priority 3)**: replace grep-style `before`/`after`
tokens with a `c4_ref_oracle_check` that:
1. Requires the agent emit `scene_state_before.json` + `scene_state_after.json`
   (or equivalent embedded blocks) alongside the result.
2. Verifies the edit actually occurred (e.g., for `remove_furniture`: the
   target item is absent in *after*; for `add_partition`: a new wall in
   *after*; for `change_material`: the material field differs).
3. Computes `analytical_delta = analytical_FSPL(after) − analytical_FSPL(before)`
   and checks `|agent_delta_pp − analytical_delta| ≤ 10pp`.
4. Sign check: if edit type is `remove_obstacle`, expect `delta ≥ 0`;
   `add_partition` or `change_to_concrete`, expect `delta ≤ 0`.

This catches the "agent reported a number without actually editing" pattern.

---

## C5 — rt_to_phy

**Task**: from the generated scene, place 1 TX + 1 RX, run RT to compute
the channel impulse response (CIR), then simulate QPSK BER at one SNR.
Report `cir_path_count`, `ber`, optionally `ber_theoretical_awgn`.

**Total subchecks**: 5 core + 4 capability = **9**

| # | Subcheck | Type | Pass criterion |
|---|---|---|---|
| 1-5 | (5 core checks) | — | see above |
| 6 | `ber` | metric_range | Read `numerical_metrics.ber`. Pass iff `ber ∈ [0, 1]` (physical) |
| 7 | `cir` | code_contains | Token `[cir]` must appear (4 chars after lowercase split; check `cir > 2 chars`) |
| 8 | `qpsk` | code_contains | Token `[qpsk]` must appear |
| 9 | `phy_oracle_check` | code_contains (magic) | `_check_phy_oracle` — read `simulation_result.json`, check (a) `ber ∈ [0, 1]`; (b) BER monotone vs SNR if both arrays present (≤1 violation tolerated); (c) `coding_gain_db ≥ 0` if coded; (d) `nmse_db ∈ [-30, +10]` if channel estimator results present |

**Capability-specific failure modes**:
- `ber` is null → agent only generated scene, didn't run BER simulation
- `phy_oracle.ber` outside [0,1] → bug in computation (e.g., negative BER, or > 1)
- `phy_oracle.coding_gain_db < 0` → coding hurts (impossible; bug)
- `phy_oracle.BER not monotone` → SNR sweep wrong direction or stochastic noise too high

---

## C6 — irc_coverage_joint

**Task**: generate a habitable room (bedroom/living/kitchen/office) with a
window meeting IRC §R303 8% floor-area aperture rule. Place AP, compute
coverage. Report `coverage_pct`, `irc_compliant` (boolean),
`total_window_aperture_m2`.

**Total subchecks**: 5 core + 5 capability = **10**

| # | Subcheck | Type | Pass criterion |
|---|---|---|---|
| 1-5 | (5 core checks) | — | see above |
| 6 | `coverage_pct` | metric_range | Pass iff `coverage_pct ∈ [30, 100]` (lower bound looser than C1 since IRC + mmWave often gives moderate coverage) |
| 7 | `irc_compliant` | code_contains | Token `[irc, compliant]` must appear |
| 8 | `perimeter` | code_contains | Token `[perimeter]` must appear (IRC requires windows on perimeter walls) |
| 9 | `aperture` | code_contains | Token `[aperture]` must appear |
| 10 | `rt_oracle_check` | code_contains (magic) | See C1 #8 |

**Capability-specific failure modes**:
- Missing IRC tokens → agent ignored the IRC §R303 rule
- `coverage_pct < 30` → agent placed AP poorly OR the IRC constraint was over-restrictive

---

## C7 — system_level_multicell

**Task**: deploy 2 (easy) or 3 (hard) cells (APs as base stations) at
evenly-spaced positions, run PF (proportional-fair) scheduling for 100 TTI,
report `mean_throughput_bps_hz`, `fairness_index` (Jain's), `per_user_avg_rate`.

**Total subchecks**: 5 core + 4 capability = **9**

| # | Subcheck | Type | Layer | Pass criterion |
|---|---|---|---|---|
| 1-5 | (5 core checks) | — | A + plausibility | see above |
| 6 | `fairness_index` | metric_threshold | C | Pass iff `fairness_index ≥ 0.7` (easy) OR `≥ 0.6` (hard) |
| 7 | `two_cells` or `three_cells` | code_contains | C | Tokens `[two, cells]` or `[three, cells]` must appear |
| 8 | `pf_scheduling` | code_contains | C | Tokens `[scheduling]` must appear |
| 9 | `sys_oracle_check` | code_contains (magic) | B | `_check_sys_oracle` — (a) `fairness_index ∈ [0, 1]`; (b) `mean_throughput_bps_hz ∈ [0, 30]`; (c) `sinr_mean ∈ [-20, +60] dB`; (d) `len(per_user_avg_rate) == num_users` if both reported |

**Capability-specific failure modes**:
- `fairness_index < 0.6` → scheduling is bad (one user starves others)
- `sys_oracle.throughput > 30` → unphysical (above LTE-Advanced cap)
- `sys_oracle.fairness > 1` → math impossible (Jain's bounded by 1)
- `len(per_user_avg_rate) != num_users` → agent reported inconsistent data

**Planned strengthening (priority 2)**: add a `c7_ref_oracle_check` that
mandates `per_user_avg_rate[]` in the result and re-computes Jain's index
from those rates:

```text
J(r1,…,rn) = (Σ r_i)² / (n · Σ r_i²)

mean_throughput_recomputed = mean(per_user_avg_rate)
```

Pass iff:
- `|agent_fairness − J(per_user_rates)| ≤ 0.05`
- `|agent_mean_throughput − mean(rates)| ≤ 0.2 bps/Hz`
- Every user scheduled at least once (`per_user_scheduled_tti[i] ≥ 1`)
- All `per_user_avg_rate[i] ≥ 0`

This is the **highest-value** new oracle because `fairness_index` is the
metric most trivially fakeable — without per-user rates, the agent can
simply emit a plausible-looking 0.85 and pass the threshold check. With
the per-user array re-computation, the only way to pass is to actually
schedule.

---

## Summary table

| Capability | Core (A) | Capability-specific (C) | Oracle (B) | Reference oracle (C) | Total | Primary metric |
|---|---|---|---|---|---|---|
| C1 single_ap_coverage | 5 | 2 (range + furniture grep) | rt | **`c1_ref_oracle` ✓** | **9** | `coverage_pct ∈ [50, 100]` + Δ vs FSPL ≤ 5pp |
| C2 multi_ap_optimization | 5 | 2 (threshold + ap grep) | rt + geometry | — *(planned)* | 9 | `min_rss_dbm ≥ -85` |
| C3 material_frequency | 5 | 3 (range + 2 freq tokens) | rt | **`c3_ref_oracle` ✓** | **10** | `coverage_diff_pp ∈ [-50, 100]` + per-freq FSPL match |
| C4 scene_edit_recompute | 5 | 4 (range + 2 before/after + action) | rt + geometry | **`c4_ref_oracle` ✓** | **12** | `coverage_delta_pp ∈ [-100, 100]` + arith + sign |
| C5 rt_to_phy | 5 | 3 (range + cir + qpsk) | phy | — *(planned)* | 9 | `ber ∈ [0, 1]` |
| C6 irc_coverage_joint | 5 | 4 (range + 3 IRC tokens) | rt | — *(planned)* | 10 | `coverage_pct ∈ [30, 100]` |
| C7 system_level_multicell | 5 | 3 (threshold + 2 grep) | sys | **`c7_ref_oracle` ✓** | **10** | `fairness_index ≥ 0.7/0.6` + Jain re-computed from rates |

---

## How the oracles are implemented

Each oracle is a standalone function in `benchmark/verifier.py` that
reads `simulation_result.json` (and optionally `scene_state.json` for
geometry oracle) and emits a `CheckResult(name, passed, detail)`. None of
them require an LLM judge — they are pure Python predicates on numeric
fields.

| Oracle | File | Key conditions |
|---|---|---|
| `rt_oracle` | `verifier.py:_check_rt_oracle` | energy conservation + RSS physical range + FSPL trend (if multi-freq) + material trend (if both reported) |
| `phy_oracle` | `verifier.py:_check_phy_oracle` | BER ∈ [0,1] + monotone vs SNR (if both arrays) + coding_gain ≥ 0 + nmse_db ∈ [-30, +10] |
| `sys_oracle` | `verifier.py:_check_sys_oracle` | fairness ∈ [0,1] + throughput ∈ [0, 30] bps/Hz + sinr_mean ∈ [-20, +60] dB + per-user array length matches num_users |
| `geometry_oracle` | `verifier.py:_check_geometry_oracle` | multi-room + per_room_coverage spread ≥ 1 pp (walls must affect coverage) |

Each oracle is **tolerant of missing fields** — if the relevant metric
isn't present in the agent's output, the oracle passes the relevant
sub-condition silently and only flags real violations.

---

## How verification.passed and pass_strict combine

```
For each subcheck in task.verifier.subchecks:
    result = dispatch(subcheck, output_dir, sim, exec_success)
    if not result.passed:
        # contributes to failed_checks list
        
verification.score = sum(c.passed for c in checks) / len(checks)
verification.passed = (score == 1.0) AND no_plausibility_failure

pass_strict = verification.passed AND (exec_success OR score == 1.0)
```

The `score == 1.0` relaxation on `pass_strict` accepts trials where the
agent produced fully-verified artifacts but the wrapping script crashed
or hit `max_turns` at the end (the `pass_strict_exec` variant keeps the
old strict semantics for back-compat analysis).

---

## Where each capability's verifier lives in code

| File | Role |
|---|---|
| `benchmark/tasks/_sources/tc_chained_gen.py` | Builds the task subcheck list (the JSON spec) per capability — see `make_c1`...`make_c7` |
| `benchmark/tasks/_sources/tc_chained.json` | The generated 210 tasks (verifier spec embedded in each) |
| `benchmark/verifier.py` | Dispatcher (`run_checks`) + magic metrics (`collision_free_check`, etc.) + oracle functions |
| `benchmark/_verifier_core.py` | Helpers — `load_all_code`, `extract_scalar`, `_check_generic_tokens` |

To audit how a specific TC trial was graded, open the corresponding
`benchmark/results/<label>/<cond>/<task_id>/t1/result.json` — the
`verification.checks[]` array lists every subcheck's `name`, `passed`,
and `detail`.

---

## Reference oracle roadmap

The reference oracle (Layer C above) compares the agent's reported metric
against a **closed-form analytical or pre-computed Sionna RT value**, not
just a plausibility band. This is the strongest layer because plausibility
checks can be passed by fabricated-but-physical-looking numbers, while
reference oracles cannot.

| Capability | Reference type | Tolerance | Status |
|---|---|---|---|
| **C1** single_ap_coverage | Analytical FSPL grid on `scene.bounds` | ±5 pp easy / [-15, +5] pp hard | **Implemented** (`_check_c1_reference_oracle`) |
| **C3** material_frequency | Analytical FSPL at two freqs + diff arithmetic + FSPL-sanity | [-10, +5] pp easy / [-20, +5] pp hard per freq; ±2 pp diff arith | **Implemented** (`_check_c3_reference_oracle`) |
| **C4** scene_edit_recompute | Before/after arithmetic + sign by edit type | ±2 pp delta arith; ±5 pp sign tolerance per action | **Implemented** (`_check_c4_reference_oracle`, takes `edit_action` from subcheck spec) |
| **C7** system_level_multicell | Re-compute Jain's + mean from `per_user_avg_rate[]` | ±0.05 fairness; ±0.2 bps/Hz throughput; no starvation | **Implemented** (`_check_c7_reference_oracle`) |
| C2 multi_ap_optimization | Analytical FSPL on agent-chosen positions | ±2 dB on min_rss | Planned |
| C5 rt_to_phy | Analytical AWGN/Rayleigh BER bounds | `ber_awgn ≤ agent ≤ 1.5 × ber_rayleigh` | Planned |
| C6 irc_coverage_joint | IRC R303 aperture re-check (boolean) + FSPL coverage | exact for IRC + ±5 pp coverage | Planned |

### Why "reference" beats "plausibility"

Plausibility-only verifiers are vulnerable to the agent emitting a
plausible band-conforming number without running the computation:

```
agent: "I checked. fairness_index = 0.85, mean_throughput = 4.2 bps/Hz."
verifier (plausibility only): 0.85 ∈ [0, 1] ✓, 4.2 ∈ [0, 30] ✓  → PASS
verifier (with c7_ref_oracle): per_user_avg_rate missing  → FAIL (Layer C)
```

The reference oracle requires the agent submit the **raw evidence** (per-user
rates, per-frequency coverage maps, before/after deltas) and then
**re-derives** the headline metric. Reward-hacking only works if the agent
fabricates the raw evidence too — which is much harder than fabricating
the summary.

### Implementation lives in `benchmark/verifier.py`

`_check_c1_reference_oracle(task, output_dir)` is the current template.
It (a) defensively parses scene bounds from 6+ schema variants
(`scene.bounds`, `scene.dimensions`, `rooms[].bounds`, `dims_m`,
`width_m,depth_m`, `bounds={x_min,x_max}`), (b) extracts AP position from
either `access_points` or `transmitters` arrays, (c) computes analytical
FSPL coverage on a 0.25 m grid at 1.5 m rx height, (d) compares with
asymmetric tolerance reflecting the asymmetry of physics: agent can be
below FSPL ceiling (walls/multipath subtract) but never above (energy
conservation).

The same defensive-parsing + analytical-recompute + asymmetric-tolerance
pattern applies to C2–C7 references; future implementations should
follow this template.

---

## SKILL.md schema invariant (v6, May 2026)

The agent's freedom to invent `scene_state.json` schemas was the dominant
failure mode in the C1 held-out test (10 ws trials, 6 distinct schemas
emerged; verifier could only parse the train-set variants). SKILL.md v6
adds a `[ACTIVE]` invariant section titled **"Scene-state schema
invariant"** that fixes the canonical key paths:

- `scene.bounds.{width,depth,height}` (NOT `scene.dimensions`)
- `rooms[].bounds.{x,y,width,depth,height}` (NOT `dims_m` / `width_m` / `{x_min,x_max}`)
- `access_points[].position = [x, y, z]` LIST (NOT dict, NOT under `transmitters`)
- `furniture[].position`, `furniture[].dimensions` both LIST not dict
- `frequency_hz` (not `freq_ghz` / `f_ghz`)
- `power_dbm` (not `tx_power_dbm`)

Plus an explicit "Forbidden variants" table the agent sees on every
load. The verifier's defensive-parsing layer still handles legacy
variants for backward compatibility, but the skill now actively discourages
schema improvisation. The contract becomes:

> The agent writes a fixed schema → the verifier reads a fixed schema →
> the reference oracle re-derives the metric → reward hacking requires
> faking the entire scene, not just the summary number.
