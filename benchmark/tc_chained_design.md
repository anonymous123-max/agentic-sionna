# TC (T-Chained) Benchmark — Design & Progress

This document records the current design of the **chained scene+simulation
benchmark** (TC), which evaluates contributions 2 (network-task agent) and
3 (skill iteration) by chaining a 3D scene generated for contribution 1
with downstream RF tasks. Last updated 2026-05-17.

---

## Design philosophy

Each task pair (scene, capability) requires the agent to:
1. Generate a 3D scene (scene_state.json) — exercising contribution 1
2. Run an RF capability on that scene (simulation_result.json) — exercising contribution 2

Same scene reused across 7 capabilities → cleaner ablation per capability,
demonstrates that contribution 1's output is genuinely simulation-ready.

```
30 scenes × 7 capabilities = 210 tasks
  Train: 20 scenes × 7 caps = 140 train tasks
  Test:  10 scenes × 7 caps =  70 test tasks
```

---

## 30 scenes — `S01`–`S30`

### Easy (15 total: 10 train + 5 test)

| Scene | Description | Dims | Furniture |
|---|---|---|---|
| S01 | home office | 5×4 | desk, office chair, bookshelf |
| S02 | living room | 6×5 | sofa, coffee table, tv stand, armchair |
| S03 | bedroom | 4×4 | double bed, nightstand, wardrobe, dresser |
| S04 | dining room | 6×4 | round table, chair, buffet |
| S05 | conference room | 5×4 | meeting table, chair |
| S06 | home library | 6×5 | bookshelf, reading chair, side table, floor lamp |
| S07 | kitchen | 5×4 | counter, refrigerator, oven, dining table |
| S08 | hobby room | 5×4 | craft table, cabinet, stool, storage shelf |
| S09 | music room | 5×4 | upright piano, piano bench, music stand, armchair |
| S10 | multipurpose room | 7×5 | foldable table, chair, storage cabinet |
| S21 | small office (test) | 4×3 | desk, office chair, filing cabinet |
| S22 | reading nook (test) | 5×4 | armchair, side table |
| S23 | home study (test) | 5×4 | desk, office chair, bookshelf |
| S24 | small bedroom (test) | 4×3 | single bed, nightstand, dresser |
| S25 | art studio (test) | 5×4 | easel, drafting table, storage cabinet |

### Hard (15 total: 10 train + 5 test)

| Scene | Description | Dims | Layout / Materials |
|---|---|---|---|
| S11 | L-shaped office + study | 8×4 | drywall, long arm 6×4 + short arm 4×3 |
| S12 | L-shaped living + dining | 8×4 | drywall, long arm 7×4 + short arm 4×3 |
| S13 | Partitioned 2-tenant office | 10×6 | drywall + 1 partition at x=5 |
| S14 | Studio with kitchenette partition | 8×6 | drywall + 1.2 m half-height partition |
| S15 | 3-bedroom apartment | 12×9 | drywall, 3 bedrooms + corridor |
| S16 | 2-bedroom apartment | 10×7 | drywall, 2 bedrooms + living + kitchen + bath |
| S17 | Two-room office with corridor | 10×5 | drywall, 2 offices 4×5 + 2×5 corridor |
| S18 | Office suite with cubicles | 14×9 | drywall + 4 fabric cubicles + meeting room partition |
| S19 | Hostel dorm cluster | 14×8 | drywall, 4 dorms + corridor + bath |
| S20 | Studio with separate bathroom | 8×5 | drywall, 6×5 studio + 2×5 bath |
| S26 | Open-plan dining + kitchen (test) | 8×5 | drywall, open-plan |
| S27 | Split office: 2 zones (test) | 10×6 | drywall + 1 partition at x=5 |
| S28 | Two-bedroom apartment (test) | 10×7 | drywall, 2 bedrooms + living |
| S29 | Large open studio (test) | 12×8 | drywall, single open zone |
| S30 | Home office + meeting room (test) | 8×5 | drywall + 1 partition (5×5 office + 3×5 meeting) |

**Scene pre-flight verification:** all 30 scenes pass under `with_skill v5`
with 100% pass rate on the full verifier (geometry + Sionna-loadable + furniture
grep).

---

## 7 capabilities — `C1`–`C7`

Each capability is applied to all 30 scenes. Per-task verifier varies by
capability.

| ID | Capability | Description | Verifier oracle layer |
|---|---|---|---|
| C1 | `single_ap_coverage` | Place 1 AP, compute coverage map, report `coverage_pct` | RT-level |
| C2 | `multi_ap_optimization` | Place 2-3 APs, maximize min-RSS, report positions + RSS | RT + SYS |
| C3 | `material_frequency` | Compare coverage at 2 frequencies, report `coverage_diff_pp` | RT-level |
| C4 | `scene_edit_recompute` | Apply 1 edit to scene, recompute coverage delta | RT + Geometry |
| C5 | `rt_to_phy` | RT → CIR → QPSK BER at one SNR | RT + PHY |
| C6 | `irc_coverage_joint` | Verify IRC §R303 8% window + compute coverage | RT + structural |
| C7 | `system_level_multicell` | 2-cell PF scheduling, report `fairness_index` | SYS-level |

---

## Verifier — 5 core checks + capability-specific oracles

Following the 3-level oracle framework (RT / PHY / SYS).

### Core (every task)

| Check | Description |
|---|---|
| `must_create_scene_state` | scene_state.json exists, not harness placeholder |
| `must_create_simulation_result` | simulation_result.json exists, not placeholder |
| `collision_free_check` | Real AABB overlap analysis on furniture |
| `in_bounds_check` | Furniture AABBs within room polygon |
| `sionna_loadable_check` | Convert scene to Mitsuba 3.0 XML and load in Sionna RT |

### Oracle-level checks (added per capability)

| Oracle | Implementation | Layer | Checks |
|---|---|---|---|
| `rt_oracle_check` | `_check_rt_oracle` | RT | (a) `max_rss ≤ tx_power + 5` (energy conservation); (b) coverage(low_freq) ≥ coverage(high_freq) − 5 pp (FSPL trend); (c) coverage(drywall) ≥ coverage(concrete) − 3 pp (material trend); (d) RSS values in [-120, +30] dBm |
| `phy_oracle_check` | `_check_phy_oracle` | PHY | (a) `ber ∈ [0, 1]`; (b) BER monotone vs SNR if both arrays present (≤1 violation tolerated); (c) `coding_gain_db ≥ 0`; (d) `nmse_db ∈ [-30, +10]` |
| `sys_oracle_check` | `_check_sys_oracle` | SYS | (a) `fairness_index ∈ [0, 1]`; (b) `throughput_bps_hz ∈ [0, 30]`; (c) `sinr_mean ∈ [-20, +60] dB`; (d) per_user array length consistent with `num_users` |
| `geometry_oracle_check` | `_check_geometry_oracle` | Cross-layer | Multi-room scene + per_room_coverage array: variance ≥ 1 pp (walls must matter) |

### Capability → Oracle assignment

| Capability | Core | Capability extras | Oracles |
|---|---|---|---|
| C1 | ✓ × 5 | `coverage_pct ∈ [50, 100]` + furniture grep | `rt_oracle` |
| C2 | ✓ × 5 | `min_rss ≥ -85` + AP-count grep | `rt_oracle` + `geometry_oracle` |
| C3 | ✓ × 5 | `coverage_diff_pp` + freq grep | `rt_oracle` |
| C4 | ✓ × 5 | `coverage_delta_pp` + before/after grep | `rt_oracle` + `geometry_oracle` |
| C5 | ✓ × 5 | `ber ∈ [0, 1]` + cir/modulation grep | `phy_oracle` |
| C6 | ✓ × 5 | `coverage_pct` + IRC grep | `rt_oracle` |
| C7 | ✓ × 5 | `fairness ≥ 0.6` + scheduler grep | `sys_oracle` |

---

## Experiment plan

Per-capability batch (run one at a time to control quota):

```
Batch | Capability             | Tasks   | Trials (×3 cond) | Wall time est.
─────────────────────────────────────────────────────────────────────────
1     | C1 single_ap_coverage  | 20 train | 60                | ~30 min
2     | C2 multi_ap_optimization | 20      | 60                | ~30 min
3     | C3 material_frequency  | 20      | 60                | ~30 min
4     | C4 scene_edit_recompute| 20      | 60                | ~30 min
5     | C5 rt_to_phy           | 20      | 60                | ~30 min
6     | C6 irc_coverage_joint  | 20      | 60                | ~30 min
7     | C7 system_level_multicell | 20    | 60                | ~30 min
Total train                    | 140     | 420               | ~3.5 h
Test (single batch later)      | 70      | 210               | ~1.5 h
```

Each batch runs three conditions:
- `with_skill` — uses curated `rf-simulator` SKILL.md v5
- `no_skill` — raw Sonnet 4.6, no SKILL
- `self_gen` — Sonnet-authored SKILL.md (generated for T0; reused here)

---

## Current progress

| Stage | Status |
|---|---|
| 30 scenes designed | ✓ |
| Scene pre-flight (30/30 pass with new Sionna check) | ✓ |
| All 4 oracle checks implemented in verifier.py | ✓ |
| C1-C7 train batches | pending |
| Skill iteration v5 → v6+ on TC-specific failures | pending |
| Test split (10 scenes × 7 cap × 3 cond) | pending |
| Paper integration | pending |

---

## Files of record

| Path | Purpose |
|---|---|
| `benchmark/tasks/_sources/tc_chained.json` | 210-task corpus (30 scenes × 7 caps) |
| `benchmark/tasks/_sources/tc_chained_gen.py` | Generator (scenes + capability builders) |
| `benchmark/tasks/_sources/tc_scene_check.json` | 30 scene-only pre-flight tasks |
| `benchmark/tasks/_sources/tc_scene_check.py` | Scene-only generator |
| `benchmark/verifier.py` | Verifier with 5 core + 4 oracle checks |
| `benchmark/results/tc_scene_check/` | 30/30 pre-flight pass results |
| `benchmark/results/tc20_c1_train/` (archived) | Prior 12-scene C1 attempt (now superseded by 20-scene plan) |

---

## How this maps to the paper's 3 contributions

| Contribution | Tested by |
|---|---|
| (1) 3D scene reconstruction | All TC tasks require scene_state.json — geometry + Sionna-loadable check verifies contribution 1 |
| (2) Network-task agent | The simulation portion of each TC task (coverage, BER, fairness, etc.) — oracle checks verify physical correctness |
| (3) Skill iteration | v5 (T0-iterated SKILL) tested on TC; if pass rate < target, iterate v5 → v6 with TC-specific edits |

This makes TC the **integrated benchmark** that simultaneously validates all
three contributions on the same task pipeline.
