# Reference Oracle Design — Per-Capability Ground Truth

This document specifies, for each capability C1–C7, the exact reference
function that produces "ground truth" and the tolerance band against which
the agent's output is compared. The design follows a layered priority:

```
A. Analytical formula      ← preferred (textbook-vouched, deterministic)
B. Sionna RT cached value  ← fallback for scenes A can't handle
C. Analytical bounds       ← for metrics with theoretical limits
```

For every task we use the **tightest applicable** reference. A and C are
preferred over B because their correctness is provable from textbook
physics rather than relying on Sionna being right.

---

## Implementation status (May 2026)

| Capability | Reference type | Status | Code location |
|---|---|---|---|
| C1 single_ap_coverage | A — analytical FSPL grid | **✓ Implemented** | `verifier.py:_check_c1_reference_oracle` (subcheck `c1_ref_oracle_check`) |
| C3 material_frequency | A — FSPL at two freqs + diff arithmetic + FSPL sanity | **✓ Implemented** | `verifier.py:_check_c3_reference_oracle` (subcheck `c3_ref_oracle_check`) |
| C4 scene_edit_recompute | A — before/after arithmetic + sign-by-edit-action | **✓ Implemented** | `verifier.py:_check_c4_reference_oracle` (subcheck `c4_ref_oracle_check`, reads `edit_action` from subcheck spec) |
| C7 system_level_multicell | A — Jain's re-computation from per_user_rates | **✓ Implemented** | `verifier.py:_check_c7_reference_oracle` (subcheck `c7_ref_oracle_check`) |
| C2 multi_ap_optimization | A — FSPL on agent positions | Planned | — |
| C5 rt_to_phy | C — AWGN/Rayleigh bounds | Planned | — |
| C6 irc_coverage_joint | A — R303 aperture + FSPL | Planned | — |

Shared helpers `_parse_scene_geometry` + `_analytical_fspl_coverage` are
factored out in `verifier.py` for reuse across C1/C3 (and future C2/C4-with-
analytical/C6 oracles). The C2/C5/C6 implementations will follow the same
defensive-parsing + analytical-recompute + asymmetric-tolerance pattern.

The C1 oracle additionally absorbed schema-defensive parsing for **6+
scene_state.json variants** observed in test trials (`scene.bounds`,
`scene.dimensions`, `rooms[].dims_m`, `rooms[].dims`, `width_m,depth_m`,
`{x_min,x_max,y_min,y_max}` bounding box). C2–C7 oracles should reuse
this defensive parser. The complementary fix lives in
[SKILL.md](../.claude/skills/rf-simulator/SKILL.md) v6 as the
"Scene-state schema invariant" — it instructs the agent to emit one
canonical schema, with the verifier's defensive parser as a backstop for
legacy/edge-case schemas.

### C1 validation summary (train, 20 scenes × 3 conditions)

| Cond | overall pass | ref_oracle pass | failure mode |
|---|---|---|---|
| with_skill (v5) | 17/20 = 85% | 20/20 numerically correct | 2 collision + 1 in_bounds (geometric) |
| with_skill (v6) | 14/20 = 70% | 20/20 numerically correct | 4 in_bounds + 2 collision (sampling noise vs v5; ref_oracle unchanged) |
| no_skill | 0/20 | 0/20 | no scene_state.json bounds at all |
| self_gen | 0/20 | 0/20 | placeholder scene_state.json |

The Δ = +70 to +85 pp **and** the cleanly-separated failure modes
(numerical correctness perfect, geometric placement still imperfect)
are direct evidence that this verifier separates *task-physics-knowledge*
from *placement-constraint-knowledge*. This is the kind of fine-grained
diagnostic the paper's contribution-3 (skill iteration) needs.

C1 test (10 held-out scenes × 3 conditions, evaluated once with the
schema-invariant locked in train-time): ws 1/10 = 10%. The ref_oracle
passes on the 3 scenes where the agent emitted a parseable schema —
on those, agent_coverage = analytical_coverage to ±5pp. The 6/10
verifier-fail-on-schema trials are an honest measurement of zero-shot
schema-improvisation under a fixed verifier, not a true capability gap.

---

## Verifier integration

A new subcheck type `reference_match_check` is added. It loads a
pre-computed reference value for the task (or computes it on-the-fly via
analytical formulas) and compares the agent's reported metric within a
tolerance band:

```python
# benchmark/verifier.py
def _check_reference_match(task: dict, output_dir: Path,
                          ref_value: float, tolerance: float,
                          mode: str = "absolute") -> CheckResult:
    """Compare agent's metric against pre-computed reference.
    mode='absolute': |agent - ref| <= tolerance
    mode='relative': |agent - ref| / |ref| <= tolerance
    mode='bound':    ref_low <= agent <= ref_high (tolerance ignored; ref is tuple)
    mode='sign':     sign(agent) == sign(ref) (numerical match optional)
    """
```

Reference values live in either:
- **Inline in task JSON** for analytical references that take seconds to compute
- **`benchmark/oracle/cache/`** for Sionna RT pre-computed references

---

## C1 — single_ap_coverage

**Reference type**: A (analytical FSPL)
**Tolerance**: ±5 pp for empty/lightly-furnished rooms

### Reference function

```python
def reference_c1_coverage_pct(scene_bounds, ap_pos, freq_hz,
                              threshold_dbm, tx_power_dbm=20.0,
                              rx_height=1.5, grid=0.1):
    """Compute analytical FSPL coverage % for single-AP scenario."""
    import numpy as np
    W, D = scene_bounds["width"], scene_bounds["depth"]
    ap_x, ap_y, ap_z = ap_pos
    
    xs = np.arange(0, W + grid, grid)
    ys = np.arange(0, D + grid, grid)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    
    d = np.sqrt((X - ap_x)**2 + (Y - ap_y)**2 + (ap_z - rx_height)**2)
    fspl_db = 20*np.log10(d) + 20*np.log10(freq_hz) - 147.55
    rss = tx_power_dbm - fspl_db
    return float(np.mean(rss > threshold_dbm) * 100)
```

### Worked example: TC1_S01 (5×4 m home office)

```
scene_bounds = {"width": 5, "depth": 4}
ap_pos       = (2.5, 2.0, 2.5)        # centroid + 2.5 m height
freq_hz      = 5e9                     # 5 GHz
threshold    = -75 dBm
tx_power     = 20 dBm

→ reference_c1_coverage_pct = 100.0%
   (FSPL at worst corner (0,0,1.5): d=3.35m, loss=56.9 dB → RSS=-37 dBm > -75)

Tolerance: ±5 pp → agent's coverage_pct ∈ [95, 100]

Agent reports 98.5% → |98.5 - 100| = 1.5 ≤ 5 → PASS
Agent reports 65%   → |65 - 100| = 35 > 5  → FAIL
```

### When this reference DOESN'T apply

For hard scenes (L-shape, partitioned, multi-room), FSPL ignores walls.
Fall back to:
- B (Sionna RT cache) — pre-run Sionna once per hard scene, store value
- OR allow larger tolerance ±15 pp for these scenes

---

## C2 — multi_ap_optimization

**Reference type**: A (analytical FSPL applied to agent's chosen AP positions)
**Tolerance**: ±2 dB on min_rss for empty rooms

### Reference function

The agent picks where to place the N APs. The reference is "given those
positions, what should the min_rss across the floor be"?

```python
def reference_c2_min_rss_dbm(scene_bounds, ap_positions, freq_hz,
                             tx_power_dbm=20.0, rx_height=1.5, grid=0.1):
    """Compute analytical FSPL min_rss across the floor for placed APs."""
    import numpy as np
    W, D = scene_bounds["width"], scene_bounds["depth"]
    
    xs = np.arange(0, W + grid, grid)
    ys = np.arange(0, D + grid, grid)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    
    rss_max = np.full_like(X, -200.0)
    for ap in ap_positions:
        ax, ay, az = ap
        d = np.sqrt((X - ax)**2 + (Y - ay)**2 + (az - rx_height)**2)
        fspl_db = 20*np.log10(d) + 20*np.log10(freq_hz) - 147.55
        rss = tx_power_dbm - fspl_db
        rss_max = np.maximum(rss_max, rss)
    
    return float(rss_max.min())
```

### Worked example: TC2_S15 (12×9 m apartment with 3 bedrooms + corridor)

```
agent_ap_positions = [(3, 4.5, 2.5), (9, 4.5, 2.5)]   # 2 APs from agent's output

→ reference_min_rss = -72.3 dBm  (corner of bedroom farthest from both APs)

Tolerance: ±2 dB → agent's min_rss_dbm ∈ [-74.3, -70.3]

Agent reports -71.5 → in band → PASS
Agent reports -85   → out of band by 11 dB → FAIL (agent miscomputed for given placement)
```

### Important note

For hard scenes with walls, the analytical FSPL OVER-estimates min_rss
(ignores wall loss). Agent's RT-based min_rss should be ≤ analytical
(lower because walls attenuate). So tolerance for hard scenes should be
asymmetric: agent's value can be ≤ analytical (real walls hurt), but
not > analytical (no free energy).

---

## C3 — material_frequency

**Reference type**: A (analytical FSPL difference)
**Tolerance**: ±5 pp on `coverage_diff_pp` for empty rooms

### Reference function

```python
def reference_c3_coverage_diff_pp(scene_bounds, ap_pos, freq_low_hz,
                                  freq_high_hz, threshold_dbm,
                                  tx_power_dbm=20.0):
    """Difference = coverage(low) - coverage(high)."""
    cov_low = reference_c1_coverage_pct(scene_bounds, ap_pos, freq_low_hz,
                                        threshold_dbm, tx_power_dbm)
    cov_high = reference_c1_coverage_pct(scene_bounds, ap_pos, freq_high_hz,
                                          threshold_dbm, tx_power_dbm)
    return cov_low - cov_high
```

### Worked example: TC3_S01 (5×4 office, 2.4 GHz vs 5 GHz)

```
freq_low_hz  = 2.4e9
freq_high_hz = 5e9

cov_low (2.4 GHz)  = 100.0% (room is small, both freqs cover fully)
cov_high (5 GHz)   = 100.0%
→ reference_coverage_diff_pp = 0.0 pp

Tolerance: ±5 pp → agent's diff ∈ [-5, +5]

Agent reports diff = 2 pp → PASS (small but agent saw some material effect)
Agent reports diff = -10 pp → FAIL (high freq covered MORE than low? unphysical)
```

### Larger room example: TC3_S15 (12×9 apartment, 5 GHz vs 28 GHz)

```
cov_low (5 GHz)    = ~95%
cov_high (28 GHz)  = ~60%   (mmWave is range-limited)
→ reference_coverage_diff_pp = +35 pp

Tolerance: ±10 pp → agent's diff ∈ [25, 45]

Agent reports diff = +5 pp  → FAIL (underestimated freq effect)
Agent reports diff = +30 pp → PASS
```

---

## C4 — scene_edit_recompute

**Reference type**: B (Sionna RT cache) OR sign-only check
**Tolerance**: ±10 pp on `coverage_delta_pp` for B; sign-only for analytical

Scene edits change geometry/materials in ways that depend on walls/materials.
Pure FSPL is too crude. Two options:

### Option 1: Sign-only check (cheap)

For each edit type, the EXPECTED sign of `coverage_delta_pp` is known:

| Edit type | Expected sign | Magnitude band |
|---|---|---|
| Remove furniture (TV stand, etc.) | `delta_pp ≥ 0` (less obstruction → more coverage) | [0, +5] pp |
| Add partition wall | `delta_pp < 0` (signal blocked) | [-30, -3] pp |
| Change wall material concrete→drywall | `delta_pp > 0` (drywall lower loss) | [+3, +15] pp |
| Change material drywall→glass | `delta_pp > 0` (glass transparent at mid-band) | [+2, +10] pp |
| Change material wood→metal | `delta_pp < 0` (metal blocks) | [-25, -3] pp |

### Option 2: Sionna RT cache (gold standard)

Pre-compute reference for each (scene, edit_type) tuple by running Sionna RT
twice (before + after the edit) with a fixed configuration. Cache the values
in `benchmark/oracle/cache/c4/<task_id>.json`.

```python
def reference_c4_coverage_delta_pp(task_id):
    cache_path = f"benchmark/oracle/cache/c4/{task_id}.json"
    if Path(cache_path).exists():
        return json.load(open(cache_path))["coverage_delta_pp"]
    # Compute on first call, save
    ref = compute_sionna_rt_delta(task)
    json.dump({"coverage_delta_pp": ref}, open(cache_path, "w"))
    return ref
```

Cost: ~30 seconds per task to pre-compute the cache. One-time.

### Recommended

Start with Option 1 (sign-only) — almost zero setup cost. Add Option 2
for tasks where sign-only is too loose.

---

## C5 — rt_to_phy BER

**Reference type**: C (analytical AWGN/Rayleigh bounds)
**Tolerance**: agent's BER must be **inside the band** [ber_awgn, ber_rayleigh × 1.5]

### Reference function

```python
def reference_c5_ber_bounds(modulation, snr_db, code_rate=1.0):
    """Compute (lower, upper) BER bounds for the given modulation and SNR.
    Lower = AWGN (no fading), Upper = Rayleigh fading.
    Real indoor multipath should fall in between."""
    import numpy as np
    from scipy.special import erfc
    snr_lin = 10**(snr_db / 10) * code_rate
    
    if modulation.lower() == "bpsk" or modulation.lower() == "qpsk":
        ber_awgn = 0.5 * erfc(np.sqrt(snr_lin))
        ber_rayleigh = 0.5 * (1 - np.sqrt(snr_lin / (snr_lin + 1)))
    elif modulation.lower() == "16qam":
        # Approximate AWGN: (3/8) erfc(sqrt(0.4 * SNR))
        ber_awgn = (3/8) * erfc(np.sqrt(0.4 * snr_lin))
        ber_rayleigh = (3/4) * (1 - np.sqrt(0.4*snr_lin / (1 + 0.4*snr_lin)))
    elif modulation.lower() == "64qam":
        ber_awgn = (7/24) * erfc(np.sqrt(snr_lin / 7))
        ber_rayleigh = (7/12) * (1 - np.sqrt(snr_lin/7 / (1 + snr_lin/7)))
    
    return float(ber_awgn), float(ber_rayleigh)
```

### Worked example: TC5_S01 (QPSK, SNR=10 dB)

```
ber_awgn     = 0.5 * erfc(sqrt(10)) = 3.87e-6   (theoretical AWGN)
ber_rayleigh = 0.5 * (1 - sqrt(10/11)) = 0.024  (worst-case Rayleigh)

Upper bound (with 1.5× safety) = 0.024 × 1.5 = 0.036

Valid band: agent's ber ∈ [3.87e-6, 0.036]

Agent reports ber = 0.005 → in band → PASS  (typical indoor channel)
Agent reports ber = 0.5   → above upper → FAIL (random guess)
Agent reports ber = 1e-10 → below lower → FAIL (better than AWGN, impossible)
```

### Coding gain check (if LDPC coded reported)

```python
def reference_c5_coding_gain_bounds(snr_db, code_rate):
    """LDPC coding gain at given SNR — usually 2-5 dB for rate-0.5 LDPC."""
    if code_rate >= 0.5:
        return (1.0, 6.0)  # at least 1 dB gain, at most 6 dB
    elif code_rate >= 0.33:
        return (2.0, 8.0)
    return (0.5, 10.0)
```

---

## C6 — irc_coverage_joint

**Reference type**: A (IRC geometric verification) + A (FSPL coverage)
**Tolerance**: IRC compliance is boolean (no tolerance); coverage ±5 pp

### IRC compliance check

```python
def reference_c6_irc_compliant(scene):
    """Verify IRC §R303 8% window aperture requirement per habitable room."""
    habitable_types = {"bedroom", "living", "kitchen", "dining", "office", "study"}
    rooms = scene.get("rooms", [])
    for room in rooms:
        rtype = room.get("type", "").lower()
        if rtype not in habitable_types:
            continue  # IRC exempt
        # Compute floor area
        dims = room.get("dimensions") or [
            room.get("bounds", {}).get("width", 0),
            room.get("bounds", {}).get("depth", 0)
        ]
        floor_area = float(dims[0]) * float(dims[1])
        target = 0.08 * floor_area
        
        # Sum window apertures on perimeter walls
        aperture = 0.0
        for w in room.get("windows", []):
            if w.get("wall") in {"north", "south", "east", "west"}:
                aperture += w.get("width", 0) * w.get("height", 0)
        
        if aperture < target:
            return False, f"{rtype}: aperture {aperture:.2f}m² < target {target:.2f}m²"
    return True, "all habitable rooms compliant"
```

### Coverage reference (same as C1)

```python
def reference_c6_coverage_pct(scene_bounds, ap_pos, freq_hz, threshold_dbm):
    return reference_c1_coverage_pct(scene_bounds, ap_pos, freq_hz, threshold_dbm)
```

### Worked example: TC6_S03 (4×4 m bedroom, 28 GHz, IRC window)

```
Required: 0.08 × 16 = 1.28 m² aperture on perimeter wall

Agent's scene has window 1.2×1.2 = 1.44 m² on perimeter → IRC ✓
Agent reports coverage_pct = 78% at 28 GHz, threshold -70 dBm
Reference coverage (FSPL 28 GHz) = 82%
|78 - 82| = 4 ≤ 5 → coverage ✓

Both pass → PASS

If agent had window 0.8×1.0 = 0.8 m² on interior wall:
  IRC check: 0.8 m² < 1.28 OR not on perimeter → FAIL
```

---

## C7 — system_level_multicell

**Reference type**: A (analytical PF fairness)
**Tolerance**: ±0.1 on fairness_index

### Reference function — Jain's fairness for PF scheduling

```python
def reference_c7_fairness_index(per_user_rates):
    """Jain's fairness index = (sum_i r_i)^2 / (N * sum_i r_i^2)."""
    import numpy as np
    rates = np.array(per_user_rates, dtype=float)
    N = len(rates)
    if N == 0 or rates.sum() == 0:
        return 1.0
    return float(rates.sum()**2 / (N * (rates**2).sum()))


def reference_c7_pf_expected_fairness(user_channel_qualities, n_cells=2):
    """For PF scheduling, expected fairness given user channel distribution."""
    # PF gives each user 1/N share of time (over long horizon)
    # Resulting rates = log2(1 + SNR_i) for each user, modulated by sharing
    import numpy as np
    snr_lin = 10**(np.array(user_channel_qualities) / 10)
    rates_per_user = np.log2(1 + snr_lin)  # bps/Hz before time-sharing
    pf_rates = rates_per_user / len(user_channel_qualities)  # equal time share
    return reference_c7_fairness_index(pf_rates)
```

### Worked example: TC7_S02 (12×8 office, 2 cells, 8 users, PF scheduling)

```
agent_per_user_rates = [2.1, 2.3, 2.0, 2.5, 1.9, 2.4, 2.2, 2.1]  # bps/Hz

→ reference_fairness = (2.1+...+2.1)^2 / (8 * (2.1^2+...+2.1^2)) = 0.991

agent reports fairness_index = 0.98 → |0.98 - 0.991| = 0.011 ≤ 0.1 → PASS

Agent reports fairness = 0.5 → 0.491 > 0.1 → FAIL (scheduling broken)
```

### Edge case: round-robin

For RR scheduling, fairness should be ≥ 0.95 (near-perfect). PF can be 0.7-1.0.

---

## Summary table

| Capability | Primary reference | Tolerance | Cost to compute | Confidence |
|---|---|---|---|---|
| C1 single AP | **A** FSPL grid | ±5 pp easy, ±15 pp hard | ~10 ms | High (textbook) |
| C2 multi-AP | **A** FSPL on agent positions | ±2 dB on min_rss | ~30 ms | High |
| C3 freq compare | **A** FSPL diff | ±5 pp empty, ±10 pp hard | ~20 ms | High |
| C4 scene_edit | **Sign-only** (Opt 1) → B (Opt 2) | sign + magnitude band | ~0 (sign) / 30 s (RT) | Medium → High |
| C5 BER | **C** AWGN/Rayleigh bounds | in [lower, upper × 1.5] | ~1 ms | High (textbook) |
| C6 IRC + coverage | **A** geometric + FSPL | boolean + ±5 pp | ~10 ms | High |
| C7 fairness | **A** Jain's analytical | ±0.1 on index | ~1 ms | High (Jain 1984) |

---

## Implementation roadmap

### Phase 1: Analytical references only (1-2 hours work)

Add `benchmark/oracle/analytical.py` with:
- `reference_c1_coverage_pct`
- `reference_c2_min_rss_dbm`
- `reference_c3_coverage_diff_pp`
- `reference_c4_sign_check`
- `reference_c5_ber_bounds`
- `reference_c6_irc_compliant`
- `reference_c7_fairness_index`

Add `_check_reference_match` to `verifier.py` that dispatches on capability.

Modify task generators to include reference values in each task's verifier
spec — e.g., for C1:

```python
"verifier": {
    "type": "composite",
    "subchecks": [
        ...core...,
        {"metric": "coverage_pct", "type": "metric_range", "min": 50, "max": 100},
        {"metric": "reference_match", "type": "code_contains",
         "spec": {"capability": "C1", "field": "coverage_pct",
                  "ref": 100.0, "tolerance": 5.0}},
        ...
    ]
}
```

### Phase 2: Sionna RT cache for C4 (optional, +1 hour)

Add `benchmark/oracle/sionna_cache.py` that pre-runs Sionna RT for each
C4 task and caches `coverage_pct_before` and `coverage_pct_after`.
Used when sign-only check is too loose.

### Phase 3: Re-run benchmarks with reference

Re-evaluate C1–C7 results with reference check applied. Expected outcome:
- with_skill v5 pass rate: drops some, since reference is stricter
- no_skill, self_gen: drops even more (or stays 0%)
- Δ_skill remains substantial, with stronger "agent actually computes correctly" interpretation

---

## What this gets us in the paper

**Before reference oracle**:
> "We verify the agent produces plausible RF simulation outputs (range checks, monotonicity, physical bounds)."

**After reference oracle**:
> "We verify the agent computes RF metrics within ±5 pp of textbook FSPL ground truth for simple scenes and within ±10 pp of Sionna RT for complex scenes — confirming that the skill teaches not just procedural format but actual physical correctness."

The second claim is **much stronger** and what RF community reviewers care about.
