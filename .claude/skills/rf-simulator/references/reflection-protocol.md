# Reflection Protocol — Diagnose Coverage Maps

Use this when the user shows you a coverage map (or you've just produced one) and asks "what's wrong" / "where are the blind spots" / "how do I improve coverage."

## Inputs you need

Before diagnosing, gather all three:

1. **Numerical metrics** from `simulation_result.json`:
   - `coverage_pct`, `rss_p5_dbm` (5th-percentile worst-case RSS), `rss_mean_dbm`, `rss_std_db`, `blind_spot_area_m2`, `blind_spot_pct`.
   - `snr_mean_db`, `snr_std_db` (mean and standard deviation of SNR over the receiver grid).
   - `path_loss_per_rx_db[]` — per-receiver path-loss values, one entry per receiver in the grid. Required so the diagnoser can correlate weak zones with specific receivers, not just aggregate statistics.
2. **Visual output**: the rendered coverage heatmap (`coverage_map.png`) overlaid on the floor plan with AP positions marked. Standardized colormap (red=strong, blue=weak).
3. **Scene context**: `scene_state.json` — AP positions/orientations/TX power, room geometry, furniture layout with material categories, the user's stated coverage target.

If any are missing, *ask* — don't guess.

## Five-dimension diagnostic walkthrough

Step through these in order. Each produces a structured fragment for the action plan.

### D1 — Coverage completeness
Does current performance meet the target? What fraction below `P_min` (default −80 dBm)? Any priority zones uncovered?

### D2 — Signal-strength uniformity
Is RSS evenly distributed or are there extreme variations? An 92%-coverage deployment with 30 dB dynamic range may still be unacceptable if weak zones overlap high-traffic areas.

### D3 — Blind-spot diagnosis (the core step)

For each identified weak zone, attribute a **single primary cause** drawn from this closed taxonomy. Each cause implies a different remedy:

| Cause | Definition | Typical remedy |
|---|---|---|
| `wall_occlusion` | Blind spot lies behind a concrete/brick wall relative to nearest AP | Reposition AP |
| `furniture_blockage` | A large object (metal cabinet, tall bookshelf) blocks LoS | Reorient antenna or raise mounting height |
| `distance_attenuation` | Far from all APs; no single obstruction responsible | Add a new AP |
| `interference_shadow` | Destructive interference / handover gap in multi-AP layout | Adjust AP frequencies or transmit power |
| `material_penetration_loss` | Path traverses RF-attenuating material (glass partition, metal-frame door) | Reposition AP to avoid the path |

If multiple causes plausibly apply to one zone, name the dominant one and note the secondary in `notes`. Do not pick more than one primary cause per zone.

### D4 — Gap analysis
Quantify: how far is current coverage from target? Where is residual gap *located* (which zones, which fraction of total area)?

### D5 — Action recommendation
Synthesize D1–D4 into one or more concrete actions. Each action is typed (see schema below) and references a specific AP or scene element.

## Action plan output

Emit your diagnosis as JSON conforming to `templates/result_schema_action_plan.json`. Save it to `action_plan.json` in the working directory.

Required fields:
- `coverage_current` (float, %), `coverage_target` (float, %)
- `blind_spots` — list of `{location: [x,y], area_m2: float, cause: <one of taxonomy>, notes: str}`
- `actions` — list of `{type: reposition|reorient|add, ap_id: str, delta: [dx,dy,dz] | null, expected_gain_pp: float}`
- `confidence` — float in [0,1]; below 0.5, escalate to user rather than auto-execute
- `stop_recommended` — bool; true when no further improvement is achievable under current scene constraints

## Action types — reference

- **reposition**: move AP `ap_id` by `delta` (m). Expected_gain_pp is the predicted coverage % improvement.
- **reorient**: change antenna pattern direction. `delta` is `[d_azimuth_deg, d_elevation_deg, 0]`.
- **add**: place a new AP. `ap_id` is the proposed name; `delta` is the absolute position vector (not delta from anything).

## Confidence calibration

Self-assessed confidence reflects how strongly the data supports your diagnosis:
- ≥0.8: ≥3 corroborating signals (numerical + spatial pattern + scene match)
- 0.5–0.8: 1–2 signals; remedy is plausible but not certain
- <0.5: ambiguous diagnosis — escalate to user, don't auto-execute the action plan

## Stop conditions

Set `stop_recommended=true` when:
- Coverage already meets target.
- Remaining blind spots are all `material_penetration_loss` through walls the user said cannot move.
- Last 2 iterations produced <2 percentage-point improvement (use planning history if iterating).