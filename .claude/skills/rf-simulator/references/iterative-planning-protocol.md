# Iterative Planning Protocol — MACRO / MICRO Mode Switching

Use this when the user wants iterative coverage optimization: "deploy 3 APs to maximize coverage in this room," "improve the current placement until I hit 95%," or any prompt that implies multiple simulate-evaluate-adjust rounds.

The protocol is a state machine you run in your own context, *not* a runtime that spawns subagents. State (current deployment, history, mode) lives in `planning_state.json` between iterations.

## State you maintain

`planning_state.json`:
```json
{
  "iteration": 3,
  "mode": "MACRO",
  "deployment": {"APs": [{"id":"AP1","x":3,"y":5,"z":2.5,"power_dbm":20}]},
  "best_so_far": {"deployment": {}, "coverage": 0.91},
  "history": [
    {"iter": 1, "coverage": 0.78, "action_taken": "initial_placement"},
    {"iter": 2, "coverage": 0.85, "action_taken": "reposition AP1 north 1.5m"},
    {"iter": 3, "coverage": 0.86, "action_taken": "reposition AP1 east 0.5m"}
  ],
  "stop": false
}
```

## Per-iteration loop (Algorithm 1)

Each iteration runs four phases in sequence:

1. **Simulate**: run `template_rt_coverage.py` on the current deployment. Save `coverage_map.npy`.
2. **Reflect**: load `references/reflection-protocol.md` and produce `action_plan.json` for the current map.
3. **Decide**: pick a strategy via the decision rules below.
4. **Execute**: apply the chosen action and update `planning_state.json`.

Terminate when *any* of these:
- `coverage ≥ target_coverage`
- `action_plan.stop_recommended == true`
- `iteration ≥ T_max` (see budget formula below)

## Decision engine (Algorithm 2)

Hard rules (deterministic — apply BEFORE any LLM judgment):

```
if coverage_current >= coverage_target:        return STOP
if action_plan.stop_recommended:               return STOP
if action_plan.confidence < 0.5:               return ESCALATE_TO_USER
if mode == MACRO and len(history) >= 2:
    delta_n  = history[-1].coverage - history[-2].coverage
    delta_n1 = history[-2].coverage - history[-3].coverage if len(history) >= 3 else 1.0
    if abs(delta_n) < 0.02 and abs(delta_n1) < 0.02:   # ε = 2 percentage points
        switch_mode(MACRO -> MICRO)
        return MICRO_OPTIMIZE
if mode == MACRO:                               return MACRO_ADJUST
return MICRO_OPTIMIZE                           # already in MICRO
```

The macro→micro switch is **one-directional**: once you enter MICRO, you don't revert.

## MACRO_ADJUST procedure

Apply the highest-priority `action` from the action plan. Position deltas are in meters; clip to room boundary before applying. Update `planning_state.deployment`. Increment iteration.

## MICRO_OPTIMIZE procedure

Run `templates/template_optimize.py` for one micro-loop pass:
- Adam optimizer, learning rate η=0.01, 50–100 gradient steps.
- Objective: maximize 5th-percentile RSS across the coverage grid (proxy for worst-case coverage; simultaneously improves average and shrinks blind spots).
- Variables: AP positions (x, y) only. Z (height) and orientation are fixed in MICRO mode.
- Constraint: clip positions to room boundary after each step.

Save the resulting deployment to `planning_state.deployment`.

## Adaptive iteration budget

Don't fix `T_max` globally — scale it to scene complexity using the formula:

```
T_max = T_base + α · N_AP + β · ⌊A_room / A_ref⌋ + γ · 1[N_furn > N_thr]
```

Named constants:

| Symbol | Value | Meaning |
|---|---|---|
| `T_base` | 3 | Baseline iteration budget |
| `α` | 1 | Per-AP contribution |
| `β` | 1 | Per-`A_ref`-of-floor contribution |
| `γ` | 2 | Furniture-density bonus when above threshold |
| `A_ref` | 30 m² | Reference floor-area unit |
| `N_thr` | 8 | Furniture-count threshold |

`N_AP` is the number of APs being optimized, `A_room` is room area in m², `N_furn` is the furniture-item count, and `1[·]` is the indicator function.

Worked examples (paper §4.5):

- 30 m² room, 1 AP, 6 furniture items → `T_max = 3 + 1·1 + 1·⌊30/30⌋ + 2·1[6>8] = 3 + 1 + 1 + 0 = 5`.
- 120 m² floor, 3 APs, 20 furniture items → `T_max = 10` (paper-mandated cap; the indicator-bonus mode is applied with both floor-scaling and AP terms damped under that ceiling for the densest scenes).

In practice, well-behaved scenes converge in 3–4 iterations; the budget is an upper bound.

## Human-in-the-loop modes

Three intervention granularities the planner must support during an active iteration loop. Each takes effect from the next phase boundary; never mid-phase.

1. **Constraint injection.** The user adds a new requirement mid-session (e.g. *"do not place an AP near the south window"*). Append the constraint to the active requirement set in `planning_state.json`. Effective from the next iteration's MACRO_ADJUST / MICRO_OPTIMIZE step; current iteration completes as already in flight.

2. **Element locking.** The user pins specific APs or furniture items by id. Locked APs are excluded from both MACRO position adjustments and MICRO gradient updates (their (x, y) become constants in the optimizer). Locked furniture items are excluded from any scene rearrangement the planner would otherwise propose. Record locks under `planning_state.locks = {"ap_ids": [...], "furniture_ids": [...]}`.

3. **Override and steering.** The user rejects the proposed typed action plan and substitutes a manual instruction in natural language (e.g. *"forget the reposition, raise AP2 to 3 m instead"*). The orchestrator parses the instruction into a structured edit operation matching the action-plan schema. The Planner applies it, logs `action_taken="user_override: <instruction>"`, and the loop resumes from the next **simulate** phase.

## Initial deployment heuristics

Before iteration 1, propose `N_AP` initial AP positions using these rules in order:

1. Place AP1 at the geometric centroid of the target coverage area.
2. For multi-AP, distribute APs to maximize mutual distance while preserving overlap (use Lloyd's algorithm or k-means on the grid).
3. Avoid placing APs directly behind large obstructions (concrete pillars, metal cabinets) per `scene_state.json`.
4. Prefer wall- or ceiling-mounted positions (z = ceiling_height − 0.3 m).
5. Respect any user-specified hard constraints (forbidden zones, fixed AP positions).

## Logging convention

After each iteration, append to `planning_state.history`:
```
{"iter": N, "coverage": <pct>, "mode": "MACRO|MICRO",
 "action_taken": "<short description>",
 "action_plan_ref": "iter_<N>_action_plan.json"}
```

Save the per-iteration action plan as `iter_<N>_action_plan.json` so the user can audit the planning trace.
