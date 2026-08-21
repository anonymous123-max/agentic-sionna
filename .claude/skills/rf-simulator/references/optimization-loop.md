# Step 6: Agent-Driven Iterative Planning

When the user requests coverage optimization (e.g., "achieve 85% coverage",
"optimize AP placement", "maximize signal strength"), use the **closed-loop
iterative planning pipeline** with structured multi-modal reflection.

## 6.0 Quick-Check Gate (Hybrid Approach)

**Before entering the full iterative loop, check if optimization is even
needed.** Many scenes meet the target with default placement — running the
full loop wastes time and risks timeout.

```
QUICK-CHECK GATE:
  1. Compute coverage at ceiling-center position
     → If coverage >= target: DONE. Report success. Skip loop.

  2. Compute coverage at 9 grid positions (3×3 across room)
     → If best position >= target: DONE. Report that position.

  3. If still below target → enter the full iterative loop (§6a onward)
     with the 9-position results as warm-start context
```

This gate eliminates unnecessary agent overhead on easy/medium scenes
(which typically meet target with center placement) and provides warm-start
data for the iterative loop on hard scenes. In evaluation, this improved
pass rate from 88.9% to 100% by preventing timeout failures on scenes
that didn't need optimization.

## Reference Files (load only if entering the iterative loop)

| File | Purpose |
|------|---------|
| `references/reflection-protocol.md` | 5-dimension evaluation protocol (D1-D5) |
| `references/action-plan-schema.md` | ActionPlan JSON schema + cause taxonomy |
| `references/simulation-result-schema.md` | SimulationResult format |

## Initial Deployment Strategy (D₀)

The quality of the initial TX placement significantly affects convergence
speed. Use these 5 heuristics instead of random or default placement:

1. **Centroid rule**: Place the first AP near the geometric centroid of
   the target coverage area (not the room centroid if they differ)
2. **Mutual distance**: For multi-AP deployments, distribute APs to
   maximize mutual distance while maintaining 20-30% coverage overlap
3. **Obstruction avoidance**: Never place an AP directly behind a large
   obstruction (concrete wall, metal cabinet, server rack). Check the
   scene_state.json for walls and furniture before placing.
4. **Practical mounting**: Prefer ceiling-mounted positions (height =
   room_height - 0.2m) or wall-mounted positions for deployability
5. **User constraints**: If the user specified constraints ("not near
   the window", "in the hallway"), apply them before optimization

These heuristics are applied by the LLM using spatial reasoning over the
scene geometry. They reduce the average iteration count by ~1.8 compared
to centroid-only placement.

## 6a. The Iterative Planning Loop (Algorithm 1)

```
T_max = adaptive_budget(scene)  # see §6b

mode = MACRO
history = []

for t = 1 to T_max:
  1. SIMULATE  — run coverage using template_rt_coverage.py
                  Output: simulation_result.json
  2. REFLECT   — follow 5-dimension protocol (references/reflection-protocol.md)
                  Evaluate D1-D5, produce ActionPlan JSON
  3. DECIDE    — route action via Decision Engine (§6c)
                  Returns: STOP | MACRO_ADJUST | MICRO_OPTIMIZE
  4. EXECUTE   — apply the decided action
  5. LOG       — append to optimization_log.json + action_plans/

  if action == STOP: break
```

## 6b. Adaptive Iteration Budget

Compute a scene-dependent budget instead of fixed MAX_ITERATIONS:

```
T_max = T_base + α·N_AP + β·floor(A_room / A_ref) + γ·𝟙[N_furn > N_thr]
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| T_base | 3 | Minimum iterations |
| α | 1 | Per AP |
| β | 1 | Per 30 m² of room area |
| A_ref | 30 m² | Reference room area |
| γ | 2 | Complex furniture penalty |
| N_thr | 8 | Furniture count threshold |

Examples:
- 30 m² office, 1 AP, 6 furniture: T_max = 3 + 1 + 1 + 0 = 5
- 120 m² open plan, 3 APs, 20 furniture: T_max = 3 + 3 + 4 + 2 = 12

## 6c. Decision Engine (Algorithm 2)

After each reflection, route the next action deterministically:

```
DECIDE(ActionPlan A, mode, history H):
  if A.coverage_current >= A.coverage_target → STOP
  if A.stop_recommended → STOP
  if A.confidence < 0.5 → flag for user review, then MACRO_ADJUST

  if mode == MACRO:
    Δ₁, Δ₂ = coverage gains of last 2 iterations from H
    if |H| >= 2 and Δ₁ < ε and Δ₂ < ε:  # stagnation (ε = 2 pp)
      → MICRO_OPTIMIZE, switch mode to MICRO
    else:
      → MACRO_ADJUST, stay in MACRO

  if mode == MICRO:
    → MICRO_OPTIMIZE
```

**Routing rules are deterministic** — the system must never skip a
convergence check or ignore a low-confidence warning due to LLM variability.

## 6d. MACRO_ADJUST Phase

When the Decision Engine selects MACRO_ADJUST:

1. Read the ActionPlan's `actions` list
2. Apply the highest-priority action:
   - `move_ap`: update TX position by `delta_position`
   - `add_ap`: add new TX at `new_position`
   - `adjust_orientation`: rotate TX antenna
3. Clip positions to room bounds (0.5m margin)
4. Re-run simulation with updated TX config
5. Show the user: "Moving TX1 from (x1,y1,z1) to (x2,y2,z2) because
   [cause_detail from blind spot diagnosis]"

**Limit to 1-2 actions per MACRO iteration** to avoid over-correction.
The agent retains control over which actions to apply and may reduce
displacement magnitudes (e.g., "suggested 3m move would exit room, reducing
to 1.5m").

## 6e. MICRO_OPTIMIZE Phase

When the Decision Engine selects MICRO_OPTIMIZE (after MACRO stagnates):

Use scipy.optimize to fine-tune TX position over the analytical coverage
model. This is gradient-free optimization (L-BFGS-B or Nelder-Mead) that
explores the local neighborhood around the MACRO solution.

```python
from scipy.optimize import minimize

def optimize_tx_micro(scene_state, initial_pos, threshold_dbm, cell_size=0.5):
    """Fine-tune TX position via scipy optimization."""
    bounds_w = scene_state["scene"]["bounds"]["width"]
    bounds_l = scene_state["scene"]["bounds"]["depth"]

    def objective(pos):
        cov, _ = compute_analytical_coverage(scene_state, pos[0], pos[1],
                                              threshold_dbm, cell_size)
        return -cov  # minimize negative coverage

    result = minimize(
        objective,
        x0=[initial_pos[0], initial_pos[1]],
        method="L-BFGS-B",
        bounds=[(0.5, bounds_w - 0.5), (0.5, bounds_l - 0.5)],
        options={"maxiter": 50, "ftol": 0.01},
    )
    return [round(result.x[0], 2), round(result.x[1], 2), initial_pos[2]]
```

MICRO mode is one-directional: once entered, the system does not revert to
MACRO. MICRO typically completes in 1-2 iterations (50 optimizer steps each).

## 6g. Convergence Tracking

Every iteration appends to `optimization_log.json`:

```json
{
  "iteration": 2,
  "timestamp": "2026-04-07T14:30:00",
  "tx_positions": [[3.5, 2.0, 2.8]],
  "coverage_pct": 84.1,
  "dead_zone_count": 1,
  "mode": "MACRO",
  "action_type": "move_ap",
  "cause": "wall_occlusion",
  "reasoning": "Moved TX west past interior wall to cover SW dead zone",
  "reflection_changed_plan": true,
  "action_plan_path": "action_plans/iteration_2.json"
}
```

Track `reflection_changed_plan`: did the structured reflection produce a
different action than a naive center-placement baseline? This metric
addresses the finding that >90% of generic reflections are confirmatory.

## 6h. Human-in-the-Loop

The loop runs autonomously by default, but supports 3 intervention modes:

- **Constraint injection**: User adds "don't place AP near the window" →
  append constraint, effective from next iteration
- **Element locking**: User locks a TX or furniture item → excluded from
  MACRO adjustment and MICRO optimization
- **Override**: User rejects the ActionPlan and substitutes a manual
  instruction → parse as structured edit, apply, resume loop

Between iterations, check if the user has provided any of these. If so,
incorporate before the next simulation.

## 6i. User Communication

- **Always show** the D1-D5 reflection reasoning — users need to understand
  *why* the agent is making each decision
- **Stream progress**: Report coverage % and mode after each iteration
- **On convergence**: Show the full optimization log as a summary table:
  ```
  Iter  Mode   TX Position      Coverage  Cause              Action
  1     MACRO  (2.5, 2.0, 2.8)  62.3%     —                  initial
  2     MACRO  (3.5, 2.0, 2.8)  71.8%     wall_occlusion     moved east
  3     MACRO  (3.2, 3.0, 2.8)  84.1%     distance_atten.    moved north
  4     MICRO  (3.3, 3.1, 2.8)  86.2%     —                  L-BFGS-B
  5     MACRO  +TX2 (1.5,1.0)   93.1%     distance_atten.    added TX2
  ```
- **On failure**: Report best achieved and suggest what might help
