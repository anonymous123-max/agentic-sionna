# Benchmark Methodology for the Sionna/RF Skill

> Sources: SkillsBench, SWE-Skills-Bench, AgentSkills.io, RewardHackingAgents,
> LiveBench, NVlabs/the-ai-telco-engineer.

## Core Principle

Benchmarking a skill is not benchmarking a model. Every task runs under
3 conditions — no skill, curated skill, self-generated skill — to isolate
the skill's contribution from the model's baseline capability.

## Three Conditions

1. **No skill**: Raw model with no RF domain context
2. **Curated skill**: The rf-simulator skill loaded normally
3. **Self-generated skill**: Model asked to write its own Sionna instructions
   before completing the task (controls for "any context helps")

## Evaluation Rules

- **Deterministic verifiers only** for pass/fail — no LLM-as-judge
- **5 trials per task**, fixed denominator (all tasks, including timeouts)
- **Fresh session per trial** — no state bleed between runs
- **Skill and no-skill in separate sessions** — prevent injection contamination
- **60/40 train/held-out split** — never tune skill on held-out set

## Metrics

### Primary

| Metric | Formula |
|---|---|
| Pass rate | tasks_passed / total_tasks (averaged across 5 trials) |
| Normalized gain | (pass_skill - pass_vanilla) / (1 - pass_vanilla) |

### Domain-Specific Continuous

| Task type | Metric | Baseline | Target |
|---|---|---|---|
| Channel estimation | NVE | LS ≈ 94 | < 50 |
| Channel estimation | NMSE | LS ≈ -3 dB | < -10 dB |
| BER simulation | Eb/N0 gap | Theory | ± 0.5 dB at BER=1e-3 |
| Radio map | MAE | Analytical ≈ 8 dB | < 5 dB |
| RIS optimization | Rx power gain | Random phase | +3 dB |
| Code correctness | Binary | — | Runs + plausible output |

### Efficiency

- Tokens used: with skill vs without
- Wall time: with skill vs without
- Token efficiency ratio: pass_rate_gain / extra_tokens

## Threats to Validity

| Threat | Defense |
|---|---|
| Oracle leakage (skill contains task answers) | Verify skill has no verbatim task parameters |
| Prompt contamination (model memorized tutorials) | Use novel parameter combinations, v2.0-specific API |
| Harness effects | Test on ≥2 configurations (Claude Code + raw API) |
| Overfitting to test set | 60/40 split, held-out evaluated once only |
| Reward hacking | Lock verifier scripts, add plausibility checks |

## Failure Classification Taxonomy

When reading trajectory logs, classify each failure:

| Class | Signs | Skill fix |
|---|---|---|
| Skill not consulted | Agent ignores skill entirely | Improve description triggering |
| Skill consulted but ignored | Agent reads then contradicts | Strengthen instruction rationale |
| Skill content wrong | Agent follows but gets wrong result | Fix the instruction |
| Skill gap | Agent struggles with uncovered topic | Add coverage |
| Model capability ceiling | Agent understands but can't execute | Not fixable by skill |
| Environment error | Docker/GPU/library issue | Fix test harness |
| Reward hacking | Verifier passes but output implausible | Strengthen verifier |

## Iteration Protocol

```
Write draft → Run TRAIN split (3 conditions × 5 trials × N tasks)
  → Read trajectories → Classify failures → Edit ONE section
  → Re-run TRAIN → Check regression → Repeat until plateau
  → Run HELD-OUT once → Report
```

Rules:
- One section per iteration (isolate cause of improvement)
- Minimum 2pp improvement per iteration or stop
- Mandatory regression check after every edit
- Stop at plateau or 5 iterations
- Never touch held-out until iteration complete

## Reporting Requirements

| Required | Why |
|---|---|
| Per-domain breakdown | Averages hide domain-specific effects |
| Negative deltas | Tasks where skill hurts — report honestly |
| Self-generated condition | Proves curated > arbitrary context |
| ≥2 models | Proves skill generalizes beyond one model |
| ≥3 ablations | Shows which sections drive improvement |
| Failure taxonomy figure | Shows where bottleneck shifts |
| Token overhead | Shows cost of skill loading |
| Iteration learning curve | Shows convergence |

## Minimum Viable Experiment Set

For a publishable result with constrained resources:

- 60 tasks × 3 conditions × 5 trials = 900 trajectories
- 2 models for generalizability
- 3 ablations (no description, no static knowledge, no ML section)
- Failure taxonomy + iteration curve + token overhead

~1,900 total trajectories. ~160 GPU-hours at 5 min/trajectory.
