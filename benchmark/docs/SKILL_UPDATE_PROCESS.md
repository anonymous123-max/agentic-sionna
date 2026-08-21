# Skill Update Process

How to improve the skill based on evaluation results. Grounded in ReAct
(Reason-Act-Observe), Memento-Skills (Read-Write Reflective Learning),
and SkillRL (recursive skill evolution).

## Update Classes

Every instruction block in the skill has an update class:

| Class | Where | Rule |
|---|---|---|
| **FROZEN** | `references/static-knowledge.md` | Domain expert review only. Physical constants, 3GPP tables, formulas. |
| **STABLE** | Most reference files | Update only when 3+ test failures point to the same instruction. |
| **ACTIVE** | SKILL.md body, templates PARAMS | Update freely based on eval feedback. |

## The Reflection-to-Knowledge Pipeline

After each eval round, convert agent reasoning traces into structured
skill updates. This is the core learning mechanism — it turns ephemeral
debugging into reusable knowledge.

### Step 1: Run evals

```bash
python benchmark/run_experiments.py run --exp correctness
python benchmark/evaluate_quality.py --method radiotwin_full_correctness
```

### Step 2: Extract reflection tuples from failed runs

For each failure, read the agent transcript and extract:

```
OBSERVATION: What the agent tried
FAILURE: What went wrong (error message or wrong output)
CORRECTION: How the agent (or human) fixed it
PRINCIPLE: Generalizable rule that prevents this failure class
UPDATE_TARGET: Which file and section to update, with update class
```

Example:
```
OBSERVATION: Agent used CDL-A channel for a 4-user MIMO scenario
FAILURE: Sionna raised ValueError — CDL only supports single TX/RX pair
CORRECTION: Switched to UMi channel model
PRINCIPLE: CDL models support point-to-point only. For NUM_TX > 1, use UMi/UMa/RMa.
UPDATE_TARGET: references/channel-models.md § Model Selection [STABLE]
```

### Step 3: Filter principles

Discard principles that are:
- Test-case-specific (tied to one scene, one parameter set)
- Already covered by an existing instruction
- Speculative (not grounded in an observed failure)

Keep principles that are:
- Generalizable to a class of tasks
- Derived from a real failure with a confirmed fix
- Not redundant with existing instructions

### Step 4: Apply updates

For ACTIVE blocks: apply directly.
For STABLE blocks: apply only if 3+ failures point to the same instruction.
For FROZEN blocks: flag for domain expert review.

### Step 5: Re-run evals to verify improvement

The updated skill should pass the failing test cases without regressing
on passing ones. If regression occurs, revert the update.

## Baseline Scores (from task-baselines.md)

Use these to determine success — code that runs is not enough:

| Task | Metric | Baseline | Target |
|---|---|---|---|
| BER (LDPC QPSK AWGN) | Eb/N0 gap | Theory | ±0.5 dB at BER=1e-4 |
| Channel estimation | NVE | LS ≈ 94 | < 50 |
| Coverage map | MAE | Analytical ≈ 8 dB | < 5 dB |
| Scene generation | Structural checks | Empty room | All checks pass |

## When to Stop Iterating

- All Tier 1 assertions in `evals.json` pass
- No new PRINCIPLE tuples extracted from failures (stable state)
- Tier 2 qualitative feedback is positive from 3+ reviewers
- Token efficiency is within 2x of without-skill baseline
