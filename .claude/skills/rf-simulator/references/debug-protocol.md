# Debug-from-Logs Protocol

This file is loaded on-demand from SKILL.md's Step 6 when an agent encounters a failure mode not classifiable from the failure-class table alone, or when the agent needs the longer-form rationale for the log-driven ReAct loop.

## Why log-driven debugging (vs blind retries)

The inner ReAct loop in Step 6 of SKILL.md follows `Reason → Act → Observe → Reason`. The most common failure mode in iter1–iter3 traces wasn't *code generation quality* — it was the agent **retrying with the same wrong fix three times in a row** because it never read the actual error. The Step 6 retry table requires the agent to cite the line from `run.log` motivating each fix; this file expands that discipline.

## Capture pattern (Act step)

Always pipe stdout + stderr to a file so the Observe step can `Read` it:

```bash
python3 simulation.py 2>&1 | tee run.log
RC=$?
echo "exit code: $RC" >> run.log
```

For the verifier:

```bash
python3 $RF_SKILL_DIR/scripts/verify_output.py --workdir . 2>&1 | tee verify.log
```

Both `run.log` and `verify.log` are now `Read`-able artifacts. **Do not** rely on the agent's memory of the last turn's tool result — re-`Read` the log on each Observe.

## Observe step: what to extract

Three slices of the log matter; everything else is noise.

1. **Traceback head**: the first `File "..."` line + the exception type. This tells you *where* and *what*.
   ```bash
   grep -A1 'Traceback' run.log | head -6
   # Or: read the last ~20 lines of stderr, which contains the bubble-up of the exception.
   tail -20 run.log
   ```
2. **Verifier `[FAIL]` lines**: each is one failed check.
   ```bash
   grep '\[FAIL\]' verify.log
   ```
3. **Numerical sanity**: if the run executed but produced bad numbers (BER > 0.5 at high SNR, NMSE > 0 dB on a denoiser), the verifier prints `[FAIL] plausibility:<reason>`. Look there before assuming the code crashed.

Skip the rest. A 2000-line `run.log` is rare and almost never necessary to read in full.

## Reason step: classify before fixing

The Step 6 failure-class table in SKILL.md has 10 rows. Match the **first line of the traceback** or the **first `[FAIL]` line** to one of them. If two rows match, pick the **earliest** one — fixing an earlier error often makes later ones disappear.

If no row matches:

1. `cat $RF_SKILL_DIR/references/error-patterns.md` — distilled symptom→root→fix mappings from past failures.
2. If still nothing, treat it as the "Unknown error" row: switch to analytical fallback rather than burn retries.

## Act step: one targeted fix per retry

The single most common anti-pattern in early traces was the agent making **multiple speculative changes per retry** ("change the carrier frequency AND switch modulation AND adjust SNR range"). When the next run fails differently, you don't know which change caused which effect.

**Rule: one fix per retry, attributed to one log line.**

If you find yourself wanting to change two things, pick the one that's earlier in the traceback and run again. The second issue may resolve itself.

## Budget discipline

- **3 retries total.** Not 3 per failure-class — 3 across the whole task.
- If the same `[FAIL]` line appears on two consecutive retries, the strategy isn't working. Switch to analytical fallback on retry 3 (don't burn it on another minor edit).
- Wall-time matters: each retry is ~30-60s of inference. After 3, you've spent 2-3 minutes; the trial timeout is 1200s (20 min). Don't soak up time on retries that have plateaued.

## When to escalate to analytical fallback

Switch to numpy/scipy analytical fallback (from the Step 5 fallback table) immediately if:

- `ModuleNotFoundError: sionna` (Sionna unavailable in this environment — common on CPU-only runners)
- `CUDA out of memory` after halving batch size once
- Same `[FAIL]` 2 retries in a row on a non-shape, non-API issue (model is fundamentally not converging)
- The trial's verifier asks for an artifact (`coverage_map.npy`, `cir.npy`) that the failing Sionna path will never produce — write the analytical version directly

The Step 5 fallback recipes in SKILL.md cover BER (Q-function), FSPL coverage, NMSE/LMMSE channel estimation, sum-rate, and synthetic CIR. They run in <5s on CPU and reliably pass the verifier's existence + plausibility checks.

## Output discipline post-failure

Even after a failed retry sequence, emit a valid `simulation_result.json` per Step 7's standardized schema. Empty / null fields fail every plausibility check; a populated analytical-fallback JSON passes existence and most threshold checks.

```python
# After exhausting retries, ensure these fields exist with real numbers:
result = {
  "schema_version": "1.0",
  "numerical_metrics": {
    # task-specific fields per templates/result_schema_<task>.json
  },
  "warnings": [
    {"kind": "fallback", "source": "agent",
     "message": "Sionna run failed (see run.log); used analytical FSPL fallback."}
  ],
}
import json
json.dump(result, open("simulation_result.json","w"))
```

The `warnings` block is how reviewers / the auto-improvement loop distinguish "got the right answer cleanly" from "got the right answer via fallback". Always populate it on fallback paths.
