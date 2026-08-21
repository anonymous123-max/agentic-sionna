# Backend Script Guidelines

## Table of Contents

1. [Runtime Requirements](#runtime-requirements) — Environment setup for Sionna scripts
2. [Script Lifecycle](#script-lifecycle) — 5-phase pattern every script follows
3. [Parameter Bounds](#parameter-bounds) — Physical limits for input validation
4. [Error Handling](#error-handling) — Domain exceptions and actionable fixes
5. [Output Format](#output-format) — Structured result format (ACI pattern)
6. [Guardrails](#guardrails) — Pre-execution precondition checks

---

## Runtime Requirements

Apply these when running Sionna scripts — omitting any of them causes
hard-to-debug failures:

1. **Unbuffered output**: Run with `python -u` or `PYTHONUNBUFFERED=1`.
   Without this, stdout buffers and the script appears to hang.

2. **Mitsuba cleanup crash (RTX 5090/Blackwell)**: End scripts with
   `os._exit(0)` to skip Mitsuba's destructor, which triggers
   `free(): invalid pointer` on sm_120 GPUs. This is a Mitsuba bug.

3. **LD_LIBRARY_PATH** (conda sionna env):
   ```bash
   LD_LIBRARY_PATH=$(conda info --base)/envs/sionna/lib:$LD_LIBRARY_PATH python -u script.py
   ```

4. **TF warnings**: Suppress with `TF_CPP_MIN_LOG_LEVEL=2`.

---

## Script Lifecycle

Every script follows 5 phases. The templates already implement this
pattern — follow it when writing custom (non-template) scripts.

```
VALIDATE → PLAN → EXECUTE → VALIDATE OUTPUT → REPORT
```

**Phase 1: Validate** — Check parameters against bounds (see table below).
Reject immediately with an actionable error message.

**Phase 2: Plan** — Log what will happen and estimate computation cost.
This lets the user cancel expensive runs early.

**Phase 3: Execute** — Wrap computation in try/except with domain
exceptions. Use `RadioMapSolver()` (not the deprecated `scene.coverage_map()`).
Provide center, orientation, and size together — they are all-or-nothing.

**Phase 4: Validate output** — Check for all-zero path gain, RSS exceeding
TX power, NaN/Inf values, and low coverage fraction. See
`references/physics-validation.md` for the full checklist.

**Phase 5: Report** — Save structured result as JSON alongside .npy data
files. Use the `standard_result()` format below.

---

## Parameter Bounds

Reject inputs outside these physical limits before any computation:

| Parameter | Min | Max | Unit | Typical values |
|-----------|-----|-----|------|----------------|
| frequency | 100e6 | 300e9 | Hz | 3.5 GHz (5G NR), 28 GHz (mmWave), 60 GHz (WiGig) |
| tx_power | -30 | 60 | dBm | 23 (small cell), 43 (macro) |
| cell_size | 0.1 | 10.0 | m | 0.5 (indoor), 2.0 (outdoor) |
| max_depth | 1 | 10 | int | 3 (LOS-dominated), 5-7 (indoor NLOS) |
| num_samples | 1e3 | 1e8 | int | 1e6 (fast preview), 1e7 (publication) |
| bandwidth | 1e3 | 400e6 | Hz | 20 MHz (typical) |

---

## Error Handling

Use domain-specific exceptions with actionable fix hints — generic
exceptions don't tell the agent how to self-correct:

```python
class SceneError(Exception):
    def __init__(self, message: str, fix_hint: str):
        super().__init__(message)
        self.fix_hint = fix_hint
```

Rules:
- Catch `RuntimeError`, `FileNotFoundError`, `ValueError` individually — not blanket `Exception`
- Include parameter values in error messages so the user knows what they passed
- Log before raising: `logger.error(...)` with context
- Check `references/error-patterns.md` for known errors before writing new handling

---

## Output Format

Every script returns a structured dict for consistent parsing. This
pattern (inspired by Agent-Computer Interface design) optimizes output
for LLM consumption:

```python
def standard_result(status, result, next_suggested, guardrail_warnings):
    return {
        "status": status,           # "success" | "error" | "warning"
        "result": result,           # domain-specific output
        "next_suggested_action": next_suggested,
        "guardrail_warnings": guardrail_warnings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
```

| After this action | Suggest next |
|---|---|
| Create scene | Place TX |
| Place TX | Run coverage |
| Run coverage | Generate heatmap |
| Run BER | Generate chart |
| Optimize TX | Verify coverage at optimal position |

---

## Guardrails

Check these preconditions before executing. Add violations to
`guardrail_warnings` — do not silently proceed with bad inputs:

| ID | Check | Default if violated |
|---|---|---|
| G-FREQ | frequency_hz set and in [100e6, 300e9] | Default to 3.5 GHz with warning |
| G-TX | At least 1 TX in scene | Error — cannot simulate without TX |
| G-SCENE | scene_state.json exists and valid | Error — create scene first |
| G-BOUNDS | TX/RX within scene bounds | Warning — clip to bounds |
| G-MATERIAL | Materials in ITU known list | Default to itu_concrete with warning |

---

## Lazy Imports

Import optional packages (sionna, osmnx, trimesh) inside the function
that uses them, not at module top level. This prevents ImportError for
users who don't have the dependency:

```python
def run_sionna_coverage(...):
    import sionna.rt as rt  # only imported when this function runs
    ...
```

---

## State Management

`scene_state.json` is the single source of truth. Update it atomically
(write to temp file, then `os.replace()` to rename):

```python
import tempfile, os
fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
with os.fdopen(fd, "w") as f:
    json.dump(state, f, indent=2)
os.replace(tmp, path)
```

---

## Related

- `sionna-v2-api.md` — Sionna API patterns used in scripts
- `error-patterns.md` — Known errors with auto-fix strategies
- `defaults.md` — Default parameter values
- `physics-validation.md` — Output validation formulas
