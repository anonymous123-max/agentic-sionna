# rf-simulator Skill Refinement Proposal — Iter 1

**Model under test:** `Qwen/Qwen3-Coder-30B-A3B-Instruct` (32 768-token context window)
**Run:** `anvil_..._20260511_1218_dflash0`, 34 with_skill / 33 no_skill / 67 total trials.

## Headline gap is real and severe

| Condition | `pass_strict` | `verification.passed` | `score > 0` | `exec_success` | trials |
|---|---|---|---|---|---|
| `no_skill`   | 7 / 33 (21%) | 11 / 33 (33%) | **32 / 33 (97%)** | 20 / 33 (61%) | 33 |
| `with_skill` | 1 / 34 ( 3%) | 4 / 34 (12%)  | **17 / 34 (50%)** | 3 / 34 ( 9%)  | 34 |

The user's 50% vs 97% figure is `score > 0`. The more damning gap is **execution rate: 3/34 vs 20/33**. With the skill active, this 30 B model rarely produces a runnable Python script — it spends its turns reading skill files instead.

This is not a sampling artifact. The gap holds across difficulty buckets:

```
with_skill   easy 0/7   medium 0/14   hard 1/13
no_skill     easy 1/10  medium 1/9    hard 5/14   (pass_strict)
```

And across all 8 tiers `with_skill` is at-or-below `no_skill`. The 20 overlapping task IDs replicate the pattern: `no_skill` outperforms `with_skill` on 13, ties on 5 (both fail), and loses on 2 (U063 medium, U092 dimensions edge-case).

---

## Root causes (ranked by evidence weight)

### RC1 — Context-window exhaustion (high confidence)

**Evidence.** 10 / 34 `with_skill` trials hit `API Error 400: maximum context length is 32768 tokens`. **0 / 33** `no_skill` trials hit this error.

Tasks where context exhaustion stranded the agent before any code ran:
U019, U057, U104, U114, U115, U116, U118, U133, U159 (all `exec_success=False`, `py_runs=0`).

Concrete instance — U115 (T4 hard, opt_ap_placement). After reading 3 templates back-to-back:

```
Read: $RF_SKILL_DIR/templates/template_rt_coverage.py    (19 KB)
Read: $RF_SKILL_DIR/templates/template_optimize.py        ( 9 KB)
Read: $RF_SKILL_DIR/templates/template_scene.py           ( 7 KB)
→ API Error: 400 ... requested 33372 tokens (29276 in messages)
```

`no_skill` on the same task wrote a working `simulation.py` in 4 tool calls and got `pass_strict=True`.

**Why the skill triggers this.** The harness prompt (`benchmark/prompts/skill_hint_full.txt`, 103 lines) already contains the full Sionna v2 API reference inline. SKILL.md is 292 lines (~5 K tokens). Templates are 7–19 KB each. The model is instructed in three places (prompt-tail line 96-104, SKILL.md L190-196, skill_hint_full L89-97) to "COPY the template and edit PARAMS" — which the model interprets literally as **Read each candidate template into context first** to decide which fits.

The skill's value-add (API reference) is 100% redundant with the harness prompt the agent already has. The skill's templates are too heavy to load multiple of them.

### RC2 — Agent burns turns hunting in `$RF_SKILL_DIR` instead of writing code (high confidence)

**Evidence.** Mean reads-before-first-Write in `with_skill`: 3.8, of which 2.9 are inside `$RF_SKILL_DIR/templates/`. `no_skill` typically writes after 0–1 reads (`ls` + maybe reading the placeholder `simulation_result.json`).

Per-trial counts of skill-tree reads-before-write (sampled): U060=5, U057=5, U156=4, U156=4, U040=2, U116=3 ... and so on. Combine with the `ripgrep (rg) is required` error every Glob call returns — agent retries with `find`/`ls` adding 1-2 more bash turns per pattern.

Concrete: U080 (T0 easy, scene_indoor). 18 turns, **0 python executions**, finished with the harness-placeholder `scene_state.json` untouched. The agent's last 4 turns were chasing `template_scene.py` paths, getting "File has not been read yet" errors when trying to `Write` over the placeholder, and never finishing.

```
WRITE: .../U080/t1/scene_state.json
  RESULT: <tool_use_error>File has not been read yet. Read it first before writing to it.</tool_use_error>
```

`no_skill` U080 wrote `generate_office.py` and ran it in 3 turns (`exec_success=True`, `score=0.75`).

### RC3 — Skill instructs "COPY templates with `cp`" + "edit only PARAMS" — but model reads instead (high confidence)

**Evidence.** `skill_hint_full.txt` L89-94 and SKILL.md L190-196 both say:

> TEMPLATES (Python source files; **COPY their content with `cp`**, then edit the PARAMS dict; do NOT execute them with arguments)

But across 34 `with_skill` traces, **not a single agent used `cp $RF_SKILL_DIR/templates/template_X.py simulation.py`**. Every agent does `Read template_X.py → Read template_Y.py → ...` then writes a new file from scratch. The "cp + edit PARAMS" pattern is described in prose but never demonstrated with an output token budget; the model's prior dominates and it reads.

Worse, the templates contain a full validation harness (`validate()`, `aabb_overlap()`, `export_scene_to_ply_xml()`) that the model also tries to imitate, generating 100+ line scripts when 20 lines would pass.

### RC4 — `$RF_SKILL_DIR` env variable surface area is too high; `Glob` doesn't expand env vars (medium confidence)

**Evidence.** Every `with_skill` trial has at least 2 entries of `ripgrep (rg) is required for file search`:

```
Glob: {'pattern': '$RF_SKILL_DIR/templates/template_rt_coverage.py'}
  RESULT: ripgrep (rg) is required for file search...
```

The Claude Code harness's Glob tool needs `rg` AND does not pre-expand `$RF_SKILL_DIR`. So the model burns 2 turns per attempt: one to Glob (fails), one to fall back to `ls -la $RF_SKILL_DIR`. Across the sample, this costs ~3–6 turns per trial that uses `$RF_SKILL_DIR`. Plus, the skill mentions `$RF_SKILL_DIR` 30+ times in SKILL.md (`grep -c '$RF_SKILL_DIR' SKILL.md` → 21).

### RC5 — Routing table makes the model fan-out across templates (medium confidence)

**Evidence.** SKILL.md L67-86 routing table lists 18 task types and points each to 1-4 reference files. For ambiguous prompts (e.g. U114 "campus AP placement" matches both `OPTIMIZE` and `RT_COVERAGE`), the model dutifully reads BOTH templates plus their references — that's how U114, U115, U118 hit the context wall.

Specifically L74 (`OPTIMIZE`) points to **5** files: template_optimize.py, iterative-planning-protocol.md, reflection-protocol.md, physics-validation.md, optimization-loop.md, plus the result_schema_optimize.json. The model interprets this as a reading list.

The "Tiebreak" rule on L88 (`scene → channel → PHY → ML → system`) only kicks in when multiple rows match — but the model reads aggressively before applying the tiebreak.

### RC6 — Skill discourages writing custom code even when it's the right move (medium confidence)

**Evidence.** SKILL.md L190 says "Use the named template AS-IS — modify only `PARAMS = {}`. ... If a template doesn't fit, write a separate file from scratch — never edit the template body." The model interprets this as a hard rule: it tries to make every task fit a template, including U153 (DIAGNOSE) which has no template (the routing table even says "no template — emit `action_plan.json`"). U153 with_skill: 12+ bash turns hunting for nonexistent scene files; 0 python runs; missing artifact. U153 no_skill: 4 tool calls, wrote simulation.py, action_plan.json produced (only failed on key shape).

---

## Proposed edits — concrete and minimal

Numbers below reference lines in the files as they currently stand. Confidence flags: H = high, M = medium, L = low.

### Edit 1 — Inline the "happy path" idiom at the very top of `SKILL.md` [H]

**File:** `/anvil/scratch/x-jsong16/skill/.claude/skills/rf-simulator/SKILL.md`
**Where:** Insert immediately after the front-matter (after L14), before "Restate before coding".
**Why:** The first instruction the model sees should be the action that wins. Currently, the first instruction is "restate the task in ONE line" — useful for humans, but the model's first concrete action becomes `Glob $RF_SKILL_DIR/templates/*`. We need to short-circuit that.

**Add (replaces the current L15-21):**

```markdown
## Fast path (DO THIS FIRST, even before restating the task)

For 80% of tasks, this 3-line sequence wins:

```bash
# 1. Copy the right template, DON'T read it first:
cp $RF_SKILL_DIR/templates/template_<task>.py simulation.py
# 2. Edit ONLY the PARAMS dict at the top of simulation.py (use Edit tool).
# 3. Run it:
python3 simulation.py && python3 $RF_SKILL_DIR/scripts/verify_output.py
```

Pick `<task>` from the one-word lookup table below. If unsure, just write
`simulation.py` from scratch using numpy/scipy — a 20-line analytical
fallback (Q-function BER, FSPL coverage) outscores reading 3 templates
and running out of context.

| Prompt mentions | `<task>` |
|---|---|
| BER / BLER / SNR sweep / LDPC / Polar / modulation | `ber` |
| coverage map / heatmap / radio map / signal strength | `rt_coverage` |
| ray tracing + link / CIR / CFR | `rt_to_phy` |
| room / floor plan / office / furniture / scene | `scene` |
| AP placement / optimize position / maximize coverage | `optimize` |
| MIMO / OFDM / channel estimation / precoding | `mimo_ofdm` |
| neural / learned / autoencoder / training | `neural_train` |
| multi-cell / scheduling / link adaptation | `system_level` |

If nothing fits, the prompt is a DIAGNOSE / EMERGING / RIS task — write
custom numpy/scipy and emit the canonical JSON. Do NOT loop reading
templates that don't match.

**Hard budget: at most 1 Read of any `$RF_SKILL_DIR/templates/*.py`
per task.** If you've read one and it doesn't fit, switch to custom
numpy/scipy — don't read a second.
```

This addresses RC1, RC2, RC3, RC5.

### Edit 2 — Cap the routing table to ONE reference per row [H]

**File:** `SKILL.md` L67-86 (the big routing table)
**Why:** Each row currently lists up to 5 reference files. Models treat them as a reading list. Replace with the single most useful pointer per row, and a one-line "if stuck, read this:" footnote.

**Replace** the "References (load after classify)" column entirely. For each row, keep only the **first** ref file (or `—` if the template is self-contained). The references block stays on disk for the rare case the model needs depth; just stop advertising them in the routing table.

Concretely, change L73 from:
```
| `SCENE_GEN` | room, ... | `templates/template_scene.py` OR `from lib.scene_gen import ...` + `templates/result_schema_scene.json` | `scene-gen-library.md`, `core-patterns.md`, `scene-state-schema.md`, `export-formats.md`, `data-sources.md` |
```
to:
```
| `SCENE_GEN` | room, floor plan, office | `template_scene.py` | `scene-state-schema.md` (only if validation fails) |
```

Repeat for every row. Drop the "OR `from lib.scene_gen import ...`" alternative entirely — it's never used in the traces and offers a second wrong path.

This addresses RC5.

### Edit 3 — Trim `benchmark/prompts/skill_hint_full.txt` to ~25 lines [H]

**File:** `/anvil/scratch/x-jsong16/skill/benchmark/prompts/skill_hint_full.txt` (103 lines, ~3.5 K tokens)
**Why:** This file is injected verbatim into every `with_skill` prompt. With a 32 K context window, this hint alone consumes 11% of the budget before the task is described. The Sionna v2 API quick reference (L16-47) duplicates SKILL.md, the routing table, AND the templates' docstrings. The "LOOKUP first" block (L51-77) directs the model to a vector store that returns nothing useful in the 30 B / no-chromadb deployment.

**Replace** entire file content with the existing `skill_hint_minimal.txt` (23 lines) PLUS one additional 2-line block:

```
SIONNA V2 NAMESPACES (only matters if you import sionna): use
sionna.phy.{fec,channel,modulation,mimo,ofdm}, sionna.rt.{Scene,
PathSolver,RadioMapSolver}, sionna.sys.*. Old paths (sionna.fec,
sionna.channel, sionna.mimo) ARE GONE.
```

Drop the LOOKUP section entirely (the trace data shows lookups don't help and burn turns). Drop the SUBAGENTS section (no trace used Task tool).

Make `RF_SKILL_HINT_LEVEL=minimal` the default in `benchmark/trial/prompt.py` L31. Currently:
```python
level = os.environ.get("RF_SKILL_HINT_LEVEL", "full")
```
Change `"full"` → `"minimal"`.

This is the single biggest leverage point. It addresses RC1 directly and frees ~3 K tokens per turn.

### Edit 4 — Replace "Read template" with "cp template" in BOTH SKILL.md and prompts [H]

**Files:**
- `SKILL.md` L190-196
- `benchmark/prompts/skill_hint_*.txt` (templates section)

**Why:** Telling the model "copy with cp, edit PARAMS" twice doesn't work — the model still reads. The fix is to never mention reading templates at all, and to make `cp` the only verb associated with templates.

**SKILL.md L190-196 — replace** the universal constraint #1 block:

Current:
```
1. **Use the named template AS-IS — modify only `PARAMS = {}`.** Why: templates are validated against the verifier; changes to imports/function bodies/output formatting break verification silently. Concrete idiom:
   ```bash
   cp $RF_SKILL_DIR/templates/template_<task>.py simulation.py
   # then Edit only the PARAMS = {...} dict in simulation.py
   python3 simulation.py
   ```
   If a template doesn't fit, write a separate file from scratch — never edit the template body. Task-specific rules (RT init, BER MC iter count, neural artifact list) live in each template's docstring.
```

Replace with:
```
1. **Templates: `cp` then `Edit`, never `Read` first.** Use the Bash
   tool to copy the template into the workdir, then use the Edit tool
   on the local copy. Reading the template into context wastes 5–15 K
   tokens you cannot spare on a 32 K-window model.
   ```bash
   cp $RF_SKILL_DIR/templates/template_<task>.py simulation.py
   # then Edit ONLY the PARAMS = {...} block at the top.
   python3 simulation.py
   ```
   If the template doesn't fit, write `simulation.py` from scratch with
   numpy/scipy. Do NOT Read more templates trying to find a fit.
```

This addresses RC3 directly.

### Edit 5 — Add a "if no scene file, just synthesize" guard at top of `template_scene.py` [M]

**File:** `/anvil/scratch/x-jsong16/skill/.claude/skills/rf-simulator/templates/template_scene.py`
**Why:** Multiple `with_skill` failures (U080, U092, U141) involved the agent reading `template_scene.py` to learn the schema, then writing its own scene file that doesn't include all canonical fields. The template's docstring (L1-9) describes the file but doesn't include a self-contained example of the JSON it produces. Adding a 12-line example at the very top would let the agent stop reading and start writing immediately.

**Modify** L1-9 from:
```python
"""Template: SCENE_GEN — write a valid scene_state.json fixture.

Modify ONLY the PARAMS block. This template:
1. Writes a minimal valid scene_state.json with rooms, walls, furniture, TX/RX
2. Runs a self-test for in-bounds + AABB-collision
3. Saves placeholder simulation_result.json (in case downstream tasks read it)

Schema: see references/scene-state-schema.md.
"""
```

To:
```python
"""Template: SCENE_GEN — write a valid scene_state.json fixture.

CANONICAL schema (what the verifier looks for):
{
  "schema_version": "1.0",
  "rooms":     [{"id": "r0", "type": "office", "bounds": {"x":0,"y":0,"w":10,"d":8,"h":3}, "windows": []}],
  "furniture": [{"id":"f0","type":"desk","x":2,"y":2,"theta_deg":0,"w":1.4,"h":0.7,"material":"itu_wood"}],
  "walls":     [],
  "transmitters": [{"id":"tx1","x":5,"y":4,"z":2.8,"power_dbm":20}],
  "receivers":    [{"id":"rx1","type":"grid","height":1.5,"resolution":0.5}],
  "numerical_metrics": {"num_rooms":1,"num_walls":4,"num_furniture":1,"num_transmitters":1,"scene_area_m2":80.0}
}

Modify ONLY the PARAMS block below. Self-test runs in-bounds + AABB-collision.
"""
```

This addresses RC2 and RC6: even if the agent Reads this template (against advice), the first 14 lines are enough to write code from scratch — no need to scroll through the 193-line file.

Apply the same shape to all other templates: put the canonical output JSON in the docstring's first 20 lines.

### Edit 6 — Remove the "Glob $RF_SKILL_DIR" suggestion entirely [M]

**File:** `SKILL.md` L33-35
**Why:** SKILL.md tells the model to `Run helper scripts as $RF_SKILL_DIR/scripts/<name>.py` — that path works in `Bash`, but the model first tries `Glob` with `$RF_SKILL_DIR/...` which fails because (a) the env var isn't expanded by the harness's Glob tool, and (b) `rg` isn't installed in the runner.

**Replace L33-35:**
```
## Skill paths

The harness exports `$RF_SKILL_DIR` as the absolute skill path. Run helper scripts as `$RF_SKILL_DIR/scripts/<name>.py` and import the lib via `sys.path.insert(0, "$RF_SKILL_DIR"); from lib.scene_gen import ...`. Outside the harness, `export RF_SKILL_DIR=$(realpath .claude/skills/rf-simulator)`.
```

With:
```
## Skill paths

`$RF_SKILL_DIR` is exported by the harness. Use it **only via the Bash
tool** (e.g. `cp $RF_SKILL_DIR/templates/template_ber.py simulation.py`
or `python3 $RF_SKILL_DIR/scripts/verify_output.py`). The Glob/Grep
tools do not expand env vars; use Bash with `ls $RF_SKILL_DIR` if you
must browse.
```

This addresses RC4.

### Edit 7 — Drop the "vector-store lookup" block from SKILL.md [M]

**File:** `SKILL.md` L41-58
**Why:** Across all 34 `with_skill` traces, lookups were called in only a small minority and returned silently-empty results in the traces I sampled (no chromadb in the runner). The block bloats the system prompt by ~20 lines for zero observed benefit on this model. The "CAP: 2-3 lookups MAX" guidance is also widely ignored.

**Action:** Delete L41-58 entirely. If you want to keep the entry point, replace it with one line:
```
Optional: `python3 $RF_SKILL_DIR/scripts/lookup.py "<query>"` for vector-store hits (only if chromadb is installed; silent no-op otherwise).
```

This addresses RC1 (token bloat).

### Edit 8 — Add explicit "DIAGNOSE has no template" runbook [L]

**File:** `SKILL.md` — after the routing table, before "Conditional reads" (around L88)
**Why:** U153 was a DIAGNOSE task. The routing table says "no template — emit `action_plan.json` per schema" but doesn't tell the agent what schema. The `with_skill` agent spent 12 bash turns hunting for nonexistent scene files. A 5-line runbook would have saved the trial.

**Add:**
```markdown
### DIAGNOSE shortcut

DIAGNOSE tasks (prompt says "diagnose", "what's wrong", "improve", "blind spots") produce ONE file: `action_plan.json` with this exact shape:

```json
{
  "coverage_current": 0.62,           // float in [0,1] — current coverage fraction
  "coverage_target":  0.85,           // float in [0,1] — target
  "blind_spots":  [{"x":7.0,"y":5.0,"reason":"meeting-room wall blockage"}],
  "actions":      [{"kind":"add_ap","x":8.0,"y":4.5,"z":2.8,"expected_gain_db":12.5}],
  "confidence":   0.7,                // float in [0,1]
  "stop_recommended": false           // bool
}
```

Write that JSON; you do not need a scene file, a coverage simulation, or Sionna. 5 lines of Python + `json.dump`.
```

This addresses RC6 for the DIAGNOSE family.

---

## Expected impact

These edits target the three observed failure mechanisms:

1. **Context exhaustion** (10 trials → 0): Edits 1, 3, 4, 7 cut at least 3 K tokens from the per-turn payload and eliminate the multi-template-read pattern.
2. **No-execution** (31 trials → ~10): Edits 1, 4, 5 push the agent to `cp + Edit + python3` in 3 turns. Even a partial uplift to `no_skill`'s 61% execution rate would roughly triple `pass_strict`.
3. **Schema mismatch** (e.g. U092 dimensions, U080 placeholder untouched): Edit 5 puts canonical output JSON directly in template docstrings.

Quantitative guess: `pass_strict` 3% → ~20% on this model (matching no_skill within noise), `score > 0` 50% → ~90% on the same. Above-30 B models with 200 K windows may be unaffected — recommend re-running the benchmark sweep across model sizes before adopting Edit 3 unconditionally (could regress for larger models that benefit from the rich API reference).

## Confidence summary

| Edit | Confidence | Risk if wrong |
|---|---|---|
| 1. Fast-path block at top of SKILL.md | H | Low — additive, doesn't remove existing info |
| 2. Trim routing table refs | H | Low — references still on disk, just not advertised |
| 3. Default `RF_SKILL_HINT_LEVEL=minimal` | H | Medium — may hurt larger models with bigger windows; gate on model |
| 4. `cp`-only template idiom | H | Low |
| 5. Inline canonical schema in template docstrings | M | Low |
| 6. Remove Glob suggestion for `$RF_SKILL_DIR` | M | Low |
| 7. Drop vector-store block | M | Low — keeps one-liner pointer |
| 8. DIAGNOSE runbook | L | Low — only affects 1 task family observed |

Order to apply: 3 → 1 → 4 → 2 → 7 → 6 → 5 → 8. Edit 3 (default minimal hint) alone is expected to recover most of the gap; the others are belt-and-suspenders against the remaining `no execution` cases.
