# Local-LLM Cycle Results — 2026-05-03

Three cycles run on sunlab (RTX 5090, 32 GB VRAM) over 8 autonomous iterations. Each cycle ran the same 4-task smoke set (U001 BER, U079 scene_gen, U019 RT placement, U061 indoor coverage) on three local LLMs via vLLM + tool-call proxy + openclaude.

## Cycle pass rates

| Cycle | Patches active | 8B | Qwen-27B | Gemma-31B AWQ |
|---|---|---|---|---|
| **cycle1** | none (baseline) | 1/4 (25%) | 1/4 (25%) | 4/7 with retry (57%) |
| **cycle2** | skill_hint v1 + sionna PATH | 1/4 (25%) | 1/4 (25%) | 2/6 with retry (33%) |
| **cycle3** | v2 prompt + Write+Edit + 32k 8B context, no retry | 0/4 (0%) — regressed | **2/4 (50%) — DOUBLED** | (in flight) |

The pass rates are misleading on their own — see "behavior wins" below.

### Cycle3 detail (partial — Gemma still in flight)

- **Qwen-27B U001 PASS (1.0, 55.9s, 14 turns):** ran `ls $RF_SKILL_DIR/scripts/run_ber_analytical.py` then executed it, `ber_gap_db: 0`. The skill_hint v2 + Write tool + sionna PATH combination works exactly as designed for this model class. **First time a small LLM cleanly used the skill's runnable script.**
- **Qwen-27B U079 PASS (1.0, 113s).**
- **8B regressed to 0/4** because the Write tool re-enable confuses its parser. **Per-model `CLAUDE_CODE_TOOLS_OVERRIDE` is needed** — 8B should stay on `Bash,Read`; Qwen-27B and Gemma-31B can use `Bash,Read,Write,Edit`. This is the next small fix to apply for cycle4.

## Behavior wins (not yet score wins)

The skill_hint patch (commit `d87900b` then `f2c63bb`) caused dramatically different agent behavior even when scores stayed identical:

- **8B before**: hallucinated `python -m sionna.simulation.uncoded_qpsk_awgn` (non-existent module) repeatedly, never wrote a Python file
- **8B after**: explicitly tried `python3 $RF_SKILL_DIR/scripts/run_ber_analytical.py` (mis-parsed inline comment as args, fixed in v2)
- **Qwen-27B before**: `import sionna` failed → gave up
- **Qwen-27B after**: read SKILL.md + template_ber.py, recognized BER_ANALYSIS task, attempted `Write` (tool was disabled in cycle1/2)
- **Gemma-31B**: wrote real `from sionna.rt import load_scene; scene.compute_paths(...)` for U019 — code_contains check passed (vs cycle1's 0.0). Just timed out on the Munich scene at 1200s.

## What worked

1. **`tool_call_proxy.py` bool coercion** (commit `fb072d8`): Llama-3.1-8B repeatedly emitted `run_in_background: "True"` (string) which tripped the Bash validator. Coercing string→bool eliminated 60+ retry loops on cycle1 8B U061.
2. **`trial.py` skill_hint with absolute paths** (commits `d87900b`, `f2c63bb`): all three model classes shifted from "hallucinate APIs" to "look at skill files." This is the highest-leverage change shipped.
3. **`/home/myid/js66916/miniconda3/envs/sionna/` in PATH for the agent** (cycle2/3 scripts only — needs to be baked into queue_local_llms.sh): without it, `import sionna` fails and analytical-only is forced. With it, Gemma actually wrote and tried to run real Sionna RT code.

## What regressed (with the v1 patch)

1. **Gemma U061 went 1.0 → 0.43.** The skill_hint v1 listed `run_ber_analytical.py` and templates without distinguishing runnable scripts from copy-only template bodies. Gemma scanned the list, didn't find a coverage script, gave up — vs cycle1 where it just wrote FSPL+erfc from scratch. The v2 patch (`f2c63bb`) split RUNNABLE vs COPY headers and added an explicit "Quickest path" line. Untested in cycle3 yet.
2. **Qwen-27B context overflow at 24576 tokens.** With skill_hint, Qwen reads SKILL.md (~4.4K tokens) + template_ber.py and reaches input ~20K, leaving only ~4K for output. Triggered `VLLMValidationError`. Cycle3 bumps 8B to 32k; Qwen+Gemma kept at 24k due to GPU memory.

## What didn't help (yet)

1. **Score parity on 8B + Qwen across cycle1 and cycle2.** Both 1/4 in both cycles. Behavior changed dramatically but the runs failed in different ways (8B mis-parsed args; Qwen couldn't Write). Cycle3 should clarify: Write tool re-enabled means Qwen can now save files; format fixes mean 8B should run scripts cleanly.

## Open infrastructure issues

1. **`with_skill` doesn't auto-inject SKILL.md** — Sonnet/Opus invoke skills via slash commands; small LLMs don't. The skill_hint patch is a workaround; a proper fix would be a system-message prepend in `invoke_claude()` for the with_skill condition.
2. **Verifier retry pass overwrites successful trials when stderr has `[TIMEOUT]`** (Gemma U061 cycle1 went pass→fail). `trial.py` retry condition should be "result.json missing or score==0", not "[TIMEOUT] in stderr." Suppressed in cycle3 by dropping `--retry-timeout`.
3. **`code_contains` checks score 0 when no .py file exists.** Could fall back to scanning shell heredoc commands. Cosmetic; not blocking.
4. **`CLAUDE_CODE_TOOLS_OVERRIDE=Bash,Read` was a v1.4-era workaround for Write parser bugs** that newer models (Qwen-27B, Gemma-31B) don't have. Cycle3 re-enables Write+Edit; if 8B also tolerates it now, we can drop the override entirely.
5. **`[context] Warning: model not in context window table`** floods stderr 50+ times per trial — openclaude harness issue, not skill. Add the local-LLM model names to `openaiContextWindows.ts` upstream.

## Recommendations to user (queued from memory)

1. **Bake `FURNITURE_CATALOG_PATH` and the sionna env on PATH into `queue_local_llms.sh`** so they survive script regeneration. The v2.8.3 plan does this via `trial.py`.
2. **Fix the retry-pass overwrite bug** in trial.py before running paper-quality numbers.
3. **Decide: prompt-level skill injection or stay with file-pointer skill_hint?** Prepending SKILL.md (~4.4K tokens) into every `with_skill` system message guarantees small LLMs see it. Costs context budget. Tradeoff judgment.
4. **Add local-LLM names to openaiContextWindows.ts** (cosmetic but reduces stderr noise during debugging).
