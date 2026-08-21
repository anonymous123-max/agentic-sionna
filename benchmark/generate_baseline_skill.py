"""Generate a self-authored SKILL.md for the Sionna/RF domain.

SkillsBench-style 2-stage self_gen protocol:
  Stage 1 (this script): ask the model to produce a SKILL.md for the domain,
          given ONLY domain-level context (no specific test tasks). Save to
          benchmark/self_gen_skill/SKILL.md (or per-model
          benchmark/self_gen_skill/<model_basename>/SKILL.md).
  Stage 2 (trial.py): every self_gen trial runs in a tmpdir whose
          .claude/skills/sionna-self-generated/SKILL.md is a copy of this
          stage-1 output. Claude Code's skill discovery then auto-loads it
          exactly like it does for the curated rf-simulator skill.

This mirrors the test of whether the model can generate useful procedural
knowledge on its own, with no leakage from (a) the curated skill or (b)
per-task information.

Run once before the benchmark:
    # Anthropic CLI backend (default — uses `claude` subprocess)
    python benchmark/generate_baseline_skill.py --model sonnet
    # Writes benchmark/self_gen_skill/SKILL.md

    # OpenAI-compatible backend (vLLM via tunnel/proxy on vast.ai)
    OPENAI_BASE_URL=http://127.0.0.1:8101/v1 \\
        python benchmark/generate_baseline_skill.py \\
        --backend openai-compat \\
        --out-file benchmark/self_gen_skill/Qwen3.6-27B/SKILL.md

Re-run with --force to regenerate (e.g., after changing the prompt).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmark/self_gen_skill"
OUT_FILE = OUT_DIR / "SKILL.md"
CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")


# Prompt the model to produce a SKILL.md given ONLY the Sionna/RF domain.
# Deliberately does NOT reference any specific benchmark task, parameter
# value, or verifier field. Provides only the capability scope.
DOMAIN_PROMPT = """You are generating a Claude Code skill file (SKILL.md) for
an agent that will complete NVIDIA Sionna wireless-simulation tasks. Write
the skill as general-purpose procedural knowledge the agent can use across
any task in this domain. DO NOT reference specific test tasks, scenario
parameters, or evaluation metrics — write only general guidance.

Domain capabilities the agent will face:
- PHY link-level: BER/BLER loops (AWGN, fading), channel coding (LDPC,
  Polar), OFDM (pilots, cyclic prefix, channel estimation, equalization),
  modulation (BPSK/QPSK/M-QAM), MIMO (point-to-point, multi-user, massive,
  precoding, stream management)
- Channel models: CDL, TDL, UMi, UMa, RMa, Rayleigh/Rician
- Ray tracing: scene loading, TX/RX placement, path solving, CIR/CFR
  extraction, radio maps, differentiable optimization
- Neural receivers: neural demappers, neural channel estimators,
  end-to-end autoencoder communication systems, unfolded decoders
- System level: multi-cell layouts, scheduling, link adaptation, power
  control, PHY abstraction
- Emerging: OTFS, near-field XL-MIMO, RIS / STAR-RIS, ISAC, THz, semantic
  communication, federated learning, channel prediction

Your output MUST follow this exact format:

---
name: sionna-self-generated
description: <one-line trigger description>
---

# Sionna RF Skill (self-generated)

<body sections covering procedural knowledge, code patterns, output schemas,
common pitfalls, and verification checks>

Output ONLY the SKILL.md content — no preamble, no explanation, no code
fences around the whole thing. Begin directly with the YAML frontmatter."""


def generate_skill_anthropic(model: str) -> str:
    """Invoke the `claude` CLI once and capture its SKILL.md output."""
    cmd = [CLAUDE_BIN, "-p", DOMAIN_PROMPT,
           "--model", model,
           "--max-turns", "5",
           "--permission-mode", "bypassPermissions",
           "--output-format", "text"]
    # Run from /tmp so the curated rf-simulator skill does NOT auto-load
    # during generation (would contaminate the "self-generated" claim).
    import tempfile
    with tempfile.TemporaryDirectory(prefix="self_gen_author_") as tmp:
        proc = subprocess.run(cmd, cwd=tmp, timeout=300,
                               capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"claude exited {proc.returncode}: {proc.stderr[-500:]}")
    return proc.stdout


def generate_skill_openai_compat(model_id: str | None = None) -> str:
    """Invoke an OpenAI-compatible endpoint (e.g. local vLLM via
    OPENAI_BASE_URL) once and return the SKILL.md content. Used to make
    each model under test author its OWN self_gen skill rather than
    sharing a Sonnet-generated one."""
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "openai-compat backend requires the `openai` Python SDK. "
            "Install with: pip install openai") from e

    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        raise SystemExit(
            "openai-compat backend requires OPENAI_BASE_URL "
            "(e.g. http://127.0.0.1:8101/v1).")

    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("OPENAI_API_KEY", "dummy-vllm"),
    )
    # Resolve served-model name: explicit arg > BENCHMARK_MODEL_ID env >
    # MODEL_HF env (the harness's source of truth). The literal string
    # "benchmark-model" is never served by vLLM and causes NotFoundError.
    model_id = (model_id
                or os.environ.get("BENCHMARK_MODEL_ID")
                or os.environ.get("MODEL_HF"))
    if not model_id:
        raise SystemExit(
            "openai-compat backend needs a served-model name. Pass --model "
            "or set MODEL_HF / BENCHMARK_MODEL_ID.")
    print(f"  openai-compat backend: base_url={base_url} model={model_id}",
          file=sys.stderr)
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": DOMAIN_PROMPT}],
        max_tokens=8000,
        timeout=300,  # generation can take several min on a 27B model
    )
    return resp.choices[0].message.content or ""


def main():
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--backend", choices=["anthropic", "openai-compat"],
                    default="anthropic",
                    help="Generation backend: anthropic (claude CLI, default) "
                         "or openai-compat (OPENAI_BASE_URL, e.g. local vLLM).")
    ap.add_argument("--model", default="sonnet",
                    help="Model for skill generation. anthropic backend: "
                         "Claude alias (sonnet/opus). openai-compat backend: "
                         "served-model id (e.g. MODEL_HF); falls back to env "
                         "BENCHMARK_MODEL_ID or MODEL_HF.")
    ap.add_argument("--out-file", default=None,
                    help="Output path for SKILL.md (default: "
                         "benchmark/self_gen_skill/SKILL.md).")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if SKILL.md already exists")
    args = ap.parse_args()

    out_file = Path(args.out_file).resolve() if args.out_file else OUT_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists() and not args.force:
        print(f"Already exists: {out_file} (use --force to regenerate)")
        return 0

    if args.backend == "anthropic":
        print(f"Generating self-authored skill (anthropic, model={args.model})...")
        skill_md = generate_skill_anthropic(args.model)
    else:
        print("Generating self-authored skill (openai-compat)...")
        # For openai-compat, --model holds the served model id (e.g. MODEL_HF).
        skill_md = generate_skill_openai_compat(args.model)

    # Basic sanity: should start with YAML frontmatter (regardless of backend)
    if not skill_md.lstrip().startswith("---"):
        # Try to salvage — maybe model wrapped in ``` fences
        lines = skill_md.splitlines()
        first_yaml = next((i for i, l in enumerate(lines)
                           if l.strip() == "---"), None)
        if first_yaml is not None:
            skill_md = "\n".join(lines[first_yaml:])
        else:
            print("WARNING: model output doesn't start with YAML frontmatter.",
                  file=sys.stderr)

    out_file.write_text(skill_md)
    print(f"Wrote {out_file} ({len(skill_md)} chars, "
          f"{len(skill_md.splitlines())} lines)")
    print()
    print("First 400 chars:")
    print(skill_md[:400])
    return 0


if __name__ == "__main__":
    sys.exit(main())
