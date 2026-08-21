"""Trial orchestration: run_one, should_retry_trial, CONTINUOUS_METRICS."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

from benchmark.trial.skeletons import pre_ship_skeleton  # noqa: F401 (re-exported)
from benchmark.trial.prompt import build_prompt
from benchmark.trial.invoke import invoke_claude, _agent_gave_up_empty
from benchmark.verifier import (
    verify,
    CheckResult,
    VerificationResult,
    extract_scalar,
    load_sim_result,
)


@dataclass(frozen=True)
class TrialConfig:
    """Knobs passed to run_one / invoke_claude. Replaces the long
    positional-arg list."""
    model: str
    max_turns: int
    timeout: int
    condition: str


# Continuous metrics extracted from simulation_result.json into result.json's
# continuous_metrics field. Each entry is (canonical_name, list_of_aliases) —
# extract_scalar tries aliases in order; first non-None hit wins.
CONTINUOUS_METRICS: list[tuple[str, list[str]]] = [
    ("ber_gap_db", ["ber_gap_db"]),
    ("ber_at_snr", ["ber_at_snr"]),
    ("nmse_db", ["nmse_db"]),
    ("doppler_hz", ["doppler_hz"]),
    ("coverage_pct", ["coverage_pct"]),
    ("path_loss_range", ["path_loss_range"]),
    ("ris_gain_db", ["ris_gain_db", "received_power_gain_db"]),
    ("noise_power_dbm", ["noise_power_dbm"]),
    ("peak_se", ["peak_se"]),
    ("snr_at_ber_1e4_db", ["snr_at_ber_1e4_db"]),
    ("nve", ["nve", "normalized_validation_error"]),
    ("map_mae_db", ["map_mae_db", "radio_map_mae", "path_loss_mae_db"]),
    ("coding_gain_db", ["coding_gain_db"]),
]


def should_retry_trial(workdir: Path) -> bool:
    """Decide whether a trial in `workdir` should be re-run with a longer
    timeout. Replaces the buggy "stderr contains [TIMEOUT]" check that
    overwrote successful trials.

    Retry only when ALL of these are true:
    1. stderr contains [TIMEOUT] (the original signal)
    2. AND result.json is missing OR score < 0.5

    A trial that produced a passing result.json should NEVER be retried,
    even if its stderr has a [TIMEOUT] marker (e.g. from a progress probe
    or a per-call deadline that the harness as a whole survived).
    """
    stderr_path = workdir / "stderr.txt"
    if not stderr_path.exists():
        return False
    if "[TIMEOUT]" not in stderr_path.read_text(errors="replace"):
        return False
    rj = workdir / "result.json"
    if not rj.exists():
        return True  # real failure — definitely retry
    try:
        score = json.loads(rj.read_text())["verification"]["score"]
    except Exception:
        return True  # malformed result, treat as failure
    return score < 0.5


def run_one(task: dict, trial: int, results_root: Path,
            config: TrialConfig) -> dict:
    task_id = task["id"]
    workdir = results_root / config.condition / task_id / f"t{trial}"
    workdir.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(task, config.condition)
    (workdir / "prompt.txt").write_text(prompt)

    ok, stdout, stderr, wall, usage = invoke_claude(
        prompt, workdir, config, task=task)

    # auto-retry hook. If the agent end_turn'd with empty
    # final text AND no real simulation_result.json (only the pre-shipped
    # placeholder), give it one more shot with stderr+nudge injected.
    # SMART GUARD: only retry if the first run ended *early* (used less
    # than half the turn budget). An agent that burned all 25 turns and
    # still produced nothing won't be saved by another 25-turn budget;
    # retrying just doubles wall time for zero gain.
    retried = False
    turns_used = usage.get("totals", {}).get("num_turns", 0)
    if (_agent_gave_up_empty(stdout, workdir)
            and 1 <= turns_used < max(2, config.max_turns // 2)):
        retried = True
        retry_prompt = (
            f"{prompt}\n\n--- HARNESS RETRY ---\n"
            f"Your previous attempt produced no real output. Stderr was:\n"
            f"{(stderr or '')[-1500:]}\n"
            f"Stdout tail:\n{(stdout or '')[-1000:]}\n"
            f"Fix the error and try again. If sionna keeps failing, write "
            f"a numpy/scipy analytical fallback (scipy.special.erfc for "
            f"BER, FSPL formula for coverage). Write real numbers to "
            f"simulation_result.json before ending."
        )
        ok2, stdout2, stderr2, wall2, usage2 = invoke_claude(
            retry_prompt, workdir, config, task=task)
        # Concatenate so analysis sees both passes; use whichever
        # exec_success is True; sum wall + turns.
        ok = ok or ok2
        stdout = (stdout or "") + "\n--- RETRY ---\n" + (stdout2 or "")
        stderr = (stderr or "") + "\n--- RETRY ---\n" + (stderr2 or "")
        wall += wall2
        # Merge usage totals
        t = usage.get("totals", {}); t2 = usage2.get("totals", {})
        for k in ("input_tokens", "output_tokens",
                  "cache_read_input_tokens", "cache_creation_input_tokens",
                  "num_turns"):
            t[k] = int(t.get(k, 0) or 0) + int(t2.get(k, 0) or 0)
        usage["totals"] = t

    # Test-split log seal: held-out test trials must NOT leave behind
    # per-trial transcripts that the agent / human iterator could read for
    # leakage. Train trials keep full logs for debugging.
    if task.get("split") == "test":
        (workdir / "stdout.txt").write_text(
            "[sealed] test-split trial — full transcript suppressed to "
            "prevent train-time leakage. Re-run with train split for full logs.\n"
        )
        (workdir / "stderr.txt").write_text("[sealed]\n")
    else:
        (workdir / "stdout.txt").write_text(stdout)
        (workdir / "stderr.txt").write_text(stderr)

    try:
        v = verify(task, workdir, exec_success=ok)
        verifier_error = None
    except Exception as e:
        import traceback
        verifier_error = f"{type(e).__name__}: {e}"
        v = VerificationResult(passed=False, score=0.0,
            checks=[CheckResult(name="verifier_internal_error",
                                passed=False, detail=verifier_error)],
            notes=[traceback.format_exc()[-500:]])

    try:
        from benchmark.trial.failure_capture import maybe_capture
        maybe_capture(task, v, workdir)
    except Exception:
        pass  # never block trial result

    continuous = {}
    try:
        sim = load_sim_result(workdir)
        if isinstance(sim, dict):
            for canonical, aliases in CONTINUOUS_METRICS:
                for alias in aliases:
                    val = extract_scalar(sim, alias, None)
                    if val is not None:
                        continuous[canonical] = val
                        break
    except Exception:
        pass  # malformed sim result — no continuous metrics, no crash

    result = {
        "task_id": task_id,
        "origin_id": task.get("origin_id", task_id),
        "tier": task["tier"],
        "capability": task["capability"],
        "difficulty": task["difficulty"],
        "harness_retried": retried,
        "condition": config.condition,
        "trial": trial,
        "model": config.model,
        "exec_success": ok,
        "wall_sec": round(wall, 2),
        "usage": usage.get("totals", {}),
        "continuous_metrics": continuous,
        "verification": v.as_dict(),
        "verifier_error": verifier_error,
        "env_snapshot": _capture_env_snapshot(),
    }
    # pass_strict: verifier must pass; AND either the agent's script exited
    # cleanly OR the verifier saw a fully complete output (score == 1.0). The
    # latter clause prevents penalizing trials where the agent produced
    # geometrically/numerically correct artifacts on disk but the wrapping
    # script hit max_turns or an end-of-script crash that didn't affect the
    # artifacts. The original strict variant remains available as
    # `pass_strict_exec` for back-compat analyses.
    _verif = result.get("verification", {})
    _exec_ok = bool(result.get("exec_success", False))
    _verif_ok = bool(_verif.get("passed", False))
    _full_score = float(_verif.get("score", 0.0)) >= 0.999
    result["pass_strict"] = bool(_verif_ok and (_exec_ok or _full_score))
    result["pass_strict_exec"] = bool(_exec_ok and _verif_ok)
    (workdir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def _capture_env_snapshot() -> dict:
    """Snapshot the runtime environment for paper-grade reproducibility.

    NOT @lru_cache'd: a long-running harness process can pip-install or
    swap deps mid-run, and we want each result.json to reflect the env at
    THAT trial's time. The snapshot is cheap (~5ms via importlib.metadata).

    Captures:
      - python version + platform + host
      - key package versions via importlib.metadata (avoids importing torch
        which would init CUDA and waste a context inside the bench worker).
        P0.7: dist-name aliases — chromadb registers as `chromadb-client`,
        sentence_transformers as `sentence-transformers`. Try both forms.
      - CUDA query gated behind BENCH_QUERY_CUDA=1 — only do it when the
        operator actually wants the GPU info captured (paper runs), since
        torch.cuda.is_available() forces driver init.
      - HF model id (from VAST_MODEL_HF or BENCH_MODEL env)
      - vLLM + openclaude versions if installed
      - harness_git_sha (best-effort, "unknown" if git not available)
      - rag_enabled boolean (computed from CLAUDE_CODE_USE_RAG env +
        chromadb + sentence_transformers presence)
    """
    import os as _os
    import platform
    import subprocess as _sp
    try:
        from importlib.metadata import version as _pkg_version
    except ImportError:
        from importlib_metadata import version as _pkg_version  # type: ignore
    snap: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "host": platform.node(),
        "hf_model_id": _os.environ.get("VAST_MODEL_HF",
                                       _os.environ.get("BENCH_MODEL", "")),
    }
    # Map (display_name, dist_name) — dist_name is what pip/PyPI registers.
    # P0.7: try several name forms because the package's pip dist name may
    # differ from its import name (sentence_transformers vs
    # sentence-transformers) AND from server vs client splits (chromadb vs
    # chromadb-client).
    pkg_map = [
        ("sionna", "sionna"),
        ("chromadb", "chromadb-client"),
        ("sentence_transformers", "sentence-transformers"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("mitsuba", "mitsuba"),
        ("drjit", "drjit"),
        ("trimesh", "trimesh"),
        ("shapely", "shapely"),
        ("pydantic", "pydantic"),
        ("vllm", "vllm"),
    ]
    for display, dist in pkg_map:
        v = "missing"
        for cand in (dist, display, dist.replace("-", "_"),
                     display.replace("_", "-")):
            try:
                v = _pkg_version(cand)
                break
            except Exception:
                continue
        snap[display] = v

    # rag_enabled: precondition flags (env var + dep presence). Cheaper than
    # actually probing the chroma server, and tells reviewers at-a-glance
    # whether RAG was operational for this trial.
    snap["rag_enabled"] = bool(
        _os.environ.get("CLAUDE_CODE_USE_RAG") == "1"
        and snap.get("chromadb") not in ("missing", "")
        and snap.get("sentence_transformers") not in ("missing", "")
    )

    # harness_git_sha: best-effort, run from the repo root if possible.
    try:
        repo_root = Path(__file__).resolve().parents[2]
        sha = _sp.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=_sp.DEVNULL, timeout=2,
        ).decode().strip()
        snap["harness_git_sha"] = sha
    except Exception:
        snap["harness_git_sha"] = "unknown"

    # openclaude version (CLI binary). Falls back to "unknown" if not on PATH.
    try:
        snap["openclaude_version"] = _sp.check_output(
            ["openclaude", "--version"], stderr=_sp.DEVNULL, timeout=3,
        ).decode().splitlines()[0].strip()
    except Exception:
        snap["openclaude_version"] = "unknown"

    # CUDA query is opt-in to avoid forcing torch import + CUDA driver init
    # in workers that don't otherwise need it. Set BENCH_QUERY_CUDA=1 in the
    # bench tmux session env to capture; default off.
    if _os.environ.get("BENCH_QUERY_CUDA") == "1":
        try:
            import torch  # noqa: PLC0415
            snap["cuda_available"] = bool(torch.cuda.is_available())
            snap["cuda_device_count"] = int(torch.cuda.device_count())
            if snap["cuda_available"]:
                snap["gpu_model"] = torch.cuda.get_device_name(0)
        except Exception:
            snap["cuda_available"] = False
            snap["cuda_device_count"] = 0
    return snap
