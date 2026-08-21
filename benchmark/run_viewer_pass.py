"""Second-pass viewer-generation benchmark.

Walks a base benchmark's results dir, finds passing trials whose tier is
visualization-relevant, and runs a `viewer.html` generation trial against
each one. Output goes to a sibling `<base-label>_viewer/` directory mirroring
the original layout (chunk/condition/task/t1/...).

Key design points:
- Reuses the existing trial.invoke / trial.run machinery (same Claude harness
  invocation, same env, same tool override) — only difference is the prompt
  and the verifier.
- Skips trials that didn't produce real sim outputs (placeholder JSON →
  nothing useful to visualize).
- Idempotent: existing viewer.html outputs are skipped on rerun.
- Runs in parallel via the same multiprocessing pool pattern as run_benchmark.

Usage:
    python3 benchmark/run_viewer_pass.py \\
        --base-results-dir benchmark/results \\
        --base-label-prefix train_Qwen_Qwen3_6_27B \\
        --model benchmark-model \\
        --workers 4 \\
        [--dry-run]   # list what WOULD run without invoking agent

Output: benchmark/results/<base-label>_chunkN_viewer/<condition>/<task>/t1/{
    viewer.html, prompt.txt, stdout.txt, stderr.txt, result.json}
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.trial.run import TrialConfig  # noqa: E402
from benchmark.trial.invoke import invoke_claude  # noqa: E402

# Tiers we run viewer-gen on (visualization-relevant).
_VIEWER_TIERS = {
    "T0_scene_gen",
    "T2_ray_tracing",
    "T2_channel_modeling",
    "T4_system_level",
}

# Sim-result status that indicates the base trial didn't actually produce
# real output — skip these (no point visualizing a placeholder).
_PLACEHOLDER_STATUS = "placeholder_pre_shipped_by_harness"


@dataclass(frozen=True)
class ViewerCandidate:
    base_workdir: Path     # original trial workdir (with sim_result + artifacts)
    base_label: str        # e.g. "train_Qwen_Qwen3_6_27B_chunk0"
    condition: str         # with_skill | no_skill | self_gen
    task_id: str           # U001 etc.
    base_task: dict        # task definition from tasks.json
    base_result: dict      # parsed result.json


def _load_tasks(repo_root: Path) -> dict[str, dict]:
    tj = repo_root / "benchmark" / "tasks" / "tasks.json"
    return {t["id"]: t for t in json.loads(tj.read_text())["tasks"]}


def find_viewer_candidates(
    base_results_dir: Path,
    label_prefix: str,
    tasks: dict[str, dict],
) -> list[ViewerCandidate]:
    """Walk results dir, return passing trials eligible for viewer-gen."""
    out: list[ViewerCandidate] = []
    for label_dir in sorted(base_results_dir.glob(f"{label_prefix}*")):
        # Skip already-done viewer dirs (suffix _viewer)
        if label_dir.name.endswith("_viewer"):
            continue
        for result_path in sorted(label_dir.glob("*/*/t*/result.json")):
            try:
                rj = json.loads(result_path.read_text())
            except Exception:
                continue
            if not rj.get("verification", {}).get("passed", False):
                continue
            task_id = result_path.parent.parent.name
            condition = result_path.parent.parent.parent.name
            task = tasks.get(task_id)
            if task is None or task.get("tier") not in _VIEWER_TIERS:
                continue
            sim_path = result_path.parent / "simulation_result.json"
            if sim_path.exists():
                try:
                    sj = json.loads(sim_path.read_text())
                    if sj.get("status") == _PLACEHOLDER_STATUS:
                        continue
                except Exception:
                    pass
            out.append(ViewerCandidate(
                base_workdir=result_path.parent,
                base_label=label_dir.name,
                condition=condition,
                task_id=task_id,
                base_task=task,
                base_result=rj,
            ))
    return out


def _viewer_workdir(base_results_dir: Path, cand: ViewerCandidate) -> Path:
    """Map a base trial workdir → viewer-pass workdir, preserving structure."""
    return (base_results_dir / f"{cand.base_label}_viewer" /
            cand.condition / cand.task_id / "t1")


def _build_viewer_prompt(cand: ViewerCandidate) -> str:
    """Render the viewer-gen prompt for one candidate."""
    base_dir = cand.base_workdir
    extras = []
    for f in sorted(base_dir.glob("*")):
        if f.name in ("result.json", "stdout.txt", "stderr.txt",
                      "prompt.txt", "viewer.html", "simulation_result.json"):
            continue
        if f.is_file():
            extras.append(f.name)
    artifacts_str = ", ".join(extras) if extras else "(no extra artifacts)"
    base_summary = (
        f"Tier: {cand.base_task.get('tier','?')}; "
        f"capability: {cand.base_task.get('capability','?')}; "
        f"original task: {cand.base_task.get('prompt','')[:300]}"
    )
    return (
        f"You are generating a viewer.html for an existing simulation. The "
        f"current directory contains a completed simulation:\n"
        f"  - simulation_result.json (contains the canonical metrics)\n"
        f"  - {artifacts_str}\n\n"
        f"Follow `$RF_SKILL_DIR/references/viewer-spec.md` to produce a "
        f"`viewer.html` in the current directory that:\n"
        f"  1. Renders the scene / data using HTML5 canvas + Plotly + Three.js "
        f"(CDN imports, no build step)\n"
        f"  2. Loads the simulation data via fetch('./simulation_result.json') "
        f"and any artifact files\n"
        f"  3. Has the dashboard theme (dark-mode CSS variables per "
        f"viewer-spec.md)\n"
        f"  4. NO chatbox (per viewer-spec.md `Chatbox policy` — no "
        f"ANTHROPIC_API_KEY in deployment)\n\n"
        f"Commit only viewer.html; do not modify simulation_result.json.\n\n"
        f"Task context for the original sim:\n{base_summary}\n"
    )


def _verify_viewer(viewer_path: Path) -> tuple[bool, list[str]]:
    """Lightweight viewer.html verifier. Returns (passed, failures)."""
    failures: list[str] = []
    if not viewer_path.exists():
        return False, ["viewer.html not produced"]
    text = viewer_path.read_text(errors="replace")
    if "<canvas" not in text and "id=\"plot" not in text and "id=\"scene-canvas" not in text:
        failures.append("no <canvas> or plot div found")
    if "<script" not in text:
        failures.append("no <script> tag")
    if "simulation_result.json" not in text:
        failures.append(
            "no fetch reference to simulation_result.json (data not bound)")
    # Chatbox forbidden in viewer-pass output.
    bad_markers = ["chat-input", "chat-messages", "/api/chat"]
    for m in bad_markers:
        if m in text:
            failures.append(
                f"chatbox marker '{m}' present (forbidden in viewer-pass)")
            break
    return len(failures) == 0, failures


def run_one_viewer(
    cand: ViewerCandidate,
    base_results_dir: Path,
    config: TrialConfig,
    dry_run: bool,
) -> dict:
    """Generate a viewer for one candidate. Returns the output result dict."""
    workdir = _viewer_workdir(base_results_dir, cand)
    workdir.mkdir(parents=True, exist_ok=True)
    # Copy the base sim outputs (BOTH top-level AND nested) into the viewer
    # workdir so the agent has them in cwd. Some trials produce nested
    # artifacts like outputs/coverage_map.npy or meshes/scene.glb — those
    # MUST come along or the viewer agent has nothing to render.
    import shutil
    _BLOCKLIST = {"stdout.txt", "stderr.txt", "result.json",
                  "prompt.txt", "viewer.html"}
    for src in cand.base_workdir.iterdir():
        if src.name in _BLOCKLIST:
            continue
        dest = workdir / src.name
        if dest.exists():
            continue
        try:
            if src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)
            elif src.is_file():
                shutil.copy2(src, dest)
        except Exception as e:
            # Log + continue — losing one artifact is better than aborting
            # the whole viewer-pass for a single trial.
            print(f"  ! could not copy {src} → {dest}: {e}",
                  file=sys.stderr)
    viewer_path = workdir / "viewer.html"
    # Idempotent skip
    if viewer_path.exists() and (workdir / "result.json").exists():
        return json.loads((workdir / "result.json").read_text())

    prompt = _build_viewer_prompt(cand)
    (workdir / "prompt.txt").write_text(prompt)

    if dry_run:
        passed, failures = False, ["dry-run"]
        result = {
            "task_id": cand.task_id,
            "condition": cand.condition,
            "base_label": cand.base_label,
            "viewer_passed": passed,
            "viewer_failures": failures,
            "wall_sec": 0.0,
            "exec_success": False,
            "dry_run": True,
        }
        (workdir / "result.json").write_text(json.dumps(result, indent=2))
        return result

    ok, stdout, stderr, wall, usage = invoke_claude(
        prompt=prompt, workdir=workdir, config=config, task=None)
    (workdir / "stdout.txt").write_text(stdout or "")
    (workdir / "stderr.txt").write_text(stderr or "")
    passed, failures = _verify_viewer(viewer_path)
    result = {
        "task_id": cand.task_id,
        "condition": cand.condition,
        "base_label": cand.base_label,
        "viewer_passed": passed,
        "viewer_failures": failures,
        "wall_sec": round(wall, 2),
        "exec_success": ok,
        "usage": usage.get("totals", {}),
    }
    (workdir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-results-dir", required=True,
                    help="Root of base benchmark results (contains "
                         "<label>/<condition>/<task>/t*/result.json)")
    ap.add_argument("--base-label-prefix", required=True,
                    help="Glob prefix for base labels (e.g. train_Qwen_Qwen3_6_27B)")
    ap.add_argument("--model", default="benchmark-model")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=400,
                    help="Per-trial wall timeout (s); viewer-gen is "
                         "lighter than sim, so default is shorter")
    ap.add_argument("--workers", type=int, default=2,
                    help="Parallel viewer-gen workers (each starts its own "
                         "Claude subprocess)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List candidates + write empty result.json without "
                         "invoking Claude")
    args = ap.parse_args()

    base_results_dir = Path(args.base_results_dir).resolve()
    tasks = _load_tasks(ROOT)
    candidates = find_viewer_candidates(
        base_results_dir, args.base_label_prefix, tasks)
    print(f"Found {len(candidates)} viewer candidates "
          f"under {args.base_label_prefix}*")
    if not candidates:
        return 0

    # We DON'T spawn a multi-worker pool here because each Claude trial
    # already runs serially within itself; the harness's existing pool
    # pattern is overkill for a typical viewer-pass (<200 trials).
    # If you need parallelism, wrap this loop in concurrent.futures.
    n_pass = 0
    for i, cand in enumerate(candidates, 1):
        cfg = TrialConfig(
            condition=cand.condition,
            model=args.model,
            max_turns=args.max_turns,
            timeout=args.timeout,
        )
        result = run_one_viewer(cand, base_results_dir, cfg, args.dry_run)
        if result.get("viewer_passed"):
            n_pass += 1
        status = "PASS" if result["viewer_passed"] else "FAIL"
        fails = ",".join(result.get("viewer_failures") or [])
        print(f"  [{i}/{len(candidates)}] {cand.base_label} "
              f"{cand.condition}/{cand.task_id} → {status} "
              f"({fails or 'all checks ok'})")

    print(f"\nViewer-pass complete: {n_pass}/{len(candidates)} "
          f"({100*n_pass/len(candidates):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
