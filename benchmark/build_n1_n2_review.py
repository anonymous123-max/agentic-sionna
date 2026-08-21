"""Export N1 + N2 trials into the review dataset for advisor inspection.

Output layout:
  benchmark/_review_dataset/
    N1/
      README.md                    # task description + headline table
      box_two_screens/
        README.md                  # scene + AP + freq + verifier config
        reference_coverage.npy     # ground truth (5 GHz)
        reference_coverage.png
        with_skill/t1..t5/
        no_skill/t1..t5/
        self_gen/t1..t5/
      box_one_screen/ ...
      simple_street_canyon/ ...
      etoile/ ...
    N2/
      README.md
      box_two_screens/
        README.md
        reference_5ghz.npy / .png  # 5 GHz GT
        reference_2ghz.npy / .png  # 2.4 GHz GT
        with_skill/t1..t5/ ...
      box_one_screen/ ...
      simple_street_canyon/ ...
      etoile/ ...

Per trial we copy: simulation.py, coverage_map.npy/.png (N1) or
coverage_{5ghz,2ghz,delta}* (N2), simulation_result.json, prompt.txt,
result.json. We also generate a verifier_report.txt summarizing the
3-layer verdict in human-readable form.

Run from repo root:
    python3 benchmark/build_n1_n2_review.py
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / "benchmark" / "_review_dataset"
RUNS = {
    "N1": {
        "results_dir": ROOT / "benchmark" / "results" / "n1_full_v10",
        "tasks_file": ROOT / "benchmark" / "tasks" / "_sources" / "n1_coverage.json",
        "scenes": ["box_two_screens", "box_one_screen",
                   "simple_street_canyon", "etoile"],
        "title": "Single-AP Coverage on Sionna built-in scenes",
        "oracle_dir": ROOT / "benchmark" / "oracles" / "n1",
        "trial_files": [
            "simulation.py", "coverage_map.npy", "coverage_map.png",
            "simulation_result.json", "prompt.txt", "result.json",
        ],
        "oracle_files_for_scene": lambda scene: [
            (f"{scene}.npy", "reference_coverage.npy"),
            (f"{scene}.png", "reference_coverage.png"),
        ],
    },
    "N2": {
        "results_dir": ROOT / "benchmark" / "results" / "n2_full_v10",
        "tasks_file": ROOT / "benchmark" / "tasks" / "_sources" / "n2_freq_edit.json",
        "scenes": ["box_two_screens", "box_one_screen",
                   "simple_street_canyon", "etoile"],
        "title": "Frequency-edit + recompute (5 → 2.4 GHz)",
        "oracle_dir": None,    # uses both n1 and n2_freq oracles
        "trial_files": [
            "simulation.py", "coverage_5ghz.npy", "coverage_5ghz.png",
            "coverage_2ghz.npy", "coverage_2ghz.png", "coverage_delta.png",
            "simulation_result.json", "prompt.txt", "result.json",
        ],
        "oracle_files_for_scene": lambda scene: [
            (ROOT / "benchmark" / "oracles" / "n1" / f"{scene}.npy",
             "reference_5ghz.npy"),
            (ROOT / "benchmark" / "oracles" / "n1" / f"{scene}.png",
             "reference_5ghz.png"),
            (ROOT / "benchmark" / "oracles" / "n2_freq" / f"{scene}_2ghz.npy",
             "reference_2ghz.npy"),
            (ROOT / "benchmark" / "oracles" / "n2_freq" / f"{scene}_2ghz.png",
             "reference_2ghz.png"),
        ],
    },
}
CONDS = ["with_skill", "no_skill", "self_gen"]


def render_verifier_report(result: dict) -> str:
    """Human-readable summary of result.json's verification block."""
    out = []
    v = result.get("verification", {})
    overall = "PASS" if v.get("passed") else "FAIL"
    out.append(f"VERDICT: {overall}   (score {v.get('score', 0.0):.3f})")
    out.append(f"task_id   : {result.get('task_id')}")
    out.append(f"condition : {result.get('condition')}")
    out.append(f"trial     : t{result.get('trial')}")
    out.append(f"model     : {result.get('model')}")
    out.append(f"exec_ok   : {result.get('exec_success')}")
    wall = result.get("wall_sec")
    if wall is not None:
        out.append(f"wall_sec  : {wall:.1f}")
    out.append("")
    out.append("Checks (deduped):")
    seen = set()
    for c in v.get("checks", []):
        nm = c.get("name", "?")
        if nm in seen:
            continue
        seen.add(nm)
        p = "PASS" if c.get("passed") else "FAIL"
        det = (c.get("detail", "") or "").strip()
        out.append(f"  [{p}]  {nm}")
        if det:
            # Wrap long details
            for line in det.split("\n"):
                line = line[:160]
                out.append(f"           {line}")
    out.append("")
    out.append("---")
    out.append("How to interpret:")
    out.append("  Layer A (artifact:*)   — required output files exist and parse")
    out.append("  Layer B (sionna_rt_used) — agent really invoked Sionna RT, not FSPL fallback")
    out.append("  Layer C (n1_ref_oracle / n2_freq_oracle) — agent's coverage map matches")
    out.append("           the precomputed Sionna-RT ground truth (cell-wise MAE ≤ 3 dB,")
    out.append("           or distribution-level fallback when grid extent differs)")
    return "\n".join(out)


def export_task(name: str, cfg: dict, tasks_by_id: dict) -> None:
    out_root = REVIEW / name
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)
    print(f"\n=== {name}: {cfg['title']} ===")
    print(f"  output: {out_root}")

    headline = defaultdict(lambda: defaultdict(list))   # cond -> scene -> [pass, ...]

    for scene in cfg["scenes"]:
        scene_dir = out_root / scene
        scene_dir.mkdir(parents=True)

        # Look up task spec (used in README)
        task_id = f"{name}_{scene}"
        task = tasks_by_id.get(task_id)
        if task is None:
            print(f"  [warn] no task spec for {task_id}")
            continue

        # Copy reference oracle(s) for this scene
        for src_arg, dst_name in cfg["oracle_files_for_scene"](scene):
            src = (cfg["oracle_dir"] / src_arg) if isinstance(src_arg, str) else Path(src_arg)
            if src.exists():
                shutil.copy(src, scene_dir / dst_name)

        # Scene README
        readme = _render_scene_readme(name, scene, task)
        (scene_dir / "README.md").write_text(readme)

        for cond in CONDS:
            for trial in range(1, 6):
                src_trial = cfg["results_dir"] / cond / task_id / f"t{trial}"
                if not src_trial.exists():
                    continue
                dst_trial = scene_dir / cond / f"t{trial}"
                dst_trial.mkdir(parents=True, exist_ok=True)
                for fname in cfg["trial_files"]:
                    src = src_trial / fname
                    if src.exists():
                        shutil.copy(src, dst_trial / fname)
                # Generate verifier_report.txt
                result_path = dst_trial / "result.json"
                if result_path.exists():
                    try:
                        r = json.loads(result_path.read_text())
                        (dst_trial / "verifier_report.txt").write_text(
                            render_verifier_report(r))
                        headline[cond][scene].append(bool(
                            r.get("verification", {}).get("passed", False)))
                    except Exception as e:
                        print(f"  [warn] could not parse {result_path}: {e}")

        scene_pass = {
            cond: f"{sum(headline[cond][scene])}/{len(headline[cond][scene])}"
            for cond in CONDS
        }
        print(f"  {scene:25s}  with_skill={scene_pass['with_skill']:>4s}  "
              f"no_skill={scene_pass['no_skill']:>4s}  "
              f"self_gen={scene_pass['self_gen']:>4s}")

    (out_root / "README.md").write_text(_render_task_readme(name, cfg, headline))


def _render_scene_readme(task_name: str, scene: str, task: dict) -> str:
    lines = [f"# {task_name} on `{scene}`", ""]
    lines.append(f"**Task** — {task.get('capability', '')}  "
                 f"(difficulty: {task.get('difficulty', '?')})")
    lines.append("")
    lines.append("## Fixed parameters (same across all conditions)")
    lines.append("")
    for k in ("scene_name", "ap_position", "frequency_hz", "freq_before_hz",
              "freq_after_hz", "tx_power_dbm", "cell_size_m", "max_depth",
              "samples_per_tx"):
        if k in task:
            lines.append(f"- `{k}` = `{task[k]}`")
    lines.append("")
    lines.append("## Reference oracle")
    if task_name == "N1":
        lines.append("- `reference_coverage.npy` — Sionna RT coverage at the prescribed frequency")
        lines.append("- `reference_coverage.png` — heatmap of the reference")
    elif task_name == "N2":
        lines.append("- `reference_5ghz.npy` / `.png` — Sionna RT at 5 GHz (baseline)")
        lines.append("- `reference_2ghz.npy` / `.png` — Sionna RT at 2.4 GHz (after frequency edit)")
        lines.append("")
        lines.append("Theoretical mean delta (FSPL): +6.4 dB. Our reference produces +5.97 to +7.72 dB across scenes.")
    lines.append("")
    lines.append("## Per-trial folders")
    lines.append("")
    lines.append("Each `{condition}/t{1..5}/` contains:")
    lines.append("- `simulation.py` — agent-written code (the artifact to review)")
    lines.append("- `coverage_*.npy/.png` — agent's coverage outputs")
    lines.append("- `simulation_result.json` — agent's reported metrics")
    lines.append("- `prompt.txt` — exact prompt the agent received")
    lines.append("- `result.json` — full verifier output")
    lines.append("- `verifier_report.txt` — human-readable 3-layer verdict")
    lines.append("")
    lines.append("## Prompt given to the agent")
    lines.append("")
    lines.append("```")
    lines.append(task.get("prompt", "").strip())
    lines.append("```")
    return "\n".join(lines)


def _render_task_readme(task_name: str, cfg: dict, headline: dict) -> str:
    lines = [f"# {task_name} — {cfg['title']}", ""]
    lines.append(f"4 scenes × 3 conditions × 5 trials = 60 trials per task.")
    lines.append("")
    lines.append("## Headline pass rates")
    lines.append("")
    lines.append("| Scene | with_skill | no_skill | self_gen |")
    lines.append("|---|---|---|---|")
    totals = defaultdict(lambda: [0, 0])
    for scene in cfg["scenes"]:
        row = [f"`{scene}`"]
        for cond in CONDS:
            r = headline.get(cond, {}).get(scene, [])
            n_pass = sum(r); n_tot = len(r)
            totals[cond][0] += n_pass
            totals[cond][1] += n_tot
            pct = 100*n_pass/max(1, n_tot)
            row.append(f"{n_pass}/{n_tot} = {pct:.0f}%")
        lines.append("| " + " | ".join(row) + " |")
    # Aggregate row
    agg_row = ["**Aggregate**"]
    for cond in CONDS:
        n_pass, n_tot = totals[cond]
        agg_row.append(f"**{n_pass}/{n_tot} = {100*n_pass/max(1,n_tot):.0f}%**")
    lines.append("| " + " | ".join(agg_row) + " |")
    lines.append("")
    lines.append("## How to review")
    lines.append("")
    lines.append("1. Open one scene folder (e.g. `box_two_screens/`).")
    lines.append("2. Read `README.md` for the prompt + fixed parameters.")
    lines.append("3. Open `reference_coverage.png` (N1) or `reference_5ghz.png` + `reference_2ghz.png` (N2) — these are the Sionna-RT ground truth.")
    lines.append("4. Drill into one trial folder, e.g. `with_skill/t1/`.")
    lines.append("5. Read `simulation.py` — that's the agent-written code (the audit target).")
    lines.append("6. Open the agent's `coverage_*.png` — compare to the reference visually.")
    lines.append("7. Read `verifier_report.txt` — the 3-layer verdict (A: artifacts, B: Sionna RT used, C: physics oracle vs reference).")
    lines.append("")
    lines.append("## Verifier design (short version)")
    lines.append("")
    lines.append("**Layer A — artifacts**: required output files exist and parse.")
    lines.append("")
    lines.append("**Layer B — `sionna_rt_used`**: agent imported `sionna` (non-commented), `simulation_result.json.method` is not in the analytical-fallback set, and either a Mitsuba scene XML or a `rt.scene.<builtin>` reference is detected. This catches FSPL-fallback agents that produce numerically plausible output without actually invoking Sionna RT.")
    lines.append("")
    lines.append("**Layer C — physics oracle**: agent's coverage grid is compared cell-wise to a precomputed Sionna-RT reference (same scene, same AP, same solver params, same frequency). PASS if MAE ≤ 3 dB and ≥80% of cells agree within ±5 dB. If grid extents differ (the agent passed an explicit `center`/`size`), we fall back to a distribution-level check on mean (≤3 dB), std (≤2 dB), and valid-cell fraction (≤25 pp).")
    if task_name == "N2":
        lines.append("")
        lines.append("**Layer C extras for N2**: (i) the two output grids must actually differ (mean absolute difference ≥ 1 dB); (ii) the mean delta lies in [+3, +20] dB (FSPL theory predicts +6.4 dB for 5 → 2.4 GHz); (iii) the agent's reported `delta_dbm_mean` matches our computed value within ±2 dB.")
    return "\n".join(lines)


def main() -> int:
    for name, cfg in RUNS.items():
        if not cfg["results_dir"].exists():
            print(f"[skip] {name}: no results at {cfg['results_dir']}")
            continue
        tasks_by_id = {t["id"]: t
                       for t in json.loads(cfg["tasks_file"].read_text())["tasks"]}
        export_task(name, cfg, tasks_by_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
