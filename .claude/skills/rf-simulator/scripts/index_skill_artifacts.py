"""Index scripts/, templates/, and SKILL.md sections into the RAG vector store.

After running, lookups like:
    python3 lookup.py "BER simulation" --kind script
    python3 lookup.py "scene generation" --kind template
    python3 lookup.py "routing table" --kind skill_module

return code/template snippets the agent can copy directly.

Usage:
    python3 .claude/skills/rf-simulator/scripts/index_skill_artifacts.py
        [--dry-run] [--force]

Idempotent: store.store_artifact() dedups via cosine similarity. Re-running
after a script edit will refresh the snippet but skip near-duplicates.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_SKILL_ROOT = _THIS.parent.parent  # .../rf-simulator/
sys.path.insert(0, str(_SKILL_ROOT / "memory"))

import store  # noqa: E402

# How many lines from each script to surface as a snippet.
SCRIPT_SNIPPET_LINES = 50
# Templates indexed in full when <= this many lines, else chunked.
TEMPLATE_FULL_LINE_LIMIT = 300


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _module_docstring(text: str) -> str:
    """Extract the first triple-quoted string in the file (module docstring)."""
    m = re.search(r'^\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')\s*$',
                  text, re.DOTALL | re.MULTILINE)
    if m:
        return m.group(1).strip()
    return ""


def _argparse_summary(text: str) -> str:
    """Find argparse arg names + helps so the agent sees the CLI shape."""
    args = []
    for m in re.finditer(
        r'add_argument\(\s*["\']([^"\']+)["\'].*?(?:help\s*=\s*["\']([^"\']*)["\'])?',
        text, re.DOTALL,
    ):
        name, help_ = m.group(1), m.group(2) or ""
        args.append(f"  {name}: {help_}" if help_ else f"  {name}")
    if not args:
        return ""
    return "CLI args:\n" + "\n".join(args)


def _params_dict(text: str) -> str:
    """Extract a PARAMS = { ... } dict if present."""
    m = re.search(r'^\s*PARAMS\s*=\s*\{(.*?)^\s*\}', text,
                  re.DOTALL | re.MULTILINE)
    if m:
        return "PARAMS = {" + m.group(1).rstrip() + "\n}"
    return ""


def _first_n_lines(text: str, n: int) -> str:
    return "\n".join(text.splitlines()[:n])


def index_scripts(dry_run: bool) -> int:
    scripts_dir = _SKILL_ROOT / "scripts"
    n = 0
    for p in sorted(scripts_dir.glob("*.py")):
        if p.name.startswith("_") or p.name == "index_skill_artifacts.py":
            continue
        text = p.read_text()
        doc = _module_docstring(text)
        cli = _argparse_summary(text)
        snippet = _first_n_lines(text, SCRIPT_SNIPPET_LINES)
        if cli:
            snippet = snippet + "\n\n# " + cli.replace("\n", "\n# ")
        # Title from docstring first line, fallback to stem.
        title = doc.splitlines()[0][:120] if doc else p.stem
        summary = doc[:400] if doc else f"Runnable script {p.name}"
        # Tags: filename keywords + common terms in docstring.
        tags = [p.stem.replace("_", " ")]
        if "BER" in doc or "ber" in p.stem:
            tags.append("BER analytical AWGN")
        if "lookup" in p.stem or "vector" in doc.lower():
            tags.append("vector search RAG")
        if "verify" in p.stem:
            tags.append("output verification")
        artifact_id = f"script_{_slug(p.stem)}"
        rel_path = f"scripts/{p.name}"
        if dry_run:
            print(f"[dry] script: {artifact_id} ({rel_path}) — '{title[:60]}'")
        else:
            store.store_artifact(
                artifact_id=artifact_id, kind="script",
                title=title, file_path=rel_path,
                snippet=snippet, summary=summary, tags=tags, dedup=True,
            )
            print(f"  ✓ {artifact_id}")
        n += 1
    return n


def index_templates(dry_run: bool) -> int:
    tmpl_dir = _SKILL_ROOT / "templates"
    n = 0
    for p in sorted(tmpl_dir.glob("template_*.py")):
        text = p.read_text()
        doc = _module_docstring(text)
        params = _params_dict(text)
        lines = text.splitlines()
        if len(lines) <= TEMPLATE_FULL_LINE_LIMIT:
            snippet = text
        else:
            # Chunk: docstring + PARAMS + first 150 lines, plus output schema
            snippet = _first_n_lines(text, 150)
            if params and params not in snippet:
                snippet += "\n\n" + params
        title = doc.splitlines()[0][:120] if doc else p.stem
        summary = doc[:400] if doc else f"Template {p.name}"
        # Tags by template family
        stem = p.stem.replace("template_", "")
        tags = [stem.replace("_", " ")]
        if stem == "ber":
            tags.extend(["BER curve LDPC AWGN modulation", "sionna.phy.fec"])
        elif "rt_" in stem:
            tags.extend(["ray tracing coverage", "sionna.rt"])
        elif stem == "scene":
            tags.extend(["scene generation room furniture",
                         "lib.scene_gen Scene Room"])
        elif "mimo" in stem or "ofdm" in stem:
            tags.extend(["MIMO OFDM channel estimation",
                         "sionna.phy.ofdm sionna.phy.mimo"])
        elif "neural" in stem:
            tags.extend(["neural receiver autoencoder training",
                         "torch.nn"])
        elif "optimize" in stem:
            tags.extend(["optimize gradient AP placement",
                         "differentiable"])
        elif "system_level" in stem:
            tags.extend(["multi-cell scheduling link adaptation",
                         "sionna.sys"])
        artifact_id = f"template_{_slug(stem)}"
        rel_path = f"templates/{p.name}"
        if dry_run:
            print(f"[dry] template: {artifact_id} ({rel_path}) — '{title[:60]}'")
        else:
            store.store_artifact(
                artifact_id=artifact_id, kind="template",
                title=title, file_path=rel_path,
                snippet=snippet, summary=summary, tags=tags, dedup=True,
            )
            print(f"  ✓ {artifact_id}")
        n += 1
    return n


def index_skill_md(dry_run: bool) -> int:
    skill_md = _SKILL_ROOT / "SKILL.md"
    if not skill_md.exists():
        return 0
    text = skill_md.read_text()
    # Split by "## " headers (top-level sections after frontmatter)
    sections = re.split(r'(?m)^## +(.+)$', text)
    # sections[0] = preamble (frontmatter + intro), then alternating heading/body
    n = 0
    if sections[0].strip():
        # Index preamble as "skill_module_overview"
        preamble = sections[0].strip()[:1500]
        artifact_id = "skill_module_overview"
        rel_path = "SKILL.md"
        title = "Sionna RF Skill — Overview & Usage"
        summary = "Skill overview: when to ask vs. proceed, RF_SKILL_DIR, " \
                  "skeleton on disk, lookup-before-guessing protocol."
        tags = ["skill overview", "RF_SKILL_DIR", "lookup", "skeleton",
                "simulation_result.json"]
        if dry_run:
            print(f"[dry] skill_module: {artifact_id} ({title})")
        else:
            store.store_artifact(
                artifact_id=artifact_id, kind="skill_module",
                title=title, file_path=rel_path,
                snippet=preamble, summary=summary, tags=tags, dedup=True,
            )
            print(f"  ✓ {artifact_id}")
        n += 1
    for i in range(1, len(sections), 2):
        heading = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        # Stop at next-level header to keep the section bounded
        body = re.split(r'(?m)^# +', body)[0].strip()
        if not body:
            continue
        snippet = (heading + "\n\n" + body)[:2000]
        slug = _slug(heading)[:40]
        artifact_id = f"skill_module_{slug}"
        title = f"SKILL.md — {heading[:80]}"
        summary = body.split("\n\n")[0][:400]
        tags = [heading.lower(), "SKILL.md section"]
        rel_path = "SKILL.md"
        if dry_run:
            print(f"[dry] skill_module: {artifact_id} — '{heading[:60]}'")
        else:
            store.store_artifact(
                artifact_id=artifact_id, kind="skill_module",
                title=title, file_path=rel_path,
                snippet=snippet, summary=summary, tags=tags, dedup=True,
            )
            print(f"  ✓ {artifact_id}")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be indexed without writing")
    args = ap.parse_args()

    if not store._ensure_initialized():
        print("ERROR: chromadb / sentence-transformers not installed",
              file=sys.stderr)
        return 2
    before = store.stats().get("n_items", 0)
    print(f"Store before: {before} items at {store.DB_PATH}")
    print()

    print("=== Indexing scripts/ ===")
    n_scripts = index_scripts(args.dry_run)
    print(f"  total: {n_scripts}")
    print()
    print("=== Indexing templates/ ===")
    n_tmpl = index_templates(args.dry_run)
    print(f"  total: {n_tmpl}")
    print()
    print("=== Indexing SKILL.md ===")
    n_skill = index_skill_md(args.dry_run)
    print(f"  total: {n_skill}")
    print()

    if not args.dry_run:
        after = store.stats().get("n_items", 0)
        print(f"Store after: {after} items (delta +{after - before})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
