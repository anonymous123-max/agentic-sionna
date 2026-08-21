"""tools/online_apis.py — agent-callable lookups against live sources.

Per master guide Part 6 Layer 3 (Online API and Dynamic Knowledge Sources).
Each function is a thin, defensive wrapper around a public endpoint:

    fetch_sionna_docs(class_name, method_name=None)
    search_arxiv(query, max_results=3)
    search_github_issues(error_message, repo="NVlabs/sionna", n=3)
    get_latest_sionna_version()
    detect_installed_sionna()

Use when local references don't have the answer:
  - Unknown Sionna class with no `references/` coverage  → fetch_sionna_docs
  - Frontier topic (ISAC, STAR-RIS, THz)                  → search_arxiv
  - Cryptic runtime error not in error-patterns.md        → search_github_issues
  - "Is my Sionna version current?"                       → get_latest_sionna_version

All functions return a list of dicts (or empty list on failure). Network
errors never raise — they log and return [] so a single API outage can't
crash the agent.

CACHING: results are cached file-backed at $XDG_CACHE_HOME/rf-skill/online_apis.json
(or ~/.cache/rf-skill/online_apis.json) with per-function TTLs. Pass
--no-cache to any subcommand to force a fresh fetch.

  TTLs:
    fetch_sionna_docs       7 days (Sionna API rarely changes)
    search_arxiv            6 hours (latest-paper relevance window)
    search_github_issues    24 hours (issue threads update slowly)
    get_latest_sionna_version  6 hours

ROBUSTNESS — fetch_sionna_docs uses three strategies in order of
reliability:
  1. `inspect.getdoc()` if Sionna is importable locally (fastest, exact)
  2. Raw GitHub source grep for `class ClassName` + docstring
  3. ReadTheDocs HTML regex (current strategy, most fragile)

Only stdlib (no new deps). httpx is NOT used; urllib only.

CLI:
    python3 online_apis.py docs RadioMapSolver
    python3 online_apis.py arxiv "STAR-RIS beamforming"
    python3 online_apis.py issues "CDL channel does not support multiple"
    python3 online_apis.py version
    python3 online_apis.py cache-stats        # show cache contents
    python3 online_apis.py cache-clear        # wipe cache
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


# ────────────────────────────────────────────────────────────────────
# File-backed cache (P2.5)
# ────────────────────────────────────────────────────────────────────

def _cache_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    p = Path(base) / "rf-skill"
    p.mkdir(parents=True, exist_ok=True)
    return p / "online_apis.json"


def _cache_load() -> dict:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _cache_save(cache: dict) -> None:
    try:
        _cache_path().write_text(json.dumps(cache))
    except Exception as e:
        print(f"[cache] save failed: {e}", file=sys.stderr)


def _cache_get(key: str, ttl_seconds: int) -> object | None:
    cache = _cache_load()
    e = cache.get(key)
    if not e:
        return None
    if time.time() - e.get("ts", 0) > ttl_seconds:
        return None
    return e.get("data")


def _cache_set(key: str, data: object) -> None:
    cache = _cache_load()
    cache[key] = {"ts": time.time(), "data": data}
    _cache_save(cache)


_NO_CACHE = False  # toggled by --no-cache CLI flag


def _cached(prefix: str, ttl: int):
    """Decorator: cache function result by (prefix, args, kwargs)."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            if _NO_CACHE:
                return fn(*args, **kwargs)
            key = f"{prefix}:{json.dumps([args, kwargs], sort_keys=True, default=str)}"
            hit = _cache_get(key, ttl)
            if hit is not None:
                return hit
            data = fn(*args, **kwargs)
            if data:  # don't cache empty failures (let them retry next time)
                _cache_set(key, data)
            return data
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return deco


# ────────────────────────────────────────────────────────────────────
# Sionna documentation lookup — multi-strategy (P2.6)
# ────────────────────────────────────────────────────────────────────

SIONNA_DOC_BASE = "https://nvlabs.github.io/sionna/api/"
SIONNA_RAW_BASE = "https://raw.githubusercontent.com/NVlabs/sionna/main/src/sionna"


def _strategy_inspect(class_name: str, method_name: str | None) -> dict | None:
    """Strategy 1: use inspect.getdoc() if Sionna is importable locally.

    This is the most reliable source — the actual Python docstring of
    the class as installed. Fails fast (ImportError) when Sionna isn't
    in the current Python env.
    """
    try:
        import importlib
        for mod in ("sionna.rt", "sionna.phy", "sionna.sys",
                    "sionna.phy.fec.ldpc", "sionna.phy.fec.polar",
                    "sionna.phy.channel", "sionna.phy.channel.tr38901",
                    "sionna.phy.mapping", "sionna.phy.ofdm",
                    "sionna.phy.mimo", "sionna.phy.signal"):
            try:
                m = importlib.import_module(mod)
            except ImportError:
                continue
            cls = getattr(m, class_name, None)
            if cls is None:
                continue
            import inspect
            target = cls
            if method_name and hasattr(cls, method_name):
                target = getattr(cls, method_name)
            doc = inspect.getdoc(target) or ""
            sig = ""
            try:
                sig = str(inspect.signature(target))
            except (ValueError, TypeError):
                pass
            return {
                "class_name": class_name,
                "method_name": method_name,
                "summary": doc[:1500],
                "signature": sig,
                "source_url": f"local::{mod}.{class_name}",
                "strategy": "inspect",
            }
    except Exception:
        return None
    return None


@_cached("github_tree", ttl=7 * 86400)
def _get_sionna_tree() -> list[str]:
    """Fetch the full Sionna v2.0 .py file list from GitHub once, cached 7d."""
    url = "https://api.github.com/repos/NVlabs/sionna/git/trees/main?recursive=1"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "rf-skill/1.0",
            "Accept": "application/vnd.github+json",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"[_get_sionna_tree] {e}", file=sys.stderr)
        return []
    return [it["path"] for it in (data.get("tree") or [])
            if it.get("path", "").startswith("src/sionna/")
            and it.get("path", "").endswith(".py")]


def _strategy_github_raw(class_name: str, method_name: str | None) -> dict | None:
    """Strategy 2: locate class via GitHub code search, then fetch raw source.

    Uses the GitHub Code Search API to find which file actually contains
    `class ClassName` — handles arbitrarily-named files (e.g. Mapper in
    mapping.py, where path-similarity heuristics miss). Falls back to
    iterating the cached tree if code search fails.
    """
    # Permissive: skip any number of `# comment` lines between class header and docstring.
    pattern = re.compile(
        rf'^class\s+{re.escape(class_name)}\b[^:]*:\s*\n'
        r'(?:\s*#[^\n]*\n)*'              # zero or more comment lines (e.g. pylint disables)
        r'\s*(?:r)?"""\s*(.+?)"""',
        re.MULTILINE | re.DOTALL,
    )
    raw_base = "https://raw.githubusercontent.com/NVlabs/sionna/main"

    # Strategy 2a: GitHub code search (one round-trip).
    candidates: list[str] = []
    try:
        q = urllib.parse.urlencode({
            "q": f'"class {class_name}" repo:NVlabs/sionna extension:py',
            "per_page": 5,
        })
        url = f"https://api.github.com/search/code?{q}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "rf-skill/1.0",
            "Accept": "application/vnd.github+json",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        candidates = [it["path"] for it in (data.get("items") or [])]
    except Exception as e:
        # Code search requires auth above ~10 req/min; gracefully fall through.
        print(f"[github code search] {e}", file=sys.stderr)

    # Strategy 2b: fall back to scanning cached tree, sorted by name similarity.
    # When name-similarity returns 0 (class is in a generically-named file like
    # encoding.py), we still scan ALL .py files. Cache hides the cost on
    # subsequent calls.
    if not candidates:
        paths = _get_sionna_tree()
        if not paths:
            return None
        name_tokens = re.findall(r"[A-Z][a-z0-9]*", class_name)
        name_lower = class_name.lower()

        def score(path: str) -> int:
            fname = path.rsplit("/", 1)[-1].lower().replace("_", "")
            s = 0
            if name_lower.replace("_", "") in fname:
                s += 100
            for tok in name_tokens:
                if tok.lower() in fname:
                    s += 10
            return s
        # Sort by score, keep ALL paths — first match wins via the regex.
        candidates = sorted(paths, key=score, reverse=True)

    for rel in candidates:
        url = f"{raw_base}/{rel}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "rf-skill/1.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                src = r.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        m = pattern.search(src)
        if not m:
            continue
        doc = m.group(1).strip()
        sig_m = re.search(
            rf'class\s+{re.escape(class_name)}\b[^:]*:[\s\S]*?def\s+__init__\s*\(([^)]*)\)',
            src, re.MULTILINE,
        )
        sig = f"({sig_m.group(1).strip()})" if sig_m else ""
        return {
            "class_name": class_name,
            "method_name": method_name,
            "summary": doc[:1500],
            "signature": sig,
            "source_url": url,
            "strategy": "github_raw",
        }
    return None


def _strategy_readthedocs_html(class_name: str, method_name: str | None) -> dict | None:
    """Strategy 3 (fallback): parse the rendered ReadTheDocs page."""
    candidates = [
        f"{SIONNA_DOC_BASE}rt.html#sionna.rt.{class_name}",
        f"{SIONNA_DOC_BASE}phy.html#sionna.phy.{class_name}",
        f"{SIONNA_DOC_BASE}sys.html#sionna.sys.{class_name}",
    ]
    for url in candidates:
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                html = r.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        m = re.search(
            rf'<dt[^>]*id="sionna\.\w+\.{re.escape(class_name)}"[^>]*>'
            r'(.*?)</dl>',
            html, re.DOTALL,
        )
        if m:
            text = re.sub(r"<[^>]+>", " ", m.group(1))
            text = re.sub(r"\s+", " ", text).strip()
            return {
                "class_name": class_name,
                "method_name": method_name,
                "summary": text[:1500],
                "signature": "",
                "source_url": url,
                "strategy": "readthedocs_html",
            }
    return None


@_cached("docs", ttl=7 * 86400)
def fetch_sionna_docs(class_name: str,
                       method_name: str | None = None) -> list[dict]:
    """Best-effort retrieval of a Sionna class docstring from live sources.

    Tries three strategies in order of reliability:
      1. inspect.getdoc() if Sionna is importable
      2. Raw GitHub source grep
      3. ReadTheDocs HTML regex

    Returns at most one entry; empty list on total failure.
    """
    for strategy in (_strategy_inspect, _strategy_github_raw,
                     _strategy_readthedocs_html):
        result = strategy(class_name, method_name)
        if result:
            return [result]
    return []


# ────────────────────────────────────────────────────────────────────
# arXiv search
# ────────────────────────────────────────────────────────────────────

ARXIV_API = "http://export.arxiv.org/api/query"


@_cached("arxiv", ttl=6 * 3600)
def search_arxiv(query: str, max_results: int = 3) -> list[dict]:
    """Search arXiv for recent papers matching `query`."""
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            xml_text = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[search_arxiv] {e}", file=sys.stderr)
        return []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    entries: list[dict] = []
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        sum_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        pub_el = entry.find("atom:published", ns)
        authors = [a.findtext("atom:name", "", ns)
                   for a in entry.findall("atom:author", ns)]
        title = (title_el.text or "").strip() if title_el is not None else ""
        summary = (sum_el.text or "").strip() if sum_el is not None else ""
        arxiv_id = ((id_el.text or "").strip().rsplit("/", 1)[-1]
                    if id_el is not None else "")
        published = (pub_el.text or "").strip() if pub_el is not None else ""
        entries.append({
            "title": re.sub(r"\s+", " ", title),
            "summary": re.sub(r"\s+", " ", summary)[:600],
            "arxiv_id": arxiv_id,
            "published": published,
            "authors": authors,
        })
    return entries


# ────────────────────────────────────────────────────────────────────
# GitHub issues search
# ────────────────────────────────────────────────────────────────────

GITHUB_SEARCH = "https://api.github.com/search/issues"


@_cached("issues", ttl=24 * 3600)
def search_github_issues(error_message: str,
                          repo: str = "NVlabs/sionna",
                          n: int = 3) -> list[dict]:
    """Search a repo's issues+PRs for the most relevant matches to an error."""
    distinctive = re.sub(r"\s+", " ", error_message).strip()[:200]
    query = f'"{distinctive}" repo:{repo}'
    params = urllib.parse.urlencode({"q": query, "per_page": n})
    url = f"{GITHUB_SEARCH}?{params}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "rf-simulator-skill/1.0",
            "Accept": "application/vnd.github+json",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        print(f"[search_github_issues] {e}", file=sys.stderr)
        return []
    items = data.get("items") or []
    out = []
    for it in items[:n]:
        body = (it.get("body") or "").strip().replace("\r\n", "\n")
        out.append({
            "title": it.get("title", ""),
            "html_url": it.get("html_url", ""),
            "state": it.get("state", ""),
            "comments": it.get("comments", 0),
            "body_preview": body[:400],
        })
    return out


# ────────────────────────────────────────────────────────────────────
# Sionna version helpers
# ────────────────────────────────────────────────────────────────────

@_cached("version_latest", ttl=6 * 3600)
def get_latest_sionna_version() -> str | None:
    """Latest tag from NVlabs/sionna releases (network call)."""
    try:
        url = "https://api.github.com/repos/NVlabs/sionna/releases/latest"
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        return data.get("tag_name")
    except Exception as e:
        print(f"[get_latest_sionna_version] {e}", file=sys.stderr)
        return None


def detect_installed_sionna() -> str | None:
    """Version of locally installed sionna, or None if not installed.
    Not cached — `pip install` could change the answer between calls."""
    try:
        import importlib.metadata
        return importlib.metadata.version("sionna")
    except Exception:
        return None


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--no-cache", action="store_true",
                    help="bypass and overwrite the file cache")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_docs = sub.add_parser("docs", help="Fetch a Sionna class docstring")
    p_docs.add_argument("class_name")
    p_docs.add_argument("--method", default=None)

    p_arx = sub.add_parser("arxiv", help="Search arXiv recent papers")
    p_arx.add_argument("query", nargs="+")
    p_arx.add_argument("--max", type=int, default=3)

    p_iss = sub.add_parser("issues", help="Search GitHub issues for an error")
    p_iss.add_argument("error", nargs="+")
    p_iss.add_argument("--repo", default="NVlabs/sionna")
    p_iss.add_argument("--n", type=int, default=3)

    sub.add_parser("version", help="Compare local vs latest Sionna version")
    sub.add_parser("cache-stats", help="Show cache file size + entry count")
    sub.add_parser("cache-clear", help="Wipe cache file")

    args = ap.parse_args()
    global _NO_CACHE
    _NO_CACHE = args.no_cache

    if args.cmd == "docs":
        out = fetch_sionna_docs(args.class_name, args.method)
    elif args.cmd == "arxiv":
        out = search_arxiv(" ".join(args.query), args.max)
    elif args.cmd == "issues":
        out = search_github_issues(" ".join(args.error), args.repo, args.n)
    elif args.cmd == "version":
        out = {
            "installed": detect_installed_sionna(),
            "latest": get_latest_sionna_version(),
        }
    elif args.cmd == "cache-stats":
        cache = _cache_load()
        out = {
            "path": str(_cache_path()),
            "entries": len(cache),
            "size_bytes": _cache_path().stat().st_size if _cache_path().exists() else 0,
            "keys": list(cache.keys())[:20],
        }
    elif args.cmd == "cache-clear":
        if _cache_path().exists():
            _cache_path().unlink()
        out = {"cleared": True, "path": str(_cache_path())}
    else:
        ap.error("unknown command")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
