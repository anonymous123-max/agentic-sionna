# Web UI restore — partial

Files restored from git history (commits `a200c63` Feb 12 + `9c69111` Feb 25 2026).

## What's here

| File | Source | Notes |
|---|---|---|
| `dashboard_app.py` | a200c63 | Monolithic Flask app (80 KB) — imports `src/models/*`, `src/optimizer/*`, `src/exporters/*` |
| `routes/__init__.py` | 9c69111 | Blueprint loader — registers chat, scenes, coverage, catalog, creation, rays, ase, segmentation |
| `routes/chat.py` | 9c69111 | Claude Haiku-backed chat agent (parses `action` JSON blocks) |
| `routes/scenes.py` | a200c63 | Scene CRUD + furniture persistence |
| `routes/pages.py` | 9c69111 | Page rendering routes |
| `routes/shared.py` | 9c69111 | Shared OUTPUTS_DIR + utilities |
| `templates/dashboard.html` | a200c63 | Three.js + Plotly frontend with chat sidebar + 3D canvas |
| `static/css/dashboard.css` | a200c63 | Dashboard styling |
| `static/js/dashboard.js` | a200c63 | Frontend logic (136 KB) — antenna draggable canvases, scene state |

## What's broken

All restored files reference the OLD package layout (`src.models.room`, `src.optimizer.layout`, `routes.coverage`, `routes.catalog`, `routes.creation`, `routes.rays`, `routes.ase`, `routes.segmentation`). The current repo:
- Renamed `src/` → `sionna_skill/`
- Removed `routes/coverage.py`, `routes/catalog.py`, `routes/creation.py`, `routes/rays.py`, `routes/ase.py`, `routes/segmentation.py`
- Refactored `lib/scene_gen/` → its own subdir under `.claude/skills/rf-simulator/`

## Minimum-viable path to working web UI

1. Replace top of `dashboard_app.py` imports:
   ```python
   from sionna_skill.lib.scene_gen.models import Scene, Room, FurnitureItem
   ```
   (vs. the old `from src.models.room import ...`)
2. Strip blueprint registrations to only the ones with restored files (chat, scenes, pages, shared) — comment out the rest in `routes/__init__.py`
3. Stub or recreate `coverage_bp`, `catalog_bp`, `creation_bp`, `rays_bp`, `ase_bp`, `segment_bp` as empty blueprints to make the app start
4. Run: `cd web && python3 dashboard_app.py --port 8080` → open http://localhost:8080

## Files NOT restored (would need separate work)

- `routes/coverage.py` — coverage map computation
- `routes/catalog.py` — 3D-FUTURE catalog browser
- `routes/creation.py` — scene creation wizard
- `routes/rays.py` — ray-tracing visualization
- `routes/ase.py` — ASE indoor scene loader
- `routes/segmentation.py` — mesh segmentation
- `templates/skills.html` — secondary skills page

## Why we restored these

User asked for "chatbox + blank canvas" — both are in `templates/dashboard.html` (chat panel in sidebar, Three.js canvas as main view). The blueprint setup is needed for the chat endpoint to resolve.

## Chatbox policy (NEW)

Per [references/viewer-spec.md](../.claude/skills/rf-simulator/references/viewer-spec.md#chatbox-policy-llm-driven-ui-elements), the chatbox **does not render by default**. It only appears when:
1. `ANTHROPIC_API_KEY` is set in the environment when `dashboard_app.py` starts
2. The user explicitly asks the skill for an interactive UI

How it's wired here:
- `routes/__init__.py` registers `chat_bp` only if `chat_enabled()` returns true
- `routes/pages.py` passes `chat_enabled=...` into `render_template`
- `templates/dashboard.html` wraps the chat sidebar in `{% if chat_enabled %}`

So with no API key, users see the canvas + scene tools but no chat input. With a key, the full chat-driven flow comes back. This matches the design goal: "for now no chatbox; in the future when we have an Anthropic API key, the chatbox should work as well as in Claude Code."
