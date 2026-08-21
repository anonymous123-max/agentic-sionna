# Interactive Viewer Specification

## Contents

1. [Output Directory Structure](#output-directory-structure) — File layout for viewer assets and simulation outputs
2. [Viewer Features](#viewer-features) — 3D navigation, coverage overlay, and interactive controls
3. [Viewer Theme](#viewer-theme) — Color palette, lighting, and visual style configuration
4. [Viewer Generation Pattern](#viewer-generation-pattern) — Code flow for producing viewer.html from scene data
5. [Serving the Viewer](#serving-the-viewer) — HTTP server setup and auto-open browser behavior
6. [Multi-Room House Layout](#multi-room-house-layout) — Handling multi-room scenes with shared walls and corridors

Every scene generates a `viewer.html` served via HTTP that auto-opens
in the user's browser. The viewer loads `scene.glb` (with real 3D-FUTURE
meshes) as an external file — no base64 embedding.

## Output Directory Structure

```
outputs/<descriptive-name>/
├── viewer.html          # Interactive 3D viewer (PRIMARY — served via HTTP)
├── generate.py          # Self-contained generation script
├── layout.png           # Static floor plan (fallback)
├── scene.xml            # Mitsuba 3.0 / Sionna RT (real OBJ meshes)
├── scene.glb            # 3D model (real 3D-FUTURE meshes, box fallback)
├── scene_state.json     # Structured scene state
├── coverage_*.png       # Coverage heatmap (if simulation ran)
└── coverage_*.npy       # Raw coverage data (if simulation ran)
```

## Second-pass benchmark protocol

When the harness invokes you for a **viewer-gen-only trial** (recognizable
because the cwd already contains `simulation_result.json` + visualization
artifacts produced by an earlier sim trial), the contract is narrower than
a normal trial:

- **Input**: `simulation_result.json` (canonical metrics) + tier-specific
  artifacts (`coverage_map.npy`, `cir.npy`, `scene_state.json`, etc.)
- **Output**: a single new file `viewer.html` in the cwd
- **Do not** modify `simulation_result.json` or any input artifact
- **Do not** include a chatbox (per `Chatbox policy` below)
- **Do** load data via `fetch('./simulation_result.json')` and any sibling
  `.npy` / `.json` files; emit a clear "loading…" placeholder if a fetch
  fails so the page never appears blank

Verifier in `benchmark/run_viewer_pass.py` checks that viewer.html parses,
contains a `<canvas>` or `<div id="plot...">`, has a `<script>` block, and
references `simulation_result.json`. Anti-patterns auto-failed: `chat-input`,
`chat-messages`, `/api/chat` substrings.

The viewer-pass costs ~5-8K tokens / ~3-5 min wall per trial. It runs only
on PASSING base trials in visualization-relevant tiers (T0_scene_gen,
T2_ray_tracing, T2_channel_modeling, T4_system_level), so most BER-style
trials are skipped — visualizing a single BER number adds no signal.

## Chatbox policy (LLM-driven UI elements)

**Default: do NOT include a chatbox** in the generated viewer. The viewer is
a static-page artifact rendered from `scene_state.json` + simulation outputs,
served via plain HTTP. A chatbox needs a live LLM backend, which:

- requires an `ANTHROPIC_API_KEY` (or equivalent) at serve time
- adds operational complexity (server-side routes, request signing, rate limits)
- is wasted effort when the user already drove the run via Claude Code

**Conditional inclusion:** ONLY include a chatbox if BOTH:
1. the user explicitly asks for one ("add a chatbox", "interactive UI"), AND
2. the runtime env has `ANTHROPIC_API_KEY` set (check `os.environ`)

When both are true, render the chatbox UI and back it with a `/api/chat`
route that proxies to Anthropic's Messages API (Haiku is fine for UI helpers).
When either condition is false, omit the chatbox HTML entirely; do not
render a disabled or stub version — the user shouldn't see a non-functional
input field.

The dashboard restored in `web/` (under `templates/dashboard.html`) follows
this rule: chat blueprint is registered conditionally on `ANTHROPIC_API_KEY`,
and the chat sidebar `<div>` is wrapped in a Jinja2 `{% if chat_enabled %}`
guard so it doesn't render when the backend is unavailable.

## Viewer Features

| Feature | Implementation |
|---------|---------------|
| 3D orbit/pan/zoom | Three.js OrbitControls |
| Real furniture meshes | GLB with 3D-FUTURE OBJ geometry (box fallback if catalog unavailable) |
| Coverage heatmap overlay | PlaneGeometry with canvas texture, jet colormap, toggleable |
| Click-to-inspect | Raycaster → object name, dimensions, material, RSS value |
| Color legend | Fixed overlay with dBm range and gradient |
| Stats panel | Mean/median/min/max RSS, coverage %, frequency, TX power |
| Dark theme | CSS variables from Viewer Theme (below) |
| Auto-reload | JS polls scene.glb mtime every 3s, reloads on change |
| Responsive | `window.resize` handler |
| Shadows | PCFSoftShadowMap, furniture casts, floor receives |
| Z-height slider | Range input swaps coverage slice, moves heatmap plane |
| Signal-colored rays | Green→yellow→red per segment based on cumulative path loss |
| Ray wall transmission | Rays pass through interior walls with loss, not just reflect |

---

## Viewer Theme

Every `viewer.html` should include these CSS variables and reference them
via `var(--*)` instead of hardcoding hex colors — this keeps the dark theme
consistent across all generated viewers and makes future theme changes trivial.

```css
:root {
  --bg: #050505;
  --surface: #0a0a0f;
  --card: #111118;
  --border: #1a1a2e;
  --text: #e0e0e8;
  --text-dim: #8888a0;
  --accent: #00d4ff;
  --green: #00ff88;
  --amber: #ffaa00;
  --red: #ff4466;
  --font: 'Inter', system-ui, sans-serif;
  --mono: 'JetBrains Mono', monospace;
  --radius: 8px;

  /* Heatmap jet colormap (from old dashboard) */
  --heat-high: #FF3B30;     /* Red — strong signal */
  --heat-mid-high: #FF9500; /* Orange */
  --heat-mid-low: #4CD964;  /* Green */
  --heat-low: #007AFF;      /* Blue — weak signal */
}
```

**Three.js mapping:**
- `scene.background = new THREE.Color(0x050505)` (matches `--bg`)
- `renderer.toneMapping = THREE.ACESFilmicToneMapping` (cinematic look)
- Grid: `new THREE.GridHelper(50, 50, 0x1a1a2e, 0x111118)` (--border, --card)

**Panel styling (frosted glass effect from old dashboard):**
- Overlay panels: `background: rgba(17,17,17,0.85); backdrop-filter: blur(8px)`
- Info panel: `background: var(--card); border: 1px solid var(--border)`
- Buttons hover: `border-color: var(--accent); color: var(--accent)`

---

## Viewer Generation Pattern

```python
import json
from pathlib import Path

def generate_viewer(out_dir, scene_state, coverage_data=None):
    coverage_json = "null"
    if coverage_data is not None:
        coverage_json = json.dumps({{
            "grid": coverage_data.tolist(),
            "width": scene_state["scene"]["bounds"]["width"],
            "length": scene_state["scene"]["bounds"]["depth"],
        }})

    html = f'''<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>{{scene_state["meta"]["name"]}}</title>
<script type="importmap">
{{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.183.1/build/three.module.js",
"three/addons/": "https://cdn.jsdelivr.net/npm/three@0.183.1/examples/jsm/"}}}}
</script>
<style>
  :root {{
    --bg: #050505; --surface: #0a0a0f; --card: #111118; --border: #1a1a2e;
    --text: #e0e0e8; --text-dim: #8888a0; --accent: #00d4ff;
    --green: #00ff88; --amber: #ffaa00; --red: #ff4466;
    --font: 'Inter', system-ui, sans-serif; --mono: 'JetBrains Mono', monospace;
    --radius: 8px;
  }}
  body {{ margin:0; background:var(--bg); color:var(--text); font-family:var(--font); overflow:hidden; }}
  .panel {{ position:absolute; background:rgba(17,17,17,0.85); backdrop-filter:blur(8px);
            -webkit-backdrop-filter:blur(8px); border:1px solid var(--border);
            border-radius:var(--radius); z-index:10; font-size:13px; }}
  #info {{ top:12px; left:12px; padding:14px 18px; max-width:320px; }}
  #stats {{ bottom:12px; left:12px; padding:14px 18px; font-family:var(--mono); font-size:12px; }}
  #controls {{ top:12px; right:12px; padding:10px 14px; display:flex; gap:4px; }}
  #colorbar {{ bottom:12px; right:12px; padding:10px 14px; width:30px; }}
  #colorbar .gradient {{ width:20px; height:120px; border-radius:3px;
    background:linear-gradient(to bottom, var(--heat-high), var(--heat-mid-high), var(--heat-mid-low), var(--heat-low)); }}
  #colorbar .label {{ font-size:10px; font-family:var(--mono); color:var(--text-dim); text-align:center; }}
  canvas {{ display:block; }}
  button {{ background:var(--surface); color:var(--text); border:1px solid var(--border);
            padding:6px 12px; border-radius:var(--radius); cursor:pointer;
            font-family:var(--font); font-size:12px; transition:all 0.15s; }}
  button:hover {{ background:var(--card); border-color:var(--accent); color:var(--accent); }}
  .dim {{ color:var(--text-dim); }}
  .accent {{ color:var(--accent); }}
</style>
</head><body>
<div id="info" class="panel">
  <h3 style="margin:0 0 6px" class="accent">{{scene_state["meta"]["name"]}}</h3>
  <div class="dim">{{scene_state["meta"].get("description","")}}</div>
</div>
<div id="stats" class="panel" style="display:none"></div>
<div id="controls" class="panel">
  <button onclick="toggleHeatmap()">Toggle Heatmap</button>
  <button onclick="toggleWalls()">Toggle Walls</button>
</div>
<div id="colorbar" class="panel" style="display:none">
  <div class="label">High</div>
  <div class="gradient"></div>
  <div class="label">Low</div>
</div>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';
import {{ TransformControls }} from 'three/addons/controls/TransformControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050505);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.1, 2000);
const renderer = new THREE.WebGLRenderer({{ antialias:true }});
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
document.body.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 1.2));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);
scene.add(new THREE.GridHelper(50, 50, 0x1a1a2e, 0x111118));

// Load GLB — real 3D-FUTURE meshes + translucent walls
const wallMeshes = [];
new GLTFLoader().load('./scene.glb', (gltf) => {{
  // Walls → translucent glass-box look (signature style)
  gltf.scene.traverse((child) => {{
    if (child.isMesh && child.name && child.name.startsWith('wall')) {{
      child.material = new THREE.MeshStandardMaterial({{
        color: 0x888888,
        transparent: true,
        opacity: 0.10,
        side: THREE.DoubleSide,
        roughness: 0.85,
        depthWrite: false,
      }});
      wallMeshes.push(child);
    }}
    // IMPORTANT: Preserve original furniture materials from GLB.
    // Do NOT override with MeshStandardMaterial. Only enable shadows:
    // child.castShadow = true; child.receiveShadow = true;
  }});
  scene.add(gltf.scene);
  const box = new THREE.Box3().setFromObject(gltf.scene);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  camera.position.set(center.x + size.x*0.8, center.y + size.y*0.8, size.z * 2);
  controls.target.copy(center);
  controls.update();
}});

// Toggle wall visibility
window.toggleWalls = () => {{
  wallMeshes.forEach(w => {{ w.visible = !w.visible; }});
}};

// Auto-reload: poll scene.glb mtime every 3s, reload if changed
let lastMod = 0;
setInterval(async () => {{
  try {{
    const res = await fetch('./scene.glb', {{ method: 'HEAD' }});
    const mod = new Date(res.headers.get('last-modified')).getTime();
    if (lastMod && mod > lastMod) location.reload();
    lastMod = mod;
  }} catch {{}}
}}, 3000);

// Furniture interaction: double-click to select, drag to move
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const transformCtrl = new TransformControls(camera, renderer.domElement);
transformCtrl.setMode('translate');
transformCtrl.setSpace('world');
// Three.js r152+: must add the helper gizmo, not the control itself
const gizmo = transformCtrl.getHelper();
scene.add(gizmo);
transformCtrl.addEventListener('dragging-changed', (e) => {{
  controls.enabled = !e.value;
}});

let selectedObj = null;
function updateInspectPanel(obj) {{
  const name = obj.name || 'unknown';
  const pos = obj.position;
  const box = new THREE.Box3().setFromObject(obj);
  const sz = box.getSize(new THREE.Vector3());
  const panel = document.getElementById('stats');
  panel.style.display = 'block';
  panel.innerHTML = `
    <div class="accent">${{name}}</div>
    <div class="dim">Pos: (${{pos.x.toFixed(2)}}, ${{pos.y.toFixed(2)}}, ${{pos.z.toFixed(2)}})</div>
    <div class="dim">Size: ${{sz.x.toFixed(2)}} x ${{sz.y.toFixed(2)}} x ${{sz.z.toFixed(2)}} m</div>
    <div style="margin-top:6px;font-size:11px;color:var(--text-dim)">
      Double-click to select/move. Esc to deselect.</div>`;
}}

renderer.domElement.addEventListener('dblclick', (e) => {{
  mouse.x = (e.clientX / innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(scene.children, true)
    .filter(h => h.object.name && !h.object.name.startsWith('wall')
                 && h.object.name !== 'floor');
  if (hits.length > 0) {{
    const obj = hits[0].object.parent?.type === 'Group'
      ? hits[0].object.parent : hits[0].object;
    selectedObj = obj;
    transformCtrl.attach(obj);
    updateInspectPanel(obj);
  }} else {{
    transformCtrl.detach();
    selectedObj = null;
    document.getElementById('stats').style.display = 'none';
  }}
}});

window.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') {{
    transformCtrl.detach();
    selectedObj = null;
    document.getElementById('stats').style.display = 'none';
  }}
}});

window.addEventListener('resize', () => {{
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});
(function animate() {{ requestAnimationFrame(animate); renderer.render(scene, camera); controls.update(); }})();
</script></body></html>'''

    viewer_path = out_dir / "viewer.html"
    viewer_path.write_text(html)
    print(f"Wrote {{viewer_path}}")
```

## Serving the Viewer (Always HTTP)

**Always serve via HTTP** — even locally. This enables external file
loading (GLB, textures) and auto-reload. Never use `file://`.

```python
import os, socket, threading, webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

def serve_viewer(out_dir):
    """Serve viewer via HTTP — always, even locally."""
    port = 8765
    for p in range(8765, 8800):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", p)) != 0:
                port = p
                break

    handler = partial(SimpleHTTPRequestHandler, directory=str(out_dir))
    server = HTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Determine the best URL to show the user
    is_ssh = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))
    if is_ssh:
        ssh_conn = os.environ.get("SSH_CONNECTION", "")
        parts = ssh_conn.split()
        ip = parts[2] if len(parts) >= 3 else socket.gethostname()
    else:
        ip = "localhost"

    url = f"http://{ip}:{port}/viewer.html"
    print(f"")
    print(f"========================================")
    print(f"  3D Viewer: {url}")
    print(f"  Auto-reloads on scene changes")
    print(f"========================================")
    print(f"")

    if not is_ssh:
        webbrowser.open(url)
```

**The script must call `serve_viewer(OUT_DIR)` at the end.** The HTTP
server runs as a daemon thread — it stays alive as long as the Python
process is running. The viewer auto-reloads when `scene.glb` is modified.

## Multi-Room House Layout

When the user requests multiple rooms or a house:

1. Define each room as a named region: `{id, name, x0, y0, x1, y1}`
2. Tag walls as `interior` (plasterboard) or `exterior` (concrete)
3. **Extend N/S walls by WALL_THICK on each end** to fill corner gaps
4. Interior walls allow ray transmission (8-15 dB loss per wall)
5. **Regenerate ALL exports** (XML, GLB, scene_state.json) from scratch —
   never patch single-room files
6. Place doors between adjacent rooms, windows on exterior walls only

## Z-Height Slider

For coverage visualization, compute coverage at multiple Z-heights
(default: 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5 m) and embed all
slices in the viewer. Add a range slider that:
- Swaps the heatmap canvas texture data
- Moves the heatPlane to the selected Y position
- Updates the stats panel with per-slice metrics

**JS ordering matters:** Create `heatPlane` mesh and `scene.add(heatPlane)`
before calling `updateHeatmap()` — calling update on a mesh that doesn't
exist yet causes a black screen (the texture has nowhere to render).

## Fallback (no browser, no HTTP)

If even HTTP serving fails, fall back to Read tool on `layout.png` and `coverage_*.png`.

## Progressive Coverage Updates (Large Scenes / Optimization Only)

For most requests, coverage computes fast enough that streaming is unnecessary.
**Only use this pattern when:**
- Outdoor scenes with large bboxes (>300m) where simulation takes >15 seconds
- TX optimization (iterating over multiple candidate positions)
- Multi-floor or multi-room sequential computation

**Do NOT use for:** Simple indoor scenes (CPU analytical runs in <1 second).

**How it works:** Decouple coverage from viewer.html. Write coverage as a
separate JSON file that the viewer fetches and updates dynamically.

**Script side** — write `coverage.json` after each pass/iteration:

```python
import json

def write_coverage_update(out_dir, coverage_grid, bounds, iteration=None):
    """Write coverage data as a separate file the viewer polls."""
    data = {
        "grid": coverage_grid.tolist(),
        "width": bounds["width"],
        "length": bounds["depth"],
        "iteration": iteration,  # null for single-pass, int for optimization
    }
    (out_dir / "coverage.json").write_text(json.dumps(data))
```

**Viewer side** — add this to the `<script>` block in viewer.html when
generating for a progressive-update scenario:

```javascript
// Poll coverage.json for updates (optimization / large scenes only)
let covMesh = null;
async function pollCoverage() {{
  try {{
    const res = await fetch('./coverage.json?t=' + Date.now());
    const data = await res.json();
    const grid = data.grid;
    const rows = grid.length, cols = grid[0].length;

    // Create or update heatmap texture
    const canvas = document.createElement('canvas');
    canvas.width = cols; canvas.height = rows;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(cols, rows);

    // Jet colormap: blue → green → yellow → red
    for (let r = 0; r < rows; r++) {{
      for (let c = 0; c < cols; c++) {{
        const v = grid[r][c];
        if (v < -200) {{ img.data[(r*cols+c)*4+3] = 0; continue; }}
        const t = Math.max(0, Math.min(1, (v + 100) / 70)); // -100...-30 dBm
        const i = (r * cols + c) * 4;
        img.data[i]   = Math.round(255 * Math.min(1, 1.5 - Math.abs(t - 0.75) * 4));
        img.data[i+1] = Math.round(255 * Math.min(1, 1.5 - Math.abs(t - 0.5) * 4));
        img.data[i+2] = Math.round(255 * Math.min(1, 1.5 - Math.abs(t - 0.25) * 4));
        img.data[i+3] = 180;
      }}
    }}
    ctx.putImageData(img, 0, 0);

    const texture = new THREE.CanvasTexture(canvas);
    if (covMesh) scene.remove(covMesh);
    const plane = new THREE.PlaneGeometry(data.width, data.length);
    const mat = new THREE.MeshBasicMaterial({{
      map: texture, transparent: true, side: THREE.DoubleSide,
    }});
    covMesh = new THREE.Mesh(plane, mat);
    covMesh.rotation.x = -Math.PI / 2;
    covMesh.position.set(data.width/2, 0.02, data.length/2);
    scene.add(covMesh);

    document.getElementById('colorbar').style.display = 'block';
  }} catch {{}}
}}

// Poll every 2s while optimization runs
const covInterval = setInterval(pollCoverage, 2000);
pollCoverage(); // initial load
```

**When to include this in generated viewer.html:**
- Optimization requests → YES (user watches TX position improve)
- Large outdoor coverage → YES (coarse-to-fine: 5m → 2m → 1m cell_size)
- Simple indoor room → NO (generate heatmap normally, embed in viewer)
