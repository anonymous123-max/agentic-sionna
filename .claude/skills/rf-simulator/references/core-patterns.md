# Core Patterns

Code patterns for generated scripts. Copy and adapt as needed.

---

## 3D-FUTURE Furniture Catalog

See `references/data-sources.md` for dataset discovery (`find_dataset()`),
download instructions, and category normalization.

**Key points:**
- `model_info.json` has null categories — always use `(m.get("category") or "").lower().strip()`
- No plain "chair" — use "dining chair", "armchair", etc.
- 3D-FUTURE = Y-up. Rotate +90 deg around X for Z-up (Sionna). **Never -90 deg** (upside down).

### Category Reference (exact 3D-FUTURE names)

| Category | Count | Use |
|----------|-------|-----|
| `pendant lamp` | 1152 | ceiling lighting |
| `three-seat / multi-seat sofa` | 852 | living room |
| `armchair` | 751 | accent seating |
| `corner/side table` | 718 | side tables |
| `coffee table` | 684 | living room |
| `lounge chair / cafe chair / office chair` | 585 | office/lounge |
| `dining chair` | 543 | dining room |
| `nightstand` | 538 | bedroom |
| `tv stand` | 529 | living room |
| `dining table` | 517 | dining room |
| `wardrobe` | 476 | bedroom |
| `bookcase / jewelry armoire` | 439 | storage |
| `desk` | 356 | office/bedroom |
| `king-size bed` | 440 | master bedroom |
| `single bed` / `double bed` | 135/141 | bedroom |

### Room Type → Furniture

| Room | Categories |
|------|-----------|
| bedroom | king-size/single bed, nightstand, wardrobe, desk |
| living room | multi-seat sofa, coffee table, tv stand, bookcase, armchair, floor lamp |
| office | desk, office chair, bookcase, drawer chest |
| dining room | dining table, dining chair (x4-6), sideboard |

---

## Load Catalog

```python
import json, random
from pathlib import Path

def load_catalog():
    if THREED_FUTURE is None:
        return {}
    with open(THREED_FUTURE / "model_info.json") as f:
        models = json.load(f)
    by_cat = {}
    for m in models:
        cat = (m.get("category") or "").lower().strip()
        if cat and (THREED_FUTURE / m["model_id"] / "raw_model.obj").exists():
            by_cat.setdefault(cat, []).append(m)
    return by_cat

def get_model_dims(model_id):
    xs, ys, zs = [], [], []
    with open(THREED_FUTURE / model_id / "raw_model.obj") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                xs.append(float(p[1])); ys.append(float(p[2])); zs.append(float(p[3]))
    return {"width": max(xs)-min(xs), "height": max(ys)-min(ys), "depth": max(zs)-min(zs)}

def find_model(catalog, cat_searches, max_w=99, max_d=99):
    for cat_name in cat_searches:
        suitable = [(m, get_model_dims(m["model_id"]))
                    for m in catalog.get(cat_name, [])
                    if (dims := get_model_dims(m["model_id"]))
                    and dims["width"] <= max_w * 1.3 and dims["depth"] <= max_d * 1.3]
        if suitable:
            return random.choice(suitable)
    return None, None
```

---

## Collision-Aware Placement

```python
def rects_overlap(ax, ay, aw, ad, bx, by, bw, bd, margin=0.1):
    return not (ax+aw/2+margin <= bx-bw/2 or bx+bw/2+margin <= ax-aw/2 or
                ay+ad/2+margin <= by-bd/2 or by+bd/2+margin <= ay-ad/2)

def in_bounds(x, y, w, d, room_w, room_l, margin=0.05):
    return (x-w/2 >= margin and x+w/2 <= room_w-margin and
            y-d/2 >= margin and y+d/2 <= room_l-margin)

# Resolve collisions with jitter:
for dx in [0, 0.5, -0.5, 1.0, -1.0, 1.5]:
    for dy in [0, 0.5, -0.5, 1.0, -1.0, 1.5]:
        if in_bounds(x+dx,y+dy,w,d,ROOM_W,ROOM_L) and not has_collision(x+dx,y+dy,w,d,placed):
            x, y = x+dx, y+dy; break
```

---

## GLB Generation (Real Meshes)

See `references/export-formats.md` for coordinate system rules (GLB=Y-up,
XML=Z-up) and orientation_offset handling.

```python
import trimesh, numpy as np

scene_glb = trimesh.Scene()

# Floor (Y-up: height along Y)
floor = trimesh.creation.box(extents=[room_w, 0.02, room_l])
floor.visual.face_colors = [40, 40, 50, 255]
floor.apply_translation([room_w/2, -0.01, room_l/2])
scene_glb.add_geometry(floor, node_name="floor")

# Walls — name must start with "wall" so the viewer applies translucent material
for wall in walls:
    w_mesh = trimesh.creation.box(extents=[wall["length"], wall["height"], 0.15])
    w_mesh.visual.face_colors = [80, 80, 100, 25]
    node_name = f"wall_{wall['id']}" if not wall["id"].startswith("wall") else wall["id"]
    scene_glb.add_geometry(w_mesh, node_name=node_name)

# Furniture — real OBJ meshes with box fallback
for item in furniture:
    model_path = THREED_FUTURE / item["model_id"] / "raw_model.obj"
    if model_path.exists():
        try:
            mesh = trimesh.load(str(model_path), force='mesh')
            # 3D-FUTURE Y-up matches GLB Y-up — no rotation needed
            mesh.apply_translation([item["x"], 0, item["y"]])
            scene_glb.add_geometry(mesh, node_name=item["id"])
        except Exception:
            _add_box_fallback(scene_glb, item)
    else:
        _add_box_fallback(scene_glb, item)

scene_glb.export(str(out_dir / "scene.glb"), file_type="glb")
```

---

## Sionna XML (Mitsuba 3.0)

```python
from xml.etree.ElementTree import Element, SubElement
root = Element("scene", version="3.0.0")
SubElement(root, "integrator", type="path")
# Materials: wall_mat=concrete, floor_mat=concrete, glass_mat=glass, wood_mat=wood
# Geometry: type="rectangle" with scale+rotate+translate
# Furniture: type="obj" with rotate x=+90 (Y-up→Z-up) + rotate z + translate
```

---

## Physics Validation

Call after computing coverage, before generating viewer. Prints warnings only.
See `references/physics-validation.md` for formulas.

```python
def validate_coverage(coverage_grid, tx_power_dbm, tx_pos, cell_size, freq_hz):
    valid = coverage_grid[coverage_grid > -200]
    if len(valid) == 0:
        warnings.warn("Coverage map is all-zero"); return

    # RSS must not exceed TX power
    if float(np.max(valid)) > tx_power_dbm + 0.1:
        warnings.warn(f"Max RSS exceeds TX power")

    # Path loss should increase with distance (Spearman rho > 0.5)
    # No path loss below free-space (FSPL - 5 dB tolerance)
    # Coverage should be spatially continuous (< 5% outlier cells)
    # See references/physics-validation.md for full implementation
```

---

## Window Placement

- Bedroom: egress window (IRC R310) on exterior wall
- Living room: ~8% glazing ratio of floor area
- In XML: `type="rectangle"` with `glass_mat`

---

## Ground Truth Calibration

Use `scipy.optimize.minimize` (Nelder-Mead) to fit tx_power and wall_loss
to user-provided measurements. See `references/physics-validation.md` for
analytical cross-check formulas.

```python
def calibrate_scene(measurements, sim_func, initial_params):
    def objective(x):
        params = {**initial_params, "tx_power_dbm": x[0], "wall_loss_db": x[1]}
        grid = sim_func(params)
        return sum((grid[r,c] - m["rss_dbm"])**2 for m in measurements
                   for r,c in [grid_index(m["position"], params["cell_size"])]) / len(measurements)
    result = minimize(objective, x0=[initial_params["tx_power_dbm"], 15.0], method="Nelder-Mead")
    return {"tx_power_dbm": result.x[0], "wall_loss_db": result.x[1], "rmse_db": result.fun**0.5}
```

---

## Outdoor Scenes (OpenStreetMap)

```python
import osmnx as ox
from pyproj import Transformer
gdf = ox.features.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})
transformer = Transformer.from_crs("EPSG:4326", f"EPSG:326{zone}", always_xy=True)
```

**CRITICAL — NaN handling:** OSM tags use `float('nan')`, not `None`.
```python
import math
def safe_float(val):
    if val is None: return None
    if isinstance(val, float) and math.isnan(val): return None
    try: return float(str(val).replace("m","").strip())
    except: return None
```

**Rules:** Always use bbox (not place names). Keep bbox ~300-500m.
Height heuristics: residential=8m, apartments=15m, office=20m.

See `references/defaults.md` for simulation parameter defaults and
`references/cpu-fallback.md` for the CPU analytical model.
