# Sensible Defaults

Applied automatically when the user doesn't specify a value. Derived from
3GPP standards, industry tools (Ekahau, Ranplan, iBwave), and Sionna tutorials.

**Every default is overridable.** User-specified values always win.

Override rules:
1. **User value wins.** Never silently override.
2. **State assumptions.** "Placed TX at ceiling center (2.5, 2.0, 2.8) at 20 dBm."
3. **Omission = default.** "Create a bedroom" uses default dims; "create a 3×3 bedroom" overrides width/length only.
4. **"Don't"/"no"/"without"/"skip" disables a default.**
5. **Partial overrides are fine.** "Use 28 GHz" keeps all other defaults.

---

## Room

| Parameter | Default | Source |
|-----------|---------|--------|
| Shape | rectangle (width × length) | Simplest case |
| Width | 5.0 m | Standard office/bedroom |
| Length | 4.0 m | Standard office/bedroom |
| Height | 3.0 m | 3GPP Indoor Office |
| Door | south wall, pos=width/2, w=0.9 m | Building code |
| Walls | `itu_concrete` (ext), `itu_plasterboard` (int) | ITU-R P.2040-3 |
| Floor | `itu_concrete` / `itu_floorboard` | ITU-R P.2040-3 |
| Ceiling | `itu_ceiling_board` | ITU-R P.2040-3 |
| Windows | `itu_glass` (6mm single, 12mm double) | ITU-R P.2040-3 |
| Doors | `itu_wood` | ITU-R P.2040-3 |

**Prefer non-rectangular rooms.** When the user doesn't specify a shape,
generate an interesting polygon (L-shaped, alcove, bay window nook) rather
than a plain rectangle. Use `floor_polygon` (list of (x,y) vertices, CCW).
Width/length are the bounding box. Only use rectangle if explicitly asked
or room is very small (< 3m on a side).

## Multi-Room Layout

| Parameter | Default | Notes |
|-----------|---------|-------|
| Interior wall material | `itu_plasterboard` | Shared walls between rooms |
| Exterior wall material | `itu_concrete` | Outer perimeter |
| Interior wall thickness | 0.12 m | Thinner than exterior |
| Exterior wall thickness | 0.15 m | Standard |
| Corner overlap | N/S walls extend by WALL_THICK each end | Eliminates corner gaps |
| `is_interior` flag | true/false per wall in scene_state | Controls transmission vs reflection |

**When expanding from single-room to multi-room: regenerate ALL exports (XML, GLB, scene_state.json) from scratch.** Never patch a single-room XML.

## Furniture Dimensions

| Category | W × D × H (m) | Affinity |
|----------|----------------|----------|
| Bed | 1.5 × 2.0 × 0.6 | center/wall |
| Desk | 1.2 × 0.6 × 0.75 | wall |
| Chair | 0.5 × 0.5 × 0.9 | near desk |
| Sofa | 2.0 × 0.9 × 0.85 | wall |
| Wardrobe | 1.2 × 0.6 × 2.0 | wall |
| Nightstand | 0.5 × 0.4 × 0.55 | beside bed |
| Bookcase | 0.8 × 0.35 × 1.8 | wall |
| Cabinet | 0.8 × 0.45 × 0.9 | wall |
| Table | 1.0 × 1.0 × 0.75 | center |
| Coffee table | 1.0 × 0.6 × 0.45 | near sofa |
| TV stand | 1.2 × 0.4 × 0.5 | wall |
| Lamp | 0.3 × 0.3 × 1.5 | corner |

## Placement Optimization

Place furniture via optimizer (SciPy SLSQP or greedy grid), not randomly —
random placement produces unrealistic layouts where furniture floats in the
middle of rooms or overlaps walls.

**Cost weights:** collision=10, bounds=5, pathway=3, wall_affinity=2.
**Constraints:** door clearance 0.6m, wall margin 0.05m, max items = max(3, floor(area/3)).
**Bounds check:** `shapely.contains()` for polygon rooms, simple AABB for rectangles.
**Collision:** rotated AABB overlap detection between all furniture pairs.

Furniture source priority:
1. 3D-FUTURE catalog (real OBJ meshes with textures)
2. ABO catalog (GLB meshes, permissive license)
3. Fallback wireframe boxes (dimensions from table above)

## TX Auto-Placement

```
x = room_width/2,  y = room_length/2,  z = room_height - 0.2
```

Multi-room: one TX per room at ceiling center.

## Simulation

| Parameter | Default | Notes |
|-----------|---------|-------|
| frequency_hz | 3.5e9 | 5G NR n78 |
| tx_power_dbm | 20.0 | Indoor AP |
| rx_height | 1.5 m | 3GPP TR 38.901 |
| cell_size | 0.2 m (indoor), 1.0 m (outdoor) | |
| max_depth | 5 | Standard indoor |
| num_samples | 1,000,000 | Sionna default |
| coverage_threshold | -70 dBm | "Good" indoor 5G |
| Multi-Z range | 0.5–2.5 m | |
| Adaptive Z slices | 6 (≤3m), 8 (≤10m), 12 (>10m) | |

## Signal Quality Thresholds

**RSRP:** excellent ≥-80, good -80 to -90, fair -90 to -105, poor <-105 dBm
**SINR:** excellent ≥20, good 13-20, fair 0-13, poor <0 dB

## Heatmap

Colormap: Blue (#007AFF) → Cyan → Green (#4CD964) → Yellow (#FF9500) → Red (#FF3B30).
Always show colorbar with min/max and unit (dBm).

## Default Panels After Coverage

Always: heatmap overlay, colorbar, coverage stats (% above threshold, mean/min/max), histogram, method badge.
On request: PDP, BER, OFDM grid, ray paths.

## 3D Viewport

Camera: 50° FOV, position (6,6,5), ACESFilmic tone mapping, #050505 background.
Layers ON: walls, floor, furniture, TX, heatmap, colorbar, stats.
Layers OFF: rays, RX grid, measurements.
Coordinate readout on hover: `X: 3.42m  Y: 2.15m  Z: 1.50m | RSS: -62.3 dBm`

**Shadows:** `renderer.shadowMap.enabled = true`, `PCFSoftShadowMap`. All directional lights `castShadow = true`. Floor `receiveShadow = true`. All furniture meshes `castShadow = true`, `receiveShadow = true`.

**Lighting:** AmbientLight (intensity 1.8), PointLight inside room at `(w/2, h*0.7, l/2)` with intensity 2.5, DirectionalLight from `(6,12,4)` intensity 2.0 with shadows, fill DirectionalLight from `(-6,8,-4)` intensity 0.6.

**Furniture materials: PRESERVE original mesh colors from 3D-FUTURE GLB. Never override with custom MeshStandardMaterial. Only enable `castShadow`/`receiveShadow`. Only walls get translucent glass-box treatment.**

## BER/OFDM (When Requested)

SNR: -2 to 10 dB (25 pts). LDPC +6 dB, Polar +5 dB, floor 1e-7.
OFDM: 14 symbols, 128 FFT, 30 kHz SCS, CP=20.

## Ray Visualization

24 rays, 3 bounces, 6-color cycling.

Clip rays at room boundaries — rays that escape through doors or windows
produce visual artifacts (lines shooting into empty space) that confuse
users about where signal actually propagates. For each ray segment:

1. Cast ray from current position in current direction.
2. Find nearest intersection: wall segment, floor (y=0), ceiling (y=height),
   or furniture AABB.
3. Reflect: `dir_reflected = dir - 2 * dot(dir, normal) * normal`
4. If no intersection found within room bounds, terminate the ray.
5. For polygon rooms, test ray against each wall segment edge
   (use `_ray_wall_segment_intersect` — 2D line intersect extruded to height).

**Ray intersection: use triangle-level mesh intersection, NOT AABB bounding boxes.** For Python (trimesh): `mesh.ray.intersects_location()` (requires `pip install rtree`). For JS (Three.js): `Raycaster` which does triangle intersection by default.

**Ray transmission through interior walls:** Interior walls (plasterboard) allow both reflection and transmission. When a ray hits an interior wall, it transmits through with material-dependent loss. Exterior walls, floor, ceiling: reflect only.

| Material | Visual Loss (viewer) | Notes |
|----------|---------------------|-------|
| Plasterboard | 8 dB | Simplified for ray coloring. Actual propagation loss is frequency-dependent (2-8 dB); see `physics-validation.md` |
| Concrete | 15 dB | |
| Glass | 3 dB | Standard clear glass; IRR/low-E glass is 25+ dB |
| Wood | 10 dB | |
| Metal | 40 dB | Effectively opaque |

**Signal strength coloring per segment:** cumulative loss = FSPL + 8 dB per reflection + material transmission loss. Color mapping: green (#00ff88) for 0-20 dB loss, yellow/amber (#ffaa00) for 20-40 dB, red (#ff4466) for >40 dB.

## Window Placement (IRC Building Codes)

Auto-place windows following International Residential Code when user
doesn't specify window positions:

| Room type | Requirement | Source |
|-----------|------------|--------|
| Bedroom | ≥1 egress window, ≥5.7 sq ft opening, sill ≤44" from floor | IRC R310 |
| Living room | Glazing area ≥ 8% of floor area | IRC R303.1 |
| Bathroom | Optional; if present, ≥3 sq ft glazing | IRC R303.3 |
| Kitchen | Glazing area ≥ 8% of floor area | IRC R303.1 |

Place windows on exterior walls, avoid walls where furniture blocks them.
Distribute evenly for natural light. Window material: `itu_glass`.

## Multi-Height Coverage Slices

Compute coverage at multiple Z heights simultaneously:

```python
heights = [0.75, 1.2, 2.8]  # desk, standing, ceiling
for h in heights:
    rm = rm_solver(scene, cell_size=[1,1], samples_per_tx=10**6,
                   center=[cx, cy, h], orientation=[0,0,0], size=[w,l])
```

Frontend: stacked heatmap planes at each height. Height slider in panel
selects which slice is opaque (others go semi-transparent at 0.2 opacity).

## Scene Save/Load/Delete

Persist scenes in `scenes/<scene_id>/` directory:
- **Save:** scene_state.json + scene.xml + last simulation results + metadata.json
- **Load:** deserialize state, regenerate XML if needed, restore frontend
- **Delete:** remove directory after user confirmation
- **List:** `GET /api/scenes` returns all saved scenes (name, date, room count)

Frontend: scene browser panel with load/delete buttons.

## Mobility and Doppler

Set velocity on receivers for Doppler analysis:

```python
rx = scene.get("rx")
rx.velocity = [1.5, 0, 0]  # 1.5 m/s walking along X
paths = solver(scene, max_depth=5, num_samples=10**6)
# paths.doppler → Doppler frequency shifts per path
```

Frontend: animate RX marker along velocity vector, show delay-Doppler
spectrum as 2D heatmap (see component-guidelines Pattern 10).

## Target Coverage Optimization

When user says "achieve 80% coverage above -70 dBm":

1. Define objective: maximize fraction of cells where RSS > threshold
2. Variables: TX position (x, y, z), optionally power and antenna orientation
3. Use differentiable radio map to compute gradient of coverage w.r.t. TX params
4. Gradient ascent with constraint (TX inside room, above furniture)
5. Stream intermediate coverage % to an optimization progress chart
6. Stop when target met or convergence (Δ < 0.1% for 3 iterations)

## Export

Default all: XML + PNG + GLTF + JSON. CSV only after sim. ASCII on request.
Output: `outputs/<scene_id>/`

**Coordinate conventions:**
- **GLB:** Y-up (glTF standard). Floor in XZ plane at Y=0. Height along +Y.
- **XML (Sionna/Mitsuba):** Z-up. Floor in XY plane at Z=0. Height along +Z.
- **3D-FUTURE models:** Y-up natively. No rotation for GLB; +90° X rotation for XML only.

## Frequency-Material Limits

Indoor (concrete/wood/glass): ≤100 GHz. Outdoor (ground): ≤10 GHz.
Brick: ≤40 GHz. Warn if exceeded.

## Undo/Redo

50-op stack. Ctrl+Z / Ctrl+Shift+Z. Captured on drag start, param change, add/remove.

## Related

- [script-guidelines.md](script-guidelines.md) — parameter bounds enforced in scripts
- [physics-validation.md](physics-validation.md) — physical bounds for validation checks
- [antenna-patterns.md](antenna-patterns.md) — default antenna array configurations
- [scene-state-schema.md](scene-state-schema.md) — default values in scene state

## Glossary — often-confused noise quantities

| Term                  | Formula              | Typical value     |
|-----------------------|----------------------|-------------------|
| Noise PSD (kT)        | kT, T=290K           | −174 dBm/Hz       |
| Thermal noise power   | kTB                  | −101 dBm (20 MHz) |
| Receiver noise floor  | kTB + NF             | −94 dBm (NF=7 dB) |

- When the task says "thermal noise power" → use **kTB alone**, no NF.
- When the task says "noise floor" or "receiver sensitivity" → **add NF**.
- Eb/N0 and Es/N0 are noise *ratios*, not absolute powers — do not confuse
  them with the table above.
