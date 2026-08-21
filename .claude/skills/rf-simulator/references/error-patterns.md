# Error Patterns

## Contents

1. [Backend / Sionna Errors](#backend--sionna-errors) — GPU OOM, Mitsuba plugin, and ray tracing runtime failures
2. [Frontend / React Errors](#frontend--react-errors) — Viewer rendering, GLB loading, and UI component issues
3. [Adding New Error Patterns](#adding-new-error-patterns) — Template for documenting new failure modes and fixes
4. [RadioMapSolver Errors](#radiomapsolver-errors) — CPU fallback solver edge cases and numerical issues

Known errors and auto-fix strategies for the RF simulator. This file grows over time as the QA validator encounters new failure modes.

Each entry follows a standard format:
- **Error Signature**: The exception type and message fragment that identifies this error.
- **Cause**: Why it happens.
- **Fix**: How to resolve it.
- **Auto-fixable**: Whether the system can fix it without user intervention.

---

## Backend / Sionna Errors

### 1. ITU Radio Material Plugin Failure

**Error Signature:**
```
RuntimeError: could not instantiate bsdf plugin of type "itu-radio-material"
```

**Cause:** Mitsuba does not know about Sionna's custom material plugins. This happens when `mi.load_dict()` or `mi.load_file()` is called before Sionna's RT module has registered its plugins.

**Fix:** Import `sionna.rt` before any scene loading call. The import triggers plugin registration as a side effect.

```python
# WRONG - fails
import mitsuba as mi
scene = mi.load_file("scene.xml")

# CORRECT - works
import sionna.rt  # registers ITU material plugins with Mitsuba
scene = sionna.rt.load_scene("scene.xml")
```

**Auto-fixable:** Yes. Prepend `import sionna.rt` before the failing line.

---

### 2. Object Not Found After Scene Load

**Error Signature:**
```
scene.get("object_name") returns None
# or
KeyError: "object_name"
```

**Cause:** By default, Sionna merges all shapes in the scene into a single object for performance. Individual named objects become inaccessible.

**Fix:** Load the scene with `merge_shapes=False` to preserve individual object identities.

```python
# WRONG - objects merged, individual access fails
scene = sionna.rt.load_scene("scene.xml")
wall = scene.get("north_wall")  # None

# CORRECT - objects preserved
scene = sionna.rt.load_scene("scene.xml", merge_shapes=False)
wall = scene.get("north_wall")  # SceneObject
```

**Auto-fixable:** Yes. Add `merge_shapes=False` to the `load_scene()` call.

---

### 3. All-Zero Path Gain

**Error Signature:**
```
WARNING: ALL_ZERO_PATH_GAIN: No energy reached any receiver cell.
# or
numpy path_gain array is all zeros
```

**Cause:** One of three things:
1. Transmitter position is outside the scene geometry (rays start in empty space).
2. `max_depth` is too low for the environment (e.g., max_depth=1 in a room with no LOS).
3. `scene.frequency` was not set before computing coverage.

**Fix:**
1. Verify TX coordinates are inside a room polygon. Print `scene.transmitters` and compare against scene bounds.
2. Increase `max_depth` to at least 3 for indoor scenes, 5 for complex NLOS.
3. Set `scene.frequency = 3.5e9` (or desired frequency) before calling `coverage_map()`.

```python
# Diagnostic checklist
print("TX positions:", [(tx.name, tx.position) for tx in scene.transmitters.values()])
print("Scene bounds:", scene.center, scene.size)
print("Frequency:", scene.frequency)
print("Max depth:", max_depth)
```

**Auto-fixable:** Partially. Can auto-check TX position and frequency. Cannot determine correct max_depth without domain knowledge.

---

### 4. RSS Exceeds TX Power

**Error Signature:**
```
WARNING: RSS_EXCEEDS_TX: Max RSS (X dBm) exceeds TX power (Y dBm).
```

**Cause:** Numerical precision issue in path gain computation, usually near the transmitter where path gain approaches 0 dB. Can also occur with constructive multipath interference in very small scenes.

**Fix:**
1. Clamp path gain to maximum 0 dB: `path_gain = np.minimum(path_gain, 1.0)`.
2. Check for NaN or Inf in the path gain array before conversion.
3. If the scene is very small (< 2m), increase cell_size to avoid near-field artifacts.

```python
path_gain = np.minimum(cm.path_gain.numpy(), 1.0)  # Clamp to 0 dB max
path_gain = np.nan_to_num(path_gain, nan=0.0, posinf=1.0, neginf=0.0)
```

**Auto-fixable:** Yes. Apply clamping automatically and log a notice.

---

### 5. Mitsuba Variant Error

**Error Signature:**
```
RuntimeError: The variant "..." has already been set.
# or
RuntimeError: Mitsuba variant must be set before importing...
```

**Cause:** Manually calling `mi.set_variant()` conflicts with Sionna's variant management. Sionna sets the variant internally based on whether CUDA is available.

**Fix:** Never call `mi.set_variant()` manually. Let Sionna handle it.

```python
# WRONG
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")  # conflicts with Sionna

# CORRECT
import sionna.rt  # handles variant selection automatically
```

**Auto-fixable:** Yes. Remove the `mi.set_variant()` call.

---

### 6. CUDA Out of Memory

**Error Signature:**
```
RuntimeError: CUDA out of memory. Tried to allocate X MiB...
# or
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**Cause:** The ray tracing computation exceeds available GPU memory. Typical triggers:
- `num_samples` too high (> 1e7 on 8GB GPU).
- `cell_size` too small with a large scene (creates millions of cells).
- Multiple transmitters computed simultaneously.

**Fix (ordered by preference):**
1. Reduce `num_samples`: try `1e5` for preview, `1e6` for production.
2. Increase `cell_size`: try `1.0` for indoor, `5.0` for outdoor.
3. Compute one transmitter at a time instead of all at once.
4. Free GPU memory: `torch.cuda.empty_cache()` before the computation.
5. Last resort: fall back to CPU computation (50x slower).

```python
import torch

# Free cached memory
torch.cuda.empty_cache()

# Reduce parameters
cm = scene.coverage_map(
    max_depth=3,
    num_samples=int(1e5),       # reduced from 1e7
    cm_cell_size=(1.0, 1.0),    # reduced resolution
)
```

**Auto-fixable:** Yes. Halve `num_samples` and retry. If still OOM, double `cell_size` and retry. Log each fallback step.

---

## Frontend / React Errors

### 7. React ErrorBoundary Crash

**Error Signature:**
```
Error: Minified React error #...
# or
Uncaught TypeError: Cannot read properties of undefined (reading '...')
```

**Cause:** A component received unexpected props, usually because:
1. Backend API response shape changed.
2. A data array is empty when the component assumes non-empty.
3. A nullable field was not checked before accessing nested properties.

**Fix:**
1. Add null checks: `data?.field ?? defaultValue`.
2. Verify the API response shape matches the TypeScript interface.
3. Add a loading state for async data.
4. Wrap the component in an ErrorBoundary with a fallback UI.

```tsx
// Always guard against undefined data
const value = result?.rss_dbm?.[0] ?? -Infinity;

// ErrorBoundary wrapper
<ErrorBoundary fallback={<ErrorCard message="Failed to render coverage map" />}>
  <HeatmapOverlay data={data} />
</ErrorBoundary>
```

**Auto-fixable:** No. Requires manual investigation of the specific prop.

---

### 8. WebSocket Disconnected

**Error Signature:**
```
WebSocket connection to 'ws://localhost:PORT/ws' failed
# or
WebSocket is already in CLOSING or CLOSED state
```

**Cause:**
1. Backend server is not running.
2. Port mismatch between frontend config and backend.
3. Backend crashed during a long simulation.
4. Firewall or proxy blocking WebSocket upgrade.

**Fix:**
1. Check that the backend server is running: `curl http://localhost:8000/health`.
2. Verify the port in the frontend `.env` matches the backend port.
3. Check backend logs for crash traces.
4. Implement automatic reconnection with exponential backoff.

```typescript
// Auto-reconnect pattern
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 1000;

function connectWebSocket(url: string, attempt = 0): WebSocket {
  const ws = new WebSocket(url);

  ws.onclose = () => {
    if (attempt < MAX_RETRIES) {
      const delay = BASE_DELAY_MS * Math.pow(2, attempt);
      console.warn(`WebSocket closed. Reconnecting in ${delay}ms (attempt ${attempt + 1})`);
      setTimeout(() => connectWebSocket(url, attempt + 1), delay);
    } else {
      console.error("WebSocket: max retries reached. Giving up.");
    }
  };

  return ws;
}
```

**Auto-fixable:** Partially. Auto-reconnect handles transient failures. Server-down requires user action.

---

### 9. Three.js Geometry Disposal Warning

**Error Signature:**
```
THREE.WebGLRenderer: Context Lost
# or
Memory usage growing continuously (visible in Chrome DevTools)
```

**Cause:** React component re-renders create new Three.js geometries and materials without disposing the old ones.

**Fix:** Dispose geometries and materials in a `useEffect` cleanup function.

```tsx
useEffect(() => {
  const geo = new THREE.PlaneGeometry(width, height, segX, segY);
  meshRef.current.geometry = geo;

  return () => {
    geo.dispose();
  };
}, [width, height, segX, segY]);
```

**Auto-fixable:** No. Requires code review of the component lifecycle.

---

### 10. Coverage Data Shape Mismatch

**Error Signature:**
```
TypeError: data.length is not a function
# or
RangeError: offset is out of bounds
```

**Cause:** The coverage data array from the backend does not match the grid dimensions expected by the heatmap component. Common when:
1. Backend changed `cell_size` without notifying frontend.
2. Grid dimensions were rounded differently on backend vs. frontend.

**Fix:** Always pass grid dimensions alongside the data array. Never compute grid shape independently on the frontend.

```typescript
// Backend response should include:
interface CoverageResult {
  data: number[];          // flattened 2D array
  grid_width: number;      // columns
  grid_height: number;     // rows
  cell_size: number;       // meters
  origin: [number, number]; // world coordinates of grid[0][0]
}

// Frontend uses backend-provided dimensions, never computes its own
const { data, grid_width, grid_height } = coverageResult;
```

**Auto-fixable:** No. Requires backend/frontend contract alignment.

---

### 11. GLB Coordinate System Mismatch (Z-Up in Y-Up Format)

**Error Signature:**
```
Furniture appears sideways, floor is vertical, room rotated 90 degrees in 3D viewer
```

**Cause:** GLB geometry was built in Z-up convention (Sionna/scene coords) but glTF and Three.js expect Y-up. The entire scene appears rotated 90 degrees.

**Fix:** Build GLB geometry directly in Y-up convention. Floor in XZ plane at Y=0, walls extend along +Y, ceiling at Y=roomHeight. 3D-FUTURE models are Y-up natively — do NOT apply the +90° X rotation when loading into GLB (that rotation is only needed for Sionna XML which is Z-up). Never build in Z-up and apply a root transform to flip.

**Auto-fixable:** No. Requires regenerating the GLB with correct coordinate convention.

---

### 12. Visible Corner Gaps in Room Walls

**Error Signature:**
```
Light leaks or visible gaps at room corners in 3D viewer
```

**Cause:** Adjacent wall boxes meet at exact endpoints without overlap. Floating-point precision and the wall thickness offset leave sub-pixel gaps at corners.

**Fix:** Extend N/S walls by `WALL_THICK` on each end so they overlap the E/W wall endpoints. North/south wall total length = `room_width + 2 * WALL_THICK`. This fills the corners completely.

```python
full_w = ROOM_W + 2 * WALL_THICK
wall_south = trimesh.creation.box(extents=[full_w, WALL_THICK, ROOM_H])
```

**Auto-fixable:** Yes. Adjust wall dimension calculations.

---

### 13. Rays Reflect Off Bounding Boxes Instead of Mesh Surface

**Error Signature:**
```
Ray paths bounce off invisible box edges surrounding furniture rather than the actual mesh surface
```

**Cause:** Ray intersection computed against AABB (axis-aligned bounding box) rather than per-triangle mesh geometry. Visible as rays bouncing off air near curved or irregular furniture.

**Fix:** Use trimesh triangle-level ray-mesh intersection: `combined_mesh.ray.intersects_location(origins, dirs)`. Requires `pip install rtree` for the spatial index. In Three.js, `Raycaster` does triangle intersection by default on Mesh objects. Never use AABB for ray path visualization.

**Auto-fixable:** No. Requires changing intersection logic from AABB planes to mesh triangles.

---

### 14. Black Screen — Heatmap Update Before Plane Creation

**Error Signature:**
```
Black screen in viewer; console shows "Cannot read properties of undefined" or "Cannot set property 'y' of undefined"
```

**Cause:** `updateHeatmap()` or similar function called before `heatPlane` mesh is created and added to the Three.js scene. The function tries to set `heatPlane.position.y` on an undefined object, crashing the entire script block.

**Fix:** Ensure object creation order: PlaneGeometry → CanvasTexture → MeshBasicMaterial → Mesh → `scene.add(mesh)` → THEN call `updateHeatmap()`. General rule: never call update functions that reference Three.js objects before those objects exist.

```javascript
// WRONG — crashes
updateHeatmap(data, 1.0);  // references heatPlane
const heatPlane = new THREE.Mesh(...);

// CORRECT
const heatPlane = new THREE.Mesh(...);
scene.add(heatPlane);
updateHeatmap(data, 1.0);  // now safe
```

**Auto-fixable:** Yes. Reorder function calls so creation precedes usage.

---

## Adding New Error Patterns

When you encounter a new error during QA validation:

1. Capture the full error signature (exception type + message).
2. Identify the root cause through log analysis.
3. Document the fix with a code example.
4. Determine if it is auto-fixable.
5. Add it to this file in the appropriate section (Backend or Frontend).
6. If auto-fixable, implement the fix in the QA validator's auto-repair pipeline.

Use this template:

```markdown
### N. Short Error Title

**Error Signature:**
\`\`\`
ExactErrorMessage
\`\`\`

**Cause:** Why it happens.

**Fix:** How to resolve it, with code example.

**Auto-fixable:** Yes/No/Partially. Details.
```

---

## RadioMapSolver Errors

### RadioMapSolver: Missing orientation Parameter

**Error Signature:**
```
ValueError: If one of `center`, `orientation`, or `size` is not None, then all of them must not be None.
```

**Cause:** `center`, `orientation`, and `size` are an all-or-nothing group in `RadioMapSolver.__call__()`. Providing `center` and `size` but omitting `orientation` triggers this error.

**Fix:** Always provide all three parameters together:
```python
radio_map = rm_solver(
    scene=scene,
    cell_size=[0.2, 0.2],
    samples_per_tx=10_000_000,
    max_depth=5,
    center=[4.0, 3.0, 1.5],
    orientation=[0.0, 0.0, 0.0],  # MUST be provided — [0,0,0] = horizontal plane
    size=[8.0, 6.0],
)
```

**Auto-fixable:** Yes. Add `orientation=[0.0, 0.0, 0.0]` to the call.

### RadioMapSolver: AttributeError on .value

**Error Signature:**
```
AttributeError: 'PlanarRadioMap' object has no attribute 'value'
```

**Cause:** `RadioMapSolver` returns a `PlanarRadioMap` object, not a tensor. There is no `.value` attribute. This was a common mistake from v1.x documentation.

**Fix:** Use `.path_gain` (linear scale, unitless) or `.rss` (linear scale, Watts):
```python
# WRONG: radio_map.value.numpy()
# RIGHT:
import numpy as np
pg = np.array(radio_map.path_gain)  # [num_tx, cells_y, cells_x], linear
rss_dbm = TX_POWER_DBM + 10.0 * np.log10(np.where(pg > 0, pg, np.nan))
```

**Auto-fixable:** Yes. Replace `.value` with `.path_gain` and add dBm conversion.

### Mitsuba Cleanup Crash (RTX 5090 / Blackwell)

**Error Signature:**
```
free(): invalid pointer
Aborted (core dumped)
```

**Cause:** Mitsuba's C++ destructor triggers a double-free on process exit when running on RTX 5090 (compute capability sm_120 / Blackwell). The simulation completes correctly — this only happens during cleanup.

**Fix:** End every Sionna script with `os._exit(0)` to skip Python's normal cleanup:
```python
# At the very end of the script, after all output is saved:
import os
os._exit(0)
```

**Auto-fixable:** Yes. Append `os._exit(0)` to every generated script.

## Related

- [script-guidelines.md](script-guidelines.md) — error handling patterns and SceneError class
- [sionna-v2-api.md](sionna-v2-api.md) — API calls that trigger these errors
