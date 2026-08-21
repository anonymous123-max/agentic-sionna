# Export Formats

## Contents

1. [Orientation Offset Rule](#orientation-offset-rule) — How orientation_offset applies to all exporters
2. [Format Matrix](#format-matrix) — Quick-reference table of all formats and their use cases
3. [Mitsuba XML](#mitsuba-xml) — Scene description for Sionna RT ray tracing backend
4. [GLTF/GLB](#gltfglb) — 3D viewer and interactive visualization export
5. [PNG Floor Plan](#png-floor-plan) — 2D top-down coverage overlay images
6. [ASCII Diagram](#ascii-diagram) — Text-based room layout for LLM context and debugging

The RF simulator supports multiple export formats, each optimized for a different downstream use case. This document specifies when to use each format, what it contains, and how to trigger it.

## Orientation Offset Rule

**ALL exporters must add `orientation_offset` to `theta` when computing
the final rotation of furniture.** This compensates for 3D models whose
"front" doesn't align with the skill's north convention (theta=0 = north).

```python
final_theta = furniture.position.theta + furniture.orientation_offset
```

Apply this in: Mitsuba XML (rotation matrix), GLTF (quaternion), PNG
(2D rotation), and Sionna scene building.

---

## Format Matrix

| Format | Extension | Primary Use | Includes Materials | Includes Mesh | 3D Viewable | Editable |
|--------|-----------|-------------|-------------------|---------------|-------------|----------|
| Mitsuba XML | `.xml` | Sionna RT re-import | ITU radio materials | Shape references | Via Sionna | XML editor |
| GLTF/GLB | `.gltf` / `.glb` | Blender, web viewers | PBR approximation | Embedded | Yes | Blender |
| PNG Floor Plan | `.png` | Documentation, papers | Color-coded by type | 2D footprint | No | Image editor |
| ASCII Diagram | `.txt` | Terminal display | Text symbols | Character art | No | Text editor |
| Coverage CSV | `.csv` | Data analysis | No | Grid only | No | Excel/Python |
| Scene State JSON | `.json` | App state save/restore | Full spec | No | No | JSON editor |

---

## Mitsuba XML (.xml)

### Purpose
Primary interchange format for Sionna RT. The exported XML can be loaded directly by `sionna.rt.load_scene()` for ray tracing simulation.

### Contents
- Scene integrator configuration
- ITU radio material definitions (concrete, glass, wood, metal, etc.)
- Shape references with material assignments
- Transmitter and receiver positions (as point sources)
- Transform matrices for positioned objects

### Structure
```xml
<?xml version="1.0" encoding="utf-8"?>
<scene version="2.1.0">
  <!-- Materials -->
  <bsdf type="itu-radio-material" id="mat-concrete">
    <string name="material_name" value="itu_concrete"/>
  </bsdf>

  <!-- Geometry -->
  <shape type="obj">
    <string name="filename" value="meshes/wall_north.obj"/>
    <ref id="mat-concrete"/>
    <transform name="to_world">
      <matrix value="1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1"/>
    </transform>
  </shape>

  <!-- Transmitters are placed via Sionna API after load -->
</scene>
```

### When to Use
- Running Sionna ray tracing simulations
- Sharing scenes with other Sionna users
- Archiving simulation-ready scenes

### Limitations
- Not human-readable for complex scenes
- Material properties are Sionna-specific (not standard PBR)
- Does not include simulation results

### Multi-Room Regeneration

When transitioning from single-room to multi-room layouts, **always regenerate the full XML from scene_state.json**. Never manually edit a single-room XML to add more rooms — missing geometry, stale wall definitions, or orphaned material references will cause Sionna `load_scene()` failures.

Common error when XML is stale:
```
RuntimeError: Found 1 unreferenced property in shape plugin of type "rectangle"
```
This means the XML contains properties that Mitsuba doesn't recognize (e.g., `<string name="id" value="wall_south"/>` — use the `id` attribute on the `<shape>` element instead).

---

## GLTF / GLB (.gltf / .glb)

### Purpose
Standard 3D format for visualization in Blender, web-based 3D viewers, and AR/VR applications.

### Contents
- Triangle meshes with vertex positions and normals
- PBR material approximation (baseColorFactor, metallicFactor, roughnessFactor)
- Scene hierarchy with named nodes
- Transform matrices

### Coordinate Convention

glTF is defined as Y-up, right-handed. Three.js loads glTF natively as Y-up.

**Build ALL GLB geometry directly in Y-up:**
- Floor: XZ plane at Y=0
- Walls: extend along +Y (height)
- Ceiling: XZ plane at Y=roomHeight
- Furniture position: scene `(x, y)` maps to GLB `(x, 0, y)` with height along Y

**3D-FUTURE models are already Y-up** and do NOT need the +90° X rotation for GLB. That rotation is only for Sionna XML (which is Z-up).

**Never build GLB in Z-up and apply a root transform to flip.** This causes confusion with Three.js cameras, lights, OrbitControls, and shadow projections which all assume Y-up.

### Material Mapping
ITU radio materials are approximated to PBR for visual fidelity:

| ITU Material | Base Color | Metallic | Roughness |
|-------------|------------|----------|-----------|
| itu_concrete | (0.65, 0.65, 0.65) | 0.0 | 0.9 |
| itu_glass | (0.85, 0.90, 0.95) | 0.0 | 0.1 |
| itu_wood | (0.55, 0.35, 0.20) | 0.0 | 0.7 |
| itu_metal | (0.80, 0.80, 0.82) | 1.0 | 0.3 |
| itu_plasterboard | (0.90, 0.88, 0.85) | 0.0 | 0.8 |
| itu_ceiling_board | (0.92, 0.90, 0.87) | 0.0 | 0.85 |
| itu_floorboard | (0.45, 0.30, 0.18) | 0.0 | 0.6 |

### When to Use
- 3D visualization in Blender for figures
- Web-based scene inspection
- Sharing with collaborators who don't have Sionna
- AR/VR demonstrations

### Variants
- `.gltf`: JSON + separate binary buffer + textures. Easier to debug.
- `.glb`: Single binary file. Easier to share.

---

## PNG Floor Plan (.png)

### Purpose
2D top-down view of the scene for documentation, papers, and quick visual reference.

### Contents
- Wall outlines with material-coded colors
- Furniture footprints (optional)
- Transmitter/receiver markers with labels
- Optional heatmap overlay (coverage, RSS, delay spread)
- Scale bar and north arrow
- Coordinate grid (optional)

### Color Coding
Walls and objects use material-based colors following design tokens:

| Element | Color |
|---------|-------|
| Concrete walls | `#868e96` (gray) |
| Glass walls | `#74c0fc` (light blue) |
| Wood walls | `#a68a64` (brown) |
| Metal walls | `#adb5bd` (silver) |
| Furniture | `#495057` (dark gray) |
| TX marker | `#4dabf7` (accent blue) |
| RX marker | `#22b8cf` (accent cyan) |
| Constraint | `#fcc419` (accent amber) |

### Heatmap Overlay Options
When exporting with a heatmap overlay, the coverage data is rendered as a semi-transparent color layer on top of the floor plan.

| Overlay | Colormap | Value Range |
|---------|----------|-------------|
| RSS (dBm) | viridis | [-120, tx_power] |
| Path gain (dB) | inferno | [-150, 0] |
| Delay spread (ns) | plasma | [0, auto] |
| SINR (dB) | coolwarm | [-10, 30] |

### When to Use
- Paper figures and presentations
- Quick scene validation
- Documentation and reports
- Embedding in markdown or LaTeX

---

## ASCII Diagram (.txt)

### Purpose
Text-only representation of the scene for terminal display and text-based communication.

### Symbol Key
```
#  Wall (concrete, default)
=  Glass wall
|  Wall (vertical)
-  Wall (horizontal)
+  Corner
.  Empty floor
T  Transmitter
R  Receiver
*  Constraint marker
F  Furniture
D  Door
W  Window
```

### Example Output
```
+------------------+
|..................|
|..T...............|
|..................|
|......F...........|
|..................|
|.........R.......|
|..................|
+--------==========+
         Window
```

### When to Use
- Terminal/CLI output
- Chat-based discussions
- Quick spatial debugging
- Environments without image rendering

---

## Coverage CSV (.csv)

### Purpose
Raw coverage data in tabular form for analysis in Excel, Python (pandas), R, or MATLAB.

### Columns
```csv
x_m,y_m,z_m,path_gain_linear,path_gain_db,rss_dbm,delay_spread_ns,sinr_db
0.5,0.5,1.5,1.23e-08,-79.1,-56.1,12.3,15.2
1.0,0.5,1.5,9.87e-09,-80.1,-57.1,14.1,13.8
...
```

### Metadata Header
The first lines of the CSV contain metadata as comments:

```csv
# RF Coverage Export
# frequency_hz: 3500000000
# tx_power_dbm: 23
# cell_size_m: 0.5
# max_depth: 5
# num_samples: 1000000
# num_tx: 1
# timestamp: 2026-03-25T14:30:00Z
x_m,y_m,z_m,path_gain_linear,path_gain_db,rss_dbm,delay_spread_ns,sinr_db
```

### When to Use
- Statistical analysis of coverage
- Comparison with measurements
- Input to external tools (MATLAB, R)
- Generating publication plots with custom styling

---

## Scene State JSON (.json)

### Purpose
Complete application state for save/restore functionality. This is the internal working format, not intended for external consumption.

### Structure
```json
{
  "version": 1,
  "scene_xml": "exports/scene.xml",
  "room": {
    "vertices": [[0,0], [10,0], [10,8], [0,8]],
    "height": 3.0,
    "materials": {
      "walls": "itu_concrete",
      "floor": "itu_floorboard",
      "ceiling": "itu_ceiling_board"
    }
  },
  "transmitters": [
    {
      "id": "tx-1",
      "name": "BS-1",
      "position": [2.0, 3.0, 2.5],
      "power_dbm": 23,
      "antenna": "iso"
    }
  ],
  "receivers": [
    {
      "id": "rx-1",
      "name": "UE-1",
      "position": [7.0, 5.0, 1.5],
      "antenna": "iso"
    }
  ],
  "furniture": [
    {
      "id": "furn-1",
      "type": "desk",
      "position": [5.0, 4.0, 0.0],
      "rotation": 0,
      "dimensions": [1.2, 0.6, 0.75],
      "material": "itu_wood"
    }
  ],
  "constraints": [
    {
      "id": "cst-1",
      "type": "min_rssi",
      "position": [7.0, 5.0, 1.5],
      "threshold": -80,
      "unit": "dBm",
      "status": "satisfied",
      "actual_value": -72.3
    }
  ],
  "materials": {
    "itu_concrete": {"relative_permittivity": 5.24, "conductivity": 0.0462},
    "itu_glass": {"relative_permittivity": 6.31, "conductivity": 0.0036}
  },
  "results": [
    {
      "id": "res-1",
      "type": "coverage_map",
      "timestamp": "2026-03-25T14:30:00Z",
      "params": {"frequency": 3.5e9, "max_depth": 5},
      "file": "results/coverage_001.npy",
      "stale": false
    }
  ]
}
```

### When to Use
- Saving/loading sessions
- Undo/redo state snapshots
- Debugging scene configuration
- Sharing exact simulation setups with collaborators

---

## Choosing the Right Format

Use this decision tree when the user asks to export:

| User Says | Export Format(s) |
|-----------|-----------------|
| "export for Sionna" / "save scene" | Mitsuba XML |
| "open in Blender" / "3D model" | GLTF/GLB |
| "figure for paper" / "screenshot" | PNG floor plan |
| "show me the layout" (in terminal) | ASCII diagram |
| "export coverage data" / "CSV" | Coverage CSV |
| "save my work" / "save session" | Scene State JSON |
| "export everything" | All of the above |
| "share with collaborator" | Mitsuba XML + Scene State JSON + PNG |

When in doubt, export Mitsuba XML (for simulation) and PNG (for visual reference) together. This covers both technical and communication needs.

---

## Export API Endpoints

All exports are triggered through the backend API:

| Endpoint | Method | Format | Parameters |
|----------|--------|--------|------------|
| `/api/scene/export/xml` | POST | Mitsuba XML | `output_path`, `include_tx` |
| `/api/scene/export/gltf` | POST | GLTF/GLB | `output_path`, `binary` (GLB if true) |
| `/api/scene/export/png` | POST | PNG | `output_path`, `width`, `height`, `overlay`, `dpi` |
| `/api/scene/export/ascii` | GET | ASCII | `width_chars` |
| `/api/scene/export/csv` | POST | Coverage CSV | `result_id`, `output_path` |
| `/api/scene/state` | GET | Scene State JSON | (none) |
| `/api/scene/state` | PUT | (restore) | Full state JSON body |
| `/api/scene/export/all` | POST | ZIP of all formats | `output_dir` |

## Related

- [scene-state-schema.md](scene-state-schema.md) — state schema mapped to export outputs
- [sionna-scene-editing.md](sionna-scene-editing.md) — loading exported scene data in Sionna
