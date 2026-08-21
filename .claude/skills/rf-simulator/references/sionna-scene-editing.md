# Sionna Scene Editing: Gotchas and Best Practices

## Contents

1. [Critical Rules](#critical-rules) — Must-follow constraints for scene manipulation
2. [What scene.add() Accepts](#what-sceneadd-accepts) — TX, RX, materials only — no geometry
3. [The merge_shapes Trap](#the-merge_shapes-trap) — When objects become inaccessible
4. [HolderMaterial Registration](#holdermaterial--itu-radio-material-registration) — Import order for material plugins
5. [Complete Recommended Workflow](#complete-recommended-workflow) — End-to-end scene setup pattern
6. [Adding Geometry](#adding-geometry-xml-modification) — How to add objects via XML reload

## Critical Rules

1. **`scene.add()` only works for TX, RX, RadioMaterial, RIS, Camera.** It does NOT work for `SceneObject`. There is no runtime API to add geometry.
2. **To add geometry, modify the Mitsuba XML and reload the scene.**
3. **Use `merge_shapes=False`** when individual objects need addressing.
4. **`import sionna.rt` before `load_scene()`** to register ITU material plugins.
5. **Never call `mi.set_variant()`** manually.
6. **Set `scene.frequency` before running solvers.**

---

## What `scene.add()` Accepts

```python
# These work:
scene.add(sionna.rt.Transmitter("tx", position=[0, 0, 3]))     # OK
scene.add(sionna.rt.Receiver("rx", position=[10, 10, 1.5]))    # OK
scene.add(sionna.rt.RadioMaterial("my_mat", ...))               # OK
scene.add(sionna.rt.RIS("ris", ...))                            # OK
scene.add(sionna.rt.Camera("cam", position=[0, 0, 50]))         # OK

# This does NOT work:
scene.add(sionna.rt.SceneObject(...))  # WILL FAIL / NOT SUPPORTED
```

---

## The `merge_shapes` Trap

### Default Behavior (merge_shapes=True)

When `merge_shapes=True` (the default), Sionna merges all shapes that share the same material into a single combined shape. This is efficient for rendering but means you cannot address individual objects.

```python
# Default: shapes merged by material
scene = sionna.rt.load_scene("room.xml")

# If wall_north and wall_south both use itu_concrete,
# they become ONE merged object. This FAILS:
wall = scene.get("wall_north")  # KeyError or returns the merged object
```

### Correct Usage (merge_shapes=False)

```python
# Individual objects preserved
scene = sionna.rt.load_scene("room.xml", merge_shapes=False)

# Now each object is individually addressable:
wall_n = scene.get("wall_north")
wall_s = scene.get("wall_south")

# You can reassign materials per-object:
wall_n.radio_material = scene.get("itu_glass")     # make north wall glass
wall_s.radio_material = scene.get("itu_concrete")  # keep south wall concrete
```

### When to Use Each

| Use Case | merge_shapes |
|---|---|
| Simple coverage simulation, no per-object edits | `True` (default) |
| Per-object material assignment | `False` |
| Per-object position queries | `False` |
| Debugging which object a ray hit | `False` |
| Large scenes where performance matters | `True` |

---

## HolderMaterial / itu-radio-material Registration

### The Problem

Sionna extends Mitsuba with a custom BSDF plugin called `itu-radio-material`. This plugin is only registered when `sionna.rt` is imported. If you try to load a scene XML containing `itu-radio-material` references before importing `sionna.rt`, Mitsuba will throw a `KeyError`.

### Wrong

```python
import mitsuba as mi
# itu-radio-material not registered yet!
scene = mi.load_file("scene.xml")  # KeyError: 'itu-radio-material'
```

### Right

```python
import sionna.rt  # registers itu-radio-material plugin
scene = sionna.rt.load_scene("scene.xml")  # works
```

### Also Right (if you need raw Mitsuba access)

```python
import sionna.rt  # registers plugins first
import mitsuba as mi
mi_scene = mi.load_file("scene.xml")  # now works because plugin is registered
```

---

## Mitsuba Variant: Never Set Manually

### Wrong

```python
import mitsuba as mi
mi.set_variant("llvm_ad_rgb")  # DO NOT DO THIS
import sionna.rt
```

### Right

```python
import sionna.rt  # handles variant selection internally
# Sionna picks the correct variant based on available backends
```

Sionna selects the Mitsuba variant automatically:
- GPU available: uses CUDA variant
- CPU only: uses LLVM variant
- Always uses AD (automatic differentiation) variant for differentiable RT

---

## Scene Frequency

Materials in Sionna are frequency-dependent (ITU-R P.2040-3). You must set `scene.frequency` before running any solver.

```python
scene = sionna.rt.load_scene("room.xml", merge_shapes=False)

# MUST set before simulation
scene.frequency = 3.5e9  # 3.5 GHz

# Materials auto-recompute their parameters
mat = scene.get("itu_concrete")
print(f"eps_r at 3.5 GHz: {mat.relative_permittivity}")

# Changing frequency updates all materials
scene.frequency = 28e9
print(f"eps_r at 28 GHz: {mat.relative_permittivity}")
```

### Forgetting to Set Frequency

If you run a solver without setting `scene.frequency`, you may get:
- Default frequency behavior (which may not match your use case)
- Incorrect material parameters
- Silently wrong results

Always set it explicitly.

---

## Complete Recommended Workflow

```python
import sionna
import sionna.rt  # Step 0: register plugins FIRST

# Step 1: Generate or prepare Mitsuba XML
# (This is where you add/modify geometry -- not at runtime)
xml_path = "/path/to/scene.xml"

# Step 2: Load scene with merge_shapes=False
scene = sionna.rt.load_scene(xml_path, merge_shapes=False)

# Step 3: Set frequency
scene.frequency = 3.5e9

# Step 4: (Optional) Modify materials on existing objects
wall = scene.get("wall_exterior")
wall.radio_material = scene.get("itu_concrete")

window = scene.get("window_north")
window.radio_material = scene.get("itu_glass")

# Step 5: Configure antenna arrays
tx_array = sionna.rt.PlanarArray(
    num_rows=4, num_cols=4,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="tr38901", polarization="VH",
)
rx_array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso", polarization="V",
)
scene.tx_array = tx_array
scene.rx_array = rx_array

# Step 6: Add TX/RX
tx = sionna.rt.Transmitter("bs", position=[5.0, 5.0, 3.0], power_dbm=23.0)
rx = sionna.rt.Receiver("ue", position=[15.0, 10.0, 1.5])
scene.add(tx)
scene.add(rx)

# Step 7: Run solver
solver = sionna.rt.PathSolver()
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=1_000_000,
    interaction_types="all",
)

# Step 8: Extract results
a, tau = paths.cir(out_type="torch")
```

---

## Adding Geometry: XML Modification

Since `scene.add(SceneObject(...))` is not supported, geometry must be added by modifying the XML file before loading.

### Adding a New Object to XML

```python
import xml.etree.ElementTree as ET

def add_obj_to_scene_xml(xml_path, obj_filename, material_id, transform=None):
    """Add an OBJ mesh to a Mitsuba scene XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    shape = ET.SubElement(root, "shape", type="obj")
    ET.SubElement(shape, "string", name="filename", value=obj_filename)
    ET.SubElement(shape, "ref", name="bsdf", id=material_id)

    if transform:
        t = ET.SubElement(shape, "transform", name="to_world")
        ET.SubElement(t, "translate", **{
            "x": str(transform.get("x", 0)),
            "y": str(transform.get("y", 0)),
            "z": str(transform.get("z", 0)),
        })

    tree.write(xml_path, xml_declaration=True, encoding="utf-8")

# Usage
add_obj_to_scene_xml(
    "scene.xml",
    "meshes/new_wall.obj",
    "itu_concrete",
    transform={"x": 5.0, "y": 0.0, "z": 0.0},
)

# Then reload
scene = sionna.rt.load_scene("scene.xml", merge_shapes=False)
```

### Using `edit_scene_shapes` (Advanced)

Sionna provides `edit_scene_shapes` for programmatic scene modification, but modifying XML and reloading is the recommended approach for most use cases.

---

## Removing TX/RX

```python
# Remove by name
scene.remove("tx-0")
scene.remove("rx-0")

# Verify
print(list(scene.transmitters.keys()))  # should not contain "tx-0"
print(list(scene.receivers.keys()))     # should not contain "rx-0"
```

---

## Listing Scene Contents

```python
# All scene objects (geometry)
print("Objects:")
for name in scene.objects:
    print(f"  {name}")

# All transmitters
print("Transmitters:")
for name, tx in scene.transmitters.items():
    print(f"  {name}: pos={tx.position}")

# All receivers
print("Receivers:")
for name, rx in scene.receivers.items():
    print(f"  {name}: pos={rx.position}")

# All materials
print("Materials:")
for name, mat in scene.radio_materials.items():
    print(f"  {name}: eps_r={mat.relative_permittivity}")
```

## Related

- [sionna-v2-api.md](sionna-v2-api.md) — full RT API including scene loading and solvers
- [data-sources.md](data-sources.md) — data import pipelines feeding scene construction
- [scene-state-schema.md](scene-state-schema.md) — state schema mapped to scene objects
