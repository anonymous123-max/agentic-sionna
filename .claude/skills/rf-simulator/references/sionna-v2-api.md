# Sionna v2.0 RT API Reference

## Contents

1. [Critical Rules](#critical-rules) — Import order, material registration, and must-follow constraints
2. [Scene Loading](#scene-loading) — Loading Mitsuba XML scenes and configuring frequency/materials
3. [Transmitters and Receivers](#transmitters-and-receivers) — Placing TX/RX with positions, orientations, and antenna arrays
4. [Editing Scene Objects](#editing-scene-objects) — Modifying geometry, materials, and object properties at runtime
5. [Path Computation](#path-computation) — Ray tracing with reflection, diffraction, and scattering
6. [Channel Impulse Response](#channel-impulse-response) — Extracting CIR, delay spread, and channel matrices from paths

## Critical Rules

1. **Always `import sionna.rt`** before calling `load_scene()`. This registers `HolderMaterial` and `itu-radio-material` BSDF plugins with Mitsuba.
2. **Never call `mi.set_variant()`** manually — Sionna sets the Mitsuba variant on import. Overriding it causes variant conflicts that produce wrong results or crashes.
3. **Cannot add geometry at runtime.** `scene.add()` accepts `Transmitter`, `Receiver`, `RadioMaterial`, `RIS`, and `Camera` only -- never `SceneObject`. To add geometry, modify the Mitsuba XML and reload.
4. **Set `scene.frequency`** before running any solver. Material parameters (permittivity, conductivity) are frequency-dependent and auto-recompute.
5. **Use `merge_shapes=False`** when individual scene objects need addressing (position changes, material reassignment). Default `True` merges same-material shapes into one.

---

## Scene Loading

### Built-in Scenes

```python
import sionna
import sionna.rt  # MUST import before load_scene

# Built-in test scenes
scene = sionna.rt.load_scene(sionna.rt.scene.simple_street_canyon)
scene = sionna.rt.load_scene(sionna.rt.scene.munich)
scene = sionna.rt.load_scene(sionna.rt.scene.etoile)
scene = sionna.rt.load_scene(sionna.rt.scene.simple_wedge)
scene = sionna.rt.load_scene(sionna.rt.scene.simple_reflector)
scene = sionna.rt.load_scene(sionna.rt.scene.floor_wall)
```

### Custom XML Scene

```python
scene = sionna.rt.load_scene("/path/to/scene.xml", merge_shapes=False)
```

### Empty Scene

```python
scene = sionna.rt.load_scene()  # empty scene, no geometry
```

### merge_shapes Parameter

```python
# Default: merge_shapes=True -- objects with same material merged, cannot address individually
scene = sionna.rt.load_scene("scene.xml")

# Use False when you need to reposition or re-material individual objects
scene = sionna.rt.load_scene("scene.xml", merge_shapes=False)

# With merge_shapes=False, you can do:
obj = scene.get("wall_north")
obj.radio_material = scene.get("my_custom_mat")
```

---

## Transmitters and Receivers

### PlanarArray

```python
# Define antenna array (shared by TX and RX, or create separate ones)
array = sionna.rt.PlanarArray(
    num_rows=1,
    num_cols=1,
    vertical_spacing=0.5,     # in wavelengths
    horizontal_spacing=0.5,   # in wavelengths
    pattern="iso",            # "iso", "dipole", "tr38901"
    polarization="V",         # "V", "H", "VH" (cross-polarized)
)
```

### Adding TX/RX

```python
# Create transmitter
tx = sionna.rt.Transmitter(
    name="tx-0",
    position=[10.0, 5.0, 3.0],   # [x, y, z] in meters
    orientation=[0.0, 0.0, 0.0],  # [alpha, beta, gamma] radians (Euler ZYX)
    power_dbm=23.0,               # transmit power in dBm (default 10)
)

# Create receiver
rx = sionna.rt.Receiver(
    name="rx-0",
    position=[50.0, 20.0, 1.5],
    orientation=[0.0, 0.0, 0.0],
)

# Add to scene
scene.add(tx)
scene.add(rx)

# Assign antenna arrays
scene.tx_array = array
scene.rx_array = array
```

### Updating TX/RX After Adding

```python
tx = scene.get("tx-0")
tx.position = [15.0, 5.0, 3.0]
tx.orientation = [0.0, 0.0, 0.0]
tx.power_dbm = 30.0
```

### Multiple TX/RX

```python
for i in range(4):
    tx = sionna.rt.Transmitter(
        name=f"tx-{i}",
        position=[10.0 + i * 5.0, 5.0, 3.0],
    )
    scene.add(tx)

for j in range(16):
    rx = sionna.rt.Receiver(
        name=f"rx-{j}",
        position=[20.0 + j * 2.0, 30.0, 1.5],
    )
    scene.add(rx)
```

---

## Editing Scene Objects

### Get and Modify Objects

```python
# Get a scene object by name (must use merge_shapes=False at load time)
obj = scene.get("floor")
print(obj.radio_material)
print(obj.position)

# Reassign material
concrete = scene.get("itu_concrete")
obj.radio_material = concrete

# List all objects
for name, obj in scene.objects.items():
    print(f"{name}: material={obj.radio_material.name}")
```

### Setting Scene Frequency

```python
# MUST set before running solvers -- materials auto-recompute
scene.frequency = 3.5e9   # 3.5 GHz (sub-6)
scene.frequency = 28e9    # 28 GHz (mmWave)
scene.frequency = 60e9    # 60 GHz (V-band)
```

---

## Path Computation (PathSolver)

### Basic Ray Tracing

```python
solver = sionna.rt.PathSolver()

# Compute paths: returns a Paths object
paths = solver(
    scene=scene,
    max_depth=5,               # max number of interactions per path
    num_samples=1_000_000,     # number of candidate rays (more = more paths found)
)
```

### Interaction Types

```python
from sionna.rt import InteractionType

# Control which interaction types to include
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=1_000_000,
    interaction_types=[
        InteractionType.SPECULAR,      # mirror reflections
        InteractionType.DIFFUSE,       # diffuse scattering
        InteractionType.REFRACTION,    # transmission through materials
        InteractionType.DIFFRACTION,   # edge diffraction (UTD)
    ],
)

# Specular-only (fastest)
paths = solver(
    scene=scene,
    max_depth=3,
    num_samples=500_000,
    interaction_types=[InteractionType.SPECULAR],
)

# All interactions
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=1_000_000,
    interaction_types="all",
)
```

### Inspecting Paths

```python
# Number of paths found per TX-RX pair
print(paths.types.shape)  # [num_tx, num_rx, num_paths, max_depth]

# Per-path data
print(paths.vertices)     # interaction points
print(paths.types)        # interaction types at each bounce
print(paths.objects)      # scene objects hit
```

---

## Channel Impulse Response (CIR)

### Converting Paths to CIR

```python
# Get CIR as PyTorch tensors
a, tau = paths.cir(out_type="torch")

# a:   complex path coefficients [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
# tau: propagation delays in seconds [batch, num_rx, num_tx, num_paths]

print(f"Path amplitudes shape: {a.shape}")
print(f"Path delays shape:     {tau.shape}")
print(f"Number of paths:       {a.shape[-1]}")
```

### CIR Output Types

```python
# PyTorch tensors (for downstream PHY processing)
a, tau = paths.cir(out_type="torch")

# Dr.Jit arrays (for RT-layer autodiff)
a, tau = paths.cir(out_type="drjit")

# NumPy arrays
a, tau = paths.cir(out_type="numpy")
```

---

## Radio Maps (RadioMapSolver)

### Computing a Coverage Map

**CRITICAL**: `center`, `orientation`, and `size` are an all-or-nothing group.
If you provide ANY of them, you MUST provide ALL THREE or you get a ValueError.

```python
rm_solver = sionna.rt.RadioMapSolver()

# Compute radio map — orientation is REQUIRED when center/size are given
radio_map = rm_solver(
    scene=scene,
    cell_size=[0.2, 0.2],        # grid resolution in meters [dx, dy]
    samples_per_tx=10_000_000,   # number of rays per transmitter
    max_depth=5,
    center=[4.0, 3.0, 1.5],     # [x, y, z] center of the measurement plane
    orientation=[0.0, 0.0, 0.0], # REQUIRED: [alpha, beta, gamma] Euler angles; [0,0,0] = horizontal
    size=[8.0, 6.0],             # [width, height] of the map in meters
)
```

### Return type: PlanarRadioMap

The solver returns a `PlanarRadioMap` object (NOT a tensor). Key properties:

| Property | Shape | Description |
|----------|-------|-------------|
| `radio_map.path_gain` | `[num_tx, cells_y, cells_x]` | Path gain in **linear scale** (unitless) |
| `radio_map.rss` | `[num_tx, cells_y, cells_x]` | RSS in **Watts** (linear) |
| `radio_map.cell_centers` | `[cells_y, cells_x, 3]` | XYZ positions of cell centers |
| `radio_map.sinr` | `[num_tx, cells_y, cells_x]` | SINR (linear) |

**There is NO `.value` attribute.** Use `.path_gain` or `.rss`.

### Converting to dBm

```python
import numpy as np

# path_gain is linear scale — convert to dB then add TX power
pg = np.array(radio_map.path_gain)          # [num_tx, cells_y, cells_x]
pg_linear = pg[0, :, :]                     # first TX
pg_linear = np.where(pg_linear > 0, pg_linear, np.nan)
rss_dbm = TX_POWER_DBM + 10.0 * np.log10(pg_linear)
```

### Rendering Radio Maps

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(
    rss_dbm,                     # [cells_y, cells_x] in dBm
    origin="lower",
    extent=[x_min, x_max, y_min, y_max],
    cmap="jet",
    vmin=-90,
    vmax=-30,
)
plt.colorbar(im, label="RSS (dBm)")
plt.savefig("radio_map.png", dpi=150)
```

### Built-in Rendering (Alternative)

```python
# PlanarRadioMap has a built-in show() method
radio_map.show()  # Opens a matplotlib figure
```

---

## Velocity and Doppler

### Setting Object Velocity

```python
# Set velocity on TX (e.g., moving base station on a vehicle)
tx = scene.get("tx-0")
tx.velocity = [0.0, 10.0, 0.0]  # 10 m/s in y-direction

# Set velocity on RX (e.g., mobile user)
rx = scene.get("rx-0")
rx.velocity = [5.0, 0.0, 0.0]   # 5 m/s in x-direction

# Set velocity on a scene object (e.g., moving wall/reflector)
obj = scene.get("reflector_panel")
obj.velocity = [0.0, 0.0, 1.0]  # 1 m/s upward

# Doppler shifts are automatically computed in paths.cir()
a, tau = paths.cir(out_type="torch")
# Doppler is embedded in the complex amplitude time evolution
```

---

## Differentiable Optimization

### Dr.Jit Autodiff (RT Layer)

Use Dr.Jit for differentiating through the ray-tracing layer (scene geometry, material parameters, TX/RX positions).

```python
import drjit as dr

# Make a material parameter differentiable
mat = scene.get("itu_concrete")
rel_perm = dr.Float32(mat.relative_permittivity)
dr.enable_grad(rel_perm)

# Forward pass through RT
paths = solver(scene=scene, max_depth=3, num_samples=500_000)
a, tau = paths.cir(out_type="drjit")

# Compute loss and backward
loss = dr.sum(dr.abs(a) ** 2)
dr.backward(loss)

# Gradient
grad_perm = dr.grad(rel_perm)
```

### PyTorch Autograd (PHY Layer)

Use PyTorch for differentiating through PHY-layer processing (beamforming weights, channel estimation, etc.).

```python
import torch

# Get CIR as PyTorch tensors
a, tau = paths.cir(out_type="torch")

# PHY processing with autograd
a_torch = a.clone().requires_grad_(True)

# Example: optimize beamforming weights
w = torch.randn(num_antennas, dtype=torch.cfloat, requires_grad=True)
signal = torch.einsum("...a,a->...", a_torch, w)
loss = -torch.abs(signal).mean()
loss.backward()

# Update weights
with torch.no_grad():
    w -= 0.01 * w.grad
```

---

## Complete End-to-End Example

```python
import sionna
import sionna.rt  # MUST import before load_scene

# 1. Load scene
scene = sionna.rt.load_scene(sionna.rt.scene.munich, merge_shapes=False)
scene.frequency = 3.5e9

# 2. Configure antennas
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

# 3. Add TX/RX
tx = sionna.rt.Transmitter("bs", position=[8.5, 21.0, 27.0], power_dbm=23.0)
rx = sionna.rt.Receiver("ue", position=[45.0, 90.0, 1.5])
scene.add(tx)
scene.add(rx)

# 4. Compute paths
solver = sionna.rt.PathSolver()
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=10_000_000,
    interaction_types="all",
)

# 5. Get CIR
a, tau = paths.cir(out_type="torch")
print(f"Found {a.shape[-1]} paths")
print(f"Path amplitudes: {a.shape}")
print(f"Path delays:     {tau.shape}")

# 6. Compute radio map
rm_solver = sionna.rt.RadioMapSolver()
radio_map = rm_solver(
    scene=scene,
    cell_size=[2.0, 2.0],
    samples_per_tx=10_000_000,
    max_depth=5,
    center=[0.0, 0.0, 1.5],
    size=[500.0, 500.0],
)
```

---

## Key Differences from Sionna v1.x

| Feature | v1.x | v2.0 |
|---|---|---|
| PHY layer | TensorFlow-based | **PyTorch-native** |
| RT layer | Dr.Jit + Mitsuba 3 | Dr.Jit + Mitsuba 3 (same) |
| CIR output | `paths.cir()` returns TF tensors | `paths.cir(out_type="torch")` returns PyTorch |
| `compute_paths()` | Method on `Scene` | Standalone `PathSolver()` callable |
| `compute_coverage_map()` | Method on `Scene` | Standalone `RadioMapSolver()` callable |
| Installation | `pip install sionna` (TF) | `pip install sionna>=2.0.0` (PyTorch) |
| GPU backend | TensorFlow + CUDA | PyTorch CUDA + Dr.Jit (llvm/cuda) |

---

## API Quick Reference

| Task | Code |
|---|---|
| Load scene | `sionna.rt.load_scene(path, merge_shapes=False)` |
| Set frequency | `scene.frequency = 3.5e9` |
| Create array | `sionna.rt.PlanarArray(num_rows, num_cols, ...)` |
| Add TX | `scene.add(sionna.rt.Transmitter(name, position))` |
| Add RX | `scene.add(sionna.rt.Receiver(name, position))` |
| Compute paths | `PathSolver()(scene, max_depth, num_samples)` |
| Get CIR | `paths.cir(out_type="torch")` -> `(a, tau)` |
| Radio map | `RadioMapSolver()(scene, cell_size, samples_per_tx)` |
| Get object | `scene.get("object_name")` |
| Set material | `obj.radio_material = scene.get("mat_name")` |
| Set velocity | `tx.velocity = [vx, vy, vz]` |

## Related

- [script-guidelines.md](script-guidelines.md) — script patterns wrapping these API calls
- [sionna-materials.md](sionna-materials.md) — RadioMaterial API and ITU catalog
- [sionna-scene-editing.md](sionna-scene-editing.md) — scene loading and object editing gotchas
- [error-patterns.md](error-patterns.md) — common API errors and auto-fix strategies
