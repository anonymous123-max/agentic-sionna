# Sionna Diffraction, RIS, and Scattering Reference

## Contents

1. [Diffraction](#diffraction) — UTD-based edge diffraction for NLOS coverage around corners
2. [Reconfigurable Intelligent Surfaces (RIS)](#reconfigurable-intelligent-surfaces-ris) — Programmable metasurface reflection for coverage steering
3. [Scattering](#scattering) — Lambertian and directive scattering models for rough surfaces
4. [Refraction](#refraction) — Through-wall transmission with material-dependent loss
5. [InteractionType Enum Reference](#interactiontype-enum-reference) — Bitmask enum values for path interaction filtering

## Diffraction

### Overview

Sionna implements UTD (Uniform Theory of Diffraction) for edge diffraction. Diffraction allows signals to propagate around corners and edges where line-of-sight is blocked.

### Enabling Diffraction

```python
from sionna.rt import InteractionType

solver = sionna.rt.PathSolver()

# Include diffraction in path computation
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=1_000_000,
    interaction_types=[
        InteractionType.SPECULAR,
        InteractionType.DIFFRACTION,
    ],
)
```

### Diffraction-Only Paths

```python
# Find only diffracted paths (useful for NLOS analysis)
paths = solver(
    scene=scene,
    max_depth=3,
    num_samples=1_000_000,
    interaction_types=[InteractionType.DIFFRACTION],
)
```

### Combined Interaction Types

Diffraction can combine with other interactions in multi-bounce paths:

```python
# All interaction types -- paths may include mixed bounces
# e.g., specular reflection -> diffraction -> specular reflection
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=2_000_000,
    interaction_types=[
        InteractionType.SPECULAR,
        InteractionType.DIFFUSE,
        InteractionType.DIFFRACTION,
        InteractionType.REFRACTION,
    ],
)

# Or use the shorthand
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=2_000_000,
    interaction_types="all",
)
```

### Inspecting Diffraction Paths

```python
# paths.types contains the interaction type at each bounce
# Check which paths include diffraction
import torch

types = paths.types  # [num_tx, num_rx, num_paths, max_depth]

# InteractionType values for filtering
# SPECULAR, DIFFUSE, REFRACTION, DIFFRACTION
diffraction_mask = (types == InteractionType.DIFFRACTION.value).any(dim=-1)
print(f"Paths with diffraction: {diffraction_mask.sum().item()}")
```

---

## Reconfigurable Intelligent Surfaces (RIS)

### Overview

RIS are programmable metasurfaces that can steer reflected signals by applying configurable phase shifts across their elements. Sionna models RIS as planar arrays of reflecting elements.

### Creating a RIS

```python
ris = sionna.rt.RIS(
    name="ris-0",
    position=[20.0, 10.0, 5.0],        # [x, y, z] in meters
    orientation=[0.0, 0.0, 0.0],        # [alpha, beta, gamma] radians
    num_rows=32,                         # number of element rows
    num_cols=32,                         # number of element columns
    element_spacing=0.5,                 # spacing in wavelengths
)

# Add to scene
scene.add(ris)
```

### Configuring RIS Phase Profile

```python
# Get the RIS object
ris = scene.get("ris-0")

# Set a focusing phase profile (steers beam toward a target point)
ris.phase_profile = sionna.rt.FocusingPhaseProfile(
    target_position=[50.0, 30.0, 1.5],  # focus toward this point
    frequency=scene.frequency,
)
```

### Manual Phase Profile

```python
import torch
import numpy as np

# Create a custom phase profile (num_rows x num_cols)
num_rows, num_cols = 32, 32
phases = torch.zeros(num_rows, num_cols)

# Example: linear phase gradient for beam steering
wavelength = 3e8 / scene.frequency
d = 0.5 * wavelength  # element spacing in meters
theta_steer = np.radians(30)  # steer 30 degrees from broadside

for col in range(num_cols):
    phases[:, col] = 2 * np.pi * d * col * np.sin(theta_steer) / wavelength

ris.phase_profile = phases
```

### RIS with Path Solver

```python
# RIS automatically participates in ray tracing after being added
solver = sionna.rt.PathSolver()
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=2_000_000,
    interaction_types="all",
)

# Paths via the RIS are included in the results
a, tau = paths.cir(out_type="torch")
```

### RIS Parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | str | Unique identifier |
| `position` | list[3] | [x, y, z] center position in meters |
| `orientation` | list[3] | [alpha, beta, gamma] Euler angles in radians |
| `num_rows` | int | Number of element rows |
| `num_cols` | int | Number of element columns |
| `element_spacing` | float | Element spacing in wavelengths (default 0.5) |
| `phase_profile` | tensor/Profile | Phase shift per element (num_rows x num_cols) |

### API Notes

- The RIS API was significantly reworked between Sionna 0.19 and 1.0+. The v2.0 API follows the 1.0+ pattern.
- In Sionna 0.19 and earlier, RIS was experimental and used a different class hierarchy.
- In v2.0, `RIS` is a first-class scene object added via `scene.add()`.

---

## Scattering

### Overview

Scattering models diffuse reflection from rough surfaces. Sionna applies a scattering coefficient to materials that splits reflected energy between specular and diffuse components.

### Scattering Coefficient

```python
# Set scattering on a material
mat = sionna.rt.RadioMaterial(
    name="rough_concrete",
    relative_permittivity=5.31,
    conductivity=0.0326,
    scattering_coefficient=0.5,  # 50% diffuse, 50% specular
)
scene.add(mat)

# Assign to an object
wall = scene.get("wall_exterior")
wall.radio_material = mat
```

### Scattering Coefficient Values

| Value | Surface Type | Example |
|---|---|---|
| 0.0 | Perfectly smooth | Polished metal, flat glass |
| 0.1-0.2 | Slightly rough | Painted drywall, smooth concrete |
| 0.3-0.4 | Moderately rough | Exposed concrete, plaster |
| 0.5-0.6 | Rough | Brick, stucco |
| 0.7-0.8 | Very rough | Rough stone, textured panels |
| 1.0 | Perfectly diffuse | Theoretical Lambertian surface |

### LambertianPattern

The default scattering pattern is Lambertian (cosine-weighted diffuse reflection).

```python
from sionna.rt import LambertianPattern

# Explicit Lambertian pattern (this is the default)
mat = sionna.rt.RadioMaterial(
    name="diffuse_wall",
    relative_permittivity=5.31,
    conductivity=0.0326,
    scattering_coefficient=0.6,
    scattering_pattern=LambertianPattern(),
)
```

### Enabling Diffuse Scattering in Path Computation

```python
# Must include DIFFUSE interaction type
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=2_000_000,
    interaction_types=[
        InteractionType.SPECULAR,
        InteractionType.DIFFUSE,
    ],
)
```

---

## Refraction (Transmission Through Materials)

### Overview

Refraction models signal transmission through thin material slabs (walls, windows, floors). The material's thickness and electrical properties determine the transmission loss.

### Enabling Refraction

```python
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=1_000_000,
    interaction_types=[
        InteractionType.SPECULAR,
        InteractionType.REFRACTION,
    ],
)
```

### Material Thickness for Refraction

Thickness is critical for refraction accuracy -- it determines how much energy passes through.

```python
# Thin glass window: more transmission
glass = sionna.rt.RadioMaterial(
    name="thin_window",
    relative_permittivity=6.27,
    conductivity=0.0043,
    thickness=0.003,  # 3mm single pane
)

# Thick concrete wall: less transmission
concrete = sionna.rt.RadioMaterial(
    name="thick_wall",
    relative_permittivity=5.31,
    conductivity=0.0326,
    thickness=0.30,  # 30cm structural wall
)
```

---

## InteractionType Enum Reference

```python
from sionna.rt import InteractionType

InteractionType.SPECULAR      # Mirror-like reflection
InteractionType.DIFFUSE       # Diffuse scattering from rough surfaces
InteractionType.REFRACTION    # Transmission through material slabs
InteractionType.DIFFRACTION   # UTD edge diffraction around corners
```

### Selecting Interaction Types

```python
# Single type
interaction_types=[InteractionType.SPECULAR]

# Multiple types (list)
interaction_types=[InteractionType.SPECULAR, InteractionType.DIFFRACTION]

# All types (shorthand string)
interaction_types="all"
```

---

## Frontend Visualization Colors

When rendering path visualizations, use these colors to distinguish path types:

| Path Type | Color | Hex |
|---|---|---|
| Line-of-sight (LOS) | Green | `#00FF00` |
| Specular reflection | Blue | `#0000FF` |
| Diffuse scattering | Orange | `#FFA500` |
| Diffraction | Red | `#FF0000` |
| Refraction | Cyan | `#00FFFF` |
| RIS reflection | Magenta | `#FF00FF` |

---

## Complete Example: All Path Types

```python
import sionna
import sionna.rt
from sionna.rt import InteractionType

# Load scene
scene = sionna.rt.load_scene("building.xml", merge_shapes=False)
scene.frequency = 28e9  # mmWave

# Configure antennas
array = sionna.rt.PlanarArray(1, 1, 0.5, 0.5, "iso", "V")
scene.tx_array = array
scene.rx_array = array

# Add TX/RX
tx = sionna.rt.Transmitter("tx", position=[5.0, 5.0, 3.0], power_dbm=30.0)
rx = sionna.rt.Receiver("rx", position=[25.0, 15.0, 1.5])
scene.add(tx)
scene.add(rx)

# Modify a wall to be rough
wall = scene.get("wall_south")
rough_mat = sionna.rt.RadioMaterial(
    name="rough_wall",
    relative_permittivity=5.31,
    conductivity=0.0326,
    scattering_coefficient=0.5,
    thickness=0.15,
)
scene.add(rough_mat)
wall.radio_material = rough_mat

# Add a RIS
ris = sionna.rt.RIS(
    name="ris",
    position=[15.0, 0.5, 3.0],
    orientation=[0.0, 0.0, 0.0],
    num_rows=16,
    num_cols=16,
    element_spacing=0.5,
)
scene.add(ris)

# Configure RIS to focus toward RX
ris.phase_profile = sionna.rt.FocusingPhaseProfile(
    target_position=rx.position,
    frequency=scene.frequency,
)

# Compute paths with all interaction types
solver = sionna.rt.PathSolver()
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=5_000_000,
    interaction_types="all",
)

# Get CIR
a, tau = paths.cir(out_type="torch")
print(f"Total paths: {a.shape[-1]}")
print(f"Max amplitude: {a.abs().max().item():.6f}")
print(f"Delay spread: {(tau.max() - tau.min()).item() * 1e9:.2f} ns")
```

## Related

- [sionna-v2-api.md](sionna-v2-api.md) — PathSolver and RIS API reference
- [antenna-patterns.md](antenna-patterns.md) — array elements used with RIS
