# Sionna Materials Reference (ITU-R P.2040-3)

## Material Model

Sionna implements the ITU-R P.2040-3 recommendation for building material electrical properties.

**Relative permittivity:**
```
epsilon_r(f) = a * f_GHz^b
```

**Conductivity (S/m):**
```
sigma(f) = c * f_GHz^d
```

Where `f_GHz` is frequency in GHz.

---

## Built-in ITU Material Catalog

| Material | a | b | c | d | Thickness (m) | Typical Use |
|---|---|---|---|---|---|---|
| `vacuum` | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | Free space |
| `concrete` | 5.31 | 0.0 | 0.0326 | 0.8095 | 0.015 | Exterior walls, floors, ceilings |
| `brick` | 3.75 | 0.0 | 0.038 | 0.0 | 0.04 | Exterior walls |
| `plasterboard` | 2.94 | 0.0 | 0.0116 | 0.7076 | 0.012 | Interior walls, partitions |
| `wood` | 1.99 | 0.0 | 0.0047 | 1.0718 | 0.015 | Doors, furniture, floors |
| `glass` | 6.27 | 0.0 | 0.0043 | 1.1925 | 0.003 | Windows, glass partitions |
| `ceiling_board` | 1.50 | 0.0 | 0.0005 | 1.3515 | 0.005 | Suspended ceilings |
| `chipboard` | 2.58 | 0.0 | 0.0217 | 0.7800 | 0.015 | Furniture panels |
| `floorboard` | 3.66 | 0.0 | 0.0044 | 1.3515 | 0.02 | Wooden floors |
| `metal` | 1.0 | 0.0 | 1.0e7 | 0.0 | 0.001 | Metal surfaces, ducts, elevator doors |
| `very_dry_ground` | 3.0 | 0.0 | 0.00015 | 2.52 | n/a | Dry terrain |
| `medium_dry_ground` | 15.0 | -0.1 | 0.035 | 1.63 | n/a | Average terrain |
| `wet_ground` | 30.0 | -0.4 | 0.15 | 1.30 | n/a | Wet terrain |
| `carpet` | 2.0 | 0.0 | 0.005 | 0.0 | 0.008 | Indoor carpeted floors (typical εr≈2.0, σ≈0.005 S/m per ITU-R P.2040-3 §2 typical values) |
| `ceiling_tile` | 1.5 | 0.0 | 0.0001 | 0.0 | 0.018 | Acoustic / suspended ceiling tile (typical εr≈1.5, σ≈0.0001 S/m per ITU-R P.2040-3 §2 typical values; lower σ than `ceiling_board` reflects the higher void-fraction of acoustic tile) |

---

## Common Semantic-to-ITU Material Mapping

Use this table when assigning materials to scene objects from semantic labels.

| Semantic Label | ITU Material | Rationale |
|---|---|---|
| exterior_wall | `itu_concrete` | Load-bearing, thick concrete |
| interior_wall | `itu_plasterboard` | Drywall/gypsum partitions |
| floor | `itu_concrete` | Concrete slab (default) |
| floor (wooden) | `itu_floorboard` | Residential wooden floors |
| ceiling | `itu_concrete` | Structural ceiling |
| ceiling (suspended) | `itu_ceiling_board` | Drop ceiling tiles |
| window | `itu_glass` | Standard window glass |
| door (wood) | `itu_wood` | Interior wooden doors |
| door (metal) | `itu_metal` | Security/elevator doors |
| roof | `itu_concrete` | Concrete roof slab |
| ground | `itu_medium_dry_ground` | Default outdoor ground |
| furniture | `itu_wood` | Generic furniture |
| partition | `itu_plasterboard` | Office dividers |
| brick_wall | `itu_brick` | Exposed brick |
| metal_surface | `itu_metal` | Ducts, pipes, panels |

---

## Mitsuba XML Syntax for ITU Materials

### Using Built-in ITU Materials in XML

```xml
<!-- Reference a built-in ITU material by name -->
<shape type="obj">
    <string name="filename" value="wall.obj"/>
    <ref name="bsdf" id="itu_concrete"/>
</shape>
```

Built-in ITU material IDs available after `import sionna.rt` (15 entries):
- `itu_vacuum`, `itu_concrete`, `itu_brick`, `itu_plasterboard`
- `itu_wood`, `itu_glass`, `itu_ceiling_board`, `itu_chipboard`
- `itu_floorboard`, `itu_metal`
- `itu_very_dry_ground`, `itu_medium_dry_ground`, `itu_wet_ground`
- `itu_carpet`, `itu_ceiling_tile`

### Defining Custom ITU Materials in XML

```xml
<bsdf type="itu-radio-material" id="custom_wall">
    <!-- Relative permittivity: epsilon_r = a * f_GHz^b -->
    <float name="relative_permittivity_a" value="5.31"/>
    <float name="relative_permittivity_b" value="0.0"/>

    <!-- Conductivity: sigma = c * f_GHz^d -->
    <float name="conductivity_a" value="0.0326"/>
    <float name="conductivity_b" value="0.8095"/>

    <!-- Slab thickness in meters -->
    <float name="thickness" value="0.02"/>

    <!-- Scattering coefficient (0.0 = perfect specular, 1.0 = fully diffuse) -->
    <float name="scattering_coefficient" value="0.3"/>
</bsdf>
```

### Complete Scene XML Example

```xml
<?xml version="1.0" encoding="utf-8"?>
<scene version="2.1.0">
    <!-- Custom materials -->
    <bsdf type="itu-radio-material" id="thick_concrete">
        <float name="relative_permittivity_a" value="5.31"/>
        <float name="relative_permittivity_b" value="0.0"/>
        <float name="conductivity_a" value="0.0326"/>
        <float name="conductivity_b" value="0.8095"/>
        <float name="thickness" value="0.30"/>
        <float name="scattering_coefficient" value="0.4"/>
    </bsdf>

    <bsdf type="itu-radio-material" id="thin_glass">
        <float name="relative_permittivity_a" value="6.27"/>
        <float name="relative_permittivity_b" value="0.0"/>
        <float name="conductivity_a" value="0.0043"/>
        <float name="conductivity_b" value="1.1925"/>
        <float name="thickness" value="0.006"/>
        <float name="scattering_coefficient" value="0.0"/>
    </bsdf>

    <!-- Geometry referencing materials -->
    <shape type="obj">
        <string name="filename" value="meshes/wall_exterior.obj"/>
        <ref name="bsdf" id="thick_concrete"/>
    </shape>

    <shape type="obj">
        <string name="filename" value="meshes/window.obj"/>
        <ref name="bsdf" id="thin_glass"/>
    </shape>

    <!-- Using built-in ITU material -->
    <shape type="obj">
        <string name="filename" value="meshes/floor.obj"/>
        <ref name="bsdf" id="itu_concrete"/>
    </shape>
</scene>
```

---

## Python API for Custom Materials

### Creating a RadioMaterial

```python
import sionna.rt

# Custom material with ITU-style parameters
custom_mat = sionna.rt.RadioMaterial(
    name="my_concrete",
    relative_permittivity=5.31,   # real part at reference freq
    conductivity=0.0326,          # S/m at reference freq
    scattering_coefficient=0.3,   # 0-1, fraction scattered diffusely
    thickness=0.25,               # slab thickness in meters
)

# Add to scene
scene.add(custom_mat)

# Assign to an object
obj = scene.get("wall_north")
obj.radio_material = custom_mat
```

### Frequency-Dependent Material (Python)

```python
# Materials auto-update when scene.frequency changes
scene.frequency = 3.5e9

mat = scene.get("itu_concrete")
print(f"At 3.5 GHz: eps_r={mat.relative_permittivity}, sigma={mat.conductivity}")

scene.frequency = 28e9
print(f"At 28 GHz:  eps_r={mat.relative_permittivity}, sigma={mat.conductivity}")
```

### Inspecting All Scene Materials

```python
for name, mat in scene.radio_materials.items():
    print(f"{name}:")
    print(f"  epsilon_r = {mat.relative_permittivity}")
    print(f"  sigma     = {mat.conductivity} S/m")
    if hasattr(mat, 'scattering_coefficient'):
        print(f"  scatter   = {mat.scattering_coefficient}")
```

---

## Scattering Configuration

### Scattering Coefficient

Controls the fraction of reflected energy that is scattered diffusely vs specularly.

```python
mat = sionna.rt.RadioMaterial(
    name="rough_concrete",
    relative_permittivity=5.31,
    conductivity=0.0326,
    scattering_coefficient=0.5,  # 50% diffuse, 50% specular
)
```

| Value | Behavior |
|---|---|
| 0.0 | Perfectly specular (smooth surface) |
| 0.3 | Lightly rough (painted concrete, drywall) |
| 0.5 | Moderately rough (raw concrete, brick) |
| 0.8 | Very rough (stucco, rough stone) |
| 1.0 | Perfectly diffuse (Lambertian) |

### Scattering Pattern

```python
from sionna.rt import LambertianPattern

# Default: Lambertian scattering (cosine-weighted)
mat = sionna.rt.RadioMaterial(
    name="rough_wall",
    relative_permittivity=5.31,
    conductivity=0.0326,
    scattering_coefficient=0.5,
    scattering_pattern=LambertianPattern(),
)
```

---

## Material Selection Guidelines

### By Frequency Band

| Band | Key Consideration |
|---|---|
| Sub-1 GHz | Penetration dominant; wall thickness matters most |
| 1-6 GHz (sub-6) | Balanced reflection/penetration; standard ITU params work well |
| 24-40 GHz (mmWave) | Reflection dominant; scattering coefficient matters more |
| 60+ GHz (V/E-band) | Almost all energy reflected; glass becomes near-opaque |

### By Scenario

| Scenario | Floor | Walls | Ceiling | Special |
|---|---|---|---|---|
| Office | `itu_concrete` | `itu_plasterboard` | `itu_ceiling_board` | Glass partitions |
| Residential | `itu_floorboard` | `itu_plasterboard` | `itu_concrete` | Wood doors |
| Warehouse | `itu_concrete` | `itu_metal` | `itu_metal` | Metal shelving |
| Outdoor urban | `itu_medium_dry_ground` | `itu_concrete` | n/a | Glass facades |
| Shopping mall | `itu_concrete` | `itu_glass` | `itu_concrete` | Mixed materials |

## Transmission (Penetration) Loss Reference

Approximate single-wall penetration losses for CPU analytical fallback and
JS ray visualization. Sionna computes these automatically from ITU material
parameters during GPU ray tracing — use these values only for CPU/JS models.

| Material | 3.5 GHz | 28 GHz | 60 GHz |
|----------|---------|--------|--------|
| Plasterboard (12mm) | 5 dB | 8 dB | 10 dB |
| Concrete (150mm) | 12 dB | 25 dB | 35 dB |
| Glass (6mm) | 2 dB | 4 dB | 4 dB |
| Wood (15mm) | 4 dB | 8 dB | 12 dB |
| Metal | >40 dB | >40 dB | >40 dB |

**For multi-room ray visualization:** When a ray hits an interior wall
(plasterboard), transmit through with the appropriate loss applied to the
ray's signal color. Exterior walls (concrete), floor, ceiling: reflect only.

## Structural-dominance rule

When assigning materials to furniture, pick the constituent whose volume × conductivity dominates RF propagation, NOT the visually-dominant surface. Examples:

- **Sofa**: wood frame, not fabric upholstery. Fabric is RF-transparent at 2.4–6 GHz; the wood frame is what blocks the signal.
- **Upholstered chair / armchair**: wood frame.
- **Bed**: wood frame, not mattress.
- **Filing cabinet / refrigerator**: metal body. Metal at 10⁷ S/m is a near-total reflector.
- **Monitor / TV**: metal/electronics chassis (the screen surface is also conductive).

Rule of thumb: which material would block a 3.5 GHz signal more if the others were absent? That's the structural dominator. The regression test at `lib/scene_gen/tests/test_materials_dominance.py` covers the most common categories.

## Related

- [physics-validation.md](physics-validation.md) — material attenuation bounds for validation
- [sionna-v2-api.md](sionna-v2-api.md) — RadioMaterial Python API
- [defaults.md](defaults.md) — default material assignments by room type
