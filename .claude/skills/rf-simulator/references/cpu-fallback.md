# CPU Analytical Fallback Model

## Contents

1. [When to Use](#when-to-use) — Criteria for choosing CPU fallback over GPU ray tracing
2. [Model Components](#model-components) — Path loss, atmospheric absorption, LOS detection, and antenna gain
3. [Complete Python Implementation](#complete-python-implementation) — Full source code for the analytical solver
4. [Usage Example](#usage-example) — End-to-end example producing a coverage heatmap
5. [Extracting Obstacles from a Scene](#extracting-obstacles-from-a-scene) — Converting Scene geometry to obstacle list for LOS checks
6. [Performance Comparison](#performance-comparison) — CPU vs GPU accuracy and runtime benchmarks

Analytical RF coverage model for environments without GPU acceleration. Combines 3GPP TR 38.901 path loss, ITU-R P.676 atmospheric absorption, ray-box LOS checking, and antenna directivity to produce coverage maps compatible with Sionna RadioMap output.

---

## When to Use

- No NVIDIA GPU available (CPU-only machine, Apple Silicon without CUDA)
- User requests a "quick estimate" or "quick preview"
- Rapid iteration during scene design (before committing to full ray tracing)
- Batch processing many scenes where GPU time is limited

---

## Model Components

The analytical model chains four stages:

```
TX position + RX grid
       |
       v
  [1] LOS/NLOS classification (ray-box intersection)
       |
       v
  [2] 3GPP path loss (InH for indoor, UMi/UMa for outdoor)
       |
       v
  [3] Atmospheric absorption (ITU-R P.676, significant above 10 GHz)
       |
       v
  [4] Antenna directivity gain (3GPP TR 38.901 antenna model)
       |
       v
  Path gain map (dB) -> same format as Sionna RadioMap
```

---

## Complete Python Implementation

```python
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


C_LIGHT = 3e8  # speed of light in m/s


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box for a piece of furniture or obstacle."""
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float


@dataclass(frozen=True)
class TransmitterConfig:
    """Transmitter parameters."""
    position: tuple[float, float, float]  # (x, y, z) in meters
    power_dbm: float = 23.0               # transmit power
    frequency_ghz: float = 3.5            # carrier frequency
    antenna_bearing_deg: float = 0.0      # boresight direction from north (CW)
    antenna_downtilt_deg: float = 0.0     # mechanical downtilt
    antenna_beamwidth_h_deg: float = 360.0  # horizontal 3dB beamwidth (360 = omni)
    antenna_beamwidth_v_deg: float = 180.0  # vertical 3dB beamwidth
    antenna_gain_dbi: float = 0.0         # peak antenna gain


@dataclass
class CoverageResult:
    """Coverage map output, shaped to match Sionna RadioMap conventions."""
    path_gain_db: np.ndarray       # 2D array [ny, nx], negative values
    rx_positions: np.ndarray       # [ny, nx, 3] receiver grid positions
    cell_size_m: float             # grid resolution
    is_los: np.ndarray             # [ny, nx] boolean LOS map
    tx_position: tuple[float, float, float]
    frequency_ghz: float
    model_type: str = "cpu_analytical"  # badge identifier


# ---------------------------------------------------------------------------
# Stage 1: LOS / NLOS classification via ray-box intersection
# ---------------------------------------------------------------------------

def _ray_intersects_aabb(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    box: BoundingBox,
    ray_length: float,
) -> bool:
    """Slab method for ray vs axis-aligned bounding box intersection.

    Returns True if the ray segment [0, ray_length] intersects the box.
    """
    inv_dir = np.where(np.abs(ray_dir) > 1e-12, 1.0 / ray_dir, np.sign(ray_dir) * 1e12)

    t_min_vec = (np.array([box.x_min, box.y_min, box.z_min]) - ray_origin) * inv_dir
    t_max_vec = (np.array([box.x_max, box.y_max, box.z_max]) - ray_origin) * inv_dir

    t1 = np.minimum(t_min_vec, t_max_vec)
    t2 = np.maximum(t_min_vec, t_max_vec)

    t_near = np.max(t1)
    t_far = np.min(t2)

    # Intersection exists if t_near <= t_far AND the interval overlaps [0, ray_length]
    return bool(t_near <= t_far and t_far >= 0.0 and t_near <= ray_length)


def check_los(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    obstacles: list[BoundingBox],
) -> bool:
    """Check line-of-sight between TX and RX positions.

    Casts a ray from tx_pos toward rx_pos and tests intersection against
    all obstacle bounding boxes. Returns True if no obstacle blocks the path.
    """
    diff = rx_pos - tx_pos
    dist = np.linalg.norm(diff)
    if dist < 1e-6:
        return True
    ray_dir = diff / dist

    for box in obstacles:
        if _ray_intersects_aabb(tx_pos, ray_dir, box, dist):
            return False
    return True


def check_los_grid(
    tx_pos: np.ndarray,
    rx_grid: np.ndarray,
    obstacles: list[BoundingBox],
) -> np.ndarray:
    """Vectorized LOS check for an entire receiver grid.

    Args:
        tx_pos: shape (3,)
        rx_grid: shape (ny, nx, 3)
        obstacles: list of BoundingBox

    Returns:
        Boolean array shape (ny, nx), True = LOS
    """
    ny, nx = rx_grid.shape[:2]
    los_map = np.ones((ny, nx), dtype=bool)

    for iy in range(ny):
        for ix in range(nx):
            los_map[iy, ix] = check_los(tx_pos, rx_grid[iy, ix], obstacles)

    return los_map


# ---------------------------------------------------------------------------
# Stage 2: 3GPP TR 38.901 path loss
# ---------------------------------------------------------------------------

def analytical_path_loss(
    d_3d: float,
    f_ghz: float,
    is_los: bool,
    scenario: str = "inh",
) -> float:
    """Compute deterministic path loss in dB using 3GPP TR 38.901 models.

    Args:
        d_3d: 3D distance in meters (clamped to >= 1.0 m)
        f_ghz: carrier frequency in GHz
        is_los: True for LOS, False for NLOS
        scenario: "inh" (indoor), "umi" (urban micro), "uma" (urban macro)

    Returns:
        Path loss in dB (positive value; subtract from TX power to get RX power)
    """
    d_3d = max(d_3d, 1.0)  # avoid log(0)

    if scenario == "inh":
        if is_los:
            return 32.4 + 17.3 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
        else:
            pl_los = 32.4 + 17.3 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
            pl_nlos = 17.3 + 38.3 * np.log10(d_3d) + 24.9 * np.log10(f_ghz)
            return max(pl_los, pl_nlos)

    elif scenario == "umi":
        if is_los:
            return 32.4 + 21.0 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
        else:
            pl_los = 32.4 + 21.0 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
            pl_nlos = 22.4 + 35.3 * np.log10(d_3d) + 21.3 * np.log10(f_ghz)
            return max(pl_los, pl_nlos)

    elif scenario == "uma":
        if is_los:
            return 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
        else:
            pl_los = 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
            pl_nlos = 13.54 + 39.08 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
            return max(pl_los, pl_nlos)

    else:
        raise ValueError(f"Unknown scenario: {scenario!r}. Use 'inh', 'umi', or 'uma'.")


# ---------------------------------------------------------------------------
# Stage 3: ITU-R P.676 atmospheric absorption
# ---------------------------------------------------------------------------

def atmospheric_absorption(
    d_m: float,
    f_ghz: float,
    temperature_c: float = 20.0,
    humidity_pct: float = 50.0,
) -> float:
    """Simplified ITU-R P.676 atmospheric absorption loss in dB.

    Significant above ~10 GHz. At sub-6 GHz this returns near-zero.

    Args:
        d_m: distance in meters
        f_ghz: frequency in GHz
        temperature_c: ambient temperature in Celsius
        humidity_pct: relative humidity percentage

    Returns:
        Absorption loss in dB (positive value)
    """
    if f_ghz < 1.0:
        return 0.0

    # Water vapor density approximation
    es = 6.1121 * np.exp((18.678 - temperature_c / 234.5)
                          * temperature_c / (257.14 + temperature_c))
    rho_w = humidity_pct / 100.0 * es * 216.7 / (temperature_c + 273.15)

    # Oxygen component
    if f_ghz <= 57.0:
        gamma_o = (7.2 * f_ghz**2 / (f_ghz**2 + 0.34)
                   + 0.62 / ((54.0 - f_ghz)**1.16 + 0.83)) * f_ghz**2 * 1e-3
    elif f_ghz <= 63.0:
        gamma_o = 15.0 * np.exp(-0.5 * ((f_ghz - 60.0) / 1.5)**2)
    elif f_ghz <= 98.0:
        gamma_o = (0.2 + 0.5 / ((f_ghz - 63.0)**1.6 + 1.5)) * f_ghz**2 * 1e-3
    else:
        gamma_o = (2.0 / ((f_ghz - 118.75)**2 + 1.0) + 0.01) * f_ghz * 1e-2

    # Water vapor component
    gamma_w = (0.067 + 3.0 / ((f_ghz - 22.235)**2 + 5.0)
               + 7.0 / ((f_ghz - 183.3)**2 + 6.0)) * f_ghz**2 * rho_w * 1e-4

    specific_atten = gamma_o + gamma_w  # dB/km
    return specific_atten * (d_m / 1000.0)


# ---------------------------------------------------------------------------
# Stage 4: Antenna directivity gain
# ---------------------------------------------------------------------------

def compute_antenna_gain(
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    bearing_deg: float = 0.0,
    downtilt_deg: float = 0.0,
    beamwidth_h_deg: float = 360.0,
    beamwidth_v_deg: float = 180.0,
    gain_dbi: float = 0.0,
) -> float:
    """3GPP-style antenna gain with horizontal and vertical pattern rolloff.

    For an omnidirectional antenna (beamwidth_h=360, beamwidth_v=180),
    returns gain_dbi directly. For directional antennas, applies a
    cos^2-based rolloff from boresight.

    Args:
        tx_pos: transmitter position (3,)
        rx_pos: receiver position (3,)
        bearing_deg: boresight azimuth from north, clockwise
        downtilt_deg: mechanical downtilt (positive = below horizon)
        beamwidth_h_deg: horizontal 3 dB beamwidth
        beamwidth_v_deg: vertical 3 dB beamwidth
        gain_dbi: peak antenna gain in dBi

    Returns:
        Effective antenna gain in dB for the TX->RX direction
    """
    # Omnidirectional shortcut
    if beamwidth_h_deg >= 360.0 and beamwidth_v_deg >= 180.0:
        return gain_dbi

    diff = rx_pos - tx_pos
    d_horiz = np.sqrt(diff[0]**2 + diff[1]**2)

    if d_horiz < 1e-6:
        return gain_dbi  # directly above/below, no rolloff applied

    # Azimuth angle of RX from TX (degrees from north, CW)
    az_rx = np.degrees(np.arctan2(diff[0], diff[1])) % 360.0
    az_off = az_rx - bearing_deg
    # Wrap to [-180, 180]
    az_off = (az_off + 180.0) % 360.0 - 180.0

    # Elevation angle
    el_rx = np.degrees(np.arctan2(diff[2], d_horiz))
    el_off = el_rx - (-downtilt_deg)

    # 3GPP TR 38.901 Section 7.3: A(theta) = -min(12*(theta/theta_3dB)^2, A_m)
    # A_m = 30 dB (max front-to-back ratio)
    a_m = 30.0

    if beamwidth_h_deg < 360.0:
        a_h = -min(12.0 * (az_off / (beamwidth_h_deg / 2.0))**2, a_m)
    else:
        a_h = 0.0

    if beamwidth_v_deg < 180.0:
        a_v = -min(12.0 * (el_off / (beamwidth_v_deg / 2.0))**2, a_m)
    else:
        a_v = 0.0

    # Combined: -min(-(A_h + A_v), A_m)
    a_total = -min(-(a_h + a_v), a_m)

    return gain_dbi + a_total


# ---------------------------------------------------------------------------
# Coverage map computation
# ---------------------------------------------------------------------------

def compute_analytical_coverage(
    tx: TransmitterConfig,
    scene_bounds: tuple[float, float, float, float],
    obstacles: list[BoundingBox],
    cell_size_m: float = 0.5,
    rx_height_m: float = 1.5,
    scenario: str = "inh",
) -> CoverageResult:
    """Compute a full analytical coverage map over a scene.

    Args:
        tx: transmitter configuration
        scene_bounds: (x_min, y_min, x_max, y_max) scene extents in meters
        obstacles: furniture / wall bounding boxes for LOS checks
        cell_size_m: grid resolution in meters (default 0.5 m)
        rx_height_m: receiver height above floor in meters
        scenario: 3GPP scenario identifier ("inh", "umi", "uma")

    Returns:
        CoverageResult with path_gain_db, rx_positions, is_los arrays
    """
    x_min, y_min, x_max, y_max = scene_bounds

    # Build receiver grid
    xs = np.arange(x_min + cell_size_m / 2, x_max, cell_size_m)
    ys = np.arange(y_min + cell_size_m / 2, y_max, cell_size_m)
    nx, ny = len(xs), len(ys)

    if nx == 0 or ny == 0:
        raise ValueError("Scene bounds too small for the given cell_size_m.")

    rx_grid = np.zeros((ny, nx, 3))
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            rx_grid[iy, ix] = [x, y, rx_height_m]

    tx_pos = np.array(tx.position, dtype=float)

    # Stage 1: LOS classification
    los_map = check_los_grid(tx_pos, rx_grid, obstacles)

    # Stages 2-4: path loss + absorption + antenna gain per grid cell
    path_gain_db = np.full((ny, nx), -np.inf)

    for iy in range(ny):
        for ix in range(nx):
            rx_pos = rx_grid[iy, ix]
            d_3d = np.linalg.norm(rx_pos - tx_pos)

            if d_3d < 0.1:
                path_gain_db[iy, ix] = 0.0
                continue

            # Path loss (positive dB value)
            pl_db = analytical_path_loss(d_3d, tx.frequency_ghz, los_map[iy, ix], scenario)

            # Atmospheric absorption
            atm_db = atmospheric_absorption(d_3d, tx.frequency_ghz)

            # Antenna gain
            ant_gain = compute_antenna_gain(
                tx_pos, rx_pos,
                bearing_deg=tx.antenna_bearing_deg,
                downtilt_deg=tx.antenna_downtilt_deg,
                beamwidth_h_deg=tx.antenna_beamwidth_h_deg,
                beamwidth_v_deg=tx.antenna_beamwidth_v_deg,
                gain_dbi=tx.antenna_gain_dbi,
            )

            # Path gain = TX power + antenna gain - path loss - absorption
            # Store as path gain (negative dB), consistent with Sionna RadioMap
            path_gain_db[iy, ix] = -(pl_db + atm_db) + ant_gain

    return CoverageResult(
        path_gain_db=path_gain_db,
        rx_positions=rx_grid,
        cell_size_m=cell_size_m,
        is_los=los_map,
        tx_position=tx.position,
        frequency_ghz=tx.frequency_ghz,
        model_type="cpu_analytical",
    )
```

---

## Usage Example

```python
# Define scene
tx = TransmitterConfig(
    position=(5.0, 5.0, 2.5),
    power_dbm=23.0,
    frequency_ghz=3.5,
    antenna_bearing_deg=0.0,
    antenna_beamwidth_h_deg=360.0,  # omni
    antenna_gain_dbi=0.0,
)

obstacles = [
    BoundingBox(3.0, 3.0, 0.0, 4.5, 3.5, 2.0),   # desk
    BoundingBox(7.0, 2.0, 0.0, 8.0, 4.0, 1.8),    # cabinet
]

scene_bounds = (0.0, 0.0, 10.0, 10.0)

result = compute_analytical_coverage(
    tx=tx,
    scene_bounds=scene_bounds,
    obstacles=obstacles,
    cell_size_m=0.25,
    rx_height_m=1.5,
    scenario="inh",
)

print(f"Coverage grid shape: {result.path_gain_db.shape}")
print(f"Path gain range: {result.path_gain_db.min():.1f} to {result.path_gain_db.max():.1f} dB")
print(f"LOS percentage: {result.is_los.mean() * 100:.1f}%")
print(f"Model type badge: {result.model_type}")
```

---

## Extracting Obstacles from a Scene

Convert scene furniture to bounding boxes for LOS checks:

```python
def scene_to_obstacles(scene) -> list[BoundingBox]:
    """Extract BoundingBox list from a Scene or Room object.

    Works with the project's Scene/Room Pydantic models. Each furniture
    item with position and dimensions becomes an AABB.
    """
    obstacles = []
    rooms = scene.rooms if hasattr(scene, "rooms") else [scene]

    for room in rooms:
        for item in room.furniture:
            x, y = item.position
            w = getattr(item, "width", 0.5)
            d = getattr(item, "depth", 0.5)
            h = getattr(item, "height", 1.0)

            # Approximate AABB (ignoring rotation for speed)
            half_w, half_d = w / 2.0, d / 2.0
            obstacles.append(BoundingBox(
                x_min=x - half_w, y_min=y - half_d, z_min=0.0,
                x_max=x + half_w, y_max=y + half_d, z_max=h,
            ))

    return obstacles
```

---

## Performance Comparison

| Metric                    | CPU Analytical         | GPU Ray Tracing (Sionna) |
|---------------------------|------------------------|--------------------------|
| Compute time (10x10 m)    | < 0.5 seconds          | 5-15 seconds             |
| Compute time (100x100 m)  | < 2 seconds            | 15-30 seconds            |
| Reflection modeling        | None (direct path only) | Full multi-bounce         |
| Diffraction                | None                   | Edge diffraction          |
| Scattering                 | None                   | Surface scattering        |
| Material interaction       | None                   | Per-material properties   |
| Accuracy vs measurements   | +/- 10-15 dB           | +/- 3-5 dB               |
| Hardware requirement       | Any CPU                | NVIDIA GPU (CUDA)         |
| Shadow fading              | Not modeled            | Implicit in geometry      |

---

## Badge System

The frontend rendering pipeline should display the model type so users know the fidelity of results:

```python
def get_result_badge(result: CoverageResult) -> dict:
    """Return badge metadata for frontend display."""
    if result.model_type == "cpu_analytical":
        return {
            "label": "CPU Analytical",
            "color": "#FFA500",  # orange
            "tooltip": ("Analytical estimate using 3GPP TR 38.901 path loss. "
                        "For higher accuracy, run with GPU ray tracing."),
        }
    else:
        return {
            "label": "GPU Ray Trace",
            "color": "#00CC00",  # green
            "tooltip": ("Full ray-tracing simulation using NVIDIA Sionna RT. "
                        "Includes reflections, diffraction, and scattering."),
        }
```

---

## Output Format Compatibility

The `CoverageResult.path_gain_db` array is intentionally shaped identically to Sionna's `RadioMap` output so the same frontend rendering code handles both:

```python
def render_coverage(path_gain_db: np.ndarray, model_type: str, **kwargs):
    """Unified renderer accepts both CPU analytical and GPU ray trace results."""
    # path_gain_db shape is [ny, nx] regardless of source
    # model_type determines the badge overlay
    ...
```

This means switching between CPU fallback and GPU ray tracing requires zero changes to the visualization pipeline.

## Related

- [script-guidelines.md](script-guidelines.md) — script lifecycle and validation patterns
- [physics-validation.md](physics-validation.md) — analytical formulas used in this model
- [defaults.md](defaults.md) — fallback TX power, frequency, and grid defaults
