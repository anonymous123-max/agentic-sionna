# Antenna Patterns and Array Configurations

## Contents

1. [Built-in Antenna Patterns](#built-in-antenna-patterns) — Isotropic, dipole, 3GPP 38.901, and custom pattern definitions
2. [Polarization](#polarization) — Slant angles and cross-pol discrimination settings
3. [PlanarArray Parameters](#planararray-parameters) — Row/column counts, spacing, and element configuration
4. [Common Configurations](#common-configurations) — Ready-to-use array configs for typical deployment scenarios
5. [Typical TX/RX Array Pairings](#typical-txrx-array-pairings) — Recommended transmitter/receiver array combinations
6. [Assigning Arrays to Scene](#assigning-arrays-to-scene) — How to attach antenna arrays to Sionna scene objects

## Built-in Antenna Patterns

### Isotropic (`"iso"`)

Radiates equally in all directions. No directivity gain. Used as baseline reference and for omnidirectional receivers.

```python
array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)
```

- Gain: 0 dBi
- Use cases: reference simulations, mobile UE, omnidirectional sensors

### Half-Wave Dipole (`"dipole"`)

Classic dipole pattern with a toroidal radiation pattern. Null along the dipole axis, maximum perpendicular to it.

```python
array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="dipole",
    polarization="V",
)
```

- Gain: 2.15 dBi
- Use cases: simple base stations, Wi-Fi APs, realistic UE modeling

### 3GPP TR 38.901 (`"tr38901"`)

Standardized antenna element pattern from 3GPP TR 38.901 Section 7.3. Directional pattern with configurable beamwidth. Used for 5G NR simulations.

```python
array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="V",
)
```

- Gain: ~8 dBi (single element)
- 3 dB beamwidth: 65 degrees (azimuth and elevation)
- Front-to-back ratio: 30 dB
- Use cases: 5G NR base stations, sector antennas, compliant simulations

---

## Polarization

| Value | Description | Antenna Count |
|---|---|---|
| `"V"` | Vertical polarization | 1 antenna per element position |
| `"H"` | Horizontal polarization | 1 antenna per element position |
| `"VH"` | Dual/cross-polarized (V+H) | 2 antennas per element position |

### Cross-Polarized Arrays

With `"VH"`, each physical element position has two antenna ports (V and H). This doubles the number of antenna ports relative to `"V"` or `"H"`.

```python
# 4x4 cross-polarized: 4*4*2 = 32 antenna ports
array = sionna.rt.PlanarArray(
    num_rows=4, num_cols=4,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="VH",
)
# Total antenna ports: num_rows * num_cols * 2 = 32
```

---

## PlanarArray Parameters

```python
sionna.rt.PlanarArray(
    num_rows,            # int: number of rows in the array
    num_cols,            # int: number of columns in the array
    vertical_spacing,    # float: row spacing in wavelengths (typically 0.5)
    horizontal_spacing,  # float: column spacing in wavelengths (typically 0.5)
    pattern,             # str: "iso", "dipole", or "tr38901"
    polarization,        # str: "V", "H", or "VH"
)
```

**Spacing convention:** Specified in wavelengths (lambda). At `scene.frequency = 3.5e9`, one wavelength is ~85.7 mm, so `0.5` spacing = ~42.9 mm.

**Total antenna ports:**
- `"V"` or `"H"`: `num_rows * num_cols`
- `"VH"`: `num_rows * num_cols * 2`

---

## Common Configurations

### Single Isotropic Element (Baseline)

```python
# 1 antenna port, omnidirectional
# Use for: baseline simulations, simple UE
array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)
# Ports: 1
```

### Single Dipole

```python
# 1 antenna port, dipole pattern
# Use for: realistic single-antenna UE
array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="dipole",
    polarization="V",
)
# Ports: 1
```

### 2x2 Patch (Small MIMO)

```python
# 4 antenna ports (or 8 with VH)
# Use for: Wi-Fi AP, small cell, UE with multiple antennas
array = sionna.rt.PlanarArray(
    num_rows=2, num_cols=2,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="V",
)
# Ports: 4
```

### 4x4 MIMO

```python
# 16 ports (or 32 with VH)
# Use for: macro cell, 5G NR gNB
array = sionna.rt.PlanarArray(
    num_rows=4, num_cols=4,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="VH",
)
# Ports: 32
```

### 8x8 Massive MIMO

```python
# 64 ports (or 128 with VH)
# Use for: massive MIMO base station, mmWave gNB
array = sionna.rt.PlanarArray(
    num_rows=8, num_cols=8,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="VH",
)
# Ports: 128
```

### 16x16 Large Array (mmWave)

```python
# 256 ports (or 512 with VH)
# Use for: mmWave massive MIMO, beamforming testbeds
array = sionna.rt.PlanarArray(
    num_rows=16, num_cols=16,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="VH",
)
# Ports: 512
```

---

## Typical TX/RX Array Pairings

| Scenario | TX Array | RX Array |
|---|---|---|
| Basic coverage study | 1x1 iso V | 1x1 iso V |
| Indoor Wi-Fi | 2x2 tr38901 VH | 1x1 iso V |
| Outdoor macro cell | 4x4 tr38901 VH | 1x1 dipole V |
| 5G NR sub-6 GHz | 8x8 tr38901 VH | 2x2 tr38901 V |
| 5G NR mmWave | 16x16 tr38901 VH | 4x4 tr38901 VH |
| Massive MIMO research | 8x8 tr38901 VH | 1x1 iso V |

---

## Assigning Arrays to Scene

```python
# TX and RX can have different arrays
scene.tx_array = tx_array
scene.rx_array = rx_array

# Or the same array for both
array = sionna.rt.PlanarArray(1, 1, 0.5, 0.5, "iso", "V")
scene.tx_array = array
scene.rx_array = array
```

---

## Beamforming

### Zero-Forcing Precoder

```python
from sionna.phy.mimo import zero_forcing_precoder

# Get channel matrix from CIR
a, tau = paths.cir(out_type="torch")

# a shape: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
# Sum over paths to get narrowband channel matrix H
H = a.sum(dim=-1)  # [batch, num_rx, num_rx_ant, num_tx, num_tx_ant]

# Reshape to [batch, num_rx * num_rx_ant, num_tx * num_tx_ant]
num_rx = H.shape[1]
num_rx_ant = H.shape[2]
num_tx = H.shape[3]
num_tx_ant = H.shape[4]
H_mat = H.reshape(-1, num_rx * num_rx_ant, num_tx * num_tx_ant)

# Compute ZF precoding weights
# W = H^H (H H^H)^{-1}
W = zero_forcing_precoder(H_mat, power=1.0)
```

### Matched Filter Beamforming

```python
import torch

# Simple matched filter (conjugate beamforming)
# H: [batch, num_rx_ant, num_tx_ant]
H = a[..., 0, :, :, :].sum(dim=-1)  # first RX, sum paths

# Matched filter weights: conjugate of channel
w_mf = H.conj()  # [batch, num_rx_ant, num_tx_ant]

# Normalize
w_mf = w_mf / torch.norm(w_mf, dim=-1, keepdim=True)
```

---

## Directivity Gain Computation (Analytical Fallback)

When RT is not available (no GPU, no scene), use analytical antenna gain for link budget calculations.

### Isotropic Gain

```python
# Isotropic: 0 dBi in all directions
G_iso_dBi = 0.0
```

### Dipole Gain

```python
import numpy as np

def dipole_gain_dBi(theta):
    """Half-wave dipole gain as function of elevation angle theta (from z-axis).

    Args:
        theta: elevation angle in radians (0 = along dipole axis, pi/2 = broadside)

    Returns:
        Gain in dBi
    """
    sin_theta = np.sin(theta)
    if sin_theta < 1e-10:
        return -np.inf  # null along axis
    # Dipole radiation pattern
    numerator = np.cos(np.pi / 2 * np.cos(theta))
    E = numerator / sin_theta
    G_linear = 1.64 * E**2  # 1.64 = max directivity of half-wave dipole
    return 10 * np.log10(G_linear)

# Peak gain (broadside)
print(f"Dipole peak gain: {dipole_gain_dBi(np.pi/2):.2f} dBi")  # ~2.15 dBi
```

### 3GPP TR 38.901 Element Gain

```python
import numpy as np

def tr38901_element_gain_dBi(theta_deg, phi_deg,
                              theta_3dB=65.0, phi_3dB=65.0,
                              A_max=30.0, G_E_max=8.0):
    """3GPP TR 38.901 Section 7.3 antenna element pattern.

    Args:
        theta_deg: elevation angle in degrees (0 = horizon, 90 = zenith)
        phi_deg: azimuth angle in degrees
        theta_3dB: vertical 3dB beamwidth in degrees (default 65)
        phi_3dB: horizontal 3dB beamwidth in degrees (default 65)
        A_max: front-to-back ratio in dB (default 30)
        G_E_max: maximum element gain in dBi (default 8)

    Returns:
        Element gain in dBi
    """
    # Vertical cut
    A_V = -min(12 * (theta_deg / theta_3dB) ** 2, A_max)
    # Horizontal cut
    A_H = -min(12 * (phi_deg / phi_3dB) ** 2, A_max)
    # Combined
    A_dB = -min(-(A_V + A_H), A_max)
    return G_E_max + A_dB

# Boresight gain
print(f"TR38.901 boresight: {tr38901_element_gain_dBi(0, 0):.1f} dBi")  # 8.0 dBi
```

### Array Gain (Analytical)

```python
def array_gain_dBi(num_elements, element_gain_dBi=0.0):
    """Maximum coherent array gain (all elements in phase).

    Args:
        num_elements: total number of antenna elements
        element_gain_dBi: single element gain in dBi

    Returns:
        Maximum array gain in dBi
    """
    return element_gain_dBi + 10 * np.log10(num_elements)

# Examples
print(f"1x1 iso:     {array_gain_dBi(1, 0.0):.1f} dBi")    # 0.0 dBi
print(f"2x2 iso:     {array_gain_dBi(4, 0.0):.1f} dBi")    # 6.0 dBi
print(f"4x4 tr38901: {array_gain_dBi(16, 8.0):.1f} dBi")   # 20.0 dBi
print(f"8x8 tr38901: {array_gain_dBi(64, 8.0):.1f} dBi")   # 26.1 dBi
print(f"8x8 VH:      {array_gain_dBi(128, 8.0):.1f} dBi")  # 29.1 dBi
```

---

## Complete Example: MIMO Link with Beamforming

```python
import sionna
import sionna.rt
import torch

# Load scene
scene = sionna.rt.load_scene(sionna.rt.scene.munich, merge_shapes=False)
scene.frequency = 3.5e9

# 4x4 cross-pol TX (gNB)
tx_array = sionna.rt.PlanarArray(
    num_rows=4, num_cols=4,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="tr38901",
    polarization="VH",
)

# 1x1 single-pol RX (UE)
rx_array = sionna.rt.PlanarArray(
    num_rows=1, num_cols=1,
    vertical_spacing=0.5, horizontal_spacing=0.5,
    pattern="iso",
    polarization="V",
)

scene.tx_array = tx_array  # 32 ports
scene.rx_array = rx_array  # 1 port

# Add TX/RX
tx = sionna.rt.Transmitter("gnb", position=[8.5, 21.0, 27.0], power_dbm=23.0)
rx = sionna.rt.Receiver("ue", position=[45.0, 90.0, 1.5])
scene.add(tx)
scene.add(rx)

# Compute paths
solver = sionna.rt.PathSolver()
paths = solver(
    scene=scene,
    max_depth=5,
    num_samples=5_000_000,
    interaction_types="all",
)

# Get CIR
a, tau = paths.cir(out_type="torch")
print(f"CIR shape: a={a.shape}, tau={tau.shape}")
# a: [1, 1, 1, 1, 32, num_paths]  (batch=1, 1 RX, 1 RX ant, 1 TX, 32 TX ant, paths)

# Narrowband channel: sum over paths
h = a.sum(dim=-1)  # [1, 1, 1, 1, 32]
h = h.squeeze()     # [32] -- channel vector from 32 TX antennas to 1 RX antenna

# Matched filter beamforming
w = h.conj() / torch.norm(h)  # normalize

# Beamforming gain
bf_gain = torch.abs(torch.dot(w, h)) ** 2
bf_gain_dB = 10 * torch.log10(bf_gain).item()
print(f"Beamforming gain: {bf_gain_dB:.1f} dB")
```

## Related

- [sionna-v2-api.md](sionna-v2-api.md) — PlanarArray API and scene antenna assignment
- [defaults.md](defaults.md) — default antenna pattern and array configuration
- [3gpp-models.md](3gpp-models.md) — 3GPP TR 38.901 antenna element pattern spec
