# 3GPP TR 38.901 Path Loss Models

## Contents

1. [Indoor Hotspot (InH)](#indoor-hotspot-inh) — Office path loss (LOS/NLOS) for short-range indoor deployments
2. [Urban Micro (UMi)](#urban-micro-umi) — Street-level small-cell models with optional O2I penetration
3. [Urban Macro (UMa)](#urban-macro-uma) — Elevated base station models for wide-area urban coverage
4. [Rural Macro (RMa)](#rural-macro-rma) — Long-range propagation in open rural terrain
5. [Shadow Fading Summary](#shadow-fading-summary) — Log-normal shadow fading sigma values per scenario
6. [ITU-R P.676 Atmospheric Absorption](#itu-r-p676-atmospheric-absorption) — Oxygen and water vapor attenuation at mmWave frequencies

Reference for radio propagation path loss models defined in 3GPP TR 38.901 V17.0.0 (2022-03), applicable from 0.5 to 100 GHz.

All formulas use:
- `d_3D` or `d`: 3D distance in meters between TX and RX
- `d_2D`: 2D (horizontal) distance in meters
- `f_c`: carrier frequency in GHz
- `h_BS`: base station height in meters
- `h_UT`: user terminal height in meters
- `PL`: path loss in dB

---

## Indoor Hotspot (InH)

Use for: indoor deployments such as offices, shopping malls, factory floors. Typical cell radius 20-75 m.

### InH-LOS (Line of Sight)

```
PL_InH_LOS = 32.4 + 17.3 * log10(d_3D) + 20.0 * log10(f_c)
```

- Valid range: 1 m <= d_3D <= 150 m
- Shadow fading: sigma_SF = 3.0 dB (log-normal)
- Applicability: 0.5 - 100 GHz

### InH-NLOS (Non Line of Sight)

```
PL_InH_NLOS = max(PL_InH_LOS, PL'_InH_NLOS)

PL'_InH_NLOS = 17.3 + 38.3 * log10(d_3D) + 24.9 * log10(f_c)
```

- Valid range: 1 m <= d_3D <= 150 m
- Shadow fading: sigma_SF = 8.03 dB (log-normal)
- The `max()` ensures NLOS loss is never less than LOS loss at the same distance.

### InH LOS Probability

```
P_LOS = 1                             if d_2D <= 5 m
P_LOS = exp(-(d_2D - 5) / 70.8)      if 5 m < d_2D <= 49 m
P_LOS = exp(-(d_2D - 49) / 211.7) * 0.54   if d_2D > 49 m
```

### Python Implementation

```python
import numpy as np

def path_loss_inh_los(d_3d: float, f_ghz: float) -> float:
    """InH-LOS path loss in dB. 3GPP TR 38.901 Table 7.4.1-1."""
    return 32.4 + 17.3 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)

def path_loss_inh_nlos(d_3d: float, f_ghz: float) -> float:
    """InH-NLOS path loss in dB. 3GPP TR 38.901 Table 7.4.1-1."""
    pl_los = path_loss_inh_los(d_3d, f_ghz)
    pl_nlos = 17.3 + 38.3 * np.log10(d_3d) + 24.9 * np.log10(f_ghz)
    return max(pl_los, pl_nlos)

def los_probability_inh(d_2d: float) -> float:
    """InH LOS probability. 3GPP TR 38.901 Table 7.4.2-1."""
    if d_2d <= 5.0:
        return 1.0
    elif d_2d <= 49.0:
        return np.exp(-(d_2d - 5.0) / 70.8)
    else:
        return np.exp(-(d_2d - 49.0) / 211.7) * 0.54
```

---

## Urban Micro (UMi) - Street Canyon

Use for: small-cell outdoor deployments in dense urban areas with BS below rooftop. Typical cell radius 50-200 m. BS height ~10 m, UT height ~1.5-2.5 m.

### UMi-LOS

```
PL_UMi_LOS = 32.4 + 21.0 * log10(d_3D) + 20.0 * log10(f_c)
```

- Valid range: 10 m <= d_2D <= 5000 m
- Shadow fading: sigma_SF = 4.0 dB
- Default heights: h_BS = 10 m, h_UT = 1.5 m

### UMi-NLOS

```
PL_UMi_NLOS = max(PL_UMi_LOS, PL'_UMi_NLOS)

PL'_UMi_NLOS = 22.4 + 35.3 * log10(d_3D) + 21.3 * log10(f_c)
               - 0.3 * (h_UT - 1.5)
```

- Valid range: 10 m <= d_2D <= 5000 m
- Shadow fading: sigma_SF = 7.82 dB

### UMi LOS Probability

```
P_LOS = 1                                          if d_2D <= 18 m
P_LOS = (18 / d_2D) + exp(-d_2D / 36) * (1 - 18 / d_2D)   if d_2D > 18 m
```

### Python Implementation

```python
def path_loss_umi_los(d_3d: float, f_ghz: float) -> float:
    """UMi-LOS path loss in dB. 3GPP TR 38.901 Table 7.4.1-1."""
    return 32.4 + 21.0 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)

def path_loss_umi_nlos(d_3d: float, f_ghz: float, h_ut: float = 1.5) -> float:
    """UMi-NLOS path loss in dB. 3GPP TR 38.901 Table 7.4.1-1."""
    pl_los = path_loss_umi_los(d_3d, f_ghz)
    pl_nlos = (22.4 + 35.3 * np.log10(d_3d) + 21.3 * np.log10(f_ghz)
               - 0.3 * (h_ut - 1.5))
    return max(pl_los, pl_nlos)

def los_probability_umi(d_2d: float) -> float:
    """UMi LOS probability. 3GPP TR 38.901 Table 7.4.2-1."""
    if d_2d <= 18.0:
        return 1.0
    return (18.0 / d_2d) + np.exp(-d_2d / 36.0) * (1.0 - 18.0 / d_2d)
```

---

## Urban Macro (UMa)

Use for: macro-cell outdoor deployments with BS above rooftop. Typical cell radius 200-2000 m. BS height ~25 m, UT height ~1.5-2.5 m.

### UMa-LOS

```
PL_UMa_LOS = 28.0 + 22.0 * log10(d_3D) + 20.0 * log10(f_c)
```

- Valid range: 10 m <= d_2D <= 5000 m
- Shadow fading: sigma_SF = 4.0 dB
- Default heights: h_BS = 25 m, h_UT = 1.5 m
- Breakpoint distance: d_BP' = (4 * h_BS_eff * h_UT_eff * f_c * 1e9) / c
  - where h_BS_eff = h_BS - h_E, h_UT_eff = h_UT - h_E, h_E = 1.0 m

For d_2D > d_BP':
```
PL_UMa_LOS = 28.0 + 40.0 * log10(d_3D) + 20.0 * log10(f_c)
              - 9.0 * log10((d_BP')^2 + (h_BS - h_UT)^2)
```

### UMa-NLOS

```
PL_UMa_NLOS = max(PL_UMa_LOS, PL'_UMa_NLOS)

PL'_UMa_NLOS = 13.54 + 39.08 * log10(d_3D) + 20.0 * log10(f_c)
                - 0.6 * (h_UT - 1.5)
```

- Valid range: 10 m <= d_2D <= 5000 m
- Shadow fading: sigma_SF = 6.0 dB

### UMa LOS Probability

```
P_LOS = 1                                        if d_2D <= 18 m
P_LOS = (18/d_2D + exp(-d_2D/63) * (1 - 18/d_2D))
        * (1 + C'(h_UT) * (5/4) * (d_2D/100)^3 * exp(-d_2D/150))   if d_2D > 18 m

C'(h_UT) = 0                                     if h_UT <= 13 m
C'(h_UT) = ((h_UT - 13) / 10)^1.5               if 13 m < h_UT <= 23 m
```

### Python Implementation

```python
C_LIGHT = 3e8  # speed of light in m/s

def path_loss_uma_los(d_3d: float, d_2d: float, f_ghz: float,
                      h_bs: float = 25.0, h_ut: float = 1.5) -> float:
    """UMa-LOS path loss in dB. 3GPP TR 38.901 Table 7.4.1-1."""
    h_e = 1.0
    h_bs_eff = h_bs - h_e
    h_ut_eff = h_ut - h_e
    d_bp = 4.0 * h_bs_eff * h_ut_eff * f_ghz * 1e9 / C_LIGHT

    if d_2d <= d_bp:
        return 28.0 + 22.0 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
    else:
        return (28.0 + 40.0 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
                - 9.0 * np.log10(d_bp**2 + (h_bs - h_ut)**2))

def path_loss_uma_nlos(d_3d: float, d_2d: float, f_ghz: float,
                       h_bs: float = 25.0, h_ut: float = 1.5) -> float:
    """UMa-NLOS path loss in dB. 3GPP TR 38.901 Table 7.4.1-1."""
    pl_los = path_loss_uma_los(d_3d, d_2d, f_ghz, h_bs, h_ut)
    pl_nlos = (13.54 + 39.08 * np.log10(d_3d) + 20.0 * np.log10(f_ghz)
               - 0.6 * (h_ut - 1.5))
    return max(pl_los, pl_nlos)

def los_probability_uma(d_2d: float, h_ut: float = 1.5) -> float:
    """UMa LOS probability. 3GPP TR 38.901 Table 7.4.2-1."""
    if d_2d <= 18.0:
        return 1.0
    c_prime = 0.0
    if h_ut > 13.0:
        c_prime = ((h_ut - 13.0) / 10.0) ** 1.5
    p_base = 18.0 / d_2d + np.exp(-d_2d / 63.0) * (1.0 - 18.0 / d_2d)
    correction = 1.0 + c_prime * (5.0 / 4.0) * (d_2d / 100.0)**3 * np.exp(-d_2d / 150.0)
    return p_base * correction
```

---

## Rural Macro (RMa)

Use for: rural deployments with large cell sizes, low building density. Typical cell radius 1-10 km. BS height ~35 m, UT height ~1.5 m.

### RMa-LOS

```
PL_RMa_LOS = 20.0 * log10(40*pi*d_3D*f_c/3) + min(0.03*h^1.72, 10) * log10(d_3D)
              - min(0.044*h^1.72, 14.77) + 0.002*log10(h)*d_3D
```

where `h` is the average building height (default 5 m).

For d_2D > d_BP (breakpoint):
```
PL_RMa_LOS = 20.0 * log10(40*pi*d_BP*f_c/3) + min(0.03*h^1.72, 10) * log10(d_BP)
              - min(0.044*h^1.72, 14.77) + 0.002*log10(h)*d_BP
              + 40.0 * log10(d_3D / d_BP)
```

- Valid range: 10 m <= d_2D <= 10000 m (21000 m for large cells)
- Shadow fading: sigma_SF = 4.0 dB (LOS, d_2D <= d_BP), 6.0 dB (LOS, d_2D > d_BP)
- Frequency range: 0.5 - 30 GHz (note: narrower than other models)
- Default heights: h_BS = 35 m, h_UT = 1.5 m, h = 5 m, W = 20 m (street width)

### RMa-NLOS

```
PL_RMa_NLOS = max(PL_RMa_LOS, PL'_RMa_NLOS)

PL'_RMa_NLOS = 161.04 - 7.1*log10(W) + 7.5*log10(h)
                - (24.37 - 3.7*(h/h_BS)^2) * log10(h_BS)
                + (43.42 - 3.1*log10(h_BS)) * (log10(d_3D) - 3)
                + 20.0*log10(f_c) - (3.2*(log10(11.75*h_UT))^2 - 4.97)
```

- Shadow fading: sigma_SF = 8.0 dB

### RMa LOS Probability

```
P_LOS = 1                          if d_2D <= 10 m
P_LOS = exp(-(d_2D - 10) / 1000)  if d_2D > 10 m
```

### Python Implementation

```python
def path_loss_rma_los(d_3d: float, d_2d: float, f_ghz: float,
                      h_bs: float = 35.0, h_ut: float = 1.5,
                      h: float = 5.0, w: float = 20.0) -> float:
    """RMa-LOS path loss in dB. 3GPP TR 38.901 Table 7.4.1-1.
    Only valid for 0.5-30 GHz.
    """
    d_bp = 2.0 * np.pi * h_bs * h_ut * f_ghz * 1e9 / C_LIGHT

    term1 = 20.0 * np.log10(40.0 * np.pi * min(d_3d, d_bp) * f_ghz / 3.0)
    term2 = min(0.03 * h**1.72, 10.0) * np.log10(min(d_3d, d_bp))
    term3 = min(0.044 * h**1.72, 14.77)
    term4 = 0.002 * np.log10(h) * min(d_3d, d_bp)

    pl = term1 + term2 - term3 + term4

    if d_2d > d_bp:
        pl += 40.0 * np.log10(d_3d / d_bp)

    return pl

def path_loss_rma_nlos(d_3d: float, d_2d: float, f_ghz: float,
                       h_bs: float = 35.0, h_ut: float = 1.5,
                       h: float = 5.0, w: float = 20.0) -> float:
    """RMa-NLOS path loss in dB. 3GPP TR 38.901 Table 7.4.1-1."""
    pl_los = path_loss_rma_los(d_3d, d_2d, f_ghz, h_bs, h_ut, h, w)
    pl_nlos = (161.04 - 7.1 * np.log10(w) + 7.5 * np.log10(h)
               - (24.37 - 3.7 * (h / h_bs)**2) * np.log10(h_bs)
               + (43.42 - 3.1 * np.log10(h_bs)) * (np.log10(d_3d) - 3.0)
               + 20.0 * np.log10(f_ghz)
               - (3.2 * (np.log10(11.75 * h_ut))**2 - 4.97))
    return max(pl_los, pl_nlos)

def los_probability_rma(d_2d: float) -> float:
    """RMa LOS probability. 3GPP TR 38.901 Table 7.4.2-1."""
    if d_2d <= 10.0:
        return 1.0
    return np.exp(-(d_2d - 10.0) / 1000.0)
```

---

## Shadow Fading Summary

| Scenario   | Condition | sigma_SF (dB) |
|------------|-----------|---------------|
| InH-LOS    | --        | 3.0           |
| InH-NLOS   | --        | 8.03          |
| UMi-LOS    | --        | 4.0           |
| UMi-NLOS   | --        | 7.82          |
| UMa-LOS    | --        | 4.0           |
| UMa-NLOS   | --        | 6.0           |
| RMa-LOS    | d <= d_BP | 4.0           |
| RMa-LOS    | d > d_BP  | 6.0           |
| RMa-NLOS   | --        | 8.0           |

Shadow fading is modeled as log-normal: add `sigma_SF * N(0,1)` dB to the deterministic PL. It is spatially correlated with decorrelation distances specified in TR 38.901 Table 7.5-6.

---

## ITU-R P.676 Atmospheric Absorption (Simplified)

Atmospheric absorption becomes significant above ~10 GHz and is critical at mmWave/sub-THz frequencies. The simplified model below is suitable for clear-air conditions at sea level.

### Key Absorption Peaks

| Frequency     | Absorber     | Specific attenuation |
|---------------|-------------|----------------------|
| 22.235 GHz    | Water vapor  | ~0.2 dB/km (moderate humidity) |
| 60 GHz        | Oxygen       | ~15 dB/km            |
| 120 GHz       | Oxygen       | ~1.5 dB/km           |
| 183.3 GHz     | Water vapor  | ~30 dB/km            |

### Simplified Model

For quick estimation in the 1-100 GHz range:

```python
def atmospheric_absorption_db_per_km(f_ghz: float,
                                      temperature_c: float = 20.0,
                                      pressure_hpa: float = 1013.25,
                                      humidity_pct: float = 50.0) -> float:
    """Simplified ITU-R P.676 atmospheric absorption in dB/km.

    Accurate to within ~20% for 1-100 GHz at sea level.
    For precise calculations above 100 GHz, use the full ITU-R P.676 model.
    """
    # Water vapor density from relative humidity (simplified)
    es = 6.1121 * np.exp((18.678 - temperature_c / 234.5)
                          * temperature_c / (257.14 + temperature_c))
    rho_w = humidity_pct / 100.0 * es * 216.7 / (temperature_c + 273.15)

    # Dry air (oxygen) component
    gamma_o = 0.0
    if f_ghz <= 57.0:
        gamma_o = (7.2 * (f_ghz**2) / (f_ghz**2 + 0.34)
                   + 0.62 / ((54.0 - f_ghz)**1.16 + 0.83)) * f_ghz**2 * 1e-3
    elif f_ghz <= 63.0:
        # O2 absorption band - very high attenuation
        gamma_o = 15.0 * np.exp(-0.5 * ((f_ghz - 60.0) / 1.5)**2)
    elif f_ghz <= 98.0:
        gamma_o = (0.2 + 0.5 / ((f_ghz - 63.0)**1.6 + 1.5)) * f_ghz**2 * 1e-3
    else:
        gamma_o = (2.0 / ((f_ghz - 118.75)**2 + 1.0) + 0.01) * f_ghz * 1e-2

    # Water vapor component
    gamma_w = (0.067 + 3.0 / ((f_ghz - 22.235)**2 + 5.0)
               + 7.0 / ((f_ghz - 183.3)**2 + 6.0)) * f_ghz**2 * rho_w * 1e-4

    return gamma_o + gamma_w

def total_atmospheric_loss(d_km: float, f_ghz: float, **kwargs) -> float:
    """Total atmospheric absorption loss over a given distance."""
    return atmospheric_absorption_db_per_km(f_ghz, **kwargs) * d_km
```

### When Atmospheric Absorption Matters

| Frequency Band | Typical Range  | Attenuation  | Impact                   |
|----------------|---------------|--------------|--------------------------|
| Sub-6 GHz      | < 6 GHz       | < 0.01 dB/km | Negligible               |
| FR2 (mmWave)   | 24-52 GHz     | 0.05-0.2 dB/km | Minor for short range  |
| 60 GHz (V-band)| 57-71 GHz     | 10-15 dB/km  | Limits range to ~200 m   |
| D-band         | 110-170 GHz   | 0.5-3 dB/km  | Moderate, usable         |
| Sub-THz        | 200-300 GHz   | 1-30 dB/km   | Window-dependent         |

---

## Model Selection Guide

```
Is the scenario indoors?
  YES -> InH (Office, Open-plan, Shopping Mall, Factory)
  NO  -> Is the BS above rooftop?
           YES -> Is the area rural?
                    YES -> RMa (farms, highways, open areas)
                    NO  -> UMa (urban/suburban macro cells)
           NO  -> UMi (street-level small cells, dense urban)
```

### Quick Reference

| Parameter               | InH        | UMi         | UMa         | RMa          |
|--------------------------|-----------|-------------|-------------|--------------|
| Frequency range          | 0.5-100   | 0.5-100     | 0.5-100     | 0.5-30       |
| d_2D range (m)           | 1-150     | 10-5000     | 10-5000     | 10-10000     |
| Default h_BS (m)         | 3         | 10          | 25          | 35           |
| Default h_UT (m)         | 1-2.5     | 1.5-2.5     | 1.5-2.5     | 1.5          |
| Typical use              | WiFi, 5G indoor | 5G small cell | 4G/5G macro | 4G rural  |

### Combined Path Loss with Atmospheric Absorption

For frequencies above 10 GHz, always add atmospheric absorption:

```python
def total_path_loss(d_3d: float, d_2d: float, f_ghz: float,
                    scenario: str, is_los: bool,
                    h_bs: float = None, h_ut: float = 1.5) -> float:
    """Combined 3GPP path loss + atmospheric absorption."""
    # Select path loss model
    models = {
        ("inh", True): lambda: path_loss_inh_los(d_3d, f_ghz),
        ("inh", False): lambda: path_loss_inh_nlos(d_3d, f_ghz),
        ("umi", True): lambda: path_loss_umi_los(d_3d, f_ghz),
        ("umi", False): lambda: path_loss_umi_nlos(d_3d, f_ghz, h_ut),
        ("uma", True): lambda: path_loss_uma_los(d_3d, d_2d, f_ghz, h_bs or 25.0, h_ut),
        ("uma", False): lambda: path_loss_uma_nlos(d_3d, d_2d, f_ghz, h_bs or 25.0, h_ut),
        ("rma", True): lambda: path_loss_rma_los(d_3d, d_2d, f_ghz, h_bs or 35.0, h_ut),
        ("rma", False): lambda: path_loss_rma_nlos(d_3d, d_2d, f_ghz, h_bs or 35.0, h_ut),
    }

    pl = models[(scenario.lower(), is_los)]()

    # Add atmospheric absorption for mmWave and above
    if f_ghz > 10.0:
        d_km = d_3d / 1000.0
        pl += total_atmospheric_loss(d_km, f_ghz)

    return pl
```

## Related

- [physics-validation.md](physics-validation.md) — validation formulas and sanity bounds for path loss
- [defaults.md](defaults.md) — default frequency, power, and scenario parameters
- [antenna-patterns.md](antenna-patterns.md) — antenna configurations used with these models
