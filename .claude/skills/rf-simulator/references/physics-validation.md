# Physics Validation Reference

Analytical formulas and sanity bounds for validating RF simulation outputs.
All formulas sourced from ITU-R, 3GPP, and standard electromagnetics references.

---

## Free-Space Path Loss (FSPL)

The theoretical minimum path loss in an unobstructed environment.

### Formula

```
FSPL(d, f) = 20*log10(d) + 20*log10(f) - 147.55
```

Where:
- `d` = distance in meters
- `f` = frequency in Hz
- Result in dB

### Python Implementation

```python
import numpy as np

def fspl(distance_m: float, frequency_hz: float) -> float:
    """Free-space path loss in dB.

    Args:
        distance_m: Distance between TX and RX in meters (must be > 0).
        frequency_hz: Carrier frequency in Hz.

    Returns:
        Path loss in dB (positive value).
    """
    if distance_m <= 0:
        return 0.0
    return 20 * np.log10(distance_m) + 20 * np.log10(frequency_hz) - 147.55
```

### Expected FSPL Values (dB)

| Distance | 900 MHz | 2.4 GHz | 3.5 GHz | 5 GHz  | 28 GHz | 60 GHz |
|----------|---------|---------|---------|--------|--------|--------|
| 1 m      | 31.5    | 40.0    | 43.3    | 46.4   | 61.4   | 68.0   |
| 5 m      | 45.5    | 54.0    | 57.3    | 60.4   | 75.4   | 82.0   |
| 10 m     | 51.5    | 60.0    | 63.3    | 66.4   | 81.4   | 88.0   |
| 20 m     | 57.5    | 66.0    | 69.3    | 72.4   | 87.4   | 94.0   |
| 50 m     | 65.5    | 74.0    | 77.3    | 80.4   | 95.4   | 102.0  |

---

## ITU-R P.1238 Indoor Propagation Model

Models indoor path loss accounting for environment type and floor penetration.

### Formula

```
L_total = 20*log10(f_MHz) + N*log10(d) + L_f(n) - 28
```

Where:
- `f_MHz` = frequency in MHz
- `N` = distance power loss coefficient (environment-dependent)
- `d` = distance in meters
- `L_f(n)` = floor penetration loss factor for `n` floors
- Result in dB

### Distance Power Loss Coefficient N

| Environment  | 900 MHz | 1.8 GHz | 2.4 GHz | 3.5 GHz | 5 GHz | 28 GHz | 60 GHz |
|--------------|---------|---------|---------|---------|-------|--------|--------|
| Residential  | 28      | 30      | 28      | 30      | 31    | 33     | 35     |
| Office       | 33      | 32      | 30      | 31      | 31    | 33     | 35     |
| Commercial   | 22      | 22      | 26      | 28      | 28    | 30     | 32     |
| Corridor     | 18      | 19      | 20      | 20      | 22    | 24     | 26     |

Typical ranges:
- Residential: 28-33
- Office: 30-33
- Commercial: 22-28
- Corridor: 18-22

### Floor Penetration Loss

For office environments (typical):

```
L_f(n) = 15 + 4*(n - 1)     [dB]
```

Where `n` is the number of floors between TX and RX.

| Floors (n) | L_f (dB) |
|------------|----------|
| 1          | 15       |
| 2          | 19       |
| 3          | 23       |
| 4          | 27       |

### Python Implementation

```python
def itu_p1238(distance_m: float, frequency_hz: float, env: str = "office",
              n_floors: int = 0) -> float:
    """ITU-R P.1238 indoor path loss in dB.

    Args:
        distance_m: Distance between TX and RX in meters.
        frequency_hz: Carrier frequency in Hz.
        env: Environment type ("residential", "office", "commercial", "corridor").
        n_floors: Number of floors between TX and RX.

    Returns:
        Path loss in dB.
    """
    N_TABLE = {
        "residential": {900e6: 28, 1.8e9: 30, 2.4e9: 28, 3.5e9: 30, 5e9: 31},
        "office":      {900e6: 33, 1.8e9: 32, 2.4e9: 30, 3.5e9: 31, 5e9: 31},
        "commercial":  {900e6: 22, 1.8e9: 22, 2.4e9: 26, 3.5e9: 28, 5e9: 28},
        "corridor":    {900e6: 18, 1.8e9: 19, 2.4e9: 20, 3.5e9: 20, 5e9: 22},
    }
    n_vals = N_TABLE.get(env, N_TABLE["office"])
    closest_freq = min(n_vals.keys(), key=lambda f: abs(f - frequency_hz))
    N = n_vals[closest_freq]
    f_mhz = frequency_hz / 1e6

    if distance_m <= 0:
        return 0.0

    L = 20 * np.log10(f_mhz) + N * np.log10(distance_m) - 28

    if n_floors > 0:
        L += 15 + 4 * (n_floors - 1)

    return L
```

---

## 3GPP TR 38.901 Reference Values

### InH (Indoor Hotspot)

**LOS (Line of Sight):**
```
PL_InH_LOS = 32.4 + 17.3*log10(d_3D) + 20*log10(f_GHz)
```
- Valid for 1 m <= d_3D <= 150 m
- Shadow fading: sigma = 3.0 dB

Expected path loss at 3.5 GHz:
| d_3D | PL (dB) |
|------|---------|
| 1 m  | 43.3    |
| 5 m  | 55.4    |
| 10 m | 60.6    |
| 20 m | 65.8    |
| 50 m | 73.0    |

**NLOS (Non-Line of Sight):**
```
PL_InH_NLOS = 38.3*log10(d_3D) + 17.3 + 24.9*log10(f_GHz)
```
- Valid for 1 m <= d_3D <= 150 m
- Shadow fading: sigma = 8.03 dB

Expected path loss at 3.5 GHz:
| d_3D | PL (dB) |
|------|---------|
| 1 m  | 30.9    |
| 5 m  | 57.7    |
| 10 m | 69.2    |
| 20 m | 80.8    |
| 50 m | 95.6    |

### UMi (Urban Micro)

**LOS:**
```
PL_UMi_LOS = 32.4 + 21.0*log10(d_3D) + 20*log10(f_GHz)
```
- Valid for 10 m <= d_3D <= 5000 m
- Shadow fading: sigma = 4.0 dB

**NLOS:**
```
PL_UMi_NLOS = 22.4 + 35.3*log10(d_3D) + 21.3*log10(f_GHz) - 0.3*(h_UT - 1.5)
```
- Valid for 10 m <= d_3D <= 5000 m
- Shadow fading: sigma = 7.82 dB
- h_UT = UE antenna height (typical 1.5 m)

### UMa (Urban Macro)

**LOS:**
```
PL_UMa_LOS = 28.0 + 22.0*log10(d_3D) + 20*log10(f_GHz)
```
- Valid for 10 m <= d_3D <= 5000 m
- Shadow fading: sigma = 4.0 dB

**NLOS:**
```
PL_UMa_NLOS = 13.54 + 39.08*log10(d_3D) + 20*log10(f_GHz) - 0.6*(h_UT - 1.5)
```
- Valid for 10 m <= d_3D <= 5000 m
- Shadow fading: sigma = 6.0 dB

### Shadow Fading Summary

| Scenario   | Condition | Sigma (dB) |
|------------|-----------|------------|
| InH        | LOS       | 3.0        |
| InH        | NLOS      | 8.03       |
| UMi        | LOS       | 4.0        |
| UMi        | NLOS      | 7.82       |
| UMa        | LOS       | 4.0        |
| UMa        | NLOS      | 6.0        |

---

## Material Attenuation Bounds (ITU-R P.2040)

Expected wall/material penetration loss by material and frequency.

| Material      | 1 GHz   | 3.5 GHz | 5 GHz   | 28 GHz  | 60 GHz  |
|---------------|---------|---------|---------|---------|---------|
| Concrete      | 10-15   | 15-20   | 18-23   | 20-25   | 22-25   |
| Glass (std)   | 2-4     | 4-6     | 5-7     | 6-8     | 8-10    |
| Glass (IRR)   | 23-25   | 25-28   | 28-30   | 30-35   | 35-40   |
| Wood          | 3-5     | 5-7     | 6-8     | 8-10    | 9-10    |
| Plasterboard  | 2-4     | 3-5     | 4-6     | 5-7     | 6-8     |
| Brick         | 5-10    | 10-15   | 12-17   | 15-20   | 18-20   |
| Metal         | 40+     | 40+     | 40+     | 40+     | 40+     |

Notes:
- Glass (std) = standard clear glass
- Glass (IRR) = infrared-reflective / low-E coated glass
- Metal is effectively opaque at all frequencies
- Values in dB per wall/layer; actual loss depends on thickness and incidence angle

---

## Physics Check Thresholds

Summary of pass/fail thresholds for the 8 physics validation checks.

| # | Check                  | Threshold                                              | Severity |
|---|------------------------|--------------------------------------------------------|----------|
| 1 | RSS <= TX power        | max(rx_power) <= tx_power_dbm + 0.1 (tolerance)       | ERROR    |
| 2 | Path loss monotonicity | Spearman r(distance, path_loss) > 0.7                 | WARNING  |
| 3 | Free-space lower bound | No path loss below FSPL - 3 dB (measurement tolerance)| ERROR    |
| 4 | Dimensional consistency| All units verified (dBm, meters, Hz)                   | ERROR    |
| 5 | Reciprocity spot-check | |PL_AB - PL_BA| < 1 dB                                 | WARNING  |
| 6 | Analytical cross-check | RMSE vs ITU-R P.1238 < 6 dB (simple scenes only)      | WARNING  |
| 7 | Coverage continuity    | < 5% outlier cells (>20 dB from median of 8-neighbors)| WARNING  |
| 8 | Near-field guard       | Flag cells within lambda = c/f of TX                   | INFO     |

### Severity Definitions

- **ERROR**: Must pass. Failure means simulation output is physically wrong and should not be trusted.
- **WARNING**: Advisory. Failure suggests potential issues worth investigating but does not invalidate the simulation.
- **INFO**: Informational. Flagged for awareness but does not affect scoring or pass/fail determination.

### Physics Score Calculation

Points are awarded for each passing check:
- ERROR check (4 checks): 15 points each = 60 points max
- WARNING check (3 checks): 10 points each = 30 points max
- INFO check (1 check): 0 points (informational only)
- Total possible: 90 points, scaled to 0-100

```
raw_score = (passed_error_checks * 15) + (passed_warning_checks * 10)
physics_score = round(raw_score / 90 * 100)
```

## Related

- [sionna-materials.md](sionna-materials.md) — ITU-R P.2040 material attenuation constants
- [3gpp-models.md](3gpp-models.md) — 3GPP propagation models validated here
- [defaults.md](defaults.md) — validation bounds for simulation parameters
- [cpu-fallback.md](cpu-fallback.md) — analytical models used in CPU fallback
