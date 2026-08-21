# Channel Models Reference

## Contents

1. [Model Selection Guide](#model-selection-guide) -- Pick the right model for the scenario
2. [CDL Models (A-E)](#cdl-models-a-e) -- Point-to-point link with spatial clusters
3. [TDL Models (A-E)](#tdl-models-a-e) -- Single-antenna or reduced-complexity channels
4. [Stochastic Multi-User: UMi, UMa, RMa](#stochastic-multi-user-umi-uma-rma) -- Multi-cell network channels
5. [Simple Channels: AWGN, Rayleigh, Rician](#simple-channels-awgn-rayleigh-rician) -- Baseline and analytical comparison
6. [CIR Output Shape](#cir-output-shape) -- Tensor dimensions for all stochastic models

---

## Model Selection Guide

> **Red flags — stop and re-read this section if any of these appear:**
> - `CDL(...)` in a script that also mentions "users", "uplink", "interference", "multi-cell", or multiple TX positions → **use UMi/UMa/RMa instead**
> - `UMi`/`UMa`/`RMa` without `channel_model.set_topology(*topology)` before the first CIR call → RuntimeError about unset topology
> - Single-antenna, delay-only simulation wrapped in CDL → wasted compute and identical tap powers → **use TDL**

When simulating a **single point-to-point MIMO link** (one TX, one RX), use CDL. Failure to do so gives unrealistic spatial correlation -- CDL models define explicit angular clusters.

When simulating a **single-antenna or delay-only** channel (no spatial structure needed), use TDL. Using CDL with 1 antenna wastes computation and gives identical tap powers.

When simulating **multi-user / multi-cell networks** with topology-dependent pathloss and LOS probability, use UMi/UMa/RMa. These embed 3GPP path loss, shadow fading, and LOS/NLOS state.

When **benchmarking against analytical BER** curves or testing coded systems in isolation, use AWGN or Rayleigh.

| Scenario | Model | Why |
|----------|-------|-----|
| 5G NR link-level, MIMO | CDL-A to E | Standardized spatial clusters |
| Quick fading test, SISO | TDL-A to E | No antenna arrays needed |
| Urban small cell, multi-user | UMi | Street-level, 10m BS height |
| Urban macro tower | UMa | 25m BS, wide-area coverage |
| Rural / highway | RMa | Long range, low density |
| Coded BER baseline | AWGN | Ideal, no fading |
| Flat fading, diversity analysis | Rayleigh / Rician | Classical block fading |

---

## CDL Models (A-E)

When using CDL, always set `delay_spread` -- it scales the per-cluster delays. Omitting it uses the default (100 ns) which may not match your scenario.

```python
import sionna as sn
from sionna.phy.channel.tr38901 import CDL, PanelArray, Antenna

carrier_frequency = 3.5e9

# TX: single UT antenna
ut_array = Antenna(polarization="single",
                   polarization_type="V",
                   antenna_pattern="omni",
                   carrier_frequency=carrier_frequency)

# RX: 4x2 dual-pol panel = 16 elements
bs_array = PanelArray(num_rows_per_panel=4,
                      num_cols_per_panel=2,
                      polarization="dual",
                      polarization_type="cross",
                      antenna_pattern="38.901",
                      carrier_frequency=carrier_frequency)

cdl = CDL("C",                          # Model: A, B, C, D, or E
          delay_spread=100e-9,           # Nominal delay spread [s]
          carrier_frequency=carrier_frequency,
          ut_array=ut_array,
          bs_array=bs_array,
          direction="uplink",            # UT transmits
          min_speed=3.0)                 # UT speed [m/s]

# Generate CIR: (a, tau) tensors
# batch_size=64, num_time_steps=14 (one per OFDM symbol)
cir = cdl(batch_size=64, num_time_steps=14, sampling_frequency=15e3 * 76)
a, tau = cir  # a: complex coeffs, tau: path delays
```

**CDL model selection:**
- A, B, C: NLOS profiles (A = short delay, B = medium, C = long delay spread)
- D: LOS with strong direct path (Rician K-factor ~13 dB)
- E: LOS with shorter delay spread than D

---

## TDL Models (A-E)

When using TDL with OFDM, pass `sampling_frequency=resource_grid.bandwidth`. Mismatched sampling rates produce incorrect tap spacing.

```python
from sionna.phy.channel.tr38901 import TDL

tdl = TDL("A",                          # Model: A, B, C, D, or E
          delay_spread=300e-9,
          carrier_frequency=3.5e9,
          min_speed=3.0,
          num_sinusoids=20,              # Sum-of-sinusoids for Doppler (default 20)
          num_rx_ant=1,                  # TDL: specify antenna count directly
          num_tx_ant=1)

cir = tdl(batch_size=64, num_time_steps=14, sampling_frequency=15e3 * 76)
```

---

## Stochastic Multi-User: UMi, UMa, RMa

When using UMi/UMa/RMa, always call `set_topology()` before generating CIR. Forgetting this raises a RuntimeError about unset topology.

When you need a quick multi-user topology without manual placement, use `gen_single_sector_topology()` or `gen_hexgrid_topology()`.

```python
from sionna.phy.channel.tr38901 import UMi, UMa, RMa, gen_single_sector_topology

channel_model = UMi(carrier_frequency=3.5e9,
                    o2i_model="low",         # Outdoor-to-indoor penetration
                    ut_array=ut_array,
                    bs_array=bs_array,
                    direction="uplink",
                    enable_pathloss=True,     # Includes distance-dependent loss
                    enable_shadow_fading=True)

# Quick single-sector topology: 10 UTs
topology = gen_single_sector_topology(
    batch_size=64,
    num_ut=10,
    scenario="umi",
    min_bs_ut_dist=10.0,
    max_bs_ut_dist=200.0)

# topology returns: (ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state)
channel_model.set_topology(*topology)

# Generate CIR
cir = channel_model(batch_size=64, num_time_steps=14)
```

**UMa vs UMi vs RMa:**
- UMi: BS height ~10m, ISD 200m, street-level. Use `o2i_model="low"` or `"high"`.
- UMa: BS height ~25m, ISD 500m, rooftop macro. Also needs `o2i_model`.
- RMa: BS height ~35m, ISD 1732m. Needs `average_street_width` and `average_building_height`.

---

## Simple Channels: AWGN, Rayleigh, Rician

```python
from sionna.phy.channel import AWGN, RayleighBlockFading, GenerateOFDMChannel

# AWGN -- no fading, additive noise only
awgn = AWGN()
y = awgn(x, no=0.1)  # no = noise variance (N0)

# Rayleigh block fading -- flat fading, constant within a block
rayleigh = RayleighBlockFading(num_rx=1, num_rx_ant=1,
                               num_tx=1, num_tx_ant=1)
cir = rayleigh(batch_size=64, num_time_steps=1)

# Convert to OFDM channel
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
frequencies = subcarrier_frequencies(fft_size=72, subcarrier_spacing=15e3)
h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True)
```

---

## CIR Output Shape

All stochastic channel models return `(a, tau)`:

```
a:   [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
tau: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
     (or [batch_size, num_rx, num_tx, num_paths] for models without per-antenna delays)
```

When converting CIR to OFDM channel with `cir_to_ofdm_channel()`, the output shape is:
```
h_freq: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
```

When batch dimension is missing from `a`, add it with `a.unsqueeze(0)`. Shape mismatches between `a` and downstream OFDM layers are the most common channel model bug.
