# Static Knowledge — RF/Wireless Domain Constants

> **Update policy: FROZEN.** This file contains physical constants, standards
> tables, and mathematical formulas that change only when standards bodies
> publish new specifications. Do not modify during automated skill updates.
> Changes require domain expert review.

## Contents

1. [Material Properties](#material-properties) — ITU-R P.2040 conductivity and permittivity
2. [3GPP Channel Model Parameters](#3gpp-channel-model-parameters) — CDL delay profiles, path loss exponents
3. [OFDM Numerology](#ofdm-numerology) — 5G NR subcarrier spacings and slot structures
4. [Path Loss Models](#path-loss-models) — Friis, 3GPP TR 38.901 formulas
5. [Fundamental Formulas](#fundamental-formulas) — Shannon, thermal noise, link budget
6. [Sionna Module Map](#sionna-module-map) — Which class handles which task

---

## Material Properties

ITU-R P.2040-3 (2023). Frequency-dependent penetration loss (dB per layer):

| Material | 1 GHz | 3.5 GHz | 5 GHz | 28 GHz | 60 GHz |
|---|---|---|---|---|---|
| Concrete | 10-15 | 15-20 | 18-23 | 20-25 | 22-25 |
| Glass (standard) | 2-4 | 4-6 | 5-7 | 6-8 | 8-10 |
| Glass (IRR/low-E) | 23-25 | 25-28 | 28-30 | 30-35 | 35-40 |
| Plasterboard | 2-4 | 3-5 | 4-6 | 5-7 | 6-8 |
| Wood | 3-5 | 5-7 | 6-8 | 8-10 | 9-10 |
| Brick | 5-10 | 10-15 | 12-17 | 15-20 | 18-20 |
| Metal | 40+ | 40+ | 40+ | 40+ | 40+ |

Relative permittivity (εr) at 3.5 GHz:

| Material | εr | σ (S/m) | Sionna name |
|---|---|---|---|
| Concrete | 5.31 | 0.0326 | `itu_concrete` |
| Brick | 3.75 | 0.038 | `itu_brick` |
| Plasterboard | 2.94 | 0.0116 | `itu_plasterboard` |
| Glass | 6.27 | 0.0043 | `itu_glass` |
| Wood | 1.99 | 0.0047 | `itu_wood` |
| Metal | 1.0 | 1e7 | `itu_metal` |

---

## 3GPP Channel Model Parameters

### CDL Delay Profiles (TR 38.901 Table 7.7.1-1)

| Model | RMS Delay Spread | Ricean K | Use case |
|---|---|---|---|
| CDL-A | 129 ns | - | NLOS, medium delay |
| CDL-B | 634 ns | - | NLOS, long delay |
| CDL-C | 171 ns | - | NLOS, short delay |
| CDL-D | 14.4 ns | 13.3 dB | LOS, strong direct path |
| CDL-E | 51.4 ns | 22.0 dB | LOS, dominant direct path |

### Path Loss Exponents (TR 38.901 Table 7.4.1-1)

| Scenario | LOS | NLOS |
|---|---|---|
| UMi Street Canyon | 2.1 | 3.19 |
| UMa | 2.2 | 2.9 |
| RMa | 2.0 | 3.4 |
| InH Office | 1.73 | 3.83 |

### Standard Antenna Patterns

| Pattern | 3dB beamwidth | Max gain | Use |
|---|---|---|---|
| Isotropic | 360° | 0 dBi | Baseline, analytical |
| Dipole | ~78° (H) | 2.15 dBi | Simple analysis |
| TR 38.901 | configurable | ~8 dBi per element | 5G NR evaluation |

---

## OFDM Numerology

5G NR (TS 38.211):

| μ | SCS (kHz) | Slot duration (ms) | Symbols/slot | CP samples (normal) | Max BW (MHz) |
|---|---|---|---|---|---|
| 0 | 15 | 1.0 | 14 | 144 | 50 |
| 1 | 30 | 0.5 | 14 | 144 | 100 |
| 2 | 60 | 0.25 | 14 | 144 | 200 |
| 3 | 120 | 0.125 | 14 | 144 | 400 |

FFT sizes: 128, 256, 512, 1024, 2048, 4096.
Guard band subcarriers: ~10% of FFT size on each edge.

---

## Path Loss Models

### Free-Space Path Loss (Friis)
```
FSPL(d, f) = 20·log₁₀(d) + 20·log₁₀(f) - 147.55  [dB]
```
where d in meters, f in Hz.

### 3GPP TR 38.901 Indoor Hotspot (InH)
```
LOS:  PL = 32.4 + 17.3·log₁₀(d₃D) + 20·log₁₀(fc)  [dB]
NLOS: PL = 17.3 + 38.3·log₁₀(d₃D) + 24.9·log₁₀(fc)  [dB]
```
where d₃D in meters, fc in GHz. Valid for 0.5-100 GHz, 1-150m.

### Shadow Fading
```
LOS:  σ_SF = 3.0 dB
NLOS: σ_SF = 8.03 dB
```
Decorrelation distance: 10m.

---

## Fundamental Formulas

**Shannon capacity:**
```
C = B · log₂(1 + SNR)  [bits/s]
```

**Thermal noise power:**
```
N = kT·B = -174 + 10·log₁₀(B)  [dBm]
```
where B in Hz, k = 1.381e-23 J/K, T = 290K.

**Eb/N0 to SNR conversion:**
```
SNR = (Eb/N0) · (k/n) · num_bits_per_symbol
```
where k = info bits, n = coded bits. In Sionna: `ebnodb2no(ebno_db, num_bits_per_symbol, coderate)`.

**Link budget:**
```
Prx = Ptx + Gtx + Grx - PL - Lmisc  [dBm]
```

---

## Sionna Module Map

| Task | v2.0 Module | Key Class |
|---|---|---|
| BER/BLER simulation | `sionna.phy.utils` | `sim_ber()` |
| Channel coding (LDPC) | `sionna.phy.fec` | `LDPC5GEncoder`, `LDPC5GDecoder` |
| Channel coding (Polar) | `sionna.phy.fec` | `Polar5GEncoder`, `Polar5GDecoder` |
| Modulation | `sionna.phy.mapping` | `Mapper`, `Demapper` |
| OFDM | `sionna.phy.ofdm` | `ResourceGrid`, `ResourceGridMapper` |
| Channel estimation | `sionna.phy.ofdm` | `LSChannelEstimator`, `LMMSEChannelEstimator` |
| Equalization | `sionna.phy.ofdm` | `LMMSEEqualizer`, `ZFEqualizer` |
| CDL/TDL channels | `sionna.phy.channel.tr38901` | `CDL`, `TDL` |
| UMi/UMa/RMa | `sionna.phy.channel.tr38901` | `UMi`, `UMa`, `RMa` |
| AWGN channel | `sionna.phy.channel` | `AWGN` |
| Scene loading | `sionna.rt` | `load_scene()` |
| Ray tracing | `sionna.rt` | `PathSolver` |
| Radio maps | `sionna.rt` | `RadioMapSolver` |
| TX/RX placement | `sionna.rt` | `Transmitter`, `Receiver` |
| Antenna arrays | `sionna.rt` | `PlanarArray` |
| Network topology | `sionna.sys.utils` | `gen_hexgrid_topology()` |
| Link adaptation | `sionna.sys` | `OuterLoopLinkAdaptation` |
| Scheduling | `sionna.sys` | `PFSchedulerSUMIMO` |
| PHY abstraction | `sionna.sys` | `PHYAbstraction` |
