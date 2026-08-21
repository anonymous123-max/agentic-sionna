# Task Baselines and Tool Cards

> **Source:** NVlabs/the-ai-telco-engineer framework. Each task has a fixed
> evaluation function, a named baseline to beat, and a deterministic metric.
> Code that runs without errors is NOT success — it must beat the baseline.

## Contents

1. [BER Simulation](#ber-simulation)
2. [Channel Estimation](#channel-estimation)
3. [OFDM Equalization](#ofdm-equalization)
4. [Radio Map / Coverage](#radio-map--coverage)
5. [RIS Optimization](#ris-optimization)
6. [Link Adaptation](#link-adaptation)
7. [Scene Generation](#scene-generation)

---

## BER Simulation

**Function:** Evaluate link reliability vs. SNR for a PHY chain.
**Inputs:** modulation (BPSK/QPSK/16QAM/64QAM/256QAM), channel_model
(AWGN/CDL/TDL/UMi), code_rate (0.1–0.95), ebno_range (min, max, steps).
**Output:** (ebno_db[], ber[]) — plot on log-scale y-axis.
**Baseline:** Analytical BER for uncoded AWGN (`Q(sqrt(2·Eb/N0))` for BPSK).
**Target:** Match theory within ±0.5 dB at BER=1e-4 for AWGN; LDPC-coded
should show 5–8 dB coding gain over uncoded at BER=1e-4.
**Metric:** Eb/N0 gap to theory at target BER.
**Sanity check:** BER should be >0.3 at minimum SNR, <1e-4 at maximum SNR
for a working system. Flat or non-monotone BER → check demapper output
scale or decoder input LLR sign convention.

---

## Channel Estimation

**Function:** Estimate channel from pilot observations.
**Inputs:** channel_model, SNR, num_rx_ant, num_tx_ant, pilot_pattern.
**Output:** Estimated channel matrix H_hat.
**Baseline:** LS (Least Squares) estimator — NVE ≈ 94.
**Target:** NVE < 50 for a novel estimator (neural or otherwise).
**Metric:** NVE = normalized validation error (ratio of estimator BLER to
perfect-CSI BLER, averaged across SNR range). Lower is better.
**Also report:** NMSE (dB) of H_hat vs H_true. Target < -10 dB.
**Common failure:** Using CDL model for multi-user scenario — CDL only
supports single TX. Switch to UMi/UMa for NUM_TX > 1.

---

## OFDM Equalization

**Function:** Recover transmitted symbols from received OFDM signal.
**Inputs:** channel_model, SNR, MIMO config, equalizer type.
**Output:** Equalized symbols, BER.
**Baseline:** LMMSE equalizer — NMSE ≈ -8 dB.
**Target:** NMSE < -12 dB (neural equalizer should improve 4+ dB).
**Metric:** NMSE of equalized symbols. Also report BER improvement.
**Sanity check:** Equalized constellation should show distinct clusters.
Smeared constellation → equalization failed or SNR too low.

---

## Radio Map / Coverage

**Function:** Compute signal coverage metrics across a spatial grid.
**Inputs:** scene (XML or scene_state.json), TX position, frequency,
cell_size.
**Output:** 2D RSS map (dBm), coverage percentage above threshold.
**Baseline:** CPU analytical model (3GPP InH FSPL + wall attenuation).
MAE vs. Sionna RT ≈ 8 dB for simple indoor scenes.
**Target:** MAE < 5 dB vs. measurement data (when available).
**Metric:** Coverage % above threshold (typically -70 dBm).
**Sanity check:** RSS must never exceed TX power. Path loss must increase
with distance. Coverage should be spatially continuous (no isolated
single-cell outliers 20+ dB from neighbors).

---

## RIS Optimization

**Function:** Optimize RIS phase shifts to maximize received power.
**Inputs:** scene with RIS, TX/RX positions, number of RIS elements.
**Output:** Optimized phase configuration, received power improvement.
**Baseline:** Random phase shifts.
**Target:** +3 dB received power gain over random baseline.
**Metric:** Received power (dBm) at RX with optimized vs. random phases.
**Sanity check:** Gain should be larger for NLoS than LoS scenarios
(RIS provides most benefit when direct path is blocked).

---

## Link Adaptation

**Function:** Select optimal MCS (Modulation and Coding Scheme) based on
channel conditions.
**Inputs:** channel model, SNR range, available MCS table.
**Output:** Selected MCS per TTI, achieved throughput, BLER.
**Baseline:** Fixed MCS (highest rate with BLER < 10%).
**Target:** OLLA-adapted throughput within 5% of genie-aided (perfect CSI).
**Metric:** Median throughput (Mbps), 5th percentile throughput, BLER.
**Sanity check:** BLER should hover around 10% target. Consistently
higher → OLLA step-down too slow. Consistently lower → too conservative.

---

## Scene Generation

**Function:** Create a 3D indoor/outdoor scene with furniture and RF materials.
**Inputs:** Room description (type, dimensions, furniture list).
**Output:** scene_state.json, scene.glb, scene.xml, viewer.html.
**Baseline:** Empty room with correct dimensions.
**Target:** Realistic furniture placement (no overlaps, wall-aligned),
correct ITU materials, viewer renders without errors.
**Metric:** Qualitative (human review) + structural checks: valid JSON,
unique IDs, positions in bounds, no furniture overlaps, materials from
ITU set.
**Sanity check:** All furniture within room bounds. Wall names start
with "wall" (viewer applies translucent material to wall-prefixed meshes).
