# Emerging Research Tasks (Tier 5 · T51-T60)

Use this file when the prompt names: channel charting, OTFS, near-field /
XL-MIMO, THz, federated learning, semantic communication, ISAC, O-RAN /
xApp, STAR-RIS, channel prediction.

| Task | Critical pattern | Distractor to avoid | Cross-link |
|---|---|---|---|
| T51 Channel charting | Use complex CSI (not magnitude) | Real-valued path gains lose phase | `neural-receivers.md` |
| T52 OTFS | 2D delay-Doppler pilot pattern | Standard OFDM pilots underperform on high-Doppler | `channel-models.md` |
| T53 Near-field XL-MIMO | Compute Rayleigh distance `2D²/λ`; spherical-wave steering inside it | Far-field steering vectors are suboptimal | `antenna-patterns.md` |
| T54 THz | Add ITU-R P.676 molecular absorption to Friis | Friis-only misses 10-30 dB at 300 GHz | `static-knowledge.md` |
| T55 Federated CE | Equal FedAvg with equal datasets ≡ centralized — vary client sizes | Otherwise federated framing is meaningless | `neural-receivers.md` |
| T56 Semantic / accuracy | Report classification accuracy (NOT BER) | BER framing misses semantic-comm point | — |
| T57 ISAC tradeoff | Pareto curve over ≥20 channel realizations | Fixed realization → overoptimistic sensing RMSE | `routing-extended.md` |
| T58 O-RAN xApp | MCS update every 10 slots (near-RT RIC), not every slot | Slot-level updates are wrong timescale | `system-level.md` |
| T59 STAR-RIS | Separate T/R coefficient matrices; enforce \|t\|² + \|r\|² = 1 | Reflect-only model on transmit side | `differentiable-optimization.md`, `sionna-diffraction-ris.md` |
| T60 Channel prediction | Normalize coefficients before LSTM/AR training | Training instability without it | `neural-receivers.md` |

## Common patterns

- **Complex tensors:** When a model expects CSI, never collapse to magnitude. Sionna paths use complex-valued tensors throughout. Stack as `(real, imag)` channels for neural input.
- **Tradeoff curves require multiple realizations:** ISAC/STAR-RIS papers consistently use ≥20 random channel realizations to draw Pareto fronts. Single-shot results are not publishable.
- **Quantization for hardware-realistic studies:** RIS phase, MCS table, beam codebook — note the bit budget (1-bit, 2-bit, 4-bit) and report quantized vs. continuous gap.
- **Self-verification for emerging tasks:**
  - Range bounds: ISAC ranging RMSE > 5 cm at 28 GHz; channel charting Pearson r ∈ [0.5, 0.95]; near-field SNR gain > 0 dB.
  - Curve shapes: prediction NMSE increases with horizon; semantic-accuracy curve has gentler slope than BER.
