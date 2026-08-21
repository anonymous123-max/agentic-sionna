# RF-3DGS: 3D Gaussian Splatting for RF Propagation

## Overview

RF-3DGS adapts 3D Gaussian Splatting (originally for novel view synthesis)
to model RF signal propagation. Instead of rendering images from novel
viewpoints, it predicts RF coverage from novel TX/RX positions using a
learned scene representation.

**Key idea:** Train a set of 3D Gaussians that encode both the geometry AND
RF propagation properties of a scene. Each Gaussian has position, covariance,
opacity, AND an RF attenuation coefficient. After training on measurement
data, the model can predict coverage for new TX positions without re-running
ray tracing.

## When to Suggest RF-3DGS

- User has **many measurements** (>50 positions with RSS values) and wants
  fast coverage prediction for different TX placements
- User needs **real-time** coverage updates (drag TX → instant heatmap)
- Traditional ray tracing is too slow for interactive optimization
- Scene geometry is complex or partially unknown

## When NOT to Suggest RF-3DGS

- User has **no measurements** (need training data)
- Scene is simple (ray tracing is fast enough)
- User needs physically interpretable results (RF-3DGS is a black box)
- Frequency changes are needed (RF-3DGS trains for a specific frequency)

## Key Papers

1. **NeRF2** (Zhao et al., 2024): Neural Radiance Fields for RF signal
   propagation. Uses NeRF architecture with RF-specific rendering equation.

2. **RF-3DGS** (various 2024-2025): Adapts 3D Gaussian Splatting for RF.
   Faster training and inference than NeRF-based approaches.

3. **WiNeRT** (Orekondy et al., 2023): Neural ray tracing for wireless
   propagation. Combines learned materials with differentiable ray tracing.

## Relationship to Sionna

Sionna provides **physics-based** ray tracing (deterministic, interpretable,
no training data needed). RF-3DGS provides **learned** propagation modeling
(fast inference, needs training data, less interpretable).

They are complementary:
- Use Sionna for initial scene setup and physics-based simulation
- Use RF-3DGS when you have measurements and need fast interactive prediction
- Use Sionna's differentiable RT to generate synthetic training data for RF-3DGS

## Implementation Status

RF-3DGS is **not implemented** in this skill. This reference exists so the
agent can discuss the technique when users ask about it and suggest
appropriate alternatives (Sionna RT for physics-based, CPU analytical for
fast approximation, calibration for measurement fitting).

## Related

- [physics-validation.md](physics-validation.md) — analytical models for comparison
- [cpu-fallback.md](cpu-fallback.md) — fast analytical alternative
- [sionna-v2-api.md](sionna-v2-api.md) — Sionna's differentiable RT
