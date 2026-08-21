# Sionna Version Guide

When generating Sionna code, confirm the version first — wrong imports
cause immediate failures with no useful error message.

## Version Detection

```python
# Run this before generating any Sionna code
import subprocess
result = subprocess.run(["pip", "show", "sionna"], capture_output=True, text=True)
# Parse "Version: X.Y.Z" from output
```

When Sionna is not installed, use the CPU analytical fallback from
`references/cpu-fallback.md` — this is not an error condition.

## Import Patterns by Version

### Sionna v2.0+ (March 2026, PyTorch)

```python
import torch
import sionna
import sionna.rt                    # ray tracing — import before load_scene
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, RadioMapSolver
from sionna.rt import PathSolver, Camera

# PHY components
from sionna.phy.fec import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.fec import Polar5GEncoder, Polar5GDecoder
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.channel import AWGN
from sionna.phy.channel.tr38901 import CDL, TDL, UMi, UMa, RMa
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
from sionna.phy.ofdm import LSChannelEstimator, LMMSEEqualizer
from sionna.phy.utils import ebnodb2no, sim_ber

# SYS components
from sionna.sys import PHYAbstraction, PFSchedulerSUMIMO
from sionna.sys import OuterLoopLinkAdaptation
from sionna.sys.utils import gen_hexgrid_topology
```

Tensors: `torch.Tensor`. GPU: `.to("cuda")`. Gradients: `torch.autograd`.
Every component is a callable PyTorch module (differentiable).

### Sionna v1.x (2025, TensorFlow)

```python
import tensorflow as tf
import sionna
from sionna.channel.tr38901 import CDL, UMi, UMa, RMa
from sionna.ofdm import ResourceGrid, ResourceGridMapper
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no, sim_ber
```

Tensors: `tf.Tensor`. GPU: automatic. Gradients: `tf.GradientTape`.

### Sionna v0.x (pre-2025, TensorFlow, different module paths)

```python
import sionna
from sionna.channel import AWGN, RayleighBlockFading
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid
```

Module paths differ significantly — `sionna.channel` instead of
`sionna.phy.channel`, `sionna.fec.ldpc` instead of `sionna.phy.fec`.

## Common Version Errors

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'sionna.phy'` | Using v2 imports on v1.x | Drop `.phy` from path: `sionna.channel` not `sionna.phy.channel` |
| `ModuleNotFoundError: No module named 'sionna.channel'` | Using v0.x imports on v2 | Add `.phy`: `sionna.phy.channel` not `sionna.channel` |
| `AttributeError: module 'sionna' has no attribute 'rt'` | Sionna RT not installed separately | v2: `pip install sionna[rt]` or `pip install sionna-rt` |
| `ImportError: cannot import name 'BinarySource'` | v2 moved utilities | Use `torch.randint(0, 2, ...)` directly |
