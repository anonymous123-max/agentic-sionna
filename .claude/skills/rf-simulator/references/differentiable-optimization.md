# Differentiable Optimization with Sionna RT

## Contents

1. [What Supports Gradients](#what-supports-gradients) -- Continuous vs discrete parameters
2. [Optimizing TX Orientation](#optimizing-tx-orientation) -- Maximize received power by steering antenna
3. [Learning Material Properties](#learning-material-properties) -- Fit permittivity/conductivity to measurements
4. [RIS Phase Optimization](#ris-phase-optimization) -- Steer reflected beam to target RX
5. [Gradient Patterns](#gradient-patterns) -- PyTorch autograd conventions for Sionna v2

---

## What Supports Gradients

Sionna v2 (PyTorch backend) supports automatic differentiation through the ray tracer for **continuous parameters only**.

**Differentiable (continuous):**
- TX/RX position and orientation
- Antenna patterns (parameterized)
- Material properties: relative permittivity, conductivity, scattering coefficient
- RIS phase profile (per-element phase shifts)
- TX power

**NOT differentiable (discrete / geometric):**
- Scene geometry (wall positions, room shape) -- changing geometry requires re-solving visibility
- Number of reflections / diffractions (discrete path count)
- LOS/NLOS state (binary)
- Object addition/removal

When you need to optimize geometry, use grid search or evolutionary methods instead of gradients. Gradient-based optimization only works for parameters that affect path coefficients continuously.

---

## Optimizing TX Orientation

When optimizing orientation, use `torch.autograd` and set `requires_grad=True` on the parameter tensor. Forgetting `requires_grad` silently produces None gradients.

```python
import torch
import sionna
import sionna.rt

scene = sionna.rt.load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 3.5e9

# TX with learnable orientation
tx_orientation = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)

scene.add(sionna.rt.Transmitter("tx", position=[0, 0, 30],
          orientation=tx_orientation,
          antenna=sionna.rt.Antenna("38.901", "V")))  # Directional pattern
scene.add(sionna.rt.Receiver("rx", position=[200, 50, 1.5],
          antenna=sionna.rt.Antenna("iso", "V")))

solver = sionna.rt.PathSolver()
optimizer = torch.optim.Adam([tx_orientation], lr=0.01)

for step in range(100):
    optimizer.zero_grad()

    # Update orientation in scene
    scene.transmitters["tx"].orientation = tx_orientation

    # Forward pass: compute paths
    paths = solver(scene, max_depth=3, num_samples=500_000)
    a, tau = paths.cir(out_type="torch")

    # Objective: maximize total received power
    received_power = torch.sum(torch.abs(a) ** 2)
    loss = -received_power  # Negate for minimization

    # Backward pass
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        print(f"Step {step}: power = {received_power.item():.4e}, "
              f"orientation = {tx_orientation.detach().numpy()}")
```

---

## Learning Material Properties

When fitting material properties to measured data, parameterize with `relative_permittivity` and `conductivity` as learnable tensors. Attach them to the scene's `RadioMaterial` before solving.

```python
import torch
import sionna
import sionna.rt

scene = sionna.rt.load_scene("/path/to/scene.xml", merge_shapes=False)
scene.frequency = 28e9

measured_power_db = torch.tensor([-65.0, -72.0, -80.0, -58.0])

# Learnable material parameters
eps_r = torch.tensor(5.0, requires_grad=True)
sigma = torch.tensor(0.02, requires_grad=True)

custom_mat = sionna.rt.RadioMaterial("learned_concrete",
                                     relative_permittivity=eps_r,
                                     conductivity=sigma)
scene.add(custom_mat)
for name, obj in scene.objects.items():
    if "wall" in name.lower():
        obj.radio_material = "learned_concrete"

solver = sionna.rt.PathSolver()
optimizer = torch.optim.Adam([eps_r, sigma], lr=0.01)

for step in range(200):
    optimizer.zero_grad()
    scene.radio_materials["learned_concrete"].relative_permittivity = eps_r
    scene.radio_materials["learned_concrete"].conductivity = sigma

    paths = solver(scene, max_depth=5, num_samples=1_000_000)
    a, _ = paths.cir(out_type="torch")
    # Compute per-RX power, compare to measured, MSE loss
    loss = ...  # torch.mean((predicted_db - measured_power_db) ** 2)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        eps_r.clamp_(min=1.0, max=80.0)
        sigma.clamp_(min=0.0, max=10.0)
```

---

## RIS Phase Optimization

When optimizing RIS phases, the gradient flows through the reflection coefficient computation. Each RIS element has a phase shift that modifies the reflected field.

```python
import torch
import sionna
import sionna.rt

scene = sionna.rt.load_scene(sionna.rt.scene.simple_reflector)
scene.frequency = 28e9

# Create RIS with learnable phase profile
num_rows, num_cols = 10, 10
phase_profile = torch.zeros(num_rows, num_cols, requires_grad=True)

ris = sionna.rt.RIS("ris",
                     position=[50, 0, 10],
                     num_rows=num_rows,
                     num_cols=num_cols,
                     phase_profile=phase_profile)
scene.add(ris)

scene.add(sionna.rt.Transmitter("tx", position=[0, 0, 10],
          antenna=sionna.rt.Antenna("iso", "V")))
scene.add(sionna.rt.Receiver("rx", position=[100, 30, 1.5],
          antenna=sionna.rt.Antenna("iso", "V")))

solver = sionna.rt.PathSolver()
optimizer = torch.optim.Adam([phase_profile], lr=0.05)

for step in range(300):
    optimizer.zero_grad()

    # Update RIS phase in scene
    scene.ris["ris"].phase_profile = phase_profile

    paths = solver(scene, max_depth=3, num_samples=500_000,
                   interaction_types=[sionna.rt.InteractionType.SPECULAR,
                                     sionna.rt.InteractionType.RIS])
    a, _ = paths.cir(out_type="torch")

    received_power = torch.sum(torch.abs(a) ** 2)
    loss = -received_power

    loss.backward()
    optimizer.step()

    # Wrap phases to [0, 2*pi]
    with torch.no_grad():
        phase_profile.data = phase_profile.data % (2 * torch.pi)
```

---

## Gradient Tips

When gradients are `None`, check that `requires_grad=True` is set and all ops are differentiable. Use `torch.nn.utils.clip_grad_norm_` for stability.

When gradients explode (common with dB-scale objectives), use linear-scale power as the objective and convert to dB only for logging. Computing `10 * log10(x)` with very small `x` produces huge gradients.
