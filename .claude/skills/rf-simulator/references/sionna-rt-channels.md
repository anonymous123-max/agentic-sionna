# Ray Tracing to Channel Representations

## Contents

1. [RT Path Output to CIR](#rt-path-output-to-cir) -- `paths.cir()` baseband channel impulse response
2. [Channel Frequency Response](#channel-frequency-response) -- `cir_to_ofdm_channel()` for OFDM systems
3. [Discrete-Time CIR](#discrete-time-cir) -- `cir_to_time_channel()` for time-domain processing
4. [Doppler and Mobility](#doppler-and-mobility) -- Moving TX/RX with velocity vectors
5. [Full Pipeline Example](#full-pipeline-example) -- Scene to BER in one script

---

## RT Path Output to CIR

When calling `paths.cir()`, always pass `out_type="torch"` in Sionna v2 (PyTorch backend). Omitting it returns Dr.Jit arrays that cannot be used with PyTorch ops.

When you need delays relative to the first arriving path, set `normalize_delays=True`. Forgetting this gives absolute propagation delays that cause incorrect OFDM channel tap alignment.

```python
import sionna
import sionna.rt  # MUST import before load_scene

scene = sionna.rt.load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 3.5e9

# Place TX/RX
scene.add(sionna.rt.Transmitter("tx", position=[0, 0, 30],
          antenna=sionna.rt.Antenna("iso", "V")))
scene.add(sionna.rt.Receiver("rx", position=[100, 50, 1.5],
          antenna=sionna.rt.Antenna("iso", "V")))

# Compute paths
solver = sionna.rt.PathSolver()
paths = solver(scene, max_depth=5, num_samples=1_000_000)

# Extract baseband CIR
a, tau = paths.cir(out_type="torch")
# a:   [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
# tau: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]

# With normalized delays (relative to first path)
a, tau = paths.cir(normalize_delays=True, out_type="torch")
```

**Shape notes (RT vs stochastic models):**
- RT paths have no batch dimension by default. Add with `a.unsqueeze(0)` before feeding to OFDM layers.
- `num_time_steps` = 1 for static scenes. Use velocity vectors for time-varying channels.
- `num_paths` varies per TX/RX pair depending on scene geometry and max_depth.

---

## Channel Frequency Response

When converting RT CIR to OFDM channel, always add a batch dimension first. `cir_to_ofdm_channel` expects `[batch, ...]` shape -- passing unbatched RT output causes a dimension mismatch error.

```python
import torch
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.ofdm import ResourceGrid

# Define OFDM parameters
resource_grid = ResourceGrid(num_ofdm_symbols=14,
                             fft_size=72,         # 6 PRBs
                             subcarrier_spacing=30e3)

frequencies = subcarrier_frequencies(resource_grid.fft_size,
                                     resource_grid.subcarrier_spacing)

# Get CIR from RT
a, tau = paths.cir(normalize_delays=True, out_type="torch")

# Add batch dimension
a = a.unsqueeze(0)      # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
tau = tau.unsqueeze(0)   # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]

# Convert to frequency domain
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
# h_freq: [1, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]
```

---

## Discrete-Time CIR

When you need time-domain channel taps (e.g., for equalization or custom processing), use `cir_to_time_channel()`.

```python
from sionna.phy.channel import cir_to_time_channel, time_lag_discrete_time_channel

bandwidth = resource_grid.bandwidth  # = fft_size * subcarrier_spacing

# Compute valid tap range
l_min, l_max = time_lag_discrete_time_channel(bandwidth)

# Convert to discrete-time taps
h_time = cir_to_time_channel(bandwidth, a, tau,
                              l_min=l_min, l_max=l_max,
                              normalize=True)
# h_time: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, l_max - l_min + 1]
```

---

## Doppler and Mobility

When simulating mobile scenarios with RT, set velocity vectors on TX/RX before solving. Without velocities, `num_time_steps=1` and no Doppler is computed.

```python
# Set receiver velocity (moving at 30 km/h = 8.33 m/s along X)
scene.receivers["rx"].velocity = [8.33, 0, 0]

# Solve with multiple time steps
paths = solver(scene, max_depth=5, num_samples=1_000_000)

# CIR now has time-varying coefficients
a, tau = paths.cir(out_type="torch")
# a shape: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
# num_time_steps > 1 when velocities are set
```

---

## Full Pipeline Example

Scene to OFDM channel to BER measurement:

```python
import torch
import sionna
import sionna.rt
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies, AWGN
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.utils import compute_ber

# --- Scene setup ---
scene = sionna.rt.load_scene(sionna.rt.scene.simple_street_canyon)
scene.frequency = 3.5e9

scene.add(sionna.rt.Transmitter("tx", position=[0, 0, 30],
          antenna=sionna.rt.Antenna("iso", "V")))
scene.add(sionna.rt.Receiver("rx", position=[100, 50, 1.5],
          antenna=sionna.rt.Antenna("iso", "V")))

solver = sionna.rt.PathSolver()
paths = solver(scene, max_depth=5, num_samples=1_000_000)

# --- CIR to OFDM channel ---
a, tau = paths.cir(normalize_delays=True, out_type="torch")
a = a.unsqueeze(0)
tau = tau.unsqueeze(0)

rg = ResourceGrid(num_ofdm_symbols=14, fft_size=72, subcarrier_spacing=30e3)
frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)

# --- Transmit through channel ---
num_bits_per_symbol = 4  # 16-QAM
mapper = Mapper("qam", num_bits_per_symbol)
demapper = Demapper("app", "qam", num_bits_per_symbol)

# Generate random bits and map to symbols
bits = torch.randint(0, 2, (1, 1, 1, rg.num_data_symbols * num_bits_per_symbol))
x = mapper(bits)

# Apply channel (simplified flat-fading per subcarrier)
h = h_freq[..., 0, :rg.num_data_symbols]  # Use first OFDM symbol
h = h.squeeze()
y = h * x.squeeze()

# Add noise
snr_db = 20.0
no = 10 ** (-snr_db / 10)
awgn = AWGN()
y_noisy = awgn(y.unsqueeze(0), no=no)

# Demapper expects channel knowledge
llr = demapper(y_noisy, h.unsqueeze(0), no)
bits_hat = (llr < 0).int()

ber = compute_ber(bits, bits_hat)
print(f"BER at {snr_db} dB SNR: {ber:.6f}")
```

**Common pitfalls:**
- When RT finds zero paths (full blockage), `a` is empty. Check `paths.num_paths > 0` before proceeding. Forgetting this causes silent NaN propagation.
- When using `normalize=True` in `cir_to_ofdm_channel`, the channel energy is normalized per resource grid. This is needed for fair SNR comparison but changes absolute power levels.
