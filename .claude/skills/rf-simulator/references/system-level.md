# System-Level Simulation (sionna.sys)

## Contents

1. [Hexagonal Grid Topology](#hexagonal-grid-topology) -- Multi-cell BS/UT placement
2. [Channel Model Setup](#channel-model-setup) -- UMi/UMa/RMa with topology
3. [PHY Abstraction](#phy-abstraction) -- SINR-to-throughput mapping
4. [Link Adaptation](#link-adaptation) -- OLLA for MCS selection
5. [Scheduling](#scheduling) -- Proportional fair resource allocation
6. [Power Control](#power-control) -- Uplink and downlink power settings
7. [Full Example: 7-Cell Throughput CDF](#full-example-7-cell-throughput-cdf)

---

## Hexagonal Grid Topology

When using `gen_hexgrid_topology()`, each cell has 3 sectors (3 BSs per cell). A 1-ring grid = 7 cells = 21 sectors. Misunderstanding this causes wrong user density calculations.

When you need wraparound to eliminate edge effects, `gen_hexgrid_topology` returns `bs_virtual_loc` with mirror positions. Use these with `set_topology()` for correct inter-cell interference.

```python
from sionna.sys import gen_hexgrid_topology

topology = gen_hexgrid_topology(
    batch_size=4,
    num_rings=1,            # 1 ring = 7 cells = 21 sectors
    num_ut_per_sector=10,   # 10 UTs per sector = 210 UTs total
    scenario="umi",
    min_bs_ut_dist=10.0,
    max_bs_ut_dist=200.0,
    isd=200.0,              # Inter-site distance [m]
    bs_height=10.0,
    indoor_probability=0.8,
    return_grid=True
)

# Unpack returns
(ut_loc, bs_loc, ut_orientations, bs_orientations,
 ut_velocities, in_state, los, bs_virtual_loc, grid) = topology

# ut_loc:       [batch_size, num_ut, 3]
# bs_loc:       [batch_size, num_sectors, 3]  (21 for 1-ring)
# bs_virtual_loc: [batch_size, num_sectors, num_ut, 3]  (wraparound mirrors)
```

---

## Channel Model Setup

When connecting topology to channel model, call `set_topology()` with the unpacked tuple (excluding `grid`). Passing the tuple directly causes argument count mismatch.

```python
from sionna.phy.channel.tr38901 import UMi, UMa, RMa, PanelArray, Antenna

carrier_frequency = 3.5e9

bs_array = PanelArray(num_rows_per_panel=4,
                      num_cols_per_panel=4,
                      polarization="dual",
                      polarization_type="VH",
                      antenna_pattern="38.901",
                      carrier_frequency=carrier_frequency)

ut_array = PanelArray(num_rows_per_panel=1,
                      num_cols_per_panel=1,
                      polarization="single",
                      polarization_type="V",
                      antenna_pattern="omni",
                      carrier_frequency=carrier_frequency)

channel_model = UMi(carrier_frequency=carrier_frequency,
                    o2i_model="low",
                    ut_array=ut_array,
                    bs_array=bs_array,
                    direction="downlink",
                    enable_pathloss=True,
                    enable_shadow_fading=True)

# Set topology (all returns except grid)
channel_model.set_topology(
    ut_loc, bs_loc, ut_orientations, bs_orientations,
    ut_velocities, in_state, los, bs_virtual_loc
)

# Generate CIR
cir = channel_model(batch_size=4, num_time_steps=1)
```

---

## PHY Abstraction

When you need throughput without full PHY simulation, use `PHYAbstraction` to map SINR to transport block success probability via MCS tables.

```python
from sionna.sys import PHYAbstraction

phy = PHYAbstraction(mcs_table_index=1)  # 3GPP Table 1 (64QAM max)
# mcs_table_index=2 for 256QAM, =3 for low-SE

# Map SINR to spectral efficiency
sinr_db = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
sinr_linear = 10 ** (sinr_db / 10)

# Get achievable rate for each SINR (after MCS selection)
# PHYAbstraction internally selects the best MCS for each SINR
spectral_eff = phy(sinr_linear)  # [num_points], bits/s/Hz
```

---

## Link Adaptation and Scheduling

```python
from sionna.sys import OuterLoopLinkAdaptation, PFSchedulerSUMIMO

# OLLA: adjusts SINR offset per ACK/NACK to target 10% BLER
olla = OuterLoopLinkAdaptation(target_bler=0.1, initial_offset=0.0, step_up=0.1)
sinr_effective = sinr_measured + olla.offset
olla.update(ack=True)   # Success -> lower offset
olla.update(ack=False)  # Failure -> raise offset

# Proportional fair scheduler: assigns PRBs by instantaneous_rate / avg_throughput
scheduler = PFSchedulerSUMIMO(num_prb=25, num_tx_ant=32, num_layers=2)
```

---

## Power Control

| Parameter | Downlink | Uplink |
|-----------|----------|--------|
| Max power | Macro 46 dBm, Small 30 dBm, Indoor 24 dBm | UE class 3: 23 dBm |
| Per-PRB | `P_total - 10*log10(num_prb)` | Open-loop: `min(P_max, P0 + alpha * PL)` |
| Typical P0 | N/A | -76 dBm |
| Alpha | N/A | 0.8 (fractional pathloss compensation) |

---

## Full Example: 7-Cell Throughput CDF

```python
import torch
import matplotlib.pyplot as plt
from sionna.phy.channel.tr38901 import UMi, PanelArray
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.sys import gen_hexgrid_topology, PHYAbstraction

carrier_frequency = 3.5e9
bandwidth = 10e6
subcarrier_spacing = 30e3
fft_size = int(bandwidth / subcarrier_spacing)
bs_power_dbm = 46.0
noise_power_dbm = -174 + 10 * torch.log10(torch.tensor(bandwidth)) + 7.0  # thermal + NF

bs_array = PanelArray(num_rows_per_panel=4, num_cols_per_panel=4,
                      polarization="dual", polarization_type="VH",
                      antenna_pattern="38.901", carrier_frequency=carrier_frequency)
ut_array = PanelArray(num_rows_per_panel=1, num_cols_per_panel=1,
                      polarization="single", polarization_type="V",
                      antenna_pattern="omni", carrier_frequency=carrier_frequency)

# 7 cells, 21 sectors, 10 UTs/sector, 10 random drops
topology = gen_hexgrid_topology(batch_size=10, num_rings=1, num_ut_per_sector=10,
    scenario="umi", isd=200.0, bs_height=10.0, indoor_probability=0.8, return_grid=False)
(ut_loc, bs_loc, ut_ori, bs_ori, ut_vel, in_state, los, bs_vloc) = topology

channel = UMi(carrier_frequency=carrier_frequency, o2i_model="low",
              ut_array=ut_array, bs_array=bs_array, direction="downlink",
              enable_pathloss=True, enable_shadow_fading=True)
channel.set_topology(ut_loc, bs_loc, ut_ori, bs_ori, ut_vel, in_state, los, bs_vloc)

a, tau = channel(batch_size=10, num_time_steps=1)
frequencies = subcarrier_frequencies(fft_size, subcarrier_spacing)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)

# SINR: strongest BS as signal, rest as interference
h_power = torch.mean(torch.abs(h_freq) ** 2, dim=(-1, -2, -3))
signal, _ = torch.max(h_power, dim=-1)
interference = torch.sum(h_power, dim=-1) - signal
bs_pw = 10 ** ((bs_power_dbm - 30) / 10)
noise = 10 ** ((noise_power_dbm - 30) / 10)
sinr = (signal * bs_pw) / (interference * bs_pw + noise)

# Map SINR to throughput via PHY abstraction
phy = PHYAbstraction(mcs_table_index=1)
throughput_mbps = phy(sinr) * bandwidth / 1e6

# Plot CDF
tp = throughput_mbps.flatten().sort().values
plt.plot(tp.numpy(), torch.linspace(0, 1, len(tp)).numpy())
plt.xlabel("Throughput [Mbps]"); plt.ylabel("CDF")
plt.title("DL Throughput CDF - 7-cell UMi"); plt.grid(True)
plt.savefig("throughput_cdf.png", dpi=150, bbox_inches="tight")

print(f"Cell-edge (5th pct): {torch.quantile(tp, 0.05):.1f} Mbps")
print(f"Median:              {torch.quantile(tp, 0.50):.1f} Mbps")
```

When `num_rings=2`: 19 cells, 57 sectors. Memory grows quadratically with UT count. When batch_size > 20 on CPU, expect minutes per CIR generation.
