# Appendix X. Layered verification — aggregated view

**Table X: Aggregated per-layer results for the same experiments as Tables IV and VI in the main paper.** For each task family and prompt tier, we report per-layer pass rates and, where applicable, the continuous Layer-3 distances (median across trials, computed on trials that produced structured output). The **L3 pass** column matches the AutoNetSim column of Tables IV / VI up to a small denominator difference due to crashed trials that leave no artifact on disk.

Notation: `L1` = artifact & schema; `L2` = executable & Sionna-namespace call detected; `L3` = numerical output within the audited tolerance window. Continuous columns report medians across trials that produced the corresponding numeric field. `n/a` = the task family has no such oracle; `–` = no trial produced the field.

| Section | Family / prompt | Condition | L1 | L2 | L3 (=paper) | Path-gain MAE (dB) | RSS MAE (dB) | SINR err (dB) | BER log-err | Thr RE (%) |
|---------|-----------------|-----------|---:|---:|------------:|-----:|-----:|-----:|-----:|-----:|
| Sim | N1 / simple | Naive | 100.0% | 16.7% | 16.7% | n/a | 6.91 | n/a | n/a | n/a |
| Sim | N1 / simple | Self-Written | 100.0% | 54.2% | 12.5% | n/a | 11.24 | n/a | n/a | n/a |
| Sim | N1 / simple | AutoNetSim | 100.0% | 85.0% | 87.5% | n/a | 0.11 | n/a | n/a | n/a |
| Sim | N2 / simple | Naive | 100.0% | 35.3% | 0.0% | n/a | 2.47 | n/a | n/a | n/a |
| Sim | N2 / simple | Self-Written | 100.0% | 25.0% | 0.0% | n/a | – | n/a | n/a | n/a |
| Sim | N2 / simple | AutoNetSim | 100.0% | 95.0% | 90.0% | n/a | 0.00 | n/a | n/a | n/a |
| Sim | N3 / simple | Naive | 100.0% | 17.6% | 0.0% | n/a | 3.34 | n/a | n/a | n/a |
| Sim | N3 / simple | Self-Written | 100.0% | 20.0% | 0.0% | n/a | – | n/a | n/a | n/a |
| Sim | N3 / simple | AutoNetSim | 100.0% | 85.0% | 85.0% | n/a | 0.00 | n/a | n/a | n/a |
| Sim | N4 / simple | Naive | 100.0% | 0.0% | 0.0% | n/a | n/a | n/a | 0.01 | – |
| Sim | N4 / simple | Self-Written | 100.0% | 0.0% | 0.0% | n/a | n/a | n/a | 0.02 | – |
| Sim | N4 / simple | AutoNetSim | 100.0% | 83.3% | 100.0% | n/a | n/a | n/a | 0.01 | 36.33 |
| Sim | N1 / enhanced | Naive | 100.0% | 35.0% | 95.0% | n/a | 0.00 | n/a | n/a | n/a |
| Sim | N1 / enhanced | Self-Written | 100.0% | 40.0% | 100.0% | n/a | 0.05 | n/a | n/a | n/a |
| Sim | N1 / enhanced | AutoNetSim | 100.0% | 100.0% | 100.0% | n/a | 0.10 | n/a | n/a | n/a |
| Sim | N2 / enhanced | Naive | 100.0% | 35.0% | 100.0% | n/a | 12.74 | n/a | n/a | n/a |
| Sim | N2 / enhanced | Self-Written | 100.0% | 55.0% | 100.0% | n/a | 12.74 | n/a | n/a | n/a |
| Sim | N2 / enhanced | AutoNetSim | 100.0% | 100.0% | 100.0% | n/a | 12.74 | n/a | n/a | n/a |
| Opt | P1 / simple | Naive | 100.0% | 0.0% | 0.0% | – | n/a | n/a | n/a | – |
| Opt | P1 / simple | Self-Written | 100.0% | 0.0% | 0.0% | – | n/a | n/a | n/a | – |
| Opt | P1 / simple | AutoNetSim | 100.0% | 84.2% | 84.2% | 0.00 | n/a | n/a | n/a | 0.00 |
| Opt | P2 / simple | Naive | 100.0% | 0.0% | 0.0% | – | n/a | n/a | n/a | – |
| Opt | P2 / simple | Self-Written | 100.0% | 0.0% | 0.0% | – | n/a | n/a | n/a | – |
| Opt | P2 / simple | AutoNetSim | 100.0% | 84.2% | 78.9% | 0.00 | n/a | n/a | n/a | 0.00 |
| Sys | S1 / simple | Naive | 100.0% | 73.3% | 0.0% | n/a | n/a | 1.29 | n/a | – |
| Sys | S1 / simple | Self-Written | 100.0% | 70.0% | 0.0% | n/a | n/a | 2.94 | n/a | – |
| Sys | S1 / simple | AutoNetSim | 100.0% | 100.0% | 100.0% | n/a | n/a | 0.11 | n/a | 1.07 |
| Sys | S2 / simple | Naive | 100.0% | 71.4% | 0.0% | – | n/a | 3.27 | n/a | – |
| Sys | S2 / simple | Self-Written | 100.0% | 66.7% | 0.0% | – | n/a | 8.34 | n/a | – |
| Sys | S2 / simple | AutoNetSim | 100.0% | 100.0% | 100.0% | 0.00 | n/a | 0.00 | n/a | 0.00 |
| Sys | S3 / simple | Naive | 100.0% | 16.7% | 0.0% | n/a | n/a | – | n/a | – |
| Sys | S3 / simple | Self-Written | 100.0% | 80.0% | 0.0% | n/a | n/a | – | n/a | – |
| Sys | S3 / simple | AutoNetSim | 100.0% | 95.0% | 50.0% | n/a | n/a | 0.00 | n/a | 0.00 |
| Sys | S4 / simple | Naive | 100.0% | 61.5% | 0.0% | 8.16 | n/a | n/a | n/a | – |
| Sys | S4 / simple | Self-Written | 100.0% | 66.7% | 0.0% | – | n/a | n/a | n/a | – |
| Sys | S4 / simple | AutoNetSim | 100.0% | 100.0% | 50.0% | 0.00 | n/a | n/a | n/a | 0.00 |
| Sys | S1 / enhanced | Naive | 100.0% | 100.0% | 100.0% | n/a | n/a | 0.08 | n/a | 0.79 |
| Sys | S1 / enhanced | Self-Written | 100.0% | 100.0% | 100.0% | n/a | n/a | 0.01 | n/a | 0.16 |
| Sys | S1 / enhanced | AutoNetSim | 100.0% | 100.0% | 100.0% | n/a | n/a | 0.00 | n/a | 0.00 |
| Sys | S2 / enhanced | Naive | 100.0% | 100.0% | 100.0% | 0.32 | n/a | 0.32 | n/a | 1.64 |
| Sys | S2 / enhanced | Self-Written | 100.0% | 100.0% | 100.0% | 0.00 | n/a | 0.00 | n/a | 0.02 |
| Sys | S2 / enhanced | AutoNetSim | 100.0% | 100.0% | 100.0% | 0.00 | n/a | 0.00 | n/a | 0.00 |
| Sys | S3 / enhanced | Naive | 100.0% | 100.0% | 43.8% | n/a | n/a | 3.00 | n/a | 23.04 |
| Sys | S3 / enhanced | Self-Written | 100.0% | 100.0% | 73.3% | n/a | n/a | 0.05 | n/a | 0.21 |
| Sys | S3 / enhanced | AutoNetSim | 100.0% | 100.0% | 100.0% | n/a | n/a | 0.00 | n/a | 0.00 |
| Sys | S4 / enhanced | Naive | 100.0% | 100.0% | 82.4% | 0.36 | n/a | n/a | n/a | 99.86 |
| Sys | S4 / enhanced | Self-Written | 100.0% | 100.0% | 100.0% | 0.01 | n/a | n/a | n/a | 4899.86 |
| Sys | S4 / enhanced | AutoNetSim | 100.0% | 100.0% | 100.0% | 0.00 | n/a | n/a | n/a | 0.00 |

## Summary — continuous metrics pooled per condition

(Pooled across every paper cell above; heavy-tail metrics — path-gain MAE and throughput RE — reported as median.)

| Condition | Path-gain MAE (dB) | RSS MAE (dB) | SINR err (dB) | BER log-err | Thr RE (%) |
|---|---:|---:|---:|---:|---:|
| Naive | 0.36 | 2.27 | 0.57 | 0.01 | 6.86 |
| Self-Written | 0.00 | 0.29 | 0.01 | 0.02 | 1.21 |
| AutoNetSim | 0.00 | 0.04 | 0.00 | 0.01 | 0.00 |
