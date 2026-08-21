# Continuous quantitative metrics — pooled across studies

Per-trial values are compared against per-family oracles. Only trials that produce the corresponding numeric field are counted in `n`. Pass-rate is the binary verifier result. Heavy-tailed metrics (marked ⋆) are reported as **median [Q1, Q3]** to guard against unit-mismatch outliers.

| Condition | Trials | Pass rate | Path-gain MAE (dB) | RSS grid MAE (dB) | SINR error (dB) | BER log-err | BLER log-err | Throughput RE (%) ⋆ |
|-----------|-------:|----------:|------:|------:|------:|------:|------:|------:|
| no_skill | 526 | 27.6% | 0.97 ± 2.58 (n=65) | 6.74 ± 9.82 (n=60) | 0.88 ± 1.45 (n=111) | 0.25 ± 0.71 (n=9) | – | 0.14 [0.00, 11.64] (n=131) |
| self_gen | 447 | 33.3% | 0.66 ± 1.45 (n=56) | 6.11 ± 9.22 (n=54) | 0.69 ± 1.34 (n=85) | 0.02 ± 0.01 (n=6) | – | 0.04 [0.00, 9.15] (n=114) |
| with_skill | 1121 | 70.5% | 0.95 ± 2.27 (n=172) | 3.24 ± 7.74 (n=101) | 0.62 ± 1.64 (n=106) | 0.01 ± 0.01 (n=21) | 0.69 ± 0.02 (n=5) | 0.00 [0.00, 1.26] (n=259) |

### Unit-error rate (throughput_re_pct > 100%)

| Condition | Extreme outliers / Trials with metric | Rate |
|---|---|---|
| no_skill | 9 / 131 | 6.9% |
| self_gen | 13 / 114 | 11.4% |
| with_skill | 15 / 259 | 5.8% |

## Per-tier breakdown (pooled across conditions)

| Tier | Path-gain MAE (dB) | RSS grid MAE (dB) | SINR error (dB) | BER log-err | BLER log-err | Throughput RE (%) ⋆ |
|------|------:|------:|------:|------:|------:|------:|
| N1 | – | 1.92 ± 6.18 (n=86) | – | – | – | – |
| N2 | – | 12.37 ± 10.85 (n=68) | – | – | – | – |
| N3 | – | 0.89 ± 1.15 (n=61) | – | – | – | – |
| N4 | – | – | – | 0.07 ± 0.36 (n=36) | 0.69 ± 0.02 (n=5) | 33.37 [33.05, 36.33] (n=8) |
| P1 | 1.58 ± 3.90 (n=45) | – | – | – | – | 0.00 [0.00, 0.00] (n=50) |
| P2 | 0.78 ± 1.36 (n=40) | – | – | – | – | 0.00 [0.00, 0.00] (n=51) |
| S1 | – | – | 0.60 ± 0.93 (n=159) | – | – | 0.07 [0.00, 2.90] (n=117) |
| S2 | 0.57 ± 0.99 (n=104) | – | 0.88 ± 1.93 (n=109) | – | – | 0.00 [0.00, 7.77] (n=104) |
| S3 | – | – | 0.93 ± 1.90 (n=34) | – | – | 0.05 [0.00, 8.00] (n=83) |
| S4 | 0.98 ± 2.32 (n=104) | – | – | – | – | 0.02 [0.00, 134.03] (n=91) |
