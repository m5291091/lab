# T3  Ablation Summary (corrected 325557; mixed-checkpoint aggregate)

| Row | Hybrid BFS Effect | Warp-Cooperative Effect | Dual-Stream Effect | Trials | Note |
|---|---|---|---|---|---|
| Synthetic-4 aggregate (mixed-checkpoint) | 1.6787x | 1.0661x | 1.3914x | 5 per configuration | Geometric mean of the 4 per-graph effects; mixed checkpoints (see notes) |
| benchmark_7000_41459 | 1.5357x | 1.1753x | 1.2335x | 5 per configuration | job 2354994 (unchanged raw) |
| benchmark_11023_62184 | 1.7824x | 1.0067x | 1.5774x | 5 per configuration | job 2354994 (unchanged raw) |
| 56438_300801 | 1.9649x | 0.9916x | 1.2377x | 5 per configuration | job 2354994 (unchanged raw) |
| 325557_3216152_corrected_v1 | 1.4767x | 1.1012x | 1.5563x | 5 per configuration | Corrected re-measurement (job 2406254, checkpoint 45352a3); supersedes old malformed 325557 |

> The synthetic-4 aggregate is a MIXED-CHECKPOINT geometric mean: three graphs from job 2354994 and the corrected 325557 from job 2406254 (checkpoint 45352a3). It is not a single-checkpoint re-measurement of all four graphs.
> Per-graph and aggregate effects are distinct. Prose rounding of the aggregate is H = 1.679x, W = 1.066x, A = 1.391x. The old malformed-325557 headline (H = 1.655, W = 1.065, A = 1.396) is retained only as a historical value and is not the current main value.
> n=5 per configuration; the per-invocation untimed H1W1A1 warm-up is excluded from the 40 formal rows (corrected 325557: H0W0A0 median 176.35 s, H1W1A1 median 69.32 s). Warp-Cooperative Accumulation is graph-dependent (56438_300801 = 0.9916x < 1.0). Not generalized to roadNet.
