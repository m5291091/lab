# T3  Ablation Summary

| Graph Group | Hybrid BFS Effect | Warp-Cooperative Effect | Dual-Stream Effect | Trials | Limitation |
|---|---|---|---|---|---|
| Synthetic (geomean, 4 graphs) | 1.655x | 1.065x | 1.396x | 5 per configuration | 4 synthetic graphs; not generalized to roadNet |
| email-EuAll (hub, real) | 1.429x | 0.970x | 1.720x | 3 per configuration | Single hub graph; Warp-Cooperative < 1.0x (harmful here) |

> Per-factor main effects are recomputed from configuration medians in the canonical raw ablation TSVs and checked against the archived contribution TSVs.
> Warp-Cooperative Accumulation is graph-dependent (range ~0.970x-1.175x across the 5 measured graphs).
