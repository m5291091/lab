# raw_data/corrected_325557 — corrected 325557 re-verification raw archive

Self-contained raw data for the **corrected** 325557 graph
(`data/325557_3216152_corrected_v1`, SHA256
`8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22`, n=325557,
m=3216152). All artifacts are Git-tracked; nothing here depends on `build_miyabi/`,
on any build binary, or on prior Git history.

The **legacy malformed** 325557 input and every result derived from it are **not**
deleted or overwritten — they remain under their original paths as historical
provenance (`result/correctness/memory_paths/`, `result/ablation/synthetic_2354994/`,
`raw_data/memory_scalability/325557_3216152/`, etc.), and are explicitly labelled as
superseded, not used in any current thesis claim.

## Contents

| Directory | Job | Checkpoint | Content |
|-----------|-----|------------|---------|
| `job_2404743/` | `2404743.opbs` (Series A/B) | `45352a3` | 6 BC vectors + 10 comparisons (all mismatch=0) + capacity-boundary feasibility |
| `job_2406254/` | `2406254.opbs` (Series C) | `45352a3` | 8×5 = 40-row H/W/A ablation + per-config stats |

See each job's `SOURCE.md` for full provenance, warmup classification, and the exact
capacity-boundary phrasing.

## Integrity ledgers

- `MANIFEST.tsv` — copy-integrity manifest for every archived file: `OriginalPath`,
  `CanonicalPath`, `SizeBytes`, `OriginalSHA256`, `CanonicalSHA256`, `HashMatch`,
  `CheckpointSHA`, `GraphSHA256`, `PBSJobID`, `Purpose`. All rows `HashMatch=yes`.
- `SHA256SUMS` — `sha256  path` for every archived file (paths relative to
  `raw_data/`). Verify with `sha256sum -c` from `raw_data/corrected_325557/` after
  rewriting the paths, or recompute directly.

76 files archived (56,140,346 bytes), including the six ≈9.3 MB per-vertex BC vectors
required for independent full-vector re-audit.

## Scope and limitations

- Series A/B/C all use checkpoint `45352a3` and the same corrected graph.
- Series B feasibility is **1 trial per boundary configuration** — a feasibility
  probe, not a performance comparison; it includes a host/cgroup memory OOM kill
  (`um_b12288`, exit 137) that is **not** a CUDA/HBM OOM.
- Series C is a **component-level** ablation of corrected 325557, not a performance
  headline. Any 4-graph aggregate that combines it with the legacy synthetic-3 graphs
  (jobs 2354994) is a **mixed-checkpoint aggregate** and must be labelled as such.
- PathMerge is an external comparator, not an independent ground truth.
- The corrected graph was reconstructed internally (symmetry-completed); the original
  generator seed is unknown. Results apply to corrected 325557 only and are not
  generalized to roadNet or to other GPUs.

## Related failure archives

The two failed attempts that preceded these successes are archived (without duplicate
vector blobs) under `failure/failed/`:

- `failure/failed/build/job_2403658/` — `BUILD_FAILED_CMAKE_BINARY_DIR_COLLISION`
  (checkpoint `193eb21`, pre-fix).
- `failure/failed/validation/job_2404249/` — `VALIDATION_FALSE_POSITIVE_OOM_MARKER`
  (checkpoint `b677d6c`, pre-fix).
