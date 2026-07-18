# SOURCE — corrected 325557 Series A/B (job 2404743)

Correctness (full-vector) and capacity-boundary (feasibility) re-run on the
**corrected** 325557 graph. Self-contained raw archive; no dependency on
`build_miyabi/` or on any prior Git history.

## Identity

| Field | Value |
|-------|-------|
| PBS job ID | `2404743.opbs` |
| Series | A (correctness vectors + comparison matrix) + B (capacity feasibility) |
| Checkpoint SHA | `45352a344aaac463283a647467b790be9b45bfb8` |
| Final status | `SUCCESS` |
| Graph | `data/325557_3216152_corrected_v1` |
| Graph SHA256 | `8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22` |
| Graph file size | 45,348,105 bytes (≈43.25 MiB) — **input file size, not GPU working set** |
| n_nodes / n_edges | 325557 / 3216152 |
| Runner | `run_benchmark`, SHA256 `3a53f77c19d6465cc645edc5fb51341f595c5a1b36277b3cd7d91a5a9b880143` |
| Tolerances | abs_tol = 1e-3, rel_tol = 1e-6 |

Graph validation (`graph_validation.json`, run.log): `len(ptr)==n+1`,
`ptr[n]==2m==6432304`, `len(adj)==2m`, 0 failed checks — VALIDATION PASS.

## PBS stdout provenance

The scheduler stdout file is gitignored at its original name (`*.o[0-9]*`), so it
is copied verbatim into this archive as `pbs_stdout.log`.

| Original filename | Original path (project-relative) | Size (bytes) | SHA256 |
|-------------------|----------------------------------|-------------:|--------|
| `bc_corr325557.o2404743` | `bc_corr325557.o2404743` | 10130 | `3c4c46680f9432b94fef79ca9344027ad77195973d075b8019379f934feb8ec5` |

Content is byte-identical to `pbs_stdout.log` (see top-level `MANIFEST.tsv`,
`HashMatch=yes`).

## Series A — correctness (6 vectors, 10 comparisons)

All six per-vertex BC vectors acquired with `runner_exit=0`, `oom_evidence=none`,
`vector_validation` status `PASS` (length 325557, missing/duplicate/out-of-range/
NaN/Inf all 0). See `vectors/`, `validation/`, `vector_inventory.tsv`.

| Config | Implementation | Vector SHA256 |
|--------|----------------|---------------|
| gpu_opt_b1024 | GPU_Opt (UM) | `adc6270b…d738796` |
| gpu_opt_b9792 | GPU_Opt (UM, oversubscribing batch) | `22f5eed2…dfcc8a9a` |
| gpu_opt_pure_b1024 | GPU_Opt_Pure | `65c109ef…e8cacbb` |
| gpu_opt_pure_chunked_b1024 | GPU_Opt_Pure_Chunked | `d53afd64…3ff7f618` |
| gpu_opt_pure_chunked_b16384 | GPU_Opt_Pure_Chunked | `222bcfcc…e27bed1f` |
| pathmerge_b4096 | PathMerge (external comparator) | `164cceb2…c7aa9496` |

The comparison matrix (`comparison_matrix.tsv`, `comparisons/`) records **10 pairwise
comparisons, MismatchedElements = 0 for every pair**:

- `same_impl_diff_batch` (the stress pairs that FAILED on the legacy malformed
  input): gpu_opt b9792-vs-b1024 max_rel = 5.32e-14; chunked b16384-vs-b1024
  max_rel = 4.75e-14 — both PASS.
- `same_batch_diff_path` (UM / Pure / Chunked at b1024): max_rel ≤ 1.08e-14 — PASS,
  though **not byte-identical** (distinct SHA256, floating-point accumulation order).
- `pathmerge_cross` (PathMerge vs GPU_Opt variants): max_abs ≈ 1.2e-3, max_rel ≈
  5.09e-13, "PASS (absolute-only warning)". **PathMerge is an external comparator,
  not an independent ground truth.**

Max BC agrees across implementations at index 272816.

## Series B — capacity-boundary feasibility (1 trial per configuration)

`feasibility_results.tsv`, `oom_evidence.tsv`, `feasibility/`. This is a
**feasibility boundary probe on the corrected 325557 graph only**, not a performance
comparison. Every failure classification uses the strong-evidence-only OOM policy
(`oom_policy=strong_evidence_only`; a word mention or advisory warning is not evidence).

| Config | Impl | Requested batch | Outcome | Runtime (s) | Runner exit | Cause |
|--------|------|----------------:|---------|------------:|------------:|-------|
| pure_b4096 | GPU_Opt_Pure | 4096 | SUCCESS | 65.889429 | 0 | — |
| pure_b8192 | GPU_Opt_Pure | 8192 | **OOM_CONFIRMED (CUDA)** | not_recorded | 1 | `cuda_oom`: `host_pure.cu:144: out of memory` (`feasibility/pure_b8192.stderr.log:5`) |
| um_b10240 | GPU_Opt | 10240 | SUCCESS | 238.672569 | 0 | — |
| um_b12288 | GPU_Opt | 12288 | **RUNTIME_FAILED** | not_recorded | 137 | SIGKILL (exit 137); `oom_evidence=none` at CUDA level → host/cgroup memory OOM kill, **not** a CUDA/HBM OOM |
| chunked_b16384 | GPU_Opt_Pure_Chunked | 16384 | SUCCESS | 66.598223 | 0 | tested upper limit |

**Correct phrasing (do not misstate):** the input file is ≈43.25 MiB; the capacity
limit is a *batch-dependent working set*, not the input graph. `pure_b8192` is a
CUDA out-of-memory. `um_b12288` (exit 137) is a host/cgroup memory OOM kill, and is
**not** labelled a CUDA OOM or an HBM OOM. Chunked succeeds up to the tested limit
b16384; this does not imply unlimited capacity. Results apply to corrected 325557 only.

## Layout

```
job_2404743/
├── SOURCE.md, MANIFEST.txt, run.log, pbs_stdout.log, graph_validation.json
├── implementation_manifest.tsv, feasibility_results.tsv, vector_inventory.tsv
├── comparison_matrix.tsv, oom_evidence.tsv
├── vectors/       6 × (bc.tsv + stderr.log)
├── validation/    6 × vector_validation.json
├── comparisons/   10 × (json + md)
└── feasibility/   5 boundary runs × (stderr.log + stdout.tsv)
```
