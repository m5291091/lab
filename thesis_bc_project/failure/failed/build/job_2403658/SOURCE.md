# SOURCE — FAILED (build) job 2403658

## Classification

```
BUILD_FAILED_CMAKE_BINARY_DIR_COLLISION
```

A build-stage failure that produced **no benchmark and no BC vector**. Preceded the
successful Series A/B run (job 2404743). Fixed in checkpoint `45352a3` line by the
W7.3B1.1 build-directory guard.

## Identity

| Field | Value |
|-------|-------|
| PBS job ID | `2403658.opbs` |
| Checkpoint SHA | `193eb21f9c1d0646a2a58826000e9c876d84fabb` (pre-fix) |
| Graph | `data/325557_3216152_corrected_v1` (SHA256 `8373244f…4eeaa22`) |
| Graph validation | PASS (0 failed checks) — the failure is **not** a graph problem |
| Runners built | 0 |
| BC vectors produced | 0 |
| Feasibility rows | 0 (header only) |
| Comparison rows | 0 (header only) |

## Root cause (exact log evidence)

`run.log` (Stage 2) ends with:

```
[Stage 1] cugraph_bc_mini: DONE
[Stage 2] Building run_benchmark...
CMake Error: The source ".../thesis_bc_project/CMakeLists.txt" does not match the
  source ".../thesis_bc_project/cugraph_bc_mini/CMakeLists.txt" used to generate
  cache.  Re-run cmake with a different source directory.
ABORTED: build failed
```

Stage 1 (the `cugraph_bc_mini` mini-build) configured the shared `build_miyabi/`
directory; Stage 2 then tried to reuse the same binary directory with a different
CMake source, so the cached source path did not match and CMake aborted. No runner
binary was produced, therefore Series A/B never started. This is the CMake binary-dir
collision fixed by W7.3B1.1 (job-specific `build_corrected_325557/<stamp>_<jobid>/`
build tree + `build_dir_guard.sh`); the fixed run is job 2404743.

## PBS stdout

**No PBS scheduler stdout file exists for this job.** No `bc_*.o2403658` (or any
`*2403658*` file other than this archive directory) is present anywhere in the tree.
The orchestrator `run.log` is retained as the primary and only stdout evidence; a
`pbs_stdout.log` is deliberately **not** fabricated.

## Retained artifacts

`MANIFEST.txt`, `run.log`, `graph_validation.json`, and the header-only
`feasibility_results.tsv` / `vector_inventory.tsv` / `implementation_manifest.tsv` /
`comparison_matrix.tsv` (all four contain a header and zero data rows, confirming
runner=0 / vector=0).
