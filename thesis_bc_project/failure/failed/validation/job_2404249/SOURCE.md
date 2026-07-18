# SOURCE — FAILED (validation false positive) job 2404249

## Classification

```
VALIDATION_FALSE_POSITIVE_OOM_MARKER
```

The runner **succeeded** (exit 0, complete BC vector), but the orchestrator's naive
OOM check matched an **advisory warning string** and wrongly aborted Series A. This is
a harness false positive, not a real out-of-memory. Fixed in checkpoint `45352a3` by
W7.3B2.2 (`oom_evidence.sh`, strong-evidence-only classification); the corrected run is
job 2404743.

## Identity

| Field | Value |
|-------|-------|
| PBS job ID | `2404249.opbs` |
| Checkpoint SHA | `b677d6cd495b450557b014c740f77d727ddb0f9a` (pre-fix) |
| Graph | `data/325557_3216152_corrected_v1` (SHA256 `8373244f…4eeaa22`) |
| Graph validation | PASS (0 failed checks) |
| Runner | `run_benchmark`, SHA256 `11be0eec…f647e50c` |
| Final status | `FAILED` (`final_reason=Series A gpu_opt_pure_b1024: oom_marker;runner_exit=0`) |

## Root cause (exact log evidence)

The orchestrator classified `gpu_opt_pure_b1024` as an OOM failure although the runner
exited 0 with a fully-valid vector. The matched line (`false_positive_match.tsv`,
`gpu_opt_pure_b1024.stderr.log:2`) is only an advisory pre-run warning:

```
> [Warn] BC_BATCH_OVERRIDE=1024 exceeds safe limit 512; may cause cudaMalloc OOM
```

The substring `OOM` in "may cause cudaMalloc OOM" tripped a naive marker check.
Evidence that the run actually succeeded (same stderr): `Elapse time = 64.732001 s`,
`GTEPS = 16.1750`, `Maximum Betweenness Centrality index 272816`, and the vector
passed length/finiteness validation. `run.log:283` records
`FAILED: Series A gpu_opt_pure_b1024: oom_marker;runner_exit=0`.

W7.3B2.2 replaced the naive check with `oom_evidence.sh`, which only accepts three
strong-evidence classes (`cuda_oom`, `host_alloc_failure`, `kernel_oom_kill`); an
advisory warning or a bare word mention is no longer treated as evidence. Under that
policy this configuration is a SUCCESS, as confirmed by job 2404743.

## PBS stdout

`pbs_stdout.log` — copied verbatim from `bc_corr325557.o2404249` (17375 bytes, SHA256
`fcf10a2287bd6a8ade1e5d5e2b358254636c1d4e2b6b87828aaa258137b5071c`).

## Vectors — recorded by SHA/size only, NOT duplicated into Git

This pre-fix run produced three `.bc.tsv` vectors before it aborted. They are on the
**superseded checkpoint `b677d6c`** and are **not byte-identical** to the canonical
job-2404743 vectors (different checkpoint and runner; batch-1024 floating-point
accumulation order differs). They are therefore **not** committed; only their path,
size, and SHA256 are recorded in `vector_provenance.tsv`:

| Vector | Size (bytes) | SHA256 | Canonical (2404743) SHA256 |
|--------|-------------:|--------|----------------------------|
| gpu_opt_b1024 | 9330105 | `08b70e46…10e85e5a` | `adc6270b…d738796` (differs) |
| gpu_opt_pure_b1024 | 9330110 | `d7f82fcc…59cab0b70` | `65c109ef…e8cacbb` (differs) |
| gpu_opt_pure_chunked_b1024 | 9330118 | `f24ca2a5…9da22fb2` | `d53afd64…3ff7f618` (differs) |

The canonical, Git-tracked corrected-325557 vectors live in
`raw_data/corrected_325557/job_2404743/vectors/`.

## Retained artifacts

`MANIFEST.txt`, `run.log`, `pbs_stdout.log`, `graph_validation.json`,
`implementation_manifest.tsv`, `vector_inventory.tsv` (2 PASS rows recorded before the
abort), header-only `feasibility_results.tsv` / `comparison_matrix.tsv`,
`gpu_opt_pure_b1024.stderr.log` (the false-positive source), `false_positive_match.tsv`,
`vector_provenance.tsv`.
