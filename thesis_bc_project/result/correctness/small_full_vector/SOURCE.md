# Source and provenance

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/correctness/small_full_vector/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

## Execution

- Purpose: small full-vector correctness only; timing values are not performance results.
- Reference / candidate: Sequential / GPU_Opt.
- SourceSnapshotID: `small_correctness_20260712`（実験時コード `code_snapshots/small_correctness_20260712/`；元 commit は `code_snapshots/_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv`）。runner は `EXPECTED_SHA=ACTUAL_SHA` を確認。
- PBS job ID: `2367583.opbs`.
- Graphs: `data/benchmark_7000_41459`, `data/benchmark_11023_62184`, `data/chain_200`.
- Runs: each configuration n=1, no warmup.
- Batch: requested 512; effective 512; `SUB_BATCH=512`; `num_subs=1`; `NS_eff=2` for all three GPU_Opt runs.
- Tolerance: `abs_tol=1e-3`, `rel_tol=1e-6`.
- Criterion: `abs_diff <= abs_tol + rel_tol * max(abs(reference), abs(candidate))`.
- Success: all runner/comparison exits 0, complete vectors, missing index 0, mismatch 0, NaN/+Inf/-Inf 0, and complete result artifacts.

PBS accounting's explicit `Exit_status` could not be retrieved because repeated history queries returned `SIM0801: Cannot connect to database`. Success was established from the terminal script PASS under `set -euo pipefail`, runner/comparison exits 0, and the complete independently rechecked artifacts. The PBS `.o` file is not archived in Git.

## Input and vector hashes

| Graph | Graph SHA256 | Sequential vector SHA256 | GPU_Opt vector SHA256 |
|:--|:--|:--|:--|
| benchmark_7000_41459 | `4a891b4de4a0df86ef73c469f1e81b6206073e7368488e74f3ee2cec43b29ddc` | `fa23bf8892bdc799a9f859d3ccbe6859a1d965fb9162dcd62030d073a6771c04` | `458d0a129e03229b1e349d8725461ef9c6cb99356636b8ebc87bacbe8a16e957` |
| benchmark_11023_62184 | `8d1df41c579de3150a155ee9cce321784723fdb1824c0a2a160d95004d4b6e31` | `98381b7869cb86719669aa78ecaeaad90054be77d43544a907cfea27f66fbae6` | `d9db54898130c8e7acefe2cb304f57a6a52c0b379edae5aa72203e622a0e21f9` |
| chain_200 | `8fe3b0e05de9eecb9999962374ce843264d040b985e85e7ddc7bfd987494db79` | `1f39b65cc3fd9cf8e79e24421ed9bf5883e2869cd3d10db956275ca82c715f4b` | `3feef9b82e3ef287a3fe010f0797c07d04e9b18206ef88a6a8b3f469c9840dda` |

The raw vectors are archived (Git-tracked) under `raw_data/` and verified by `raw_data/SHA256SUMS`:

```text
raw_data/correctness/small_full_vector/<graph>/sequential/seq/job_2367583_20260712/sequential.bc.tsv
raw_data/correctness/small_full_vector/<graph>/gpu_opt/um_b512/job_2367583_20260712/gpu_opt.bc.tsv
```

They were copied byte-identically (SHA256 verified) from the Git-ignored originals
`build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/<graph>/{sequential,gpu_opt}.bc.tsv`
(the build_miyabi originals are retained, not moved/deleted). `MANIFEST.txt`, `correctness_summary.tsv`, `run.log`, all `comparison.md` files, and all runner stderr logs are byte-identical copies from that result directory; paths inside them therefore still identify the external originals (resolvable via `raw_data/MANIFEST.tsv`).

## Reproduction

```bash
cd thesis_bc_project
# 実験時コード = code_snapshots/small_correctness_20260712/（元 commit は _legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv）
qsub -v BC_BATCH_OVERRIDE=512 scripts/run_small_correctness.sh
```

This command documents reproduction only; no job was submitted during archival. The scope excludes email-EuAll, roadNet-PA/TX/CA independent-reference full-vector correctness, GPU_Opt_Pure, GPU_Opt_Pure_Chunked, and UM oversubscription-specific paths. It also does not prove Hybrid BFS, warp, or other internal branches were exercised through dedicated counters.
