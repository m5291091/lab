# Source and provenance — memory_paths

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/correctness/memory_paths/325557_3216152/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

## Common

- Graph: `data/325557_3216152` (n=325557, m=3216152).
- Graph SHA256: `a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584`.
- Runs: each configuration n=1, no warmup. Timing is `correctness_only_not_performance`.
- Tolerance: `abs_tol=1e-3`, `rel_tol=1e-6`.
- Criterion: `abs_diff <= abs_tol + rel_tol * max(abs(a), abs(b))`.
- PathMerge is an **external comparator, not ground truth**.
- All raw BC vectors are **archived (Git-tracked) under `../../raw_data/correctness/memory_paths/`**
  and verified by `../../raw_data/SHA256SUMS`; their `OriginalPath`→`RawPath`+SHA256 mapping is in
  `../../raw_data/MANIFEST.tsv` (and `../../EXTERNAL_ARTIFACTS.tsv`, now the raw_data migration map).
  They were copied byte-identically from the Git-ignored `build_miyabi/` originals (retained, not moved).
  Copied `MANIFEST.txt` / `*_summary` / `run.log` / `*.md` / stderr are byte-identical from the source
  result directories, so the paths inside them still identify the external originals.

## Job termination status

The PBS accounting `Exit_status` could not be retrieved independently, so it is **not**
asserted as `0`. The following are kept separate (measured values / runner status unchanged):

| Field | canonical_job_2368587 | diagnostic_job_2369632 |
|:--|:--|:--|
| `PBSExitStatus` | `not_recorded` | `not_recorded` |
| `ScriptCompletion` | `normal_completion_marker_observed` (FINAL_STATUS written under `set -euo pipefail`) | `normal_completion_marker_observed` |
| `RunnerExitStatus` | `0` for all 6 configurations | `0` for all 3 configurations |
| `ScriptExitCode` | `1` (deliberate `CORE_FAIL`, not a runtime failure) | `0` |
| `OverallStatus` | `CORE_FAIL` | `DIAGNOSTIC_COMPLETE` |

`PBSExitStatus=not_recorded` is not the same as success; success/`CORE_FAIL` is established
from the completion marker, the per-configuration `RunnerExitStatus`, and the complete
artifacts, not from PBS accounting.

## canonical_job_2368587

- Checkpoint: `memory_correctness_20260712`.
- PBS job ID: `2368587.opbs`.
- Design: comparison matrix; external comparator = `pathmerge_b4096`.
- **All 6 configurations completed with runner_exit=0** (`configs_ok=6 configs_fail=0`).
- Formal overall status: **`CORE_FAIL`** (exit=1). This is preserved, not hidden:
  CORE_MEMORY_PATH pass=3 fail=2; PATHMERGE_CROSS_IMPL_DIAGNOSTIC pass=0 fail=5.

| config | impl / path | requested/effective batch | SUB_BATCH | num_subs | NS_eff | mode / note |
|:--|:--|:--|:--|:--:|:--:|:--|
| gpu_opt_b9792 | gpu_opt / um | 9792 / 9792 | 6596 | 2 | 1 | oversubscribed; route evidence=PASS |
| gpu_opt_b1024 | gpu_opt / um | 1024 / 1024 | 1024 | 1 | 2 | in_capacity control |
| gpu_opt_pure_chunked_b16384 | gpu_opt_pure_chunked / chunked | 16384 / 16384 | 6596 | 3 | 1 | chunked, completed |
| gpu_opt_pure_chunked_b1024 | gpu_opt_pure_chunked / chunked | 1024 / 1024 | 1024 | 1 | 2 | non_chunk |
| gpu_opt_pure_b1024 | gpu_opt_pure / pure | 1024 / 1024 | n/a | n/a | n/a | explicit cudaMalloc |
| pathmerge_b4096 | pathmerge_bc / pathmerge | 4096 / 4096 | n/a | 80 | n/a | external comparator |

Comparison outcomes (see `canonical_job_2368587/comparison_matrix.tsv`):

- **`same_batch_diff_path` (b1024): mismatch=0** for UM vs Pure, UM vs Chunked, Pure vs Chunked.
  This agreement is within mixed tolerance and is **NOT a byte (SHA256) match** — the three
  vector SHA256 differ.
- **`same_impl_diff_batch`: strict `rel_tol=1e-6` exceeded** — `gpu_opt_b9792` vs `gpu_opt_b1024`
  and `chunked_b16384` vs `chunked_b1024` each mismatch=6 (max_rel ≈ 2.23e-6 / 2.85e-6). The
  affected-index union across the two is **8 vertices**.
- **`pathmerge_cross` (diagnostic, not required for core): all 5 FAIL** — PathMerge b4096 vs each
  proposed vector mismatch ≈ 11027–11030, max_rel ≈ 2.0e-3. **Unresolved; correctness undecided.**
- UM `b9792` completed under the host-memory-limited 100 GiB configuration and satisfied the combined oversubscription route
  evidence (est=102.02 GB > free_before=101.4 GB, HBM3 streaming, NS_eff=1, num_subs=2,
  SUB_BATCH=6596<9792, prefetch_cum=33.18 s). **This is not a migration-byte measurement.**

## diagnostic_job_2369632

- Checkpoint: `memory_diagnostic_20260713`.
- PBS job ID: `2369632.opbs`.
- Purpose: one-factor-each diagnosis (T-RESET, T-NSEFF); not performance.
- Old (canonical) vectors reused as reference: `old_b1024` / `old_b9792` / `old_chunk_b16384`
  from job `2368587`.

| config | env | requested/effective batch | NS_eff | num_subs | route counters (full_memset/visited) |
|:--|:--|:--|:--:|:--:|:--|
| CONTROL | none | 1024 / 1024 | 2 | 1 | 3 / 315 |
| T-RESET | `BC_DIAG_FORCE_FULL_RESET=1` | 1024 / 1024 | 2 | 1 | 318 / 0 |
| T-NSEFF | `BC_DIAG_FORCE_NS_EFF_ONE=1` | 1024 / 1024 | 1 | 1 | 2 / 316 |

- `non_interference=verified_mismatch0` (CONTROL vs old_b1024 mismatch=0).
- `reset_status=RESET_NOT_DISTINGUISHED`, `nseff_status=NS_EFF_NOT_DISTINGUISHED`:
  **full memset forcing and NS_eff=1 forcing each produced no difference vs the b1024 CONTROL**
  (CONTROL vs T-RESET and CONTROL vs T-NSEFF both mismatch=0). Neither single factor
  distinguished the stress difference; the cause is unspecified.
- CONTROL / T-RESET / T-NSEFF vs `old_b9792` (and T-NSEFF vs `old_chunk_b16384`) all reproduce the
  same 6-element stress difference (max_abs ≈ 5.49 @ index 289277), confirming the stress
  difference is a property of the large-batch configuration, not of the diagnostic instrumentation.

### DIAGNOSIS.md repair

The archived `diagnostic_job_2369632/DIAGNOSIS.md` is a **repaired regeneration**. In the original
job output three bullet lines were dropped because `scripts/run_memory_correctness_diagnostic.sh`
used `printf '- ...'`, whose format string begins with `-` and was parsed by `printf` as an option
(`printf: - : invalid option`). The fix (`printf -- '- ...'`) affects **document generation only**;
GPU computation, judgments, tolerances and diagnostic switches are unchanged. The three restored
lines carry the already-recorded measured values (from `FINAL_STATUS.txt` / `.judgment.env`):

```text
- non_interference (CONTROL vs old_b1024): verified_mismatch0
- CONTROL vs T-RESET mixed-tol mismatch: 0
- CONTROL vs T-NSEFF mixed-tol mismatch: 0
```

No measured value was hand-entered or altered; only the previously-missing lines were restored.

## analysis/

Read-only Gate G2.2/G2.3 analysis. Files `run_to_run_comparison.tsv`,
`stress_direct_comparison.tsv`, `six_vertex_detail.tsv`, `tolerance_sensitivity.tsv`, and
`Gate_G2_2_analysis.md` are **regenerable byte-identically** from the external raw vectors by:

```bash
cd thesis_bc_project
python3 scripts/analyze_memory_correctness.py \
    --build-dir build_miyabi --graph data/325557_3216152 \
    --outdir result/correctness/memory_paths/analysis
```

`Gate_G2_3_audit.md` is a **static code audit (human narrative)** — hypothesis ranking, source-file
references, and rounding-bound estimates. It is not computable from vectors and is therefore not
regenerated by the script; its numeric claims (SHA256, mismatch counts) are cross-verified by the
regenerated TSV/Markdown above.

### Input raw vectors (archived under raw_data/, SHA256-verified) and SHA256

| logical name | source (under `build_miyabi/`) | SHA256 |
|:--|:--|:--|
| pm_269 | `result_memory_correctness_20260712_204001_2368269.opbs/pathmerge_b4096.bc.tsv` | `1569b9e341f1baecaca010aa36f01d1b9bc1e97530a6156de3d6a9874acf9f84` |
| pm_398 | `result_memory_correctness_20260712_211738_2368398.opbs/pathmerge_b4096.bc.tsv` | `c895c6121671af6605d834e68bc1443b84624ca595506267901c3a59a643e454` |
| pure_398 | `result_memory_correctness_20260712_211738_2368398.opbs/gpu_opt_pure_b1024.bc.tsv` | `b4f0674431bc4c918b7b5a597ca98be6132c6f9ad06b2f31b300ff31f6b23e95` |
| pm_587 | `result_memory_correctness_20260712_220331_2368587.opbs/pathmerge_b4096.bc.tsv` | `94e6379ac52e76025052ff98e97274a16b6467b57454d73c6d68aaa9eeeebd9d` |
| pure_587 | `result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_pure_b1024.bc.tsv` | `fc95255fe422693d5f7c8d80d39624015b3f9ad2020b54916f61358718439984` |
| gpu_b1024 | `result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_b1024.bc.tsv` | `4a40a553a388ba2cb29d4ea366db979983fa398c55bb8a694882f260efd431cb` |
| gpu_b9792 | `result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_b9792.bc.tsv` | `be8f52d32ac03cd495a08c5c6cd138fcdcec916a16830e62e7bc8d3c968d25c5` |
| chunk_b1024 | `result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_pure_chunked_b1024.bc.tsv` | `5ff0bef083dff51e0c6f15894912549dd5e28bb42e5eaabf626654edbf4627de` |
| chunk_b16384 | `result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_pure_chunked_b16384.bc.tsv` | `618ffdc4108f0c24a148bc1aa2a18b83d14b48319cd18438f89196b40930a86d` |

The originals of the analysis files were produced ad-hoc in a prior agent session workspace
(`~/.copilot/session-state/.../files/`, Git-untracked). The archived copies are byte-identical to
those originals and to `scripts/analyze_memory_correctness.py` output.

## Constraints (summary)

- graph=`325557_3216152`; graph SHA256 as above; each configuration trials=1, warmup=none.
- checkpoints: canonical `memory_correctness_20260712`, diagnostic `memory_diagnostic_20260713`; jobs `2368587` / `2369632`.
- `abs_tol=1e-3`, `rel_tol=1e-6`; all runner configurations succeeded (runner_exit=0).
- Formal overall status for the canonical job is `CORE_FAIL` (not hidden).
- Same-batch `mismatch=0` is **within mixed tolerance, not a byte (SHA256) match**.
- Stress conditions exceed the strict `rel_tol=1e-6` over a union of 8 vertices.
- The PathMerge difference is unresolved; PathMerge is an external comparator, not ground truth.
- `reset` and `NS_eff` single-factor changes were **not distinguished** (each mismatch=0 vs CONTROL);
  the cause is unspecified and not attributable to a single factor.
- Raw BC vectors are archived under `../../raw_data/` (SHA256-verified; `raw_data/MANIFEST.tsv`; migration map `EXTERNAL_ARTIFACTS.tsv`).
