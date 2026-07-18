# SOURCE — corrected 325557 Series C ablation (job 2406254)

Factorial H/W/A ablation (8 compile-time configurations) re-run on the **corrected**
325557 graph. Self-contained raw archive; independently audited in Gate W7.3C1.

## Identity

| Field | Value |
|-------|-------|
| PBS job ID | `2406254.opbs` |
| Series | C (ablation, component-level analysis) |
| Checkpoint SHA | `45352a344aaac463283a647467b790be9b45bfb8` |
| Final status | `SUCCESS_COMPLETE_40` |
| Graph | `data/325557_3216152_corrected_v1` |
| Graph SHA256 | `8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22` |
| Graph file size | 45,348,105 bytes (≈43.25 MiB) — input file size, not GPU working set |
| n_nodes / n_edges | 325557 / 3216152 |
| Runner | `run_ablation`, SHA256 `70f2df8f7faa435264ceadbb86975e34a5ef7d3edaff4a1323b534aef4164606` |
| Configurations | `Ablation_H{0,1}_W{0,1}_A{0,1}` (8, compile-time templates) |
| Trials | 5 per configuration |
| Formal rows | 40 (8 × 5) |

## PBS stdout provenance

| Original filename | Original path (project-relative) | Size (bytes) | SHA256 |
|-------------------|----------------------------------|-------------:|--------|
| `bc_corr_abl.o2406254` | `bc_corr_abl.o2406254` | 4514 | `5adb78594cf98b072fe7b44770dd7dd39dfbf74a9cdbb111bfe09258af369b18` |

Copied verbatim as `pbs_stdout.log` (top-level `MANIFEST.tsv`, `HashMatch=yes`).

## Warmup classification

```
Warmup: one global untimed H1W1A1 warmup per trial invocation
Warmup executions: 5
Measured trials per configuration: 5
Warmup included in statistics: no
```

Each of the 5 `run_ablation <graph> all` invocations begins with **one untimed
global H1W1A1 warmup** (`=== Warmup (untimed, H1W1A1) ===`, 5 markers in
`stderr/ablation.stderr.log`), then runs the 8 timed configurations. Warmup rows are
**not** written to the formal TSV; each configuration has exactly 5 measured trials.
(Do **not** describe this as "one warmup for the whole job".)

## Formal results

`ablation_results.tsv` — 40 rows, all `RunnerExit=0` / `Status=SUCCESS`, no NaN/Inf/
zero/OOM/timeout. `completeness.json` = PASS (`row_count=40`, `duplicate_trial_count=0`,
`configuration_count=8`). `ablation_results.partial.tsv` is a completion-time snapshot,
**byte-identical** to the formal TSV (not a distinct input). `ablation_per_config_stats.tsv`
holds per-config median/mean/sample-SD(ddof=1)/min/max, independently reproduced in
Gate W7.3C1 (max abs diff 4.75e-7, rounding).

### Median runtime (s) and H/W/A main effects (corrected 325557)

Main effect (existing definition, `scripts/summarize_ablation.py`): for each factor,
geometric mean over the 4 combinations of the other two factors of the per-config
median ratio T(F=0)/T(F=1).

| Factor | Main effect |
|--------|------------:|
| H (Hybrid BFS) | **1.4767** |
| W (Warp-cooperative accumulation) | **1.1012** |
| A (Dual streams) | **1.5563** |

Fastest→slowest by median: H1W1A1 (69.32 s) < H1W0A1 (78.87) < H0W1A1 (100.77) <
H1W1A0 (107.84) < H0W0A1 (112.09) < H1W0A0 (116.33) < H0W1A0 (163.74) < H0W0A0 (176.35).

## Layout

```
job_2406254/
├── SOURCE.md, MANIFEST.txt, run.log, pbs_stdout.log, graph_validation.json
├── ablation_results.tsv, ablation_results.partial.tsv
├── ablation_per_config_stats.tsv, completeness.json
└── stderr/  ablation.stderr.log + trial_{1..5}.stdout.tsv + completeness_after_trial_{1..5}.json
```
