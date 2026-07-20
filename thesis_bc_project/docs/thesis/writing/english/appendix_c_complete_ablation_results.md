# Appendix C Complete Ablation Results

This appendix compiles all 8 configurations of the H/W/A factorial ablation, every retained formal trial, descriptive statistics by configuration, and recalculated main effects. Its purpose is to make the summary values in Chapter 7 auditable. It introduces no new performance claim, statistical-significance result, or causal decomposition of the end-to-end speedup in Chapter 6.

## C.1 Experimental Scope

Table C.1 separates the measurements into 3 series. Synthetic and email use the experiment-time snapshot `phase_def_block_20260710`, whereas Corrected uses commit `45352a344aaac463283a647467b790be9b45bfb8`. The series therefore do not share one checkpoint. The former `325557_3216152` in the Synthetic raw record is a historical record on malformed input and is not used for current conclusions. The current Synthetic-4 aggregate retains the other 3 graphs but replaces the former 325557 graph with `325557_3216152_corrected_v1` from the Corrected series. It is thus a mixed-checkpoint descriptive aggregate.

**Table C.1: Scope and provenance of the three ablation series.**

| Series | Job | Graphs | $N_{\mathrm{trials}}$ per Configuration | Formal Rows | Checkpoint |
|:--|:--|:--|--:|--:|:--|
| Synthetic | 2354994 | benchmark_7000_41459; benchmark_11023_62184; 56438_300801; 325557_3216152 (historical malformed input only) | 5 | 160 | `phase_def_block_20260710` |
| email | 2354999 | email-EuAll | 3 | 24 | `phase_def_block_20260710` |
| Corrected | 2406254 | 325557_3216152_corrected_v1 | 5 | 40 | `45352a344aaac463283a647467b790be9b45bfb8` |

<!-- Source: result/ablation/synthetic_2354994/SOURCE.md; result/ablation/email_2354999/SOURCE.md; raw_data/corrected_325557/job_2406254/SOURCE.md; result/COVERAGE.md -->

## C.2 Configuration Definition

H, W, and A are not 3 independent proposed methods. They are 3 factors that switch internal behavior within one proposed execution framework. The experiment-time `host_ablation.cu` uses 3 C++ template booleans to generate 8 dedicated instantiations at compile time. Although the runner selects a configuration at runtime, no kernel branch switches H, W, or A. Table C.2 lists the disabled and enabled states of each factor.

**Table C.2: Compile-time ablation factor definitions.**

| Factor | Name | Disabled State (0) | Enabled State (1) | Target Phase |
|:--|:--|:--|:--|:--|
| H | Hybrid BFS | Top-down traversal only | Direction-optimizing top-down / bottom-up switching | Forward BFS |
| W | Warp-Cooperative Accumulation | Thread-per-vertex accumulation kernel | Warp-cooperative accumulation kernel | Backward accumulation |
| A | Dual-Stream Execution | One CUDA stream (NS=1) | Two CUDA streams (NS=2) with double buffering | Initialization / execution pipeline |

Table C.3 lists every configuration. The value 0 denotes Disabled, and 1 denotes Enabled. H0W0A0 is the baseline, and H1W1A1 is the full configuration.

**Table C.3: Complete factorial configuration set.**

| Configuration | H | W | A | Role |
|:--|--:|--:|--:|:--|
| H0W0A0 | 0 | 0 | 0 | Baseline |
| H0W0A1 | 0 | 0 | 1 | Factorial cell |
| H0W1A0 | 0 | 1 | 0 | Factorial cell |
| H0W1A1 | 0 | 1 | 1 | Factorial cell |
| H1W0A0 | 1 | 0 | 0 | Factorial cell |
| H1W0A1 | 1 | 0 | 1 | Factorial cell |
| H1W1A0 | 1 | 1 | 0 | Factorial cell |
| H1W1A1 | 1 | 1 | 1 | Full configuration |

The main effects obtained from this factorial design are observed quantities. The 3 values are not additive contribution percentages and are not interpreted as a causal decomposition that disregards interactions.

<!-- Source: code_snapshots/phase_def_block_20260710/experiments/run_ablation.cu; code_snapshots/phase_def_block_20260710/src/proposed/host_ablation.cu; commit 45352a344aaac463283a647467b790be9b45bfb8 versions of the same files -->

## C.3 Formal Trial Completeness

The audit independently read the header and every row of each canonical raw TSV. It checked the expected graph set, 8 configurations, trial identifiers, numeric finiteness, success status, and unique key `(Series, Graph, Configuration, Trial)`. Table C.4 presents the results. Synthetic contained the expected 4 graphs, and the total number of formal rows was 224. `Time_sec` and `GTEPS` were finite and positive in every row. Every Corrected row had `RunnerExit=0` and `Status=SUCCESS`.

**Table C.4: Formal-row completeness and validity checks.**

| Series | Expected Rows | Observed Rows | Missing Rows | Duplicate Rows | Unknown Configurations | Non-Finite Runtime / GTEPS | Failed Rows | Result |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| Synthetic | 160 | 160 | 0 | 0 | 0 | 0 | 0 | Pass |
| email | 24 | 24 | 0 | 0 | 0 | 0 | 0 | Pass |
| Corrected | 40 | 40 | 0 | 0 | 0 | 0 | 0 | Pass |
| Total | 224 | 224 | 0 | 0 | 0 | 0 | 0 | Pass |

Table C.5 gives the number of formal trials for every graph and configuration. Each cell is $N_{\mathrm{trials}}$. Trial identifiers formed consecutive unique sets of 1--5 for Synthetic and Corrected and 1--3 for email.

**Table C.5: Formal trial counts for every graph and configuration.**

| Series | Graph | H0W0A0 | H0W0A1 | H0W1A0 | H0W1A1 | H1W0A0 | H1W0A1 | H1W1A0 | H1W1A1 | Trial IDs |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| Synthetic | benchmark_7000_41459 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1--5 |
| Synthetic | benchmark_11023_62184 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1--5 |
| Synthetic | 56438_300801 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1--5 |
| Synthetic | 325557_3216152 (historical malformed input only) | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1--5 |
| email | email-EuAll | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 1--3 |
| Corrected | 325557_3216152_corrected_v1 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1--5 |

All 3 series execute `one untimed global H1W1A1 warmup` 1 time per runner invocation. Synthetic has 20 invocations, 20 markers, and 160 formal rows. The corresponding counts are 3, 3, and 24 for email and 5, 5, and 40 for Corrected. The warmup does not pass through the formal TSV output path in `run_brandes`. Therefore, every H1W1A1 row below is a formal trial, and the 28 warmup executions are not added.

<!-- Source: raw_data/ablation/synthetic/job_2354994_20260710/ablation.log; raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv; raw_data/ablation/email-EuAll/job_2354999_20260710/ablation.log; raw_data/ablation/email-EuAll/job_2354999_20260710/ablation_results.tsv; raw_data/corrected_325557/job_2406254/stderr/ablation.stderr.log; raw_data/corrected_325557/job_2406254/ablation_results.tsv; code_snapshots/phase_def_block_20260710/experiments/run_ablation.cu -->

## C.4 Synthetic-Graph Results

The tables in this section transcribe the canonical raw TSV from Synthetic job 2354994. The first 3 graphs remain in the current Synthetic-4 descriptive aggregate. Graph 4, the former `325557_3216152`, appears as a historical table to preserve the completeness of the 160-row archive. It is excluded from the current conclusion, current corrected result, and current Synthetic-4 aggregate.

### C.4.1 benchmark_7000_41459

**Table C.6: All formal ablation trials on benchmark_7000_41459.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| benchmark_7000_41459 | 1 | 0 | 0 | 0 | 0.056044 | 5.1784 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 0 | 0 | 1 | 0.043375 | 6.6907 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 0 | 1 | 0 | 0.047983 | 6.0483 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 0 | 1 | 1 | 0.039475 | 7.3518 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 1 | 0 | 0 | 0.037516 | 7.7357 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 1 | 0 | 1 | 0.030033 | 9.6631 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 1 | 1 | 0 | 0.030794 | 9.4243 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 1 | 1 | 1 | 0.024220 | 11.9823 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 0 | 0 | 0 | 0.055944 | 5.1876 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 0 | 0 | 1 | 0.047010 | 6.1735 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 0 | 1 | 0 | 0.047848 | 6.0653 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 0 | 1 | 1 | 0.040408 | 7.1821 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 1 | 0 | 0 | 0.037555 | 7.7277 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 1 | 0 | 1 | 0.030664 | 9.4644 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 1 | 1 | 0 | 0.030817 | 9.4173 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 1 | 1 | 1 | 0.024701 | 11.7492 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 0 | 0 | 0 | 0.056000 | 5.1824 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 0 | 0 | 1 | 0.043951 | 6.6031 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 0 | 1 | 0 | 0.047935 | 6.0543 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 0 | 1 | 1 | 0.040490 | 7.1675 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 1 | 0 | 0 | 0.037762 | 7.6853 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 1 | 0 | 1 | 0.030730 | 9.4438 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 1 | 1 | 0 | 0.030817 | 9.4173 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 1 | 1 | 1 | 0.025249 | 11.4938 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 0 | 0 | 0 | 0.055571 | 5.2224 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 0 | 0 | 1 | 0.044580 | 6.5099 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 0 | 1 | 0 | 0.047647 | 6.0909 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 0 | 1 | 1 | 0.041129 | 7.0562 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 1 | 0 | 0 | 0.037430 | 7.7536 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 1 | 0 | 1 | 0.029892 | 9.7088 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 1 | 1 | 0 | 0.030631 | 9.4744 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 1 | 1 | 1 | 0.024660 | 11.7684 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 0 | 0 | 0 | 0.055777 | 5.2031 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 0 | 0 | 1 | 0.044221 | 6.5628 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 0 | 1 | 0 | 0.047784 | 6.0734 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 0 | 1 | 1 | 0.040967 | 7.0840 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 1 | 0 | 0 | 0.037290 | 7.7826 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 1 | 0 | 1 | 0.030251 | 9.5934 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 1 | 1 | 0 | 0.030506 | 9.5133 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 1 | 1 | 1 | 0.024595 | 11.7996 | 2354994 | `phase_def_block_20260710` |

### C.4.2 benchmark_11023_62184

**Table C.7: All formal ablation trials on benchmark_11023_62184.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| benchmark_11023_62184 | 1 | 0 | 0 | 0 | 0.155606 | 4.4051 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 0 | 0 | 1 | 0.096309 | 7.1173 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 0 | 1 | 0 | 0.153860 | 4.4550 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 0 | 1 | 1 | 0.094285 | 7.2700 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 1 | 0 | 0 | 0.083936 | 8.1664 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 1 | 0 | 1 | 0.054717 | 12.5274 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 1 | 1 | 0 | 0.083598 | 8.1994 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 1 | 1 | 1 | 0.054758 | 12.5178 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 0 | 0 | 0 | 0.157071 | 4.3640 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 0 | 0 | 1 | 0.094649 | 7.2421 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 0 | 1 | 0 | 0.151187 | 4.5338 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 0 | 1 | 1 | 0.096477 | 7.1049 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 1 | 0 | 0 | 0.084188 | 8.1419 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 1 | 0 | 1 | 0.054144 | 12.6598 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 1 | 1 | 0 | 0.083897 | 8.1702 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 1 | 1 | 1 | 0.054671 | 12.5377 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 0 | 0 | 0 | 0.153957 | 4.4523 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 0 | 0 | 1 | 0.095471 | 7.1797 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 0 | 1 | 0 | 0.153494 | 4.4657 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 0 | 1 | 1 | 0.093254 | 7.3504 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 1 | 0 | 0 | 0.083900 | 8.1699 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 1 | 0 | 1 | 0.054673 | 12.5372 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 1 | 1 | 0 | 0.084048 | 8.1555 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 1 | 1 | 1 | 0.055756 | 12.2937 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 0 | 0 | 0 | 0.156449 | 4.3813 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 0 | 0 | 1 | 0.095976 | 7.1420 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 0 | 1 | 0 | 0.152473 | 4.4956 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 0 | 1 | 1 | 0.094130 | 7.2820 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 1 | 0 | 0 | 0.084426 | 8.1190 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 1 | 0 | 1 | 0.054575 | 12.5598 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 1 | 1 | 0 | 0.084350 | 8.1263 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 1 | 1 | 1 | 0.055315 | 12.3917 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 0 | 0 | 0 | 0.155951 | 4.3953 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 0 | 0 | 1 | 0.097498 | 7.0305 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 0 | 1 | 0 | 0.153300 | 4.4713 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 0 | 1 | 1 | 0.092800 | 7.3863 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 1 | 0 | 0 | 0.084500 | 8.1119 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 1 | 0 | 1 | 0.055738 | 12.2978 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 1 | 1 | 0 | 0.084560 | 8.1062 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 1 | 1 | 1 | 0.055550 | 12.3394 | 2354994 | `phase_def_block_20260710` |

### C.4.3 56438_300801

**Table C.8: All formal ablation trials on 56438_300801.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| 56438_300801 | 1 | 0 | 0 | 0 | 3.976097 | 4.2697 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 0 | 0 | 1 | 3.277375 | 5.1799 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 0 | 1 | 0 | 3.963516 | 4.2832 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 0 | 1 | 1 | 3.227384 | 5.2602 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 1 | 0 | 0 | 2.012951 | 8.4337 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 1 | 0 | 1 | 1.620168 | 10.4783 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 1 | 1 | 0 | 2.083709 | 8.1473 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 1 | 1 | 1 | 1.647959 | 10.3016 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 0 | 0 | 0 | 3.975052 | 4.2708 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 0 | 0 | 1 | 3.278142 | 5.1787 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 0 | 1 | 0 | 3.968872 | 4.2774 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 0 | 1 | 1 | 3.221999 | 5.2690 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 1 | 0 | 0 | 2.011607 | 8.4393 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 1 | 0 | 1 | 1.623107 | 10.4593 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 1 | 1 | 0 | 2.081964 | 8.1541 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 1 | 1 | 1 | 1.646102 | 10.3132 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 0 | 0 | 0 | 3.979156 | 4.2664 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 0 | 0 | 1 | 3.276283 | 5.1817 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 0 | 1 | 0 | 3.971102 | 4.2750 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 0 | 1 | 1 | 3.217845 | 5.2758 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 1 | 0 | 0 | 2.012500 | 8.4356 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 1 | 0 | 1 | 1.619469 | 10.4828 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 1 | 1 | 0 | 2.083729 | 8.1472 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 1 | 1 | 1 | 1.651386 | 10.2802 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 0 | 0 | 0 | 3.977577 | 4.2681 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 0 | 0 | 1 | 3.277497 | 5.1797 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 0 | 1 | 0 | 3.965917 | 4.2806 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 0 | 1 | 1 | 3.225628 | 5.2630 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 1 | 0 | 0 | 2.012922 | 8.4338 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 1 | 0 | 1 | 1.617956 | 10.4926 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 1 | 1 | 0 | 2.081052 | 8.1577 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 1 | 1 | 1 | 1.649608 | 10.2913 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 0 | 0 | 0 | 3.986843 | 4.2582 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 0 | 0 | 1 | 3.276129 | 5.1819 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 0 | 1 | 0 | 3.971307 | 4.2748 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 0 | 1 | 1 | 3.224815 | 5.2644 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 1 | 0 | 0 | 2.009440 | 8.4484 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 1 | 0 | 1 | 1.619542 | 10.4823 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 1 | 1 | 0 | 2.085697 | 8.1395 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 1 | 1 | 1 | 1.648082 | 10.3008 | 2354994 | `phase_def_block_20260710` |

### C.4.4 Historical 325557_3216152 Record

The former `325557_3216152` is malformed input that stored 1-based vertex IDs as 0-based and lacked 7 adjacency elements. Table C.9 is included solely to present all 224 rows of the formal archive without omission. It is not used to compare a performance improvement or degradation against the corrected version.

**Table C.9: Historical formal trials on the malformed 325557_3216152 input, excluded from current conclusions.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| 325557_3216152 | 1 | 0 | 0 | 0 | 176.499175 | 5.9323 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 0 | 0 | 1 | 112.107097 | 9.3396 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 0 | 1 | 0 | 163.700166 | 6.3961 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 0 | 1 | 1 | 100.931930 | 10.3737 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 1 | 0 | 0 | 124.594405 | 8.4036 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 1 | 0 | 1 | 81.684899 | 12.8180 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 1 | 1 | 0 | 115.602630 | 9.0572 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 1 | 1 | 1 | 73.160699 | 14.3115 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 0 | 0 | 0 | 176.273764 | 5.9399 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 0 | 0 | 1 | 112.121695 | 9.3384 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 0 | 1 | 0 | 163.758087 | 6.3938 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 0 | 1 | 1 | 100.747634 | 10.3927 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 1 | 0 | 0 | 124.550105 | 8.4066 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 1 | 0 | 1 | 81.702279 | 12.8153 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 1 | 1 | 0 | 115.679890 | 9.0512 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 1 | 1 | 1 | 73.361727 | 14.2723 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 0 | 0 | 0 | 176.377429 | 5.9364 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 0 | 0 | 1 | 112.126686 | 9.3380 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 0 | 1 | 0 | 163.775986 | 6.3931 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 0 | 1 | 1 | 100.774364 | 10.3900 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 1 | 0 | 0 | 124.607520 | 8.4027 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 1 | 0 | 1 | 81.658002 | 12.8223 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 1 | 1 | 0 | 115.627324 | 9.0553 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 1 | 1 | 1 | 73.171411 | 14.3094 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 0 | 0 | 0 | 176.371341 | 5.9366 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 0 | 0 | 1 | 112.140839 | 9.3368 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 0 | 1 | 0 | 163.723856 | 6.3952 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 0 | 1 | 1 | 100.920807 | 10.3749 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 1 | 0 | 0 | 124.580846 | 8.4045 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 1 | 0 | 1 | 81.911587 | 12.7826 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 1 | 1 | 0 | 115.678980 | 9.0513 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 1 | 1 | 1 | 73.110497 | 14.3213 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 0 | 0 | 0 | 176.369826 | 5.9366 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 0 | 0 | 1 | 112.345169 | 9.3199 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 0 | 1 | 0 | 163.865469 | 6.3896 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 0 | 1 | 1 | 100.784210 | 10.3889 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 1 | 0 | 0 | 124.448354 | 8.4135 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 1 | 0 | 1 | 81.685797 | 12.8179 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 1 | 1 | 0 | 115.678792 | 9.0513 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 1 | 1 | 1 | 73.225436 | 14.2989 | 2354994 | `phase_def_block_20260710` |

<!-- Source: raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv; SHA256 ef0f787086d4dcfb4bba8181aa248ddecef74bda2759d2456af563e5bb5193eb -->

## C.5 email-EuAll Results

The email series is job 2354999, with $N_{\mathrm{trials}}=3$ for each configuration. It is not aggregated together with Synthetic or Corrected. Table C.10 presents all 24 formal trials.

**Table C.10: All formal ablation trials on email-EuAll.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| email-EuAll | 1 | 0 | 0 | 0 | 72.107592 | 1.3395 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 0 | 0 | 1 | 42.495161 | 2.2730 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 0 | 1 | 0 | 73.294355 | 1.3178 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 0 | 1 | 1 | 42.899488 | 2.2516 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 1 | 0 | 0 | 50.350710 | 1.9184 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 1 | 0 | 1 | 28.750180 | 3.3597 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 1 | 1 | 0 | 52.481184 | 1.8405 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 1 | 1 | 1 | 30.421410 | 3.1751 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 0 | 0 | 0 | 72.070743 | 1.3402 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 0 | 0 | 1 | 42.363314 | 2.2801 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 0 | 1 | 0 | 73.243350 | 1.3188 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 0 | 1 | 1 | 42.881928 | 2.2525 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 1 | 0 | 0 | 50.340890 | 1.9187 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 1 | 0 | 1 | 28.753547 | 3.3593 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 1 | 1 | 0 | 52.481763 | 1.8405 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 1 | 1 | 1 | 30.254042 | 3.1927 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 0 | 0 | 0 | 72.001806 | 1.3415 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 0 | 0 | 1 | 42.535161 | 2.2708 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 0 | 1 | 0 | 73.264358 | 1.3184 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 0 | 1 | 1 | 42.893899 | 2.2519 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 1 | 0 | 0 | 50.289317 | 1.9207 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 1 | 0 | 1 | 28.765522 | 3.3579 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 1 | 1 | 0 | 52.524467 | 1.8390 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 1 | 1 | 1 | 30.424089 | 3.1748 | 2354999 | `phase_def_block_20260710` |

<!-- Source: raw_data/ablation/email-EuAll/job_2354999_20260710/ablation_results.tsv; SHA256 77fd81068e345888b98368e4f88162b4a75780e7f25c6786deb7bd6a62b0c45a -->

## C.6 Corrected 325557 Results

Corrected is a separate series that remeasured `325557_3216152_corrected_v1` under job 2406254 and checkpoint `45352a3`, with $N_{\mathrm{trials}}=5$ for each configuration. All 40 rows succeeded, and none came from the former malformed input.

**Table C.11: All formal ablation trials on 325557_3216152_corrected_v1.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| 325557_3216152_corrected_v1 | 1 | 0 | 0 | 0 | 176.358035 | 5.9370 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 0 | 0 | 1 | 112.073926 | 9.3424 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 0 | 1 | 0 | 163.735910 | 6.3947 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 0 | 1 | 1 | 100.765035 | 10.3909 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 1 | 0 | 0 | 116.418547 | 8.9938 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 1 | 0 | 1 | 78.896392 | 13.2711 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 1 | 1 | 0 | 107.844926 | 9.7088 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 1 | 1 | 1 | 69.290090 | 15.1110 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 0 | 0 | 0 | 176.290382 | 5.9393 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 0 | 0 | 1 | 112.090565 | 9.3410 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 0 | 1 | 0 | 163.769977 | 6.3934 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 0 | 1 | 1 | 100.839434 | 10.3832 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 1 | 0 | 0 | 116.288659 | 9.0038 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 1 | 0 | 1 | 78.868771 | 13.2757 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 1 | 1 | 0 | 107.840803 | 9.7091 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 1 | 1 | 1 | 69.376091 | 15.0922 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 0 | 0 | 0 | 176.301738 | 5.9389 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 0 | 0 | 1 | 112.087905 | 9.3412 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 0 | 1 | 0 | 163.729329 | 6.3949 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 0 | 1 | 1 | 100.763793 | 10.3910 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 1 | 0 | 0 | 116.271917 | 9.0051 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 1 | 0 | 1 | 78.879567 | 13.2739 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 1 | 1 | 0 | 107.809662 | 9.7119 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 1 | 1 | 1 | 69.324039 | 15.1036 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 0 | 0 | 0 | 176.350609 | 5.9373 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 0 | 0 | 1 | 112.138071 | 9.3371 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 0 | 1 | 0 | 163.799227 | 6.3922 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 0 | 1 | 1 | 100.779244 | 10.3894 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 1 | 0 | 0 | 116.330316 | 9.0006 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 1 | 0 | 1 | 78.865354 | 13.2763 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 1 | 1 | 0 | 107.847937 | 9.7085 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 1 | 1 | 1 | 69.376184 | 15.0922 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 0 | 0 | 0 | 176.413246 | 5.9352 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 0 | 0 | 1 | 112.144524 | 9.3365 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 0 | 1 | 0 | 163.738336 | 6.3946 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 0 | 1 | 1 | 100.751330 | 10.3923 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 1 | 0 | 0 | 116.363404 | 8.9980 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 1 | 0 | 1 | 78.836244 | 13.2812 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 1 | 1 | 0 | 107.893349 | 9.7044 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 1 | 1 | 1 | 69.299777 | 15.1089 | 2406254 | `45352a3` |

<!-- Source: raw_data/corrected_325557/job_2406254/ablation_results.tsv; SHA256 ef96297cf4cf62addac0664636f79125e221b6fe8625973aab734fc16e36df04; full checkpoint 45352a344aaac463283a647467b790be9b45bfb8 -->

## C.7 Configuration-Level Statistics

For each graph and configuration, the audit recalculated the median, mean, sample standard deviation $s_T$ with ddof=1, minimum, maximum, and median trial-level GTEPS from the raw trials. It did not use the single fastest trial as the representative value. The calculation uses the GTEPS recorded for each trial with the Chapter 5 definition, $|V||E|/(T\times10^9)$. The series are not aggregated together.

### C.7.1 Current Synthetic Graphs

**Table C.12: Per-configuration statistics for the three retained synthetic graphs.**

| Graph | Configuration | $N_{\mathrm{trials}}$ | Median (s) | Mean (s) | $s_T$ (s) | Min (s) | Max (s) | Median GTEPS |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| benchmark_7000_41459 | H0W0A0 | 5 | 0.055944 | 0.055867 | 0.000194 | 0.055571 | 0.056044 | 5.1876 |
| benchmark_7000_41459 | H0W0A1 | 5 | 0.044221 | 0.044627 | 0.001403 | 0.043375 | 0.047010 | 6.5628 |
| benchmark_7000_41459 | H0W1A0 | 5 | 0.047848 | 0.047839 | 0.000132 | 0.047647 | 0.047983 | 6.0653 |
| benchmark_7000_41459 | H0W1A1 | 5 | 0.040490 | 0.040494 | 0.000647 | 0.039475 | 0.041129 | 7.1675 |
| benchmark_7000_41459 | H1W0A0 | 5 | 0.037516 | 0.037511 | 0.000173 | 0.037290 | 0.037762 | 7.7357 |
| benchmark_7000_41459 | H1W0A1 | 5 | 0.030251 | 0.030314 | 0.000373 | 0.029892 | 0.030730 | 9.5934 |
| benchmark_7000_41459 | H1W1A0 | 5 | 0.030794 | 0.030713 | 0.000139 | 0.030506 | 0.030817 | 9.4243 |
| benchmark_7000_41459 | H1W1A1 | 5 | 0.024660 | 0.024685 | 0.000369 | 0.024220 | 0.025249 | 11.7684 |
| benchmark_11023_62184 | H0W0A0 | 5 | 0.155951 | 0.155807 | 0.001172 | 0.153957 | 0.157071 | 4.3953 |
| benchmark_11023_62184 | H0W0A1 | 5 | 0.095976 | 0.095981 | 0.001054 | 0.094649 | 0.097498 | 7.1420 |
| benchmark_11023_62184 | H0W1A0 | 5 | 0.153300 | 0.152863 | 0.001066 | 0.151187 | 0.153860 | 4.4713 |
| benchmark_11023_62184 | H0W1A1 | 5 | 0.094130 | 0.094189 | 0.001419 | 0.092800 | 0.096477 | 7.2820 |
| benchmark_11023_62184 | H1W0A0 | 5 | 0.084188 | 0.084190 | 0.000274 | 0.083900 | 0.084500 | 8.1419 |
| benchmark_11023_62184 | H1W0A1 | 5 | 0.054673 | 0.054769 | 0.000587 | 0.054144 | 0.055738 | 12.5372 |
| benchmark_11023_62184 | H1W1A0 | 5 | 0.084048 | 0.084091 | 0.000377 | 0.083598 | 0.084560 | 8.1555 |
| benchmark_11023_62184 | H1W1A1 | 5 | 0.055315 | 0.055210 | 0.000479 | 0.054671 | 0.055756 | 12.3917 |
| 56438_300801 | H0W0A0 | 5 | 3.977577 | 3.978945 | 0.004679 | 3.975052 | 3.986843 | 4.2681 |
| 56438_300801 | H0W0A1 | 5 | 3.277375 | 3.277085 | 0.000856 | 3.276129 | 3.278142 | 5.1799 |
| 56438_300801 | H0W1A0 | 5 | 3.968872 | 3.968143 | 0.003379 | 3.963516 | 3.971307 | 4.2774 |
| 56438_300801 | H0W1A1 | 5 | 3.224815 | 3.223534 | 0.003727 | 3.217845 | 3.227384 | 5.2644 |
| 56438_300801 | H1W0A0 | 5 | 2.012500 | 2.011884 | 0.001470 | 2.009440 | 2.012951 | 8.4356 |
| 56438_300801 | H1W0A1 | 5 | 1.619542 | 1.620048 | 0.001893 | 1.617956 | 1.623107 | 10.4823 |
| 56438_300801 | H1W1A0 | 5 | 2.083709 | 2.083230 | 0.001797 | 2.081052 | 2.085697 | 8.1473 |
| 56438_300801 | H1W1A1 | 5 | 1.648082 | 1.648627 | 0.001981 | 1.646102 | 1.651386 | 10.3008 |

### C.7.2 email-EuAll

**Table C.13: Per-configuration statistics for email-EuAll.**

| Graph | Configuration | $N_{\mathrm{trials}}$ | Median (s) | Mean (s) | $s_T$ (s) | Min (s) | Max (s) | Median GTEPS |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| email-EuAll | H0W0A0 | 3 | 72.070743 | 72.060047 | 0.053698 | 72.001806 | 72.107592 | 1.3402 |
| email-EuAll | H0W0A1 | 3 | 42.495161 | 42.464545 | 0.089921 | 42.363314 | 42.535161 | 2.2730 |
| email-EuAll | H0W1A0 | 3 | 73.264358 | 73.267354 | 0.025634 | 73.243350 | 73.294355 | 1.3184 |
| email-EuAll | H0W1A1 | 3 | 42.893899 | 42.891772 | 0.008971 | 42.881928 | 42.899488 | 2.2519 |
| email-EuAll | H1W0A0 | 3 | 50.340890 | 50.326972 | 0.032978 | 50.289317 | 50.350710 | 1.9187 |
| email-EuAll | H1W0A1 | 3 | 28.753547 | 28.756416 | 0.008063 | 28.750180 | 28.765522 | 3.3593 |
| email-EuAll | H1W1A0 | 3 | 52.481763 | 52.495805 | 0.024824 | 52.481184 | 52.524467 | 1.8405 |
| email-EuAll | H1W1A1 | 3 | 30.421410 | 30.366514 | 0.097413 | 30.254042 | 30.424089 | 3.1751 |

### C.7.3 Corrected 325557

**Table C.14: Per-configuration statistics for corrected 325557.**

| Graph | Configuration | $N_{\mathrm{trials}}$ | Median (s) | Mean (s) | $s_T$ (s) | Min (s) | Max (s) | Median GTEPS |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 325557_3216152_corrected_v1 | H0W0A0 | 5 | 176.350609 | 176.342802 | 0.049218 | 176.290382 | 176.413246 | 5.9373 |
| 325557_3216152_corrected_v1 | H0W0A1 | 5 | 112.090565 | 112.106998 | 0.032024 | 112.073926 | 112.144524 | 9.3410 |
| 325557_3216152_corrected_v1 | H0W1A0 | 5 | 163.738336 | 163.754556 | 0.029498 | 163.729329 | 163.799227 | 6.3946 |
| 325557_3216152_corrected_v1 | H0W1A1 | 5 | 100.765035 | 100.779767 | 0.034790 | 100.751330 | 100.839434 | 10.3909 |
| 325557_3216152_corrected_v1 | H1W0A0 | 5 | 116.330316 | 116.334569 | 0.059023 | 116.271917 | 116.418547 | 9.0006 |
| 325557_3216152_corrected_v1 | H1W0A1 | 5 | 78.868771 | 78.869266 | 0.022068 | 78.836244 | 78.896392 | 13.2757 |
| 325557_3216152_corrected_v1 | H1W1A0 | 5 | 107.844926 | 107.847335 | 0.029939 | 107.809662 | 107.893349 | 9.7088 |
| 325557_3216152_corrected_v1 | H1W1A1 | 5 | 69.324039 | 69.333236 | 0.041069 | 69.290090 | 69.376184 | 15.1036 |

### C.7.4 Historical Malformed-Input Statistics

Table C.15 gives descriptive statistics for the former raw archive, not the current corrected result. Differences between the former and corrected values are not interpreted as a causal effect of input repair.

**Table C.15: Historical per-configuration statistics for the malformed 325557_3216152 input, excluded from current conclusions.**

| Graph | Configuration | $N_{\mathrm{trials}}$ | Median (s) | Mean (s) | $s_T$ (s) | Min (s) | Max (s) | Median GTEPS |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 325557_3216152 | H0W0A0 | 5 | 176.371341 | 176.378307 | 0.080093 | 176.273764 | 176.499175 | 5.9366 |
| 325557_3216152 | H0W0A1 | 5 | 112.126686 | 112.168297 | 0.099607 | 112.107097 | 112.345169 | 9.3380 |
| 325557_3216152 | H0W1A0 | 5 | 163.758087 | 163.764713 | 0.063558 | 163.700166 | 163.865469 | 6.3938 |
| 325557_3216152 | H0W1A1 | 5 | 100.784210 | 100.831789 | 0.087458 | 100.747634 | 100.931930 | 10.3889 |
| 325557_3216152 | H1W0A0 | 5 | 124.580846 | 124.556246 | 0.063970 | 124.448354 | 124.607520 | 8.4045 |
| 325557_3216152 | H1W0A1 | 5 | 81.685797 | 81.728513 | 0.103565 | 81.658002 | 81.911587 | 12.8179 |
| 325557_3216152 | H1W1A0 | 5 | 115.678792 | 115.653523 | 0.036257 | 115.602630 | 115.679890 | 9.0513 |
| 325557_3216152 | H1W1A1 | 5 | 73.171411 | 73.205954 | 0.096174 | 73.110497 | 73.361727 | 14.3094 |

## C.8 Main-Effect Calculation

The experiment-time generator `summarize_ablation.py` first obtains the median formal-trial runtime $T_g^{\mathrm{med}}$ for each graph and configuration. For factor $F\in\{\mathrm{H},\mathrm{W},\mathrm{A}\}$ and the other 2 factors $(G_1,G_2)$, it calculates the within-graph main effect as

$$
\mathrm{ME}_g(F)=\left(\prod_{(b_1,b_2)\in\{0,1\}^2}
\frac{T_g^{\mathrm{med}}(F{=}0,G_1{=}b_1,G_2{=}b_2)}
{T_g^{\mathrm{med}}(F{=}1,G_1{=}b_1,G_2{=}b_2)}\right)^{1/4}
$$

This is the geometric mean, rather than the arithmetic mean, of the 4 corresponding `factor OFF median / factor ON median` ratios. A ratio above 1 indicates that enabling the factor shortened the median runtime.

The generator loader skips `FAIL`, `TIMEOUT`, blank, and nonnumeric entries. It returns `None` when a denominator is 0 or a value is missing, and excludes `None` and nonpositive values from the geometric mean. This appendix does not rely on that permissive behavior. The completeness check in C.3 first requires the full trial set, finite positive runtime and GTEPS, and success status for every cell. Therefore, every main effect below uses all 4 ratios for each graph.

Table C.16 compares the independent recalculation with the formal contribution values retained to 4 decimal places. The contribution from the former malformed graph is not included in the current main-effect table.

**Table C.16: Current per-graph main effects, comparing independent recalculation with formal values.**

| Graph | Factor | Independent Value (Unrounded) | Formal Value (4 d.p.) | $N_{\mathrm{trials}}$ per Configuration | Checkpoint | Match |
|:--|:--|--:|--:|--:|:--|:--|
| benchmark_7000_41459 | H | 1.5356581539 | 1.5357 | 5 | `phase_def_block_20260710` | Yes |
| benchmark_7000_41459 | W | 1.1753491521 | 1.1753 | 5 | `phase_def_block_20260710` | Yes |
| benchmark_7000_41459 | A | 1.2335242686 | 1.2335 | 5 | `phase_def_block_20260710` | Yes |
| benchmark_11023_62184 | H | 1.7824071122 | 1.7824 | 5 | `phase_def_block_20260710` | Yes |
| benchmark_11023_62184 | W | 1.0066612430 | 1.0067 | 5 | `phase_def_block_20260710` | Yes |
| benchmark_11023_62184 | A | 1.5774308935 | 1.5774 | 5 | `phase_def_block_20260710` | Yes |
| 56438_300801 | H | 1.9649122517 | 1.9649 | 5 | `phase_def_block_20260710` | Yes |
| 56438_300801 | W | 0.9915651711 | 0.9916 | 5 | `phase_def_block_20260710` | Yes |
| 56438_300801 | A | 1.2376965147 | 1.2377 | 5 | `phase_def_block_20260710` | Yes |
| email-EuAll | H | 1.4285540284 | 1.4286 | 3 | `phase_def_block_20260710` | Yes |
| email-EuAll | W | 0.9695242840 | 0.9695 | 3 | `phase_def_block_20260710` | Yes |
| email-EuAll | A | 1.7198628708 | 1.7199 | 3 | `phase_def_block_20260710` | Yes |
| 325557_3216152_corrected_v1 | H | 1.4766622574 | 1.4767 | 5 | `45352a3` | Yes |
| 325557_3216152_corrected_v1 | W | 1.1011590412 | 1.1012 | 5 | `45352a3` | Yes |
| 325557_3216152_corrected_v1 | A | 1.5562810447 | 1.5563 | 5 | `45352a3` | Yes |

The current Synthetic-4 aggregate takes a further geometric mean over the graph-level main effects of the 4 graphs listed in Table C.18:

$$
\mathrm{ME}_{\mathrm{Synthetic-4}}(F)=
\left(\prod_{g\in\mathcal{G}_{\mathrm{current}}}\mathrm{ME}_g(F)\right)^{1/4}
$$

Table C.17 compares the independent recalculation with the formal values.

**Table C.17: Current Synthetic-4 mixed-checkpoint aggregate main effects.**

| Factor | Independent Value (Unrounded) | Formal Value (4 d.p.) | Chapter 7 Display | Aggregation | Match |
|:--|--:|--:|--:|:--|:--|
| H | 1.6787323050 | 1.6787 | 1.679 | Geometric mean across four graph-level main effects | Yes |
| W | 1.0661182797 | 1.0661 | 1.066 | Geometric mean across four graph-level main effects | Yes |
| A | 1.3913937847 | 1.3914 | 1.391 | Geometric mean across four graph-level main effects | Yes |

The generator outputs `InteractionRel`, the relative difference between `AddOne` and `LeaveOneOut`, as an auxiliary check for indications of interaction. It is not an estimate of a formal interaction term. This appendix does not conclude that interactions are absent. It also does not treat the main effects as mutually independent causal effects or additive percentages.

<!-- Source: code_snapshots/phase_def_block_20260710/scripts/summarize_ablation.py; result/ablation/synthetic_2354994/ablation_contributions.tsv; result/ablation/email_2354999/ablation_contributions.tsv; result/ablation/corrected_325557/ablation_contributions.tsv; result/ablation/corrected_325557/synthetic4_aggregate.tsv -->

## C.9 Cross-Series Interpretation

Within the current results, the observed main effects of H=1.4767 and A=1.5563 on Corrected 325557 exceeded W=1.1012. On email-EuAll, A=1.7199 and H=1.4286, whereas W=0.9695. The factor effects therefore differed by graph. In particular, W ranged from 0.9695 to 1.1753 and was graph-dependent. These descriptions are limited to the measured graphs, GPU, batch, and trial counts. They do not establish that H or A always accelerates execution, that W is unnecessary, or that unmeasured graphs behave identically. No significance test was conducted.

### C.9.1 Mixed-Checkpoint Aggregate

Table C.18 lists the members of the Synthetic-4 aggregate. Only Corrected 325557 uses a different checkpoint, so this is not a same-checkpoint controlled comparison of 4 graphs. The influence of checkpoint differences cannot be separated from graph differences or factor effects. Accordingly, this aggregate is a cross-series descriptive summary, not a rigorous estimate of causal effects.

**Table C.18: Membership and provenance of the current Synthetic-4 mixed-checkpoint aggregate.**

| Graph | Job | Checkpoint | Role in Current Aggregate |
|:--|:--|:--|:--|
| benchmark_7000_41459 | 2354994 | `phase_def_block_20260710` | Included, unchanged raw result |
| benchmark_11023_62184 | 2354994 | `phase_def_block_20260710` | Included, unchanged raw result |
| 56438_300801 | 2354994 | `phase_def_block_20260710` | Included, unchanged raw result |
| 325557_3216152_corrected_v1 | 2406254 | `45352a344aaac463283a647467b790be9b45bfb8` | Included, corrected replacement |

### C.9.2 Historical Malformed-Input Result

The job 2354994 ablation on the former `325557_3216152` was not deleted. Its raw data, derived summary and contribution, and failure and provenance records remain as historical evidence on malformed legacy input. The current corrected result and current main-effect aggregate exclude those values and use job 2406254 on `325557_3216152_corrected_v1` instead. Tables C.9 and C.15 are historical tables that make every archived formal row and its descriptive statistics auditable. They are not used to assess performance improvement or degradation by comparing the former and corrected values. The retained locations are `raw_data/ablation/synthetic/job_2354994_20260710/`, `result/ablation/synthetic_2354994/`, `failure/early_terminated/memory_correctness_2368398/`, `failure/failed/oom/memory_correctness_2368269/`, and `result/provenance/GRAPH_325557_INTEGRITY_AUDIT.md`.

## C.10 Recalculation and Validation

The independent recalculation used only the standard library and reread the canonical raw TSV on every run. It calculated completeness, median, mean, $s_T$, minimum, maximum, and median GTEPS by configuration, graph-level main effects, and the current Synthetic-4 aggregate. 2 executions of the same procedure produced byte-identical output. Both outputs had SHA256 `e77716fb31b76228824d9bfe04a9ef3fe35c6cf31a181ca04ff416d88cf384a5`.

The retained experiment-time `summarize_ablation.py` used population standard deviation (population SD) and does not byte-for-byte reproduce the formal summary. The current derivation script (`scripts/summarize_ablation.py`) regenerated Synthetic and email 2 times each. The `ablation_summary.md` and `ablation_contributions.tsv` outputs were byte-identical across runs, and the formal repository artifacts are reproduced by the current derivation script. The formal summary's numerical convention is sample standard deviation (sample SD, ddof=1). All raw trials and published Appendix C formal numerical values remain unchanged. This discrepancy concerns generator provenance and SD convention, not the experimental results. Historical authorship of the formal summary should not be inferred beyond the retained evidence. For Corrected, the independently calculated configuration statistics matched the formal `ablation_per_config_stats.tsv` within its rounding precision. The main effects and aggregate matched the formal contribution and aggregate TSVs to 4 decimal places.

Table C.19 summarizes the role of each canonical source. The raw TSVs are authoritative for trial values, and the experiment-time code of each series is authoritative for implementation conditions. The contribution and aggregate TSVs are authoritative for the formal rounded main effects.

**Table C.19: Canonical sources used for this appendix.**

| Source Class | Canonical Source | Use |
|:--|:--|:--|
| Thesis plan and method | `docs/thesis/writing/plan.md`; `docs/thesis/writing/japanese/05_experimental_methodology.md`; `docs/thesis/writing/japanese/appendix_a_experimental_parameters.md` | Scope, terminology, aggregation policy |
| Chapter alignment | `docs/thesis/writing/japanese/07_ablation_and_kernel_analysis.md`; `docs/thesis/writing/japanese/10_discussion.md` | Display values and interpretation limits |
| Synthetic raw | `raw_data/ablation/synthetic/job_2354994_20260710/` | 160 formal rows, raw logs, PBS stdout |
| email raw | `raw_data/ablation/email-EuAll/job_2354999_20260710/` | 24 formal rows, raw logs, PBS stdout |
| Corrected raw | `raw_data/corrected_325557/job_2406254/` | 40 formal rows, status, manifest, SHA256 evidence |
| Raw integrity indexes | `raw_data/MANIFEST.tsv`; `raw_data/RAW_DATA_INDEX.tsv`; `raw_data/SHA256SUMS`; `raw_data/corrected_325557/SHA256SUMS` | Job attribution, archive identity, SHA256 verification |
| Formal ablation results | `result/ablation/synthetic_2354994/`; `result/ablation/email_2354999/`; `result/ablation/corrected_325557/` | Summaries, contributions, corrected statistics, mixed aggregate |
| Thesis table and figure | `result/tables/thesis/T3_ablation_summary.tsv`; `result/tables/thesis/T3_ablation_summary.md`; `result/figures/thesis/ablation_contributions.pdf`; `result/figures/thesis/ablation_contributions.png`; `result/figures/thesis/ablation_contributions.svg` | Current displayed effects and provenance |
| Result catalog | `result/TABLES_AND_FIGURES.md`; `result/MANIFEST.md`; `result/COVERAGE.md`; `result/coverage_matrix.tsv` | Jobs, checkpoints, coverage, limitations |
| Thesis evidence catalog | `docs/thesis/thesis_values.tsv`; `docs/thesis/evidence_matrix.tsv` | Cross-document value and claim alignment |
| Experiment-time code | `code_snapshots/phase_def_block_20260710/experiments/run_ablation.cu`; `code_snapshots/phase_def_block_20260710/src/proposed/host_ablation.cu`; `code_snapshots/phase_def_block_20260710/scripts/summarize_ablation.py` | Synthetic/email implementation and calculation definition |
| Corrected checkpoint code | Commit `45352a344aaac463283a647467b790be9b45bfb8` versions of `run_ablation.cu`, `host_ablation.cu`, and `summarize_ablation.py` | Corrected implementation and calculation definition |
| Historical-input provenance | `result/provenance/GRAPH_325557_INTEGRITY_AUDIT.md`; `failure/early_terminated/memory_correctness_2368398/`; `failure/failed/oom/memory_correctness_2368269/` | Separation of malformed-input history |

Table C.20 presents the final validation results. The raw SHA256 values were Synthetic `ef0f7870...193eb`, email `77fd8106...c45a`, and Corrected `ef96297c...df04`. Each matched the raw SHA256 index. The formal and independently calculated values showed no differences. They also agreed with the Chapter 7 display values: H/W/A = 1.679/1.066/1.391 for Synthetic-4, 1.429/0.970/1.720 for email, and 1.4767/1.1012/1.5563 for Corrected 325557.

**Table C.20: Recalculation and document validation summary.**

| Check | Expected | Observed | Result |
|:--|:--|:--|:--|
| Synthetic formal rows | 160 | 160 | Pass |
| email formal rows | 24 | 24 | Pass |
| Corrected formal rows | 40 | 40 | Pass |
| Total formal rows | 224 | 224 | Pass |
| Missing graph-configuration cells | 0 | 0 | Pass |
| Duplicate formal rows | 0 | 0 | Pass |
| Unknown configurations | 0 | 0 | Pass |
| Non-finite or non-positive runtime / GTEPS | 0 | 0 | Pass |
| Failed formal rows | 0 | 0 | Pass |
| Warmup rows included | 0 | 0 | Pass |
| Per-configuration median / mean / sample standard deviation / min / max / median GTEPS | Exact recomputation | Match | Pass |
| Per-graph main effects | Formal value at 4 d.p. | Match | Pass |
| Current Synthetic-4 aggregate | H=1.6787; W=1.0661; A=1.3914 | Match | Pass |
| Independent recalculation repeatability | Byte-identical | SHA256 identical | Pass |
| Derivation script repeatability | Byte-identical | Synthetic and email outputs identical | Pass |
| Chapter 7 value alignment | Exact at displayed precision | Match | Pass |
| Mixed-checkpoint disclosure | Required | C.1, C.8, C.9.1 | Pass |
| Historical malformed-input separation | Required | C.4.4, C.7.4, C.9.2 | Pass |

This appendix adds no citation keys and uses only canonical repository paths in Source notes. Every numerical value was obtained or recalculated from a raw record or formal derived artifact. No runtime was inferred from a rounded main effect.
