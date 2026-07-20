# Appendix A Complete Experimental Parameters

This appendix lists the execution parameters required to reconstruct each experimental series in this study. It covers all batch sizes, environment variables, PBS resource records, checkpoints, timing scopes, and correctness tolerances. It does not reinterpret the experimental results. It records only the execution conditions corresponding to the methods defined in Chapter 5 and the results reported in Chapters 6 through 9. Runtime values and speedups are not repeated except where required to describe a condition. Appendix B contains all trial values from the sweeps, Appendix C contains values for all ablation configurations, Section A.8 specifies the correctness-validation conditions, and the T5 correctness table provides the detailed correctness metrics.

All entries are based on retained canonical materials. If the experiment-time code differs from a current script, the experiment-time snapshot (`code_snapshots/<SourceSnapshotID>/`) is authoritative for the execution conditions. A value unavailable from the retained records is marked `Not recorded`. A value that does not apply to a series is marked `N/A`. A record that exists but cannot be established independently is marked `Not independently verifiable`. These 3 states remain distinct, and unknown values are not estimated.

## A.1 Parameter Interpretation

Table A.1 defines the terms used in this appendix.

**Table A.1: Definitions of the experimental parameters recorded in this appendix.**

| Term | Definition | Recorded in |
|---|---|---|
| Requested Batch | Batch size requested at execution time through an environment variable or a default value. | Script arguments, run logs |
| Effective Batch | Batch size actually adopted by the implementation after evaluating the memory budget. It is clamped when the requested value exceeds the budget. | Run logs, `implementation_manifest.tsv` |
| `SUB_BATCH` | Size of the source sub-batch resident concurrently. This is a log field for the GPU_Opt family. | Run logs (`[Mem] SUB_BATCH=`) |
| `num_subs` | Number of source-sub-batch iterations used to process the Effective Batch. | Run logs (`num_subs=`) |
| $NS_{\mathrm{eff}}$ | Number of stream buffers active concurrently. The retained field name is `EffectiveNS`. | `implementation_manifest.tsv` (`EffectiveNS`) |
| $M_{\mathrm{source}}$ | State size per 1 source [bytes]. The retained field name is `PerSourceStateBytes`. | `implementation_manifest.tsv` (`PerSourceStateBytes`) |
| Trials | Number of formal trials $N_{\mathrm{trials}}$ recorded for the configuration. Warmup is excluded. | `SOURCE.md`, result TSV row counts |
| Warmup | Whether and how a preliminary execution excluded from formal trials was performed. | `SOURCE.md`, experiment-time scripts, raw logs |
| Aggregation | Method used to aggregate the primary value. This study uses the median as the primary value. | `SOURCE.md`, Chapter 5 |
| Timing Scope | Interval recorded as `Time_sec`, as defined in Section A.11. | `src/core/runner.cpp` |
| Checkpoint | Identifier of the experiment-time code: a SourceSnapshotID or commit SHA. | `code_snapshots/`, provenance TSV |
| PBS Job ID | PBS job identifier for the execution. | `SOURCE.md`, `result/MANIFEST.md` |
| Failure Status | Classification used when execution did not complete normally. It has no runtime value. | `feasibility_boundary.tsv`, `oom_evidence.tsv` |

The following interpretive convention applies to batches. A batch groups source vertices; it is not a graph partition. The outer batch loop processes every source vertex, so batching neither approximates BC nor omits sources. The GPU_Opt_Pure_Chunked sub-batch likewise groups sources rather than partitioning the graph edge set.

The following 2 capacity quantities also remain distinct. Graph file size is static on-disk storage. The batch-dependent working set is a code-derived allocation estimate: `EffectiveBatch`, or `SUB_BATCH` for Chunked, multiplied by $M_{\mathrm{source}}$ and $NS_{\mathrm{eff}}$. These are different concepts, and the latter determines the capacity boundaries in Section A.7.

For notation, this thesis represents the retained field `EffectiveNS` as $NS_{\mathrm{eff}}$ and `PerSourceStateBytes` as $M_{\mathrm{source}}$. The tables in this appendix use the thesis notation and provide the original field name where necessary.

## A.2 Hardware and Software Environment

Table A.2 presents the experimental environment. The entries are limited to records in `result/environment/environment.md`, `result/MANIFEST.md`, and `result/tables/thesis/T6_experimental_environment.tsv`; nominal specifications are not used to fill missing values.

**Table A.2: Hardware and software environment.**

| Component | Specification | Basis |
|---|---|---|
| System | Miyabi-G supercomputer, GPU compute node | Environment record |
| GPU | NVIDIA GH200 Grace Hopper Superchip (sm_90) | Environment record |
| CPU memory | Grace LPDDR5X, coupled to HBM3 via NVLink-C2C (900 GB/s coherent) | Environment record |
| On-package GPU memory (HBM3), nominal | 96 GB | NVIDIA specification |
| On-package GPU memory (HBM3), recorded | 97,871 MiB (approx. 95.6 GiB; approx. 102.6 decimal GB) | Environment record |
| GPU memory reported by runtime query at run start | total approx. 102.0 GB; free (`free_before`) approx. 101.4 GB (decimal GB) | Saved run logs |
| Host physical memory | Not recorded | — |
| Host-memory resource limit (memory-path experiments) | Host-memory-limited 100 GiB configuration | Environment record |
| NVIDIA Driver | 595.58.03 | Environment record |
| CUDA Toolkit (nvcc) | release 13.0, V13.0.48 | Environment record |
| Host C++ Compiler | g++ (GCC) 11.4.1 | Environment record |
| CMake | 4.3.4 | Environment record |
| Nsight Systems (nsys) | 2025.5.1.121 | Environment record |
| Scheduler | PBS batch system (Miyabi-G) | Environment record |
| Group | `gj17` | Environment record, PBS directives |
| Queue | Not independently verifiable from retained job logs | See A.3 |
| Graph format | Undirected, unweighted three-line text CSR | `data/README.md`, `Graph::readGraph()` |

The GPU-memory description separates 4 quantities. Quantity 1 is the nominal HBM3 capacity of 96 GB. Quantity 2 is the 97,871 MiB of device memory in the environment record. Quantity 3 comprises the runner's runtime-query values at launch: approximately 102.0 GB total and approximately 101.4 GB `free_before`. Quantity 4 is the host-memory-limited 100 GiB configuration imposed on the memory-path experiments. Quantities 1 through 3 describe the same on-package HBM3 through different unit systems and acquisition methods; they are not separate memory regions or tiers. Quantity 4 is a host-side resource constraint independent of HBM3 capacity. The retained records do not establish whether 100 GiB was a queue name, submission resource limit, or node configuration. It is therefore recorded only as a host-memory-limited configuration.

Section A.10 consolidates the implementation checkpoints. The complete `result/` tree does not correspond to a single checkpoint.

## A.3 PBS Resource Records

Table A.3 presents the PBS resource specifications for each experimental series. Values come from the `#PBS` directives in the submission scripts, using the experiment-time snapshot when one exists.

**Table A.3: PBS resource records per experiment series. Queue values are directive records, not confirmed queue usage.**

| Experiment Series | PBS Job ID | Queue Directive | Select | CPUs | GPUs | Host Memory Limit | Walltime | Evidence |
|---|---|---|---:|---:|---|---|---|---|
| Main performance | 2356120, 2357334, 2357335, 2357336, 2357337 | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 12:00:00 | `code_snapshots/phase_def_block_20260710/scripts/run_benchmark_targeted.sh` |
| PathMerge batch sweep | 2355000, 2355001, 2359080, 2359081, 2359096, 2359169, 2360072, 2360073, 2361040, 2361041, 2362006 | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 6:00:00 | `code_snapshots/phase_def_block_20260710/scripts/run_pathmerge_sweep.sh` |
| Kernel selection | 2354329, 2354330 | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 6:00:00 | `code_snapshots/phase_def_block_20260710/scripts/run_kernel_selection.sh` |
| Ablation (synthetic, email) | 2354994, 2354999 | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 6:00:00 | `code_snapshots/phase_def_block_20260710/scripts/run_ablation.sh` |
| Small full-vector correctness | 2367583.opbs | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 2:00:00 | `code_snapshots/small_correctness_20260712/scripts/run_small_correctness.sh` |
| Corrected 325557 validation (Series A/B) | 2404743.opbs | `regular-g` | 1 | 72 | Not specified in directive | Host-memory-limited 100 GiB configuration | 24:00:00 | `scripts/run_corrected_325557_validation.sh` |
| Corrected 325557 ablation (Series C) | 2406254.opbs | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 6:00:00 | `scripts/run_corrected_325557_ablation.sh` |
| Profiling (nsys, bandwidth) | 2359175 | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 2:00:00 | `code_snapshots/phase_def_block_20260710/scripts/run_profiling.sh` |
| Legacy memory scalability | Not recorded | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 24:00:00 | `code_snapshots/oldtree_f05ec52_20260512/scripts/run_um_oversubscribe*.sh` |
| Legacy memory-path correctness (historical) | 2368587.opbs, 2368269.opbs, 2368398.opbs, 2369632.opbs | `regular-g` | 1 | 72 | Not specified in directive | Host-memory-limited 100 GiB configuration | 6:00:00 | `code_snapshots/memory_correctness_20260712/scripts/run_memory_correctness.sh` |

The Queue Directive column records the submission script's `#PBS -q` directive. The actual queue name cannot be independently verified from retained job logs. The presence of the directive therefore does not establish that the job ran in that queue (Chapter 5, Section 5.12). The `select` specification contains no GPU count. Consequently, the directive cannot establish that count, and the table reports `Not specified in directive`. The environment record identifies the execution-node GPU as an NVIDIA GH200, but that is not a directive-based resource request.

`Host-memory-limited 100 GiB configuration` in the Host Memory Limit column is a resource condition recorded in the retained memory-path documentation. Comparable records do not exist for the other series, which are marked `Not recorded`. The legacy memory-scalability series ran from the old tree. Its logs do not record individual PBS job IDs, so they are marked `Not recorded`.

## A.4 Main Performance Parameters

Table A.4 presents the execution parameters for the RQ1 main performance comparison in Chapter 6. GPU_Opt used the same fixed batch on all 4 graphs and was not tuned graph by graph. PathMerge tuned used graph-specific batches, so the settings are asymmetric.

**Table A.4: Execution parameters of the main performance comparison.**

| Graph | Implementation | Requested Batch | Effective Batch | `SUB_BATCH` | `num_subs` | $NS_{\mathrm{eff}}$ | Trials | Warmup | Aggregation | Checkpoint | Timing Scope |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| email-EuAll | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 5 | None | Median | `phase_def_block_20260710` | Implementation function |
| roadNet-PA | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 3 | None | Median | `phase_def_block_20260710` | Implementation function |
| roadNet-TX | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 3 | None | Median | `phase_def_block_20260710` | Implementation function |
| roadNet-CA | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 3 | None | Median | `phase_def_block_20260710` | Implementation function |
| email-EuAll | PathMerge (tuned) | 2048 | 2048 | N/A | N/A | N/A | 3 | None | Median | `phase_def_block_20260710` | Implementation function |
| roadNet-PA | PathMerge (tuned) | 64 | 64 | N/A | N/A | N/A | 3 | Not recorded | Median | `oldtree_f05ec52_20260512` | Implementation function |
| roadNet-TX | PathMerge (tuned) | 64 | 64 | N/A | N/A | N/A | 3 | Not recorded | Median | `oldtree_f05ec52_20260512` | Implementation function |
| roadNet-CA | PathMerge (tuned) | 32 | 32 | N/A | N/A | N/A | 3 | None | Median | `phase_def_block_20260710` | Implementation function |

GPU_Opt used a fixed batch of 512 source vertices per 1 CUDA stream. Dual-Stream Execution gave $NS_{\mathrm{eff}}=2$. Thus, the number of source vertices processed concurrently in 1 batch iteration corresponds to $2\times512$. All 4 graphs were in capacity, so the requested and effective batches matched and no clamp occurred.

PathMerge uses an int2 frontier and per-source arrays. It has no source-sub-batching mechanism, so `SUB_BATCH`, `num_subs`, and $NS_{\mathrm{eff}}$ do not apply. For roadNet-PA/TX, the sweep confirmed that the best batch was the default b64, but the denominator adopted a separate legacy measurement under the same b64 setting. Therefore, the PathMerge checkpoint for these 2 graphs is `oldtree_f05ec52_20260512`, unlike the GPU_Opt checkpoint `phase_def_block_20260710`. The legacy series has no explicit warmup record and is marked `Not recorded`.

The GPU_Opt side uses the environment variable `BC_BATCH_OVERRIDE`, and the PathMerge side uses `PATHMERGE_BC_BATCH_SIZE`. Submission-script variables are `GRAPHS_STR`, `IMPLS_STR` (default `gpu_opt gpu_opt_pure gpu_opt_pure_chunked`), `TRIALS`, `TIMEOUT_SEC` (default 21600), and `SKIP_BUILD`.

## A.5 PathMerge Tuning Parameters

Table A.5 presents the PathMerge tuning procedure specified in Chapter 5, Section 5.7. This section covers sweep settings; Appendix B contains each trial's runtime value.

**Table A.5: PathMerge batch-sweep configuration per graph.**

| Graph | Requested Batch Candidates | Trials per Batch | Screening Job | Confirmation Job | Recorded Clamp | Adopted Tuned Batch | Checkpoint |
|---|---|---|---|---|---|---|---|
| roadNet-PA | 8, 16, 32, 64, 128, 256, 512 | Screening 1 (b8/b16/b32); Confirmation 3 (b64/b128/b256/b512) + 1 additional trial at b64; b64 pooled 4 | Job 2359080 (early terminated) | Job 2355001 | None recorded | 64 | `phase_def_block_20260710` (sweep) |
| roadNet-TX | 32, 64, 128 | Screening 1 (b32/b64/b128); Confirmation 2 (b32/b64); b32 and b64 pooled 3 | Job 2360072 | Job 2361040 | None recorded | 64 | `phase_def_block_20260710` (sweep) |
| roadNet-CA | 16, 32, 64, 128 | Screening 1 (b32/b64/b128); Extension 1 (b16); Confirmation 2 (b32/b64); b32 and b64 pooled 3 | Job 2360073 | Job 2362006 (b32/b64); Job 2361041 (b16 extension) | None recorded | 32 | `phase_def_block_20260710` |
| email-EuAll | 8, 16, 64, 256, 512, 1024, 2048, 4096, 8192 | Screening 1 (b8/b16/b64/b256/b1024); Confirmation 3 (b512–b8192); b1024 pooled 4 | Job 2359096 (early terminated) | Job 2359169 | Requested 8192 to effective 7393 | 2048 | `phase_def_block_20260710` |
| 325557_3216152 | 32, 64, 256, 512, 1024, 2048, 4096, 8192 | Initial exploration 1–2 (b32/b64/b256/b512); Screening 3 (b512/b1024/b2048); Confirmation 3 (b4096/b8192); b512 pooled 4 | Job 2355000 | Job 2359081 (b4096, b8192) | Requested 8192 to effective 6018 | 4096 | `phase_def_block_20260710` |

The tuned-batch selection rule is as follows. The median runtime is calculated for each candidate batch on each graph, and the batch with the lowest median is the best observed in the sweep. The final denominator uses the faster of the sweep best and default b64 (`scripts/merge_final_tables.py`). The sweeps used no warmup and median aggregation.

The relationship between the adopted and sweep-confirmation values for roadNet-PA and roadNet-TX requires clarification. The sweeps confirmed that the best batch for both graphs was the default b64, but the final table adopted separate legacy measurements under the same b64 setting. Thus, the confirmation measurement and final adopted value are distinct measurements of the same batch setting; they are neither missing nor contradictory. The legacy measurements were slightly faster than the sweep-confirmation measurements, so their adoption makes the speedup estimate for these 2 graphs smaller. For roadNet-CA and email-EuAll, the best observed sweep values were adopted directly as tuned.

The records contain 2 clamps. The requested email-EuAll b8192 exceeded the HBM3 budget and was reduced to an effective 7393. The requested 325557_3216152 b8192 was reduced to an effective 6018. Both records are based on warning lines in the retained logs. No clamp was recorded for the other candidate batches. 325557_3216152 is outside the RQ1 main performance comparison; its sweep supports the Tier B external-comparator setting b4096.

The following 2 points clarify the trial counts. Point 1 concerns the stage-wise entries in the Trials column of Table A.5. If the same requested batch was measured in screening and confirmation, these were separate measurements in separate jobs. The table lists them separately rather than adding them, and it separately reports the `pooled` $N_{\mathrm{trials}}$ used for sweep ranking. email-EuAll b1024 is an example: 1 screening trial in job 2359096 and 3 confirmation trials in job 2359169 give 4 trials in total. The `n=4` summary for email-EuAll b1024 in `result/tables/final_speedup_tables.md` is a pooled descriptive statistic over the screening 1 trial and confirmation 3 trials; it does not represent 4 trials from a single job. Similarly, roadNet-PA b64 pools 3 confirmation trials and 1 additional trial for 4. roadNet-TX and roadNet-CA b32/b64 pool 1 screening trial and 2 confirmation trials for 3. 325557_3216152 b512 pools 1 initial trial and 3 screening trials for 4. Appendix B provides each trial and both stage-wise and pooled aggregation.

Point 2 concerns the roadNet-PA and email-EuAll screenings. Both are recorded as trial 1 of intentionally early-terminated jobs, 2359080 and 2359096. Trial 1 completed and remains in the raw TSV with a value, but neither job has records for trial 2 or later. Appendix B documents this treatment.

Related environment variables are `PATHMERGE_BC_BATCH_SIZE` for the runner batch and the submission-script variables `BATCH_LIST` (default `1,2,4,8,16,32,64,128,256`), `TRIALS` (default 1), `GRAPHS_STR`, and `TIMEOUT_SEC` (default 21600).

## A.6 Ablation and Kernel-Selection Parameters

The RQ2 factor analysis comprises 2 series: ablation and kernel selection. Their purposes, target graphs, and checkpoints differ, so they are recorded separately.

### A.6.1 Ablation

The 3 factors are H = Hybrid BFS, W = Warp-Cooperative Accumulation, and A = Dual-Stream Execution. Compile-time templates produced the 8 configurations in $\mathrm{H}\{0,1\}\times\mathrm{W}\{0,1\}\times\mathrm{A}\{0,1\}$. Table A.6 presents their settings.

**Table A.6: Ablation experiment parameters by measurement series.**

| Series | Graphs | Configurations | Requested Batch | Effective Batch | `SUB_BATCH` | Trials per Configuration | Warmup | Aggregation | PBS Job ID | Checkpoint |
|---|---|---:|---:|---:|---|---:|---|---|---|---|
| Synthetic (three graphs) | benchmark_7000_41459, benchmark_11023_62184, 56438_300801 | 8 | 512 | 512 | Not recorded | 5 | One global untimed H1W1A1 per invocation; excluded from formal rows | Median per configuration | 2354994 | `phase_def_block_20260710` |
| Corrected 325557 | 325557_3216152_corrected_v1 | 8 | 512 | 512 | Not recorded | 5 | One global untimed H1W1A1 per invocation; excluded from formal rows | Median per configuration | 2406254 | `45352a344aaac463283a647467b790be9b45bfb8` |
| email-EuAll | email-EuAll | 8 | 512 | 512 | Not recorded | 3 | One global untimed H1W1A1 per invocation; excluded from formal rows | Median per configuration | 2354999 | `phase_def_block_20260710` |
| Historical (malformed 325557) | 325557_3216152 | 8 | 512 | 512 | Not recorded | 5 | One global untimed H1W1A1 per invocation | Median per configuration | 2354994 | `phase_def_block_20260710` |

No environment variable specified the batch. `src/proposed/host_ablation.cu` rounds the memory-budget-derived value to an upper bound of 512. Every series was in capacity at b512, as confirmed by `[Ablation H* W* A*] BATCH=512` in the logs. The ablation logs do not output `SUB_BATCH` or `num_subs`, so these values are `Not recorded`.

All 3 series use the same warmup procedure. Each `run_ablation <graph> all` runner invocation begins with 1 global, untimed H1W1A1 execution, which is excluded from formal TSV rows. The submission script starts the runner 1 time for each graph-trial pair. Therefore, the warmup occurs 1 time per runner invocation, or 1 time per trial, rather than 1 time per PBS job. Table A.6a lists the primary records for each series.

**Table A.6a: Warmup evidence per ablation series (marker counts from the raw logs, formal rows from the result TSVs).**

| Series | PBS Job ID | Runner Invocations | Warmup Markers | Formal Rows | Rows per Configuration | Warmup in Formal TSV |
|---|---|---:|---:|---:|---:|---|
| Synthetic (four graphs, incl. historical 325557) | 2354994 | 20 (4 graphs × 5 trials) | 20 | 160 | 20 | No |
| email-EuAll | 2354999 | 3 (1 graph × 3 trials) | 3 | 24 | 3 | No |
| Corrected 325557 | 2406254 | 5 (1 graph × 5 trials) | 5 | 40 | 5 | No |

Both the row counts and output path confirm that warmup is absent from the formal TSVs. Each series contains exactly 8 times the number of invocation rows and has no additional warmup rows. The runner directly calls the implementation function for warmup instead of using `run_brandes`, which performs timing and output. Therefore, warmup generates no TSV row on stdout. In the raw logs, the warmup marker follows each invocation header and precedes the 8 formal configurations.

The runner code is identical across the 3 series. At corrected-series checkpoint `45352a3`, `experiments/run_ablation.cu` is byte-identical to the corresponding file under `code_snapshots/phase_def_block_20260710/`. The warmup path does not depend on execution mode. It is suppressed only when `BC_ABLATION_WARMUP=0` is set. No retained record indicates suppression in these 3 series, and the markers confirm execution.

The main effect is calculated as follows. For factor $F$, the ratio $T(F{=}0)/T(F{=}1)$ is calculated from the per-configuration medians at each of the 4 level combinations of the other 2 factors. The geometric mean of those ratios is the main effect. Variation is reported with the sample standard deviation using ddof=1; for $N_{\mathrm{trials}}<2$, it is `n/a`.

The aggregate over the 4 synthetic graphs is mixed-checkpoint. benchmark_7000_41459, benchmark_11023_62184, and 56438_300801 come from job 2354994. The corrected 325557 measurement comes from job 2406254 at checkpoint `45352a3`. These are not measurements of all 4 graphs at the same checkpoint. The former malformed-325557 values remain as historical results and were not overwritten.

### A.6.2 Kernel Selection

Table A.7 presents the direct comparison of the BFS kernels. An environment variable forced each kernel; this is a forced comparison, not an evaluation of an automatic selection rule.

**Table A.7: Kernel-selection (forced shared/block) experiment parameters.**

| Graph | Forced Kernels | Requested Batch | Effective Batch | `SUB_BATCH` | `num_subs` | Trials per Kernel | Warmup | Aggregation | Correctness Level | PBS Job ID | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| roadNet-PA | shared, block | 512 | 512 | 512 | 1 | 3 | None | Median with sample standard deviation | `max_bc_only` | 2354329 | `phase_def_block_20260710` |
| roadNet-TX | shared, block | 512 | 512 | 512 | 1 | 3 | None | Median with sample standard deviation | `max_bc_only` | 2354330 | `phase_def_block_20260710` |

The environment variable `BC_FORCE_BFS_KERNEL=shared|block` forces the kernel. The submission script sets `BC_BATCH_OVERRIDE=512` by default. The aggregation script `scripts/summarize_kernel_selection.py` reports only forced shared/block measurements: median, sample standard deviation, $N_{\mathrm{trials}}$, the faster side, speedup, and Max BC agreement. It does not assess a selection rule.

A former implementation had a mean-degree-based automatic selection rule that chose shared when `avg_deg < 5`. The current method does not use that rule. Neither this appendix nor this thesis presents the former rule as current. Kernel selection covers only the 2 graphs roadNet-PA and roadNet-TX and is not generalized to other graphs.

## A.7 Memory-Scalability Parameters

The RQ3 capacity evaluation strictly separates the current corrected 325557 series from the legacy series. Current conclusions use only the corrected series from job 2404743.

### A.7.1 Corrected 325557 Targeted Boundary Series

**Table A.8: Corrected 325557 targeted feasibility parameters (job 2404743). Each condition was executed once.**

| Implementation | Requested Batch | Effective Batch | `SUB_BATCH` | `num_subs` | $NS_{\mathrm{eff}}$ | Trials | Warmup | Memory Mode | Status | Failure Class |
|---|---:|---:|---|---:|---:|---:|---|---|---|---|
| GPU_Opt_Pure | 4096 | 4096 | N/A | N/A | 2 | 1 | None | `explicit_device_memory` | Success | None |
| GPU_Opt_Pure | 8192 | 8192 | N/A | N/A | 2 | 1 | None | `explicit_device_memory` | Failure | CUDA device-memory OOM, exit 1 |
| GPU_Opt | 10240 | 10240 | 6596 | 2 | 1 | 1 | None | `managed_unified_memory` | Success | None |
| GPU_Opt | 12288 | 12288 | 6596 | 2 | 1 | 1 | None | `managed_unified_memory` | Failure | Cgroup host-memory OOM kill, exit 137 |
| GPU_Opt_Pure_Chunked | 16384 | 16384 | 6596 | 3 | 1 | 1 | None | `explicit_device_memory_chunked` | Success | None |

$M_{\mathrm{source}}=10{,}418{,}856$ bytes for $D_{est}=256$. The code-derived allocation estimate is $NS_{\mathrm{eff}}\times\mathrm{EffectiveBatch}\times M_{\mathrm{source}}$. For Chunked, the estimate uses `SUB_BATCH` and is $NS_{\mathrm{eff}}\times\mathrm{SUB\_BATCH}\times M_{\mathrm{source}}$. These estimates come from array dimensions; they are not measurements of process RSS, physical HBM residency, or migration bytes.

The status rules are as follows. OOM and kill conditions are not treated as 0 seconds; their runtime is `N/A`. OOM classification requires strong evidence: `cuda_oom`, `host_alloc_failure`, or `kernel_oom_kill`. Advisory warnings or mere mentions do not qualify as evidence. Pure b8192 was a CUDA device-memory OOM: `cudaMalloc` at `host_pure.cu:144` returned `out of memory`. UM b12288 recorded `oom_evidence=none` and exit 137 in the runtime record, while the PBS epilogue established a cgroup host-memory OOM kill. These are different failure classes and remain distinct.

Chunked b16384 `SUB_BATCH=6596` is not determined solely by the HBM3 budget. `host_chunked.cu` takes the smaller of the HBM-budget-derived limit and $\lfloor\mathrm{INT\_MAX}/n\rfloor=6596$, which prevents index overflow. The latter was the binding constraint for the corrected 325557 graph.

The tested-range interpretation is explicit. Chunked succeeded through b16384, the tested upper bound. This is neither a maximum nor unlimited capacity; it records success through the tested upper bound. UM is also finite and stopped at b12288 under the host-memory constraint. UM was not adopted because the input graph file exceeded 96 GB. The input file is 45,348,105 bytes, approximately 43.25 MiB. The capacity issue concerns placement and capacity evaluation of the batch-dependent working set.

### A.7.2 Legacy Memory-Scalability Series

**Table A.9: Legacy oversubscription sweep parameters (historical, malformed 325557 input).**

| Implementation | Requested Batches Tested | Trials | Warmup | Highest Successful Tested Batch | Batches Recorded as `OOM_OR_FAIL` | PBS Job ID | Checkpoint |
|---|---|---:|---|---:|---|---|---|
| GPU_Opt | 512, 1024, 2048, 4096, 8192, 10240, 12288 | 5 per batch (12288: 1) | None | 10240 | 12288 | Not recorded | `oldtree_f05ec52_20260512` |
| GPU_Opt_Pure | 512, 1024, 2048, 4096, 8192, 10240, 12288, 16384 | 5 per batch | Not recorded | 4096 | 8192, 10240, 12288, 16384 | Not recorded | `oldtree_f05ec52_20260512` |
| GPU_Opt_Pure_Chunked | 512, 1024, 2048, 4096, 8192, 10240, 12288, 16384 | 5 per batch | None | 16384 | None | Not recorded | `oldtree_f05ec52_20260512` |

The legacy series was measured on the former malformed input `325557_3216152` and is not used for the current RQ3 boundary. Warmup records differ by implementation. The experiment-time snapshot scripts for GPU_Opt and GPU_Opt_Pure_Chunked contain no warmup loop, and the raw-log execution counts match the TSV trial rows. Therefore, those implementations had no warmup. GPU_Opt_Pure raw logs have no trial headers, and its generating driver is absent from the snapshot, so its warmup is `not_recorded`.

`OOM_OR_FAIL` is a historical label in the legacy archive. It is coarser than the current distinction between CUDA device-memory OOM and cgroup host-memory OOM kill. In particular, GPU_Opt b12288 has `OOM_OR_FAIL` with $N_{\mathrm{trials}}=1$, but no independent record distinguishes CUDA OOM, host OOM kill, or scheduler OOM. Its cause is therefore not asserted. Legacy runtimes are not used as performance values for the current block implementation.

### A.7.3 Legacy Memory-Path Correctness Series

The legacy memory-path correctness series, job 2368587 and related jobs, was measured on the former malformed input. Results containing `CORE_FAIL` remain as historical evidence. The batches were GPU_Opt b1024/b9792, GPU_Opt_Pure b1024, GPU_Opt_Pure_Chunked b1024/b16384, and PathMerge b4096. Each configuration had 1 trial and no warmup, and the checkpoint was `memory_correctness_20260712`. These results are not used for current conclusions.

## A.8 Correctness-Validation Parameters

The RQ4 correctness validation has 2 evidence tiers, recorded separately because the independence of their references differs.

The decision uses a mixed absolute-relative tolerance. Each reference element $r_i$ and candidate element $c_i$ must satisfy

$$
|r_i-c_i|\le\mathrm{abs\_tol}+\mathrm{rel\_tol}\max(|r_i|,|c_i|)
$$

The canonical tolerances are `abs_tol = 1e-3` and `rel_tol = 1e-6`. They are not changed post hoc to produce a PASS. For large BC values, the absolute tolerance alone may be exceeded. Such cases are separated as WARN and do not independently constitute failure.

### A.8.1 Tier A: Independent CPU Reference

**Table A.10: Tier A validation parameters (independent Sequential CPU reference).**

| Graph | Vector Length | Reference | Candidate | Requested Batch | Effective Batch | `SUB_BATCH` | `num_subs` | $NS_{\mathrm{eff}}$ | Trials | Warmup | `abs_tol` | `rel_tol` | PBS Job ID | Checkpoint |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| benchmark_7000_41459 | 7000 | Sequential | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 1 | None | 1e-3 | 1e-6 | 2367583.opbs | `small_correctness_20260712` |
| benchmark_11023_62184 | 11023 | Sequential | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 1 | None | 1e-3 | 1e-6 | 2367583.opbs | `small_correctness_20260712` |
| chain_200 | 200 | Sequential | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 1 | None | 1e-3 | 1e-6 | 2367583.opbs | `small_correctness_20260712` |

Tier A compares every BC element against an independent Sequential CPU implementation. Its scope is limited to 3 small graphs. It excludes email-EuAll, the roadNet graphs, GPU_Opt_Pure, GPU_Opt_Pure_Chunked, and the UM-specific oversubscription path. Each configuration had $N_{\mathrm{trials}}=1$ and no warmup. The timing values from these executions do not support performance claims.

### A.8.2 Tier B: Cross-Implementation Consistency

**Table A.11: Tier B validation parameters on the corrected 325557 graph (job 2404743, checkpoint 45352a3).**

| Vector | Implementation | Requested Batch | Effective Batch | `SUB_BATCH` | `num_subs` | $NS_{\mathrm{eff}}$ | Vector Length | Batch Environment Variable |
|---|---|---:|---:|---|---:|---|---:|---|
| gpu_opt_b1024 | GPU_Opt | 1024 | 1024 | 1024 | 1 | 2 | 325557 | `BC_BATCH_OVERRIDE` |
| gpu_opt_b9792 | GPU_Opt | 9792 | 9792 | 6596 | 2 | 1 | 325557 | `BC_BATCH_OVERRIDE` |
| gpu_opt_pure_b1024 | GPU_Opt_Pure | 1024 | 1024 | N/A | N/A | 2 | 325557 | `BC_BATCH_OVERRIDE` |
| gpu_opt_pure_chunked_b1024 | GPU_Opt_Pure_Chunked | 1024 | 1024 | 1024 | 1 | 2 | 325557 | `BC_BATCH_OVERRIDE` |
| gpu_opt_pure_chunked_b16384 | GPU_Opt_Pure_Chunked | 16384 | 16384 | 6596 | 3 | 1 | 325557 | `BC_BATCH_OVERRIDE` |
| pathmerge_b4096 | PathMerge | 4096 | 4096 | N/A | N/A | N/A | 325557 | `PATHMERGE_BC_BATCH_SIZE` |

These 6 vectors form 10 pairs: 2 `same_impl_diff_batch`, 3 `same_batch_diff_path`, and 5 `pathmerge_cross`. Each configuration had $N_{\mathrm{trials}}=1$, no warmup, and the same tolerances as Tier A. The graph is `325557_3216152_corrected_v1`, SHA256 `8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22`, with $n=325{,}557$ and $m=3{,}216{,}152$.

Tier B records cross-implementation consistency; it is not independent ground truth. PathMerge is the evaluated third-party external comparator, not a reference truth.

This input corrects only out-of-range vertex IDs and the CSR-element-count inconsistency. It retains 87,442 self-loops and 866,924 duplicate directed adjacency pairs at multiplicity 2. The adjacency contains $2m=6{,}432{,}304$ elements. Therefore, the 10 Tier B comparisons evaluate agreement on this retained adjacency representation, not agreement with independent ground truth for simple-graph semantics.

### A.8.3 Recorded Validation Checks

**Table A.12: Validation checks recorded for every compared vector.**

| Check | Recorded Quantity | Tier A | Tier B |
|---|---|---|---|
| Vector length | Number of BC elements compared against $n$ | Recorded | Recorded |
| Missing index | Count of indices absent from the vector | Recorded | Recorded |
| Duplicate index | Count of repeated indices | Recorded | Recorded |
| Out-of-range index | Count of indices outside $[0,n)$ | Recorded | Recorded |
| NaN / Inf | Count of non-finite values | Recorded | Recorded |
| Mismatched elements | Count of elements violating the mixed tolerance | Recorded | Recorded |
| Byte identity | SHA256 equality of the compared vectors | Recorded | Recorded |

A mixed-tolerance PASS and byte identity are separate decisions. Across all 13 comparisons, this study recorded `ToleranceResult=PASS` and `ByteIdentical=No`. Numerical agreement within the mixed tolerance does not imply bitwise identity.

The `CORE_FAIL` from canonical job 2368587 on the former malformed input remains separately retained as historical invalid-input evidence and is not mixed into the current corrected-input conclusion. Conversely, the corrected-input PASS is not applied retroactively to the former input.

## A.9 Profiling Parameters

Table A.13 presents the settings for each profiling series. Only retained records are included.

**Table A.13: Profiling capture parameters (PBS job 2359175).**

| Capture | Profiler | Traced Binary / Implementation | Graph | Batch | `SUB_BATCH` | `num_subs` | $NS_{\mathrm{eff}}$ | Duration / Scope | Artifact Type | Checkpoint |
|---|---|---|---|---:|---|---|---|---|---|---|
| `ablation_H1W1A0` | Nsight Systems 2025.5.1.121 | `run_ablation`, configuration H1W1A0 | 56438_300801 | 512 | Not recorded | Not recorded | 1 | Full process duration; includes the untimed H1W1A1 warmup in the same process | `.nsys-rep`, `.stats.txt`, `.console.log` | `phase_def_block_20260710` |
| `ablation_H1W1A1` | Nsight Systems 2025.5.1.121 | `run_ablation`, configuration H1W1A1 | 56438_300801 | 512 | Not recorded | Not recorded | 2 | Full process duration | `.nsys-rep`, `.stats.txt`, `.console.log` | `phase_def_block_20260710` |
| `um_prefetch_gpu_opt` | Nsight Systems 2025.5.1.121 | `run_benchmark gpu_opt` | 325557_3216152 (pre-repair input) | 512 | 512 | 1 | 2 | Partial trace, `--duration=25` (25 seconds) | `.nsys-rep`, `.stats.txt`, `.console.log` | `phase_def_block_20260710` |
| `bandwidth` | `bandwidth_benchmark` | Device bandwidth measurement | N/A | N/A | N/A | N/A | N/A | Single measurement run | `bandwidth.log` | `phase_def_block_20260710` |

`um_prefetch_gpu_opt` is a 25-second partial trace captured with `--duration=25`; it is not a full-execution profile. Migration and fault counts in the trace are partial values, not totals for the complete execution. `ablation_H1W1A0` and `ablation_H1W1A1` are full-duration traces without a duration option. However, the `ablation_H1W1A0` scope includes the untimed H1W1A1 warmup at the start of the same process, so it does not isolate only the formal configuration.

The metadata discrepancy concerning the traced graphs has been resolved. The ablation traces target `56438_300801`, and the UM-prefetch trace targets `325557_3216152`. `SOURCE.md`, `result/MANIFEST.md`, and the raw `pbs_stdout.log` and `console.log` agree on this mapping. However, the UM-prefetch trace uses the former pre-repair input, not the corrected `325557_3216152_corrected_v1`. This remains an explicit interpretive limitation.

The `.stats.txt` files were regenerated from the retained `.nsys-rep` files with `nsys stats`; the original `.nsys-rep` files remain unchanged.

## A.10 Checkpoints and Provenance

Table A.14 maps each experimental series to its experiment-time code. The canonical provenance reference is SourceSnapshotID rather than commit SHA. The corrected 325557 series has no corresponding `code_snapshots/` entry and is identified by commit SHA, so the table distinguishes the identifier types.

**Table A.14: Experiment series, code snapshots, checkpoints, and canonical artifacts.**

| Experiment Series | Code Snapshot | Checkpoint | PBS Job IDs | Canonical Result | Canonical Raw Data |
|---|---|---|---|---|---|
| Main performance | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2356120, 2357334–2357337 | `result/main_performance/proposed_variants/` | `raw_data/main_performance/proposed_variants/` |
| PathMerge batch sweep | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2355000, 2355001, 2359080, 2359081, 2359096, 2359169, 2360072, 2360073, 2361040, 2361041, 2362006 | `result/tuning/pathmerge/` | `raw_data/tuning/pathmerge/`, `raw_data/unsuccessful/early_terminated/pathmerge_sweep/` |
| Kernel selection | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2354329, 2354330 | `result/tuning/kernel_selection/` | `raw_data/tuning/kernel_selection/` |
| Ablation (synthetic three graphs) | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2354994 | `result/ablation/synthetic_2354994/` | `raw_data/ablation/synthetic/` |
| Ablation (email-EuAll) | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2354999 | `result/ablation/email_2354999/` | `raw_data/ablation/email-EuAll/` |
| Ablation (corrected 325557) | No `code_snapshots/` entry | Commit `45352a344aaac463283a647467b790be9b45bfb8` | 2406254.opbs | `result/ablation/corrected_325557/` | `raw_data/corrected_325557/job_2406254/` |
| Correctness Tier A | `code_snapshots/small_correctness_20260712/` | `small_correctness_20260712` | 2367583.opbs | `result/correctness/small_full_vector/` | `raw_data/correctness/small_full_vector/` |
| Correctness Tier B and feasibility | No `code_snapshots/` entry | Commit `45352a344aaac463283a647467b790be9b45bfb8` | 2404743.opbs | `result/correctness/corrected_325557/`, `result/memory_scalability/corrected_325557/` | `raw_data/corrected_325557/job_2404743/` |
| Profiling | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2359175 | `result/profiling/` | `raw_data/profiling/job_2359175_20260711/` |
| Legacy memory scalability | `code_snapshots/oldtree_f05ec52_20260512/` | `oldtree_f05ec52_20260512` | Not recorded | `result/memory_scalability/` | `raw_data/memory_scalability/325557_3216152/` |
| Legacy memory-path correctness | `code_snapshots/memory_correctness_20260712/` | `memory_correctness_20260712` | 2368587.opbs | `result/correctness/memory_paths/` | `raw_data/unsuccessful/`, `raw_data/correctness/` |
| PathMerge default baseline (roadNet-PA/TX denominator) | `code_snapshots/oldtree_f05ec52_20260512/` | `oldtree_f05ec52_20260512` | Not recorded | `result/main_performance/seven_implementations/legacy_partial/` | `raw_data/main_performance/seven_implementations/legacy_partial/` |

The reader-facing table presents canonical paths at directory granularity. Complete file paths, SHA256 values, and generation commands appear in `raw_data/MANIFEST.tsv`, `raw_data/RAW_DATA_INDEX.tsv`, `raw_data/SHA256SUMS`, and `result/CORRECTED_325557_ARTIFACT_PROVENANCE.tsv`.

The 3 groups—the current block series, legacy tree, and corrected validation—use different checkpoints. The complete `result/` tree does not correspond to a single checkpoint. The repository's current HEAD is also not an experimental checkpoint, and the base commit at the time of writing is not treated as an execution condition. The authoritative execution conditions are the checkpoints in the table above and their corresponding snapshots or commits.

## A.11 Timing and Aggregation Conventions

The timing scope is the interval defined by `run_brandes()` in `src/core/runner.cpp`. Timing begins immediately before each implementation function call and ends immediately after it returns. As Table A.15 shows, graph-file reading, CSR construction, and input validation are outside the interval. Device allocation, host-device transfer, kernel execution, result retrieval, and synchronization performed inside the implementation function are within it.

**Table A.15: Timing scope of the recorded `Time_sec`.**

| Operation | Included in `Time_sec` |
|---|---|
| Graph file reading and CSR construction | No |
| Input CSR validation | No |
| Device allocation and managed allocation | Yes |
| Host-to-device transfer and prefetch | Yes |
| Kernel execution (BFS and backward phases) | Yes |
| Synchronization inside the implementation function | Yes |
| Device-to-host copy-out of the BC vector | Yes |
| Warmup runs | No |
| Result file writing and reporting | No |

The aggregation conventions are as follows. The primary value is the median. Supplementary values are the arithmetic mean, sample standard deviation $s_T$, minimum, and maximum. For $N_{\mathrm{trials}}$ trials, $s_T$ is the unbiased estimator with ddof=1:

$$
s_T=\sqrt{\frac{1}{N_{\mathrm{trials}}-1}\sum_{i=1}^{N_{\mathrm{trials}}}\left(t_i-\bar{t}\right)^2}
$$

Here, $t_i$ is each trial's runtime and $\bar{t}$ is the sample mean. Trial count uses $N_{\mathrm{trials}}$, distinct from the vertex count $n=|V|$. Conventional labels `n=3` and `n=5` in tables and figures denote $N_{\mathrm{trials}}$. A single fastest trial is not used as the representative value.

Speedup is a ratio of medians. The main comparison in this study uses

$$
\mathrm{Speedup}=\frac{T_{\mathrm{PathMerge,\ tuned}}}{T_{\mathrm{GPU\_Opt}}}
$$

Both numerator and denominator are median runtimes; no ratio mixes a median with a mean. Throughput for every implementation is

$$
\mathrm{GTEPS}=\frac{n\cdot m}{T\cdot10^{9}}
$$

and is calculated from the median runtime. Here, $n$ is the number of vertices and $m$ is the number of undirected edges.

Warmup is excluded from formal trials. For series with warmup, specifically the ablation series in Section A.6.1, the warmup execution is absent from formal TSV rows. If a series has no warmup record, it is marked `not_recorded`; no uniform default is assumed.

OOM, TIMEOUT, and FAIL are not aggregated as 0 seconds. An unavailable runtime is `N/A`, and failure classes distinguish CUDA device-memory OOM from cgroup host-memory OOM kill. This convention prevents infeasible conditions from appearing as fast performance values. Timing values from correctness-only executions, which had $N_{\mathrm{trials}}=1$ per configuration and no warmup, do not support performance claims.
