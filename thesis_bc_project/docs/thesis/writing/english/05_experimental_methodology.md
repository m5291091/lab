# Chapter 5 Experimental Methodology

This chapter defines the evaluation methodology used in this study. The target is the exact computation of Betweenness Centrality (BC) with the Brandes algorithm based on all-sources Breadth-First Search (BFS) [@brandes2001]. The chapter describes the computing environment, evaluated graphs, compared implementations, execution settings, timing and statistical procedures, factor analysis, capacity evaluation, correctness validation, and reproducibility management. These procedures compare the proposed batch-based GPU execution framework and its main implementation, GPU_Opt, with the evaluated third-party PathMerge implementation and supplementary baselines. PathMerge here denotes only the evaluated snapshot and configuration. It is an external comparator, not ground truth, and was not confirmed as the original authors' official implementation. Its upstream license was not independently verified, and the comparison is not generalized to PathMerge as an algorithm. This chapter addresses evaluation methods; Chapters 6 through 9 present the numerical results. Numerical values in this chapter are generally limited to settings needed to specify the experimental conditions, including batch sizes, trial counts, and tolerances.

All descriptions are based on retained canonical materials. The code used for each experiment is frozen under a SourceSnapshotID, and raw data, derived tables and figures, and unsuccessful data are managed separately. Section 5.11 and Appendix A describe the storage structure and provenance.

## 5.1 Research Questions

The evaluation in this study is organized around the following 4 research questions.

- Scope: 4 evaluated graphs. RQ1: On the four evaluated graphs, is the block-based GPU_Opt implementation with a fixed batch size of 512 faster than the graph-wise tuned third-party PathMerge implementation?
- RQ2: To what extent do Hybrid BFS, Warp-Cooperative Accumulation, Dual-Stream Execution, and the Block Kernel contribute to the observed performance?
- RQ3: On the evaluated corrected 325557 graph, how do the memory-management approaches of GPU_Opt, GPU_Opt_Pure, and GPU_Opt_Pure_Chunked affect the feasible batch size and the observed memory constraints?
- RQ4: To what extent do the BC vectors produced by the proposed implementations agree with an independent reference and across different memory paths, and what numerical-representation and provenance limitations remain?

This evaluation separates performance evaluation (RQ1), factor analysis (RQ2), capacity evaluation (RQ3), and numerical correctness (RQ4) as independent perspectives. Performance evaluation examines median runtime, speedup, and GTEPS, whereas correctness validation examines numerical agreement between BC vectors. The measured quantities and decision criteria therefore differ. Runtime values from correctness runs, which used `n=1` per configuration and no warmup, are not used for performance claims. Conversely, performance runs are not used as correctness evidence. Factor analysis and capacity evaluation are also dedicated experiments limited to synthetic or specific graphs. Their graphs and purposes differ from those of the main RQ1 performance comparison.

Table 5.1 summarizes the mapping between each RQ and its experiments, primary metrics, aggregation methods, and trial counts. Section 5.6 provides details of trial counts and aggregation.

**Table 5.1: Evaluation protocol summary (mapping of research questions to experiments).**

| RQ | Experiment | Graphs | Primary metric | Aggregation | Trials (n) |
|---|---|---|---|---|---|
| RQ1 | Main performance: GPU_Opt vs tuned PathMerge | email-EuAll, roadNet-PA, roadNet-TX, roadNet-CA | runtime, speedup, GTEPS | median | GPU_Opt: email 5 / road 3; PathMerge: 3 |
| RQ2 | Ablation (H/W/A) | benchmark_7000_41459, benchmark_11023_62184, 56438_300801, 325557_3216152_corrected_v1, email-EuAll | main effect | median | 5 per synthetic configuration / 3 per email configuration |
| RQ2 | Kernel selection (forced shared/block) | roadNet-PA, roadNet-TX | runtime, speedup | median (+ sample standard deviation) | 3 |
| RQ3 | Memory scalability (UM / Pure / Chunked) | 325557_3216152_corrected_v1 | targeted feasibility and failure class | none (single run) | 1 per targeted condition |
| RQ4 | Correctness (Tier A / Tier B) | benchmark_7000_41459, benchmark_11023_62184, chain_200, 325557_3216152_corrected_v1 | missing, mismatch, max abs/rel error, byte identity | single-run full-vector comparison | 1 per comparison |

<!-- Source note (internal): trial counts and aggregation from result/main_performance/proposed_variants/SOURCE.md; result/tuning/{pathmerge,kernel_selection}/SOURCE.md; result/ablation/*/SOURCE.md; result/memory_scalability/SOURCE.md; result/correctness/*/README.md. -->

## 5.2 Hardware and Software Environment

The evaluation was conducted on GPU compute nodes of the Miyabi-G supercomputer. The system uses the NVIDIA GH200 Grace Hopper Superchip (sm_90). GH200 uses a coherent memory model that connects the Grace CPU LPDDR5X memory and GPU HBM3 through NVLink-C2C with 900 GB/s coherence [@nvidiaGh200Product; @nvidiaGraceHopperInDepth]. Unified Memory (UM) uses managed allocation and migration between the CPU and GPU to handle a working set that may exceed device memory. This capability underlies the RQ3 memory-capacity evaluation, but this study did not measure migration bytes or physical HBM residency.

Values for GPU memory capacity are distinguished according to their evidence and unit systems. The nominal capacity of on-package HBM3 is 96 GB [@nvidiaGraceHopperInDepth]. The retained run-environment record reports 97871 MiB of device memory, corresponding to approximately 95.6 GiB or approximately 102.6 GB in decimal (base-10) units. The nominal 96 GB and recorded 97871 MiB describe the same on-package HBM3 using different units and acquisition methods. They do not represent separate memory regions or tiers. Separately, the runner's memory query reported approximately 102.0 GB total and approximately 101.4 GB free (`free_before`) at run start, both in decimal (base-10) GB. The difference between approximately 102.0 GB and 97871 MiB (approximately 102.6 GB) results from the different acquisition methods; both refer to the same HBM3. The approximately 101.4 GB value is available memory at run start, not total capacity. This evaluation uses it only as the memory-budget reference for clamping the effective batch by comparison with the estimated working-set size. The primary environment record is `result/environment/environment.md`, and the run-start memory-query values are in the retained logs for each run.

The software environment comprised NVIDIA driver 595.58.03, CUDA Toolkit (nvcc) release 13.0 (V13.0.48) [@nvidiaCudaProgrammingGuide], g++ (GCC) 11.4.1 as the host C++ compiler, CMake 4.3.4, and Nsight Systems (nsys) 2025.5.1.121. The main experiments ran through the PBS (Portable Batch System) on Miyabi-G under group `gj17`. The RQ3 and RQ4 memory-path experiments used a host-memory-limited 100 GiB resource configuration. Because this configuration has a different host-memory limit from the legacy capacity evaluation, their OOM boundaries do not coincide (Section 5.9). Thus, not every experiment in this study used a single resource specification. The actual queue name could not be determined independently from the retained logs, so it is excluded from the canonical experimental conditions (Section 5.12).

Effective bandwidth for each GH200 memory path—HBM3 device-to-device, pinned host-device, and NVLink-C2C prefetch—was measured separately with a bandwidth benchmark. These values are supplementary platform measurements and are retained in the environment record, the T6 canonical artifact in Table 5.2. This study does not use them to explain the performance or failure cause of an individual experiment. Section 5.11 describes the mapping between experiment groups and SourceSnapshotIDs. The complete set of derived results does not correspond to a single SourceSnapshotID.

<!-- Source note (internal): raw_data/profiling/job_2359175_20260711/bandwidth.log. -->

**Table 5.2: Experimental environment.**

| Component | Specification |
|---|---|
| GPU | NVIDIA GH200 Grace Hopper Superchip (sm_90) |
| On-package GPU memory (HBM3), nominal | 96 GB (NVIDIA specification) |
| On-package GPU memory (HBM3), recorded for the run environment | 97871 MiB (= approx. 95.6 GiB = approx. 102.6 decimal GB; same HBM3 as the nominal 96 GB) |
| GPU memory reported by the runtime query at run start | total approx. 102.0 GB; free (`free_before`) approx. 101.4 GB (decimal GB) |
| CPU memory | Grace LPDDR5X, coupled to HBM3 via NVLink-C2C (900 GB/s coherent) |
| NVIDIA Driver | 595.58.03 |
| CUDA Toolkit (nvcc) | release 13.0, V13.0.48 |
| Host C++ Compiler | g++ (GCC) 11.4.1 |
| CMake | 4.3.4 |
| Nsight Systems (nsys) | 2025.5.1.121 |
| Scheduler / Group | PBS batch system (Miyabi-G), group gj17 |
| Resource configuration — memory-path experiments (RQ3/RQ4) | Host-memory-limited 100 GiB configuration |

<!-- canonical artifact: T6_experimental_environment -->
<!-- Source note (internal): result/environment/environment.md; result/MANIFEST.md; nominal HBM3 capacity from nvidiaGraceHopperInDepth; run-start total/free_before from retained logs. Queue name is excluded because it cannot be independently confirmed. -->

## 5.3 Graph Datasets

Table 5.3 lists the properties of the graphs used in this evaluation. The values come from the canonical records in `result/datasets/graph_catalog.tsv` and `docs/graph_stats.tsv`; none were inferred or reconstructed from rounded values. All graphs are retained in undirected, unweighted CSR format. Input identity is managed with the graph SHA256 recorded in `graph_catalog.tsv`.

The graphs are divided into real-world and synthetic datasets. The real-world datasets are 4 graphs obtained from the Stanford Network Analysis Project (SNAP) [@snapnets]. email-EuAll was originally a directed email communication network [@leskovec2007graphevolution] and was symmetrized for this evaluation (`Symmetrized=yes`). roadNet-PA/TX/CA were originally undirected road networks [@leskovec2009community] and were used without symmetrization (`Symmetrized=no`). These 4 graphs are the RQ1 targets. Synthetic graphs were used for factor analysis, capacity evaluation, and correctness validation. The former `325557_3216152` stored 1-based identifiers as if they were 0-based and was therefore malformed input. Current experiments use only `325557_3216152_corrected_v1`, reconstructed through a deterministic repair procedure. The former input and its results are retained as historical provenance.

<!-- Source note (internal): deterministic repair tool tools/repair_325557_graph.py. -->

The structure of the corrected 325557 graph requires explicit qualification. The repair corrected out-of-range vertex identifiers and an inconsistency in the number of CSR elements; it did not normalize the graph. Self-loops and duplicate adjacency pairs were not removed and remain in the retained representation. The corrected graph preserves $n=325{,}557$, $m=3{,}216{,}152$, and $2m=6{,}432{,}304$ adjacency elements. These elements include 87,442 self-loops and 866,924 duplicate ordered pairs at multiplicity 2 (`result/provenance/GRAPH_325557_INTEGRITY_AUDIT.md`). The node and edge counts in Table 5.3 and the mean degree of 19.758 are based on this retained adjacency representation.

Consequently, the RQ2 ablation, RQ3 capacity boundary, and RQ4 Tier B cross-implementation consistency results all apply to this retained adjacency representation. In particular, RQ4 Tier B measures numerical consistency among implementations on the same input. It is not agreement with an independent reference for simple-graph semantics after removing self-loops and parallel edges. The corrected 325557 graph is not part of the RQ1 main comparison, which uses the 4 graphs email-EuAll and roadNet-PA/TX/CA. Therefore, its structure does not affect the RQ1 speedups of 3.17, 1.31, 1.51, and 1.45. This study does not claim that the self-loops or duplicate edges caused a performance difference among implementations.

The graphs were selected for the following purposes. The 4 RQ1 graphs include 2 contrasting structures. email-EuAll has a high-variance hub structure and shallow BFS depth, whereas roadNet-PA/TX/CA have more uniform degrees and deep BFS traversals. This contrast supports evaluation across regions with different degree distributions and traversal depths. The corrected 325557 graph was selected because it can create conditions that retain source-local state for many source vertices in 1 batch, not because its input file is large. It is used for the RQ2 ablation, RQ3 capacity boundary, and RQ4 cross-implementation consistency, but not for the RQ1 main comparison. benchmark_7000_41459, benchmark_11023_62184, and chain_200 were used for full-vector validation against an independent CPU reference.

**Table 5.3: Graph datasets and static storage sizes. Input File is the on-disk text CSR size; CSR is the theoretical int32 topology size.**

| Graph | Nodes | Edges | Input File [MiB] | CSR [MiB] | Used For |
|---|---:|---:|---:|---:|---|
| email-EuAll | 265009 | 364481 | 5.59 | 3.79 | Main performance (RQ1); Ablation |
| roadNet-PA | 1088092 | 1541898 | 28.43 | 15.91 | Main performance (RQ1); Kernel selection |
| roadNet-TX | 1379917 | 1921660 | 36.53 | 19.93 | Main performance (RQ1); Kernel selection |
| roadNet-CA | 1965206 | 2766607 | 53.83 | 28.60 | Main performance (RQ1) |
| 325557_3216152_corrected_v1 | 325557 | 3216152 | 43.25 | 25.78 | Ablation; Memory scalability; Correctness |
| 56438_300801 | 56438 | 300801 | 3.72 | 2.51 | Ablation |
| benchmark_7000_41459 | 7000 | 41459 | 0.39 | 0.34 | Ablation; Correctness |
| benchmark_11023_62184 | 11023 | 62184 | 0.61 | 0.52 | Ablation; Correctness |
| benchmark_85830 | 85830 | 241080 | 3.18 | 2.17 | Auxiliary |
| chain_200 | 200 | 199 | <0.01 | <0.01 | Correctness |
| random | 32212 | 101805 | 1.30 | 0.90 | Auxiliary |
| 325557_3216152 | 325557 | 3216152 | 43.25 | 25.78 | Historical (superseded by corrected_v1) |

<!-- canonical artifact: T1_graph_metadata -->
<!-- Source note (internal): result/tables/thesis/T1_graph_metadata.tsv generated from docs/graph_stats.tsv and result/datasets/graph_catalog.tsv. Corrected graph SHA256 8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22; jobs 2404743/2406254; checkpoint 45352a3. -->

Input File, CSR topology, and the BC output vector are static storage, not the GPU working set. None of the input graph files exceeds the nominal HBM capacity, and graph-file size is not the reason for using UM. For the corrected 325557 graph, the static sizes are 45,348,105 bytes, 27,031,448 bytes, and 2,604,456 bytes, respectively. For the implementation-selected $D_{est}=256$, derived from the mean degree of 19.758, the state size for 1 source vertex is

$$
M_{\mathrm{source}}=32n+4D_{est}+8=10{,}418{,}856\ \mathrm{bytes}
$$

and the code-derived working-set estimate is defined as

$$
M_{\mathrm{work}}=NS_{\mathrm{eff}}\times \mathrm{EffectiveBatch}\times M_{\mathrm{source}}
$$

Chunked uses the concurrently resident `SUB_BATCH` in place of $\mathrm{EffectiveBatch}$. The retained manifest fields `PerSourceStateBytes` and `EffectiveNS` are denoted by $M_{\mathrm{source}}$ and $NS_{\mathrm{eff}}$, respectively, in this thesis. These quantities are allocation estimates, not measurements of process RSS, physical HBM residency, or migration bytes. A batch groups source vertices; it does not partition the graph. Iteration over outer batches and sub-batches processes every source vertex, so batching neither approximates BC nor omits source vertices.

## 5.4 Evaluated Implementations

Table 5.4 lists the implementations compared in this evaluation. The main implementation is GPU_Opt, which uses Unified Memory and the block kernel. Its primary comparator is the graph-wise tuned third-party PathMerge implementation. Sequential, OpenMP, and cuGraph serve as supplementary baselines. GPU_Opt, GPU_Opt_Pure, and GPU_Opt_Pure_Chunked are not 3 independently proposed methods; they are memory-management variants of the same batch-based GPU execution framework. GPU_Opt_Pure provides a device-only control using explicit `cudaMalloc` and `cudaMemcpy` and cannot handle a working set beyond HBM3 capacity. GPU_Opt_Pure_Chunked divides the working set into source sub-batches to extend the feasible batch range. All 3 implementations use the block kernel (see the sources in Table 5.4).

The status of the primary PathMerge baseline requires several qualifications. The evaluated PathMerge code is a third-party implementation of Galliot, a path-merging BC algorithm [@zheng2023galliot; @zheng2023jsac], obtained upstream from `gobardhanm/path-merging-bc` [@pathmergeRepo]. It was not confirmed as the original authors' official implementation. The upstream license was not independently verified because no explicit license notice was identified.

The vendored baseline in this study (`src/baseline/pathmerge.cu` plus `galliot*.cu`) adapts that upstream code and divides final BC values by 2 to follow the convention for undirected graphs.

PathMerge is an external comparator, not ground truth. The comparison applies only to the evaluated snapshot, configuration, environment, and graphs. It should not be generalized to PathMerge or Galliot as algorithms, or interpreted as a comparison with an official implementation by the original authors.

The supplementary baselines are as follows. Sequential (`src/baseline/sequential.cpp`) and OpenMP (`src/baseline/omp.cpp`) are serial and parallel CPU implementations of the Brandes algorithm [@brandes2001; @openmp52]. cuGraph (`src/baseline/cugraph_bc.cu`) invokes `betweenness_centrality` from RAPIDS cuGraph [@rapidsCugraph]. The code configures cuGraph for exact computation (`vertices=std::nullopt`, all-sources), without normalization (`normalized=false`) or endpoints (`include_endpoints=false`), and with undirected handling (`is_symmetric=true`). These settings align with the proposed implementation and PathMerge [@rapidsCugraphBcDocs]. However, the cuGraph adapter in this study does not explicitly divide the result by 2, and cuGraph's internal treatment of undirected symmetry was not independently confirmed in this environment. Its timing scope also includes the complete function, including initialization, transfers, BC computation, and retrieval, and therefore includes initialization overhead. For these reasons, cuGraph is a supplementary baseline limited to small graphs. It is not used for the main comparison or as ground truth for correctness.

Comparisons that include the supplementary Sequential, OpenMP, and cuGraph baselines are limited to small graphs. Sequential, OpenMP, and cuGraph measurements are unavailable for medium and large graphs (`result/main_performance/seven_implementations/README.md`). This study therefore does not present a unified 7-implementation comparison across all graphs using the current block implementation.

**Table 5.4: Compared implementations and their roles.**

| Implementation | Algorithm / basis | Memory strategy | Role in this study | Source |
|---|---|---|---|---|
| GPU_Opt | Proposed batch-based framework (block kernel) | Unified Memory (managed) | Main implementation | `src/proposed/host_um.cu` |
| PathMerge (tuned) | Galliot path-merging (third-party implementation) | Device (int2 frontier + per-source arrays) | Primary baseline / external comparator | `src/baseline/pathmerge.cu`, `galliot*.cu` |
| GPU_Opt_Pure | Proposed framework (block kernel) | Explicit cudaMalloc / cudaMemcpy | Device-only memory control | `src/proposed/host_pure.cu` |
| GPU_Opt_Pure_Chunked | Proposed framework (block kernel) | Chunked working set (sub-batch) | Capacity-extension variant | `src/proposed/host_chunked.cu` |
| Sequential | Brandes (CPU serial) | Host | Supplementary baseline (small only) | `src/baseline/sequential.cpp` |
| OpenMP | Brandes (CPU parallel) | Host | Supplementary baseline (small only) | `src/baseline/omp.cpp` |
| cuGraph | RAPIDS primitives (exact) | Managed (RAPIDS Memory Manager) | Supplementary baseline (small only) | `src/baseline/cugraph_bc.cu` |

## 5.5 Parameter Settings

This section specifies the execution settings. All implementations compute exact BC on undirected, unweighted graphs and align with the convention of dividing the final BC values by 2.

GPU_Opt, the main RQ1 implementation, used a fixed requested batch `b512`: a batch size of 512 source vertices per stream with the block kernel (1 block = 1 source). The batch quantity is defined for 1 stream, while `NS_eff` represents stream concurrency separately. Under in-capacity conditions, both the requested batch and effective batch were 512, with `SUB_BATCH=512` and `num_subs=1`. Dual-stream execution used two streams and `NS_eff=2` (`result/main_performance/proposed_variants/SOURCE.md`). These implementation-defined quantities are not collapsed into a single total batch. Importantly, the GPU_Opt results did not come from a search for the best-performing batch on each graph. No batch sweep was conducted for the proposed implementation; the same fixed `b512` was used on all 4 graphs.

PathMerge tuned, the RQ1 denominator, was selected by measuring candidate batches separately for each graph. The selected batches were `b2048` for email-EuAll, `b64` for roadNet-PA/TX, and `b32` for roadNet-CA (Section 5.7). GPU_Opt used fixed `b512` across all graphs, whereas PathMerge was tuned for each graph. The evaluation makes this asymmetry explicit.

The distinction between requested and effective batches is important. If a requested batch, specified by `BC_BATCH_OVERRIDE` or `PATHMERGE_BC_BATCH_SIZE`, exceeds the HBM3 budget, the effective batch is reduced. For the GPU_Opt variants, oversubscription dynamically reduces `SUB_BATCH`, producing `num_subs>1`. This evaluation distinguishes requested batch, effective batch, `SUB_BATCH`, `num_subs`, and `NS_eff`. When clamping occurs, the condition is described from the corresponding stderr or `execution_summary` record. Chapters 6 and 8 present the observed clamp values.

The proposed variants, kernel selection, PathMerge sweep, and small correctness experiments under SourceSnapshotID `phase_def_block_20260710` used no warmup and retained every trial. Ablation is the exception. In each of its 3 series, 1 global, untimed H1W1A1 warmup was executed at the start of every graph/trial runner invocation and excluded from formal TSV rows. The runner invokes the warmup outside the timing function, so it produces no TSV row on stdout. The former 4-synthetic-graph series retains 20 markers and 160 formal rows; the email-EuAll series retains 3 markers and 24 formal rows; and the corrected 325557 series retains 5 markers and 40 formal rows. The current synthetic-4 aggregate uses 120 rows for the 3 valid graphs from the former series and 40 rows from the corrected series. Runner code is byte-identical across the 3 series. If an explicit warmup record is unavailable for a legacy baseline, the value is `not_recorded`.

<!-- Source note (internal): ablation jobs 2354994, 2354999, and 2406254. -->

Performance measurements are distinguished from correctness-only runs. Performance measurements retain multiple trials for median aggregation, and their runtime values support performance claims. Correctness-only runs for small-graph correctness and memory-path correctness used `n=1` per configuration and no warmup; their runtimes are not used for performance evaluation. Runtime controls—`BC_BATCH_OVERRIDE`, `PATHMERGE_BC_BATCH_SIZE`, `CUGRAPH_BC_MAX_SOURCES_PER_BATCH`, and `BC_FORCE_BFS_KERNEL`—are part of the standard experimental procedure. They specify conditions deliberately in RQ3/RQ4 and the kernel comparison.

## 5.6 Timing and Statistical Method

The timing scope is the interval reported as `Time_sec` on stdout by the runner (`src/core/runner.cpp`) around each complete implementation function, including host control and kernel execution. Phase-level breakdowns such as BFS and Backward are reported separately on stderr. Warmup is excluded from formal trials.

Trial counts differ among experiment groups, as summarized in Table 5.1. For RQ1, GPU_Opt used `n=5` on email-EuAll and `n=3` on the road graphs, while PathMerge tuned used `n=3`. Ablation used `n=5` for each synthetic configuration and `n=3` for each email configuration; kernel selection used `n=3`. The targeted corrected 325557 boundary conditions and each correctness comparison used `n=1`, while profiling used a single trace. The `n=5` legacy capacity results on the former malformed input are historical records, not the current RQ3 trial count.

The median is the primary value. The mean, sample standard deviation, minimum, and maximum are supplementary values. The trial count is denoted by $N_{\mathrm{trials}}$ and distinguished from the vertex count $n=|V|$. Labels such as `n=3` and `n=5` in tables and figures conventionally indicate $N_{\mathrm{trials}}$. The sample standard deviation of runtime is denoted by $s_T$ and distinguished from a source vertex $s$. It is the unbiased estimator with ddof=1 defined by

$$
s_T = \sqrt{\frac{1}{N_{\mathrm{trials}}-1}\sum_{i=1}^{N_{\mathrm{trials}}}\left(t_i - \bar{t}\right)^2}
$$

where $t_i$ is the runtime of each trial and $\bar{t}$ is the sample mean. A single lowest-runtime trial is not used as the representative value.

Speedup is computed as a median-to-median comparison. For the median runtimes of the baseline and proposed implementation, $T^{\mathrm{med}}_{\mathrm{baseline}}$ and $T^{\mathrm{med}}_{\mathrm{proposed}}$, it is defined as follows. Because $S$ denotes the traversal stack in Chapter 2, this thesis uses $\mathrm{Speedup}$ for speedup.

$$
\mathrm{Speedup} = \frac{T^{\mathrm{med}}_{\mathrm{baseline}}}{T^{\mathrm{med}}_{\mathrm{proposed}}}
$$

The speedup calculation does not mix medians and means.

Throughput is expressed as GTEPS (Giga Traversed Edges Per Second). For the vertex count $n$, undirected edge count $m$, and runtime $T$ in seconds, every implementation uses

$$
\mathrm{GTEPS} = \frac{n \cdot m}{T \cdot 10^{9}}
$$

The treatment of OOM, TIMEOUT, and FAIL is important in this evaluation. They are not treated as 0 seconds, and an unavailable runtime is reported as `N/A`. Current feasibility records the CUDA device-memory OOM for Pure `b8192` separately from the cgroup host-memory OOM kill for UM `b12288`. The label `OOM_OR_FAIL` is limited to the historical archive for the former malformed input. This convention prevents infeasible conditions from being incorrectly aggregated as low runtimes.

## 5.7 Performance Comparison and PathMerge Tuning Procedure

This section describes the RQ1 performance procedure and the tuning procedure for PathMerge tuned, which forms the denominator. Chapter 6 reports the speedup values. The main comparison evaluates the proposed block-based GPU_Opt against PathMerge tuned on 4 graphs: email-EuAll, roadNet-PA, roadNet-TX, and roadNet-CA. GPU_Opt, the numerator, uses fixed `b512`, median aggregation, no warmup, and measurements under SourceSnapshotID `phase_def_block_20260710`. PathMerge tuned, the denominator, uses median aggregation with a graph-wise tuned batch. Speedup and GTEPS are calculated from the two median runtimes using the equations in Section 5.6.

The PathMerge tuning procedure consists of sweep-based screening followed by selection of the tuned value. Candidate batches were swept for each graph: `b8`–`b512` for roadNet-PA, `b32`–`b128` for roadNet-TX, `b16`–`b128` for roadNet-CA, and `b8`–`b8192` for email-EuAll. The batch with the lowest median runtime was selected as tuned. Sweep trial counts were uneven, ranging from 1 to 4 per batch, and each graph's `SOURCE.md` records the count for every batch. The sweeps used no warmup and median aggregation. `b2048` for email-EuAll and `b32` for roadNet-CA were selected as the best observed values in their sweeps.

The denominators for roadNet-PA/TX require an additional measurement qualification. The sweeps confirmed that the selected batch for both graphs was the default `b64`. The final denominator nevertheless uses retained legacy baseline measurements with the same `b64` setting from `result/main_performance/seven_implementations/legacy_partial/` at checkpoint `oldtree_f05ec52_20260512` as a conservative reference. These legacy measurements are slightly faster than the sweep confirmation measurements, which makes the reported speedups conservative, or smaller, for both graphs. Thus, the PathMerge denominators for roadNet-PA/TX come from a legacy checkpoint, whereas the proposed numerators for all 4 graphs come from `phase_def_block_20260710`. This evaluation states the checkpoint difference explicitly.

The speedup against the tuned reference is distinguished from the comparison against default `b64`, and the tuned reference is central to this study's claim. Chapter 6 compares the numerical values under both references. As noted previously, the small-graph comparison that includes Sequential, OpenMP, and cuGraph is supplementary. Its proposed implementations were measured with the former shared kernel, so it is not a unified 7-implementation comparison using the current block implementation.

## 5.8 Ablation and Kernel Analysis Method

The RQ2 factor analysis consists of 2 procedures: ablation and kernel selection.

The ablation measures the contributions of 3 features: Hybrid BFS (H), a bidirectional top-down/bottom-up BFS [@beamer2012]; Warp-Cooperative Accumulation (W), warp-cooperative accumulation in the Backward phase; and Dual-Stream Execution (A), which overlaps asynchronous initialization and computation with 2 streams. Compile-time templates enabled or disabled these features to form 8 configurations, $\mathrm{H}\{0,1\}\mathrm{W}\{0,1\}\mathrm{A}\{0,1\}$. They were measured with fixed `b512` and median aggregation. Each configuration used `n=5` on benchmark_7000_41459, benchmark_11023_62184, 56438_300801, and corrected 325557, and `n=3` on email-EuAll. The synthetic-4 aggregate is mixed-checkpoint; it is not a remeasurement of all 4 graphs under one checkpoint. The initial global, untimed H1W1A1 warmup in each 8-configuration set is excluded from the formal trials. Appendix A identifies the measurement series.

Each feature's contribution is evaluated as a main effect. For a feature $F$, the main effect is the geometric mean of the runtime ratio $T(F{=}0)/T(F{=}1)$, averaged over every level of the other 2 axes. The synthetic-graph main effect is summarized by the geometric mean across 4 graphs; email-EuAll is treated separately. The factor-analysis main effects should not be generalized to unmeasured graphs such as roadNet. In particular, Warp-Cooperative Accumulation is graph-dependent and may be neutral or slightly unfavorable on email-EuAll; this behavior is treated as a scope condition. Chapter 7 presents the numerical main effects.

Kernel selection directly compares the BFS kernel choices. `BC_FORCE_BFS_KERNEL=shared|block` forces the shared-frontier or block kernel (1 block = 1 source), which were compared on roadNet-PA and roadNet-TX. The settings were batch 512, `SUB_BATCH=512`, `num_subs=1`, `n=3`, no warmup, and median aggregation with sample standard deviation (`result/tuning/kernel_selection/SOURCE.md`). The comparison uses only the forced shared/block measurements and does not depend on an automatic selection rule. A former implementation had a mean-degree rule (`avg_deg < 5 → shared`), but the current method does not use it. Consequently, the kernel-selection conclusion is limited to the forced roadNet-PA/TX comparison and should not be generalized to other graphs.

Profiling uses an `ablation_H1W1A0` nsys trace from the ablation binary on 56438_300801 to obtain the runtime proportions of the BFS and Backward kernels. This single trace also includes the untimed H1W1A1 warmup at the start of the same process. Therefore, 63.9% / 36.1% describes the composition of CUDA GPU kernel time including warmup, not an isolated formal measurement. The same measurement series contains a separate `ablation_H1W1A1` trace and a separate GPU_Opt UM-prefetch trace, but neither contributes to the 63.9% / 36.1% proportions. This evaluation does not infer causality from the phase breakdown and limits the observed distribution to this 1 graph and 1 trace.

<!-- Source note (internal): raw_data/profiling/job_2359175_20260711/ablation_H1W1A0.stats.txt; PBS job 2359175. -->

## 5.9 Memory Scalability Protocol

The RQ3 memory-scalability evaluation is limited to 1 corrected graph, `325557_3216152_corrected_v1`. The input file itself is 45.35 MB and does not exceed HBM3 capacity. Increasing the batch enlarges the `batch × per-source state` working set. The evaluation compares the feasibility of GPU_Opt (Unified Memory), GPU_Opt_Pure (device-only), and GPU_Opt_Pure_Chunked (source sub-batching).

The target is capacity feasibility, not an execution-time performance comparison. The targeted boundary validation ran Pure `b4096`/`b8192`, UM `b10240`/`b12288`, and Chunked `b16384` with `n=1` per condition. OOM and kill events are not treated as 0 seconds. The CUDA device-memory OOM at Pure `b8192` is distinguished from the cgroup host-memory OOM kill at UM `b12288`, which exited with 137. The runtime per-configuration classifier inspected only that configuration's stdout and stderr and recorded `oom_evidence=none` with the runner exit. After the classifier's scope ended, the retained PBS epilogue appended direct evidence of a cgroup OOM kill. The two records have different observation scopes and are retained as separate evidence layers; the UM failure is not described as a CUDA or HBM OOM. Single-run runtimes are not used for a formal performance comparison among the memory approaches. Appendix A identifies the runs.

The legacy sweep and `CORE_FAIL` results for the former malformed input remain retained as historical evidence but are not used for the current RQ3 boundary. The current conclusion is based only on formal validation of the corrected input. GPU_Opt, Pure, and Chunked are memory-management variants of one common GPU execution framework, not 3 independent methods. UM uses managed allocation and migration to handle a working set that may exceed device memory; it does not partition the input graph for storage. Chunked divides a source batch into sub-batches to limit the concurrently resident working set.

<!-- Source note (internal): historical job 2368587; corrected formal job 2404743; checkpoint 45352a3. The two-layer UM evidence is documented in the corrected-graph README under result/memory_scalability/; the runtime classifier record and retained PBS epilogue are preserved in the canonical corrected-graph archive. -->

This evaluation does not claim that Unified Memory extends capacity without limit. It also does not claim that Chunked avoids OOM under unmeasured conditions. The principal benefit of Chunked is control of the resident working set and capacity extension within the tested range, not peak performance. This conclusion is limited to GH200, the corrected 325557 graph, and `n=1` per condition.

## 5.10 Correctness Validation

RQ4 correctness validation uses distinct evidence levels. Agreement only in the maximum BC index and value is not treated as a complete correctness demonstration. Table 5.5 therefore defines the validation levels and identifies the level of each comparison.

**Table 5.5: Correctness validation levels.**

| Evidence Tier | Definition | Applied to |
|---|---|---|
| Tier A: Independent CPU reference | All BC elements compared against an independent Sequential implementation | benchmark_7000_41459, benchmark_11023_62184, chain_200 |
| Tier B: Cross-implementation consistency | All BC elements compared across batches and memory paths on the corrected graph | corrected 325557: 6 vectors, 10 comparisons |
| Supplementary max_bc_only | Only the maximum BC index/value compared | Headline performance and kernel-selection runs |
| None | No BC comparison recorded | Ablation runs |

<!-- Source note (internal): result/tables/thesis/T5_correctness_summary.tsv and result/CLAIMS.md. -->

The canonical T5 evaluation comprises 2 tiers. Tier A contains 3 full-vector comparisons on 3 small graphs, using Sequential as the independent reference and GPU_Opt as the candidate. Tier B contains 10 cross-implementation comparisons among 6 UM, Pure, Chunked, and PathMerge vectors for the corrected 325557 graph. Tier B is cross-implementation consistency, not an independent reference or ground truth, and PathMerge remains an external comparator. Across the 13 comparisons in both tiers, `MissingIndices=0`, `MismatchedElements=0`, `ToleranceResult=PASS`, and `ByteIdentical=No`.

The recorded correctness metrics extend beyond the maximum BC index and value. They include vector length, the number of missing indices and mismatched elements, maximum absolute and relative errors, reference and candidate values at the corresponding indices, the presence of NaN or Inf, and SHA256 values for the vectors and input graph. These metrics record element-level differences that agreement in the maximum BC alone cannot capture.

The decision uses a mixed absolute-relative tolerance criterion. Every index must satisfy

$$
\lvert r_i - c_i \rvert \le \mathrm{abs\_tol} + \mathrm{rel\_tol} \cdot \max\!\left(\lvert r_i \rvert, \lvert c_i \rvert\right)
$$

where $r_i$ is the reference value and $c_i$ is the candidate value. The canonical tolerances are $\mathrm{abs\_tol}=1\mathrm{e}{-3}$ and $\mathrm{rel\_tol}=1\mathrm{e}{-6}$ (`result/correctness/small_full_vector/correctness_summary.tsv`; `result/correctness/memory_paths/README.md`). When BC values are as large as $\sim 10^{10}$, an absolute tolerance alone is inappropriate. Exceeding only the absolute tolerance is therefore recorded separately as WARN and does not by itself constitute failure. The tolerances are not modified after the comparison to change a decision to PASS.

The fact that the canonical series for the former malformed input produced `CORE_FAIL` is retained as a historical invalid-input result. That historical classification is neither relabeled as PASS nor removed. The current active conclusion instead comes from the formal series on the corrected input, in which all 10 comparisons passed the mixed tolerance criterion. This result does not establish bitwise identity or correctness against a large-scale independent reference. The corrected graph is internally reconstructed data, and its original generation seed and complete upstream original remain unconfirmed provenance limitations.

## 5.11 Reproducibility and Data Provenance

The archive is organized so that it can be verified from one retained tree without relying only on Git history. Raw data, experiment-time code for each SourceSnapshotID, derived tables, figures, and summaries, and unsuccessful data are separated by role.

The canonical provenance reference is the SourceSnapshotID, which identifies an experiment-time code snapshot, rather than a raw commit SHA. Manifests and internal Source notes record the content, generation conditions, checksums, and experiment identifiers for each raw dataset. Tables and figures can be regenerated from canonical raw inputs, and primary values are recomputed from the source TSV without alteration.

Failed results are retained and classified. Unsuccessful data, including OOM, intentional early termination, and fail-fast comparison mismatches, are separated from summaries and preserved without content changes. SHA256 checksums verify the retained contents.

The full set of derived results does not correspond to a single SourceSnapshotID. The main performance experiments use SourceSnapshotID `phase_def_block_20260710`, and small-graph correctness uses `small_correctness_20260712`. Correctness, feasibility, and ablation on the corrected 325557 graph were measured at a separate formal checkpoint. Because the ablation on the other 3 synthetic graphs and the corrected 325557 graph comes from different measurement series, the synthetic-4 aggregate is mixed-checkpoint. Legacy results for the former malformed input are separated as a historical archive. Text within figures, including axis labels, legends, column names, and captions, is standardized in English.

<!-- Source note (internal): raw_data/MANIFEST.tsv; raw_data/SHA256SUMS; code_snapshots/<SourceSnapshotID>/; result/TABLES_AND_FIGURES.md; failure/; raw_data/unsuccessful/. Corrected jobs 2404743/2406254 at checkpoint 45352a3; unchanged three-graph ablation job 2354994. -->

## 5.12 Scope and Methodological Limitations

The methodological limitations described throughout this chapter are summarized below. Chapter 10 discusses their implications for threats to validity in detail.

- Scope: The central performance claim is limited to 4 graphs. Memory scalability and Tier B are limited to 1 corrected 325557 graph, and full-vector correctness against an independent reference is limited to 3 small graphs. None of these results should be generalized to other graphs or GPUs.
- Comparator: PathMerge is a third-party implementation and an external comparator, not ground truth. Results should not be generalized to PathMerge or Galliot in general or to an official implementation by the original authors. No unified 7-implementation comparison across all graphs was conducted with the current block implementation.
- Configuration asymmetry: GPU_Opt uses fixed `b512` on all graphs, whereas PathMerge is tuned separately for each graph. Their batch configurations are asymmetric.
- Evidence limitations: Tier B demonstrates cross-implementation consistency within the mixed tolerance criterion; it is neither byte-identical nor an independent ground-truth comparison. Physical HBM residency, process RSS, and migration bytes were not measured.
- Provenance: The corrected 325557 graph is internally reconstructed data, and its original seed and upstream original remain unconfirmed. The `CORE_FAIL` result for the former malformed input is retained as historical evidence and is not conflated with the current conclusion.
- Environment-record limitation: The retained experimental documents and submission scripts disagree on the queue name, and the actual queue could not be determined independently from retained job logs. The queue name is therefore neither treated as a controlled variable nor interpreted as a cause of an observed performance difference. Resource information such as group, GPU, CPU, and memory is reported only to the extent supported by canonical records.

Subject to these limitations, the conclusions of this study are limited to the evaluated environment, graphs, and configurations. The following chapters present performance (Chapter 6), factor analysis (Chapter 7), memory scalability (Chapter 8), and correctness and numerical behavior (Chapter 9) under this methodology.
