# Chapter 6 Performance Evaluation

This chapter answers RQ1 on performance: On the four evaluated graphs, is the block-based GPU_Opt implementation with a fixed batch size of 512 faster than the graph-wise tuned third-party PathMerge implementation? (Section 5.1). For this 4-graph scope, the evaluation methodology, including the computing environment, graphs, execution settings, timing and statistical procedures, and tuning procedure, is defined in Chapter 5. This chapter introduces no new experimental conditions.

The main comparison is between GPU_Opt, the principal implementation of the proposed batch-based GPU execution framework, and tuned PathMerge. GPU_Opt uses Unified Memory, always uses the block kernel, and uses fixed `b512` on every graph. PathMerge uses a graph-wise tuned batch size. The scope is limited to 4 graphs: email-EuAll, roadNet-PA, roadNet-TX, and roadNet-CA. The reported metrics are median runtime, GTEPS, and speedup. All aggregations follow the definitions in Section 5.6: the median is the primary value, speedup is a median-to-median comparison, and the same GTEPS equation is used for every implementation.

The PathMerge comparator in this chapter is a third-party implementation of Galliot, a path-merging BC algorithm [@zheng2023galliot; @zheng2023jsac]. The evaluated snapshot of the upstream `gobardhanm/path-merging-bc` repository [@pathmergeRepo] was adapted and measured as described in Section 5.4. This implementation was not confirmed as the original authors' official implementation, and its upstream license was not independently verified. It is an external comparator, not an independent reference or ground truth. Therefore, the results in this chapter are limited to the evaluated snapshot, configuration, environment, and 4 graphs. They do not establish superiority over PathMerge or Galliot as algorithms or over an official implementation by the original authors.

<!-- Source note (internal): evaluated upstream commit 9c231b46f7499380d4495262c1ec75a11cdaae7a; see references.bib:pathmergeRepo and SOURCE_AUDIT.tsv:S08. -->

This chapter is limited to describing observed performance. Chapter 7 evaluates the contributions of Hybrid BFS, Warp-Cooperative Accumulation, Dual-Stream Execution, and the block kernel. Chapter 8 examines the capacity characteristics of the memory-management approaches, Chapter 9 evaluates numerical agreement among BC outputs, and Chapter 10 provides an integrated discussion of the associated factors. GPU_Opt_Pure and GPU_Opt_Pure_Chunked are memory-management variants within the same execution framework rather than independent proposals (Section 5.4), so they are evaluated primarily in Chapter 8.

## 6.1 Main Runtime Comparison

Table 6.1 presents the main performance results, and Figure 6.1 compares the median runtimes. The GPU_Opt measurements, which form the numerator, used fixed `b512` under SourceSnapshotID `phase_def_block_20260710`: a batch-size value of 512 for 1 stream, in-capacity execution, requested and effective batches of 512, `SUB_BATCH=512`, `num_subs=1`, and `NS_eff=2`. The tuned PathMerge measurements, which form the denominator, used the graph-wise batches selected by the procedure in Section 5.7: `b2048` for email-EuAll, `b64` for roadNet-PA/TX, and `b32` for roadNet-CA. GPU_Opt used `n=5` on email-EuAll and `n=3` on each roadNet graph, while PathMerge used `n=3` on each graph. No warmup was performed (Sections 5.5 and 5.6).

**Table 6.1: Main performance results on the four evaluated graphs. Times are medians over the recorded trials; Speedup = tuned-PathMerge median / GPU_Opt median; GTEPS = Nodes × Edges / Time_sec / 10^9 computed from the median time.**

| Graph | Nodes | Edges | GPU_Opt Batch | GPU_Opt Trials | GPU_Opt Median Time [s] | GPU_Opt GTEPS | PathMerge Tuned Batch | PathMerge Trials | PathMerge Median Time [s] | PathMerge GTEPS | Speedup |
|---|--:|--:|:--:|--:|--:|--:|:--:|--:|--:|--:|--:|
| email-EuAll | 265009 | 364481 | b512 | 5 | 30.81 | 3.14 | b2048 | 3 | 97.80 | 0.99 | 3.17 |
| roadNet-PA | 1088092 | 1541898 | b512 | 3 | 699.52 | 2.40 | b64 | 3 | 918.67 | 1.83 | 1.31 |
| roadNet-TX | 1379917 | 1921660 | b512 | 3 | 980.13 | 2.71 | b64 | 3 | 1482.68 | 1.79 | 1.51 |
| roadNet-CA | 1965206 | 2766607 | b512 | 3 | 2129.10 | 2.55 | b32 | 3 | 3079.72 | 1.77 | 1.45 |

<!-- canonical artifact: T2_main_performance (internal ID: T2); Nodes/Edges from T1_graph_metadata -->
<!-- Source note (internal): result/tables/thesis/T2_main_performance.tsv; GPU_Opt raw_data/main_performance/proposed_variants/<graph>/_run/job_2357334_20260711/results.tsv; PathMerge tuning and retained legacy baseline TSVs. GPU_Opt is fixed at 512 sources per stream with NS_eff=2. -->

GPU_Opt had a shorter median runtime than tuned PathMerge on each of the 4 evaluated graphs. On email-EuAll, the runtimes were 30.81 s and 97.80 s, respectively, which was the largest difference. On roadNet-PA/TX/CA, the corresponding runtimes were 699.52 s versus 918.67 s, 980.13 s versus 1482.68 s, and 2129.10 s versus 3079.72 s; GPU_Opt was shorter in each comparison.

Figure 6.1 compares the same median runtimes in absolute terms. Because the runtimes of the 4 graphs span approximately 2 orders of magnitude, from approximately 31 s to approximately 3080 s, the y-axis uses a logarithmic scale and is measured in seconds. Although the email-EuAll values are more than 1 order of magnitude smaller than those of the roadNet graphs, the logarithmic scale keeps every bar legible. Annotations above the bars identify the selected tuned PathMerge batches, and the error bars show the sample standard deviation.

![Figure 6.1: Main runtime comparison](../../../../result/figures/thesis/main_runtime_comparison.png)

**Figure 6.1: Median runtime of GPU_Opt (fixed b512) and tuned PathMerge on the four evaluated graphs. Bars show the median of per-trial runtimes; error bars show the sample standard deviation; annotations give the tuned PathMerge batch. The y-axis is logarithmic because the values span about two orders of magnitude.**

<!-- canonical artifact: main_runtime_comparison.{png,pdf,svg} (internal ID: F1); see result/figures/thesis/FIGURE_MANIFEST.tsv -->

The numbers of nodes and edges increase in order across the 3 roadNet graphs (Table 6.1). Within this evaluated range, the observed advantage of GPU_Opt remained present as graph size increased. This observation is limited to the 3 evaluated roadNet graphs and is not generalized to larger graphs or graphs with other structures.

Table 6.2 reports the variation across trials. The median is the primary value, while the mean, sample standard deviation with `ddof=1`, minimum, and maximum are supplementary values (Section 5.6). The sample standard deviation ranged from approximately 0.2% to approximately 1.7% of the median across the configurations; the largest percentage was observed for PathMerge on roadNet-TX. On each of the 4 graphs, the maximum GPU_Opt trial runtime was shorter than the minimum tuned PathMerge trial runtime, so the trial-level runtime ranges did not overlap. However, the small number of trials, `n=3` or `n=5` per configuration, limits statistical inference. No significance test was conducted, and this study makes no claim of statistical significance. It also does not use the single fastest trial as the representative value.

**Table 6.2: Trial-level runtime statistics of the main comparison. All times are in seconds; SD denotes the sample standard deviation (ddof=1).**

| Implementation (Batch) | Graph | n | Median [s] | Mean [s] | Sample Standard Deviation [s] | Min [s] | Max [s] |
|---|---|--:|--:|--:|--:|--:|--:|
| GPU_Opt (b512) | email-EuAll | 5 | 30.81 | 30.81 | 0.061 | 30.75 | 30.91 |
| GPU_Opt (b512) | roadNet-PA | 3 | 699.52 | 700.49 | 1.695 | 699.50 | 702.45 |
| GPU_Opt (b512) | roadNet-TX | 3 | 980.13 | 983.66 | 6.352 | 979.85 | 990.99 |
| GPU_Opt (b512) | roadNet-CA | 3 | 2129.10 | 2127.38 | 4.021 | 2122.78 | 2130.25 |
| PathMerge tuned (b2048) | email-EuAll | 3 | 97.80 | 97.90 | 0.988 | 96.96 | 98.93 |
| PathMerge tuned (b64) | roadNet-PA | 3 | 918.67 | 923.26 | 9.593 | 916.82 | 934.28 |
| PathMerge tuned (b64) | roadNet-TX | 3 | 1482.68 | 1493.46 | 24.855 | 1475.81 | 1521.88 |
| PathMerge tuned (b32) | roadNet-CA | 3 | 3079.72 | 3083.85 | 25.511 | 3060.66 | 3111.18 |

<!-- provenance: recomputed from the same canonical raw TSVs as Table 6.1 (median/mean/SD cross-checked against docs/thesis/thesis_values.tsv) -->
<!-- Source note (internal): computed from per-trial Time_sec values in the canonical raw TSVs listed for Table 6.1; cross-checked with docs/thesis/thesis_values.tsv. -->

## 6.2 Speedup over Tuned PathMerge

Following the definition in Section 5.6, speedup is the ratio of the tuned PathMerge median runtime to the GPU_Opt median runtime, a median-to-median comparison. The calculation does not mix means and medians. Before rounding, the values were approximately 3.1743 for email-EuAll, 1.3133 for roadNet-PA, 1.5127 for roadNet-TX, and 1.4465 for roadNet-CA. The text, tables, and figures display these values to 2 decimal places.

![Figure 6.2: Speedup over tuned PathMerge](../../../../result/figures/thesis/main_speedup_over_tuned_pathmerge.png)

**Figure 6.2: Speedup of GPU_Opt (fixed b512) over tuned PathMerge on the four evaluated graphs (median/median). The dashed line marks parity (1.0x) with tuned PathMerge.**

<!-- canonical artifact: main_speedup_over_tuned_pathmerge.{png,pdf,svg} (internal ID: F2) -->

As Figure 6.2 shows, the speedup exceeded 1.0 on each of the 4 evaluated graphs and ranged from 1.31× to 3.17×. The largest observed speedup was 3.17× on email-EuAll. The speedups on the 3 roadNet graphs were smaller than on email-EuAll: 1.31× for PA, 1.51× for TX, and 1.45× for CA.

The batch settings are asymmetric (Sections 5.5 and 5.12). PathMerge, the denominator, was allowed graph-wise tuning as described in Section 6.3. In contrast, GPU_Opt, the numerator, used fixed `b512` on all 4 graphs and was not subjected to a graph-wise search for the fastest batch. The reported speedups therefore do not include a benefit from graph-wise tuning for GPU_Opt. Performance with other GPU_Opt batch settings was not measured and is not discussed in this chapter. Providing graph-wise tuning to the comparator establishes a conservative comparison against the best observed PathMerge setting under the stated adoption procedure.

The claim that the observed speedup ranged from 1.31× to 3.17× is limited to the 4 evaluated graphs and the tuned third-party PathMerge implementation evaluated on the GH200 system in this study. It does not claim the same speedup or advantage for unevaluated graphs, other PathMerge implementations, or other computing environments.

## 6.3 PathMerge Batch-Size Sensitivity

This section presents the batch-size sensitivity of PathMerge, which forms the denominator of the comparison, and distinguishes its tuned and default settings. The tuning procedure is defined in Section 5.7: screening followed by confirmation over `b8`–`b512` for roadNet-PA, `b32`–`b128` for roadNet-TX, `b16`–`b128` for roadNet-CA, and `b8`–`b8192` for email-EuAll, with `n=1`–`4` trials per batch. Appendix B reports every value in the sweep. Single-trial screening was used only to locate candidate ranges and was not used by itself as a formal performance conclusion. The tuned batch was adopted from confirmed measurements using the lowest observed median, subject to the same-configuration conservative selection for roadNet-PA/TX described below.

![Figure 6.3: PathMerge batch-size sweep](../../../../result/figures/thesis/pathmerge_batch_sweep.png)

**Figure 6.3: PathMerge batch-size sweep: median runtime versus requested batch size (log2 axis) per graph. Circled markers denote the tuned batch used in the main comparison; marker styles encode the number of recorded trials per batch (n=1 screening, n=2, n>=3 confirmation with sample-standard-deviation error bars; see the in-figure legend); squares annotate recorded clamping of the effective batch.**

<!-- canonical artifact: pathmerge_batch_sweep.{png,pdf,svg} (internal ID: F3) -->

As Figure 6.3 shows, the PathMerge median runtime depended on the requested batch size, and the best observed batch differed by graph. The best observed sweep values were `b2048` for email-EuAll at 97.80 s, `b64` for roadNet-PA/TX, and `b32` for roadNet-CA at 3079.72 s. On roadNet-CA, `b32` was better than the PA/TX selection; the `b64` sweep median was 3491.64 s with `n=3`. Thus, the best observed batch for PA/TX did not transfer directly to CA. The email-EuAll panel shows sweep points from `b512` through `b8192`. Smaller requested batches from `b8` through `b256`, each with `n=1` screening, recorded substantially longer runtimes, including 226.0 s at `b64` (Appendix B).

The records distinguish requested batch from effective batch (Section 5.5). Two clamps were recorded. On email-EuAll, requested `b8192` exceeded the HBM3 budget and was reduced to an effective batch of 7393. On the historical `325557_3216152` graph, requested `b8192` was reduced to an effective batch of 6018. Both observations are based on warning lines in the retained logs and are annotated in Figure 6.3. The `325557_3216152` panel in that figure is not part of the RQ1 main performance comparison and must not be mixed into the RQ1 conclusion. Its best observed sweep batch was `b4096`, which is used as the PathMerge external-comparator setting in the memory-path comparison in Chapter 9. An observed clamp is not an OOM result, and OOM, FAIL, and unavailable trials are not treated as zero seconds.

The roadNet-PA/TX denominators require the measurement qualification described in Section 5.7. The sweeps confirmed that the best batch for both graphs was the default `b64`; the confirmation medians were 941.39 s with `n=4` for PA and 1491.13 s with `n=3` for TX. The final denominator used the retained legacy baseline measurement with the same `b64` setting: 918.67 s for PA and 1482.68 s for TX at checkpoint `oldtree_f05ec52_20260512`. The sweep-confirmation and adopted values are separate measurements of the same setting, not missing or contradictory results. Adopting the faster of these measurements for the comparator gives PathMerge the favorable condition and makes the resulting speedup smaller for PA/TX. The numerator for these 2 graphs was measured at `phase_def_block_20260710`, whereas the denominator was measured at `oldtree_f05ec52_20260512`, as stated in Section 5.7.

Table 6.3 distinguishes the default and tuned settings. The central claim in this study uses only the 1.31×–3.17× speedup over tuned PathMerge; the ratio against default `b64` is supplementary. For email-EuAll, the speedup was 7.15× against the default and 3.17× against the tuned setting. For roadNet-CA, it was 1.64× against the default and 1.45× against the tuned setting. The ratios are identical for roadNet-PA/TX because their tuned and default batches are both `b64`. The default-comparison ratios are not used as the headline values.

**Table 6.3: Default (b64) and tuned PathMerge medians and the corresponding GPU_Opt speedups. The headline claim uses the tuned column only.**

| Graph | PathMerge Default b64 Median [s] | Speedup vs Default | Tuned Batch | PathMerge Tuned Median [s] | Speedup vs Tuned (headline) |
|---|--:|--:|:--:|--:|--:|
| email-EuAll | 220.39 | 7.15 | b2048 | 97.80 | 3.17 |
| roadNet-PA | 918.67 | 1.31 | b64 | 918.67 | 1.31 |
| roadNet-TX | 1482.68 | 1.51 | b64 | 1482.68 | 1.51 |
| roadNet-CA | 3499.03 | 1.64 | b32 | 3079.72 | 1.45 |

<!-- provenance: default/tuned separation from result/tables/final_speedup_tables.md (merge_final_tables.py); default n: email-EuAll 5, roadNet-PA/TX/CA 3 -->
<!-- Source note (internal): result/tables/final_speedup_tables.md; default b64 legacy baseline TSVs under raw_data/main_performance/seven_implementations/legacy_partial/; tuned values as in Table 6.1. -->

GPU_Opt was not subjected to a graph-wise batch sweep like the one conducted for PathMerge. Only fixed `b512` was measured on all 4 graphs (Section 5.5), and this asymmetry is part of the comparison conditions described in Section 6.2. Chapter 9, Section 9.3, presents the full-vector validation showing that the tuned and default `b64` PathMerge batches produced the same BC vectors for email-EuAll (`b64` versus `b2048`) and roadNet-CA (`b32` versus `b64`).

## 6.4 Throughput Analysis

Throughput is reported using the GTEPS definition established in Section 5.6: $\mathrm{GTEPS} = n \cdot m / (T \cdot 10^{9})$. Here, $n$ and $m$ are the numbers of nodes and undirected edges in Table 6.1, and $T$ is the median runtime. GPU_Opt and PathMerge use the same graph sizes, the same timing scope—the complete implementation function measured by the runner (Section 5.6)—and the same equation. All GTEPS values are displayed to 2 decimal places.

As shown in the GTEPS columns of Table 6.1, GPU_Opt ranged from 2.40 GTEPS on roadNet-PA to 3.14 GTEPS on email-EuAll. Tuned PathMerge ranged from 0.99 GTEPS on email-EuAll to 1.83 GTEPS on roadNet-PA. GPU_Opt had the higher derived throughput on each of the 4 evaluated graphs. GTEPS normalizes runtime by the quantity $n \cdot m$, so on the same graph, a higher GTEPS value indicates a shorter runtime per unit of this derived work quantity. On the same graph, the unrounded GTEPS ratio is equal by definition to the speedup. Because the displayed GTEPS values in Table 6.1 are rounded to 2 decimal places, ratios calculated from those displayed values can differ from the speedup column within the rounding error.

GTEPS is a derived throughput metric and has limited value for comparisons across graphs. For example, roadNet-CA had the longest runtime, yet its GPU_Opt GTEPS of 2.55 exceeded the roadNet-PA value of 2.40. Thus, the ordering of GTEPS does not directly determine the ordering of runtimes. GTEPS also does not guarantee that the implementations perform identical algorithmic work. It is a supplementary interpretation of the runtime results based on $n \cdot m$, not a measurement of hardware memory bandwidth or effective bandwidth. This chapter does not attribute differences across graphs to degree distribution, BFS depth, or other mechanisms; their integrated interpretation is deferred to Chapter 10. Separate effective-bandwidth measurements for the memory paths are retained as supplementary environment data and are not used to explain the GTEPS values in this chapter.

## 6.5 Supplementary Baseline Results

The multi-implementation comparison that includes the supplementary Sequential, OpenMP, and cuGraph [@rapidsCugraph] baselines is a legacy partial dataset limited to small graphs, as described in Section 5.4. This section reports its complete values and limitations. It differs from the main performance comparison in 2 respects. (1) No Sequential, OpenMP, or cuGraph measurements exist for medium or large graphs, so a unified 7-implementation comparison across all graphs using the current block implementation cannot be presented. (2) The proposed-family implementation in this legacy dataset used the former shared kernel. Its execution path therefore differs from the current block implementation in Table 6.1, and these values are not used for the headline claim.

**Table 6.4: Supplementary legacy small-graph baseline comparison (mean ± sample standard deviation over n=10 trials; legacy shared-kernel measurements from the old tree). `N/A` denotes a measurement that does not exist; it is not zero.**

| Graph | Sequential [s] | OpenMP [s] | cuGraph_BC [s] | GPU_Opt_Pure (legacy shared) [s] | PathMerge_BC [s] |
|---|--:|--:|--:|--:|--:|
| benchmark_7000_41459 | 5.63 ± 0.05 | 0.10 ± 0.00 | 1.12 ± 0.04 | 0.22 ± 0.01 | 0.41 ± 0.02 |
| benchmark_11023_62184 | 11.39 ± 0.08 | 0.16 ± 0.02 | 2.81 ± 0.08 | 0.25 ± 0.01 | 1.02 ± 0.03 |
| random (32212/101805) | 171.17 ± 2.60 | 3.36 ± 0.03 | 17.23 ± 0.43 | 0.90 ± 0.01 | 0.90 ± 0.01 |
| 56438_300801 | N/A | 13.35 ± 0.13 | 71.85 ± 1.36 | 2.02 ± 0.01 | 4.64 ± 0.01 |

<!-- provenance: result/main_performance/seven_implementations/legacy_partial/small/statistical_test_no_gpu_opt.md (mean±SD, n=10); raw: raw_data/main_performance/seven_implementations/legacy_partial/small/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv -->
<!-- Source note (internal): result/main_performance/seven_implementations/legacy_partial/small/statistical_test_no_gpu_opt.md. Aggregation is mean ± sample standard deviation (n=10); GPU_Opt_Pure is the legacy shared-kernel measurement; Sequential on 56438_300801 was not measured. -->

The following observations are limited to this supplementary comparison. Sequential required approximately 1–2 orders of magnitude more time than the GPU implementations on the 3 graphs for which a measurement was recorded. OpenMP was faster than the GPU implementations—the legacy shared proposed implementation and PathMerge—on the 2 smallest graphs, benchmark_7000_41459 and benchmark_11023_62184, but slower on random and 56438_300801. cuGraph was slower than the legacy shared proposed implementation and PathMerge on each of the 4 evaluated small graphs. The cuGraph timing scope, however, is the complete function and includes initialization (Section 5.4).

This table uses the mean ± sample standard deviation over `n=10` trials, unlike the median aggregation used for the headline comparison, so it is not combined directly with Table 6.1. It is a supplementary legacy small-graph result, not a main headline baseline. `N/A` is not interpolated or treated as zero seconds, and the table is not extrapolated to relationships among implementations on medium or large graphs.

## 6.6 Answer to RQ1

The answer to RQ1 is limited to the evaluated email-EuAll and roadNet-PA, roadNet-TX, and roadNet-CA graphs under the GH200 evaluation configuration. Within this scope, the block-based GPU_Opt implementation with a fixed batch size of 512 per stream was 1.31×–3.17× faster than the graph-wise tuned third-party PathMerge implementation, based on median-to-median runtime comparisons. The largest observed speedup was 3.17× on email-EuAll, while the speedups on the 3 roadNet graphs ranged from 1.31× to 1.51×. GPU_Opt also had higher GTEPS under the same graph sizes and equation on each of the 4 evaluated graphs.

<!-- English version (plan.md 8.7): "On the four evaluated graphs, the fixed-batch block-based GPU_Opt implementation was 1.31x to 3.17x faster than the tuned third-party PathMerge implementation evaluated in this study." -->

This answer is subject to the following limitations.

- The scope is limited to the 4 evaluated graphs—email-EuAll and roadNet-PA/TX/CA—and is not generalized to other graphs.
- The comparator is limited to the evaluated snapshot and configuration of the third-party PathMerge implementation from upstream `gobardhanm/path-merging-bc`. It is an external comparator, not ground truth, and the result is not generalized to PathMerge or Galliot as algorithms or to an official implementation by the original authors. Its upstream license was not independently verified.
- The GPU_Opt values were observed with fixed `b512`, whose batch-size quantity is defined for 1 stream, without a graph-wise search for the fastest batch. Only PathMerge received graph-wise tuning, so the comparison uses asymmetric batch configurations.
- The speedups against default `b64` PathMerge—7.15× for email-EuAll and 1.64× for roadNet-CA—are supplementary. The headline uses only the tuned-comparison speedups.
- The factors associated with this performance difference and the contributions of individual optimizations are not determined in this chapter. Chapter 7 presents the ablation and kernel analyses, and Chapter 10 provides the integrated discussion. Chapter 8 examines the capacity characteristics of the memory-management approaches, and Chapter 9 examines numerical agreement among BC outputs.
