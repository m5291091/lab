# Chapter 7 Ablation and Kernel Analysis

This chapter answers RQ2 on optimization contributions: To what extent do Hybrid BFS, Warp-Cooperative Accumulation, Dual-Stream Execution, and the Block Kernel contribute to the observed performance? (Section 5.1). The evaluation follows the 2 procedures defined in Section 5.8: a factorial ablation that toggles the 3 H/W/A factors and a forced direct comparison of the shared and block BFS kernels. This chapter introduces no new experimental conditions. It also describes the observed phase breakdown of runtime.

The role of this chapter must be clear at the outset. This chapter does not uniquely decompose the 1.31×–3.17× end-to-end speedup observed in Chapter 6 into causal components. The quantities examined here are (i) observed main effects calculated from the factorial design, (ii) median runtimes of individual configurations, (iii) observed proportions in the phase breakdown, and (iv) runtime ratios from the forced kernel comparison. They are distinct observations with graph sets and definitions that differ from those of the end-to-end speedup in Chapter 6. The H/W/A main effects must therefore neither be added together nor interpreted as percentage contributions to that speedup. Chapter 10 provides an integrated discussion of the factors associated with the observed effects; this chapter is limited to describing the observations and defining their scope.

## 7.1 Ablation Design

The ablation evaluates 3 features of the proposed execution framework. Hybrid BFS (H) is a direction-optimizing BFS that switches between top-down and bottom-up traversal during the forward search [@beamer2012]. Warp-Cooperative Accumulation (W) processes the Backward dependency-accumulation phase cooperatively within a warp by using warp-level shuffle reduction. Dual-Stream Execution (A) uses 2 CUDA streams (NS=2) and double buffering to overlap asynchronous initialization of the next batch with computation. Table 7.1 shows the behavior when each factor is disabled (0) or enabled (1). The disabled behaviors were verified from the implementation (`src/proposed/host_ablation.cu` and `include/proposed/brandes_kernels.cuh`): H=0 uses only top-down BFS, W=0 uses a thread-per-vertex Backward kernel, and A=0 uses 1 stream (NS=1) with synchronous initialization. C++ templates select the 3 factors at compile time rather than through runtime branches inside CUDA kernels, instantiating a dedicated branch-free kernel for each of the 8 configurations.

**Table 7.1: Ablation factor definitions. Disabled/enabled behaviors are taken from the ablation implementation.**

| Factor | Disabled State (0) | Enabled State (1) | Target Phase |
|---|---|---|---|
| H (Hybrid BFS) | Top-down traversal only | Direction-optimizing top-down / bottom-up switching | Forward BFS |
| W (Warp-Cooperative Accumulation) | Thread-per-vertex accumulation kernel | Warp-cooperative accumulation (warp-level shuffle reduction) | Backward (dependency accumulation) |
| A (Dual-Stream Execution) | Single CUDA stream (NS=1), synchronous initialization | Two CUDA streams (NS=2), asynchronous initialization overlapped with computation (double buffering) | Batch initialization / kernel pipeline |

<!-- Source note (internal): src/proposed/host_ablation.cu; include/proposed/brandes_kernels.cuh; result/ablation/{corrected_325557,synthetic_2354994,email_2354999}/. -->

The experiment measured all 8 configurations in $\mathrm{H}\{0,1\} \times \mathrm{W}\{0,1\} \times \mathrm{A}\{0,1\}$. H0W0A0 is the baseline, and H1W1A1 is the full configuration. Each configuration used `n=5` on benchmark_7000_41459, benchmark_11023_62184, 56438_300801, and the corrected `325557_3216152_corrected_v1` graph, and `n=3` on email-EuAll. At the start of every runner invocation for a set of 8 configurations, the runner executed 1 global, untimed H1W1A1 warmup. The warmup was outside the timing function and was not included in any formal TSV row. The Synthetic-4 aggregate is mixed-checkpoint and is not a remeasurement of all 4 graphs at one checkpoint. Every configuration used fixed `b512`, and the primary value is the configuration median. Because the ablation runs recorded no BC vector comparison, their correctness level is `none`. Appendix A gives the provenance of the measurement series, marker counts, and formal row counts. As Section 5.3 explains, the corrected graph retains a stored adjacency representation containing 87,442 self-loops and 866,924 duplicate ordered pairs. It is outside the RQ1 graph set and is used here only for the RQ2 component study; this chapter does not attribute any runtime difference to those retained structures.

<!-- Source note (internal): jobs 2354994, 2354999, and 2406254; corrected-series checkpoint 45352a3; marker counts 20/3/5 and formal row counts 160/24/40. -->

The synthetic graphs and email-EuAll are aggregated separately. The 2 groups have contrasting degree distributions and traversal depths (Section 5.3) and different trial counts (`n=5` and `n=3`), so they are not combined into a single aggregate. The 4 synthetic graphs are summarized with a geometric mean, whereas email-EuAll is reported individually. The ablation also forms a dedicated measurement series, so its absolute runtimes must not be equated with the main performance values in Chapter 6. For example, the full H1W1A1 configuration on email-EuAll had a median of 30.42 s in this series, whereas GPU_Opt had a median of 30.81 s in Table 6.1; these values come from different measurement series.

The contribution of each factor is evaluated as a main effect. For a factor $F \in \{\mathrm{H}, \mathrm{W}, \mathrm{A}\}$, the main effect is the geometric mean of the median runtime ratios with $F$ disabled and enabled over all 4 level combinations of the remaining 2 factors $(G_1, G_2)$. Following the definition in the aggregation script (`scripts/summarize_ablation.py`), it is calculated for graph $g$ as

$$
\mathrm{ME}_g(F) = \left( \prod_{(b_1, b_2) \in \{0,1\}^2} \frac{T^{\mathrm{med}}_g(F{=}0,\, G_1{=}b_1,\, G_2{=}b_2)}{T^{\mathrm{med}}_g(F{=}1,\, G_1{=}b_1,\, G_2{=}b_2)} \right)^{1/4}
$$

where $T^{\mathrm{med}}_g(\cdot)$ is the median runtime of the corresponding configuration. A value of $\mathrm{ME}_g(F) > 1$ means that enabling $F$ shortened the observed median runtime, whereas $\mathrm{ME}_g(F) < 1$ means that it lengthened the observed median runtime. The summary for the synthetic graphs is the geometric mean of the main effects over the 4 graphs,

$$
\mathrm{ME}_{\mathrm{synth}}(F) = \left( \prod_{g \in \mathcal{G}_{\mathrm{synth}}} \mathrm{ME}_g(F) \right)^{1/4}
$$

The main-effect values in this chapter were recalculated with these equations from the unrounded canonical trial data and verified against the official aggregation. The text and Figure 7.1 show 3 decimal places; Table 7.2 shows 4 decimal places to provide consistent precision within that table.

<!-- Source note (internal): raw_data/ablation/ and result/ablation/*/ablation_contributions.tsv. -->

This evaluation does not estimate formal interaction terms among the factors. As a supplementary check for indications of interaction, the aggregation script compares add-one contribution, $T(\mathrm{H0W0A0})/T(F\text{ only enabled})$, with leave-one-out contribution, $T(\text{full with only }F\text{ disabled})/T(\mathrm{H1W1A1})$. Across all factors and graphs, the largest relative difference was 9.2%, below the 10% decision threshold. This is only a limited check; this chapter does not conclude that the effects of the 3 factors are independent. For the same reason, a main effect is not an additive allocation of contribution. The product $\mathrm{ME}(\mathrm{H}) \cdot \mathrm{ME}(\mathrm{W}) \cdot \mathrm{ME}(\mathrm{A})$ is not guaranteed to equal the improvement from the baseline to the full configuration.

Table 7.2 summarizes the observed main effects, and Figure 7.1 gives the graph-wise breakdown. Appendix C provides the median runtimes and all trial values for the 8 configurations. Inter-trial variation was small. Recalculation from the raw data showed that the sample standard deviation (`ddof=1`) was at most 0.33% of the median on email-EuAll and at most 3.2% among the synthetic graphs. The latter maximum occurred for H0W0A1 on the smallest graph, benchmark_7000_41459, with a standard deviation of 0.0014 s and a median of 0.0442 s. For configurations with runtimes of at least 1 second, the sample standard deviation was below 1% of the median.

**Table 7.2: Observed main effects of the three ablation factors, rounded to four decimal places.**

| Dataset Group | H Main Effect | W Main Effect | A Main Effect | Trials per Configuration | Aggregation |
|---|--:|--:|--:|--:|---|
| Synthetic (4 graphs, mixed-checkpoint geometric mean) | 1.6787 | 1.0661 | 1.3914 | 5 | Median per configuration; factorial main effect; geometric mean across 4 graphs |
| 325557_3216152_corrected_v1 | 1.4767 | 1.1012 | 1.5563 | 5 | Corrected re-measurement |
| email-EuAll | 1.4286 | 0.9695 | 1.7199 | 3 | Median per configuration; factorial main effect |

<!-- canonical artifact: T3_ablation_summary (internal ID: T3) -->
<!-- Source note (internal): result/tables/thesis/T3_ablation_summary.tsv. The synthetic-4 aggregate combines unchanged three-graph job 2354994 data with corrected-325557 job 2406254 data; it is not a same-checkpoint four-graph remeasurement. -->

![Figure 7.1: Ablation main effects](../../../../result/figures/thesis/ablation_contributions.png)

**Figure 7.1: Per-factor main-effect speedups of the H/W/A factorial ablation. The synthetic-4 aggregate is mixed-checkpoint rather than a same-checkpoint four-graph remeasurement. Bars use configuration medians (synthetic: n=5; email-EuAll: n=3; fixed b512).**

<!-- canonical artifact: ablation_contributions.{png,pdf,svg} (internal ID: F4); see result/figures/thesis/FIGURE_MANIFEST.tsv. The figure number is provided by the thesis caption; the exported figure does not embed an internal artifact ID. -->

## 7.2 Effect of Hybrid BFS

The main effect of Hybrid BFS (H) was 1.536, 1.782, and 1.965 on the first 3 synthetic graphs and 1.4767 on the corrected 325557 graph, producing a mixed-checkpoint geometric mean of 1.679. It was 1.429 on email-EuAll. H showed a substantial observed main effect: its value exceeded 1 on all 5 evaluated graphs. This observation is not generalized to unmeasured graphs.

In the synthetic and email-EuAll measurement series that retain phase attribution, enabling H shortened the accumulated BFS time on 56438_300801 and email-EuAll. The formal artifact for the corrected 325557 graph retains wall-time medians and shows a reduction from 176.35 s for H0W0A0 to 116.33 s for H1W0A0, but it does not retain per-phase timers. Phase values from the former malformed 325557 graph are therefore not used as causal evidence for the corrected graph.

These values are observations on the 4 evaluated synthetic graphs and email-EuAll. They do not represent an effect on synthetic graphs in general and are not generalized to unmeasured graphs, including roadNet-PA/TX/CA, on which the H/W/A factorial ablation was not conducted.

## 7.3 Effect of Warp-Cooperative Accumulation

The main effect of Warp-Cooperative Accumulation (W) was 1.175, 1.007, and 0.992 on the first 3 synthetic graphs and 1.1012 on the corrected 325557 graph, producing a mixed-checkpoint geometric mean of 1.066. The evaluated graphs included both near-neutral and positive observed effects.

On email-EuAll, the main effect of W was 0.970. A value below 1 means that enabling W was associated with an approximately 3.1% increase in the observed median runtime ($1/0.970 \approx 1.031$); it does not represent a performance improvement. This observed difference was larger than the inter-trial variation within each configuration, for which the sample standard deviation was at most 0.33% of the median. However, each configuration had the small sample size `n=3`, and no significance test was conducted, so this study makes no claim of statistical significance. Nor does it conclude that the difference was measurement error. The supported observation is that W was slightly unfavorable on email-EuAll under the evaluated conditions.

W was 0.970 on email-EuAll, and its main effects ranged from 0.970 to 1.175 over the 5 evaluated graphs. Its value was 1.1012 on the corrected 325557 graph. The differences in direction and magnitude indicate that the observed effect depended on the evaluated graph. This conclusion does not generalize the causes of the graph-dependent behavior to unmeasured graphs.

## 7.4 Effect of Dual-Stream Execution

The main effect of Dual-Stream Execution (A) was 1.234, 1.577, and 1.238 on the first 3 synthetic graphs and 1.5563 on the corrected 325557 graph, producing a mixed-checkpoint geometric mean of 1.391. It was 1.720 on email-EuAll. A showed a substantial observed main effect, with a value above 1 on all 5 evaluated graphs.

Under the evaluated conditions that retain phase logs for A=1, the gap—the residual after subtracting accumulated BFS and Backward kernel times from wall time—was negative. This is consistent with temporal overlap because the accumulated BFS and Backward times for A=1 (NS=2) sum the times from both streams. The formal artifact for the corrected 325557 graph does not retain per-phase timers, so phase values from the former input are not transferred to the corrected graph. This chapter also does not establish why the observed effect of A was large on email-EuAll.

## 7.5 Shared and Block Kernels

This section directly compares the shared-frontier BFS kernel (shared) with the 1-block-per-source kernel (block). As specified in Section 5.8, the environment variable `BC_FORCE_BFS_KERNEL=shared|block` forced each kernel; this is a forced comparison, not an evaluation of an automatic selection rule. The targets were the 2 graphs roadNet-PA and roadNet-TX. The settings were batch 512 (`SUB_BATCH=512`, `num_subs=1`, in-capacity), `n=3` per kernel, no warmup, and median aggregation with the sample standard deviation. The SourceSnapshotID was `phase_def_block_20260710`. Table 7.3 and Figure 7.2 present the results.

**Table 7.3: Forced shared/block BFS kernel comparison on roadNet-PA and roadNet-TX. Speedup = slower kernel median / faster kernel median. Max BC Match compares the maximum-BC index and value between the two kernels.**

| Graph | Shared Trials | Shared Median [s] | Block Trials | Block Median [s] | Faster Kernel | Speedup | Max BC Match |
|---|--:|--:|--:|--:|---|--:|---|
| roadNet-PA | 3 | 1063.71 | 3 | 701.57 | Block | 1.52 | Yes (index and value) |
| roadNet-TX | 3 | 1639.16 | 3 | 984.59 | Block | 1.66 | Yes (index and value) |

<!-- canonical artifact: kernel_selection_contributions.tsv (internal table ID: T-KSEL) -->
<!-- Source note (internal): raw_data/tuning/kernel_selection/roadNet-PA/gpu_opt_forced/job_2354329_20260710/kernel_selection_results.tsv and raw_data/tuning/kernel_selection/roadNet-TX/gpu_opt_forced/job_2354330_20260710/kernel_selection_results.tsv; medians and sample standard deviations cross-checked against the corresponding result/tuning/kernel_selection tables. -->

![Figure 7.2: Forced shared vs block kernel comparison](../../../../result/figures/thesis/shared_vs_block_kernel.png)

**Figure 7.2: Median runtime of the forced shared and forced block BFS kernels on roadNet-PA and roadNet-TX (n=3 per kernel per graph, fixed b512). Error bars show the sample standard deviation.**

<!-- canonical artifact: shared_vs_block_kernel.{png,pdf,svg} (internal ID: F6). The figure number is provided by the thesis caption; the exported figure does not embed an internal artifact ID. -->

The observed results were as follows. On roadNet-PA, the shared median was 1063.71 s and the block median was 701.57 s, so the forced block kernel was 1.52× faster. On roadNet-TX, the respective medians were 1639.16 s and 984.59 s, so the forced block kernel was 1.66× faster. Speedup is the ratio of the slower median to the faster median. The sample standard deviations were 0.060 s for shared and 3.574 s for block on PA, and 0.284 s for shared and 7.260 s for block on TX. Although the block measurements had greater relative variation, the variation was small relative to the differences between the kernel medians, approximately 362 s on PA and 655 s on TX.

For correctness, the maximum-BC index and value agreed between shared and block on both graphs: index 557532 and value 151395302679.08 on roadNet-PA, and index 400570 and value 164495142042.45 on roadNet-TX. The evidence level of this comparison is nevertheless only `max_bc_only` (Table 5.5 in Section 5.10). Agreement in the maximum BC does not establish full-vector correctness for all BC elements.

The current kernel policy is as follows. The current implementation, including the GPU_Opt evaluated in Chapter 6, always uses the block BFS kernel. A former implementation had an automatic mean-degree rule that selected shared when `avg_deg < 5`, but that rule is not used in the current implementation. The forced comparison provides design support for the current adoption of block only within the roadNet-PA/TX scope; it does not evaluate whether the former rule was correct. The result is limited to the 2 measured graphs and does not imply the relative performance of shared and block on unmeasured graphs, including email-EuAll and the synthetic graphs. It therefore does not establish a general best-kernel rule.

## 7.6 Phase Breakdown

This section describes the observed runtime phase breakdown using 2 types of evidence. The first is the full-duration component breakdown for the 4 headline graphs, obtained from phase-timing records of the same GPU_Opt fixed-`b512` runs used for the main performance measurements in Chapter 6 (Figure 7.3). The second is the composition of GPU kernel time in a single nsys trace. Both describe observed time distributions. Neither the phase proportions nor the trace proportions are treated as H/W/A main-effect contribution percentages or as causal evidence for performance differences.

Figure 7.3 reports BFS wall time and Backward wall time measured by the runner, together with Other. Other is the per-trial residual obtained by subtracting BFS and Backward from total runtime; it includes initialization, CSR loading, result retrieval, and host-side overhead. GPU_Opt, the Unified Memory variant, does not separately time host-to-device or device-to-host transfers or initialization. This breakdown therefore decomposes the available measured components but is not a complete breakdown that includes transfer quantities. The aggregation uses the median of each component across trials (`n=5` for email-EuAll and `n=3` for each roadNet graph).

![Figure 7.3: Phase breakdown of GPU_Opt](../../../../result/figures/thesis/phase_breakdown.png)

**Figure 7.3: Median phase components (BFS, Backward, Other) of GPU_Opt (fixed b512) on the four headline graphs. Other is the per-trial residual: total time minus BFS minus Backward (initialization, CSR load, copy-out, and host overhead). Components are medians over trials (email-EuAll n=5; roadNet-PA/TX/CA n=3).**

<!-- canonical artifact: phase_breakdown.{png,pdf,svg} (internal ID: F7). The figure number is provided by the thesis caption; the exported figure does not embed an internal artifact ID. -->

The observed median components were as follows. On email-EuAll, BFS was 9.12 s, Backward was 20.07 s, and Other was 1.60 s; Backward was the largest component, at approximately 65% of the component sum. On roadNet-PA, the respective values were 348.91 s, 290.61 s, and 62.60 s. They were 355.43 s, 477.56 s, and 145.58 s on roadNet-TX, and 816.37 s, 950.28 s, and 362.46 s on roadNet-CA. On the 3 road graphs, BFS and Backward were of similar order, each accounting for approximately 36%–50% of the component sum, while Other accounted for approximately 9%–17%. These proportions describe the measured `b512` runs and do not represent other batch settings or implementations.

As supplementary kernel-level evidence, a single nsys trace showed that the Backward kernel (`brandes_back_kernel_opt`) accounted for 63.9% and the BFS kernel (`brandes_bfs_kernel_opt`) for 36.1% of CUDA GPU kernel time. The measured configuration in this `ablation_H1W1A0` trace was H1W1A0 in the ablation binary, and the target graph was 56438_300801. However, the trace scope also included the untimed H1W1A1 warmup at the beginning of the same process. These proportions therefore describe the entire single-trace scope including the warmup, not the isolated formal H1W1A0 measurement. They cover only GPU kernel time and exclude host and transfer time, so they are not generalized to the phase composition of other experiments. A separate `ablation_H1W1A1` full-duration trace is retained in the same series but is not used for the quantitative values in this chapter. The separate `um_prefetch_gpu_opt` artifact is a partial `--duration=25` trace of the historical, former 325557 input rather than the corrected graph; it does not cover the full execution. This chapter does not use its partial migration or fault quantities and does not infer effects from Unified Memory migration volume, HBM residency, cache behavior, bandwidth limitations, or NVLink-C2C transfer.

<!-- Source note (internal): phase components from raw_data/main_performance/proposed_variants/*/_run/job_2357334_20260711/phase_timing.log; kernel-time shares and target attribution from raw_data/profiling/job_2359175_20260711/. -->

## 7.7 Answer to RQ2

RQ2 is answered as `SUPPORTED_WITH_LIMITATIONS`. On the corrected 325557 graph, the observed main effects were H=1.4767, W=1.1012, and A=1.5563. The mixed-checkpoint aggregate over the 4 synthetic graphs was H=1.679, W=1.066, and A=1.391. The factorial comparison supports the conclusion that Hybrid BFS and Dual-Stream Execution showed substantial observed effects, whereas Warp-Cooperative Accumulation was graph-dependent. The Synthetic-4 aggregate replaces only the 325557 measurement with the corrected-input series and is not a same-checkpoint remeasurement. In the forced comparison on roadNet-PA/TX, the block kernel was 1.52× and 1.66× faster than shared, respectively; this supports the block result only for PA/TX and does not extend the H/W/A findings causally to roadNet. In this RQ, “contribute” refers to these observed factorial and forced-comparison results and does not establish a universal causal effect.

<!-- English version (plan.md 8.8): "Hybrid BFS and dual-stream execution provided the main observed improvements, whereas warp-cooperative accumulation was graph-dependent." -->

This answer is subject to the following limitations.

- The full H/W/A factorial ablation is limited to the 4 synthetic graphs and email-EuAll; it was not conducted on roadNet-PA/TX/CA. The H/W/A contributions on roadNet cannot be estimated from the values in this chapter.
- The main effect is an observed quantity in the factorial design, not an additive contribution percentage. The end-to-end speedup in Chapter 6 cannot be explained as the sum or product of the main effects.
- Formal interaction terms were not estimated. The supplementary check found a maximum relative difference of 9.2%, but this does not establish that the effects of the 3 factors are independent.
- The forced shared/block comparison is limited to the 2 graphs roadNet-PA/TX and is not generalized to unmeasured graphs. Agreement in the maximum BC does not establish full-vector correctness.
- The evidence is limited by the evaluated graph set, the small trial counts (`n=3` or `n=5` per configuration), and the mixed-checkpoint Synthetic-4 aggregate. The corrected 325557 measurement and the other 3 synthetic measurements do not form a same-checkpoint series.
- The phase breakdown in Figure 7.3 and the nsys kernel proportions are observed distributions for specific configurations, graphs, and runs; they are not allocations of the observed speedup.
- This chapter does not establish the causes of the observed effects, including causal relationships with graph structure, the larger A value on email-EuAll, or the unfavorable W result on email-EuAll. Chapter 10 discusses these observations. The conclusions are not generalized beyond the evaluated GH200 environment, graphs, and conditions.
