# Chapter 8 Memory Scalability

This chapter answers RQ3: On the evaluated corrected 325557 graph, how do the memory-management approaches of GPU_Opt, GPU_Opt_Pure, and GPU_Opt_Pure_Chunked affect the feasible batch size and the observed memory constraints? The evaluation target is the corrected `325557_3216152_corrected_v1` graph ($n=325{,}557$, $m=3{,}216{,}152$). The formal results are based on targeted boundary validation with 1 trial per condition. This is not a sweep or a performance comparison among the approaches; it evaluates feasibility at the tested boundary conditions. Appendix A provides the run identifiers and input SHA256 in the provenance records.

<!-- Source note (internal): corrected graph SHA256 8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22; PBS job 2404743; checkpoint 45352a3. -->

## 8.1 Capacity Terms and Evaluation Scope

### 8.1.1 Static Graph Sizes

The input graph's file size must be distinguished from the GPU working set. Table 8.1 presents the static storage sizes of the corrected 325557 graph.

**Table 8.1: Static storage and per-source state for the corrected 325557 graph.**

| Quantity | Definition | Bytes | Decimal Size | Binary Size |
|---|---|---:|---:|---:|
| Input graph file | On-disk text CSR measured by `stat` | 45,348,105 | 45.35 MB | 43.25 MiB |
| CSR topology | $((n+1)+2m)\times4$ | 27,031,448 | 27.03 MB | 25.78 MiB |
| BC output vector | $n\times8$ | 2,604,456 | 2.60 MB | 2.48 MiB |
| Per-source state $M_{\mathrm{source}}$ | $32n+4D_{est}+8$, $D_{est}=256$ | 10,418,856 | 10.42 MB | 9.94 MiB |

<!-- Source note (internal): result/datasets/graph_catalog.tsv; raw_data/corrected_325557/job_2404743/implementation_manifest.tsv; src/proposed/host_pure.cu:141-157. -->

The on-disk graph file, in-memory CSR topology, and output BC vector are all smaller than the nominal 96 GB HBM3 capacity. The capacity pressure instead arises from retaining state arrays for several source vertices concurrently. These arrays include distance, shortest-path count, dependency, frontier, and stack state, and together form the batch-dependent source-local working set.

The corrected graph uses the retained adjacency representation described in Chapter 5, including self-loops and duplicate ordered pairs. RQ3 evaluates only this graph, which is not part of RQ1 and is not representative of every graph structure. This chapter does not claim that the retained self-loops or duplicate ordered pairs caused the observed memory boundary.

### 8.1.2 Batch-Dependent Working Set

Following the implementation and retained manifest, the code-derived allocation estimate is defined as

$$
M_{\mathrm{work}}=NS_{\mathrm{eff}}\times \mathrm{EffectiveBatch}\times M_{\mathrm{source}}.
$$

The requested batch is the number of source vertices specified for a condition, whereas the effective batch is the value used after any implementation clamping. The retained manifest fields `EffectiveNS` and `PerSourceStateBytes` are denoted by $NS_{\mathrm{eff}}$ and $M_{\mathrm{source}}$, respectively, in this thesis. Here, $NS_{\mathrm{eff}}$ is effective stream concurrency. For Chunked's concurrently resident estimate, `SUB_BATCH` is the source count in one sub-batch and replaces $\mathrm{EffectiveBatch}$. The value of `num_subs` is the number of source sub-batches. The resulting working-set value is an estimate derived from the array dimensions in the code. It is not measured process RSS, measured physical HBM residency, or measured migration bytes; these 3 quantities were not collected in this experiment.

A batch is not a graph partition. Each batch is an execution unit that groups a subset of all source vertices, and the outer loop processes every batch. Chunked further divides each source batch into sub-batches, and those iterations process every source vertex in that batch. Consequently, batching and source sub-batching neither approximate BC nor omit source vertices.

### 8.1.3 HBM3 Capacity Reference

The GH200 has a nominal 96 GB of on-package HBM3 [@nvidiaGraceHopperInDepth]. The retained runtime query recorded approximately 102.0 decimal GB in total and approximately 101.4 decimal GB in `free_before` at run start. GB and MB are decimal units, whereas GiB and MiB are binary units. The nominal value and runtime decimal values describe the same HBM3 under different unit systems and acquisition methods, not separate memory tiers. This chapter uses the approximately 101.4 GB `free_before` value from the same retained record system when comparing the code-derived estimate. The host-memory-limited configuration is a separate resource condition, not nominal HBM capacity or a measurement of physical host memory.

## 8.2 Memory-Management Variants

GPU_Opt, GPU_Opt_Pure, and GPU_Opt_Pure_Chunked are not 3 independent algorithms. They are memory-management variants of the common GPU execution framework and use the same all-sources Brandes computation.

**Table 8.2: Memory-management variants of the common GPU execution framework.**

| Implementation | Allocation Strategy | Purpose | Finite Limit |
|---|---|---|---|
| GPU_Opt_Pure | Device allocation with `cudaMalloc` | Device-only control | CUDA device-memory capacity |
| GPU_Opt | Managed allocation with `cudaMallocManaged` | Handle a working set that may exceed device memory through managed placement and migration | HBM3, host memory, cgroup, runtime resources |
| GPU_Opt_Pure_Chunked | Device allocation for one source sub-batch | Bound the simultaneously resident working set | Sub-batch buffers and all remaining finite resources |

GPU_Opt is the main implementation and uses Unified Memory (UM) for managed source-local state in the Unified Memory address space. UM is not used to partition and store a large input graph. The input is approximately 45.35 MB; UM instead handles managed state allocation that grows with the batch. Its capacity across GPU HBM, host memory, and associated runtime resources remains finite, and it is not guaranteed to avoid OOM. Although the implementation uses managed placement and migration, this study measured neither memory placement nor migration volume.

GPU_Opt_Pure is the device-only comparison and uses explicit device-memory allocation in GPU HBM. The observed failed boundary provides CUDA device-memory OOM evidence. This role does not establish it as the universally fastest method within capacity.

GPU_Opt_Pure_Chunked uses source sub-batching to control the resident source-local working set. It does not partition the graph and does not provide unlimited capacity. Its principal value is the extension of the feasible range through the tested upper bound, not proven peak performance.

## 8.3 Corrected Targeted Boundary Validation

Table 8.3 and Figure 8.1 present all conditions in the targeted boundary validation. A failure is represented as a separate status, not as 0 seconds.

**Table 8.3: Targeted memory-feasibility boundary on the corrected 325557 graph. Each condition was run once.**

| Implementation | Requested Batch | Code-Derived Working Set | Outcome | Failure Class | Runtime [s] |
|---|---:|---:|---|---|---:|
| GPU_Opt_Pure | 4096 | 85,351,268,352 bytes (85.35 GB) | Success | None | 65.89 |
| GPU_Opt_Pure | 8192 | 170,702,536,704 bytes (170.70 GB) | Failure | CUDA device-memory OOM, exit 1 | N/A |
| GPU_Opt | 10240 | 106,689,085,440 bytes (106.69 GB) | Success | None | 238.67 |
| GPU_Opt | 12288 | 128,026,902,528 bytes (128.03 GB) | Failure | Cgroup host-memory OOM kill, exit 137 | N/A |
| GPU_Opt_Pure_Chunked | 16384 | 68,722,774,176 bytes (68.72 GB) resident estimate | Success | None | 66.60 |

<!-- Source note (internal): result/tables/thesis/T4_memory_scalability.tsv; result/memory_scalability/corrected_325557/feasibility_boundary.tsv; raw_data/corrected_325557/job_2404743/{implementation_manifest,feasibility_results,oom_evidence}.tsv. For UM b12288, the direct post-hoc evidence is raw_data/corrected_325557/job_2404743/pbs_stdout.log:146, SHA256 3c4c46680f9432b94fef79ca9344027ad77195973d075b8019379f934feb8ec5. -->

![Figure 8.1: Corrected 325557 targeted memory-feasibility boundary](../../../../result/figures/thesis/memory_scalability_325557.png)

**Figure 8.1: Targeted feasibility boundary on the corrected 325557 graph (one trial per condition). Failure markers distinguish CUDA device-memory out-of-memory from a cgroup host-memory OOM kill; failed runs are not plotted as zero-second runtimes.**

For Pure b8192, the `cudaMalloc` call at `host_pure.cu:144` returned `out of memory`. The retained records contain `oom_evidence=cuda_oom` and runner exit 1. This was a CUDA device-memory OOM, and its runtime was unavailable rather than zero.

UM b10240 succeeded when its 106.69 GB estimate with $NS_{\mathrm{eff}}=1$ exceeded the approximately 101.4 GB run-start `free_before` value. This success did not result from a large input file. It occurred under a condition in which the batch-dependent managed allocation could exceed free HBM. The UM code uses managed placement and migration, but this study measured neither physical residency nor migration bytes. The physical placement during the successful run therefore cannot be determined.

The UM b12288 run was terminated by a host-memory cgroup OOM kill. The retained PBS epilogue provides the direct evidence; the runtime per-configuration classifier reported none because that epilogue was appended after the configuration had completed. The runtime record retained runner exit 137 and a SIGKILL-compatible status. It contained no CUDA OOM string and recorded `oom_evidence=none`, so this failure is not described as a CUDA or HBM OOM. UM was not unlimited under this condition.

Within the tested range, the observed ordering of the largest successful batches was Pure b4096, UM b10240, and Chunked b16384. Each point came from 1 targeted validation trial. These observations are neither general capacity limits nor a runtime ranking; the largest Chunked batch was only the tested upper bound.

## 8.4 Chunked Source Sub-Batches

Chunked b16384 succeeded with `SUB_BATCH=6596`, `num_subs=3`, and $NS_{\mathrm{eff}}=1$. Its concurrently resident estimate is

$$
6596\times10{,}418{,}856=68{,}722{,}774{,}176\ \mathrm{bytes}
$$

`SUB_BATCH=6596` was not determined from HBM3 capacity alone. `host_chunked.cu` takes the smaller of the HBM-budget-derived limit and the following index-safety limit. This limit prevents index overflow.

$$
safe\_sub\_batch=\left\lfloor\frac{INT\_MAX}{n}\right\rfloor
=\left\lfloor\frac{2{,}147{,}483{,}647}{325{,}557}\right\rfloor=6596
$$

For the corrected 325557 graph, integer division gave an HBM-budget limit of 7783 source vertices from the retained `free_before` value and the code formula. The index-safety limit was 6596 and was therefore the binding constraint. This observation is not generalized to a different $n$ or an unmeasured GPU.

The 3 sub-batches reuse the same buffer while advancing the source offset. This is exact BC processing of every source vertex in the requested batch, not an approximate partition of the graph edge set. The outer batch loop also processes all source vertices.

## 8.5 Performance Interpretation and Measurement Limits

Feasibility asks whether a configuration completed, whereas performance asks how quickly repeated, comparable configurations executed. The runtimes in Table 8.3 are single-run wall-clock times accompanying the feasibility observations. They are descriptive records without trial aggregation. The batches, allocation modes, and numbers of sub-batches differ, so these values are not used for a formal performance comparison among the approaches.

In particular, the 238.67 s runtime of UM b10240 is not used to infer migration cost. The approximately 66 s Pure and Chunked records do not establish a speed ranking. Success at a larger requested batch does not mean faster execution, and no statistical inference is made from these single-run boundary timings. UM success beyond the Pure boundary supports a feasibility observation, while the Chunked capacity extension is not a peak-performance claim.

The following quantities were not collected in this study:

- Full-run process RSS
- Physical HBM residency
- Total host residency
- Full-run migration bytes

The 25-second partial trace is supplementary material from a different condition, not the full-run migration total for the formal boundary validation.

## 8.6 Historical Malformed-Input Results

The legacy sweep and the labels `CORE_FAIL` and `OOM_OR_FAIL` for the former `325557_3216152` input remain in the archive as historical invalid-input evidence. That input was malformed. Formal validation on the corrected input replaces it as the current RQ3 boundary. The failure classes, checkpoints, resource conditions, timings, and implementation-dependent warmup records of historical and current corrected results must not be combined. The current conclusion is derived only from the current corrected series; the legacy series is supplementary evidence.

<!-- Source note (internal): historical canonical job 2368587; current corrected-input job 2404743. -->

## 8.7 Answer to RQ3

RQ3 is answered as `SUPPORTED_WITH_LIMITATIONS`. In the targeted boundary validation on the corrected 325557 graph, Pure succeeded at b4096 and encountered a CUDA device-memory OOM at b8192. UM succeeded at b10240 and was terminated by a host-memory cgroup OOM kill at b12288. Chunked succeeded at the tested upper bound of b16384 with `SUB_BATCH=6596` and `num_subs=3`. Thus, the observed feasible range differed among the variants. UM managed a larger batch than the observed Pure boundary through managed allocation. Chunked further extended the tested range by controlling the concurrently resident working set.

This conclusion is limited to the corrected graph, the GH200, and 1 trial per boundary condition. The runtimes are descriptive records and do not answer a performance-ranking question. The result does not generalize to larger untested batches, other graphs, or other GPUs. In particular, the Chunked success is not an absolute maximum, UM does not provide unlimited capacity, and Chunked is not guaranteed to avoid OOM under unmeasured conditions. The principal design implication is that capacity pressure arises from `batch × per-source state`, not the approximately 45.35 MB on-disk input graph file. Source grouping and resident-state management are therefore the control points for extending capacity.
