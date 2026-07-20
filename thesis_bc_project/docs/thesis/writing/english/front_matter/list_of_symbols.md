# List of Symbols

This list includes only the symbols that are actually defined and used repeatedly in the text and equations of Chapters 1 through 11 and the Abstract. Code identifiers, recorded column names in saved TSVs and manifests, and local temporary variables are excluded; necessary terms of this type are handled in the `Recorded Fields` and `Units and Conventions` sections. As a principle, the same character is not assigned to different concepts.

## Graph and Betweenness Centrality Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $G=(V,E)$ | A graph consisting of a vertex set $V$ and an edge set $E$. The input of this study is an undirected, unweighted graph. | — |
| $V$ | The set of vertices. | Set |
| $E$ | The set of edges. | Set |
| $n=\lvert V\rvert$ | The number of vertices. It is not used for the number of trials. | Count |
| $m=\lvert E\rvert$ | The number of undirected edges. | Count |
| $u,\ v,\ w$ | General vertices. $w$ primarily represents a vertex being processed in the Backward phase. | Element of $V$ |
| $s$ | A source vertex. The search origin for the Brandes algorithm. It is not used for sample standard deviation. | Element of $V$ |
| $t$ | A target vertex. | Element of $V$ |
| $d(s,t)$ | The shortest path length from $s$ to $t$. | Edges |
| $d_s(v)=d(s,v)$ | The shortest distance from source $s$ to vertex $v$. An unvisited state is represented by $-1$. | Edges |
| $\sigma_{st}$ | The total number of shortest paths from $s$ to $t$. | Count |
| $\sigma_{st}(v)$ | The number of shortest paths from $s$ to $t$ that pass through $v$ as an internal vertex. | Count |
| $\sigma_s(v)=\sigma_{sv}$ | The number of shortest paths from source $s$ to vertex $v$. | Count |
| $P_s(w)$ | The predecessor set of $w$ in the shortest-path DAG rooted at source $s$. | Set |
| $Succ_s(w)$ | The set of adjacent vertices of $w$ satisfying $d_s(v)=d_s(w)+1$. The proposed implementation uses this relationship without materializing $P_s$. | Set |
| $\delta_s(v)$ | The dependency of source $s$ on vertex $v$. | Real number |
| $C_B^{\mathrm{dir}}(v)$ | Unnormalized directed BC based on ordered pairs. | Real number |
| $C_B^{\mathrm{undir}}(v)$ | Unnormalized undirected BC applying the $1/2$ correction. The quantity computed by this study. | Real number |
| $\widehat{C}_B^{\mathrm{dir}}(v),\ \widehat{C}_B^{\mathrm{undir}}(v)$ | Normalized directed/undirected BC. Not applied in this study. | Real number |
| $CB$ | The BC output array storing $C_B^{\mathrm{undir}}$. Used in Algorithm 2.1 and Algorithm 4.1. | Array of length $n$ |
| $S$ | The traversal stack storing vertices in non-decreasing order of BFS distance. $S_s$ represents the stack for source $s$. It is not used for speedup. | Array |
| $R$ | The CSR row pointer array. $n+1$ elements. | Array |
| $C$ | The CSR adjacency array. Symmetrized to $2m$ elements. | Array |

## Hybrid BFS Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $Q$ | The BFS frontier. The implementation maintains the current and next frontiers in separate arrays. | Array |
| $\lvert Q\rvert$ | The size of the current frontier. Used for the bottom-up to top-down reversion check. | Count |
| $m_f$ | An approximation of the number of edges outgoing from the current frontier. | Count |
| $m_u$ | An approximation of the number of remaining edges outgoing from the unexplored side. | Count |
| $\alpha$ | The threshold parameter for switching from top-down to bottom-up. The switching condition is $m_f > m_u/\alpha$. The implementation uses $\alpha=14$. | Dimensionless |
| $\beta$ | The threshold parameter for reverting from bottom-up to top-down. The reversion condition is $\lvert Q\rvert < n/\beta$. The implementation uses $\beta=24$. | Dimensionless |

$\alpha=14$ and $\beta=24$ are switching parameters used in the evaluation by Beamer et al. and adopted in this implementation; they are not universal optimums for all graphs and hardware.

## Batch and Memory Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $\mathrm{EffectiveBatch}$ | The number of sources per stream actually used by the outer batch loop. It is reduced from the requested batch when exceeding the HBM budget. | Sources |
| $q$ | The source offset within a batch. The block with `blockIdx.x=q` handles source $s_{start}+q$. | Index |
| $s_{start}$ | The starting source number of an outer batch. | Index |
| $\texttt{SUB\_BATCH}$ | The maximum number of sources processed by a single sub-launch when dividing a batch. | Sources |
| $\texttt{num\_subs}$ | The number of sub-launches. $\lceil \mathrm{EffectiveBatch}/\texttt{SUB\_BATCH}\rceil$. | Count |
| $NS_{\mathrm{eff}}$ | The number of concurrently active stream buffers. 2 for in-capacity execution, 1 for the oversubscription path. | Count |
| $D_{est}$ | The upper bound of the BFS depth estimated by the implementation. $D_{est}=256$ for the corrected 325557 graph. | Levels |
| $M_{\mathrm{source}}$ | The source-local state size. $M_{\mathrm{source}}=32n+4D_{est}+8$ bytes. | bytes |
| $M_{\mathrm{work}}$ | The batch-dependent working-set conceptual quantity (code-derived allocation estimate). | bytes |

The working set is conceptually modeled as follows:

$$
M_{\mathrm{work}}
\approx
NS_{\mathrm{eff}}
\times
\mathrm{EffectiveBatch}
\times
M_{\mathrm{source}}.
$$

The concurrent resident estimate for Chunked uses $\texttt{SUB\_BATCH}$ instead of $\mathrm{EffectiveBatch}$. These are allocation estimates derived from array dimensions, not measured process RSS, physical HBM residency, or migration bytes. A batch or sub-batch is a grouping of sources, not a graph partition.

## Performance Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $T$ | The execution time. The total time for the implementation function measured by the runner. | s |
| $t_i$ | The execution time of the $i$-th trial. | s |
| $\bar{t}$ | The sample mean of the trials. | s |
| $N_{\mathrm{trials}}$ | The number of trials. A different symbol is used from the vertex count $n$. `n=3` and `n=5` in tables and figures are conventional labels indicating this quantity. | Count |
| $s_T$ | The sample standard deviation of runtime (unbiased estimator, ddof=1). A different symbol is used from the source vertex $s$. | s |
| $T^{\mathrm{med}}_{\mathrm{baseline}}$ | The median execution time of the baseline. Tuned PathMerge for RQ1. | s |
| $T^{\mathrm{med}}_{\mathrm{proposed}}$ | The median execution time of the proposed method. Fixed `b512` GPU_Opt for RQ1. | s |
| $\mathrm{Speedup}$ | The speedup defined as the ratio of medians. A different symbol is used from the traversal stack $S$. | Dimensionless |
| $\mathrm{GTEPS}$ | The throughput. $\mathrm{GTEPS}=n\cdot m/(T\cdot 10^{9})$. | $10^{9}$ edges/s |

Speedup is defined as the ratio of medians:

$$
\mathrm{Speedup} = \frac{T^{\mathrm{med}}_{\mathrm{baseline}}}{T^{\mathrm{med}}_{\mathrm{proposed}}}
$$

The primary values are medians, and speedup is not calculated by mixing medians and means. The sample standard deviation is the unbiased estimator (ddof=1):

$$
s_T = \sqrt{\frac{1}{N_{\mathrm{trials}}-1}\sum_{i=1}^{N_{\mathrm{trials}}}\left(t_i-\bar{t}\right)^2}
$$

## Ablation Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $\mathrm{H}$ | The binary configuration flag for Hybrid BFS. 0 for top-down only, 1 for top-down/bottom-up switching. | $\{0,1\}$ |
| $\mathrm{W}$ | The binary configuration flag for Warp-Cooperative Accumulation. 0 for thread-per-vertex, 1 for warp-cooperative accumulation. | $\{0,1\}$ |
| $\mathrm{A}$ | The binary configuration flag for Dual-Stream Execution. 0 for a single stream, 1 for 2 streams. | $\{0,1\}$ |
| $F$ | The target factor whose main effect is evaluated. $F\in\{\mathrm{H},\mathrm{W},\mathrm{A}\}$. | Factor |
| $G_1,\ G_2$ | The remaining two factors other than $F$. | Factor |
| $T^{\mathrm{med}}_g(\cdot)$ | The median execution time of the configuration on graph $g$. | s |
| $\mathrm{ME}_g(F)$ | The main effect of factor $F$ on graph $g$. A value $>1$ indicates that enabling the factor reduced the median execution time. | Dimensionless |
| $\mathcal{G}_{\mathrm{synth}}$ | The set of synthetic graphs used to aggregate main effects via geometric mean. | Set |
| $\mathrm{ME}_{\mathrm{synth}}(F)$ | The geometric mean of the main effects across $\mathcal{G}_{\mathrm{synth}}$. | Dimensionless |

H, W, and A are compile-time binary configuration flags with uniform meanings throughout this thesis, forming the 8 configurations $\mathrm{H}\{0,1\}\times\mathrm{W}\{0,1\}\times\mathrm{A}\{0,1\}$. A main effect is an observed quantity in a factorial design and is not an additive allocation of total speedup.

## Correctness Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $r_i$ | The BC value of the $i$-th element on the reference side. | Real number |
| $c_i$ | The BC value of the $i$-th element on the candidate side. | Real number |
| $\mathrm{abs\_tol}$ | The absolute tolerance value of the mixed tolerance criterion. $\mathrm{abs\_tol}=10^{-3}$. | Real number |
| $\mathrm{rel\_tol}$ | The relative tolerance value of the mixed tolerance criterion. $\mathrm{rel\_tol}=10^{-6}$. | Dimensionless |

The evaluation uses the following mixed absolute and relative tolerance criterion:

$$
\lvert r_i-c_i\rvert \le \mathrm{abs\_tol}+\mathrm{rel\_tol}\cdot\max\!\left(\lvert r_i\rvert,\lvert c_i\rvert\right)
$$

A PASS satisfying this criterion indicates numerical consistency within the mixed tolerance; it does not indicate that the results are byte-identical.

## Recorded Fields

The following are formal recorded column names in saved manifests and TSVs, or code identifiers. They are not included in the symbol list above because they are not mathematical symbols. When cited in the text, they are accompanied by the corresponding thesis symbols.

| Recorded Field | Thesis Symbol or Term | Explanatory Section |
|:--|:--|:--|
| `PerSourceStateBytes` | $M_{\mathrm{source}}$ | Sections 4.2, 5.3, 8.1.2 |
| `EffectiveNS` | $NS_{\mathrm{eff}}$ | Sections 5.3, 8.1.2 |
| `EffectiveBatch` | $\mathrm{EffectiveBatch}$ | Section 4.2 |
| `RequestedBatch` | Requested batch | Sections 4.2, 5.5 |
| `SubBatch` / `SUB_BATCH` | $\texttt{SUB\_BATCH}$ | Sections 4.2, 8.4 |
| `NumSubs` / `num_subs` | $\texttt{num\_subs}$ | Sections 4.2, 8.4 |
| `INT_MAX` | The maximum value of a 32-bit signed integer, 2,147,483,647 | Sections 4.7.3, 8.4 |
| `safe_sub_batch` | The index-safety upper bound using $\lfloor \texttt{INT\_MAX}/n\rfloor$ | Section 8.4 |

These column names and identifiers are the representations used in saved records and code, and they remain unchanged in this manuscript.

## Units and Conventions

| Notation | Meaning |
|:--|:--|
| `s` | Seconds. The unit of execution time. |
| `GB` | Decimal gigabytes ($10^{9}$ bytes). |
| `MB` | Decimal megabytes ($10^{6}$ bytes). |
| `GiB` | Binary gibibytes ($2^{30}$ bytes). |
| `MiB` | Binary mebibytes ($2^{20}$ bytes). |
| `GB/s` | Decimal gigabytes per second. Used for interconnect bandwidth. |
| `GTEPS` | $10^{9}$ traversed edges per second. |
| `b512` | An experimental shorthand denoting a requested batch size of 512. `b<number>` indicates the requested batch size. |
| `n=3`, `n=5` | Conventional labels indicating the number of trials $N_{\mathrm{trials}}$ for a given configuration. |
| `n/a` | Not applicable (the field is undefined). |
| `not recorded` | Cannot be confirmed from the saved records. |
| `N/A (failed)` | The numerical value is undefined due to execution failure. It is not treated as 0 seconds of execution time. |

Decimal units (GB, MB) and binary units (GiB, MiB) are presented alongside each other without numerical conversion and are not mixed. For GPU memory capacity, the nominal HBM3 capacity, the saved record value of the execution environment, the reported value from runtime queries, and the host-memory resource limit are not treated as the same conceptual quantity. The nominal value and the saved record value represent the same on-package HBM3 using different unit systems and acquisition methods; they are not distinct memory regions.

