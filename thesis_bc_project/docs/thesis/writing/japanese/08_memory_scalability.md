# Chapter 8 Memory Scalability

本章では RQ3 に回答する。評価対象は修正版 `325557_3216152_corrected_v1`（$n=325{,}557$、$m=3{,}216{,}152$）である。正式結果は targeted boundary validation に基づき、各条件 1 試行である。これは sweep や方式間の性能比較ではなく、試験した境界条件での feasibility 評価である。実行識別子と入力 SHA256 は Appendix A の provenance に示す。

<!-- Source note (internal): corrected graph SHA256 8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22; PBS job 2404743; checkpoint 45352a3. -->

## 8.1 Capacity Terms and Evaluation Scope

### 8.1.1 Static Graph Sizes

入力グラフのファイル容量と GPU working set を区別する。修正版 325557 の static storage を Table 8.1 に示す。

**Table 8.1: Static storage and per-source state for the corrected 325557 graph.**

| Quantity | Definition | Bytes | Decimal Size | Binary Size |
|---|---|---:|---:|---:|
| Input graph file | On-disk text CSR measured by `stat` | 45,348,105 | 45.35 MB | 43.25 MiB |
| CSR topology | $((n+1)+2m)\times4$ | 27,031,448 | 27.03 MB | 25.78 MiB |
| BC output vector | $n\times8$ | 2,604,456 | 2.60 MB | 2.48 MiB |
| Per-source state $M_{\mathrm{source}}$ | $32n+4D_{est}+8$, $D_{est}=256$ | 10,418,856 | 10.42 MB | 9.94 MiB |

<!-- Source note (internal): result/datasets/graph_catalog.tsv; raw_data/corrected_325557/job_2404743/implementation_manifest.tsv; src/proposed/host_pure.cu:141-157. -->

入力ファイル、CSR topology、BC output vector はいずれも公称 96 GB の HBM3 を超えない。容量問題を作るのは、source ごとの距離、最短経路数、依存度、frontier、stack などの状態配列を複数 source について同時保持することによる batch-dependent working set である。

### 8.1.2 Batch-Dependent Working Set

実装と保存 manifest に従い、code-derived allocation estimate を次で定義する。

$$
M_{\mathrm{work}}=NS_{\mathrm{eff}}\times \mathrm{EffectiveBatch}\times M_{\mathrm{source}}.
$$

保存 manifest の記録列 `EffectiveNS` と `PerSourceStateBytes` は、本論文ではそれぞれ $NS_{\mathrm{eff}}$ と $M_{\mathrm{source}}$ と表す。Chunked の同時 resident estimate では $\mathrm{EffectiveBatch}$ の代わりに `SUB_BATCH` を用いる。この値はコードの配列寸法から導いた estimate であり、measured process RSS、measured physical HBM residency、measured migration bytes ではない。これら 3 量は本実験では取得していない。

batch は graph partition ではない。各 batch は全 source 集合の一部を grouping する実行単位であり、outer loop が全 batch を処理する。Chunked ではさらに各 source batch を sub-batch に分けるが、`num_subs` 回の反復により batch 内の全 source を処理する。したがって、batch/sub-batch は BC を近似せず、source を省略しない。

### 8.1.3 HBM3 Capacity Reference

GH200 のオンパッケージ HBM3 は公称 96 GB である [@nvidiaGraceHopperInDepth]。保存された runtime query は total 約 102.0 decimal GB、実行前の `free_before` 約 101.4 decimal GB を記録する。公称値と runtime の decimal 表示は同一 HBM3 の異なる単位系・取得方法であり、別のメモリ階層ではない。本章の code-derived estimate との比較には、同じ保存記録系の `free_before` 約 101.4 GB を用いる。

## 8.2 Memory-Management Variants

GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunked は独立した 3 アルゴリズムではない。共通 GPU 実行基盤と同じ全始点 Brandes 計算を用い、memory management のみを変える。

**Table 8.2: Memory-management variants of the common GPU execution framework.**

| Implementation | Allocation Strategy | Purpose | Finite Limit |
|---|---|---|---|
| GPU_Opt_Pure | Device allocation with `cudaMalloc` | Device-only control | CUDA device-memory capacity |
| GPU_Opt | Managed allocation with `cudaMallocManaged` | Handle a working set that may exceed device memory through managed placement and migration | HBM3, host memory, cgroup, runtime resources |
| GPU_Opt_Pure_Chunked | Device allocation for one source sub-batch | Bound the simultaneously resident working set | Sub-batch buffers and all remaining finite resources |

UM の目的は大きな入力グラフを分割して格納することではない。約 45.35 MB の入力に対し、batch に比例して増える managed state allocation を扱うために使用する。Chunked の目的は source batch を sub-batch に分け、同時 resident state を明示的に制限することである。

## 8.3 Corrected Targeted Boundary Validation

Table 8.3 と Figure 8.1 に targeted boundary validation の 5 条件を示す。failure は 0 秒ではなく独立した status として表す。

**Table 8.3: Targeted memory-feasibility boundary on the corrected 325557 graph. Each condition was run once.**

| Implementation | Requested Batch | Code-Derived Working Set | Outcome | Failure Class | Runtime [s] |
|---|---:|---:|---|---|---:|
| GPU_Opt_Pure | 4096 | 85,351,268,352 bytes (85.35 GB) | Success | None | 65.89 |
| GPU_Opt_Pure | 8192 | 170,702,536,704 bytes (170.70 GB) | Failure | CUDA device-memory OOM, exit 1 | N/A |
| GPU_Opt | 10240 | 106,689,085,440 bytes (106.69 GB) | Success | None | 238.67 |
| GPU_Opt | 12288 | 128,026,902,528 bytes (128.03 GB) | Failure | Cgroup host-memory OOM kill, exit 137 | N/A |
| GPU_Opt_Pure_Chunked | 16384 | 68,722,774,176 bytes (68.72 GB) resident estimate | Success | None | 66.60 |

<!-- Source note (internal): result/tables/thesis/T4_memory_scalability.tsv; result/memory_scalability/corrected_325557/feasibility_boundary.tsv; raw_data/corrected_325557/job_2404743/{implementation_manifest,feasibility_results,oom_evidence}.tsv. -->

![Figure 8.1: Corrected 325557 targeted memory-feasibility boundary](../../../../result/figures/thesis/memory_scalability_325557.png)

**Figure 8.1: Targeted feasibility boundary on the corrected 325557 graph (one trial per condition). Failure markers distinguish CUDA device-memory out-of-memory from a cgroup host-memory OOM kill; failed runs are not plotted as zero-second runtimes.**

Pure b8192 では `host_pure.cu:144` の `cudaMalloc` が `out of memory` を返し、`oom_evidence=cuda_oom`、runner exit 1 が保存されている。これは CUDA device-memory OOM である。

UM b10240 は $NS_{\mathrm{eff}}=1$ の estimate 106.69 GB が run-start `free_before` 約 101.4 GB を上回る条件で成功した。これは入力ファイルが大きいためではなく、batch-dependent managed allocation が free HBM を超え得る領域に達した条件での成功である。UM のコードは managed placement と migration を使用するが、本研究は物理 residency や migration bytes を測定していないため、成功時の物理配置量を断定しない。

UM b12288 は runner exit 137、SIGKILL として記録され、cgroup host-memory OOM kill に分類される。CUDA OOM 文字列はなく、`oom_evidence=none` である。したがって、これを CUDA/HBM OOM と記述しない。UM はこの条件で無制限ではなかった。

最大成功 batch の観測順序は試験範囲内で Pure b4096、UM b10240、Chunked b16384 である。ただし、各点 1 試行の targeted validation であり、これらを一般的な capacity limit や runtime ranking として扱わない。

## 8.4 Chunked Source Sub-Batches

Chunked b16384 は `SUB_BATCH=6596`、`num_subs=3`、$NS_{\mathrm{eff}}=1$ で成功した。同時 resident estimate は

$$
6596\times10{,}418{,}856=68{,}722{,}774{,}176\ \mathrm{bytes}
$$

である。`SUB_BATCH=6596` は HBM3 容量だけから決めた値ではない。`host_chunked.cu` は HBM budget 由来上限と、index overflow を防ぐ

$$
safe\_sub\_batch=\left\lfloor\frac{INT\_MAX}{n}\right\rfloor
=\left\lfloor\frac{2{,}147{,}483{,}647}{325{,}557}\right\rfloor=6596
$$

の小さい方を採る。修正版 325557 では保存された `free_before` とコード式による HBM budget 上限が整数除算で 7783 source であるのに対し、index-safety 上限が 6596 であり、後者が binding constraint であった。この観測を異なる $n$ や未測定 GPU へ一般化しない。

3 sub-batch は source offset を進めながら同じ buffer を再利用する。これは graph edge set の分割近似ではなく、要求 batch に含まれる全 source の exact BC 処理である。外側の batch loop も全 source を処理する。

## 8.5 Performance Interpretation and Measurement Limits

Table 8.3 の runtime は feasibility に付随する single-run wall-clock time である。trial aggregation がなく、batch、allocation mode、sub-batch 数が異なるため、方式間の正式な性能比較に用いない。UM b10240 の 238.67 s から migration cost の大きさを逆算せず、Pure/Chunked の約 66 s との速度順位を headline claim にしない。

本研究で取得していない量は次のとおりである。

- Full-run process RSS
- Physical HBM residency
- Total host residency
- Full-run migration bytes

25 秒の部分 trace は別条件の補助資料であり、formal boundary validation の full-run migration total ではない。

## 8.6 Historical Malformed-Input Results

旧 `325557_3216152` 上の legacy sweep、`CORE_FAIL`、および `OOM_OR_FAIL` 表記は削除せず historical invalid-input evidence として archive に保持する。旧入力は malformed であり、current RQ3 の boundary は修正版入力の formal validation で置き換える。historical result と current formal result の failure class、checkpoint、resource condition を混合しない。

<!-- Source note (internal): historical canonical job 2368587; current corrected-input job 2404743. -->

## 8.7 Answer to RQ3

RQ3 は `SUPPORTED_WITH_LIMITATIONS` と回答する。修正版 325557 の targeted boundary validation では、Pure は b4096 で成功し b8192 で CUDA device-memory OOM、UM は b10240 で成功し b12288 で cgroup host-memory OOM kill、Chunked は `SUB_BATCH=6596`・`num_subs=3` により tested upper bound の b16384 で成功した。UM は managed allocation により Pure より大きい batch を扱い、Chunked は同時 resident working set を制限して試験範囲をさらに拡張した。

この結論は修正版 325557、GH200、各条件 1 試行に限定される。各条件の実行時間は記録値であり、方式間の正式な性能比較には用いない。したがって Chunked の価値を最高性能としては主張しない。UM も無制限ではなく、Chunked が未測定条件で OOM を回避する保証もない。主な設計上の含意は、容量問題が約 45.35 MB の input graph file ではなく `batch × per-source state` から生じ、source grouping と resident-state management が容量拡張の制御点になることである。
