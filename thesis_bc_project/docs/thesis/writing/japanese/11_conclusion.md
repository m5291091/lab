# Chapter 11 Conclusion

## 11.1 Summary

本研究は、NVIDIA GH200 上で無向・非重みグラフの exact all-sources Betweenness Centrality を計算する batch-based GPU execution framework を設計し、性能、最適化要因、memory scalability、correctness を評価した。GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunked は独立した 3 手法ではなく、共通 GPU execution framework の memory-management variants である。

## 11.2 Answers to the Research Questions

### 11.2.1 RQ1 Performance

RQ1 は `SUPPORTED` である。固定 b512 の block-based GPU_Opt は、評価した第三者実装の tuned PathMerge に対し、email-EuAll で 3.17 倍、roadNet-PA で 1.31 倍、roadNet-TX で 1.51 倍、roadNet-CA で 1.45 倍高速であった。結果は評価した 4 グラフと GH200 に限定される。修正版 325557 は RQ1 の主性能比較に使用していない。

### 11.2.2 RQ2 Optimization Contributions

RQ2 は `SUPPORTED_WITH_LIMITATIONS` である。修正版 325557 の main effect は Hybrid BFS 1.4767、Warp-Cooperative Accumulation 1.1012、Dual-Stream Execution 1.5563 であった。合成 4 グラフ集約はそれぞれ 1.679、1.066、1.391 である。ただしこの集約は、グラフにより測定系列が異なる mixed-checkpoint aggregate である。Hybrid BFS と Dual-Stream Execution が主要な正の効果を示し、Warp-Cooperative Accumulation の効果は graph-dependent であったが、未測定の roadNet 系グラフへ因果を一般化しない。

### 11.2.3 RQ3 Memory Scalability

RQ3 は `SUPPORTED_WITH_LIMITATIONS` である。修正版 325557 の targeted boundary validation では、Pure b4096 は成功し b8192 は CUDA device-memory OOM、UM b10240 は成功し b12288 は cgroup host-memory OOM kill、Chunked b16384 は `SUB_BATCH=6596`・`num_subs=3` で成功した。容量制約は 45.35 MB の input graph file ではなく、10,418,856 bytes の per-source state を同時 source 数だけ保持する working set に起因する。各条件 1 試行の feasibility 評価であり、runtime は方式間の正式な性能比較ではない。UM は無制限ではなく、Chunked が未測定条件で OOM を回避する保証もない。

### 11.2.4 RQ4 Correctness and Numerical Behavior

RQ4 の Tier A は `SUPPORTED` である。小規模 3 グラフの Sequential CPU independent reference と GPU_Opt の full-vector 比較はすべて PASS した。Tier B は `SUPPORTED_WITH_LIMITATIONS` であり、修正版 325557 の 6 vector・10 cross-implementation comparisons はすべて MissingIndices=0、MismatchedElements=0、ToleranceResult=PASS であった。全 13 比較は `ByteIdentical=No` であるため、結論は mixed tolerance 内の numerical consistency であって bitwise identity ではない。PathMerge は external comparator であり independent ground truth ではなく、修正版 graph にも provenance と adjacency representation の制約が残る。

## 11.3 Contributions

本研究の貢献は、既存の GPU 最適化要素を GH200 向け batch-based exact BC framework として統合したこと、tuned third-party PathMerge に対する評価範囲内の性能を示したこと、H/W/A の寄与を graph-dependent な制約とともに定量化したこと、そして memory capacity と numerical consistency を failure class・provenance を含めて分離評価したことである。

## 11.4 Final Remarks

旧 malformed `325557_3216152` 上の `CORE_FAIL`、stress mismatch、PathMerge difference は historical invalid-input evidence として保存するが、current active conclusion には用いない。本研究の結論は、評価したグラフ、GH200、保存された実装 snapshot、trial 条件に限定される。この範囲で、batch-dependent source-local state を明示的に管理する共通 framework が、性能、容量、数値挙動を分離して評価する基盤となることを示した。
