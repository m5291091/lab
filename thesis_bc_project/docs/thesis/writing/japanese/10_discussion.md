# Chapter 10 Discussion

本章では RQ1〜RQ4 の関係を統合し、性能、最適化要因、容量、数値整合性の相互関係を論じる。観測事実と未検証の解釈を区別し、修正版 325557 の結果を旧 malformed input の historical evidence と混同しない。

## 10.1 Integrated Interpretation of the Research Questions

RQ1 の主性能結果は変更しない。固定 b512 の block GPU_Opt は、評価した第三者実装の tuned PathMerge に対し、email-EuAll で 3.17 倍、roadNet-PA で 1.31 倍、roadNet-TX で 1.51 倍、roadNet-CA で 1.45 倍高速であった。修正版 325557 はこの主性能比較に含まれない。

RQ2 は性能差の内部要因を、RQ1 とは別のグラフ集合で評価した。修正版 325557 の主効果は H=1.4767、W=1.1012、A=1.5563、合成 4 グラフ集約は H=1.679、W=1.066、A=1.391 である。後者は他 3 グラフが job 2354994、修正版 325557 が job 2406254 / checkpoint `45352a3` の mixed-checkpoint 集約である。Hybrid BFS と Dual-Stream Execution の正の効果、Warp-Cooperative Accumulation の graph dependence は評価条件内の観測であり、roadNet の RQ1 speedup をこのアブレーションだけで因果分解しない。

RQ3 は同じ計算基盤の memory-management variants を比較した。修正版 325557 の targeted boundary では Pure b4096、UM b10240、Chunked b16384 が成功し、Pure b8192 は CUDA device-memory OOM、UM b12288 は cgroup host-memory OOM kill であった。各条件 1 試行であり、runtime を方式間性能の formal ranking に用いない。

RQ4 は 2 tier である。Tier A は小規模 3 グラフの独立 Sequential CPU 参照比較が `SUPPORTED`、Tier B は修正版 325557 の 10 cross-implementation 比較が `SUPPORTED_WITH_LIMITATIONS` である。全 13 比較は mixed tolerance 内で PASS したが、すべて `ByteIdentical=No` である。

## 10.2 Performance and Graph Characteristics

email-EuAll と roadNet の speedup 差は、グラフ構造の違いと整合する。email-EuAll はハブを含み BFS depth が浅い一方、roadNet は低次数で探索が深い。email-EuAll では Dual-Stream Execution の main effect が 1.720 と大きく、initialization と kernel execution の overlap が end-to-end time に寄与した可能性がある。一方、roadNet で H/W/A factorial ablation は実施していないため、roadNet の speedup 差の原因を断定しない。

Warp-Cooperative Accumulation は email-EuAll で 0.970、56438 で 0.992、benchmark_7000 で 1.175、修正版 325557 で 1.1012 であった。隣接走査量や lane utilization が効果に影響する可能性はあるが、専用 hardware counter による因果検証はない。したがって、平均次数や次数分布だけから W の効果を予測する一般則は提示しない。

## 10.3 The Capacity Problem Is Batch-Dependent State

修正版 325557 の input graph file は 45,348,105 bytes、CSR topology は 27,031,448 bytes、BC output vector は 2,604,456 bytes である。これらは HBM3 容量を超えない。容量問題を作るのは、1 source あたり 10,418,856 bytes の state（$M_{\mathrm{source}}$）を同時 source 数だけ複製する code-derived working set である。

$$
M_{\mathrm{work}}=NS_{\mathrm{eff}}\times \mathrm{EffectiveBatch}\times M_{\mathrm{source}}.
$$

この関係は設計上重要である。入力 graph file を小さくしても batch-dependent state が支配的なら容量問題は残る。逆に、source grouping と同時 resident 数を制御すれば、graph topology を partition した近似計算に変えずに容量を調整できる。batch/sub-batch は全 source を反復処理する exact execution unit であり、graph partition や source sampling ではない。

## 10.4 Unified Memory and Chunked Execution

UM の利点は managed allocation により device memory を超え得る working set を扱えることである。修正版 325557 の UM b10240 は code-derived estimate 106.69 GB が run-start free HBM 約 101.4 GB を上回る条件で成功した。しかし b12288 は cgroup host-memory OOM kill であり、UM は有限の host memory、cgroup、runtime resource に依存する。UM の目的を「96 GB を超える graph file の格納」と表現しない。

Chunked は source batch を `SUB_BATCH=6596` の sub-batch に分け、b16384 に対する同時 resident estimate を 68.72 GB に制限した。`SUB_BATCH=6596` は HBM budget だけでなく `INT_MAX/n` の index-safety upper bound により決まり、修正版 325557 では後者が binding constraint であった。Chunked の利点は最高速度ではなく explicit resident-working-set control と tested capacity extension である。未試行条件で OOM を回避する保証はない。

UM と Chunked の選択は単純な優劣ではない。UM は managed placement の簡潔さを持つが host-memory pressure を受ける。Chunked は resident amount を制御できるが sub-batch loop と launch control を要する。本研究は measured RSS、physical HBM residency、full-run migration bytes を取得していないため、両者の物理配置量や migration total を定量比較しない。

## 10.5 Correctness, Numerical Consistency, and Provenance

Tier A の独立参照 PASS は、小規模 3 グラフにおける GPU_Opt の full-vector numerical validity を支持する。Tier B の 10 PASS は、修正版 325557 で UM/Pure/Chunked/PathMerge の異なる batch・memory path が混合許容内で整合したことを示す。ただし、Tier B は独立 reference ではなく、PathMerge も ground truth ではない。

全 13 比較が `ByteIdentical=No` であることは、floating-point accumulation order の違いを許容する判定設計と整合する。ここから特定の差の原因を断定しない。混合許容内 PASS と bitwise identity を区別することが、性能・容量の変更を評価する際の construct validity に必要である。

修正版 325557 は決定的に再構成されたが、元 seed・上流の完全な原本が未確認である。したがって Tier B の結論には provenance limitation が残る。旧 malformed 入力上の `CORE_FAIL` は current conclusion から除外する一方、failure archive と canonical job 2368587 を historical invalid-input evidence として保存する。この履歴は、入力 validation と再検証手続きの重要性を示す。

## 10.6 Threats to Validity

### 10.6.1 Internal Validity

RQ2 の合成 4 集約は mixed-checkpoint であり、same-checkpoint four-graph remeasurement ではない。修正版 325557 の RQ3/RQ4 は job 2404743、RQ2 は job 2406254 である。manifest、checkpoint、job ID を明示し、異なる実験群を同一 run として扱わない。

OOM class は保存 evidence に従う。Pure b8192 の CUDA error と UM b12288 の cgroup host-memory OOM kill を区別する。failure を 0 秒の runtime として扱わない。

### 10.6.2 External Validity

RQ1 は 4 グラフ、RQ2 は synthetic 4 + email、RQ3/Tier B は修正版 325557 の 1 グラフ、Tier A は小規模 3 グラフに限定される。評価 GPU は GH200 1 台である。1 graph、1 trial の targeted boundary を他 graph、他 GPU、異なる host-memory limit へ一般化しない。

### 10.6.3 Construct Validity

RQ3 の working-set values は code-derived allocation estimates であり measured memory usage ではない。process RSS、physical residency、migration bytes が未取得である。RQ4 の PASS は mixed tolerance 内の数値整合性で、bitwise identity ではない。

### 10.6.4 Baseline and Data Validity

PathMerge は第三者実装の external comparator であり原著者公式実装でも ground truth でもない。修正版 graph は内部再構成データで元 seed・上流原本未確認である。旧 malformed result を current evidence に混入させない。

## 10.7 Future Work

今後の課題は、headline graph の独立 full-vector reference、追加 graph と他 GPU での boundary validation、複数 trial の容量評価、full-run RSS/HBM residency/migration measurement、同一 checkpoint での synthetic-4 ablation remeasurement、上流原本または生成 seed による修正版 graph provenance の強化である。これらは現行結果の制約を緩和するための課題であり、本論文の current claims を事後に拡張しない。
