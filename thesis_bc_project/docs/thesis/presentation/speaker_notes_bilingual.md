# Speaker Notes (Bilingual)

Presentation timing is intentionally not fixed.
The user will adjust slide selection and speaking time after rehearsal.

スライド面はすべて英語である。英語原稿と日本語説明は同じ主張と限定を述べる。

## Main

## Slide 1 — Design and Evaluation of a Batch-Based GPU Execution Framework for Betweenness Centrality on GH200

### Duration

[USER TO EDIT]

### English Script

The main result comes first. GPU_Opt achieved 1.31 to 3.17 times speedup on four evaluated graphs. The comparison used one GH200 GPU and median runtimes.

### 日本語説明

最初に主結果を示します。GPU_Optは評価した4グラフで1.31倍から3.17倍の高速化を達成しました。比較はGH200一台と実行時間の中央値を用いました。

### English Transition

I will outline the talk.

### 日本語トランジション

発表の流れを示します。

### Limitations to State

- The result applies only to the evaluated conditions.
- 結果は評価した条件にのみ適用されます。

### Expected Questions

- What is the comparator? PathMerge is a tuned third-party external GPU implementation.
- 比較対象は何か。PathMergeは調整済みの第三者GPU実装です。

## Slide 2 — Agenda

### Duration

[USER TO EDIT]

### English Script

I first explain BC and Brandes. I then show the GPU design, evaluation, results, and limits.

### 日本語説明

最初にBCとBrandesを説明します。その後、GPU設計、評価、結果、限界を示します。

### English Transition

First, what does BC measure?

### 日本語トランジション

まず、BCが何を測るかを説明します。

### Limitations to State

- The agenda does not fix the speaking time.
- このAgendaは発表時間を固定しません。

### Expected Questions

- Why start with results? The audience can see the outcome before the technical details.
- なぜ結果から始めるのか。技術詳細の前に成果を共有できるからです。

## Slide 3 — Betweenness Centrality Finds Important Bridge Vertices

### Duration

[USER TO EDIT]

### English Script

BC measures how often a vertex lies on shortest paths. In this path graph, C has BC four. C connects the two sides most often.

### 日本語説明

BCは頂点が最短経路上に現れる頻度を表します。このパスグラフではCのBCが4で、両側を最も多く結びます。

### English Transition

Next, I explain the exact BC algorithm.

### 日本語トランジション

次に、厳密BCアルゴリズムを説明します。

### Limitations to State

- This graph is a simple teaching example.
- このグラフは説明用の簡単な例です。

### Expected Questions

- Why is C important? More shortest paths pass through C than through any other vertex.
- なぜCが重要か。他のどの頂点よりも多くの最短経路がCを通るからです。

## Slide 4 — Brandes' Algorithm Reduces the Cost of Exact BC

### Duration

[USER TO EDIT]

### English Script

Direct path enumeration is expensive. Brandes reuses shortest-path information from each BFS. For unweighted graphs, exact BC costs O of V times V plus E.

### 日本語説明

経路を直接列挙すると高コストです。Brandesは各BFSの最短経路情報を再利用します。無重みグラフの厳密BCはO(|V|(|V|+|E|))です。

### English Transition

The next slide separates Brandes from this study's contribution.

### 日本語トランジション

次のスライドでBrandesと本研究の貢献を分けます。

### Limitations to State

- I do not improve Brandes' asymptotic complexity.
- 本研究はBrandesの漸近計算量を改善しません。

### Expected Questions

- Does this study propose a new BC algorithm? No. It keeps Brandes' mathematical algorithm.
- 本研究は新しいBCアルゴリズムを提案するのか。いいえ。Brandesの数理アルゴリズムを維持します。

## Slide 5 — I Improve How Brandes Runs on the GPU

### Duration

[USER TO EDIT]

### English Script

Brandes defines the exact computation. I improve its GPU execution. I batch sources, map blocks, switch BFS direction, cooperate within warps, and use two streams.

### 日本語説明

Brandesが厳密計算を定義します。本研究はGPU実行を改善します。始点のバッチ化、block割当、BFS方向切替、warp協調、2 streamを用います。

### English Transition

I now show the integrated framework.

### 日本語トランジション

次に統合実行基盤を示します。

### Limitations to State

- The BC definition and Brandes equations stay unchanged.
- BCの定義とBrandesの数式は変えません。

### Expected Questions

- What is new? The contribution is the integrated GPU execution framework and its evaluation.
- 何が新しいのか。統合GPU実行基盤とその評価が貢献です。

## Slide 6 — The Proposal Is an Integrated GPU Execution Framework

### Duration

[USER TO EDIT]

### English Script

The framework combines four execution components. Each component targets a different GPU bottleneck. I claim the integration and evaluation, not each component's first use.

### 日本語説明

実行基盤は4つの実行要素を組み合わせます。各要素は異なるGPUボトルネックを対象にします。個別要素の初出ではなく統合と評価を主張します。

### English Transition

Source batching also creates a memory constraint.

### 日本語トランジション

始点バッチ化はメモリ制約も生みます。

### Limitations to State

- Individual component novelty is not claimed.
- 個々の要素の新規性は主張しません。

### Expected Questions

- Why integrate the components? Their bottlenecks occur in one shared execution flow.
- なぜ統合するのか。各ボトルネックが同じ実行フロー内に現れるからです。

## Slide 7 — Source Batching Creates a Batch-Dependent Working Set

### Duration

[USER TO EDIT]

### English Script

Source batching groups source vertices. It does not split the graph. At b512 and two effective streams, the code-derived allocation estimate is about 10.67 GB.

### 日本語説明

始点バッチ化は始点頂点をまとめます。グラフを分割しません。b512と実効2 streamでは、コード由来の割当推定は約10.67 GBです。

### English Transition

Three variants manage this working set differently.

### 日本語トランジション

3つのvariantはこのworking setを異なる方法で管理します。

### Limitations to State

- The estimate is not a measured memory footprint.
- この推定値は実測メモリ使用量ではありません。

### Expected Questions

- Is source batching graph partitioning? No. The input graph stays fixed.
- 始点バッチ化はgraph partitioningか。いいえ。入力グラフは固定です。

## Slide 8 — Three Memory-Management Variants Share One Framework

### Duration

[USER TO EDIT]

### English Script

GPU_Opt uses Unified Memory. GPU_Opt_Pure uses device memory. GPU_Opt_Pure_Chunked splits the source batch. All variants share one execution framework.

### 日本語説明

GPU_OptはUnified Memory、GPU_Opt_Pureはdevice memoryを使います。GPU_Opt_Pure_Chunkedは始点バッチを分割します。全variantは同じ実行基盤を共有します。

### English Transition

I next separate the evaluation scopes.

### 日本語トランジション

次に評価範囲を分けます。

### Limitations to State

- These are memory variants, not separate proposals.
- これらはメモリvariantであり、別々の提案ではありません。

### Expected Questions

- Does Chunked split the graph? No. It splits the source batch.
- Chunkedはグラフを分割するのか。いいえ。始点バッチを分割します。

## Slide 9 — I Separate Performance and Capacity Studies

### Duration

[USER TO EDIT]

### English Script

I use four graphs for main performance. I use corrected 325557 for capacity tests. Every run uses one GH200 GPU.

### 日本語説明

主性能評価には4グラフを使います。容量試験にはcorrected 325557を使います。すべての実行はGH200一台です。

### English Transition

The first result is runtime.

### 日本語トランジション

最初の結果は実行時間です。

### Limitations to State

- Input file size is not working-set size.
- 入力ファイルサイズはworking setサイズではありません。

### Expected Questions

- Why separate the studies? Performance and capacity use different graph scopes.
- なぜ試験を分けるのか。性能と容量ではグラフ範囲が異なるからです。

## Slide 10 — GPU_Opt Reduced Runtime on All Four Evaluated Graphs

### Duration

[USER TO EDIT]

### English Script

GPU_Opt reduced median runtime on all four graphs. GPU_Opt uses fixed b512. PathMerge uses a tuned batch for each graph. The chart uses a log scale.

### 日本語説明

GPU_Optは4グラフすべてで実行時間中央値を短縮しました。GPU_Optはb512固定です。PathMergeは各グラフで調整したバッチを使います。縦軸は対数です。

### English Transition

The next slide shows the speedup ratios.

### 日本語トランジション

次に高速化率を示します。

### Limitations to State

- Trial counts are small, so I report medians.
- 試行数が少ないため中央値を報告します。

### Expected Questions

- Why use a log scale? Runtime spans a wide range across the graphs.
- なぜ対数軸を使うのか。グラフ間で実行時間の範囲が広いからです。

## Slide 11 — GPU_Opt Achieved 1.31–3.17× Speedup

### Duration

[USER TO EDIT]

### English Script

GPU_Opt achieved 3.17, 1.31, 1.51, and 1.45 times speedup. PathMerge is an external comparator, not ground truth.

### 日本語説明

GPU_Optは3.17倍、1.31倍、1.51倍、1.45倍の高速化を達成しました。PathMergeは外部比較対象でありground truthではありません。

### English Transition

I next examine the execution components.

### 日本語トランジション

次に実行要素の効果を確認します。

### Limitations to State

- I do not generalize this result to PathMerge in general.
- この結果をPathMerge一般へは一般化しません。

### Expected Questions

- Were both batch sizes tuned? No. GPU_Opt stayed at b512; PathMerge was tuned per graph.
- 両方のバッチを調整したのか。いいえ。GPU_Optはb512固定で、PathMergeは各グラフで調整しました。

## Slide 12 — Hybrid BFS and Dual Streams Gave the Largest Observed Effects

### Duration

[USER TO EDIT]

### English Script

On corrected 325557, H was 1.4767 times, W was 1.1012 times, and A was 1.5563 times. Hybrid BFS and dual streams had the largest observed effects.

### 日本語説明

corrected 325557ではHが1.4767倍、Wが1.1012倍、Aが1.5563倍でした。Hybrid BFSとdual streamsが最大の観測効果を示しました。

### English Transition

The next result concerns tested memory capacity.

### 日本語トランジション

次は試験したメモリ容量の結果です。

### Limitations to State

- The aggregate mixes checkpoints. No roadNet factor analysis was run.
- aggregateはcheckpointが混在します。roadNetの要因分析は未実施です。

### Expected Questions

- Do these factors explain roadNet speedups? No. This analysis did not use roadNet graphs.
- これらの要因はroadNetの高速化を説明するのか。いいえ。この分析ではroadNetを使っていません。

## Slide 13 — Memory Variants Expanded the Tested Batch Range

### Duration

[USER TO EDIT]

### English Script

Pure passed b4096 and failed at b8192 with CUDA OOM. UM passed b10240 and ended at b12288 with a host OOM kill. Chunked passed b16384.

### 日本語説明

Pureはb4096に成功しb8192でCUDA OOMとなりました。UMはb10240に成功しb12288でhost OOM killとなりました。Chunkedはb16384に成功しました。

### English Transition

These results have clear limits.

### 日本語トランジション

これらの結果には明確な限界があります。

### Limitations to State

- A tested upper bound is not an unlimited capacity claim.
- 試験上限は無制限の容量を意味しません。

### Expected Questions

- Are failures zero-second runtimes? No. They are categorical failure outcomes.
- 失敗は0秒の実行時間か。いいえ。カテゴリとしての失敗結果です。

## Slide 14 — The Results Have Clear Limits

### Duration

[USER TO EDIT]

### English Script

The evidence uses one GH200, four main graphs, one capacity graph, and small trial counts. I do not generalize beyond these conditions.

### 日本語説明

根拠はGH200一台、主性能4グラフ、容量1グラフ、少数試行に基づきます。これらの条件を超えて一般化しません。

### English Transition

Within these limits, I summarize four contributions.

### 日本語トランジション

この限界の範囲で4つの貢献をまとめます。

### Limitations to State

- PathMerge is a retained third-party snapshot.
- PathMergeは保存された第三者snapshotです。

### Expected Questions

- Does the result cover other GPUs? No. Only one GH200 was evaluated.
- 他のGPUも対象か。いいえ。評価したGPUはGH200一台です。

## Slide 15 — Contributions

### Duration

[USER TO EDIT]

### English Script

I contribute an integrated exact-BC GPU framework, an external comparison, component measurements, and a tested capacity comparison. Correctness remains required validation, not a main contribution.

### 日本語説明

貢献は厳密BCの統合GPU基盤、外部比較、要素効果の測定、試験容量の比較です。正確性は必要な検証であり、主要貢献ではありません。

### English Transition

I will close with the result and scope.

### 日本語トランジション

最後に結果と適用範囲を述べます。

### Limitations to State

- No new BC algorithm is claimed.
- 新しいBCアルゴリズムは主張しません。

### Expected Questions

- What is the central contribution? The integration and evaluation of GPU execution methods.
- 中心的な貢献は何か。GPU実行方法の統合と評価です。

## Slide 16 — The Framework Improved Performance and Expanded the Tested Batch Range

### Duration

[USER TO EDIT]

### English Script

The framework achieved 1.31 to 3.17 times speedup. Hybrid BFS and dual streams had the largest observed effects. Memory variants changed the tested feasible batch range.

### 日本語説明

実行基盤は1.31倍から3.17倍の高速化を達成しました。Hybrid BFSとdual streamsが最大の観測効果を示しました。メモリvariantは試験上の実行可能バッチ範囲を変えました。

### English Transition

Questions are welcome. Backup evidence follows.

### 日本語トランジション

質問を受けます。以降はBackup資料です。

### Limitations to State

- The conclusions apply only to the evaluated conditions.
- 結論は評価した条件にのみ適用されます。

### Expected Questions

- What should be tested next? More GPUs, graphs, trials, and roadNet factor analysis.
- 次に何を試験すべきか。GPU、グラフ、試行数の拡大とroadNetの要因分析です。

## Backup

Backup は質疑応答用である。

## Slide 17 — Detailed Experimental Environment

### Duration

[USER TO EDIT]

### English Script

This backup lists the hardware, software, bandwidth records, and aggregation method. The main experiments use one GH200 GPU.

### 日本語説明

このBackupはhardware、software、帯域記録、集計方法を示します。主実験はGH200一台を使います。

### English Transition

The next backup lists graph and batch parameters.

### 日本語トランジション

次のBackupはグラフとバッチの設定です。

### Limitations to State

- Undetermined items stay undetermined.
- 未確定項目は未確定のままです。

### Expected Questions

- Why are two memory capacity values shown? They come from different records and units.
- なぜメモリ容量値が2つあるのか。異なる記録と単位に由来するからです。

## Slide 18 — Graph and Batch Parameters

### Duration

[USER TO EDIT]

### English Script

This backup lists graph batches, medians, trial counts, and the working-set formula. The per-source state is code-derived, not measured.

### 日本語説明

このBackupはグラフ別バッチ、中央値、試行数、working set式を示します。始点ごとの状態量はコード由来で、実測ではありません。

### English Transition

The next backup shows PathMerge tuning.

### 日本語トランジション

次のBackupはPathMergeの調整を示します。

### Limitations to State

- GPU_Opt stays at b512 on every main graph.
- GPU_Optは主性能の全グラフでb512固定です。

### Expected Questions

- Why not tune GPU_Opt? The fixed setting avoids a proposal-favoring comparison.
- なぜGPU_Optを調整しないのか。提案側に有利な比較を避けるためです。

## Slide 19 — PathMerge Batch-Size Sweep

### Duration

[USER TO EDIT]

### English Script

This sweep selects the tested PathMerge batch for each graph. The historical malformed graph is excluded.

### 日本語説明

このsweepは各グラフのPathMergeバッチを選びます。履歴的なmalformed graphは除外します。

### English Transition

The next backup compares two kernels.

### 日本語トランジション

次のBackupは2つのkernelを比較します。

### Limitations to State

- The sweep supports only the tested batch points.
- sweepが裏づけるのは試験点だけです。

### Expected Questions

- Could another batch be faster? It is possible outside the tested points.
- 別のバッチがより速い可能性はあるか。試験点の外ではあり得ます。

## Slide 20 — Forced Block-vs-Shared Kernel Comparison

### Duration

[USER TO EDIT]

### English Script

This backup forces block and shared kernels on PA and TX. The block kernel was faster on both tested graphs.

### 日本語説明

このBackupはPAとTXでblock kernelとshared kernelを強制比較します。試験した両グラフでblock kernelが高速でした。

### English Transition

The next backup shows phase timing.

### 日本語トランジション

次のBackupはphase timingを示します。

### Limitations to State

- No selector rule is inferred for unmeasured graphs.
- 未測定グラフのselector ruleは導きません。

### Expected Questions

- Can shared memory win elsewhere? It was not tested on other graphs.
- 他のグラフでshared memoryが勝つか。他のグラフでは試験していません。

## Slide 21 — Phase Breakdown and Profiling Scope

### Duration

[USER TO EDIT]

### English Script

This backup separates BFS, backward accumulation, and other time. The values come from complete b512 wall-clock runs.

### 日本語説明

このBackupはBFS、backward accumulation、otherの時間を分けます。値は完全なb512 wall-clock runから得ています。

### English Transition

The next backup gives detailed correctness validation.

### 日本語トランジション

次のBackupは詳細な正確性検証です。

### Limitations to State

- The phase values are not partial Nsight totals.
- phase値は部分的なNsight合計ではありません。

### Expected Questions

- Why use wall-clock data? A partial trace does not cover the complete run.
- なぜwall-clock dataを使うのか。部分traceは完全な実行を覆わないからです。

## Slide 22 — Detailed Correctness Evidence

### Duration

[USER TO EDIT]

### English Script

Correctness is required validation, not the main research result. All 13 comparisons passed mixed tolerance. Tier A has three independent CPU comparisons. Tier B has ten path comparisons.

### 日本語説明

正確性は必要な検証であり、主要な研究成果ではありません。13比較はすべてmixed toleranceに合格しました。Tier Aは独立CPU比較3件、Tier Bは実装経路比較10件です。

### English Transition

The final backup preserves the malformed-input history.

### 日本語トランジション

最後のBackupはmalformed inputの履歴を保存します。

### Limitations to State

- Tier B is not an independent ground-truth evaluation.
- Tier Bは独立したground truth評価ではありません。

### Expected Questions

- Was correctness checked? Yes. The detailed table retains all validation data.
- 正確性は確認したのか。はい。詳細表にすべての検証データを保持しています。

## Slide 23 — Historical Record of the Malformed Input

### Duration

[USER TO EDIT]

### English Script

This backup preserves the malformed-input record. The old input failed validation. The corrected input passed all 13 comparisons. Current conclusions use only the corrected input.

### 日本語説明

このBackupはmalformed inputの記録を保存します。旧入力は検証に失敗しました。修正入力は13比較すべてに合格しました。現在の結論は修正入力だけを使います。

### English Transition

This ends the backup material.

### 日本語トランジション

Backup資料は以上です。

### Limitations to State

- Historical evidence is not part of the current results.
- 履歴的根拠は現在の結果の一部ではありません。

### Expected Questions

- Why keep the old result? It documents input validation and correction.
- なぜ旧結果を残すのか。入力検証と修正の経緯を記録するためです。
