# Speaker Notes (Bilingual)

スライド面はすべて英語である。日本語はこのノートにのみ存在する。

各スライドは **完全な英語スクリプト** と **完全な日本語説明** の両方を持つ。この二つは一本の読み上げ原稿の前半・後半ではなく、発表言語に応じて **どちらか一方だけを読む** ための代替である。両方を続けて読むことは想定していない。

本編 15 枚の想定時間合計は 900 秒（15.0 分）である。英語版のみを読んだ場合の推定合計は約 802 秒、日本語版のみを読んだ場合の推定合計は約 830 秒であり、いずれか一方だけで発表が成立する。

推定は英語 2.5 words/秒、日本語 5.5 文字/秒の発表ペースによる計画値であり、実測ではない。

発表時間 15 分はリポジトリに公式指定がないための暫定値であり、`scripts/generate_thesis_presentation.py` の `TALK_MINUTES` と各スライドの想定秒数で調整する。

同じ内容は PPTX のノートペインにも埋め込まれており、スライド面には表示されない。

## Main

## Slide 1 — Design and Evaluation of a Batch-Based GPU Execution Framework for Betweenness Centrality on GH200

### Duration

25 seconds

### English Script

Thank you for the introduction. I will present the design and evaluation of a batch-based GPU execution framework for betweenness centrality on the GH200. The target is exact all-sources betweenness centrality, not an approximation, and every number I report was measured on a single GH200 GPU. I will cover the problem, the proposed framework, four evaluation axes, and the boundaries of the evidence.

### 日本語説明

ご紹介ありがとうございます。本日は、GH200上の媒介中心性計算に向けたバッチ型GPU実行基盤の設計と評価について発表します。対象は近似ではなく厳密な全始点媒介中心性であり、報告する数値はすべてGH200一台で測定したものです。問題設定、提案基盤、四つの評価軸、根拠の限界の順で説明します。

### English Transition

Let me begin with why exact all-sources betweenness centrality is expensive.

### 日本語トランジション

まず、厳密な全始点媒介中心性がなぜ計算コストが高いのかから説明します。

### Limitations to State

- All results come from a single GH200 GPU.
- すべての結果はGH200一台での測定に基づく。

### Expected Questions

- Where does this sit relative to existing GPU BC work? It integrates known components and evaluates them under consistent conditions; it claims no novelty for any single component.
- 既存のGPU BC研究との位置づけは。既知の要素を統合し一貫した条件で評価したものであり、個々の要素の初出性は主張しない。

## Slide 2 — Exact All-Sources BC Is Computationally Expensive

### Duration

55 seconds

### English Script

Betweenness centrality measures how often a vertex lies on shortest paths between other vertices. Computing it exactly uses Brandes' algorithm: from every source vertex we run a breadth-first search, then accumulate dependencies in reverse level order, and add the result into the global betweenness values. Because that entire loop repeats for every vertex, the cost is big O of |V| times, in parentheses, |V| plus |E|. Beyond the raw cost, the work is irregular. The breadth-first frontier changes size at every level, and skewed degree distributions mean a fixed parallel granularity either starves threads or serializes them. That combination of high cost and irregular load is what motivates a GPU execution framework rather than a single optimized kernel.

### 日本語説明

媒介中心性は、ある頂点が他の頂点間の最短経路上にどれだけ現れるかを表す指標です。厳密に求めるにはBrandesのアルゴリズムを用い、各始点で幅優先探索を行い、レベルの逆順に依存度を累積し、結果を全体の媒介中心性へ加算します。これを全頂点で繰り返すため、計算量はラージオー、|V|かける、括弧、|V|＋|E|となります。さらに負荷が不規則です。フロンティアサイズはレベルごとに変化し、次数分布の偏りにより、並列化の粒度を固定するとスレッドが遊ぶか直列化します。計算コストの高さと負荷の不規則性が、単一カーネルではなく実行基盤を必要とする理由です。

### English Transition

Given that cost, GPU parallelism is the natural response, but it raises a second question.

### 日本語トランジション

このコストを踏まえるとGPU並列化が自然な対応ですが、そこで第二の問題が生じます。

### Limitations to State

- The target is exact computation, not approximation.
- 対象は近似ではなく厳密計算である。

### Expected Questions

- Why not use an approximation algorithm? The thesis targets exact all-sources BC, so approximate methods are out of scope.
- なぜ近似ではなく厳密計算か。本研究は厳密な全始点BCを対象としており、近似手法は範囲外である。

## Slide 3 — Performance Alone Is Not Enough

### Duration

60 seconds

### English Script

Source batching, which groups many source vertices and processes them together, raises parallelism. But it also holds per-source state for every concurrent source, so memory capacity becomes a constraint that grows with the batch size. That means runtime alone cannot tell us whether an execution framework is sound. We need to know which execution components actually contributed, how far the feasible batch range extends before the run fails, and whether the betweenness vectors that come out are numerically valid. These four axes, performance, component contribution, memory scalability, and numerical correctness, are the four research questions of this work, and I evaluate all four on the same framework under consistent conditions. The other three axes are not decoration around the performance number; they are what makes it interpretable.

### 日本語説明

始点をまとめて処理するsource batchingは並列性を高めます。しかし実行中の各始点の状態を保持するため、バッチサイズとともにメモリ容量が制約になります。つまり実行時間だけでは実行基盤として妥当か判断できません。どの実行要素が寄与したのか、失敗するまでバッチをどこまで大きくできるのか、出力されるBCベクトルが数値的に妥当かを知る必要があります。性能、要因寄与、メモリ容量、数値的正確性というこの四観点が本研究の四つのResearch Questionであり、同一基盤に対し一貫した条件で四つすべてを評価します。残る三観点は性能値の装飾ではなく、それを解釈可能にするものです。

### English Transition

With those four axes fixed, let me show the framework itself.

### 日本語トランジション

この四つの評価軸を踏まえて、実行基盤そのものを説明します。

### Limitations to State

- A runtime number without capacity and correctness evidence does not establish a framework as sound.
- 容量と正確性の根拠を欠いた実行時間だけでは、実行基盤としての妥当性は示せない。

### Expected Questions

- How do the four axes relate? Capacity bounds which batch sizes performance can even be measured at, and correctness bounds whether any of those runs are meaningful.
- 四つの観点はどう関係するか。容量はどのバッチで性能を測定できるかを規定し、正確性はその実行結果が意味を持つかを規定する。

## Slide 4 — The Proposal Is an Integrated GPU Execution Framework

### Duration

85 seconds

### English Script

The proposal is a single GPU execution framework that integrates four components into one execution flow. First, block-based source assignment: each source vertex in the batch is mapped to one thread block, so the per-source traversal state stays block-local. Second, hybrid breadth-first search, which switches between top-down and bottom-up expansion according to frontier size, so that the wide levels in the middle of a traversal do not dominate. Third, warp-cooperative accumulation, which lets a warp cooperate on the reverse dependency accumulation instead of leaving one thread per vertex. Fourth, dual-stream execution, where two CUDA streams each hold independent source-local buffers and overlap initialization with computation. I want to be explicit about the claim here: none of these four techniques is claimed as new. The contribution is that they are integrated into one coherent framework and evaluated together under consistent conditions. The diagram shows the two streams running that same pipeline in parallel, each merging its partial results into the global betweenness array in the middle. The buffers are source-local, which is the property the overlap is designed around.

### 日本語説明

提案手法は、四つの実行要素を一つの実行フローへ統合したGPU実行基盤です。第一にblock-based source assignment。各始点を一スレッドブロックへ割り当て、探索状態をブロック内に閉じます。第二にHybrid BFS。フロンティアサイズに応じてtop-downとbottom-upを切り替え、中盤の広いレベルの支配を避けます。第三にWarp-Cooperative Accumulation。依存度累積を頂点あたり一スレッドではなくワープで協調処理します。第四にDual-Stream Execution。二つのCUDAストリームが独立した始点ローカルバッファを持ち初期化と計算を重ねます。四つとも新規性は主張しません。貢献はこれらを一貫した基盤へ統合し一貫した条件で評価した点です。図は二つのストリームが同じパイプラインを並列実行し、中央の全体BC配列へ部分結果を統合する様子です。バッファは始点ローカル、重ね合わせはこれを前提とします。

### English Transition

The batching that makes this framework fast is also what creates its capacity constraint.

### 日本語トランジション

この基盤を高速にしているバッチ化は、同時に容量制約を生む要因でもあります。

### Limitations to State

- No novelty is claimed for the individual components; the contribution is the integration and its evaluation.
- 個々の要素技術の初出性は主張しない。貢献は統合とその評価である。

### Expected Questions

- How does this differ from prior work? The components are known individually; the framework and the four-axis evaluation on one GH200 are what this work adds.
- 既存研究とどこが異なるか。各要素は個別には既知であり、本研究が加えるのは統合された基盤とGH200上での四観点評価である。

## Slide 5 — Source Batching Creates a Batch-Dependent Working Set

### Duration

75 seconds

### English Script

This slide defines where the capacity constraint actually comes from, because it is easy to get wrong. Source batching groups source vertices; it does not partition the graph. Every source in the batch still traverses the whole graph. What grows with the batch is the per-source state: the distance array, the sigma counts, the delta values, and the frontier structures, one copy per concurrent source. So the working set is the product of the effective number of streams, the effective batch size, and the per-source state. The consequence is the point I want you to take away: input graph size and the batch-dependent working set are different quantities. Even a small input graph can require a much larger batch-dependent working set. At b512 with two effective streams, the code-derived allocation estimate is about 10.67 gigabytes; this is an estimate, not a measured footprint. In the diagram, the box on the lower left is the input graph file, which does not grow with the batch, and the box on the lower right is the batch-dependent working set, which does.

### 日本語説明

容量制約がどこから生じるかを定義します。誤解されやすい点です。source batchingは始点をまとめるもので、グラフを分割しません。各始点は依然として全体を探索します。バッチとともに増えるのは始点ごとの状態、すなわち距離配列、シグマ、デルタ、フロンティア構造であり、同時実行する始点の数だけ必要です。したがってworking setは、実効ストリーム数、実効バッチサイズ、始点ごとの状態量の積です。要点は、入力グラフのサイズとbatch依存のworking setが別の量だということです。入力が小さくてもworking setははるかに大きくなり得ます。b512かつ実効ストリーム数2では、コード由来の割り当て推定値は約10.67ギガバイトです。これは推定値であり実測値ではありません。図の左下の箱が入力ファイルでバッチとともに増えず、右下の箱がbatch依存のworking setです。

### English Transition

That distinction is what separates the three memory-management variants.

### 日本語トランジション

この区別が、三つのメモリ管理方式を分ける基準になります。

### Limitations to State

- Source batching is not graph partitioning; the input file size is not the working set.
- source batchingはgraph partitioningではなく、入力ファイルサイズはworking setではない。

### Expected Questions

- Does a larger batch always run faster? No. Larger batches raise parallelism but also enlarge the working set, and past a point the run fails outright.
- バッチを大きくすれば必ず速くなるのか。ならない。並列性は上がるがworking setも増大し、ある点を超えると実行自体が失敗する。

## Slide 6 — Three Memory-Management Variants Share One Framework

### Duration

60 seconds

### English Script

Because the working set is what binds, the three implementations differ only in how they manage memory. GPU_Opt uses unified memory, so allocations are managed and the runtime places pages between HBM and host memory. GPU_Opt_Pure uses explicit device allocation and is bounded by device capacity. GPU_Opt_Pure_Chunked sub-batches the source set so that only part of the source state is resident at a time. I want to be precise about two things. These are not three separate proposals; they are memory-management variants of one common framework, sharing the same kernels. And what Chunked subdivides is the source set, not the graph itself. The variants were designed to isolate differences in memory-management behavior while sharing the common execution framework, which is how the capacity comparison on the later slides should be read.

### 日本語説明

制約はworking setであるため、三実装はメモリの扱い方だけが異なります。GPU_OptはUnified Memoryを用い、ランタイムがHBMとホスト間にページを配置します。GPU_Opt_Pureは明示的なデバイス割り当てを用い、デバイス容量に制限されます。GPU_Opt_Pure_Chunkedは始点集合を分割し、始点状態の一部だけを常駐させます。二点重要です。これらは独立した三提案ではなく、同一カーネルを共有する共通基盤上のメモリ管理の違いです。またChunkedが分割するのは始点集合でありグラフではありません。共通基盤を保ったままメモリ管理の挙動差を切り分ける設計であり、容量比較もその観点で読みます。

### English Transition

With the framework and its variants defined, let me turn to how they were evaluated.

### 日本語トランジション

基盤とその変種を定義したので、次にどのように評価したかを説明します。

### Limitations to State

- The three are variants of one framework, and Chunked subdivides sources, not the graph.
- 三つは同一基盤の変種であり、Chunkedが分割するのは始点であってグラフではない。

### Expected Questions

- Which variant should be used in practice? The comparison characterises capacity behaviour rather than ranking them; the right choice depends on the batch size needed.
- 実運用ではどれを選ぶべきか。この比較は優劣を決めるものではなく容量特性を示すものであり、必要なバッチサイズに依存する。

## Slide 7 — The Evaluation Separates Performance and Capacity Studies

### Duration

55 seconds

### English Script

The evaluation deliberately separates two studies. Main performance uses four graphs: email-EuAll, and the three road networks roadNet-PA, roadNet-TX, and roadNet-CA. The memory and correctness study uses one graph, the corrected 325557 graph. The file sizes in this table, from about five to about fifty-four mebibytes, are the static size of the input on disk. As I said on the previous slide, that is not the batch-dependent working set, and I show both here so the two are not conflated. All measurements are on one GH200 GPU, reported values are medians over the recorded trials, and every speedup is a median-to-median ratio. The scope is deliberately small, and I will return to what that bounds at the end.

### 日本語説明

評価は二つの研究を意図的に分離しています。主性能比較にはemail-EuAllと道路ネットワークroadNet-PA、TX、CAの四グラフ、メモリと正確性の検証には修正版325557を用います。表のファイルサイズ、およそ5から54メビバイトはディスク上の静的サイズです。前スライドのとおりbatch依存のworking setではなく、混同しないよう併記しています。測定はすべてGH200一台、報告値は記録した試行のmedian、speedupはmedian同士の比です。評価範囲は意図的に小さく、それが何を制約するかは最後に述べます。

### English Transition

Let me start with the main performance result.

### 日本語トランジション

まず主性能の結果から示します。

### Limitations to State

- Input graph size is not the batch-dependent working set.
- 入力グラフサイズはbatch依存のworking setではない。

### Expected Questions

- Why only this many graphs? The scope was set by what could be measured and verified under consistent conditions; the conclusions are limited accordingly.
- なぜこのグラフ数なのか。一貫した条件で測定・検証できる範囲で設定しており、結論もその範囲に限定される。

## Slide 8 — GPU_Opt Reduced Runtime on All Four Evaluated Graphs

### Duration

75 seconds

### English Script

This is the main performance result. The blue bars are GPU_Opt at a fixed batch size of 512, and the purple bars are PathMerge, the external comparator, at the best batch size found for each graph by a separate sweep. Note the vertical axis is logarithmic. GPU_Opt reduced runtime on all four evaluated graphs. I want to be clear about how the comparison is set up, because it is deliberately unfavourable to the proposal: our side runs one fixed configuration across every graph, while the comparator is tuned per graph. The bars are medians of raw trials, five trials for email-EuAll and three for each road graph. Those are small samples, and I will return to that when I state the limitations. Reading the chart, the leftmost pair is email-EuAll, where the gap is widest, and the three pairs to the right are the road networks, where the gap is narrower but consistent. Because the axis is logarithmic, equal visual gaps mean equal ratios rather than equal differences in seconds.

### 日本語説明

これが主性能の結果です。青い棒がバッチサイズ512に固定したGPU_Opt、紫の棒が外部比較対象のPathMergeで、後者は別途の掃引によりグラフごとに最良バッチを与えています。縦軸が対数目盛である点にご注意ください。GPU_Optは評価した四グラフすべてで実行時間を短縮しました。この比較設定は意図的に提案側へ不利です。提案側は全グラフで一つの固定設定を用いる一方、比較対象はグラフごとに調整されているためです。棒は生の試行のmedianで、試行数はemail-EuAllが5回、各道路グラフが3回です。小標本であり、限界の説明で改めて触れます。図の左端がemail-EuAllで差が最も大きく、右側の三組が道路ネットワークで、差は小さいものの一貫しています。対数目盛であるため、見た目の間隔が等しいことは秒数差ではなく比が等しいことを意味します。

### English Transition

Expressing the same data as a ratio gives the headline number.

### 日本語トランジション

同じデータを比として表すと、主要な数値が得られます。

### Limitations to State

- Trial counts are small: five for email-EuAll and three per road graph.
- 試行数は小さく、email-EuAllが5回、各道路グラフが3回である。

### Expected Questions

- Isn't the trial count too small? Yes, it is a real limitation; the medians are reported as such and no distributional claim is made.
- 試行数が少ないのではないか。実際に限界であり、medianとして報告し分布に関する主張はしない。

## Slide 9 — GPU_Opt Achieved 1.31–3.17× Speedup over the Tuned Comparator

### Duration

90 seconds

### English Script

Expressed as a ratio, GPU_Opt achieved a 3.17 times speedup on email-EuAll, 1.31 times on roadNet-PA, 1.51 times on roadNet-TX, and 1.45 times on roadNet-CA. Each is a median-to-median ratio, and the dashed line marks parity at one. The tuned batch sizes for the comparator were b2048 for email-EuAll, b64 for roadNet-PA, b64 for roadNet-TX, and b32 for roadNet-CA. Now the important qualification. PathMerge is an evaluated third-party implementation and an external comparator, not ground truth. It is a snapshot we retained and measured; it is not the original authors' official implementation, and it is not a correctness reference. The result does not generalize to PathMerge implementations in general, and it does not generalize beyond these four graphs on this one GPU. The selected batch sizes differed substantially across the four graphs. This evaluation records that sensitivity but does not establish a single structural cause. Our side used b512 throughout, which is not the tuned setting for any of them.

### 日本語説明

比として表すと、GPU_Optはemail-EuAllで3.17倍、PAで1.31倍、TXで1.51倍、CAで1.45倍のspeedupを達成しました。いずれもmedian同士の比で、破線が等速の1.0倍です。比較対象の調整済みバッチはemail-EuAllがb2048、PAとTXがb64、CAがb32でした。重要な限定です。PathMergeは第三者実装かつ外部の比較対象であってground truthではなく、保存して測定したsnapshotで、公式実装でも正確性の参照でもありません。結果はPathMerge実装一般へも、この四グラフと一台のGPUを超えても一般化しません。採用バッチは四グラフ間で大きく異なりました。本評価はその感度を記録しますが、単一の構造的原因を確定しません。提案側はb512固定で、いずれのグラフの調整済み設定でもありません。

### English Transition

The next question is which parts of the framework produced that gain.

### 日本語トランジション

次の問いは、この向上が基盤のどの部分から生じたのかです。

### Limitations to State

- PathMerge is an external comparator, not ground truth, and the result does not generalize to PathMerge in general.
- PathMergeは外部比較対象でありground truthではない。結果はPathMerge一般へ一般化しない。

### Expected Questions

- Was the comparator tuned enough? A dedicated batch sweep, shown in the backup slides, selected its best batch per graph, while our side stayed fixed at b512.
- 比較対象の調整は十分か。Backupに示す専用の掃引でグラフごとの最良バッチを選定しており、提案側はb512に固定している。

## Slide 10 — Multiple Execution Components Contributed to Performance

### Duration

75 seconds

### English Script

To attribute the gain, I ran a factor decomposition over the three compile-time components: H for hybrid BFS, W for warp-cooperative accumulation, and A for dual-stream execution. On the corrected 325557 graph the main effects were 1.4767 for hybrid BFS, 1.1012 for warp-cooperative accumulation, and 1.5563 for dual-stream execution. On the synthetic-four aggregate they were 1.6787, 1.0661, and 1.3914. So hybrid BFS and dual-stream execution contributed most, while the warp-cooperative effect was smaller and differed between the two settings. Two caveats. The synthetic-four figure is an aggregate across a mixed set of checkpoints, which I note on the slide. And no factor decomposition was run on the road networks at all, so these numbers do not explain the main performance gap I just showed. Each bar is a main effect: the speedup attributable to enabling that one component, averaged across the full factorial of the three flags. Because these are main effects, they do not simply multiply together to give the overall speedup.

### 日本語説明

要因分解は、三つのコンパイル時要素、HがHybrid BFS、WがWarp-Cooperative Accumulation、AがDual-Stream Executionについて行いました。修正版325557では主効果がH 1.4767倍、W 1.1012倍、A 1.5563倍、合成4グラフの集約では1.6787倍、1.0661倍、1.3914倍です。HとAの寄与が大きく、Wの効果は相対的に小さく条件により異なりました。注意点が二つ。合成4グラフの値はcheckpoint混在の集約であり、スライドにも明記しています。また道路網では要因分解を実施しておらず、これらの数値は主性能差を説明しません。各棒は主効果、すなわち三フラグの完全実施要因計画で平均した、その要素の有効化によるspeedupで、積算しても全体のspeedupにはなりません。

### English Transition

That covers performance. The next axis is capacity.

### 日本語トランジション

性能についてはここまでです。次の観点は容量です。

### Limitations to State

- No factor decomposition was run on roadNet, so these values do not explain the main performance gap.
- roadNetでは要因分解を実施しておらず、これらの値は主性能差を説明しない。

### Expected Questions

- Why did the warp effect differ between the two settings? The measured values differ, but this evaluation does not establish a single cause, and no general rule is inferred from them.
- Warpの効果が二条件で異なったのはなぜか。測定値は異なるが、本評価は単一の原因を確定するものではなく、これらから一般則も導かない。

## Slide 11 — Memory Variants Expanded the Tested Feasible Batch Range

### Duration

80 seconds

### English Script

This is the capacity result on the corrected 325557 graph, one row per memory variant against requested batch size. Pure succeeded at b4096 and failed at b8192 with a CUDA device out-of-memory error. Unified memory succeeded at b10240 and failed at b12288, and that failure was different in kind: it was recorded as a host cgroup memory limit out-of-memory kill, exit 137, and not a device error. The two failure classes are reported separately and are not treated as one boundary. Chunked succeeded at b16384. Two things to read carefully. Failures are categorical outcomes, not zero-second runtimes, so a cross on this chart is a classification, not a measurement of zero. And Chunked succeeded at the tested upper bound of b16384; this does not imply unlimited capacity. That is the largest batch we tried, not a limit we found. Every point is a single targeted feasibility run. The point to take away is that the three rows failed at different batch sizes and with different failure classes, and exposing that difference is why the three memory-management variants exist.

### 日本語説明

修正版325557の容量結果を、要求バッチサイズに対し方式ごとに示します。Pureはb4096で成功、b8192ではCUDAのデバイスout-of-memoryエラーで失敗しました。UMはb10240で成功、b12288で失敗しましたが、この失敗は種類が異なり、デバイス側エラーではなくホストのcgroupメモリ制限によるOOM kill、exit 137です。二つの失敗クラスは分けて報告し、単一境界としては扱いません。Chunkedはb16384で成功しました。注意点が二つ。失敗は区分された結果で0秒の実行時間ではなく、×印は分類で測定値0ではありません。またChunkedは試験上限で成功しましたが、容量が無制限とは言えません。これは試した最大のバッチであり限界ではありません。各点は1回のtargetedな実行可能性確認です。要点は三行が異なるバッチと失敗クラスで失敗したことで、その差の可視化が三方式の理由です。

### English Transition

That is capacity. The remaining axis is whether the output is numerically valid.

### 日本語トランジション

容量については以上です。残る観点は、出力が数値的に妥当かどうかです。

### Limitations to State

- Failures are categorical outcomes, not zero-second runtimes, and b16384 is a tested upper bound, not a demonstrated capacity limit.
- 失敗は0秒の実行時間ではなく区分された結果であり、b16384は試験上限であって容量限界の証明ではない。

### Expected Questions

- Does unified memory remove the memory constraint? No. It moved the boundary and changed the failure mode, but it still failed, at b12288, on the host side.
- UMを使えばメモリ制約は解消するのか。しない。境界と失敗の様態が変わっただけで、b12288でホスト側で失敗している。

## Slide 12 — Numerical Results Matched within Tolerance but Were Not Byte-Identical

### Duration

60 seconds

### English Script

Correctness was checked in two tiers. Tier A uses an independent Sequential CPU reference, comparing full betweenness vectors on three small graphs. Tier B compares implementation paths against each other on the corrected 325557 graph, ten comparisons. That gives thirteen comparisons in total, with zero missing indices and zero mismatched elements. All 13 comparisons passed the mixed tolerance, but none was byte-identical. Two qualifications matter here. Agreement under a tolerance is numerical agreement, not bitwise identity. The non-byte-identical results are consistent with different floating-point operation orders, but this evaluation does not establish a single cause. And Tier B compares implementation paths and is not an independent ground-truth evaluation, so the independent evidence is the three Tier A comparisons. The table breaks this down by tier, with the missing and mismatched columns reading zero on every row and the byte-identical column reading No on every row.

### 日本語説明

正確性は二つのTierで検証しました。Tier Aは独立したSequential CPU参照を用い、小規模な三グラフでBCベクトル全体を比較します。Tier Bは修正版325557上で実装経路どうしを比較する10件です。合計13比較で、欠損インデックス0件、不一致要素0件。すべてmixed toleranceの下で合格しましたが、byte一致は一つもありません。二点の限定が重要です。許容誤差下の一致は数値的一致でありビット単位の同一性ではありません。byte非一致は浮動小数点演算順序の違いと整合しますが、本評価は単一の原因を確定しません。またTier Bは実装経路間の比較で、独立したground truth評価ではありません。独立な根拠はTier Aの3比較です。

### English Transition

Having covered all four axes, let me state where the evidence stops.

### 日本語トランジション

四つの観点をすべて説明したので、次に根拠がどこで止まるかを述べます。

### Limitations to State

- Tier B is not an independent ground-truth evaluation, and no comparison was byte-identical.
- Tier Bは独立したground truthとの評価ではなく、byte一致した比較は一つもない。

### Expected Questions

- Is the absence of byte-identity a problem? It is not treated as a failure here because all comparisons passed the predefined mixed tolerance with zero mismatched elements. The exact cause of the byte difference is not independently established.
- byte一致でないことは問題か。ここでは失敗として扱っていない。全比較が事前に定めたmixed toleranceを不一致要素0で満たしたためである。byte差の正確な原因は独立には確定していない。

## Slide 13 — The Evidence Has Clear Boundaries

### Duration

50 seconds

### English Script

Let me be explicit about the boundaries. The evaluation used a single GH200 GPU and does not generalize to other GPUs. Main performance covers four graphs, and the memory and correctness study uses corrected 325557 only. Trial counts are small, and each capacity boundary is a single targeted validation run. PathMerge is a retained third-party snapshot, so nothing here is a claim about PathMerge in general. And unified memory and Chunked have finite capacity limits; I do not claim they avoid out-of-memory failures. Stated together: the conclusions are limited to one GH200 GPU, the evaluated graphs, the retained implementation snapshots, and the recorded experimental conditions. These are not retractions; they mark the range within which the conclusions hold, and I would rather state them plainly than leave them to be inferred.

### 日本語説明

根拠の境界を述べます。評価はGH200一台であり、他のGPUへ一般化しません。主性能は四グラフ、メモリと正確性の検証は修正版325557のみです。試行数は小さく、容量境界は各条件1回のtargeted validationです。PathMergeは保存した第三者実装のsnapshotであり、PathMerge一般に関する主張ではありません。UMとChunkedの容量にも上限があり、OOM回避は主張しません。本研究の結論はGH200一台、評価したグラフ、保存した実装snapshot、記録された実験条件の範囲に限定されます。これらは結論の否定ではなく、成立範囲を示すものです。

### English Transition

Within those boundaries, the contributions are as follows.

### 日本語トランジション

この範囲の内側で、貢献は次のとおりです。

### Limitations to State

- The limitations mark the range in which the conclusions hold; they are not retractions.
- 限定は結論が成立する範囲を示すものであり、結論の撤回ではない。

### Expected Questions

- What can be claimed under these limits? That on this hardware and these graphs, the integrated framework was faster and its capacity boundaries were characterised.
- この限定の下で何が言えるか。このハードウェアとこれらのグラフにおいて、統合基盤が高速であり、その容量境界を特徴づけたということである。

## Slide 14 — Contributions

### Duration

30 seconds

### English Script

The contributions are four. First, the design and implementation of a GPU execution framework integrating existing components. Second, a performance evaluation against a tuned third-party comparator. Third, a component-level contribution analysis quantifying the H, W, and A effects including their graph dependence. Fourth, a boundary analysis that treats memory capacity and numerical correctness separately. Again, no novelty is claimed for the individual components. Each of the four is evaluated in the body of the thesis under the conditions I have described.

### 日本語説明

貢献は四点です。第一に、既存の要素を統合したGPU実行基盤の設計と実装です。第二に、調整済みの第三者実装を比較対象とした性能評価です。第三に、H、W、Aの効果をグラフ依存性を含めて定量化した要素レベルの寄与分析です。第四に、メモリ容量と数値的正確性を分離して扱った境界分析です。繰り返しになりますが、個々の要素技術の初出性は主張しません。

### English Transition

Let me close with the central conclusion.

### 日本語トランジション

最後に中心となる結論を述べます。

### Limitations to State

- Novelty is claimed for the integration and evaluation, not for the components.
- 新規性は統合と評価について主張するものであり、個々の要素についてではない。

### Expected Questions

- Which contribution is the most essential? The integration together with the four-axis evaluation, since no single component is claimed as new.
- 最も本質的な貢献はどれか。個々の要素の新規性を主張しない以上、統合と四観点評価である。

## Slide 15 — The Integrated Framework Improved Performance and Clarified Capacity Limits

### Duration

25 seconds

### English Script

To conclude: the integrated block-based GPU implementation achieved a 1.31 to 3.17 times speedup over the tuned comparator on the four evaluated graphs, and the memory-path experiments clarified both the feasible batch ranges and the remaining numerical limitations. The gain came from multiple components of the framework rather than one technique. Thank you for your attention, and I am happy to take questions.

### 日本語説明

結論です。統合されたblock-basedのGPU実装は、評価した四グラフで調整済みの比較対象より1.31倍から3.17倍高速であり、メモリ経路の実験は実行可能なバッチ範囲と数値的限界の双方を明らかにしました。この向上は単一技術ではなく複数要素から生じました。ご清聴ありがとうございました。

### English Transition

(End of the main talk; backup slides follow for questions.)

### 日本語トランジション

（本編はここまでです。以降のBackupは質疑応答用です。）

### Limitations to State

- The conclusion holds within the evaluated GPU, graphs, and implementation snapshots.
- 結論は評価したGPU、グラフ、実装snapshotの範囲で成立する。

### Expected Questions

- What is the future work? Broader graph coverage, more trials, and factor decomposition on the road networks, which was not performed here.
- 今後の課題は何か。グラフ範囲の拡大、試行数の増加、そして本研究では未実施の道路ネットワークにおける要因分解である。

## Backup

Backup は本編では使用せず、質疑応答時にのみ参照する。想定時間の合計 280 秒は本編 900 秒には含まれない。

## Slide 16 — Detailed Experimental Environment

### Duration

40 seconds

### English Script

This backup slide gives the full environment. The GPU is an NVIDIA GH200 Grace Hopper Superchip at sm_90 with a nominal 96 gigabytes of HBM3, driver 595.58.03, CUDA 13.0, and g++ 11.4.1. Measured device-to-device HBM3 bandwidth was 1818.6 gigabytes per second, and NVLink-C2C prefetch bandwidth was 177.7. The main experiments report the median of all recorded trials with no warmup and nothing discarded. Items that could not be established independently from the records were left undetermined rather than filled in.

### 日本語説明

このBackupスライドは実験環境の全体を示します。GPUはNVIDIA GH200 Grace Hopper Superchipのsm_90で、公称HBM3容量は96ギガバイト、ドライバは595.58.03、CUDAは13.0、g++は11.4.1です。実測のデバイス間HBM3帯域は毎秒1818.6ギガバイト、NVLink-C2Cのプリフェッチ帯域は177.7でした。主実験は記録した全試行のmedianを報告し、warmupは行わず、破棄した試行もありません。記録から独立に確定できない項目は、補完せず未確定のまま扱っています。

### English Transition

(Backup slide, shown on request.)

### 日本語トランジション

（Backupスライド。質疑応答で必要に応じて表示する。）

### Limitations to State

- Values that could not be independently established from the records are left undetermined.
- 記録から独立に確定できない値は未確定のままとしている。

### Expected Questions

- Why is the runtime-reported memory larger than 96 GB? The nominal capacity and runtime-reported values come from different records and units. They are shown separately without asserting that they are equivalent or identifying a single cause for the difference.
- 実行時報告値が96GBより大きいのはなぜか。公称容量と実行時報告値は異なる記録と単位に由来する。両者は同値と断定せず、差の単一原因も特定せずに分けて示している。

## Slide 17 — Graph and Batch Parameters

### Duration

40 seconds

### English Script

This backup slide lists the per-graph batch settings, medians, and trial counts, together with the working-set formula: the effective number of streams times the effective batch size times the per-source state, which is 10,418,856 bytes. The hybrid BFS alpha and beta thresholds are also given. Two points: the per-source state is an allocation estimate derived from the code, not a measured memory footprint, and GPU_Opt uses a fixed b512 on every graph while only PathMerge is tuned per graph.

### 日本語説明

このBackupスライドは、グラフごとのバッチ設定、median、試行数に加えて、working setの算出式を示します。式は実効ストリーム数かける実効バッチサイズかける始点ごとの状態量であり、状態量は10,418,856バイトです。Hybrid BFSのalphaとbetaの閾値も併せて示しています。二点補足します。始点ごとの状態量はコードから導いた割り当ての推定値であり、実測のメモリ使用量ではありません。またGPU_Optは全グラフでb512に固定されており、グラフごとに調整しているのはPathMergeのみです。

### English Transition

(Backup slide, shown on request.)

### 日本語トランジション

（Backupスライド。質疑応答で必要に応じて表示する。）

### Limitations to State

- Per-Source State is a code-derived allocation estimate, not measured memory usage.
- 始点ごとの状態量はコード由来の割当推定であり、実測メモリ使用量ではない。

### Expected Questions

- Why not tune GPU_Opt per graph as well? Leaving it fixed keeps the comparison unfavourable to the proposal rather than favourable.
- なぜGPU_Optもグラフごとに調整しないのか。固定のままとすることで、比較が提案側に有利にならないようにしている。

## Slide 18 — PathMerge Batch-Size Sweep

### Duration

40 seconds

### English Script

This backup slide shows the PathMerge batch sweep that justifies the tuned comparator settings. Median runtime is plotted against requested batch size for each of the four graphs, and the minimum of each curve is the batch reported on the speedup slide. One detail: for email-EuAll the requested batch is clamped to a smaller effective batch, which is noted on the slide. The historical malformed 325557 graph is excluded from this sweep.

### 日本語説明

このBackupスライドは、比較対象の調整設定を裏づけるPathMergeのバッチ掃引を示します。四つのグラフそれぞれについて、要求バッチサイズに対するmedian実行時間を示しており、各曲線の最小値がspeedupのスライドで報告したバッチです。一点補足すると、email-EuAllでは要求バッチがより小さい実効バッチへクランプされており、その旨をスライドに注記しています。履歴的なmalformed版の325557グラフはこの掃引から除外しています。

### English Transition

(Backup slide, shown on request.)

### 日本語トランジション

（Backupスライド。質疑応答で必要に応じて表示する。）

### Limitations to State

- The sweep is what gives PathMerge its best per-graph batch; the malformed graph is excluded.
- この掃引がPathMergeへグラフごとの最良バッチを与えている。malformed版グラフは除外している。

### Expected Questions

- Could a batch outside the swept range be faster? Possibly; the sweep covers the tested points only and no claim is made beyond them.
- 掃引範囲外のバッチがより速い可能性はあるか。あり得る。掃引は試験した点のみを対象とし、その外側については主張しない。

## Slide 19 — Forced Block-vs-Shared Kernel Comparison

### Duration

40 seconds

### English Script

This backup slide compares the block kernel against the shared-memory kernel by forcing each one, on roadNet-PA and roadNet-TX. The chart gives the median runtime for each kernel on each graph, with the per-graph ratio annotated beneath it. The block kernel was faster on both measured graphs. These measurements support the current block-kernel choice within the measured PA and TX scope, but they do not define a selector rule for unmeasured graphs.

### 日本語説明

このBackupスライドは、blockカーネルとshared memoryカーネルをそれぞれ強制的に選択して、roadNet-PAとroadNet-TXで比較したものです。図は各グラフ・各カーネルのmedian実行時間を示し、下部にグラフごとの比を注記しています。測定した両グラフでblockカーネルの方が高速でした。この測定は、測定範囲であるPAとTXの内側では現行のblockカーネル選択を支持しますが、測定していないグラフに対する選択規則を定めるものではありません。

### English Transition

(Backup slide, shown on request.)

### 日本語トランジション

（Backupスライド。質疑応答で必要に応じて表示する。）

### Limitations to State

- Measured on two graphs only; no selector rule is inferred for unmeasured graphs.
- 測定は二グラフのみであり、未測定グラフへの選択規則は導かない。

### Expected Questions

- Would the shared kernel ever win? It might on graphs not measured here; that was not tested, so no claim is made.
- sharedカーネルが勝つことはあるか。ここで測定していないグラフではあり得るが、未検証であり主張しない。

## Slide 20 — Phase Breakdown and Profiling Scope

### Duration

40 seconds

### English Script

This backup slide breaks the runtime into BFS, backward accumulation, and other, for each graph. These come from complete b512 wall-clock runs, where other is the total minus BFS minus backward; they are not partial Nsight trace totals. Where Nsight profiling is referenced elsewhere, that is a single trace of a single graph and does not generalize to all graphs.

### 日本語説明

このBackupスライドは、各グラフの実行時間をBFS、逆方向の累積、その他へ分解したものです。これらはb512での完全な実行の実時間に基づいており、その他は総時間からBFSと逆方向累積を引いた値です。部分的なNsightトレースの合計ではありません。他所でNsightプロファイルに言及している箇所については、単一グラフの単一トレースであり、全グラフへ一般化しません。

### English Transition

(Backup slide, shown on request.)

### 日本語トランジション

（Backupスライド。質疑応答で必要に応じて表示する。）

### Limitations to State

- These are complete wall-clock runs, not partial Nsight trace totals.
- これは完全実行の実時間であり、部分的なNsightトレースの合計ではない。

### Expected Questions

- Why not report Nsight totals directly? A single partial trace would not account for the whole run, so the wall-clock decomposition is reported instead.
- なぜNsightの合計を直接報告しないのか。単一の部分トレースでは実行全体を説明できないため、実時間による分解を報告している。

## Slide 21 — Detailed Correctness Evidence

### Duration

40 seconds

### English Script

This backup slide lists all thirteen correctness comparisons individually: the tier, the graph, the reference and candidate implementations, their batch sizes, the maximum relative error, and the tolerance result. Across all thirteen, missing indices are zero, mismatched elements are zero, and byte-identical is No. The three Tier A rows are the independent evidence; Tier B compares implementation paths and is not an independent ground-truth evaluation.

### 日本語説明

このBackupスライドは、13件の正確性比較を個別に列挙したものです。Tier、グラフ、参照実装と候補実装、それぞれのバッチサイズ、最大相対誤差、許容判定を示しています。13件すべてにおいて、欠損インデックスは0、不一致要素は0、byte一致はNoです。Tier Aの3行が独立な根拠であり、Tier Bは実装経路どうしの比較であって、独立したground truthとの評価ではありません。

### English Transition

(Backup slide, shown on request.)

### 日本語トランジション

（Backupスライド。質疑応答で必要に応じて表示する。）

### Limitations to State

- Tier B is implementation-path consistency, not comparison against independent ground truth.
- Tier Bは実装経路間の整合であり、独立したground truthとの比較ではない。

### Expected Questions

- Why is Tier A limited to small graphs? The independent CPU reference is sequential, so full-vector comparison is only tractable at small scale.
- なぜTier Aは小規模グラフに限られるのか。独立参照がSequential CPU実装であり、ベクトル全体の比較は小規模でのみ現実的だからである。

## Slide 22 — Historical Record of the Malformed Input

### Duration

40 seconds

### English Script

This backup slide is a historical record and is deliberately separated from the current results. An earlier version of the 325557 graph was malformed: it used 1-based vertex identifiers, was missing seven elements, and contained out-of-range identifiers. On that input, correctness reported CORE_FAIL with a stress mismatch at the 1e-6 level. The graph was reconstructed by symmetry, a validator was added, and on the corrected input all thirteen comparisons pass. The old results are retained as evidence about invalid input, and they are not used in any current conclusion. The banner on this slide says so explicitly.

### 日本語説明

このBackupスライドは履歴的な記録であり、現在の結果とは意図的に分離しています。325557グラフの旧版はmalformedでした。頂点IDが1始まりで、要素が7個欠落しており、範囲外のIDを含んでいました。この入力では正確性検証がCORE_FAILとなり、1e-6の水準でstress不一致が生じていました。その後、対称性からグラフを再構成し、検証器を追加した結果、修正版の入力では13比較すべてが合格しています。旧結果は不正な入力に関する記録として保存していますが、現在のいかなる結論にも用いていません。このスライドのバナーはその点を明示しています。

### English Transition

(Backup slide, shown on request.)

### 日本語トランジション

（Backupスライド。質疑応答で必要に応じて表示する。）

### Limitations to State

- Historical evidence only; it is not part of the current results.
- 履歴的な記録にすぎず、現在の結果の一部ではない。

### Expected Questions

- Why keep the malformed-input results at all? Detecting the defect and re-validating on the corrected input is itself part of the reproducibility record.
- malformed入力の結果をなぜ残すのか。不整合を検出し修正版で再検証した経緯自体が再現性の記録だからである。
