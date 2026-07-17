# Chapter 10 Discussion

本章では、Chapter 6からChapter 9の性能、構成要因、容量、数値的挙動を横断して解釈する。本研究の中心は、GH200向けバッチ型Betweenness Centrality（BC）GPU実行基盤を設計し、その性能の成立要因と適用限界を一つの実験系で明らかにすることにある。

## 10.1 Interpretation of the Performance Results

Chapter 6の中心結果は、固定b512のblockベースGPU_Optが、評価したemail-EuAllおよびroadNet-PA/TX/CAにおいて、グラフごとに調整した第三者実装PathMergeより1.31〜3.17倍高速だったことである。内訳はemail-EuAllで3.17倍、roadNet-PAで1.31倍、roadNet-TXで1.51倍、roadNet-CAで1.45倍であり、speedupはmedian/medianである。GPU_Optは1ストリーム当たりb512、NS_eff=2の固定設定で、グラフ別の最速バッチ探索結果ではない。PathMergeはemail-EuAllでb2048、roadNet-PA/TXでb64、roadNet-CAでb32へ調整されている。この非対称性の下でもGPU_Optが高速だったことは、GPU_Opt側にグラフ別調整の利益を与えない保守的な比較と解釈できる。

性能差の幅はemail-EuAllとroadNet群で異なる。4件で同じ方向の差が観測されたことは、未評価条件で同じ倍率が再現することを意味しない。試行数も小さく、有意差や分布一般に関する結論は置かない。本研究が支持するのは、評価した4グラフ、1台のNVIDIA GH200、保存されたsnapshotと測定条件におけるmedianベースの比較である。

GTEPSでもGPU_Optが4グラフで上回ったが、これは本研究のBCワークロード定義に基づく正規化値である。GTEPSはruntimeを補足する一方、実効メモリ帯域やカーネル効率を直接測るものではなく、性能差の機構を単独で説明しない。

PathMergeは評価に用いた第三者実装で、原著者の公式実装とは確認されておらず、上流の明示的ライセンス表記も確認できない。PathMergeはground truthではなくexternal comparatorであり、本結果はPathMerge/Galliot一般や別実装に対する性能関係を示さない。

### 10.1.1 Contributions of the Execution Components

Chapter 7のfactorial ablationは、end-to-endの性能差を一つの工夫へ還元するのではなく、共通実行基盤を構成する要素の観測上の寄与を分けて示した。合成4グラフの主効果の幾何平均はHybrid BFSが1.655倍、Warp-Cooperative Accumulationが1.065倍、Dual-Stream Executionが1.396倍であった。email-EuAllではそれぞれ1.429倍、0.970倍、1.720倍であった。この範囲ではHybrid BFSと2ストリームが主要な正の寄与を示し、warp協調はグラフによって効果の大きさと方向が変わった。特にemail-EuAllの0.970倍は、warp協調を有効にした構成が測定上わずかに遅い方向であったことを示す。

主効果はChapter 6のspeedupを分配した値ではなく、因子間interactionも正式には推定されていない。単一要素だけでPathMergeとの差を説明することはできない。支持される解釈は、統合基盤の中でHybrid BFSと2ストリームが主要な観測要因であり、warp協調にはグラフ依存性が残るというものである。

forced shared/block比較では、roadNet-PA/TXでblockがそれぞれ1.52倍、1.66倍高速であり、現行のblock常用を2グラフについて支持した。ただし正確性は`max_bc_only`で、未測定グラフへ広げられない。旧実装の`avg_deg < 5`でsharedを選ぶ規則は現行方式では用いていない。

Backward 63.9%とBFS 36.1%は、56438_300801の`ablation_H1W1A0`単一nsysトレースにおけるGPUカーネル時間比で、同一process冒頭のuntimedな`H1W1A1` warmupを含む。本測定だけのphase比率や他グラフの代表値ではなく、因果的な寄与率にも用いない。

## 10.2 Relationship with Graph Characteristics

email-EuAllとroadNet群は、規模、次数分布、BFS深さ、フロンティア形状が同時に異なる。email-EuAllはハブを含み探索が浅く、roadNet群は低次数で探索が深い。性能差がemailで大きかったことはこの対照と併存し、emailのablationでは2ストリームの主効果が大きく、warp協調はわずかに不利であった。

しかし、複数の構造量とPathMergeのtuned batchが同時に変化し、roadNet群ではH/W/A ablationもない。よってspeedup幅を特定の構造や単一要因へ帰属できない。

warp協調の方向がグラフごとに異なり、低次数のroadNet-PA/TXではforced blockが優位だったことからも、平均次数だけで方式を決める根拠はない。探索、累積、同期、メモリ配置を含む基盤全体として評価する必要があり、4グラフ程度の関係から選択則は導かない。

## 10.3 Performance-Capacity Trade-Off

### 10.3.1 Positioning of the Memory-Management Variants

GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunkedは独立した提案ではない。共通基盤はHybrid BFS、block単位始点処理、warp/thread経路、2ストリームを統合する。その上でGPU_OptはUMを用いる主実装、Pureはdevice-only memoryの対照、Chunkedはsub-batch分割による容量拡張方式である。

325557_3216152を対象としたlegacy feasibility系列では、Pureはb4096まで成功し、b8192以降にCUDA OOMが記録された。UMはb10240まで成功し、b12288は`OOM_OR_FAIL`、exit 137で停止したが、その原因は独立に確認されていない。したがって、b12288を確認済みCUDA OOMとして扱うことはできない。Chunkedは試験範囲のb16384まで成功した。この順序は、評価した1グラフとlegacy環境の試験範囲内で、UMがPureより大きなバッチへ到達し、Chunkedがさらに試験上限まで到達したというexecution feasibilityを示す。

性能と容量は異なる評価軸である。UMはPureの容量境界を越えた一方、oversubscription域で時間が増加した。Chunkedの利点は最高性能ではなく、resident working setを制限して実行可能範囲を広げた点にある。UMの容量は有限で、Chunkedの未試験条件も保証されないため、方式選択には時間、メモリ余裕、分割回数を併せて考える必要がある。

legacy feasibilityとcurrent memory-path correctnessはcheckpoint、目的、資源構成が異なるため、境界を結合しない。legacy時間値は現行blockのheadline性能値ではなく、25秒のUM部分traceのmigration量も全実行総量ではない。

### 10.3.2 Numerical Behavior and Reliability

最も強い数値証拠は、小規模3グラフでGPU_Optの全BCベクトルが独立参照Sequentialと混合許容内でPASSし、mismatch、missing、NaN/Infが0だったことである。この範囲をemail、roadNet、325557、Pure、Chunked、UM固有経路へ拡張できない。

PathMergeではemailのb64対b2048が歴史的comparison summary上PASSし、CAのb32対b64もabsolute-only warning付きでPASSした。ただし4本のvector本体は現在取得不能で、NaN/Infとduplicate indexは`not_recorded`である。PA/TXは`max_bc_only`にとどまる。

325557のsame-batch b1024ではUM/Pure/Chunkedの3組がmismatch=0だったが、SHA256は異なる。異なる経路の出力bitが異なっても混合許容内で整合する場合を示すが、byte identityや独立参照による正しさを意味しない。

stress比較は2組ともmismatch 6、影響indexの和集合8でFAILし、overall statusは`CORE_FAIL`である。補助許容で差が消えても正式判定は変わらない。T-RESET/T-NSEFF診断はfull memsetと`NS_eff=1`の単独変更で差を識別できず、原因は未解決である。加算順序などと整合的な可能性はあっても、特定原因へ確定できない。

PathMergeとのcross比較には多数の不一致があるが、大規模な独立参照がない。PathMergeはground truthではなくexternal comparatorであり、正誤は未決定である。smallとsame-batchのPASS、stressのFAIL、crossの未決定を同時に保持する必要がある。

## 10.4 Implications for GH200

GH200上で計算構成の工夫と容量管理を同じBC基盤として扱うと、in-capacityでは実行要素が性能を担い、working set増大時にはUMまたはChunkedが実行可否を左右する。runtimeだけの評価では、容量拡張時の時間増加、数値挙動、失敗境界を捉えられない。

一方、HBM容量、CPU–GPU接続、page migration、host memory上限が異なれば結果も変わり得る。本研究が示すのはGH200上の設計と評価であり、他GPUやmulti-GPUの倍率・容量境界ではない。

### 10.4.1 Research Contributions and Novelty

新規性は個々の要素の初出性ではなく、BC向けGPU基盤として統合し、性能、要因、容量、数値的限界を一貫した実験系で評価した点にある。

1. GH200向けバッチ型BC GPU実行基盤の設計と実装。
2. Hybrid BFS、block単位始点処理、warp/thread経路、2ストリームの統合。
3. グラフ別に調整した第三者GPU comparatorに対する性能評価。
4. H/W/A ablationによる構成要因の定量評価とforced kernel比較。
5. UM、Pure、Chunkedによるメモリ方式と容量特性の比較。
6. full-vector比較による数値挙動の検証と、未解決領域の明示。
7. OOM、`OOM_OR_FAIL`、`CORE_FAIL`、取得不能artifactを成功結果とともに保持する再現可能な研究アーカイブ。

これにより、単なるruntime比較ではなく、performance、components、capacity、numerical behaviorを相互に制約するシステム研究となる。主張できない領域をstatusとartifact状態まで含めて記録したことも成果である。

## 10.5 Threats to Validity

### 10.5.1 Internal Validity

主性能比較はn=3またはn=5、memory-pathはn=1で、warmupも系列によりなしまたは`not_recorded`である。主性能の一部baselineと容量評価にはlegacy checkpointが含まれ、snapshotで系列を識別できても環境変動や実装差を排除しにくい。

queue名と一部PBS資源は保存ログから独立に確定できず、scheduler条件の追試に制約がある。profilingも単一traceまたは25秒部分traceで、warmupを含むscopeや時間窓の制約を受ける。

emailとCAのPathMerge vector本体は現在取得不能で、summaryを越える再検査はできない。stress差の原因も未解決である。比較対象の第三者snapshotとライセンス監査上の制約も追試可能性に影響するため、`not_recorded`、FAIL、`CORE_FAIL`を維持する。

### 10.5.2 External Validity

評価は主にGH200 1台、性能4グラフ、ablation 5グラフ、memory scalability 1グラフである。他GPU、multi-GPU、別のhost memory構成、他BC実装、別グラフ族へ性能関係や容量順序を拡張できない。

PathMergeも保存された第三者実装に限られ、原著者公式とは確認されていないため、PathMerge/Galliot全体を代表しない。

### 10.5.3 Construct Validity

runtime、speedup、GTEPSは異なる側面を測り、GTEPSはmemory bandwidthではない。要求batch、effective batch、`SUB_BATCH`、`num_subs`、`NS_eff`も異なる量で、1ストリーム当たりb512と2ストリームの並行性を区別する必要がある。

CUDA OOM、`OOM_OR_FAIL`、`TIMEOUT`、runnerの`FAIL`、correctness FAILは別状態である。feasibilityとperformanceも別軸で、Max BC一致はfull-vector一致を、混合許容PASSはbyte identityを示さない。

### 10.5.4 Baseline Validity

PathMergeをグラフ別に調整した点はbaseline妥当性を高めるが、第三者snapshotとライセンス監査上の制約は再配布・追試を難しくする。PA/TXではlegacy b64を分母に採用し、分子とcheckpointが異なるため、provenanceの明示が必要である。

補助baselineは現行blockによるmedium/large統一比較を欠き、legacy提案系は旧shared経路である。Sequential、OpenMP、cuGraphの欠測とcuGraph条件の未確認点があるため、中心主張へ結合しない。

### 10.5.5 Conclusion Validity

小標本のため統計的検出力は限られ、Wilcoxonで差を検出できない場合も同等性を示さない。合成4グラフの幾何平均も広いグラフ母集団へ拡張できない。

単一の最速trialではなくmedianを主値とし、speedupもmedian/medianで計算したが、小標本自体は解消されない。よって観測方向と効果量を評価条件付きで報告する。

## 10.6 Limitations and Future Work

Chapter 11へ送るlimitationsは、headlineと大規模memory-pathでの独立参照不足、stress原因とPathMerge cross差の未解決、memory scalabilityの1グラフ・legacy依存、他GPU・multi-GPU未評価、現行blockの統一baseline不足である。

今後はlarge batch、分割、grid/occupancyの交絡を切り分け、大規模な独立参照を導入する必要がある。加えて、再配布条件と公式性を確認できるPathMerge実装、現行block統一比較、追加グラフ、他GPU・multi-GPU、full-duration migration計測が課題となる。これらを現在の結果として先取りしない。

## 10.7 Integrated Answers to the Research Questions

Table 10.1にRQ1〜RQ4の統合回答を示す。各回答は、Chapter 6〜9のformal statusと適用範囲を保持する。

**Table 10.1: Integrated answers to the research questions.**

| Research Question | Main Finding | Evidence Status | Evaluation Scope | Primary Limitation |
|---|---|---|---|---|
| RQ1 | Fixed-batch block GPU_Opt outperformed the tuned third-party PathMerge implementation on the evaluated graphs. | SUPPORTED | Four evaluated graphs on one GH200; GPU_Opt fixed b512; PathMerge graph-wise tuned | Small trial counts; no generalization to other graphs, GPUs, or PathMerge implementations |
| RQ2 | Hybrid BFS and dual-stream execution were the main observed contributors; warp-cooperative accumulation was graph-dependent; block was faster in the forced kernel tests. | SUPPORTED_WITH_LIMITATIONS | H/W/A on four synthetic graphs and email-EuAll; forced shared/block on roadNet-PA/TX | Observed main effects are not a causal decomposition and do not cover roadNet H/W/A |
| RQ3 | Unified Memory extended the feasible batch range beyond Pure, and Chunked extended it further within the tested range. | SUPPORTED_WITH_LIMITATIONS | Legacy feasibility on 325557_3216152 and one GH200 | One graph; environment-dependent boundaries; feasibility differs from performance and correctness |
| RQ4 | Small independent-reference comparisons passed, while broader stress and cross-implementation agreement remain unresolved. | SUPPORTED (small independent-reference scope); NOT_YET_SUPPORTED (stress and PathMerge-cross scope); CORE_FAIL (overall memory-path matrix) | Three small independent-reference graphs; same-batch and stress comparisons on 325557_3216152 | No large independent reference; stress cause unresolved; PathMerge is not ground truth |

> Source: Chapters 6–9; formal evidence status follows the archived claims and coverage records.

RQ1はChapter 6の4グラフとGH200環境で`SUPPORTED`である。小標本、第三者PathMergeへの限定、GPU_Optのbatch sweep未実施が残る。

RQ2はChapter 7の範囲で`SUPPORTED_WITH_LIMITATIONS`である。H/W/AのroadNet測定がなく、main effectは単独の因果証明ではない。

RQ3はChapter 8の325557 legacy feasibility範囲で`SUPPORTED_WITH_LIMITATIONS`である。容量境界は1グラフ、GH200、資源構成、tested rangeに依存し、current系列と統合しない。

RQ4はChapter 9の小規模独立参照範囲で`SUPPORTED`である。一方、stressとPathMerge crossは`NOT_YET_SUPPORTED`、overall memory-path statusは`CORE_FAIL`であり、包括的な正確性は未確立である。
