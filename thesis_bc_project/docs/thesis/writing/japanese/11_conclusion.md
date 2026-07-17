# Chapter 11 Conclusion

## 11.1 Summary

本研究は、大規模グラフに対する厳密なBetweenness Centrality（BC）計算を対象とした。全始点から探索と依存度累積を繰り返すBC計算では、実行時間に加えて、始点ごとの状態を保持するメモリ容量が制約となる。そこで本研究は、GPU上で始点をバッチ処理し、実行性能と実行可能なメモリ容量を両立させることを課題とした。

この課題に対し、共通のbatch-based GPU execution frameworkを設計した。この基盤は、Hybrid BFS、block単位の始点処理、Warp-Cooperative Accumulationを含むwarp/thread経路、およびDual-Stream Executionを統合する。GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunkedは独立した提案ではない。これらは、それぞれUnified Memory（UM）、device-only memory、sub-batch分割を用いる、共通基盤上のメモリ管理方式である。

評価では、固定設定のGPU_Optを、グラフごとに調整した第三者実装PathMergeと主要性能比較した。さらに、H/W/Aの全構成アブレーション、forced shared/block比較、UM/Pure/Chunkedの容量評価を行った。数値的正確性については、小規模グラフの独立参照とのfull-vector比較、PathMergeの異バッチ比較、同一バッチのメモリ経路比較、および大バッチstress条件とcross-implementation条件の比較を区別して検証した。これにより、性能結果だけでなく、構成要因、容量境界、数値的な成立範囲と未解決範囲を同じ評価体系で整理した。

## 11.2 Answers to the Research Questions

**RQ1（Performance）.** 固定b512のblock GPU_Optは、評価したemail-EuAllおよびroadNet-PA/TX/CAにおいて、グラフごとに調整した第三者実装のPathMergeより1.31〜3.17倍高速だった。内訳はemail-EuAllが3.17倍、roadNet-PAが1.31倍、roadNet-TXが1.51倍、roadNet-CAが1.45倍であり、speedupはmedian/medianである。GPU_Optは1ストリーム当たりb512、NS_eff=2であり、PathMergeの調整バッチは順にb2048、b64、b64、b32であった。比較対象は原著者の公式実装と確認されたものではない第三者実装であり、external comparatorであってground truthではない。この回答は、評価したsnapshot、4グラフ、1台のGH200、および保存された測定条件に限定され、PathMerge一般への性能一般化を意味しない。また、GPU_Optについてグラフ別のbatch探索は実施していない。

**RQ2（Optimization Contributions）.** 評価した範囲では、Hybrid BFSとDual-Stream Executionが主要な正の寄与を示した。合成4グラフにおける主効果の幾何平均は、Hybrid BFSが1.655倍、Dual-Stream Executionが1.396倍であった。Warp-Cooperative Accumulationは1.065倍であり、email-EuAllでは0.970倍であったため、その効果はグラフ依存である。forced比較では、blockカーネルがroadNet-PA/TXでsharedカーネルよりそれぞれ1.52倍、1.66倍高速であった。ただし、H/W/Aのアブレーション結果をroadNet全体や未測定グラフへ拡張せず、forced比較も評価した2グラフに限定する。各主効果とカーネル比較は異なる実験で得た観測量であり、単一要因だけで主要性能比較の差全体を説明するものではない。

**RQ3（Memory Scalability）.** 325557_3216152を用いたlegacy feasibility評価では、GPU_Opt_Pureはb4096まで、UM方式のGPU_Optはb10240まで成功し、GPU_Opt_Pure_Chunkedは試験上限のb16384まで成功した。したがって、この1グラフと試験範囲では、UMはPureより大きな要求バッチへ到達し、Chunkedはさらに大きな要求バッチへ到達した。Chunkedの主要な利点は、最高性能ではなく容量拡張性にある。一方、GPU_Optのb12288はexit 137を伴う`OOM_OR_FAIL`であり、原因は独立に確認されていない。UMの容量は有限であり、Chunkedもb16384より先を試していない。このため、いずれの方式にも無制限な実行可能性を認めず、legacyのfeasibility境界を現行block実装の性能結果へ読み替えない。

**RQ4（Correctness and Numerical Behavior）.** 小規模3グラフでは、GPU_Optと独立参照Sequentialのfull-vector比較がすべてPASSであり、mismatch、missing、NaN/Infはいずれも0であった。PathMergeのemail-EuAllとroadNet-CAにおける異バッチの歴史的比較summaryもPASSを記録するが、比較vector本体は現在取得不能である。roadNet-PA/TXの証拠水準はMax BCのみであり、full-vector検証ではない。325557_3216152の同一b1024では、UM/Pure/Chunkedの3経路が混合許容内でmismatch=0であったが、SHA256は異なりbyte-identicalではない。一方、stress比較は2組でそれぞれmismatch=6、影響頂点の和集合は8であり、正式判定はFAILである。PathMergeとのcross-implementation比較にも差があり、PathMergeはground truthではないため正誤は未決定である。canonicalなmemory-path比較のoverall statusは`CORE_FAIL`であり、stress差の原因も確定していない。以上から、小規模3グラフにおける独立参照との数値的一致は支持されるが、広いmemory-pathおよびcross-implementation範囲の正確性は未解決である。

## 11.3 Contributions

本研究の貢献は、個々の要素技術の初出性ではなく、BC向けGPU実行基盤としての統合と、その成立範囲を一貫して評価した点にある。具体的な貢献は次の4項目である。

1. **Integrated GPU Execution Framework.** 大規模な全始点BC計算に向けたbatch-based GPU execution frameworkを設計し、Hybrid BFS、block単位の始点処理、warp/threadによる依存度累積経路、および2ストリーム実行を一つの基盤として統合した。さらに、UM、device-only memory、sub-batch分割を、共通基盤上のメモリ管理方式として位置づけた。

2. **Performance Evaluation against a Tuned GPU Comparator.** 固定b512のblock GPU_Optと、グラフごとに調整した第三者実装PathMergeを比較した。評価した4グラフとGH200環境において、median/medianで1.31〜3.17倍の性能差を確認した。比較対象の第三者性、測定snapshot、ハードウェア、グラフ、およびバッチ設定を明示し、この範囲を越える性能関係とは区別した。

3. **Component-Level Analysis.** Hybrid BFS、Warp-Cooperative Accumulation、Dual-Stream Executionの全8構成を用いたH/W/Aアブレーションと、roadNet-PA/TXにおけるforced shared/block比較を実施した。これにより、主要な観測効果とグラフ依存の効果を区別し、end-to-endの性能差を単一要因へ還元しない構成要因分析を示した。

4. **Memory Scalability and Numerical-Boundary Analysis.** UM/Pure/Chunkedの実行可能バッチ範囲と失敗境界を比較し、メモリ管理方式による容量特性を示した。同時に、小規模独立参照full-vectorのPASS、same-batchの混合許容内一致、stress条件の差、PathMergeとのcross-implementation差、および`CORE_FAIL`を区別して記録した。これにより、容量拡張の実行成功と数値的正確性を別の評価軸として扱った。

## 11.4 Final Remarks

本研究は、評価した4グラフとGH200環境において主要な性能目標を達成し、メモリ管理方式による容量拡張の可能性と方式間の違いを示した。同時に、成功した実行だけでなく、OOM、原因未確認の`OOM_OR_FAIL`、現在取得不能なartifact、および`CORE_FAIL`を保存された証拠水準のまま記録した。これらの記録は、実行可能性、性能、数値的正確性を相互に代替できない評価軸として扱うために重要である。

大バッチstress条件とPathMergeとのcross-implementation比較には、未解決の数値差が残る。また、他GPU、他グラフ、および現行block実装による統一baselineへの一般化には追加検証が必要である。本研究は、これらの制約を保持しながら、性能、メモリ容量、正確性を同時に検討するための実装・評価基盤を提供した。本研究は、評価範囲に限定した性能向上を実証するとともに、GPUによるBC計算では、性能、メモリ容量、数値整合性を統合して評価する必要があることを示した。
