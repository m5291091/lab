# Abstract

媒介中心性（Betweenness Centrality, BC）は、各頂点が他の頂点対の最短経路上にどれだけ現れるかを表す指標である。厳密な全始点 BC は、始点ごとの幅優先探索と逆順の依存度累積を全頂点について反復するため計算コストが高く、GPU 並列化が有力な高速化手段となる。しかしグラフ処理は不規則で、フロンティアサイズの変動と次数の偏りが負荷不均衡を生む。さらに、複数の始点をまとめる source batching は並列性を高める一方、始点ごとの状態配列を同時始点数だけ保持するため、バッチ依存の working set が GPU メモリ容量を制約する。よって実行基盤は、実行時間に加え容量範囲と数値的正確性も評価する必要がある。

本研究は、NVIDIA GH200 を対象に、無向・非重みグラフの厳密な全始点 BC を計算する batch-based GPU execution framework を設計・実装した。共通基盤は block-based source assignment、Hybrid BFS、Warp-Cooperative Accumulation、Dual-Stream Execution を統合する。個別要素の初出は主張せず、一貫した実行フローへ統合した点を貢献とする。主実装は Unified Memory を用いる GPU_Opt であり、GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunked は独立した三提案ではなく、共通基盤上の memory-management variants である。

主性能比較は GH200 1 台、email-EuAll と roadNet-PA/TX/CA の 4 グラフで行った。1 ストリームあたり固定バッチ 512 の GPU_Opt は、グラフごとにバッチサイズを調整した評価対象の第三者実装 PathMerge に対し、median/median の比較で 1.31〜3.17 倍高速であった。アブレーションでは、Hybrid BFS と Dual-Stream Execution が主要な観測寄与を示し、Warp-Cooperative Accumulation の効果はグラフに依存した。

メモリ特性は修正版 325557 グラフで比較した。Pure は b4096 まで成功し b8192 で CUDA device OOM、UM は b10240 まで成功し b12288 で host メモリ資源制限に伴う OOM kill となった。Chunked は source sub-batching で同時 resident な working set を制限し、試験上限 b16384 まで成功した。その主な価値は最高性能ではなく実行可能バッチ範囲の拡大にある。正確性は二層で検証し、小規模 3 グラフにおける独立 CPU 参照実装との全ベクトル比較、および修正版 325557 の 10 件の実装間比較は、いずれも mixed tolerance の範囲で一致した。ただし全 13 比較は byte-identical ではなく、実装間比較は独立した ground truth との一致ではない。

本研究は、共通の GPU 実行基盤について性能、要因寄与、容量、数値挙動を一貫した条件下で評価し、評価範囲において性能上の優位と容量拡張性を確認した。結果は GH200、評価したグラフ、保存した実装 snapshot に限定され、他の GPU、未測定グラフ、PathMerge 一般へ一般化しない。UM および Chunked が無制限に OOM を回避するとも主張しない。
