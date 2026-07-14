# 00 研究の位置づけ

## 研究対象
NVIDIA GH200 Grace Hopper Superchip（Miyabi-G, sm_90）上での、**厳密（exact）な全始点
媒介中心性（Betweenness Centrality, BC）計算**の GPU 実行基盤。無向・非重みグラフを対象と
し、全頂点をソースとする Brandes アルゴリズムをバッチ型で GPU 実行する。

## 問題設定
BC は各頂点が最短経路上でどれだけ「橋渡し」になるかを表す中心性指標であり、通信網・
ソーシャルネットワーク・道路網などの解析に用いられる。厳密 BC は全頂点対の最短経路構造を
必要とするため計算コストが高い。

### なぜ BC が高コストか
Brandes アルゴリズムは、各ソース頂点 s について (1) s からの最短経路 DAG を BFS で構築し
最短経路数 σ を求める前向きフェーズと、(2) 逆順に依存度 δ を累積する後向きフェーズを行う。
1 ソースあたり O(V + E)、全始点で **O(V·E)**（重みなし）。本研究の評価グラフでは V が
2.65×10^5〜1.97×10^6、E が 3.6×10^5〜2.77×10^6 であり、全始点計算は数十秒〜数千秒規模に
なる（`result/main_performance/proposed_variants/`、例：roadNet-CA で提案手法 median 2129.10 s）。
このため、ソースを跨いだ並列化とメモリ管理が実行時間を支配する。

### なぜ GH200 で扱うのか
GH200 は GPU（Hopper, HBM3 ≈96–102 GB）と Grace CPU（LPDDR5X）を NVLink-C2C で密結合し、
Unified Memory（UM）を通じて HBM3 を超える working set を透過的に扱える。全始点 BC の
バッチ処理はソース数に比例して動的状態（距離・σ・δ・BFS フロンティア・スタック）が
V に比例して増大し、大きなバッチでは HBM3 容量を容易に超える。GH200 の C2C 帯域と UM は、
この容量制約に対して「デバイスメモリのみ（Pure）では OOM する領域でも実行を継続する」
選択肢を与える。したがって GH200 は、**計算最適化とメモリ容量管理を同一土俵で評価できる**
プラットフォームである。実測帯域は HBM3 DtoD 1818.6 GB/s、NVLink-C2C Prefetch 177.7 GB/s
（`raw_data/profiling/job_2359175_20260711/bandwidth.log`）。

## 単なる性能評価との違い
本研究を「GPU_Opt の速度を測っただけの性能評価」として構成しない。速度は 4 つの
研究質問（[01_research_questions.md](01_research_questions.md)）のうちの 1 つ（RQ1）に
すぎず、以下を一貫した実験条件（同一 checkpoint `phase_def_block_20260710`・同一 GH200・同一集計規約）で
横断的に扱う：

1. **性能（RQ1）**：調整済み GPU baseline（PathMerge tuned）との比較。
2. **要因分析（RQ2）**：Hybrid BFS・warp 協調・2 ストリームの寄与のアブレーション分解。
3. **メモリ容量（RQ3）**：UM/Pure/Chunked の実行可能バッチ範囲と OOM 境界。
4. **数値整合性（RQ4）**：参照実装・他実装との一致範囲と、未解決の構成依存差。

## システム研究としての位置づけ
本研究は**アルゴリズム理論の新規性を主張しない**。Brandes、direction-optimizing BFS
（Beamer ら）、warp 協調還元、multi-source/batched BC、UM オーバーサブスクリプションは
いずれも既存の要素技術である。本研究の位置づけは、これらを **GH200 の計算・メモリ特性を
考慮した「バッチ型全始点 BC 実行基盤」として統合し、その振る舞いを体系的に測定した**
アーキテクチャ指向の HPC・システム実装研究である。

## 過度に主張しない新規性
新規性は次の一点に限定する：

> 既存の GPU 最適化要素（Hybrid BFS・block 単位の始点処理・依存度計算の GPU 並列化・
> 2 ストリーム実行・UM/Pure/Chunked のメモリ管理方式）を、GH200 向けのバッチ型全始点 BC
> 実行基盤として統合し、**調整済み GPU baseline との性能・各要因の寄与・メモリ容量・
> 数値的制約を、一貫した実験条件で明らかにした点**。

以下は主張**しない**（根拠がない、または実験範囲外）：
- Hybrid BFS / warp 協調 / 2 ストリーム / UM を個別の新発明として主張しない。
- 「あらゆるグラフで高速」「常に PathMerge に勝つ」とは主張しない（評価は 4 グラフ限定）。
- 「UM が容量制約を完全に解消する」とは主張しない（UM も `b12288`/`b10240` で OOM の事例あり）。
- 提案実装を「厳密性が全条件で検証済み」とは主張しない（full-vector 独立参照は小規模3グラフ限定）。

参照：`result/CLAIMS.md`, `result/COVERAGE.md`, `result/environment/environment.md`。
