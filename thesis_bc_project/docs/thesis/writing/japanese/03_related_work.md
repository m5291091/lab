# Chapter 3 Related Work

本章では、厳密 Betweenness Centrality（BC）の計算、GPU による BC 計算、direction-optimizing BFS、比較対象とした GPU baseline、およびメモリ容量を越える処理に関する研究を整理する。Chapter 2 では BC の定義と Brandes algorithm の計算過程を示した。本章ではそれらを繰り返すのではなく、並列化とメモリ管理がどのように発展してきたかを中心に述べる。最後に、本研究が既存要素をどの範囲で統合し、何を評価対象とするかを明確にする。

## 3.1 Exact Betweenness Centrality

Brandes algorithm は、全頂点対の最短路を個別に列挙せず、各 source に対する最短路探索と依存度の逆向き累積を組み合わせる。非重みグラフでは、1 source の処理を BFS と辺走査で実行できるため、all-sources の時間計算量は $O(|V||E|)$、補助空間は $O(|V|+|E|)$ となる [@brandes2001]。これは、単純な全頂点対処理に比べて厳密 BC の実用性を大きく高めた。計算式と依存度漸化式は Chapter 2 で示したため、ここでは source-by-source processing が並列化の単位を与える点に注目する。

ある source の寄与は、他の source の探索状態を参照せずに計算できる。したがって、複数の source を CPU thread、GPU、あるいは分散した計算資源へ割り当て、最後に頂点ごとの部分 BC を加算する source-level parallelism が成立する。ただし、各 worker が距離、最短路数、依存度、frontier などの source-local state を必要とする。source 数を増やすほど並列性は高まる一方、同時処理数に比例して作業領域が増える場合がある。

Sarıyüce らは、この粗粒度並列を CPU 実装に用いた。各 source の寄与を 1 thread が担当し、thread ごとに探索・累積用の配列を保持することで、source 間の同期を抑えている。評価には multi-core CPU の複数 thread が含まれ、同じ研究の GPU 側では source 内をさらに細粒度に並列化している [@sariyuce2013]。この構成は、multi-core parallelization では source の独立性が扱いやすい一方、per-thread state と最終 reduction が必要になることを示す。

distributed-memory processing にも同じ分解を適用できる。すなわち、source 集合を rank または node 間で分割し、各計算資源が部分 BC vector を生成した後に加算する。具体例として、McLaughlin and Bader は root の部分集合を multi-node・multi-GPU 環境へ分配し、各 GPU の結果を統合する厳密 BC 計算を示した [@mclaughlin2014]。この方式では source-level parallelism が通信の少ない外側の並列性を与えるが、グラフ複製の容量、部分 vector の reduction、および source ごとの負荷差が分散実行の設計条件となる。

source 単位の並列化と source 内の並列化は排他的ではない。前者は独立した traversal を並行実行する coarse-grained parallelism であり、後者は 1 source の frontier や辺を複数 worker で処理する fine-grained parallelism である。CPU の thread 数、GPU の実行幅、グラフの frontier 幅、および利用可能メモリに応じて、両者を階層的に組み合わせる余地がある。

厳密 BC と approximate BC は区別する必要がある。厳密 all-sources BC は全 source の寄与を処理する。これに対し、source sampling を用いる近似法は、選択した一部の source から全体を推定し、計算量と誤差の trade-off を導入する。例えば cuGraph の公式 API も、全 source を指定した exact calculation と、seed source を指定する approximation を別の実行条件としている [@rapidsCugraphBcDocs]。近似 BC は大規模化の重要な方向であるが、本研究の直接比較対象ではない。本研究は、無向・非重みグラフに対する exact all-sources BC を対象とし、source を省略しない。

## 3.2 GPU-Based BC Computation

GPU 上の BC 計算では、Brandes algorithm の source 間並列性に加えて、各 source の BFS と dependency accumulation を SIMT 実行へ写像する必要がある。主要な設計軸は、同時に処理する source 数、frontier の表現、vertex または edge への thread 割当て、最短路数の競合更新、および source-local state の配置である。グラフ処理は frontier サイズと頂点次数が段階ごとに変化するため、規則的な dense computation と異なる負荷分散問題を持つ。

Sarıyüce らは、coarse-grained CPU parallelism と fine-grained GPU parallelism を組み合わせた heterogeneous BC を検討した。GPU 側では、frontier の vertex に thread を割り当てる vertex-based 方式と、frontier から出る edge に thread を割り当てる edge-based 方式を比較している。vertex-based 方式は frontier vertex を起点として隣接辺を走査する。高次数頂点を担当した thread の処理が長くなるため、次数分布が偏るグラフでは load imbalance が生じやすい。edge-based 方式は辺単位の並列性と memory access の整列を得やすい一方、辺 frontier の管理と追加の atomic operation を必要とする [@sariyuce2013]。

最短路数 $\sigma$ は、同一 level の複数 predecessor から同じ vertex へ寄与が到達すると競合する。このため、細粒度 GPU 実装では atomic update が必要になる。dependency accumulation でも、複数 successor からの寄与を安全に統合する必要がある。Sarıyüce らは predecessor edge list を明示的に保存せず、距離の level relation から逆向きの関係を判定する。また、高次数頂点を仮想的な複数 vertex に分ける vertex virtualization により、処理量の偏りと atomic contention を緩和する [@sariyuce2013]。これらは、GPU BC の性能が演算数だけでなく、frontier 構造、次数分布、同期頻度、および memory layout に左右されることを示す。

McLaughlin and Bader は、厳密 BC の GPU 実装について、work-efficient traversal と edge-parallel traversal をグラフ構造に応じて選ぶ方法を示した。work-efficient 方式は、明示的な vertex frontier を用いて到達済み領域の不要な再走査を避ける。edge-parallel 方式はより広い並列性と連続的な memory access を得やすい。両者の選択は、frontier の変化や初期 traversal から得る構造情報に基づく [@mclaughlin2014]。同研究は CACM Research Highlight としても公表されている [@mclaughlin2018]。本章では、2018 publication page を書誌的な補強にのみ用い、手法詳細と実験条件は全文を確認した 2014 conference paper に基づく。これは、後述する top-down と bottom-up の切替とは別の hybridization であり、BC における作業効率と GPU 利用率の trade-off を扱う。

同研究は、source を GPU 間へ割り当てる source-level parallelism と、各 source の探索を GPU thread で処理する fine-grained parallelism を階層的に用いる。単一 GPU の評価には Kepler 世代の GTX Titan、multi-node 評価には Fermi 世代の Tesla M2090 が用いられた [@mclaughlin2014]。Sarıyüce らの評価は Tesla C2050 と multi-core CPU を用いている [@sariyuce2013]。対象 GPU、CUDA 世代、グラフ集合、正規化の有無、計時範囲、および出力定義が本研究と同一ではない。そのため、これらの既報性能値を本研究の GH200 上の値と直接比較しない。

GPU memory capacity は source-level parallelism の上限を決める。複数 source の距離、$\sigma$、dependency、frontier、および traversal order を同時に保持する設計では、batch size と頂点数の積に比例する領域が支配的になり得る。batch を大きくすると並列性を増やせるが、同時 resident な作業集合も増える。逆に batch を小さくすると容量は抑えられるが、kernel launch、初期化、および同期の相対負担が増える。この問題は、グラフ本体のサイズだけでは説明できない。

既存研究の結果を総合すると、GPU BC の性能は graph-dependent である。幅の広い frontier は fine-grained parallelism を供給しやすいが、atomic contention を増やす場合がある。高 diameter で frontier が狭いグラフでは GPU occupancy が低下し、不要な走査の影響も大きくなる。次数の偏りは vertex-based assignment の負荷差を拡大する。したがって、ある GPU BC 実装が特定のグラフで高性能であっても、その順位を異なる実装条件とグラフ構造へ一般化することはできない。本研究は、既存 GPU BC 全般を対象とする性能順位を主張しない。

## 3.3 Direction-Optimizing BFS

level-synchronous な top-down BFS は、現在の frontier に含まれる vertex から外向きに辺を調べ、未訪問の隣接 vertex を次の frontier へ追加する。frontier が小さい探索初期や、高 diameter で frontier が狭く推移するグラフでは、到達済み領域だけを起点にするこの方式が自然である。一方、探索中盤で frontier が大きくなると、同じ未訪問 vertex に多くの frontier edge が到達し、既訪問 vertex への辺も繰り返し調べる可能性がある。

bottom-up BFS は探索方向を反転する。未訪問 vertex を候補とし、その隣接 vertex のいずれかが現在の frontier に含まれるかを調べる。frontier neighbor を一つ見つければその vertex の探索を打ち切れるため、frontier が大きく、未訪問領域が急速に縮む段階では調べる辺数を減らせる。ただし、frontier が小さいと、多数の未訪問 vertex が frontier neighbor を見つけられず、bottom-up の走査が無駄になる。

Beamer らの direction-optimizing BFS は、top-down と bottom-up を level ごとに切り替える。論文では、$m_f$ を現在の frontier から出る辺数、$m_u$ を未訪問 vertex から出る辺数の推定量として扱う。top-down の予想 work が大きくなり、

$$
m_f > \frac{m_u}{\alpha}
$$

を満たすと bottom-up へ切り替える。bottom-up から top-down へ戻る条件には、frontier vertex 数 $n_f$ と全 vertex 数 $n$ を用い、$n_f < n/\beta$ を用いる [@beamer2012]。$\alpha$ は top-down と bottom-up の作業量を比較する尺度、$\beta$ は frontier が縮小した後に bottom-up を継続しすぎないための尺度である。

Beamer らは、論文中の評価グラフ群に対する parameter study を行い、平均性能と最悪側の挙動を踏まえて $\alpha=14$ を選び、多くのグラフで良好であった値として $\beta=24$ を選んでいる [@beamer2012]。したがって、この二つの数値が一次論文に明記されていることは確認できる。ただし、あらゆるグラフやハードウェアに対する普遍的な最適値ではない。これらは、Beamer らの評価で用いられ、本実装でも採用した切替パラメータである。

direction optimization の効果は graph structure に依存する。低 diameter で frontier が急拡大するグラフでは bottom-up が不要な edge examination を削減しやすい。これに対し、高 diameter で frontier が小さいグラフでは切替機会が少なく、frontier representation の変換、$m_f$ の計算、および未訪問領域の走査が overhead になり得る。Beamer らも、方向切替の利益と bitmap・queue 変換などの費用を区別して評価している [@beamer2012]。

本研究の Hybrid BFS は、この既存の方向切替という考え方を、BC の source-local BFS へ統合したものである。各 source の最短路数と traversal order を保持しながら、frontier と未訪問領域の関係に応じて top-down と bottom-up を切り替える。Hybrid BFS の起源を本研究に帰属させない。本研究の対象は BFS 単体ではなく、後続の dependency accumulation を含む exact all-sources BC であるため、切替が BC 全体へ与える効果は component ablation によって評価する。

## 3.4 PathMerge and GPU Baselines

Galliot は、path-merging により GPU 上の BC 計算における補助メモリ消費を抑えることを目的として公表された研究である [@zheng2023galliot; @zheng2023jsac]。本研究では、Galliot を、Brandes-based な per-source state の保持方法とは異なる memory-oriented な GPU BC の研究方向として位置づける。一方、入手できた一次資料では論文本文の独立照合が完了していないため、path merging の内部手順、詳細な BFS strategy、実装仕様、性能値、および実装上の漸近量を本章で補完しない。

本研究で PathMerge と呼ぶ評価対象は、Galliot 論文のコードそのものではなく、第三者実装である。評価対象とした上流の第三者repositoryは `gobardhanm/path-merging-bc` であり、原著論文著者の公式実装とは確認されていない [@pathmergeRepo]。論文との実装忠実性と、評価対象 snapshot の exact commit identity も確認されていない。したがって、保存したコードと論文上のアルゴリズムが完全に同一であるとは仮定しない。この実装は提案実装と性能・出力を対比する external comparator であり、正解を決める ground truth ではない。

評価対象とした上流の第三者repositoryには、明示的な LICENSE または COPYING file によるライセンスを確認できていない [@pathmergeRepo]。これは再配布・改変許諾に関する provenance 上の制約であり、測定された性能値の有効性とは別の問題である。詳細な取扱いは再現性と制約の議論へ委ねる。

性能比較の対象は、本研究で保存・評価した第三者実装 snapshot、評価した 4 グラフ、および GH200 環境に限定される。得られた結果を PathMerge algorithm 一般、Galliot の原著者実装、別の repository、別の GPU、または未評価グラフへ一般化しない。特に、評価対象が第三者実装であること、原著論文著者の公式実装とは確認されていないこと、external comparator であること、ground truth ではないこと、exact commit identity・論文との実装忠実性・上流ライセンスが未確認であることを、性能比較の解釈から分離して扱う。

cuGraph は、広く利用される GPU graph analytics library として補助 baseline に採用した [@rapidsCugraph]。公式 API では、source list を指定しない場合に全 source の traversal による exact BC を計算し、一部の source を指定した場合は approximate BC を計算すると区別される [@rapidsCugraphBcDocs]。本研究の評価では exact 条件を用いた。ただし、保存されている cuGraph 比較は小規模グラフの legacy 部分データに限られ、現行 block 実装を含む全グラフ統一比較ではない。

Sequential baseline は CPU 上の逐次 Brandes algorithm、OpenMP baseline は source-level parallelism を用いる CPU 並列実装である [@brandes2001; @openmp52]。これらは、GPU 化による実行方式の違いを理解するための補助対照である。cuGraph と同様に、medium および large graph を含む現行実装系列の統一測定は存在しない。したがって、Sequential、OpenMP、cuGraph の保存値から大規模グラフにおける順位を推定せず、Chapter 6 の主比較にも用いない。

以上の baseline は役割が異なる。第三者実装 PathMerge は主要な external comparator である。cuGraph は library implementation との補助比較である。Sequential と OpenMP は CPU execution との補助比較である。いずれも本研究の数値的正解を定義する ground truth ではなく、正確性は独立 CPU reference または full-vector の相互比較として別に検証する。

## 3.5 Unified Memory and Out-of-Core Processing

CUDA Unified Memory（UM）は、CPU と GPU から利用できる統一されたアドレス空間と、その配置・移動を支える memory management mechanism である。`cudaMallocManaged` による managed allocation は、host code と device code の双方から参照できる領域を確保する API である [@nvidiaCudaProgrammingGuide; @nvidiaCudaRuntimeApi]。これに対し、pure device memory は `cudaMalloc` などで device memory に領域を確保し、必要な host-device transfer を明示的に管理する。両者は addressability、placement control、および容量制約の扱いが異なる。

page migration を備える環境では、GPU が未 resident page にアクセスした際、runtime が page fault に応じて host memory と device memory の間で page を移動できる。`cudaMemPrefetchAsync` は、指定した processor の近くへ data を置くための stream-ordered hint であり、実行順序に従って migration を開始できる [@nvidiaCudaProgrammingGuide; @nvidiaCudaRuntimeApi]。prefetch は correctness を変える機能ではなく、access pattern が予測と一致する場合に page fault を減らすための性能上の手段である。

UM の初期世代については、Li らが CUDA application を対象とした評価を報告している [@li2015um]。ただし、この研究は CUDA 6.x 世代であり、後の GPU memory oversubscription を評価した研究として扱わない。Knap and Czarnul は Pascal および Volta GPU 上で、standard memory management、基本的な UM、prefetch 付き UM、および oversubscription を複数の CUDA application で比較した。prefetch の利益は application、stream の利用、および GPU に依存し、すべての条件で高速化するわけではない [@knap2019um]。したがって、prefetch の指定だけから性能向上を仮定することはできない。

oversubscription は、単一 processor の物理メモリ容量より大きい managed working set を割り当て、必要な data を device memory と host memory の間で扱う能力を指す [@nvidiaCudaProgrammingGuide]。これは device memory の容量制約を緩和するが、無制限の容量を提供しない。実行可能性は host memory の利用可能量、process または system の memory limit、page migration または remote access の cost、および access locality に依存する。頻繁な往復移動が起これば、容量を確保できても性能が低下し得る。

out-of-core graph processing は、GPU memory に graph data 全体を常駐できない場合に、graph partition、active subgraph の生成、または streaming によって必要部分を順次 GPU へ供給する研究領域である。Subway は、out-of-GPU-memory の vertex-centric graph processing に対し、各 iteration で active edge を含む SubCSR を生成し、転送量を抑える。さらに、適用可能な algorithm では subgraph processing の非同期化も扱う [@nodehisabet2020subway]。同研究は UM についても、page-fault overhead と page 単位の不要な data transfer が application と graph により性能を制限することを示している。

Shirahata らの研究も、MapReduce-based large-scale graph processing に対する out-of-core GPU memory management の既存例である [@shirahata2014outofcore]。ただし、一次資料本文を独立に取得できていないため、本章では title と監査済み metadata が支持する範囲を越えて、partitioning policy、streaming granularity、または性能特性を記述しない。

本研究の Chunked は、これらの graph partitioning または active-subgraph streaming と同じ方式ではない。Chunked が分割するのは input graph ではなく、同時に処理する source 集合である。outer batch を source sub-batch に分け、各 sub-batch で input graph 全体を使用する。すべての source を最終的に処理するため、exact all-sources BC を維持し、sampling による approximation も行わない。

この区別により、graph file size と batch-dependent working set も分けて考えられる。input graph が device memory に収まっていても、多数の source-local state を同時に確保すれば working set は device memory を越え得る。逆に、graph file 自体が大きい out-of-core graph processing では、graph topology や active subgraph の配置が主要問題になる。本研究の Chunked は前者を source sub-batching で制御する。UM は同じ source-local working set を managed allocation で扱い、Pure は device memory 内に明示配置する。このため、本研究を Subway と同一方式、または out-of-core graph processing そのものとは位置づけない。

## 3.6 Positioning of This Work

本研究は、無向・非重みグラフの exact all-sources BC を対象とする。外側では source を batch としてまとめ、block-based source assignment により 1 block が 1 source を担当する。内側では、source-local BFS に Hybrid BFS を統合し、dependency accumulation に Warp-Cooperative Accumulation を用いる。さらに、Dual-Stream Execution により初期化と計算の overlap を図る。これらは、既存の GPU parallelization、direction-optimizing BFS、warp-level primitive、および CUDA stream を BC 向けの共通実行基盤へ統合したものである [@beamer2012; @nvidiaCudaProgrammingGuide]。

メモリ方式は、この共通基盤上の variation として扱う。UM は managed allocation と runtime の migration mechanism を用いる。Pure は pure device memory と明示転送を用いる。Chunked は source batch を sub-batch に分け、同時 resident な source-local working set を制限する。したがって、UM、Pure、Chunked を三つの独立した提案とはしない。また、Chunked は graph partitioning ではなく、input graph 全体を用いる exact BC の source sub-batching である。

本研究の新規性は、Brandes algorithm、Hybrid BFS、CUDA stream、Unified Memory、または個々の warp primitive の初出性にはない。位置づけの中心は、これらを batch-based・block-based な BC execution framework に統合し、同じ実装系列の上で UM、Pure、Chunked を比較可能にした点にある。さらに、end-to-end performance だけでなく、component ablation、memory scalability、および full-vector numerical validation を同じ設計系列に結び付ける。

Table 3.1 は、関連研究と本研究の比較軸をまとめる。不明な項目は論文 title や abstract から推測せず、`Not reported` または `Not independently verified` とした。Evaluation Scope は各研究で報告された範囲または本論文での利用範囲を示し、異なる hardware 間の性能順位を表すものではない。

**Table 3.1 Comparison with Related Work**

| Work | Target | Platform | Exact BC | Parallelism | BFS Strategy | Memory Strategy | Evaluation Scope | Role in This Thesis |
|---|---|---|---|---|---|---|---|---|
| Brandes (2001) [@brandes2001] | Vertex BC | General algorithm | Yes | Source-by-source | BFS on unweighted graphs | Linear auxiliary space | Algorithmic analysis | Exact BC foundation |
| Sarıyüce et al. (2013) [@sariyuce2013] | Vertex BC | Multi-core CPU and Tesla C2050 | Yes | Source-level CPU; vertex- and edge-level GPU | Level-synchronous frontier processing | Per-thread or per-source state; no predecessor edge list | Eight social-network graphs | Heterogeneous and fine-grained GPU BC |
| McLaughlin and Bader (2014) [@mclaughlin2014] | Vertex BC | GTX Titan and Tesla M2090 cluster | Yes | Root-level multi-GPU; fine-grained GPU | Work-efficient and edge-parallel switching | Per-source state; no predecessor list | Real and synthetic graphs; single and multi-GPU | Scalable exact GPU BC |
| Beamer et al. (2012) [@beamer2012] | BFS | Multi-core CPU | N/A | Vertex- and edge-parallel BFS | Top-down and bottom-up switching | Queue and bitmap frontiers | Synthetic and real graphs | Basis of Hybrid BFS |
| Galliot published research (2023) [@zheng2023galliot; @zheng2023jsac] | Vertex BC | NVIDIA GPU | Yes in published research scope | Path merging | Not independently verified | On-board memory reduction; details not independently verified | Not independently verified beyond audited publication records | Published algorithmic context; not the evaluated implementation |
| Evaluated third-party PathMerge snapshot [@pathmergeRepo] | Vertex BC | GH200 | Yes in evaluated configuration | Batched sources | Not independently verified against Galliot | Device-resident per-source working state | Saved snapshot on four graphs and GH200; paper fidelity and exact commit identity not independently verified | Upstream license not verified; external comparator; not verified as an original-authors implementation; not ground truth; no algorithm-level generalization |
| RAPIDS cuGraph [@rapidsCugraph; @rapidsCugraphBcDocs] | Vertex BC | CUDA GPU | Exact or sampled | Not reported | Not reported | Library-managed GPU memory | Small legacy thesis subset | Supplementary library baseline |
| Subway (2020) [@nodehisabet2020subway] | General graph processing | CPU and NVIDIA GPU | N/A | Vertex-centric processing | Application-dependent | Active-subgraph generation and transfer | Six graph applications | Out-of-GPU-memory contrast |
| This work | Vertex BC | GH200 | Yes | Batched sources; one block per source; warp cooperation | Source-local hybrid top-down and bottom-up BFS | Unified Memory; pure device memory; source sub-batching | Performance, ablation, memory scalability, and full-vector validation | Unified exact BC execution framework |

Table 3.1 の本研究行は、結果値ではなく design と evaluation scope を示す。性能比較は、本研究で保存・評価した第三者実装 snapshot、4 グラフ、および GH200 に限定する。component ablation はその測定対象に、memory scalability は修正済み入力に、full-vector validation は保存された比較範囲にそれぞれ限定される。cuGraph、Sequential、OpenMP を含む補助比較は小規模 legacy data に限られ、現行 block implementation の全グラフ統一比較ではない。

以上より、本研究は個別技術の初出を主張するのではなく、exact BC の処理単位、GPU kernel、direction switching、asynchronous execution、および memory management を共通基盤上で結び付ける研究として位置づけられる。その基盤に対し、性能の差だけでなく、どの component が寄与したか、どの memory path がどの容量条件で実行可能か、BC vector が数値的にどの程度一致するかを評価する。同時に、未評価グラフ、別 hardware、別 PathMerge implementation、UM の未測定 migration behavior、および stress condition の一般化を未解決条件として残す。

<!--
Source notes (internal):
- Chapter definitions and novelty wording were aligned with docs/thesis/writing/japanese/02_background.md and 04_proposed_gpu_execution_framework.md.
- Evaluation-scope limitations were checked against Chapters 5--11, result/CLAIMS.md, evidence_matrix.tsv, thesis_values.tsv, and result/provenance/.
- PathMerge provenance was checked against docs/thesis/SOURCE_AUDIT.tsv, the repository PathMerge README/SOURCE records, and the upstream third-party repository gobardhanm/path-merging-bc. Citation source type for pathmergeRepo: Upstream third-party repository page.
- Galliot details are intentionally limited because the audited author/publisher records did not provide independently checked full text. Shirahata is limited to verified bibliographic metadata for the same reason.
- Li et al. is limited to the audited CUDA 6.x evaluation scope because the publisher full text was not independently readable. No dedicated audited citation for CPU-only distributed exact BC exists in references.bib; the distributed-memory discussion is therefore limited to the source decomposition implied by Brandes and the concrete multi-node multi-GPU study by McLaughlin and Bader.
- The Beamer et al. author PDF, Section VI-B, “Tuning alpha and beta,” selects alpha=14 to maximize the average and minimum performance in the evaluated suite and beta=24 because it works well for the majority of those graphs. This chapter therefore describes them as switching parameters used in Beamer et al.'s evaluation and also adopted by this implementation, not as universal optima.
- The McLaughlin and Bader 2018 publication page is used only as a bibliographic record of the CACM Research Highlight; detailed method and experimental claims rely on the independently checked 2014 conference paper.
-->
