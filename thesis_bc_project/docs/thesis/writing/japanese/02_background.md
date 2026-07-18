# Chapter 2 Background

本章では、本研究を理解するために必要な Betweenness Centrality（BC）、Brandes アルゴリズム、BC 計算の並列性、CUDA の実行モデル、および NVIDIA GH200 のメモリアーキテクチャを整理する。ここでは一般的な定義と技術的背景を扱う。提案する batch-based GPU execution framework の具体的な設計は Chapter 4 で述べる。

## 2.1 Graphs and Betweenness Centrality

グラフを $G=(V,E)$ と表す。$V$ は vertex の有限集合、$E$ は edge の集合である。頂点数を $n=|V|$、辺数を $m=|E|$ とする。directed graph では edge は順序対 $(u,v)$ であり、$u$ から $v$ への向きをもつ。undirected graph では edge $\{u,v\}$ に向きはなく、$u$ と $v$ の隣接関係は対称である。本研究が計算時に扱う入力は undirected graph である。

頂点 $s$ から $t$ への path は、頂点列 $p=(v_0,v_1,\ldots,v_k)$ であり、$v_0=s$、$v_k=t$、かつ連続する各頂点間に対応する edge が存在する。非重みグラフでは path length を edge 数 $k$ とする。$s$ と $t$ を結ぶ path のうち length が最小のものを shortest path と呼び、その長さを distance $d(s,t)$ とする。directed graph では各 edge の向きに従う path のみを許す。undirected graph では path を逆向きにたどることもできる。

$s$ から $t$ への shortest path の総数を $\sigma_{st}$ とする。また、$s$ と $t$ 以外の頂点 $v$ を内部頂点として通る shortest path の数を $\sigma_{st}(v)$ とする。$s$ から $t$ へ到達できない場合、その頂点対の寄与を 0 とする。したがって、$\sigma_{st}>0$ である頂点対について、比 $\sigma_{st}(v)/\sigma_{st}$ は shortest path を一様に数えたときに $v$ を通る割合を表す。

端点を含めない directed BC は、ordered source--target pair に対して次式で定義できる [@brandes2001]。

$$
C_B^{\mathrm{dir}}(v)
=
\sum_{\substack{
s,t\in V\\
s\ne t\\
s\ne v\\
t\ne v
}}
\frac{\sigma_{st}(v)}{\sigma_{st}}.
$$

この式では $v=s$ または $v=t$ となる pair を除いている。このため、source または target であること自体は BC に加算されない。endpoints を含む別定義では端点としての寄与も加えるが、本研究ではその定義を用いない。

undirected graph では、shortest path の向きを反転した $(s,t)$ と $(t,s)$ が同じ非順序頂点対を表す。全 source から ordered pair を数えると、両者は同じ内部頂点へ同じ寄与を与える。この重複を除く undirected BC は、次の $1/2$ 補正で表される。

$$
C_B^{\mathrm{undir}}(v)
=
\frac{1}{2}
\sum_{\substack{
s,t\in V\\
s\ne t\\
s\ne v\\
t\ne v
}}
\frac{\sigma_{st}(v)}{\sigma_{st}}.
$$

係数 $1/2$ は、undirected graph における $(s,t)$ と $(t,s)$ の重複計数を補正するためである。本研究の実装も全頂点を source として ordered pair 相当の依存度を累積し、各 source の寄与を global BC に加える際に $1/2$ を適用する。

normalization は、異なる頂点数のグラフ間で値の尺度をそろえるための追加の scaling である。$n>2$ かつ endpoints を含めない場合、ordered pairs を用いる directed BC と、ordered-pair sum に $1/2$ を適用した undirected BC は、それぞれ次のように正規化できる。

$$
\widehat{C}_B^{\mathrm{dir}}(v)
=\frac{C_B^{\mathrm{dir}}(v)}{(n-1)(n-2)},
\qquad
\widehat{C}_B^{\mathrm{undir}}(v)
=\frac{C_B^{\mathrm{undir}}(v)}{(n-1)(n-2)/2}.
$$

本研究で使用する BC は、undirected、unweighted、exact all-sources、unnormalized、endpoints excluded である。すなわち、$C_B^{\mathrm{undir}}$ を計算し、上式の normalization は適用しない。入力データの由来上の directedness ではなく、実験へ渡されるグラフ表現の directedness を基準とする。たとえば directed な原データを前処理で無向化した場合、計算対象は undirected graph である。これらの条件は Chapter 5 の評価設定、および Sequential、GPU_Opt、PathMerge の集計処理と整合する。

**Table 2.1: Symbols used in this thesis.**

| Symbol | Meaning |
|---|---|
| $G=(V,E)$ | Graph with vertex set $V$ and edge set $E$ |
| $n=|V|$ | Number of vertices |
| $m=|E|$ | Number of edges |
| $s,t$ | Source and target vertices |
| $d(s,t)$ | Shortest-path distance from $s$ to $t$ |
| $d_s(v)=d(s,v)$ | Shortest-path distance from source $s$ to vertex $v$ |
| $\sigma_{st}$ | Number of shortest paths from $s$ to $t$ |
| $\sigma_{st}(v)$ | Number of shortest paths from $s$ to $t$ that contain $v$ as an internal vertex |
| $\sigma_s(v)=\sigma_{sv}$ | Number of shortest paths from source $s$ to vertex $v$ |
| $P_s(w)$ | Predecessors of $w$ in the shortest-path DAG rooted at $s$ |
| $\delta_s(v)$ | Dependency of source $s$ on vertex $v$ |
| $C_B^{\mathrm{dir}}(v)$ | Unnormalized directed betweenness centrality based on ordered pairs |
| $C_B^{\mathrm{undir}}(v)$ | Unnormalized undirected betweenness centrality with the $1/2$ correction |
| $\widehat{C}_B^{\mathrm{dir}}(v),\ \widehat{C}_B^{\mathrm{undir}}(v)$ | Normalized directed and undirected betweenness centrality |
| $S$ | Stack of vertices in nondecreasing BFS distance order (traversal stack; $S_s$ denotes the stack of source $s$) |
| $M_{\mathrm{work}}$ | Conceptual batch-dependent working-set size |
| $NS_{\mathrm{eff}}$ | Effective number of simultaneously active stream buffers |
| $\mathrm{EffectiveBatch}$ | Number of sources provisioned per effective stream buffer |
| $M_{\mathrm{source}}$ | Source-local state size |

speedup には $\mathrm{Speedup}$ を用い、traversal stack $S$ と記号を共有しない（5.6 節）。

## 2.2 Brandes Algorithm

BC の定義式を頂点対ごとに直接評価すると、各 pair の shortest path を個別に列挙する必要がある。Brandes アルゴリズムは、1 source から得た shortest-path DAG を共有し、target ごとの寄与を dependency としてまとめて逆向きに累積する [@brandes2001]。非重みグラフでは、各 source $s\in V$ に対して Forward BFS と Backward Dependency Accumulation の 2 phase を実行する。

Forward phase では、$s$ から Breadth-First Search（BFS）を行う。各頂点 $v$ について、distance $d_s(v)=d(s,v)$、$s$ から $v$ への shortest-path count $\sigma_s(v)=\sigma_{sv}$、および shortest-path DAG 上の predecessor set $P_s(v)$ を求める。初期状態は $d_s(s)=0$、$\sigma_s(s)=1$ であり、未訪問頂点の distance は $-1$、path count は 0 とする。

BFS で edge $(v,w)$ を調べたとき、$w$ が未訪問なら $d_s(w)=d_s(v)+1$ とし、次の frontier へ追加する。さらに $d_s(w)=d_s(v)+1$ なら、この edge は shortest-path DAG に属する。この場合、$\sigma_s(w)$ に $\sigma_s(v)$ を加え、$v$ を $P_s(w)$ へ追加する。同じ level の $w$ に複数の predecessor がある場合、そのすべての path count が加算される。

BFS から取り出した頂点は traversal stack $S$ に格納する。$S$ の順序は distance の非減少順である。したがって、$S$ を後ろから取り出せば、source から遠い頂点を先に処理できる。この逆順が Backward phase の依存関係を満たす。

source $s$ の vertex dependency $\delta_s(v)$ は、$v$ より 1 level 深い successor から次式で計算される。

$$
P_s(w)=\{v\in V\mid (v,w)\in E,\ d_s(w)=d_s(v)+1\},
$$

$$
\delta_s(v)
=
\sum_{w:\,v\in P_s(w)}
\frac{\sigma_s(v)}{\sigma_s(w)}
\left(1+\delta_s(w)\right).
$$

$1$ は target $w$ 自身に対応し、$\delta_s(w)$ は $w$ より先にある target 群の寄与を表す。deep level から shallow level へ処理することで、右辺の $\delta_s(w)$ は使用時点ですでに確定している。各 $w\ne s$ について $\delta_s(w)$ を $C_B(w)$ に加えれば、source $s$ に関する全 target の寄与をまとめて反映できる。すべての source についてこの処理を反復し、undirected graph では最後に ordered pair の重複を $1/2$ で補正する。

**Algorithm 2.1: Brandes Algorithm for Unweighted Graphs**

```text
Input: Unweighted graph G = (V, E)
Output: Unnormalized, endpoint-excluded betweenness vector CB

1:  CB[v] <- 0 for every v in V
2:  for each source s in V do
3:      S <- empty stack
4:      P[w] <- empty list for every w in V
5:      sigma[w] <- 0 and dist[w] <- -1 for every w in V
6:      sigma[s] <- 1; dist[s] <- 0
7:      Q <- queue containing s
8:      while Q is not empty do
9:          v <- Q.pop()
10:         S.push(v)
11:         for each neighbor w of v do
12:             if dist[w] < 0 then
13:                 dist[w] <- dist[v] + 1
14:                 Q.push(w)
15:             end if
16:             if dist[w] = dist[v] + 1 then
17:                 sigma[w] <- sigma[w] + sigma[v]
18:                 P[w].append(v)
19:             end if
20:         end for
21:     end while
22:     delta[v] <- 0 for every v in V
23:     while S is not empty do
24:         w <- S.pop()
25:         for each v in P[w] do
26:             delta[v] <- delta[v] + (sigma[v] / sigma[w]) * (1 + delta[w])
27:         end for
28:         if w != s then CB[w] <- CB[w] + delta[w]
29:     end while
30: end for
31: if G is undirected then CB[v] <- CB[v] / 2 for every v in V
32: return CB
```

1 source の Forward BFS と Backward phase は、shortest-path DAG の vertex と edge を高々定数回処理するため、合わせて $O(|V|+|E|)$ time である。全 $|V|$ source を処理する非重み Brandes アルゴリズムの一般的な time complexity は

$$
O\!\left(|V|(|V|+|E|)\right)
$$

である [@brandes2001]。connected graph など $|E|=\Omega(|V|)$ が成り立つ場合に限り、これは $O(|V||E|)$ と簡略化できる。

標準的な source-by-source Brandes 法の記憶量には、graph の adjacency representation $O(|V|+|E|)$、predecessor lists $O(|V|+|E|)$、および distance、path count、dependency、queue、stack の各 $O(|V|)$ が含まれる。これらを合わせた working memory は $O(|V|+|E|)$ である。input graph を auxiliary space から除外する慣習を採っても、predecessor lists が最大 $O(|V|+|E|)$ を要するため、algorithm-specific auxiliary space は $O(|V|+|E|)$ である。最終 BC vector には $O(|V|)$ を要し、同じ漸近上界に含まれる。

Algorithm 2.1 は標準 Brandes アルゴリズムを説明するための擬似コードである。本研究の GPU 実装では、predecessor lists をそのまま materialize せず、distance と CSR adjacency から次 level の関係を判定する。また、複数 source の状態を同時に保持する。これらは Chapter 4 の実装設計であり、本節の標準アルゴリズムおよびその source-by-source space complexity と区別する。

## 2.3 Parallelism in BC Computation

Brandes 型 BC には、粒度と同期条件の異なる複数の parallelism が存在する。GPU BC の先行研究でも、source 単位の coarse-grained parallelism と、頂点・edge に処理を分ける fine-grained parallelism が区別されている [@sariyuce2013; @mclaughlin2014]。

source-level parallelism は、異なる source の BFS と dependency accumulation を並行して実行する。各 source は独自の distance、$\sigma$、predecessor または traversal order、$\delta$ をもつため、source-local computation は相互に独立しやすい。一方、最終的には全 source の寄与を同じ global BC vector へ加える必要がある。並行 source が同じ $C_B(v)$ を更新する場合は、reduction、atomic operation、または source ごとの部分和と後段の merge が必要になる。

vertex-level parallelism は、同じ BFS level に属する複数頂点や、Backward phase の同じ dependency level に属する複数頂点を並行に処理する。BFS では level 間に順序依存がある。Backward phase でも、深い level の $\delta$ を確定してから浅い level を処理しなければならない。このため、level 内には並列性があっても、level 間には synchronization が必要である。

edge-level parallelism は、frontier vertex の adjacency list や、dependency 計算で参照する successor edges を複数 thread へ分配する。高次数頂点では 1 vertex の edge scan を細分化できる。一方、低次数頂点へ多くの thread を割り当てると、処理すべき edge をもたない thread が増える。degree distribution が偏るグラフでは、頂点間の仕事量が不均一になり、thread または warp の利用効率が変化する。

frontier-level parallelism は、現在の BFS frontier に含まれる active vertices と、それらに接続する edges を対象とする。frontier size は source、BFS level、連結成分、グラフ構造によって変化する。探索初期や細長いグラフでは frontier が小さく、利用できる並列度が低い場合がある。探索中盤で frontier が急に広がると、多数の edge を並行に処理できる一方、同じ未訪問頂点への競合や atomic update が増え得る。この変動性は、固定された thread 配置だけで全 level を効率よく処理することを難しくする。

batch-based source processing は、複数 source を 1 batch にまとめ、source-level parallelism を一定の実行単位として GPU へ供給する。これは graph partition や source sampling ではない。batch を順に反復して全 source を処理する限り、計算する BC の定義は変わらない。小さい frontier をもつ source でも、同時に別 source を処理すれば GPU 全体の並列度を補える可能性がある。

ただし、batch size を増やすと source-local state も増える。各 source は少なくとも distance、shortest-path count、dependency、frontier、traversal order を必要とする。したがって、batch 拡大は source-level parallelism を増やし得る一方、memory capacity、初期化量、global BC への競合、同時実行 resource を消費する。大きい batch が常に高い性能を与えるとは限らない。適切な粒度はグラフ構造、frontier の推移、degree skew、GPU resource、および source-local state の容量に依存する。

atomic operation は、この並列化で複数の役割をもつ。BFS では未訪問頂点の発見を一意に確定し、複数 predecessor からの $\sigma$ を加算するために用いられる。全 source の dependency を global BC へ merge する際にも用いられる。atomic update は read--modify--write を不可分にするが、競合する update の実行順序を固定するものではない。また、同じ address への競合が多い場合は serialization の要因となる。

## 2.4 GPU Execution Model

CUDA の基本モデルでは、CPU 側を host、CUDA-capable GPU 側を device と呼ぶ。host program は device memory の確保やデータ移動を指示し、GPU で実行する kernel を launch する。kernel は多数の thread によって実行され、その thread 群は grid、thread block、thread の階層に配置される [@nvidiaCudaProgrammingGuide]。

grid は 1 回の kernel launch に含まれる全 block の集合である。thread block は、同じ kernel を実行し、block 内の高速な shared memory と block-level synchronization を共有できる thread 群である。block は GPU 上で独立に schedule されるため、通常の kernel 内で異なる block 間の実行順序を仮定できない。thread は階層の最小のプログラム実行主体であり、`threadIdx` と `blockIdx` により担当データを決める。

thread は通常、32 thread からなる warp 単位で実行される。warp 内の thread は Single Instruction, Multiple Threads（SIMT）として命令を進める。条件分岐で lane ごとの経路が異なると、複数経路を分けて実行する必要があり、同時に有効な lane 数が減る場合がある。一方、warp shuffle を用いると、shared memory を介さず lane 間で値を交換し、reduction などの cooperative processing を実装できる [@nvidiaCudaProgrammingGuide]。

CUDA の主な memory space は、global memory、shared memory、register である。global memory は device 上の多数の thread からアクセスでき、容量は大きいが、access pattern と locality が性能へ影響する。shared memory は block 内で共有される on-chip memory であり、block 内の再利用や thread 間交換に用いられる。register は原則として各 thread に専有され、局所変数を低 latency で保持する。ただし、register と shared memory は block の同時 resident 数を左右する有限 resource でもある [@nvidiaCudaProgrammingGuide]。

synchronization は、その適用範囲を区別する必要がある。block-level barrier は同じ block の thread をそろえ、shared data の phase 間整合を取る。異なる block 間の global dependency は、一般には kernel boundary、別 kernel の launch order、event などで構成する。atomic operation は特定 address への update を不可分にするが、block 全体または grid 全体の barrier ではない。

CUDA stream は、host から device へ投入する operations の順序付き sequence である。同一 stream 内では command order が保たれる。依存関係のない異なる streams の operations は、hardware resource と memory dependency が許せば concurrent に実行され得る。asynchronous memory operation、memory transfer、kernel execution を別 streams へ配置することで、transfer または initialization と computation の overlap を構成できる。ただし asynchronous launch は即時完了や overlap を保証しない。実際の重畳は device capability、copy engine、memory type、利用可能 resource、同期点に依存する [@nvidiaCudaProgrammingGuide; @nvidiaCudaRuntimeApi]。

これらの概念は Chapter 4 の設計へ直接つながる。本研究の `1 block = 1 source` は、各 block に source-local state を対応させ、block 内 thread で frontier と adjacency を処理する。Warp-Cooperative Accumulation は 1 vertex の successor scan と部分和を warp lanes に分配する。Dual-Stream Execution は 2 組の state buffers と CUDA streams を用い、ある batch の computation と次 batch の initialization の重畳を試みる。source batching は grid に複数の source blocks を供給する。これらの mapping、同期境界、切替条件の詳細は Chapter 4 に委ねる。

## 2.5 GH200 Memory Architecture

NVIDIA GH200 Grace Hopper Superchip は、Grace CPU と Hopper GPU を同一 superchip に統合する。Hopper GPU は on-package HBM3 を主な device memory として使用し、Grace CPU は CPU-side LPDDR5X memory を使用する。両者は coherent な NVLink-C2C で接続される。NVIDIA の製品資料は、この interface の帯域を 900 GB/s とし、CPU と GPU の coherent memory model を説明している [@nvidiaGh200Product; @nvidiaGraceHopperInDepth]。

HBM3 と CPU-side memory は、同じ物理メモリ階層ではない。HBM3 は Hopper GPU に近い有限の on-package memory であり、CPU-side memory は Grace CPU 側の有限の物理 memory である。NVLink-C2C と coherent addressing は両者の連携を可能にするが、容量や access cost を同一にするものではない。CPU と GPU から共通の address space を扱えることと、すべての data が同じ物理場所に resident であることも同義ではない。

NVIDIA の一般製品資料では、対象構成の HBM3 を最大 96 GB と記載している [@nvidiaGraceHopperInDepth]。これは製品の nominal specification である。これに対し Chapter 5 の実験環境記録には、device memory 97,871 MiB、runtime query の total 約 102.0 decimal GB、実行開始時の free memory 約 101.4 decimal GB が残る。これらは同じ on-package HBM3 を異なる単位系または取得方法で表した値であり、加算可能な別領域ではない。また、free memory は実行開始時点の観測値であり、常時自由に使用できる保証容量ではない。

device memory allocation と managed memory allocation も区別する必要がある。`cudaMalloc` による device allocation は、device で使用する領域を device memory に確保する。必要量が利用可能な device memory を超えれば allocation は失敗し得る。`cudaMallocManaged` による managed allocation は Unified Memory（UM）の対象となり、CPU と GPU の双方からアクセス可能な address を提供する。CUDA runtime と driver は access、memory advice、prefetch などに応じて物理配置を管理する [@nvidiaCudaProgrammingGuide; @nvidiaCudaRuntimeApi]。

migration は、managed data の page を CPU-side memory と GPU memory の間で移動させる動作である。page fault による on-demand movement が生じる場合もあれば、program が `cudaMemPrefetchAsync` で将来の access location へ prefetch を要求する場合もある。prefetch は data placement と latency を制御する手段であり、全 page が要求どおり恒久的に resident になる保証ではない。allocation、access pattern、同時実行、available memory によって挙動は変化する。

oversubscription は、GPU で利用しようとする managed working set が、その時点で利用可能な device memory を上回る状態を指す。GH200 では NVLink-C2C と coherent memory model を通じ、Grace CPU memory を含む placement を利用できる [@nvidiaGraceHopperInDepth]。しかし UM は追加の無制限な物理容量ではない。HBM3、CPU-side physical memory、page-management overhead、runtime resource のいずれも有限である。さらに、process が使用可能な host memory は、job resource configuration や cgroup limit により物理搭載量より小さく制限され得る。したがって UM を使用しても、device allocation failure、host-memory pressure、cgroup OOM（Out of Memory）kill、または実行時間上の制約は残る。

本研究で中心となる容量は、graph file size ではなく batch-dependent working set である。on-disk graph file、in-memory CSR topology、最終 BC vector は static graph storage であり、source batch size に比例しない。一方、distance、$\sigma$、$\delta$、frontier、traversal stack、level metadata などは source ごとに必要である。1 source 当たりの状態量を $M_{\mathrm{source}}$、同時に用意する有効 stream buffer 数を $NS_{\mathrm{eff}}$、各 buffer の source 数を $\mathrm{EffectiveBatch}$ とすると、概念的な working set は次式で表される。

$$
M_{\mathrm{work}}
\approx
NS_{\mathrm{eff}}
\times \mathrm{EffectiveBatch}
\times M_{\mathrm{source}}.
$$

これは source-local arrays の code-derived allocation を理解するための概念式であり、measured process RSS、physical HBM residency、host residency、または migration bytes を表す実測式ではない。実際の必要量には static graph storage、BC vector、allocator および runtime overhead が別途加わる。source sub-batch を使う方式では、同時 resident estimate に `EffectiveBatch` ではなく実際に確保する `SUB_BATCH` を用いる。

したがって、本研究における UM の採用理由は、on-disk graph file の容量ではない。対象となるのは、多数 source の state を同時に用意することで増加する working set である。UM の役割は、その有限な working set に managed placement、migration、prefetch の選択肢を与えることである。その有効性と cost は workload と resource configuration に依存するため、実測なしに一般化できない。具体的な allocation policy は Chapter 4、実験環境と容量評価方法は Chapter 5、観測された境界は Chapter 8 で扱う。

## 2.6 Challenges Addressed in This Thesis

第 1 の課題は、exact all-sources BC が各頂点を source とする BFS を反復する点である。1 source の処理が線形時間でも、全 source の一般的な time complexity は $O(|V|(|V|+|E|))$ となる。$|E|=\Omega(|V|)$ の条件下では $O(|V||E|)$ と簡略化できる。source 間の独立性を利用できる一方、各 source は独自の shortest-path state と Backward phase を必要とする。

第 2 の課題は、Forward BFS の並列度と効率が探索中に変化する点である。frontier は source と level によって小さくも大きくもなる。top-down traversal は current frontier から outgoing edges を展開する。bottom-up traversal は未訪問側から frontier への接続を調べる。どちらが適するかは frontier と未探索領域の関係により変わるため、探索方向を固定するだけでは多様な phase を表現しにくい [@beamer2012]。一方、方向切替にも判定、frontier representation、追加走査の overhead があり、Hybrid BFS がすべてのグラフで有利であるとは限らない。

第 3 の課題は、Backward Dependency Accumulation の不規則性である。頂点ごとの successor 数は degree と shortest-path DAG に依存する。高次数頂点の adjacency scan は複数 thread または warp lanes へ分配しやすいが、低次数頂点では協調単位の一部が仕事をもたない場合がある。degree distribution と BFS level 幅の違いは、thread-per-vertex と warp-cooperative processing の利用効率を変化させる。また level 間の dependency により、完全に順序制約を除くことはできない。

第 4 の課題は、source-level parallelism と memory capacity の trade-off である。batch processing は複数 source を同時に実行して並列度を増やせる可能性がある。しかし batch を拡大すると、source-local distance、$\sigma$、$\delta$、frontier、stack も比例して増える。性能上望ましい同時 source 数と、HBM3 および CPU-side memory が許す working set の間に調整が必要になる。大きい batch が常に高速であるとも、UM がこの trade-off を消去するとも仮定できない。

第 5 の課題は、初期化、計算、global accumulation、synchronization の overhead である。各 source state の reset、BFS level 間の同期、Backward level 間の同期、および global BC への atomic update は、BC の算術演算以外の cost を生む。複数 stream による asynchronous execution は一部の initialization や memory operation と kernel execution を重畳できる可能性があるが、buffer reuse の依存関係と有限 resource を守る同期点が必要である。

第 6 の課題は、並列実行に伴う numerical behavior である。$\sigma$、$\delta$、global BC は多数の floating-point additions を含む。floating-point addition は結合則を厳密には満たさないため、thread scheduling、atomic update、batch grouping、reduction tree が変わると、数学的に同じ和でも丸め順序が変わり得る。このため、algorithmic equivalence は byte-identical output を自動的には意味しない。正確性評価では、定義、vector 全体、許容基準、NaN/Inf、provenance を分離して確認する必要がある。

以上の課題は互いに独立ではない。source batch を増やすと並列性と state 容量の双方が変わり、thread/warp の割当は degree skew と dependency phase の双方に影響する。streams による overlap は初期化 overhead を隠せる可能性がある一方、追加 buffer を必要とする。Chapter 4 では、これらの背景課題に対する実行基盤の設計を示す。本節では課題の所在だけを整理し、各設計要素の有効性や性能上の結論は先取りしない。

<!--
Source notes (not reader-facing):

- Section 2.1 conditions were cross-checked against writing/japanese/05_experimental_methodology.md (Sections 5.3--5.5), src/baseline/sequential.cpp, src/baseline/omp.cpp, src/baseline/cugraph_bc.cu, src/baseline/pathmerge.cu, and include/proposed/brandes_kernels.cuh.
- Section 2.2 standard algorithm and complexity follow references.bib:brandes2001 and SOURCE_AUDIT.tsv:S01. Implementation-specific successor reconstruction is described only as a transition to Chapter 4.
- Sections 2.3--2.4 were cross-checked against writing/japanese/04_proposed_gpu_execution_framework.md, docs/thesis/04_method_design.md, include/proposed/brandes_kernels.cuh, and the CUDA primary-source audit entries S17--S20.
- Section 2.5 product claims follow references.bib:nvidiaGh200Product and nvidiaGraceHopperInDepth, audited in SOURCE_AUDIT.tsv:S15--S16. Run-environment distinctions follow result/environment/environment.md and result/tables/thesis/T6_experimental_environment.tsv.
- The working-set distinction follows writing/japanese/04_proposed_gpu_execution_framework.md, writing/japanese/05_experimental_methodology.md, result/datasets/graph_catalog.tsv, and the current corrected-input records summarized in Chapters 8--11. Historical malformed-input conclusions are not used.
- Candidate captions, with no image assets created for this draft: "Figure 2.1: Brandes Algorithm Overview.", "Figure 2.2: GPU Execution Hierarchy.", and "Figure 2.3: GH200 Memory Architecture."
-->
