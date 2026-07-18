# List of Symbols

本一覧は Chapter 1 から Chapter 11 および Abstract の本文と数式で実際に定義・反復使用されている記号のみを収録する。コード上の識別子、保存 TSV・manifest の記録列名、および局所的な一時変数は収録せず、必要なものは `Recorded Fields` および `Units and Conventions` の節で扱う。同一文字を異なる概念へ割り当てないことを原則とする。

## Graph and Betweenness Centrality Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $G=(V,E)$ | 頂点集合 $V$ と辺集合 $E$ からなるグラフ。本研究の入力は無向・非重みグラフである。 | — |
| $V$ | 頂点集合。 | 集合 |
| $E$ | 辺集合。 | 集合 |
| $n=\lvert V\rvert$ | 頂点数。試行数には用いない。 | 個数 |
| $m=\lvert E\rvert$ | 無向辺数。 | 個数 |
| $u,\ v,\ w$ | 一般の頂点。$w$ は主に Backward phase で処理中の頂点を表す。 | $V$ の元 |
| $s$ | source 頂点。Brandes アルゴリズムの探索始点。標本標準偏差には用いない。 | $V$ の元 |
| $t$ | target 頂点。 | $V$ の元 |
| $d(s,t)$ | $s$ から $t$ への最短経路長。 | 辺数 |
| $d_s(v)=d(s,v)$ | source $s$ から頂点 $v$ への最短距離。未訪問は $-1$ で表す。 | 辺数 |
| $\sigma_{st}$ | $s$ から $t$ への最短経路の総数。 | 個数 |
| $\sigma_{st}(v)$ | $s$ から $t$ への最短経路のうち $v$ を内部頂点として通るものの数。 | 個数 |
| $\sigma_s(v)=\sigma_{sv}$ | source $s$ から頂点 $v$ への最短経路数。 | 個数 |
| $P_s(w)$ | source $s$ を根とする最短経路 DAG における $w$ の predecessor 集合。 | 集合 |
| $Succ_s(w)$ | $d_s(v)=d_s(w)+1$ を満たす $w$ の隣接頂点集合。提案実装は $P_s$ を materialize せずこの関係を用いる。 | 集合 |
| $\delta_s(v)$ | source $s$ の頂点 $v$ に対する依存度（dependency）。 | 実数 |
| $C_B^{\mathrm{dir}}(v)$ | ordered pair に基づく非正規化 directed BC。 | 実数 |
| $C_B^{\mathrm{undir}}(v)$ | $1/2$ 補正を適用した非正規化 undirected BC。本研究が計算する量。 | 実数 |
| $\widehat{C}_B^{\mathrm{dir}}(v),\ \widehat{C}_B^{\mathrm{undir}}(v)$ | 正規化した directed / undirected BC。本研究では適用しない。 | 実数 |
| $CB$ | $C_B^{\mathrm{undir}}$ を格納する BC 出力配列。Algorithm 2.1 および Algorithm 4.1 で用いる。 | 長さ $n$ の配列 |
| $S$ | BFS 距離の非減少順に頂点を格納する traversal stack。$S_s$ は source $s$ の stack を表す。speedup には用いない。 | 配列 |
| $R$ | CSR の row pointer 配列。要素数 $n+1$。 | 配列 |
| $C$ | CSR の adjacency array。対称化して要素数 $2m$。 | 配列 |

## Hybrid BFS Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $Q$ | BFS の frontier。実装では現在 frontier と次 frontier を別配列で保持する。 | 配列 |
| $\lvert Q\rvert$ | 現在 frontier のサイズ。bottom-up から top-down への復帰判定に用いる。 | 個数 |
| $m_f$ | 現在 frontier から出る辺数の近似量。 | 個数 |
| $m_u$ | 未探索側から出る残り辺数の近似量。 | 個数 |
| $\alpha$ | top-down から bottom-up への切替閾値パラメータ。切替条件は $m_f > m_u/\alpha$。本実装では $\alpha=14$ を用いた。 | 無次元 |
| $\beta$ | bottom-up から top-down への復帰閾値パラメータ。復帰条件は $\lvert Q\rvert < n/\beta$。本実装では $\beta=24$ を用いた。 | 無次元 |

$\alpha=14$ と $\beta=24$ は Beamer らの評価で用いられ本実装でも採用した切替パラメータであり、あらゆるグラフ・ハードウェアに対する普遍的な最適値ではない。

## Batch and Memory Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $\mathrm{EffectiveBatch}$ | outer batch loop が実際に用いる 1 stream 当たりの source 数。HBM 予算超過時は要求バッチから縮小される。 | source 数 |
| $q$ | batch 内の source offset。`blockIdx.x=q` の block が source $s_{start}+q$ を担当する。 | 添字 |
| $s_{start}$ | outer batch の先頭 source 番号。 | 添字 |
| $\texttt{SUB\_BATCH}$ | batch を分割するとき 1 回の sub-launch が処理する source 数の上限。 | source 数 |
| $\texttt{num\_subs}$ | sub-launch 回数。$\lceil \mathrm{EffectiveBatch}/\texttt{SUB\_BATCH}\rceil$。 | 回数 |
| $NS_{\mathrm{eff}}$ | 同時に有効な stream buffer 数。in-capacity 実行では 2、oversubscription 経路では 1。 | 個数 |
| $D_{est}$ | 実装が推定する BFS depth の上限。修正版 325557 では $D_{est}=256$。 | level 数 |
| $M_{\mathrm{source}}$ | 1 source 当たりの状態量。$M_{\mathrm{source}}=32n+4D_{est}+8$ bytes。 | bytes |
| $M_{\mathrm{work}}$ | batch 依存の working-set 概念量（code-derived allocation estimate）。 | bytes |

working set の概念式は次のとおりである。

$$
M_{\mathrm{work}}
\approx
NS_{\mathrm{eff}}
\times
\mathrm{EffectiveBatch}
\times
M_{\mathrm{source}}.
$$

Chunked の同時 resident estimate では $\mathrm{EffectiveBatch}$ の代わりに $\texttt{SUB\_BATCH}$ を用いる。これらは配列寸法から導いた allocation estimate であり、measured process RSS、physical HBM residency、migration bytes ではない。batch および sub-batch は source 集合の grouping であり、graph partition ではない。

## Performance Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $T$ | 実行時間。runner が計測する実装関数全体の時間。 | s |
| $t_i$ | 第 $i$ 試行の実行時間。 | s |
| $\bar{t}$ | 試行の標本平均。 | s |
| $N_{\mathrm{trials}}$ | 試行数。頂点数 $n$ とは別の記号を用いる。表・図の `n=3`、`n=5` はこの量を示す慣用ラベルである。 | 回数 |
| $s_T$ | runtime の標本標準偏差（不偏推定量、ddof=1）。source 頂点 $s$ とは別の記号を用いる。 | s |
| $T^{\mathrm{med}}_{\mathrm{baseline}}$ | baseline の median 実行時間。RQ1 では tuned PathMerge。 | s |
| $T^{\mathrm{med}}_{\mathrm{proposed}}$ | 提案手法の median 実行時間。RQ1 では固定 b512 の GPU_Opt。 | s |
| $\mathrm{Speedup}$ | median 同士の比として定義する高速化率。traversal stack $S$ とは別の記号を用いる。 | 無次元 |
| $\mathrm{GTEPS}$ | スループット。$\mathrm{GTEPS}=n\cdot m/(T\cdot 10^{9})$。 | $10^{9}$ edges/s |

speedup は median 同士の比として定義する。

$$
\mathrm{Speedup} = \frac{T^{\mathrm{med}}_{\mathrm{baseline}}}{T^{\mathrm{med}}_{\mathrm{proposed}}}
$$

主値は median であり、median と mean を混在させて speedup を計算しない。標本標準偏差は不偏推定量（ddof=1）である。

$$
s_T = \sqrt{\frac{1}{N_{\mathrm{trials}}-1}\sum_{i=1}^{N_{\mathrm{trials}}}\left(t_i-\bar{t}\right)^2}
$$

## Ablation Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $\mathrm{H}$ | Hybrid BFS の有効・無効を表す binary configuration flag。0 は top-down のみ、1 は top-down / bottom-up 切替。 | $\{0,1\}$ |
| $\mathrm{W}$ | Warp-Cooperative Accumulation の有効・無効を表す binary configuration flag。0 は thread-per-vertex、1 は warp 協調累積。 | $\{0,1\}$ |
| $\mathrm{A}$ | Dual-Stream Execution の有効・無効を表す binary configuration flag。0 は単一 stream、1 は 2 stream。 | $\{0,1\}$ |
| $F$ | 主効果を評価する対象因子。$F\in\{\mathrm{H},\mathrm{W},\mathrm{A}\}$。 | 因子 |
| $G_1,\ G_2$ | $F$ 以外の残り 2 因子。 | 因子 |
| $T^{\mathrm{med}}_g(\cdot)$ | グラフ $g$ における当該構成の median 実行時間。 | s |
| $\mathrm{ME}_g(F)$ | グラフ $g$ における因子 $F$ の主効果（main effect）。$>1$ は有効化が median 実行時間を短縮したことを表す。 | 無次元 |
| $\mathcal{G}_{\mathrm{synth}}$ | 主効果を幾何平均で集約する合成グラフ集合。 | 集合 |
| $\mathrm{ME}_{\mathrm{synth}}(F)$ | $\mathcal{G}_{\mathrm{synth}}$ にわたる主効果の幾何平均。 | 無次元 |

H/W/A は本論文全体で同じ意味をもつ compile-time の binary configuration flag であり、8 構成 $\mathrm{H}\{0,1\}\times\mathrm{W}\{0,1\}\times\mathrm{A}\{0,1\}$ を構成する。主効果は factorial design 上の観測量であって、加算可能な寄与配分ではない。

## Correctness Notation

| Symbol | Meaning | Unit or Domain |
|:--|:--|:--|
| $r_i$ | reference 側の第 $i$ 要素の BC 値。 | 実数 |
| $c_i$ | candidate 側の第 $i$ 要素の BC 値。 | 実数 |
| $\mathrm{abs\_tol}$ | 混合許容基準の絶対許容値。$\mathrm{abs\_tol}=10^{-3}$。 | 実数 |
| $\mathrm{rel\_tol}$ | 混合許容基準の相対許容値。$\mathrm{rel\_tol}=10^{-6}$。 | 無次元 |

判定は次の混合絶対・相対許容による。

$$
\lvert r_i-c_i\rvert \le \mathrm{abs\_tol}+\mathrm{rel\_tol}\cdot\max\!\left(\lvert r_i\rvert,\lvert c_i\rvert\right)
$$

この基準を満たす PASS は混合許容内の numerical consistency を意味し、byte-identical であることを意味しない。

## Recorded Fields

次は保存 manifest・TSV の正式な記録列名、またはコード上の識別子である。数式記号ではないため上記の記号一覧には収録しない。本文で引用する場合は、対応する論文記号を併記する。

| Recorded Field | Thesis Symbol or Term | 対応の説明箇所 |
|:--|:--|:--|
| `PerSourceStateBytes` | $M_{\mathrm{source}}$ | 4.2 節、5.3 節、8.1.2 項 |
| `EffectiveNS` | $NS_{\mathrm{eff}}$ | 5.3 節、8.1.2 項 |
| `EffectiveBatch` | $\mathrm{EffectiveBatch}$ | 4.2 節 |
| `RequestedBatch` | Requested Batch（要求バッチ） | 4.2 節、5.5 節 |
| `SubBatch` / `SUB_BATCH` | $\texttt{SUB\_BATCH}$ | 4.2 節、8.4 節 |
| `NumSubs` / `num_subs` | $\texttt{num\_subs}$ | 4.2 節、8.4 節 |
| `INT_MAX` | 32 bit 符号付き整数の最大値 2,147,483,647 | 4.7.3 項、8.4 節 |
| `safe_sub_batch` | $\lfloor \texttt{INT\_MAX}/n\rfloor$ による index-safety 上限 | 8.4 節 |

これらの列名・識別子は保存記録およびコードの表記であり、本 Gate では変更しない。

## Units and Conventions

| Notation | Meaning |
|:--|:--|
| `s` | seconds（秒）。実行時間の単位。 |
| `GB` | decimal gigabytes（$10^{9}$ bytes）。 |
| `MB` | decimal megabytes（$10^{6}$ bytes）。 |
| `GiB` | binary gibibytes（$2^{30}$ bytes）。 |
| `MiB` | binary mebibytes（$2^{20}$ bytes）。 |
| `GB/s` | decimal gigabytes per second。interconnect の帯域表記に用いる。 |
| `GTEPS` | $10^{9}$ traversed edges per second。 |
| `b512` | requested batch size 512 を表す実験上の略記。`b<数値>` は要求バッチサイズを示す。 |
| `n=3`, `n=5` | 当該構成の試行数 $N_{\mathrm{trials}}$ を示す慣用ラベル。 |
| `n/a` | not applicable（当該欄が定義されない）。 |
| `not recorded` | 保存記録から確認できない。 |
| `N/A (failed)` | 実行失敗のため数値を定義しない。実行時間 0 秒として扱わない。 |

decimal 単位（GB、MB）と binary 単位（GiB、MiB）は換算せずに併記し、混在させない。GPU メモリ容量については、公称 HBM3 容量、実行環境の保存記録値、runtime query の報告値、および host-memory resource limit を同一の概念として扱わない。公称値と保存記録値は同一の on-package HBM3 を異なる単位系・取得方法で表したものであり、別個のメモリ領域ではない。
