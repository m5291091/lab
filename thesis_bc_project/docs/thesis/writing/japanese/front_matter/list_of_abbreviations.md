# List of Abbreviations

本一覧は Chapter 1 から Chapter 11 および Abstract の本文で反復して使用され、読者が参照する必要のある略語のみを収録する。本文で使用していない略語、および初出箇所の説明だけで完結する 1 回限りの略語は掲載しない。`Full Term` は正式な英語名称、`Description` は簡潔な日本語説明である。

| Abbreviation | Full Term | Description |
|:--|:--|:--|
| API | Application Programming Interface | ライブラリや実行環境が提供する呼び出し仕様。cuGraph および CUDA runtime の関数仕様を指す文脈で使用する。 |
| BC | Betweenness Centrality | 各頂点が他の頂点対の最短経路上に現れる度合いを表す中心性指標。本研究の計算対象。 |
| BFS | Breadth-First Search | 幅優先探索。Brandes アルゴリズムの Forward phase で最短距離と最短経路数を求めるために用いる。 |
| CPU | Central Processing Unit | 中央処理装置。本研究では GH200 の Grace CPU、および host 側処理を指す。 |
| CSR | Compressed Sparse Row | 疎グラフの隣接構造を row pointer と adjacency array で保持する形式。本研究の入力および内部表現。 |
| CUDA | Compute Unified Device Architecture | NVIDIA GPU 向けの並列計算プラットフォームおよびプログラミングモデル。 |
| DAG | Directed Acyclic Graph | 有向非巡回グラフ。ある source を根とする最短経路構造（shortest-path DAG）を指す。 |
| GPU | Graphics Processing Unit | 画像処理装置。本研究では GH200 の Hopper GPU を指し、BC 計算の実行主体である。 |
| GTEPS | Giga Traversed Edges Per Second | 1 秒あたり $10^{9}$ 辺の走査量として定義するスループット指標。 |
| HBM3 | High Bandwidth Memory 3 | Hopper GPU の on-package 高帯域メモリ。対象構成の公称容量は 96 GB である。 |
| LPDDR5X | Low-Power Double Data Rate 5X | Grace CPU 側の低消費電力メモリ規格。 |
| NVLink-C2C | NVLink Chip-to-Chip | Grace CPU と Hopper GPU を結合する coherent interconnect。 |
| OOM | Out of Memory | メモリ不足による割り当て失敗またはプロセス強制終了。本研究では CUDA device-memory OOM と cgroup host-memory OOM kill を区別する。 |
| PBS | Portable Batch System | Miyabi-G のジョブスケジューラ。実験ジョブの投入に使用した。 |
| RQ | Research Question | 本研究の評価軸を規定する研究設問。RQ1 から RQ4 を設定する。 |
| RSS | Resident Set Size | プロセスが物理メモリ上に常駐させている量。本研究では未取得の指標として言及する。 |
| SD | Standard Deviation | 標準偏差。本研究では runtime の標本標準偏差 $s_T$（不偏推定量、ddof=1）を補助値として報告する。 |
| SHA256 | Secure Hash Algorithm 256-bit | 入力グラフおよび BC ベクトルの同一性を管理するハッシュ値。 |
| SIMT | Single Instruction, Multiple Threads | warp 内の複数 thread が同一命令列を進める GPU の実行方式。 |
| TSV | Tab-Separated Values | 生データおよび派生表の保存形式。 |
| UM | Unified Memory | CPU と GPU から同一アドレスでアクセスできる managed allocation の仕組み。追加の物理容量ではない。 |

## Implementation Names

次の名称は略語ではなく、本研究の実装名・方式名である。したがって上記の略語一覧には含めない。

| Name | Description |
|:--|:--|
| GPU_Opt | 提案するバッチ型 GPU 実行基盤の主実装。Unified Memory を用いる。 |
| GPU_Opt_Pure | 同一基盤の memory-management variant。device memory のみを明示的に使用する。 |
| GPU_Opt_Pure_Chunked | 同一基盤の memory-management variant。source sub-batch により同時 resident な working set を制限する。 |
| PathMerge | 比較対象として評価した第三者実装。原著論文著者の公式実装ではなく、external comparator であって ground truth ではない。 |
| Pure | GPU_Opt_Pure の本文中の短縮表記。 |
| Chunked | GPU_Opt_Pure_Chunked の本文中の短縮表記。 |

GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunked は 3 つの独立した提案ではなく、共通の GPU 実行基盤におけるメモリ管理方式のバリエーションである。

## Abbreviations Defined Only at First Use

次の略語は本文で 1 回のみ使用され、その箇所で正式名称を併記して説明が完結するため、本一覧には収録しない。

| Abbreviation | Full Term | 初出箇所 |
|:--|:--|:--|
| SNAP | Stanford Network Analysis Project | 5.3 節 |
| RAPIDS Memory Manager | — | 5.4 節 Table 5.4（略語 `RMM` は使用しない） |

`H2D` / `D2H` は 7.6 節で host-to-device / device-to-host と平文表記へ改めたため、略語としては使用しない。
