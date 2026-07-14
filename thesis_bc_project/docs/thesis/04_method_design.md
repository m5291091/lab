# 04 提案手法の設計（コード準拠）

本章の記述はすべて `src/proposed/*.cu`・`include/proposed/*.cuh`・`src/core/*` の実コードに
基づく。実装と一致しない機能は書かない。共通計算基盤の上に 3 つのメモリ管理方式（UM/Pure/
Chunked）が載る構成である（[00](00_thesis_positioning.md) の「共通 GPU 実行基盤」）。

## 4.1 入力グラフ表現
- 無向・非重み。CSR テキスト 3 行形式：`n m` / `ptr[0..n]` / `adj[0..2m-1]`
  （`src/core/graph.cpp`, `Graph::readGraph()`）。
- `R = getAdjacencyListPointers()`（長さ n+1）、`C = getAdjacencyList()`（長さ 2m）。
  有向 nnz は `R[n]`（= `offsets[n_nodes]`）から取得する（`getEdgeCount()` は無向辺数 m）。
- 提案側は `edge_size = 2 * getEdgeCount()` を隣接配列長として使う（`host_um.cu:652`）。

## 4.2 バッチ処理と block 割当
- 全 n 頂点をソースとし、`BATCH_PER_STREAM` 個ずつのバッチで処理する（`host_um.cu:418` の
  `for (s_start ...; s_start += BATCH_PER_STREAM)`）。
- **block 単位の始点処理**：BFS/backward カーネルはグリッド次元 = バッチ内ソース数、
  `blockIdx.x` が各ソースに対応（`brandes_bfs_kernel_opt`：`int s = s_start + batch_idx;`
  `brandes_kernels.cuh:266`）。すなわち **1 CUDA block = 1 ソース**。
- 動的状態は「ソース × n_nodes」でオフセットされる（`batch_node_offset(batch_idx, n_nodes)`）。
  1 バッチが確保する配列：`d_d`(int), `d_sigma`(double), `d_delta`(double), `d_Q_curr`(int),
  `d_Q_next`(int), `d_S`(int), `d_S_ends`(int, レベル境界), `d_depth`(int)。

### block カーネルと shared-frontier カーネル
- 既定は **block（1 block = 1 source, `brandes_bfs_kernel_opt`）**。旧実装には平均次数に基づく
  自動選択則があったが、現在は常時 block（`host_um.cu:250-261`, `heuristic_shared=false`）。
- `BC_FORCE_BFS_KERNEL=shared|block` で強制切替可能（再現実験用に温存, `host_um.cu:238-249`）。
  shared 版 `brandes_bfs_kernel_shared_frontier` は 1 block に K=16 ソース、ソースあたり
  `THREADS_PER_SOURCE=16`（`common.hpp`）。kernel 選択の寄与は
  [07](07_results_ablation.md) の kernel_selection（forced 比較）で扱う。
- スレッド数/block（tpb）は avg_deg に依存：<5→128, <20→256, ≥20→512（`choose_tpb_for_graph`,
  `host_um.cu:80-93`）。

## 4.3 前向き BFS（Hybrid: top-down/bottom-up 方向切替）
`find_shortest_paths_opt`（`brandes_kernels.cuh:24-150`）。**Hybrid BFS は BFS の探索方向
（top-down / bottom-up）をレベルごとに切り替える direction-optimizing BFS であり、CPU–GPU
hybrid ではない**。
- `direction`（0=TOP_DOWN, 1=BOTTOM_UP）を各レベルで判定。Beamer らの推奨値 `alpha=14, beta=24`。
  top-down→bottom-up は `m_f > m_u/alpha`、bottom-up→top-down は `Q_curr_len < n/beta`
  （`brandes_kernels.cuh:54-56`）。`USE_HYBRID_BFS=false` なら常に top-down。
- top-down：フロンティア各頂点の隣接を走査し、`atomicCAS(d_d,-1,depth+1)` で未訪問を確定、
  `atomicAdd(d_sigma[w], d_sigma[v])` で σ 累積。
- bottom-up：未訪問頂点 w について隣接 v の距離が depth なら σ を集約。
- 各レベルで S（探索順スタック）と S_ends（レベル境界）を記録。`max_depth_estimate` 超過時は
  `d_overflow` を立て、host 側でエラー終了（`host_um.cu:590-594`）。`max_depth_estimate` は
  avg_deg 依存：<5→4096, <20→256, ≥20→64（`host_um.cu:231-233`）。

## 4.4 後向き依存度累積（thread/warp 協調のグラフ依存切替）
- **warp 協調版** `accumulate_dependencies_opt`（`brandes_kernels.cuh:156-198`）：レベルを深い
  順にたどり、頂点 w を warp（32 lane）に割当。隣接を lane 分割で走査し
  `δ += (σ_w/σ_v)·(1+δ_v)` を **warp shuffle 還元**（`warp.shfl_down`）で集約。
- **thread-per-vertex 版** `accumulate_dependencies_tpv_opt`（低密度向け）。
- 切替則：`avg_deg < 8.0` なら tpv、それ以外は warp（`host_um.cu:534-548`）。
- warp 協調（W）の寄与はアブレーションで測定（[07](07_results_ablation.md)）。効果はグラフ依存
  （高次数で有利、低次数では中立〜わずかに不利）。

## 4.5 2 ストリーム構造（ダブルバッファリング）
- `NS=2` の CUDA ストリームでバッチを交互処理（`host_um.cu:264, 418-419`）。
- 初期化はホスト側 `cudaMemsetAsync`（`memset_subbatch`）を Copy Engine（DMA）に投入し、
  片ストリームのカーネル実行中にもう一方の memset を重畳（`host_um.cu:121-135` のコメント、
  `brandes_bfs_kernel_opt` はソース 1 点のみ O(1) セットアップ）。
- バッファ再利用前に前回イベント時間を回収し、当該ストリームのみ同期（他ストリームとの重畳は維持,
  `host_um.cu:426-433`）。
- **buffer 再利用最適化**：同一 sub-batch 範囲を再利用する場合は full memset ではなく到達済み頂点
  のみリセット（`reset_visited_batch_kernel`, `host_um.cu:459-476`）。full memset か visited-reset
  かの分岐計数を診断出力する（`full_memset_calls` / `visited_reset_calls`）。

## 4.6 BC 依存度の集計と無向補正
- 各ソースの backward 完了後、`v != s` について `CB[v] += δ[v]`（無向は `δ[v]/2.0`）を
  `atomicAdd`（`brandes_kernels.cuh:318-325`, `IS_UNDIRECTED=true`）。**無向グラフでは最終 BC を
  1/2 にする**のが本プロジェクトの規約（提案・PathMerge とも適用）。
- 計算量：1 ソース O(V+E)、全始点 **O(V·E)**（重みなし）。

## 4.7 メモリ管理方式（UM / Pure / Chunked）
共通のバッチ計算に対し、動的状態の確保方法だけが異なる。3 方式とも `NS=2`・同一
`per_batch_mem` 公式・同一カーネルを用いる。

| 項目 | GPU_Opt（`host_um.cu`） | GPU_Opt_Pure（`host_pure.cu`） | GPU_Opt_Pure_Chunked（`host_chunked.cu`） |
|:--|:--|:--|:--|
| 確保 API | `cudaMallocManaged`（UM） | `cudaMalloc`（device only） | `cudaMalloc`（SUB_BATCH 単位） |
| 実バッファ量 | `BATCH_PER_STREAM × n_nodes`（UM, HBM3 超過分は LPDDR5X へ spill） | `BATCH_PER_STREAM × n_nodes`（HBM3 に直接） | `SUB_BATCH × n_nodes`（常に HBM3 内） |
| oversubscription | 対応（`oversubscribed=dynamic_bytes>free_mem*0.90`）。SUB_BATCH 分割 + prefetch/evict, NS_eff=1 | 非対応（大 batch で cudaMalloc OOM） | SUB_BATCH 分割で常に HBM3 内、NS_eff=1（oversub 時） |
| CSR(R,C) 配置 | UM + SetReadMostly、topo<35%×totalGlobalMem なら HBM3、超なら LPDDR5X + AccessedBy | device | device |
| 論文での位置づけ | **主実装**（主要性能・RQ1） | device-only 対照 | 容量拡張（sub-batch 分割） |

- UM の oversubscription 経路（`host_um.cu:439-555`）：SUB_BATCH ごとに prefetch → memset/reset →
  BFS → 並行 prefetch（次 sub-batch）→ backward → evict。`prefetch_subbatch`/`evict_subbatch_to_host`
  が NVLink-C2C 経由の移動を担う。**prefetch/evict 時間は計測するが、移動 byte 総量は直接計測しない。**
- Pure は BATCH_PER_STREAM×n の device 確保を直接行うため、これが HBM3 を超えると OOM（RQ3）。
- Chunked は BATCH_PER_STREAM をいくら大きくしても実確保は SUB_BATCH 単位なので HBM3 上限を
  超えない（`host_chunked.cu:119-143`）。

## 4.8 バッチサイズの意味（requested / effective / SUB_BATCH / num_subs / NS_eff）
`host_um.cu:266-336` に対応。用語を厳密に区別する。

| 用語 | 意味 |
|:--|:--|
| requested batch | ユーザ指定値（`BC_BATCH_OVERRIDE`）または自動計算した `BATCH_PER_STREAM`。既定は `min(available/(NS·per_batch_mem), 512)`（**上限 512**）。 |
| effective batch | 実際に処理単位となった値。in-capacity では requested と一致。 |
| SUB_BATCH | oversubscribed 時に HBM3 予算（`free_mem*0.80` − topology − CB）から算出する 1 launch あたりソース数（in-capacity では = batch）。 |
| num_subs | 1 バッチを分割する sub-launch 数 `⌈BATCH_PER_STREAM/SUB_BATCH⌉`（in-capacity では 1）。 |
| NS_eff | 有効ストリーム数。oversubscribed（または診断 `BC_DIAG_FORCE_NS_EFF_ONE`）で 1、それ以外 2。 |

- 主要性能実験（proposed_variants）は in-capacity：requested/effective/SUB_BATCH=512/512/512、
  num_subs=1、NS_eff=2（`coverage_matrix.tsv`）。
- 例（325557, UM `b9792`, `memory_paths` job 2368587）：oversubscribed, SUB_BATCH=6596,
  num_subs=2, NS_eff=1（`execution_summary.tsv`）。Chunked `b16384` は num_subs=3。
- `per_batch_mem = n·(3·int + 2·double) + n·int(d_S) + (max_depth+1)·int(d_S_ends) + int(d_depth)`
  （`host_um.cu:271-275`）。

## 4.9 タイミングと GTEPS
- `run_brandes`（`src/core/runner.cpp:88-96`）が `omp_get_wtime()` で **実装関数全体**（host 制御 +
  H2D + カーネル + D2H）を `Time_sec` として計測。phase 内訳（BFS/backward）は stderr。
- `GTEPS = n_nodes × n_edges / Time_sec / 1e9`（`n_edges` は無向辺数 m, `runner.cpp:94`）。
  全実装で同一定義。

## 4.10 baseline 実装（比較対象）
- **PathMerge（Galliot）**：`src/baseline/pathmerge.cu` + `galliot.cu`/`galliot_kernels.cu`。
  CSR を `cudaMallocManaged` に構築、int2 フロンティア、batch（既定 64, `PATHMERGE_BC_BATCH_SIZE`
  で上書き、人為的 64 上限は撤廃）。HBM3 予算にクランプ（`clamp_batch_to_memory`）。最終 BC を
  **/2.0**（無向補正, `pathmerge.cu:175`）。**external comparator であり ground truth ではない。**
- **cuGraph**：`src/baseline/cugraph_bc.cu`。`cugraph::betweenness_centrality` を
  vertices=`nullopt`（全始点＝**exact**）, `normalized=false`, `include_endpoints=false`,
  `is_symmetric=true`（無向）で呼ぶ。RMM `managed_memory_resource`。**この adapter は明示的な /2 を
  適用しない**（提案・PathMerge は /2 する）ため、cuGraph の BC スケール整合は本環境で未確立
  → [05](05_experimental_setup.md) の制約に記載。small 限定の補助 baseline。
- **Sequential / OpenMP**：CPU 参照（`src/baseline/sequential.cpp`, `omp.cpp`）。Sequential は
  small 正確性の独立参照（[09](09_results_correctness.md)）。

## 4.11 アブレーション実装（H/W/A）
`include/proposed/ablation_config.hpp` + `src/proposed/host_ablation.cu`。3 工夫（`hybrid_bfs`,
`warp_coop`, `async_init`）を **C++ テンプレートのコンパイル時分岐**で 8 実体に振り分ける
（カーネル内 if を避け branch divergence を排除）。ラベル例 `H1_W0_A1`。
