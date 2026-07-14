# 12 関連研究とギャップ

**方針**：一次資料を優先する。書誌（著者・題目・会議/雑誌・年・DOI）は権威ある一次書誌
メタデータ（Crossref）で照合して確定した。二次まとめサイトのみを根拠にしない。GPU 実験・
性能再測定・論文数値の変更は行わない（Stage I1 での関連研究補完のみ）。照合できた参照は `§12.5 参照文献` に確定書誌としてまとめ、`§12.2` の各行から参照する。**一次資料で
裏付けられない主張は、未確認の引用で代替せず主張自体を除去する**（本 Stage で全カテゴリの書誌を確定し、
未解決の引用プレースホルダは残していない）。

> 補足（照合時の注意）：AI 検索が返す DOI・著者名にはハルシネーションが含まれたため、全 DOI を
> Crossref で再照合した。誤りを検出・訂正した例：GPU BC（SC14）は `10.1109/SC.2014.51` では
> なく **`10.1109/SC.2014.52`**（`.51` は Pardicle）、Subway は `…3387556` ではなく
> **`10.1145/3342195.3387537`**（`…3387556` は Mousse）、UM 評価は `10.1109/CCGrid.2015.39`
> ではなく **`10.1109/CCGrid.2015.105`**。

## 12.1 リポジトリ内で確認できる参照（確定アンカー）
| 参照 | リポジトリ内の根拠 | 本研究での役割 |
|:--|:--|:--|
| Direction-Optimizing BFS（Beamer ら, top-down/bottom-up, α=14/β=24） | `src/proposed/host_um.cu:140`, `include/proposed/brandes_kernels.cuh:40,54-56`（"Beamer 推奨値"） | Hybrid BFS（H）の直接の設計根拠 |
| Brandes 型 BC（前向き σ / 後向き δ 累積） | プロジェクト全体のアルゴリズム基盤（`brandes_kernels.cuh` の σ/δ 更新式） | 提案・全 baseline の共通アルゴリズム |
| Galliot / path-merging-bc（PathMerge） | `src/baseline/galliot.cu`, `galliot_kernels.cu`, `pathmerge.cu`（vendored baseline, int2 frontier, batch） | 主要比較 baseline（external comparator） |
| RAPIDS cuGraph `betweenness_centrality` | `src/baseline/cugraph_bc.cu`, `third_party/cugraph/`（vendored, RMM managed_memory_resource） | 補助 baseline（small 限定） |
| GH200 メモリ階層最適化（内部レポート） | `host_um.cu` コメントの `BC_Miyabi_report.pdf §3`, `Miyabi.pdf §5` | UM 配置・帯域活用の設計背景（内部資料） |

## 12.2 関連研究カテゴリと確定書誌
列：`Work / Platform / BCType / Parallelism / MemoryStrategy / DifferenceFromThisWork / Source`
（Source は `§12.5` の確定書誌 [R#] を指す。書誌は Crossref で照合済）

| Work | Platform | BCType | Parallelism | MemoryStrategy | DifferenceFromThisWork | Source |
|:--|:--|:--|:--|:--|:--|:--|
| Brandes アルゴリズム（厳密 BC） | CPU（原著） | exact, 全始点 | 逐次 | in-memory | 本研究は同アルゴリズムを不変のまま GPU バッチ型全始点並列として実装 | [R1] |
| Direction-Optimizing BFS（Beamer ら） | 多様 | BFS（BC ではない） | 並列 | in-memory | 本研究は BC の BFS フェーズに適用（H）。**top-down/bottom-up 切替であり CPU–GPU hybrid ではない** | [R2] |
| GPU BC（一般・高性能化） | GPU | exact | work-efficient/edge-parallel 切替 | device メモリ | GPU BC・並列戦略切替は既存（新発明ではない）。本研究の焦点は GH200 UM オーバーサブスクリプション統合と容量・数値整合性の体系評価 | [R3][R4] |
| Multi-source / batched BC | GPU/heterogeneous | exact | 複数ソース同時（vertex virtualization） | device/UM | バッチ型多ソース BC 自体は既存。本研究は固定 b512 block + 2-stream init + UM/Chunked を GH200 で統合 | [R5] |
| Galliot / PathMerge（主要比較 baseline） | GPU | exact | path-merging batch（int2 frontier） | managed CSR, O(n) 補助メモリ | 本研究の external comparator（vendored）。提案 block が tuned Galliot より median 1.31–3.17×。Galliot は direction-optimizing hybrid BFS・2-stream・UM オーバーサブスクリプション統合を扱わない | [R6][R7] |
| RAPIDS cuGraph BC | GPU | exact（`vertices` 未指定=全頂点 traversal）/approx | 汎用グラフプリミティブ | RMM（managed 可） | 本研究は small 限定の補助 baseline として利用（`vertices=nullopt, normalized=false, include_endpoints=false`）。**内部実装への formal な per-level 計算量比較は主張しない** | [R8] |
| GPU Unified Memory / オーバーサブスクリプション | GPU（UM） | 一般（BC 特化でない） | — | UM prefetch/page-fault/oversubscription | 既存は一般/選定 CUDA アプリのベンチ。本研究は**全始点 BC の working set** に UM を適用し GH200 NVLink-C2C 上の HBM3 超過 feasibility を BC 文脈で評価（migration byte 直接計測なし） | [R9][R10] |
| Chunking / out-of-core graph processing | GPU | 一般 | — | 分割・データ転送最小化 | 既存はグラフ本体の out-of-core 処理。本研究は BC の**バッチ（同時ソース数）を SUB_BATCH 分割**（グラフ CSR 分割ではない） | [R11][R12] |

## 12.3 ギャップの主張（誇張しない）
- 本研究の新規性は**要素技術の発明ではなく統合と体系的評価**にある（[00](00_thesis_positioning.md)）。
  照合した一次資料が示すとおり、GPU 上の厳密 BC（[R3][R4]）、バッチ型多ソース BC（[R5]）、
  direction-optimizing BFS（[R2]）、path-merging バッチ BC（[R6][R7]）、UM オーバーサブスクリプ
  ション（[R9][R10]）、out-of-core グラフ処理（[R11][R12]）はいずれも**既存**である。したがって
  関連研究節では「各要素は既存であり、本研究はそれらを GH200 向けバッチ型全始点 BC 実行基盤として
  統合し、性能・要因・容量・数値整合性を一貫条件で測定した」と位置づける。
- **既存研究が既に同一の組合せ**（バッチ型全始点 BC × direction-optimizing BFS × 2-stream ×
  GH200 UM オーバーサブスクリプション）**を提案している場合、新規性を誇張せず位置づけを修正する**。
  本 Stage で照合した一次資料の範囲では、この 4 要素すべてを単一研究で統合し GH200 で評価した先行
  研究は確認できなかった（ただしこれは網羅的探索の証明ではなく、調査範囲内での所見）。特に主要
  比較 baseline の Galliot（[R6][R7]）は path-merging バッチ BC だが、direction-optimizing hybrid
  BFS・2-stream 初期化・UM オーバーサブスクリプション統合は含まない（vendored コードで確認）。
- cuGraph は GPU 向け BC 実装を提供する（vendored 公式ヘッダ
  `third_party/cugraph/cpp/include/cugraph/algorithms.hpp` で確認：`vertices` 未指定時は全頂点
  traversal による exact betweenness）。**本研究はその内部実装に対する formal な per-level 計算量
  比較を主張しない**。実験で条件を確認できた小規模グラフの補助 baseline として位置づける
  （[06](06_results_performance.md) §6.4）。cuGraph を全グラフの baseline とはせず、未測定の優位も
  主張しない。

## 12.4 執筆時の TODO（Stage I1 で照合済 / 残課題）
- [x] Brandes 原著の書誌を確定（[R1], DOI 10.1080/0022250X.2001.9990249）。O(VE) の定式化は原著参照。
- [x] Beamer らの direction-optimizing BFS の書誌を確定（[R2], DOI 10.1109/SC.2012.50）。α/β の値は
      コード（`brandes_kernels.cuh`）が実装、原著が出典。
- [x] Galliot/path-merging-bc の原著・リポジトリを確定（[R6] INFOCOM 2023, [R7] JSAC 2023,
      upstream `gobardhanm/path-merging-bc`）。差分は §12.2/§12.5 に整理。
- [x] GPU BC・batched BC・UM オーバーサブスクリプション・out-of-core の代表研究を各 1–2 件確定
      （[R3][R4][R5][R9][R10][R11][R12]）。
- [x] cuGraph BC の利用条件を vendored 公式ヘッダ（`third_party/cugraph/cpp/include/cugraph/algorithms.hpp`）
      と本研究の呼び出し（`src/baseline/cugraph_bc.cu`: `vertices=nullopt`→exact, `normalized=false`,
      `include_endpoints=false`）で確定。**内部の per-level 計算量は formal な出典を確認できないため、
      計算量比較の主張自体を削除**（未確認引用で代替しない）。
- [x] 同一組合せの先行研究の有無を確認（調査範囲では未確認 → §12.3 と [00](00_thesis_positioning.md)
      の「統合が新規」の位置づけを維持。網羅探索ではない点を明記）。

## 12.5 参照文献（Crossref 照合済）
各項目：`[R#] Authors, "Title," Venue, Year. DOI / URL — SupportedStatement / DifferenceFromThisWork`

- **[R1]** Ulrik Brandes, "A faster algorithm for betweenness centrality," *Journal of Mathematical
  Sociology*, 25(2):163–177, 2001. DOI: 10.1080/0022250X.2001.9990249 —
  *Supported*: 提案・全 baseline が採用する厳密 BC の基盤（前向き σ / 後向き δ 累積、非重み付き
  O(VE)）。 *Difference*: 本研究は同アルゴリズムを不変のまま GPU バッチ型全始点並列として実装。
- **[R2]** Scott Beamer, Krste Asanović, David Patterson, "Direction-optimizing Breadth-First
  Search," *SC'12: Int. Conf. for High Performance Computing, Networking, Storage and Analysis*,
  2012. DOI: 10.1109/SC.2012.50 — *Supported*: Hybrid BFS（H）の top-down/bottom-up 切替（α=14,
  β=24）の設計根拠（コードに「Beamer 推奨値」と明記）。 *Difference*: 本研究は BC の BFS フェーズに
  適用（原著は BFS 単体）。**CPU–GPU hybrid ではなく方向切替**。
- **[R3]** Adam McLaughlin, David A. Bader, "Scalable and High Performance Betweenness Centrality
  on the GPU," *SC'14: Int. Conf. for High Performance Computing, Networking, Storage and
  Analysis*, 2014. DOI: 10.1109/SC.2014.52 — *Supported*: GPU 上の厳密 BC が高性能化され既存で
  あること（work-efficient と edge-parallel の切替、scale-free/high-diameter 両対応）。
  *Difference*: GPU BC 化・並列戦略切替は新発明ではない。本研究は UM オーバーサブスクリプション統合と
  容量・数値整合性の体系評価が焦点で、McLaughlin/Bader は UM オーバーサブスクリプション/バッチ分割を
  扱わない。
- **[R4]** Adam McLaughlin, David A. Bader, "Accelerating GPU betweenness centrality,"
  *Communications of the ACM*, 61(8):85–92, 2018. DOI: 10.1145/3230485 —
  *Supported*: [R3] の一般向け改訂（GPU BC の確立を補強）。 *Difference*: 同上。
- **[R5]** Ahmet Erdem Sariyüce, Kamer Kaya, Erik Saule, Ümit V. Çatalyürek, "Betweenness centrality
  on GPUs and heterogeneous architectures," *GPGPU-6 (Workshop on General Purpose Processing Using
  GPUs)*, 2013. DOI: 10.1145/2458523.2458531 — *Supported*: 複数ソースをまとめて処理する batched
  BC・ヘテロ環境 BC が既存であること（vertex virtualization）。 *Difference*: バッチ型多ソース BC
  自体は既存。本研究は固定 b512 block + 2-stream init + UM/Chunked を GH200 で統合し容量境界を評価。
- **[R6]** Zhigao Zheng, Chen Zhao, Peichen Xie, Bo Du, "Galliot: Path Merging Based Betweenness
  Centrality Algorithm on GPU," *IEEE INFOCOM 2023 - IEEE Conference on Computer Communications*,
  pp. 1–10, 2023. DOI: 10.1109/INFOCOM53939.2023.10229018. Upstream code:
  https://github.com/gobardhanm/path-merging-bc — *Supported*: path-merging により補助メモリを O(n)
  に抑えた GPU 厳密 BC。本研究の **external comparator**（`src/baseline/` に vendored）。
  *Difference*: 提案 block GPU_Opt が tuned Galliot より median 1.31–3.17×（4グラフ）。Galliot は
  direction-optimizing hybrid BFS・2-stream・UM オーバーサブスクリプション統合を扱わない。
  **PathMerge は ground truth ではない**。
- **[R7]** Zhigao Zheng, Bo Du, Chen Zhao, Peichen Xie, "Path Merging Based Betweenness Centrality
  Algorithm in Delay Tolerant Networks," *IEEE Journal on Selected Areas in Communications*, 2023.
  DOI: 10.1109/JSAC.2023.3310071 — *Supported*: [R6] の path-merging BC の雑誌版。 *Difference*:
  同上（本研究の baseline アルゴリズムの出典）。
- **[R8]** RAPIDS Development Team, "RAPIDS cuGraph — GPU Graph Analytics Library"（オープンソース
  ソフトウェア; 本研究は BC 関連サブセットを `third_party/cugraph/` に vendored, rapids-cmake を
  v26.04.00 に固定）. URL: https://github.com/rapidsai/cugraph — *Supported*: cuGraph は
  `betweenness_centrality` を提供し、`vertices` 未指定で全頂点 traversal による exact BC を計算する
  （vendored 公式ヘッダ `cpp/include/cugraph/algorithms.hpp` で確認）。本研究では
  `vertices=nullopt, normalized=false, include_endpoints=false`、RMM managed_memory_resource で
  **小規模グラフの補助 baseline** として利用（`src/baseline/cugraph_bc.cu`）。 *Difference*: 本研究は
  cuGraph の内部実装に対する **formal な per-level 計算量比較を主張しない**（査読付き原著 DOI も
  特定できないため、ソフトウェア/公式ヘッダとして引用）。cuGraph を全グラフ baseline とはせず、
  未測定の優位も主張しない。
- **[R9]** Wenqiang Li, Guanghao Jin, Xuewen Cui, Simon See, "An Evaluation of Unified Memory
  Technology on NVIDIA GPUs," *CCGrid'15: 15th IEEE/ACM Int. Symp. on Cluster, Cloud and Grid
  Computing*, 2015. DOI: 10.1109/CCGrid.2015.105 — *Supported*: UM の利便性と、アクセスパターン/
  オーバーサブスクリプション依存の性能特性の既存評価。 *Difference*: 本研究は**全始点 BC の working
  set** に UM を適用し GH200 で feasibility を評価。
- **[R10]** Marcin Knap, Paweł Czarnul, "Performance evaluation of Unified Memory with prefetching
  and oversubscription for selected parallel CUDA applications on NVIDIA Pascal and Volta GPUs,"
  *The Journal of Supercomputing*, 75:7625–7645, 2019. DOI: 10.1007/s11227-019-02966-8 —
  *Supported*: prefetching・オーバーサブスクリプション下の UM 性能の既存評価（本研究の UM/prefetch
  設計背景）。 *Difference*: 既存は選定 CUDA アプリのベンチ。本研究は BC 文脈で NVLink-C2C 上の
  HBM3 超過 feasibility を評価（migration byte 直接計測なし）。
- **[R11]** Amir Hossein Nodehi Sabet, Zhijia Zhao, Rajiv Gupta, "Subway: minimizing data transfer
  during out-of-GPU-memory graph processing," *EuroSys'20: Fifteenth European Conference on
  Computer Systems*, pp. 1–16, 2020. DOI: 10.1145/3342195.3387537 — *Supported*: GPU メモリを
  超えるグラフの out-of-core 処理・データ転送最小化。 *Difference*: 既存はグラフ本体の out-of-core
  処理。本研究は BC の**バッチ（同時ソース数）を SUB_BATCH 分割**（グラフ CSR 分割ではない）。
- **[R12]** Koichi Shirahata, Hitoshi Sato, Satoshi Matsuoka, "Out-of-core GPU memory management
  for MapReduce-based large-scale graph processing," *CLUSTER'14: IEEE Int. Conf. on Cluster
  Computing*, 2014. DOI: 10.1109/CLUSTER.2014.6968748 — *Supported*: GPU メモリを超えるグラフ処理の
  out-of-core メモリ管理の既存例。 *Difference*: 同上（本研究は BC バッチ分割で working set を制御）。
