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

> **Stage L0（2026-07-15, 独立再検証）**：[R1]–[R12] の全 DOI・著者・誌名・巻号頁を Crossref API
> で再照合し（相違 0 件）、可能な限り第二系統（出版社ページ・著者公開版 PDF・著者公式ページ・
> 公式リポジトリ）でも照合した。[R1][R2][R5] は本文レベルで主張を確認（Brandes の Theorem 6 /
> Corollary 4、Beamer の「we suggest α = 14 and β = 24」、Sarıyüce らの vertex virtualization と
> 粗粒度＝ソース単位並列）。この過程で 2 点を訂正した：(1) **[R9] は CUDA 6.x 世代の UM 評価で
> あり oversubscription（Pascal 以降の機能）は対象外**のため、oversubscription の既存評価は [R10]
> のみに帰属させる。(2) vendored PathMerge の上流リポジトリは**論文著者の公式実装ではなく第三者
> 実装**（[R6] 参照）。全対応は BibTeX（`references.bib`）と出典監査表（`SOURCE_AUDIT.tsv`）に
> 記録した。技術仕様・データセットの公式一次資料は §12.6（[R13]–[R20]）に追加した。

## 12.1 リポジトリ内で確認できる参照（確定アンカー）
| 参照 | リポジトリ内の根拠 | 本研究での役割 |
|:--|:--|:--|
| Direction-Optimizing BFS（Beamer ら, top-down/bottom-up, α=14/β=24） | `src/proposed/host_um.cu:140`, `include/proposed/brandes_kernels.cuh:40,54-56`（"Beamer 推奨値"） | Hybrid BFS（H）の直接の設計根拠 |
| Brandes 型 BC（前向き σ / 後向き δ 累積） | プロジェクト全体のアルゴリズム基盤（`brandes_kernels.cuh` の σ/δ 更新式） | 提案・全 baseline の共通アルゴリズム |
| Galliot / path-merging-bc（PathMerge） | `src/baseline/galliot.cu`, `galliot_kernels.cu`, `pathmerge.cu`（vendored baseline, int2 frontier, batch） | 主要比較 baseline（external comparator; 上流は第三者実装 → §12.5 [R6]） |
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
| Multi-source / batched BC | GPU/heterogeneous | exact | 粗粒度（ソース単位）／細粒度並列, vertex virtualization | device/UM | ソースを跨いだ並列処理・ヘテロ分担は既存。本研究は固定 b512 block + 2-stream init + UM/Chunked を GH200 で統合 | [R5] |
| Galliot / PathMerge（主要比較 baseline） | GPU | exact | path-merging batch（int2 frontier） | managed CSR, per-source O(n) 補助配列（vendored 実装で確認） | 本研究の external comparator（vendored）。提案 block が tuned Galliot より median 1.31–3.17×。Galliot は direction-optimizing hybrid BFS・2-stream・UM オーバーサブスクリプション統合を扱わない | [R6][R7] |
| RAPIDS cuGraph BC | GPU | exact（`vertices` 未指定=全頂点 traversal）/approx | 汎用グラフプリミティブ | RMM（managed 可） | 本研究は small 限定の補助 baseline として利用（`vertices=nullopt, normalized=false, include_endpoints=false`）。**内部実装への formal な per-level 計算量比較は主張しない** | [R8] |
| GPU Unified Memory / オーバーサブスクリプション | GPU（UM） | 一般（BC 特化でない） | — | UM prefetch/page-fault/oversubscription | 既存は一般/選定 CUDA アプリのベンチ（[R9]=CUDA 6.x 世代の UM 評価, [R10]=prefetch/oversubscription 評価）。本研究は**全始点 BC の working set** に UM を適用し GH200 NVLink-C2C 上の HBM3 超過 feasibility を BC 文脈で評価（migration byte 直接計測なし） | [R9][R10] |
| Chunking / out-of-core graph processing | GPU | 一般 | — | 分割・データ転送最小化 | 既存はグラフ本体の out-of-core 処理。本研究は BC の**バッチ（同時ソース数）を SUB_BATCH 分割**（グラフ CSR 分割ではない） | [R11][R12] |

## 12.3 ギャップの主張（誇張しない）
- 本研究の新規性は**要素技術の発明ではなく統合と体系的評価**にある（[00](00_thesis_positioning.md)）。
  照合した一次資料が示すとおり、GPU 上の厳密 BC（[R3][R4]）、バッチ型多ソース BC（[R5]）、
  direction-optimizing BFS（[R2]）、path-merging バッチ BC（[R6][R7]）、UM の評価（[R9]）と
  UM オーバーサブスクリプション（[R10], 公式仕様は [R14][R15]）、out-of-core グラフ処理
  （[R11][R12]）はいずれも**既存**である。したがって
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

### Stage L0（2026-07-15）で追加照合・訂正した項目
- [x] [R1]–[R12] の全 DOI・書誌を Crossref API で独立再照合（相違 0 件）。第二系統（出版社ページ・
      著者公開版・著者公式ページ・公式リポジトリ）も可能な範囲で照合し、`SOURCE_AUDIT.tsv` に記録。
- [x] [R1] Brandes の依存度累積（Theorem 6）と非重み O(nm)（Corollary 4）を原文で確認。
- [x] [R2] Beamer らの「we suggest α = 14 and β = 24」を著者公開版の本文で確認
      （`brandes_kernels.cuh:40` の「Beamer 推奨値」の直接の出典）。
- [x] **[R9] の主張範囲を訂正**：CCGrid'15 は CUDA 6.x 世代の UM 評価であり、oversubscription
      （Pascal 以降）は対象外。oversubscription/prefetch の既存評価は [R10] のみに帰属。
- [x] **vendored PathMerge 上流の第三者性を確認**：`gobardhanm/path-merging-bc` は論文著者の公式
      実装ではない（README に正式引用・ライセンス表記なし）。[R6] に明記し、アルゴリズム帰属は
      名称・設計の一致に基づくことを記載。
- [x] cuGraph の exact/近似の定義を公式 API ドキュメント（cugraph-docs 26.06.00）でも確認
      （vendored ヘッダと一致; `rapidsCugraphBcDocs`）。
- [x] GH200・CUDA（UM/streams/warp shuffle/atomics）・OpenMP・SNAP の公式一次資料を §12.6
      （[R13]–[R20]）として追加。BibTeX を `references.bib` に整備（key は §12.5/§12.6 に併記）。
- [x] [06](06_results_performance.md) §6.4 の cuGraph per-level 計算量（O(M log M)）の記述を削除
      （§12.3 の「formal な出典を確認できない計算量主張はしない」方針と統一; 実測の優劣記述のみ残す）。

## 12.5 参照文献（Crossref 照合済; Stage L0 で独立再検証・BibTeX key 併記）
各項目：`[R#] Authors, "Title," Venue, Year. DOI / URL — SupportedStatement / DifferenceFromThisWork`
（BibTeX key は `references.bib` のエントリに対応）

- **[R1]**（BibTeX: `brandes2001`）Ulrik Brandes, "A faster algorithm for betweenness centrality,"
  *The Journal of Mathematical Sociology*, 25(2):163–177, 2001. DOI: 10.1080/0022250X.2001.9990249 —
  *Supported*: 提案・全 baseline が採用する厳密 BC の基盤（前向き σ / 後向き δ 累積、非重み付き
  O(VE)）。依存度累積の再帰式（Theorem 6）と非重み O(nm)（Corollary 4）を原文で確認（Stage L0,
  著者側公開記録 kops.uni-konstanz.de でも書誌一致）。 *Difference*: 本研究は同アルゴリズムを
  不変のまま GPU バッチ型全始点並列として実装。
- **[R2]**（BibTeX: `beamer2012`）Scott Beamer, Krste Asanović, David Patterson,
  "Direction-optimizing Breadth-First Search," *SC'12: Int. Conf. for High Performance Computing,
  Networking, Storage and Analysis*, pp. 1–10, 2012. DOI: 10.1109/SC.2012.50 —
  *Supported*: Hybrid BFS（H）の top-down/bottom-up 切替（α=14, β=24）の設計根拠（コードに
  「Beamer 推奨値」と明記）。著者公開版の本文に「we suggest α = 14 and β = 24」を確認（Stage L0）。
  *Difference*: 本研究は BC の BFS フェーズに適用（原著は BFS 単体）。**CPU–GPU hybrid ではなく
  方向切替**。
- **[R3]**（BibTeX: `mclaughlin2014`）Adam McLaughlin, David A. Bader, "Scalable and High
  Performance Betweenness Centrality on the GPU," *SC'14: Int. Conf. for High Performance
  Computing, Networking, Storage and Analysis*, pp. 572–583, 2014. DOI: 10.1109/SC.2014.52 —
  *Supported*: GPU 上の厳密 BC が高性能化され既存であること（work-efficient と edge-parallel の
  グラフ特性に応じた選択、scale-free/high-diameter 両対応; 著者公式ページの要旨で「hybrid
  implementation が graph 特性に応じて戦略を選択する」ことを確認）。実装は著者公式リポジトリ
  `Adam27X/hybrid_BC`。 *Difference*: GPU BC 化・並列戦略切替は新発明ではない。本研究は UM
  オーバーサブスクリプション統合と容量・数値整合性の体系評価が焦点で、McLaughlin/Bader は UM
  オーバーサブスクリプション/バッチ分割を扱わない。
- **[R4]**（BibTeX: `mclaughlin2018`）Adam McLaughlin, David A. Bader, "Accelerating GPU
  betweenness centrality," *Communications of the ACM*, 61(8):85–92, 2018. DOI: 10.1145/3230485 —
  *Supported*: [R3] の一般向け改訂（GPU BC の確立を補強; work-efficient / edge-parallel の選択を
  含む）。 *Difference*: 同上。
- **[R5]**（BibTeX: `sariyuce2013`）Ahmet Erdem Sarıyüce, Kamer Kaya, Erik Saule,
  Ümit V. Çatalyürek, "Betweenness centrality on GPUs and heterogeneous architectures,"
  *GPGPU-6 (Workshop on General Purpose Processor Using Graphics Processing Units)*, pp. 76–85,
  2013. DOI: 10.1145/2458523.2458531 — *Supported*: BC の粗粒度（ソース単位; 各ソースの寄与を
  1 スレッドが計算）／細粒度並列の両方式、GPU 向け vertex virtualization（高次数頂点の仮想分割）、
  CPU/GPU ヘテロ同時実行が既存であること（本文で確認, Stage L0）。 *Difference*: ソースを跨いだ
  並列化・GPU 内細粒度並列は既存。本研究は固定 b512 block + 2-stream init + UM/Chunked を GH200 で
  統合し容量境界を評価。
- **[R6]**（BibTeX: `zheng2023galliot`）Zhigao Zheng, Chen Zhao, Peichen Xie, Bo Du, "Galliot:
  Path Merging Based Betweenness Centrality Algorithm on GPU," *IEEE INFOCOM 2023 - IEEE
  Conference on Computer Communications*, pp. 1–10, 2023. DOI: 10.1109/INFOCOM53939.2023.10229018
  （著者公式ページ whu-zhigao.github.io でも書誌確認, Stage L0）—
  *Supported*: path-merging により補助メモリ消費を抑えた GPU 厳密 BC（公式要旨は on-board メモリ
  消費の最小化と大規模グラフ対応を主張）。本研究の **external comparator**（`src/baseline/` に
  vendored）。 *Difference*: 提案 block GPU_Opt が tuned Galliot より median 1.31–3.17×（4グラフ）。
  Galliot は direction-optimizing hybrid BFS・2-stream・UM オーバーサブスクリプション統合を扱わない。
  **PathMerge は ground truth ではない**。
  **上流実装の注記（Stage L0 で確認）**：vendored 実装の上流
  https://github.com/gobardhanm/path-merging-bc（BibTeX: `pathmergeRepo`）は**論文著者の公式実装
  ではなく第三者実装**である（README は「Galloit algorithm: Path-merging-based betweenness
  centrality」と記すのみで正式な論文引用・ライセンス表記なし）。本研究は PathMerge/Galliot の
  帰属を**アルゴリズム名と設計（path-merging・バッチ並列 BFS）の一致**に基づき [R6][R7] とし、
  上流実装の論文への忠実性は検証していない（external comparator という位置づけはこの点でも妥当）。
- **[R7]**（BibTeX: `zheng2023jsac`）Zhigao Zheng, Bo Du, Chen Zhao, Peichen Xie, "Path Merging
  Based Betweenness Centrality Algorithm in Delay Tolerant Networks," *IEEE Journal on Selected
  Areas in Communications*, 41(10):3133–3145, 2023. DOI: 10.1109/JSAC.2023.3310071 —
  *Supported*: [R6] の path-merging BC の雑誌版（巻号頁は Crossref と著者公式ページで一致）。
  *Difference*: 同上（本研究の baseline アルゴリズムの出典）。
- **[R8]**（BibTeX: `rapidsCugraph`, `rapidsCugraphBcDocs`）RAPIDS Development Team,
  "RAPIDS cuGraph — GPU Graph Analytics Library"（オープンソースソフトウェア, Apache-2.0;
  本研究は BC 関連サブセットを `third_party/cugraph/` に vendored, rapids-cmake を v26.04.00 に
  固定）. URL: https://github.com/rapidsai/cugraph — *Supported*: cuGraph は
  `betweenness_centrality` を提供し、`vertices` 未指定で全頂点 traversal による exact BC を計算する
  （vendored 公式ヘッダ `cpp/include/cugraph/algorithms.hpp` で確認。公式 API ドキュメント
  cugraph-docs 26.06.00 も `k=None`（既定）で全頂点による exact、`k` 指定でサンプリング近似と明記,
  参照 2026-07-15）。本研究では `vertices=nullopt, normalized=false, include_endpoints=false`、
  RMM managed_memory_resource で**小規模グラフの補助 baseline** として利用
  （`src/baseline/cugraph_bc.cu`）。 *Difference*: 本研究は cuGraph の内部実装に対する **formal な
  per-level 計算量比較を主張しない**（査読付き原著 DOI も特定できないため、ソフトウェア/公式
  ドキュメントとして引用）。cuGraph を全グラフ baseline とはせず、未測定の優位も主張しない。
- **[R9]**（BibTeX: `li2015um`）Wenqiang Li, Guanghao Jin, Xuewen Cui, Simon See, "An Evaluation
  of Unified Memory Technology on NVIDIA GPUs," *CCGrid'15: 15th IEEE/ACM Int. Symp. on Cluster,
  Cloud and Grid Computing*, pp. 1092–1098, 2015. DOI: 10.1109/CCGrid.2015.105 —
  *Supported*: UM（CUDA 6.x 世代）の利便性とアプリケーション依存の性能特性の既存評価。
  **注（Stage L0 訂正）**：本論文は CUDA 6.x 世代の UM 評価であり、**oversubscription（Pascal
  以降の機能）は対象外**。oversubscription/prefetch の既存評価としては [R10] のみを引く。
  *Difference*: 本研究は**全始点 BC の working set** に UM を適用し GH200 で feasibility を評価。
- **[R10]**（BibTeX: `knap2019um`）Marcin Knap, Paweł Czarnul, "Performance evaluation of Unified
  Memory with prefetching and oversubscription for selected parallel CUDA applications on NVIDIA
  Pascal and Volta GPUs," *The Journal of Supercomputing*, 75(11):7625–7645, 2019.
  DOI: 10.1007/s11227-019-02966-8（Springer 公式ページの要旨で prefetching・oversubscription の
  評価を確認, Stage L0）— *Supported*: prefetching・オーバーサブスクリプション下の UM 性能の
  既存評価（本研究の UM/prefetch 設計背景）。 *Difference*: 既存は選定 CUDA アプリのベンチ。
  本研究は BC 文脈で NVLink-C2C 上の HBM3 超過 feasibility を評価（migration byte 直接計測なし）。
- **[R11]**（BibTeX: `nodehisabet2020subway`）Amir Hossein Nodehi Sabet, Zhijia Zhao, Rajiv Gupta,
  "Subway: minimizing data transfer during out-of-GPU-memory graph processing," *EuroSys'20:
  Fifteenth European Conference on Computer Systems*, pp. 1–16, 2020.
  DOI: 10.1145/3342195.3387537（著者研究室公式実装 `AutomataLab/Subway` を確認, Stage L0）—
  *Supported*: GPU メモリを超えるグラフの out-of-core 処理・データ転送最小化。 *Difference*:
  既存はグラフ本体の out-of-core 処理。本研究は BC の**バッチ（同時ソース数）を SUB_BATCH 分割**
  （グラフ CSR 分割ではない）。
- **[R12]**（BibTeX: `shirahata2014outofcore`）Koichi Shirahata, Hitoshi Sato, Satoshi Matsuoka,
  "Out-of-core GPU memory management for MapReduce-based large-scale graph processing,"
  *CLUSTER'14: IEEE Int. Conf. on Cluster Computing*, pp. 221–229, 2014.
  DOI: 10.1109/CLUSTER.2014.6968748 — *Supported*: GPU メモリを超えるグラフ処理の out-of-core
  メモリ管理の既存例（主張はタイトル・公式メタデータの範囲; 本文照合は未実施 →
  `SOURCE_AUDIT.tsv` に記録）。 *Difference*: 同上（本研究は BC バッチ分割で working set を制御）。

## 12.6 技術仕様・データセットの公式一次資料（Stage L0 追加, [R13]–[R20]）
本研究の設計・環境記述（[00](00_thesis_positioning.md) の GH200 動機、
[04](04_method_design.md) §4.3–4.7/§4.10 の API 記述、[05](05_experimental_setup.md)
§5.1/§5.4 のハードウェア・データセット）が依拠する公式資料。**論文の実測値の出典ではない**
（実測値は repo 内 raw データのみを用い、公称値と混在させない）。Web 資料はすべて
参照日 2026-07-15 を `references.bib` に記録した。

- **[R13]**（BibTeX: `nvidiaGh200Product`）NVIDIA, "NVIDIA GH200 Grace Hopper Superchip"
  （製品ページ）. URL: https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/ —
  *Supported*: Grace CPU と Hopper GPU を NVLink-C2C（900 GB/s の coherent interface）で結合し、
  CPU+GPU coherent memory model（単一 per-process ページテーブル、CPU/GPU 双方から system-allocated
  memory へアクセス可能）を提供するという GH200 のアーキテクチャ記述。
- **[R14]**（BibTeX: `nvidiaGraceHopperInDepth`）Jonathon Evans, Michael Andersch, Vikram Sethi,
  Gonzalo Brito, Vishal Mehta, "NVIDIA Grace Hopper Superchip Architecture In-Depth," *NVIDIA
  Technical Blog*, 2022-11-10. URL:
  https://developer.nvidia.com/blog/nvidia-grace-hopper-superchip-architecture-in-depth/ —
  *Supported*: HBM3（最大 96 GB）・LPDDR5X・ATS（Address Translation Services による単一
  per-process ページテーブル）・「アプリケーションが GPU メモリを oversubscribe し Grace CPU
  メモリを高帯域で直接利用できる」という公式説明（[00] の「HBM3 を超える working set を UM で
  扱える」という動機と、[04] §4.7/[08] の UM oversubscription 設計の前提）。
- **[R15]**（BibTeX: `nvidiaCudaProgrammingGuide`）NVIDIA, "CUDA C++ Programming Guide,"
  Release 13.0. URL: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/ —
  *Supported*: Unified Memory Programming（memory oversubscription・data prefetching・memory
  advise）、Asynchronous Concurrent Execution（stream、データ転送とカーネル実行の重畳）、Warp
  Shuffle Functions（`__shfl_down_sync` 系）、Atomic Functions（`atomicAdd`/`atomicCAS`）の公式
  仕様。実験環境の CUDA 13.0（[05] §5.2）と同版のアーカイブを参照。
- **[R16]**（BibTeX: `nvidiaCudaRuntimeApi`）NVIDIA, "CUDA Runtime API," v13.0.0. URL:
  https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/ — *Supported*:
  `cudaMallocManaged` / `cudaMemPrefetchAsync` / `cudaMemAdvise`（`SetReadMostly`,
  `SetAccessedBy`）/ `cudaMemsetAsync` / Stream Management の関数仕様（[04] §4.5/§4.7 の
  API 呼び出しの根拠）。
- **[R17]**（BibTeX: `openmp52`）OpenMP Architecture Review Board, "OpenMP Application
  Programming Interface," Version 5.2, November 2021. URL:
  https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf — *Supported*:
  CPU baseline（[04] §4.10, `src/baseline/omp.cpp`）が用いる `parallel`/`for`/`atomic`/
  `reduction`/`omp_get_wtime` の公式仕様（いずれも OpenMP 2.x 系以降の中核構文であり、
  版の選択に依存しない）。
- **[R18]**（BibTeX: `snapnets`）Jure Leskovec, Andrej Krevl, "SNAP Datasets: Stanford Large
  Network Dataset Collection," June 2014. URL: http://snap.stanford.edu/data — *Supported*:
  email-EuAll / roadNet-PA / roadNet-TX / roadNet-CA の公式配布元（SNAP が要請する公式引用形式に
  準拠）。email-EuAll が原本 directed であること（[05] §5.4 の無向化前処理の前提）、roadNet-CA が
  undirected であることを公式ページで確認。※公式ページのノード数（例: email-EuAll 265,214）と
  本研究の CSR（265,009）の差は前処理（無向化・整形）によるもので、`result/datasets/
  graph_catalog.tsv` に SHA256 とともに記録済み。
- **[R19]**（BibTeX: `leskovec2007graphevolution`）Jure Leskovec, Jon Kleinberg, Christos
  Faloutsos, "Graph evolution: Densification and shrinking diameters," *ACM Transactions on
  Knowledge Discovery from Data*, 1(1), Article 2, 2007. DOI: 10.1145/1217299.1217301 —
  *Supported*: email-EuAll データセットの原著（SNAP 公式ページの出典表示に基づく）。
- **[R20]**（BibTeX: `leskovec2009community`）Jure Leskovec, Kevin J. Lang, Anirban Dasgupta,
  Michael W. Mahoney, "Community Structure in Large Networks: Natural Cluster Sizes and the
  Absence of Large Well-Defined Clusters," *Internet Mathematics*, 6(1):29–123, 2009.
  DOI: 10.1080/15427951.2009.10129177 — *Supported*: roadNet-PA/TX/CA データセットの原著
  （SNAP 公式ページの出典表示に基づく）。
