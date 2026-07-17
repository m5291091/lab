# Chapter 5 Experimental Methodology

本章では、本研究の評価方法を規定する。対象とする厳密な Betweenness Centrality（BC, 媒介中心性）は、全始点からの Breadth-First Search（BFS, 幅優先探索）に基づく Brandes アルゴリズム [@brandes2001] で計算される。以下では、提案するバッチ型 GPU 実行基盤とその主実装 GPU_Opt を、評価対象の第三者実装 PathMerge および補助 baseline と比較するための計算機環境、評価グラフ、比較実装、実行設定、計時・統計処理、要因分析、容量評価、正確性検証、再現性管理の各手続きを述べる。本章は評価の「方法」を対象とし、具体的な数値結果は Chapter 6 から Chapter 9 で示す。本章で数値を示すのは、原則として実験条件の記述に必要な設定値（バッチサイズ、試行数、許容値など）に限る。

すべての記述は `thesis_bc_project/` 配下の正式資料に基づく。実験時コードは checkpoint（SourceSnapshotID）で凍結保存され、生データは `raw_data/`、派生表・図は `result/` に分離される。一次情報は各 `SOURCE.md`・`README.md`・`*.tsv` を参照する。

## 5.1 Research Questions

本研究の評価は、次の 4 つの Research Question に沿って構成される。

- **RQ1（性能）**: 評価した 4 グラフにおいて、固定バッチ b512 の block ベース GPU_Opt は、グラフごとに調整した第三者実装 PathMerge より高速か。
- **RQ2（最適化の寄与）**: Hybrid BFS、Warp-Cooperative Accumulation、Dual-Stream Execution、および block カーネルは、観測された性能にどの程度寄与するか。
- **RQ3（メモリ容量拡張性）**: GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunked のメモリ管理方式は、実行可能なバッチサイズにどのような影響を与えるか。
- **RQ4（正確性と数値的挙動）**: 提案実装の BC ベクトルは独立参照および異なるメモリ経路とどこまで一致し、どの条件で未解決の差が残るか。

本評価では、性能評価（RQ1）・要因分析（RQ2）・容量評価（RQ3）・数値的正確性（RQ4）を互いに独立した観点として分離する。性能評価は median 実行時間・speedup・GTEPS を対象とし、正確性検証は BC ベクトルの数値一致を対象とするため、測定対象も判定基準も異なる。したがって、正確性検証に用いた実行（各構成 n=1、warmup なし）の時間値を性能主張に用いず、逆に性能測定の実行を正確性の根拠にも用いない。要因分析と容量評価も、それぞれ合成グラフ・特定グラフに限定した専用実験であり、RQ1 の主性能比較とは対象グラフも目的も異なる。

各 RQ と実験項目の対応、主指標、集計方法、試行数の概要を Table 5.1 に示す。試行数と集計の詳細は 5.6 節で述べる。

**Table 5.1: Evaluation protocol summary (mapping of research questions to experiments).**

| RQ | Experiment | Graphs | Primary metric | Aggregation | Trials (n) |
|---|---|---|---|---|---|
| RQ1 | Main performance: GPU_Opt vs tuned PathMerge | email-EuAll, roadNet-PA, roadNet-TX, roadNet-CA | runtime, speedup, GTEPS | median | GPU_Opt: email 5 / road 3; PathMerge: 3 |
| RQ2 | Ablation (H/W/A) | benchmark_7000_41459, benchmark_11023_62184, 56438_300801, 325557_3216152, email-EuAll | main effect | median | synthetic 5 / email 3 |
| RQ2 | Kernel selection (forced shared/block) | roadNet-PA, roadNet-TX | runtime, speedup | median (+ sample SD) | 3 |
| RQ3 | Memory scalability (UM / Pure / Chunked) | 325557_3216152 | feasibility (SUCCESS/OOM), max success batch | median (+ SUCCESS/OOM status) | legacy 5; memory-path 1 |
| RQ4 | Correctness (independent reference & memory paths) | benchmark_7000_41459, benchmark_11023_62184, chain_200, 325557_3216152 | mismatch, max abs/rel error | single-run comparison | 1 |

> Source: trial counts and aggregation from `result/main_performance/proposed_variants/SOURCE.md`, `result/tuning/{pathmerge,kernel_selection}/SOURCE.md`, `result/ablation/*/SOURCE.md`, `result/memory_scalability/SOURCE.md`, `result/correctness/*/README.md`.

## 5.2 Hardware and Software Environment

本評価は Miyabi-G スーパーコンピュータの GPU 計算ノードで実施した。計算機は NVIDIA GH200 Grace Hopper Superchip（sm_90）である。GH200 は Grace CPU の LPDDR5X メモリと GPU の HBM3 を NVLink-C2C（900 GB/s coherent）で結合する coherent memory model を採用しており [@nvidiaGh200Product; @nvidiaGraceHopperInDepth]、Unified Memory（UM）による HBM3 容量超過時の Grace CPU メモリへのスピルを可能にする。この特性は RQ3 のメモリ容量評価の前提となる。

GPU のメモリ容量については、根拠と単位系の異なる値を区別して扱う。オンパッケージ HBM3 の公称容量は 96 GB である [@nvidiaGraceHopperInDepth]。実行環境の記録に残るデバイスメモリ容量は 97871 MiB であり、これは約 95.6 GiB、10 進表記では約 102.6 GB に相当する。公称 96 GB と記録された 97871 MiB は、同一のオンパッケージ HBM3 を異なる単位系・取得方法で示したものであり、別個のメモリ領域や異なるメモリ階層を表すものではない。これとは別に、runner 自身のメモリ照会は実行開始時に総量約 102.0 GB、空き（`free_before`）約 101.4 GB（いずれも 10 進 GB）を報告した。総量約 102.0 GB と 97871 MiB（約 102.6 GB）の差は取得方法の違いによるものであり、対象とする HBM3 は同一である。約 101.4 GB は総容量ではなく実行開始時点の利用可能量であり、本評価では実効バッチのクランプ判定に用いるメモリ予算の基準（推定作業集合量との比較）としてのみ扱う。実行環境の一次記録は `result/environment/environment.md`、実行開始時のメモリ照会値は各実行の保存ログである。

ソフトウェア環境は、NVIDIA driver 595.58.03、CUDA Toolkit（nvcc）release 13.0（V13.0.48）[@nvidiaCudaProgrammingGuide]、ホスト C++ コンパイラ g++（GCC）11.4.1、CMake 4.3.4、Nsight Systems（nsys）2025.5.1.121 である。主実験は Miyabi-G 上の PBS batch system（group `gj17`）を通じて実行した。RQ3・RQ4 のメモリ経路実験は、ホストメモリ容量を明示的に制限した 100 GiB 資源構成で実行した。この構成はホストメモリ上限が legacy 容量評価と異なるため、両者の OOM 境界は一致しない（5.9 節）。したがって、本研究の全実験が単一の資源指定で実行されたわけではない。なお、実際の queue 名は保存ログから独立に確定できないため、本評価の正式な実験条件には含めない（5.12 節）。

GH200 の各メモリ経路（HBM3 の device-to-device、pinned host-device、NVLink-C2C prefetch）の実効帯域は、bandwidth ベンチマークで別途測定した（`raw_data/profiling/job_2359175_20260711/bandwidth.log`）。測定された帯域値は platform の性能特性であるため本章では示さず、メモリ経路の議論（Chapter 8）で扱う。実験群と checkpoint（SourceSnapshotID）の対応は 5.11 節で述べる。`result/` 全体は単一 checkpoint に対応しない。

**Table 5.2: Experimental environment.**

| Component | Specification |
|---|---|
| GPU | NVIDIA GH200 Grace Hopper Superchip (sm_90) |
| On-package GPU memory (HBM3), nominal | 96 GB (NVIDIA specification) |
| On-package GPU memory (HBM3), recorded for the run environment | 97871 MiB (= approx. 95.6 GiB = approx. 102.6 decimal GB; same HBM3 as the nominal 96 GB) |
| GPU memory reported by the runtime query at run start | total approx. 102.0 GB; free (`free_before`) approx. 101.4 GB (decimal GB) |
| CPU memory | Grace LPDDR5X, coupled to HBM3 via NVLink-C2C (900 GB/s coherent) |
| NVIDIA Driver | 595.58.03 |
| CUDA Toolkit (nvcc) | release 13.0, V13.0.48 |
| Host C++ Compiler | g++ (GCC) 11.4.1 |
| CMake | 4.3.4 |
| Nsight Systems (nsys) | 2025.5.1.121 |
| Scheduler / Group | PBS batch system (Miyabi-G), group gj17 |
| Resource configuration — memory-path experiments (RQ3/RQ4) | Host-memory-limited 100 GiB configuration |

<!-- canonical artifact: T6_experimental_environment -->
> Source: `result/environment/environment.md`, `result/MANIFEST.md`; nominal HBM3 capacity from [@nvidiaGraceHopperInDepth]; run-start memory query values (`total`, `free_before`) from the saved run logs. The nominal 96 GB and the recorded 97871 MiB denote the same on-package HBM3 in different units and query methods, not separate memory regions. The queue name cannot be independently confirmed from the saved job logs and is therefore excluded from the reported experimental conditions (5.12). Memory-path bandwidth values are reported in Chapter 8.

## 5.3 Graph Datasets

本評価で用いたグラフの属性を Table 5.3 に示す。数値は `result/datasets/graph_catalog.tsv` および `docs/graph_stats.tsv` の正式記録から取得したものであり、丸め値からの逆算や推定は行っていない。すべてのグラフは無向・非重みの CSR 形式で保持し、入力の同一性は `graph_catalog.tsv` に記録された graph SHA256 で管理する。

グラフは実データと合成グラフに大別される。実データは SNAP から取得した 4 グラフである [@snapnets]。email-EuAll は原本が directed の電子メール通信網であり [@leskovec2007graphevolution]、本評価では無向化して用いた（`Symmetrized=yes`）。roadNet-PA/TX/CA は原本が undirected の道路網であり [@leskovec2009community]、対称化せずそのまま用いた（`Symmetrized=no`）。これら 4 グラフが RQ1 の対象である。合成グラフは `tools/gen_graph.py` で生成した無向グラフで、要因分析・容量評価・正確性検証に用いた（325557_3216152 は 1-indexed の高次数グラフ）。SNAP グラフの self-loop・重複辺処理、および生成グラフの directed 素性・対称化情報は記録が無いため `unknown` とし、推定による補完は行わない。

各グラフの選択目的は次のとおりである。RQ1 の 4 グラフは対照的な 2 種の構造を含む。email-EuAll は変動係数の大きいハブ構造をもち BFS 深さが浅く、roadNet-PA/TX/CA は次数が均質で BFS 深さが深い（詳細は Table 5.3）。この対照により、次数分布と探索深さの異なる領域で RQ1 を評価する。325557_3216152 は平均次数・辺数が本研究で最大級の作業集合をもつため、HBM3 容量を人為的に超過させる RQ3 のメモリ容量評価および RQ4 のメモリ経路正確性検証に用いた。benchmark_7000_41459、benchmark_11023_62184、chain_200 の 3 グラフは小規模であり、独立参照との全ベクトル正確性検証（RQ4）に用いた。

**Table 5.3: Graph datasets. Degree statistics are from `docs/graph_stats.tsv`; provenance columns are from `result/datasets/graph_catalog.tsv`.**

| Graph | Category | Nodes | Edges | Avg. Degree | Max Degree | Directed Input | Symmetrized | Primary use |
|---|---|---|---|---|---|---|---|---|
| email-EuAll | real (hub) | 265009 | 364481 | 2.751 | 7636 | directed | yes | RQ1 |
| roadNet-PA | road network | 1088092 | 1541898 | 2.834 | 9 | undirected | no | RQ1 |
| roadNet-TX | road network | 1379917 | 1921660 | 2.785 | 12 | undirected | no | RQ1 |
| roadNet-CA | road network | 1965206 | 2766607 | 2.816 | 12 | undirected | no | RQ1 |
| 325557_3216152 | synthetic (high degree) | 325557 | 3216152 | 19.758 | 18280 | unknown | unknown | RQ3 / RQ4 |
| 56438_300801 | synthetic | 56438 | 300801 | 10.660 | 604 | unknown | unknown | RQ2 |
| benchmark_7000_41459 | synthetic | 7000 | 41459 | 11.845 | 589 | unknown | unknown | RQ2 / RQ4 |
| benchmark_11023_62184 | synthetic | 11023 | 62184 | 11.283 | 2109 | unknown | unknown | RQ2 / RQ4 |
| chain_200 | synthetic (chain) | 200 | 199 | 1.990 | 2 | unknown | unknown | RQ4 |

<!-- canonical artifact: T1_graph_metadata -->
> Source: Nodes / Edges / degrees from `docs/graph_stats.tsv` (undirected edge count m); Directed Input / Symmetrized / preprocessing / SHA256 from `result/datasets/graph_catalog.tsv` ("unknown" = not recorded for generated graphs).

## 5.4 Evaluated Implementations

本評価で比較した実装を Table 5.4 に示す。本研究の主実装は GPU_Opt（Unified Memory、常時 block カーネル）であり、その主要な比較対象は、グラフごとに調整した第三者実装 PathMerge（tuned）である。補助 baseline として Sequential、OpenMP、cuGraph を用いる。GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunked は 3 つの独立した提案ではなく、同一のバッチ型 GPU 実行基盤におけるメモリ管理方式のバリエーションである。GPU_Opt_Pure は明示的な `cudaMalloc`/`cudaMemcpy` によるデバイス専用管理の対照であり、HBM3 容量を超える作業集合を扱えない。GPU_Opt_Pure_Chunked は作業集合を sub-batch に分割して実行可能バッチを拡張する。これら 3 実装はいずれも block カーネルを用いる（ソースは Table 5.4）。

主要 baseline の PathMerge については、次を明記する。本研究で評価対象とした PathMerge は、Galliot（path-merging 型 BC アルゴリズム）[@zheng2023galliot; @zheng2023jsac] の**第三者実装**であり、上流は `gobardhanm/path-merging-bc` である [@pathmergeRepo]。これは原著論文著者による公式実装ではなく、上流リポジトリには明示的なライセンス表記が確認できていない。本研究の vendored baseline（`src/baseline/pathmerge.cu` + `galliot*.cu`）はこの上流を adapter 化した派生であり、無向グラフの慣習に合わせて最終 BC 値を 2 で除する。PathMerge は external comparator であって ground truth ではない。したがって比較結果は、評価に用いたこの実装・環境・グラフに限定され、PathMerge/Galliot アルゴリズム一般や原著者の公式実装への優劣を主張しない。

補助 baseline は次のとおりである。Sequential（`src/baseline/sequential.cpp`）と OpenMP（`src/baseline/omp.cpp`）は Brandes アルゴリズム [@brandes2001] の CPU 逐次・並列実装である [@openmp52]。cuGraph（`src/baseline/cugraph_bc.cu`）は RAPIDS cuGraph [@rapidsCugraph] の `betweenness_centrality` を呼び出す。cuGraph は実コードにおいて exact 計算（`vertices=std::nullopt`、全始点）、非正規化（`normalized=false`）、端点非包含（`include_endpoints=false`）、無向扱い（`is_symmetric=true`）で構成され、この設定は提案・PathMerge と整合する [@rapidsCugraphBcDocs]。ただし、本研究の cuGraph adapter は明示的な 2 除算を適用しておらず、cuGraph 内部の無向対称性の扱いは本環境で未確認である。また cuGraph の計時範囲は関数全体（初期化・転送・BC 計算・回収）を含み、初期化オーバヘッドを内包する。以上の理由から、cuGraph は小規模グラフに限定した補助 baseline として扱い、提案の主比較や正確性の ground truth には用いない。

補助 baseline（Sequential/OpenMP/cuGraph）を含む複数実装の比較は小規模グラフに限定され、medium/large 規模では Sequential/OpenMP/cuGraph の測定を欠く（`result/main_performance/seven_implementations/README.md`）。したがって本研究は、現行の block 実装による全グラフ統一の 7 実装比較表を提示しない。

**Table 5.4: Compared implementations and their roles.**

| Implementation | Algorithm / basis | Memory strategy | Role in this study | Source |
|---|---|---|---|---|
| GPU_Opt | Proposed batch-based framework (block kernel) | Unified Memory (managed) | Main implementation | `src/proposed/host_um.cu` |
| PathMerge (tuned) | Galliot path-merging (third-party implementation) | Device (int2 frontier + per-source arrays) | Primary baseline / external comparator | `src/baseline/pathmerge.cu`, `galliot*.cu` |
| GPU_Opt_Pure | Proposed framework (block kernel) | Explicit cudaMalloc / cudaMemcpy | Device-only memory control | `src/proposed/host_pure.cu` |
| GPU_Opt_Pure_Chunked | Proposed framework (block kernel) | Chunked working set (sub-batch) | Capacity-extension variant | `src/proposed/host_chunked.cu` |
| Sequential | Brandes (CPU serial) | Host | Supplementary baseline (small only) | `src/baseline/sequential.cpp` |
| OpenMP | Brandes (CPU parallel) | Host | Supplementary baseline (small only) | `src/baseline/omp.cpp` |
| cuGraph | RAPIDS primitives (exact) | Managed (RMM) | Supplementary baseline (small only) | `src/baseline/cugraph_bc.cu` |

## 5.5 Parameter Settings

本評価における実行設定を規定する。すべての実装は無向・非重み BC を対象とし、最終 BC 値を 2 で除する慣習で整列されている。

GPU_Opt（RQ1 主実装）は、1 ストリーム当たり 512 ソースの固定バッチ b512、block カーネル（1 block = 1 source）で実行した。in-capacity 条件では、要求バッチ・実効バッチともに 512、SUB_BATCH=512、num_subs=1 であり、dual-stream 実行により NS_eff=2 であった（`result/main_performance/proposed_variants/SOURCE.md`）。重要な点として、GPU_Opt の結果はグラフごとに最速バッチを探索したものではない。提案手法のバッチ掃引は実施せず、全 4 グラフで同一の固定 b512 を用いた。

PathMerge tuned（RQ1 分母）は、グラフごとに候補バッチを測定して選択した。採用バッチは email-EuAll が b2048、roadNet-PA/TX が b64、roadNet-CA が b32 である（手続きは 5.7 節）。GPU_Opt が全グラフ固定 b512 であるのに対し PathMerge はグラフごとに調整しており、両者のバッチ設定は非対称である。本評価はこの非対称性を明示した上で比較する。

要求バッチと実効バッチの区別は本評価で重要である。要求バッチ（`BC_BATCH_OVERRIDE` または `PATHMERGE_BC_BATCH_SIZE` で指定）が HBM3 予算を超える場合、実効バッチは縮小される。GPU_Opt 系では、oversubscription 時に SUB_BATCH が動的に縮小して num_subs>1 となる。本評価では、要求バッチ・実効バッチ・SUB_BATCH・num_subs・NS_eff を区別し、clamp が生じた場合はその記録（各実行の stderr / `execution_summary`）に基づいて条件を記述する。具体的な clamp 値の観測結果は Chapter 6 および Chapter 8 で示す。

warmup については、SourceSnapshotID `phase_def_block_20260710` の新規測定（proposed_variants、kernel_selection、PathMerge 掃引、correctness）では warmup を行わず、ベンチマークスクリプトは全試行を記録して discard しない。ただし ablation synthetic（job 2354994）は例外で、各 `run_ablation <graph> all` invocation、すなわち各 graph/trial の8構成セットの先頭で、全構成に対する global・untimed H1W1A1 warmup を1回実行し、TSV本試行には含めない。これは実験時script、raw logの20 warmup marker、8構成×4 graph×5 trial=160行のraw TSV、runner snapshotから確認した。legacy baseline では明示的な warmup 記録が無いため `not_recorded` として扱う。旧ツリーの UM feasibility sweep（Series A）は方式別であり、GPU_Opt と GPU_Opt_Pure_Chunked は warmup なし（実験時 snapshot `oldtree_f05ec52_20260512` の実行スクリプトに warmup ループが無く全実行を試行として記録し、保存ログの実行数と raw TSV の試行行数が 1:1 で一致する）、GPU_Opt_Pure はログに試行 header が無く生成ドライバが snapshot に未収録のため `not_recorded` とする。

性能測定と correctness-only 実行は区別する。性能測定は median 集計のために複数試行を記録し、その時間値を性能主張に用いる。一方、correctness-only 実行（小規模正確性・メモリ経路正確性）は各構成 n=1、warmup なしで実施し、その時間値は性能評価に用いない。実行時の調整ノブ（`BC_BATCH_OVERRIDE`、`PATHMERGE_BC_BATCH_SIZE`、`CUGRAPH_BC_MAX_SOURCES_PER_BATCH`、`BC_FORCE_BFS_KERNEL`）は通常の実験手続きの一部であり、条件を人為的に指定する RQ3/RQ4 やカーネル比較で用いる。

## 5.6 Timing and Statistical Method

計時範囲（timing scope）は、runner（`src/core/runner.cpp`）が各実装関数の全体（ホスト制御 + カーネル実行）を `Time_sec` として stdout に出力する範囲である。BFS/Backward などのフェーズ内訳は stderr に別途出力される。warmup を本試行に含めない。

試行数（trials）は実験群ごとに異なり、その一覧は Table 5.1 に示す。すなわち RQ1 の GPU_Opt は email-EuAll で n=5・roadNet で n=3、PathMerge tuned は n=3、ablation は合成 4 種で n=5・email で n=3、kernel selection は n=3、legacy 容量評価は各バッチ n=5、メモリ経路実験と profiling は各 1 である。

主値には median（中央値）を用いる。補助値として mean、標本標準偏差（sample standard deviation）、min、max を扱う。標本標準偏差は次式で定義される不偏推定量（ddof=1）である。

$$
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}\left(t_i - \bar{t}\right)^2}
$$

ここで $t_i$ は各試行の実行時間、$\bar{t}$ は標本平均、$n$ は試行数である。単一の最速試行を代表値としない。

speedup は median 同士の比として計算する。baseline と提案手法の median 実行時間 $T^{\mathrm{med}}_{\mathrm{baseline}}$、$T^{\mathrm{med}}_{\mathrm{proposed}}$ に対して次式で定義する。

$$
S = \frac{T^{\mathrm{med}}_{\mathrm{baseline}}}{T^{\mathrm{med}}_{\mathrm{proposed}}}
$$

median と mean を混在させて speedup を計算しない。

throughput は GTEPS（Giga Traversed Edges Per Second）で表す。ノード数 $n$、無向辺数 $m$、実行時間 $T$（秒）に対して全実装で次式に統一する。

$$
\mathrm{GTEPS} = \frac{n \cdot m}{T \cdot 10^{9}}
$$

OOM・TIMEOUT・FAIL の扱いは本評価で重要である。これらは 0 秒として扱わない。feasibility を評価する表では、実行不能を `Status=OOM_OR_FAIL` として記録し、`Time_sec=0` は「未達」を表すマーカーであって性能値ではない。取得不能な値は `N/A` とする。この規約により、実行不能条件が誤って高速な性能値として集計されることを防ぐ。

## 5.7 Performance Comparison and PathMerge Tuning Procedure

本節では、RQ1 の性能評価手続きと、その分母となる PathMerge tuned の調整手続きを述べる。具体的な speedup 値は Chapter 6 で示す。主性能比較は、提案の block ベース GPU_Opt と PathMerge tuned を email-EuAll、roadNet-PA、roadNet-TX、roadNet-CA の 4 グラフで比較する。GPU_Opt（分子）は固定 b512、median 集計、warmup なし、SourceSnapshotID `phase_def_block_20260710` の測定であり、PathMerge tuned（分母）はグラフごとの調整バッチによる median 集計である。両者の median 実行時間から 5.6 節の式で speedup と GTEPS を算出する。

PathMerge の tuning 手続きは、掃引による screening と tuned 値の確定からなる。各グラフについて候補バッチを掃引し（掃引範囲は roadNet-PA が b8–512、roadNet-TX が b32–128、roadNet-CA が b16–128、email-EuAll が b8–8192）、median 実行時間が最良のバッチを tuned として採用した。掃引の試行数はバッチごとに 1〜4 で不均一であり、各バッチの試行数は各グラフの `SOURCE.md` に記録される。warmup は行わず、集計は median である。email-EuAll では b2048、roadNet-CA では b32 が掃引実測の最良値として採用された。

roadNet-PA/TX の分母には、注意すべき測定条件がある。両グラフでは、掃引によって最適バッチが既定の b64 と同一であることを確認した上で、最終的な分母には同一 b64 設定の legacy baseline 実測値（`result/main_performance/seven_implementations/legacy_partial/`、checkpoint `oldtree_f05ec52_20260512`）を保守的な基準として採用した。この legacy 実測値は掃引の確認測定値よりわずかに速いため、両グラフの speedup は保守的（過小）に見積もられる方向に働く。すなわち、roadNet-PA/TX の PathMerge 分母は legacy checkpoint 由来である一方、提案手法（分子）は全 4 グラフで `phase_def_block_20260710` の測定であり、両者の測定 checkpoint が異なる。本評価はこの相違を明示する。

なお、本 tuned 基準の speedup は既定 b64 に対する比較とは区別され、本研究は tuned 基準の値を主張の中心とする。両基準の具体値は Chapter 6 で対比する。前述のとおり、Sequential/OpenMP/cuGraph を含む小規模比較は補助的な位置づけであり、その提案系は旧 shared カーネルで測定されているため、現行 block 実装による統一 7 実装比較表ではない。

## 5.8 Ablation and Kernel Analysis Method

RQ2 の要因分析は、ablation と kernel selection の 2 つの手続きからなる。

Ablation は、提案手法の 3 つの工夫、すなわち Hybrid BFS（H、top-down/bottom-up の二方向 BFS [@beamer2012]）、Warp-Cooperative Accumulation（W、Backward phase の warp 協調累積）、Dual-Stream Execution（A、2 ストリームによる非同期初期化と計算の重畳）の寄与を測定する。これらを compile-time テンプレートで有効・無効に切り替えた 8 構成（$\mathrm{H}\{0,1\}\mathrm{W}\{0,1\}\mathrm{A}\{0,1\}$）を固定バッチ b512、median 集計で測定した。合成4グラフのjob 2354994では各8構成セットの前にglobal・untimed H1W1A1 warmupを1回行い、TSV本試行から除外した。対象は合成グラフ 4 種（benchmark_7000_41459、benchmark_11023_62184、56438_300801、325557_3216152）で各構成 n=5、および email-EuAll で n=3 である（`result/ablation/{synthetic_2354994,email_2354999}/SOURCE.md`）。

各工夫の寄与は主効果（main effect）で評価する。ある工夫 $F$ の主効果は、他の 2 軸の全水準にわたって平均した、$F$ を無効にした場合と有効にした場合の実行時間比 $T(F{=}0)/T(F{=}1)$ の幾何平均として計算する。合成グラフの主効果は 4 グラフの幾何平均で要約し、email-EuAll は個別に扱う。本評価では、要因分析の主効果を roadNet などの未測定グラフへ一般化しない。特に、Warp 協調累積の効果はグラフ依存であり、email-EuAll では中立的または僅かに不利となり得るため、この点を限定条件として扱う。具体的な主効果値は Chapter 7 で示す。

Kernel selection は、BFS カーネルの選択に関する直接比較である。`BC_FORCE_BFS_KERNEL=shared|block` によって shared-frontier カーネルと block カーネル（1 block = 1 source）を強制実行し、roadNet-PA と roadNet-TX で比較した。設定はバッチ 512、SUB_BATCH=512、num_subs=1、n=3、warmup なし、median 集計（標本標準偏差併記）であり、自動選択則に依存せず forced shared/block の実測のみを対象とする（`result/tuning/kernel_selection/SOURCE.md`）。旧実装には平均次数に基づく自動選択則（`avg_deg < 5 → shared`）が存在したが、現行方式では使用していない。したがって kernel selection の結論は roadNet-PA/TX の強制比較に限定され、他グラフへ一般化しない。

Profiling は、56438_300801 に対する ablation バイナリの `ablation_H1W1A0` nsys トレースにより、BFS カーネルと Backward カーネルの実行時間比率を得るものである（単一トレース、`raw_data/profiling/job_2359175_20260711/ablation_H1W1A0.stats.txt`）。このトレースの本測定は H1W1A0 構成であるが、同一 process 冒頭の untimed H1W1A1 warmup もtrace scopeに含むため、63.9% / 36.1% は本測定だけを分離した値ではなく、warmupを含む単一トレース全体の CUDA GPU カーネル時間構成比である。同じ PBS job 2359175 には、同じ 56438_300801 を対象とする `ablation_H1W1A1` の別トレースと、325557_3216152 を対象とする GPU_Opt の UM prefetch 別トレースも含まれるが、63.9% / 36.1% の構成比には用いない。本評価では、フェーズ内訳から因果を断定せず、この観測された配分を、当該 1 グラフ・1 トレースに限定して記述する。

## 5.9 Memory Scalability Protocol

RQ3 のメモリ容量拡張性は、325557_3216152 の 1 グラフに限定して評価する。このグラフは本研究で最大級の作業集合をもつため、人為的にバッチを増加させて HBM3 容量を超過させ、GPU_Opt（Unified Memory）、GPU_Opt_Pure（デバイス専用）、GPU_Opt_Pure_Chunked（sub-batch 分割）の実行可能性を比較する。

本評価が対象とするのは最速実行時間ではなく容量到達性（feasibility）である。各実装が各バッチで SUCCESS するか OOM するか、および最大成功バッチを記録する。判定規則は 5.6 節の OOM 規約に従い、OOM は 0 秒として扱わない。具体的な feasibility 境界（最大成功バッチと OOM 到達バッチ）は Chapter 8 で示す。

本評価では、この feasibility 評価に 2 つの測定系が関与し、両者の OOM 境界は直接比較できないことを明示する。第 1 は legacy 容量評価（各バッチ n=5、SourceSnapshotID `oldtree_f05ec52_20260512`、旧ツリー、2026-05-12 測定）である。この legacy 結果は、時間値を最新 block 性能値として採用せず、feasibility（SUCCESS/OOM 傾向）のみを限定的に採用する。これはメモリサイジングのコードが現行 checkpoint と文字単位で同一であることに基づく再利用であり、現行 checkpoint で同じ境界を再測定したものではない。第 2 はメモリ経路正確性実験（canonical checkpoint `memory_correctness_20260712`、job 2368587、各構成 n=1、5.2 節の 100 GiB 資源構成）であり、ホストメモリ上限が異なるため legacy 系とは OOM 境界が異なる。

本評価では次を主張しない。Unified Memory が無制限に容量を拡張するとは主張しない（GPU_Opt も十分大きなバッチで OOM する）。Chunked があらゆる条件で OOM を完全に回避するとも主張しない。Chunked の主な利点は最高性能ではなく実行可能バッチの拡大であり、その結論は GH200・325557・試験バッチ範囲に限定される。

## 5.10 Correctness Validation

RQ4 の正確性検証は、水準を分けて実施する。本評価では、最大 BC の index/value のみの一致は完全な正確性証明とはみなさない。したがって、正確性の水準を Table 5.5 のように定義し、各比較がどの水準に属するかを区別する。

**Table 5.5: Correctness validation levels.**

| Level | Definition | Applied to |
|---|---|---|
| full_vector_independent_reference | All BC elements compared against an independent implementation | benchmark_7000_41459, benchmark_11023_62184, chain_200 (Sequential vs GPU_Opt) |
| full_vector_same_implementation | All BC elements compared across configurations of the same implementation | PathMerge default b64 vs tuned; memory-path UM/Pure/Chunked same-batch |
| max_bc_only | Only the maximum BC index/value compared | Headline 4-graph cross-implementation; kernel selection |
| none | No BC comparison recorded | Ablation runs |

> Source: correctness-level ordering from `result/CLAIMS.md`.

本評価は 4 種の比較を用いる。第 1 に、独立参照による小規模全ベクトル比較（`full_vector_independent_reference`）は、Sequential を独立参照、GPU_Opt を candidate として benchmark_7000_41459、benchmark_11023_62184、chain_200 の全 BC 要素を比較する（checkpoint `small_correctness_20260712`、job `2367583.opbs`、各構成 n=1、warmup なし；時間値は性能主張に用いない）。第 2 に、同一実装の異バッチ間比較は、PathMerge の tuned バッチが既定 b64 と同一の BC を出すことを `--dump-bc` の全ベクトル比較で検証する（email-EuAll の b64 vs b2048、roadNet-CA の b32 vs b64）。第 3 に、メモリ経路の same-batch 比較は、325557 の同一バッチ b1024 で UM/Pure/Chunked の 3 経路の全ベクトルを比較する。第 4 に、stress 条件は、同一実装の大バッチ・oversubscription 出力（gpu_opt b9792、chunked b16384）を b1024 と比較する。

本評価で記録する正確性指標は、最大 BC index/value のみではなく、ベクトル長、欠損 index 数、不一致要素数、最大絶対・相対誤差とその index における reference / candidate 値、NaN/Inf の有無、ベクトルおよび入力グラフの SHA256 を含む。これにより、最大 BC の一致だけでは捉えられない要素単位の差異を記録する。

判定は混合許容基準（mixed absolute-relative tolerance）による。各 index について次を満たすことを要求する。

$$
\lvert r_i - c_i \rvert \le \mathrm{abs\_tol} + \mathrm{rel\_tol} \cdot \max\!\left(\lvert r_i \rvert, \lvert c_i \rvert\right)
$$

ここで $r_i$ は reference、$c_i$ は candidate の値である。正式な許容値は $\mathrm{abs\_tol}=1\mathrm{e}{-3}$、$\mathrm{rel\_tol}=1\mathrm{e}{-6}$ である（`result/correctness/small_full_vector/correctness_summary.tsv`、`result/correctness/memory_paths/README.md`）。BC 値が $\sim 10^{10}$ と大きい場合、絶対許容単独は不適切であるため、絶対許容の超過は WARN として分離し、単独の失敗判定にはしない。許容値を事後に変更して判定を PASS に変える操作は行わない。

メモリ経路の未解決不一致（Core Fail）を隠さない。メモリ経路正確性の canonical 実験（`memory_correctness_20260712`、job 2368587）の formal overall status は `CORE_FAIL` である。同一バッチ b1024 の 3 経路比較は混合許容内で不一致 0 であった一方、stress 条件では正式な $\mathrm{rel\_tol}=1\mathrm{e}{-6}$ を超える構成依存の差が影響頂点の和集合（計 8 頂点）で観測された。診断（`memory_diagnostic_20260713`、job 2369632）では、full memset の強制と NS_eff=1 の強制のいずれの単独変更でも b1024 CONTROL との差を再現できず、大バッチ・sub-batch 分割・grid/occupancy またはその組合せとの関連が残った。本評価では、この差の原因を確定しておらず、原因未確定の差を浮動小数点誤差と断定しない。許容感度分析（$\mathrm{rel\_tol}=3\mathrm{e}{-6}$ で差が消失すること）は補助情報であり、正式な FAIL を PASS に変更しない。また、PathMerge（external comparator）との差は本 stress 差とは別の regime であり、正誤は未決定であって、この差から提案実装が誤りであるとは断定しない。stress 全ベクトル正確性、および UM/Chunked の全条件正確性は証明していない。具体的な誤差値と影響頂点の内訳は Chapter 9 で示す。

## 5.11 Reproducibility and Data Provenance

本研究のアーカイブは、Git 履歴（commit）に依存せず、現在の単一 commit tree のみで検証可能なように構成されている。生データ本体は `raw_data/`、実験時コードは `code_snapshots/<SourceSnapshotID>/`、派生表・図・要約は `result/`、非正常データは `failure/`（要約）と `raw_data/unsuccessful/`（生データ本体）に分離して保持する。

Provenance の正式参照は、commit SHA ではなく SourceSnapshotID（実験時コードのスナップショット識別子）である。各生データの内容・生成条件は、RawPath・`raw_data/MANIFEST.tsv`・`raw_data/SHA256SUMS`・`code_snapshots/<SourceSnapshotID>/` から判断でき、各実験群の PBS job ID は `SOURCE.md` および `MANIFEST` に記録される。図表は canonical raw を入力とし `scripts/` の再生成スクリプトで再生成可能であり、主要値は元 TSV から再生成して変更しない（`result/TABLES_AND_FIGURES.md`）。

失敗結果は削除せず分類している。OOM・意図的早期打切り・比較不一致 fail-fast などの非正常データは `failure/` に分類・保持され、その生データは `raw_data/unsuccessful/{oom,failed,early_terminated}/` に内容不変で保持される。SHA256 は `sha256sum -c SHA256SUMS` で検証可能である。

`result/` 全体は単一 checkpoint に対応しない。主実験は SourceSnapshotID `phase_def_block_20260710`、legacy baseline と UM 容量 feasibility は `oldtree_f05ec52_20260512`、小規模正確性は `small_correctness_20260712`、メモリ経路正確性は canonical `memory_correctness_20260712` と診断 `memory_diagnostic_20260713` で測定した。実験群と SourceSnapshotID の完全な対応表は再現性資料（Appendix F）に置く。図表内の文字（軸名、凡例、列名、caption）は英語で統一する。

## 5.12 Scope and Methodological Limitations

各節で述べた方法上の制約を、以下に要約する。詳細な妥当性への影響は Chapter 10 の Threats to Validity で論じる。

- 対象範囲: 性能の中心主張は 4 グラフに限定され、メモリ容量拡張性は 325557_3216152 の 1 グラフ、独立参照との全ベクトル正確性検証は小規模 3 グラフに限定される（他は cross-implementation の max_bc_only）。いずれも他グラフ・他 GPU へ一般化しない。
- 比較対象: PathMerge は第三者実装かつ external comparator であって ground truth ではなく、PathMerge/Galliot 一般や原著者の公式実装へ一般化しない。現行 block による全グラフ統一の 7 実装比較は行っていない。
- 設定の非対称性: GPU_Opt は全グラフ固定 b512、PathMerge はグラフごとに調整しており、両者のバッチ設定は非対称である。
- 未解決事項: メモリ経路の stress 全ベクトル正確性および PathMerge との cross-implementation 一致は未解決で、canonical のメモリ経路正確性は formal status `CORE_FAIL` として保存され、本研究はこれを隠さない。UM oversubscription の経路証拠は migration byte 量の直接計測ではなく、profiling は部分トレースである。
- legacy 依存: PathMerge の roadNet-PA/TX 分母と UM 容量 feasibility は、測定 checkpoint（`oldtree_f05ec52_20260512`）と制約（時間値非採用、境界の再実測なし）を明示した上で用いる。
- 実行環境記録の限界: 保存された実験文書と投入スクリプトの間で queue 名の記録が一致せず、保存済みジョブログから実際の queue 名を独立に確定できない。このため queue 名は本評価の統制変数として扱わず、観測された性能差の要因としても解釈しない。group・GPU・CPU・メモリなどの資源情報は正式記録から確認できる範囲で記述する。

以上の制約の下で、本研究の結論は評価した環境・グラフ・条件の範囲に限定される。次章以降では、この方法に基づく性能（Chapter 6）、要因分析（Chapter 7）、メモリ容量拡張性（Chapter 8）、正確性と数値的挙動（Chapter 9）の結果を示す。
