# Appendix A Complete Experimental Parameters

本付録は、本研究の各実験系列を再構成するために必要な実行パラメータを一覧する。収録対象は、全バッチサイズ、環境変数、PBS 資源記録、checkpoint、計時範囲、正確性許容値である。本付録は実験結果の再解釈を目的とせず、Chapter 5 で規定した方法と Chapter 6 から Chapter 9 で報告した結果に対応する実行条件のみを記述する。runtime 値や speedup を本付録で再掲することはせず、条件の記述に必要な範囲に限る。掃引の全 trial 値は Appendix B、ablation の全構成値は Appendix C、正確性の詳細指標は Appendix D に置く。

すべての記載は保存された正式資料に基づく。実験時コードと現行スクリプトが異なる場合は、実験時 snapshot（`code_snapshots/<SourceSnapshotID>/`）を実行条件の正本とする。値が保存記録から確認できない場合は `Not recorded`、当該系列に適用されない場合は `N/A`、記録は存在するが独立に確定できない場合は `Not independently verifiable` と記す。これら 3 者を区別し、不明値を推定して埋めることはしない。

## A.1 Parameter Interpretation

本付録で用いる用語を Table A.1 に定義する。

**Table A.1: Definitions of the experimental parameters recorded in this appendix.**

| Term | Definition | Recorded in |
|---|---|---|
| Requested Batch | 実行時に要求したバッチサイズ。環境変数または既定値で指定する。 | Script arguments, run logs |
| Effective Batch | 実装がメモリ予算判定の後に実際に採用したバッチサイズ。要求値が予算を超える場合は縮小（clamp）される。 | Run logs, `implementation_manifest.tsv` |
| `SUB_BATCH` | 同時に resident となる source sub-batch の大きさ。GPU_Opt 系のログ列。 | Run logs (`[Mem] SUB_BATCH=`) |
| `num_subs` | Effective Batch を処理するための sub-batch 反復回数。 | Run logs (`num_subs=`) |
| $NS_{\mathrm{eff}}$ | 同時に有効な stream buffer 数。保存記録の列名は `EffectiveNS` である。 | `implementation_manifest.tsv` (`EffectiveNS`) |
| $M_{\mathrm{source}}$ | 1 source 当たりの状態量 [bytes]。保存記録の列名は `PerSourceStateBytes` である。 | `implementation_manifest.tsv` (`PerSourceStateBytes`) |
| Trials | 当該構成で記録した本試行数 $N_{\mathrm{trials}}$。warmup を含まない。 | `SOURCE.md`, result TSV row counts |
| Warmup | 本試行に含めない事前実行の有無と方式。 | `SOURCE.md`, experiment-time scripts, raw logs |
| Aggregation | 主値の集計方法。本研究では median を主値とする。 | `SOURCE.md`, Chapter 5 |
| Timing Scope | `Time_sec` として記録する計測区間。A.11 節で定義する。 | `src/core/runner.cpp` |
| Checkpoint | 実験時コードの識別子。SourceSnapshotID または commit SHA。 | `code_snapshots/`, provenance TSV |
| PBS Job ID | 当該実行の PBS ジョブ識別子。 | `SOURCE.md`, `result/MANIFEST.md` |
| Failure Status | 実行が正常終了しなかった場合の分類。runtime 値を持たない。 | `feasibility_boundary.tsv`, `oom_evidence.tsv` |

バッチに関する解釈上の規約を明記する。バッチは source vertex の grouping であり、graph partition ではない。外側の batch loop が全 source を処理するため、バッチ分割によって BC を近似することも source を省略することもない。GPU_Opt_Pure_Chunked の sub-batch も同様に source の grouping であり、graph edge set の分割ではない。

容量に関しても 2 つの量を区別する。graph file size は on-disk の静的容量であり、batch-dependent working set は `EffectiveBatch`（Chunked では `SUB_BATCH`）と $M_{\mathrm{source}}$ の積に $NS_{\mathrm{eff}}$ を乗じた code-derived allocation estimate である。両者は別概念であり、容量境界（A.7 節）は後者によって決まる。

記号については、保存記録の列名 `EffectiveNS` を論文記号 $NS_{\mathrm{eff}}$、列名 `PerSourceStateBytes` を論文記号 $M_{\mathrm{source}}$ と表す。本付録の表では論文記号を用い、必要な箇所で元の列名を併記する。

## A.2 Hardware and Software Environment

実験環境を Table A.2 に示す。値は `result/environment/environment.md`、`result/MANIFEST.md`、`result/tables/thesis/T6_experimental_environment.tsv` の記録に限り、公称仕様による補完は行わない。

**Table A.2: Hardware and software environment.**

| Component | Specification | Basis |
|---|---|---|
| System | Miyabi-G supercomputer, GPU compute node | Environment record |
| GPU | NVIDIA GH200 Grace Hopper Superchip (sm_90) | Environment record |
| CPU memory | Grace LPDDR5X, coupled to HBM3 via NVLink-C2C (900 GB/s coherent) | Environment record |
| On-package GPU memory (HBM3), nominal | 96 GB | NVIDIA specification |
| On-package GPU memory (HBM3), recorded | 97,871 MiB (approx. 95.6 GiB; approx. 102.6 decimal GB) | Environment record |
| GPU memory reported by runtime query at run start | total approx. 102.0 GB; free (`free_before`) approx. 101.4 GB (decimal GB) | Saved run logs |
| Host physical memory | Not recorded | — |
| Host-memory resource limit (memory-path experiments) | Host-memory-limited 100 GiB configuration | Environment record |
| NVIDIA Driver | 595.58.03 | Environment record |
| CUDA Toolkit (nvcc) | release 13.0, V13.0.48 | Environment record |
| Host C++ Compiler | g++ (GCC) 11.4.1 | Environment record |
| CMake | 4.3.4 | Environment record |
| Nsight Systems (nsys) | 2025.5.1.121 | Environment record |
| Scheduler | PBS batch system (Miyabi-G) | Environment record |
| Group | `gj17` | Environment record, PBS directives |
| Queue | Not independently verifiable from retained job logs | See A.3 |
| Graph format | Undirected, unweighted three-line text CSR | `data/README.md`, `Graph::readGraph()` |

GPU メモリの記載では 4 つの量を分離している。第 1 は公称 HBM3 容量 96 GB、第 2 は実行環境に記録されたデバイスメモリ 97,871 MiB、第 3 は runner 自身の runtime query が実行開始時に報告した total 約 102.0 GB と `free_before` 約 101.4 GB、第 4 は memory-path 実験に課された host-memory-limited 100 GiB configuration である。第 1 から第 3 は同一のオンパッケージ HBM3 を異なる単位系・取得方法で示したものであり、別のメモリ領域や階層ではない。第 4 はホスト側の資源制約であり、HBM3 容量とは独立の量である。100 GiB が queue 名、submission resource limit、node configuration のいずれであるかは保存記録から断定できないため、host-memory-limited configuration としてのみ記録する。

実装 checkpoint は A.10 節に一括して示す。`result/` 全体は単一 checkpoint に対応しない。

## A.3 PBS Resource Records

各実験系列の PBS 資源指定を Table A.3 に示す。値は投入スクリプト（実験時 snapshot が存在する系列はその snapshot）の `#PBS` directive から取得した。

**Table A.3: PBS resource records per experiment series. Queue values are directive records, not confirmed queue usage.**

| Experiment Series | PBS Job ID | Queue Directive | Select | CPUs | GPUs | Host Memory Limit | Walltime | Evidence |
|---|---|---|---:|---:|---|---|---|---|
| Main performance | 2356120, 2357334, 2357335, 2357336, 2357337 | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 12:00:00 | `code_snapshots/phase_def_block_20260710/scripts/run_benchmark_targeted.sh` |
| PathMerge batch sweep | 2355000, 2355001, 2359080, 2359081, 2359096, 2359169, 2360072, 2360073, 2361040, 2361041, 2362006 | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 6:00:00 | `code_snapshots/phase_def_block_20260710/scripts/run_pathmerge_sweep.sh` |
| Kernel selection | 2354329, 2354330 | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 6:00:00 | `code_snapshots/phase_def_block_20260710/scripts/run_kernel_selection.sh` |
| Ablation (synthetic, email) | 2354994, 2354999 | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 6:00:00 | `code_snapshots/phase_def_block_20260710/scripts/run_ablation.sh` |
| Small full-vector correctness | 2367583.opbs | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 2:00:00 | `code_snapshots/small_correctness_20260712/scripts/run_small_correctness.sh` |
| Corrected 325557 validation (Series A/B) | 2404743.opbs | `regular-g` | 1 | 72 | Not specified in directive | Host-memory-limited 100 GiB configuration | 24:00:00 | `scripts/run_corrected_325557_validation.sh` |
| Corrected 325557 ablation (Series C) | 2406254.opbs | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 6:00:00 | `scripts/run_corrected_325557_ablation.sh` |
| Profiling (nsys, bandwidth) | 2359175 | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 2:00:00 | `code_snapshots/phase_def_block_20260710/scripts/run_profiling.sh` |
| Legacy memory scalability | Not recorded | `regular-g` | 1 | 72 | Not specified in directive | Not recorded | 24:00:00 | `code_snapshots/oldtree_f05ec52_20260512/scripts/run_um_oversubscribe*.sh` |
| Legacy memory-path correctness (historical) | 2368587.opbs, 2368269.opbs, 2368398.opbs, 2369632.opbs | `regular-g` | 1 | 72 | Not specified in directive | Host-memory-limited 100 GiB configuration | 6:00:00 | `code_snapshots/memory_correctness_20260712/scripts/run_memory_correctness.sh` |

Queue 欄は投入スクリプトの `#PBS -q` directive の記録である。保存されたジョブログからは実際に使用された queue 名を独立に確認できないため、directive の存在をもって当該 queue での実行を断定しない（Chapter 5、5.12 節）。GPU 数は `select` 指定に含まれておらず、directive からは確定できないため `Not specified in directive` とする。環境記録は実行ノードの GPU が NVIDIA GH200 であることを示すが、これは directive による資源要求の記録ではない。

Host Memory Limit 欄の `Host-memory-limited 100 GiB configuration` は、memory-path 系列の保存文書に記録された資源条件である。他系列については同種の記録がないため `Not recorded` とする。legacy memory scalability 系列は旧ツリーでの実行であり、PBS job ID が当時のログに個別記録されていないため `Not recorded` である。

なお、本環境ではシェル初期化時に `.zshenv` からの `.cargo/env` 読み込み警告が出力されるが、これは runner の実行・計測・結果に影響しない環境固有の初期化メッセージであり、実験の失敗記録として扱わない。

## A.4 Main Performance Parameters

RQ1 の主性能比較（Chapter 6）の実行パラメータを Table A.4 に示す。GPU_Opt は全 4 グラフで共通の固定バッチであり、グラフごとの最速バッチ探索は行っていない。PathMerge tuned はグラフごとに調整されたバッチであり、両者の設定は非対称である。

**Table A.4: Execution parameters of the main performance comparison.**

| Graph | Implementation | Requested Batch | Effective Batch | `SUB_BATCH` | `num_subs` | $NS_{\mathrm{eff}}$ | Trials | Warmup | Aggregation | Checkpoint | Timing Scope |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| email-EuAll | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 5 | None | Median | `phase_def_block_20260710` | Implementation function |
| roadNet-PA | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 3 | None | Median | `phase_def_block_20260710` | Implementation function |
| roadNet-TX | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 3 | None | Median | `phase_def_block_20260710` | Implementation function |
| roadNet-CA | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 3 | None | Median | `phase_def_block_20260710` | Implementation function |
| email-EuAll | PathMerge (tuned) | 2048 | 2048 | N/A | N/A | N/A | 3 | None | Median | `phase_def_block_20260710` | Implementation function |
| roadNet-PA | PathMerge (tuned) | 64 | 64 | N/A | N/A | N/A | 3 | Not recorded | Median | `oldtree_f05ec52_20260512` | Implementation function |
| roadNet-TX | PathMerge (tuned) | 64 | 64 | N/A | N/A | N/A | 3 | Not recorded | Median | `oldtree_f05ec52_20260512` | Implementation function |
| roadNet-CA | PathMerge (tuned) | 32 | 32 | N/A | N/A | N/A | 3 | None | Median | `phase_def_block_20260710` | Implementation function |

GPU_Opt の設定は 1 CUDA ストリーム当たり 512 source の固定バッチであり、dual-stream 実行により $NS_{\mathrm{eff}}=2$ である。したがって 1 回の batch 反復で同時に処理する source 数は $2\times512$ に相当する。4 グラフすべてで in-capacity であり、要求バッチと実効バッチは一致し、clamp は発生していない。

PathMerge は int2 frontier と per-source 配列による実装であり、sub-batch 分割の機構を持たないため `SUB_BATCH`、`num_subs`、$NS_{\mathrm{eff}}$ はいずれも適用されない。roadNet-PA/TX の分母は、掃引によって最適バッチが既定と同一の b64 であることを確認した上で、同一 b64 設定の legacy 実測値を採用したものである。このため当該 2 グラフの PathMerge 側 checkpoint は `oldtree_f05ec52_20260512` であり、GPU_Opt 側の `phase_def_block_20260710` とは異なる。legacy 系列には明示的な warmup 記録がないため `Not recorded` とする。

環境変数としては、GPU_Opt 側で `BC_BATCH_OVERRIDE`、PathMerge 側で `PATHMERGE_BC_BATCH_SIZE` を用いる。投入時のスクリプト変数は `GRAPHS_STR`、`IMPLS_STR`（既定 `gpu_opt gpu_opt_pure gpu_opt_pure_chunked`）、`TRIALS`、`TIMEOUT_SEC`（既定 21600）、`SKIP_BUILD` である。

## A.5 PathMerge Tuning Parameters

PathMerge の tuning 手続き（Chapter 5、5.7 節）の設定を Table A.5 に示す。本節は掃引の「設定」を対象とし、各 trial の runtime 値は Appendix B に収録する。

**Table A.5: PathMerge batch-sweep configuration per graph.**

| Graph | Requested Batch Candidates | Trials per Batch | Screening Job | Confirmation Job | Recorded Clamp | Adopted Tuned Batch | Checkpoint |
|---|---|---|---|---|---|---|---|
| roadNet-PA | 8, 16, 32, 64, 128, 256, 512 | 1 (b8/b16/b32), 4 (b64), 3 (b128/b256/b512) | Small-batch screening | Job 2355001 | None recorded | 64 | `phase_def_block_20260710` (sweep) |
| roadNet-TX | 32, 64, 128 | 3 (b32, b64), 1 (b128) | Job 2360072 | Job 2361040 | None recorded | 64 | `phase_def_block_20260710` (sweep) |
| roadNet-CA | 16, 32, 64, 128 | 1 (b16), 3 (b32, b64), 1 (b128) | Job 2360073 | Job 2362006, Job 2361041 (b16) | None recorded | 32 | `phase_def_block_20260710` |
| email-EuAll | 8, 16, 64, 256, 512, 1024, 2048, 4096, 8192 | 1 (b8/b16/b64/b256), 3 (b512–b8192) | Small-batch screening | Job 2355000 series | Requested 8192 to effective 7393 | 2048 | `phase_def_block_20260710` |
| 325557_3216152 | 32, 64, 256, 512, 1024, 2048, 4096, 8192 | 1–4 (batch dependent) | Job 2355000 | Job 2359081 (b4096, b8192) | Requested 8192 to effective 6018 | 4096 | `phase_def_block_20260710` |

tuned batch の選択規則は次のとおりである。各グラフについて候補バッチごとの median 実行時間を求め、最小の median を与えるバッチを掃引実測の最良とする。最終的な分母は、掃引最良と既定 b64 のうち速い方を採用する（`scripts/merge_final_tables.py`）。掃引は warmup を行わず、集計は median である。

roadNet-PA および roadNet-TX については、最終採用値と掃引確認値の関係を明記する。両グラフの掃引では最適バッチが既定と同一の b64 であることを確認したが、最終表の分母には同一 b64 設定の legacy 実測値を採用した。すなわち、掃引の確認測定と最終採用値は同一バッチ設定に対する別々の測定であり、欠損でも矛盾でもない。legacy 実測値の方が掃引確認値よりわずかに速いため、この採用は当該 2 グラフの speedup を過小方向に見積もる。roadNet-CA と email-EuAll では、掃引実測の最良値がそのまま tuned として採用されている。

clamp は 2 件記録されている。email-EuAll の要求 b8192 は HBM3 予算超過により実効 7393 へ、325557_3216152 の要求 b8192 は実効 6018 へ縮小された。いずれも保存ログの警告行に基づく記録である。その他の候補バッチでは clamp の記録はない。325557_3216152 は RQ1 の主性能比較の対象ではなく、その掃引結果は Tier B の external comparator 設定（b4096）の根拠として用いる。

関連する環境変数は `PATHMERGE_BC_BATCH_SIZE`（runner のバッチ指定）、および投入スクリプト変数 `BATCH_LIST`（既定 `1,2,4,8,16,32,64,128,256`）、`TRIALS`（既定 1）、`GRAPHS_STR`、`TIMEOUT_SEC`（既定 21600）である。

## A.6 Ablation and Kernel-Selection Parameters

RQ2 の要因分析は ablation と kernel selection の 2 系列からなる。両者は目的・対象グラフ・checkpoint が異なるため分離して記述する。

### A.6.1 Ablation

3 因子は H = Hybrid BFS、W = Warp-Cooperative Accumulation、A = Dual-Stream Execution であり、compile-time テンプレートにより $\mathrm{H}\{0,1\}\times\mathrm{W}\{0,1\}\times\mathrm{A}\{0,1\}$ の 8 構成を測定した。設定を Table A.6 に示す。

**Table A.6: Ablation experiment parameters by measurement series.**

| Series | Graphs | Configurations | Requested Batch | Effective Batch | `SUB_BATCH` | Trials per Configuration | Warmup | Aggregation | PBS Job ID | Checkpoint |
|---|---|---:|---:|---:|---|---:|---|---|---|---|
| Synthetic (three graphs) | benchmark_7000_41459, benchmark_11023_62184, 56438_300801 | 8 | 512 | 512 | Not recorded | 5 | One global untimed H1W1A1 per invocation; excluded from formal rows | Median per configuration | 2354994 | `phase_def_block_20260710` |
| Corrected 325557 | 325557_3216152_corrected_v1 | 8 | 512 | 512 | Not recorded | 5 | One global untimed H1W1A1 per invocation; excluded from formal rows | Median per configuration | 2406254 | `45352a344aaac463283a647467b790be9b45bfb8` |
| email-EuAll | email-EuAll | 8 | 512 | 512 | Not recorded | 3 | One global untimed H1W1A1 per invocation; excluded from formal rows | Median per configuration | 2354999 | `phase_def_block_20260710` |
| Historical (malformed 325557) | 325557_3216152 | 8 | 512 | 512 | Not recorded | 5 | One global untimed H1W1A1 per invocation | Median per configuration | 2354994 | `phase_def_block_20260710` |

バッチは環境変数で指定していない。`src/proposed/host_ablation.cu` はメモリ予算から算出した値を上限 512 で丸めるため、いずれの系列も in-capacity で b512 となり、ログの `[Ablation H* W* A*] BATCH=512` で確認できる。`SUB_BATCH` と `num_subs` は ablation のログに出力されないため `Not recorded` である。

warmup は 3 系列すべてで同一の方式である。`run_ablation <graph> all` の各 runner invocation の先頭に global・untimed の H1W1A1 実行を 1 回置き、formal TSV 行に含めない。投入スクリプトは graph × trial ごとに runner を 1 回起動するため、この warmup は「PBS job 当たり 1 回」ではなく「runner invocation 当たり 1 回」、すなわち trial 当たり 1 回である。系列別の一次記録は Table A.6a のとおりである。

**Table A.6a: Warmup evidence per ablation series (marker counts from the raw logs, formal rows from the result TSVs).**

| Series | PBS Job ID | Runner Invocations | Warmup Markers | Formal Rows | Rows per Configuration | Warmup in Formal TSV |
|---|---|---:|---:|---:|---:|---|
| Synthetic (four graphs, incl. historical 325557) | 2354994 | 20 (4 graphs × 5 trials) | 20 | 160 | 20 | No |
| email-EuAll | 2354999 | 3 (1 graph × 3 trials) | 3 | 24 | 3 | No |
| Corrected 325557 | 2406254 | 5 (1 graph × 5 trials) | 5 | 40 | 5 | No |

warmup が formal TSV に混入しないことは、行数と出力経路の双方から確認できる。formal 行数は各系列で 8 × invocation 数に厳密に一致し、warmup 分の追加行は存在しない。また runner の warmup は計時・出力を担う `run_brandes` を経由せずに実装関数を直接呼ぶため、stdout の TSV 行を生成しない。warmup marker は raw log 上で各 invocation header の直後、8 構成の本試行の前に出現する。

3 系列の runner コードは同一である。corrected 系列の checkpoint `45352a3` における `experiments/run_ablation.cu` は、`code_snapshots/phase_def_block_20260710/` の同ファイルと byte 単位で一致する。warmup 経路は実行モードに依存せず、環境変数 `BC_ABLATION_WARMUP=0` を指定した場合にのみ抑止されるが、上記 3 系列のいずれでも抑止された記録はなく、marker の存在が実行を裏づける。

主効果の算出方法は次のとおりである。因子 $F$ について、他 2 因子の 4 通りの水準組合せそれぞれで per-configuration median の比 $T(F{=}0)/T(F{=}1)$ を求め、その幾何平均を主効果とする。ばらつきは標本標準偏差（ddof=1）で表し、$N_{\mathrm{trials}}<2$ の場合は `n/a` とする。

合成 4 グラフの集約は mixed-checkpoint である。benchmark_7000_41459、benchmark_11023_62184、56438_300801 は job 2354994、修正版 325557 は job 2406254 / checkpoint `45352a3` の測定であり、同一 checkpoint で 4 グラフを再測定した結果ではない。旧 malformed 325557 の測定値は上書きせず historical result として保持する。

### A.6.2 Kernel Selection

BFS カーネルの直接比較の設定を Table A.7 に示す。これは環境変数によって各カーネルを強制実行した forced 比較であり、自動選択則の評価ではない。

**Table A.7: Kernel-selection (forced shared/block) experiment parameters.**

| Graph | Forced Kernels | Requested Batch | Effective Batch | `SUB_BATCH` | `num_subs` | Trials per Kernel | Warmup | Aggregation | Correctness Level | PBS Job ID | Checkpoint |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| roadNet-PA | shared, block | 512 | 512 | 512 | 1 | 3 | None | Median with sample SD | `max_bc_only` | 2354329 | `phase_def_block_20260710` |
| roadNet-TX | shared, block | 512 | 512 | 512 | 1 | 3 | None | Median with sample SD | `max_bc_only` | 2354330 | `phase_def_block_20260710` |

カーネルの強制切替は環境変数 `BC_FORCE_BFS_KERNEL=shared|block` による。バッチは投入スクリプトが `BC_BATCH_OVERRIDE=512` を既定として設定する。集計スクリプト `scripts/summarize_kernel_selection.py` は forced shared/block の実測値（median、標本標準偏差、$N_{\mathrm{trials}}$、速い側、速度向上、Max BC 一致）のみを出力し、選択則の当否判定を含まない。

旧実装には平均次数に基づく自動選択則（`avg_deg < 5` の場合に shared）が存在したが、現行方式では使用していない。本付録および本論文は、この旧選択則を現行の方式として再導入しない。kernel selection の対象グラフは roadNet-PA と roadNet-TX の 2 件であり、他グラフへ一般化しない。

## A.7 Memory-Scalability Parameters

RQ3 の容量評価は、current の corrected 325557 系列と legacy 系列を厳密に分離する。current の結論は corrected 系列（job 2404743）のみから導く。

### A.7.1 Corrected 325557 Targeted Boundary Series

**Table A.8: Corrected 325557 targeted feasibility parameters (job 2404743). Each condition was executed once.**

| Implementation | Requested Batch | Effective Batch | `SUB_BATCH` | `num_subs` | $NS_{\mathrm{eff}}$ | Trials | Warmup | Memory Mode | Status | Failure Class |
|---|---:|---:|---|---:|---:|---:|---|---|---|---|
| GPU_Opt_Pure | 4096 | 4096 | N/A | N/A | 2 | 1 | None | `explicit_device_memory` | Success | None |
| GPU_Opt_Pure | 8192 | 8192 | N/A | N/A | 2 | 1 | None | `explicit_device_memory` | Failure | CUDA device-memory OOM, exit 1 |
| GPU_Opt | 10240 | 10240 | 6596 | 2 | 1 | 1 | None | `managed_unified_memory` | Success | None |
| GPU_Opt | 12288 | 12288 | 6596 | 2 | 1 | 1 | None | `managed_unified_memory` | Failure | Cgroup host-memory OOM kill, exit 137 |
| GPU_Opt_Pure_Chunked | 16384 | 16384 | 6596 | 3 | 1 | 1 | None | `explicit_device_memory_chunked` | Success | None |

$M_{\mathrm{source}}=10{,}418{,}856$ bytes（$D_{est}=256$）である。code-derived allocation estimate は $NS_{\mathrm{eff}}\times\mathrm{EffectiveBatch}\times M_{\mathrm{source}}$、Chunked では `SUB_BATCH` を用いた $NS_{\mathrm{eff}}\times \mathrm{SUB\_BATCH}\times M_{\mathrm{source}}$ である。これらは配列寸法から導いた estimate であり、process RSS、physical HBM residency、migration bytes の実測値ではない。

status 規則は次のとおりである。OOM および kill は 0 秒として扱わず、runtime 値を `N/A` とする。OOM 判定は strong evidence（`cuda_oom`、`host_alloc_failure`、`kernel_oom_kill`）に限定し、助言的警告や語の言及を証拠としない。Pure b8192 は `host_pure.cu:144` の `cudaMalloc` が `out of memory` を返した CUDA device-memory OOM であり、UM b12288 は `oom_evidence=none`・exit 137 の cgroup host-memory OOM kill である。両者は異なる failure class であり、混同しない。

Chunked b16384 の `SUB_BATCH=6596` は HBM3 予算のみで決まる値ではない。`host_chunked.cu` は HBM budget 由来の上限と、index overflow を防ぐ $\lfloor\mathrm{INT\_MAX}/n\rfloor=6596$ の小さい方を採る。修正版 325557 では後者が binding constraint であった。

試験範囲の解釈について明記する。Chunked は試験した上限である b16384 まで成功した。これは「上限」でも「無制限」でもなく、試験上限まで成功したという記述である。UM もまた無制限ではなく、b12288 では host-memory 側の制約により停止した。UM を採用する理由は入力グラフファイルが 96 GB を超えるためではない。入力ファイルは 45,348,105 bytes（約 43.25 MiB）であり、容量問題は batch-dependent working set の配置と容量評価に起因する。

### A.7.2 Legacy Memory-Scalability Series

**Table A.9: Legacy oversubscription sweep parameters (historical, malformed 325557 input).**

| Implementation | Requested Batches Tested | Trials | Warmup | Highest Successful Tested Batch | Batches Recorded as `OOM_OR_FAIL` | PBS Job ID | Checkpoint |
|---|---|---:|---|---:|---|---|---|
| GPU_Opt | 512, 1024, 2048, 4096, 8192, 10240, 12288 | 5 per batch (12288: 1) | None | 10240 | 12288 | Not recorded | `oldtree_f05ec52_20260512` |
| GPU_Opt_Pure | 512, 1024, 2048, 4096, 8192, 10240, 12288, 16384 | 5 per batch | Not recorded | 4096 | 8192, 10240, 12288, 16384 | Not recorded | `oldtree_f05ec52_20260512` |
| GPU_Opt_Pure_Chunked | 512, 1024, 2048, 4096, 8192, 10240, 12288, 16384 | 5 per batch | None | 16384 | None | Not recorded | `oldtree_f05ec52_20260512` |

legacy 系列は旧 malformed 入力 `325557_3216152` 上の測定であり、current RQ3 の境界には用いない。warmup は方式別に記録が異なる。GPU_Opt と GPU_Opt_Pure_Chunked は実験時 snapshot のスクリプトに warmup ループが存在せず、raw log の実行数と TSV の試行行数が一致することから warmup なしと確認できる。GPU_Opt_Pure は raw log に trial header がなく、生成ドライバが snapshot に収録されていないため `not_recorded` とする。

`OOM_OR_FAIL` は legacy archive の historical label であり、current の failure class 分類（CUDA device-memory OOM と cgroup host-memory OOM kill の区別）とは異なる粒度である。特に GPU_Opt b12288 の `OOM_OR_FAIL`（$N_{\mathrm{trials}}=1$）は、CUDA OOM・host OOM kill・scheduler OOM のいずれについても独立記録がないため、原因を断定していない。legacy 系列の runtime 値は current の block 実装の性能値として使用しない。

### A.7.3 Legacy Memory-Path Correctness Series

memory-path correctness の legacy 系列（job 2368587 ほか）は旧 malformed 入力上の測定であり、`CORE_FAIL` を含む結果を historical evidence として保持する。使用したバッチは GPU_Opt b1024/b9792、GPU_Opt_Pure b1024、GPU_Opt_Pure_Chunked b1024/b16384、PathMerge b4096 であり、各構成 1 試行・warmup なし、checkpoint は `memory_correctness_20260712` である。これらは current の結論には用いない。

## A.8 Correctness-Validation Parameters

RQ4 の正確性検証は 2 つの evidence tier に分かれる。両者は参照の独立性が異なるため分離して記述する。

判定は混合絶対・相対許容による。reference $a_i$ と candidate $b_i$ の各要素について次を要求する。

$$
|a_i-b_i|\le\mathrm{abs\_tol}+\mathrm{rel\_tol}\max(|a_i|,|b_i|)
$$

正式な許容値は `abs_tol = 1e-3`、`rel_tol = 1e-6` である。許容値を事後に変更して判定を PASS に変えることはしない。BC 値が大きい領域では絶対許容単独の超過が生じ得るため、これは WARN として分離し、単独の失敗判定としない。

### A.8.1 Tier A: Independent CPU Reference

**Table A.10: Tier A validation parameters (independent Sequential CPU reference).**

| Graph | Vector Length | Reference | Candidate | Requested Batch | Effective Batch | `SUB_BATCH` | `num_subs` | $NS_{\mathrm{eff}}$ | Trials | Warmup | `abs_tol` | `rel_tol` | PBS Job ID | Checkpoint |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|
| benchmark_7000_41459 | 7000 | Sequential | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 1 | None | 1e-3 | 1e-6 | 2367583.opbs | `small_correctness_20260712` |
| benchmark_11023_62184 | 11023 | Sequential | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 1 | None | 1e-3 | 1e-6 | 2367583.opbs | `small_correctness_20260712` |
| chain_200 | 200 | Sequential | GPU_Opt | 512 | 512 | 512 | 1 | 2 | 1 | None | 1e-3 | 1e-6 | 2367583.opbs | `small_correctness_20260712` |

Tier A の比較範囲は、Sequential CPU 実装を独立参照とする全 BC 要素の比較である。scope は小規模 3 グラフに限定され、email-EuAll、roadNet 系、GPU_Opt_Pure、GPU_Opt_Pure_Chunked、UM oversubscription 固有経路は含まない。各構成 $N_{\mathrm{trials}}=1$・warmup なしであり、これらの実行の時間値は性能主張に用いない。

### A.8.2 Tier B: Cross-Implementation Consistency

**Table A.11: Tier B validation parameters on the corrected 325557 graph (job 2404743, checkpoint 45352a3).**

| Vector | Implementation | Requested Batch | Effective Batch | `SUB_BATCH` | `num_subs` | $NS_{\mathrm{eff}}$ | Vector Length | Batch Environment Variable |
|---|---|---:|---:|---|---:|---|---:|---|
| gpu_opt_b1024 | GPU_Opt | 1024 | 1024 | 1024 | 1 | 2 | 325557 | `BC_BATCH_OVERRIDE` |
| gpu_opt_b9792 | GPU_Opt | 9792 | 9792 | 6596 | 2 | 1 | 325557 | `BC_BATCH_OVERRIDE` |
| gpu_opt_pure_b1024 | GPU_Opt_Pure | 1024 | 1024 | N/A | N/A | 2 | 325557 | `BC_BATCH_OVERRIDE` |
| gpu_opt_pure_chunked_b1024 | GPU_Opt_Pure_Chunked | 1024 | 1024 | 1024 | 1 | 2 | 325557 | `BC_BATCH_OVERRIDE` |
| gpu_opt_pure_chunked_b16384 | GPU_Opt_Pure_Chunked | 16384 | 16384 | 6596 | 3 | 1 | 325557 | `BC_BATCH_OVERRIDE` |
| pathmerge_b4096 | PathMerge | 4096 | 4096 | N/A | N/A | N/A | 325557 | `PATHMERGE_BC_BATCH_SIZE` |

比較は上記 6 vector から構成される 10 組であり、comparison class は `same_impl_diff_batch` が 2 組、`same_batch_diff_path` が 3 組、`pathmerge_cross` が 5 組である。各構成 $N_{\mathrm{trials}}=1$・warmup なし、許容値は Tier A と同一である。対象グラフは `325557_3216152_corrected_v1`（SHA256 `8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22`、$n=325{,}557$、$m=3{,}216{,}152$）である。

Tier B は cross-implementation consistency であり、independent ground truth ではない。PathMerge は評価した第三者実装の external comparator であって参照真値ではない。

### A.8.3 Recorded Validation Checks

**Table A.12: Validation checks recorded for every compared vector.**

| Check | Recorded Quantity | Tier A | Tier B |
|---|---|---|---|
| Vector length | Number of BC elements compared against $n$ | Recorded | Recorded |
| Missing index | Count of indices absent from the vector | Recorded | Recorded |
| Duplicate index | Count of repeated indices | Recorded | Recorded |
| Out-of-range index | Count of indices outside $[0,n)$ | Recorded | Recorded |
| NaN / Inf | Count of non-finite values | Recorded | Recorded |
| Mismatched elements | Count of elements violating the mixed tolerance | Recorded | Recorded |
| Byte identity | SHA256 equality of the compared vectors | Recorded | Recorded |

mixed-tolerance の PASS と byte-identical は別の判定である。本研究では全 13 比較で `ToleranceResult=PASS` かつ `ByteIdentical=No` であり、混合許容内の数値的一致は bitwise identity を意味しない。

旧 malformed 入力上の canonical job 2368587 における `CORE_FAIL` は historical invalid-input evidence として分離保持し、修正版入力に基づく current conclusion へ混入させない。逆に、修正版の PASS を旧入力の判定へ遡及適用することもしない。

## A.9 Profiling Parameters

profiling 系列の設定を Table A.13 に示す。保存されている範囲のみを記載する。

**Table A.13: Profiling capture parameters (PBS job 2359175).**

| Capture | Profiler | Traced Binary / Implementation | Graph | Batch | `SUB_BATCH` | `num_subs` | $NS_{\mathrm{eff}}$ | Duration / Scope | Artifact Type | Checkpoint |
|---|---|---|---|---:|---|---|---|---|---|---|
| `ablation_H1W1A0` | Nsight Systems 2025.5.1.121 | `run_ablation`, configuration H1W1A0 | 56438_300801 | 512 | Not recorded | Not recorded | 1 | Full process duration; includes the untimed H1W1A1 warmup in the same process | `.nsys-rep`, `.stats.txt`, `.console.log` | `phase_def_block_20260710` |
| `ablation_H1W1A1` | Nsight Systems 2025.5.1.121 | `run_ablation`, configuration H1W1A1 | 56438_300801 | 512 | Not recorded | Not recorded | 2 | Full process duration | `.nsys-rep`, `.stats.txt`, `.console.log` | `phase_def_block_20260710` |
| `um_prefetch_gpu_opt` | Nsight Systems 2025.5.1.121 | `run_benchmark gpu_opt` | 325557_3216152 (pre-repair input) | 512 | 512 | 1 | 2 | Partial trace, `--duration=25` (25 seconds) | `.nsys-rep`, `.stats.txt`, `.console.log` | `phase_def_block_20260710` |
| `bandwidth` | `bandwidth_benchmark` | Device bandwidth measurement | N/A | N/A | N/A | N/A | N/A | Single measurement run | `bandwidth.log` | `phase_def_block_20260710` |

`um_prefetch_gpu_opt` は `--duration=25` による 25 秒の部分トレースであり、実行全体の profiling ではない。したがって、そこに記録された migration 量や fault 数は部分値であって全実行総量ではない。`ablation_H1W1A0` と `ablation_H1W1A1` は duration 指定のない full-duration トレースであるが、`ablation_H1W1A0` の trace scope には同一 process 冒頭の untimed H1W1A1 warmup が含まれるため、本測定構成だけを分離した値ではない。

トレース対象グラフについては、保存 metadata 上の不整合は解消済みである。ablation 系トレースの対象は `56438_300801`、UM prefetch トレースの対象は `325557_3216152` であり、この対応は `SOURCE.md`、`result/MANIFEST.md`、raw の `pbs_stdout.log` および `console.log` で一致する。ただし UM prefetch トレースの対象は修復前の旧入力であり、修正版 `325557_3216152_corrected_v1` ではない。この点は解釈上の制約として明示する。

`.stats.txt` は保存された `.nsys-rep` から `nsys stats` により再生成したものであり、元の `.nsys-rep` は不変である。

## A.10 Checkpoints and Provenance

各実験系列と実験時コードの対応を Table A.14 に示す。provenance の正式参照は commit SHA ではなく SourceSnapshotID であるが、corrected 325557 系列は `code_snapshots/` に対応 snapshot を持たず commit SHA で識別されるため、両者を区別して記載する。

**Table A.14: Experiment series, code snapshots, checkpoints, and canonical artifacts.**

| Experiment Series | Code Snapshot | Checkpoint | PBS Job IDs | Canonical Result | Canonical Raw Data |
|---|---|---|---|---|---|
| Main performance | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2356120, 2357334–2357337 | `result/main_performance/proposed_variants/` | `raw_data/main_performance/proposed_variants/` |
| PathMerge batch sweep | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2355000, 2355001, 2359081, 2360072, 2360073, 2361040, 2361041, 2362006 | `result/tuning/pathmerge/` | `raw_data/tuning/pathmerge/` |
| Kernel selection | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2354329, 2354330 | `result/tuning/kernel_selection/` | `raw_data/tuning/kernel_selection/` |
| Ablation (synthetic three graphs) | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2354994 | `result/ablation/synthetic_2354994/` | `raw_data/ablation/synthetic/` |
| Ablation (email-EuAll) | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2354999 | `result/ablation/email_2354999/` | `raw_data/ablation/email-EuAll/` |
| Ablation (corrected 325557) | No `code_snapshots/` entry | Commit `45352a344aaac463283a647467b790be9b45bfb8` | 2406254.opbs | `result/ablation/corrected_325557/` | `raw_data/corrected_325557/job_2406254/` |
| Correctness Tier A | `code_snapshots/small_correctness_20260712/` | `small_correctness_20260712` | 2367583.opbs | `result/correctness/small_full_vector/` | `raw_data/correctness/small_full_vector/` |
| Correctness Tier B and feasibility | No `code_snapshots/` entry | Commit `45352a344aaac463283a647467b790be9b45bfb8` | 2404743.opbs | `result/correctness/corrected_325557/`, `result/memory_scalability/corrected_325557/` | `raw_data/corrected_325557/job_2404743/` |
| Profiling | `code_snapshots/phase_def_block_20260710/` | `phase_def_block_20260710` | 2359175 | `result/profiling/` | `raw_data/profiling/job_2359175_20260711/` |
| Legacy memory scalability | `code_snapshots/oldtree_f05ec52_20260512/` | `oldtree_f05ec52_20260512` | Not recorded | `result/memory_scalability/` | `raw_data/memory_scalability/325557_3216152/` |
| Legacy memory-path correctness | `code_snapshots/memory_correctness_20260712/` | `memory_correctness_20260712` | 2368587.opbs | `result/correctness/memory_paths/` | `raw_data/unsuccessful/`, `raw_data/correctness/` |
| PathMerge default baseline (roadNet-PA/TX denominator) | `code_snapshots/oldtree_f05ec52_20260512/` | `oldtree_f05ec52_20260512` | Not recorded | `result/main_performance/seven_implementations/legacy_partial/` | `raw_data/main_performance/seven_implementations/legacy_partial/` |

読者向けの表では canonical path をディレクトリ単位で示す。ファイル単位の完全な path、SHA256、生成コマンドは `raw_data/MANIFEST.tsv`、`raw_data/RAW_DATA_INDEX.tsv`、`raw_data/SHA256SUMS`、および `result/CORRECTED_325557_ARTIFACT_PROVENANCE.tsv` に記録される。

現行ブロック、legacy tree、corrected validation の 3 者は checkpoint が異なる。`result/` 全体は単一 checkpoint に対応しない。また、repository の現在の HEAD は実験 checkpoint ではなく、本付録の執筆時点の base commit を実験条件として扱うこともしない。実験条件の正本は上表の checkpoint と、対応する snapshot または commit である。

## A.11 Timing and Aggregation Conventions

計時範囲は `src/core/runner.cpp` の `run_brandes()` が定義する区間である。計測は各実装関数の呼び出し直前に開始し、関数からの復帰直後に終了する。したがって Table A.15 のとおり、グラフファイルの読み込みと CSR 構築、および入力検証は計測区間の外であり、デバイス確保、host-device 転送、カーネル実行、結果回収、実装関数内で行われる同期は計測区間の内である。

**Table A.15: Timing scope of the recorded `Time_sec`.**

| Operation | Included in `Time_sec` |
|---|---|
| Graph file reading and CSR construction | No |
| Input CSR validation | No |
| Device allocation and managed allocation | Yes |
| Host-to-device transfer and prefetch | Yes |
| Kernel execution (BFS and backward phases) | Yes |
| Synchronization inside the implementation function | Yes |
| Device-to-host copy-out of the BC vector | Yes |
| Warmup runs | No |
| Result file writing and reporting | No |

集計の規約は次のとおりである。主値は median である。補助値として mean、標本標準偏差 $s_T$、min、max を扱う。$s_T$ は $N_{\mathrm{trials}}$ 試行に対する不偏推定量（ddof=1）であり、

$$
s_T=\sqrt{\frac{1}{N_{\mathrm{trials}}-1}\sum_{i=1}^{N_{\mathrm{trials}}}\left(t_i-\bar{t}\right)^2}
$$

で定義される。ここで $t_i$ は各試行の実行時間、$\bar{t}$ は標本平均である。試行数の記号は $N_{\mathrm{trials}}$ であり、頂点数 $n=|V|$ とは別の記号を用いる。表・図中の `n=3`、`n=5` は $N_{\mathrm{trials}}$ を示す慣用ラベルである。単一の最速試行を代表値としない。

speedup は median 同士の比として算出し、本研究の主比較では次式に統一する。

$$
\mathrm{Speedup}=\frac{T_{\mathrm{PathMerge,\ tuned}}}{T_{\mathrm{GPU\_Opt}}}
$$

分子・分母はいずれも median 実行時間であり、median と mean を混在させた比は用いない。throughput は全実装で

$$
\mathrm{GTEPS}=\frac{n\cdot m}{T\cdot10^{9}}
$$

に統一し、median 実行時間から算出する。ここで $n$ は頂点数、$m$ は無向辺数である。

warmup は本試行に含めない。warmup を実施した系列（A.6.1 節の ablation 系列）では、warmup 実行を formal TSV 行に記録しない。warmup の記録がない系列については `not_recorded` とし、一律の既定値を仮定して補うことはしない。

失敗の扱いについては、OOM、TIMEOUT、FAIL を 0 秒として集計しない。取得不能な runtime は `N/A` とし、failure class（CUDA device-memory OOM と cgroup host-memory OOM kill）を区別して記録する。この規約により、実行不能条件が高速な性能値として誤って集計されることを防ぐ。正確性検証のみを目的とする実行（各構成 $N_{\mathrm{trials}}=1$、warmup なし）の時間値は、性能主張に用いない。
