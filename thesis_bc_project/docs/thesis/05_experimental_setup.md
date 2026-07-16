# 05 実験方法

不明値は推定せず `not_recorded` とする。数値は `result/environment/environment.md`・
`result/MANIFEST.md`・`docs/graph_stats.md` と一致。

## 5.1 ハードウェア（T-ENV）
| 項目 | 値 |
|:--|:--|
| GPU | NVIDIA GH200（Grace Hopper Superchip, sm_90） |
| 公称 HBM3 | 96 GB |
| 記録されたデバイスメモリ | 97,871 MiB（約95.6 GiB、約102.6 decimal GB；公称96 GBと同一のHBM3） |
| 実行開始時の runtime 照会 | total 約102.0 GB、free (`free_before`) 約101.4 GB（decimal GB；freeは総容量ではなくメモリ予算計算の基準） |
| CPU メモリ | Grace LPDDR5X（NVLink-C2C 結合） |
| 実測帯域 | HBM3 DtoD 1818.6 GB/s、Pinned HtoD 424.1 GB/s、Pinned DtoH 297.6 GB/s、NVLink-C2C Prefetch 177.7 GB/s（`raw_data/profiling/job_2359175_20260711/bandwidth.log`） |
| PBS system | Miyabi-G PBS batch system |
| Group | `gj17` |
| Queue | Not independently verifiable from retained job logs |
| memory-path実験の資源構成 | Host-memory-limited 100 GiB configuration |

公称96 GB、記録値97,871 MiB、runtime照会のtotal約102.0 GBは、同一のオンパッケージHBM3を異なる単位系または取得方法で示したものであり、別個のメモリ階層ではない。実験はMiyabi-G上のPBS batch systemを通じて実行したが、保存された正式文書と投入スクリプトの間でqueue名の記録が一致せず、保存済みジョブログから実際のqueue名を独立に確定できないため、queue名は統制変数として扱わない。

## 5.2 ソフトウェア
| 項目 | 値 |
|:--|:--|
| NVIDIA driver | 595.58.03 |
| CUDA (nvcc) | release 13.0, V13.0.48 |
| CMake | 4.3.4（`~/.local/bin/cmake`） |
| C++ コンパイラ | g++ (GCC) 11.4.1 |
| nsys | 2025.5.1.121 |

## 5.3 checkpoint SHA（実験群ごとに異なる）
`result/` 全体は単一 checkpoint ではない（`result/environment/environment.md`）。

| 実験群 | checkpoint | 備考 |
|:--|:--|:--|
| 提案 block（proposed_variants）・kernel_selection・PathMerge 掃引・correctness(small)・profiling | `phase_def_block_20260710` | 常時 block 化後 |
| ablation（synthetic/email） | `phase_def_block_20260710`（測定 2026-07-10） | build_miyabi から curate |
| legacy baseline（seven_implementations, PathMerge 既定 b64 / 旧 shared 提案） | 旧 tree（oldtree_f05ec52_20260512-era, pre-consolidation） | 旧 mylab/research 由来 |
| UM feasibility（memory_scalability） | `oldtree_f05ec52_20260512`（2026-05-12, 旧 tree） | **時間値非採用**, feasibility のみ |
| memory-path canonical | `memory_correctness_20260712`（job 2368587） | Host-memory-limited 100 GiB configuration |
| memory-path diagnostic | `memory_diagnostic_20260713`（job 2369632） | Host-memory-limited 100 GiB configuration |
| memory-path OOM（failed） | `memory_correctness_oom_20260712`（job 2368269） | UM b10240 OOM |
| memory-path fail-fast（early_terminated） | `memory_correctness_failfast_20260712`（job 2368398） | 比較不一致で打切り |

## 5.4 データセット（T-GRAPH）
すべて無向・非重み CSR。統計は `docs/graph_stats.md`、素性は `result/datasets/graph_catalog.tsv`。

| グラフ | 分類 | n | m | avg_deg | max_deg | CV(次数) | BFS深さ中央 | 用途 |
|:--|:--|--:|--:|--:|--:|--:|--:|:--|
| email-EuAll | 実データ(ハブ有) | 265,009 | 364,481 | 2.75 | 7,636 | 13.93 | 10 | RQ1 主 |
| roadNet-PA | 道路網 | 1,088,092 | 1,541,898 | 2.83 | 9 | 0.36 | 674 | RQ1 主 |
| roadNet-TX | 道路網 | 1,379,917 | 1,921,660 | 2.79 | 12 | 0.36 | 798 | RQ1 主 |
| roadNet-CA | 道路網 | 1,965,206 | 2,766,607 | 2.82 | 12 | 0.35 | 608 | RQ1 主 |
| 325557_3216152 | 合成(高次数) | 325,557 | 3,216,152 | 19.76 | 18,280 | 11.15 | 27 | RQ3/RQ4（UM/memory-path） |
| 56438_300801 | 合成 | 56,438 | 300,801 | 10.66 | 604 | 2.01 | 8 | ablation, 7impl(small) |
| benchmark_7000_41459 | 合成 | 7,000 | 41,459 | 11.85 | 589 | 1.37 | 5 | ablation, 正確性(small) |
| benchmark_11023_62184 | 合成 | 11,023 | 62,184 | 11.28 | 2,109 | 4.60 | 8 | ablation, 正確性(small) |
| chain_200 | 合成(鎖) | 200 | 199 | 1.99 | 2 | 0.05 | 171 | 正確性(small) |
| random | 合成 | 32,212 | 101,805 | 6.32 | 22 | 0.19 | 12 | 7impl(small) |

### 前処理・directed/undirected（`graph_catalog.tsv`）
- email-EuAll：SNAP 由来（原本 directed）を **無向化**して使用（`Symmetrized=yes`）。
- roadNet-PA/TX/CA：SNAP 由来（原本 undirected）を無向で使用（`Symmetrized=no`）。
- 合成グラフ：`tools/gen_graph.py` 生成、無向。325557 は 1-indexed。
- SelfLoop/DupEdge 処理：SNAP グラフは `unknown`（原データ準拠, 推定しない）。
- SHA256 は `graph_catalog.tsv` に記録（同一性確認可能）。

## 5.5 集計規約
- **Aggregation = median（中央値）**。主値は median。mean と混在させない。速度向上は median/median。
- 標準偏差は**標本標準偏差（sample SD, ddof=1）**を併記（[06](06_results_performance.md)）。
- **Warmup**：
  - 新規測定（proposed_variants / kernel_selection / PathMerge 掃引 / correctness, `phase_def_block_20260710`）＝
    **なし**（ベンチスクリプトは全 TRIALS を記録・discard なし）。
  - legacy baseline（旧 tree）＝当時ログ準拠（明示的 warmup 記録なし → `not_recorded`）。
  - UM feasibility sweep（旧 tree, Series A）＝方式別（Gate W4.1 監査）：gpu_opt / gpu_opt_pure_chunked ＝
    **なし**（実験時 snapshot `code_snapshots/oldtree_f05ec52_20260512/scripts/run_um_oversubscribe*.sh` に
    warmup ループ無し・全実行を trial 記録、log/TSV 1:1 で確認）；gpu_opt_pure ＝ `not_recorded`
    （log に trial header 無し・生成ドライバ未保存のため確認不能）。
- **試行数（trials）**：proposed email n=5 / road n=3；PathMerge tuned n=3（掃引最良）；
  ablation synthetic n=5 / email n=3；memory_scalability n=5；memory-path 各構成 **n=1**；
  profiling n=1（トレース）。
- **TimingScope**：runner が実装関数全体を `Time_sec` として stdout 出力
  （`src/core/runner.cpp`）。phase 内訳は stderr。
- **GTEPS**：`n_nodes × n_edges / Time_sec`（`n_edges`＝無向辺数 m）で統一。

## 5.6 正確性の許容（correctness tolerance）
- 混合許容：`abs_diff ≤ abs_tol + rel_tol · max(|ref|,|cand|)`。
- 値：`abs_tol=1e-3`, `rel_tol=1e-6`（small_full_vector, memory_paths とも）。
- 巨大 magnitude の BC（〜10^10）では絶対許容単独は不適切なため、絶対許容超過は WARN として
  分離し単独の失敗判定にしない（`correctness/pathmerge_tuned/README.md`）。
- **許容値は事後に変更しない**。`rel_tol=3e-6` 感度分析は補助情報で、正式 FAIL を PASS 化しない。

## 5.7 OOM / TIMEOUT の扱い
- OOM は **0 秒として扱わない**。feasibility 表では `Status=OOM_OR_FAIL`（`Time_sec=0` は
  「未達」を表すマーカーであり性能値ではない）。取得不能値は `N/A`。
- `failed/{build,runtime,timeout}` は該当なし（`failure/README.md`）。PBS `.o` の
  `Timeout: 21600s` はジョブ設定行であり実タイムアウトではない。
- UM b10240 の OOM はhost-memory-limited 100 GiB configurationで発生（runner_exit=137,
  `failure/failed/oom/memory_correctness_2368269/`）。

## 5.8 バッチ選択方法
- 提案手法（RQ1）：固定 b512（in-capacity 自動計算値, 上限 512）。**提案手法の batch sweep は未実施**。
- PathMerge tuned（RQ1 分母）：グラフごとに掃引最良を採用。email=b2048（掃引実測）、
  CA=b32（掃引実測）、PA/TX=b64（掃引で最適を確認し legacy 既定 b64 実測を保守的に採用）。
  掃引確認値 ≠ 最終採用値の関係は [06](06_results_performance.md) に明記。
- memory-path/feasibility（RQ3/RQ4）：`BC_BATCH_OVERRIDE`/`PATHMERGE_BC_BATCH_SIZE` で人為指定。

## 5.9 cuGraph の設定と既存実験との整合（制約）
`src/baseline/cugraph_bc.cu` の実コードから確認：
| 項目 | cuGraph 設定 | 提案/PathMerge との整合 |
|:--|:--|:--|
| exact / approximate | **exact**（vertices=`nullopt`＝全始点） | 整合（提案も全始点 exact） |
| normalized | **false**（非正規化） | 整合 |
| include_endpoints | **false** | 整合 |
| directed / undirected | undirected（`is_symmetric=true`） | 整合 |
| 無向 /2 補正 | adapter は**明示的 /2 を適用しない** | **不明**（提案/PathMerge は /2）。cuGraph が内部で対称性を扱うか未確認 |
| timing scope | 関数全体（Init + H2D+Build + BC + D2H）。BC-only 時間は stderr に別出力 | 提案も関数全体を計測するが、cuGraph は CUDA/RMM/RAFT 初期化を含む点で 1 回分の初期化オーバヘッドを内包 |

**制約**：cuGraph の /2 補正の有無と BC スケール整合が本環境で未確認のため、cuGraph は
small 限定の**補助 baseline** に留め、提案の headline 比較や正確性 ground truth には用いない。
timing scope も cuGraph は初期化を含むため、cuGraph の絶対時間は他実装と厳密には同条件でない
可能性がある（この点を論文に一文添える）。
