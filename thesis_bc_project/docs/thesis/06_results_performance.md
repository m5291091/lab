# 06 性能評価（RQ1）

主値は median。速度向上は median/median。GTEPS は `n_nodes × n_edges / median_time`
（`n_edges`＝無向辺数 m）で統一して算出。数値は元 TSV から再計算し、`result/CLAIMS.md`・
`result/tables/final_speedup_tables.md` と一致（変更禁止）。

## 6.1 主性能表（T-PERF・必須）
提案 = block GPU_Opt（UM, 固定 b512, in-capacity）。PathMerge = tuned（グラフごとに調整）。

| Graph | Nodes | Edges | GPU_Opt Req.Batch | GPU_Opt Eff.Batch | GPU_Opt Trials | GPU_Opt Median[s] | GPU_Opt Mean[s] | GPU_Opt SampleSD | GPU_Opt GTEPS | PathMerge Batch | PathMerge Trials | PathMerge Median[s] | PathMerge Mean[s] | PathMerge SampleSD | PathMerge GTEPS | Speedup | SupportingFiles |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|--:|--:|--:|--:|--:|--:|:--|
| email-EuAll | 265009 | 364481 | 512 | 512 | 5 | **30.81** | 30.81 | 0.061 | 3.14 | b2048 | 3 | **97.80** | 97.90 | 0.988 | 0.99 | **3.17×** | ①⑤ |
| roadNet-PA | 1088092 | 1541898 | 512 | 512 | 3 | **699.52** | 700.49 | 1.695 | 2.40 | b64 | 3 | **918.67** | 923.26 | 9.593 | 1.83 | **1.31×** | ②⑥ |
| roadNet-TX | 1379917 | 1921660 | 512 | 512 | 3 | **980.13** | 983.66 | 6.352 | 2.71 | b64 | 3 | **1482.68** | 1493.46 | 24.855 | 1.79 | **1.51×** | ③⑥ |
| roadNet-CA | 1965206 | 2766607 | 512 | 512 | 3 | **2129.10** | 2127.38 | 4.021 | 2.55 | b32 | 3 | **3079.72** | 3083.85 | 25.511 | 1.77 | **1.45×** | ④⑦ |

出典：
- ① `raw_data/main_performance/proposed_variants/email-EuAll/_run/job_2357334_20260711/results.tsv`（GPU_Opt, n=5）
- ② `raw_data/main_performance/proposed_variants/roadNet-PA/_run/job_2357334_20260711/results.tsv`（GPU_Opt, n=3）
- ③ `raw_data/main_performance/proposed_variants/roadNet-TX/_run/job_2357334_20260711/results.tsv`（GPU_Opt, n=3）
- ④ `raw_data/main_performance/proposed_variants/roadNet-CA/_run/job_2357334_20260711/results.tsv`（GPU_Opt, n=3）
- ⑤ `raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`（PathMerge b2048, n=3）
- ⑥ `raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv`（PathMerge_BC b64, n=3, legacy 既定）
- ⑦ `raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`（PathMerge b32, n=3）

**再計算の一致確認**：median/median の speedup は 3.1743 / 1.3133 / 1.5127 / 1.4465 で、公式値
3.17 / 1.31 / 1.51 / 1.45 と一致（丸め）。**期待値と一致しない場合は停止し、逆算・修正しない**。

## 6.2 tuned バッチの根拠と「掃引確認値 ≠ 最終採用値」
- **email-EuAll**：掃引 b2048=97.80 s が最良（b1024=99.9, b4096=101.6…）。採用 b2048。
- **roadNet-CA**：掃引 b32=3079.72 s が最良（b64=3491.6, b16=3610.0…）。採用 b32。既定 b64
  比較の 1.64×（3499.03/2129.10）と区別し、tuned は **1.45×**。
- **roadNet-PA/TX**：掃引で最適は既定と同一の **b64**。最終表は同一 b64 設定の legacy 既定実測
  （PA 918.67, TX 1482.68）を保守的 baseline として採用。掃引確認値（PA≈941.4, TX≈1491.13）と
  一致しないのは**欠損・矛盾ではなく、別測定（保守的 baseline の採用）**である
  （`result/main_performance/proposed_vs_pathmerge/README.md`,
  `result/tuning/pathmerge/{roadNet-PA,roadNet-TX}/SOURCE.md`）。
- PathMerge tuned の正確性（tuned batch が既定 b64 と同一 BC を出すこと）は
  `result/correctness/pathmerge_tuned/`（email b64 vs b2048: max_rel 4.9e-14, mismatch 0;
  CA b32 vs b64: max_rel 3.9e-13, mismatch 0）で確認済み。

## 6.3 既定 b64 基準との区別（誤用防止）
| グラフ | vs 既定 b64 | vs tuned | 論文で使う値 |
|:--|--:|--:|:--|
| email-EuAll | 7.15×（220.39/30.81） | **3.17×** | tuned |
| roadNet-PA | 1.31×（918.67/699.52） | **1.31×** | tuned（同一 b64） |
| roadNet-TX | 1.51×（1482.68/980.13） | **1.51×** | tuned（同一 b64） |
| roadNet-CA | 1.64×（3499.03/2129.10） | **1.45×** | tuned |

**主軸は tuned 基準**。既定 b64 比較の 7.15×/1.64× を主張として混同しない。

## 6.4 補助 baseline（Sequential / OpenMP / cuGraph, small 限定）
**注意（必読）**：これは現行 block による全グラフ統一 7 実装比較では**ない**。legacy（旧 shared
カーネル・旧 tree 測定）の部分データであり、medium/large では Sequential/OpenMP/cuGraph が欠ける。
比較可能な行のみを補助表として掲載し、欠損は `N/A` とする。

### small 補助比較（mean±SD, n=10; 出典 `.../small/statistical_test_no_gpu_opt.md`）
| Graph | n / m | Sequential[s] | OpenMP[s] | cuGraph_BC[s] | GPU_Opt_Pure[s]※ | PathMerge_BC[s] |
|:--|:--|--:|--:|--:|--:|--:|
| benchmark_7000_41459 | 7000/41459 | 5.63±0.05 | 0.10±0.00 | 1.12±0.04 | 0.22±0.01 | 0.41±0.02 |
| benchmark_11023_62184 | 11023/62184 | 11.39±0.08 | 0.16±0.02 | 2.81±0.08 | 0.25±0.01 | 1.02±0.03 |
| random | 32212/101805 | 171.17±2.60 | 3.36±0.03 | 17.23±0.43 | 0.90±0.01 | 0.90±0.01 |
| 56438_300801 | 56438/300801 | **N/A** | 13.35±0.13 | 71.85±1.36 | 2.02±0.01 | 4.64±0.01 |

※ GPU_Opt_Pure（small）は**旧 shared 経路**の legacy 値であり、headline の block 値ではない。
Sequential は 56438 で欠（非現実的コストのため未測定 → `N/A`）。

**この補助表から言えること/言えないこと**：
- 言える：small では Sequential/OpenMP（CPU）に対し GPU 実装が桁で高速。cuGraph は
  `sort_by_key + reduce_by_key` の per-level O(M log M) により提案系より低速（`CLAUDE.md` の
  性能リファレンスと整合）。
- 言えない：medium/large の統一 7 実装比較（Seq/OMP/cuGraph 欠 + 提案系が旧 shared →
  `NOT_YET_SUPPORTED`）。cuGraph の BC スケール整合は未確認（[05](05_experimental_setup.md)
  §5.9）なので cuGraph を正確性の基準にはしない。

## 6.5 phase breakdown（F-PHASE）
提案 block の BFS/backward 内訳は `result/main_performance/proposed_variants/<g>/phase_timing.log`
に基づき、図 `result/phase_breakdown/phase_breakdown.pdf`（既存）で提示。生成スクリプトは
scipy+numpy+matplotlib 依存で本環境では未再検証（[13](13_tables_and_figures.md)）。
