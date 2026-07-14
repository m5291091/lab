# 最終ベンチマーク表 (提案 block × PathMerge 既定/tuned)

集計: 全て中央値 (median)。実測 TSV のみ (比率逆算なし)。
入力は git 管理下の `raw_data/` (raw) と `result/` (派生) のみ (build_miyabi 非依存・新規 clone から再生成可能)。
提案手法は GPU_Opt (UM, 常時 block)。

### medium (email-EuAll)

| グラフ | 提案(block) [s] | PathMerge既定 b64 [s] | PathMerge tuned [s] (batch) | vs 既定 | vs tuned |
|:-----|------:|------:|------:|------:|------:|
| email-EuAll | 30.81 | 220.39 | 97.80 (b2048) | 7.15× | 3.17× |

### large (roadNet 3種)

| グラフ | 提案(block) [s] | PathMerge既定 b64 [s] | PathMerge tuned [s] (batch) | vs 既定 | vs tuned |
|:-----|------:|------:|------:|------:|------:|
| roadNet-PA | 699.52 | 918.67 | 918.67 (b64) | 1.31× | 1.31× |
| roadNet-TX | 980.13 | 1482.68 | 1482.68 (b64) | 1.51× | 1.51× |
| roadNet-CA | 2129.10 | 3499.03 | 3079.72 (b32) | 1.64× | 1.45× |

### 旧表との差分 (BFS カーネル shared→block の逆転)

旧提案手法値は shared 経路 (legacy)、新値は常時 block (Phase D 再計測)。
speedup 基準は PathMerge 既定 (b64, legacy 実測) で固定。

| グラフ | 旧 提案(shared) [s] | 新 提案(block) [s] | 旧 speedup | 新 speedup | 逆転 |
|:-----|------:|------:|------:|------:|:---:|
| email-EuAll | 190.95 | 30.81 | 1.15× | 7.15× | ↑ |
| roadNet-PA | 1062.56 | 699.52 | 0.86× | 1.31× | ○ |
| roadNet-TX | 1636.10 | 980.13 | 0.91× | 1.51× | ○ |
| roadNet-CA | 3494.98 | 2129.10 | 1.00× | 1.64× | ↑ |

### PathMerge バッチ掃引 詳細 (batch 別 median 実行時間 [s])

- **roadNet-PA**: b8=2715.0(n1) b16=1573.9(n1) b32=1016.0(n1) b64=941.4(n4) b128=1105.6(n3) b256=1155.3(n3) b512=1207.4(n3)  → 最良 b64=941.4s
- **roadNet-TX**: b32=1621.0(n3) b64=1491.1(n3) b128=1668.7(n1)  → 最良 b64=1491.1s
- **roadNet-CA**: b16=3610.0(n1) b32=3079.7(n3) b64=3491.6(n3) b128=3830.9(n1)  → 最良 b32=3079.7s
- **email-EuAll**: b8=786.9(n1) b16=491.0(n1) b64=226.0(n1) b256=125.9(n1) b512=106.4(n3) b1024=99.9(n4) b2048=97.8(n3) b4096=101.6(n3) b8192=103.3(n3)  → 最良 b2048=97.8s
- **325557_3216152**: b32=1292.3(n2) b64=770.8(n1) b256=324.3(n1) b512=240.2(n4) b1024=195.2(n3) b2048=175.4(n3) b4096=167.6(n3) b8192=168.3(n3)  → 最良 b4096=167.6s

### 出典・追跡可能性 (各数値の入力ファイルと集計)

集計方法は全て中央値 (median)。入力は git 管理下のみ。

| グラフ | 提案(block) 出典 (impl=GPU_Opt, n) | PathMerge既定 出典 (impl=PathMerge_BC, n) | tuned 出典 (batch, n) |
|:-----|:-----|:-----|:-----|
| email-EuAll | `raw_data/main_performance/proposed_variants/email-EuAll/_run/job_2357334_20260711/results.tsv` (n=5) | `raw_data/main_performance/seven_implementations/legacy_partial/medium/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv` (n=5) | 掃引 (実測) b2048 (n=3); raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/email_smallbatch_trial1.tsv; raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv |
| roadNet-PA | `raw_data/main_performance/proposed_variants/roadNet-PA/_run/job_2357334_20260711/results.tsv` (n=3) | `raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv` (n=3) | legacy 既定 (掃引で最適確認) b64 (n=4); raw_data/tuning/pathmerge/roadNet-PA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv |
| roadNet-TX | `raw_data/main_performance/proposed_variants/roadNet-TX/_run/job_2357334_20260711/results.tsv` (n=3) | `raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv` (n=3) | legacy 既定 (掃引で最適確認) b64 (n=3); raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv |
| roadNet-CA | `raw_data/main_performance/proposed_variants/roadNet-CA/_run/job_2357334_20260711/results.tsv` (n=3) | `raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv` (n=3) | 掃引 (実測) b32 (n=3); raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv |

- **提案手法 (block)**: `raw_data/main_performance/proposed_variants/<graph>/_run/*/results.tsv` の `GPU_Opt` 行 Time_sec 中央値。
- **PathMerge 既定 (b64)**: `raw_data/main_performance/seven_implementations/legacy_partial/{medium,large}/no_gpu_opt/*/results_no_gpu_opt.tsv` の `PathMerge_BC` 行 Time_sec 中央値。
- **PathMerge tuned**: `raw_data/tuning/pathmerge/<graph>/pathmerge_bc/*/*.tsv` の掃引実測。採用値はグラフ別に固定 (email=掃引 b2048 / roadNet-PA・TX=掃引で b64 最適を確認し legacy b64 中央値 / roadNet-CA=掃引 b32)。**現在の主要4グラフは全て実測であり、推定値は使用していない。**

