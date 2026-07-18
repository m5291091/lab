# MANIFEST — Phase D/E/F 実験検証・再現性 (thesis_bc_project)

本 manifest は最終表・アーカイブの各数値を、実行環境・投入ジョブ・入力ファイルまで
追跡可能にするための記録である。性能値は実測 TSV の中央値、正確性値は全要素比較の
直接出力であり、比率からの逆算・補間・捏造は含まない。

## 1. 実行環境

| 項目 | 値 |
|:--|:--|
| checkpoint commit (実験コード確定) | `phase_def_block_20260710` |
| GPU | NVIDIA GH200 |
| 公称 HBM3 | 96 GB |
| 記録されたデバイスメモリ | 97,871 MiB（約95.6 GiB、約102.6 decimal GB；公称96 GBと同一のHBM3） |
| 実行開始時の runtime 照会 | total 約102.0 GB、free (`free_before`) 約101.4 GB（decimal GB；freeは総容量ではなくメモリ予算計算の基準） |
| NVIDIA driver | 595.58.03 |
| CUDA (nvcc) | release 13.0, V13.0.48 |
| CMake | 4.3.4 (`~/.local/bin/cmake`) |
| C++ コンパイラ | g++ (GCC) 11.4.1 |
| nsys | 2025.5.1.121 |
| PBS system | Miyabi-G PBS batch system |
| Group | `gj17` |
| Queue | Not independently verifiable from retained job logs |
| memory-path実験の資源構成 | Host-memory-limited 100 GiB configuration |
| 実験時 git HEAD | checkpoint SHA と一致 (実験中に experiment 影響コードの commit なし) |

## 2. 投入ジョブ

| 用途 | PBS job ID | グラフ | 実装/バッチ | 試行 | 結果 |
|:--|:--|:--|:--|:--:|:--|
| T0.3 クリーンビルド+block smoke test | 2360062 | benchmark_7000_41459 | gpu_opt (非強制) | 1 | auto=block 確認 (OK) |
| T1 325557 screening (historical) | 2355000 | 325557_3216152 (旧 malformed) | PathMerge b512/b1024/b2048 | 3 | b512=240.07 b1024=195.15 b2048=175.43 (historical invalid-input evidence) |
| T1 PA confirmation | 2355001 | roadNet-PA | PathMerge b64/b128/b256/b512 | 3 | 当 job のみ (n=3): b64=943.47 b128=1105.57 b256=1155.30 b512=1207.41。b64 は別 job の追加 1 試行 (job `Not recorded`, 912.18) を含む **pooled n=4 で 941.39(最適)** |
| T1 PA screening (early terminated) | 2359080 | roadNet-PA | PathMerge b8/b16/b32 | 3 投入 / trial1 のみ完了 | b8=2714.98 b16=1573.87 b32=1015.98。trial2 の b8 実行中に終了、trial2/3 の記録なし。`failure/early_terminated/` |
| T1 325557 confirmation (historical) | 2359081 | 325557_3216152 (旧 malformed) | PathMerge b4096/b8192 | 3 | b4096=167.57(最小) b8192(実効6018)=168.27。要求 b8192 → **実効 6018 にクランプ** |
| T1 email screening (early terminated) | 2359096 | email-EuAll | PathMerge b8/b16/b64/b256/b1024 | 3 投入 / trial1 のみ完了 | b8=786.91 b16=491.01 b64=226.05 b256=125.91 b1024=97.69。trial2 の b8 実行中に終了、trial2/3 の記録なし。`failure/early_terminated/` |
| T1 email confirmation | 2359169 | email-EuAll | PathMerge b512/b1024/b2048/b4096/b8192 | 3 | **b2048=97.80(最適)** b512=106.43 b1024=99.93 b4096=101.58 b8192(実効7393)=103.27。要求 b8192 → **実効 7393 にクランプ** |
| T1 TX screening | 2360072 | roadNet-TX | PathMerge b32/b64/b128 | 1 | b32=1620.96 b64=1493.69 b128=1668.68 |
| T1 CA screening | 2360073 | roadNet-CA | PathMerge b32/b64/b128 | 1 | b32=3111.18 b64=3588.39 b128=3830.86 |
| T1.7 correctness (email) | 2360074 | email-EuAll | PathMerge b64 vs b2048 --dump-bc | 1 | 当時の比較 summary に基づく **PASS** (max_rel_err 4.9e-14, 混合許容不一致0)。vector 本体は `currently_unavailable`（比較 summary のみ保存, archive-time 再検証なし）→ NaN/Inf=`not_recorded` |
| T1.7 correctness (CA) | 2362965 | roadNet-CA | PathMerge b32 (最適) vs b64 (既定) --dump-bc | 1 | 当時の比較 summary に基づく **PASS (absolute-only warning)** (max_rel_err 3.9e-13, 混合許容不一致0)。vector 本体は `currently_unavailable`（比較 summary のみ保存, archive-time 再検証なし）→ NaN/Inf=`not_recorded` |
| T1 TX confirmation | 2361040 | roadNet-TX | b32 + b64 | 各 +2 (計 n=3) | b32=1620.96 **b64=1491.13(最適)** |
| T1 CA b16 内部最小確認 | 2361041 | roadNet-CA | b16 | 1 | b16=3609.95 (> b32 → b32 内部最小) |
| T1 CA confirmation | 2362006 | roadNet-CA | b32 + b64 | 各 +2 (計 n=3) | **b32=3079.72(最適)** b64=3491.64 |
| Stage 3 small full-vector correctness | 2367583.opbs | benchmark_7000_41459; benchmark_11023_62184; chain_200 | Sequential vs GPU_Opt, requested/effective/SUB=512/512/512, num_subs=1, NS_eff=2 | 各1, warmupなし | **PASS** (length=n, missing=0, NaN/Inf=0, mixed mismatch=0) |
| Stage 4C memory-path canonical | 2368587.opbs (ckpt memory_correctness_20260712) | 325557_3216152 | gpu_opt b1024/b9792, pure b1024, chunked b1024/b16384, pathmerge b4096（全6構成） | 各1, warmupなし | runner全PASS。formal `CORE_FAIL`: same_batch mismatch=0（非byte一致）、same_impl_diff_batch stress超過（和集合8頂点）、pathmerge_cross 5 DIFF（未解決） |
| Stage 4C memory-path diagnostic | 2369632.opbs (ckpt memory_diagnostic_20260713) | 325557_3216152 | CONTROL/T-RESET/T-NSEFF（b1024, 一因子各） | 各1, warmupなし | `DIAGNOSTIC_COMPLETE`: full reset/NS_eff=1 単独では CONTROL と差なし（`RESET/NS_EFF_NOT_DISTINGUISHED`, mismatch=0） |
| Stage 4C memory-path OOM (failed) | 2368269.opbs (ckpt memory_correctness_oom_20260712) | 325557_3216152 | gpu_opt b10240（UM oversubscribe）, pathmerge b4096 | 各1, warmupなし | UM b10240 がhost-memory-limited 100 GiB configurationでOOM（runner_exit=137）。`failure/failed/oom/`。空vector=OOM証跡 |
| Stage 4C memory-path fail-fast (early_terminated) | 2368398.opbs (ckpt memory_correctness_failfast_20260712) | 325557_3216152 | pathmerge b4096 → gpu_opt_pure b1024（以降未実行） | 各1, warmupなし | pure vs PathMerge 比較不一致で fail-fast（後続 chunked/UM 未実行）。`failure/early_terminated/` |
| **W7.4 corrected correctness+feasibility** | **2404743.opbs (ckpt 45352a3)** | **325557_3216152_corrected_v1** | gpu_opt b1024/b9792, pure b1024, chunked b1024/b16384, pathmerge b4096; feasibility pure b4096/b8192, um b10240/b12288, chunked b16384 | 各1, warmupなし | `SUCCESS`: 6ベクトル完全・**10比較すべて mismatch=0**（stress含む, 非byte）。境界: pure_b8192=CUDA OOM, um_b12288=host/cgroup OOM kill(exit137,非CUDA), chunked b16384=success。`result/{correctness,memory_scalability}/corrected_325557/` |
| **W7.4 corrected ablation** | **2406254.opbs (ckpt 45352a3)** | **325557_3216152_corrected_v1** | Ablation_H{0,1}_W{0,1}_A{0,1} (b512) | n=5 + 各invocation先頭untimed H1W1A1 warmup×5(統計非算入) | `SUCCESS_COMPLETE_40`: 40行完全。H=1.4767 W=1.1012 A=1.5563。合成4集約(mixed-checkpoint)=H1.6787/W1.0661/A1.3914。`result/ablation/corrected_325557/` |
| build 失敗 (failed/build) | 2403658.opbs (ckpt 193eb21) | 325557_3216152_corrected_v1 | — (Stage2 CMake binary-dir 衝突) | — | `BUILD_FAILED_CMAKE_BINARY_DIR_COLLISION`: runner=0/vector=0。W7.3B1.1で修正。`failure/failed/build/` |
| validation 誤判定 (failed/validation) | 2404249.opbs (ckpt b677d6c) | 325557_3216152_corrected_v1 | gpu_opt_pure b1024 (runner exit0) | — | `VALIDATION_FALSE_POSITIVE_OOM_MARKER`: 助言警告の"OOM"語を素朴検査が誤検出(実際は成功)。W7.3B2.2で修正。`failure/failed/validation/` |

- **PathMerge batch sweep の既知 job は上表に 11 件すべて収録**: email-EuAll = 2359096(screening, 早期打切り) / 2359169(confirmation)、roadNet-PA = 2359080(screening, 早期打切り) / 2355001(confirmation)、roadNet-TX = 2360072(screening) / 2361040(confirmation)、roadNet-CA = 2360073(screening) / 2361041(extension b16) / 2362006(confirmation)、historical 325557 = 2355000(screening) / 2359081(confirmation)。
  次の 2 件は PBS Job ID が保存記録に存在せず **`Not recorded`** とする（推定補完しない）: roadNet-PA b64 の追加 1 試行、historical 325557 の初期実行 (b32/b64/b256/b512)。
  **2360074 と 2362965 は `--dump-bc` の正確性確認 job であり、batch sweep job ではない**（上表でも別行として区別）。stage 別 job 対応の正本は `tuning/pathmerge/SOURCE.md`。
- **確定した最適バッチ (n=3 中央値)**: roadNet-TX = **b64** (1491.13s), roadNet-CA = **b32** (3079.72s)。
  TX は PA と同じ b64 が内部最小。CA は実測で b32 が最速 (PA/TX から推定した b64 は CA に一般化せず)。
- confirmation は「異なる 2 候補 (片方は必ず b64)」を各 n=3: TX={b64, b32}, CA={b32, b64}。
  差はいずれも 3% 超 (TX 8.7% / CA 13.4%) のため b64 優先の保守ルールは適用外。
- **T1.7 の PathMerge BC ベクトル 4 本 (Gate W5.2)**: original runtime path は
  `build_miyabi/{t1_correctness,t1_ca_correctness}/bc_b*.txt`（historical build output path）。
  4 本とも `currently_unavailable` で、`result/EXTERNAL_ARTIFACTS.tsv` に
  `RetentionStatus=not_retained` / `Availability=currently_unavailable` として登録済み。
  保存されているのは比較 summary のみであり、archive-time に vector を再解析していない。
  PA/TX は tuned/default とも b64 で別の full-vector comparison artifact がなく、`max_bc_only`。

### Stage 3 小規模 full-vector 正確性

- canonical archive: `result/correctness/small_full_vector/`
- checkpoint: `small_correctness_20260712`; PBS job: `2367583.opbs`。
- Sequential を独立参照、GPU_Opt を candidate とし、3グラフすべて runner/comparison exit 0、vector length=n、missing=0、NaN/+Inf/-Inf=0、混合許容不一致=0。
- `abs_tol=1e-3`, `rel_tol=1e-6`; 判定式は `abs_diff <= abs_tol + rel_tol * max(abs(reference),abs(candidate))`。
- requested/effective batch=512/512、SUB_BATCH=512、num_subs=1、NS_eff=2（stderr実測）。各 n=1、warmupなしで時間値は性能結果に使用しない。
- PBS accounting の明示的 Exit_status は SIM0801 により未取得。`set -euo pipefail` の最終PASS、runner/comparison exit 0、完全成果物で成功を確認。
- scope は小規模3グラフのみ。email/roadNet、Pure/Chunked、oversubscription 固有経路、専用カウンタによる Hybrid BFS/warp 経路確認は含まない。

## 3. 最終表の数値と入力ファイル (追跡可能性)

集計は全て中央値 (median)。詳細な per-数値の出典は
`result/tables/final_speedup_tables.md` の「出典・追跡可能性」節を参照。

| 数値種別 | 入力ファイル (git 管理下) | impl/行 | 集計 |
|:--|:--|:--|:--|
| 提案手法 (block) [s] | `raw_data/main_performance/proposed_variants/<graph>/_run/job_2357334_20260711/results.tsv` | `GPU_Opt` Time_sec | median |
| PathMerge 既定 b64 [s] | `raw_data/main_performance/seven_implementations/legacy_partial/{medium,large}/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv` | `PathMerge_BC` Time_sec | median |
| PathMerge tuned [s] | `raw_data/tuning/pathmerge/<graph>/pathmerge_bc/job_multi_20260710/*.tsv` | 掃引最良バッチ vs 既定 b64 の速い方 | median |
| 旧提案 shared [s] | `raw_data/main_performance/seven_implementations/legacy_partial/{medium,large}/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv` | `GPU_Opt_Pure` (旧 shared 経路) | median |

- **email 掃引 (b8-b8192)**: screening = `raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/email_smallbatch_trial1.tsv`
  (job 2359096, 早期打切りの trial1 のみ), confirmation = 同 dir `pathmerge_sweep_results.tsv` (job 2359169, 各 n=3)。
  中央値 **b2048=97.80s (最小)**。tuned=b2048。要求 b8192 は**実効 7393 にクランプ**。
  b1024 は screening 1 + confirmation 3 の pooled 4 (`final_speedup_tables.md` の n=4 はこの pooled)。
  vs tuned=**3.17×** (既定 b64 の 7.15× とは区別)。詳細は `tuning/pathmerge/email-EuAll/SOURCE.md`。
- **PA 掃引 (b8-b512)**: `raw_data/tuning/pathmerge/roadNet-PA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`
  (`SOURCE.md` に build_miyabi 出典・n を記録)。中央値 b64=941.4s (最小)。
  screening (b8/b16/b32) = job 2359080 の早期打切り trial1、confirmation (b64-b512) = job 2355001。
  b64 の追加 1 試行は PBS Job ID が保存されておらず `Not recorded`。
- **325557 掃引**: `raw_data/tuning/pathmerge/325557/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`
  中央値 b4096=167.574s (最小), b8192(実効6018)=168.266s (約 0.41% 悪化)。
- **TX 掃引**: `raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`
  中央値 b64=1491.1s (最小=最適設定の確認測定)。最適は既定と**同一の b64 設定**のため、
  最終表 tuned は既定の実測値 legacy b64=1482.68s を採用 (掃引の 1491.1s は別測定; 詳細は
  同ディレクトリ SOURCE.md)。vs tuned=1.51×。
- **CA 掃引**: `raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`
  中央値 **b32=3079.7s (最小)**。tuned=b32 (3079.72s)。vs tuned=**1.45×** (b64 既定の 1.64× から低下)。

## 4. プロファイリング

- PBS job 2359175 内の nsys は、ablation H1W1A0/H1W1A1 が `56438_300801`、別の `um_prefetch_gpu_opt` が `325557_3216152` を対象とする。backward 63.9% / bfs 36.1% は `ablation_H1W1A0` の単一トレースにおける CUDA GPU カーネル時間のみの比率であり、本測定 H1W1A0 に加えて同一 process 冒頭の untimed H1W1A1 warmupを含む。
- `.stats.txt` は `.nsys-rep` から再生成 (元 `.nsys-rep` は不変; md5 検証済み):
  `nsys stats --force-export=true --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum
  --report cuda_api_sum <rep>` (H1W1A0/A1 同一レポート)。
- `um_prefetch_gpu_opt` の HtoD migration 27.918 MB / CPU faults 85 / GPU faults 9 は
  **`--duration=25` の 25 秒部分トレースの値** (実行全体の総量ではない)。

## 5. 再現手順 (新規 clone 相当)

```bash
# --- 最終表の冪等性確認 (生成前保存 → 生成 → diff → 再生成 → diff) ---
cp result/tables/final_speedup_tables.md /tmp/ft_before.md   # 生成前を一時保存
python3 scripts/merge_final_tables.py                        # 1回目生成
diff /tmp/ft_before.md result/tables/final_speedup_tables.md # 生成前後の差分 (パス更新等が無ければ空)
python3 scripts/merge_final_tables.py                        # 2回目生成
diff /tmp/ft_before.md result/tables/final_speedup_tables.md # 2回目も同一なら冪等 (空)
# ※ 必須入力 (提案 results.tsv×4 / PathMerge 掃引×4 / legacy b64×2) が欠けると非0終了 (fail-fast, 推定なし)

# --- 結果整合性検査 (FAIL/OOM/TIMEOUT/欠損) ---
python3 scripts/check_results_integrity.py raw_data/tuning/pathmerge/*/pathmerge_bc/*/pathmerge_sweep_results.tsv

# --- 統計・図 (scipy/numpy/matplotlib が必要。無い環境では再生成しない) ---
uv run --with scipy --with matplotlib --with numpy python3 scripts/statistical_analysis.py \
  --results result/main_performance/proposed_variants/*/results.tsv \
  --phases  result/main_performance/proposed_variants/*/phase_timing.log \
  --outdir  result/figures
```
