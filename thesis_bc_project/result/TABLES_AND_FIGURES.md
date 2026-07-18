# TABLES_AND_FIGURES — 論文表・図の入力と再生成可否

すべて **canonical raw（新パス）** を入力とし `scripts/` で再生成する。主要値は元 TSV から再生成し変更しない。

## Gate W7.4.1 — 修正版 325557 を反映した表・図（T3/F4, T4/F5, T5）

`scripts/generate_thesis_artifacts.py` の T3/F4（ablation）, T4/F5（memory feasibility）, T5（correctness）を
**修正版 325557**（`data/325557_3216152_corrected_v1`, SHA256 `8373244f…4eeaa22`, checkpoint `45352a3`）の
正式入力へ切り替えた。旧 malformed 325557 の値・旧 CORE_FAIL は **historical としてのみ保持**（現行主値ではない）。
生成コマンド:

```bash
THESIS_FIGS=F4,F5 python3 scripts/generate_thesis_artifacts.py
```

`THESIS_FIGS` で再生成する図を選択する。matplotlib の binary 出力は toolchain 間で byte 再現しないため、
本 Gate では **corrected データの F4/F5 のみ再生成**し、F1/F2/F3/F6/F7 は既存 byte を保持する。表（T1–T6）は
決定論的テキストで、T1/T2/T6 は byte 不変、T3/T4/T5 のみ更新される。artifact SHA256 等の来歴は
`result/CORRECTED_325557_ARTIFACT_PROVENANCE.tsv`、および `result/tables/thesis/TABLE_MANIFEST.tsv` /
`result/figures/thesis/FIGURE_MANIFEST.tsv` に記録する。

| 更新 artifact | 正式入力（canonical, W7.4 pending-commit を含む） | PBS job | 主な値 / 分類 |
|:--|:--|:--|:--|
| **T3 / F4**（ablation） | `raw_data/corrected_325557/job_2406254/ablation_results.tsv`; `raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv`; `result/ablation/corrected_325557/{ablation_contributions,synthetic4_aggregate}.tsv` | 2406254 (+2354994) | 修正版325557 H=1.4767x W=1.1012x A=1.5563x; 合成4 mixed 集約 H=1.679x W=1.066x A=1.391x（**mixed-checkpoint**, n=5, warmupは40行から除外） |
| **T4 / F5**（memory feasibility） | `result/memory_scalability/corrected_325557/feasibility_boundary.tsv`; `raw_data/corrected_325557/job_2404743/{feasibility_results,oom_evidence}.tsv` | 2404743 | Pure b4096 SUCCESS(65.89s); Pure b8192 **CUDA OOM**; UM b10240 SUCCESS(238.67s); UM b12288 **cgroup host-memory OOM kill (exit137)**; Chunked b16384 SUCCESS(66.60s)（n=1, targeted boundary, runtimeは性能比較でない） |
| **T5**（correctness） | `result/correctness/small_full_vector/correctness_summary.tsv`（Tier A）; `result/correctness/corrected_325557/{comparison_summary,vector_summary}.tsv`; `raw_data/corrected_325557/job_2404743/comparisons/*.json`（Tier B） | 2367583 / 2404743 | **Tier A（独立CPU参照）**: 3 小規模グラフ Sequential vs GPU_Opt 全 PASS（missing/mismatch 0）; **Tier B（実装間整合）**: 修正版325557 で 6 vector 全 PASS・10 比較すべて MismatchedElements=0; 計 13 行すべて MissingIndices=0, ToleranceResult=PASS, **ByteIdentical=No**（混合許容 abs_tol1e-3/rel_tol1e-6; PathMerge は external comparator、ground truth ではない） |

> **注記**: 合成4集約は mixed-checkpoint（他3グラフ=job2354994、325557=job2406254）であり同一checkpoint全再測ではない。
> memory feasibility は **CUDA OOM（GPU device, exit1）** と **cgroup host-memory OOM kill（exit137）** を別分類・別 marker とし、
> failure を 0 秒として描かない・未測定バッチを線で補間しない。Chunked b16384 は試験上限であり無制限容量を意味しない。
> T5 は 2 種類の証拠（Tier A=独立 Sequential CPU 参照、Tier B=修正版325557 の実装間整合）で構成する。PASS は混合許容下で mismatch=0 の意味であり byte-identical を含意しない。
> Tier B/PathMerge を独立参照・ground truth とは書かない。旧 malformed 325557 の CORE_FAIL は現行 T5 から除外し、
> `result/correctness/memory_paths/canonical_job_2368587/` に historical invalid-input result（provenance）として保持する。

## 再生成ステータスの凡例
- **REGENERATABLE**: スクリプト+入力が揃い再生成可能。依存不要な表は Gate B2、依存（numpy/scipy/matplotlib）を要する図表は **2026-07-14 (Gate J2) に本環境で実行確認・冪等検証済み**。
- **（履歴注記）REGENERATABLE_NOT_REVALIDATED_IN_GATE_B2**: Gate B2 当時は本環境に numpy/scipy/matplotlib が無く未再検証だった**過去状態**。**2026-07-14 (Gate J2) に該当図表を本環境で再検証し全て REGENERATABLE へ更新済み**（現在この状態のエントリは無い）。
- **NOT_YET_REGENERATABLE**: スクリプトまたは入力が未整備（現状該当なし）。

## 表
| 表 | 入力（新パス） | 生成 | 出力 | 再生成ステータス |
|:--|:--|:--|:--|:--|
| 最終速度向上表 | `proposed_variants/<g>/results.tsv`; `seven_implementations/legacy_partial/{medium,large}/results_no_gpu_opt.tsv`; `tuning/pathmerge/<g>/*.tsv` | `merge_final_tables.py`（依存なし） | `result/tables/final_speedup_tables.md` | **REGENERATABLE**（Gate B2 で2回生成し冪等確認・値不変） |
| 主軸比較表 | 上記 canonical link | 手動整形（元TSV） | `proposed_vs_pathmerge/{README.md,comparison.tsv}` | REGENERATABLE（元TSVから確認） |
| ablation 寄与表 | `raw_data/ablation/{synthetic/job_2354994_20260710,email-EuAll/job_2354999_20260710}/ablation_results.tsv` | `summarize_ablation.py`（**numpy不要**、trial summary は Sample SD, ddof=1） | `result/ablation/{synthetic_2354994,email_2354999}/{ablation_summary.md,ablation_contributions.tsv}` | **REGENERATABLE**（Gate W3.1 で2回生成し冪等性を再確認） |
| kernel選択表 | `tuning/kernel_selection/<g>/kernel_selection_results.tsv` + `kernel_selection_max_bc.tsv` | `summarize_kernel_selection.py`（**numpy不要・選択則非依存**） | `kernel_selection_summary.md`, `kernel_selection_contributions.tsv` | **REGENERATABLE**（選択則非依存の forced shared/block 比較。Gate B2.1 で PA/TX を再生成し2回実行で差分0=冪等確認済み） |
| UM feasibility 表 | `memory_scalability/oversubscribe_results_*.tsv` | `summarize_oversubscribe.py`（**numpy 必要**） | 集計md（未生成; stdout 集計） | **REGENERATABLE** — raw_data入力から2026-07-14に再検証済み。2回生成で冪等。 |
| 小規模 full-vector 正確性表 | `correctness/small_full_vector/{correctness_summary.tsv,*/comparison.md}` | `compare_bc_vectors.py`（実行済み出力を正式配置） | `correctness/small_full_vector/README.md` | **ARCHIVED_VERIFIED**（3グラフ、独立参照、全要素、mismatch/missing/NaN/Inf=0） |
| memory-path 分析表（G2.2） | 外部 raw BC vectors（`EXTERNAL_ARTIFACTS.tsv` 登録; jobs 2368269/2368398/2368587）+ `data/325557_3216152` | `analyze_memory_correctness.py`（**標準ライブラリのみ**） | `correctness/memory_paths/analysis/{run_to_run_comparison.tsv,stress_direct_comparison.tsv,six_vertex_detail.tsv,tolerance_sensitivity.tsv,Gate_G2_2_analysis.md}` | **REGENERATABLE**（raw vector から byte-identical 再生成を確認。`Gate_G2_3_audit.md` は静的監査で対象外） |
| T6 Experimental Environment | `result/environment/environment.md`; `raw_data/main_performance/proposed_variants/email-EuAll/_run/job_2357334_20260711/phase_timing.log`; `raw_data/profiling/job_2359175_20260711/bandwidth.log` | `generate_thesis_artifacts.py`（numpy+matplotlib） | `result/tables/thesis/T6_experimental_environment.{md,tsv}` | **REGENERATABLE**（Gate W1.3で同一コマンドを2回実行し、生成物の差分0を確認） |

## 図
| 図 | 入力 | 生成 | 出力 | 再生成ステータス |
|:--|:--|:--|:--|:--|
| phase_breakdown | `proposed_variants/<g>/phase_timing.log` | `statistical_analysis.py`（**scipy+numpy+matplotlib**） | `result/phase_breakdown/phase_breakdown.pdf`（既存） | **REGENERATABLE** — raw_data入力から2026-07-14に再検証済み。描画内容・PDF content streamsは公式出力と一致。通常生成では埋め込み生成日時のみ異なり得る。SOURCE_DATE_EPOCH固定時は冪等。 |
| statistical_test | proposed_variants results | `statistical_analysis.py`（同上） | `result/figures/statistical_test.md`（既存） | **REGENERATABLE** — raw_data入力から2026-07-14に再検証済み。公式出力とbyte-identical、2回生成で冪等。 |
| um_oversubscribe | `memory_scalability/oversubscribe_results_*.tsv` | `generate_um_figures.py`（**numpy+matplotlib**; 入力パスは raw_data 対応済） | `result/figures/fig_um_oversubscribe.pdf`（**未生成**） | **REGENERATABLE** — raw_data入力から生成成功。ただし論文用公式出力としては未採用・未追跡。 |

## Gate B2 で実行確認したコマンド（依存不要）
```bash
python3 scripts/merge_final_tables.py                 # → result/tables/final_speedup_tables.md（2回生成で冪等・値不変）
python3 scripts/check_results_integrity.py raw_data/tuning/pathmerge/*/pathmerge_bc/*/pathmerge_sweep_results.tsv   # ✅ 異常なし
python3 scripts/summarize_ablation.py raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv /tmp/out   # 公式版と一致
python3 scripts/summarize_kernel_selection.py raw_data/tuning/kernel_selection/roadNet-PA/gpu_opt_forced/job_2354329_20260710/kernel_selection_results.tsv result/tuning/kernel_selection/roadNet-PA   # PA: 2回実行で差分0
python3 scripts/summarize_kernel_selection.py raw_data/tuning/kernel_selection/roadNet-TX/gpu_opt_forced/job_2354330_20260710/kernel_selection_results.tsv result/tuning/kernel_selection/roadNet-TX   # TX: 2回実行で差分0
```

## 依存パッケージ（numpy/scipy/matplotlib）を用いる再生成（2026-07-14 本環境で再検証済み）
```bash
python3 scripts/generate_thesis_artifacts.py   # T6を含むthesis用全図表
python3 scripts/generate_um_figures.py raw_data/memory_scalability result/figures   # numpy+matplotlib
python3 scripts/summarize_oversubscribe.py raw_data/memory_scalability            # numpy
python3 scripts/statistical_analysis.py --results raw_data/main_performance/proposed_variants/*/_run/*/results.tsv \
        --phases raw_data/main_performance/proposed_variants/*/_run/*/phase_timing.log --outdir result/figures  # scipy+numpy+matplotlib
```

## 重要な注記
- **kernel選択 summary（選択則非依存）**: `summarize_kernel_selection.py` は **forced shared / forced block の直接比較のみ**を集計する（自動選択則・平均次数ヒューリスティクス・「正しい選択/誤選択」の判定を含まない）。Gate B2.1 で PA/TX を再生成し、2回実行で差分0（冪等）を確認済み。formal 主張は **roadNet-PA/TX の強制比較で block が shared よりそれぞれ 1.52倍・1.66倍高速、Max BC 一致**に限定し、**未測定グラフへ一般化しない**。
- **PA/TX の掃引確認値 ≠ 最終採用値**（PA 掃引≈941.4/採用918.67, TX 掃引1491.13/採用1482.68）: 同一 b64 設定の既存 default 実測を保守的 baseline として採用したため（矛盾ではない, `proposed_vs_pathmerge/README.md`）。
- **um_prefetch のプロファイル値**は `--duration=25` の 25秒部分トレース（全実行総量ではない）。
- **主軸(A)は tuned 基準**（既定 b64 比較 7.15×/1.64× と混同しない）。
- 小規模正確性ジョブの時間値は性能表・性能図の入力に使用しない。支持範囲は benchmark_7000_41459 / benchmark_11023_62184 / chain_200 のみ。
- **memory-path（correctness/memory_paths）は正確性・診断のみ**（325557 限定、各構成 n=1、warmup なし）。時間値は性能表・性能図に**追加しない**。canonical の formal overall status は `CORE_FAIL`（隠さない）、PathMerge は external comparator（ground truth ではない）。分析 TSV/Markdown は `analyze_memory_correctness.py` で raw vector から byte-identical 再生成できる（`Gate_G2_3_audit.md` は静的コード監査で再生成対象外）。
