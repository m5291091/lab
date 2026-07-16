# TABLES_AND_FIGURES — 論文表・図の入力と再生成可否

すべて **canonical raw（新パス）** を入力とし `scripts/` で再生成する。主要値は元 TSV から再生成し変更しない。

## 再生成ステータスの凡例
- **REGENERATABLE**: スクリプト+入力が揃い再生成可能。依存不要な表は Gate B2、依存（numpy/scipy/matplotlib）を要する図表は **2026-07-14 (Gate J2) に本環境で実行確認・冪等検証済み**。
- **（履歴注記）REGENERATABLE_NOT_REVALIDATED_IN_GATE_B2**: Gate B2 当時は本環境に numpy/scipy/matplotlib が無く未再検証だった**過去状態**。**2026-07-14 (Gate J2) に該当図表を本環境で再検証し全て REGENERATABLE へ更新済み**（現在この状態のエントリは無い）。
- **NOT_YET_REGENERATABLE**: スクリプトまたは入力が未整備（現状該当なし）。

## 表
| 表 | 入力（新パス） | 生成 | 出力 | 再生成ステータス |
|:--|:--|:--|:--|:--|
| 最終速度向上表 | `proposed_variants/<g>/results.tsv`; `seven_implementations/legacy_partial/{medium,large}/results_no_gpu_opt.tsv`; `tuning/pathmerge/<g>/*.tsv` | `merge_final_tables.py`（依存なし） | `result/tables/final_speedup_tables.md` | **REGENERATABLE**（Gate B2 で2回生成し冪等確認・値不変） |
| 主軸比較表 | 上記 canonical link | 手動整形（元TSV） | `proposed_vs_pathmerge/{README.md,comparison.tsv}` | REGENERATABLE（元TSVから確認） |
| ablation 寄与表 | `ablation/{synthetic_2354994,email_2354999}/ablation_results.tsv` | `summarize_ablation.py`（**numpy不要**） | `ablation_summary.md`, `ablation_contributions.tsv` | **REGENERATABLE**（Gate B2 で再生成し公式版と一致=冪等） |
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
