# 13 表・図の一覧

各表・図を **必須 / 推奨 / 補助 / 付録** に分類。列：`ID / Title / Purpose / InputFiles /
GenerationCommand / Status / Chapter`。再生成可否は `result/TABLES_AND_FIGURES.md` に準拠。

## 凡例（Status）
- `REGENERATABLE`：スクリプト+入力が揃い依存不要（Gate B2 で冪等確認済）。
- `REGEN_DEP_MISSING`：入力は整備済だが numpy/matplotlib/scipy が本環境に無く未再検証（公式出力は据置）。
- `ARCHIVED_VERIFIED`：実行済み出力を正式配置（全要素検証済）。
- `MANUAL_FROM_TSV`：元 TSV から手動整形（逆算なし）。

## 必須（本文の主張を直接支える）
| ID | Title | Purpose | InputFiles | GenerationCommand | Status | Chapter |
|:--|:--|:--|:--|:--|:--|:--|
| T-PERF | 主性能表（提案 vs PathMerge tuned） | RQ1 の中心主張 1.31〜3.17× | `main_performance/proposed_variants/<g>/results.tsv`; `tuning/pathmerge/<g>/*`; `seven_implementations/legacy_partial/large/results_no_gpu_opt.tsv` | `scripts/merge_final_tables.py` | REGENERATABLE | 6 |
| T-ABL | アブレーション寄与表（H/W/A 主効果） | RQ2 の要因分解 | `ablation/{synthetic_2354994,email_2354999}/ablation_results.tsv` | `scripts/summarize_ablation.py`（numpy 不要） | REGENERATABLE | 7 |
| T-MEM | メモリ feasibility 表（UM/Pure/Chunked × batch） | RQ3 の OOM 境界 | `memory_scalability/oversubscribe_results_*.tsv` | 手動整形（Status 行列, 逆算なし） | MANUAL_FROM_TSV | 8 |
| T-CORR | 正確性区分表（5 区分） | RQ4 の支持範囲 | `correctness/small_full_vector/*`; `correctness/memory_paths/canonical_job_2368587/comparison_matrix.tsv` | 手動整形（元出力） | MANUAL_FROM_TSV | 9 |

## 推奨（主張を補強）
| ID | Title | Purpose | InputFiles | GenerationCommand | Status | Chapter |
|:--|:--|:--|:--|:--|:--|:--|
| T-GRAPH | グラフ統計表 | 実験対象の素性 | `docs/graph_stats.tsv`; `result/datasets/graph_catalog.tsv` | 手動整形 | MANUAL_FROM_TSV | 5 |
| T-ENV | 実行環境表 | 再現性 | `result/environment/environment.md`; `result/MANIFEST.md` | 手動整形 | MANUAL_FROM_TSV | 5 |
| T-KSEL | BFS カーネル選択表（forced shared/block） | H の block 優位 | `tuning/kernel_selection/<g>/kernel_selection_results.tsv` | `scripts/summarize_kernel_selection.py`（numpy 不要） | REGENERATABLE | 7 |
| F-PHASE | phase breakdown 図 | BFS/backward 内訳 | `main_performance/proposed_variants/<g>/phase_timing.log` | `scripts/statistical_analysis.py`（scipy+numpy+matplotlib） | REGEN_DEP_MISSING（既存 PDF あり） | 6 |
| F-NSYS | nsys カーネル要約 | backward 63.9%/bfs 36.1% | `profiling/ablation_H1W1A0.stats.txt` | `nsys stats`（既生成, .nsys-rep から） | ARCHIVED_VERIFIED | 7 |

## 補助（small 限定・文脈提供）
| ID | Title | Purpose | InputFiles | GenerationCommand | Status | Chapter |
|:--|:--|:--|:--|:--|:--|:--|
| T-AUX | 補助 baseline 表（Seq/OMP/cuGraph, small） | CPU/汎用 GPU との桁比較 | `seven_implementations/legacy_partial/small/statistical_test_no_gpu_opt.md` | 手動整形（欠損 N/A） | MANUAL_FROM_TSV | 6 |
| T-BW | 帯域表（HBM3/C2C） | A の物理的裏付け | `profiling/bandwidth.log` | 手動整形 | MANUAL_FROM_TSV | 2/5 |
| F-UM | UM oversubscribe 図 | feasibility の可視化 | `memory_scalability/oversubscribe_results_*.tsv` | `scripts/generate_um_figures.py`（numpy+matplotlib） | REGEN_DEP_MISSING（出力未生成） | 8 |
| T-STRESS | stress 差 8 頂点詳細 | RQ4 の未解決差 | `correctness/memory_paths/analysis/{six_vertex_detail,tolerance_sensitivity}.tsv` | `scripts/analyze_memory_correctness.py`（標準ライブラリ） | REGENERATABLE | 9 |

## 付録
| ID | Title | Purpose | InputFiles | GenerationCommand | Status | Chapter |
|:--|:--|:--|:--|:--|:--|:--|
| A-SWEEP | PathMerge 全バッチ掃引 | tuned 選定の透明性 | `tuning/pathmerge/*/pathmerge_sweep_results.tsv` | `scripts/merge_final_tables.py`（掃引詳細節） | REGENERATABLE | 付録A |
| A-ABL8 | アブレーション全 8 構成 | 完全開示 | `ablation/*/ablation_summary.md` | `scripts/summarize_ablation.py` | REGENERATABLE | 付録B |
| A-MEMMTX | memory-path 比較行列 | 正確性の一次情報 | `correctness/memory_paths/canonical_job_2368587/comparison_matrix.tsv` | `scripts/analyze_memory_correctness.py` | REGENERATABLE | 付録C |
| A-FAIL | 失敗・OOM 記録 | 透明性 | `failure/README.md`; `failure/MANIFEST.tsv` | 手動整形 | MANUAL_FROM_TSV | 付録E |

## 本環境で再生成できない図（明記）
`result/TABLES_AND_FIGURES.md` の通り、以下は numpy/matplotlib/scipy 依存で **本環境では未再検証**
（公式出力は据置、`N/A` で上書きしない）：
- F-PHASE（`phase_breakdown.pdf`, 既存）
- statistical_test（`figures/statistical_test.md`, 既存）
- F-UM（`fig_um_oversubscribe.pdf`, **未生成**）
- UM feasibility 集計 md（`summarize_oversubscribe.py`, numpy 必要, 未生成）

依存が揃った環境での再生成コマンドは `result/TABLES_AND_FIGURES.md` の該当節にある。

## Gate B2 で実行確認済み（依存不要）のコマンド
```
python3 scripts/merge_final_tables.py                     # T-PERF（2回生成で冪等・値不変）
python3 scripts/summarize_ablation.py <ablation_results.tsv> <out>   # T-ABL（公式版と一致）
python3 scripts/summarize_kernel_selection.py <ksel_results.tsv> <dir>  # T-KSEL（差分0）
python3 scripts/check_results_integrity.py raw_data/tuning/pathmerge/*/pathmerge_bc/*/pathmerge_sweep_results.tsv
```
