# SOURCE — tuning/kernel_selection（BFS カーネル forced shared/block 比較）

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/tuning/kernel_selection/<graph>/gpu_opt_forced/job_<jid>_20260710/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

- **目的**: BFS カーネル（shared-frontier vs 1ブロック=1ソース）を **強制実行した直接比較**（forced shared/block）。自動選択則には依存しない集計。
- **支える論文主張**: A 設計根拠（block 採用の裏付け。PA/TX に限定）
- **グラフ**: roadNet-PA, roadNet-TX
- **実装**: gpu_opt（`BC_FORCE_BFS_KERNEL=shared|block` で強制切替）
- **要求バッチ / 実効バッチ**: 512 / 512
- **SUB_BATCH**: 512 / **num_subs**: 1（in-capacity）
- **試行数**: 3
- **warmup**: なし
- **集計方法**: median + **標本標準偏差**。`summarize_kernel_selection.py` は forced shared/block の実測（中央値・標本SD・n・速い側・速度向上・Max BC 一致）のみを出力し、**選択則・「正しい選択/誤選択」判定は含まない**。
- **SourceSnapshotID**: `phase_def_block_20260710`
- **PBS job ID**: 2354329, 2354330
- **入力**: `data/snap/{roadNet-PA,roadNet-TX}`
- **出力**: `kernel_selection_results.tsv`, `kernel_selection_contributions.tsv`, `kernel_selection_max_bc.tsv`, `kernel_selection_summary.md`, `kernel_selection.log`
- **正確性**: `max_bc_only`（shared == block の Max BC 一致: PA=151395302679.08, TX=164495142042.45）
- **再現コマンド**: 測定 `qsub scripts/run_kernel_selection.sh`。集計 `python3 scripts/summarize_kernel_selection.py <dir>/kernel_selection_results.tsv <dir>`（依存なし・2回で冪等）。
- **formal 主張**: roadNet-PA/TX の強制比較で block が shared よりそれぞれ **1.52倍・1.66倍高速**、**Max BC 一致**。**未測定グラフへの一般化はしない**。
- **旧選択則の扱い**: 旧実装には平均次数（avg_deg<5→shared）に基づく自動選択則が存在したが、**現在は使用していない**（設計経緯のみ）。
