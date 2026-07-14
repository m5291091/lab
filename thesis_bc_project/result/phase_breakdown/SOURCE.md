# SOURCE — phase_breakdown（BFS/Backward 内訳）

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/main_performance/proposed_variants/<graph>/_run/job_2357334_20260711/ (phase_timing.log)` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

- **目的**: 提案手法の phase 内訳（BFS / Backward）と可視化
- **支える論文主張**: A（提案手法の時間内訳）
- **グラフ**: email-EuAll, roadNet-PA/TX/CA
- **実装**: GPU_Opt（block, UM）
- **要求/実効バッチ / SUB_BATCH / num_subs**: 512 / 512 / 512 / 1（in-capacity, 源泉ログ確認）
- **試行数**: email 5 / road 3 / **warmup**: なし / **集計方法**: median
- **SourceSnapshotID**: `phase_def_block_20260710`
- **PBS job ID**: 2356120, 2357334–2357337（proposed_variants と同一実行の phase_timing.log 由来）
- **入力（源泉）**: `../main_performance/proposed_variants/<graph>/phase_timing.log`（源泉はそちらに残置。重複コピーなし）
- **出力**: `phase_breakdown.pdf`
- **正確性**: `not_applicable`（時間内訳）
- **再現コマンド**: `python3 scripts/statistical_analysis.py --phases raw_data/main_performance/proposed_variants/*/_run/*/phase_timing.log --outdir result/phase_breakdown`
- **制約（重要）**: BFS/Backward の**計測成分**内訳。UM 版は H2D/D2H 転送や init を個別計測しないため、end-to-end 総時間との差は **Other（未計測）** として明示（転送込み完全内訳ではない）。
