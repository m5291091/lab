# raw_data/main_performance

主性能実験の **raw**（実行ランナー直接出力）。派生表・図・要約は `result/main_performance/` に保持。

## proposed_variants（提案 block 再計測: UM/Pure/Chunked）
- 配置: `proposed_variants/<graph>/_run/job_2357334_20260711/{results.tsv, phase_timing.log, max_bc.tsv, benchmark.log}`
- graph: email-EuAll, roadNet-PA, roadNet-TX, roadNet-CA
- SourceSnapshotID: `phase_def_block_20260710` / PBS job: 2356120, 2357334–2357337 / RunDate 2026-07-11
- 派生: `result/main_performance/proposed_variants/<graph>/{summary_table.md,speedup_table.md,correctness.md}`

## seven_implementations（legacy 部分データ, 旧 shared / 旧ツリー）
- 配置: `seven_implementations/legacy_partial/<size>/{no_gpu_opt,gpu_opt_and_gpu_opt_pure_chunked}/job_notrecorded_legacy/`
- size: small, medium, large / SourceSnapshotID: `oldtree_f05ec52_20260512`（近似）/ PBS job 未記録
- 派生: `result/main_performance/seven_implementations/legacy_partial/`

正式参照 = `raw_data/RAW_DATA_INDEX.tsv`。主性能値（3.17 / 1.31 / 1.51 / 1.45）= `result/tables/final_speedup_tables.md`。
実験時コードは `../../code_snapshots/<SourceSnapshotID>/`。
