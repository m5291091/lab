# raw_data/tuning

チューニング実験の **raw**（実測 TSV・生ログ）。派生要約は `result/tuning/`・`result/correctness/pathmerge_tuned/` に保持。

## pathmerge（PathMerge バッチ掃引, tuned 分母）
- 配置: `pathmerge/<graph>/pathmerge_bc/job_multi_20260710/{pathmerge_sweep_results.tsv, pathmerge_sweep.log, email_smallbatch_trial1.tsv}`
- graph: 325557, email-EuAll, roadNet-PA, roadNet-TX, roadNet-CA
- `job_multi` = 複数 PBS job から組成（2355000,2355001,2359081,2359169,… 詳細は `result/tuning/pathmerge/<graph>/SOURCE.md`）
- 意図的早期打切りの掃引は `../unsuccessful/early_terminated/pathmerge_sweep/`

## kernel_selection（BFS forced shared/block 比較）
- 配置: `kernel_selection/<graph>/gpu_opt_forced/job_<jid>_20260710/{kernel_selection_results.tsv, kernel_selection.log, kernel_selection_max_bc.tsv}`
- graph/job: roadNet-PA=2354329, roadNet-TX=2354330 / SourceSnapshotID `phase_def_block_20260710`
- 派生: `result/tuning/kernel_selection/<graph>/kernel_selection_summary.md`

正式参照 = `raw_data/RAW_DATA_INDEX.tsv`。実験時コードは `../../code_snapshots/phase_def_block_20260710/`。
