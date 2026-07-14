# SOURCE — proposed_variants（提案手法 block 再計測）

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/main_performance/proposed_variants/<graph>/_run/job_2357334_20260711/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

- **目的**: 提案手法（常時 block カーネル）を UM/Pure/Chunked の3実装で再計測（主軸(A)の分子）
- **支える論文主張**: A 性能（vs PathMerge tuned）
- **グラフ**: email-EuAll, roadNet-PA, roadNet-TX, roadNet-CA
- **実装**: GPU_Opt(UM) / GPU_Opt_Pure(cudaMalloc) / GPU_Opt_Pure_Chunked（すべて block）
- **要求バッチ / 実効バッチ**: 512 / 512（in-capacity, ログ `[Mem] BATCH=512` 確認）
- **SUB_BATCH**: 512（ログ `[Mem] SUB_BATCH=512` 確認）
- **num_subs**: 1（ログ `num_subs=1`, `NS_eff=2` 確認）
- **試行数**: email-EuAll = 5, roadNet-* = 3（results.tsv 実測行数で確認）
- **warmup**: なし（`scripts/run_benchmark_targeted.sh` は全 TRIALS を記録・discard なし）
- **集計方法**: median
- **SourceSnapshotID**: `phase_def_block_20260710`
- **PBS job ID**: 2356120（gpu_opt targeted）, 2357334 / 2357335 / 2357336 / 2357337（3実装 targeted）
- **入力**: `data/snap/{email-EuAll,roadNet-PA,roadNet-TX,roadNet-CA}`
- **出力**: `results.tsv`, `phase_timing.log`, `max_bc.tsv`, `benchmark.log`, `correctness.md`, `summary_table.{md,tsv}`, `speedup_table.md`
- **正確性**: `max_bc_only`（`correctness.md`: 提案3実装間の Max BC 一致。独立参照 PathMerge の Max BC とも一致: PA=151395302679.08, CA=686380725021.27）
- **再現コマンド**: `qsub -v 'GRAPHS_STR=snap/roadNet-CA,TRIALS=3,SKIP_BUILD=1' scripts/run_benchmark_targeted.sh`
- **制約**: in-capacity のみ（oversubscribe 経路は本実験では非該当）。独立参照との全ベクトル比較は未実施（→ `../../CLAIMS.md`）
