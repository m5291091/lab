# SOURCE — memory_scalability（副次(B) UM オーバーサブスクリプション）

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/memory_scalability/325557_3216152/<impl>/job_notrecorded_20260512/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

- **目的**: HBM3 容量を超える working set での実行可能性（pure/UM/chunked 3者 feasibility）
- **支える論文主張**: B メモリスケーラビリティ（feasibility のみ。時間値は非採用）
- **グラフ**: 325557_3216152（合成, 人為的バッチ強制で oversubscribe）
- **実装**: gpu_opt(UM) / gpu_opt_pure(single cudaMalloc) / gpu_opt_pure_chunked（手動 chunk）
- **要求バッチ / 実効バッチ**: 512–16384 / 動的（oversub 時 SUB_BATCH<Batch）。例: gpu_opt b8192 → SUB_BATCH=6596, num_subs=2, NS_eff=1（ログ確認）
- **SUB_BATCH**: batch 依存（in-capacity は =Batch, oversub は動的縮小）。ログ `[Mem] SUB_BATCH=...` 参照
- **num_subs**: batch 依存（in-capacity 1, oversub >1。例 b8192=2）
- **試行数**: 5（各 batch）
- **warmup**: なし
- **集計方法**: median（+ SUCCESS/OOM Status 行列）
- **SourceSnapshotID**: **測定=`oldtree_f05ec52_20260512`（2026-05-12, 旧ツリー）**（phase_def_block_20260710 ではない）
- **PBS job ID**: UMv2（旧ツリー; PBS job ID は当時ログに個別記録なし → `not_recorded`）
- **入力**: `data/325557_3216152`
- **出力**: `oversubscribe_results_gpu_opt{,_pure,_pure_chunked}.tsv`, `um_experiment_*.log`
- **正確性**: `max_bc_only`（log `Maximum Betweenness Centrality ==> 39343001000.11` = 独立参照一致。FP順序差3値 ≤5.7e-9。**全ベクトル未照合**）
- **再現コマンド**: `qsub scripts/run_um_oversubscribe_experiment.sh`（現行 checkpoint 版）
- **制約（重要）**:
  - **時間値は最新 block 性能値として使用しない**（旧セッション未再検証; `../provenance/um_code_diff_audit.md`）。
  - **feasibility(SUCCESS/OOM)を限定的に採用**（メモリサイジングコードが checkpoint と文字単位同一のため再利用可。ただし phase_def_block_20260710 で境界を再実測したものではない）。
  - feasibility: pure OOM@b8192+ / UM→b10240(b12288 OOM) / chunked→b16384 全SUCCESS。「UM 無制限」は偽。
  - Chunked の主効果は最高性能でなく**実行可能バッチの拡大**。GH200・325557・試験バッチ範囲に限定。
