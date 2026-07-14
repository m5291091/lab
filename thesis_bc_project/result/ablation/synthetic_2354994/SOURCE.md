# SOURCE — ablation synthetic_2354994（H/W/A 8構成, 合成4グラフ）

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/ablation/synthetic/job_2354994_20260710/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

- **目的**: 提案手法の要素分解（H=hybrid BFS, W=warp協調累積, A=async 2-stream init）の寄与測定
- **支える論文主張**: A 内部（提案手法がなぜ速いか）
- **グラフ**: benchmark_7000_41459, benchmark_11023_62184, 56438_300801, 325557_3216152
- **実装**: `Ablation_H{0,1}_W{0,1}_A{0,1}`（8構成, コンパイル時テンプレート）
- **要求バッチ / 実効バッチ**: 512 / 512（ログ `[Ablation ...] BATCH=512` 確認）
- **SUB_BATCH**: `not_recorded`（ablation.log に SUB_BATCH 出力なし。in-capacity）
- **num_subs**: `not_recorded`
- **試行数**: 5（各構成×グラフ）
- **warmup**: なし
- **集計方法**: median（寄与は `ablation_contributions.tsv` の MainEffect/InteractionRel）
- **SourceSnapshotID**: `phase_def_block_20260710`（測定 2026-07-10, 常時 block 化後）
- **PBS job ID**: 2354994
- **入力**: `data/{benchmark_7000_41459,benchmark_11023_62184,56438_300801,325557_3216152}`
- **出力**: `ablation_results.tsv`, `ablation_contributions.tsv`, `ablation_summary.md`, `ablation.log`
- **正確性**: `none`（本 dir に max_bc ファイルなし。ablation.log 内に Max BC 出力はあるが未集計）
- **再現コマンド**: `qsub scripts/run_ablation.sh`（`./run_ablation <graph> all`）
- **制約**: build_miyabi（gitignore）からコピー。原本 `build_miyabi/result_ablation_20260710_182735_2354994/`。H が最大寄与（MainEffect 1.40〜2.17×）。
