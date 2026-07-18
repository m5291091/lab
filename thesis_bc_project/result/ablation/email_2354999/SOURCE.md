# SOURCE — ablation email_2354999（H/W/A 8構成, email-EuAll）

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/ablation/email-EuAll/job_2354999_20260710/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

- **目的**: 要素分解（H/W/A）の寄与測定（ハブ有り実データ email-EuAll）
- **支える論文主張**: A 内部（提案手法がなぜ速いか）
- **グラフ**: email-EuAll
- **実装**: `Ablation_H{0,1}_W{0,1}_A{0,1}`（8構成）
- **要求バッチ / 実効バッチ**: 512 / 512（ログ確認, in-capacity）
- **SUB_BATCH**: `not_recorded` / **num_subs**: `not_recorded`
- **試行数**: 3（各構成）
- **warmup**: 各 `run_ablation <graph> all` invocation（=各 trial）の先頭で、**untimed な global warmup H1W1A1 を1回**実行（計測外・TSV不収録）。
  job 2354999 は 1 graph × 3 trial の3 invocation なので raw log の `=== Warmup (untimed, H1W1A1) ===` は3件、
  直後にH0W0A0〜H1W1A1の本試行8件が並び、TSVは8×3=24行。
  （Gate W13.1 訂正: 本項は当初「なし」と記載していたが、一次記録の再監査により誤りと確定した。
  実験時 runner `code_snapshots/phase_def_block_20260710/experiments/run_ablation.cu:111-118` の warmup 経路は
  mode に依存せず `BC_ABLATION_WARMUP=0` の指定時のみ抑止される経路であり、当該 job では抑止されていない。
  本試行値・主効果・trial数は変更なし。）
- **warmup一次根拠**: 実験時script `code_snapshots/phase_def_block_20260710/scripts/run_ablation.sh:88-108`（graph×trial ごとに runner を1回起動）、
  raw log `raw_data/ablation/email-EuAll/job_2354999_20260710/ablation.log`（marker 3件、行1/61/121 の invocation header 直後）、
  本試行TSV `raw_data/ablation/email-EuAll/job_2354999_20260710/ablation_results.tsv`（8構成×3=24行）、
  runner snapshot `code_snapshots/phase_def_block_20260710/experiments/run_ablation.cu:111-118`（warmup）
  および `:120-125`（本試行）。warmup は `run_brandes` を経由しないため stdout の TSV 行を生成せず、formal 行に混入しない。
- **集計方法**: median（trial summary のばらつきは Sample SD, ddof=1、n<2 は n/a）
- **SourceSnapshotID**: `phase_def_block_20260710`（測定 2026-07-10, 常時block化後）
- **PBS job ID**: 2354999
- **入力**: `data/snap/email-EuAll`
- **出力**: `ablation_results.tsv`, `ablation_contributions.tsv`, `ablation_summary.md`, `ablation.log`
- **正確性**: `none`（本 dir に max_bc ファイルなし）
- **再現コマンド**: `qsub scripts/run_ablation.sh`
- **制約**: build_miyabi（gitignore）からコピー。原本 `build_miyabi/result_ablation_20260710_182735_2354999/`。
