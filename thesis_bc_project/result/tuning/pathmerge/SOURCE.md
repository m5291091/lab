# SOURCE — tuning/pathmerge（PathMerge バッチ掃引 = tuned 分母）

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/tuning/pathmerge/<graph>/pathmerge_bc/job_multi_20260710/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

各グラフ別の詳細出典・n・batch別中央値は **各グラフ dir の `SOURCE.md`** を参照。
roadNet-PA / roadNet-TX / roadNet-CA / 325557 の 4 件は掃引時に同梱されたもの、
email-EuAll の 1 件は Gate W14.1 で raw（`raw_data/tuning/pathmerge/email-EuAll/`、
`raw_data/unsuccessful/early_terminated/pathmerge_sweep/email-EuAll/`）から再構成したものであり、
新規の測定値は含まない。

各グラフの stage 別 job 対応は次のとおり（一次記録 = 各 job の PBS stdout の `Graphs :` 行と
`raw_data/MANIFEST.tsv` の Graph 列で相互確認済み）。

| Graph | Screening | Confirmation | その他 |
|:--|:--|:--|:--|
| email-EuAll | 2359096（早期打切り, trial1 のみ） | 2359169 | 正確性 dump 2360074 |
| roadNet-PA | 2359080（早期打切り, trial1 のみ） | 2355001 | b64 追加 1 試行の job ID は `Not recorded` |
| roadNet-TX | 2360072 | 2361040 | — |
| roadNet-CA | 2360073 | 2362006 | extension 2361041 (b16); 正確性 dump 2362965 |
| 325557（historical） | 2355000 | 2359081 | 初期実行の job ID は `Not recorded` |

- **目的**: PathMerge(Galliot) のバッチサイズ掃引で tuned（グラフ別最適）を確定（主軸(A)の分母）
- **支える論文主張**: A 性能（PathMerge tuned 基準）
- **グラフ**: roadNet-PA, roadNet-TX, roadNet-CA, email-EuAll, 325557_3216152
- **実装**: PathMerge_BC（`src/baseline/pathmerge.cu` + `galliot.cu` + `galliot_kernels.cu`）
- **要求バッチ / 実効バッチ**: グラフ別掃引範囲（PA b8–512 / TX b32–128 / CA b16–128 / email b8–8192 / 325557 b32–8192）。**325557 b8192 → 実効 6018 にクランプ**（HBM3 予算超過, ログ警告）
- **SUB_BATCH / num_subs**: `not_applicable`（PathMerge は int2 frontier + per-source 配列 [batch×N]、サブバッチ分割なし）
- **試行数**: batch 毎 1〜4（掃引探索のため不均一。各 SOURCE.md に n 記録）
- **warmup**: なし / **集計方法**: median
- **SourceSnapshotID**: `phase_def_block_20260710`
- **PBS job ID**: 2355000, 2355001, 2359080, 2359081, 2359096, 2359169, 2360072, 2360073, 2361040, 2361041, 2362006
- **入力**: `data/snap/{roadNet-*,email-EuAll}`, `data/325557_3216152`
- **出力**: `<graph>/pathmerge_sweep_results.tsv`, `<graph>/pathmerge_sweep.log`, `<graph>/SOURCE.md`
- **正確性**: tuned vs 既定 b64 の全ベクトル一致は `../../correctness/pathmerge_tuned/`（email, roadNet-CA）
- **再現コマンド**: `qsub scripts/run_pathmerge_sweep.sh`（`BATCH_LIST`, `TRIALS`）
- **制約**: tuned 実測 = PA/TX b64（既定と同一設定）/ CA b32 / email b2048。PA/TX の掃引確認値≠最終採用値（保守的 default b64 採用, `../../main_performance/proposed_vs_pathmerge/README.md`）。意図的早期打切り(2359080/2359096)は `../../../failure/early_terminated/`。
