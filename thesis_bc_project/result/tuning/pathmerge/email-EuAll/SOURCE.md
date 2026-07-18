# email-EuAll PathMerge 掃引 元データの出典 (実測)

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

checkpoint `phase_def_block_20260710` のバイナリで GH200 実測。捏造・補間・比率逆算なし。

- **グラフ**: email-EuAll (`data/snap/email-EuAll`, nodes=265009, edges=364481, Symmetrized=yes)
- **実装**: PathMerge_BC（`src/baseline/pathmerge.cu` + `galliot.cu` + `galliot_kernels.cu`）
- **実験種別**: pathmerge_sweep(tuned) — 主軸(A) の分母確定
- **screening job**: 2359096（要求 b8,16,64,256,1024 / 各 n=1 / **意図的早期打切り**、trial 1 のみ完了）
- **confirmation job**: 2359169（要求 b512,1024,2048,4096,8192 / 各 n=3）
- **要求バッチ**: 8, 16, 64, 256, 512, 1024, 2048, 4096, 8192（9 種）
- **実効バッチ**: b8–b4096 は要求値と同一（保存ログの `batch_size=` 行で確認）。**b8192 → 実効 7393 にクランプ**
- **clamp**: `WARNING: batch_size=8192 exceeds HBM3 budget; clamping to 7393 (free=101.4 GB, 11660396 B/source)`、num_batches=36、全 3 trial に記録
- **SUB_BATCH / num_subs**: `not_applicable`（PathMerge は int2 frontier + per-source 配列 [batch×N]、サブバッチ分割なし）
- **試行数**: screening 各 1、confirmation 各 3。**b1024 のみ両 stage に出現**（screening 1 + confirmation 3 = pooled 4）
- **warmup**: なし / **集計方法**: median
- **SourceSnapshotID**: `phase_def_block_20260710`
- **正確性**: `full_vector_same_implementation`（b64 対 b2048、job 2360074、`../../../correctness/pathmerge_tuned/`）。比較 vector 本体は `currently_unavailable`（台帳 = `EXTERNAL_ARTIFACTS.tsv`、比較 summary のみ保存）
- **再現コマンド**: `qsub scripts/run_pathmerge_sweep.sh`（`BATCH_LIST`, `TRIALS`）
- **PathMerge provenance**: 上流 `gobardhanm/path-merging-bc`（評価時 snapshot `9c231b46`）の第三者実装を adapter 化したもの。原著者の公式実装とは確認されていない。**external comparator であり ground truth ではない**（`../../../../docs/thesis/SOURCE_AUDIT.tsv`、`result/CLAIMS.md`）

## batch 別中央値 (raw から再計算)

| stage | 要求 batch | 実効 batch | n | median [s] |
|:--|--:|--:|--:|--:|
| screening | 8 | 8 | 1 | 786.91 |
| screening | 16 | 16 | 1 | 491.01 |
| screening | 64 | 64 | 1 | 226.05 |
| screening | 256 | 256 | 1 | 125.91 |
| screening | 1024 | 1024 | 1 | 97.69 |
| confirmation | 512 | 512 | 3 | 106.43 |
| confirmation | 1024 | 1024 | 3 | 99.93 |
| confirmation | **2048** | **2048** | 3 | **97.80 (最小)** |
| confirmation | 4096 | 4096 | 3 | 101.58 |
| confirmation | 8192 | 7393 (clamp) | 3 | 103.27 |
| pooled (screening+confirmation) | 1024 | 1024 | 4 | 99.86 |

掃引形状は b8 から b2048 まで単調に短縮し、b4096・b8192 で再び長くなる内部最小である。
**最適バッチ = b2048（97.80 s, n=3）**。PathMerge 既定（legacy b64）の 220.39 s より速いため、
**tuned = b2048 (97.80 s)** を最終表の分母として採用する（`scripts/merge_final_tables.py`）。

`result/tables/final_speedup_tables.md` が b1024 を `n=4` と要約しているのは、上表の
pooled 行（screening 1 + confirmation 3）に対応する記述統計であり、単一 job で 4 試行を
測定したものではない。いずれの粒度でも b1024 は最良ではなく、tuned 選択（b2048）は変わらない。

## canonical artifacts

| 区分 | パス |
|:--|:--|
| raw TSV (screening) | `raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/email_smallbatch_trial1.tsv` |
| raw TSV (confirmation) | `raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv` |
| raw log (confirmation) | `raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/pathmerge_sweep.log` |
| raw log (screening) | `raw_data/unsuccessful/early_terminated/pathmerge_sweep/email-EuAll/job_2359096_20260711/pathmerge_sweep.log` |
| PBS stdout (confirmation) | `raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_2359169_20260711/pbs_stdout.log` |
| PBS stdout (screening) | `raw_data/unsuccessful/early_terminated/pathmerge_sweep/email-EuAll/job_2359096_20260711/pbs_stdout.log` |
| PBS stdout (correctness dump) | `raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_2360074_20260711/pbs_stdout.log` |
| 派生 (最終比較) | `result/main_performance/proposed_vs_pathmerge/comparison.tsv` |
| 派生 (全 trial 表) | `docs/thesis/writing/japanese/appendix_b_pathmerge_batch_sweeps.md`（B.3 節） |

## 制約

- screening（job 2359096）は trial 1 完了後、trial 2 の b8 実行中に終了しており、trial 2・trial 3 の記録は存在しない。`result/tuning/pathmerge/SOURCE.md` に意図的早期打切りとして記録されるが、**終了機構そのものは保存ログから独立に確認できない**（`Cause not independently confirmed`）。OOM の記録はなく、timeout と断定できる証拠もない。記録のない trial を 0 秒として集計しない。
- b8192 の clamp は実装がメモリ予算に合わせてバッチを縮小した助言的警告であり、OOM ではない。
- screening の b64（226.05 s）は、同一 b64 の legacy default 測定（220.39 s, n=5, checkpoint `oldtree_f05ec52_20260512`）とは別 checkpoint の別測定であり、同一系列として結合しない。
- 本掃引の結果は、保存 snapshot・email-EuAll・GH200 環境に限定される。PathMerge/Galliot アルゴリズム一般や原著者の公式実装へ一般化しない。
