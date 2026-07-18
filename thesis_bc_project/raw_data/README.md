# raw_data — Git履歴に依存しない生データアーカイブ

このディレクトリは、修士論文（GH200 上の厳密 Betweenness Centrality 実装比較）の
**生データ本体**を、**Git commit SHA を参照せずに** 内容・生成条件を判断できる形で保持する。

- **正式な参照先** = `RawPath`（このディレクトリ以下のパス）。Git 内 raw の正式索引は `RAW_DATA_INDEX.tsv`。
- 各生データの内容・生成条件は **`RawPath` + `MANIFEST.tsv` + `SHA256SUMS` + `code_snapshots/<SourceSnapshotID>/`** から判断できる。
- **commit SHA は生データアクセスの必須条件ではない**。過去の commit SHA と `SourceSnapshotID` の対応は
  `../code_snapshots/_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv`（raw 取得・再生成には不要な旧履歴対応表）にのみ保持する。
- 旧 `result/`・`failure/` 内 raw → 新 raw パスの移行対応は `../result/provenance/RAW_DATA_MIGRATION.tsv`。
- **Gate W7.4 追加**: 修正版325557 の再検証 raw は `corrected_325557/`（自己完結サブアーカイブ）に集約。
  job 2404743（Series A/B: 6ベクトル+10比較+容量境界）と job 2406254（Series C: 40行ablation）を
  `corrected_325557/job_<jobid>/` 配下に保持し、`corrected_325557/{README.md, MANIFEST.tsv, SHA256SUMS}`
  で自己完結の copy-integrity/検証を提供する（top-level `RAW_DATA_INDEX.tsv`/`MANIFEST.tsv`/`SHA256SUMS`
  にも 76 raw が追加され計 240 件）。gitignore された PBS `.o<jobid>` は内容不変で `pbs_stdout.log` に複製。

## パス命名規約

```
raw_data/<Experiment>/<Graph>/<Implementation>/<Configuration>/job_<PBSJobID>_<Date>/<file>
```

失敗データは `<Experiment>` の前に失敗種別が入る:

```
raw_data/unsuccessful/<oom|failed|early_terminated>/<Experiment>/<Graph>/<Implementation>/<Configuration>/job_<PBSJobID>_<Date>/<file>
```

ディレクトリ名・ファイル名だけで、実験・グラフ・実装・メモリ方式・バッチ・job ID・実験日が分かる。

### Configuration（メモリ方式 + バッチ）の読み方

| Configuration 接頭辞 | Implementation | メモリ方式 | 出典 |
|:--|:--|:--|:--|
| `um_b<N>`      | `gpu_opt`              | Unified Memory（managed, `host_um.cu`）           | `um_b512`, `um_b1024`, `um_b9792`, `um_b10240` |
| `pure_b<N>`    | `gpu_opt_pure`         | 明示的 `cudaMalloc`/`cudaMemcpy`（`host_pure.cu`） | `pure_b1024` |
| `chunked_b<N>` | `gpu_opt_pure_chunked` | 手動チャンク working set（`host_chunked.cu`）      | `chunked_b1024`, `chunked_b16384` |
| `pathmerge_b<N>` | `pathmerge_bc`       | Galliot path-merging（`src/baseline/pathmerge.cu`）| `pathmerge_b4096` |
| `seq`          | `sequential`           | CPU 逐次参照（バッチなし）                        | 独立参照 |
| `um_b1024_<control\|treset\|tnseff>` | `gpu_opt` | UM 診断（1因子スイッチ） | `BC_DIAG_FORCE_FULL_RESET=1`(treset) / `BC_DIAG_FORCE_NS_EFF_ONE=1`(tnseff) / none(control) |

`<N>` は `BC_BATCH_OVERRIDE`（gpu_opt 系）または `PATHMERGE_BC_BATCH_SIZE`（pathmerge_bc）で指定した要求バッチ。
実効バッチ・SUB_BATCH・num_subs・NS_eff は `DerivedResultPath` の stderr / execution_summary を参照。

## ファイル種別

- `*.bc.tsv` / `vector.bc.tsv` — 頂点別 BC 値（`node_idx<TAB>bc_value`, 先頭 1 行はヘッダ）。`--dump-bc` の stdout。
- `results.tsv` / `results_no_gpu_opt.tsv` — `Impl<TAB>Graph<TAB>…Time_sec<TAB>GTEPS`（ランナー stdout）。
- `phase_timing*.log` / `max_bc*.tsv` — フェーズ時間 / Max BC（ランナー stderr 由来）。
- `benchmark*.log` / `ablation.log` / `pathmerge_sweep.log` / `kernel_selection.log` / `um_experiment_*.log` — 実行ドライバ/ランナーの生ログ。
- `ablation_results.tsv` / `pathmerge_sweep_results.tsv` / `kernel_selection_results.tsv` / `oversubscribe_results_*.tsv` — 実測 TSV。
- `*.nsys-rep` / `*.stats.txt` / `*.console.log` / `bandwidth.log` — nsys プロファイル / 帯域測定。
- `*.stderr.log` / `stderr.log` / `run.log` — 実行 stdout/stderr。
- `MANIFEST.txt` — ジョブ生成マニフェスト（`checkpoint_sha`, `pbs_job_id`, tolerances 等; 内容不変の生ログ）。
- `pbs_stdout*.log` — PBS ジョブ stdout/stderr（元の `.o<jobid>` を内容変更せずコピー）。

## MANIFEST.tsv（列）

`RawPath`, `Experiment`, `Graph`, `Implementation`, `Configuration`, `Status`, `RunDate`,
`PBSJobID`, `SourceSnapshotID`, `OriginalPath`, `OriginalFilename`, `SizeBytes`, `SHA256`,
`UsedInThesis`, `DerivedResultPath`, `FailureSummaryPath`, `Notes`

- `SourceSnapshotID` — 実験時コードのスナップショット（`../code_snapshots/<SourceSnapshotID>/`）。
- `OriginalPath` — コピー元（`build_miyabi/…` は gitignore の外部生成物。移動・削除はしていない）。
- `DerivedResultPath` — この生データから導出された論文採用結果・比較（`result/…`）。
- `FailureSummaryPath` — 失敗要約（`failure/…`, 成功データは空）。

## SHA256 検証

```bash
cd thesis_bc_project/raw_data
sha256sum -c SHA256SUMS
```

全 `.bc.tsv` / `pbs_stdout` の SHA256 は `OriginalPath` の外部生成物と一致する
（コピー前後で一致確認済み。`result/EXTERNAL_ARTIFACTS.tsv` の記録値とも一致）。

## Experiment 別の内訳

| Experiment | raw の種別 | 説明 |
|:--|:--|:--|
| `correctness/small_full_vector` | `.bc.tsv` / stderr / run.log / MANIFEST.txt / pbs_stdout.log | 小グラフ独立参照 full-vector 正確性（job 2367583） |
| `correctness/memory_paths` | `.bc.tsv` / stderr / run.log / MANIFEST.txt / pbs_stdout | メモリ経路正確性 canonical（job 2368587）+ 診断（job 2369632） |
| `main_performance/proposed_variants` | `results.tsv` / `phase_timing.log` / `max_bc.tsv` / `benchmark.log` | 提案 block 再計測（UM/Pure/Chunked, job 2357334–2357337） |
| `main_performance/seven_implementations` | `results*.tsv` / `phase_timing*.log` / `max_bc*.tsv` / `benchmark*.log` | legacy 部分データ（旧 shared / 旧ツリー, size別） |
| `ablation` | `ablation_results.tsv` / `ablation.log` | H/W/A 8構成分解（job 2354994, 2354999） |
| `tuning/pathmerge` | `pathmerge_sweep_results.tsv` / `pathmerge_sweep.log` / `pbs_stdout.log` | PathMerge バッチ掃引（tuned 分母, job_multi 集約 + 各 job の起源 PBS ログ） |
| `tuning/kernel_selection` | `kernel_selection_results.tsv` / `.log` / `_max_bc.tsv` / `bc_*.txt` / `*.err` / `pbs_stdout.log` | BFS forced shared/block（job 2354329/2354330）+ phaseB block/shared BC ベクトル検証（job 2355971） |
| `memory_scalability` | `oversubscribe_results_*.tsv` / `um_experiment_*.log` | UM オーバーサブスクリプション feasibility（旧ツリー） |
| `profiling` | `*.nsys-rep` / `*.stats.txt` / `*.console.log` / `bandwidth.log` | 帯域 + nsys タイムライン（job 2359175） |
| `unsuccessful/oom` | `.bc.tsv`（空=証跡）/ stderr / run.log | UM b10240 の OOM（job 2368269） |
| `unsuccessful/early_terminated` | `.bc.tsv` / stderr / run.log / `pathmerge_sweep_*` | 比較不一致 fail-fast（job 2368398）+ 掃引早期打切り（2359080/2359096） |
| `unsuccessful/failed` | `ablation_H1W1A1.stats.txt`（271B 不正） | nsys stats 途中切れの不正出力 |

**Gate J1 以降、全 raw（Git 管理された既存 raw を含む）を本ディレクトリへ集約した。**
`result/` は派生表・図・集計・主張文書のみを保持する（`RAW_DATA_INDEX.tsv` が正式参照）。
旧 `result/`・`failure/` 内 raw → 新 raw パスの対応は `../result/provenance/RAW_DATA_MIGRATION.tsv`。

### Gate J1.1: 残存 external raw の追加救出（133 → 164）

Gate J1 で `result/EXTERNAL_ARTIFACTS.tsv` に残っていた external 42 件を全件監査し
（`../result/provenance/EXTERNAL_ARTIFACTS_AUDIT.tsv`）、次を本ディレクトリへ内容不変で救出した:

- **PBS 起源ログ 23 件** — 保持実験の `.o<jobid>`（gitignored）を対応 raw 実験dir へ `pbs_stdout.log` としてコピー。
  元ファイル名は `RAW_DATA_INDEX.tsv` / `MANIFEST.tsv` の `OriginalPath` 列に記録（真の job ID もここに保持;
  targeted の `_run` dir はバッチ代表 job 2357334 表記、真の job は `OriginalPath` の `.o<jobid>` と
  `../result/provenance/RAW_DATA_MIGRATION.tsv` の `PBSJobID`）。
- **phaseB kernel-selection BC ベクトル/err 8 件** — `tuning/kernel_selection/<Graph>/<block|shared>/gpu_opt/job_2355971_20260710/` へ原名コピー。

監査後、external に残すのは **再生成可能な .sqlite 3 件**（`../result/provenance/SQLITE_REGENERATION.tsv`; 追跡済み
`profiling/*.nsys-rep` から `nsys export --type sqlite` で再生成可, nsys 2025.5.1.121 で実証）と
**ビルド成果物ディレクトリの補助記録 1 件**のみ。build/smoke/superseded/旧ツリーの 7 件は
`EXCLUDE_NOT_USED`（論文非引用・データ非保持）として監査表に記録した。

## 重複について

`pathmerge_b4096` / `gpu_opt_pure_b1024` は job 2368587 / 2368398 / 2368269 で
**別 run（SHA256 相異）** であり、run-to-run 変動（Gate G2.2）の入力として各 job 配下に
保持している（同一データの複製ではない）。
`RAW_DATA_INDEX.tsv` / `MANIFEST.tsv` / `SHA256SUMS` の SHA256 は移行前後で一致（内容不変移動）。

### 意図的な同一 SHA256（screening run の二役; **意図しない複製ではない**）

早期打切りの screening run の出力は、**チューニング入力**と**早期打切り証跡**の 2 つの
実験ビューから参照されるため、同一 SHA256 が 2 箇所に存在する（元 commit `08204f8` でも
同一 blob。J1 で新規に複製したものではない）:

| SHA256 (先頭) | 役割A (tuning 入力) | 役割B (早期打切り証跡) |
|:--|:--|:--|
| `4ac33b39…` | `tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/email_smallbatch_trial1.tsv` | `unsuccessful/early_terminated/pathmerge_sweep/email-EuAll/job_2359096_20260711/pathmerge_sweep_results.tsv` |
| `59418cba…` | `tuning/pathmerge/roadNet-PA/pathmerge_bc/job_multi_20260710/pathmerge_sweep.log` | `unsuccessful/early_terminated/pathmerge_sweep/roadNet-PA/job_2359080_20260711/pathmerge_sweep.log` |

役割A は `merge_final_tables.py` の掃引入力（screening 値を tuned 掃引に取込）、役割B は
job 2359080 / 2359096（意図的早期打切り）の完全な生出力証跡。両者は同一 screening run に
由来し、実験的役割が異なるため各ビューに保持する（`failure/MANIFEST.tsv` が役割B を参照）。
これ以外に同一 SHA256 の raw 重複は存在しない。
