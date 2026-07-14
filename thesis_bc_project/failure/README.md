# failure/ — 失敗・早期終了・不完全・置換済データの**要約**

Gate B 整理で `result/`（正常・論文使用データ）から分離した非正常データ。
**Gate J1 以降、失敗の raw（run.log / stderr / MANIFEST.txt / 掃引生出力 / 空ベクトル / 不正 stats 等）は
`raw_data/unsuccessful/` へ集約した。`failure/` は失敗要約と raw への参照のみを保持する。**

分類規則:
- `failed/` … 実行失敗（build / runtime / oom / timeout）
- `early_terminated/` … 意図的早期終了
- `incomplete/` … 空・欠損・不正出力
- `superseded_success/` … 正常だが新結果に置換済（派生表）

各エントリの出所・SHA256・job・`SourceSnapshotID`・**新 raw パス**は `MANIFEST.tsv`（`Path` 列＝現在の所在）を参照。
raw の正式索引は `../raw_data/RAW_DATA_INDEX.tsv`、旧→新対応は `../result/provenance/RAW_DATA_MIGRATION.tsv`。

## failure/ に残す要約（raw は raw_data/unsuccessful/）
- `failed/oom/memory_correctness_2368269/correctness_summary.tsv` … OOM 実行の要約
- `early_terminated/memory_correctness_2368398/{correctness_summary.tsv, gpu_opt_pure_b1024_vs_reference.md}` … fail-fast 要約・不一致比較 md
- `superseded_success/**` … 置換済の旧 headline 派生表（派生物; raw ではない）
- `MANIFEST.tsv`, `README.md`

## カテゴリ別の状況
### failed/{build,runtime,timeout}: **該当なし**
- 実行失敗（コンパイル失敗・実行時クラッシュ・タイムアウト）は検出されていない。
- PBS `.o` の `Timeout: 21600s` はジョブ**設定行**であり、実タイムアウトではない。空のため Git 非保存。

### failed/oom: memory_correctness_2368269（UM b10240 が 100 GiB queue でOOM）
- job `2368269`（`SourceSnapshotID=memory_correctness_oom_20260712`）: 325557 の正確性実行で `gpu_opt` UM `b10240`（dynamic(UM)=213.38 GB）がホスト常駐 >100 GiB で SIGKILL（runner_exit=137）。PathMerge b4096（reference）は成功。
- **100 GiB queue のホストメモリ上限に起因する当該正確性実行の失敗**であり、`data/325557`・b10240 に限定。mem 上限のないノードでは legacy 実験で b10240 は SUCCESS（`raw_data/memory_scalability/`）。
- raw（空ベクトル `gpu_opt_b10240.bc.tsv`＝OOM 証跡・stderr・run.log・MANIFEST.txt・pbs_stdout）は
  `raw_data/unsuccessful/oom/memory_paths/325557_3216152/` に保持。failure/ には `correctness_summary.tsv` のみ。

### failed/oom（意図的 feasibility 境界）: raw_data/memory_scalability/ が正
- UM 掃引実験の OOM は**意図的な feasibility 境界の証拠**で、`raw_data/memory_scalability/`（SUCCESS と同居）に保持（**failure へデータ複製しない**）。
- `MANIFEST.tsv` に `ExpectedFailure=yes, UsedInThesis=yes, Path=raw_data/memory_scalability/...` の参照行のみ記録。

### early_terminated/: PathMerge掃引の意図的早期打切り（job 2359080 / 2359096）
- roadNet-PA(b8/16/32) と email-EuAll は、探索方向確定後の**意図的な早期打切り**（失敗ではない）。
- trial1 実測値は `raw_data/tuning/pathmerge/` に取り込み済み。打切りジョブの生出力は
  `raw_data/unsuccessful/early_terminated/pathmerge_sweep/<graph>/job_<jid>_20260711/` に保持。

### early_terminated/: memory_correctness_2368398（比較不一致による fail-fast）
- job `2368398`（`SourceSnapshotID=memory_correctness_failfast_20260712`）: 実行順 pathmerge_b4096 → gpu_opt_pure_b1024 → chunked_b16384 → gpu_opt_b9792。
  reference の PathMerge b4096 は成功、続く `gpu_opt_pure_b1024` の full-vector 比較が混合許容で不一致 11027 件（max_rel≈2.0e-3, comparison_exit=3）となり、**fail-fast で残り構成（chunked/UM）は未実行**。
- `Reason=comparison mismatch caused fail-fast before remaining configurations`。**「Pure runner 失敗」や「OOM」とは誤記しない**（runner は exit0、失敗したのは比較判定）。
- raw（ベクトル・stderr・run.log・MANIFEST.txt・pbs_stdout）は `raw_data/unsuccessful/early_terminated/memory_paths/325557_3216152/` に保持。failure/ には `correctness_summary.tsv` と `gpu_opt_pure_b1024_vs_reference.md`（比較 md）のみ。canonical（成功した後続構成の等価データ）は job `2368587`（`raw_data/correctness/memory_paths/`）。

### incomplete/: 271バイト不正 stats（`ablation_H1W1A1.stats.txt`）
- nsys stats 生成が途中で切れた**不正な旧出力**（271B）。
- raw は `raw_data/unsuccessful/failed/profiling/ablation_H1W1A1_incomplete/job_2359175_20260711/` に保持。
  正版は `raw_data/profiling/job_2359175_20260711/ablation_H1W1A1.stats.txt`（5197B, `.nsys-rep` から再生成）。

### superseded_success/: 置換済の旧 headline 派生表（派生物; raw ではない）
- 旧 final table（`final_speedup_tables_OLD.md`, roadNet-CA tuned=b64 1.64× の旧世代）→ 現行 `result/tables/final_speedup_tables.md`（CA tuned=b32 1.45×）に置換。
- legacy large の旧 shared headline 派生表（speedup/summary）→ 現行 block（`result/main_performance/proposed_variants/` + `result/tables/`）に置換。
- **生 TSV は置換していない**（`raw_data/main_performance/seven_implementations/legacy_partial/` に保存, baseline/provenance）。置換したのは headline 派生表のみ。
