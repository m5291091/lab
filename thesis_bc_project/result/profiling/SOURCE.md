# SOURCE — profiling（帯域 + nsys）

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/profiling/job_2359175_20260711/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

- **目的**: HBM3/NVLink-C2C 帯域測定と nsys タイムライン（A=async 2-stream, UM prefetch）
- **支える論文主張**: A（帯域根拠）/ B（prefetch 隠蔽の定性観察）
- **グラフ**: nsys ablation H1W1A0/H1W1A1=`56438_300801` / nsys UM prefetch=`325557_3216152` / 帯域=`not_applicable`
- **実装**: `bandwidth_benchmark`（帯域）, `run_ablation`/`gpu_opt`（nsys）
- **要求バッチ / 実効バッチ / SUB_BATCH / num_subs**: `not_applicable`（帯域）/ nsys は実行時設定
- **試行数**: 1（各トレース） / **warmup**: ablation は untimed H1W1A1 を同一 process・trace 内に含む、UM prefetch は `not_recorded` / **集計方法**: `not_applicable`（トレース）
- **SourceSnapshotID**: `phase_def_block_20260710`
- **PBS job ID**: 2359175
- **入力**: 帯域=デバイス測定 / nsys=提案手法実行
- **出力**: `bandwidth.log`, `ablation_H1W1A0.{nsys-rep,stats.txt,console.log}`, `ablation_H1W1A1.{...}`, `um_prefetch_gpu_opt.{...}`
- **正確性**: `not_applicable`
- **再現コマンド**: `qsub scripts/run_profiling.sh`
- **制約（重要）**:
  - backward 63.9% / bfs 36.1% は `ablation_H1W1A0` の**単一トレース**における CUDA GPU カーネル時間のみの構成比（`56438_300801`）。本測定は H1W1A0 だが、同一 process 冒頭の untimed H1W1A1 warmup もtrace scopeに含む。別の `ablation_H1W1A1` または UM prefetch のトレース値ではなく、他グラフへ一般化しない。
  - **`um_prefetch_gpu_opt` は `--duration=25` の 25秒部分トレース**。HtoD migration 27.918 MB / CPU faults 85 / GPU faults 9 は**部分値であり全実行総量ではない**。
  - `.stats.txt` は `.nsys-rep` から `nsys stats` で再生成（H1W1A0/A1 同一レポート）。
  - `.sqlite`（再生成可）は Git 非追加（`../EXTERNAL_ARTIFACTS.tsv`）。
  - 帯域: HBM3 DtoD 1818.6 GB/s, NVLink-C2C Prefetch 177.7 GB/s。
  - 不正 271B `ablation_H1W1A1.stats.txt`（旧出力）は `../../raw_data/unsuccessful/failed/profiling/ablation_H1W1A1_incomplete/job_2359175_20260711/`（Gate J1 で `failure/incomplete/` から移動; 要約は `../../failure/README.md`）。
