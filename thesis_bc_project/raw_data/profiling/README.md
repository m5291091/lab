# raw_data/profiling

プロファイリング実験（HBM3/NVLink-C2C 帯域測定、nsys タイムライン）の **raw**。

- 配置: `profiling/job_2359175_20260711/`
  - `bandwidth.log`（帯域: HBM3 DtoD 1818.6 GB/s, NVLink-C2C Prefetch 177.7 GB/s）
  - `ablation_H1W1A0.{nsys-rep,stats.txt,console.log}`, `ablation_H1W1A1.{...}`, `um_prefetch_gpu_opt.{...}`
- SourceSnapshotID: `phase_def_block_20260710` / PBS job 2359175
- `.sqlite` は `.nsys-rep` から `nsys stats` で再生成可のため Git 非追加（`result/EXTERNAL_ARTIFACTS.tsv`）
- 不正 271B の旧 `ablation_H1W1A1.stats.txt` は `../unsuccessful/failed/profiling/`（失敗証跡）
- 派生: `result/profiling/SOURCE.md`

正式参照 = `raw_data/RAW_DATA_INDEX.tsv`。実験時コードは `../../code_snapshots/phase_def_block_20260710/`。
