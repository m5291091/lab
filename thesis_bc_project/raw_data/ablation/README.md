# raw_data/ablation

Ablation 実験（H=hybrid BFS / W=warp 協調累積 / A=async 2-stream init の 8 構成分解）の **raw**。

- 配置: `ablation/<graph>/job_<jid>_20260710/{ablation_results.tsv, ablation.log}`
  - `synthetic/job_2354994_20260710/`（合成4グラフ, PBS 2354994）
  - `email-EuAll/job_2354999_20260710/`（PBS 2354999）
- SourceSnapshotID: `phase_def_block_20260710`（`src/proposed/host_ablation.cu`, コンパイル時テンプレート）
- 派生（要約・寄与分解）: `result/ablation/<dir>/{ablation_summary.md, ablation_contributions.tsv}`

正式参照 = `raw_data/RAW_DATA_INDEX.tsv`。実験時コードは `../../code_snapshots/phase_def_block_20260710/`。
