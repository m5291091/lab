# 325557_3216152 PathMerge 掃引 元データの出典

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/tuning/pathmerge/325557/pathmerge_bc/job_multi_20260710/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

`pathmerge_sweep_results.tsv` は build_miyabi (git 管理外) の実測を忠実に移送したもの。
比率逆算・補間・捏造は含まない。

| batch | n | 出典 (build_miyabi result ディレクトリ) |
|:--|:--:|:--|
| b32 | 2 | result_pathmerge_sweep_20260709_160712 (t1 は GTEPS 欄が破損していたため time から runner と同式で再計算) |
| b64/b256 | 1 | result_pathmerge_sweep_20260709_160712 |
| b512 | 4 | result_pathmerge_sweep_20260709_160712 (1) + result_pathmerge_sweep_20260710_183730_2355000 (3) |
| b1024/b2048 | 3 | result_pathmerge_sweep_20260710_183730_2355000 |
| b4096/b8192 | 3 | result_pathmerge_sweep_20260711_083659_2359081 |

中央値: b32=1292.27, b64=770.82, b256=324.27, b512=240.16, b1024=195.15,
b2048=175.43, **b4096=167.574 (最小)**, b8192(実効6018)=168.266。
b4096→b8192 は約 0.41% 悪化のみで内部最小に到達。最適 b4096。
