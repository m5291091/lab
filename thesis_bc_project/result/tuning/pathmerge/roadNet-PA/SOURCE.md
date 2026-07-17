# roadNet-PA PathMerge 掃引 元データの出典

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/tuning/pathmerge/roadNet-PA/pathmerge_bc/job_multi_20260710/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

`pathmerge_sweep_results.tsv` は以下の実測 (build_miyabi, git 管理外) から移送した実測値である。
比率からの逆算・補間・捏造は一切含まない。各バッチの試行数 (n) は下表の通り。

| batch | n | 出典 (build_miyabi の元 result ディレクトリ) |
|:--|:--:|:--|
| b8/b16/b32 | 1 | result_pathmerge_sweep 小バッチ screening (trial1) |
| b64 | 4 | result_pathmerge_sweep_20260710_193014_2355001 (trial1-3) + result_pathmerge_sweep_20260709_160712 (追加1試行) |
| b128/b256/b512 | 3 | result_pathmerge_sweep_20260710_193014_2355001 (trial1-3) |

中央値 (再現性検証用): b8=2714.98, b16=1573.87, b32=1015.98, **b64=941.39 (最小)**,
b128=1105.57, b256=1155.30, b512=1207.41 [秒]。
最適バッチは b64。PathMerge 既定 (legacy result_paper b64) の中央値 918.67s と比較し、
tuned は速い方 (918.67s, b64) を採用する。

## 正確性証拠の範囲

tuned と default はともに b64 で、最終表の tuned 値には default と同じ legacy b64 測定を採用する。
掃引には別の timing 実行が含まれるが、PA 用の `--dump-bc` full vector、vector A/B の path・SHA256、
全 index の comparison summary は保存されていない。したがって b64 を自分自身と比較したものとして
full-vector 検証へ格上げせず、正式な `CorrectnessLevel` は、legacy の PathMerge と GPU_Opt_Pure の
Max BC index/value 一致記録に基づく `max_bc_only` とする。

Tuned and default configurations both use b64; no distinct full-vector comparison artifact is available.
