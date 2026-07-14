# roadNet-CA PathMerge 掃引 元データの出典 (実測)

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

checkpoint phase_def_block_20260710 のバイナリ (SKIP_BUILD=1) で GH200 実測。捏造・補間・比率逆算なし。

| batch | n | 出典 job | 内訳 |
|:--|:--:|:--|:--|
| b16 | 1 | ca_b16 2361041 | 3609.95 (内部最小判定用の 1 段延長) |
| b32 | 3 | screening 2360073 (t1) + confirmation 2362006 (t2,t3) | 3111.18, 3079.72, 3060.66 |
| b64 | 3 | screening 2360073 (t1) + confirmation 2362006 (t2,t3) | 3588.39, 3490.24, 3491.64 |
| b128 | 1 | screening 2360073 | 3830.86 (内部最小判定用) |

中央値: b16=3609.95, **b32=3079.72 (最小)**, b64=3491.64, b128=3830.86 [秒]。
掃引形状: b16 > **b32** < b64 < b128 → 内部最小 b32。**最適バッチ = b32** (実測 n=3)。
roadNet-CA では b32 が実測最適となり、PA/TX から推定した b64 最適は CA には一般化しなかった
(b32 vs b64 差 13.4% >3%)。
PathMerge 既定 (legacy b64) 3499.03s より b32 (3079.72s) が速いため、**tuned = b32 (3079.72s)**。
