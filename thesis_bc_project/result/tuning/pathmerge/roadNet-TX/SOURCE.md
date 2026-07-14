# roadNet-TX PathMerge 掃引 元データの出典 (実測)

> **生データ所在 (Gate J1)**: この文書が説明する派生結果の raw は `raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_multi_20260710/` に集約（正式参照 = `raw_data/RAW_DATA_INDEX.tsv`）。`result/` には派生表・図・要約のみを保持する。旧 `result/` 内 raw → 新 raw パスの対応は `result/provenance/RAW_DATA_MIGRATION.tsv`。

checkpoint phase_def_block_20260710 のバイナリ (SKIP_BUILD=1) で GH200 実測。捏造・補間・比率逆算なし。

| batch | n | 出典 job | 内訳 |
|:--|:--:|:--|:--|
| b32 | 3 | screening 2360072 (t1) + confirmation 2361040 (t2,t3) | 1620.96, 1615.62, 1631.80 |
| b64 | 3 | screening 2360072 (t1) + confirmation 2361040 (t2,t3) | 1493.69, 1491.13, 1466.16 |
| b128 | 1 | screening 2360072 | 1668.68 (内部最小判定用) |

中央値: b32=1620.96, **b64=1491.13 (最小)**, b128=1668.68 [秒]。
内部最小 (b32>b64<b128) のため **最適バッチ = b64** (実測 n=3)。
b64 vs b32 の差 8.7% (>3%) のため保守ルール適用外。

## tuned と既定 b64 の関係 (重要)

掃引で確認した TX の最適設定は **b64** であり、これは PathMerge 既定 (legacy) の b64 と
**同一のバッチ設定**である。両者は独立に測定した値で数値がわずかに異なる:

| 出所 | バッチ | 中央値 [s] | 用途 |
|:--|:--|--:|:--|
| 本掃引 (checkpoint phase_def_block_20260710, n=3) | b64 | 1491.13 | 最適設定が b64 であることの**確認測定** |
| legacy result_paper (既定, n=3) | b64 | 1482.68 | 最終表の PathMerge 既定/tuned に使用する実測値 |

最終表では、**同一の b64 設定**であるため tuned にも既定の実測値 1482.68s を用いる
(`merge_final_tables.py` は掃引最良と既定 b64 の速い方を採用 → 1482.68s)。掃引の 1491.13s は
「最適設定が b64 である」ことを確認した別測定であり、最終表の tuned 値そのものではない。
どちらも b64 のため、この選択は tuned を悪化させない。
