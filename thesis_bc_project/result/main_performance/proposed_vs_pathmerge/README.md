# proposed_vs_pathmerge — 主軸(A) 最終比較（提案 block vs PathMerge tuned）

**生データは重複コピーしない。** canonical raw は下記に一度だけ保存され、本フォルダは比較表とリンクのみ。

- 提案手法 (block, UM) canonical raw: `../proposed_variants/<graph>/results.tsv`
- PathMerge tuned canonical raw:
  - 掃引実測: `../../tuning/pathmerge/<graph>/pathmerge_sweep_results.tsv`
  - 既定 b64 実測(legacy baseline): `../seven_implementations/legacy_partial/large/results_no_gpu_opt.tsv`
- 最終表: `../../tables/final_speedup_tables.md`（`scripts/merge_final_tables.py` 生成）

## 最終比較表（中央値・元TSVから再生成・変更禁止）

| グラフ | 提案(block)[s] | n | PathMerge tuned[s] | batch | n | speedup | provenance |
|:--|--:|:--:|--:|:--:|:--:|--:|:--|
| email-EuAll | 30.81 | 5 | 97.80 | b2048 | 3 | **3.17×** | 提案=proposed_variants; tuned=tuning/pathmerge(掃引最良) |
| roadNet-PA | 699.52 | 3 | 918.67 | b64 | 3 | **1.31×** | 提案=proposed_variants; tuned=legacy 既定 b64 実測 |
| roadNet-TX | 980.13 | 3 | 1482.68 | b64 | 3 | **1.51×** | 提案=proposed_variants; tuned=legacy 既定 b64 実測 |
| roadNet-CA | 2129.10 | 3 | 3079.72 | b32 | 3 | **1.45×** | 提案=proposed_variants; tuned=tuning/pathmerge(掃引最良) |

集計=median、warmup=なし、checkpoint=`phase_def_block_20260710`。tuned 基準（既定 b64 比較 7.15×/1.64× とは区別）。

## PA/TX: 掃引確認値 ≠ 最終採用値（矛盾ではない）

- roadNet-PA: 掃引 b64 中央値 ≈ **941.4s**、最終採用 **918.67s**
- roadNet-TX: 掃引 b64 中央値 = **1491.13s**、最終採用 **1482.68s**

いずれも **最適設定は既定と同一の b64**。最終表では**同一 b64 設定の既存 default 実測（legacy, より速い保守的値）**を baseline として採用したため、掃引の確認測定値と一致しない。**欠損・矛盾ではなく、保守的 baseline の採用による別測定**である（詳細は `../../tuning/pathmerge/{roadNet-PA,roadNet-TX}/SOURCE.md`）。

## 補足
- email/roadNet の正確性は現状 `max_bc_only`（提案3実装間 + 独立参照 PathMerge の Max BC 一致）。独立参照との全ベクトル比較は未実施（`../../CLAIMS.md`）。
