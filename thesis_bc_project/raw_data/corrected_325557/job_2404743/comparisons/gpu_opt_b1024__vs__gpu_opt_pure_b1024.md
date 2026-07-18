# BC ベクトル数値比較: gpu_opt_b1024 vs gpu_opt_pure_b1024

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_corrected_325557_20260717_215817_2404743.opbs/gpu_opt_b1024.bc.tsv`  (header: impl=GPU_Opt graph=325557_3216152_corrected_v1 nodes=325557)
  - SHA256: `adc6270b15aba9201c7667460b235d0e3762dcb9e529e6390bc5db218d738796`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_corrected_325557_20260717_215817_2404743.opbs/gpu_opt_pure_b1024.bc.tsv`  (header: impl=GPU_Opt_Pure graph=325557_3216152_corrected_v1 nodes=325557)
  - SHA256: `65c109ef96f418be91cfb0acf5891504356783a172e28f703192b3594e8cacbb`

| 項目 | 値 |
|:--|:--|
| 期待ベクトル長 | 325557 |
| ベクトル長 A | 325557 |
| ベクトル長 B | 325557 |
| データ行数 A / B | 325557 / 325557 |
| 共通 index 数 | 325557 |
| 欠損 index 数 (A のみ) | 0 |
| 欠損 index 数 (B のみ) | 0 |
| expected domain 欠損 A / B | 0 / 0 |
| duplicate index A / B | 0 / 0 |
| out-of-range index A / B | 0 / 0 |
| parse error A / B | 0 / 0 |
| 値列欠損 A / B | 0 / 0 |
| NaN A / B | 0 / 0 |
| +Inf A / B | 0 / 0 |
| -Inf A / B | 0 / 0 |
| 最大絶対誤差 | 1.907349e-06 (index 44622) |
| 最大相対誤差 | 1.010044e-14 (index 260168) |
| 最大絶対誤差 index の値 | A=4013628796.616262, B=4013628796.616264 |
| 最大相対誤差 index の値 | A=167123.6552127306, B=167123.6552127323 |
| Max BC A | index 272816, value 39343117052.539436 |
| Max BC B | index 272816, value 39343117052.539436 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | OK |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 0 |
| **総合判定** | **PASS** |

