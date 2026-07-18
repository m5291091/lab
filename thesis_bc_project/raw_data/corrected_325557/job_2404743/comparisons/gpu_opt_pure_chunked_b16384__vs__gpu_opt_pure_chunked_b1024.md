# BC ベクトル数値比較: gpu_opt_pure_chunked_b16384 vs gpu_opt_pure_chunked_b1024

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_corrected_325557_20260717_215817_2404743.opbs/gpu_opt_pure_chunked_b16384.bc.tsv`  (header: impl=GPU_Opt_Pure_Chunked graph=325557_3216152_corrected_v1 nodes=325557)
  - SHA256: `222bcfccd085da93b13aefbda83e597020688a1afd5fab3321df14b9e27bed1f`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_corrected_325557_20260717_215817_2404743.opbs/gpu_opt_pure_chunked_b1024.bc.tsv`  (header: impl=GPU_Opt_Pure_Chunked graph=325557_3216152_corrected_v1 nodes=325557)
  - SHA256: `d53afd646d3895a60f065ba98561ccfca522bb240681610c915dfabd3ff7f618`

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
| 最大絶対誤差 | 3.051758e-05 (index 272816) |
| 最大相対誤差 | 4.751990e-14 (index 186859) |
| 最大絶対誤差 index の値 | A=39343117052.53946, B=39343117052.53943 |
| 最大相対誤差 index の値 | A=2929987.687586053, B=2929987.687586192 |
| Max BC A | index 272816, value 39343117052.539459 |
| Max BC B | index 272816, value 39343117052.539429 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | OK |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 0 |
| **総合判定** | **PASS** |

