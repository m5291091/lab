# BC ベクトル数値比較: pathmerge_b4096 vs gpu_opt_b1024

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_corrected_325557_20260717_215817_2404743.opbs/pathmerge_b4096.bc.tsv`  (header: impl=PathMerge_BC graph=325557_3216152_corrected_v1 nodes=325557)
  - SHA256: `164cceb2c179ce1a6552685663e6d66745a2483c877e0afe52e9f94bc7aa9496`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_corrected_325557_20260717_215817_2404743.opbs/gpu_opt_b1024.bc.tsv`  (header: impl=GPU_Opt graph=325557_3216152_corrected_v1 nodes=325557)
  - SHA256: `adc6270b15aba9201c7667460b235d0e3762dcb9e529e6390bc5db218d738796`

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
| 最大絶対誤差 | 1.222134e-03 (index 48184) |
| 最大相対誤差 | 5.088914e-13 (index 309697) |
| 最大絶対誤差 index の値 | A=3750319918.790966, B=3750319918.789744 |
| 最大相対誤差 index の値 | A=117887775.6666132, B=117887775.6666732 |
| Max BC A | index 272816, value 39343117052.538231 |
| Max BC B | index 272816, value 39343117052.539436 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | WARN (超過; 巨大 magnitude で不適切な場合あり) |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 0 |
| **総合判定** | **PASS (absolute-only warning)** |

