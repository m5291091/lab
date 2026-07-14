# BC ベクトル数値比較: T-RESET vs old_b9792

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_memory_diagnostic_20260713_012328_2369632.opbs/T-RESET/vector.bc.tsv`  (header: impl=GPU_Opt graph=325557_3216152 nodes=325557)
  - SHA256: `95395d1c5cf4a0cd75af0a2ec078edb5770c3ac201fb6ecaca2118fc63b4ce28`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_b9792.bc.tsv`  (header: impl=GPU_Opt graph=325557_3216152 nodes=325557)
  - SHA256: `be8f52d32ac03cd495a08c5c6cd138fcdcec916a16830e62e7bc8d3c968d25c5`

- checkpoint_sha: 43d1cf5542f3234dddc93c88c5fdd72761f52271
- pbs_job_id: 2369632.opbs

| 項目 | 値 |
|:--|:--|
| ベクトル長 A | 325557 |
| ベクトル長 B | 325557 |
| 共通 index 数 | 325557 |
| 欠損 index 数 (A のみ) | 0 |
| 欠損 index 数 (B のみ) | 0 |
| 非有限値数 A | 0 |
| 非有限値詳細 A (vector A: T-RESET) | なし |
| 非有限値数 B | 0 |
| 非有限値詳細 B (vector B: old_b9792) | なし |
| 最大絶対誤差 | 5.486364e+00 (index 289277) |
| 最大絶対誤差 index の値 | A=41284037.72298335, B=41284043.20934694 |
| 最大相対誤差 | 2.230184e-06 (index 7954) |
| 最大相対誤差 index の値 | A=224196.7897792182, B=224196.2897792182 |
| Max BC A | index 272817, value 39343001000.108521 |
| Max BC B | index 272817, value 39343001000.108543 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | WARN (超過; 巨大 magnitude で不適切な場合あり) |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 6 |
| **総合判定** | **FAIL (混合許容で不一致 6 件)** |

