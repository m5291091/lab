# BC ベクトル数値比較: T-NSEFF vs old_chunk_b16384

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_memory_diagnostic_20260713_012328_2369632.opbs/T-NSEFF/vector.bc.tsv`  (header: impl=GPU_Opt graph=325557_3216152 nodes=325557)
  - SHA256: `45edeb2c6b8dc65cbb98280f863407d146b2d2007d3f92b1668475713f759d35`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_pure_chunked_b16384.bc.tsv`  (header: impl=GPU_Opt_Pure_Chunked graph=325557_3216152 nodes=325557)
  - SHA256: `618ffdc4108f0c24a148bc1aa2a18b83d14b48319cd18438f89196b40930a86d`

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
| 非有限値詳細 A (vector A: T-NSEFF) | なし |
| 非有限値数 B | 0 |
| 非有限値詳細 B (vector B: old_chunk_b16384) | なし |
| 最大絶対誤差 | 5.358586e+00 (index 289277) |
| 最大絶対誤差 index の値 | A=41284037.72298337, B=41284043.08156917 |
| 最大相対誤差 | 2.847745e-06 (index 95156) |
| 最大相対誤差 index の値 | A=21947.19184601125, B=21947.12934601126 |
| Max BC A | index 272817, value 39343001000.108452 |
| Max BC B | index 272817, value 39343001000.108582 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | WARN (超過; 巨大 magnitude で不適切な場合あり) |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 6 |
| **総合判定** | **FAIL (混合許容で不一致 6 件)** |

