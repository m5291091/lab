# BC ベクトル数値比較: CONTROL vs T-RESET

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_memory_diagnostic_20260713_012328_2369632.opbs/CONTROL/vector.bc.tsv`  (header: impl=GPU_Opt graph=325557_3216152 nodes=325557)
  - SHA256: `e4458d31c904d904d1b3f4e0c21bf35ee279324e64b39444e8cd1b44585fdede`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_memory_diagnostic_20260713_012328_2369632.opbs/T-RESET/vector.bc.tsv`  (header: impl=GPU_Opt graph=325557_3216152 nodes=325557)
  - SHA256: `95395d1c5cf4a0cd75af0a2ec078edb5770c3ac201fb6ecaca2118fc63b4ce28`

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
| 非有限値詳細 A (vector A: CONTROL) | なし |
| 非有限値数 B | 0 |
| 非有限値詳細 B (vector B: T-RESET) | なし |
| 最大絶対誤差 | 2.288818e-05 (index 272817) |
| 最大絶対誤差 index の値 | A=39343001000.10854, B=39343001000.10852 |
| 最大相対誤差 | 4.249754e-14 (index 325328) |
| 最大相対誤差 index の値 | A=405422.7133861769, B=405422.7133861597 |
| Max BC A | index 272817, value 39343001000.108543 |
| Max BC B | index 272817, value 39343001000.108521 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | OK |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 0 |
| **総合判定** | **PASS** |

