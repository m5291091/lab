# BC ベクトル数値比較: gpu_opt_b1024 vs gpu_opt_pure_b1024

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_b1024.bc.tsv`  (header: impl=GPU_Opt graph=325557_3216152 nodes=325557)
  - SHA256: `4a40a553a388ba2cb29d4ea366db979983fa398c55bb8a694882f260efd431cb`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_memory_correctness_20260712_220331_2368587.opbs/gpu_opt_pure_b1024.bc.tsv`  (header: impl=GPU_Opt_Pure graph=325557_3216152 nodes=325557)
  - SHA256: `fc95255fe422693d5f7c8d80d39624015b3f9ad2020b54916f61358718439984`

- checkpoint_sha: ac2b409c25c49c41608749afba8c7081871bfe45
- pbs_job_id: 2368587.opbs
- comparison_class: same_batch_diff_path
- graph_sha256: a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584
- n: 325557
- m: 3216152

| 項目 | 値 |
|:--|:--|
| ベクトル長 A | 325557 |
| ベクトル長 B | 325557 |
| 共通 index 数 | 325557 |
| 欠損 index 数 (A のみ) | 0 |
| 欠損 index 数 (B のみ) | 0 |
| 非有限値数 A | 0 |
| 非有限値詳細 A (vector A: gpu_opt_b1024) | なし |
| 非有限値数 B | 0 |
| 非有限値詳細 B (vector B: gpu_opt_pure_b1024) | なし |
| 最大絶対誤差 | 1.525879e-05 (index 272817) |
| 最大絶対誤差 index の値 | A=39343001000.10853, B=39343001000.10854 |
| 最大相対誤差 | 3.015985e-14 (index 260562) |
| 最大相対誤差 index の値 | A=165977.6043583227, B=165977.6043583277 |
| Max BC A | index 272817, value 39343001000.108528 |
| Max BC B | index 272817, value 39343001000.108543 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | OK |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 0 |
| **総合判定** | **PASS** |

