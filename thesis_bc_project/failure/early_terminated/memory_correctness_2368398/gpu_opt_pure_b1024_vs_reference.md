# BC ベクトル数値比較: PathMerge_b4096 vs gpu_opt_pure_b1024

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_memory_correctness_20260712_211738_2368398.opbs/pathmerge_b4096.bc.tsv`  (header: impl=PathMerge_BC graph=325557_3216152 nodes=325557)
  - SHA256: `c895c6121671af6605d834e68bc1443b84624ca595506267901c3a59a643e454`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_memory_correctness_20260712_211738_2368398.opbs/gpu_opt_pure_b1024.bc.tsv`  (header: impl=GPU_Opt_Pure graph=325557_3216152 nodes=325557)
  - SHA256: `b4f0674431bc4c918b7b5a597ca98be6132c6f9ad06b2f31b300ff31f6b23e95`

- checkpoint_sha: 29d28c50dec5e70f8d3a9a2341904e1ee94c65f3
- pbs_job_id: 2368398.opbs
- graph_path: /work/gj17/j17000/m5291091/lab/thesis_bc_project/data/325557_3216152
- graph_sha256: a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584
- n: 325557
- m: 3216152
- reference_impl: pathmerge_bc
- reference_batch: 4096
- candidate_impl: gpu_opt_pure
- path_type: pure
- requested_batch: 1024
- effective_batch: 1024
- SUB_BATCH: not_applicable
- num_subs: not_applicable
- NS_eff: not_applicable
- oversubscribed: not_applicable
- uses_um: false

| 項目 | 値 |
|:--|:--|
| ベクトル長 A | 325557 |
| ベクトル長 B | 325557 |
| 共通 index 数 | 325557 |
| 欠損 index 数 (A のみ) | 0 |
| 欠損 index 数 (B のみ) | 0 |
| 非有限値数 A | 0 |
| 非有限値詳細 A (vector A: PathMerge_b4096) | なし |
| 非有限値数 B | 0 |
| 非有限値詳細 B (vector B: gpu_opt_pure_b1024) | なし |
| 最大絶対誤差 | 6.638161e+03 (index 289444) |
| 最大絶対誤差 index の値 | A=8903157.975814402, B=8909796.137074383 |
| 最大相対誤差 | 2.000197e-03 (index 325556) |
| 最大相対誤差 index の値 | A=892937.5267641286, B=894727.1570761405 |
| Max BC A | index 272817, value 39343001000.107368 |
| Max BC B | index 272817, value 39343001000.108551 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | WARN (超過; 巨大 magnitude で不適切な場合あり) |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 11027 |
| **総合判定** | **FAIL (混合許容で不一致 11027 件)** |

