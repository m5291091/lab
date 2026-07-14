# BC ベクトル数値比較: Sequential vs GPU_Opt

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/chain_200/sequential.bc.tsv`  (header: impl=Sequential graph=chain_200 nodes=200)
  - SHA256: `1f39b65cc3fd9cf8e79e24421ed9bf5883e2869cd3d10db956275ca82c715f4b`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/chain_200/gpu_opt.bc.tsv`  (header: impl=GPU_Opt graph=chain_200 nodes=200)
  - SHA256: `3feef9b82e3ef287a3fe010f0797c07d04e9b18206ef88a6a8b3f469c9840dda`

- checkpoint_sha: e32b03e9b73e9eb294685c58e488ce2a92521852
- pbs_job_id: 2367583.opbs
- graph_path: /work/gj17/j17000/m5291091/lab/thesis_bc_project/data/chain_200
- graph_sha256: 8fe3b0e05de9eecb9999962374ce843264d040b985e85e7ddc7bfd987494db79
- n: 200
- m: 199
- requested_batch: 512
- effective_batch: 512
- SUB_BATCH: 512
- num_subs: 1
- NS_eff: 2

| 項目 | 値 |
|:--|:--|
| ベクトル長 A | 200 |
| ベクトル長 B | 200 |
| 共通 index 数 | 200 |
| 欠損 index 数 (A のみ) | 0 |
| 欠損 index 数 (B のみ) | 0 |
| 非有限値数 A | 0 |
| 非有限値詳細 A (vector A: Sequential) | なし |
| 非有限値数 B | 0 |
| 非有限値詳細 B (vector B: GPU_Opt) | なし |
| 最大絶対誤差 | 0.000000e+00 (index 0) |
| 最大絶対誤差 index の値 | A=0.0, B=0.0 |
| 最大相対誤差 | 0.000000e+00 (index 0) |
| 最大相対誤差 index の値 | A=0.0, B=0.0 |
| Max BC A | index 99, value 9900.000000 |
| Max BC B | index 99, value 9900.000000 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | OK |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 0 |
| **総合判定** | **PASS** |

