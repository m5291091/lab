# BC ベクトル数値比較: Sequential vs GPU_Opt

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/benchmark_7000_41459/sequential.bc.tsv`  (header: impl=Sequential graph=benchmark_7000_41459 nodes=7000)
  - SHA256: `fa23bf8892bdc799a9f859d3ccbe6859a1d965fb9162dcd62030d073a6771c04`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/benchmark_7000_41459/gpu_opt.bc.tsv`  (header: impl=GPU_Opt graph=benchmark_7000_41459 nodes=7000)
  - SHA256: `458d0a129e03229b1e349d8725461ef9c6cb99356636b8ebc87bacbe8a16e957`

- checkpoint_sha: e32b03e9b73e9eb294685c58e488ce2a92521852
- pbs_job_id: 2367583.opbs
- graph_path: /work/gj17/j17000/m5291091/lab/thesis_bc_project/data/benchmark_7000_41459
- graph_sha256: 4a891b4de4a0df86ef73c469f1e81b6206073e7368488e74f3ee2cec43b29ddc
- n: 7000
- m: 41459
- requested_batch: 512
- effective_batch: 512
- SUB_BATCH: 512
- num_subs: 1
- NS_eff: 2

| 項目 | 値 |
|:--|:--|
| ベクトル長 A | 7000 |
| ベクトル長 B | 7000 |
| 共通 index 数 | 7000 |
| 欠損 index 数 (A のみ) | 0 |
| 欠損 index 数 (B のみ) | 0 |
| 非有限値数 A | 0 |
| 非有限値詳細 A (vector A: Sequential) | なし |
| 非有限値数 B | 0 |
| 非有限値詳細 B (vector B: GPU_Opt) | なし |
| 最大絶対誤差 | 6.053597e-09 (index 0) |
| 最大絶対誤差 index の値 | A=2549196.725646447, B=2549196.725646441 |
| 最大相対誤差 | 4.563145e-15 (index 1186) |
| 最大相対誤差 index の値 | A=11161.53593043016, B=11161.53593043011 |
| Max BC A | index 4, value 3935437.257858 |
| Max BC B | index 4, value 3935437.257858 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | OK |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 0 |
| **総合判定** | **PASS** |

