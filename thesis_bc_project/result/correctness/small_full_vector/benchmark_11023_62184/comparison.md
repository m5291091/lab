# BC ベクトル数値比較: Sequential vs GPU_Opt

- 入力A: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/benchmark_11023_62184/sequential.bc.tsv`  (header: impl=Sequential graph=benchmark_11023_62184 nodes=11023)
  - SHA256: `98381b7869cb86719669aa78ecaeaad90054be77d43544a907cfea27f66fbae6`
- 入力B: `/work/gj17/j17000/m5291091/lab/thesis_bc_project/build_miyabi/result_small_correctness_20260712_181140_2367583.opbs/benchmark_11023_62184/gpu_opt.bc.tsv`  (header: impl=GPU_Opt graph=benchmark_11023_62184 nodes=11023)
  - SHA256: `d9db54898130c8e7acefe2cb304f57a6a52c0b379edae5aa72203e622a0e21f9`

- checkpoint_sha: e32b03e9b73e9eb294685c58e488ce2a92521852
- pbs_job_id: 2367583.opbs
- graph_path: /work/gj17/j17000/m5291091/lab/thesis_bc_project/data/benchmark_11023_62184
- graph_sha256: 8d1df41c579de3150a155ee9cce321784723fdb1824c0a2a160d95004d4b6e31
- n: 11023
- m: 62184
- requested_batch: 512
- effective_batch: 512
- SUB_BATCH: 512
- num_subs: 1
- NS_eff: 2

| 項目 | 値 |
|:--|:--|
| ベクトル長 A | 11023 |
| ベクトル長 B | 11023 |
| 共通 index 数 | 11023 |
| 欠損 index 数 (A のみ) | 0 |
| 欠損 index 数 (B のみ) | 0 |
| 非有限値数 A | 0 |
| 非有限値詳細 A (vector A: Sequential) | なし |
| 非有限値数 B | 0 |
| 非有限値詳細 B (vector B: GPU_Opt) | なし |
| 最大絶対誤差 | 2.980232e-08 (index 10) |
| 最大絶対誤差 index の値 | A=11951000.93285756, B=11951000.93285759 |
| 最大相対誤差 | 1.789722e-14 (index 3092) |
| 最大相対誤差 index の値 | A=11789.69389924664, B=11789.69389924643 |
| Max BC A | index 10, value 11951000.932858 |
| Max BC B | index 10, value 11951000.932858 |
| 許容値 | rel_tol=1e-06, abs_tol=0.001 |
| 絶対許容のみ (abs_diff ≤ abs_tol) | OK |
| 混合許容 abs_diff ≤ abs_tol+rel_tol·max(\|a\|,\|b\|) 不一致要素数 | 0 |
| **総合判定** | **PASS** |

