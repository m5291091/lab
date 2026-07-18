# Correctness — corrected 325557 (job 2404743)

修正版 325557 グラフ (`data/325557_3216152_corrected_v1`, SHA256
`8373244f…4eeaa22`, n=325557, m=3216152) 上の全 BC ベクトル正確性・相互比較の
**正式（current）結果**。checkpoint `45352a3`、PBS job `2404743`。

派生元 raw: `raw_data/corrected_325557/job_2404743/`（6 ベクトル本体 + 10 比較 md/json）。
本ディレクトリの表は `raw_data/corrected_325557/job_2404743/{comparison_matrix,vector_inventory}.tsv`
から再計算した派生要約である。判定は `abs_tol=1e-3`, `rel_tol=1e-6`。

## 正式結果（支持できる結論）

- **6 ベクトル完全**（`vector_summary.tsv`）: gpu_opt_b1024 / gpu_opt_b9792 /
  gpu_opt_pure_b1024 / gpu_opt_pure_chunked_b1024 / gpu_opt_pure_chunked_b16384 /
  pathmerge_b4096。各 length=325557、missing/duplicate/out-of-range/NaN/Inf=0。
- **10 比較すべて mismatch=0**（`comparison_summary.tsv`）:
  - `same_batch_diff_path`（UM/Pure/Chunked @ b1024）: max_rel ≤ 1.08e-14。
  - `same_impl_diff_batch`（**旧 malformed 入力で FAIL していた stress ペア**）:
    gpu_opt b9792-vs-b1024 max_rel=5.32e-14、chunked b16384-vs-b1024 max_rel=4.75e-14。
    修正版では **PASS（mismatch=0）**。
  - `pathmerge_cross`（PathMerge vs GPU_Opt 系）: max_abs≈1.2e-3, max_rel≈5.09e-13,
    "PASS (absolute-only warning)"。
- 混合許容内で一致するが **byte 一致（SHA256 一致）ではない**（浮動小数点の累積順差）。
- Max BC は全実装で index 272816 において一致。

## スコープと非主張

- 対象は修正版 325557 のみ。他グラフ・他バッチ・他 GPU へ一般化しない。
- **PathMerge は external comparator であり independent ground truth ではない。**
- 修正版グラフは対称性から内部再構成したもので original generator seed は不明。
- 容量境界（feasibility）は `result/memory_scalability/corrected_325557/` を参照。

## 旧 malformed 入力の CORE_FAIL は current claim に使用しない

旧 `data/325557_3216152`（malformed）上の stress `CORE_FAIL`
（`result/correctness/memory_paths/canonical_job_2368587/`）は **historical** として
保持（削除しない）。`UsedInCurrentThesisClaim=No`,
`SupersededByCorrectedInputJob=2404743`。current claim は本ディレクトリ（mismatch=0）のみ。
