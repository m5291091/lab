# Chapter 9 Correctness and Numerical Behavior

本章では RQ4 に回答する。正確性証拠を Tier A の independent CPU reference と Tier B の corrected-325557 cross-implementation consistency に分ける。PathMerge は評価した第三者実装の external comparator であり、ground truth ではない。混合許容内の `MismatchedElements=0` は bitwise identity を意味しない。

## 9.1 Validation Criterion and Evidence Tiers

reference $r_i$ と candidate $c_i$ の各 BC 要素に対し、次の混合絶対・相対許容を適用する。

$$
|r_i-c_i|\le abs\_tol+rel\_tol\max(|r_i|,|c_i|),
$$

ここで $abs\_tol=10^{-3}$、$rel\_tol=10^{-6}$ である。vector length、missing index、mismatched element、最大絶対・相対誤差、NaN/Inf、vector SHA256 を記録する。許容値を事後に変更して判定を変えない。

**Table 9.1: Correctness evidence tiers used in this study.**

| Tier | Evidence | Graphs | Comparisons | Interpretation | Status |
|---|---|---|---:|---|---|
| A | Independent Sequential CPU reference vs GPU_Opt | benchmark_7000_41459, benchmark_11023_62184, chain_200 | 3 | Independent full-vector validation on small graphs | SUPPORTED |
| B | Cross-implementation consistency on corrected 325557 | UM, Pure, Chunked, and PathMerge vectors | 10 | Numerical consistency, not independent ground truth | SUPPORTED_WITH_LIMITATIONS |

T5 は Tier A の 3 行と Tier B の 10 行、合計 13 行で構成する。全 13 行で `MissingIndices=0`、`MismatchedElements=0`、`ToleranceResult=PASS`、`ByteIdentical=No` である。

## 9.2 Tier A: Independent CPU Reference

Tier A は job `2367583.opbs` の full-vector 比較である。Sequential CPU 実装を独立参照、GPU_Opt b512 を candidate とし、各グラフ 1 回の vector を比較した。

**Table 9.2: Tier A full-vector comparisons against the independent Sequential CPU reference.**

| Graph | Vector Length | Missing Indices | Mismatched Elements | Max Absolute Error | Max Relative Error | Byte Identical | Result |
|---|---:|---:|---:|---:|---:|---|---|
| benchmark_7000_41459 | 7000 | 0 | 0 | 6.054e-09 | 4.563e-15 | No | PASS |
| benchmark_11023_62184 | 11023 | 0 | 0 | 2.980e-08 | 1.790e-14 | No | PASS |
| chain_200 | 200 | 0 | 0 | 0.000e+00 | 0.000e+00 | No | PASS |

> Source: `result/correctness/small_full_vector/correctness_summary.tsv` and T5 Panel A.

3 比較はいずれも全 index が存在し、混合許容を超える要素と NaN/Inf はなかった。Sequential と GPU_Opt の vector SHA256 は異なるため byte-identical ではない。Tier A が独立参照に基づき `SUPPORTED` とする範囲は、この小規模 3 グラフに限定される。headline roadNet、修正版 325557、すべての UM/Chunked 条件へ外挿しない。

## 9.3 Tier B: Corrected 325557 Cross-Implementation Consistency

Tier B は修正版 `325557_3216152_corrected_v1`、job 2404743、checkpoint `45352a3` の 6 vector から構成する。対象は GPU_Opt b1024/b9792、GPU_Opt_Pure b1024、GPU_Opt_Pure_Chunked b1024/b16384、PathMerge b4096 である。

**Table 9.3: Tier B full-vector comparisons on the corrected 325557 graph.**

| Comparison Class | Reference | Candidate | Mismatched Elements | Max Absolute Error | Max Relative Error | Byte Identical | Result |
|---|---|---|---:|---:|---:|---|---|
| Same implementation, different batch | GPU_Opt b9792 | GPU_Opt b1024 | 0 | 3.052e-05 | 5.316e-14 | No | PASS |
| Same implementation, different batch | Chunked b16384 | Chunked b1024 | 0 | 3.052e-05 | 4.752e-14 | No | PASS |
| Same batch, different path | GPU_Opt b1024 | Pure b1024 | 0 | 1.907e-06 | 1.010e-14 | No | PASS |
| Same batch, different path | GPU_Opt b1024 | Chunked b1024 | 0 | 7.629e-06 | 1.080e-14 | No | PASS |
| Same batch, different path | Pure b1024 | Chunked b1024 | 0 | 7.629e-06 | 1.033e-14 | No | PASS |
| PathMerge cross (5 comparisons) | PathMerge b4096 | Five proposed vectors | 0 each | 1.222e-03 to 1.236e-03 | 5.089e-13 | No | PASS |

> Source: `result/correctness/corrected_325557/comparison_summary.tsv`, `result/correctness/corrected_325557/vector_summary.tsv`, and T5 Panel B.

10 比較すべてで vector length は 325,557、missing は 0、mismatch は 0 である。same-implementation different-batch の 2 比較には UM b9792 と Chunked b16384 の large-batch/sub-batch 条件が含まれる。旧 malformed 入力で観測された stress mismatch は、修正版では再現しなかった。

一方、6 vector の SHA256 は相異するため `ByteIdentical=No` である。本結果は混合許容内の numerical consistency を示すが、exactly identical または bitwise identical という主張ではない。全実装の最大 BC index は 272816 で一致した。

## 9.4 Role of PathMerge

PathMerge は Galliot 系の第三者実装を adapter 化した external comparator である。原著者の公式実装ではなく、独立 ground truth として扱わない。Tier B の 5 pathmerge-cross 比較が PASS したことは、修正版入力に対して両実装群が混合許容内で整合したことを示す。それだけから PathMerge または提案実装の一般的な正しさを証明しない。

## 9.5 Historical Malformed-Input Result

旧 `data/325557_3216152`（SHA256 `a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584`）は、1-based を 0-based として格納し、隣接要素 7 個の不足、範囲外 ID、頂点行の欠落を含む malformed input であることが integrity audit で判明した。

この入力上の canonical job 2368587 では、same-implementation different-batch の stress 比較が formal tolerance を超え、PathMerge cross でも差が観測され、`FINAL_STATUS.txt` は `overall_status=CORE_FAIL` を記録した。これらを削除または PASS へ relabel しない。次に historical invalid-input evidence として保持する。

- `result/correctness/memory_paths/canonical_job_2368587/`
- `failure/`
- `result/provenance/GRAPH_325557_INTEGRITY_AUDIT.md`
- 関連 raw vector、log、SHA256 records

current active conclusion は `tools/repair_325557_graph.py` で決定的に再構成した修正版入力を job 2404743 で再検証した Tier B である。経緯は「過去の結果を誤りとして削除した」のではなく、「入力不整合を発見し、旧 evidence を保持した上で修正版を再検証した」と記述する。旧 `CORE_FAIL` を current failure として扱わず、修正版 PASS を旧入力の判定へ遡及適用しない。

## 9.6 Provenance Limitation of the Corrected Graph

修正版は旧入力から 1-based→0-based の全体 relabelling と、対称性に基づく欠落最終行 7 要素の再構成を行った内部修復データである。修復結果は決定的で独立 2 回の生成が byte-identical であった。しかし、元の生成 seed または上流の完全な原本を独立に確認できず、`ProvenanceStatus=internally_reconstructed_no_original_seed` である。

この制約のため、Tier B は `SUPPORTED_WITH_LIMITATIONS` とする。修復手続きの決定性は上流 provenance の欠損を解消しない。

## 9.7 T5 Summary

**Table 9.4: T5 summary across both evidence tiers.**

| Panel | Evidence Tier | Rows | Missing Indices | Mismatched Elements | Tolerance Result | Byte Identical |
|---|---|---:|---:|---:|---|---|
| A | Independent CPU reference | 3 | 0 in all rows | 0 in all rows | PASS in all rows | No in all rows |
| B | Corrected-325557 cross-implementation consistency | 10 | 0 in all rows | 0 in all rows | PASS in all rows | No in all rows |

> Canonical artifact: `result/tables/thesis/T5_correctness_summary.{md,tsv}`.

## 9.8 Answer to RQ4

RQ4 は evidence tier ごとに回答する。Tier A は、小規模 3 グラフにおける独立 Sequential CPU 参照との full-vector 比較がすべて PASS したため `SUPPORTED` である。Tier B は、修正版 325557 の 6 vector・10 比較がすべて混合許容内で PASS したため `SUPPORTED_WITH_LIMITATIONS` である。

Tier B は bitwise identity ではなく、独立 ground truth でもない。PathMerge は external comparator であり、修正版グラフには元 seed・上流原本未確認の provenance 制約が残る。旧 malformed 入力の `CORE_FAIL` は historical evidence として保存されるが、current active conclusion には含めない。
