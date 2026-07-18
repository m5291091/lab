# 09 正確性・制約（RQ4）

支持範囲を階層で示す。許容は `abs_diff ≤ abs_tol + rel_tol·max(|a|,|b|)`,
`abs_tol=1e-3`, `rel_tol=1e-6`。**PathMerge は external comparator（ground truth ではない）。**
**mismatch=0（PASS）は byte（SHA256）一致ではない。** 正確性証拠は 2 種類の tier で構成する
（T5 の Panel A / Panel B）。数値は `result/CLAIMS.md`・`result/correctness/` と一致。

## 9.1 正確性証拠の 2 tier（T5・必須）
| Tier | 区分 | グラフ | 参照/候補 | job | 行数 | 支持状態 |
|:--|:--|:--|:--|:--|--:|:--|
| A | 独立 CPU 参照 | bench_7000, bench_11023, chain_200 | Sequential(独立参照) vs GPU_Opt | 2367583 | 3 | `SUPPORTED` |
| B | 実装間整合 | 325557_3216152_corrected_v1 | UM/Pure/Chunked/PathMerge 相互（6 vector, 10 比較） | 2404743 | 10 | `SUPPORTED_WITH_LIMITATIONS` |

- **Tier A は独立 ground-truth 参照との比較**、**Tier B は同一修正版入力に対する実装間整合**であり
  独立正解との一致ではない。両 tier 計 13 行すべて MissingIndices=0・MismatchedElements=0・
  ToleranceResult=PASS・ByteIdentical=No。

## 9.2 Tier A：小規模独立 CPU 参照（`SUPPORTED`）
`result/correctness/small_full_vector/`（job `2367583.opbs`, requested/effective/SUB_BATCH=512,
num_subs=1, NS_eff=2, 各 n=1）。Sequential を**独立参照**、GPU_Opt を候補として全 BC ベクトル比較。

| Graph | 長さ ref/cand | missing | mismatch | NaN/Inf | max_abs_error | max_rel_error | ByteIdentical | 結果 |
|:--|:--|--:|--:|--:|--:|--:|:--:|:--|
| benchmark_7000_41459 | 7000/7000 | 0 | 0 | 0 | 6.05e-9 | 4.56e-15 | No | PASS |
| benchmark_11023_62184 | 11023/11023 | 0 | 0 | 0 | 2.98e-8 | 1.79e-14 | No | PASS |
| chain_200 | 200/200 | 0 | 0 | 0 | 0.0 | 0.0 | No | PASS |

- **full-vector 検証範囲**：この 3 グラフのみ。email/roadNet、UM oversubscription 固有経路には
  **一般化しない**。ByteIdentical=No（Sequential と GPU_Opt の vector SHA256 は相異）。

## 9.3 Tier B：修正版 325557 の実装間整合（`SUPPORTED_WITH_LIMITATIONS`）
`result/correctness/corrected_325557/`（job 2404743, checkpoint `45352a3`）。6 実装ベクトル
（UM/Pure/Chunked の複数 batch, PathMerge）にわたる **10 比較すべて MismatchedElements=0・
PASS・ByteIdentical=No**、max_rel_error ≤ **5.089e-13**。

| 区分 | 比較 | max_rel | 結果 |
|:--|:--|--:|:--|
| same_impl_diff_batch | gpu_opt b9792 vs b1024 | 5.316e-14 | PASS |
| same_impl_diff_batch | chunked b16384 vs b1024 | 4.752e-14 | PASS |
| same_batch_diff_path | gpu_opt vs pure（b1024） | 1.010e-14 | PASS |
| same_batch_diff_path | gpu_opt vs chunked（b1024） | 1.080e-14 | PASS |
| same_batch_diff_path | pure vs chunked（b1024） | 1.033e-14 | PASS |
| pathmerge_cross | PathMerge b4096 vs 提案各実装（5 ペア） | ≤ 5.089e-13 | PASS |

- **byte 一致ではない**：per-implementation の vector SHA256 は相異（混合許容内の数値整合性）。
- **large batch / sub-batch 分割条件（same_impl_diff_batch）も PASS**：旧 malformed 入力で観測
  された stress divergence（§9.5）は修正版入力では再現せず、mismatch=0 に解消した。
- Max BC は全実装で **index 272816** 一致。

## 9.4 PathMerge の位置づけ
PathMerge（第三者実装, Galliot 由来）は **external comparator であり ground truth ではない**。
Tier B の pathmerge_cross 行は数値整合性チェックであって exact-match の主張ではない。
「byte 一致」「exact match」「PathMerge ground truth」とは書かない。

## 9.5 Historical：旧 malformed 入力の発見と再検証
旧 325557 入力（`325557_3216152`, SHA256 `a095b2e7...`）は 1-based を 0-based として格納し、
隣接配列に 7 要素不足・範囲外 ID・行欠落を含む **malformed input** であることが判明した
（`result/provenance/GRAPH_325557_INTEGRITY_AUDIT.md`）。この malformed 入力上で得られた

- same_impl_diff_batch の rel_tol 超過（和集合 8 頂点, former stress FAIL）、
- PathMerge cross の約 11027 要素差（max_rel≈2.0e-3）、
- `FINAL_STATUS.txt` の `overall_status=CORE_FAIL`

は、**current active conclusion から外す**。ただし削除せず、
`result/correctness/memory_paths/canonical_job_2368587/`・`failure/`・provenance 文書に
**historical invalid-input result** として保持する。`tools/repair_325557_graph.py` で
決定的に修復した入力（`325557_3216152_corrected_v1`, SHA256 `8373244f...`）で再検証した結果が
§9.3 であり、malformed 入力で観測された divergence は修正版では再現しない。これは
「過去に誤っていたため削除した」のではなく、**入力不整合を発見し修正版で再検証した経緯**である。

## 9.6 修正版グラフの provenance 制約
`325557_3216152_corrected_v1` は旧入力からの**内部修復データ**であり、元生成 seed または上流の
完全な原本を独立に確認できない（`ProvenanceStatus=internally_reconstructed_no_original_seed`）。
修復は対称性から一意に定まり決定的（独立 2 回実行で byte 一致）だが、この provenance 制約は
残る。Tier B の結論はこの制約付きで `SUPPORTED_WITH_LIMITATIONS` とする。

## 9.7 支持範囲まとめ
- `SUPPORTED`：Tier A 小規模 3 グラフの Sequential vs GPU_Opt full-vector（独立参照）。
- `SUPPORTED_WITH_LIMITATIONS`：Tier B 修正版 325557 の 10 比較（実装間整合, 非 byte, provenance 制約）。
- **除外（historical）**：旧 malformed 入力の CORE_FAIL / stress FAIL / PathMerge DIFF は現行 T5 に
  含めず provenance として保持する。
