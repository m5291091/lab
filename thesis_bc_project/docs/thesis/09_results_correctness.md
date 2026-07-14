# 09 正確性・制約（RQ4）

失敗を隠さず、支持範囲を階層で示す。許容は `abs_diff ≤ abs_tol + rel_tol·max(|a|,|b|)`,
`abs_tol=1e-3`, `rel_tol=1e-6`。**PathMerge は external comparator（ground truth ではない）。**
**mismatch=0 は byte（SHA256）一致ではない。** 数値は `result/CLAIMS.md`・
`result/correctness/` と一致。

## 9.1 正確性区分表（T-CORR・必須）
| # | 区分 | グラフ | 実装 | 正確性レベル | mismatch | 支持状態 |
|:--|:--|:--|:--|:--|--:|:--|
| 1 | 小規模独立参照 | bench_7000, bench_11023, chain_200 | Sequential(参照) vs GPU_Opt | full_vector_independent_reference | 0 | `SUPPORTED` |
| 2 | same-batch メモリ経路 | 325557 | UM/Pure/Chunked b1024 | full_vector_same_implementation | 0（非 byte 一致） | `SUPPORTED_WITH_LIMITATIONS` |
| 3 | stress 条件 | 325557 | 同一実装 別バッチ（b9792 vs b1024, b16384 vs b1024） | full_vector_same_implementation | 6 / 6（和集合 8 頂点） | `NOT_YET_SUPPORTED` |
| 4 | PathMerge cross-impl | 325557 | PathMerge b4096 vs 提案各実装 | max_bc_only（vector は DIFF） | 約11027（max_rel≈2.0e-3） | `NOT_YET_SUPPORTED` |
| 5 | run-to-run 再現性 | 325557 | Pure/PathMerge 同一構成の実行間 | full_vector_same_implementation | 0 | 補助（`within_mixed_tol`） |

正確性レベルの序列：`full_vector_independent_reference` > `full_vector_same_implementation` >
`max_bc_only` > `structural_only` > `none`。

## 9.2 区分1：小規模独立参照（`SUPPORTED`）
`result/correctness/small_full_vector/`（checkpoint `small_correctness_20260712`, job `2367583.opbs`,
requested/effective/SUB_BATCH=512/512/512, num_subs=1, NS_eff=2, 各 n=1, warmup なし）。
Sequential を**独立参照**、GPU_Opt を candidate として全 BC ベクトルを比較。

| Graph | 長さ ref/cand | missing | mismatch | NaN/Inf | max_abs_error | max_rel_error | Max BC ref/cand |
|:--|:--|--:|--:|--:|--:|--:|:--|
| benchmark_7000_41459 | 7000/7000 | 0 | 0 | 0 | 6.05e-9 | 4.56e-15 | 一致（idx4, 3935437.257858） |
| benchmark_11023_62184 | 11023/11023 | 0 | 0 | 0 | 2.98e-8 | 1.79e-14 | 一致（idx10, 11951000.932858） |
| chain_200 | 200/200 | 0 | 0 | 0 | 0.0 | 0.0 | 一致（idx99, 9900.000000） |

- **full-vector 検証範囲**：この 3 グラフのみ。email/roadNet、Pure/Chunked、UM oversubscription
  固有経路には**一般化しない**。個別の Hybrid BFS/warp 経路を専用カウンタで検証したものでもない。
- vector SHA256 は `correctness_summary.tsv` に記録（例 bench_7000 GPU_Opt=`458d0a12...`）。

## 9.3 区分2：same-batch memory-path consistency（`SUPPORTED_WITH_LIMITATIONS`）
`result/correctness/memory_paths/canonical_job_2368587/comparison_matrix.tsv`（`same_batch_diff_path`）。
b1024 の UM/Pure/Chunked を相互比較：**3 ペアすべて mismatch=0**（PASS）。max_rel_error は
2.6e-14〜3.0e-14。
- **byte 一致ではない**：SHA256 は 3 実装で相異（例 UM=`4a40a553...`, Pure=`fc95255f...`,
  Chunked=`5ff0bef0...`）。「事前設定した混合許容内で一致」とだけ述べる。
- 各 n=1、warmup なし、325557 限定。

## 9.4 区分3：stress full-vector（`NOT_YET_SUPPORTED`）
`same_impl_diff_batch`（同一実装・別バッチ）：
- gpu_opt b9792 vs b1024：mismatch=**6**, max_abs=5.49, max_rel=**2.23e-6**（>rel_tol=1e-6）。
- gpu_opt_pure_chunked b16384 vs b1024：mismatch=**6**, max_abs=5.36, max_rel=**2.85e-6**。
- 影響 index 集合の**和=8 頂点**（`analysis/six_vertex_detail.tsv`：7954, 95156, 143358,
  165886, 226184, 228350, 289284, 325556）。
- **診断**（`diagnostic_job_2369632`, checkpoint `memory_diagnostic_20260713`）：full memset 強制（T-RESET）と
  NS_eff=1 強制（T-NSEFF）の**単独変更では CONTROL(b1024) と差が出ず**（mismatch=0,
  `RESET/NS_EFF_NOT_DISTINGUISHED`）。**原因未特定**。large batch / sub-batch 分割 /
  grid・occupancy またはその組合せとの関連が残る。
- **許容感度**（`analysis/tolerance_sensitivity.tsv`）：`1e-6` で 6 頂点、`2e-6` で 1、
  `3e-6` で 0。**これは補助情報であり、正式 FAIL（rel_tol=1e-6）を PASS に変更しない**。
- stress 差を「FP 累積順序が原因」と**確定しない**。

## 9.5 区分4：PathMerge cross-implementation（`NOT_YET_SUPPORTED`）
`pathmerge_cross`（PathMerge b4096 vs 提案各実装, 5 ペア）：**5/5 DIFF**、約 11027〜11030 要素、
max_abs≈6.64e3、max_rel≈**2.0e-3（0.2%）**。
- Max BC 値は全実装で ≈39343001000.108（一致；提案系は …108, PathMerge は …107 の末尾差）。
- **正誤未決定**。この差から提案実装が誤りとは**断定しない**。PathMerge は external comparator。
- fail-fast 事例：job `2368398` で pure vs PathMerge の比較不一致（11027 件）により後続構成が
  未実行（`failure/early_terminated/`）。**「Pure runner 失敗」や「OOM」とは誤記しない**
  （runner は exit0、失敗したのは比較判定）。

## 9.6 区分5：run-to-run 再現性（補助）
`analysis/run_to_run_comparison.tsv`：同一構成の実行間比較（Pure b1024 の 2 run, PathMerge の
3 run）はいずれも mismatch=0（`within_mixed_tol`）だが **byte 一致ではない**（SHA256 相異）。
通常の丸め変動範囲内。ただし stress の 8 頂点差は通常 run-to-run 変動だけでは説明困難
（診断で単一因子が特定できず）。

## 9.7 canonical の formal status（隠さない）
`FINAL_STATUS.txt`：`overall_status=CORE_FAIL`（core_pass=3, core_fail=2 of 5;
cross 0/5 pass）。この `CORE_FAIL` を隠さず記載する。memory-path は正確性・診断のみで、
時間値は性能表・性能図に追加しない。

## 9.8 支持範囲まとめ
- `SUPPORTED`：小規模3グラフの Sequential vs GPU_Opt full-vector。
- `SUPPORTED_WITH_LIMITATIONS`：325557 same-batch UM/Pure/Chunked（非 byte 一致）、
  headline 4グラフの max_bc_only（提案3実装間 + 独立参照 PathMerge の Max BC 一致）。
- `NOT_YET_SUPPORTED`：stress full-vector、PathMerge cross-impl、headline 独立参照 full-vector。
