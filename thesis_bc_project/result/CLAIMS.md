# CLAIMS — 論文主張と支持状態（Gate F0 時点）

本ファイルは各主張の**現時点**の支持状態を記録する。**未実行の検証を「完了」と書かない。**
主要値は元 TSV（canonical raw）から再生成しており、本ファイルで勝手に変更しない。

## 主軸 (A): 提案手法(常時block GPU_Opt) vs PathMerge tuned

| グラフ | 提案(block)[s] | PathMerge tuned[s] (batch) | 倍率 | canonical 出典 |
|:--|--:|--:|--:|:--|
| email-EuAll | 30.81 | 97.80 (b2048) | **3.17×** | `main_performance/proposed_variants/email-EuAll/` + `tuning/pathmerge/email-EuAll/` |
| roadNet-PA | 699.52 | 918.67 (b64) | **1.31×** | `proposed_variants/roadNet-PA/` + `seven_implementations/legacy_partial/large/`(PathMerge_BC b64) |
| roadNet-TX | 980.13 | 1482.68 (b64) | **1.51×** | `proposed_variants/roadNet-TX/` + `seven_implementations/legacy_partial/large/`(PathMerge_BC b64) |
| roadNet-CA | 2129.10 | 3079.72 (b32) | **1.45×** | `proposed_variants/roadNet-CA/` + `tuning/pathmerge/roadNet-CA/` |

- 集計=中央値(median)、warmup=なし、checkpoint=`phase_def_block_20260710`。**tuned 基準**（既定 b64 比較の 7.15×/1.64× とは区別）。
- 本表の「PathMerge」は**評価に使用した第三者実装**（上流 `gobardhanm/path-merging-bc`, 論文著者の公式実装ではない; 上流に明示的ライセンス表記なし）を指す。倍率はこの第三者実装（tuned）に対する実測であり、PathMerge/Galliot アルゴリズム一般や原著者公式実装への優劣ではない。external comparator であり ground truth ではない（Stage L0.1）。

## 支持状態

| 主張 | 状態 | 根拠 / 検証範囲 |
|:--|:--|:--|
| **A 性能** | `SUPPORTED` | 全4グラフで提案 block が PathMerge tuned を上回る（上表, 追跡下・median） |
| **A 小規模 full-vector 正確性** | `SUPPORTED` | benchmark_7000_41459、benchmark_11023_62184、chain_200 において、Sequential を独立参照として GPU_Opt の全 BC ベクトルを比較し、混合許容基準で不一致 0、missing index 0、NaN/Inf 0 を確認（`correctness/small_full_vector/`, checkpoint `small_correctness_20260712`, job `2367583.opbs`）。**この3グラフだけに限定** |
| **A email/roadNet 正確性** | `SUPPORTED_WITH_LIMITATIONS` | **Reason**: 提案3実装間(GPU_Opt/Pure/Chunked)の Max BC 一致のみ（`proposed_variants/*/correctness.md`）+ 独立参照 PathMerge と Max BC 一致。**独立参照との全ベクトル比較は未実施** |
| **B OOM/容量 feasibility** | `SUPPORTED_WITH_LIMITATIONS` | legacy UM(測定 `oldtree_f05ec52_20260512`) と checkpoint `phase_def_block_20260710` のメモリロジックが**文字単位同一**（`provenance/um_code_diff_audit.md`）のため legacy 結果を限定的 feasibility 根拠として再利用（phase_def_block_20260710 で境界を再実測したものではない）。**時間値は非採用**（旧セッション未再検証）。GH200・325557・試験バッチ範囲に限定 |
| **B BC計算としての feasibility** | `NOT_YET_SUPPORTED` | 現状 max_bc_only（39343001000.11 が独立参照一致, FP順序差≤5.7e-9）。代表 oversub/chunk の full-vector 確認後に `SUPPORTED`（C1: PathMerge b4096 / GPU_Opt b10240 / Pure_Chunked b16384） |
| **B 最新block性能値** | `NOT_YET_SUPPORTED` | UM掃引の block 全面再測定は未実施（副次のため必須でない） |
| **7実装 small 比較** | `SUPPORTED_WITH_LIMITATIONS` | `seven_implementations/legacy_partial/small/`。提案系=**旧shared**、legacy baseline。README の制約参照 |
| **7実装 medium/large 比較** | `NOT_YET_SUPPORTED` | medium/large は Sequential/OpenMP/cuGraph 欠。現行 block での完全統一表は未測定 |

### Stage 3 小規模検証の適用範囲
小規模3グラフの正確性だけを `SUPPORTED` とする。email/roadNet は `SUPPORTED_WITH_LIMITATIONS` を**維持**する（headline 4グラフの独立参照 full-vector は未実施）。GPU_Opt_Pure、GPU_Opt_Pure_Chunked、UM oversubscription 固有経路にも一般化しない。また Hybrid BFS、warp 等の個別経路を専用カウンタで確認した検証ではない。

## メモリ経路 (Stage 4C / Gate H0; 325557 限定)

memory-path 関連は主軸(A)・副次(B)の性能主張から分離する。canonical=checkpoint `memory_correctness_20260712` / job `2368587`、診断=checkpoint `memory_diagnostic_20260713` / job `2369632`。各構成 n=1、warmup なし、判定 `abs_tol=1e-3` `rel_tol=1e-6`。canonical の formal overall status は `CORE_FAIL`（隠さない）。詳細は `correctness/memory_paths/`。

| 主張 | 状態 | 根拠 / 検証範囲 |
|:--|:--|:--|
| **UM oversubscription 実行可能性** | `SUPPORTED_WITH_LIMITATIONS` | graph=325557、GPU_Opt `b9792`、checkpoint `memory_correctness_20260712`、100 GiB queue で完走し oversubscription 経路証拠あり（est>free_before, HBM3 streaming, NS_eff=1, num_subs=2, SUB_BATCH<batch, Prefetch cum>0）。**migration byte 量の直接計測ではない**。`b10240` は 100 GiB 制限で OOM（`failure/failed/oom/memory_correctness_2368269/`） |
| **Chunked 実行可能性** | `SUPPORTED_WITH_LIMITATIONS` | graph=325557、`b16384`、`num_subs=3` で試験範囲内で完走。**あらゆる条件で OOM 回避とは書かない** |
| **Same-batch memory-path consistency** | `SUPPORTED_WITH_LIMITATIONS` | UM/Pure/Chunked `b1024` が `abs_tol=1e-3` `rel_tol=1e-6` で mismatch=0（`same_batch_diff_path`）。各1回。**非 byte（SHA256）一致**。325557 限定 |
| **Stress full-vector correctness** | `NOT_YET_SUPPORTED` | 正式 `rel_tol=1e-6` を超える構成依存差（`same_impl_diff_batch`, 影響 index 集合の和=8）。原因未特定。full reset と NS_eff=1 の**単独変更では再現せず**（診断 `RESET/NS_EFF_NOT_DISTINGUISHED`）。許容値感度分析（`rel_tol=3e-6` で消失）は補助情報で**正式 FAIL を変更しない** |
| **PathMerge cross-implementation 一致** | `NOT_YET_SUPPORTED` | PathMerge b4096 vs 提案各実装で約 11027 要素、最大相対差約 0.2%。**正誤未決定**。PathMerge は external comparator（ground truth ではない） |

- 小規模 Sequential vs GPU_Opt の `SUPPORTED`（上記「A 小規模 full-vector 正確性」）は**変更しない**。
- stress 差を「FP 累積順序が原因」と確定しない。stress full-vector 正確性や UM/Chunked の全条件正確性は**証明していない**。

## 正確性レベルの凡例
`full_vector_independent_reference` > `full_vector_same_implementation` > `max_bc_only` > `structural_only` > `none`
- 小規模3グラフの Sequential vs GPU_Opt: `full_vector_independent_reference`。headline 4グラフ: `max_bc_only`（cross-impl）。PathMerge tuned batch 間: `full_vector_same_implementation`。
