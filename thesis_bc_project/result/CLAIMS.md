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

## メモリ経路 (Stage 4C / Gate H0; 325557 限定) — **[HISTORICAL: malformed legacy input]**

> **この節の表は旧 malformed 入力 `data/325557_3216152` 上の履歴結果である。**
> 現行の memory-path 正確性 claim は修正版入力（job 2404743, `data/325557_3216152_corrected_v1`）の
> 下記「Gate W7.4」節を参照。canonical の `CORE_FAIL`（stress `same_impl_diff_batch` 超過）は
> **修正版で mismatch=0 となり置換された**が、履歴として削除・改変せず保持する
> (`UsedInCurrentThesisClaim=No`, `SupersededByCorrectedInputJob=2404743`)。

memory-path 関連は主軸(A)・副次(B)の性能主張から分離する。canonical=checkpoint `memory_correctness_20260712` / job `2368587`、診断=checkpoint `memory_diagnostic_20260713` / job `2369632`。各構成 n=1、warmup なし、判定 `abs_tol=1e-3` `rel_tol=1e-6`。canonical の formal overall status は `CORE_FAIL`（隠さない）。詳細は `correctness/memory_paths/`。**（以下は malformed legacy input 上の履歴; 現行 active claim ではない）**

| 主張 | 状態 | 根拠 / 検証範囲 |
|:--|:--|:--|
| **UM oversubscription 実行可能性** | `SUPPORTED_WITH_LIMITATIONS` | graph=325557、GPU_Opt `b9792`、checkpoint `memory_correctness_20260712`、host-memory-limited 100 GiB configurationで完走し oversubscription 経路証拠あり（est>free_before, HBM3 streaming, NS_eff=1, num_subs=2, SUB_BATCH<batch, Prefetch cum>0）。**migration byte 量の直接計測ではない**。`b10240` は同構成でOOM（`failure/failed/oom/memory_correctness_2368269/`） |
| **Chunked 実行可能性** | `SUPPORTED_WITH_LIMITATIONS` | graph=325557、`b16384`、`num_subs=3` で試験範囲内で完走。**あらゆる条件で OOM 回避とは書かない** |
| **Same-batch memory-path consistency** | `SUPPORTED_WITH_LIMITATIONS` | UM/Pure/Chunked `b1024` が `abs_tol=1e-3` `rel_tol=1e-6` で mismatch=0（`same_batch_diff_path`）。各1回。**非 byte（SHA256）一致**。325557 限定 |
| **Stress full-vector correctness** | `NOT_YET_SUPPORTED` | 正式 `rel_tol=1e-6` を超える構成依存差（`same_impl_diff_batch`, 影響 index 集合の和=8）。原因未特定。full reset と NS_eff=1 の**単独変更では再現せず**（診断 `RESET/NS_EFF_NOT_DISTINGUISHED`）。許容値感度分析（`rel_tol=3e-6` で消失）は補助情報で**正式 FAIL を変更しない** |
| **PathMerge cross-implementation 一致** | `NOT_YET_SUPPORTED` | PathMerge b4096 vs 提案各実装で約 11027 要素、最大相対差約 0.2%。**正誤未決定**。PathMerge は external comparator（ground truth ではない） |

- 小規模 Sequential vs GPU_Opt の `SUPPORTED`（上記「A 小規模 full-vector 正確性」）は**変更しない**。
- stress 差を「FP 累積順序が原因」と確定しない。stress full-vector 正確性や UM/Chunked の全条件正確性は**証明していない**。

## 正確性レベルの凡例
`full_vector_independent_reference` > `full_vector_same_implementation` > `max_bc_only` > `structural_only` > `none`
- 小規模3グラフの Sequential vs GPU_Opt: `full_vector_independent_reference`。headline 4グラフ: `max_bc_only`（cross-impl）。PathMerge tuned/default の別設定間比較は email（b64 vs b2048）と roadNet-CA（b32 vs b64）だけが `full_vector_same_implementation`。roadNet-PA/TX は tuned/default とも b64 で別 full-vector artifact がないため `max_bc_only`。
- email/CA の `full_vector_same_implementation` は**実験時の比較 summary に基づく**判定であり（email=PASS, CA=PASS with absolute-only warning）、この水準を維持する。比較に用いた BC ベクトル 4 本は現在 `currently_unavailable`（`EXTERNAL_ARTIFACTS.tsv`）で、archive-time に vector を再解析していないため NaN/Inf/duplicate index は `not_recorded`。**vector が現存しないことを理由に当時の比較 summary を無効化しない**。

---

## Gate W7.4 — 修正版325557 再検証 **完了**後の RQ 支持状態

`data/325557_3216152` は **malformed** と確定し（`provenance/GRAPH_325557_INTEGRITY_AUDIT.md`）、
修復版 `data/325557_3216152_corrected_v1`（`ProvenanceStatus=internally_reconstructed_no_original_seed`）
で再検証を実施した。GPU 実行済み（Series A/B=job 2404743, Series C=job 2406254, いずれも checkpoint
`45352a3`, `SUCCESS`）、Gate W7.3C1 で独立監査済み。正式結果は
`result/{correctness,memory_scalability,ablation}/corrected_325557/`、raw は
`raw_data/corrected_325557/`。旧 malformed 入力上の結果は **historical** として保存（削除・置換せず）。

| 主張群 | 状態 | 根拠 / 検証範囲・制約 |
|:--|:--|:--|
| **RQ1 main performance**（email-EuAll / roadNet-PA/TX/CA） | `SUPPORTED` | **不変**。325557 を使用しない。主要値 **3.17 / 1.31 / 1.51 / 1.45**（上表）は変更しない |
| **RQ2 synthetic aggregate**（ablation H/W/A） | `SUPPORTED_WITH_LIMITATIONS` | 修正版325557 H=1.4767 / W=1.1012 / A=1.5563（job 2406254, 40行完全）。合成4集約（他3グラフ raw 不変 + 325557 修正版）= **H=1.679 / W=1.066 / A=1.391**。**制約**: 4 synthetic graphs、**mixed checkpoints**（他3=job2354994, 325557=job2406254）、325557 のみ修正版再測定、roadNet へ一般化しない |
| **RQ3 memory feasibility**（UM/Pure/Chunked 容量境界） | `SUPPORTED_WITH_LIMITATIONS` | corrected 325557 の targeted boundary confirmation（job 2404743 Series B）。pure_b8192=**CUDA OOM**、um_b10240=success、**um_b12288=host/cgroup memory OOM kill（exit137, CUDA/HBM OOM ではない）**、chunked_b16384=success。**制約**: **各境界 1 trial**、feasibility であり性能比較ではない、host/cgroup memory limit を含む、入力≈43.25 MiB で容量問題は batch 依存 working set、他グラフ・他 GPU へ一般化しない |
| **RQ4a 小規模独立参照 full-vector 正確性** | `SUPPORTED` | **不変**。benchmark_7000_41459 / benchmark_11023_62184 / chain_200 の Sequential 独立参照 vs GPU_Opt（`correctness/small_full_vector/`） |
| **RQ4b 修正版325557 memory-path / cross consistency** | `SUPPORTED_WITH_LIMITATIONS` | job 2404743: 6ベクトル完全・**10比較すべて mismatch=0**（stress `same_impl_diff_batch` b9792/b1024・b16384/b1024 を含め混合許容内, max_rel<=5.09e-13）。**制約**: 混合許容内 mismatch=0 だが **byte-identical ではない**、PathMerge は独立正解ではない、corrected graph は内部再構成で original seed 不明、対象は 325557 のみ |

- 旧 `CORE_FAIL`（stress 超過, malformed input）は **current active claim から外し**、
  historical malformed-input result として保存（`UsedInCurrentThesisClaim=No`,
  `SupersededByCorrectedInputJob=2404743`）。上記「メモリ経路 (Stage 4C)」節参照。
- **stress 差の原因を GPU 数値計算へ帰属しない**。旧入力には範囲外添字アクセスがあったが（監査 §2.3）、
  修正版で差が消えたことは「malformed 入力が差の必要条件だった」ことと整合するが、**単一因果とは断定しない**。
- 失敗系列（build 失敗 2403658 / OOM マーカー誤判定 2404249）は `failure/failed/{build,validation}/` に保存。
  W7.3B1.1 / W7.3B2.2 で修正済み、成功再実行は job 2404743。
- **RQ3/RQ4b は corrected 325557 に限定し、他グラフ・他 GPU・block 一般へ一般化しない。**
  PathMerge は external comparator（ground truth ではない）。追加 GPU 実験は完了（Gate W7.3C1）。
