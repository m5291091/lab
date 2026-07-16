# 01 研究質問（RQ1〜RQ4）

各 RQ について `RQ / Question / RequiredEvidence / AvailableEvidence / Answer / Limitations`
を記録する。回答は実測範囲を超えて一般化しない。数値は `result/CLAIMS.md`・元 TSV と一致。

---

## RQ1：性能

- **RQ**: RQ1
- **Question**: 提案した GPU 実行基盤（block GPU_Opt, UM, 固定 b512）は、グラフごとに
  調整した既存 GPU baseline（PathMerge tuned）より高速か。
- **RequiredEvidence**: 主要4グラフでの提案 median と PathMerge tuned median、同一 checkpoint、
  同一集計、tuned バッチの根拠。
- **AvailableEvidence**:
  - `result/main_performance/proposed_variants/{email-EuAll,roadNet-PA,roadNet-TX,roadNet-CA}/results.tsv`（GPU_Opt, median; email n=5, road n=3, checkpoint `phase_def_block_20260710`）。
  - PathMerge tuned: `result/tuning/pathmerge/*`（email b2048, CA b32 は掃引実測）+
    `raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv`
    （PA/TX は掃引で b64 最適を確認し legacy 既定 b64 実測を採用）。
  - 元 TSV から再計算：email 30.81/97.80=**3.17×**、PA 699.52/918.67=**1.31×**、
    TX 980.13/1482.68=**1.51×**、CA 2129.10/3079.72=**1.45×**（median/median）。
- **Answer**: **評価した 4 グラフすべてで、提案 block GPU_Opt が PathMerge tuned より
  1.31〜3.17 倍高速だった**（`SUPPORTED`）。ハブ有り実データ（email-EuAll）で最大 3.17×、
  低次数の道路網 3 種で 1.31〜1.51×。
- **Limitations**: 評価は 4 グラフに限定。提案手法側は固定 b512（提案手法の batch sweep は
  未実施）。PathMerge は tuned（グラフごとに調整＝提案手法に不利な保守的比較）。既定 b64
  比較の 7.15×/1.64× とは区別する。正確性は headline では `max_bc_only`（RQ4 参照）。

---

## RQ2：最適化要因

- **RQ**: RQ2
- **Question**: Hybrid BFS（H）・warp 協調（W）・2 ストリーム（A）は性能へどのように寄与するか。
- **RequiredEvidence**: H/W/A の 8 構成（2^3）アブレーション、フェーズ内訳、プロファイル。
- **AvailableEvidence**:
  - `result/ablation/synthetic_2354994/`（benchmark_7000 / 11023 / 56438 / 325557 × 8 構成 × n=5）。
  - `result/ablation/email_2354999/`（email-EuAll × 8 構成 × n=3）。
  - 主効果（幾何平均）：synthetic で H≈1.655×、A≈1.396×、W≈1.065×；email で H≈1.429×、
    A≈1.720×、W≈0.970×（`ablation_contributions.tsv` / `ablation_summary.md`）。
  - フェーズ内訳（`ablation_summary.md`）：H は BFS cum を短縮、A は wall を短縮（2 stream 重畳）。
  - プロファイル：`raw_data/profiling/job_2359175_20260711/ablation_H1W1A0.stats.txt`（56438_300801、本測定 H1W1A0 と untimed H1W1A1 warmupを含む単一トレースの CUDA GPU カーネル時間のみ：backward 63.9% / bfs 36.1%）、
    帯域 `bandwidth.log`。
- **Answer**: 評価したアブレーション条件では、**Hybrid BFS と 2 ストリームが主要な性能寄与を
  示し、warp 協調の効果はグラフ依存だった**（W は email で 0.970×＝わずかに悪化、56438 で
  0.992×、benchmark_7000 で 1.175×）。交互作用チェックでは全構成が閾値 10% 未満で強い交互
  作用は検出されず。
- **Limitations**: アブレーションは synthetic 4 + email の計 5 グラフ。因果を実験範囲外
  （headline roadNet 全体など）へ一般化しない。フェーズ内訳の gap は A=1 で 2 stream 合算の
  ため負値になり得る（バグではなく重畳の証拠）。専用ハードウェアカウンタで個別経路を検証した
  ものではない。

---

## RQ3：メモリ容量

- **RQ**: RQ3
- **Question**: UM・Pure・Chunked は性能と実行可能バッチ範囲へどのような影響を与えるか。
- **RequiredEvidence**: 同一グラフでの 3 方式のバッチ掃引、OOM 境界、oversubscription 経路証拠、
  Chunked の num_subs。
- **AvailableEvidence**:
  - `result/memory_scalability/oversubscribe_results_gpu_opt{,_pure,_pure_chunked}.tsv`
    （325557_3216152, batch 512–16384, n=5, checkpoint `oldtree_f05ec52_20260512`＝旧 tree, **時間値非採用**）：
    feasibility は Pure が **b8192 以上で OOM**、UM が **b10240 まで SUCCESS（b12288 は OOM_OR_FAIL, exit 137, 原因独立未確認）**、
    Chunked が **b16384 まで全 SUCCESS**。
  - `result/correctness/memory_paths/canonical_job_2368587/`（325557, checkpoint `memory_correctness_20260712`,
    **Host-memory-limited 100 GiB configuration**）：UM **b9792 完走**（oversubscribed, SUB_BATCH=6596, num_subs=2, NS_eff=1,
    prefetch cum=33.18 s）、Chunked **b16384 完走**（num_subs=3）。
  - `failure/failed/oom/memory_correctness_2368269/`：UM **b10240 がhost-memory-limited 100 GiB configurationでOOM**
    （dynamic(UM)=213.38 GB, runner_exit=137）。
- **Answer**: UM は Pure がデバイスメモリ確保で OOM する領域（b8192+）でも oversubscription に
  より実行を継続できるが、UM も無制限ではない（旧 tree で b12288 が OOM_OR_FAIL(exit 137)、host-memory-limited 100 GiB configurationで
  b10240 OOM）。Chunked は working set を SUB_BATCH 単位に分割することで、試験範囲で最大の
  実行可能バッチ（b16384, num_subs=3）に到達した。**メモリ方式の主な差は最高性能ではなく
  「実行可能バッチ範囲の拡大」にある**（`SUPPORTED_WITH_LIMITATIONS`）。
- **Limitations**: グラフは 325557 の 1 件のみ。feasibility は `oldtree_f05ec52_20260512` 測定を限定的に再利用
  （メモリサイジングコードが `phase_def_block_20260710` と文字単位同一だが `phase_def_block_20260710` で境界を再実測してはいない）。
  **migration byte 量の直接計測はしていない**。「あらゆる条件で OOM を完全回避」とは書かない。
  Host-memory-limited 100 GiB configurationと旧 tree のホストメモリ条件は異なる（境界が環境依存）。

---

## RQ4：数値整合性

- **RQ**: RQ4
- **Question**: 提案実装の数値結果は、どの範囲で参照実装・他実装と整合するか。
- **RequiredEvidence**: 独立参照との full-vector 比較、same-batch のメモリ経路一致、stress
  条件の差、PathMerge との差、run-to-run 再現性。
- **AvailableEvidence**:
  - `result/correctness/small_full_vector/`：benchmark_7000 / 11023 / chain_200 で Sequential
    （独立参照）vs GPU_Opt の全 BC ベクトル比較、mismatch=0・missing=0・NaN/Inf=0（`SUPPORTED`）。
  - `result/correctness/memory_paths/canonical_job_2368587/`：same_batch（UM/Pure/Chunked b1024）
    mismatch=0（**非 byte 一致**, `SUPPORTED_WITH_LIMITATIONS`）；same_impl_diff_batch（b9792 vs
    b1024, b16384 vs b1024）は `rel_tol=1e-6` を和集合 8 頂点で超過（`NOT_YET_SUPPORTED`）；
    pathmerge_cross は 5/5 DIFF（約 11027 要素, max_rel≈2.0e-3, `NOT_YET_SUPPORTED`）。
  - `result/correctness/memory_paths/analysis/`（run-to-run mismatch=0, tolerance sensitivity）。
- **Answer**: **整合の範囲は階層的である**。(1) 小規模3グラフでは独立参照 Sequential と
  full-vector で一致（`SUPPORTED`）。(2) 325557 の same-batch では 3 メモリ方式が事前設定
  許容内で一致するが byte 一致ではない（`SUPPORTED_WITH_LIMITATIONS`）。(3) stress（大 batch /
  分割）条件では厳格許容を超える構成依存差が残り原因未特定（`NOT_YET_SUPPORTED`）。
  (4) PathMerge（external comparator）との差は別 regime で正誤未決定（`NOT_YET_SUPPORTED`）。
- **Limitations**: headline 4 グラフの独立参照 full-vector は未実施。PathMerge を ground truth
  としない。stress 差を「FP 累積順序が原因」と確定しない。`rel_tol=3e-6` で差が消えることは
  補助情報であり、正式 FAIL を PASS に変更しない。canonical の formal overall status は
  `CORE_FAIL`（隠さない）。
