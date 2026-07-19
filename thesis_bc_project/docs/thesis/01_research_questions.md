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
  - `result/ablation/corrected_325557/`（修正版 325557_3216152_corrected_v1 × 8 構成 × n=5,
    job 2406254, checkpoint `45352a3`）。旧 malformed 325557 の測定を置換する。
  - `result/ablation/synthetic_2354994/`（benchmark_7000 / 11023 / 56438 × 8 構成 × n=5,
    job 2354994, 生データ不変）。
  - `result/ablation/email_2354999/`（email-EuAll × 8 構成 × n=3）。
  - 主効果（幾何平均）：修正版 325557 で H=**1.4767×**、W=**1.1012×**、A=**1.5563×**；
    合成 4 グラフの mixed-checkpoint 集約で H≈**1.679×**、W≈**1.066×**、A≈**1.391×**；
    email で H≈1.429×、A≈1.720×、W≈0.970×（`ablation_contributions.tsv` / T3）。
  - フェーズ内訳（`ablation_summary.md`）：H は BFS cum を短縮、A は wall を短縮（2 stream 重畳）。
  - プロファイル：`raw_data/profiling/job_2359175_20260711/ablation_H1W1A0.stats.txt`（56438_300801、本測定 H1W1A0 と untimed H1W1A1 warmupを含む単一トレースの CUDA GPU カーネル時間のみ：backward 63.9% / bfs 36.1%）、
    帯域 `bandwidth.log`。
- **Answer**: 評価したアブレーション条件では、**Hybrid BFS と 2 ストリームが主要な性能寄与を
  示し、warp 協調の効果はグラフ依存だった**（W は email で 0.970×＝わずかに悪化、56438 で
  0.992×、benchmark_7000 で 1.175×）。交互作用チェックでは全構成が閾値 10% 未満で強い交互
  作用は検出されず（`SUPPORTED_WITH_LIMITATIONS`）。
- **Limitations**: アブレーションは synthetic 4 + email の計 5 グラフ。**合成 4 集約は
  mixed-checkpoint**（他 3 グラフ = job 2354994、修正版 325557 = job 2406254 / checkpoint
  `45352a3`）であり、同一 checkpoint で 4 グラフを再測定したものではない。因果を実験範囲外
  （headline roadNet 全体など）へ一般化しない。旧 malformed 325557 の集約値（H≈1.655 /
  W≈1.065 / A≈1.396）は historical として保持し、現行主値ではない。フェーズ内訳の gap は
  A=1 で 2 stream 合算のため負値になり得る（バグではなく重畳の証拠）。専用ハードウェア
  カウンタで個別経路を検証したものではない。

---

## RQ3：メモリ容量

- **RQ**: RQ3
- **Question**: 評価した修正版325557グラフにおいて、GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunkedのメモリ管理方式は、実行可能なbatch sizeと観測されたメモリ制約にどのような影響を与えるか。
- **RequiredEvidence**: 同一グラフでの 3 方式の feasibility 境界、OOM 種別、working-set 見積り、
  Chunked の SUB_BATCH / num_subs。
- **AvailableEvidence**（修正版 325557_3216152_corrected_v1, job 2404743, checkpoint `45352a3`,
  各条件 n=1 の targeted boundary validation）:
  - `result/memory_scalability/corrected_325557/feasibility_boundary.tsv` /
    `raw_data/corrected_325557/job_2404743/{feasibility_results,oom_evidence,implementation_manifest}.tsv`：
    Pure **b4096 SUCCESS（65.89 s）/ b8192 CUDA out-of-memory（device, `host_pure.cu:144`, exit 1）**、
    UM/GPU_Opt **b10240 SUCCESS（238.67 s）/ b12288 cgroup host-memory OOM kill（exit 137, oom_evidence=none）**、
    Chunked **b16384 SUCCESS（66.60 s, SUB_BATCH=6596, num_subs=3）**。
  - working-set 見積り（`EffectiveNS × EffectiveBatch × PerSourceStateBytes`,
    per-source = `32n + 4·D_est + 8 = 10,418,856` bytes; code 由来 allocation, 実測 RSS/residency ではない）：
    Pure b8192 ≈ 170.70 GB（free HBM 約 101.4 GB 超で device OOM）、UM b10240 ≈ 106.69 GB、
    UM b12288 ≈ 128.03 GB、Chunked b16384 は同時 resident `6596 × PSS ≈ 68.72 GB`。
- **Answer**: 入力グラフファイル（約 45.35 MB）や CSR topology（約 27.03 MB）が HBM3 を超えるの
  ではなく、**始点ごとの状態配列を複数始点分同時に保持する batch 依存 working set** が容量問題を
  作る。UM は managed allocation と migration により、Pure が device メモリ確保で OOM する領域
  （b8192, 見積り ≈ 170.70 GB）を超えて b10240（≈ 106.69 GB）まで成功できるが、UM も無制限では
  なく b12288（≈ 128.03 GB）で **cgroup host-memory OOM kill（exit 137, CUDA/HBM OOM ではない）**
  になる。Chunked は source batch を SUB_BATCH 単位の sub-batch に分割して同時 resident working set
  を制限し、試験上限 b16384（num_subs=3）まで成功した。いずれの方式も全始点を処理し BC を
  近似・省略しない。**メモリ方式の主な差は最高性能ではなく「実行可能バッチ範囲の拡大と
  resident working-set 制御」にある**（`SUPPORTED_WITH_LIMITATIONS`）。
- **Limitations**: グラフは修正版 325557 の 1 件のみ、各条件 n=1 の feasibility 評価であり、
  実行時間を方式間の正式な性能比較に用いない。**migration byte 量・実測 HBM residency・実測 RSS
  は直接計測していない**（working set は code 由来の allocation 見積り）。「あらゆる条件で OOM を
  完全回避」「Chunked は常に OOM を回避」とは書かない。旧 tree / 旧 malformed 325557 の
  feasibility（b12288 OOM_OR_FAIL 等）は historical として保持し、現行境界ではない。

---

## RQ4：数値整合性

- **RQ**: RQ4
- **Question**: 提案実装の数値結果は、どの範囲で参照実装・他実装と整合するか。
- **RequiredEvidence**: 独立参照との full-vector 比較（Tier A）、修正版 325557 の実装間
  full-vector 整合（Tier B）、PathMerge との差、run-to-run 再現性。
- **AvailableEvidence**:
  - **Tier A（独立 CPU 参照）** `result/correctness/small_full_vector/`（job 2367583）：
    benchmark_7000 / 11023 / chain_200 で Sequential（独立参照）vs GPU_Opt の全 BC ベクトル比較、
    MissingIndices=0・MismatchedElements=0・NaN/Inf=0・PASS（`SUPPORTED`）。
  - **Tier B（実装間整合）** `result/correctness/corrected_325557/`（修正版 325557, job 2404743）：
    6 実装ベクトルにわたる **10 比較すべて MissingIndices=0・MismatchedElements=0・
    ToleranceResult=PASS・ByteIdentical=No**（max_rel ≤ 5.089e-13; same_impl_diff_batch を含む）。
    Max BC は全実装で index 272816 一致。PathMerge は external comparator であり ground truth ではない。
  - T5 は上記 Tier A（3 行）+ Tier B（10 行）の計 13 行で構成する。
- **Answer**: **整合の範囲は階層的である**。(1) Tier A：小規模 3 グラフで独立参照 Sequential と
  full-vector 一致（`SUPPORTED`）。(2) Tier B：修正版 325557 で 6 実装・10 比較が事前設定の
  混合許容内で一致するが byte 一致ではない（`SUPPORTED_WITH_LIMITATIONS`）。これは独立 ground
  truth との一致ではなく同一修正版入力に対する実装間整合であり、PathMerge との一致も数値整合性
  チェックである。
- **Limitations**: headline 4 グラフの独立参照 full-vector は未実施。PathMerge を ground truth と
  しない。「byte 一致」「exact match」とは書かない（SHA256 は相異）。修正版グラフは内部修復
  データであり、元生成 seed / 上流原本を確認できない provenance 制約が残る。旧 malformed 入力で
  得られた stress divergence（former `CORE_FAIL`, `canonical_job_2368587`）は current active
  conclusion から外し、`result/correctness/memory_paths/canonical_job_2368587/` に historical
  invalid-input result として保持する（削除しない）。malformed 入力の発見と修正版での再検証の
  経緯として位置づける。
