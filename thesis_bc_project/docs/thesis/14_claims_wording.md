# 14 主張の表現ガイド

## 14.1 RQ1 性能

- **使用可能**：「固定 b512 の block GPU_Opt は、評価した email-EuAll および roadNet-PA/TX/CA において、グラフごとに調整した評価対象の第三者実装 PathMerge より 1.31〜3.17 倍高速だった（email 3.17×、PA 1.31×、TX 1.51×、CA 1.45×）。」
- **避ける**：all graphs、always faster、universally、最速の BC 実装、PathMerge/Galliot 一般または原著者公式実装への一般化。
- **制約**：4 graph、GPU_Opt fixed b512、PathMerge tuned。修正版 325557 は RQ1 に不使用。

## 14.2 RQ2 アブレーション

- **使用可能**：「修正版 325557 の main effect は H=1.4767 / W=1.1012 / A=1.5563。合成 4 graph aggregate は H=1.679 / W=1.066 / A=1.391。」
- **必須注記**：aggregate は mixed-checkpoint（他 3 = job 2354994、修正版 325557 = job 2406254 / checkpoint `45352a3`）。same-checkpoint four-graph remeasurement ではない。
- **使用可能**：「評価した ablation 条件では Hybrid BFS と Dual Streams が主要な正の効果を示し、Warp-Cooperative Accumulation は graph-dependent だった。」
- **避ける**：旧 1.655 / 1.065 / 1.396 を current 値として使用、roadNet への因果一般化、W が常に有効/有害という表現。

## 14.3 Graph File Size and Working Set

- **使用可能**：「修正版 325557 の input graph file は 45,348,105 bytes（45.35 MB / 43.25 MiB）、CSR は 27,031,448 bytes、BC vector は 2,604,456 bytes である。」
- **使用可能**：「容量問題は input graph file ではなく、`EffectiveNS × EffectiveBatch × PerSourceStateBytes` の batch-dependent working set から生じる。per-source state は 10,418,856 bytes である。」
- **避ける**：96 GB graph、graph larger than HBM、入力 graph が HBM3 を超えた、96 GB 以上の graph を UM で格納した。
- **必須区別**：allocation estimate と measured RSS / physical HBM residency / migration bytes。後 3 者は未取得。

## 14.4 Batch and Sub-Batch

- **使用可能**：「batch は source grouping の実行単位で、outer loop が全 source を処理する。Chunked は source batch を sub-batch に分けて同時 resident state を制限する。」
- **避ける**：the graph was split into batches、batch で graph を分割、一部の graph/source だけを計算、近似計算。
- **必須説明**：batch/sub-batch を使用しても全 source を exact に処理し、BC を省略しない。

## 14.5 Unified Memory

- **使用可能**：「UM は managed allocation と migration により device memory を超え得る working set を扱う。修正版 325557 の b10240 は code-derived estimate 106.69 GB が `free_before≈101.4 GB` を上回る条件で成功した。」
- **必須注記**：入力 graph file の格納目的ではない。physical residency / migration bytes は未測定。
- **使用可能**：「UM b12288 は cgroup host-memory OOM kill（exit 137）であり、UM は無制限ではない。」
- **避ける**：UM completely avoids OOM、UM unlimited、CUDA/HBM OOM と cgroup kill の混同。

## 14.6 Pure and Chunked

- **Pure**：「b4096 success、b8192 confirmed CUDA device-memory OOM（exit 1）。b8192 estimate は 170,702,536,704 bytes。」
- **Chunked**：「b16384 success、`SUB_BATCH=6596`、`num_subs=3`、resident estimate 68,722,774,176 bytes。」
- **binding constraint**：「修正版 325557 では `safe_sub_batch=INT_MAX/n=6596` が HBM-budget upper bound より小さく、index-safety が binding。」
- **避ける**：Chunked eliminates OOM、Chunked always avoids OOM、Chunked always fastest。

## 14.7 RQ3 の範囲

- **使用可能**：「試験範囲内の maximum successful batch は Pure b4096 < UM b10240 < Chunked b16384。」
- **必須注記**：corrected 325557、job 2404743 / checkpoint `45352a3`、各条件 1 trial、targeted feasibility only、runtime は formal performance comparison ではない。

## 14.8 RQ4 正確性

- **Tier A**：「小規模 3 graph の independent Sequential CPU reference comparison は full-vector PASS (`SUPPORTED`)。」
- **Tier B**：「修正版 325557 の 6 vector・10 comparison は mixed tolerance 内で全て missing 0 / mismatch 0 / PASS (`SUPPORTED_WITH_LIMITATIONS`)。」
- **全体**：「T5 の 13 comparison はすべて `ByteIdentical=No`。」
- **避ける**：bitwise identical、exactly identical、ground truth、all conditions verified。
- **PathMerge**：external comparator であり ground truth ではない。

## 14.9 Historical Malformed Input

- **使用可能**：「旧 malformed input の `CORE_FAIL` と stress/pathmerge differences は historical invalid-input evidence として保存し、修正版 job 2404743 で再検証した。」
- **必須保持**：`result/correctness/memory_paths/canonical_job_2368587/`、`failure/`、provenance documents、raw vectors/logs。
- **避ける**：「過去に誤っていたため削除した」、旧 `CORE_FAIL` を current failure とする表現、修正版 PASS を旧判定へ遡及適用する表現。

## 14.10 Corrected Graph Provenance

- **使用可能**：「修正版は deterministic internal reconstruction である。」
- **必須制約**：original generation seed / complete upstream original は未確認。`internally_reconstructed_no_original_seed`。
- **避ける**：外部原本と完全同一、provenance が完全に確立したという表現。

## 14.11 Submission Checks

提出前に次を確認する。

- RQ1 値 3.17 / 1.31 / 1.51 / 1.45 が不変。
- RQ2 current 値 1.4767 / 1.1012 / 1.5563 と 1.679 / 1.066 / 1.391 が一致。
- mixed-checkpoint、1 graph、1 trial、repair provenance の制約が記載される。
- CUDA device OOM と cgroup host-memory OOM kill が区別される。
- graph file / CSR / BC vector / per-source state / batch working set が区別される。
- historical `CORE_FAIL` が保存され current conclusion と分離される。
- prohibited wording（all graphs、always faster、universally、ground truth、bitwise/exactly identical、UM completely avoids OOM、Chunked eliminates OOM、96 GB graph、graph larger than HBM、graph split into batches、same-checkpoint four-graph ablation）が current claim に 0 件。
