# Chapter 9 Correctness and Numerical Behavior

本章では、RQ4（正確性と数値的挙動）に回答する。RQ4 は「提案実装の BC ベクトルは独立参照および異なるメモリ経路とどこまで一致し、どの条件で未解決の差が残るか」である（5.1 節）。中心となる問いは、提案 GPU 実装が、独立参照および異なる memory/batch 経路に対して、どの範囲で数値的に整合した BC ベクトルを生成するか、である。検証の方法、正確性水準の定義、混合許容基準（mixed absolute-relative tolerance）、構造検査、SHA256 記録はすべて 5.10 節で規定したとおりであり、本章では新しい実験条件や新しい許容値を導入しない。

本章は、正確性を単一の PASS/FAIL に集約しない。検証は証拠の強さが異なる複数の水準からなり、本章では次の 7 つに分解して報告する。(1) 小規模グラフにおける独立参照との全ベクトル比較、(2) 同一実装・異バッチの全ベクトル比較、(3) 同一バッチ・異メモリ経路の比較、(4) stress 条件のメモリ経路比較、(5) PathMerge との cross-implementation 比較、(6) 最大 BC のみの補助証拠、(7) 構造検査と非有限値（NaN/Inf）検査である。各比較の実行は correctness-only（各構成 n=1、warmup なし）であり、その時間値は性能評価に用いない（5.5 節）。実行可能性（execution feasibility）と数値的正確性の分離は Chapter 8 で述べたとおりであり、本章は後者のみを扱う。

本章で比較対象として現れる PathMerge は、Chapter 5・6 で述べたとおり、Galliot（path-merging 型 BC アルゴリズム）の第三者実装（上流 `gobardhanm/path-merging-bc`）であり、原著論文著者による公式実装ではない。PathMerge は external comparator であって ground truth ではなく、本章のいかなる比較においても PathMerge の出力を正解として扱わない（5.4 節）。

本章の全数値は、canonical な比較記録（`result/correctness/`、`result/tables/thesis/T5_correctness_summary.tsv`）に基づく。全ベクトル比較の基礎となる raw BC ベクトルは Git 追跡下の `raw_data/correctness/`（失敗系列は `raw_data/unsuccessful/`）に SHA256 検証付きで保存されており、本章の各指標（不一致数、最大誤差、誤差 index、当該 index の両値、最大 BC、非有限値数、SHA256）はこれらの raw ベクトルから再計算して公式記録と一致することを確認した。ただし PathMerge の tuned バッチ比較（9.3 節）のみ、巨大 dump 本体は保存されず、実験時の比較サマリと入力 SHA256 が正式記録である（当該 4 本は currently unavailable として `result/EXTERNAL_ARTIFACTS.tsv` に登録済みであり、archive-time の再解析は実施していない）。

## 9.1 Validation Levels

### 9.1.1 Levels and Comparison Axes

正確性の水準は 5.10 節 Table 5.5 で定義した 4 水準（`full_vector_independent_reference`、`full_vector_same_implementation`、`max_bc_only`、`none`）を用いる。証拠の強さの順序は `full_vector_independent_reference` > `full_vector_same_implementation` > `max_bc_only` > `none` である（`result/CLAIMS.md`）。

比較の軸は 3 種類あり、意味が異なる。第 1 の独立参照比較（independent reference）は、実装系統が異なる Sequential（CPU 逐次）を参照とする比較であり、アルゴリズム実装の誤りを検出する力が最も強い。第 2 の同一実装内比較（same implementation）は、同じ実装のバッチサイズやメモリ経路だけを変えた出力同士の比較であり、構成変更に対する数値的一貫性（numerical consistency）を検証する。この比較は両構成に共通する実装誤りを検出できないため、numerical agreement（数値的整合）を示しても algorithmic correctness（アルゴリズム的正しさ）の証明にはならない。第 3 の cross-implementation 比較は、独立に開発された別実装（本研究では PathMerge）との比較であるが、どちらの実装も正解と保証されないため、差が観測された場合にどちらが正しいかを単独では決定できない。

さらに、tolerance-based agreement と byte identity を区別する。混合許容基準を満たす一致（mismatch=0）は、丸め差程度の範囲で数値が整合していることを意味するが、bit 単位の同一出力を意味しない。byte identity は dump ファイルの SHA256 で管理する。なお `--dump-bc` の出力は 1 行目に実装名を含む header を持つため、異なる実装間ではペイロードが同一でもファイル SHA256 は一致しない。したがって値レベルの同一性は最大絶対誤差が 0 かどうかで判断し、SHA256 は各成果物の同一性検証（provenance）に用いる。

Table 9.1 に、本研究で用いた検証水準と対象、証拠の強さ、限界を整理する。

**Table 9.1: Validation levels used in this thesis, their scope, and evidence strength.**

| Level | Reference | Candidate | Scope | Evidence Strength | Limitation |
|---|---|---|---|---|---|
| full_vector_independent_reference | Sequential (CPU) | GPU_Opt (b512) | benchmark_7000_41459, benchmark_11023_62184, chain_200 | Strongest in this study: all BC elements vs an independently coded reference | Small graphs only; single run (n=1) |
| full_vector_same_implementation (batch consistency) | PathMerge (default batch) | PathMerge (tuned batch) | email-EuAll (b64 vs b2048), roadNet-CA (b32 vs b64) | All elements; detects batch-dependent corruption | Same implementation on both sides; no independent reference |
| full_vector_same_implementation (memory paths) | GPU_Opt / Pure / Chunked (b1024, stress batches) | Same framework, different path or batch | 325557_3216152 | All elements; detects path/batch-dependent divergence | Shared framework code; no independent reference; n=1 |
| Cross-implementation diagnostic (full-vector comparison; achieved agreement: max_bc_only) | PathMerge (b4096) | GPU_Opt / Pure / Chunked vectors | 325557_3216152 | Detects disagreement between independent implementations | Neither side is ground truth; disagreement alone decides nothing |
| max_bc_only | Cross-implementation / cross-variant | Max BC index and value only | Headline 4 graphs; kernel selection (roadNet-PA/TX); legacy feasibility sweep (325557_3216152) | Supporting evidence only | Not a full-vector comparison; not a correctness proof |
| none | — | — | Ablation runs | No BC comparison recorded | — |

> Source: level definitions and ordering from Table 5.5 and `result/CLAIMS.md`; scope from `result/coverage_matrix.tsv`.

### 9.1.2 Mixed Tolerance and Recorded Metrics

全ベクトル比較の判定は、5.10 節で規定した混合許容基準による。reference 値 $r_i$、candidate 値 $c_i$ の各 index $i$ について次を要求する。

$$
\lvert r_i - c_i \rvert \le \mathrm{abs\_tol} + \mathrm{rel\_tol} \cdot \max\!\left(\lvert r_i \rvert, \lvert c_i \rvert\right)
$$

正式な許容値は $\mathrm{abs\_tol}=1\mathrm{e}{-3}$、$\mathrm{rel\_tol}=1\mathrm{e}{-6}$ であり、本章のすべての判定でこの値を変更しない。absolute tolerance（$\mathrm{abs\_tol}$）は BC 値が 0 近傍のときの丸め差を吸収し、relative tolerance（$\mathrm{rel\_tol}$）は値の大きさに比例して許容幅を広げる。本評価の BC 値は最大で $10^{10}$～$10^{11}$ に達するため（例えば roadNet-CA の最大 BC ≈ $6.9\times10^{11}$、325557_3216152 の最大 BC ≈ $3.9\times10^{10}$）、この規模では相対的には $10^{-13}$ 程度に過ぎない差でも絶対差が $10^{-3}$ を超え得る。したがって絶対許容単独の判定は大きい BC 値に対して不適切であり、混合許容は値の大きさに応じた scale-aware な判定を与える。絶対許容の超過は WARN として分離記録し、単独の失敗判定にはしない（5.10 節）。mismatched elements（不一致要素数）は、混合許容を満たさない index の個数である。

各比較で記録する指標は次のとおりである：両ベクトルの長さ（vector length）、片側にしか存在しない index の数（missing indices）、非有限値（NaN/+Inf/−Inf）の個数と位置、混合許容の不一致要素数、最大絶対誤差とその index および当該 index の reference/candidate 両値、最大相対誤差とその index および両値、各ベクトルの最大 BC の index/value、両入力ファイルと入力グラフの SHA256。非有限値が存在する場合、および長さ不一致・欠損 index が存在する場合は、許容判定に関わらず無条件 FAIL である（`scripts/compare_bc_vectors.py`）。

許容値の感度確認（例えば $\mathrm{rel\_tol}=3\mathrm{e}{-6}$ での再判定）は補助情報としてのみ用いる。補助許容で差が消えても正式判定は変更せず、結果に合わせて許容値を事後調整することはしない（5.10 節）。

本章の全比較の要約を Table 9.2 に示す。以降の各節はこの表の各行を検証水準ごとに説明する。

**Table 9.2: Correctness comparison summary. Status is the mixed-tolerance judgment of each comparison (abs_tol=1e-3, rel_tol=1e-6); T5 renders these outcomes as Pass / Core Fail, and labels the cross-implementation diagnostic row "Supported with Limitations" (the observed difference is the supported finding; agreement is undetermined). "n/r" = not recorded in the archived comparison summary. Max BC Match compares the maximum-BC index between the two vectors.**

| Graph | Reference | Candidate | Batch (Ref / Cand) | Vector Length | Missing | Non-Finite | Mismatches | Max Absolute Error | Max Relative Error | Max BC Match | Status |
|---|---|---|---|--:|--:|--:|--:|--:|--:|---|---|
| benchmark_7000_41459 | Sequential (CPU) | GPU_Opt | n/a / b512 | 7000 | 0 | 0 | 0 | 6.05e-09 | 4.56e-15 | Yes (index 4) | PASS |
| benchmark_11023_62184 | Sequential (CPU) | GPU_Opt | n/a / b512 | 11023 | 0 | 0 | 0 | 2.98e-08 | 1.79e-14 | Yes (index 10) | PASS |
| chain_200 | Sequential (CPU) | GPU_Opt | n/a / b512 | 200 | 0 | 0 | 0 | 0.00e+00 | 0.00e+00 | Yes (index 99) | PASS |
| email-EuAll | PathMerge (b64) | PathMerge (b2048) | b64 / b2048 | 265009 | 0 | n/r | 0 | 1.94e-05 | 4.91e-14 | Yes (index 10) | PASS |
| roadNet-CA | PathMerge (b32) | PathMerge (b64) | b32 / b64 | 1965206 | 0 | n/r | 0 | 3.34e-03 | 3.88e-13 | Yes (index 1584888) | PASS (absolute-only warning) |
| 325557_3216152 | GPU_Opt (b1024) | GPU_Opt_Pure (b1024) | b1024 / b1024 | 325557 | 0 | 0 | 0 | 1.53e-05 | 3.02e-14 | Yes (index 272817) | PASS |
| 325557_3216152 | GPU_Opt (b1024) | GPU_Opt_Pure_Chunked (b1024) | b1024 / b1024 | 325557 | 0 | 0 | 0 | 2.29e-05 | 2.87e-14 | Yes (index 272817) | PASS |
| 325557_3216152 | GPU_Opt_Pure (b1024) | GPU_Opt_Pure_Chunked (b1024) | b1024 / b1024 | 325557 | 0 | 0 | 0 | 7.63e-06 | 2.63e-14 | Yes (index 272817) | PASS |
| 325557_3216152 | GPU_Opt (b9792) | GPU_Opt (b1024) | b9792 / b1024 | 325557 | 0 | 0 | 6 | 5.49e+00 | 2.23e-06 | Yes (index 272817) | FAIL (Core Fail) |
| 325557_3216152 | GPU_Opt_Pure_Chunked (b16384) | GPU_Opt_Pure_Chunked (b1024) | b16384 / b1024 | 325557 | 0 | 0 | 6 | 5.36e+00 | 2.85e-06 | Yes (index 272817) | FAIL (Core Fail) |
| 325557_3216152 | PathMerge (b4096) | GPU_Opt (b1024) | b4096 / b1024 | 325557 | 0 | 0 | 11027 | 6.64e+03 | 2.00e-03 | Yes (index 272817) | FAIL (difference observed; correctness undetermined) |

<!-- canonical artifact: T5_correctness_summary (internal ID: T5); augmented here with the two PathMerge batch-consistency rows, vector length, missing, max absolute error, batch, and Max BC Match from the same canonical comparison records -->
> Source: `result/tables/thesis/T5_correctness_summary.tsv`; small full-vector rows from `result/correctness/small_full_vector/correctness_summary.tsv`; PathMerge batch rows from `result/correctness/pathmerge_tuned/{email-EuAll_b64_vs_b2048,roadNet-CA_b32_vs_b64}.md`; memory-path rows from `result/correctness/memory_paths/canonical_job_2368587/comparison_matrix.tsv`. All memory-path and small-graph metrics were recomputed from the Git-tracked raw vectors under `raw_data/correctness/` and agree with the canonical records. The PathMerge cross-implementation row is the representative comparison in T5; all five cross comparisons are enumerated in 9.5.3. "Max BC Match" refers to the index; the values at that index agree within the reported error columns (exact recorded values in the sources and Appendix D).

## 9.2 Small-Graph Independent Validation

小規模グラフの独立参照検証は、Sequential（`src/baseline/sequential.cpp`）を独立参照、GPU_Opt を candidate として、benchmark_7000_41459、benchmark_11023_62184、chain_200 の全 BC 要素を比較したものである（水準 `full_vector_independent_reference`）。実行は checkpoint `small_correctness_20260712`、PBS job `2367583.opbs`、各構成 1 回（n=1）、warmup なしの correctness-only 実行であり、時間値は性能測定として扱わない。GPU_Opt は要求・実効バッチとも 512、`SUB_BATCH=512`、`num_subs=1`、`NS_eff=2` の in-capacity 条件で実行された。

結果は Table 9.2 の第 1～3 行のとおりであり、3 グラフすべてで、ベクトル長は $n$ と一致（7000 / 11023 / 200）、欠損 index 0、非有限値 0、混合許容の不一致要素 0 であった。判定はいずれも PASS である。誤差の詳細は次のとおりである。

- **benchmark_7000_41459**: 最大絶対誤差 6.053597e-09（index 0；Sequential 2549196.725646447、GPU_Opt 2549196.725646441）、最大相対誤差 4.563145e-15（index 1186；11161.53593043016 対 11161.53593043011）。最大 BC は両者とも index 4、値 3935437.257858。
- **benchmark_11023_62184**: 最大絶対誤差 2.980232e-08（index 10；11951000.93285756 対 11951000.93285759）、最大相対誤差 1.789722e-14（index 3092；11789.69389924664 対 11789.69389924643）。最大 BC は両者とも index 10、値 11951000.932858。
- **chain_200**: 最大絶対誤差・最大相対誤差ともに厳密に 0。最大 BC は両者とも index 99、値 9900。この 1 件では dump 本文（header 行を除く全 200 行）が byte 単位でも一致しており、ファイル SHA256 の相違は実装名を含む header 行のみに由来する。

観測された最大相対誤差は $10^{-14}$～$10^{-15}$ のオーダーであり、倍精度演算の丸め水準にある。両入力グラフ・両出力ベクトルの SHA256 は `result/correctness/small_full_vector/SOURCE.md` に記録され、raw ベクトルは `raw_data/correctness/small_full_vector/` に Git 追跡で保存されている。

> Source: `result/correctness/small_full_vector/{correctness_summary.tsv,README.md,SOURCE.md,*/comparison.md}`; raw vectors under `raw_data/correctness/small_full_vector/<graph>/{sequential,gpu_opt}/`; all metrics recomputed from the raw vectors for this chapter and found identical.

この結果が支持する主張は次に限定される：評価した 3 つの小規模グラフにおいて、GPU_Opt の全 BC ベクトルは独立参照 Sequential と混合許容内で全要素一致した。この主張の formal status は `SUPPORTED` である（`result/CLAIMS.md`）。一方、次は主張しない。この結果は email-EuAll・roadNet-PA/TX/CA を含む未検証グラフでの独立参照一致を意味しない。GPU_Opt_Pure、GPU_Opt_Pure_Chunked、および UM oversubscription 固有経路はこの検証の対象外である。また、Hybrid BFS や warp 協調などの内部分岐が専用カウンタで通過検証されたものではなく、この 3 グラフの一致だけで実装全体の完全な正しさが証明されたとは主張しない（5.10 節、`result/COVERAGE.md`）。

## 9.3 Tuned-Configuration Consistency

### 9.3.1 PathMerge Batch-Size Consistency

Chapter 6 の主性能比較では、分母の PathMerge にグラフごとの tuned バッチ（email-EuAll b2048、roadNet-CA b32）を用いた。本節の検証は、この tuned バッチへの変更が既定 b64 と数値的に整合した BC ベクトルを生成することを、同一実装（PathMerge）内の全ベクトル比較（水準 `full_vector_same_implementation`）で確認したものである。roadNet-PA/TX は tuned バッチと default バッチがともに b64 であり、最終表では同じ legacy b64 測定を tuned 値にも採用している。両グラフには別の vector A/B を用いた full-vector comparison artifact が存在しないため、b64 の自己比較を full-vector 検証とは扱わず、正式水準を既存の Max BC 記録に基づく `max_bc_only` とする。本節の full-vector 対象は email-EuAll と roadNet-CA の 2 件だけであり、いずれも checkpoint `phase_def_block_20260710` における各 1 回の correctness-only 実行である。

**email-EuAll（b64 対 b2048、PBS job 2360074）**: 既定 b64（要求 64／実効 64、4141 バッチ）と tuned b2048（要求 2048／実効 2048、130 バッチ）を比較した。clamp は発生していない。両ベクトル長は 265009 で一致、欠損 index 0、混合許容の不一致要素 0 である。最大絶対誤差は 1.943111e-05（index 47159）、最大相対誤差は 4.913736e-14（index 237）であった。最大 BC は両者とも index 10 であり、値は 2384894520.796642（b64）対 2384894520.796650（b2048）と、末尾桁のみが異なる。総合判定は PASS である。

**roadNet-CA（b32 対 b64、PBS job 2362965）**: tuned b32（要求 32／実効 32、61413 バッチ）と既定 b64（要求 64／実効 64、30707 バッチ）を比較した。clamp は発生していない。両ベクトル長は 1965206 で一致、欠損 index 0、混合許容の不一致要素 0 である。最大絶対誤差は 3.339767e-03（index 1423587、当該 index の BC 値は約 $8.6\times10^{9}$）、最大相対誤差は 3.878449e-13（同 index）であった。最大 BC は両者とも index 1584888 であり、値は 686380725021.268311（b32）対 686380725021.267822（b64）である。この比較では最大絶対誤差が絶対許容 $\mathrm{abs\_tol}=1\mathrm{e}{-3}$ を超えるため absolute-only warning が記録されたが、当該 index の BC 値の大きさ（$\sim10^{9}$）に対する相対誤差は $10^{-13}$ オーダーであり、混合許容の不一致は 0 である。総合判定は PASS（absolute-only warning）である。これは 9.1.2 節で述べた「絶対許容単独では超過しても混合許容を満たす場合がある」ことの実例であり、警告の分離記録によって判定の scale-aware 性を保っている。

この 2 比較では、巨大な dump 本体（email 約 7.5 MB×2、roadNet-CA 約 58 MB×2）は保存されず、実験時に生成された比較サマリ（`result/correctness/pathmerge_tuned/`）に両入力の SHA256 を記録した上で破棄された。4 本の original runtime path は email が `build_miyabi/t1_correctness/{bc_b64.txt,bc_b2048.txt}`、roadNet-CA が `build_miyabi/t1_ca_correctness/{bc_b32.txt,bc_b64.txt}` であり、これは実験時の historical build output path を指す記録であって、現在のアーカイブ内の保存場所ではない。email-EuAll および roadNet-CA の比較サマリには、実験時に算出されたベクトル長、欠損数、不一致数および誤差指標が保存されている。一方、比較に用いた BC ベクトル本体は現在のアーカイブに保存されていない。この 4 本は currently unavailable として `result/EXTERNAL_ARTIFACTS.tsv` に登録した（`RetentionStatus=not_retained`、`Availability=currently_unavailable`；SizeBytes と SHA256 は当時の記録に基づく）。このため、NaN、Inf および重複 index の有無をアーカイブ整理時に再検査することはできず、これらを not recorded として扱う（Table 9.2 で n/r と表記）。既存サマリのベクトル長、欠損数、不一致数、誤差指標および PASS 判定は当時の混合許容比較で記録された範囲に限定され、archive-time に vector を再解析して得た値ではない。この点は、保存 vector から非有限値数を再確認できる小規模検証（9.2 節）およびメモリ経路検証（9.4・9.5 節）と異なる。

この検証が示すのは次の 2 点である。第 1 に、評価した PathMerge 実装において、バッチサイズの変更（batch decomposition の変更）が混合許容を超える BC 差を生じなかったこと。第 2 に、Chapter 6 で tuned バッチを採用したことによる明らかなベクトル破損は検出されなかったことである。一方、この検証が示さないことを明確にする。これは同一実装内の比較であるため、PathMerge の独立参照に対する正確性を示さず、PathMerge を ground truth とする根拠にもならない。GPU_Opt の正確性とも無関係であり、評価した第三者実装以外の PathMerge/Galliot 実装一般への一般化もしない。

> Source: comparison summaries `result/correctness/pathmerge_tuned/{README.md,email-EuAll_b64_vs_b2048.md,roadNet-CA_b32_vs_b64.md}`; Git-tracked raw logs `raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_2360074_20260711/pbs_stdout.log` and `raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_2362965_20260711/pbs_stdout.log` (these record the vectors' original runtime paths and byte sizes). The four BC vectors are currently unavailable and are registered as such in `result/EXTERNAL_ARTIFACTS.tsv`; their original runtime paths are listed above. All metrics in this section come from the comparison summaries recorded at experiment time — the vectors were not re-analyzed at archive time.

### 9.3.2 Max-BC-Only Supporting Evidence

全ベクトル比較が存在しない構成では、最大 BC の index/value の一致のみを確認した記録が存在する。この水準は `max_bc_only` であり、supporting evidence（補助証拠）ではあるが、全要素の一致（full-vector agreement）ではなく、独立参照検証でもなく、完全な正確性証明として扱わない（5.10 節）。該当する記録は次の 3 群である。

第 1 に、headline 4 グラフ（email-EuAll、roadNet-PA/TX/CA）では、提案 3 実装（GPU_Opt、GPU_Opt_Pure、GPU_Opt_Pure_Chunked）の最大 BC の index/value が一致し（例えば email-EuAll は index 10・2384894520.80、roadNet-CA は index 1584888・686380725021.27；`result/main_performance/proposed_variants/<graph>/correctness.md`）、さらに第三者実装 PathMerge との最大 BC 一致も記録されている（roadNet-PA/TX/CA は `result/main_performance/seven_implementations/legacy_partial/large/correctness_no_gpu_opt.md`、email-EuAll・roadNet-CA は 9.3.1 節の比較記録の Max BC 行）。この根拠に基づく headline 4 グラフの正確性主張の formal status は `SUPPORTED_WITH_LIMITATIONS` であり、その限定理由は「独立参照との全ベクトル比較は未実施」である（`result/CLAIMS.md`）。第 2 に、kernel selection の forced shared/block 比較（roadNet-PA/TX）における両カーネルの最大 BC 一致であり、これは 7.5 節で述べた。第 3 に、legacy feasibility sweep（Chapter 8 Series A、325557_3216152）の各成功点で記録された最大 BC の一致であり、この水準しかないため「BC 計算としての feasibility」主張は全ベクトル未確認の `NOT_YET_SUPPORTED` に留まる（`result/CLAIMS.md`）。

最大 BC の一致は、最も値の大きい 1 頂点の一致しか保証しない。9.5 節で示すとおり、325557_3216152 の cross-implementation 比較では、最大 BC index が一致したまま約 11,000 頂点の混合許容超過が併存した。この事実は、max_bc_only 水準を full-vector 水準の代替として扱ってはならないことの直接の実例である。

## 9.4 Memory-Path Same-Batch Comparison

同一バッチ・異メモリ経路の比較は、325557_3216152（$n=325{,}557$、graph SHA256 は `result/correctness/memory_paths/SOURCE.md` に記録）において、同一の要求・実効バッチ b1024 で実行した GPU_Opt（Unified Memory）、GPU_Opt_Pure（デバイス専用）、GPU_Opt_Pure_Chunked（sub-batch 分割型、b1024 では非分割）の 3 経路の全ベクトルを、3 組の pairwise 比較で検証したものである（水準 `full_vector_same_implementation`）。実行は canonical checkpoint `memory_correctness_20260712`、PBS job `2368587.opbs`、host-memory-limited 100 GiB 資源構成（5.2 節）、各構成 n=1、warmup なしである。GPU_Opt と Chunked は `SUB_BATCH=1024`、`num_subs=1`、`NS_eff=2` の in-capacity 実行であり、Pure には sub-batch 機構がないため該当値は記録上 `not_applicable` である。runner は 3 構成すべて exit 0 で全長ベクトルを出力した（8.4.3 節 Table 8.4）。

結果は Table 9.2 の第 6～8 行のとおりである。3 組の pairwise 比較（UM 対 Pure、UM 対 Chunked、Pure 対 Chunked）はいずれも、ベクトル長 325557 で一致、欠損 index 0、非有限値 0、混合許容の不一致要素 0 であり、判定は 3 組とも PASS である。最大絶対誤差は 1.525879e-05／2.288818e-05／7.629395e-06 で、いずれも最大 BC 頂点（index 272817、BC ≈ $3.93\times10^{10}$）で観測され、相対では $10^{-16}$ オーダーに相当する。最大相対誤差は 3.015985e-14／2.872713e-14／2.628983e-14 であり、倍精度丸め水準にある。最大 BC は 3 経路とも index 272817 である。

一方、この一致は byte identity ではない。3 経路のベクトル SHA256 は相互に異なり、最大絶対誤差が非零であることから、header 行の差異を除いた値レベルでも 3 出力は同一でない。すなわち、同一バッチであってもメモリ経路によって出力 bit は異なり、その差が混合許容内に収まっている、というのが正確な記述である。混合許容 PASS を「完全一致」と表現しない。

この検証が支持する主張は、「同一 b1024 条件では、3 つの memory-management variants が生成した全ベクトルは混合許容内で一致した」であり、formal status は `SUPPORTED_WITH_LIMITATIONS` である（`result/CLAIMS.md` の Same-batch memory-path consistency）。限定は次のとおりである。これは同一実行基盤内の相互比較であって独立参照との比較ではなく、3 経路に共通する誤りは検出できない。byte identity は成立していない。b1024 の in-capacity 条件に限られ、stress 条件（9.5 節）の正確性を示さない。また 325557_3216152 の 1 グラフ・各構成 1 回の実行に限られ、UM/Pure/Chunked の全条件・全グラフでの正確性証明ではない。

> Source: `result/correctness/memory_paths/canonical_job_2368587/{comparison_matrix.tsv,execution_summary.tsv}`; raw vectors under `raw_data/correctness/memory_paths/325557_3216152/`; recomputed from the raw vectors and found identical.

## 9.5 Stress-Condition Core Failures

### 9.5.1 Same-Implementation Stress Comparisons

stress 条件の比較は、同一実装の大バッチ出力を同一 job 内の b1024 control と比較したものである（水準 `full_vector_same_implementation`、同一実装・異バッチ）。対象は次の 2 組である。第 1 に GPU_Opt b9792（要求・実効 9792、`SUB_BATCH=6596`、`num_subs=2`、`NS_eff=1`、oversubscribed、prefetch 累積 33.1807 s）対 GPU_Opt b1024（要求・実効 1024、`SUB_BATCH=1024`、`num_subs=1`、`NS_eff=2`）。第 2 に GPU_Opt_Pure_Chunked b16384（要求・実効 16384、`SUB_BATCH=6596`、`num_subs=3`、`NS_eff=1`）対 Chunked b1024（同上の in-capacity 設定）である。これらの構成値は各実行の stderr ログに記録された実測値である。UM 側の stress バッチが b9792 であるのは、先行 job `2368269.opbs` で b10240 が本資源構成で OOM となったためである（8.4.3 節）。

結果は Table 9.2 の第 9～10 行および Table 9.3 のとおりである。両比較とも、ベクトル長 325557・欠損 0・非有限値 0 と構造は健全である一方、混合許容の不一致要素が各 6 個観測され、正式判定は FAIL である。不一致 index の集合は、GPU_Opt 側が {7954, 143358, 165886, 228350, 289284, 325556}、Chunked 側が {95156, 143358, 165886, 226184, 228350, 289284} であり、両者は同一でない（共通 4 頂点）。和集合は 8 頂点 {7954, 95156, 143358, 165886, 226184, 228350, 289284, 325556} である。最大相対誤差は GPU_Opt 側が 2.230184e-06（index 7954；b9792 値 224196.2897792182、b1024 値 224196.789779218）、Chunked 側が 2.847745e-06（index 95156；b16384 値 21947.12934601126、b1024 値 21947.19184601126）であり、いずれも正式な $\mathrm{rel\_tol}=1\mathrm{e}{-6}$ を超過する。

誤差の分布には次の特徴が観測された。最大絶対誤差は両比較とも index 289277（BC ≈ $4.13\times10^{7}$）で約 5.49／5.36 であるが、この頂点の相対誤差は約 $1.3\times10^{-7}$ であり混合許容内である。つまり最大絶対誤差の頂点は不一致 6 頂点に含まれず、判定は値の大きさに対して scale-aware に機能している。各比較で不一致と判定された頂点における絶対差は 0.0625～1.5 の範囲にあり、当該頂点の BC 値（約 $2.2\times10^{4}$～$8.9\times10^{5}$）に対する相対誤差は $1.4\times10^{-6}$～$2.9\times10^{-6}$ であった。また、2 つの stress 出力同士（GPU_Opt b9792 対 Chunked b16384）の直接比較でも不一致 4・最大相対誤差 2.847745e-06 が記録されており（`result/correctness/memory_paths/analysis/stress_direct_comparison.tsv`）、stress 側の 2 出力も相互に一致していない。最大 BC は全構成で index 272817 のまま一致しており、この差は最大 BC のみの検査では検出できない。

**Table 9.3: Memory-path stress comparisons and single-factor diagnostics on 325557_3216152 (each configuration n=1; mixed tolerance abs_tol=1e-3, rel_tol=1e-6). Configuration fields are shown as "A / B" for the two compared runs; diagnostics reuse the canonical job 2368587 vectors as "old" references.**

| Comparison | Requested Batch | Effective Batch | SUB_BATCH | Num Subs | NS_eff | Mismatches | Max Relative Error | Status |
|---|---|---|---|---|---|--:|--:|---|
| GPU_Opt b9792 vs GPU_Opt b1024 (canonical stress) | 9792 / 1024 | 9792 / 1024 | 6596 / 1024 | 2 / 1 | 1 / 2 | 6 | 2.23e-06 | FAIL (Core Fail) |
| Chunked b16384 vs Chunked b1024 (canonical stress) | 16384 / 1024 | 16384 / 1024 | 6596 / 1024 | 3 / 1 | 1 / 2 | 6 | 2.85e-06 | FAIL (Core Fail) |
| CONTROL vs old b1024 (non-interference) | 1024 / 1024 | 1024 / 1024 | 1024 / 1024 | 1 / 1 | 2 / 2 | 0 | 2.67e-14 | PASS |
| CONTROL vs T-RESET (forced full memset) | 1024 / 1024 | 1024 / 1024 | 1024 / 1024 | 1 / 1 | 2 / 2 | 0 | 4.25e-14 | PASS (RESET_NOT_DISTINGUISHED) |
| CONTROL vs T-NSEFF (forced NS_eff=1) | 1024 / 1024 | 1024 / 1024 | 1024 / 1024 | 1 / 1 | 2 / 1 | 0 | 6.19e-14 | PASS (NS_EFF_NOT_DISTINGUISHED) |
| CONTROL vs old b9792 (stress reproduction) | 1024 / 9792 | 1024 / 9792 | 1024 / 6596 | 1 / 2 | 2 / 1 | 6 | 2.23e-06 | DIFF (same 6 indices) |
| T-NSEFF vs old chunk b16384 (stress reproduction) | 1024 / 16384 | 1024 / 16384 | 1024 / 6596 | 1 / 3 | 1 / 1 | 6 | 2.85e-06 | DIFF (same 6 indices) |

> Source: stress rows from `result/correctness/memory_paths/canonical_job_2368587/comparison_matrix.tsv`; diagnostic rows from `result/correctness/memory_paths/diagnostic_job_2369632/{comparison_matrix.tsv,execution_summary.tsv,FINAL_STATUS.txt}`; configuration fields from the archived runner stderr logs; affected-index sets and union from `result/correctness/memory_paths/analysis/six_vertex_detail.tsv` (recomputed from the raw vectors for this chapter and found identical).

比較の位置づけとして、同一構成の run-to-run 再現性が記録されている。GPU_Opt_Pure b1024 の別 job 間（`2368398.opbs` 対 `2368587.opbs`）、および PathMerge b4096 の 3 job 間（`2368269`/`2368398`/`2368587`）の同一構成比較は、いずれも不一致 0・最大相対誤差 $9.5\times10^{-14}$ 以下であった（`result/correctness/memory_paths/analysis/run_to_run_comparison.tsv`；formal status `SUPPORTED_WITH_LIMITATIONS`）。stress 比較で観測された相対 $10^{-6}$ 台の差は、この観測された run-to-run 変動より数桁大きい。ただしこの対比は観測の記述であり、差の原因を特定するものではない。

許容値の感度確認（補助情報）では、$\mathrm{abs\_tol}=1\mathrm{e}{-3}$ を固定したまま $\mathrm{rel\_tol}$ を緩めると、不一致数は両 stress 比較とも $2\mathrm{e}{-6}$ で 1、$3\mathrm{e}{-6}$ で 0 となる（`result/correctness/memory_paths/analysis/tolerance_sensitivity.tsv`）。すなわち観測された差は相対 $3\times10^{-6}$ 未満に収まる。しかし正式基準は $\mathrm{rel\_tol}=1\mathrm{e}{-6}$ であり、この感度確認によって正式判定 FAIL を PASS に変更しない。

canonical job の formal な集計は、CORE_MEMORY_PATH の 5 比較中 same-batch 3 PASS・stress 2 FAIL であり、この stress 2 比較の FAIL によって overall status は `CORE_FAIL`（script exit 1 は意図的な判定反映）となった。PathMerge cross 比較（9.5.3 節）は core 判定に必須でない診断（`RequiredForCoreMemoryPath=no`）であり、別途 `DIFFERENCE_OBSERVED` として記録される。runner の実行成功（6/6 構成 exit 0）と correctness PASS は別の状態であり、本研究はこの `CORE_FAIL` を隠さない。同時に、`CORE_FAIL` の適用範囲を正確に述べる。これは 325557_3216152 における canonical メモリ経路比較行列の総合判定であり、本章の全実装・全グラフの出力が不正確であることを意味せず、9.2 節の小規模独立参照 PASS や 9.4 節の same-batch PASS を無効化するものでもない。

原因については、次のいずれにも断定しない。観測された差を浮動小数点誤差だけによるものと断定せず、reset 方式・`NS_eff`・chunking・UM migration のいずれか単独が原因であるとも断定しない（次項の診断参照）。原因に関する仮説の整理と考察は Chapter 10 で行う。

### 9.5.2 T-RESET and T-NSEFF Diagnostics

stress 差の一因子切り分けとして、checkpoint `memory_diagnostic_20260713`、PBS job `2369632.opbs` の診断実験が実施された（事前の静的コード監査で提案された最小診断のうち 2 項目、各構成 n=1、warmup なし）。b1024 の 3 構成、すなわち CONTROL（環境変数なし）、T-RESET（`BC_DIAG_FORCE_FULL_RESET=1`：バッファ再初期化を visited-only reset から full memset に強制）、T-NSEFF（`BC_DIAG_FORCE_NS_EFF_ONE=1`：`NS_eff=1` を強制）を実行し、canonical job `2368587` の保存ベクトル（old b1024 / old b9792 / old chunk b16384）を参照として比較した。実行時の経路カウンタは、CONTROL が full memset 3 回・visited reset 315 回、T-RESET が 318 回・0 回、T-NSEFF が 2 回・316 回であり、各診断スイッチが意図した経路を実際に通したことが記録されている。

診断の結果は次のとおりである（Table 9.3 第 3～7 行）。

- **非干渉の確認**: CONTROL 対 old b1024 は不一致 0（最大相対誤差 2.67e-14）であり、診断計装自体が b1024 の出力を run-to-run 変動を超えて変化させていないことが確認された（`non_interference=verified_mismatch0`）。
- **T-RESET**: CONTROL 対 T-RESET は不一致 0。full memset の強制単独では、b1024 出力に混合許容を超える変化を生じなかった（`RESET_NOT_DISTINGUISHED`）。
- **T-NSEFF**: CONTROL 対 T-NSEFF は不一致 0。`NS_eff=1` の強制単独でも、b1024 出力に混合許容を超える変化を生じなかった（`NS_EFF_NOT_DISTINGUISHED`）。
- **stress 差の再現**: CONTROL・T-RESET・T-NSEFF の各ベクトルと old b9792 の比較は、いずれも同一の 6 index で不一致（最大相対誤差 2.23e-06）となり、T-NSEFF 対 old chunk b16384 も 6 index で不一致（同 2.85e-06）となった。すなわち 9.5.1 節の stress 差は診断側の変更に関わらず同一パターンで再現され、診断計装の副作用ではなく大バッチ構成側の性質であることが確認された。

診断 job の overall status は `DIAGNOSTIC_COMPLETE` である。この診断が切り分けたのは次の範囲に限られる。b1024 において、reset 方式（full memset か visited-only か）の単独変更、および `NS_eff` の単独変更は、いずれも stress 差を再現しなかった。したがって「reset 方式単独が原因」「`NS_eff` 単独が原因」という単純な説明は、この診断の範囲では支持されない。一方、この診断は原因を特定したものではない。単独因子の検査は b1024 側でのみ行われており、大バッチ・sub-batch 分割（`num_subs>1`）・grid/occupancy の変化、またはこれらと他因子の組合せが関与する可能性は排除されていない。未測定の要因を診断結果から消去することはできず、結論として stress 差の原因は未確定である。

> Source: `result/correctness/memory_paths/diagnostic_job_2369632/{comparison_matrix.tsv,execution_summary.tsv,FINAL_STATUS.txt,DIAGNOSIS.md}`; static audit `result/correctness/memory_paths/analysis/Gate_G2_3_audit.md`; all comparison rows recomputed from the raw vectors and found identical.

### 9.5.3 Cross-Implementation Differences with PathMerge

canonical job `2368587.opbs` には、PathMerge b4096（325557_3216152 の掃引実測最良バッチ、6.3 節；80 バッチで全始点を処理）を external comparator とする cross-implementation 比較が診断目的で含まれる（`RequiredForCoreMemoryPath=no`）。比較は 5 行であり、PathMerge b4096 を reference、提案側 5 ベクトル（GPU_Opt b1024／GPU_Opt b9792／GPU_Opt_Pure b1024／Chunked b1024／Chunked b16384）を candidate とする。

5 比較すべてで大規模なベクトル差が観測された。不一致要素数は 11027（対 GPU_Opt b1024）、11030（対 GPU_Opt b9792）、11027（対 Pure b1024）、11027（対 Chunked b1024）、11028（対 Chunked b16384）であり、全体の約 3.4% の頂点に相当する。最大絶対誤差はいずれも約 6.64e+03（index 289444、PathMerge 8903157.975814406 対提案側約 8909796～8909801）、最大相対誤差は 2.000197e-03～2.001870e-03（index 325556、PathMerge 892937.5267641294 対提案側約 894727～894729）である。構造は健全であり（長さ 325557 一致、欠損 0、非有限値 0）、最大 BC は 5 比較すべてで index 272817 が一致する（値は PathMerge 39343001000.107368 対提案側 39343001000.108528～108582 で、末尾桁のみ相違）。正式判定は 5 比較すべて FAIL、FINAL_STATUS 上の cross status は `DIFFERENCE_OBSERVED` である。先行 job `2368398.opbs` は、この同種の差（Pure b1024 対 PathMerge b4096、不一致 11027）を検出した時点で意図的に fail-fast した実行であり、`failure/early_terminated/` に保存されている。

この cross-implementation 差は、9.5.1 節の stress 差とは別 regime である。stress 差が 8 頂点・相対 $3\times10^{-6}$ 未満であるのに対し、cross 差は約 11,000 頂点・相対最大約 $2\times10^{-3}$ であり、規模が 3 桁異なる。補助の許容感度でも、$\mathrm{rel\_tol}=1\mathrm{e}{-5}$ に緩めて 283 頂点の不一致が残る（`result/correctness/memory_paths/analysis/tolerance_sensitivity.tsv`）。また PathMerge 自身の run-to-run 再現性は不一致 0（9.5.1 節）であり、提案側の same-batch 相互一致（9.4 節）も成立しているため、この差は両実装それぞれの内部では再現的な、実装間の系統的な差である。

重要なのは、この差から正誤を決定できないことである。PathMerge は第三者実装の external comparator であって ground truth ではない。325557_3216152 の規模では独立参照（Sequential など）による全ベクトル値は取得されておらず、9.2 節の小規模 Sequential 参照の PASS を 325557 へ外挿することもできない。したがって、この差をもって提案実装が誤りであるとは断定せず、PathMerge が誤りであるとも断定しない。正式な結論は次のとおりである：325557_3216152 における PathMerge との cross-implementation 比較では大規模なベクトル差が観測されたが、独立参照が存在しないため、どちらの結果が正しいかは決定できない。PathMerge との一致という主張の formal status は `NOT_YET_SUPPORTED` である（`result/CLAIMS.md`）。差の性質に関する仮説（実装間の慣習差・アルゴリズム経路差など）の検討は Chapter 10 で行う。

> Source: `result/correctness/memory_paths/canonical_job_2368587/{comparison_matrix.tsv,pathmerge_b4096__vs__*.md,FINAL_STATUS.txt}`; fail-fast record `failure/early_terminated/memory_correctness_2368398/`; all five comparisons recomputed from the raw vectors and found identical.

## 9.6 Answer to RQ4

以上より、RQ4 へ検証水準別に次のとおり回答する。

1. **独立参照との全ベクトル一致（最強の水準）**: 評価した 3 つの小規模グラフ（benchmark_7000_41459、benchmark_11023_62184、chain_200）において、GPU_Opt の全 BC ベクトルは独立参照 Sequential と混合許容内で全要素一致した（不一致 0、欠損 0、NaN/Inf 0、最大相対誤差 $\le 1.79\times10^{-14}$）。この水準の一致はこの 3 グラフに限定される。
2. **同一実装のバッチ整合性**: 評価した PathMerge 実装では、tuned バッチと既定 b64 の全ベクトルが email-EuAll・roadNet-CA で混合許容内一致し（不一致 0）、tuned 採用によるベクトル破損は検出されなかった。roadNet-CA では絶対許容単独の超過（absolute-only warning）が記録されたが、混合許容判定は PASS である。
3. **同一バッチ・異メモリ経路**: 325557_3216152 の b1024 では、UM／Pure／Chunked の 3 経路の全ベクトルが 3 組の pairwise 比較すべてで混合許容内一致した（不一致 0）。ただし SHA256 は異なり、byte-identical ではない。
4. **stress 条件**: 同一実装の大バッチ比較（GPU_Opt b9792 対 b1024、Chunked b16384 対 b1024）では、正式な $\mathrm{rel\_tol}=1\mathrm{e}{-6}$ を超える差が各 6 頂点（和集合 8 頂点、相対 $3\times10^{-6}$ 未満）で観測され、正式判定は FAIL、canonical 行列の overall status は `CORE_FAIL` である。診断では full memset 強制・`NS_eff=1` 強制のいずれの単独変更でも差を再現できず、原因は未確定である。
5. **cross-implementation**: PathMerge b4096 との 5 比較すべてで約 11,027～11,030 頂点・最大相対約 $2\times10^{-3}$ の系統的な差が観測された。独立参照が存在しないため正誤は未決定であり、いずれの実装の誤りとも断定しない。
6. **最大 BC のみの補助証拠**: headline 4 グラフ・kernel selection・legacy feasibility の max_bc_only 一致は補助証拠として記録されるが、全ベクトル一致の代替ではない。

要約すると、提案 GPU 実装の数値的整合は、独立参照に対しては評価した小規模 3 グラフで、同一バッチのメモリ経路間では 325557 の b1024 で、それぞれ混合許容内で確認された。一方、大バッチ stress 条件の同一実装内一致と、PathMerge との cross-implementation 一致は確立されておらず、前者は原因未確定の `CORE_FAIL`、後者は正誤未決定として残る。

<!-- English version (plan.md 8.10): "Independent full-vector agreement was confirmed on the three evaluated small graphs, but numerical agreement under all large memory-stress conditions was not established." -->

検証範囲ごとの formal status を Table 9.4 に要約する。

**Table 9.4: Formal status of the correctness validation scopes. "Recorded Judgment" is the comparison-level outcome; "Claim Status" is the formal status in `result/CLAIMS.md` / `result/coverage_matrix.tsv` where a dedicated claim row exists.**

| Validation Scope | Graphs | Level | Recorded Judgment | Claim Status |
|---|---|---|---|---|
| Small independent full-vector | benchmark_7000_41459; benchmark_11023_62184; chain_200 | full_vector_independent_reference | PASS (3/3) | SUPPORTED |
| PathMerge same-implementation batch consistency | email-EuAll; roadNet-CA | full_vector_same_implementation | PASS; PASS (absolute-only warning) | Recorded as comparison-level PASS (no dedicated claim row) |
| Headline max-BC-only evidence | email-EuAll; roadNet-PA/TX/CA | max_bc_only | Max BC index/value agreement | SUPPORTED_WITH_LIMITATIONS |
| Memory-path same-batch agreement (b1024) | 325557_3216152 | full_vector_same_implementation | PASS (3/3; not byte-identical) | SUPPORTED_WITH_LIMITATIONS |
| Same-configuration run-to-run repeatability | 325557_3216152 | full_vector_same_implementation | PASS (4 pairs, mismatch=0) | SUPPORTED_WITH_LIMITATIONS |
| Memory-path stress full-vector | 325557_3216152 | full_vector_same_implementation | FAIL (2/2; union of 8 vertices) | NOT_YET_SUPPORTED |
| PathMerge cross-implementation agreement | 325557_3216152 | Full-vector comparison (achieved agreement: max BC index only) | FAIL (5/5; DIFFERENCE_OBSERVED) | NOT_YET_SUPPORTED (undetermined) |
| Overall canonical memory-path matrix | 325557_3216152 | — | CORE_FAIL (preserved) | CORE_FAIL (not relabeled) |

> Source: `result/CLAIMS.md`, `result/coverage_matrix.tsv`, `docs/thesis/evidence_matrix.tsv`, `result/correctness/memory_paths/canonical_job_2368587/FINAL_STATUS.txt`, `result/tables/thesis/T5_correctness_summary.tsv`.

この回答には次の限定が付く。

- 独立参照との全ベクトル一致は小規模 3 グラフに限定され、headline 4 グラフや 325557_3216152 を含む他グラフへ拡張しない。3 グラフの PASS は実装全体の完全な正しさの証明ではない。
- すべての correctness 実行は各構成 n=1・warmup なしであり、時間値は性能結果ではない。
- 混合許容内の一致は byte identity を意味しない。許容値（$\mathrm{abs\_tol}=1\mathrm{e}{-3}$、$\mathrm{rel\_tol}=1\mathrm{e}{-6}$）は事前設定のまま変更しておらず、補助の許容感度確認によって正式判定を変更しない。
- `CORE_FAIL` は 325557_3216152 の canonical メモリ経路行列に対する総合判定であり、全実装・全グラフの不正確を意味しない。stress 差の原因は未確定であり、浮動小数点誤差・reset・`NS_eff`・chunking・UM migration のいずれにも断定しない。
- PathMerge は第三者実装の external comparator であり、ground truth ではない。cross-implementation 差の正誤は未決定である。
- stress 差の原因仮説、accumulation order や batch decomposition の影響に関する検討、tolerance 設計の妥当性、追加検証の計画、および論文全体の主張への影響は Chapter 10 で論じる。
