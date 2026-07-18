# 02 貢献項目

本研究の貢献は次の 4 項目に限定する。「コードを書いた」「測定した」を貢献としては挙げない
（それらは手段であって貢献ではない）。各項目は実測・実装で確認できる範囲に限定する。

---

## 貢献 1：GH200 向けバッチ型全始点 BC 実行基盤の設計・実装
GH200 の計算・メモリ特性を考慮し、**Hybrid BFS（top-down/bottom-up 方向切替）・block 単位の
始点処理（1 CUDA block = 1 ソース）・依存度計算の GPU 並列化（warp 協調／thread-per-vertex の
グラフ依存切替）・2 ストリーム非同期実行（ダブルバッファリング）**を統合した、無向・非重み
グラフ向けのバッチ型全始点 BC 実行基盤を設計・実装した。3 層分離（データ load / host 制御 /
CUDA カーネル）により、共通カーネル（`include/proposed/brandes_kernels.cuh`）の上に 3 つの
メモリ管理方式（UM / Pure / Chunked）を差し替え可能な形で構築した。
- 根拠：`src/proposed/host_um.cu`, `host_pure.cu`, `host_chunked.cu`,
  `include/proposed/brandes_kernels.cuh`, `include/proposed/ablation_config.hpp`。
- 主張しないこと：個々の要素（Hybrid BFS 等）を新発明とは主張しない（[00](00_thesis_positioning.md)）。

## 貢献 2：調整済み GPU baseline に対する主要4グラフでの性能実証
固定 b512 の block GPU_Opt が、**グラフごとに batch を調整した PathMerge tuned に対して、
評価した email-EuAll と roadNet-PA/TX/CA で 1.31〜3.17 倍高速**であることを、同一 checkpoint・
同一 GH200・median 集計・追跡可能な出典で実証した。PathMerge を提案手法に有利化せず、
むしろ tuned（保守的）にした比較でこの結果を得た。
- 根拠：`result/main_performance/proposed_vs_pathmerge/`,
  `result/tables/final_speedup_tables.md`, `result/CLAIMS.md`（主軸 A）。
- 主張しないこと：「あらゆるグラフ」「常に高速」とは書かない。

## 貢献 3：アブレーションによる最適化要因分析
Hybrid BFS・warp 協調・2 ストリームを個別に ON/OFF する 2^3=8 構成のアブレーションにより、
**Hybrid BFS と 2 ストリームが主要な性能寄与を示し、warp 協調の効果はグラフ依存である**ことを、
5 グラフ（synthetic 4 + email）で定量化した。修正版 325557 の主効果は H=1.4767× / W=1.1012× /
A=1.5563×（job 2406254, checkpoint `45352a3`）、合成 4 集約は H≈1.679× / W≈1.066× / A≈1.391×
だが、これは **mixed-checkpoint**（他 3 グラフ = job 2354994、325557 のみ修正版）である。
フェーズ内訳（BFS/backward）とプロファイル（56438_300801・本測定 H1W1A0 と untimed H1W1A1
warmupを含む単一トレースにおける CUDA GPU カーネル時間 backward 63.9%/bfs 36.1%、帯域）を補助資料とした。
- 根拠：`result/ablation/{corrected_325557,synthetic_2354994,email_2354999}/`, `result/profiling/`,
  `docs/kernel_selection_decision.md`。
- 主張しないこと：因果を評価外グラフへ一般化しない。W が常に有効/有害とは書かない。同一
  checkpoint で 4 グラフを再測定したとは書かない（mixed-checkpoint を明記）。

## 貢献 4：UM/Pure/Chunked の容量特性と数値的限界の明確化
共通計算基盤に対する 3 つのメモリ管理方式について、修正版 325557（job 2404743, checkpoint
`45352a3`, targeted boundary, 各 n=1）で **(a) 実行可能バッチ範囲と OOM 種別の区別
（Pure b4096 成功 / b8192 CUDA device OOM、UM b10240 成功 / b12288 cgroup host-memory OOM kill
(exit 137)、Chunked b16384 成功 `SUB_BATCH=6596` num_subs=3）、(b) 実装間 full-vector 整合
（Tier B, 10 比較すべて mismatch=0・非 byte 一致）と (c) 独立 CPU 参照との一致（Tier A, 3 グラフ）**を、
一貫した許容基準（`abs_tol=1e-3`, `rel_tol=1e-6`）で明確化した。容量問題は入力グラフファイル
（≈ 45.35 MB）ではなく **batch × per-source state** の working set が作ること、および容量拡張の便益と
限界（UM も b12288 で host-memory OOM kill）を隠さず記述した点が貢献である。
- 根拠：`result/memory_scalability/corrected_325557/`, `result/correctness/{corrected_325557,small_full_vector}/`,
  `raw_data/corrected_325557/job_2404743/`, `result/CORRECTED_325557_ARTIFACT_PROVENANCE.tsv`。
- 主張しないこと：「UM が容量制約を完全に解消」「Chunked が全条件で OOM 回避」「96 GB を超える
  グラフを格納」とは書かない。CUDA device OOM と cgroup host-memory OOM kill を混同しない。
  migration byte 量を直接計測したとは書かない。PathMerge を ground truth としない。旧 malformed
  入力の CORE_FAIL は historical として保持し現行結論に含めない。

---

### 貢献にしないもの（明示）
- 「GPU で BC を実装した」——手段。貢献 1 は**統合設計**であって実装作業そのものではない。
- 「速度を測った」——手段。貢献 2 は**保守的 baseline に対する実証**である。
- legacy 7 実装比較——現行 block での統一比較は未達（`NOT_YET_SUPPORTED`）のため貢献に含めない。
