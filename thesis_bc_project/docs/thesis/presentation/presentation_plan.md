# Presentation Plan

## 1. Narrative

本発表は次の流れで構成する。

> Problem → Proposed Framework → Evaluation → Evidence → Limitations → Conclusion

聴衆が発表終了時に理解すべき中心メッセージは次の1文である。

> 固定b512のblock-based GPU_Optは、評価した4グラフにおいて、グラフごとに調整した第三者実装PathMergeより1.31〜3.17倍高速だった。性能向上は統合GPU実行基盤の複数要素から生じ、UM・Pure・Chunkedの比較では性能だけでなくbatch-dependent working setに対する容量特性と数値整合性も明らかになった。

## 2. Time budget

本編 15 スライド、Backup 7 スライド。本編の想定時間合計は 900 秒（15.0 分）である。

英語版のみを読んだ場合の推定合計は約 802 秒、日本語版のみを読んだ場合の推定合計は約 830 秒である。二つのスクリプトは代替であり、合算して読むことは想定していない。

発表時間 **15 分は暫定値**である。リポジトリ内に公式の発表時間指定は存在しない（`docs/`・`result/`・`scripts/` を横断検索して該当なし）。20 分等へ変更する場合は、`scripts/generate_thesis_presentation.py` の `TALK_MINUTES` と `NOTES` の想定秒数を更新して再生成する。

## 3. Slide map

| # | Section | Title | Narrative purpose | Figure |
|---:|---|---|---|---|
| 1 | Main | Design and Evaluation of a Batch-Based GPU Execution Framework for Betweenness Centrality on GH200 | Frame the talk and identify the work | — |
| 2 | Main | Exact All-Sources BC Is Computationally Expensive | Establish the cost and irregularity of exact all-sources BC | F02 |
| 3 | Main | Performance Alone Is Not Enough | Motivate four evaluation axes instead of runtime alone | — |
| 4 | Main | The Proposal Is an Integrated GPU Execution Framework | Present the proposal as one integrated execution framework | F04 |
| 5 | Main | Source Batching Creates a Batch-Dependent Working Set | Locate the capacity constraint in batch-dependent state | F05 |
| 6 | Main | Three Memory-Management Variants Share One Framework | Show the three memory variants as one shared framework | F08 |
| 7 | Main | The Evaluation Separates Performance and Capacity Studies | Separate main-performance graphs from the capacity study | — |
| 8 | Main | GPU_Opt Reduced Runtime on All Four Evaluated Graphs | Report the headline runtime reduction on four graphs | F09 |
| 9 | Main | GPU_Opt Achieved 1.31–3.17× Speedup over the Tuned Comparator | Quantify speedup over the tuned external comparator | F10 |
| 10 | Main | Multiple Execution Components Contributed to Performance | Attribute the gain to multiple execution components | F12 |
| 11 | Main | Memory Variants Expanded the Tested Feasible Batch Range | Report tested feasible batch ranges and failure classes | F13 |
| 12 | Main | Numerical Results Matched within Tolerance but Were Not Byte-Identical | Report numerical agreement and its explicit limits | — |
| 13 | Main | The Evidence Has Clear Boundaries | State the boundaries of the evidence | — |
| 14 | Main | Contributions | Summarize the four contributions | — |
| 15 | Main | The Integrated Framework Improved Performance and Clarified Capacity Limits | Restate the central conclusion and invite questions | — |
| 16 | Backup | Detailed Experimental Environment | Backup: full hardware and software environment | — |
| 17 | Backup | Graph and Batch Parameters | Backup: graph, batch, and working-set parameters | — |
| 18 | Backup | PathMerge Batch-Size Sweep | Backup: PathMerge batch sweep justifying tuning | F11 |
| 19 | Backup | Forced Block-vs-Shared Kernel Comparison | Backup: forced block-vs-shared kernel comparison | F14 |
| 20 | Backup | Phase Breakdown and Profiling Scope | Backup: phase breakdown and profiling scope | F15 |
| 21 | Backup | Detailed Correctness Evidence | Backup: per-comparison correctness detail | — |
| 22 | Backup | Historical Record of the Malformed Input | Backup: historical malformed-input evidence, separated | — |

## 4. Language and design rules

- **スライド面に表示される文字はすべて英語**とする。タイトル、本文、bullet、callout、caption、footnote、表ヘッダ、表セル、chart のタイトル・軸・凡例・データラベル、diagram のノード、注釈、Backup スライドを含む。
- 日本語が存在してよいのは speaker notes のみである。スライド面に仮名・漢字を残さない。
- speaker notes は各スライドに **完全な英語スクリプト** と **完全な日本語説明** の両方を持つ。二つは代替であり、一方だけで発表が成立する。
- 表示書体は Arial に統一する（Yu Gothic 依存はスライド面から除去済み）。
- 16:9、白背景。配色は編集可能図ライブラリ（`docs/thesis/figures/editable/`）から継承する。
- deck title 30 pt 以上、slide title 28 pt 以上、本文 20 pt 以上、図表内文字 16 pt 以上、footnote 14 pt 以上（footnote の使用は最小限）。
- 英語化で行が長くなる場合は文を短くする。font size を下げて収めない。
- 1 スライド 1 メッセージ、bullet は 5 項目以内。
- 結果スライドのタイトルは値の羅列ではなく「何が分かったか」を述べる。

## 5. Claim boundaries enforced in this deck

- PathMerge は評価用に保存した第三者実装の external comparator であり、原著者公式実装でも ground truth でもない。
- graph file size と batch-dependent working set を明確に分離する。
- source batching は graph partitioning ではない。
- UM・Chunked が無制限に OOM を回避するとは記載しない。試験上限を明示する。
- 旧 malformed 入力の結果は Backup B7 に履歴的記録として分離し、現在の結論に混入させない。
- 評価していない GPU・グラフ・PathMerge 一般へ一般化しない。
