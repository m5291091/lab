# docs/thesis — 修士論文 執筆パッケージ（研究設計・結果章）

本ディレクトリは、`thesis_bc_project/` の実測結果（`result/` / `failure/`）と実装
（`src/` / `include/`）だけを正規入力として、修士論文の**研究設計・主張整理・結果章の
下書き**を構成した文書群である。GPU 実験・qsub・性能再測定・コード変更は一切行わず、
既存の実測値のみを参照する（Stage I0）。

## 位置づけ（一行）
> NVIDIA GH200 向けに、バッチ型全始点媒介中心性（BC）計算の GPU 実行基盤を設計し、
> 計算最適化・メモリ管理方式・性能寄与・容量拡張・数値整合性と限界を、一貫した
> 実験条件で体系的に評価した、アーキテクチャ指向の HPC・システム実装研究。

個々の Hybrid BFS・warp 処理・2 ストリーム・Unified Memory を単独の新発明とは主張しない
（[00_thesis_positioning.md](00_thesis_positioning.md)）。

## ファイル一覧
| ファイル | 内容 |
|:--|:--|
| [00_thesis_positioning.md](00_thesis_positioning.md) | 研究の位置づけ・問題設定・システム研究としての新規性 |
| [01_research_questions.md](01_research_questions.md) | RQ1〜RQ4（質問・必要根拠・利用可能根拠・回答・限界） |
| [02_contributions.md](02_contributions.md) | 貢献項目（4件） |
| [03_chapter_outline.md](03_chapter_outline.md) | 章構成と各節の表・図・根拠ファイル |
| [04_method_design.md](04_method_design.md) | 提案 GPU 実行基盤の設計（コード準拠） |
| [05_experimental_setup.md](05_experimental_setup.md) | ハードウェア・ソフトウェア・データ・集計規約 |
| [06_results_performance.md](06_results_performance.md) | RQ1 性能（主要4グラフ + 補助表） |
| [07_results_ablation.md](07_results_ablation.md) | RQ2 最適化要因（H/W/A アブレーション） |
| [08_results_memory.md](08_results_memory.md) | RQ3 メモリ容量（UM/Pure/Chunked） |
| [09_results_correctness.md](09_results_correctness.md) | RQ4 数値整合性（5区分） |
| [10_discussion.md](10_discussion.md) | 考察 |
| [11_limitations.md](11_limitations.md) | 限界 |
| [12_related_work_gap.md](12_related_work_gap.md) | 関連研究とギャップ（一次資料 [R1]–[R20]。Stage L0 で全書誌を独立再検証・公式技術資料を追加） |
| [13_tables_and_figures.md](13_tables_and_figures.md) | 論文掲載の表・図（必須/推奨/補助/付録） |
| [14_claims_wording.md](14_claims_wording.md) | 各主張の使用可能/回避表現と根拠 |
| [evidence_matrix.tsv](evidence_matrix.tsv) | 主張×根拠×支持状態の機械可読行列 |
| [thesis_values.tsv](thesis_values.tsv) | 論文使用の全数値の一元台帳 |
| [references.bib](references.bib) | 確定書誌の BibTeX（[R1]–[R20] + 付随ソフトウェア資料; Stage L0） |
| [SOURCE_AUDIT.tsv](SOURCE_AUDIT.tsv) | 出典監査表（主張×出典×照合方法×検証状態; Stage L0） |

### 出典の検証状態（Stage L0, 2026-07-15）
`TODO_SOURCE` プレースホルダは 0 件（Stage I1 で解消済み）。Stage L0 では、(1) [R1]–[R12] の
全 DOI・書誌を Crossref API と第二系統（出版社ページ・著者公開版・著者公式ページ・公式
リポジトリ）で独立再照合（相違 0 件）、(2) GH200 / CUDA（UM・streams・warp shuffle・atomic）/
OpenMP / SNAP / cuGraph の公式一次資料を [R13]–[R20] として追加、(3) 出典で支持できない記述
2 点を訂正（[R9] の oversubscription 帰属除去、[06](06_results_performance.md) §6.4 の cuGraph
per-level 計算量主張の削除）した。`SOURCE_AUDIT.tsv` の `TodoFile`/`TodoLine` 列は、
TODO マーカーが存在しないため**出典を要する主張の所在**（ファイルと行）を指す。

## 正規入力（これ以外を数値源にしない）
```
result/       … 成功結果・canonical raw（数値源）
failure/      … 失敗・制約・再試行の履歴（成功数値源には使わない）
docs/         … graph_stats, kernel_selection_decision
src/ include/ … 実装（method_design の根拠）
scripts/      … 生成・集計スクリプト
```
`build_miyabi/` を論文数値の直接入力にしない。legacy データは `result/` 内で正式な
legacy baseline として指定されたものだけを使う。

## 中心主張（変更禁止の主要値）
> 固定 b512 の block GPU_Opt は、評価した email-EuAll および roadNet-PA/TX/CA において、
> グラフごとに調整した PathMerge tuned より **1.31〜3.17 倍**高速だった
> （email-EuAll 3.17×, roadNet-PA 1.31×, roadNet-TX 1.51×, roadNet-CA 1.45×; median/median）。

「あらゆるグラフ」「常に」「一般に高速」とは書かない。数値は `result/CLAIMS.md` と一致。

## 一貫して守る記述規則
- 集計は median。median と mean を混在させない。speedup は median/median。
- GTEPS は `n_nodes × n_edges / time`（`n_edges` は無向辺数 m）で統一。
- requested batch と effective batch を区別する。warmup は本試行に含めない。
- OOM を 0 秒として扱わない。取得不能値は `N/A`（`not_recorded` は未記録の意）。
- **PathMerge は external comparator であり ground truth ではない。**
- `mismatch=0` を byte（SHA256）一致と書かない。
- 許容値（`abs_tol=1e-3`, `rel_tol=1e-6`）を事後に変更して PASS 化しない。
- **Hybrid BFS は BFS の top-down/bottom-up 方向切替であり、CPU–GPU hybrid ではない。**
- メモリ2実験を混同しない：`memory_scalability`（checkpoint `oldtree_f05ec52_20260512`, feasibility のみ・
  時間値非採用）と `correctness/memory_paths`（checkpoint `memory_correctness_20260712`/`memory_diagnostic_20260713`, host-memory-limited 100 GiB configuration,
  正確性・診断）は別物。
