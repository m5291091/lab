# 03 章構成

各章に使用する表・図・根拠ファイルを併記する。表・図 ID は
[13_tables_and_figures.md](13_tables_and_figures.md) と対応。

| 章 | タイトル | 主な内容 | 使用する表・図（ID） | 根拠ファイル |
|:--|:--|:--|:--|:--|
| 1 | 序論 | 研究背景・問題設定・BC の高コスト性・GH200 の動機・貢献・論文構成 | （図なし可） | [00](00_thesis_positioning.md), [02](02_contributions.md) |
| 2 | 背景 | Brandes アルゴリズム、前向き/後向きフェーズ、GH200 メモリ階層、UM、CSR 表現、GTEPS の定義 | T-BW（帯域）, F-ARCH（概念図・任意） | `raw_data/profiling/job_2359175_20260711/bandwidth.log`, [04](04_method_design.md) |
| 3 | 関連研究 | GPU BC、direction-optimizing BFS、batched/multi-source BC、Galliot/PathMerge、cuGraph、UM オーバーサブスクリプション、out-of-core | T-RELATED（関連研究比較表） | [12](12_related_work_gap.md) |
| 4 | 提案する GPU 実行基盤 | 入力表現・バッチ処理・block 割当・Hybrid BFS・backward・stream 構造・UM/Pure/Chunked・無向補正・計算量/メモリ量 | T-BATCHDEF（用語定義表）, F-ARCH, F-STREAM（stream 重畳図・任意） | `src/proposed/*.cu`, `include/proposed/*.cuh` |
| 5 | 実験方法 | ハードウェア・ソフトウェア・checkpoint・データセット・前処理・集計規約・timing scope・correctness tolerance・OOM/TIMEOUT の扱い・バッチ選択 | T-ENV（環境表）, T-GRAPH（グラフ統計表） | [05](05_experimental_setup.md), `result/environment/`, `docs/graph_stats.md` |
| 6 | 性能評価（RQ1） | 主要4グラフの提案 vs PathMerge tuned、補助 baseline（Seq/OMP/cuGraph, small 限定） | **T-PERF（主性能表・必須）**, T-AUX（補助 baseline 表）, F-PHASE（phase breakdown） | [06](06_results_performance.md), `result/main_performance/`, `result/tuning/pathmerge/` |
| 7 | 最適化要因分析（RQ2） | H/W/A の定義・主効果・交互作用・フェーズ帰属・プロファイル・warp のグラフ依存性 | **T-ABL（アブレーション寄与表・必須）**, T-KSEL（kernel 選択表）, F-NSYS（nsys 要約） | [07](07_results_ablation.md), `result/ablation/`, `result/profiling/`, `result/tuning/kernel_selection/` |
| 8 | メモリ容量評価（RQ3） | UM/Pure/Chunked の feasibility 掃引・OOM 境界・oversubscription 経路証拠・容量と性能のトレードオフ | **T-MEM（feasibility 表・必須）**, F-UM（um_oversubscribe 図） | [08](08_results_memory.md), `result/memory_scalability/`, `result/correctness/memory_paths/` |
| 9 | 正確性・制約（RQ4） | 5 区分の正確性（独立参照/same-batch/stress/PathMerge cross/run-to-run）と支持範囲 | **T-CORR（正確性区分表・必須）**, T-STRESS（stress 差の詳細） | [09](09_results_correctness.md), `result/correctness/` |
| 10 | 考察 | roadNet の高速化差、構造と BFS/backward 負荷、固定 b512 vs tuned の意味、H/A の寄与、W のグラフ依存性、UM/Chunked の用途、性能/容量トレードオフ、PathMerge 差の解釈 | （前章の表を再参照） | [10](10_discussion.md) |
| 11 | 結論 | 貢献の要約・RQ 回答・限界・今後 | （表なし） | [02](02_contributions.md), [11](11_limitations.md) |

## 付録（Appendix）候補
| 付録 | 内容 | 根拠 |
|:--|:--|:--|
| A | PathMerge バッチ掃引の詳細（掃引した各グラフ・各バッチの median） | `result/tables/final_speedup_tables.md`, `result/tuning/pathmerge/*` |
| B | アブレーション全 8 構成の実行時間・フェーズ内訳 | `result/ablation/*/ablation_summary.md` |
| C | memory-path 比較行列・8 頂点詳細・許容感度 | `result/correctness/memory_paths/analysis/*` |
| D | 実行環境・provenance・checkpoint 別出典 | `result/environment/`, `result/provenance/`, `result/MANIFEST.md` |
| E | 失敗・早期終了・OOM の記録 | `failure/README.md`, `failure/MANIFEST.tsv` |

## 各章の完成度（追加実験の要否）
- 追加実験なしで完成可能：2, 3（一次資料調査は必要）, 4, 5, 6, 7, 8, 9, 10, 11。
- 追加実験があると強化される（必須ではない）：headline 独立参照 full-vector（→ RQ4 を
  `SUPPORTED` に格上げ）、現行 block の 7 実装統一表（→ 補助表を格上げ）、提案手法の batch sweep。
- 詳細は [11_limitations.md](11_limitations.md) の「追加実験を要する主張」節。
