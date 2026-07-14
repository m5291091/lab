# 07 最適化要因分析（RQ2）

観測事実（実測値）と解釈（考察）を明確に分ける。数値は
`result/ablation/{synthetic_2354994,email_2354999}/` の TSV/summary から。

## 7.1 H/W/A の定義（`ablation_config.hpp`）
- **H（Hybrid BFS）**：BFS の top-down/bottom-up **方向切替**（direction-optimizing）。ON=切替、
  OFF=常に top-down。※CPU–GPU hybrid ではない。
- **W（Warp 協調蓄積）**：後向き依存度累積を warp（32 lane）で協調し `shfl_down` 還元。
  OFF=非 warp 協調経路。
- **A（2 ストリーム非同期初期化）**：`NS=2` のダブルバッファリングと host 側 `cudaMemsetAsync`
  重畳。OFF=単一ストリーム相当。
- 8 構成（2^3）をコンパイル時テンプレートで切替（branch divergence 排除）。

## 7.2 実験対象と試行
- synthetic_2354994：benchmark_7000_41459 / benchmark_11023_62184 / 56438_300801 /
  325557_3216152 × 8 構成 × **n=5**（checkpoint `phase_def_block_20260710`, 2026-07-10）。
- email_2354999：email-EuAll × 8 構成 × **n=3**。
- 計 5 グラフ。ablation ディレクトリに max_bc は無し（正確性レベル `none`）。

## 7.3 観測事実：主効果（フル要因, 幾何平均）
他 2 軸で平均した `T(F=0)/T(F=1)` の幾何平均（`ablation_contributions.tsv` MainEffect）。

| 工夫 | bench_7000 | bench_11023 | 56438 | 325557 | email | synthetic幾何平均 |
|:--|--:|--:|--:|--:|--:|--:|
| H（Hybrid BFS） | 1.536 | 1.782 | 1.965 | 1.395 | 1.429 | **1.655** |
| W（Warp 協調） | 1.175 | 1.007 | 0.992 | 1.096 | **0.970** | **1.065** |
| A（2 ストリーム） | 1.234 | 1.577 | 1.238 | 1.576 | 1.720 | **1.396** |

（synthetic 幾何平均は 4 グラフの MainEffect の幾何平均。email は単一グラフの値。）

### 中央値実行時間の代表例（`ablation_summary.md`）
| 構成 | bench_7000[s] | 56438[s] | 325557[s] | email[s] |
|:--|--:|--:|--:|--:|
| H0W0A0（baseline） | 0.0559 | 3.9776 | 176.4 | 72.07 |
| H1W0A0 | 0.0375 | 2.0125 | 124.6 | 50.34 |
| H0W0A1 | 0.0442 | 3.2774 | 112.1 | 42.50 |
| H1W1A1（full） | 0.0247 | 1.6481 | 73.17 | 30.42 |

## 7.4 観測事実：交互作用チェック
単独寄与（add-one）と除外寄与（leave-one-out）の相対差が閾値 10% 超なら「交互作用あり」。
`ablation_summary.md` の判定：**全 15 セル（synthetic 12 + email 3）が 10% 未満**で「—」
（強い交互作用は検出されず）。最大は bench_7000 の H（9.2%）と bench_11023 の H（8.9%）。

> 注：本アブレーションは主効果と単独/除外寄与を測定し、上記の相対差で交互作用の有無を
> 判定した。厳密な要因計画の交互作用項（分散分析等）は算出していない。

## 7.5 観測事実：フェーズ帰属（`ablation_summary.md`「フェーズ帰属」）
full 構成から各工夫を OFF→ON した際の帰属（例）：
- **H は BFS cum を短縮**：325557 で BFS cum 139.8→82.33 s（Δ=57.52 s）、56438 で 3.363→1.257 s
  （Δ=2.106 s）。email で 40.22→18.27 s（Δ=21.95 s）。
- **A は wall を短縮**：325557 で wall 115.7→73.17 s（Δ=42.51 s）、email で 52.48→30.42 s
  （Δ=22.06 s）。
- **W の Backward cum 短縮は小さい/グラフ依存**：325557 で Bwd 76.64→62.48 s（Δ=14.16 s）だが、
  bench_11023 で Δ=0.0009 s、56438 で Δ=0.0048 s とほぼ無効。
- **gap の注記**：A=1（NS=2）では BFS cum/Backward cum が 2 ストリーム分の合算のため
  gap=wall−(BFS+Backward) が負になり得る。これはバグではなく 2 ストリーム重畳の証拠（‡ 付き）。

## 7.6 観測事実：プロファイル（`result/profiling/`）
- nsys（ablation_H1W1A0）：GPU カーネル時間の内訳は **backward 63.9%（3.36 s）/ bfs 36.1%
  （1.90 s）**。memops は memset が 99.7%。
- 帯域：HBM3 DtoD 1818.6 GB/s、NVLink-C2C Prefetch 177.7 GB/s（A の重畳が効く物理的根拠）。
- um_prefetch は `--duration=25` の **25 秒部分トレース**（HtoD 27.918 MB, CPU faults 85,
  GPU faults 9）。**総量主張には使えない**（部分値）。

## 7.7 解釈（考察・実験範囲内）
- 評価したアブレーション条件では、**Hybrid BFS（H）と 2 ストリーム（A）が主要な性能寄与**を
  示した。H は BFS フェーズの探索コストを削減し、A は初期化とカーネルの重畳で wall を隠蔽する、
  という帰属がフェーズ内訳と整合する。
- **warp 協調（W）の効果はグラフ依存**：高次数寄りの bench_7000（1.175×）・325557（1.096×）で
  正、email（0.970×）・56438（0.992×）で中立〜わずかに悪化。W の便益は後向きフェーズの隣接
  走査が warp 分割で得をするかに依存し、次数分布に敏感と考えられる（[10](10_discussion.md)）。

## 7.8 一般化しないこと
- 因果を **roadNet 全体や headline 4 グラフへ一般化しない**（アブレーションは synthetic 4 +
  email の 5 グラフのみ）。
- 「W は常に有効/常に有害」とは書かない（グラフ依存が観測事実）。
- 専用ハードウェアカウンタで H/W の個別カーネル経路を検証したものではない（正確性レベル `none`）。

## 7.9 補足：BFS カーネル選択（forced shared/block, 選択則非依存）
`result/tuning/kernel_selection/`（roadNet-PA/TX の 2 グラフ, 強制比較, median, n=3）。
| グラフ | shared median[s] | block median[s] | 速度向上 | Max BC 一致 |
|:--|--:|--:|--:|:--:|
| roadNet-PA | 1063.71 | 701.57 | **1.52×** | ✓（index 557532, 一致） |
| roadNet-TX | 1639.16 | 984.59 | **1.66×** | ✓ |
これは自動選択則に依存しない forced 比較。**未測定グラフへ一般化しない**。現行実装は常時 block。
