# BFS カーネル: 常時 block の採用と forced shared/block 比較

本ドキュメントは、提案手法 (`gpu_opt` / `gpu_opt_pure` / `gpu_opt_pure_chunked`) の
BFS フェーズにおけるカーネルを **「常時 block (1 ブロック = 1 ソース)」** とした
根拠を、roadNet-PA/TX の **forced shared/block 直接比較**（実測）とともに記録する。

## 1. 結論

- BFS カーネルを **常時 block** とした。**アーカイブ済みの formal 根拠は roadNet-PA/TX の forced shared/block 強制計測**（`result/tuning/kernel_selection/`）に限定する。両グラフで block が優位（PA 1.52× / TX 1.66×）かつ Max BC 一致を確認した。**未測定/未アーカイブのグラフへは一般化しない。**
- **旧選択則の扱い**: 旧実装には平均次数に基づく自動選択則（`avg_deg < 5 → shared-frontier`）が存在したが、**現在は使用していない**（設計経緯のみ。現行方式ではない）。
- 実装: `src/proposed/host_um.cu` の `heuristic_shared` を `false` 化 (既定 block)。
  `host_pure.cu` / `host_chunked.cu` は BFS の shared 分岐を `if (false)` で無効化。
  再現実験用に `BC_FORCE_BFS_KERNEL=shared|block` の強制切替と shared カーネル呼び出しは温存。
- `max_depth_estimate` / `choose_tpb` / バックワード分岐 (`avg_deg < 8`) は BFS カーネルとは
  独立のため**変更していない**。

## 2. BFS カーネル 2×2 実測 (アーカイブ済み = roadNet-PA/TX)

計測条件: `gpu_opt` (host_um.cu)、`BC_FORCE_BFS_KERNEL` で shared / block を強制、
`BC_BATCH_OVERRIDE=512`、GH200 単一 GPU。各セルは複数試行 (n=3) の中央値 [秒]。
**出典・再現可能なアーカイブは以下の2グラフのみ**（`result/tuning/kernel_selection/{roadNet-PA,roadNet-TX}/`）。

| グラフ | 種別 | avg_deg | max_deg | shared (s) | block (s) | 勝者 | 倍率 (shared/block) |
|:-----|:---|------:|------:|------:|------:|:----:|------:|
| roadNet-PA      | 道路網           |  2.83 |    9 | 1063.712 | 701.573 | **block** | 1.52× |
| roadNet-TX      | 道路網           |  2.79 |   12 | 1639.165 | 984.587 | **block** | 1.66× |

- **アーカイブ済みの roadNet-PA/TX で block が shared に勝利**。avg_deg が閾値 5 を下回るハブなし道路網でも block 優位。**この結論を未測定グラフへ一般化しない。**
- shared / block で **Max BC は完全一致** (roadNet-PA は shared/block とも
  index 557532, BC=151395302679.08)。正しさは同等。
- 常時 block 化後の正確性再確認 (job 2355971): `benchmark_7000_41459` / `email-EuAll` で
  auto(=block) と 強制 shared の `--dump-bc` が数値的に一致
  (max_rel_diff ~1e-14、浮動小数点の加算順序差のみ)。
- 注: email-EuAll・合成グラフに対する 2×2 強制計測も試行中に取得したが、**それらの生 TSV は `result/` に出典付きでアーカイブしていない**ため、本ドキュメントでは **formal な倍率主張として用いない**（build_miyabi の未追跡データ。必要なら別途アーカイブが前提）。

## 3. forced shared/block の速度向上（アーカイブ済み roadNet-PA/TX）

強制実行した shared / block の中央値実行時間から、速度向上 = 遅い側 / 速い側 を求める
（選択則には依存しない直接比較）。

| グラフ | shared 中央値 [s] | block 中央値 [s] | 速い側 | 速度向上 (遅/速) |
|:-----|------:|------:|:----:|------:|
| roadNet-PA | 1063.7 | 701.6 | **block** | **1.52×** |
| roadNet-TX | 1639.2 | 984.6 | **block** | **1.66×** |

- roadNet-PA/TX とも block が速く、それぞれ 1.52× / 1.66×。Max BC は shared/block で一致（§2）。
- **未測定グラフへの一般化・幾何平均化はしない**（アーカイブ済みは PA/TX の2点のみ）。
- 旧実装の平均次数選択則（`avg_deg < 5 → shared`）は現在使用していないため、「選択則ペナルティ」としては論じない（forced 比較の実測のみを示す）。

- **アーカイブ済みの道路網 (roadNet-PA/TX)** ではハブがなく (max_deg 9〜12)、block 優位幅は
  PA 1.52× / TX 1.66× と小さいが依然 block が優位。
- ハブを持つグラフ (email 等) では block 優位幅がさらに大きい傾向が試行計測で観察されたが、
  **それらの生データは `result/` に出典付きでアーカイブしていないため、具体倍率を formal 主張として用いない**。
- block はアーカイブ済み PA/TX で勝つため、**max_deg は選択基準としては不要**（アーカイブ範囲での結論）。
- 旧実装で観測された「道路網では block が約 3 倍負ける」という過去計測は、
  現行 block 実装 (hybrid top-down/bottom-up BFS・tpb 調整込み) では**再現しない**
  (現行では roadNet-PA/TX で block が 1.5× 前後勝つ)。計測が旧設計の誤りを検出した形。

## 5. legacy 影響分の再計測と対 PathMerge 勝敗 (Phase D 確定)

常時 block 化後のバイナリで email-EuAll / roadNet-PA/TX/CA を再計測した (実装 `gpu_opt`,
中央値, n=3〜5)。旧 legacy 値は全て shared 経路のもの。PathMerge は **既定バッチ (b64,
legacy result_paper 実測)** と **掃引で得た最良バッチ (tuned)** の双方に対して speedup を示す。
全数値は実測 TSV の中央値であり、比率からの逆算値は用いていない。

### 5.1 再計測値 (常時 block) と対 PathMerge speedup

| グラフ | 旧提案 shared [s] | 新提案 block [s] | 短縮 | PathMerge 既定 b64 [s] | vs 既定 | PathMerge tuned [s] (batch) | vs tuned |
|:-----|------:|------:|:---:|------:|------:|------:|------:|
| email-EuAll | 190.95 | 30.81 | 6.20× | 220.39 | **7.15×** | 97.80 (b2048, 実測) | **3.17×** |
| roadNet-PA  | 1062.56 | 699.52 | 1.52× | 918.67 | **1.31×** | 918.67 (b64, 実測) | **1.31×** |
| roadNet-TX  | 1636.10 | 980.13 | 1.67× | 1482.68 | **1.51×** | 1482.68 (b64, 実測) | **1.51×** |
| roadNet-CA  | 3494.98 | 2129.10 | 1.64× | 3499.03 | **1.64×** | 3079.72 (b32, 実測) | **1.45×** |

> ※ TX/CA の PathMerge tuned は**掃引実測で確定** (GH200, n=3 中央値, checkpoint phase_def_block_20260710)。
> TX 最適 = b64 (PA と一致)。**CA 最適 = b32** (実測; PA/TX から推定した b64 最適は CA に
> 一般化せず; 掃引 b16=3610 > b32=3080 < b64=3492 < b128=3831 の内部最小)。CA は tuned が b64→b32 に
> 下がるため vs tuned は 1.64×→**1.45×** に低下するが提案手法が依然勝利。

- **勝敗の逆転**: shared 経路では roadNet-PA が PathMerge に対し **0.86× (敗北)**、TX 0.91×、
  CA 1.00× と拮抗/敗北していた。常時 block 化により PA **1.31×**、TX **1.51×**、CA **1.64×** と
  全 4 グラフで勝利に転じた。email-EuAll は **1.15× → 7.15×** と大幅改善。
- **PathMerge tuned との比較 (実測確定)**: **PA・TX は掃引実測で最適バッチ = b64** と既定に
  一致するため vs tuned = vs 既定 (PA 1.31× / TX 1.51×)。**CA は実測で最適 = b32**
  (実測で b32 最適) のため tuned が b64→b32 に下がり、vs tuned は 1.64×→**1.45×**
  に低下するが依然勝利。一方 email-EuAll は PathMerge が大バッチ (b2048) で 220→98s と
  約 2.25× 高速化するため vs tuned は 3.17× に低下するが、提案手法が依然勝利。

### 5.2 PathMerge バッチ掃引 (最適バッチの確定・収束)

各グラフで PathMerge のバッチサイズを掃引し最適点を確定した (中央値, `run_pathmerge_sweep`)。

| グラフ | 掃引バッチ別 median 実行時間 [s] | 最適 | 収束根拠 |
|:-----|:---|:---:|:---|
| roadNet-PA | b8=2715, b16=1574, b32=1016, **b64=941**, b128=1106, b256=1155, b512=1207 | **b64** (実測) | b32→b64 で 7.4% 改善・b64→b128 で悪化する内部最小。小バッチ (b8/16/32) ほど低速のため b4 探索は不要 |
| roadNet-TX | b32=1621, **b64=1491**, b128=1669 (b32/b64 は n=3) | **b64** (実測) | b32>b64<b128 の内部最小。PA と一致 |
| roadNet-CA | b16=3610, **b32=3080**, b64=3492, b128=3831 (b32/b64 は n=3) | **b32** (実測) | b16>b32<b64 の内部最小 (実測)。PA/TX から推定した b64 最適は CA に一般化せず |
| email-EuAll | b8=787, b16=491, b64=226, b256=126, b512=106, b1024=100, **b2048=98**, b4096=102, b8192=103 | **b2048** | b2048 前後の U 字。b1024→b2048 で 2%・b2048→b4096 で悪化 |
| 325557 (参考) | b512=240.2, b1024=195.2, b2048=175.4, **b4096=167.574**, b8192(実効6018)=168.266 | **b4096** | b4096→b8192 は約 **0.41% 悪化** (167.574→168.266s)。b8192 は HBM3 予算で実効 6018 にクランプされこれが実質的上限。b4096 が内部最小のため追加探索不要 |

- **道路網 (PA/TX/CA)**: 平均次数が小さくフロンティアが浅いため、極端に小さいバッチでは逐次
  バッチ数が増えて GPU 使用率が落ち低速化する。実測最適は **PA=b64 / TX=b64 / CA=b32**。
  PA/TX は b64 が内部最小だが、**CA は実測で b32 が最速**となった (b16>b32<b64<b128)。
  PA/TX から推定した「b64 一律」は CA には一般化せず、掃引実測により CA=b32 と判明した
  (実測の意義が現れた箇所)。
- **ハブ有り (email) / 密 (325557)**: 大バッチほど並列度が上がり高速化し、HBM3 のメモリ予算が
  実質的上限となる (email は b2048、325557 は実効 6018 付近)。

### 5.3 forced shared/block 速度向上の再確認 (アーカイブ済み roadNet-PA/TX)

§2・§3 のアーカイブ済み forced 実測 (roadNet-PA/TX) を再掲する（選択則には依存しない）。

| グラフ (アーカイブ済み) | shared/block 速度向上 (遅/速) |
|:-----|------:|
| roadNet-PA | 1.52× |
| roadNet-TX | 1.66× |

- roadNet-PA/TX とも block が shared より 1.52× / 1.66× 高速（Max BC 一致）。
- **未測定グラフを含む幾何平均・最悪値の主張はしない**（アーカイブは PA/TX の2点のみ）。

## 6. 卒論用記述案

> 提案手法の BFS フェーズについて、shared / block を強制した制御実験 (forced 比較) を
> アーカイブした道路網 roadNet-PA/TX の2グラフで行ったところ、**block-per-source カーネルが
> shared-frontier より高速 (それぞれ 1.52 倍・1.66 倍) で、両者の Max BC は完全に一致する**
> ことを確認した (未測定グラフへの一般化はしない)。旧実装には平均次数に基づく自動選択則
> (avg_deg < 5 → shared) が存在したが、現在は使用しておらず、BFS カーネルは常に block で
> 実行する。この block 実装により、道路網 roadNet-PA における先行手法 PathMerge に対する
> 勝敗が敗北 (0.86 倍) から勝利 (1.31 倍) へ逆転した。
> PathMerge のバッチサイズを掃引しても roadNet-PA・TX の最適バッチは既定 (b64) に一致する
> (実測)。roadNet-CA は実測により最適が **b32** と判明した (PA/TX から推定した b64 は CA に
> 一般化せず) が、最適化後 (tuned) の PathMerge に対しても提案手法が 1.45 倍勝利し、勝敗は
> 変わらない。すなわち、制御実験による計測が旧実装の設計誤りを検出し、その修正が最終的な
> 性能比較の結論を変えた事例である。

---
*計測データ出典 (全て git 管理下): `result/tuning/kernel_selection/` (2×2 強制計測 = roadNet-PA/TX のみ),
`docs/graph_stats.tsv` (次数統計),
`result/main_performance/proposed_variants/<graph>/results.tsv` (Phase D 常時 block 再計測),
`result/tuning/pathmerge/<graph>/*.tsv` (PathMerge バッチ掃引),
`result/main_performance/seven_implementations/legacy_partial/{medium,large}/results_no_gpu_opt.tsv` (PathMerge 既定 b64 実測),
`result/tables/final_speedup_tables.md` (マージ済み最終表, scripts/merge_final_tables.py 生成).*
