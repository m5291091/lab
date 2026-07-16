# 14 主張の表現ガイド

各主張について「使用可能な表現 / 避ける表現 / 根拠と制約」を示す。査読・口頭試問での
過大主張を防ぐためのチェックリスト。数値は `result/CLAIMS.md`・元 TSV と一致。

---

## 14.1 性能 1.31〜3.17×（中心主張）
- **使用可能**：「固定 b512 の block GPU_Opt は、評価した email-EuAll および roadNet-PA/TX/CA に
  おいて、グラフごとに調整した「評価した第三者実装の PathMerge」（tuned; 上流
  `gobardhanm/path-merging-bc`, 論文著者の公式実装ではない）より 1.31〜3.17 倍高速だった（median/median;
  email 3.17×, PA 1.31×, TX 1.51×, CA 1.45×）」。
- **避ける**：「あらゆるグラフで高速」「常に PathMerge より速い」「一般に高速」「最速の BC 実装」。
  既定 b64 比較の「7.15×/1.64×」を tuned 主張と混同する表現。評価した第三者実装に対する結果を、
  PathMerge/Galliot アルゴリズム一般や原著者の公式実装に対する優劣へ一般化する表現。
- **根拠と制約**：`proposed_variants` / `tuning/pathmerge` / legacy b64（PA/TX）。4 グラフ限定、
  提案は固定 b512、PathMerge は tuned（保守的比較）。正確性は headline で `max_bc_only`。

## 14.2 Hybrid BFS
- **使用可能**：「BFS の top-down/bottom-up 方向切替（Beamer らの direction-optimizing）を採用し、
  評価したアブレーション条件で主要な性能寄与（synthetic 主効果幾何平均 ≈1.66×）を示した」。
- **避ける**：「CPU–GPU ハイブリッド」（誤り）。「Hybrid BFS を発明した」。「全グラフで最大の寄与」。
- **根拠と制約**：`ablation/*`、`brandes_kernels.cuh`（α=14/β=24）。既存手法の適用。email では
  1.429×、325557 で 1.395× 等グラフで幅がある。

## 14.3 2 ストリーム（async 2-stream）
- **使用可能**：「host 側 cudaMemsetAsync とカーネルの 2 ストリーム重畳により、評価条件で主要な
  性能寄与（synthetic 主効果 ≈1.40×, email ≈1.72×）を示した」。
- **避ける**：「常に 2 倍高速化」。gap の負値を「バグ」と書く。
- **根拠と制約**：`ablation/*`、`profiling`（HBM3/C2C 帯域差, Copy Engine 独立実行）。gap 負値は
  2 stream 重畳の証拠（‡）。

## 14.4 warp 協調
- **使用可能**：「warp 協調（shfl 還元）による後向き累積の効果は**グラフ依存**であり、高次数寄りの
  グラフで正（bench_7000 1.175×, 325557 1.096×）、email/56438 では中立〜わずかに悪化
  （0.970×/0.992×）だった」。
- **避ける**：「warp 協調が常に高速化」「warp 協調は無効」（どちらも過度）。
- **根拠と制約**：`ablation_contributions.tsv`。専用カウンタ検証はしていない。

## 14.5 Unified Memory（UM）
- **使用可能**：「UM（cudaMallocManaged）により、Pure が OOM する領域（325557, b8192+）でも
  oversubscription で実行を継続できた。ただし UM も無制限ではなく、旧 tree で b12288 が
  OOM_OR_FAIL（exit 137, 原因独立未確認）で失敗し、host memoryを100 GiBに制限した構成で
  b10240 が OOM した」。
- **避ける**：「UM が容量制約を完全に解消」「UM は無制限」「UM でどんな working set も扱える」。
- **根拠と制約**：`memory_scalability`（feasibility, 時間値非採用）、`memory_paths`（b9792 完走）、
  `failure/.../2368269`（b10240 OOM）。境界は環境依存。

## 14.6 Chunked
- **使用可能**：「Chunked は実確保を SUB_BATCH 単位に抑え、試験範囲で最大の実行可能バッチ
  （325557, b16384, num_subs=3）に到達した。主効果は最高性能ではなく実行可能バッチの拡大」。
- **避ける**：「Chunked は全条件で OOM を完全回避」「Chunked が常に最速」。
- **根拠と制約**：`memory_scalability`（b16384 全 SUCCESS）、`memory_paths`（num_subs=3）。325557
  限定。

## 14.7 OOM 回避
- **使用可能**：「試験したバッチ範囲で、UM は Pure より、Chunked は UM より大きなバッチまで到達した
  （feasibility 順序 Pure<UM<Chunked）」。
- **避ける**：「OOM を完全に回避」「容量制約を解決」。
- **根拠と制約**：feasibility 表（[08](08_results_memory.md)）。OOM を 0 秒扱いしない。

## 14.8 exactness（厳密性）
- **使用可能**：「小規模 3 グラフ（bench_7000/11023/chain_200）で独立参照 Sequential と全 BC
  ベクトルが混合許容内で一致（mismatch/missing/NaN/Inf=0）」。「headline 4 グラフでは提案 3 実装
  間 + 独立参照 PathMerge の Max BC が一致（max_bc_only）」。
- **避ける**：「提案実装は厳密性が全条件で検証済み」「全グラフで正しい」。「mismatch=0 だから
  byte 一致」。stress 差を「FP 累積が原因」と断定。
- **根拠と制約**：`small_full_vector`（SUPPORTED, 3 グラフ限定）、`memory_paths`（same-batch は
  SUPPORTED_WITH_LIMITATIONS, stress は NOT_YET_SUPPORTED）。

## 14.9 PathMerge
- **使用可能**：「評価に使用した第三者実装の PathMerge を tuned baseline かつ external comparator と
  して用いた」。「PathMerge と提案の差（325557, 約 11027 要素, max_rel≈0.2%）は正誤未決定」。
- **避ける**：「PathMerge を ground truth として提案の正しさを証明」「PathMerge と一致＝提案が正しい」。
  「PathMerge を固定設定にして提案だけ最適化」（誤り: PathMerge が tuned）。「原著者の公式実装」や
  「PathMerge アルゴリズム一般」と断定する表現。
- **根拠と制約**：`pathmerge_cross`（5/5 DIFF, 未解決）。ground truth ではない。上流
  `gobardhanm/path-merging-bc`（@ `9c231b46`）は論文著者の公式実装ではない第三者実装で、上流に
  明示的ライセンス表記なし（§12.5 [R6]; 再配布可否は未確定でユーザー判断）。

## 14.10 cuGraph
- **使用可能**：「cuGraph（exact, normalized=false, endpoints=false, undirected）を small 限定の
  補助 baseline として掲載した」。
- **避ける**：「cuGraph と厳密に同条件で全グラフ比較」「cuGraph を正確性基準に採用」。
- **根拠と制約**：`cugraph_bc.cu`。/2 補正・timing scope（初期化含む）の同条件性は未確認
  ([05](05_experimental_setup.md) §5.9)。medium/large は欠。

---

## 14.11 禁止表現チェック（提出前に grep）
以下が本文に 0 件であることを確認する：
- 「全グラフ」「あらゆるグラフ」「常に」「一般に高速」「最速」
- 「完全に OOM 回避」「容量制約を解消」「UM 無制限」
- 「PathMerge を ground truth」「PathMerge と一致＝正しい」
- 「byte 一致」（same-batch/ run-to-run について。SHA256 は相異）
- stress 差を「FP 累積順序が原因」と断定する表現
- 主要性能値 3.17 / 1.31 / 1.51 / 1.45 以外への差し替え・逆算
