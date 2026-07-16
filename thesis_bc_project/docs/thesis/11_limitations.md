# 11 限界

過小評価も過大評価もせず、実測範囲に即して列挙する。各項目は `result/CLAIMS.md`・
`result/COVERAGE.md` の欠損・制約と一致。

## 11.1 評価範囲の限界
1. **評価グラフ4件への限定**：性能主張（RQ1）は email-EuAll と roadNet-PA/TX/CA の 4 グラフのみ。
   「あらゆるグラフ」「常に高速」とは主張しない。
2. **提案手法の batch sweep 未実施**：提案は固定 b512。提案側の最適バッチ探索は行っていない
   （PathMerge のみ tuned）。提案を sweep すれば差が変わる可能性は測定していない。
3. **自然大規模グラフ（V≥5M）不在**：oversubscription 評価は合成 325557 の人為的バッチ強制に
   よる。自然な超大規模グラフでの feasibility は未評価（`COVERAGE.md`「任意強化」）。
4. **アブレーションの範囲**：H/W/A 分解は synthetic 4 + email の 5 グラフ。headline roadNet 全体
   へ因果を一般化しない。

## 11.2 比較の統一性の限界
5. **7 実装統一比較の不足**：現行 block による全グラフ統一 7 実装比較は `NOT_YET_SUPPORTED`。
   medium/large では Sequential/OpenMP/cuGraph が欠け、legacy 提案系は旧 shared 経路。補助 baseline
   は small 限定で提示（[06](06_results_performance.md) §6.4）。legacy shared データと現行 block を
   混同しない。
6. **cuGraph 設定の未確認点**：cuGraph adapter は明示的 /2 補正を持たず、BC スケール整合と
   timing scope（CUDA/RMM/RAFT 初期化を含む）の同条件性が本環境で未確認
   （[05](05_experimental_setup.md) §5.9）。cuGraph を正確性 ground truth や headline 比較に用いない。

## 11.3 正確性の限界
7. **main graph での独立参照 full-vector 不足**：headline 4 グラフの正確性は現状 `max_bc_only`
   （提案3実装間 + 独立参照 PathMerge の Max BC 一致）。独立参照との全ベクトル比較は未実施。
8. **stress full-vector 未支持**：325557 の大 batch / 分割条件で `rel_tol=1e-6` を和集合 8 頂点で
   超過する構成依存差（max_rel≈2.85e-6）。full reset・NS_eff=1 の単独変更では再現せず**原因未特定**。
   `rel_tol=3e-6` で消えることは補助情報で、正式 FAIL を変更しない。
9. **PathMerge 差未解決**：PathMerge b4096 と提案各実装で約 11027 要素・max_rel≈0.2% の差。
   正誤未決定。PathMerge は external comparator（ground truth ではない）。
10. **mismatch=0 ≠ byte 一致**：same-batch の一致は混合許容内であって SHA256 一致ではない。

## 11.4 メモリ評価の限界
11. **migration byte 直接測定なし**：oversubscription 経路証拠は est>free・SUB_BATCH<batch・
    prefetch cum>0 等の間接証拠。UM の HtoD/DtoH migration 総量は直接計測していない
    （um_prefetch は 25 秒部分トレースのみ）。
12. **UM b10240 の host memoryを100 GiBに制限した構成での OOM**：dynamic(UM)=213.38 GB で SIGKILL（runner_exit=137）。
    OOM 境界は環境（ホストメモリ上限）依存であり、旧 tree（b12288 OOM_OR_FAIL, exit 137, 原因独立未確認）と host memoryを100 GiBに制限した構成
    （b10240 OOM）で異なる。単一固定境界として述べない。「完全に OOM 回避」とは書かない。
13. **legacy 結果の利用範囲**：memory_scalability の feasibility は `oldtree_f05ec52_20260512` 測定を限定的に再利用
    （メモリサイジングコードが `phase_def_block_20260710` と文字単位同一）。**時間値は headline 性能に採用しない**。
    `phase_def_block_20260710` で境界を再実測したものではない（`provenance/um_code_diff_audit.md`）。

## 11.5 プラットフォーム・一般化の限界
14. **GH200 単一環境**：全測定は NVIDIA GH200 1 台。マルチ GPU・他アーキテクチャは未評価。
15. **他 GPU への一般化不可**：性能・feasibility 境界は GH200 の HBM3 容量・NVLink-C2C 帯域・
    LPDDR5X に依存。他 GPU（HBM 容量やホスト結合が異なる）へ数値を外挿しない。
16. **統計的検出力**：主要 road/PathMerge は各 n=3、memory-path は各 n=1。分散は小さい（SD は
    平均の数 % 以下）が、少数試行の限界を認識する。

## 11.6 追加実験を要する主張（本 Stage では未達）
| 主張 | 現状 | 追加実験（Stage 外） |
|:--|:--|:--|
| headline 4グラフの厳密性（full-vector） | `max_bc_only` | Sequential（または独立参照）vs 提案の全ベクトル比較（road は Seq が高コスト） |
| stress full-vector correctness | `NOT_YET_SUPPORTED` | 原因切り分け（grid/occupancy/分割の交絡分解）+ 再測定 |
| PathMerge cross 一致の正誤 | `NOT_YET_SUPPORTED` | 高精度独立参照との三者比較 |
| 現行 block の 7 実装統一表 | `NOT_YET_SUPPORTED` | medium/large で Seq/OMP/cuGraph 再測定 |
| UM block の最新性能値 | `NOT_YET_SUPPORTED` | UM 掃引の block 全面再測定 |
| migration byte 総量 | 未測定 | full-run UM profiling（部分トレースでなく全区間） |

## 11.7 追加実験なしで完成できる章（本 Stage の成果で執筆可能）
序論・背景・関連研究（一次資料調査は必要）・提案手法（4章）・実験方法（5章）・性能評価
（6章, RQ1）・要因分析（7章, RQ2）・メモリ容量（8章, RQ3）・正確性（9章, RQ4, 支持範囲を
明記して）・考察（10章）・結論（11章）。すなわち**論文本体は現行の実測値で構成可能**であり、
追加実験は主張の格上げ（`SUPPORTED_WITH_LIMITATIONS`→`SUPPORTED` 等）に限って必要となる。
