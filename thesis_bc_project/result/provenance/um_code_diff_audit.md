# UM legacy データ コード差分監査（oldtree_f05ec52_20260512 ↔ phase_def_block_20260710）

副次(B) の legacy UM データ（`../memory_scalability/`）が現行 checkpoint と比較可能かを、
**実コード差分**で検証した記録（推定ではない）。

## 測定 commit
- legacy UM データ（`oversubscribe_results_gpu_opt{,_pure,_pure_chunked}.tsv` + `um_experiment_*.log`）を
  コミットした本体 = **`oldtree_f05ec52_20260512`（2026-05-12, 旧 mylab/research ツリー）**。
- 測定コード = 旧 `brandes_gpu_opt{,_pure,_pure_chunked}.cu` @ oldtree_f05ec52_20260512。

## diff oldtree_f05ec52_20260512 → phase_def_block_20260710（メモリ/OOM/BC 関連）
| 項目 | 検証結果 |
|:--|:--|
| メモリサイジング（`per_batch_mem` 公式, `safety=free_mem*0.15`, `BATCH_PER_STREAM`, `dynamic_bytes`, `oversubscribed = dynamic_bytes > free_mem*0.90`, `NS_eff`, `hbm3_budget=free_mem*0.80`, `SUB_BATCH`, `num_subs`） | **文字単位で完全一致**（旧 brandes_gpu_opt.cu L758–800 ≡ 新 host_um.cu L269–311） |
| メモリ確保（`cudaMallocManaged` 対象配列） | 同一 |
| prefetch/evict（`prefetch_subbatch`/`evict_subbatch_to_host`） | 同一 |
| BFS カーネル選択（325557: avg_deg=19.758 ≥ 5.0） | 旧ヒューリスティクスでも **block**。新は常時 block。→ **325557 は旧新とも block** |
| BC 集計（`accumulate_dependencies_opt`/`_tpv_opt`, `δ*(1+δ)`, 無向 `/2.0`） | 保存（3層分割で `brandes_kernels.cuh` へ抽出） |
| reset_visited gating（[P0-A]） | 旧新とも存在 |

## 結論
- **メモリサイジングコードに変更なし**（`per_batch_mem` / `oversubscribed` 判定 / `NS_eff` / `SUB_BATCH` 等が oldtree_f05ec52_20260512 ↔ phase_def_block_20260710 で文字単位同一）。**したがって legacy UM の SUCCESS/OOM 結果を、限定的な feasibility 根拠として再利用する**。ただし **phase_def_block_20260710 で同じ境界を再実測したものではない**（あくまで旧 tree oldtree_f05ec52_20260512 の測定）。
- **BC 数値結果に影響する変更: なし**（同一集計カーネル + `/2.0`）。
- **旧時間値を現在性能値として使用できない理由**: oldtree_f05ec52_20260512 は checkpoint の約2ヶ月前・別セッション測定で、`phase_def_block_20260710` 上で未再検証。時間はセッション要因（driver/CMake/thermal/co-tenancy）依存。(A) は checkpoint 測定値を使うため方法論的整合性の観点でも混在不可。
- **feasibility として再利用する根拠と限界**: SUCCESS/OOM は上記の文字単位同一メモリサイジング + 同一割当で定まる部分が支配的で、reset/prefetch/block 差は時間のみに影響し割当量には無影響。よって feasibility 傾向（pure が先に OOM、UM がより大きなバッチまで到達、chunked が最大）は再利用可能と判断する。**ただし phase_def_block_20260710 上での境界の再実測は行っていない**ため、断定的な「境界不変」ではなく限定的根拠として扱う（厳密確認は C1 の代表 full-vector + 再実測）。
- **正確性根拠**: um_experiment_*.log の各 run に `Maximum Betweenness Centrality ==> 39343001000.11`（独立参照 PathMerge と一致）。FP 順序差で3値（.11/.61/1223.28, 相対誤差 ≤5.7e-9）。全ベクトル照合は未実施（→ C1）。
