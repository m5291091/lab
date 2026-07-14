# Gate G2.3 — stress条件8頂点差の静的コード監査 (read-only)

対象 checkpoint: memory_correctness_20260712
graph: 325557_3216152 (n=325557, avg_deg=19.76)
正式許容: abs_tol=1e-3, rel_tol=1e-6 (不変)。PathMerge は external comparator (ground truth ではない)。

## 監査ファイルと主要関数
- include/proposed/brandes_kernels.cuh
  - find_shortest_paths_opt (BFS top-down/bottom-up), brandes_bfs_kernel_opt
  - accumulate_dependencies_opt (warp-per-vertex backward), accumulate_dependencies_tpv_opt (tpv)
  - brandes_back_kernel_opt / _tpv_opt (CB atomicAdd, undirected /2)
  - reset_visited_batch_kernel (visited-only reset)
- src/proposed/host_um.cu (gpu_opt): choose_tpb_for_graph, memset_subbatch, prefetch_subbatch,
  evict_subbatch_to_host, brandes_gpu_opt_impl (2-stream/sub-batch loop)
- src/proposed/host_chunked.cu (gpu_opt_pure_chunked): 同等 sub-batch loop (SUB_BATCH device buffer 再利用)
- src/proposed/host_pure.cu (gpu_opt_pure): NS=2, sub-batch なし
- src/core/runner.cpp (計測/CB→result コピー)

## 構成差一覧 (「バッチだけ違う」ではない)
| 項目 | GPU_Opt b1024 | GPU_Opt b9792 | Chunked b1024 | Chunked b16384 |
|:--|:--|:--|:--|:--|
| Implementation | host_um | host_um | host_chunked | host_chunked |
| RequestedBatch | 1024 | 9792 | 1024 | 16384 |
| EffectiveBatch | 1024 | 9792 | 1024 | 16384 |
| NS (constant) | 2 | 2 | 2 | 2 |
| NS_eff | 2 | 1 | 2 | 1 |
| SUB_BATCH | 1024 | 6596 | 1024 | 6596 |
| NumSubs | 1 | 2 | 1 | 3 |
| UsesUM | yes(managed) | yes(managed) | no | no |
| UsesChunking | no | 実質 sub-batch(2) | no | yes(3 sub) |
| Streams | 2 | 1 | 2 | 1 |
| AllocationSize | BATCH×N ×2 | BATCH×N ×1 | SUB×N ×2 | SUB×N ×1 (再利用) |
| ResetMethod | **visited-only** (連続 (0,1024) 一致) | **full memset** (sub_off 交互で不一致) | **visited-only** | **full memset** |
| PrefetchMethod | 事前一括 prefetch(GPU) | per-sub-batch prefetch(同期) | 無(device 常駐) | 無(device 常駐, 再利用) |
| EvictMethod | 無 | per-sub-batch evict→host(同期) | 無 | 無 |
| CBUpdateMethod | atomicAdd(managed CB) | atomicAdd(managed CB) | atomicAdd(device CB) | atomicAdd(device CB) |
| UndirectedCorrection | kernel 内 delta/2 (source毎) | 同 | 同 | 同 |
- tpb (choose_tpb_for_graph), max_depth_estimate(=256), use_shared_bfs(=false→block),
  backward kernel(avg_deg≥8→warp-per-vertex) は **全構成同一** (グラフ依存, batch 非依存)。
- SUB_BATCH=6596 は safe_sub_batch = INT_MAX/n = 6596 による制限 (両 stress で同値)。

## source coverage (機械検証; GPU 不使用の index 再現)
| config | outer_batches | sub_batches | missing | dup | 全source厳密1回 | 最終 |
|:--|--:|--:|--:|--:|:--|:--|
| b1024 | 318 | 318 | 0 | 0 | yes | s_start=324608,sub_off=0,sub_n=949 |
| b9792 | 34 | 67 | 0 | 0 | yes | s_start=323136,sub_off=0,sub_n=2421 |
| b16384 | 20 | 60 | 0 | 0 | yes | s_start=311296,sub_off=13192,sub_n=1069 |
→ source 欠落/重複は無い。offset 計算は size_t 昇格で int overflow なし。

## バッファ初期化・再利用
- memset_subbatch / cudaMemset: d_d(0xFF=-1), d_sigma(0), d_delta(0)。d_S/d_S_ends/d_Q_*/d_depth は
  kernel が source 毎に上書き。
- reset_visited_batch_kernel: 前 source の到達頂点 (d_S[0..d_S_ends[depth+1])) のみ d_d/sigma/delta を -1/0/0。
- CB は全実装で開始時に 0 初期化 (um:memset, chunked/pure:cudaMemset)。
- b1024 三実装 (UM/Pure/Chunk) は visited-only reset を使い、相互に混合許容内一致 (mismatch=0)
  → visited-only reset 経路は少なくとも三実装間で整合。
- b9792/b16384 は full memset。full memset と visited-only は「到達頂点のみ dirty」不変条件が
  保たれる限り等価だが、stress 側が b1024 と相違 → 等価性が崩れる条件がある可能性 (未断定)。

## stream/event 同期
- 同一 stream 上で memset→BFS→backward→(evict) は逐次順序保証。
- oversubscribed prefetch は cudaEventSynchronize で毎回同期 (async race は実質無い)。
- pref_done event + cudaStreamWaitEvent で compute stream が prefetch を待機。
- outer batch 間はバッファ再利用前に cudaEventSynchronize(ev_back_e)。
- CB への複数 source/stream 同時 atomicAdd の**順序は非決定** (NS_eff=2 で 2 stream 同時, =1 で単一)。

## CB 累積・無向補正
- brandes_back_kernel_opt: v!=s の各頂点に atomicAdd(CB[v], delta[v]/2) (IS_UNDIRECTED=true)。
- source s 自身は除外、各 source の寄与は 1 回。/2 は指数減算で exact。
- PathMerge は末尾で result[i]=d_bc[i]/2 (提案は source 毎 /2)。数学的に同値, FP 差は無視可。
- 観測差 (BC 後): 0.0625, 0.5, 1.5 → /2 前で 0.125, 1.0, 3.0 (clean dyadic/整数)。

## FP 誤差規模の評価
- run-to-run: max_rel≈1e-14, max_abs≈1e-5 (@最大BC~3.9e10, rel~1e-15), 符号対称。
- stress 差: max_rel≤2.85e-6, abs差 0.0625〜1.5 (値~1e6)。
- CB 和 (325557 項, 部分和最大~1e6) の**加算順序のみ**による誤差上限概算 (仮定: per-source delta が
  全構成で同一, 順序のみ差):  (n-1)·u·max|partial| ≈ 3.26e5 · 1.11e-16 · 1e6 ≈ **3.6e-5 (absolute)**。
- 観測 stress 差 (0.0625〜1.5) は上限 3.6e-5 を **3〜4 桁上回る** → **丸め順序のみでは説明困難**。
  かつ差が clean dyadic (0.125/1.0/3.0) である点も, ランダム丸めより
  「離散的な最短経路寄与差 (sigma or 予測子辺の計数差)」と整合的。※原因は静的監査では未断定。

## 仮説ランキング
| Rank | Hypothesis | Supporting | Contradicting | RelevantCode | MinimalTest |
|:--|:--|:--|:--|:--|:--|
| 1 | 大 sigma(>2^53) の atomicAdd 累積が occupancy(NS_eff)依存で非exact→delta 差 | clean dyadic 差, stress=NS_eff1/b1024=NS_eff2 で系統差, source coverage 完全 | sigma>2^53 を静的に確認できず, road/small-world で稀 | kernels.cuh top-down `atomicAdd(d_sigma)` | 8頂点に効く source の sigma を dump し 2^53 超を確認 |
| 2 | full memset vs visited-only reset の非等価 (stress=memset, b1024=visited) | reset 経路が stress↔b1024 で系統的に異なる | b1024 三実装(visited) 一致 & 概念上は等価; memset は上位集合を消すはずで劣化しにくい | host_um/chunked reset gating, reset_visited_batch_kernel | b1024 で full memset 強制→b9792 パターンに変化するか |
| 3 | num_subs>1 の sub-batch 分割で per-source 計算が変化 (境界/バッファ再利用) | stress は num_subs>1, b1024 は 1; b9792/b16384 共通 | source coverage 完全, 各 source は独立 block で自己完結のはず | host_um/chunked sub-batch loop | b1024 で SUB_BATCH<batch 強制(num_subs=2, NS_eff=2 維持) |
| 4 | global CB atomicAdd 順序差 (batch grouping/stream 数) | 順序は確かに構成依存, ~1e-9 noise floor と整合 | 上限 3.6e-5≪観測 0.5〜1.5 → 単独主要因では説明不可 | back kernel `atomicAdd(CB[v])` | 単一 stream・決定的順序で CB 更新し差消えるか |
| 5 | prefetch/evict と kernel の可視性 race (UM oversubscribed 限定) | b9792 のみ該当 | chunked b16384(prefetch/evict 無)も同種差 → 共通因でない | host_um prefetch/evict | (低優先) evict 無効化 |
| 6 | BFS top-down 非atomic 読み `if(d_d==depth+1)` の可視性 race | 理屈上 race 面 | CAS がこのスレッドに coherence 供給→未計上は起きにくい, sigma 整数 exact | find_shortest_paths_opt | (低優先) sigma 決定性を 2 回実行で確認 |
| — | source 範囲欠落/重複, partial final batch, overflow, undirected 補正位置, PathMerge 固有差 | — | source coverage 完全 / overflow 無(depth≪256) / 補正は全提案同一 / PathMerge は別 regime(~0.2%) | — | 除外済 |

## 最小診断実験案 (この Gate では実装・実行しない; 一度に1因子)
1. **T-RESET**: b1024 で reset gating を full memset に強制 (visited-only 無効化)。他factor(NS_eff=2,num_subs=1)維持。
   → 結果が通常 b1024 と一致なら reset 方式は主要因でない。コード変更: reset if 分岐を無効化(数行)。GPU~2分。
2. **T-NSEFF**: b1024 で NS_eff=1 (単一 stream) 強制。reset=visited, num_subs=1 維持。
   → 変化すれば stream/CB 順序が関与。コード変更: NS_eff=1 固定(数行)。GPU~3分。
3. **T-PROBE**: b1024 と b9792 で affected 8 頂点(7954,95156,143358,165886,226184,228350,289284,325556)に
   寄与する source を特定し、各 source の delta[v] と sigma を dump 比較。per-source 差か CB 順序差かを確定。
   コード変更: デバッグ dump 追加(中程度)。GPU~5分×2。
- 推奨順: まず T-RESET と T-NSEFF (各1因子) → 差の所在を CORE/CB に絞ってから T-PROBE。複数因子同時変更はしない。

## 診断しない場合の論文上の制約 (Stage 4C 記載案)
- メモリ経路(UM/Pure/Chunk) は同一 batch で混合許容内一致 (mismatch=0, 非 bitwise/SHA256 相違)。
- stress(大 batch, NS_eff=1, num_subs>1) は厳格 rel_tol=1e-6 を **和集合8頂点** で超過 (≤2.85e-6, rel_tol=3e-6 で消失)。
  この差は「バッチ・NS_eff・sub-batch 構成に依存する差を観測」と記す。原因(丸め/離散計数/race)は未特定。
- FP 加算順序のみでは観測 stress 差 (0.0625〜1.5) を説明困難 (上限≈3.6e-5)。
- 範囲: 325557・checkpoint memory_correctness_20260712・各構成1回。他グラフ/他 batch/最新 block へ一般化しない。
- PathMerge 差 (~0.2%, 11027 頂点) は本 stress 差とは別 regime, 未解決 (正誤未決定)。
