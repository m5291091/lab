# lab v2 (HBM3 ストリーミング適用版) のコードベース評価と修正指示

このドキュメントは、`lab/mylab/research/brandes_gpu_opt.cu`（HBM3 ストリーミング設計を反映した版）と
`lab/mylab/research/brandes_gpu_opt_pure.cu`（変更なし）の現状を実コードで読み、
**「研究として成立させるために何が問題で、何をどう直すか」**を、
すべてファイルパス + 行番号 + 具体コードで指示するためのもの。
別 AI に渡して機械的に実装させる前提。

---

## 0. 結論サマリ

| | 状況 |
|---|---|
| アルゴリズム正当性 | OK（カーネル無改造、シフトポインタ方式、結果不変） |
| pure 版が触られていない | OK（意図どおり、OOM が論文主張の根拠） |
| HBM3 streaming の構造 | OK（設計どおり実装、3 ヘルパ + サブバッチループ） |
| **in-capacity 性能の保全** | **NG（reset_visited 最適化が消失）** ← 最優先で修正 |
| **prefetch overhead の定量計測** | **NG（prefetch_ms ログなし）** ← 論文の数値根拠が取れない |
| **比較対象が pure と UM の 2 つしかない** | **NG（手動 chunking pure 版が必要）** ← reviewer が必ず聞く |
| 実用シナリオの欠如 | NG（人為的 oversubscribe のみ、自然な大規模グラフがない） |
| 軽微なリーク・ガード抜け | 4 箇所（修正容易） |

研究として通すには **(1) reset_visited 復活 (2) prefetch_ms 計測 (3) 手動 chunking pure 版の追加**
の 3 つを揃える必要がある。これがあれば論文の主張軸が成立する。

---

## 1. 実装で正しく入っている部分（変更不要）

参考までに、設計どおり動いている箇所:

- `prefetch_subbatch` / `memset_subbatch` / `evict_subbatch_to_host` ヘルパ
  （`brandes_gpu_opt.cu:676-728`）
- `oversubscribed = (dynamic_bytes > free_mem * 0.90)` 判定（行 778）
- `NS_eff = oversubscribed ? 1 : NS` で in-capacity 時は NS=2 維持（行 779）
- `per_batch_mem` への `+ sizeof(int)`（d_depth 補正、行 760）
- `hbm3_budget` から CSR + CB を控除（行 781-783）
- `SUB_BATCH = min(BATCH, hbm3_budget / per_source_bytes)` の動的計算（行 784-795）
- `pref_st` 専用ストリーム（行 858）
- シフトポインタによる kernel launch（`bufs[si].d_d + sub_off * n_nodes`、行 929-941, 958-971）
- 次サブバッチを並行 prefetch（行 947-952）
- evict（行 974-977）
- `d_overflow` チェックと exit（行 1005-1009）
- pure 版（`brandes_gpu_opt_pure.cu`、945 行）に変更なし — OK

---

## 2. 問題点と具体修正（優先度順）

### [P0-A] in-capacity 性能の劣化リスク: `reset_visited` 経路が呼ばれていない

#### 何が問題か

`reset_visited_batch_kernel` のカーネル関数自体は `brandes_gpu_opt.cu:377-398` に**残っている**が、
メインループから**呼ばれていない**。新版のメインループ（行 893-979）は毎サブバッチで
`memset_subbatch`（行 909）でフル memset するだけ。

旧コード（HBM3 streaming 化前）では、バッファ再利用 2 回目以降で次のロジックが走っていた:

```cpp
if (!buf_used[si]) {
    cudaMemsetAsync(...);  // 初回のみフル memset
} else {
    reset_visited_batch_kernel<<<bufs[si].prev_batch, tpb>>>(...);  // 2 回目以降は到達済みのみ
}
```

これは `[最適化6a]` として `optimization_proposal.md` §3.6(a) で導入された性能最適化。
**新コードでこの経路が消失しているため、in-capacity ケース（SUB_BATCH=BATCH）でも
旧版より遅くなる**。BATCH=512 で v1 は 13.44 GTEPS だったのが、v2 で 12.X GTEPS に
落ちると、論文の主張「UM streaming は in-capacity で従来パスと同等性能」が崩れる。

#### 具体修正（`brandes_gpu_opt.cu`）

(1) `DynBuf` に「直前のサブバッチ実 batch サイズ」を記録するフィールドを追加（行 666-674）:

```cpp
struct DynBuf {
    int    *d_d, *d_Q_curr, *d_Q_next, *d_S, *d_S_ends;
    double *d_sigma, *d_delta;
    int    *d_depth;
    cudaEvent_t ev_bfs_s, ev_bfs_e, ev_back_e;
    float bfs_ms, back_ms;
    int prev_batch;
+   int prev_sub_batch_n;   // 直前のサブバッチで処理した sub_n
+   int prev_sub_batch_off; // 直前のサブバッチの sub_off
};
```

初期化（行 842 付近）に追加:

```cpp
bufs[i].prev_batch = 0;
+ bufs[i].prev_sub_batch_n   = 0;
+ bufs[i].prev_sub_batch_off = 0;
```

(2) サブバッチループ内で memset を分岐（行 908-910 を差し替え）:

```cpp
nvtxRangePushA("Memset_or_reset_subbatch_opt");
- memset_subbatch(bufs[si], sub_off, sub_n, n_nodes, st);
+ if (bufs[si].prev_sub_batch_n > 0
+     && bufs[si].prev_sub_batch_off == sub_off
+     && bufs[si].prev_sub_batch_n   == sub_n) {
+     // 同じ範囲の再利用 → 到達済み頂点のみリセット
+     reset_visited_batch_kernel<<<sub_n, tpb, 0, st>>>(
+         bufs[si].d_d     + (size_t)sub_off * n_nodes,
+         bufs[si].d_sigma + (size_t)sub_off * n_nodes,
+         bufs[si].d_delta + (size_t)sub_off * n_nodes,
+         bufs[si].d_S     + (size_t)sub_off * n_nodes,
+         n_nodes,
+         bufs[si].d_depth  + sub_off,
+         bufs[si].d_S_ends + (size_t)sub_off * (max_depth_estimate + 1),
+         max_depth_estimate);
+ } else {
+     memset_subbatch(bufs[si].d_d     + (size_t)sub_off * n_nodes,
+                     bufs[si].d_sigma + (size_t)sub_off * n_nodes,
+                     bufs[si].d_delta + (size_t)sub_off * n_nodes,
+                     sub_n, n_nodes, st);
+ }
nvtxRangePop();
```

注: `memset_subbatch` のシグネチャもポインタ受取りに変更すると上の呼び出しが綺麗になる
（または現状のままで `bufs[si]` を渡しても OK、ただし内部で同じオフセット計算が必要）。

(3) サブバッチループ末尾で記録（行 977 直後、`cudaEventRecord(bufs[si].ev_back_e, st)` の前）:

```cpp
+ bufs[si].prev_sub_batch_n   = sub_n;
+ bufs[si].prev_sub_batch_off = sub_off;
CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_back_e, st));
```

#### 検証

BATCH=512（in-capacity）、5 試行で v1 結果（gpu_opt: 13.44 GTEPS, gpu_opt_pure: 13.47 GTEPS）と
**±0.5% 以内**で一致することを確認。一致しなければ reset_visited のロジックがまだ
壊れている。max BC = `39343001000.11` の完全一致は別途確認。

---

### [P0-B] prefetch overhead の定量計測がない

#### 何が問題か

`grep prefetch_ms` で 0 件。設計書 §3.7 で
`[GPU Phase] BFS wall=..., Backward wall=..., Prefetch wall=..., sub_batch=..., num_subs=...`
とログするよう提案したが、実装で抜けている。

これが致命的なのは、論文の核心的主張**「GH200 の C2C 帯域が高いから tiling overhead が小さい」**
を支える定量データが**取れない**こと。reviewer が「prefetch にどれだけ時間がかかっているの?
それは kernel と隠蔽できているの?」と聞いた瞬間に詰む。

#### 具体修正（`brandes_gpu_opt.cu`）

(1) `DynBuf` に prefetch event と累積 ms を追加（行 666-674）:

```cpp
struct DynBuf {
    ...
    cudaEvent_t ev_bfs_s, ev_bfs_e, ev_back_e;
+   cudaEvent_t ev_pref_s, ev_pref_e;
    float bfs_ms, back_ms;
+   float pref_ms;   // この buf のサブバッチ累積 prefetch 時間
    int prev_batch;
    ...
};
```

(2) 初期化（行 837-842 付近）:

```cpp
CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_back_e));
+ CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_pref_s));
+ CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_pref_e));
bufs[i].bfs_ms = 0.0f;
bufs[i].back_ms = 0.0f;
+ bufs[i].pref_ms = 0.0f;
```

(3) `prefetch_subbatch` の前後を event で挟む（行 896-906、初回 prefetch 部）:

```cpp
if (oversubscribed) {
    if (sub_off == 0) {
+       CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_pref_s, pref_st));
        prefetch_subbatch(bufs[si], sub_off, sub_n,
                          n_nodes, max_depth_estimate, gpu_id, pref_st);
+       CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_pref_e, pref_st));
+       float p_ms = 0.0f;
+       CUDA_ERR_CHK(cudaEventSynchronize(bufs[si].ev_pref_e));
+       CUDA_ERR_CHK(cudaEventElapsedTime(&p_ms, bufs[si].ev_pref_s, bufs[si].ev_pref_e));
+       bufs[si].pref_ms += p_ms;
    }
    cudaEvent_t pref_done;
    ...
}
```

同様に並行 prefetch 部（行 946-953）にも event を挟む。

**注**: 並行 prefetch のレイテンシは kernel と重なるので「実時間貢献は max(prefetch_ms, kernel_ms)」
になる。この測定で取れるのは「累積 prefetch 時間」（隠蔽前の生時間）であって、
それを kernel 時間と並べて表示することで隠蔽率を議論できる。

(4) ログ出力（行 1016-1018 を差し替え）:

```cpp
+ float wall_pref_ms = 0.0f, total_pref_ms = 0.0f;
+ for (int i = 0; i < NS_eff; i++) {
+     if (!buf_used[i]) continue;
+     wall_pref_ms  = max(wall_pref_ms, bufs[i].pref_ms);
+     total_pref_ms += bufs[i].pref_ms;
+ }

fprintf(stderr,
-   "  > [GPU Phase] BFS wall=%.4f s (cum=%.4f s), Backward wall=%.4f s (cum=%.4f s)\n",
-   wall_bfs_ms / 1000.0f, total_bfs_ms / 1000.0f,
-   wall_back_ms / 1000.0f, total_back_ms / 1000.0f);
+   "  > [GPU Phase] BFS wall=%.4f s (cum=%.4f s), Backward wall=%.4f s (cum=%.4f s), "
+   "Prefetch cum=%.4f s, SUB_BATCH=%d, num_subs=%d\n",
+   wall_bfs_ms / 1000.0f, total_bfs_ms / 1000.0f,
+   wall_back_ms / 1000.0f, total_back_ms / 1000.0f,
+   total_pref_ms / 1000.0f, SUB_BATCH, num_subs);
```

#### 検証

BATCH=8192（oversubscribed）で `Prefetch cum=` が非ゼロかつ kernel 時間に対して
ある比率（10–30% 程度）で出力されること。BATCH=512（in-capacity）では 0 になること。

---

### [P0-C] 手動 chunking pure 版が存在しない

#### 何が問題か

現状の比較構図は次の 2 種類:

1. **Proposal-Gen** (`brandes_gpu_opt_pure.cu`, `cudaMalloc` 一発で BATCH 分確保)
2. **Proposal-GH200** (`brandes_gpu_opt.cu`, UM + サブバッチ prefetch)

reviewer が必ず聞くのは:

> **「pure 版を `cudaMalloc(SUB_BATCH × N)` だけ確保して、
>  outer loop で手動 chunking すれば同じことできるんじゃないの?
>  cudaMemcpy を毎チャンクで発行するだけ。
>  それと比べて UM の prefetch は速いの? 同等なの?」**

これに答える 3 番目の比較対象がないと、**UM 経由の貢献を分離できない**。
論文は通らない。

#### 具体修正（新規ファイル `brandes_gpu_opt_pure_chunked.cu`）

(1) `brandes_gpu_opt_pure.cu` を複製して `brandes_gpu_opt_pure_chunked.cu` を作る:

```bash
cp lab/mylab/research/brandes_gpu_opt_pure.cu \
   lab/mylab/research/brandes_gpu_opt_pure_chunked.cu
```

(2) シンボル名を全部 `_pure_chunked` 接尾辞に置換:

```bash
sed -i 's/brandes_gpu_opt_pure/brandes_gpu_opt_pure_chunked/g' \
       lab/mylab/research/brandes_gpu_opt_pure_chunked.cu
sed -i 's/_pure(/_pure_chunked(/g' \
       lab/mylab/research/brandes_gpu_opt_pure_chunked.cu
```

(3) メインループ部分を以下のように改造（疑似コード、新版の §3.5 と対称構造）:

```cpp
// (a) per_source_bytes と SUB_BATCH を計算 (gpu_opt.cu §4.2 と同じロジック)
size_t hbm3_budget = (size_t)(free_mem * 0.80) - topology_bytes - cb_bytes;
int SUB_BATCH = min(BATCH_PER_STREAM, (int)(hbm3_budget / ((size_t)NS * per_batch_mem)));

// (b) cudaMalloc は SUB_BATCH 分だけ
for (int i = 0; i < NS; i++) {
    cudaMalloc(&bufs[i].d_d, (size_t)SUB_BATCH * n_nodes * sizeof(int));
    // ... 他 7 種も同様
}

// (c) 入力ホストバッファ (pinned) を別途 BATCH 分用意
double *h_d_buf, *h_sigma_buf, ...;
cudaMallocHost(&h_d_buf, (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int));
// ... (実際は memset 値だけなのでホストバッファ不要かもしれない、初期化方法による)

// (d) メインループ: BATCH を SUB_BATCH 単位で chunking
for (int s_start = ...; ...) {
    int curr_batch = min(BATCH_PER_STREAM, n_nodes - s_start);
    for (int sub_off = 0; sub_off < curr_batch; sub_off += SUB_BATCH) {
        int sub_n = min(SUB_BATCH, curr_batch - sub_off);

        // (d1) 必要なら HtoD で chunk を転送 (pure 版は cudaMemset で初期化するので転送不要)
        cudaMemsetAsync(bufs[si].d_d, 0xFF, (size_t)sub_n * n_nodes * sizeof(int), st);
        cudaMemsetAsync(bufs[si].d_sigma, 0, (size_t)sub_n * n_nodes * sizeof(double), st);
        cudaMemsetAsync(bufs[si].d_delta, 0, (size_t)sub_n * n_nodes * sizeof(double), st);

        // (d2) kernel launch (元から SUB_BATCH 分しかバッファないので shifted pointer 不要)
        brandes_bfs_kernel_opt<<<sub_n, tpb, 0, st>>>(
            R, C, n_nodes, max_depth_estimate, d_overflow,
            bufs[si].d_d, bufs[si].d_sigma, ...,
            s_start + sub_off);  // s_start のシフトのみ
        brandes_back_kernel_opt<<<sub_n, tpb, 0, st>>>(...);

        // (d3) もし結果を chunk 単位でホストに回収する必要があれば cudaMemcpyAsync
        //      BC は GPU 側に蓄積するので不要 (CB のみ最後に DtoH)
    }
}
```

**ポイント**: pure 版は元々ホスト→デバイス転送を `cudaMemcpy(R/C, H2D)` の 1 回しか
していない（CSR 用）。動的バッファは GPU 上で `cudaMemset` で初期化するだけ。
だから手動 chunking 版で増えるオーバーヘッドは「`cudaMalloc` 後のバッファを毎 sub-batch
で使い回し」だけで、追加の cudaMemcpy は発生しない。**実質的にはバッファサイズを縮めて
ループ回数を増やすだけ**。

これで gpu_opt（UM streaming）との純粋比較ができる:

- 同じ SUB_BATCH × num_subs で動かす
- 違いは「UM の prefetch overhead」 vs 「pure 版は何もしない（バッファが既に HBM3 にある）」
- 期待される結果: pure_chunked が gpu_opt と**同等またはわずかに速い**。
  もし大差つくなら UM の overhead が想像より大きい証拠。

(4) `scripts/run_benchmark_*.sh` に `Proposal-Gen-Chunked` ラベルで pure_chunked を追加。
`brandes_runner` の dispatcher にも該当の case を追加。

#### 検証

3 者比較表が完成し、in-capacity と over-capacity 両方で測定が取れる。

---

### [P1-A] サブバッチ最初の prefetch が逐次（オーバーラップしない）

#### 何が問題か

行 897-906 で `if (sub_off == 0)` の prefetch を発行 → `cudaStreamWaitEvent` で kernel 側を待たせる。
これは毎 outer batch の先頭で kernel が prefetch 完了まで stall することを意味する。
num_subs=2 のケースでは「最初の sub-batch の prefetch」が overhead の半分を占める。

#### 具体修正

`s_start` ループの 1 つ前のバッチの最後で、次バッチの先頭サブバッチを pre-prefetch する。
ループの構造を「次バッチのキューイング」に変える必要があり、やや侵襲的。

簡易版（実装 1 時間以内）:

```cpp
// メインループの直前に「次に使うバッファの sub_off=0 を pre-prefetch」しておく
if (oversubscribed && n_nodes > 0) {
    int initial_curr_batch = min(BATCH_PER_STREAM, n_nodes);
    int initial_sub_n = min(SUB_BATCH, initial_curr_batch);
    prefetch_subbatch(bufs[0], 0, initial_sub_n, n_nodes, max_depth_estimate, gpu_id, pref_st);
}
```

これで最初のバッチ最初のサブバッチの prefetch だけ stall が消える。
2 番目のバッチ以降の sub_off=0 stall は残るが、複雑な実装を避けつつ実質的に 1/N 改善。

完全版（実装 4-8 時間）: 「`bufs[1-si]` の次バッチを `bufs[si]` の最終サブバッチ並行で prefetch」。
今回はスコープ外だが TODO として書き残す。

---

### [P1-B] `cudaEventCreate(&pref_done)` の毎回 create/destroy

#### 何が問題か

行 901-905 で毎サブバッチごとに `pref_done` event を create / destroy。
synth_326K で BATCH=8192 → 約 40 outer batches × 2 sub = 80 回。
1 回 ~10 μs の event create コスト × 80 = ~1 ms。微小だが冗長。

#### 具体修正

`pref_done` event を impl 関数の冒頭で 1 個作って再利用。

行 859 直後に追加:

```cpp
cudaEvent_t pref_st = 0;
cudaEvent_t pref_done = 0;
if (oversubscribed) {
    CUDA_ERR_CHK(cudaStreamCreate(&pref_st));
+   CUDA_ERR_CHK(cudaEventCreate(&pref_done));
}
```

メインループ内（行 901-905）を:

```cpp
- cudaEvent_t pref_done;
- CUDA_ERR_CHK(cudaEventCreate(&pref_done));
  CUDA_ERR_CHK(cudaEventRecord(pref_done, pref_st));
  CUDA_ERR_CHK(cudaStreamWaitEvent(st, pref_done, 0));
- CUDA_ERR_CHK(cudaEventDestroy(pref_done));
```

最後（行 938 周辺、stream destroy の隣）に:

```cpp
if (oversubscribed) {
    CUDA_ERR_CHK(cudaStreamDestroy(pref_st));
+   CUDA_ERR_CHK(cudaEventDestroy(pref_done));
}
```

---

### [P1-C] 最後のサブバッチでも evict が走る

#### 何が問題か

行 974-977 の `evict_subbatch_to_host` に gating がない。最後のサブバッチで evict すると
「次の outer batch で別バッファに移る」または「全体終了」のどちらかなので、
evict された内容を**読み返さない**。LPDDR5X への書き戻しコストが無駄。

#### 具体修正

行 974 を:

```cpp
- if (oversubscribed) {
+ // 次のサブバッチで HBM3 を空ける必要がある場合のみ evict
+ // 最後のサブバッチ (sub_off + SUB_BATCH >= curr_batch) では evict 不要
+ if (oversubscribed && (sub_off + SUB_BATCH < curr_batch)) {
    evict_subbatch_to_host(bufs[si], sub_off, sub_n,
                           n_nodes, max_depth_estimate, st);
}
```

#### 検証

`Backward wall` 時間が数 % 短縮される（特に num_subs が小さいケース）。

---

### [P1-D] `nvtxRangePush` の不整合（軽微）

#### 何が問題か

確認したところ、行 956 (`nvtxRangePushA("Backward_kernel_opt")`) と行 972 (`nvtxRangePop`) は
`if/else` の外側で対称に括っているので**実は問題なし**。修正不要。
（私の前回の評価で誤認していた箇所。本問題はキャンセル。）

---

### [P2-A] reset_visited 経路の入力検証が緩い

#### 何が問題か

[P0-A] で復活させた reset_visited 分岐では、`prev_sub_batch_n == sub_n` が必要。
oversubscribed で SUB_BATCH 単位で動いているとき、「同じ buf を再利用するときに、
かつ前回と同じ sub_off / sub_n」のときだけ reset_visited が使える。
それ以外（curr_batch が SUB_BATCH の倍数でない最終サブバッチなど）は
フル memset にフォールバック。実装上は問題ないが、テストケース不足だと
バグが残る可能性。

#### 具体修正

[P0-A] の差分で `prev_sub_batch_off == sub_off && prev_sub_batch_n == sub_n` の条件で
gating している。これで安全。`bufs[si].prev_sub_batch_n = 0` で初期化されているので、
1 回目は必ずフル memset 経路に入る。

#### 検証

correctness.md の max BC 値が in-capacity / over-capacity 両方で
`39343001000.11` と完全一致することを確認。

---

### [P2-B] `BATCH_PER_STREAM = max(1, min(..., 512))` の上限ハードコード

#### 何が問題か

行 765 で auto-calc 値が `min(..., 512)` でクランプされているが、override がそれを上書きする。
ロジックとして「auto 計算で 1024 が出ても 512 にクランプし、override がさらに上書き」となり、
in-capacity でも 512 を超えない。論文の Future Work で「Hopper の 132 SM を活かすには
BATCH を上げたい」と書く場合、この cap は外す価値あり。

#### 具体修正

```cpp
- BATCH_PER_STREAM = max(1, min(BATCH_PER_STREAM, 512));
+ BATCH_PER_STREAM = max(1, BATCH_PER_STREAM);  // auto-calc 値をそのまま採用
```

ただし変更すると in-capacity の挙動が変わるので、論文の Sec 4 の数値が変わる。
**v2 ベンチを取る前にこれを変えるか/変えないか決める**。今回は触らない方が安全。

---

## 3. 研究としての成立性: 何があれば論文が通るか

### 3.1 現状の主張軸（pure vs UM の 2 種比較）

| | 評価 |
|---|---|
| in-capacity で UM=pure ±0.2% | ◯ 既に v1 データで裏付け済み |
| over-capacity で pure OOM、UM 完走 | ◯ v2 で再確認 |
| GH200 アーキテクチャの活用主張 | △ NVLink-C2C 帯域を直接示すデータがない |
| UM 固有の貢献の分離 | ✗ pure_chunked がないため示せない |
| アルゴリズム的 novelty | △ tiling 自体は既存技法 |

### 3.2 [P0-C] 追加後の主張軸（3 種比較）

| | 評価 |
|---|---|
| 同上 in-capacity, over-capacity | ◯ |
| pure_chunked vs UM streaming の比較 | ◯ UM の overhead を定量化可能 |
| 「UM の便利さ vs 性能 trade-off」議論 | ◯ プログラミングコスト軸で UM 優位 |
| reviewer の "manual chunking" 質問への回答 | ◯ |

### 3.3 さらに novelty を上げるなら（必須ではない）

- **bandwidth_benchmark.cu との連携**: C2C 実効帯域 (700-800 GB/s) を測定し、
  prefetch 時間 (`Prefetch cum=...`) との一致を理論モデルで示す。
  Sec 3 か Sec 4.6 に "C2C-Bandwidth-Bound Prefetch Model" 節を追加。
- **より大きいグラフでの自然な oversubscribe**: V=5M クラスのグラフで BATCH=512 でも
  oversubscribe するケースを 1 つ用意（`tools/gen_graph.py` で生成）。
  「人為的 BATCH 強制ではなく、実応用で起きる」根拠になる。

---

## 4. 実装タスクの順序（受け取った AI 向け）

### Phase A (1–2 時間): in-capacity 性能の保全

1. [P0-A] reset_visited 復活 — `brandes_gpu_opt.cu` の DynBuf 拡張 + メインループ分岐追加。
2. ビルド + BATCH=512 で 5 試行。v1 結果（13.44 GTEPS）と ±0.5% で一致確認。
3. max BC = `39343001000.11` 完全一致確認。

**Pass 条件:** in-capacity 性能が v1 と同等に戻る。

### Phase B (1 時間): 計測強化

4. [P0-B] prefetch_ms 計測 — DynBuf に event 追加 + ログ出力修正。
5. ビルド + BATCH=8192 で 5 試行。`Prefetch cum=...` がログに出ることを確認。
6. [P1-B] pref_done event 再利用、[P1-C] 最終サブバッチ evict gating も同時に当てる。

**Pass 条件:** ログに `Prefetch cum=X.XXXX s, SUB_BATCH=Y, num_subs=Z` が出る。

### Phase C (3–4 時間): 手動 chunking pure 版

7. [P0-C] `brandes_gpu_opt_pure_chunked.cu` 新規作成。
8. `scripts/run_benchmark_*.sh` と `brandes_runner` ディスパッチに追加。
9. ビルド + BATCH=512, 4096, 8192, 16384 で各 5 試行。

**Pass 条件:** 3 者比較表が `oversubscribe_results_v2.tsv` に出力される。

### Phase D (実験 + 論文反映): v2 ベンチ取得

10. `result_um_oversubscribe_v2/` に全 BATCH × 3 実装 × 5 試行を取る。
11. `summarize_oversubscribe.py` を実装し、median ± std + status マトリクス + prefetch% 列を生成。
12. `draft/Chapter/Evaluation.tex` Sec 4.4 を 3 者比較に書き直し、
    Sec 4.10.3 の future work を本文に格上げ。

### Phase E (任意、論文を強くする):

13. C2C 帯域モデルを Sec 4.6 に追加。
14. V=5M クラスの自然 oversubscribe グラフを 1 つ追加。

---

## 5. 触ってはいけないファイル

- `brandes_gpu_opt_pure.cu`: OOM が論文主張の根拠なので変更禁止。
- `brandes_sequential.cpp`, `brandes_pathmerge.cu`, `brandes_cugraph.cu`, OpenMP 系:
  ベースラインなので変更禁止、再ベンチも不要。
- `tools/bandwidth_benchmark.cu`: ハード測定なので 1 回取れば使い回し。

---

## 6. 期待される v2 結果マトリクス

`synth_326K` (V=325K, E=3.2M):

| BATCH | dynamic(UM/Malloc) | pure (cudaMalloc) | pure_chunked (P0-C) | gpu_opt (UM streaming) | Prefetch cum / Total |
|------:|---:|:---|:---|:---|:---|
| 512   | 10.7 GB  | ✓ ~13.5 GTEPS | ✓ ~13.5 GTEPS | ✓ ~13.4 GTEPS（[P0-A] 修正後） | 0% |
| 1024  | 21.3 GB  | ✓ ~15.6       | ✓ ~15.6       | ✓ ~15.5 | 0% |
| 2048  | 42.7 GB  | ✓ ~15.5       | ✓ ~15.5       | ✓ ~15.4 | 0% |
| 4096  | 85.4 GB  | ✓ ~15.3       | ✓ ~15.3       | ✓ ~15.2 | 0% |
| 8192  | 170.7 GB | ✗ OOM          | ✓ ~14–15      | ✓ ~13–14 | 5–10% |
| 10240 | 213 GB   | ✗ OOM          | ✓ ~13–14      | ✓ ~13–14 | 8–12% |
| 12288 | 256 GB   | ✗ OOM          | ✓ ~13–14      | ✓ ~12–13 | 10–15% |
| 16384 | 341 GB   | ✗ OOM          | ✓ ~12–13      | ✓ ~12–13 | 15–20% |

**この表が取れれば論文の Sec 4.4 が完成**。論調は:

> "Within HBM3 capacity, the three implementations achieve essentially identical throughput
> (within ±0.5%), confirming that UM's read-mostly advisories impose no penalty.
> Beyond HBM3 capacity, both `Proposal-Gen-Chunked` and `Proposal-GH200` continue
> to operate by streaming sub-batches through HBM3, while `Proposal-Gen` (single-allocation)
> fails with OOM. `Proposal-GH200` matches `Proposal-Gen-Chunked` within X% in throughput
> (prefetch overhead $\le$ Y%), demonstrating that GH200's NVLink-C2C bandwidth is
> sufficient to make UM-managed software caching competitive with manual explicit chunking,
> while removing the implementation burden of the latter."

---

## 7. 受け取った AI への注意

- **私の指示の行番号を信頼せず**、必ず実コードを読んで最新の行番号と照合してから修正してください
  （前回の修正で行がずれている可能性があります）。
- すべてのカーネル変更は行わない（シフトポインタで動くため）。Host コードのみ変更。
- 修正後は **必ず BATCH=512 で in-capacity 同等性能を確認**してから BATCH=8192+ に進む。
- `correctness.md` の max BC 完全一致は**毎ビルドで確認**。
- `brandes_gpu_opt_pure.cu` は**絶対に**触らない。手動 chunking 版は別ファイル。

---

## 8. 参照ファイル

- `lab/mylab/research/brandes_gpu_opt.cu` (主修正対象、行 666-1131)
- `lab/mylab/research/brandes_gpu_opt_pure.cu` (変更禁止)
- `lab/mylab/research/brandes_gpu_opt_pure_chunked.cu` (新規作成、Phase C)
- `lab/mylab/research/scripts/run_benchmark_*.sh` (Phase C で追加)
- `lab/mylab/research/scripts/summarize_oversubscribe.py` (新規作成、Phase D)
- `lab/mylab/research/build_miyabi/result_um_oversubscribe/oversubscribe_results.tsv` (v1 結果、参考)
- `lab/mylab/research/build_miyabi/result_um_oversubscribe_v2/` (v2 出力先、新規)
- `draft/Chapter/Evaluation.tex` (Phase D で書き直し)
- `draft/Chapter/Methodology.tex` (HBM3 streaming 節を追加)
- `draft/scripts/generate_figures_tables.py` (新 figure)
