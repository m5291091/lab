# 実験フェーズ前 最終チェックレポート

> **目的**: lab v2 (HBM3 ストリーミング適用版) を実機ベンチマークに投入する前に、
> ブロッカーとなる箇所を全て洗い出し、修正手順を 1 ファイルで提示する。
> 別 AI が機械的に修正タスクを実行できる粒度で書く。

---

## 0. 一目でわかる結論

| 区分 | 件数 | 対応 |
|---|---|---|
| **ブロッカー (実験前に必須修正)** | **2 件** | [BUG-1], [BUG-2] |
| 推奨修正 (実験は流せるが集計に影響) | 3 件 | [MINOR-1], [MINOR-2], [MINOR-3] |
| 反映済み・問題なし | 11 件 | §2 のチェックリスト |

ブロッカー 2 件 + リビルドを片付ければ実験 GO。所要時間目安: **30 分以内**。

論文の主張軸:
> 「UM により HBM3 を超える working set でも Exact BC が実行可能。
>  pure は OOM、UM streaming は完走（manual chunking と同等性能までは出ない）。
>  GH200 の UM + NVLink-C2C により oversubscription 自体が成立することを実証する。」

この主張は**現状の実装 + ブロッカー修正で実証可能**。reviewer の典型的な質問
「manual chunking との比較は?」にも 3 者比較で答えられる構成になっている。

---

## 1. ブロッカー (必ず修正してから実験を流す)

### [BUG-1] `summarize_oversubscribe.py` の prefetch regex が実ログと一致しない

#### 問題

`scripts/summarize_oversubscribe.py:72` の正規表現:

```python
r"(?:.*?Prefetch wall=([0-9.]+)\s*s)?"
```

しかし実コードのログ出力 (`brandes_gpu_opt.cu:1066`):

```c
"Prefetch cum=%.4f s, SUB_BATCH=%d, num_subs=%d\n"
```

**`Prefetch wall=` を探しているが実際は `Prefetch cum=`**。
このため P0-B で計測した prefetch 時間が**全試行で `None` として捨てられる**。
集計表の `prefetch_ratio` 列が空になり、論文の核心データが取れない。

#### 影響

- 実験は走るが、**`prefetch_ms` / `prefetch_ratio` が全て None**。
- "GH200 の C2C 帯域で tiling overhead がどこまで隠せるか" の議論が定量化できない。
- 論文 Sec 4.4 の「UM streaming のオーバーヘッドは X%」が書けない。

#### 修正

ファイル: `lab/mylab/research/scripts/summarize_oversubscribe.py:72`

```diff
 GPU_PHASE_RE = re.compile(
     r"\[GPU Phase\]\s*"
     r"BFS wall=([0-9.]+)\s*s.*?"
     r"Backward wall=([0-9.]+)\s*s"
-    r"(?:.*?Prefetch wall=([0-9.]+)\s*s)?"
+    r"(?:.*?Prefetch cum=([0-9.]+)\s*s)?"
 )
```

#### 検証

修正後、サンプルログ 1 行 (`Prefetch cum=2.3456 s`) を `re.search` で検証:

```python
import re
RE = re.compile(r"\[GPU Phase\].*?BFS wall=([0-9.]+).*?Backward wall=([0-9.]+).*?(?:Prefetch cum=([0-9.]+))?")
m = RE.search("  > [GPU Phase] BFS wall=12.3 s (cum=24.5 s), Backward wall=10.1 s (cum=20.0 s), Prefetch cum=1.2 s, SUB_BATCH=4096, num_subs=2")
assert m and m.group(3) == "1.2", "regex broken"
```

---

### [BUG-2] `summarize_oversubscribe.py` が `gpu_opt_pure_chunked` を認識しない

#### 問題

`scripts/summarize_oversubscribe.py:57-60`:

```python
IMPL_DISPLAY = {
    "gpu_opt": "gpu_opt (UM)",
    "gpu_opt_pure": "gpu_opt_pure",
}
IMPL_ORDER = ["gpu_opt_pure", "gpu_opt"]
```

`pure_chunked` のエントリが**ない**。`normalize_impl_name` の mapping にも該当キーがない。
3 種類目の実装 (P0-C で追加した手動 chunking 版) が**集計から完全に抜け落ちる**。

#### 影響

- ベンチで pure_chunked のデータは TSV に書き込まれるが、
  集計スクリプトが**スキップする**。
- 3 者比較表 (pure vs pure_chunked vs UM streaming) が**作れない**。
- 論文の reviewer 必須質問「手動 chunking との比較は?」に**答えられない**。
- せっかく作った `brandes_gpu_opt_pure_chunked.cu` の意味が消える。

#### 修正

ファイル: `lab/mylab/research/scripts/summarize_oversubscribe.py`

(1) `IMPL_DISPLAY` と `IMPL_ORDER` の更新 (行 57-60):

```diff
 IMPL_DISPLAY = {
-    "gpu_opt": "gpu_opt (UM)",
-    "gpu_opt_pure": "gpu_opt_pure",
+    "gpu_opt": "gpu_opt (UM streaming)",
+    "gpu_opt_pure": "gpu_opt_pure (single-alloc)",
+    "gpu_opt_pure_chunked": "gpu_opt_pure_chunked (manual chunking)",
 }
-IMPL_ORDER = ["gpu_opt_pure", "gpu_opt"]
+IMPL_ORDER = ["gpu_opt_pure", "gpu_opt_pure_chunked", "gpu_opt"]
```

(2) `normalize_impl_name` の mapping 拡張 (行 80 付近):

```diff
 def normalize_impl_name(name: str) -> str:
     lowered = name.strip().lower()
     mapping = {
         "gpu_opt": "gpu_opt",
         "gpu_opt_pure": "gpu_opt_pure",
+        "gpu_opt_pure_chunked": "gpu_opt_pure_chunked",
+        "gpuoptpurechunked": "gpu_opt_pure_chunked",
+        "gpu_opt_pure_chunked_(manual)": "gpu_opt_pure_chunked",
+        "gpu_opt_pure_chunked...": "gpu_opt_pure_chunked",
+        "gpu_opt_pure_chunked:": "gpu_opt_pure_chunked",
         ...
     }
```

注: ベンチスクリプトと `main.cpp:176` での出力名は `"GPU_Opt_Pure_Chunked"`。
`lowered = name.strip().lower()` で `"gpu_opt_pure_chunked"` に正規化されるので、
mapping にこのキーがあれば通る。

#### 検証

修正後、テストデータで集計:

```bash
cd lab/mylab/research
python scripts/summarize_oversubscribe.py \
    build_miyabi/result_um_oversubscribe/oversubscribe_results.tsv
```

出力に **3 行**（pure / pure_chunked / gpu_opt）が出ること。
v1 データ (pure と gpu_opt のみ) でも 2 行は出るはず。

---

## 2. 反映済みで問題なし (確認のみ、変更不要)

実コードを精査して確認した、**設計どおり入っている部分**。

| 項目 | 確認場所 (file:line) |
|---|---|
| reset_visited 復活 (P0-A) | `brandes_gpu_opt.cu:927-940` |
| `prev_sub_batch_n/off` フィールド追加 | DynBuf `:676-677`, init `:850-851`, 記録 `:1022-1023` |
| `ev_pref_s/e` + `pref_ms` (P0-B) | DynBuf `:672-674`, init `:844-848` |
| prefetch 時間記録 (初回 / 並行) | `:911-919` および `:983-991` |
| ログ出力 `Prefetch cum=...` | `:1064-1069` |
| `brandes_gpu_opt_pure_chunked.cu` 新規作成 (P0-C) | 993 行、SUB_BATCH×N で `cudaMalloc` |
| dispatcher 登録 | `main.cpp:175-176`, `brandes.h:21`, `CMakeLists.txt:141` 周辺 |
| ベンチスクリプト統合 | `scripts/run_um_oversubscribe_experiment.sh:38` で 3 実装 × 8 BATCH × 5 試行 |
| `pref_done` event 再利用 (P1-B) | 行 `:867-870` で 1 回 create、ループ内で再利用 |
| 最終サブバッチの evict gating (P1-C) | 行 `:1015` `(sub_off + SUB_BATCH < curr_batch)` |
| pure 版に変更が入っていない | `brandes_gpu_opt_pure.cu` は 945 行のまま、シグネチャ未変更 |

これらは触らない。

---

## 3. 推奨修正 (実験は流せるが集計の質が落ちる)

### [MINOR-1] スクリプトのログに `RUN_HEADER` 行が出力されていない

#### 問題

`summarize_oversubscribe.py:65` は次の形式の行を期待:

```
=== gpu_opt batch=8192 trial=1 rc=0 ===
```

しかし `run_um_oversubscribe_experiment.sh:43` は:

```bash
echo "[RUN] Method: ${method}, BC_BATCH_OVERRIDE: ${batch}, Trial: ${trial}"
```

**`=== ... batch=... trial=... rc=... ===` フォーマットを 1 度も出力していない**。
このため stderr ログ (`um_experiment.log`) と TSV のマッチングが取れず、
prefetch 時間が試行単位で結びつかない可能性がある。

#### 影響

`summarize_oversubscribe.py` が `[GPU Phase]` 行から prefetch を取れても、
それがどの試行のものか identify できない。集計が壊れる。

#### 修正

ファイル: `lab/mylab/research/scripts/run_um_oversubscribe_experiment.sh`

行 50 (rc 取得直後) に追加:

```diff
             rc=0
             "${RUNNER}" "${method}" "${GRAPH}" > "${tmp_stdout}" 2> "${tmp_stderr}" || rc=$?
+
+            # summarize_oversubscribe.py が parse できる形式でヘッダ行をログに追加
+            echo "=== ${method} batch=${batch} trial=${trial} rc=${rc} ===" \
+                >> "${RESULT_DIR}/um_experiment.log"
             
             if [ ${rc} -ne 0 ]; then
                 ...
```

#### 検証

実験 1 試行だけ手動で走らせ、`um_experiment.log` に `=== gpu_opt batch=512 trial=1 rc=0 ===` 行が
出力されることを確認。

---

### [MINOR-2] `pure_chunked` のログフォーマットが `gpu_opt` と一致するか確認

#### 問題

`summarize_oversubscribe.py` の `GPU_PHASE_RE` は `Prefetch cum=...` を optional matching
(`(?:...)?`) しているので、`pure_chunked` が `Prefetch cum=` を出力しなくても
parse は通る。**ただし出力形式が `gpu_opt` と微妙に違うと regex がマッチせず、
BFS/Backward 時間まで取れなくなる**リスクがある。

#### 修正

ファイル: `lab/mylab/research/brandes_gpu_opt_pure_chunked.cu`

ログ出力部 (`gpu_opt_pure_chunked_impl` 内の `[GPU Phase]` 出力) を確認し、
最低でも次のフォーマットに揃える:

```c
fprintf(stderr,
    "  > [GPU Phase] BFS wall=%.4f s (cum=%.4f s), Backward wall=%.4f s (cum=%.4f s), "
    "SUB_BATCH=%d, num_subs=%d\n",
    wall_bfs_ms / 1000.0f, total_bfs_ms / 1000.0f,
    wall_back_ms / 1000.0f, total_back_ms / 1000.0f,
    SUB_BATCH, num_subs);
```

`Prefetch cum=` は出力しなくて良い (UM 不使用のため意味がない)。
ただし regex の optional matching に依存するなら、整合性のために
`Prefetch cum=0.0000 s` をダミー出力する手もある。**推奨**: 出さずに省略。

#### 検証

`grep "\[GPU Phase\]" um_experiment.log` で全 3 実装が同じ形式 (BFS wall + Backward wall) で
出ていること、summarize がそれぞれを parse できることを確認。

---

### [MINOR-3] `brandes_runner` バイナリのリビルドが必要

#### 問題

```
build_miyabi/brandes_runner: May 10 10:22 (古い)
brandes_gpu_opt_pure_chunked.cu: May 11 01:42 (新しい)
```

**現在のバイナリには `pure_chunked` がリンクされていない**。
このまま実験を走らせると `./brandes_runner gpu_opt_pure_chunked ...` が
"Unknown implementation" で落ちる。

#### 修正

実機 (Miyabi) で:

```bash
cd lab/mylab/research/build_miyabi
cmake --build . --target brandes_runner -j
```

または対話ビルドスクリプト:

```bash
bash lab/mylab/research/scripts/build_miyabi_interactive.sh
```

#### 検証

リビルド後:

```bash
./build_miyabi/brandes_runner gpu_opt_pure_chunked lab/data/325557_3216152
```

が "Unknown implementation" を出さず、最後まで走ること。

リビルド前の `brandes_runner` のサイズと比較し、若干増えていることを確認:

```bash
ls -la build_miyabi/brandes_runner
# 旧: ~14 MB
# 新: ~14.5–15 MB (pure_chunked カーネル分の増加)
```

---

## 4. 実行手順 (実験フェーズへ進むまで)

### Phase 0: ブロッカー修正 (15 分)

```bash
cd lab/mylab/research

# [BUG-1] regex 修正
sed -i 's/Prefetch wall=/Prefetch cum=/' scripts/summarize_oversubscribe.py

# [BUG-2] 手動編集 (sed では難しいのでエディタで開く)
$EDITOR scripts/summarize_oversubscribe.py
# 行 57-60 と normalize_impl_name に pure_chunked を追加 (本ドキュメント §1 [BUG-2] 参照)
```

### Phase 1: 推奨修正 (10 分)

```bash
# [MINOR-1] RUN_HEADER 行追加
$EDITOR scripts/run_um_oversubscribe_experiment.sh
# 行 50 周辺に echo "=== ${method} batch=${batch} trial=${trial} rc=${rc} ==="

# [MINOR-2] pure_chunked のログ確認
grep -n "GPU Phase" brandes_gpu_opt_pure_chunked.cu
# gpu_opt と同じフォーマットになっているか確認、必要なら修正
```

### Phase 2: リビルド (5 分)

```bash
cd build_miyabi
cmake --build . --target brandes_runner -j

# 動作確認
./brandes_runner gpu_opt_pure_chunked ../../data/325557_3216152
# → "Unknown implementation" が出ず最後まで走ること
```

### Phase 3: 集計スクリプトの動作確認 (5 分)

```bash
# v1 データで動かして 3 実装ぶんの行が出るか確認
# (v1 には pure_chunked がないので 2 実装しか出ないはず)
python scripts/summarize_oversubscribe.py \
    build_miyabi/result_um_oversubscribe/oversubscribe_results.tsv
```

期待: pure と gpu_opt の 2 行 (v1 データに pure_chunked はない)。
prefetch 列は v1 では出力されていないので空のまま。

### Phase 4: 実験投入

```bash
# Miyabi で qsub
qsub scripts/run_um_oversubscribe_experiment.sh
```

ジョブ完了後:

```bash
python scripts/summarize_oversubscribe.py \
    build_miyabi/result_um_oversubscribe/oversubscribe_results.tsv \
    > build_miyabi/result_um_oversubscribe/summary_v2.md
```

期待出力: 3 実装 × 8 BATCH の median ± std + status マトリクス + prefetch_ratio 列。

---

## 5. 期待される結果マトリクス (実験 GO 後の照合用)

`synth_326K` (V=325K, E=3.2M):

| BATCH | dynamic | pure | pure_chunked | gpu_opt (UM) | Prefetch ratio (UM) |
|------:|---:|:---|:---|:---|:---|
| 512   | 10.7 GB  | ✓ ~13.5 GTEPS | ✓ ~13.5 GTEPS | ✓ ~13.4 GTEPS | 0% |
| 1024  | 21.3 GB  | ✓ ~15.6       | ✓ ~15.6       | ✓ ~15.5       | 0% |
| 2048  | 42.7 GB  | ✓ ~15.5       | ✓ ~15.5       | ✓ ~15.4       | 0% |
| 4096  | 85.4 GB  | ✓ ~15.3       | ✓ ~15.3       | ✓ ~15.2       | 0% |
| 8192  | 170.7 GB | ✗ OOM         | ✓ ~14–15      | ✓ ~12–14      | 5–15% |
| 10240 | 213 GB   | ✗ OOM         | ✓ ~13–14      | ✓ ~12–14      | 8–18% |
| 12288 | 256 GB   | ✗ OOM         | ✓ ~13–14      | ✓ ~11–13      | 10–20% |
| 16384 | 341 GB   | ✗ OOM         | ✓ ~12–13      | ✓ ~10–13      | 15–25% |

論文 Sec 4.4 の論調案:

> "Beyond HBM3 capacity, `Proposal-Gen` (single-allocation) fails with OOM
> at BATCH ≥ 8192. Both `Proposal-Gen-Chunked` (manual chunking) and
> `Proposal-GH200` (UM streaming) continue to operate by streaming sub-batches
> through HBM3. While manual chunking achieves slightly higher throughput
> (X% advantage) due to absence of UM overhead, GH200's NVLink-C2C bandwidth
> keeps the prefetch overhead of UM streaming within Y% of the kernel time,
> demonstrating that UM-based oversubscription is feasible on GH200 at the
> cost of a modest performance penalty in exchange for programming-model
> simplicity (no manual chunking required)."

---

## 6. GO / NO-GO チェックリスト

実験投入前に**すべて ✓ になっていること**を確認。

- [ ] [BUG-1] `summarize_oversubscribe.py:72` の `Prefetch wall` → `Prefetch cum` 修正
- [ ] [BUG-2] `summarize_oversubscribe.py` の `IMPL_DISPLAY` / `IMPL_ORDER` / `normalize_impl_name` に `pure_chunked` 追加
- [ ] [MINOR-1] `run_um_oversubscribe_experiment.sh` に `RUN_HEADER` 行追加
- [ ] [MINOR-2] `brandes_gpu_opt_pure_chunked.cu` の `[GPU Phase]` 出力フォーマット確認
- [ ] [MINOR-3] `brandes_runner` をリビルドし、`pure_chunked` で動作確認
- [ ] Phase 3 で v1 データを集計して 2 行 (pure / gpu_opt) が出ることを確認
- [ ] テストラン 1 試行で `um_experiment.log` に `=== ... rc=0 ===` と `[GPU Phase] ... Prefetch cum=...` が出ることを確認

すべて ✓ なら Phase 4 (qsub) へ。

---

## 7. 触ってはいけないファイル (再確認)

| ファイル | 理由 |
|---|---|
| `brandes_gpu_opt_pure.cu` | OOM が論文主張の根拠なので変更禁止 |
| `brandes_sequential.cpp`, `brandes_pathmerge.cu`, `brandes_cugraph.cu`, OpenMP 系 | ベースラインなので変更禁止、再ベンチも不要 |
| `tools/bandwidth_benchmark.cu` | ハード測定なので 1 回取れば使い回し |
| 既存の v1 結果ディレクトリ `build_miyabi/result_um_oversubscribe/` | 上書き厳禁。v2 は別名で保存 |

v2 結果の保存先は `run_um_oversubscribe_experiment.sh` 内の `RESULT_DIR` を
`result_um_oversubscribe_v2` に変更しておくと安全。

---

## 8. 受け取った AI への指示

1. **本ドキュメント §1 の [BUG-1], [BUG-2] を最優先で修正**。これがないと実験データを取っても
   集計できない。
2. **§3 の [MINOR-1], [MINOR-2], [MINOR-3] も同時に修正**。実験を取り直すコストの方が高い。
3. **行番号は変動している可能性があるので**、必ず `grep` で対象箇所を再確認してから編集。
4. 修正後は §6 のチェックリストを順に潰す。**全項目 ✓ になるまで `qsub` しない**。
5. 修正完了の報告には、各修正の **diff (修正前後) を提示**すること。
6. **`brandes_gpu_opt_pure.cu` は絶対に触らない**。

---

## 9. 参照ファイルパス一覧

### 修正対象
- `lab/mylab/research/scripts/summarize_oversubscribe.py` ← [BUG-1], [BUG-2]
- `lab/mylab/research/scripts/run_um_oversubscribe_experiment.sh` ← [MINOR-1]
- `lab/mylab/research/brandes_gpu_opt_pure_chunked.cu` ← [MINOR-2] (確認のみ、変更不要かも)

### 確認のみ (変更禁止)
- `lab/mylab/research/brandes_gpu_opt.cu` ← v2 設計どおり、変更不要
- `lab/mylab/research/brandes_gpu_opt_pure.cu` ← OOM 主張の根拠
- `lab/mylab/research/main.cpp` ← dispatcher 登録済み
- `lab/mylab/research/brandes.h` ← 宣言登録済み
- `lab/mylab/research/CMakeLists.txt` ← ビルド対象登録済み

### リビルド対象
- `lab/mylab/research/build_miyabi/brandes_runner` ← May 10 10:22 (古い、要リビルド)

### 出力先 (実験後)
- `lab/mylab/research/build_miyabi/result_um_oversubscribe_v2/oversubscribe_results.tsv`
- `lab/mylab/research/build_miyabi/result_um_oversubscribe_v2/um_experiment.log`
- `lab/mylab/research/build_miyabi/result_um_oversubscribe_v2/summary_v2.md`

### 論文反映 (実験後の Phase で対応)
- `draft/Chapter/Evaluation.tex` ← Sec 4.4 を 3 者比較に書き直し
- `draft/Chapter/Methodology.tex` ← HBM3 streaming 節を追加
- `draft/scripts/generate_figures_tables.py` ← `fig:um_oversubscribe` 追加
