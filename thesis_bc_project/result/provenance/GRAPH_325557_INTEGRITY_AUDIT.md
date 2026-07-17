# グラフ 325557_3216152 整合性監査（Gate W7.2 / W7.3A）

`data/325557_3216152` は **1-based 頂点番号の CSR を 0-based として格納した不正入力**である。
本書はその確認内容、修復版の作成方法、両者の関係を記録する。

- 旧ファイルは **削除・上書き・修正しない**（`DoNotDelete=yes`）。既存実験の入力として保存する。
- 修復版は **別名** `data/325557_3216152_corrected_v1` として新規追加した。
- 既存の raw 結果・既存の `CORE_FAIL`・既存の vector・既存の SHA256 は**変更していない**。

検査は `tools/validate_graph_csr.py`、修復は `tools/repair_325557_graph.py` で再現できる（いずれも読み取り専用で旧ファイルを扱う）。

---

## 1. 旧ファイル（legacy malformed input）

| 項目 | 値 |
|:--|:--|
| パス | `data/325557_3216152` |
| SHA256 | `a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584` |
| サイズ | 45,348,142 bytes（約 43.25 MiB） |
| ValidationStatus | **malformed** |
| header | `n=325557`, `m=3216152` |
| expected adjacency count (2m) | 6,432,304 |
| actual adjacency count | 6,432,297 |
| **shortage** | **7** |
| out-of-range vertex IDs | **7**（すべて値 `325557`。0-based の有効範囲は `[0, 325556]`） |
| 頂点 ID の実体 | **実質 1-based**（`adj` の値域は `[1, 325557]`、値 `0` の出現は 0 個） |
| 頂点 0 | **孤立**（`ptr[0]=ptr[1]=0`。孤立行は頂点 0 のみ） |
| 最終 source 行 | **次数 7 の行が欠落**（下記 §1.1） |
| self-loop | **存在**（隣接要素 174,884 個 = 87,442 本。両方向 2 要素として格納） |
| duplicate edge | **存在**（多重度 2 の有向ペア 866,924 個。最大多重度 2） |
| 行内の並び | 昇順ソートされていない行が 146,642 / 325,557 |
| 元の generator と乱数 seed | **入手不能**（`ProvenanceStatus` は §4 参照） |
| 外部 provenance | **独立検証不能** |

### 1.1 「最終 source 行の欠落」の意味

row pointer は `ptr[0..n]`（325,558 個）しか持たない。1-based では頂点 `1..325557` が実体だが、
頂点 `325557` の行は `ptr[325558]` を要するため **格納場所そのものが存在しない**。
その結果:

- `ptr[n] = 6432297 = len(adj)`（ptr と adj は互いに整合しているが、ともに 2m に 7 不足）
- 欠落した旧頂点 `325557` の次数 = `2m - ptr[n]` = **7**
- 他行に現れる値 `325557` の個数 = **7**

両者が一致するため、旧頂点 `325557` の self-loop は `(7 - 7) / 2 = 0` 本と**確定**する
（self-loop は両方向 2 要素として格納される規約。この規約は `len(adj)/2 = 3216152 = m` が
成立することと整合する）。

### 1.2 非対称性は欠落行のみに起因する

有向ペアの多重集合比較で、`count(u,v) != count(v,u)` となる有向ペアは **6 個**であり、
そのすべてが頂点 `325557` を含む。頂点 `325557` を含むペアを除くと非対称は **0 個**。
つまり旧ファイルの唯一の欠損は最終行であり、他の行は完全に対称である。

---

## 2. 既存 reader の挙動（実行時に何が起きていたか）

Gate W7.2 の「既存 reader が範囲外頂点を検査していない」を、コードとして確認した記録。

### 2.1 切り詰めは 0 で埋められていた

修正前の `Graph::readGraph()`（`src/core/graph.cpp`）は `std::cin >> ...` で `2*edgeCount` 個を読むだけで、
抽出の成否を検査していなかった。C++11 以降、抽出に失敗した `operator>>` は**対象を 0 に設定する**ため、
不足していた 7 要素は `adj[6432297..6432303] = 0` として読み込まれていた。
ただしこの 7 スロットは `ptr[n] = 6432297` より後方にあり、**どの行からも参照されない死領域**である。
したがって「0 の辺が 7 本増えた」わけではない。

### 2.2 範囲外 ID `325557` は live な行の内部にある

値 `325557` の 7 個の出現位置は `[5509468, 5510015, 5510512, 5511009, 5511506, 6432295, 6432296]` で、
すべて `ptr[n] = 6432297` **未満**（= 実際に走査される領域）。所有行は
`289277, 289278, 289279, 289280, 289281, 325556, 325556` であり、すべて有効な source 頂点である。

### 2.3 範囲外 ID は無検査で per-source 配列の添字になっていた

`include/proposed/brandes_kernels.cuh` の前進 BFS は隣接値を境界検査なしに添字として使う:

```
const size_t base = batch_node_offset(batch_idx, n_nodes);   // = batch_idx * n_nodes
w = C[j];
if (atomicCAS(&d_d[base + w], -1, depth + 1) == -1) { ... d_Q_next[base + pos] = w; }
atomicAdd(&d_sigma[base + w], d_sigma[base + v]);
```

`w = 325557 = n_nodes` のとき `base + w = (batch_idx + 1) * n_nodes` となり、これは
**次の source スライスの先頭要素**（= 次 source から見た頂点 0 の状態）を指す。
バッチ内最後のスライスでは確保領域の末尾を 1 要素超える。
さらに `w = 325557` が frontier に入り `v` として取り出されると `R[v+1] = R[325558]` を読むが、
`R` の要素数は `n+1 = 325558`（有効添字 `0..325557`）であり、これも範囲外読み出しである。

**確認済みの事実として言えること**: 旧入力での実行は、範囲外の添字アクセス（source 間の状態への書き込みを含む）を行っていた。
**言えないこと**: それが報告済み BC 値をどの程度変えたか、fault したか、無害だったか。保存ログ・成果物からは決定できない。

> **帰属の制限**: 既存の `same_impl_diff_batch`（stress）差は `CLAIMS.md` で「原因未特定」である。
> この構造的欠陥は batch 構成に依存する汚染経路であり observed の batch 依存性と**矛盾しない**が、
> 本 Gate では因果を確定していない。stress 差を GPU 数値計算へ帰属しないのと同様に、
> **本欠陥が原因であるとも断定しない**。修正版での再検証（Gate W7.3A §8 Series A）後に判断する。

---

## 3. 修復版（corrected）

| 項目 | 値 |
|:--|:--|
| new graph ID | `325557_3216152_corrected_v1` |
| corrected file path | `data/325557_3216152_corrected_v1` |
| corrected SHA256 | `8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22` |
| corrected size | 45,348,105 bytes |
| legacy source path | `data/325557_3216152` |
| legacy SHA256 | `a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584` |
| correction script path | `tools/repair_325557_graph.py` |
| correction method | 1-based → 0-based の**グラフ全体の relabelling**（頂点行を含む）+ 対称性による欠落行の再構成 |
| n / m | `325557` / `3216152`（**維持**） |
| self-loop policy | **preserve**（87,442 本 / 174,884 要素、増減なし） |
| duplicate-edge policy | **preserve**（多重度そのまま。dedup しない） |
| ValidationStatus | **PASS**（`tools/validate_graph_csr.py`） |
| ProvenanceStatus | `internally_reconstructed_no_original_seed` |
| UsedForNewExperiments | **yes** |
| UsedForHistoricalExperiments | **no** |

### 3.1 修復アルゴリズム

1. 旧 CSR を **1-based 頂点による無向 multigraph** として解釈する。
2. 欠落した旧頂点 `325557` の行を**対称性から再構成**する（§3.2）。
3. source と destination の**両方**を `k -> k-1` で 0-based へ変換する。
   - `new_ptr[i] = old_ptr[i+1]` (`i = 0..n-1`)、`new_ptr[n] = 2m = 6432304`
   - `new_adj = concat(old_adj, reconstructed_row) - 1`
   - 旧行 0（孤立した 1-based 欠番）が消え、旧頂点 `325557` の行が末尾に入るため、
     行数はちょうど `n = 325557` に一致する。
4. self-loop と duplicate edge の多重度を保持する。
5. dedup・self-loop 除去・辺の追加削除・行内ソートなどの**正規化はしない**。
   （修復版でも未ソート行は 146,642 のまま = 旧構造を保存している証拠）
6. `n=325557`, `m=3216152` を維持する。
7. 同じ入力から常に **byte-identical** な出力を生成する（乱数・辞書順依存なし）。

`tools/repair_325557_graph.py` は、旧ファイルの SHA256 と §1 の異常条件が一致しない場合、
**出力せず非 0 終了**する。

### 3.2 欠落 7 要素と一意性

再構成した旧頂点 `325557` の隣接（多重度込み）:

| | 1-based（旧） | 0-based（修復版） |
|:--|:--|:--|
| 隣接多重集合 | `289277, 289278, 289279, 289280, 289281, 325556, 325556` | `289276, 289277, 289278, 289279, 289280, 325555, 325555` |

**多重集合の一意性**: 次数 7 と値 `325557` の出現数 7 が一致するため self-loop は 0 本と確定し、
隣接多重集合は「他行で `325557` を持つ行の所有者」の多重集合として一意に定まる。
`325556` が 2 回現れるのは多重度 2 の辺であり、これも保存する。

**並び順の一意性**: 逆辺の出現位置順と昇順ソートが**一致する**ことを検証済み
（どちらの規約でも同一の並びになるため、順序の選択は結果を左右しない）。
スクリプトは両者の一致を検査し、一致しない場合は停止する。

### 3.3 修復版の検証結果

| 検査 | 旧 | 修復版 |
|:--|:--|:--|
| ValidationStatus | **FAIL**（3 checks） | **PASS**（0 failed） |
| `len(adj)` vs `2m` | 6,432,297 / 6,432,304（不足 7） | 6,432,304 / 6,432,304 |
| `ptr[n]` | 6,432,297 | 6,432,304 |
| 頂点 ID 値域 | `[1, 325557]`（範囲外 7 個） | `[0, 325556]`（範囲外 **0** 個） |
| 対称性 | `not_computed`（構造破損のため打ち切り） | **symmetric** |
| self-loop | （同上） | 87,442 本 / 174,884 要素 |
| duplicate ordered pairs | （同上） | 866,924（多重度ヒスト `{1: 4698456, 2: 866924}`） |
| 孤立頂点 | 1（頂点 0） | **0** |
| 連結成分 | （同上） | **1**（最大成分 325,557） |
| 未ソート行 | 146,642 | 146,642（**保存**） |

多重度ヒストグラムの差 `{1: 4698451→4698456, 2: 866923→866924}` は、
再構成行が「多重度 1 のペア 5 個 + 多重度 2 のペア 1 個」を追加したことと厳密に一致する
（`4698456 + 2 × 866924 = 6432304 = 2m`）。

---

## 4. Provenance（出所）

### 4.1 `tools/gen_graph.py` は本グラフを生成していない

`result/datasets/graph_catalog.tsv` の旧 `Preprocessing` 値は
`tools/gen_graph.py 生成 (合成, 1-indexed)` であったが、これは**ファイル内容と矛盾する**。
`tools/gen_graph.py` の構造から、以下は**いずれも生成不可能**である:

| 観測された性質 | `gen_graph.py` の該当箇所 | 生成可能か |
|:--|:--|:--|
| self-loop 87,442 本 | `ba`: `if t != new_node` / `er`: `adj[v].add(w)` は `w < v` / `grid`: 自明に無し | **不可** |
| duplicate edge 866,924 | 全モデルが `adj = [set() ...]`（集合） | **不可** |
| 未ソート行 146,642 | `write_csr`: `adj_sorted = [sorted(adj[i]) ...]` | **不可** |
| 1-based 頂点番号 | `write_csr` は 0-based で書き出す | **不可** |

対照として、カタログ上の他の合成グラフ 6 本（`benchmark_7000_41459`, `benchmark_11023_62184`,
`56438_300801`, `benchmark_85830.data`, `chain_200`, `random`）は**すべて未ソート行 0・self-loop 0・
duplicate 0・0-based** であり、`gen_graph.py` の出力と整合する。
**カタログ全 11 グラフのうち、self-loop または duplicate edge を持つのは 325557 のみである。**

したがって 325557 は `gen_graph.py` 由来ではなく、**出所不明の外部入力**である。
旧カタログの generator 帰属は誤りであり、本 Gate で `provenance_unverified` へ訂正した（行は削除していない）。

### 4.2 確定できないこと

- 元の generator（プログラム・モデル・パラメータ）
- 乱数 seed
- 取得元・取得日時
- 「1-based で書かれた理由」が生成側の仕様か、変換時の事故か

`ProvenanceStatus`:

- 旧: `unverified_external_origin`（generator/seed とも不明、独立検証不能）
- 修復版: `internally_reconstructed_no_original_seed`
  （旧ファイルのみを入力とする決定的再構成。原本 seed からの再生成ではない）

---

## 5. 履歴結果の扱い

- 旧入力上で得られた既存の raw 結果・`CORE_FAIL`・vector・SHA256 は**保存し、変更しない**。
- それらは **malformed legacy input 上の履歴結果**として記録する（`UsedForHistoricalExperiments=yes`）。
- 修復版での再検証（Gate W7.3A §8）が完了するまで、`CLAIMS.md` / `COVERAGE.md` の
  該当 status を `SUPPORTED` へ戻さない。

### 5.1 履歴結果の再現方法（fail-fast 導入後）

Gate W7.3A §5 で `load_graph()` にホスト側検証を追加したため、**現行コードは旧入力を実行前に拒否する**。
旧入力での実行を再現する必要がある場合は、fail-fast を持たない
アーカイブ済み checkpoint（`code_snapshots/<SourceSnapshotID>/`、本 Gate で未変更）を使う。
現行コードに検証の回避スイッチは**設けない**（不正入力を黙って skip / clamp / 補正しないため）。

---

## 6. 参照

| 目的 | パス |
|:--|:--|
| 汎用 CSR 検証器 | `tools/validate_graph_csr.py` |
| 修復スクリプト | `tools/repair_325557_graph.py` |
| ホスト側 fail-fast | `src/core/runner.cpp`（`validate_graph`）, `src/core/graph.cpp`（`readGraph`） |
| 限定再検証ジョブ | `scripts/run_corrected_325557_validation.sh`, `scripts/run_corrected_325557_ablation.sh` |
| データセット台帳 | `result/datasets/graph_catalog.tsv`, `result/datasets/graph_metadata.md` |
