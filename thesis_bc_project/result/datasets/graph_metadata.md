# graph_metadata — グラフ来歴と整合性（叙述）

機械可読版は `graph_catalog.tsv`（24 列）。ここでは来歴・修復・検証状態のうち
TSV に収まらない内容を記録する。

## サイズ定義（`graph_catalog.tsv` の後半 5 列）

各グラフについて次を推定なしで記録する（`FileSizeBytes` は canonical 入力を `stat` した実測値）。

| 列 | 定義 |
|:--|:--|
| `FileSizeBytes` | ディスク上の CSR テキスト入力ファイルの容量（`stat`）。GPU メモリ使用量ではない。 |
| `FileSizeMB` | `FileSizeBytes / 1,000,000`（decimal MB） |
| `FileSizeMiB` | `FileSizeBytes / 1,048,576`（binary MiB） |
| `CSRBytes` | コードが読み込む CSR 配列の理論容量 `((n + 1) + 2m) × 4`（int32） |
| `BCVectorBytes` | BC 出力ベクトルの容量 `n × 8`（double） |

これらは**静的なグラフ表現**であり、GPU の working set とは別概念である。1 始点の
BC 処理に必要な per-source state は `32n + 4·D_est + 8`（`D_est = max_depth_estimate`）で、
batch 依存 working set は `EffectiveNS × EffectiveBatch × PerSourceStateBytes`（Chunked は
同時 resident を `SUB_BATCH` に制限）で決まる。これは code 由来の allocation 見積りであり、
実測 RSS・実測 HBM residency・実測 migration bytes とは区別する。

- 検証器: `tools/validate_graph_csr.py`（読み取り専用。PASS=exit 0 / FAIL=exit 1）
- 全 12 行の検証状態: **valid 11 / malformed 1**（malformed は `325557_3216152` のみ）

## ValidationStatus 一覧（`tools/validate_graph_csr.py`）

| graph | status | n | m | self-loop | duplicate | isolated | components |
|:--|:--|--:|--:|--:|--:|--:|--:|
| email-EuAll | PASS | 265,009 | 364,481 | 0 | 0 | 0 | 15,631 |
| roadNet-PA | PASS | 1,088,092 | 1,541,898 | 0 | 0 | 0 | 206 |
| roadNet-TX | PASS | 1,379,917 | 1,921,660 | 0 | 0 | 0 | 424 |
| roadNet-CA | PASS | 1,965,206 | 2,766,607 | 0 | 0 | 0 | 2,638 |
| benchmark_85830.data | PASS | 85,830 | 241,080 | 0 | 0 | 0 | 4 |
| benchmark_7000_41459 | PASS | 7,000 | 41,459 | 0 | 0 | 0 | 1 |
| benchmark_11023_62184 | PASS | 11,023 | 62,184 | 0 | 0 | 0 | 1 |
| 56438_300801 | PASS | 56,438 | 300,801 | 0 | 0 | 0 | 1 |
| chain_200 | PASS | 200 | 199 | 0 | 0 | 0 | 1 |
| random | PASS | 32,212 | 101,805 | 0 | 0 | 0 | 1 |
| **325557_3216152** | **FAIL** | 325,557 | 3,216,152 | n/c | n/c | 1 | n/c |
| **325557_3216152_corrected_v1** | **PASS** | 325,557 | 3,216,152 | 87,442 | 866,924 | 0 | 1 |

`n/c` = not computed（構造破損のため多重集合解析を打ち切り）。
`duplicate` は多重度 > 1 の有向ペア数。

> **注記**: self-loop または duplicate edge を持つのは `325557` 系のみである。
> 他の合成 6 グラフは未ソート行 0・self-loop 0・duplicate 0・0-based であり `tools/gen_graph.py` の出力と整合する。

---

## 325557_3216152（旧 / legacy malformed input）

| 項目 | 値 |
|:--|:--|
| path | `data/325557_3216152` |
| SHA256 | `a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584` |
| size | 45,348,142 bytes |
| ValidationStatus | **malformed** |
| 不整合 | 1-based を 0-based として格納。隣接要素が 2m に **7 個不足**、範囲外 ID `325557` が **7 個**、頂点 0 が孤立、最終頂点（1-based `325557`）の行が欠落 |
| ProvenanceStatus | `unverified_external_origin`（generator・seed とも不明、外部 provenance は独立検証不能） |
| UsedForHistoricalExperiments | **yes** |
| UsedForNewExperiments | **no** |
| DoNotDelete | **yes** |

- 旧カタログの `Preprocessing = tools/gen_graph.py 生成 (合成, 1-indexed)` は**ファイル内容と矛盾する**ため
  `provenance_unverified` へ訂正した（行は削除していない）。根拠は
  `../provenance/GRAPH_325557_INTEGRITY_AUDIT.md` §4.1。
- 本ファイル上で得られた既存の raw 結果・`CORE_FAIL`・vector・SHA256 は**保存し変更しない**。
- 現行コードは fail-fast により本ファイルを実行前に拒否する。履歴実験の再現には
  アーカイブ済み checkpoint（`code_snapshots/<SourceSnapshotID>/`）を使う。

## 325557_3216152_corrected_v1（修復版 / new）

| 項目 | 値 |
|:--|:--|
| new graph ID | `325557_3216152_corrected_v1` |
| corrected file path | `data/325557_3216152_corrected_v1` |
| corrected SHA256 | `8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22` |
| size | 45,348,105 bytes |
| legacy source path | `data/325557_3216152` |
| legacy SHA256 | `a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584` |
| correction script path | `tools/repair_325557_graph.py` |
| correction method | 1-based → 0-based の**グラフ全体の relabelling**（頂点行を含む）+ 対称性による欠落最終行（7 要素）の再構成 |
| n / m | 325,557 / 3,216,152（**維持**） |
| self-loop policy | **preserve**（87,442 本 / 174,884 要素） |
| duplicate-edge policy | **preserve**（多重度 2 の有向ペア 866,924。dedup しない） |
| ProvenanceStatus | `internally_reconstructed_no_original_seed` |
| UsedForNewExperiments | **yes** |
| UsedForHistoricalExperiments | **no** |
| ValidationStatus | **valid**（PASS, 0 failed check） |

### 修復の一意性

- 欠落行の**多重集合**は一意: 次数 `2m - ptr[n] = 7` と値 `325557` の出現数 `7` が一致するため
  self-loop は 0 本と確定し、隣接は「他行で `325557` を持つ行の所有者」として定まる。
- 欠落行の**並び順**は、逆辺の出現位置順と昇順ソートが一致するため、規約の選択に依存しない。
- 再構成した隣接（0-based）: `289276, 289277, 289278, 289279, 289280, 325555, 325555`。
- **決定性**: 同一入力から byte-identical な出力（独立 2 回実行で SHA256 一致を確認）。
- **非正規化の証拠**: 未ソート行数が旧 146,642 → 修復版 146,642 で不変。行内ソート・dedup・
  self-loop 除去などの副次的正規化を行っていない。

### メモリ見積りへの影響

修復版の静的サイズ: 入力ファイル `45,348,105 bytes`（45.35 MB / 43.25 MiB）、
CSR topology `((n+1)+2m)×4 = 27,031,448 bytes`（27.03 MB / 25.78 MiB）、
BC 出力ベクトル `n×8 = 2,604,456 bytes`（2.60 MB / 2.48 MiB）。いずれも HBM3 を超えない。

`n` と `m` が不変のため、`avg_deg = 19.758`（< 20）→ `max_depth_estimate (D_est) = 256` も不変であり、
per-source state `32n + 4·D_est + 8`（`src/proposed/host_pure.cu:141-157` で確認）は旧入力と**同一**
（= 10,418,856 bytes）。修正版 job 2404743 の implementation manifest に基づく code-derived
working-set estimate（`EffectiveNS × EffectiveBatch × PerSourceStateBytes`）は、Pure b4096
（EffectiveNS=2）が 85,351,268,352 bytes、Pure b8192 が 170,702,536,704 bytes、UM b10240
（EffectiveNS=1）が 106,689,085,440 bytes、UM b12288 が 128,026,902,528 bytes である。
Chunked b16384 は同時 resident を `SUB_BATCH=6596` に制限し、68,722,774,176 bytes と見積もる。
入力ファイル自体は約 43.25 MiB であり、大容量になるのは **batch 依存 working set** であって、
グラフファイル・CSR・BC ベクトルではない。これらは allocation estimate であり、実測 RSS、
physical HBM residency、migration bytes ではない。

詳細は `../provenance/GRAPH_325557_INTEGRITY_AUDIT.md`。
