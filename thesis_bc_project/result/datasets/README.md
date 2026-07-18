# datasets — グラフカタログ

実験に使用したグラフの一覧。**グラフ本体は `thesis_bc_project/data/` に格納**（巨大ファイルは複製しない）。
`graph_catalog.tsv`（24 列）に n/m/avg_deg/path/SHA256 に加え SourceURL/DirectedOriginal/UsedAsDirected/Symmetrized/SelfLoopHandling/DupEdgeHandling/Preprocessing、
ValidationStatus/ProvenanceStatus/UsedForNewExperiments/UsedForHistoricalExperiments/DoNotDelete、
および入力サイズ 5 列 FileSizeBytes/FileSizeMB/FileSizeMiB/CSRBytes/BCVectorBytes を記録（不明値は unknown、推定なし）。
叙述版は `graph_metadata.md`。ValidationStatus は `tools/validate_graph_csr.py` の判定（valid 11 / malformed 1）。

- CSR テキスト形式（3行: `n m` / `ptr[0..n]` / `adj[0..2m-1]`）。無向・非重み。
- サイズ定義: `FileSizeBytes` = ディスク上の入力ファイル容量（GPU メモリ使用量ではない）、`MB = /1,000,000`、`MiB = /1,048,576`、`CSRBytes = ((n+1)+2m)×4`、`BCVectorBytes = n×8`。GPU working set は batch 依存の per-source state（`graph_metadata.md`）であり、入力ファイル容量とは別概念。
- 主軸(A) headline: email-EuAll(265K), roadNet-PA(1.09M)/TX(1.38M)/CA(1.96M)。
- 副次(B) UM: **旧入力** `325557_3216152` は historical malformed input（1-based を0-basedとして格納、`ValidationStatus=malformed`）。既存のmemory feasibility / memory-path / synthetic ablationはこの旧入力を使用した履歴結果である。**新規Series A/B/Cは修正版** `325557_3216152_corrected_v1` のみを使用する。
- 小規模正確性(planned): benchmark_7000_41459 / benchmark_11023_62184 / chain_200。
- historical ablation: benchmark_7000/11023 / 56438_300801 / 旧 `325557_3216152`（+ email_2354999）。新規Series Cの325557対象は修正版であり、旧入力を新規ablation対象にしない。

旧入力は履歴実験の再現性と監査証跡を保つため削除しない（`DoNotDelete=yes`）。修正版は
`tools/repair_325557_graph.py` が旧入力から別名へ決定的に生成し、SHA256は
`8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22`。来歴・検証は
`graph_metadata.md`、監査は `../provenance/GRAPH_325557_INTEGRITY_AUDIT.md`。

SNAP グラフは `tools/download_snap_graphs.sh` で再取得可（`data/snap/` は gitignore）。SHA256 で同一性確認可能。
