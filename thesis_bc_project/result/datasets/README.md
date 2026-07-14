# datasets — グラフカタログ

実験に使用したグラフの一覧。**グラフ本体は `thesis_bc_project/data/` に格納**（巨大ファイルは複製しない）。
`graph_catalog.tsv` に n/m/avg_deg/path/SHA256 に加え SourceURL/DirectedOriginal/UsedAsDirected/Symmetrized/SelfLoopHandling/DupEdgeHandling/Preprocessing を記録（不明値は unknown、推定なし）。

- CSR テキスト形式（3行: `n m` / `ptr[0..n]` / `adj[0..2m-1]`）。無向・非重み。
- 主軸(A) headline: email-EuAll(265K), roadNet-PA(1.09M)/TX(1.38M)/CA(1.96M)。
- 副次(B) UM: 325557_3216152（合成 325K, 人為的バッチ強制で oversubscribe）。
- 小規模正確性(planned): benchmark_7000_41459 / benchmark_11023_62184 / chain_200。
- ablation: benchmark_7000/11023 / 56438_300801 / 325557_3216152（+ email_2354999）。

SNAP グラフは `tools/download_snap_graphs.sh` で再取得可（`data/snap/` は gitignore）。SHA256 で同一性確認可能。
