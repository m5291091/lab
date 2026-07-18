# 11 限界

## 11.1 評価範囲

1. RQ1 は email-EuAll と roadNet-PA/TX/CA の 4 グラフ、固定 b512、GH200 1 台に限定される。
2. RQ2 は synthetic 4 + email の 5 グラフ。合成 4 集約は他 3 = job 2354994、修正版 325557 = job 2406254 / checkpoint `45352a3` の mixed-checkpoint であり、same-checkpoint remeasurement ではない。
3. RQ3 と RQ4 Tier B は修正版 325557 の 1 グラフのみ。RQ3 の targeted boundary は各条件 1 試行で、runtime を方式間性能比較に用いない。
4. RQ4 Tier A の independent full-vector reference は小規模 3 グラフのみ。headline 4 グラフの独立 full-vector reference は未取得。

## 11.2 容量評価

5. working-set values はコードの allocation 寸法から導出した estimate であり、measured process RSS、physical HBM residency、host residency、migration bytes ではない。
6. 公称 96 GB と runtime total 約 102.0 decimal GB は同一 HBM3 の異なる単位・取得方法である。capacity comparison は保存記録系の `free_before≈101.4 GB` を用いる。
7. Pure b8192 の CUDA device-memory OOM と UM b12288 の cgroup host-memory OOM kill は異なる failure class であり、単一の HBM OOM 境界として扱わない。
8. UM は b12288 で失敗しており無制限ではない。Chunked は b16384 が tested upper success で、その先の limit は未測定。
9. `SUB_BATCH=6596` は修正版 325557 で `INT_MAX/n` の index-safety bound が binding だった結果であり、他 graph へ一般化しない。

## 11.3 正確性

10. Tier A は小規模 independent reference、Tier B は corrected-325557 cross-implementation consistency であり証拠強度が異なる。
11. 全 13 比較は mixed tolerance 内で PASS したが `ByteIdentical=No`。bitwise identity を主張しない。
12. PathMerge は第三者実装の external comparator であり ground truth ではない。
13. 修正版 325557 は internally reconstructed data であり、original generation seed / complete upstream original を確認できない provenance limitation が残る。

## 11.4 Historical Evidence

14. 旧 malformed `325557_3216152` の `CORE_FAIL`、stress mismatch、PathMerge difference、legacy `OOM_OR_FAIL` は current active conclusion ではない。ただし削除せず `result/correctness/memory_paths/canonical_job_2368587/`、`failure/`、provenance documents に historical invalid-input evidence として保存する。
15. historical result を current result へ混入させず、修正版 PASS を旧判定へ遡及適用しない。

## 11.5 比較・一般化

16. PathMerge は graph ごとに tuned、GPU_Opt は固定 b512 で設定が非対称である。
17. 現行 block による medium/large の統一 7 実装比較は未達。cuGraph は small の補助 baseline に限定する。
18. H/W/A の因果を roadNet、memory boundary を他 GPU・他 host limit、corrected provenance を外部原本の完全性へ一般化しない。

## 11.6 追加実験

| 目的 | 必要な追加証拠 |
|:--|:--|
| Headline graph の独立 full-vector 検証 | Sequential または別の独立 reference |
| 容量境界の統計的強化 | 同一条件の複数 trial と追加 boundary points |
| 物理メモリ挙動 | Full-run RSS, HBM/host residency, migration bytes |
| RQ2 checkpoint 統一 | 4 synthetic graphs の same-checkpoint remeasurement |
| Provenance 強化 | Original seed または upstream complete original |
| 一般化 | 追加 graph、他 GPU、異なる host-memory configuration |
