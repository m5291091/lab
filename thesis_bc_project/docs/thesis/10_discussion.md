# 10 考察

観測事実、解釈、未測定の可能性を分ける。数値は修正版 325557 の正式 artifact と RQ1 の不変値に基づく。

## 10.1 RQ1〜RQ4 の統合

- **RQ1 (`SUPPORTED`)**：固定 b512 の block GPU_Opt は、評価した第三者実装 PathMerge tuned に対し email 3.17×、PA 1.31×、TX 1.51×、CA 1.45×。修正版 325557 は RQ1 に使用していない。
- **RQ2 (`SUPPORTED_WITH_LIMITATIONS`)**：修正版 325557 は H=1.4767 / W=1.1012 / A=1.5563、合成 4 mixed-checkpoint 集約は H=1.679 / W=1.066 / A=1.391。Hybrid BFS と 2 streams が主要な正の効果を示し、warp は graph-dependent。ただし roadNet へ因果を一般化しない。
- **RQ3 (`SUPPORTED_WITH_LIMITATIONS`)**：corrected targeted boundary は Pure b4096 success / b8192 CUDA device OOM、UM b10240 success / b12288 cgroup host-memory OOM kill、Chunked b16384 success (`SUB_BATCH=6596`, `num_subs=3`)。各条件 n=1 で runtime 比較ではない。
- **RQ4**：Tier A は小規模 3 グラフの独立 CPU 参照で `SUPPORTED`。Tier B は修正版 325557 の 10 比較で `SUPPORTED_WITH_LIMITATIONS`。全 13 比較 PASS / mismatch 0 / byte-identical No。

## 10.2 性能要因の解釈

email と roadNet の speedup 差は graph structure の違いと整合するが、roadNet では H/W/A factorial ablation を行っていないため原因を断定しない。W の主効果は 0.970〜1.175 に分布し、修正版 325557 では 1.1012 であった。専用 counter による因果検証はなく、次数分布だけから一般則を導かない。

合成 4 集約は他 3 グラフ = job 2354994、修正版 325557 = job 2406254 / checkpoint `45352a3` の mixed-checkpoint であり、same-checkpoint four-graph remeasurement ではない。

## 10.3 容量問題の設計上の含意

修正版 325557 の input file / CSR / BC vector は 45,348,105 / 27,031,448 / 2,604,456 bytes で HBM3 を超えない。容量を支配するのは per-source state 10,418,856 bytes を batch 内で複製する working set である。

```
Working-set estimate = EffectiveNS × EffectiveBatch × PerSourceStateBytes
```

batch/sub-batch は graph partition や近似ではなく source grouping であり、全 source を反復処理する。このため容量制御点は graph file size ではなく同時 source state 数にある。

## 10.4 UM と Chunked

UM は managed allocation と migration により device memory を超え得る working set を扱う。b10240 の code-derived estimate 106.69 GB は `free_before≈101.4 GB` を超える条件で成功したが、b12288 は cgroup host-memory OOM kill であり無制限ではない。入力 graph を分割格納する仕組みとして説明しない。

Chunked は source batch を sub-batch に分け、b16384 の同時 resident estimate を 68.72 GB に制限した。`SUB_BATCH=6596` は `INT_MAX/n` の index-safety 上限が binding constraint であった。Chunked の主価値は最高速度ではなく resident working-set control と tested capacity extension であり、未測定条件の OOM 回避を保証しない。

## 10.5 正確性と provenance

Tier B は混合許容内の cross-implementation consistency で、独立 ground truth ではない。PathMerge は external comparator であり、全 13 比較は byte-identical ではない。修正版 graph は内部再構成データで元 seed・上流原本を確認できず、この provenance limitation を残す。

旧 malformed 入力の `CORE_FAIL` は current active conclusion から外すが、`canonical_job_2368587`、`failure/`、provenance documents に historical invalid-input evidence として保存する。旧判定を削除・relabel せず、修正版 job 2404743 の PASS を遡及適用しない。

## 10.6 妥当性への脅威

- **Internal**：RQ2 mixed-checkpoint、job/checkpoint 分離、CUDA OOM と cgroup OOM の evidence class。
- **External**：RQ1 4 graph、RQ2 5 graph、RQ3/Tier B 1 corrected graph、GH200 1 台。
- **Statistical**：corrected boundary は各条件 1 trial。runtime ranking を主張しない。
- **Construct**：working-set は code-derived estimate。process RSS、physical HBM residency、migration bytes は未取得。
- **Data provenance**：corrected graph は deterministic repair だが original seed/upstream original 不明。
- **Baseline**：PathMerge は第三者実装で ground truth ではない。

## 10.7 今後の課題

headline graph の独立 full-vector reference、同一 checkpoint の synthetic-4 remeasurement、複数 trial の容量境界、full-run RSS/residency/migration、追加 graph・他 GPU、upstream original/seed による provenance 強化が必要である。
