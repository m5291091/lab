# 08 メモリ容量評価（RQ3）

**実行時間の軸と容量拡張の軸を分けて説明する。** 本章は 2 つの別実験に基づく。混同しない。

- **実験 A：`result/memory_scalability/`**（checkpoint `oldtree_f05ec52_20260512`＝旧 tree, n=5）——
  feasibility（SUCCESS/OOM 境界）を限定的に採用。**時間値は headline block 性能値として非採用**。
- **実験 B：`result/correctness/memory_paths/`**（checkpoint `memory_correctness_20260712`/`memory_diagnostic_20260713`, 100 GiB queue,
  各構成 n=1）——正確性・診断が主目的。ここでは oversubscription 経路証拠と OOM を引用。

グラフはいずれも 325557_3216152（合成, n=325,557, m=3,216,152, avg_deg 19.76, 人為的バッチ
強制で oversubscribe）の 1 件のみ。

## 8.1 GPU_Opt / Pure / Chunked の違い（設計）
[04](04_method_design.md) §4.7 の通り。要点：Pure は `cudaMalloc` で BATCH×n を device 直接
確保（oversubscription 非対応）、UM は `cudaMallocManaged` で HBM3 超過分を LPDDR5X へ spill、
Chunked は実確保を SUB_BATCH 単位に抑え常に HBM3 内。

## 8.2 実行可能バッチ範囲と OOM 境界（T-MEM・必須, 実験 A）
`Status` は SUCCESS/OOM。**OOM は 0 秒ではなく「未達」を表す**（`Time_sec=0` はマーカー）。

| BatchSize | GPU_Opt(UM) | GPU_Opt_Pure | GPU_Opt_Pure_Chunked |
|:--|:--:|:--:|:--:|
| 512 | SUCCESS | SUCCESS | SUCCESS |
| 1024 | SUCCESS | SUCCESS | SUCCESS |
| 2048 | SUCCESS | SUCCESS | SUCCESS |
| 4096 | SUCCESS | SUCCESS | SUCCESS |
| 8192 | SUCCESS | **OOM** | SUCCESS |
| 10240 | SUCCESS | OOM | SUCCESS |
| 12288 | **OOM** | OOM | SUCCESS |
| 16384 | （未測定/―） | OOM | SUCCESS |

- **最大成功バッチ**：Pure = **b4096**、UM = **b10240**（b12288 で OOM）、Chunked = **b16384**（全成功）。
- 出典：`result/memory_scalability/oversubscribe_results_gpu_opt{,_pure,_pure_chunked}.tsv`。
- 「UM は無制限」は**偽**。UM も b12288 で OOM する（実験 A）。

## 8.3 容量拡張の経路証拠（実験 B, 100 GiB queue）
`result/correctness/memory_paths/canonical_job_2368587/execution_summary.tsv` より：
- **UM b9792 完走**：oversubscribed=true, free_before=101.4 GB, managed_alloc_estimate=102.02 GB
  （> free_before）, SUB_BATCH=6596, num_subs=2, NS_eff=1, Prefetch cum=33.18 s。
  → est>free_before・HBM3 streaming・NS_eff=1・num_subs=2・SUB_BATCH<batch・Prefetch cum>0 の
  oversubscription 経路証拠を満たす。**ただし migration byte 量の直接計測ではない**。
- **Chunked b16384 完走**：num_subs=3（`gpu_opt_pure_chunked`）。
- **UM b10240 は 100 GiB queue で OOM**：dynamic(UM)=213.38 GB, runner_exit=137（SIGKILL）。
  PathMerge b4096（reference）は成功（`failure/failed/oom/memory_correctness_2368269/`）。

> **環境差の注意**：実験 A（旧 tree）で UM は b10240 SUCCESS / b12288 OOM。実験 B（100 GiB
> queue）で UM は b9792 SUCCESS / b10240 OOM。**OOM 境界は環境（ホストメモリ上限）依存**であり、
> 単一の固定境界として述べない。共通して言えるのは「Pure < UM < Chunked の順に到達バッチが
> 大きい」という feasibility の順序である。

## 8.4 Chunked の num_subs（分割数）
- 実験 B：Chunked b16384 → **num_subs=3**、b1024 → num_subs=1（`execution_summary.tsv`）。
- 分割により実確保を HBM3 予算内に保つのが Chunked の主機能。

## 8.5 性能と容量のトレードオフ（実験 A の within-experiment 傾向のみ）
**以下の時間値は実験 A（`oldtree_f05ec52_20260512`, 旧 tree）内の相対傾向であり、headline block 性能値ではない**
（`result/provenance/um_code_diff_audit.md`, `result/environment/environment.md`）。同一実験内の
バッチ間相対比較としてのみ引用する。

| BatchSize | UM median[s]（実験A, 非headline） | 状態 |
|:--|--:|:--|
| 4096 | 67.65 | in-capacity |
| 8192 | 109.82 | oversubscribed（HBM3 超過, spill 開始） |
| 10240 | 324.22 | oversubscribed（LPDDR5X spill 増大） |

- **観測（実験 A 内）**：UM は oversubscription 領域（b8192→b10240）で in-capacity 比に対し
  実行時間が増大する（b4096≈67.7 s → b10240≈324 s）。これは HBM3 を超えた working set が
  NVLink-C2C 経由で LPDDR5X にスピルするコストと解釈できる（帯域：HBM3 1818.6 vs C2C
  Prefetch 177.7 GB/s, [05](05_experimental_setup.md)）。
- **Chunked は oversubscription 領域でも実行時間が概ね平坦**（b8192≈70.7, b16384≈69.3 s）。
  Chunked の主効果は最高性能ではなく**実行可能バッチの拡大**。
- **解釈の限界**：これは実験 A 内の相対傾向であり、絶対時間・headline 性能とは切り離す。
  migration byte 量は直接計測していない（[11](11_limitations.md)）。

## 8.6 まとめ（RQ3 回答）
- **feasibility（到達バッチ）**：Pure（≤b4096）< UM（実験Aで≤b10240 / 実験Bで≤b9792）<
  Chunked（≤b16384）。UM は Pure が OOM する領域でも実行を継続するが無制限ではない。
- **性能**：in-capacity では 3 方式は同等（[06](06_results_performance.md) の proposed_variants
  でも email/road で GPU_Opt ≈ Pure ≈ Chunked）。oversubscription 領域では UM が spill コストで
  減速、Chunked は平坦（実験 A 内傾向）。
- **用途の違い**：UM は「コード変更なしで容量超過に耐える簡潔さ」、Chunked は「最大バッチ拡張と
  安定した実行時間」。Pure は容量内での対照。
- **書かないこと**：「あらゆる条件で OOM 回避」「UM が容量制約を完全解消」「migration byte を
  直接計測」。
