# Ablation — corrected 325557 (job 2406254, Series C)

修正版 325557 グラフ (`data/325557_3216152_corrected_v1`) 上の H/W/A 要因分解
**正式結果**。checkpoint `45352a3`、PBS job `2406254`、8 構成 × 5 trial = 40 formal rows。
派生元 raw: `raw_data/corrected_325557/job_2406254/`。Gate W7.3C1 で独立監査済み。

主効果（既存定義, `scripts/summarize_ablation.py`）: 各因子について、他 2 因子の 4 組合せ
での per-config median 比 T(F=0)/T(F=1) の幾何平均。ばらつきは sample SD (ddof=1)。
warmup（trial invocation ごとの untimed H1W1A1 × 5）は統計に含めない。

## 修正版 325557 の H/W/A（`ablation_contributions.tsv`）

| Factor | AddOne | LeaveOneOut | **MainEffect** |
|--------|-------:|------------:|---------------:|
| H (Hybrid BFS) | 1.5159 | 1.4535 | **1.4767** |
| W (Warp-cooperative accumulation) | 1.0770 | 1.1377 | **1.1012** |
| A (Dual streams) | 1.5733 | 1.5557 | **1.5563** |

per-config median/mean/sample-SD/min/max は `ablation_per_config_stats.tsv`。
40 行完全・全 SUCCESS・NaN/Inf/0秒/OOM/timeout なし。

## 更新後の合成4グラフ集約（mixed-checkpoint, `synthetic4_aggregate.tsv`）

旧 malformed 325557 の値**だけ**を修正版へ置換し、他 3 グラフ
（benchmark_7000_41459 / benchmark_11023_62184 / 56438_300801, job 2354994）の raw
MainEffect は変更しない。

| Factor | 新 geomean | 旧 headline | 差 |
|--------|-----------:|-----------:|----|
| H | **1.6787** (本文丸め 1.679) | 1.655 | +0.0237 |
| W | **1.0661** (本文丸め 1.066) | 1.065 | +0.0011 |
| A | **1.3914** (本文丸め 1.391) | 1.396 | −0.0046 |

> **⚠ mixed-checkpoint aggregate**: 他 3 グラフは job 2354994（checkpoint 別）、
> 325557 は job 2406254（checkpoint `45352a3`）由来。4 グラフ全再測ではなく、325557 のみ
> 修正版へ差し替えた**混成集約**である。論文では必ずこの点を明記する。

## 旧 malformed 325557 の値は上書きしない

旧 325557 の MainEffect（H=1.3952, W=1.0956, A=1.5756,
`result/ablation/synthetic_2354994/ablation_contributions.tsv`）は legacy malformed
result として保持（削除・上書きしない）。current claim は本ディレクトリの修正版値を使用。

## スコープ

4 synthetic graphs、mixed checkpoints、325557 のみ修正版再測定。roadNet へ一般化しない。
