# Memory scalability — corrected 325557 (job 2404743, Series B)

修正版 325557 グラフ (`data/325557_3216152_corrected_v1`) 上の容量境界
（feasibility）**正式結果**。checkpoint `45352a3`、PBS job `2404743`。
派生元 raw: `raw_data/corrected_325557/job_2404743/{feasibility_results,oom_evidence}.tsv`。

各構成 **1 trial**。これは **feasibility 境界の確認**であり、性能比較ではない。
OOM 判定は strong-evidence-only（`cuda_oom` / `host_alloc_failure` / `kernel_oom_kill`
のみ；助言的警告や語の言及は証拠としない）。

## 正式な修正版境界（`feasibility_boundary.tsv`）

| Config | Impl | Requested batch | Observed | Runtime (s) | Exit | Cause |
|--------|------|----------------:|----------|------------:|-----:|-------|
| pure_b4096 | GPU_Opt_Pure | 4096 | SUCCESS | 65.889429 | 0 | — |
| pure_b8192 | GPU_Opt_Pure | 8192 | **CUDA OOM confirmed** | not_recorded | 1 | `cuda_oom` (`host_pure.cu:144: out of memory`) |
| um_b10240 | GPU_Opt | 10240 | SUCCESS | 238.672569 | 0 | — |
| um_b12288 | GPU_Opt | 12288 | **cgroup/host memory OOM kill** | not_recorded | 137 | SIGKILL (exit 137); CUDA-level `oom_evidence=none` |
| chunked_b16384 | GPU_Opt_Pure_Chunked | 16384 | SUCCESS | 66.598223 | 0 | tested upper limit |

## 正式表現（誤記禁止）

- 入力ファイルは **≈43.25 MiB**（45,348,105 bytes）。容量問題は **batch-dependent
  working set** であり、入力グラフではない。
- `pure_b8192` は **CUDA out-of-memory**。
- `um_b12288`（exit 137）は **host/cgroup memory limit 超過による OOM kill** であり、
  **CUDA OOM でも HBM OOM でもない**。
- Chunked は試験上限 `b16384` まで成功。これは**無制限容量を意味しない**。
- UM は無制限ではない。対象は修正版 325557 のみ、他グラフ・他 GPU へ一般化しない。

### 禁止表現（本結果では用いない）
- 「input graph が 96 GB を超えた」
- 「UM b12288 は CUDA OOM」／「UM b12288 は HBM OOM」
- 「UM は無制限」
- 「Chunked はあらゆる条件で OOM を回避」

## 旧 legacy oversubscription 掃引との関係

旧 `raw_data/memory_scalability/325557_3216152/`（oldtree, malformed 入力・
非 headline）の掃引は削除せず保持。current の容量境界主張は本ディレクトリ（修正版）を使用。
