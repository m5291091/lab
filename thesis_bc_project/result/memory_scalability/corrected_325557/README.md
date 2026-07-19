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
| um_b12288 | GPU_Opt | 12288 | **cgroup/host memory OOM kill** | not_recorded | 137 | SIGKILL (exit 137); runtime classifier `oom_evidence=none`、post-hoc PBS epilogue に直接的 cgroup OOM 記録（下記） |
| chunked_b16384 | GPU_Opt_Pure_Chunked | 16384 | SUCCESS | 66.598223 | 0 | tested upper limit |

## UM b12288 の証拠 2 層（Gate T1B1.1 で登録）

`um_b12288` の失敗根拠は **検査対象範囲の異なる 2 層**として記録する。後者が前者を
「訂正」するのではなく、**見られる範囲が違った**という関係である。

### 層 1 — Runtime classifier record（実験中・変更しない）

実験中の per-configuration classifier
（`scripts/run_corrected_325557_validation.sh` の `classify_observed`）は、
当該構成の stdout/stderr **のみ**を検査した（`"${CFG_STDERR}" "${CFG_OUTPUT}"`）。

| 項目 | 値 |
|---|---|
| `OOMEvidenceClass` | `none` |
| `runner_exit` | `137` |
| 補足 | SIGKILL 整合（`sigkill_exit_137`） |
| 記録先 | `feasibility_boundary.tsv`; `raw_data/corrected_325557/job_2404743/oom_evidence.tsv` |

この保存済み記録は **raw provenance として変更しない**。

### 層 2 — Post-hoc archive audit（ジョブ終了後の監査）

ジョブ終了時に PBS epilogue が追記した行に、**直接的な cgroup OOM 記録**が存在する。

| 項目 | 値 |
|---|---|
| `PostHocEvidenceClass` | `kernel_oom_kill` |
| `EvidenceSource` | `raw_data/corrected_325557/job_2404743/pbs_stdout.log` |
| `EvidenceLine` | `146` |
| `EvidenceSHA256` | `3c4c46680f9432b94fef79ca9344027ad77195973d075b8019379f934feb8ec5` |
| `PBSJobID` | `2404743` |
| `Configuration` | `um_b12288`（GPU_Opt, requested batch 12288） |

記録行（原文）:

```text
Cgroup memsw limit exceeded: oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=2404743.opbs,mems_allowed=0-1,oom_memcg=/pbs_jobs.service/jobid/2404743.opbs,task_memcg=/pbs_jobs.service/jobid/2404743.opbs,task=run_benchmark,pid=673064,uid=14102
```

**構成への帰属根拠**（`pbs_stdout.log` 内で独立に確認できる）:

- 行 137: `[B 4/5] um_b12288: gpu_opt batch=12288 expectation=um_failure_status_not_assumed_oom`
- 行 138: 当該 job 唯一の `Killed` 記録（`673063 Killed env ... timeout ... ${RUNNER}`）
- 行 139: `observed=RUNTIME_FAILED; outcome=EXPECTED_FAILURE_STATUS; runtime=not_recorded`
- 行 142: `=== Complete ===`（検証スクリプトの終了）
- 行 146: 上記 cgroup OOM 行（**ファイル最終行**）
- job 内に `Killed` 行は 1 本のみ、cgroup OOM 行も 1 本のみ。他構成は exit 0 または
  exit 1（`pure_b8192` の CUDA OOM）であり、取り違えの余地がない。
- PID 対応: 行 138 の `673063` は `env`/`timeout` ラッパ、行 146 の
  `task=run_benchmark,pid=673064` はその子プロセスであり整合する。

**なぜ層 1 が `none` だったか**: cgroup OOM 行は `=== Complete ===`（行 142）より後、
ファイル最終行に PBS epilogue が追記したものである。実行中の classifier は
per-configuration の stdout/stderr しか走査せず、かつ分類時点でこの行は
まだ存在しなかった。したがって層 1 の `none` は **走査範囲による artifact** であり、
誤判定ではない。

**検証**: `scripts/generate_thesis_artifacts.py` は T4 生成時に上記の path・SHA256・
行番号・job ID・構成 context・cgroup token を保存アーカイブから再検証する。
いずれかが不一致なら生成は **FAIL** し、静かに `none` へ戻ることはない。

**runtime classifier は変更しない**: `scripts/run_corrected_325557_validation.sh` と
`scripts/oom_evidence.sh` は本ゲートでは変更しない。PBS epilogue は同一 job の実行中
classifier からは原理的に参照できないため、事後証拠で runtime 実装を書き換えない。

## 正式表現（誤記禁止）

- 入力ファイルは **≈43.25 MiB**（45,348,105 bytes）。容量問題は **batch-dependent
  working set** であり、入力グラフではない。
- `pure_b8192` は **CUDA out-of-memory**。
- `um_b12288`（exit 137）は **host/cgroup memory limit 超過による OOM kill** であり、
  **CUDA OOM でも HBM OOM でもない**。直接証拠は post-hoc PBS epilogue 記録
  （`pbs_stdout.log:146`, `kernel_oom_kill`）であり、実行中 classifier の記録は
  `oom_evidence=none` である。両層を併記し、片方だけを示さない（上節参照）。
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
