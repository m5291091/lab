# Appendix C Complete Ablation Results

本付録は、H/W/A factorial ablation の全 8 構成、保存された全 formal trial、構成別記述統計、および main effect の再計算結果をまとめる。Chapter 7 の要約値を監査可能にすることが目的であり、新しい性能主張、統計的有意性、または Chapter 6 の end-to-end speedup に対する因果的な寄与分解を導入するものではない。

## C.1 Experimental Scope

測定系列を Table C.1 の 3 系列に分離する。Synthetic と email は実験時 snapshot `phase_def_block_20260710`、Corrected は commit `45352a344aaac463283a647467b790be9b45bfb8` に基づくため、同一 checkpoint の系列ではない。Synthetic raw record に含まれる旧 `325557_3216152` は malformed input 上の履歴記録であり、現在の結論には用いない。現在の Synthetic-4 aggregate は、同系列の旧 325557 を含む集約ではなく、他の 3 graph を維持したまま Corrected 系列の `325557_3216152_corrected_v1` へ差し替えた mixed-checkpoint 記述集約である。

**Table C.1: Scope and provenance of the three ablation series.**

| Series | Job | Graphs | $N_{\mathrm{trials}}$ per Configuration | Formal Rows | Checkpoint |
|:--|:--|:--|--:|--:|:--|
| Synthetic | 2354994 | benchmark_7000_41459; benchmark_11023_62184; 56438_300801; 325557_3216152 (historical malformed input only) | 5 | 160 | `phase_def_block_20260710` |
| email | 2354999 | email-EuAll | 3 | 24 | `phase_def_block_20260710` |
| Corrected | 2406254 | 325557_3216152_corrected_v1 | 5 | 40 | `45352a344aaac463283a647467b790be9b45bfb8` |

<!-- Source: result/ablation/synthetic_2354994/SOURCE.md; result/ablation/email_2354999/SOURCE.md; raw_data/corrected_325557/job_2406254/SOURCE.md; result/COVERAGE.md -->

## C.2 Configuration Definition

H、W、A は 3 つの独立した提案手法ではなく、一つの提案実行基盤の内部動作を切り替える 3 因子である。実験時 `host_ablation.cu` は 3 個の C++ template boolean を用い、8 通りの専用実体を compile time に生成する。runner が runtime に構成を選択しても、kernel 内で H/W/A を切り替える分岐を行うものではない。各因子の無効・有効状態を Table C.2 に示す。

**Table C.2: Compile-time ablation factor definitions.**

| Factor | Name | Disabled State (0) | Enabled State (1) | Target Phase |
|:--|:--|:--|:--|:--|
| H | Hybrid BFS | Top-down traversal only | Direction-optimizing top-down / bottom-up switching | Forward BFS |
| W | Warp-Cooperative Accumulation | Thread-per-vertex accumulation kernel | Warp-cooperative accumulation kernel | Backward accumulation |
| A | Dual-Stream Execution | One CUDA stream (NS=1) | Two CUDA streams (NS=2) with double buffering | Initialization / execution pipeline |

全構成を Table C.3 に列挙する。0 は Disabled、1 は Enabled を表し、H0W0A0 が baseline、H1W1A1 が full configuration である。

**Table C.3: Complete factorial configuration set.**

| Configuration | H | W | A | Role |
|:--|--:|--:|--:|:--|
| H0W0A0 | 0 | 0 | 0 | Baseline |
| H0W0A1 | 0 | 0 | 1 | Factorial cell |
| H0W1A0 | 0 | 1 | 0 | Factorial cell |
| H0W1A1 | 0 | 1 | 1 | Factorial cell |
| H1W0A0 | 1 | 0 | 0 | Factorial cell |
| H1W0A1 | 1 | 0 | 1 | Factorial cell |
| H1W1A0 | 1 | 1 | 0 | Factorial cell |
| H1W1A1 | 1 | 1 | 1 | Full configuration |

この factorial design から得る main effect は観測量である。3 値を加算可能な寄与率として扱わず、interaction を無視した因果分解とも解釈しない。

<!-- Source: code_snapshots/phase_def_block_20260710/experiments/run_ablation.cu; code_snapshots/phase_def_block_20260710/src/proposed/host_ablation.cu; commit 45352a344aaac463283a647467b790be9b45bfb8 versions of the same files -->

## C.3 Formal Trial Completeness

canonical raw TSV の header と全行を独立に読み取り、期待する graph 集合、8 構成、trial identifier、数値の有限性、成功状態、および一意キー `(Series, Graph, Configuration, Trial)` を検査した。結果を Table C.4 に示す。Synthetic の graph 数は期待どおり 4、総 formal row 数は 224 である。`Time_sec` と `GTEPS` は全行で有限かつ正であり、Corrected の全行は `RunnerExit=0`、`Status=SUCCESS` であった。

**Table C.4: Formal-row completeness and validity checks.**

| Series | Expected Rows | Observed Rows | Missing Rows | Duplicate Rows | Unknown Configurations | Non-Finite Runtime / GTEPS | Failed Rows | Result |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| Synthetic | 160 | 160 | 0 | 0 | 0 | 0 | 0 | Pass |
| email | 24 | 24 | 0 | 0 | 0 | 0 | 0 | Pass |
| Corrected | 40 | 40 | 0 | 0 | 0 | 0 | 0 | Pass |
| Total | 224 | 224 | 0 | 0 | 0 | 0 | 0 | Pass |

graph と configuration ごとの formal trial 数を Table C.5 に示す。各 cell は $N_{\mathrm{trials}}$ であり、trial identifier は Synthetic と Corrected が 1--5、email が 1--3 の連続した一意集合である。

**Table C.5: Formal trial counts for every graph and configuration.**

| Series | Graph | H0W0A0 | H0W0A1 | H0W1A0 | H0W1A1 | H1W0A0 | H1W0A1 | H1W1A0 | H1W1A1 | Trial IDs |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| Synthetic | benchmark_7000_41459 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1--5 |
| Synthetic | benchmark_11023_62184 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1--5 |
| Synthetic | 56438_300801 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1--5 |
| Synthetic | 325557_3216152 (historical malformed input only) | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1--5 |
| email | email-EuAll | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 1--3 |
| Corrected | 325557_3216152_corrected_v1 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 1--5 |

3 系列はいずれも、runner invocation ごとに `one untimed global H1W1A1 warmup` を 1 回実行する。Synthetic は 20 invocation / 20 marker / 160 formal rows、email は 3 / 3 / 24、Corrected は 5 / 5 / 40 である。warmup は `run_brandes` の formal TSV 出力経路を通らないため、以下の H1W1A1 行はすべて本試行であり、warmup の 28 実行を追加していない。

<!-- Source: raw_data/ablation/synthetic/job_2354994_20260710/ablation.log; raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv; raw_data/ablation/email-EuAll/job_2354999_20260710/ablation.log; raw_data/ablation/email-EuAll/job_2354999_20260710/ablation_results.tsv; raw_data/corrected_325557/job_2406254/stderr/ablation.stderr.log; raw_data/corrected_325557/job_2406254/ablation_results.tsv; code_snapshots/phase_def_block_20260710/experiments/run_ablation.cu -->

## C.4 Synthetic-Graph Results

本節の表は Synthetic job 2354994 の canonical raw TSV を転記する。最初の 3 graph は現在の Synthetic-4 descriptive aggregate に継続使用する。4 番目の旧 `325557_3216152` は 160-row archive の完全性を保つため履歴表として収録するが、current conclusion、current corrected result、および current Synthetic-4 aggregate から除外する。

### C.4.1 benchmark_7000_41459

**Table C.6: All formal ablation trials on benchmark_7000_41459.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| benchmark_7000_41459 | 1 | 0 | 0 | 0 | 0.056044 | 5.1784 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 0 | 0 | 1 | 0.043375 | 6.6907 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 0 | 1 | 0 | 0.047983 | 6.0483 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 0 | 1 | 1 | 0.039475 | 7.3518 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 1 | 0 | 0 | 0.037516 | 7.7357 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 1 | 0 | 1 | 0.030033 | 9.6631 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 1 | 1 | 0 | 0.030794 | 9.4243 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 1 | 1 | 1 | 1 | 0.024220 | 11.9823 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 0 | 0 | 0 | 0.055944 | 5.1876 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 0 | 0 | 1 | 0.047010 | 6.1735 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 0 | 1 | 0 | 0.047848 | 6.0653 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 0 | 1 | 1 | 0.040408 | 7.1821 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 1 | 0 | 0 | 0.037555 | 7.7277 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 1 | 0 | 1 | 0.030664 | 9.4644 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 1 | 1 | 0 | 0.030817 | 9.4173 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 2 | 1 | 1 | 1 | 0.024701 | 11.7492 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 0 | 0 | 0 | 0.056000 | 5.1824 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 0 | 0 | 1 | 0.043951 | 6.6031 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 0 | 1 | 0 | 0.047935 | 6.0543 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 0 | 1 | 1 | 0.040490 | 7.1675 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 1 | 0 | 0 | 0.037762 | 7.6853 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 1 | 0 | 1 | 0.030730 | 9.4438 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 1 | 1 | 0 | 0.030817 | 9.4173 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 3 | 1 | 1 | 1 | 0.025249 | 11.4938 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 0 | 0 | 0 | 0.055571 | 5.2224 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 0 | 0 | 1 | 0.044580 | 6.5099 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 0 | 1 | 0 | 0.047647 | 6.0909 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 0 | 1 | 1 | 0.041129 | 7.0562 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 1 | 0 | 0 | 0.037430 | 7.7536 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 1 | 0 | 1 | 0.029892 | 9.7088 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 1 | 1 | 0 | 0.030631 | 9.4744 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 4 | 1 | 1 | 1 | 0.024660 | 11.7684 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 0 | 0 | 0 | 0.055777 | 5.2031 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 0 | 0 | 1 | 0.044221 | 6.5628 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 0 | 1 | 0 | 0.047784 | 6.0734 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 0 | 1 | 1 | 0.040967 | 7.0840 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 1 | 0 | 0 | 0.037290 | 7.7826 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 1 | 0 | 1 | 0.030251 | 9.5934 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 1 | 1 | 0 | 0.030506 | 9.5133 | 2354994 | `phase_def_block_20260710` |
| benchmark_7000_41459 | 5 | 1 | 1 | 1 | 0.024595 | 11.7996 | 2354994 | `phase_def_block_20260710` |

### C.4.2 benchmark_11023_62184

**Table C.7: All formal ablation trials on benchmark_11023_62184.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| benchmark_11023_62184 | 1 | 0 | 0 | 0 | 0.155606 | 4.4051 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 0 | 0 | 1 | 0.096309 | 7.1173 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 0 | 1 | 0 | 0.153860 | 4.4550 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 0 | 1 | 1 | 0.094285 | 7.2700 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 1 | 0 | 0 | 0.083936 | 8.1664 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 1 | 0 | 1 | 0.054717 | 12.5274 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 1 | 1 | 0 | 0.083598 | 8.1994 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 1 | 1 | 1 | 1 | 0.054758 | 12.5178 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 0 | 0 | 0 | 0.157071 | 4.3640 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 0 | 0 | 1 | 0.094649 | 7.2421 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 0 | 1 | 0 | 0.151187 | 4.5338 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 0 | 1 | 1 | 0.096477 | 7.1049 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 1 | 0 | 0 | 0.084188 | 8.1419 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 1 | 0 | 1 | 0.054144 | 12.6598 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 1 | 1 | 0 | 0.083897 | 8.1702 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 2 | 1 | 1 | 1 | 0.054671 | 12.5377 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 0 | 0 | 0 | 0.153957 | 4.4523 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 0 | 0 | 1 | 0.095471 | 7.1797 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 0 | 1 | 0 | 0.153494 | 4.4657 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 0 | 1 | 1 | 0.093254 | 7.3504 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 1 | 0 | 0 | 0.083900 | 8.1699 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 1 | 0 | 1 | 0.054673 | 12.5372 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 1 | 1 | 0 | 0.084048 | 8.1555 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 3 | 1 | 1 | 1 | 0.055756 | 12.2937 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 0 | 0 | 0 | 0.156449 | 4.3813 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 0 | 0 | 1 | 0.095976 | 7.1420 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 0 | 1 | 0 | 0.152473 | 4.4956 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 0 | 1 | 1 | 0.094130 | 7.2820 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 1 | 0 | 0 | 0.084426 | 8.1190 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 1 | 0 | 1 | 0.054575 | 12.5598 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 1 | 1 | 0 | 0.084350 | 8.1263 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 4 | 1 | 1 | 1 | 0.055315 | 12.3917 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 0 | 0 | 0 | 0.155951 | 4.3953 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 0 | 0 | 1 | 0.097498 | 7.0305 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 0 | 1 | 0 | 0.153300 | 4.4713 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 0 | 1 | 1 | 0.092800 | 7.3863 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 1 | 0 | 0 | 0.084500 | 8.1119 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 1 | 0 | 1 | 0.055738 | 12.2978 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 1 | 1 | 0 | 0.084560 | 8.1062 | 2354994 | `phase_def_block_20260710` |
| benchmark_11023_62184 | 5 | 1 | 1 | 1 | 0.055550 | 12.3394 | 2354994 | `phase_def_block_20260710` |

### C.4.3 56438_300801

**Table C.8: All formal ablation trials on 56438_300801.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| 56438_300801 | 1 | 0 | 0 | 0 | 3.976097 | 4.2697 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 0 | 0 | 1 | 3.277375 | 5.1799 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 0 | 1 | 0 | 3.963516 | 4.2832 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 0 | 1 | 1 | 3.227384 | 5.2602 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 1 | 0 | 0 | 2.012951 | 8.4337 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 1 | 0 | 1 | 1.620168 | 10.4783 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 1 | 1 | 0 | 2.083709 | 8.1473 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 1 | 1 | 1 | 1 | 1.647959 | 10.3016 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 0 | 0 | 0 | 3.975052 | 4.2708 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 0 | 0 | 1 | 3.278142 | 5.1787 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 0 | 1 | 0 | 3.968872 | 4.2774 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 0 | 1 | 1 | 3.221999 | 5.2690 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 1 | 0 | 0 | 2.011607 | 8.4393 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 1 | 0 | 1 | 1.623107 | 10.4593 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 1 | 1 | 0 | 2.081964 | 8.1541 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 2 | 1 | 1 | 1 | 1.646102 | 10.3132 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 0 | 0 | 0 | 3.979156 | 4.2664 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 0 | 0 | 1 | 3.276283 | 5.1817 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 0 | 1 | 0 | 3.971102 | 4.2750 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 0 | 1 | 1 | 3.217845 | 5.2758 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 1 | 0 | 0 | 2.012500 | 8.4356 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 1 | 0 | 1 | 1.619469 | 10.4828 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 1 | 1 | 0 | 2.083729 | 8.1472 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 3 | 1 | 1 | 1 | 1.651386 | 10.2802 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 0 | 0 | 0 | 3.977577 | 4.2681 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 0 | 0 | 1 | 3.277497 | 5.1797 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 0 | 1 | 0 | 3.965917 | 4.2806 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 0 | 1 | 1 | 3.225628 | 5.2630 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 1 | 0 | 0 | 2.012922 | 8.4338 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 1 | 0 | 1 | 1.617956 | 10.4926 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 1 | 1 | 0 | 2.081052 | 8.1577 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 4 | 1 | 1 | 1 | 1.649608 | 10.2913 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 0 | 0 | 0 | 3.986843 | 4.2582 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 0 | 0 | 1 | 3.276129 | 5.1819 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 0 | 1 | 0 | 3.971307 | 4.2748 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 0 | 1 | 1 | 3.224815 | 5.2644 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 1 | 0 | 0 | 2.009440 | 8.4484 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 1 | 0 | 1 | 1.619542 | 10.4823 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 1 | 1 | 0 | 2.085697 | 8.1395 | 2354994 | `phase_def_block_20260710` |
| 56438_300801 | 5 | 1 | 1 | 1 | 1.648082 | 10.3008 | 2354994 | `phase_def_block_20260710` |

### C.4.4 Historical 325557_3216152 Record

旧 `325557_3216152` は 1-based vertex ID を 0-based として格納し、隣接要素が 7 個不足した malformed input である。Table C.9 は formal archive 224 行を欠落なく提示する目的に限って収録し、修正版との性能改善・劣化比較には用いない。

**Table C.9: Historical formal trials on the malformed 325557_3216152 input, excluded from current conclusions.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| 325557_3216152 | 1 | 0 | 0 | 0 | 176.499175 | 5.9323 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 0 | 0 | 1 | 112.107097 | 9.3396 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 0 | 1 | 0 | 163.700166 | 6.3961 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 0 | 1 | 1 | 100.931930 | 10.3737 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 1 | 0 | 0 | 124.594405 | 8.4036 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 1 | 0 | 1 | 81.684899 | 12.8180 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 1 | 1 | 0 | 115.602630 | 9.0572 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 1 | 1 | 1 | 1 | 73.160699 | 14.3115 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 0 | 0 | 0 | 176.273764 | 5.9399 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 0 | 0 | 1 | 112.121695 | 9.3384 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 0 | 1 | 0 | 163.758087 | 6.3938 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 0 | 1 | 1 | 100.747634 | 10.3927 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 1 | 0 | 0 | 124.550105 | 8.4066 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 1 | 0 | 1 | 81.702279 | 12.8153 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 1 | 1 | 0 | 115.679890 | 9.0512 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 2 | 1 | 1 | 1 | 73.361727 | 14.2723 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 0 | 0 | 0 | 176.377429 | 5.9364 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 0 | 0 | 1 | 112.126686 | 9.3380 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 0 | 1 | 0 | 163.775986 | 6.3931 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 0 | 1 | 1 | 100.774364 | 10.3900 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 1 | 0 | 0 | 124.607520 | 8.4027 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 1 | 0 | 1 | 81.658002 | 12.8223 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 1 | 1 | 0 | 115.627324 | 9.0553 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 3 | 1 | 1 | 1 | 73.171411 | 14.3094 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 0 | 0 | 0 | 176.371341 | 5.9366 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 0 | 0 | 1 | 112.140839 | 9.3368 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 0 | 1 | 0 | 163.723856 | 6.3952 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 0 | 1 | 1 | 100.920807 | 10.3749 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 1 | 0 | 0 | 124.580846 | 8.4045 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 1 | 0 | 1 | 81.911587 | 12.7826 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 1 | 1 | 0 | 115.678980 | 9.0513 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 4 | 1 | 1 | 1 | 73.110497 | 14.3213 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 0 | 0 | 0 | 176.369826 | 5.9366 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 0 | 0 | 1 | 112.345169 | 9.3199 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 0 | 1 | 0 | 163.865469 | 6.3896 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 0 | 1 | 1 | 100.784210 | 10.3889 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 1 | 0 | 0 | 124.448354 | 8.4135 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 1 | 0 | 1 | 81.685797 | 12.8179 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 1 | 1 | 0 | 115.678792 | 9.0513 | 2354994 | `phase_def_block_20260710` |
| 325557_3216152 | 5 | 1 | 1 | 1 | 73.225436 | 14.2989 | 2354994 | `phase_def_block_20260710` |

<!-- Source: raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv; SHA256 ef0f787086d4dcfb4bba8181aa248ddecef74bda2759d2456af563e5bb5193eb -->

## C.5 email-EuAll Results

email 系列は job 2354999、各構成 $N_{\mathrm{trials}}=3$ であり、Synthetic および Corrected と混在集計しない。全 24 formal trial を Table C.10 に示す。

**Table C.10: All formal ablation trials on email-EuAll.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| email-EuAll | 1 | 0 | 0 | 0 | 72.107592 | 1.3395 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 0 | 0 | 1 | 42.495161 | 2.2730 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 0 | 1 | 0 | 73.294355 | 1.3178 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 0 | 1 | 1 | 42.899488 | 2.2516 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 1 | 0 | 0 | 50.350710 | 1.9184 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 1 | 0 | 1 | 28.750180 | 3.3597 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 1 | 1 | 0 | 52.481184 | 1.8405 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 1 | 1 | 1 | 1 | 30.421410 | 3.1751 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 0 | 0 | 0 | 72.070743 | 1.3402 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 0 | 0 | 1 | 42.363314 | 2.2801 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 0 | 1 | 0 | 73.243350 | 1.3188 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 0 | 1 | 1 | 42.881928 | 2.2525 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 1 | 0 | 0 | 50.340890 | 1.9187 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 1 | 0 | 1 | 28.753547 | 3.3593 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 1 | 1 | 0 | 52.481763 | 1.8405 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 2 | 1 | 1 | 1 | 30.254042 | 3.1927 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 0 | 0 | 0 | 72.001806 | 1.3415 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 0 | 0 | 1 | 42.535161 | 2.2708 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 0 | 1 | 0 | 73.264358 | 1.3184 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 0 | 1 | 1 | 42.893899 | 2.2519 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 1 | 0 | 0 | 50.289317 | 1.9207 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 1 | 0 | 1 | 28.765522 | 3.3579 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 1 | 1 | 0 | 52.524467 | 1.8390 | 2354999 | `phase_def_block_20260710` |
| email-EuAll | 3 | 1 | 1 | 1 | 30.424089 | 3.1748 | 2354999 | `phase_def_block_20260710` |

<!-- Source: raw_data/ablation/email-EuAll/job_2354999_20260710/ablation_results.tsv; SHA256 77fd81068e345888b98368e4f88162b4a75780e7f25c6786deb7bd6a62b0c45a -->

## C.6 Corrected 325557 Results

Corrected 系列は job 2406254 / checkpoint `45352a3` により、`325557_3216152_corrected_v1` を各構成 $N_{\mathrm{trials}}=5$ で再測定した独立系列である。全 40 行は成功し、旧 malformed input の行を混入していない。

**Table C.11: All formal ablation trials on 325557_3216152_corrected_v1.**

| Graph | Trial | H | W | A | Runtime (s) | GTEPS | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|--:|:--|:--|
| 325557_3216152_corrected_v1 | 1 | 0 | 0 | 0 | 176.358035 | 5.9370 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 0 | 0 | 1 | 112.073926 | 9.3424 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 0 | 1 | 0 | 163.735910 | 6.3947 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 0 | 1 | 1 | 100.765035 | 10.3909 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 1 | 0 | 0 | 116.418547 | 8.9938 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 1 | 0 | 1 | 78.896392 | 13.2711 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 1 | 1 | 0 | 107.844926 | 9.7088 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 1 | 1 | 1 | 1 | 69.290090 | 15.1110 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 0 | 0 | 0 | 176.290382 | 5.9393 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 0 | 0 | 1 | 112.090565 | 9.3410 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 0 | 1 | 0 | 163.769977 | 6.3934 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 0 | 1 | 1 | 100.839434 | 10.3832 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 1 | 0 | 0 | 116.288659 | 9.0038 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 1 | 0 | 1 | 78.868771 | 13.2757 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 1 | 1 | 0 | 107.840803 | 9.7091 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 2 | 1 | 1 | 1 | 69.376091 | 15.0922 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 0 | 0 | 0 | 176.301738 | 5.9389 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 0 | 0 | 1 | 112.087905 | 9.3412 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 0 | 1 | 0 | 163.729329 | 6.3949 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 0 | 1 | 1 | 100.763793 | 10.3910 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 1 | 0 | 0 | 116.271917 | 9.0051 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 1 | 0 | 1 | 78.879567 | 13.2739 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 1 | 1 | 0 | 107.809662 | 9.7119 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 3 | 1 | 1 | 1 | 69.324039 | 15.1036 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 0 | 0 | 0 | 176.350609 | 5.9373 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 0 | 0 | 1 | 112.138071 | 9.3371 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 0 | 1 | 0 | 163.799227 | 6.3922 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 0 | 1 | 1 | 100.779244 | 10.3894 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 1 | 0 | 0 | 116.330316 | 9.0006 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 1 | 0 | 1 | 78.865354 | 13.2763 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 1 | 1 | 0 | 107.847937 | 9.7085 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 4 | 1 | 1 | 1 | 69.376184 | 15.0922 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 0 | 0 | 0 | 176.413246 | 5.9352 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 0 | 0 | 1 | 112.144524 | 9.3365 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 0 | 1 | 0 | 163.738336 | 6.3946 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 0 | 1 | 1 | 100.751330 | 10.3923 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 1 | 0 | 0 | 116.363404 | 8.9980 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 1 | 0 | 1 | 78.836244 | 13.2812 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 1 | 1 | 0 | 107.893349 | 9.7044 | 2406254 | `45352a3` |
| 325557_3216152_corrected_v1 | 5 | 1 | 1 | 1 | 69.299777 | 15.1089 | 2406254 | `45352a3` |

<!-- Source: raw_data/corrected_325557/job_2406254/ablation_results.tsv; SHA256 ef96297cf4cf62addac0664636f79125e221b6fe8625973aab734fc16e36df04; full checkpoint 45352a344aaac463283a647467b790be9b45bfb8 -->

## C.7 Configuration-Level Statistics

各 graph と構成について、raw trial から median、mean、標本標準偏差 $s_T$（ddof=1）、min、max、および trial-level GTEPS の median を再計算した。単一の最速 trial を代表値には用いない。GTEPS は Chapter 5 と同じ $|V||E|/(T\times10^9)$ で各 trial に記録された値を用いる。系列間は混在集計しない。

### C.7.1 Current Synthetic Graphs

**Table C.12: Per-configuration statistics for the three retained synthetic graphs.**

| Graph | Configuration | $N_{\mathrm{trials}}$ | Median (s) | Mean (s) | $s_T$ (s) | Min (s) | Max (s) | Median GTEPS |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| benchmark_7000_41459 | H0W0A0 | 5 | 0.055944 | 0.055867 | 0.000194 | 0.055571 | 0.056044 | 5.1876 |
| benchmark_7000_41459 | H0W0A1 | 5 | 0.044221 | 0.044627 | 0.001403 | 0.043375 | 0.047010 | 6.5628 |
| benchmark_7000_41459 | H0W1A0 | 5 | 0.047848 | 0.047839 | 0.000132 | 0.047647 | 0.047983 | 6.0653 |
| benchmark_7000_41459 | H0W1A1 | 5 | 0.040490 | 0.040494 | 0.000647 | 0.039475 | 0.041129 | 7.1675 |
| benchmark_7000_41459 | H1W0A0 | 5 | 0.037516 | 0.037511 | 0.000173 | 0.037290 | 0.037762 | 7.7357 |
| benchmark_7000_41459 | H1W0A1 | 5 | 0.030251 | 0.030314 | 0.000373 | 0.029892 | 0.030730 | 9.5934 |
| benchmark_7000_41459 | H1W1A0 | 5 | 0.030794 | 0.030713 | 0.000139 | 0.030506 | 0.030817 | 9.4243 |
| benchmark_7000_41459 | H1W1A1 | 5 | 0.024660 | 0.024685 | 0.000369 | 0.024220 | 0.025249 | 11.7684 |
| benchmark_11023_62184 | H0W0A0 | 5 | 0.155951 | 0.155807 | 0.001172 | 0.153957 | 0.157071 | 4.3953 |
| benchmark_11023_62184 | H0W0A1 | 5 | 0.095976 | 0.095981 | 0.001054 | 0.094649 | 0.097498 | 7.1420 |
| benchmark_11023_62184 | H0W1A0 | 5 | 0.153300 | 0.152863 | 0.001066 | 0.151187 | 0.153860 | 4.4713 |
| benchmark_11023_62184 | H0W1A1 | 5 | 0.094130 | 0.094189 | 0.001419 | 0.092800 | 0.096477 | 7.2820 |
| benchmark_11023_62184 | H1W0A0 | 5 | 0.084188 | 0.084190 | 0.000274 | 0.083900 | 0.084500 | 8.1419 |
| benchmark_11023_62184 | H1W0A1 | 5 | 0.054673 | 0.054769 | 0.000587 | 0.054144 | 0.055738 | 12.5372 |
| benchmark_11023_62184 | H1W1A0 | 5 | 0.084048 | 0.084091 | 0.000377 | 0.083598 | 0.084560 | 8.1555 |
| benchmark_11023_62184 | H1W1A1 | 5 | 0.055315 | 0.055210 | 0.000479 | 0.054671 | 0.055756 | 12.3917 |
| 56438_300801 | H0W0A0 | 5 | 3.977577 | 3.978945 | 0.004679 | 3.975052 | 3.986843 | 4.2681 |
| 56438_300801 | H0W0A1 | 5 | 3.277375 | 3.277085 | 0.000856 | 3.276129 | 3.278142 | 5.1799 |
| 56438_300801 | H0W1A0 | 5 | 3.968872 | 3.968143 | 0.003379 | 3.963516 | 3.971307 | 4.2774 |
| 56438_300801 | H0W1A1 | 5 | 3.224815 | 3.223534 | 0.003727 | 3.217845 | 3.227384 | 5.2644 |
| 56438_300801 | H1W0A0 | 5 | 2.012500 | 2.011884 | 0.001470 | 2.009440 | 2.012951 | 8.4356 |
| 56438_300801 | H1W0A1 | 5 | 1.619542 | 1.620048 | 0.001893 | 1.617956 | 1.623107 | 10.4823 |
| 56438_300801 | H1W1A0 | 5 | 2.083709 | 2.083230 | 0.001797 | 2.081052 | 2.085697 | 8.1473 |
| 56438_300801 | H1W1A1 | 5 | 1.648082 | 1.648627 | 0.001981 | 1.646102 | 1.651386 | 10.3008 |

### C.7.2 email-EuAll

**Table C.13: Per-configuration statistics for email-EuAll.**

| Graph | Configuration | $N_{\mathrm{trials}}$ | Median (s) | Mean (s) | $s_T$ (s) | Min (s) | Max (s) | Median GTEPS |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| email-EuAll | H0W0A0 | 3 | 72.070743 | 72.060047 | 0.053698 | 72.001806 | 72.107592 | 1.3402 |
| email-EuAll | H0W0A1 | 3 | 42.495161 | 42.464545 | 0.089921 | 42.363314 | 42.535161 | 2.2730 |
| email-EuAll | H0W1A0 | 3 | 73.264358 | 73.267354 | 0.025634 | 73.243350 | 73.294355 | 1.3184 |
| email-EuAll | H0W1A1 | 3 | 42.893899 | 42.891772 | 0.008971 | 42.881928 | 42.899488 | 2.2519 |
| email-EuAll | H1W0A0 | 3 | 50.340890 | 50.326972 | 0.032978 | 50.289317 | 50.350710 | 1.9187 |
| email-EuAll | H1W0A1 | 3 | 28.753547 | 28.756416 | 0.008063 | 28.750180 | 28.765522 | 3.3593 |
| email-EuAll | H1W1A0 | 3 | 52.481763 | 52.495805 | 0.024824 | 52.481184 | 52.524467 | 1.8405 |
| email-EuAll | H1W1A1 | 3 | 30.421410 | 30.366514 | 0.097413 | 30.254042 | 30.424089 | 3.1751 |

### C.7.3 Corrected 325557

**Table C.14: Per-configuration statistics for corrected 325557.**

| Graph | Configuration | $N_{\mathrm{trials}}$ | Median (s) | Mean (s) | $s_T$ (s) | Min (s) | Max (s) | Median GTEPS |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 325557_3216152_corrected_v1 | H0W0A0 | 5 | 176.350609 | 176.342802 | 0.049218 | 176.290382 | 176.413246 | 5.9373 |
| 325557_3216152_corrected_v1 | H0W0A1 | 5 | 112.090565 | 112.106998 | 0.032024 | 112.073926 | 112.144524 | 9.3410 |
| 325557_3216152_corrected_v1 | H0W1A0 | 5 | 163.738336 | 163.754556 | 0.029498 | 163.729329 | 163.799227 | 6.3946 |
| 325557_3216152_corrected_v1 | H0W1A1 | 5 | 100.765035 | 100.779767 | 0.034790 | 100.751330 | 100.839434 | 10.3909 |
| 325557_3216152_corrected_v1 | H1W0A0 | 5 | 116.330316 | 116.334569 | 0.059023 | 116.271917 | 116.418547 | 9.0006 |
| 325557_3216152_corrected_v1 | H1W0A1 | 5 | 78.868771 | 78.869266 | 0.022068 | 78.836244 | 78.896392 | 13.2757 |
| 325557_3216152_corrected_v1 | H1W1A0 | 5 | 107.844926 | 107.847335 | 0.029939 | 107.809662 | 107.893349 | 9.7088 |
| 325557_3216152_corrected_v1 | H1W1A1 | 5 | 69.324039 | 69.333236 | 0.041069 | 69.290090 | 69.376184 | 15.1036 |

### C.7.4 Historical Malformed-Input Statistics

Table C.15 は旧 raw archive の記述統計であり、current corrected result ではない。旧値と修正版値の差を入力修復の因果効果として解釈しない。

**Table C.15: Historical per-configuration statistics for the malformed 325557_3216152 input, excluded from current conclusions.**

| Graph | Configuration | $N_{\mathrm{trials}}$ | Median (s) | Mean (s) | $s_T$ (s) | Min (s) | Max (s) | Median GTEPS |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 325557_3216152 | H0W0A0 | 5 | 176.371341 | 176.378307 | 0.080093 | 176.273764 | 176.499175 | 5.9366 |
| 325557_3216152 | H0W0A1 | 5 | 112.126686 | 112.168297 | 0.099607 | 112.107097 | 112.345169 | 9.3380 |
| 325557_3216152 | H0W1A0 | 5 | 163.758087 | 163.764713 | 0.063558 | 163.700166 | 163.865469 | 6.3938 |
| 325557_3216152 | H0W1A1 | 5 | 100.784210 | 100.831789 | 0.087458 | 100.747634 | 100.931930 | 10.3889 |
| 325557_3216152 | H1W0A0 | 5 | 124.580846 | 124.556246 | 0.063970 | 124.448354 | 124.607520 | 8.4045 |
| 325557_3216152 | H1W0A1 | 5 | 81.685797 | 81.728513 | 0.103565 | 81.658002 | 81.911587 | 12.8179 |
| 325557_3216152 | H1W1A0 | 5 | 115.678792 | 115.653523 | 0.036257 | 115.602630 | 115.679890 | 9.0513 |
| 325557_3216152 | H1W1A1 | 5 | 73.171411 | 73.205954 | 0.096174 | 73.110497 | 73.361727 | 14.3094 |

## C.8 Main-Effect Calculation

実験時 generator `summarize_ablation.py` の実装では、まず各 graph・構成の formal trial runtime の median $T_g^{\mathrm{med}}$ を求める。因子 $F\in\{\mathrm{H},\mathrm{W},\mathrm{A}\}$ と残り 2 因子 $(G_1,G_2)$ に対し、graph 内 main effect は

$$
\mathrm{ME}_g(F)=\left(\prod_{(b_1,b_2)\in\{0,1\}^2}
\frac{T_g^{\mathrm{med}}(F{=}0,G_1{=}b_1,G_2{=}b_2)}
{T_g^{\mathrm{med}}(F{=}1,G_1{=}b_1,G_2{=}b_2)}\right)^{1/4}
$$

と計算する。すなわち、対応する 4 個の `factor OFF median / factor ON median` 比の算術平均ではなく幾何平均である。比が 1 より大きいとき、当該因子の有効化時に median runtime が短縮したことを表す。

generator の loader は `FAIL`、`TIMEOUT`、空欄、非数値を読み飛ばし、比の分母が 0 または値が欠ける場合を `None` とし、geometric mean から `None` と非正値を除外する。本付録ではこの寛容な挙動に依存せず、C.3 の完全性検査で全 cell の trial 集合、有限かつ正の runtime/GTEPS、成功状態を先に要求した。そのため、以下の main effect は各 graph で 4 比をすべて使用している。

独立再計算値と、小数第 4 位まで保存された正式 contribution 値との照合を Table C.16 に示す。旧 malformed graph の contribution は current main-effect 表に含めない。

**Table C.16: Current per-graph main effects, comparing independent recalculation with formal values.**

| Graph | Factor | Independent Value (Unrounded) | Formal Value (4 d.p.) | $N_{\mathrm{trials}}$ per Configuration | Checkpoint | Match |
|:--|:--|--:|--:|--:|:--|:--|
| benchmark_7000_41459 | H | 1.5356581539 | 1.5357 | 5 | `phase_def_block_20260710` | Yes |
| benchmark_7000_41459 | W | 1.1753491521 | 1.1753 | 5 | `phase_def_block_20260710` | Yes |
| benchmark_7000_41459 | A | 1.2335242686 | 1.2335 | 5 | `phase_def_block_20260710` | Yes |
| benchmark_11023_62184 | H | 1.7824071122 | 1.7824 | 5 | `phase_def_block_20260710` | Yes |
| benchmark_11023_62184 | W | 1.0066612430 | 1.0067 | 5 | `phase_def_block_20260710` | Yes |
| benchmark_11023_62184 | A | 1.5774308935 | 1.5774 | 5 | `phase_def_block_20260710` | Yes |
| 56438_300801 | H | 1.9649122517 | 1.9649 | 5 | `phase_def_block_20260710` | Yes |
| 56438_300801 | W | 0.9915651711 | 0.9916 | 5 | `phase_def_block_20260710` | Yes |
| 56438_300801 | A | 1.2376965147 | 1.2377 | 5 | `phase_def_block_20260710` | Yes |
| email-EuAll | H | 1.4285540284 | 1.4286 | 3 | `phase_def_block_20260710` | Yes |
| email-EuAll | W | 0.9695242840 | 0.9695 | 3 | `phase_def_block_20260710` | Yes |
| email-EuAll | A | 1.7198628708 | 1.7199 | 3 | `phase_def_block_20260710` | Yes |
| 325557_3216152_corrected_v1 | H | 1.4766622574 | 1.4767 | 5 | `45352a3` | Yes |
| 325557_3216152_corrected_v1 | W | 1.1011590412 | 1.1012 | 5 | `45352a3` | Yes |
| 325557_3216152_corrected_v1 | A | 1.5562810447 | 1.5563 | 5 | `45352a3` | Yes |

current Synthetic-4 aggregate は、Table C.18 に示す 4 graph の graph-level main effect をさらに幾何平均して

$$
\mathrm{ME}_{\mathrm{Synthetic-4}}(F)=
\left(\prod_{g\in\mathcal{G}_{\mathrm{current}}}\mathrm{ME}_g(F)\right)^{1/4}
$$

とする。独立再計算と正式値の照合結果を Table C.17 に示す。

**Table C.17: Current Synthetic-4 mixed-checkpoint aggregate main effects.**

| Factor | Independent Value (Unrounded) | Formal Value (4 d.p.) | Chapter 7 Display | Aggregation | Match |
|:--|--:|--:|--:|:--|:--|
| H | 1.6787323050 | 1.6787 | 1.679 | Geometric mean across four graph-level main effects | Yes |
| W | 1.0661182797 | 1.0661 | 1.066 | Geometric mean across four graph-level main effects | Yes |
| A | 1.3913937847 | 1.3914 | 1.391 | Geometric mean across four graph-level main effects | Yes |

generator が出力する `AddOne` と `LeaveOneOut` の相対差 `InteractionRel` は interaction の兆候を点検する補助量であり、正式な interaction term の推定ではない。本付録でも interaction が存在しないとは断定せず、main effect を相互独立な因果効果や加算可能な百分率として扱わない。

<!-- Source: code_snapshots/phase_def_block_20260710/scripts/summarize_ablation.py; result/ablation/synthetic_2354994/ablation_contributions.tsv; result/ablation/email_2354999/ablation_contributions.tsv; result/ablation/corrected_325557/ablation_contributions.tsv; result/ablation/corrected_325557/synthetic4_aggregate.tsv -->

## C.9 Cross-Series Interpretation

current result に限定すると、Corrected 325557 では H=1.4767 と A=1.5563 の観測 main effect が W=1.1012 より大きい。email-EuAll では A=1.7199、H=1.4286 に対し W=0.9695 であり、因子ごとの効果は graph によって異なる。特に W は 0.9695 から 1.1753 の範囲にあり、graph-dependent である。これらは測定した graph、GPU、batch、trial 数に限定した記述であり、H/A が常に高速化する、W が不要である、または未測定 graph でも同じであるとは結論しない。有意差検定も行っていない。

### C.9.1 Mixed-Checkpoint Aggregate

Synthetic-4 aggregate の構成要素を Table C.18 に示す。Corrected 325557 だけが別 checkpoint であり、4 graph の same-checkpoint controlled comparison ではない。checkpoint 差の影響を graph 差や因子効果から分離できないため、この aggregate は横断的な記述要約であり、因果効果の厳密な推定ではない。

**Table C.18: Membership and provenance of the current Synthetic-4 mixed-checkpoint aggregate.**

| Graph | Job | Checkpoint | Role in Current Aggregate |
|:--|:--|:--|:--|
| benchmark_7000_41459 | 2354994 | `phase_def_block_20260710` | Included, unchanged raw result |
| benchmark_11023_62184 | 2354994 | `phase_def_block_20260710` | Included, unchanged raw result |
| 56438_300801 | 2354994 | `phase_def_block_20260710` | Included, unchanged raw result |
| 325557_3216152_corrected_v1 | 2406254 | `45352a344aaac463283a647467b790be9b45bfb8` | Included, corrected replacement |

### C.9.2 Historical Malformed-Input Result

旧 `325557_3216152` に対する job 2354994 の ablation は削除せず、malformed legacy input 上の historical evidence として raw、派生 summary/contribution、failure/provenance 記録を保存する。ただし現在の corrected result と current main-effect aggregate には混入させず、`325557_3216152_corrected_v1` の job 2406254 に置き換えた。Table C.9 と Table C.15 は archive の全 formal row とその記述統計を監査可能にするための履歴表であり、旧値と修正版値を並べた性能改善・劣化の評価には使用しない。保存場所は `raw_data/ablation/synthetic/job_2354994_20260710/`、`result/ablation/synthetic_2354994/`、`failure/early_terminated/memory_correctness_2368398/`、`failure/failed/oom/memory_correctness_2368269/`、および `result/provenance/GRAPH_325557_INTEGRITY_AUDIT.md` である。

## C.10 Recalculation and Validation

独立再計算は標準ライブラリだけを用い、canonical raw TSV を毎回新たに読み込んで、完全性、構成別 median/mean/$s_T$/min/max/median GTEPS、graph 別 main effect、および current Synthetic-4 aggregate を計算した。同一処理を 2 回実行した出力は byte-identical で、両方の SHA256 は `e77716fb31b76228824d9bfe04a9ef3fe35c6cf31a181ca04ff416d88cf384a5` であった。

保存された実験時 `summarize_ablation.py` は母集団標準偏差 (population SD) を用いており、正式要約を byte-identical には再現しない。現在の derivation スクリプト (`scripts/summarize_ablation.py`) を用いて Synthetic と email をそれぞれ 2 回再生成した結果、各回の `ablation_summary.md` と `ablation_contributions.tsv` は相互に byte-identical であり、repository の正式成果物は現在の derivation スクリプトによって再現される。正式要約の数値規約は標本標準偏差 (sample SD, ddof=1) である。すべての生 trial データと付録 C の正式な数値は変更されていない。この不一致は generator の来歴と標準偏差の規約に関するものであり、実験結果には影響しない。保存された証拠を超えて、正式要約の歴史的な作成者を推論するべきではない。Corrected は独立計算した構成別統計を正式 `ablation_per_config_stats.tsv` と丸め精度内で照合し、main effect と aggregate は正式 contribution/aggregate TSV の小数第 4 位と一致した。

canonical source の役割を Table C.19 に整理する。raw trial 値は raw TSV、実装条件は各系列の実験時コード、正式な丸め済み main effect は contribution/aggregate TSV を正本とした。

**Table C.19: Canonical sources used for this appendix.**

| Source Class | Canonical Source | Use |
|:--|:--|:--|
| Thesis plan and method | `docs/thesis/writing/plan.md`; `docs/thesis/writing/japanese/05_experimental_methodology.md`; `docs/thesis/writing/japanese/appendix_a_experimental_parameters.md` | Scope, terminology, aggregation policy |
| Chapter alignment | `docs/thesis/writing/japanese/07_ablation_and_kernel_analysis.md`; `docs/thesis/writing/japanese/10_discussion.md` | Display values and interpretation limits |
| Synthetic raw | `raw_data/ablation/synthetic/job_2354994_20260710/` | 160 formal rows, raw logs, PBS stdout |
| email raw | `raw_data/ablation/email-EuAll/job_2354999_20260710/` | 24 formal rows, raw logs, PBS stdout |
| Corrected raw | `raw_data/corrected_325557/job_2406254/` | 40 formal rows, status, manifest, SHA256 evidence |
| Raw integrity indexes | `raw_data/MANIFEST.tsv`; `raw_data/RAW_DATA_INDEX.tsv`; `raw_data/SHA256SUMS`; `raw_data/corrected_325557/SHA256SUMS` | Job attribution, archive identity, SHA256 verification |
| Formal ablation results | `result/ablation/synthetic_2354994/`; `result/ablation/email_2354999/`; `result/ablation/corrected_325557/` | Summaries, contributions, corrected statistics, mixed aggregate |
| Thesis table and figure | `result/tables/thesis/T3_ablation_summary.tsv`; `result/tables/thesis/T3_ablation_summary.md`; `result/figures/thesis/ablation_contributions.pdf`; `result/figures/thesis/ablation_contributions.png`; `result/figures/thesis/ablation_contributions.svg` | Current displayed effects and provenance |
| Result catalog | `result/TABLES_AND_FIGURES.md`; `result/MANIFEST.md`; `result/COVERAGE.md`; `result/coverage_matrix.tsv` | Jobs, checkpoints, coverage, limitations |
| Thesis evidence catalog | `docs/thesis/thesis_values.tsv`; `docs/thesis/evidence_matrix.tsv` | Cross-document value and claim alignment |
| Experiment-time code | `code_snapshots/phase_def_block_20260710/experiments/run_ablation.cu`; `code_snapshots/phase_def_block_20260710/src/proposed/host_ablation.cu`; `code_snapshots/phase_def_block_20260710/scripts/summarize_ablation.py` | Synthetic/email implementation and calculation definition |
| Corrected checkpoint code | Commit `45352a344aaac463283a647467b790be9b45bfb8` versions of `run_ablation.cu`, `host_ablation.cu`, and `summarize_ablation.py` | Corrected implementation and calculation definition |
| Historical-input provenance | `result/provenance/GRAPH_325557_INTEGRITY_AUDIT.md`; `failure/early_terminated/memory_correctness_2368398/`; `failure/failed/oom/memory_correctness_2368269/` | Separation of malformed-input history |

最終検査結果を Table C.20 に示す。raw SHA256 は Synthetic `ef0f7870...193eb`、email `77fd8106...c45a`、Corrected `ef96297c...df04` で、それぞれ raw SHA256 index と一致した。正式値と独立値に差異はなく、Chapter 7 の表示値 H/W/A = 1.679/1.066/1.391（Synthetic-4）、1.429/0.970/1.720（email）、および 1.4767/1.1012/1.5563（Corrected 325557）と整合する。

**Table C.20: Recalculation and document validation summary.**

| Check | Expected | Observed | Result |
|:--|:--|:--|:--|
| Synthetic formal rows | 160 | 160 | Pass |
| email formal rows | 24 | 24 | Pass |
| Corrected formal rows | 40 | 40 | Pass |
| Total formal rows | 224 | 224 | Pass |
| Missing graph-configuration cells | 0 | 0 | Pass |
| Duplicate formal rows | 0 | 0 | Pass |
| Unknown configurations | 0 | 0 | Pass |
| Non-finite or non-positive runtime / GTEPS | 0 | 0 | Pass |
| Failed formal rows | 0 | 0 | Pass |
| Warmup rows included | 0 | 0 | Pass |
| Per-configuration median / mean / sample standard deviation / min / max / median GTEPS | Exact recomputation | Match | Pass |
| Per-graph main effects | Formal value at 4 d.p. | Match | Pass |
| Current Synthetic-4 aggregate | H=1.6787; W=1.0661; A=1.3914 | Match | Pass |
| Independent recalculation repeatability | Byte-identical | SHA256 identical | Pass |
| Derivation script repeatability | Byte-identical | Synthetic and email outputs identical | Pass |
| Chapter 7 value alignment | Exact at displayed precision | Match | Pass |
| Mixed-checkpoint disclosure | Required | C.1, C.8, C.9.1 | Pass |
| Historical malformed-input separation | Required | C.4.4, C.7.4, C.9.2 | Pass |

本付録は既存の引用 key を追加せず、repository 内の canonical path だけを source note として用いる。すべての数値は raw record または正式派生成果物から取得・再計算しており、丸め済み main effect から runtime を逆算していない。
