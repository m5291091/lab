# Appendix B Complete PathMerge Batch Sweeps

This appendix presents every trial value recorded in the retained raw TSV files for the PathMerge batch-size sweeps that determined the comparison denominator. Chapter 6, Section 6.3, reports only the sweep summary and adopted batches, while this appendix provides the complete values. It reports each trial's runtime and GTEPS, distinguishes requested and effective batches, separates the screening and confirmation stages, and gives the median, mean, and sample standard deviation for each batch.

This appendix introduces neither new experimental conditions nor new performance claims. The sweeps were a tuning procedure for determining the denominator (Chapter 5, Section 5.7). This appendix documents that procedure transparently. Every reported value comes from a raw TSV record. No value was back-calculated from a rounded table, interpolated, or estimated.

## B.1 Evaluation Scope

The PathMerge code evaluated in this appendix is a third-party implementation of Galliot, a path-merging BC algorithm. The evaluated snapshot of the upstream `gobardhanm/path-merging-bc` repository was retained with an adapter (Section 5.4). It was not confirmed as the original authors' official implementation. No explicit upstream license notice was identified, so the upstream license was not independently verified.

<!-- Source note (internal): evaluated upstream commit 9c231b46f7499380d4495262c1ec75a11cdaae7a; see references.bib:pathmergeRepo and SOURCE_AUDIT.tsv:S08. -->

This implementation is an external comparator in this study, not ground truth. It is not used as a correctness criterion for BC output, and the sweep results do not provide correctness evidence. The observed values are limited to this retained snapshot, the evaluated graphs, and the GH200 (Miyabi-G) environment.

Therefore, the values in this appendix do not generalize to the PathMerge or Galliot algorithms, an official implementation by the original authors, another implementation, or another computing environment. This study also does not claim that the best batch on one graph is best on another graph or in another environment. As Sections 6.3 and B.6 show, the best batch for roadNet-PA/TX did not apply to roadNet-CA.

This appendix does not change the Chapter 6 headline. The headline speedup is 1.31–3.17 times with tuned PathMerge as the denominator (Section 6.6). This appendix describes only how that denominator was determined.

## B.2 Sweep Stages and Selection Rules

The stages and terms used in this appendix are defined below.

**Table B.1: Definitions of the sweep stages and terms used in this appendix.**

| Term | Definition |
|---|---|
| Screening | The stage that evaluates the candidate batch range in 1 pass with $N_{\mathrm{trials}}=1$ for each batch to identify the trend and candidates for further evaluation. |
| Confirmation | The stage that evaluates the batches retained after screening with $N_{\mathrm{trials}}\ge2$ to confirm their ranking. |
| Extension | The stage that adds 1 measurement outside the candidate range to establish an internal minimum. |
| Additional trial | 1 additional trial in another job that belongs to none of the preceding stages. |
| Requested Batch | Batch size requested at runtime through `PATHMERGE_BC_BATCH_SIZE` or the sweep script's `BATCH_LIST`. |
| Effective Batch | Batch size actually adopted by the implementation after evaluating the memory budget. |
| Clamp | An event in which the effective batch was reduced because the requested batch exceeded the HBM3 budget. The evidence is a warning line in a retained log. |
| Successful trial | A completed execution for which the raw TSV records `Time_sec` and `GTEPS`. |
| Failed trial | A trial that did not complete and has no runtime value. It is excluded from aggregation. |
| Median | Median `Time_sec` over the successful trials in the group. This is the primary value. |
| Mean | Arithmetic mean of the samples in the same group. This is a supplementary value. |
| Sample Standard Deviation | Sample standard deviation $s_T$ (ddof=1, unbiased estimator). This is a supplementary value. |
| $N_{\mathrm{trials}}$ | Number of successful trials in the group. This symbol differs from the vertex count $n$. |
| Final adopted baseline | Measurement adopted as the denominator in the final comparison (Table 6.1). |

The aggregation rules follow Chapter 5, Section 5.6, and Appendix A, Section A.11. The median is the primary value. The mean, $s_T$, minimum, and maximum are supplementary values. The sample standard deviation is defined as

$$
s_T=\sqrt{\frac{1}{N_{\mathrm{trials}}-1}\sum_{i=1}^{N_{\mathrm{trials}}}\left(t_i-\bar{t}\right)^2}.
$$

For a group with $N_{\mathrm{trials}}=1$, $s_T$ is not calculated and is reported as `N/A (n=1)`. More generally, it is reported as `N/A (n<2)` when $N_{\mathrm{trials}}<2$. No unavailable value is replaced with 0 for calculation.

The trial-level tables do not merge stages. Measurements from different jobs remain separate rows even when their requested batches are identical. Requested and effective batches always appear in separate columns, and an effective value after clamping is not reported as the requested value.

In contrast, the canonical per-batch statistics used to rank the sweep and select the tuned batch pool every successful trial at that batch across screening, confirmation, extension, and additional trials. Each graph's `SOURCE.md` records this aggregation. The pooled values match those in Chapter 6, including 1491.13 s for roadNet-TX b64 and 941.39 s for roadNet-PA b64. The summary tables show both stage-specific rows and explicitly labeled `Pooled` rows, allowing recalculation from the raw data at either granularity.

The tuned-batch selection rule follows Section 5.7 and Appendix A, Section A.5. The candidate with the lowest pooled median was the best observed sweep batch. The final denominator was the faster of the sweep best and default b64 (`scripts/merge_final_tables.py`). Therefore, the sweep best was adopted directly for email-EuAll and roadNet-CA. For roadNet-PA and roadNet-TX, the sweep confirmed that the best batch was the default b64, and a legacy baseline measurement under the same setting was adopted. The confirmation measurement and final adopted value for each of these graphs are separate measurements under the same batch setting. They are neither missing nor contradictory (Section 6.3 and Section B.8).

The availability of effective-batch records differs among series. When a retained run log contains a `[PathMerge] free_mem=..., batch_size=...` line, this appendix reports that recorded value as the Effective Batch. When only the TSV was retained without this line, the effective batch cannot be confirmed independently and is reported as `Not recorded`. In those cases, the retained job logs also contain no clamp warning. Only 2 clamps were recorded across the sweeps (Sections B.3 and B.7).

No sweep series used a warmup, so every recorded trial is included in the aggregation.

## B.3 email-EuAll

The email-EuAll sweep had 2 stages. Screening comprised trial 1 of job 2359096, which measured requested batches 8, 16, 64, 256, and 1024 with $N_{\mathrm{trials}}=1$ each. Confirmation comprised job 2359169, which measured requested batches 512, 1024, 2048, 4096, and 8192 with $N_{\mathrm{trials}}=3$ each. The complete set contained 9 requested batches: 8, 16, 64, 256, 512, 1024, 2048, 4096, and 8192. Batch b1024 appeared in both stages.

Table B.2 lists every trial. The retained logs for both stages contain `batch_size=` lines, so every effective batch is a recorded value.

**Table B.2: All recorded PathMerge sweep trials on email-EuAll.**

| Stage | Requested Batch | Effective Batch | Trial | Runtime (s) | GTEPS | Status | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|:--|:--|:--|
| Screening | 8 | 8 | 1 | 786.914750 | 0.1227 | Success | 2359096 | `phase_def_block_20260710` |
| Screening | 16 | 16 | 1 | 491.012537 | 0.1967 | Success | 2359096 | `phase_def_block_20260710` |
| Screening | 64 | 64 | 1 | 226.049570 | 0.4273 | Success | 2359096 | `phase_def_block_20260710` |
| Screening | 256 | 256 | 1 | 125.914082 | 0.7671 | Success | 2359096 | `phase_def_block_20260710` |
| Screening | 1024 | 1024 | 1 | 97.692453 | 0.9887 | Success | 2359096 | `phase_def_block_20260710` |
| Confirmation | 512 | 512 | 1 | 106.434775 | 0.9075 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 512 | 512 | 2 | 106.564268 | 0.9064 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 512 | 512 | 3 | 105.908648 | 0.9120 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 1024 | 1024 | 1 | 100.101618 | 0.9649 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 1024 | 1024 | 2 | 99.797694 | 0.9679 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 1024 | 1024 | 3 | 99.929143 | 0.9666 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 2048 | 2048 | 1 | 97.798748 | 0.9876 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 2048 | 2048 | 2 | 96.959824 | 0.9962 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 2048 | 2048 | 3 | 98.928186 | 0.9764 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 4096 | 4096 | 1 | 101.160326 | 0.9548 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 4096 | 4096 | 2 | 101.691590 | 0.9498 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 4096 | 4096 | 3 | 101.580514 | 0.9509 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 8192 | 7393 (clamped) | 1 | 103.013717 | 0.9376 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 8192 | 7393 (clamped) | 2 | 103.270797 | 0.9353 | Success | 2359169 | `phase_def_block_20260710` |
| Confirmation | 8192 | 7393 (clamped) | 3 | 104.200792 | 0.9270 | Success | 2359169 | `phase_def_block_20260710` |

<!-- Source: raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/email_smallbatch_trial1.tsv (screening); .../pathmerge_sweep_results.tsv and .../pathmerge_sweep.log (confirmation); .../job_2359096_20260711/ and .../job_2359169_20260711/pbs_stdout.log -->
> Source: screening rows from `raw_data/tuning/pathmerge/email-EuAll/pathmerge_bc/job_multi_20260710/email_smallbatch_trial1.tsv`; confirmation rows from `.../job_multi_20260710/pathmerge_sweep_results.tsv`. Effective batches and the clamp warning are read from `.../job_multi_20260710/pathmerge_sweep.log` and `raw_data/unsuccessful/early_terminated/pathmerge_sweep/email-EuAll/job_2359096_20260711/pathmerge_sweep.log`. Job identifiers from the retained `pbs_stdout.log` of each job.

1 clamp occurred at requested b8192. All 3 trials recorded that the requested batch exceeded the HBM3 budget and was reduced to 7393 based on 101.4 GB of free memory and 11,660,396 bytes per 1 source. The recorded `num_batches` was 36. This was not a failure that stopped execution. The implementation reduced the batch to meet its memory budget and completed normally. No clamp was recorded for another requested batch.

<!-- Source note (internal): exact retained warning: WARNING: batch_size=8192 exceeds HBM3 budget; clamping to 7393 (free=101.4 GB, 11660396 B/source). -->

Table B.3 presents the per-batch aggregates.

**Table B.3: Per-batch aggregates of the email-EuAll sweep. Median, mean, min, and max are in seconds; SD is the sample standard deviation (ddof=1).**

| Stage | Requested Batch | Effective Batch | Valid Trials | Median (s) | Mean (s) | Sample Standard Deviation (s) | Min (s) | Max (s) | Selection |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| Screening | 8 | 8 | 1 | 786.91 | 786.91 | N/A (n=1) | 786.91 | 786.91 | Not selected |
| Screening | 16 | 16 | 1 | 491.01 | 491.01 | N/A (n=1) | 491.01 | 491.01 | Not selected |
| Screening | 64 | 64 | 1 | 226.05 | 226.05 | N/A (n=1) | 226.05 | 226.05 | Not selected |
| Screening | 256 | 256 | 1 | 125.91 | 125.91 | N/A (n=1) | 125.91 | 125.91 | Not selected |
| Screening | 1024 | 1024 | 1 | 97.69 | 97.69 | N/A (n=1) | 97.69 | 97.69 | Not selected |
| Confirmation | 512 | 512 | 3 | 106.43 | 106.30 | 0.347 | 105.91 | 106.56 | Not selected |
| Confirmation | 1024 | 1024 | 3 | 99.93 | 99.94 | 0.152 | 99.80 | 100.10 | Not selected |
| Confirmation | 2048 | 2048 | 3 | 97.80 | 97.90 | 0.988 | 96.96 | 98.93 | **Selected (tuned)** |
| Confirmation | 4096 | 4096 | 3 | 101.58 | 101.48 | 0.280 | 101.16 | 101.69 | Not selected |
| Confirmation | 8192 | 7393 (clamped) | 3 | 103.27 | 103.50 | 0.625 | 103.01 | 104.20 | Not selected |
| Pooled (screening + confirmation) | 1024 | 1024 | 4 | 99.86 | 99.38 | 1.132 | 97.69 | 100.10 | Not selected |

<!-- Recomputed from the per-trial Time_sec values of Table B.2; no value is derived from a rounded table. -->
> Source: recomputed from the per-trial `Time_sec` values in Table B.2; cross-checked against `result/tuning/pathmerge/email-EuAll/SOURCE.md`. The `Pooled` row corresponds to the aggregation used in `result/tables/final_speedup_tables.md`, which reports b1024 as `n=4`; the stage-separated rows above it are the same trials at finer granularity.

The best observed sweep result was b2048 with a median of 97.80 s, which was adopted as tuned. The median decreased monotonically as the requested batch increased from 8 to 2048. Beyond b2048, it increased to 101.58 s at b4096 and 103.27 s at b8192, whose effective batch was 7393. This pattern forms an internal minimum. The email-EuAll panel in Figure 6.3 shows only points at b512 and above. Therefore, this table is the only presentation of requested batches below that range: b8 through b256, each with $N_{\mathrm{trials}}=1$ in screening. The b64 value of 226.05 s is a different measurement at a different checkpoint from the legacy default b64 value of 220.39 s with $N_{\mathrm{trials}}=5$ in Table 6.3. These two values are not combined into one series.

Batch b1024 appeared in both screening, with 97.69 s and $N_{\mathrm{trials}}=1$, and confirmation, with 99.93 s and $N_{\mathrm{trials}}=3$. The single screening trial was shorter than each of the 3 confirmation trials. However, Section 5.6 does not treat a single trial with $N_{\mathrm{trials}}=1$ as representative. At either the stage-specific or pooled granularity, b2048 remained the best batch.

### Incomplete or Failed Runs

Job 2359096 submitted 3 trials for requested batches 8, 16, 64, 256, and 1024. It ended during b8 of trial 2 after trial 1 had completed, leaving no record for the remainder of trial 2 or for trial 3. `result/tuning/pathmerge/SOURCE.md` and `result/tuning/pathmerge/email-EuAll/SOURCE.md` classify this job as intentionally early terminated. The termination mechanism cannot be confirmed independently from the retained logs, so its classification is `Cause not independently confirmed`. The records contain no OOM evidence and no evidence sufficient to establish a timeout.

The 5 measurements in trial 1 completed and have both `Time_sec` and `GTEPS` records in the raw TSV. This appendix treats only these 5 completed measurements as successful screening trials. It does not aggregate the unrecorded trials 2 and 3 as 0 seconds.

## B.4 roadNet-PA

The roadNet-PA sweep comprised screening in job 2359080, which measured requested batches 8, 16, and 32 with $N_{\mathrm{trials}}=1$ each. Confirmation in job 2355001 measured requested batches 64, 128, 256, and 512 with $N_{\mathrm{trials}}=3$ each. 1 additional b64 trial from another job was also recorded. The complete set contained 7 requested batches: 8, 16, 32, 64, 128, 256, and 512.

Table B.4 lists every trial.

**Table B.4: All recorded PathMerge sweep trials on roadNet-PA.**

| Stage | Requested Batch | Effective Batch | Trial | Runtime (s) | GTEPS | Status | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|:--|:--|:--|
| Screening | 8 | 8 | 1 | 2714.978259 | 0.6180 | Success | 2359080 | `phase_def_block_20260710` |
| Screening | 16 | 16 | 1 | 1573.869732 | 1.0660 | Success | 2359080 | `phase_def_block_20260710` |
| Screening | 32 | 32 | 1 | 1015.980496 | 1.6513 | Success | 2359080 | `phase_def_block_20260710` |
| Confirmation | 64 | Not recorded | 1 | 943.467356 | 1.7783 | Success | 2355001 | `phase_def_block_20260710` |
| Confirmation | 64 | Not recorded | 2 | 939.313473 | 1.7861 | Success | 2355001 | `phase_def_block_20260710` |
| Confirmation | 64 | Not recorded | 3 | 946.632125 | 1.7723 | Success | 2355001 | `phase_def_block_20260710` |
| Additional trial | 64 | Not recorded | 4 | 912.180815 | 1.8392 | Success | Not recorded | `phase_def_block_20260710` |
| Confirmation | 128 | Not recorded | 1 | 1097.682982 | 1.5284 | Success | 2355001 | `phase_def_block_20260710` |
| Confirmation | 128 | Not recorded | 2 | 1105.566512 | 1.5175 | Success | 2355001 | `phase_def_block_20260710` |
| Confirmation | 128 | Not recorded | 3 | 1106.721195 | 1.5159 | Success | 2355001 | `phase_def_block_20260710` |
| Confirmation | 256 | Not recorded | 1 | 1155.295821 | 1.4522 | Success | 2355001 | `phase_def_block_20260710` |
| Confirmation | 256 | Not recorded | 2 | 1155.547670 | 1.4519 | Success | 2355001 | `phase_def_block_20260710` |
| Confirmation | 256 | Not recorded | 3 | 1152.637307 | 1.4556 | Success | 2355001 | `phase_def_block_20260710` |
| Confirmation | 512 | Not recorded | 1 | 1206.022574 | 1.3911 | Success | 2355001 | `phase_def_block_20260710` |
| Confirmation | 512 | Not recorded | 2 | 1209.399142 | 1.3872 | Success | 2355001 | `phase_def_block_20260710` |
| Confirmation | 512 | Not recorded | 3 | 1207.414580 | 1.3895 | Success | 2355001 | `phase_def_block_20260710` |

<!-- Source: raw_data/tuning/pathmerge/roadNet-PA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv; effective batches for b8/b16/b32 from .../pathmerge_sweep.log -->
> Source: `raw_data/tuning/pathmerge/roadNet-PA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`. Effective batches for the screening rows are read from `.../job_multi_20260710/pathmerge_sweep.log`, which is byte-identical to `raw_data/unsuccessful/early_terminated/pathmerge_sweep/roadNet-PA/job_2359080_20260711/pathmerge_sweep.log`. The retained `pbs_stdout.log` of job 2355001 preserves only the result TSV, so the effective batch of those rows is `Not recorded`; no clamp warning is present in the retained logs of this graph. The trial-4 row at b64 originates from an earlier sweep run whose PBS job identifier is not preserved (`result/tuning/pathmerge/roadNet-PA/SOURCE.md`).

Table B.5 presents the per-batch aggregates.

**Table B.5: Per-batch aggregates of the roadNet-PA sweep. Median, mean, min, and max are in seconds; SD is the sample standard deviation (ddof=1).**

| Stage | Requested Batch | Effective Batch | Valid Trials | Median (s) | Mean (s) | Sample Standard Deviation (s) | Min (s) | Max (s) | Selection |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| Screening | 8 | 8 | 1 | 2714.98 | 2714.98 | N/A (n=1) | 2714.98 | 2714.98 | Not selected |
| Screening | 16 | 16 | 1 | 1573.87 | 1573.87 | N/A (n=1) | 1573.87 | 1573.87 | Not selected |
| Screening | 32 | 32 | 1 | 1015.98 | 1015.98 | N/A (n=1) | 1015.98 | 1015.98 | Not selected |
| Confirmation | 64 | Not recorded | 3 | 943.47 | 943.14 | 3.670 | 939.31 | 946.63 | — |
| Additional trial | 64 | Not recorded | 1 | 912.18 | 912.18 | N/A (n=1) | 912.18 | 912.18 | — |
| Pooled (confirmation + additional) | 64 | Not recorded | 4 | 941.39 | 935.40 | 15.766 | 912.18 | 946.63 | **Selected (sweep best)** |
| Confirmation | 128 | Not recorded | 3 | 1105.57 | 1103.32 | 4.919 | 1097.68 | 1106.72 | Not selected |
| Confirmation | 256 | Not recorded | 3 | 1155.30 | 1154.49 | 1.613 | 1152.64 | 1155.55 | Not selected |
| Confirmation | 512 | Not recorded | 3 | 1207.41 | 1207.61 | 1.697 | 1206.02 | 1209.40 | Not selected |

<!-- Recomputed from the per-trial Time_sec values of Table B.4. The pooled b64 median 941.39 s (n=4) is the value quoted in 6.3. -->
> Source: recomputed from the per-trial `Time_sec` values in Table B.4. The pooled b64 row is the aggregation quoted in Chapter 6 (6.3) and in `result/tuning/pathmerge/roadNet-PA/SOURCE.md`.

The best observed sweep result was b64, with a pooled median of 941.39 s and $N_{\mathrm{trials}}=4$. The median decreased as the requested batch increased from 8 to 64 and then increased again from b128. This pattern formed an internal minimum.

This batch is the same as the PathMerge default b64. The final comparison in Table 6.1 adopted the legacy baseline median of 918.67 s under the same b64 setting, with $N_{\mathrm{trials}}=3$ at checkpoint `oldtree_f05ec52_20260512`. It did not adopt the pooled sweep median of 941.39 s. These are separate measurements under the same batch setting. The selection follows the conservative rule in Section 5.7, which uses the faster of the sweep best and default b64 as the denominator. Because the legacy value was faster than the sweep-confirmation value, its adoption reduced the reported roadNet-PA speedup. The headline uses the final adopted value of 918.67 s (Section B.8).

### Incomplete or Failed Runs

Job 2359080 submitted 3 trials for requested batches 8, 16, and 32. It ended during b8 of trial 2 after trial 1 had completed, leaving no record for the remainder of trial 2 or for trial 3. `result/tuning/pathmerge/SOURCE.md` classifies this job as intentionally early terminated. The termination mechanism cannot be confirmed independently from the retained logs, so its classification is `Cause not independently confirmed`. The records contain no OOM evidence and no evidence sufficient to establish a timeout.

The 3 measurements in trial 1 completed and appear as the screening rows in Table B.4. This appendix does not aggregate the unrecorded trials 2 and 3 as 0 seconds.

## B.5 roadNet-TX

The roadNet-TX sweep comprised screening in job 2360072, which measured requested batches 32, 64, and 128 with $N_{\mathrm{trials}}=1$ each, and confirmation in job 2361040, which measured requested batches 32 and 64 with $N_{\mathrm{trials}}=2$ each. The complete set contained 3 requested batches: 32, 64, and 128.

Table B.6 lists every trial.

**Table B.6: All recorded PathMerge sweep trials on roadNet-TX.**

| Stage | Requested Batch | Effective Batch | Trial | Runtime (s) | GTEPS | Status | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|:--|:--|:--|
| Screening | 32 | Not recorded | 1 | 1620.960440 | 1.6359 | Success | 2360072 | `phase_def_block_20260710` |
| Screening | 64 | Not recorded | 1 | 1493.690042 | 1.7753 | Success | 2360072 | `phase_def_block_20260710` |
| Screening | 128 | Not recorded | 1 | 1668.676282 | 1.5891 | Success | 2360072 | `phase_def_block_20260710` |
| Confirmation | 32 | Not recorded | 1 | 1615.624785 | 1.6413 | Success | 2361040 | `phase_def_block_20260710` |
| Confirmation | 32 | Not recorded | 2 | 1631.801685 | 1.6250 | Success | 2361040 | `phase_def_block_20260710` |
| Confirmation | 64 | Not recorded | 1 | 1491.127569 | 1.7783 | Success | 2361040 | `phase_def_block_20260710` |
| Confirmation | 64 | Not recorded | 2 | 1466.158928 | 1.8086 | Success | 2361040 | `phase_def_block_20260710` |

<!-- Source: raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv; job attribution from job_2360072 / job_2361040 pbs_stdout.log -->
<!-- Source note (internal): raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv. Stage and job attribution are preserved in the retained pbs_stdout.log files for jobs 2360072 and 2361040. Both logs record commit 88faffa391026852a4440e5b9a063c08c29624f7, whose canonical SourceSnapshotID is phase_def_block_20260710. Effective batches are Not recorded; no clamp warning is present. -->

Table B.7 presents the per-batch aggregates.

**Table B.7: Per-batch aggregates of the roadNet-TX sweep. Median, mean, min, and max are in seconds; SD is the sample standard deviation (ddof=1).**

| Stage | Requested Batch | Effective Batch | Valid Trials | Median (s) | Mean (s) | Sample Standard Deviation (s) | Min (s) | Max (s) | Selection |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| Screening | 32 | Not recorded | 1 | 1620.96 | 1620.96 | N/A (n=1) | 1620.96 | 1620.96 | — |
| Confirmation | 32 | Not recorded | 2 | 1623.71 | 1623.71 | 11.439 | 1615.62 | 1631.80 | — |
| Pooled (screening + confirmation) | 32 | Not recorded | 3 | 1620.96 | 1622.80 | 8.243 | 1615.62 | 1631.80 | Not selected |
| Screening | 64 | Not recorded | 1 | 1493.69 | 1493.69 | N/A (n=1) | 1493.69 | 1493.69 | — |
| Confirmation | 64 | Not recorded | 2 | 1478.64 | 1478.64 | 17.655 | 1466.16 | 1491.13 | — |
| Pooled (screening + confirmation) | 64 | Not recorded | 3 | 1491.13 | 1483.66 | 15.209 | 1466.16 | 1493.69 | **Selected (sweep best)** |
| Screening | 128 | Not recorded | 1 | 1668.68 | 1668.68 | N/A (n=1) | 1668.68 | 1668.68 | Not selected |

<!-- Recomputed from the per-trial Time_sec values of Table B.6. The pooled b64 median 1491.13 s (n=3) is the value quoted in 6.3. -->
> Source: recomputed from the per-trial `Time_sec` values in Table B.6. The pooled rows are the aggregation recorded in `result/tuning/pathmerge/roadNet-TX/SOURCE.md` and quoted in Chapter 6 (6.3).

The best observed sweep result was b64, with a pooled median of 1491.13 s and $N_{\mathrm{trials}}=3$. The sweep pattern, b32 > b64 < b128, formed an internal minimum. The median difference between b64 and b32 was approximately 8.7%, which exceeded the 3% threshold of the conservative rule in Section 5.7.

As on roadNet-PA, this best batch was the default b64. The final denominator adopted the legacy baseline median of 1482.68 s under the same b64 setting, with $N_{\mathrm{trials}}=3$ at checkpoint `oldtree_f05ec52_20260512`. It did not adopt the pooled sweep median of 1491.13 s. These are separate measurements under the same setting. Because the legacy value was faster than the sweep-confirmation value, its adoption reduced the reported roadNet-TX speedup. The headline uses the final adopted value of 1482.68 s (Section B.8).

### Incomplete or Failed Runs

The roadNet-TX sweep has no record of early termination, an incomplete sweep, timeout, OOM, or runtime failure. No artifact corresponding to this graph's sweep exists under `failure/`.

## B.6 roadNet-CA

The roadNet-CA sweep had 3 stages. Screening in job 2360073 measured requested batches 32, 64, and 128 with $N_{\mathrm{trials}}=1$ each. Confirmation in job 2362006 measured requested batches 32 and 64 with $N_{\mathrm{trials}}=2$ each. Extension in job 2361041 added 1 measurement at requested batch 16 to establish an internal minimum. The complete set contained 4 requested batches: 16, 32, 64, and 128.

Table B.8 lists every trial.

**Table B.8: All recorded PathMerge sweep trials on roadNet-CA.**

| Stage | Requested Batch | Effective Batch | Trial | Runtime (s) | GTEPS | Status | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|:--|:--|:--|
| Extension | 16 | Not recorded | 1 | 3609.950435 | 1.5061 | Success | 2361041 | `phase_def_block_20260710` |
| Screening | 32 | Not recorded | 1 | 3111.176829 | 1.7476 | Success | 2360073 | `phase_def_block_20260710` |
| Confirmation | 32 | Not recorded | 1 | 3079.716622 | 1.7654 | Success | 2362006 | `phase_def_block_20260710` |
| Confirmation | 32 | Not recorded | 2 | 3060.659395 | 1.7764 | Success | 2362006 | `phase_def_block_20260710` |
| Screening | 64 | Not recorded | 1 | 3588.386622 | 1.5152 | Success | 2360073 | `phase_def_block_20260710` |
| Confirmation | 64 | Not recorded | 1 | 3490.242337 | 1.5578 | Success | 2362006 | `phase_def_block_20260710` |
| Confirmation | 64 | Not recorded | 2 | 3491.644750 | 1.5571 | Success | 2362006 | `phase_def_block_20260710` |
| Screening | 128 | Not recorded | 1 | 3830.858410 | 1.4193 | Success | 2360073 | `phase_def_block_20260710` |

<!-- Source: raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv; job attribution from job_2360073 / job_2361041 / job_2362006 pbs_stdout.log -->
> Source: `raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`. Stage and job attribution from `.../job_2360073_20260711/pbs_stdout.log` (screening, `Batches : 32,64,128  Trials: 1`), `.../job_2361041_20260711/pbs_stdout.log` (extension, `Batches : 16  Trials: 1`), and `.../job_2362006_20260711/pbs_stdout.log` (confirmation, `Batches : 32,64  Trials: 2`). The retained logs preserve only the result TSV, so effective batches are `Not recorded`; no clamp warning is present. Checkpoint as recorded by `checkpoint_sha` in each log.

Table B.9 presents the per-batch aggregates.

**Table B.9: Per-batch aggregates of the roadNet-CA sweep. Median, mean, min, and max are in seconds; SD is the sample standard deviation (ddof=1).**

| Stage | Requested Batch | Effective Batch | Valid Trials | Median (s) | Mean (s) | Sample Standard Deviation (s) | Min (s) | Max (s) | Selection |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| Extension | 16 | Not recorded | 1 | 3609.95 | 3609.95 | N/A (n=1) | 3609.95 | 3609.95 | Not selected |
| Screening | 32 | Not recorded | 1 | 3111.18 | 3111.18 | N/A (n=1) | 3111.18 | 3111.18 | — |
| Confirmation | 32 | Not recorded | 2 | 3070.19 | 3070.19 | 13.475 | 3060.66 | 3079.72 | — |
| Pooled (screening + confirmation) | 32 | Not recorded | 3 | 3079.72 | 3083.85 | 25.511 | 3060.66 | 3111.18 | **Selected (tuned)** |
| Screening | 64 | Not recorded | 1 | 3588.39 | 3588.39 | N/A (n=1) | 3588.39 | 3588.39 | — |
| Confirmation | 64 | Not recorded | 2 | 3490.94 | 3490.94 | 0.992 | 3490.24 | 3491.64 | — |
| Pooled (screening + confirmation) | 64 | Not recorded | 3 | 3491.64 | 3523.42 | 56.263 | 3490.24 | 3588.39 | Not selected |
| Screening | 128 | Not recorded | 1 | 3830.86 | 3830.86 | N/A (n=1) | 3830.86 | 3830.86 | Not selected |

<!-- Recomputed from the per-trial Time_sec values of Table B.8. Pooled b32 median 3079.72 s is the adopted tuned value of Table 6.1. -->
> Source: recomputed from the per-trial `Time_sec` values in Table B.8. The pooled rows are the aggregation recorded in `result/tuning/pathmerge/roadNet-CA/SOURCE.md` and quoted in Chapter 6 (6.3).

The best observed sweep result was b32, with a pooled median of 3079.72 s and $N_{\mathrm{trials}}=3$, and it was adopted as tuned. The pattern, b16 > b32 < b64 < b128, formed an internal minimum. The median difference between b32 and b64 was approximately 13.4%, which exceeded the 3% threshold of the conservative rule in Section 5.7. Batch b64, which was best on roadNet-PA/TX, was not best on roadNet-CA. The best batch therefore did not generalize across graphs.

The tuned b32 value of 3079.72 s must be distinguished from the roadNet-CA default b64 measurement. The default b64 is the legacy baseline median of 3499.03 s in Table 6.3, with $N_{\mathrm{trials}}=3$ at checkpoint `oldtree_f05ec52_20260512`. It differs from the pooled b64 sweep median of 3491.64 s at checkpoint `phase_def_block_20260710`. Because the sweep-best b32 value of 3079.72 s was faster than the default value of 3499.03 s, the selection rule in Section 5.7 made b32 the final adopted configuration. The 1.64-fold comparison against the default is supplementary, while the headline is 1.45-fold against tuned PathMerge (Section 6.3).

### Incomplete or Failed Runs

The roadNet-CA sweep has no record of early termination, an incomplete sweep, timeout, OOM, or runtime failure. No corresponding artifact exists under `failure/`.

## B.7 Historical 325557 Sweep

This section concerns a sweep conducted on the former, pre-repair `325557_3216152` input. The former `325557_3216152` input was malformed because it stored 1-based vertex identifiers as 0-based identifiers (Section 5.3). Current experiments use only `325557_3216152_corrected_v1`, which `tools/repair_325557_graph.py` reconstructed deterministically. The retained sweep logs identify the input as `data/325557_3216152`, with `num_sources=325557` and `edges=3216152`, rather than the corrected graph.

Therefore, this section presents historical invalid-input evidence. Its runtimes and GTEPS are not used as performance results for the corrected graph or as correctness evidence. These values are excluded from the RQ1 main performance comparison in Chapter 6 and from the current formal headline. This study also does not generalize the selected batch to the corrected graph.

The sweep had 3 stages. Initial exploration measured requested batches 32, 64, 256, and 512 in an early run whose PBS job identifier was not retained. Screening in job 2355000 measured requested batches 512, 1024, and 2048 with $N_{\mathrm{trials}}=3$ each. Confirmation in job 2359081 measured requested batches 4096 and 8192 with $N_{\mathrm{trials}}=3$ each. The complete set contained 8 requested batches: 32, 64, 256, 512, 1024, 2048, 4096, and 8192.

Table B.10 lists every trial.

**Table B.10: All recorded PathMerge sweep trials on the historical (pre-repair, malformed) 325557_3216152 input.**

| Stage | Requested Batch | Effective Batch | Trial | Runtime (s) | GTEPS | Status | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|:--|:--|:--|
| Initial exploration | 32 | Not recorded | 1 | 1292.211744 | 0.8103 | Success (GTEPS recomputed) | Not recorded | `phase_def_block_20260710` |
| Initial exploration | 32 | Not recorded | 2 | 1292.320377 | 0.8102 | Success | Not recorded | `phase_def_block_20260710` |
| Initial exploration | 64 | Not recorded | 1 | 770.822531 | 1.3583 | Success | Not recorded | `phase_def_block_20260710` |
| Initial exploration | 256 | Not recorded | 1 | 324.266406 | 3.2290 | Success | Not recorded | `phase_def_block_20260710` |
| Initial exploration | 512 | Not recorded | 1 | 240.241917 | 4.3583 | Success | Not recorded | `phase_def_block_20260710` |
| Screening | 512 | Not recorded | 2 | 240.304285 | 4.3571 | Success | 2355000 | `phase_def_block_20260710` |
| Screening | 512 | Not recorded | 3 | 240.068205 | 4.3614 | Success | 2355000 | `phase_def_block_20260710` |
| Screening | 512 | Not recorded | 4 | 239.368826 | 4.3742 | Success | 2355000 | `phase_def_block_20260710` |
| Screening | 1024 | Not recorded | 1 | 194.879643 | 5.3728 | Success | 2355000 | `phase_def_block_20260710` |
| Screening | 1024 | Not recorded | 2 | 195.154679 | 5.3652 | Success | 2355000 | `phase_def_block_20260710` |
| Screening | 1024 | Not recorded | 3 | 196.762783 | 5.3213 | Success | 2355000 | `phase_def_block_20260710` |
| Screening | 2048 | Not recorded | 1 | 174.291804 | 6.0074 | Success | 2355000 | `phase_def_block_20260710` |
| Screening | 2048 | Not recorded | 2 | 175.430535 | 5.9684 | Success | 2355000 | `phase_def_block_20260710` |
| Screening | 2048 | Not recorded | 3 | 176.324096 | 5.9382 | Success | 2355000 | `phase_def_block_20260710` |
| Confirmation | 4096 | 4096 | 1 | 169.270083 | 6.1856 | Success | 2359081 | `phase_def_block_20260710` |
| Confirmation | 4096 | 4096 | 2 | 167.574471 | 6.2482 | Success | 2359081 | `phase_def_block_20260710` |
| Confirmation | 4096 | 4096 | 3 | 167.408767 | 6.2544 | Success | 2359081 | `phase_def_block_20260710` |
| Confirmation | 8192 | 6018 (clamped) | 1 | 168.716513 | 6.2059 | Success | 2359081 | `phase_def_block_20260710` |
| Confirmation | 8192 | 6018 (clamped) | 2 | 168.130241 | 6.2276 | Success | 2359081 | `phase_def_block_20260710` |
| Confirmation | 8192 | 6018 (clamped) | 3 | 168.266215 | 6.2225 | Success | 2359081 | `phase_def_block_20260710` |

<!-- Source: raw_data/tuning/pathmerge/325557/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv; effective batches and clamp from .../pathmerge_sweep.log (job 2359081 scope) -->
> Source: `raw_data/tuning/pathmerge/325557/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`. Job attribution from `.../job_2355000_20260710/pbs_stdout.log` (`Batches : 512,1024,2048  Trials: 3`) and `.../job_2359081_20260711/pbs_stdout.log` (`Batches : 4096,8192  Trials: 3`). The retained `pathmerge_sweep.log` covers only the job-2359081 batches, so effective batches are recorded for b4096 and b8192 and are `Not recorded` elsewhere. The trial-1 GTEPS at b32 was corrupted in the original TSV column and was recomputed from `Time_sec` with the runner formula (`result/tuning/pathmerge/325557/SOURCE.md`); the recomputation reproduces the tabulated 0.8103. Runs whose originating result directory carries no preserved PBS identifier are recorded as `Not recorded`.

1 clamp occurred at requested b8192. All 3 trials recorded that the requested batch exceeded the HBM3 budget and was reduced to 6018 based on 101.4 GB of free memory and 14,324,508 bytes per 1 source. The recorded `num_batches` was 55. This event with effective 6018 and the email-EuAll clamp to 7393 are the only 2 recorded clamps across the sweeps. The state per 1 source was larger for this graph than for email-EuAll, at 14,324,508 versus 11,660,396 bytes. Therefore, the effective batch after reduction was smaller for the same requested b8192.

<!-- Source note (internal): exact retained warning: WARNING: batch_size=8192 exceeds HBM3 budget; clamping to 6018 (free=101.4 GB, 14324508 B/source). -->

Table B.11 presents the per-batch aggregates.

**Table B.11: Per-batch aggregates of the historical 325557_3216152 sweep. Median, mean, min, and max are in seconds; SD is the sample standard deviation (ddof=1). These values are historical invalid-input evidence and are not used for any current performance or correctness claim.**

| Stage | Requested Batch | Effective Batch | Valid Trials | Median (s) | Mean (s) | Sample Standard Deviation (s) | Min (s) | Max (s) | Selection |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| Initial exploration | 32 | Not recorded | 2 | 1292.27 | 1292.27 | 0.077 | 1292.21 | 1292.32 | Not selected |
| Initial exploration | 64 | Not recorded | 1 | 770.82 | 770.82 | N/A (n=1) | 770.82 | 770.82 | Not selected |
| Initial exploration | 256 | Not recorded | 1 | 324.27 | 324.27 | N/A (n=1) | 324.27 | 324.27 | Not selected |
| Initial exploration | 512 | Not recorded | 1 | 240.24 | 240.24 | N/A (n=1) | 240.24 | 240.24 | — |
| Screening | 512 | Not recorded | 3 | 240.07 | 239.91 | 0.486 | 239.37 | 240.30 | — |
| Pooled (initial + screening) | 512 | Not recorded | 4 | 240.16 | 240.00 | 0.430 | 239.37 | 240.30 | Not selected |
| Screening | 1024 | Not recorded | 3 | 195.15 | 195.60 | 1.017 | 194.88 | 196.76 | Not selected |
| Screening | 2048 | Not recorded | 3 | 175.43 | 175.35 | 1.019 | 174.29 | 176.32 | Not selected |
| Confirmation | 4096 | 4096 | 3 | 167.57 | 168.08 | 1.030 | 167.41 | 169.27 | **Selected (sweep best)** |
| Confirmation | 8192 | 6018 (clamped) | 3 | 168.27 | 168.37 | 0.307 | 168.13 | 168.72 | Not selected |

<!-- Recomputed from the per-trial Time_sec values of Table B.10; historical malformed input, excluded from current claims. -->
> Source: recomputed from the per-trial `Time_sec` values in Table B.10. The pooled b512 row is the aggregation recorded in `result/tuning/pathmerge/325557/SOURCE.md`.

The best observed sweep result was b4096, with a median of 167.57 s. The change from b4096 to b8192, whose effective batch was 6018, increased the median by only approximately 0.41%. The sweep therefore reached an internal minimum within the tested range.

The current use of this selection is limited. Batch b4096 was retained as the PathMerge setting for the Tier B cross-implementation consistency comparison on the corrected 325557 graph in Chapter 9, job 2404743. Only the configuration value was retained; the runtime and correctness results in this section were not. This study did not measure whether b4096 is the best batch on the corrected graph and makes no such claim.

### Incomplete or Failed Runs

The former 325557 sweep has no record of early termination, an incomplete sweep, timeout, OOM, or runtime failure. The b8192 clamp was an advisory warning that reduced the batch to meet the memory budget; it was not an OOM. The former 325557 failures recorded under `failure/`, including the OOM in job 2368269 and fail-fast early termination in job 2368398, belong to the memory-path correctness series and not to this batch sweep.

## B.8 Adopted Tuned Configurations

Table B.12 lists only the configurations adopted as denominators in the final comparison in Table 6.1. The former 325557 graph is excluded because it is not part of the RQ1 main performance comparison.

**Table B.12: PathMerge configurations adopted as the denominator of the main comparison.**

| Graph | Adopted Requested Batch | Adopted Effective Batch | Adopted Median (s) | Measurement Source | Reason for Adoption |
|:--|--:|--:|--:|:--|:--|
| email-EuAll | 2048 | 2048 | 97.80 | Sweep confirmation, job 2359169, checkpoint `phase_def_block_20260710`, $N_{\mathrm{trials}}=3$ | Sweep best; faster than the default b64 legacy measurement (220.39 s) |
| roadNet-PA | 64 | Not recorded | 918.67 | Legacy baseline, checkpoint `oldtree_f05ec52_20260512`, $N_{\mathrm{trials}}=3$ | Sweep best batch equals the default b64; the faster same-batch legacy measurement is adopted as the conservative denominator |
| roadNet-TX | 64 | Not recorded | 1482.68 | Legacy baseline, checkpoint `oldtree_f05ec52_20260512`, $N_{\mathrm{trials}}=3$ | Sweep best batch equals the default b64; the faster same-batch legacy measurement is adopted as the conservative denominator |
| roadNet-CA | 32 | Not recorded | 3079.72 | Sweep confirmation pooled with screening, jobs 2360073 and 2362006, checkpoint `phase_def_block_20260710`, $N_{\mathrm{trials}}=3$ | Sweep best; faster than the default b64 legacy measurement (3499.03 s) |

<!-- Adopted medians recomputed from raw per-trial values; email/CA from the sweep TSVs, PA/TX from the legacy baseline TSV. Matches Table 6.1 and docs/thesis/thesis_values.tsv. -->
> Source: email-EuAll and roadNet-CA medians recomputed from `raw_data/tuning/pathmerge/<graph>/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`; roadNet-PA and roadNet-TX medians recomputed from `raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv`. Effective batches are `Not recorded` where the retained logs preserve only the result TSV; no clamp is recorded for any adopted configuration.

The reason for adopting the legacy baseline values instead of the sweep-confirmation values on roadNet-PA and roadNet-TX requires emphasis. The sweeps confirmed that the best batch for both graphs was the default b64. 2 independent measurements existed under the same batch setting: the sweep-confirmation measurement and the legacy baseline. Following the rule in Section 5.7, the faster legacy measurement was used as the denominator. The sweep-confirmation values, 941.39 s for PA and 1491.13 s for TX, were slower than the legacy values, 918.67 s for PA and 1482.68 s for TX. Using the legacy values therefore reduced the denominator and the reported speedup. This choice agrees with Chapter 6, Section 6.3; Chapter 5, Section 5.7; Appendix A, Section A.5; and `result/main_performance/proposed_vs_pathmerge/README.md`. Only these final adopted values are used for the headline ratios. The sweep-confirmation values are not used in the headline. The headline comparison is median-to-median; it neither mixes means and medians nor substitutes a fastest trial or a screening-only value.

For these 2 graphs, the measurement checkpoints also differ between the numerator and denominator. GPU_Opt used checkpoint `phase_def_block_20260710`, while PathMerge used checkpoint `oldtree_f05ec52_20260512`. Section 5.7 and Appendix A, Section A.5, state this difference explicitly.

## B.9 Recalculation and Validation

The aggregate values in this appendix were recalculated independently from the retained raw TSV files. For every graph, stage, and requested batch, the recalculation produced the trial count, median, mean, sample standard deviation with ddof=1, minimum, and maximum. The pooled aggregates used the same procedure. Rounded table values were not used as calculation inputs. Every displayed summary value was rounded from the raw per-trial values to the shown precision.

The recalculation was executed 2 times with the same procedure, and the two outputs were identical, including their hashes. Temporary calculation files were kept outside the repository and were not added to it.

Table B.13 lists the validation checks and results.

**Table B.13: Independent recalculation and cross-checks against the canonical records.**

| Check | Result |
|:--|:--|
| Trial row count per graph matches the raw TSV | Yes (email-EuAll 20, roadNet-PA 16, roadNet-TX 7, roadNet-CA 8, historical 325557 20; total 71) |
| All requested batches recorded | Yes (email-EuAll 9, roadNet-PA 7, roadNet-TX 3, roadNet-CA 4, historical 325557 8) |
| All effective batches recorded or explicitly marked `Not recorded` | Yes |
| Recorded clamps captured | Yes, 2 of 2 (email-EuAll 8192 to 7393; historical 325557 8192 to 6018) |
| Screening, confirmation, extension, and additional trials kept separate | Yes; pooled aggregates shown as explicitly labelled rows |
| Sample standard deviation omitted where $N_{\mathrm{trials}}<2$ | Yes, reported as `N/A (n=1)` |
| Failed or unrecorded trials excluded, never treated as 0 s | Yes |
| Per-batch medians reproduce each graph's `SOURCE.md` | Yes, all batches |
| Adopted medians reproduce Table 6.1 and `thesis_values.tsv` | Yes (97.80, 918.67, 1482.68, 3079.72) |
| Sweep confirmation medians reproduce the values quoted in 6.3 | Yes (roadNet-PA 941.39 with $N_{\mathrm{trials}}=4$; roadNet-TX 1491.13 with $N_{\mathrm{trials}}=3$; roadNet-CA b64 3491.64) |
| Adopted mean and sample standard deviation reproduce Table 6.2 | Yes (email-EuAll 97.90 / 0.988; roadNet-PA 923.26 / 9.593; roadNet-TX 1493.46 / 24.855; roadNet-CA 3083.85 / 25.511) |
| Default b64 medians reproduce Table 6.3 | Yes (email-EuAll 220.39; roadNet-CA 3499.03) |
| Recomputed GTEPS of the historical b32 trial 1 reproduces the tabulated value | Yes (0.8103) |
| Recalculation repeated twice with identical output | Yes |

<!-- Recalculation performed outside the repository; no aggregation artifact was added to result/ or raw_data/. -->
> Source: recomputation from the raw TSVs cited in B.3 through B.8, cross-checked against `result/tuning/pathmerge/<graph>/SOURCE.md`, `result/main_performance/proposed_vs_pathmerge/comparison.tsv`, `result/tables/final_speedup_tables.md`, and `docs/thesis/thesis_values.tsv`.

1 qualification concerns aggregation granularity. `result/tables/final_speedup_tables.md` summarizes email-EuAll b1024 with $N_{\mathrm{trials}}=4$. This is a pooled descriptive statistic over 1 screening trial from job 2359096 and 3 confirmation trials from job 2359169. It does not represent 4 trials from a single job. Appendix A, Section A.5, and Table B.3 report this relationship as screening 1, confirmation 3, and pooled 4. Batch b1024 was not the best under any of these granularities, so the tuned selection of b2048 is unchanged.
