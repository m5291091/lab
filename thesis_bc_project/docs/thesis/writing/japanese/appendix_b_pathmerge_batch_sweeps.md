# Appendix B Complete PathMerge Batch Sweeps

本付録は、比較の分母である PathMerge のバッチサイズ掃引について、保存された raw TSV に記録された全 trial 値を収録する。Chapter 6（6.3 節）では掃引の要約と採用バッチのみを述べ、全数値を本付録へ委ねた。ここでは各 trial の実行時間と GTEPS、要求バッチと実効バッチの区別、stage（screening と confirmation）の区別、および batch 別の median・mean・標本標準偏差を示す。

本付録は新しい実験条件を導入せず、新しい性能主張も追加しない。掃引は分母の設定を確定するための tuning 手続きであり（Chapter 5、5.7 節）、その透明性を示すことが本付録の目的である。収録値はすべて raw TSV の記録値であり、丸め済みの表から逆算した値、補間値、推定値を含まない。

## B.1 Evaluation Scope

本付録が対象とする PathMerge は、Galliot（path-merging 型 BC アルゴリズム）の第三者実装であり、上流リポジトリ `gobardhanm/path-merging-bc`（評価時 snapshot `9c231b46`）を adapter 化して保存した snapshot である（5.4 節）。これは原著論文著者による公式実装であるとは確認されていない。

この実装は本研究における external comparator であって、ground truth ではない。BC 出力の正しさの基準として用いることはせず、掃引結果を正解性の根拠とすることもしない。掃引で得た値は、評価に用いたこの保存 snapshot、評価した graph、GH200（Miyabi-G）環境における観測に限定される。

したがって本付録の数値は、PathMerge/Galliot アルゴリズム一般、原著者の公式実装、他の実装、他の計算機環境に対して一般化しない。あるグラフで最良となったバッチが他のグラフや他の環境で最良であることも主張しない（実際に 6.3 節および B.6 節のとおり、roadNet-PA/TX の最良バッチは roadNet-CA へ当てはまらなかった）。

本付録は Chapter 6 の headline を変更しない。headline の speedup は tuned PathMerge を分母とする 1.31〜3.17 倍であり（6.6 節）、本付録はその分母の確定過程を記述するにとどまる。

## B.2 Sweep Stages and Selection Rules

本付録で用いる stage と用語を次のとおり定義する。

**Table B.1: Definitions of the sweep stages and terms used in this appendix.**

| Term | Definition |
|---|---|
| Screening | 候補バッチ範囲を各 $N_{\mathrm{trials}}=1$ で 1 巡し、傾向と絞り込み対象を決める段階。 |
| Confirmation | screening で絞り込んだバッチについて $N_{\mathrm{trials}}\ge2$ を測定し、順位を確認する段階。 |
| Extension | 内部最小の判定のために候補範囲外へ 1 点だけ追加測定した段階。 |
| Additional trial | 上記いずれの stage にも属さない、別 job での追加 1 試行。 |
| Requested Batch | 実行時に要求したバッチサイズ（`PATHMERGE_BC_BATCH_SIZE` または掃引スクリプトの `BATCH_LIST`）。 |
| Effective Batch | 実装がメモリ予算判定の後に実際に採用したバッチサイズ。 |
| Clamp | 要求バッチが HBM3 予算を超えるため実効バッチが縮小された事象。保存ログの警告行を根拠とする。 |
| Successful trial | 実行が完了し、raw TSV に `Time_sec` と `GTEPS` が記録された trial。 |
| Failed trial | 実行が完了せず runtime 値をもたない試行。集計に含めない。 |
| Median | 当該グループの successful trial の `Time_sec` の中央値。主値。 |
| Mean | 同グループの標本平均。補助値。 |
| Sample Standard Deviation | 標本標準偏差 $s_T$（ddof=1、不偏推定量）。補助値。 |
| $N_{\mathrm{trials}}$ | 当該グループの successful trial 数。頂点数 $n$ とは別の記号である。 |
| Final adopted baseline | 最終比較（Table 6.1）で分母として採用した測定値。 |

集計規則は Chapter 5（5.6 節）および Appendix A（A.11 節）に従う。主値は median、補助値は mean・$s_T$・min・max であり、$s_T$ は

$$
s_T=\sqrt{\frac{1}{N_{\mathrm{trials}}-1}\sum_{i=1}^{N_{\mathrm{trials}}}\left(t_i-\bar{t}\right)^2}
$$

で定義される。$N_{\mathrm{trials}}=1$ のグループでは $s_T$ を算出せず `N/A (n=1)` と記す。$N_{\mathrm{trials}}<2$ 一般については `N/A (n<2)` と記す。値を 0 で代替して計算することはしない。

trial 単位の表では、stage を混在させない。同一の要求バッチであっても、別 job・別測定であれば別行として記録する。要求バッチと実効バッチは常に別列に置き、clamp 後の実効値を要求値として記載しない。

一方、掃引の順位判定と tuned バッチの選択に用いる canonical な batch 別集計は、当該バッチの screening・confirmation・extension・additional trial を通した全 successful trial に対して行われている（各グラフの `SOURCE.md` の記録）。この pooled 集計は Chapter 6 の記述値（例えば roadNet-TX b64 の 1491.13 s、roadNet-PA b64 の 941.39 s）と一致する。本付録の summary table は、stage 別行と `Pooled` 行の双方を掲げ、どちらの粒度でも raw から再計算できる形にする。

tuned バッチの選択規則は 5.7 節および Appendix A（A.5 節）のとおりである。すなわち、候補バッチごとの pooled median のうち最小を掃引実測の最良とし、最終的な分母には掃引最良と既定 b64 のうち速い方を採用する（`scripts/merge_final_tables.py`）。この規則により、掃引最良がそのまま final adopted となるグラフ（email-EuAll、roadNet-CA）と、掃引で最良バッチが既定 b64 と同一であることを確認した上で同一設定の legacy baseline 実測値を採用したグラフ（roadNet-PA、roadNet-TX）が生じる。後者について、掃引の確認測定値と final adopted 値は同一バッチ設定に対する別々の測定であり、欠損でも矛盾でもない（6.3 節、B.8 節）。

実効バッチの記録可能性には系列差がある。保存された run log が `[PathMerge] free_mem=..., batch_size=...` の行を含む場合は、その記録値を Effective Batch として掲げる。保存された log が TSV のみで当該行を含まない場合は、実効バッチを独立に確認できないため `Not recorded` と記す。この場合も、当該 job の保存ログに clamp 警告行は存在しない。掃引全体を通じて記録されている clamp は 2 件のみである（B.3 節および B.7 節）。

掃引はいずれの系列でも warmup を実施しておらず、記録された全 trial を集計対象とする。

## B.3 email-EuAll

email-EuAll の掃引は 2 stage からなる。screening は要求バッチ 8, 16, 64, 256, 1024 を各 $N_{\mathrm{trials}}=1$ で測定した job 2359096 の trial 1 であり、confirmation は要求バッチ 512, 1024, 2048, 4096, 8192 を各 $N_{\mathrm{trials}}=3$ で測定した job 2359169 である。要求バッチの全体は 8, 16, 64, 256, 512, 1024, 2048, 4096, 8192 の 9 種であり、b1024 は両 stage に現れる。

全 trial を Table B.2 に示す。両 stage とも保存 log が `batch_size=` 行を含むため、実効バッチはすべて記録値である。

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

clamp は要求 b8192 の 1 件である。保存ログには `WARNING: batch_size=8192 exceeds HBM3 budget; clamping to 7393 (free=101.4 GB, 11660396 B/source)` が全 3 trial について記録されており、実効バッチは 7393、`num_batches` は 36 であった。これは実行を停止させる失敗ではなく、実装がメモリ予算に合わせてバッチを縮小した上で正常に完了した記録である。他の要求バッチでは clamp の記録はない。

batch 別集計を Table B.3 に示す。

**Table B.3: Per-batch aggregates of the email-EuAll sweep. Median, mean, min, and max are in seconds; SD is the sample standard deviation (ddof=1).**

| Stage | Requested Batch | Effective Batch | Valid Trials | Median (s) | Mean (s) | Sample SD (s) | Min (s) | Max (s) | Selection |
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

掃引実測の最良は b2048（median 97.80 s）であり、これが tuned として採用された。要求バッチを 8 から 2048 へ増やす間 median は単調に短縮し、b2048 を超えると b4096（101.58 s）、b8192（実効 7393、103.27 s）と再び長くなる内部最小の形状である。Figure 6.3 の email-EuAll panel には b512 以上の点のみを示しているため、それ未満の要求バッチ（b8 から b256、各 $N_{\mathrm{trials}}=1$ の screening）の値は本表が唯一の掲載箇所である。b64 の 226.05 s は、同じ b64 における legacy default 測定（220.39 s、$N_{\mathrm{trials}}=5$、Table 6.3）とは別 checkpoint の別測定であり、両者を同一系列として結合しない。

b1024 は screening（97.69 s、$N_{\mathrm{trials}}=1$）と confirmation（99.93 s、$N_{\mathrm{trials}}=3$）の双方に現れる。screening の単一試行は confirmation の 3 試行いずれよりも短いが、$N_{\mathrm{trials}}=1$ の単一試行を代表値としない規約（5.6 節）に従い、選択は pooled median を含めていずれの粒度でも b2048 が最良である。

### Incomplete or Failed Runs

job 2359096 は要求バッチ 8, 16, 64, 256, 1024 を 3 trial で投入したが、trial 1 完了後、trial 2 の b8 実行中に終了しており、trial 2 と trial 3 の記録は存在しない。この job は `result/tuning/pathmerge/SOURCE.md` および `result/tuning/pathmerge/email-EuAll/SOURCE.md` に意図的な早期打切りとして記録されている。終了機構そのものは保存ログから独立に確認できないため `Cause not independently confirmed` とする。OOM の記録はなく、timeout と断定できる証拠もない。

trial 1 の 5 測定は完了しており、raw TSV に `Time_sec` と `GTEPS` を伴って記録されている。本付録ではこの完了済み 5 測定のみを screening の successful trial として扱い、記録の存在しない trial 2・trial 3 を 0 秒として集計することはしない。

## B.4 roadNet-PA

roadNet-PA の掃引は、要求バッチ 8, 16, 32 を各 $N_{\mathrm{trials}}=1$ で測定した screening（job 2359080）と、要求バッチ 64, 128, 256, 512 を各 $N_{\mathrm{trials}}=3$ で測定した confirmation（job 2355001）からなる。これに加えて、b64 には別 job の追加 1 試行が記録されている。要求バッチの全体は 8, 16, 32, 64, 128, 256, 512 の 7 種である。

全 trial を Table B.4 に示す。

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

batch 別集計を Table B.5 に示す。

**Table B.5: Per-batch aggregates of the roadNet-PA sweep. Median, mean, min, and max are in seconds; SD is the sample standard deviation (ddof=1).**

| Stage | Requested Batch | Effective Batch | Valid Trials | Median (s) | Mean (s) | Sample SD (s) | Min (s) | Max (s) | Selection |
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

掃引実測の最良は b64（pooled median 941.39 s、$N_{\mathrm{trials}}=4$）である。要求バッチを 8 から 64 へ増やす間 median は短縮し、b128 以上では再び長くなる内部最小の形状であった。

このバッチは PathMerge の既定と同一の b64 である。最終比較（Table 6.1）の分母には、同一 b64 設定の legacy baseline 実測値である median 918.67 s（$N_{\mathrm{trials}}=3$、checkpoint `oldtree_f05ec52_20260512`）を採用しており、掃引の pooled median 941.39 s ではない。両者は同一バッチ設定に対する別々の測定であり、5.7 節の保守的採用規則（掃引最良と既定 b64 の速い方を分母とする）に従った結果である。掃引確認値より速い legacy 値を分母としたため、この採用は roadNet-PA の speedup を過小方向に見積もる。headline には final adopted 値 918.67 s を用いる（B.8 節）。

### Incomplete or Failed Runs

job 2359080 は要求バッチ 8, 16, 32 を 3 trial で投入したが、trial 1 完了後、trial 2 の b8 実行中に終了しており、trial 2 と trial 3 の記録は存在しない。この job は `result/tuning/pathmerge/SOURCE.md` に意図的な早期打切りとして記録されている。終了機構は保存ログから独立に確認できないため `Cause not independently confirmed` とする。OOM の記録はなく、timeout と断定できる証拠もない。

trial 1 の 3 測定は完了しており、Table B.4 の screening 行として収録した。記録の存在しない trial 2・trial 3 を 0 秒として集計することはしない。

## B.5 roadNet-TX

roadNet-TX の掃引は、要求バッチ 32, 64, 128 を各 $N_{\mathrm{trials}}=1$ で測定した screening（job 2360072）と、要求バッチ 32, 64 を各 $N_{\mathrm{trials}}=2$ で測定した confirmation（job 2361040）からなる。要求バッチの全体は 32, 64, 128 の 3 種である。

全 trial を Table B.6 に示す。

**Table B.6: All recorded PathMerge sweep trials on roadNet-TX.**

| Stage | Requested Batch | Effective Batch | Trial | Runtime (s) | GTEPS | Status | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|:--|:--|:--|
| Screening | 32 | Not recorded | 1 | 1620.960440 | 1.6359 | Success | 2360072 | `88faffa3` |
| Screening | 64 | Not recorded | 1 | 1493.690042 | 1.7753 | Success | 2360072 | `88faffa3` |
| Screening | 128 | Not recorded | 1 | 1668.676282 | 1.5891 | Success | 2360072 | `88faffa3` |
| Confirmation | 32 | Not recorded | 1 | 1615.624785 | 1.6413 | Success | 2361040 | `88faffa3` |
| Confirmation | 32 | Not recorded | 2 | 1631.801685 | 1.6250 | Success | 2361040 | `88faffa3` |
| Confirmation | 64 | Not recorded | 1 | 1491.127569 | 1.7783 | Success | 2361040 | `88faffa3` |
| Confirmation | 64 | Not recorded | 2 | 1466.158928 | 1.8086 | Success | 2361040 | `88faffa3` |

<!-- Source: raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv; job attribution from job_2360072 / job_2361040 pbs_stdout.log -->
> Source: `raw_data/tuning/pathmerge/roadNet-TX/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`. Stage and job attribution from `.../job_2360072_20260711/pbs_stdout.log` (screening, `Trials: 1`) and `.../job_2361040_20260711/pbs_stdout.log` (confirmation, `Trials: 2`). Both retained logs preserve only the result TSV, so effective batches are `Not recorded`; no clamp warning is present. The checkpoint is the `checkpoint_sha` recorded in both logs (`88faffa391026852a4440e5b9a063c08c29624f7`), belonging to SourceSnapshotID `phase_def_block_20260710`.

batch 別集計を Table B.7 に示す。

**Table B.7: Per-batch aggregates of the roadNet-TX sweep. Median, mean, min, and max are in seconds; SD is the sample standard deviation (ddof=1).**

| Stage | Requested Batch | Effective Batch | Valid Trials | Median (s) | Mean (s) | Sample SD (s) | Min (s) | Max (s) | Selection |
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

掃引実測の最良は b64（pooled median 1491.13 s、$N_{\mathrm{trials}}=3$）である。掃引形状は b32 > b64 < b128 の内部最小であり、b64 と b32 の median 差は約 8.7% と、5.7 節の保守ルールの適用閾値（3%）を上回る。

roadNet-PA と同様、この最良バッチは既定と同一の b64 である。最終比較の分母には、同一 b64 設定の legacy baseline 実測値である median 1482.68 s（$N_{\mathrm{trials}}=3$、checkpoint `oldtree_f05ec52_20260512`）を採用しており、掃引の pooled median 1491.13 s ではない。両者は同一設定の別測定である。掃引確認値より速い legacy 値を分母としたため、この採用は roadNet-TX の speedup を過小方向に見積もる。headline には final adopted 値 1482.68 s を用いる（B.8 節）。

### Incomplete or Failed Runs

roadNet-TX の掃引には、早期終了・不完全掃引・timeout・OOM・runtime failure の記録がない。`failure/` 以下に本グラフの掃引に対応する成果物は存在しない。

## B.6 roadNet-CA

roadNet-CA の掃引は 3 stage からなる。screening は要求バッチ 32, 64, 128 を各 $N_{\mathrm{trials}}=1$ で測定した job 2360073、confirmation は要求バッチ 32, 64 を各 $N_{\mathrm{trials}}=2$ で測定した job 2362006、extension は内部最小の判定のために要求バッチ 16 を 1 点追加した job 2361041 である。要求バッチの全体は 16, 32, 64, 128 の 4 種である。

全 trial を Table B.8 に示す。

**Table B.8: All recorded PathMerge sweep trials on roadNet-CA.**

| Stage | Requested Batch | Effective Batch | Trial | Runtime (s) | GTEPS | Status | Job ID | Checkpoint |
|:--|--:|--:|--:|--:|--:|:--|:--|:--|
| Extension | 16 | Not recorded | 1 | 3609.950435 | 1.5061 | Success | 2361041 | `88faffa3` |
| Screening | 32 | Not recorded | 1 | 3111.176829 | 1.7476 | Success | 2360073 | `88faffa3` |
| Confirmation | 32 | Not recorded | 1 | 3079.716622 | 1.7654 | Success | 2362006 | `88faffa3` |
| Confirmation | 32 | Not recorded | 2 | 3060.659395 | 1.7764 | Success | 2362006 | `88faffa3` |
| Screening | 64 | Not recorded | 1 | 3588.386622 | 1.5152 | Success | 2360073 | `88faffa3` |
| Confirmation | 64 | Not recorded | 1 | 3490.242337 | 1.5578 | Success | 2362006 | `88faffa3` |
| Confirmation | 64 | Not recorded | 2 | 3491.644750 | 1.5571 | Success | 2362006 | `88faffa3` |
| Screening | 128 | Not recorded | 1 | 3830.858410 | 1.4193 | Success | 2360073 | `88faffa3` |

<!-- Source: raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv; job attribution from job_2360073 / job_2361041 / job_2362006 pbs_stdout.log -->
> Source: `raw_data/tuning/pathmerge/roadNet-CA/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`. Stage and job attribution from `.../job_2360073_20260711/pbs_stdout.log` (screening, `Batches : 32,64,128  Trials: 1`), `.../job_2361041_20260711/pbs_stdout.log` (extension, `Batches : 16  Trials: 1`), and `.../job_2362006_20260711/pbs_stdout.log` (confirmation, `Batches : 32,64  Trials: 2`). The retained logs preserve only the result TSV, so effective batches are `Not recorded`; no clamp warning is present. Checkpoint as recorded by `checkpoint_sha` in each log.

batch 別集計を Table B.9 に示す。

**Table B.9: Per-batch aggregates of the roadNet-CA sweep. Median, mean, min, and max are in seconds; SD is the sample standard deviation (ddof=1).**

| Stage | Requested Batch | Effective Batch | Valid Trials | Median (s) | Mean (s) | Sample SD (s) | Min (s) | Max (s) | Selection |
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

掃引実測の最良は b32（pooled median 3079.72 s、$N_{\mathrm{trials}}=3$）であり、これが tuned として採用された。掃引形状は b16 > b32 < b64 < b128 の内部最小である。b32 と b64 の median 差は約 13.4% であり、5.7 節の保守ルールの閾値（3%）を上回る。roadNet-PA/TX で最良となった b64 は roadNet-CA では最良ではなく、最良バッチはグラフ間で一般化しなかった。

roadNet-CA では、tuned の b32（3079.72 s）と default の b64 を明確に区別する必要がある。ここでいう default 測定とは、Table 6.3 に示した legacy baseline の b64 median 3499.03 s（$N_{\mathrm{trials}}=3$、checkpoint `oldtree_f05ec52_20260512`）であり、本掃引の b64 pooled median 3491.64 s（checkpoint `phase_def_block_20260710`）とは別の測定である。掃引最良 b32（3079.72 s）は default 測定 3499.03 s より速いため、5.7 節の選択規則により b32 が final adopted となった。default 比 1.64 倍は補助結果であり、headline は tuned 比 1.45 倍である（6.3 節）。

### Incomplete or Failed Runs

roadNet-CA の掃引には、早期終了・不完全掃引・timeout・OOM・runtime failure の記録がない。`failure/` 以下に本グラフの掃引に対応する成果物は存在しない。

## B.7 Historical 325557 Sweep

本節の掃引は、修復前の旧 `325557_3216152` を入力として実施したものである。旧 `325557_3216152` は 1-based の頂点 ID を 0-based として格納した malformed input であり（5.3 節）、現行の実験では `tools/repair_325557_graph.py` により決定的に再構成した `325557_3216152_corrected_v1` のみを使用する。保存された掃引ログの実行対象は `data/325557_3216152`（`num_sources=325557`、`edges=3216152`）であり、修正版ではない。

したがって本節は historical invalid-input evidence である。ここに示す実行時間・GTEPS を修正版グラフの性能結果として用いることはせず、正確性の根拠として用いることもしない。これらの値は RQ1 の主性能比較（Chapter 6）にも current formal headline にも算入されていない。本節の選定結果を修正版グラフへ一般化することもしない。

掃引は 3 stage からなる。initial exploration は要求バッチ 32, 64, 256, 512 を測定した初期実行（保存ログに PBS job 識別子が残されていない）、screening は要求バッチ 512, 1024, 2048 を各 $N_{\mathrm{trials}}=3$ で測定した job 2355000、confirmation は要求バッチ 4096, 8192 を各 $N_{\mathrm{trials}}=3$ で測定した job 2359081 である。要求バッチの全体は 32, 64, 256, 512, 1024, 2048, 4096, 8192 の 8 種である。

全 trial を Table B.10 に示す。

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

clamp は要求 b8192 の 1 件である。保存ログには `WARNING: batch_size=8192 exceeds HBM3 budget; clamping to 6018 (free=101.4 GB, 14324508 B/source)` が全 3 trial について記録されており、実効バッチは 6018、`num_batches` は 55 であった。email-EuAll の clamp（実効 7393）と本件（実効 6018）が、掃引全体で記録されている clamp の 2 件である。1 source 当たりの状態量が本グラフの方が大きいため（14,324,508 B/source 対 11,660,396 B/source）、同一の要求 b8192 に対して縮小後の実効バッチが小さくなっている。

batch 別集計を Table B.11 に示す。

**Table B.11: Per-batch aggregates of the historical 325557_3216152 sweep. Median, mean, min, and max are in seconds; SD is the sample standard deviation (ddof=1). These values are historical invalid-input evidence and are not used for any current performance or correctness claim.**

| Stage | Requested Batch | Effective Batch | Valid Trials | Median (s) | Mean (s) | Sample SD (s) | Min (s) | Max (s) | Selection |
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

掃引実測の最良は b4096（median 167.57 s）であった。b4096 から b8192（実効 6018）への変化は約 0.41% の悪化にとどまり、掃引範囲内で内部最小に到達したと判定された。

この選定結果の現行での用途は限定される。b4096 は、Chapter 9 の Tier B（修正版 325557 における実装間整合比較、job 2404743）で PathMerge を external comparator として実行する際の設定として引き継がれている。すなわち引き継がれたのは設定値であって、本節の実行時間・正確性の結果ではない。修正版グラフ上で b4096 が最良バッチであることは測定していないため主張しない。

### Incomplete or Failed Runs

旧 325557 の掃引には、早期終了・不完全掃引・timeout・OOM・runtime failure の記録がない。b8192 の clamp は実装がメモリ予算に合わせてバッチを縮小した助言的警告の記録であり、OOM ではない。`failure/` 以下に記録された旧 325557 関連の失敗（job 2368269 の OOM、job 2368398 の fail-fast 早期終了）は memory-path correctness 系列に属し、本節の batch sweep とは別の実験である。

## B.8 Adopted Tuned Configurations

最終比較（Table 6.1）で分母として採用した設定のみを Table B.12 に示す。旧 325557 は RQ1 の主性能比較の対象ではないため、本表に含めない。

**Table B.12: PathMerge configurations adopted as the denominator of the main comparison.**

| Graph | Adopted Requested Batch | Adopted Effective Batch | Adopted Median (s) | Measurement Source | Reason for Adoption |
|:--|--:|--:|--:|:--|:--|
| email-EuAll | 2048 | 2048 | 97.80 | Sweep confirmation, job 2359169, checkpoint `phase_def_block_20260710`, $N_{\mathrm{trials}}=3$ | Sweep best; faster than the default b64 legacy measurement (220.39 s) |
| roadNet-PA | 64 | Not recorded | 918.67 | Legacy baseline, checkpoint `oldtree_f05ec52_20260512`, $N_{\mathrm{trials}}=3$ | Sweep best batch equals the default b64; the faster same-batch legacy measurement is adopted as the conservative denominator |
| roadNet-TX | 64 | Not recorded | 1482.68 | Legacy baseline, checkpoint `oldtree_f05ec52_20260512`, $N_{\mathrm{trials}}=3$ | Sweep best batch equals the default b64; the faster same-batch legacy measurement is adopted as the conservative denominator |
| roadNet-CA | 32 | Not recorded | 3079.72 | Sweep confirmation pooled with screening, jobs 2360073 and 2362006, checkpoint `phase_def_block_20260710`, $N_{\mathrm{trials}}=3$ | Sweep best; faster than the default b64 legacy measurement (3499.03 s) |

<!-- Adopted medians recomputed from raw per-trial values; email/CA from the sweep TSVs, PA/TX from the legacy baseline TSV. Matches Table 6.1 and docs/thesis/thesis_values.tsv. -->
> Source: email-EuAll and roadNet-CA medians recomputed from `raw_data/tuning/pathmerge/<graph>/pathmerge_bc/job_multi_20260710/pathmerge_sweep_results.tsv`; roadNet-PA and roadNet-TX medians recomputed from `raw_data/main_performance/seven_implementations/legacy_partial/large/no_gpu_opt/job_notrecorded_legacy/results_no_gpu_opt.tsv`. Effective batches are `Not recorded` where the retained logs preserve only the result TSV; no clamp is recorded for any adopted configuration.

roadNet-PA と roadNet-TX について、掃引確認値ではなく legacy baseline 値を採用した理由を明記する。両グラフでは掃引により最良バッチが既定と同一の b64 であることを確認した。同一バッチ設定に対して 2 つの独立した測定（掃引確認値と legacy baseline）が存在するため、5.7 節の規則に従い速い方、すなわち legacy baseline を分母とした。掃引確認値（PA 941.39 s、TX 1491.13 s）は legacy 値（PA 918.67 s、TX 1482.68 s）より遅いため、legacy 値の採用は分母を小さくし、speedup を過小方向に見積もる。この選択は Chapter 6（6.3 節）、Chapter 5（5.7 節）、Appendix A（A.5 節）、および `result/main_performance/proposed_vs_pathmerge/README.md` の記述と一致する。headline の倍率にはこの final adopted 値のみを用い、掃引確認値を headline に用いることはしない。

また、この 2 グラフでは分子（GPU_Opt、checkpoint `phase_def_block_20260710`）と分母（PathMerge、checkpoint `oldtree_f05ec52_20260512`）の測定 checkpoint が異なる。この相違は 5.7 節および Appendix A（A.5 節）で明示されているとおりである。

## B.9 Recalculation and Validation

本付録の集計値は、保存された raw TSV から独立に再計算した。再計算は各グラフ・各 stage・各要求バッチについて、trial 数、median、mean、標本標準偏差（ddof=1）、min、max を算出し、pooled 集計も同一手順で求めた。丸め済みの表を計算入力とすることはせず、summary table の表示値は raw の per-trial 値から算出した値を表示桁へ丸めたものである。

再計算は同一手順で 2 回実行し、出力が完全に一致することを確認した（両回の出力ハッシュが一致）。集計に用いた一時ファイルはリポジトリ外に置いており、リポジトリへは追加していない。

検証項目と結果を Table B.13 に示す。

**Table B.13: Independent recalculation and cross-checks against the canonical records.**

| Check | Result |
|:--|:--|
| Trial row count per graph matches the raw TSV | Yes (email-EuAll 20, roadNet-PA 16, roadNet-TX 7, roadNet-CA 8, historical 325557 20; total 71) |
| All requested batches recorded | Yes (email-EuAll 9, roadNet-PA 7, roadNet-TX 3, roadNet-CA 4, historical 325557 8) |
| All effective batches recorded or explicitly marked `Not recorded` | Yes |
| Recorded clamps captured | Yes, 2 of 2 (email-EuAll 8192 to 7393; historical 325557 8192 to 6018) |
| Screening, confirmation, extension, and additional trials kept separate | Yes; pooled aggregates shown as explicitly labelled rows |
| Sample SD omitted where $N_{\mathrm{trials}}<2$ | Yes, reported as `N/A (n=1)` |
| Failed or unrecorded trials excluded, never treated as 0 s | Yes |
| Per-batch medians reproduce each graph's `SOURCE.md` | Yes, all batches |
| Adopted medians reproduce Table 6.1 and `thesis_values.tsv` | Yes (97.80, 918.67, 1482.68, 3079.72) |
| Sweep confirmation medians reproduce the values quoted in 6.3 | Yes (roadNet-PA 941.39 with $N_{\mathrm{trials}}=4$; roadNet-TX 1491.13 with $N_{\mathrm{trials}}=3$; roadNet-CA b64 3491.64) |
| Adopted mean and sample SD reproduce Table 6.2 | Yes (email-EuAll 97.90 / 0.988; roadNet-PA 923.26 / 9.593; roadNet-TX 1493.46 / 24.855; roadNet-CA 3083.85 / 25.511) |
| Default b64 medians reproduce Table 6.3 | Yes (email-EuAll 220.39; roadNet-CA 3499.03) |
| Recomputed GTEPS of the historical b32 trial 1 reproduces the tabulated value | Yes (0.8103) |
| Recalculation repeated twice with identical output | Yes |

<!-- Recalculation performed outside the repository; no aggregation artifact was added to result/ or raw_data/. -->
> Source: recomputation from the raw TSVs cited in B.3 through B.8, cross-checked against `result/tuning/pathmerge/<graph>/SOURCE.md`, `result/main_performance/proposed_vs_pathmerge/comparison.tsv`, `result/tables/final_speedup_tables.md`, and `docs/thesis/thesis_values.tsv`.

集計粒度に関する注意を 1 点記す。`result/tables/final_speedup_tables.md` は email-EuAll の b1024 を $N_{\mathrm{trials}}=4$ と要約している。これは screening 1 試行（job 2359096）と confirmation 3 試行（job 2359169）を pooled した記述統計であり、単一 job で 4 試行を測定したものではない。Appendix A（A.5 節）と本付録の Table B.3 は、この関係を stage 別（screening 1 / confirmation 3）と pooled（4）に分けて記述する。いずれの粒度でも b1024 は最良ではなく、tuned バッチの選択（b2048）は変わらない。

本付録は既存の測定値、checkpoint、batch selection、RQ status、および PathMerge の位置づけを変更しない。`raw_data/`、`failure/`、`code_snapshots/`、`scripts/`、および Chapter 1 から Chapter 11 に対する変更は行っていない。
