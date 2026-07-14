# raw_data/unsuccessful

**失敗・不成功に終わった実行の生データ**。成功データ（`../correctness/` 等）とは
物理的に分離し、混在させない。3 分類:

| 分類 | 意味 | 該当 job |
|:--|:--|:--|
| `oom/` | メモリ不足による強制終了（OOM, exit 137）。空ベクトル=証跡 | 2368269（UM b10240） |
| `early_terminated/` | 比較不一致等による意図的 fail-fast 早期打切り | 2368398（pure vs PathMerge mismatch） |
| `failed/` | 上記以外のハード失敗（現時点で該当なし） | — |

各ファイルの `Status`（`OOM` / `SUCCESS_IN_FAILED_JOB` / `COMPARISON_MISMATCH` /
`FAILED_OOM` / `EARLY_TERMINATED`）と失敗要約先（`FailureSummaryPath`）は `../MANIFEST.tsv` を参照。

失敗 job 内で参照 comparator（PathMerge 等）が個別に PASS した場合も、その job 全体は
不成功実験であるため、当該ベクトルは `Status=SUCCESS_IN_FAILED_JOB` として本ディレクトリ
配下（別 Implementation フォルダ）に保持する（`UsedInThesis=NO(failure)`）。
