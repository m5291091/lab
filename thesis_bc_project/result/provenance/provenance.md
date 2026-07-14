# provenance — 出所・コードスナップショット・コード差分監査

## 正式参照 = SourceSnapshotID（commit 非依存）

実験時コードの正式参照は **`SourceSnapshotID`**（`../../code_snapshots/<id>/`）。commit SHA は
Git 履歴を削除すると解決不能になるため、**監査目的の対応表**（下表 / `../../code_snapshots/_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv`）
にのみ保持する。生データ・コードへのアクセスに commit SHA は不要。

## SourceSnapshotID 対応表（正式参照）
| 実験 | SourceSnapshotID | code_snapshots パス |
|:--|:--|:--|
| 主実験（提案 block / PathMerge 掃引 / kernel 選択 / ablation / profiling） | `phase_def_block_20260710` | `code_snapshots/phase_def_block_20260710/` |
| small full-vector 正確性 | `small_correctness_20260712` | `code_snapshots/small_correctness_20260712/` |
| memory-path 正確性 canonical | `memory_correctness_20260712` | `code_snapshots/memory_correctness_20260712/` |
| memory-path 診断 | `memory_diagnostic_20260713` | `code_snapshots/memory_diagnostic_20260713/` |
| memory-path OOM | `memory_correctness_oom_20260712` | `code_snapshots/memory_correctness_oom_20260712/` |
| memory-path fail-fast | `memory_correctness_failfast_20260712` | `code_snapshots/memory_correctness_failfast_20260712/` |
| UM オーバーサブスクリプション（旧ツリー）+ legacy seven_implementations | `oldtree_f05ec52_20260512` | `code_snapshots/oldtree_f05ec52_20260512/` |

## 監査用 commit 対応表（Git 履歴削除後は監査目的のみ・アクセス必須ではない）
| 役割 | 元 commit | 日付 | SourceSnapshotID |
|:--|:--|:--|:--|
| 実験 checkpoint（実験コード確定） | `88faffa` | 2026-07-11 | `phase_def_block_20260710` |
| Gate B 開始時 base commit（整理作業の基点） | `4b41eab` | 2026-07-12 | （整理作業; スナップショット対象外） |
| 副次(B) UM 測定 commit（旧ツリー） | `f05ec52` | 2026-05-12 | `oldtree_f05ec52_20260512` |
| 常時 block 化 | `1ae987c` | 2026-07-10 | （`phase_def_block_20260710` に含まれる） |
| thesis_bc_project 集約/独立化 | `834d1d9` / `e7b86de` | 2026-07-04 | （集約作業; スナップショット対象外） |
| small full-vector 正確性 | `e32b03e9` | 2026-07-12 | `small_correctness_20260712` |
| memory-path canonical / 診断 / OOM / fail-fast | `ac2b409` / `43d1cf5` / `6282798` / `29d28c50` | 2026-07-12〜13 | `memory_correctness_20260712` / `memory_diagnostic_20260713` / `memory_correctness_oom_20260712` / `memory_correctness_failfast_20260712` |

`88faffa` は base commit `4b41eab` の祖先。`88faffa`↔`4b41eab` で `src/proposed/host_um.cu` 等は差分ゼロ。

## commit 時系列（主要）
```
f05ec52 (05-12) UM オーバーサブスクリプション性能データ確定（旧ツリー）
   └─ 834d1d9/e7b86de (07-04) thesis_bc_project 集約・3層分割・独立化
1ae987c (07-10) 常時 block 化
caca85b (07-11) phaseDEF: block 再計測・PathMerge掃引・最終表・profiling
88faffa (07-11) checkpoint
4b41eab (07-12) Gate B 開始時 base commit: PathMerge tuned TX/CA・アーカイブ自己完結化
```

## ファイル名対応表（旧ツリー → 現行）
| 旧（mylab/research） | 現行（thesis_bc_project） |
|:--|:--|
| `brandes_gpu_opt.cu` | `src/proposed/host_um.cu` |
| `brandes_gpu_opt_pure.cu` | `src/proposed/host_pure.cu` |
| `brandes_gpu_opt_pure_chunked.cu` | `src/proposed/host_chunked.cu` |
| `main.cpp` | `experiments/run_benchmark.cu` |
| `brandes.h` | `include/proposed/brandes_gpu.hpp` |
| （インライン kernel） | `include/proposed/brandes_kernels.cuh`（3層分割で抽出） |

## stale 文書（旧パス参照・歴史的経緯）
- `lab_evaluation_v2.md` / `pre_experiment_final_check.md`（トップレベル）は旧 `lab/mylab/research/` を参照する **ARCHIVED（歴史的）** 文書。現行コードは上表の対応で読む。UM 実験の設計背景の記録として保持。

## コード差分監査（副次B）
- `um_code_diff_audit.md` を参照（f05ec52 ↔ 88faffa のメモリ/OOM/BC ロジック検証）。
