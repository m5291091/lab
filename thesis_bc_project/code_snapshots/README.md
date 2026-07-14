# code_snapshots — 実験時コードのスナップショット

Git 履歴（commit）を削除した後でも **実験時のコードを特定・再構築できる** よう、各実験の
実験時 checkpoint のソース・使用スクリプト・ビルド条件を凍結保存する。

- **正式な参照** = `SourceSnapshotID`（各サブディレクトリ名）。`raw_data/MANIFEST.tsv` /
  `result/*/SOURCE.md` / `result/MIGRATION_MAP.tsv` / `result/coverage_matrix.tsv` は
  この `SourceSnapshotID` を参照する。
- 元 commit SHA は **監査用対応表** `_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv` にのみ保持する（生データ／コード
  アクセスの必須条件ではない）。

## 自己完結性（Gate J1.1: 依存固定後 = DEPENDENCIES_CAPTURED）

各 snapshot は **実験時ソース**（`src/` + `include/` + `experiments/` + `scripts/` + `CMakeLists.txt`）を凍結保持し、
CMakeLists が参照するソースは全て snapshot 内に存在する（欠損 0）。Gate J1 で「snapshot 外」としていた外部コード依存は、
**Gate J1.1 で `_dependencies/<DependencyID>/` へ実体固定**した（履歴削除後も実験時と同一版を特定可能）:

| DependencyID | 内容 | UsedByTarget | 参照 snapshot |
|:--|:--|:--|:--|
| `cugraph_bc_subset_20260710` | vendored cuGraph subset（`third_party/cugraph`, tree `eb339d4`）+ `cugraph_bc_mini/CMakeLists.txt`（`c286a48`） | run_benchmark（cugraph_bc baseline; Stage1+2） | 全 7（cuGraph subset は 7 checkpoint 同一） |
| `bandwidth_tool_20260710` | `tools/bandwidth_benchmark.cu`（`c0d4945`, CUDA のみ） | bandwidth_benchmark（Stage2 configure; 実行は profiling） | thesis 6 件 |

- vendored cuGraph subset（tree `eb339d4`）は **全 7 checkpoint で git tree 同一**（e32b03e9/88faffa/ac2b409/43d1cf5/6282798/29d28c5/f05ec52; oldtree は top-level `cugraph/` として同一）。抽出は commit `88faffa` の git blob と照合済。
- 各 dependency は `SOURCE_MANIFEST.tsv` / `SHA256SUMS` / `README.md` を持つ。各 snapshot の `BUILD_ENV.md`（`GATE_J1_1_DEPENDENCIES` 節）に `DependencyID` / `DependencyPath` / `DependencySHA256` / `UsedByTarget` を記録。
- グラフ入力 `data/` は canonical（`../result/datasets/graph_catalog.tsv`）を参照し、各 `BUILD_ENV.md` に path / SHA256 / Nodes / Edges / directed / symmetrization を固定（snapshot へは非複製）。

**分類結果**（`SELF_CONTAINMENT.tsv`; 実検証 = snapshot 内容 SHA + dependency 実体 SHA + canonical data SHA の全照合）:
全 7 snapshot = **`DEPENDENCIES_CAPTURED`**（`INCOMPLETE` 0）。`run_ablation` / `run_pathmerge_sweep` は cuGraph 非依存で
snapshot + `data/` のみで可、`run_benchmark` / `bandwidth_benchmark` は上記固定済 dependency + canonical data で再構築可能。
GPU 実行・qsub は行っていない（静的検査 + SHA256 照合のみ）。

## スナップショット一覧

| SourceSnapshotID | 元 commit | 日付 | 実験 |
|:--|:--|:--|:--|
| `small_correctness_20260712` | e32b03e9 | 2026-07-12 | small full-vector 正確性（job 2367583） |
| `phase_def_block_20260710` | 88faffa | 2026-07-11 | 主実験: proposed_variants / pathmerge 掃引 / kernel 選択 / ablation / profiling / phase_breakdown（実施 07-10〜11） |
| `memory_correctness_20260712` | ac2b409 | 2026-07-12 | memory-path 正確性 canonical（job 2368587） |
| `memory_diagnostic_20260713` | 43d1cf5 | 2026-07-13 | memory-path 診断 T-RESET/T-NSEFF（job 2369632） |
| `memory_correctness_oom_20260712` | 6282798 | 2026-07-12 | memory-path OOM（job 2368269） |
| `memory_correctness_failfast_20260712` | 29d28c50 | 2026-07-12 | memory-path fail-fast（job 2368398） |
| `oldtree_f05ec52_20260512` | f05ec52 | 2026-05-12 | UM オーバーサブスクリプション feasibility + legacy seven_implementations（旧 `mylab/research` ツリー） |

## 各スナップショットの中身

thesis_bc_project ベース（6 個）:

```
<id>/
├── src/               # ホスト制御 + CUDA（3層分割）
├── include/           # ヘッダ + カーネルテンプレート
├── experiments/       # runner エントリポイント（run_benchmark.cu 等）
├── CMakeLists.txt     # ビルド定義
├── scripts/           # 当該実験で使用した PBS 実行 / ビルド / 集計スクリプト
├── BUILD_ENV.md       # ビルド条件・コンパイラ/CUDA 環境・再現コマンド
├── SOURCE_MANIFEST.tsv# 各ファイルの出所（元 commit・元パス・SHA256・役割・git blob 整合性）
└── SHA256SUMS         # コードファイルの SHA256（メタデータ 3 ファイルは除く）
```

旧ツリー `oldtree_f05ec52_20260512` はフラット構成（`brandes_*.cu` 直下 + `main.cpp`）。
新旧ファイル名対応は `../result/provenance/provenance.md`。

固定済み外部コード依存（Gate J1.1）:

```
_dependencies/
├── cugraph_bc_subset_20260710/   # third_party/cugraph（eb339d4）+ cugraph_bc_mini/CMakeLists.txt
│   ├── SOURCE_MANIFEST.tsv  SHA256SUMS  README.md
├── bandwidth_tool_20260710/      # tools/bandwidth_benchmark.cu
│   ├── SOURCE_MANIFEST.tsv  SHA256SUMS  README.md
└── (各 snapshot の BUILD_ENV.md から DependencyID で参照)
```

## 出所・整合性

- 各ファイルは対応 commit から `git archive` で抽出し、内容 SHA256 を **その commit の git blob と
  照合済み**（`SOURCE_MANIFEST.tsv` の `GitBlobIntegrity=MATCH`, 全 220 ファイル一致）。
- ビルド生成物（ELF/`.o`/`.a`/`.so`）、CMake 生成物、`__pycache__`/`*.pyc` は **含めない**。
- 使用スクリプトは実験に必要なもの（実行 PBS + ビルド + 集計）に限定して抽出（フル scripts/ は非同梱）。

## 検証

```bash
cd thesis_bc_project/code_snapshots/<id>
sha256sum -c SHA256SUMS
```
