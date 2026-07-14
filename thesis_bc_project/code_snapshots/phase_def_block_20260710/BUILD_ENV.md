# BUILD_ENV — phase_def_block_20260710

**用途**: 主実験 checkpoint（提案 block・PathMerge 掃引・kernel 選択・ablation・profiling）

- SourceSnapshotID: `phase_def_block_20260710`
- 元 commit（監査用対応表 `../_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv`）: 参照
- 実験対象: main_performance/proposed_variants; tuning/pathmerge; tuning/kernel_selection; ablation; profiling; phase_breakdown; correctness/pathmerge_tuned
- PBS job ID: `2356120;2357334-2357337;2355000;2355001;2359080;2359081;2359096;2359169;2360072;2360073;2361040;2361041;2362006;2354329;2354330;2354994;2354999;2359175`
- queue: regular-g / small-g

## 再現コマンド（GPU 計算ノード, group gj17）

```bash
qsub scripts/run_benchmark_targeted.sh   # proposed_variants (2356120,2357334-2357337)
qsub scripts/run_pathmerge_sweep.sh      # pathmerge tuned
qsub scripts/run_kernel_selection.sh     # forced shared/block (2354329,2354330)
qsub scripts/run_ablation.sh             # H/W/A 8構成 (2354994,2354999)
qsub scripts/run_profiling.sh            # nsys + 帯域 (2359175)
```

## ビルド条件（thesis_bc_project 二段ビルド）

GPU 計算ノード上で実行（ログインノード不可）。まず新しめの CMake を用意（cugraph_bc_mini は CMake ≥3.30.4 を要求）。

```bash
cd thesis_bc_project
uv tool install cmake            # or: AUTO_INSTALL_CMAKE=1
bash scripts/build_miyabi_interactive.sh
```

- Stage 1: `cugraph_bc_mini/build/libcugraph_bc_mini.a`（rapids-cmake を v26.04.00 に固定）
- Stage 2: `build_miyabi/{run_benchmark,run_ablation,run_pathmerge_sweep,bandwidth_benchmark}`
- `run_ablation` / `run_pathmerge_sweep` は cuGraph 非依存（Stage 1 不要）。`run_benchmark` のみ Stage 1 必要。
- CUDA アーキテクチャ: **sm_90**（GH200）。

## コンパイラ・CUDA 環境（checkpoint 実測, `result/environment/environment.md`）

| 項目 | 値 |
|:--|:--|
| GPU | NVIDIA GH200 120GB（HBM3 97871 MiB / total≈102 GB, free≈101.4 GB） |
| GPU アーキテクチャ | sm_90 |
| NVIDIA driver | 595.58.03 |
| CUDA (nvcc) | release 13.0, V13.0.48 |
| C++ コンパイラ | g++ (GCC) 11.4.1 |
| CMake | 4.3.4（`~/.local/bin/cmake`; cugraph_bc_mini は ≥3.30.4 必須） |
| nsys | 2025.5.1.121 |
| スケジューラ / group | PBS (Miyabi-G), group `gj17` |
| 集計 | median / warmup なし（新規測定, ベンチスクリプトは全 TRIALS 記録） |

## 備考

この checkpoint は複数実験の実験時コードを兼ねる（`1ae987c` 常時 block 化後）。各実験の詳細は該当 `result/*/SOURCE.md`。

## 依存関係と自己完結性 (Gate J1 静的解析)

静的解析（CMakeLists.txt / PBS スクリプトのソース・インクルード・入力参照の実在検査; GPU 実行・qsub なし）。

**snapshot 内に存在する（自己完結）**:
- `src/`（core / proposed / baseline）, `include/`（core / proposed / baseline）, `experiments/`, `scripts/`, `CMakeLists.txt`
- CMakeLists が参照するソースは **全て snapshot 内に存在**（欠損 0）。

**snapshot 外の依存（thesis_bc_project 主 tree から取得; snapshot には含めない）**:
- `tools/bandwidth_benchmark.cu`（bandwidth_benchmark ターゲット; profiling のみ）
- `third_party/cugraph/cpp/include`（vendored cuGraph API ヘッダ; `run_benchmark` のみ）
- `cugraph_bc_mini/build/libcugraph_bc_mini.a`（`run_benchmark` の cuGraph baseline; **Stage 1 ビルドで生成**）
- グラフ入力 `data/`（実行時; 実験スクリプトの `GRAPHS_STR`/env で指定（`data/snap/*`, `data/benchmark_*`, 合成グラフ））

**自己完結性の判定**:
- `run_ablation` / `run_pathmerge_sweep`: **cuGraph 非依存**。snapshot の `src/`+`include/`+`experiments/`+`CMakeLists.txt` と `data/` のみでビルド・実行可能（Git 履歴不要）。
- `run_benchmark`: 上記に加え **外部の `third_party/cugraph` + `cugraph_bc_mini`（Stage 1）** が必要。これらは vendored 依存であり、サイズ上 snapshot には複製せず主 tree を参照する（「完全自己完結」とは称さない）。
- CUDA/GH200 実行環境（sm_90, nvcc 13.0, GH200）は上記「コンパイラ・CUDA 環境」表の通り。

<!-- GATE_J1_1_DEPENDENCIES:BEGIN -->

## 依存固定 (Gate J1.1)

履歴削除後も実験時と同一版を特定できるよう、外部コード依存を `code_snapshots/_dependencies/<DependencyID>/`（内容+用途名, commit SHA非依存）へ一度だけ固定し、本 snapshot から参照する。各 dependency は `SOURCE_MANIFEST.tsv` / `SHA256SUMS` / `README.md` を持つ。

### コード依存

| DependencyID | DependencyPath | DependencySHA256 | UsedByTarget |
|:--|:--|:--|:--|
| `cugraph_bc_subset_20260710` | `code_snapshots/_dependencies/cugraph_bc_subset_20260710/` | `bd0bfa117a0b118fca63448362f2ade6b3e5be5ef6de395f55543a1abd67087d` | run_benchmark（cugraph_bc baseline; Stage1 libcugraph_bc_mini.a + Stage2 link） |
| `bandwidth_tool_20260710` | `code_snapshots/_dependencies/bandwidth_tool_20260710/` | `b2d1edd69c414fbb765acaffc78b81784e8bafa7eef43a8c2162b61249c6080c` | bandwidth_benchmark（Stage2 configure に必須; 実行は profiling のみ） |

- `DependencySHA256` = 各 dependency の `SHA256SUMS` の SHA256（manifest digest）。個別ファイルの SHA256 は dependency 内 `SHA256SUMS` / `SOURCE_MANIFEST.tsv` を参照。
- vendored cuGraph subset（`third_party/cugraph`, tree `eb339d4`）は **全 7 checkpoint で同一内容**（抽出 commit `88faffa` の git blob と照合済）。

### canonical graph data（`result/datasets/graph_catalog.tsv` と一致; snapshot へは非複製）

| Graph | canonical path | SHA256 | Nodes | Edges | UsedAsDirected | Symmetrized | Preprocessing |
|:--|:--|:--|:--|:--|:--|:--|:--|
| email-EuAll | `data/snap/email-EuAll` | `62799296e5a01b1a6ebd29c1b17702a2415d718991fbafd18e3a3fd8d0a52ece` | 265009 | 364481 | undirected | yes | SNAP download + 無向化 (tools/download_snap_graphs.sh) |
| roadNet-PA | `data/snap/roadNet-PA` | `60bd70115d2c7e1ea74642e916666dc8d85e7ab85c0fd1b7dc6590617c4a5e28` | 1088092 | 1541898 | undirected | no | SNAP download (tools/download_snap_graphs.sh) |
| roadNet-TX | `data/snap/roadNet-TX` | `9bf091d2936202302265fda6f573017ae816878df373c17cdd73a2eae796a969` | 1379917 | 1921660 | undirected | no | SNAP download (tools/download_snap_graphs.sh) |
| roadNet-CA | `data/snap/roadNet-CA` | `daabf77ed106166937cacb5556a40149c0fffd0ebc641967781896e7301eac0a` | 1965206 | 2766607 | undirected | no | SNAP download (tools/download_snap_graphs.sh) |
| benchmark_7000 | `data/benchmark_7000_41459` | `4a891b4de4a0df86ef73c469f1e81b6206073e7368488e74f3ee2cec43b29ddc` | 7000 | 41459 | undirected | unknown | tools/gen_graph.py 生成 (合成ベンチ) |
| benchmark_11023 | `data/benchmark_11023_62184` | `8d1df41c579de3150a155ee9cce321784723fdb1824c0a2a160d95004d4b6e31` | 11023 | 62184 | undirected | unknown | tools/gen_graph.py 生成 (合成ベンチ) |
| 56438_300801 | `data/56438_300801` | `4668fc586e17dccbf7888bdf8a823eb4430878bc6eefc421bde797303318d0d0` | 56438 | 300801 | undirected | unknown | tools/gen_graph.py 生成 (合成ベンチ) |
| benchmark_85830 | `data/benchmark_85830.data` | `cb8884200f8099971c88bfdbc06ab8fc6133cd45fd7791091bffb985cabbbbd0` | 85830 | 241080 | undirected | unknown | tools/gen_graph.py 生成 (合成ベンチ) |
| 325557_3216152 | `data/325557_3216152` | `a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584` | 325557 | 3216152 | undirected | unknown | tools/gen_graph.py 生成 (合成, 1-indexed) |

グラフ入力は canonical path（Git 内 `data/`）から取得し、SHA256 は上表（= `graph_catalog.tsv`）で固定。全実装は無向・非重み BC（CPU は accumulation 時 /2、pathmerge adapter は最終 /2）。

<!-- GATE_J1_1_DEPENDENCIES:END -->
