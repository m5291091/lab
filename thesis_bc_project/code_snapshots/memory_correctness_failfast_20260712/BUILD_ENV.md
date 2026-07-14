# BUILD_ENV — memory_correctness_failfast_20260712

**用途**: memory-path 正確性（比較不一致 fail-fast）

- SourceSnapshotID: `memory_correctness_failfast_20260712`
- 元 commit（監査用対応表 `../_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv`）: 参照
- 実験対象: unsuccessful/early_terminated
- PBS job ID: `2368398.opbs`
- queue: regular-g(→ small-g)

## 再現コマンド（GPU 計算ノード, group gj17）

```bash
qsub scripts/run_memory_correctness.sh   # pure vs PathMerge mismatch で fail-fast (job 2368398)
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

`fix(correctness): adapt UM validation to Miyabi-G memory limit`。gpu_opt_pure_b1024 vs PathMerge の全ベクトル比較不一致（11027, max_rel≈2e-3）で fail-fast、後続構成は未実行。

## 依存関係と自己完結性 (Gate J1 静的解析)

静的解析（CMakeLists.txt / PBS スクリプトのソース・インクルード・入力参照の実在検査; GPU 実行・qsub なし）。

**snapshot 内に存在する（自己完結）**:
- `src/`（core / proposed / baseline）, `include/`（core / proposed / baseline）, `experiments/`, `scripts/`, `CMakeLists.txt`
- CMakeLists が参照するソースは **全て snapshot 内に存在**（欠損 0）。

**snapshot 外の依存（thesis_bc_project 主 tree から取得; snapshot には含めない）**:
- `tools/bandwidth_benchmark.cu`（bandwidth_benchmark ターゲット; profiling のみ）
- `third_party/cugraph/cpp/include`（vendored cuGraph API ヘッダ; `run_benchmark` のみ）
- `cugraph_bc_mini/build/libcugraph_bc_mini.a`（`run_benchmark` の cuGraph baseline; **Stage 1 ビルドで生成**）
- グラフ入力 `data/`（実行時; `data/325557_3216152`）

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
| 325557_3216152 | `data/325557_3216152` | `a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584` | 325557 | 3216152 | undirected | unknown | tools/gen_graph.py 生成 (合成, 1-indexed) |

グラフ入力は canonical path（Git 内 `data/`）から取得し、SHA256 は上表（= `graph_catalog.tsv`）で固定。全実装は無向・非重み BC（CPU は accumulation 時 /2、pathmerge adapter は最終 /2）。

<!-- GATE_J1_1_DEPENDENCIES:END -->
