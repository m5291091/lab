# BUILD_ENV — oldtree_f05ec52_20260512

**用途**: UM オーバーサブスクリプション（旧 mylab/research ツリー）

- SourceSnapshotID: `oldtree_f05ec52_20260512`
- 元 commit（監査用対応表 `../_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv`）: 参照
- 実験対象: memory_scalability（feasibility）; main_performance/seven_implementations（legacy, 近似）
- PBS job ID: `UMv2（旧ツリー; PBS job ID は当時ログに個別記録なし → not_recorded）`
- queue: （旧ツリー測定; queue 記録なし）

## 再現コマンド（GPU 計算ノード, group gj17）

```bash
qsub scripts/run_um_oversubscribe_experiment.sh
qsub scripts/run_um_oversubscribe_gpu_opt.sh
qsub scripts/run_um_oversubscribe_gpu_opt_pure_chunked.sh
```

## ビルド条件（旧 mylab/research ツリー, フラット構成）

旧ツリーは `src/include/experiments` 分割前のフラット構成（`brandes_*.cu` 直下 + `main.cpp`）。
ファイル名対応は `result/provenance/provenance.md` の対応表を参照:
`brandes_gpu_opt.cu`→`host_um.cu`, `brandes_gpu_opt_pure.cu`→`host_pure.cu`,
`brandes_gpu_opt_pure_chunked.cu`→`host_chunked.cu`, `main.cpp`→`experiments/run_benchmark.cu`。

```bash
cd mylab/research
bash scripts/build_miyabi_interactive.sh
```

- CUDA アーキテクチャ: sm_90（GH200）。

## コンパイラ・CUDA 環境（旧ツリー測定, 限定）

- 測定 commit `f05ec52`（2026-05-12, 旧 `mylab/research`）。同一 GH200 系ハードウェア。
- **メモリサイジング/OOM ロジックは checkpoint `88faffa` と文字単位同一**（`result/provenance/um_code_diff_audit.md`）。
- **時間値は最新 block 性能値として非採用**（旧セッションの driver/CMake/thermal/co-tenancy は独立再検証していない）。
- **feasibility（SUCCESS/OOM 傾向）のみ限定採用**。文字単位同一のメモリサイジング + 同一割当により、
  pure が先に OOM / UM がより大きなバッチまで到達 / chunked が最大、という傾向を再利用（境界の 88faffa 再実測は未実施）。
- 正確性: um_experiment_*.log の各 run で Max BC = `39343001000.11`（独立参照 PathMerge と一致）。

## 依存関係と自己完結性 (Gate J1 静的解析)

静的解析（CMakeLists.txt / PBS スクリプトのソース・インクルード・入力参照の実在検査; GPU 実行・qsub なし）。

**snapshot 内に存在する（自己完結）**:
- `src/`（core / proposed / baseline）, `include/`（core / proposed / baseline）, `experiments/`, `scripts/`, `CMakeLists.txt`
- CMakeLists が参照するソースは **全て snapshot 内に存在**（欠損 0）。

**snapshot 外の依存（thesis_bc_project 主 tree から取得; snapshot には含めない）**:
- `tools/bandwidth_benchmark.cu`（bandwidth_benchmark ターゲット; profiling のみ）
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
| `cugraph_bc_subset_20260710` | `code_snapshots/_dependencies/cugraph_bc_subset_20260710/` | `bd0bfa117a0b118fca63448362f2ade6b3e5be5ef6de395f55543a1abd67087d` | cuGraph baseline（vendored subset のみ; mini CMake は snapshot 内同梱; 保持 UM 実験は gpu_opt で cuGraph 非使用） |

- `DependencySHA256` = 各 dependency の `SHA256SUMS` の SHA256（manifest digest）。個別ファイルの SHA256 は dependency 内 `SHA256SUMS` / `SOURCE_MANIFEST.tsv` を参照。
- vendored cuGraph subset（`third_party/cugraph`, tree `eb339d4`）は **全 7 checkpoint で同一内容**（抽出 commit `88faffa` の git blob と照合済）。

### canonical graph data（`result/datasets/graph_catalog.tsv` と一致; snapshot へは非複製）

| Graph | canonical path | SHA256 | Nodes | Edges | UsedAsDirected | Symmetrized | Preprocessing |
|:--|:--|:--|:--|:--|:--|:--|:--|
| 325557_3216152 | `data/325557_3216152` | `a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584` | 325557 | 3216152 | undirected | unknown | tools/gen_graph.py 生成 (合成, 1-indexed) |

グラフ入力は canonical path（Git 内 `data/`）から取得し、SHA256 は上表（= `graph_catalog.tsv`）で固定。全実装は無向・非重み BC（CPU は accumulation 時 /2、pathmerge adapter は最終 /2）。

<!-- GATE_J1_1_DEPENDENCIES:END -->
