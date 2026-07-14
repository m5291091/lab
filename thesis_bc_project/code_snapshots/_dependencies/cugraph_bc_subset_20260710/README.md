# Dependency: cugraph_bc_subset_20260710

**用途**: `run_benchmark` の cuGraph BC ベースライン（`cugraph_bc`）をビルド・実行するための
vendored cuGraph サブセット + BC 専用ミニビルドの CMake レシピ。

## 内容

- `third_party/cugraph/` — vendored cuGraph サブセット（BC 関連のみ, upstream RAPIDS）。
  Git tree = `eb339d4ae02a0d6f6d9f75658f74fcae59079666`。**全 7 checkpoint で同一**
  （thesis 6 件の `thesis_bc_project/third_party/cugraph`、および oldtree `f05ec52` の
  top-level `cugraph/` すべて同一 tree）。`run_benchmark` は `cpp/include` を API として使用。
- `cugraph_bc_mini/CMakeLists.txt` — BC 専用ミニビルドのレシピ（rapids-cmake を v26.04.00 に固定,
  RMM/RAFT/cuCo/CCCL/spdlog/NVTX3/rapids_logger を CPM 取得）。blob = `c286a48…`
  （thesis 6 件で同一）。oldtree `f05ec52` は別版（SHA256 `67b8dee…`, git blob `f101978`）を **snapshot 内に同梱**
  （`oldtree_f05ec52_20260512/cugraph_bc_mini/CMakeLists.txt`; SOURCE_MANIFEST.tsv で SHA256 照合済）。

## 参照するスナップショット / ターゲット

| Snapshot | UsedByTarget |
|:--|:--|
| small_correctness_20260712 | run_benchmark (Stage1 libcugraph_bc_mini.a + Stage2 link) |
| phase_def_block_20260710 | run_benchmark |
| memory_correctness_20260712 | run_benchmark |
| memory_diagnostic_20260713 | run_benchmark |
| memory_correctness_oom_20260712 | run_benchmark |
| memory_correctness_failfast_20260712 | run_benchmark |
| oldtree_f05ec52_20260512 | cuGraph baseline（`../../cugraph` 参照; mini CMake は snapshot 内同梱; 保持 UM 実験は gpu_opt のみで cuGraph 非使用） |

## ビルド

Stage 1（`cugraph_bc_mini/CMakeLists.txt` + `third_party/cugraph`）で
`cugraph_bc_mini/build/libcugraph_bc_mini.a` を生成 → `run_benchmark` が IMPORTED static lib として link。
詳細は各 snapshot の `BUILD_ENV.md` と `scripts/build_cugraph_bc_mini.sh`。

## 検証

```bash
cd code_snapshots/_dependencies/cugraph_bc_subset_20260710
sha256sum -c SHA256SUMS
```
`SOURCE_MANIFEST.tsv` の各 SHA256 は commit `88faffa` の対応 git blob と一致（抽出時照合済; 全 checkpoint 同一内容）。
