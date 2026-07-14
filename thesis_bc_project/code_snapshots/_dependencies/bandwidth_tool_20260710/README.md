# Dependency: bandwidth_tool_20260710

**用途**: `bandwidth_benchmark` ターゲット（HBM3 / NVLink-C2C 帯域計測; profiling 実験）。

## 内容

- `tools/bandwidth_benchmark.cu` — 帯域計測ツール（`CUDA::cudart` のみに依存, cuGraph 非依存）。
  blob = `c0d4945…`（thesis 6 checkpoint で同一）。

主 `CMakeLists.txt` は `add_executable(bandwidth_benchmark tools/bandwidth_benchmark.cu)` を
**無条件**に定義するため、Stage 2 の configure には本ファイルの存在が必要（実行は profiling のみ）。

## 参照するスナップショット / ターゲット

| Snapshot | UsedByTarget |
|:--|:--|
| small_correctness_20260712 | bandwidth_benchmark (Stage2 configure) |
| phase_def_block_20260710 | bandwidth_benchmark (profiling job 2359175 で実行) |
| memory_correctness_20260712 | bandwidth_benchmark (Stage2 configure) |
| memory_diagnostic_20260713 | bandwidth_benchmark (Stage2 configure) |
| memory_correctness_oom_20260712 | bandwidth_benchmark (Stage2 configure) |
| memory_correctness_failfast_20260712 | bandwidth_benchmark (Stage2 configure) |

oldtree `f05ec52` は本ターゲットを持たない（該当なし）。

## 検証

```bash
cd code_snapshots/_dependencies/bandwidth_tool_20260710
sha256sum -c SHA256SUMS
```
