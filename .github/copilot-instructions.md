# Copilot instructions

## Project context

This repository benchmarks **exact betweenness centrality (BC)** implementations on the NVIDIA GH200 / Miyabi-G environment. It is a **self-contained project** rooted at `thesis_bc_project/`: all source, a vendored cuGraph subset (`third_party/cugraph/`), datasets (`data/`), and scripts live there with no dependency on any other tree. The main deliverables are the runners `thesis_bc_project/build_miyabi/{run_benchmark,run_ablation,run_pathmerge_sweep}`.

`thesis_bc_project/README.md` is the definitive guide.

Build and runtime commands assume a **GPU compute node**, not the login node:

```bash
qsub -I -q interact-g -l select=1:ncpus=72 -l walltime=02:00:00 -W group_list=gj17
```

## Build and test commands

Run all primary commands from `thesis_bc_project/`.

### Build

```bash
cd thesis_bc_project

# First build — the cugraph_bc_mini build needs CMake >= 3.30.4, so install a newer one first
uv tool install cmake                 # or: AUTO_INSTALL_CMAKE=1 (installs via pip)
bash scripts/build_miyabi_interactive.sh

# Incremental rebuild
bash scripts/build_miyabi_interactive.sh

# Reconfigure if the cache is stale
CLEAN_CACHE=1 bash scripts/build_miyabi_interactive.sh
```

The unified build script is the authoritative entrypoint. It:

1. Builds `cugraph_bc_mini` in `thesis_bc_project/cugraph_bc_mini/build/` (produces `libcugraph_bc_mini.a`)
2. Builds `run_benchmark`, `run_ablation`, `run_pathmerge_sweep`, and `bandwidth_benchmark` in `thesis_bc_project/build_miyabi/`

`run_ablation` and `run_pathmerge_sweep` do not depend on cuGraph, so they can be built without Stage 1.

### Single-test / smoke-test commands

```bash
cd thesis_bc_project/build_miyabi

# Smallest quick smoke test
./run_benchmark sequential ../data/chain_200

# Single implementation on the main small benchmark graph
./run_benchmark cugraph_bc ../data/benchmark_7000_41459

# Compare per-vertex BC output between implementations
./run_benchmark gpu_opt    ../data/benchmark_7000_41459 --dump-bc > bc_gpu.txt
./run_benchmark cugraph_bc ../data/benchmark_7000_41459 --dump-bc > bc_cugraph.txt
diff bc_gpu.txt bc_cugraph.txt

# Ablation (H = hybrid BFS, W = warp-cooperative accumulation, A = async 2-stream init)
./run_ablation ../data/benchmark_7000_41459 all      # all 8 configs; also full / baseline / H1W0A0 ...

# PathMerge batch-size sweep (no 64 cap)
./run_pathmerge_sweep ../data/benchmark_7000_41459
```

`scripts/smoke_test.sh` is also available.

### Full benchmark and analysis workflows

```bash
cd thesis_bc_project

# Preview benchmark commands without running them
DRY_RUN=1 SKIP_BUILD=1 bash scripts/run_benchmark_small.sh

# Submit PBS jobs
qsub scripts/run_benchmark_small.sh
qsub scripts/run_benchmark_medium.sh
qsub scripts/run_benchmark_large.sh
qsub scripts/run_ablation.sh
qsub scripts/run_pathmerge_sweep.sh
qsub scripts/run_um_oversubscribe_experiment.sh

# Optional gpu_opt / gpu_opt_pure_chunked comparison reruns
qsub scripts/run_benchmark_small_gpu_opt_compare.sh
qsub scripts/run_benchmark_medium_gpu_opt_compare.sh
qsub scripts/run_benchmark_large_gpu_opt_compare.sh
qsub scripts/run_um_oversubscribe_gpu_opt.sh
qsub scripts/run_um_oversubscribe_gpu_opt_pure_chunked.sh

# Postprocess completed benchmark outputs
python3 scripts/statistical_analysis.py \
    --results build_miyabi/result_benchmark_*/results.tsv \
    --phases build_miyabi/result_benchmark_*/phase_timing.log \
    --oversubscribe build_miyabi/result_um_oversubscribe/oversubscribe_results.tsv \
    --outdir ./thesis_figures
```

## High-level architecture

- `thesis_bc_project/experiments/run_benchmark.cu` is the main CLI entrypoint. It resolves graph paths by walking up ancestor directories for a matching `data/` folder, loads a `Graph` once (`src/core/graph.cpp`), then reuses that same `Graph` for the selected implementation(s) via a dispatch table (CLI key → label → function).
- `include/proposed/brandes_gpu.hpp` and `include/baseline/brandes_baseline.hpp` declare every implementation with the shared signature `std::vector<double>(Graph&)`, which lets `run_brandes()` (in `src/core/runner.cpp`) time implementations uniformly and run `all` with shared reporting.
- `run_brandes()` defines the external contract consumed by scripts: stdout is the machine-readable result stream, while stderr carries phase timing, max-BC diagnostics, and progress messages.
- The code uses a **3-layer split**: data load (`src/core/graph.cpp`), host control (`src/{proposed,baseline}/*.cu`), and templated CUDA kernels (`include/proposed/brandes_kernels.cuh`). Ablation toggles kernel features at compile time via templates, not in-kernel branches (no branch-divergence overhead).
- `thesis_bc_project/CMakeLists.txt` builds the runners. It links `cugraph_bc_mini` as an IMPORTED static library (`cugraph_bc_mini/build/libcugraph_bc_mini.a`) and takes cuGraph API headers from the vendored `third_party/cugraph/cpp/include`. The `baseline` library is cuGraph-free; only `run_benchmark` (via `baseline_cugraph`) needs the mini lib.
- `cugraph_bc_mini/CMakeLists.txt` compiles only the BC-related cuGraph sources plus the minimum RAPIDS dependencies, pinning rapids-cmake dependency resolution to `v26.04.00`.
- The Galliot path-merging baseline is vendored as `src/baseline/galliot.cu` + `galliot_kernels.cu` (compiled directly into the runners); `src/baseline/pathmerge.cu` is the adapter layer from this repo's `Graph` representation to Galliot's CSR layout.
- `data/` stores graph datasets in the repo's CSR text format (bundled small/medium graphs; large SNAP graphs are fetched via `tools/download_snap_graphs.sh` into `data/snap/`, which is gitignored).

## Key conventions

- `thesis_bc_project/` is the whole, self-contained project. The old `mylab/research/`, top-level `cugraph/`, `path-merging-bc/`, and the `data/` submodule no longer exist — they were consolidated into `thesis_bc_project/` (cuGraph → `third_party/cugraph/`, path-merging → `src/baseline/`, graphs → `data/`). Archived prior thesis results live in `legacy_results_miyabi/result_paper/`.
- Preserve the stdout/stderr contract from `run_benchmark.cu` / `runner.cpp`:
  - normal stdout: `Impl<TAB>Graph<TAB>Time_sec<TAB>GTEPS`
  - `--dump-bc` stdout: header line plus `node_idx<TAB>bc_value`
  - stderr: phase timing and `Maximum Betweenness Centrality` summary lines
  Benchmark scripts parse those streams directly into `results.tsv`, `phase_timing.log`, and `max_bc.tsv`.
- Graph files are 3-line CSR text files. `Graph::readGraph()` reads:
  1. `n_nodes n_edges`
  2. `ptr[0..n_nodes]`
  3. `adj[0..2*n_edges-1]`
- When an implementation or adapter needs the actual adjacency-array length / directed nnz, prefer `R[n_nodes]` / `offsets[n_nodes]` over recomputing from `edgeCount`.
- All implementations are aligned around **undirected, unweighted** BC output. CPU implementations divide by 2 during accumulation, and the path-merging adapter explicitly divides its final BC values by 2 so outputs stay comparable with the other modes.
- The GH200 memory strategy is intentional, not accidental:
  - `gpu_opt` (`src/proposed/host_um.cu`) uses managed memory and can intentionally oversubscribe HBM3
  - `gpu_opt_pure` (`host_pure.cu`) uses explicit `cudaMalloc` / `cudaMemcpy`; `gpu_opt_pure_chunked` (`host_chunked.cu`) chunks the working set
  - `cugraph_bc` also uses managed memory via RMM's `managed_memory_resource`
  Do not "simplify" those differences away unless the task is specifically about changing the experiment design.
- Runtime tuning knobs are part of normal workflow, not dead code:
  - `BC_BATCH_OVERRIDE` for `gpu_opt` / `gpu_opt_pure` / ablation
  - `CUGRAPH_BC_MAX_SOURCES_PER_BATCH` for `cugraph_bc`
  - `PATHMERGE_BC_BATCH_SIZE` for `pathmerge_bc` (default 64, no hard upper bound)
