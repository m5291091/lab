# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project context

This repository compares multiple **Exact Betweenness Centrality (BC)** implementations on the NVIDIA GH200 (Miyabi-G supercomputer, sm_90). It is a **self-contained project** rooted at `thesis_bc_project/`: all source, a vendored cuGraph subset, datasets, and scripts live under that directory and it builds/runs with no dependency on any other tree.

All commands assume the Miyabi-G environment (PBS scheduler, group `gj17`). Build & run must happen on a GPU compute node, **not** on the login node.

`thesis_bc_project/README.md` is the definitive (Japanese) walkthrough; this file is a quick orientation for agents.

## Repository layout

```
lab/
├── CLAUDE.md, GEMINI.md, *.md      # top-level notes (this file, evaluations, etc.)
└── thesis_bc_project/              # THE project — self-contained, no external deps
    ├── CMakeLists.txt              # top-level build definition
    ├── include/                    # headers (host API + kernel templates)
    │   ├── core/                   #   graph.hpp, common.hpp, runner.hpp
    │   ├── proposed/               #   brandes_gpu.hpp, ablation_config.hpp, brandes_kernels.cuh
    │   └── baseline/               #   brandes_baseline.hpp, galliot headers
    ├── src/                        # implementations (3-layer split)
    │   ├── core/                   #   graph.cpp (CSR load), runner.cpp (timing/report)
    │   ├── proposed/               #   host_um / host_pure / host_chunked / host_ablation (.cu)
    │   └── baseline/               #   sequential / omp / gpu_unopt / cugraph_bc / galliot / pathmerge
    ├── experiments/                # runner entrypoints
    │   ├── run_benchmark.cu        #   all impls + correctness (--dump-bc)
    │   ├── run_ablation.cu         #   ablation (H/W/A compile-time flags)
    │   └── run_pathmerge_sweep.cu  #   PathMerge batch-size sweep
    ├── third_party/cugraph/        # vendored cuGraph subset (BC-related); cpp/include is the API
    ├── cugraph_bc_mini/            # BC-only mini cuGraph build → libcugraph_bc_mini.a
    ├── scripts/                    # build / benchmark / analysis scripts (PBS jobs)
    ├── tools/                      # bandwidth benchmark, graph generators, SNAP downloader
    ├── data/                       # bundled small/medium graphs (CSR text); data/README.md is the catalog
    ├── build_miyabi/               # build output (gitignored)
    └── legacy_results_miyabi/      # archived thesis-ready results (result_paper) from the old tree
```

**3-layer separation (HPC/CUDA best practice):** data load (`src/core/graph.cpp`), host control (`src/{proposed,baseline}/*.cu`: memory management, kernel launch, timing), CUDA kernels (`include/proposed/brandes_kernels.cuh`, templated). Every implementation has the shared signature `std::vector<double>(Graph&)`, so `run_brandes()` (in `src/core/runner.cpp`) times them uniformly.

## Build

Two-stage build via one script. **Must run on a GPU compute node.** The `cugraph_bc_mini` build needs CMake ≥3.30.4 (Miyabi default is older), so install a newer CMake first.

Get an interactive GPU node:
```bash
qsub -I -q interact-g -l select=1:ncpus=72 -l walltime=02:00:00 -W group_list=gj17
```

From `thesis_bc_project/`:
```bash
# First build — install a newer CMake first (uv recommended)
uv tool install cmake                     # or: AUTO_INSTALL_CMAKE=1 (installs via pip)
bash scripts/build_miyabi_interactive.sh

# Incremental rebuild
bash scripts/build_miyabi_interactive.sh

# When the CMake cache breaks
CLEAN_CACHE=1 bash scripts/build_miyabi_interactive.sh
```

Stage 1 produces `cugraph_bc_mini/build/libcugraph_bc_mini.a` (~10 min cold, skipped when up-to-date). Stage 2 produces the runners under `build_miyabi/`:
- `run_benchmark` — all impls + correctness verification
- `run_ablation` — ablation experiment (**no** cuGraph dependency)
- `run_pathmerge_sweep` — PathMerge batch-size sweep (**no** cuGraph dependency)
- `bandwidth_benchmark` — bandwidth measurement

Useful env vars: `JOBS` (parallel build, default 8), `SKIP_BUILD=1`.

## Run

```bash
cd thesis_bc_project/build_miyabi
./run_benchmark <impl> <graph_path> [--dump-bc]
```

Implementations: `sequential`, `omp`, `gpu`, `gpu_opt`, `gpu_opt_pure`, `gpu_opt_pure_chunked`, `cugraph_bc`, `pathmerge_bc`, `all`.

`--dump-bc` writes every per-vertex BC value to stdout (diff against another impl for correctness). Without it, stdout is one TSV line: `Impl\tGraph\tTime_sec\tGTEPS`. Per-phase timing and progress go to stderr.

Graph path resolution walks up ancestor directories for a `data/` folder, so `../data/benchmark_7000_41459` and bare basenames both work.

Ablation (compile-time flags H = hybrid top-down/bottom-up BFS, W = warp-cooperative accumulation, A = async 2-stream init):
```bash
./run_ablation ../data/benchmark_7000_41459 all        # all 8 configs
./run_ablation ../data/benchmark_7000_41459 full       # H1W1A1
./run_ablation ../data/benchmark_7000_41459 baseline   # H0W0A0
```

PathMerge batch sweep (no 64 cap anymore — int2 frontier + per-source arrays, bounded only by GPU memory):
```bash
./run_pathmerge_sweep ../data/benchmark_7000_41459 [32,64,128,256,512]
```

Runtime knobs (env vars):
- `BC_BATCH_OVERRIDE` — `gpu_opt` / `gpu_opt_pure` / ablation batch size
- `PATHMERGE_BC_BATCH_SIZE` — `pathmerge_bc` batch size (default 64, no hard upper bound)
- `CUGRAPH_BC_MAX_SOURCES_PER_BATCH` — `cugraph_bc` batch size

## Benchmark workflows

PBS jobs in `thesis_bc_project/scripts/` (`#PBS -q regular-g`), submitted with `qsub` from `thesis_bc_project/`.

```bash
cd thesis_bc_project

# Full benchmark (submit size tiers in parallel)
qsub scripts/run_benchmark_small.sh
qsub scripts/run_benchmark_medium.sh
qsub scripts/run_benchmark_large.sh

# gpu_opt / gpu_opt_pure_chunked comparison reruns
qsub scripts/run_benchmark_small_gpu_opt_compare.sh
qsub scripts/run_benchmark_medium_gpu_opt_compare.sh
qsub scripts/run_benchmark_large_gpu_opt_compare.sh

# Ablation & PathMerge sweep
qsub scripts/run_ablation.sh
qsub scripts/run_pathmerge_sweep.sh

# UM oversubscription
qsub scripts/run_um_oversubscribe_experiment.sh
qsub scripts/run_um_oversubscribe_gpu_opt.sh
qsub scripts/run_um_oversubscribe_gpu_opt_pure_chunked.sh

# Dry run (print commands without executing)
DRY_RUN=1 SKIP_BUILD=1 bash scripts/run_benchmark_small.sh
```

Each job writes a timestamped `build_miyabi/result_*` dir with `results.tsv`, `benchmark.log`, `phase_timing.log`, `max_bc.tsv`, and auto-runs `scripts/summarize_benchmark.py`.

Thesis figures (scipy + matplotlib; `uv run --with scipy --with matplotlib python ...` also works):
```bash
python3 scripts/statistical_analysis.py \
    --results build_miyabi/result_benchmark_*/results.tsv \
    --phases build_miyabi/result_benchmark_*/phase_timing.log \
    --oversubscribe build_miyabi/result_um_oversubscribe/oversubscribe_results.tsv \
    --outdir ./thesis_figures
```

PBS: `qstat -u $USER`, `qstat -f <JOB_ID>`, `qdel <JOB_ID>`. Job stdout/stderr land as `bc_*.oNNNNNN`.

## Architecture notes

**Single graph, many algorithms.** `experiments/run_benchmark.cu` loads the CSR graph once into a `Graph`, then dispatches the selected impl(s) through `run_brandes()` (in `src/core/runner.cpp`) via a table mapping CLI key → label → `vector<double>(Graph&)` function. Timing, GTEPS (`n_nodes × n_edges / time`), and reporting are shared, which makes "run `all` and diff outputs" practical for correctness checking.

**cuGraph integration.** `cugraph_bc_mini` is linked as an IMPORTED static library (`cugraph_bc_mini/build/libcugraph_bc_mini.a`); the cuGraph API headers come from the vendored `third_party/cugraph/cpp/include`. The mini build (`cugraph_bc_mini/`) compiles only the centrality kernels plus the minimum CPM deps (CCCL, RMM, RAFT, cuCo, spdlog, NVTX3, rapids_logger) and pins rapids-cmake to `v26.04.00`. The `baseline` CMake library is cuGraph-free, so `run_ablation` and `run_pathmerge_sweep` build without Stage 1; only `run_benchmark` (via `baseline_cugraph`) needs `libcugraph_bc_mini.a`.

**GPU memory strategies are the experimental variable.** `gpu_opt` (Unified Memory, `src/proposed/host_um.cu`) vs `gpu_opt_pure` (manual `cudaMalloc`/`cudaMemcpy`, `host_pure.cu`) vs `gpu_opt_pure_chunked` (chunked working set, `host_chunked.cu`). The UM oversubscription experiments measure spilling to LPDDR5X over NVLink-C2C when the working set exceeds HBM3 (~96 GB) — `pure` OOMs, `gpu_opt` keeps running.

**path-merging-bc (Galliot) is a vendored baseline compiled into the runners.** Its sources are `src/baseline/galliot.cu` + `galliot_kernels.cu`; `src/baseline/pathmerge.cu` is the adapter that converts our `Graph` to Galliot's CSR layout and divides final BC by 2 for the undirected convention. The batch size is **not** capped at 64 (int2 frontier + per-source arrays, bounded only by GPU memory).

## Key conventions

- `thesis_bc_project/` is the whole, self-contained project. There is no separate `mylab/research/`, top-level `cugraph/`, `path-merging-bc/`, or `data/` submodule anymore — they were consolidated into `thesis_bc_project/` (cuGraph → `third_party/cugraph/`, path-merging → `src/baseline/`, graphs → `data/`). Archived prior results live in `legacy_results_miyabi/result_paper/`.
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
- All implementations are aligned around **undirected, unweighted** BC output and divide final BC values by 2 (the path-merging adapter divides explicitly).
- The GH200 memory strategy is intentional, not accidental:
  - `gpu_opt` uses managed memory and can intentionally oversubscribe HBM3
  - `gpu_opt_pure` uses explicit `cudaMalloc` / `cudaMemcpy`; `gpu_opt_pure_chunked` chunks the working set
  - `cugraph_bc` also uses managed memory via RMM's `managed_memory_resource`
  Do not "simplify" those differences away unless the task is specifically about changing the experiment design.
- Runtime tuning knobs are part of normal workflow, not dead code:
  - `BC_BATCH_OVERRIDE` for `gpu_opt` / `gpu_opt_pure` / ablation
  - `CUGRAPH_BC_MAX_SOURCES_PER_BATCH` for `cugraph_bc`
  - `PATHMERGE_BC_BATCH_SIZE` for `pathmerge_bc`

## Performance reference

| Implementation | Algorithm | Approx. GH200 throughput |
|----------------|-----------|--------------------------|
| `gpu_opt` | Custom CUDA kernels, queue-based BFS | ~10 GTEPS |
| `cugraph_bc` | Generic primitives + thrust sort/reduce per BFS level | ~0.5–2 GTEPS |

`cugraph_bc` is slower because `sort_by_key + reduce_by_key` per BFS level is O(M log M); `gpu_opt` is O(M) per level.

## Authoritative references

- `thesis_bc_project/README.md` — full Japanese walkthrough (definitive)
- `thesis_bc_project/scripts/HOWTO.md` — benchmark workflow and figure generation
- `thesis_bc_project/data/README.md` — graph catalog, CSR format spec, recommended graph per experiment type
- `thesis_bc_project/legacy_results_miyabi/result_paper/` — archived thesis-ready results from the prior tree
