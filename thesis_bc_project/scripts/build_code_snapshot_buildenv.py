#!/usr/bin/env python3
"""code_snapshots/<id>/BUILD_ENV.md を生成する（ビルド条件・コンパイラ/CUDA 環境・再現コマンド）。"""
import os
CS = "/work/gj17/j17000/m5291091/lab/thesis_bc_project/code_snapshots"

COMMON_TBP_BUILD = """\
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
"""

COMMON_ENV = """\
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
"""

# id -> (title, covers, jobs, queue, run_cmds[list], extra_note)
S = {
"small_correctness_20260712": (
  "small full-vector correctness（独立参照）","correctness/small_full_vector","2367583.opbs","small-g",
  ["qsub -v EXPECTED_SHA=e32b03e9b73e9eb294685c58e488ce2a92521852,BC_BATCH_OVERRIDE=512 scripts/run_small_correctness.sh"],
  "Sequential（CPU 独立参照）vs GPU_Opt(UM, b512)。comparison は `scripts/compare_bc_vectors.py`。"),
"phase_def_block_20260710": (
  "主実験 checkpoint（提案 block・PathMerge 掃引・kernel 選択・ablation・profiling）",
  "main_performance/proposed_variants; tuning/pathmerge; tuning/kernel_selection; ablation; profiling; phase_breakdown; correctness/pathmerge_tuned",
  "2356120;2357334-2357337;2355000;2355001;2359080;2359081;2359096;2359169;2360072;2360073;2361040;2361041;2362006;2354329;2354330;2354994;2354999;2359175",
  "regular-g / small-g",
  ["qsub scripts/run_benchmark_targeted.sh   # proposed_variants (2356120,2357334-2357337)",
   "qsub scripts/run_pathmerge_sweep.sh      # pathmerge tuned",
   "qsub scripts/run_kernel_selection.sh     # forced shared/block (2354329,2354330)",
   "qsub scripts/run_ablation.sh             # H/W/A 8構成 (2354994,2354999)",
   "qsub scripts/run_profiling.sh            # nsys + 帯域 (2359175)"],
  "この checkpoint は複数実験の実験時コードを兼ねる（`1ae987c` 常時 block 化後）。各実験の詳細は該当 `result/*/SOURCE.md`。"),
"memory_correctness_20260712": (
  "memory-path 正確性 canonical（比較行列）","correctness/memory_paths（canonical）","2368587.opbs","regular-g(→ small-g 100GiB)",
  ["qsub scripts/run_memory_correctness.sh   # 6構成 comparison matrix (job 2368587)"],
  "UM/Pure/Chunked/PathMerge を 325557 で比較。全 6 構成 runner_exit=0、job 形式判定は CORE_FAIL（比較判定であり実行失敗ではない）。"),
"memory_diagnostic_20260713": (
  "memory-path 診断（T-RESET/T-NSEFF）","correctness/memory_paths（diagnostic）","2369632.opbs","regular-g(→ small-g)",
  ["qsub scripts/run_memory_correctness_diagnostic.sh   # CONTROL/T-RESET/T-NSEFF (job 2369632)"],
  "`BC_DIAG_FORCE_FULL_RESET=1`(T-RESET) / `BC_DIAG_FORCE_NS_EFF_ONE=1`(T-NSEFF) の 1 因子スイッチを追加した診断ビルド。"),
"memory_correctness_oom_20260712": (
  "memory-path 正確性（UM b10240 OOM）","unsuccessful/oom","2368269.opbs","small-g(100GiB, mem 制限)",
  ["qsub scripts/run_memory_correctness.sh   # UM b10240 が 100GiB queue で OOM (exit 137)"],
  "100 GiB mem 上限の queue で UM b10240（dynamic 213.38 GB）がホスト常駐超過し SIGKILL。空ベクトル=OOM 証跡。PathMerge b4096 参照は PASS。"),
"memory_correctness_failfast_20260712": (
  "memory-path 正確性（比較不一致 fail-fast）","unsuccessful/early_terminated","2368398.opbs","regular-g(→ small-g)",
  ["qsub scripts/run_memory_correctness.sh   # pure vs PathMerge mismatch で fail-fast (job 2368398)"],
  "`fix(correctness): adapt UM validation to Miyabi-G memory limit`。gpu_opt_pure_b1024 vs PathMerge の全ベクトル比較不一致（11027, max_rel≈2e-3）で fail-fast、後続構成は未実行。"),
"oldtree_f05ec52_20260512": (
  "UM オーバーサブスクリプション（旧 mylab/research ツリー）",
  "memory_scalability（feasibility）; main_performance/seven_implementations（legacy, 近似）",
  "UMv2（旧ツリー; PBS job ID は当時ログに個別記録なし → not_recorded）","（旧ツリー測定; queue 記録なし）",
  ["qsub scripts/run_um_oversubscribe_experiment.sh",
   "qsub scripts/run_um_oversubscribe_gpu_opt.sh",
   "qsub scripts/run_um_oversubscribe_gpu_opt_pure_chunked.sh"],
  "OLDTREE"),
}

OLDTREE_BUILD = """\
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
"""

OLDTREE_ENV = """\
## コンパイラ・CUDA 環境（旧ツリー測定, 限定）

- 測定 commit `f05ec52`（2026-05-12, 旧 `mylab/research`）。同一 GH200 系ハードウェア。
- **メモリサイジング/OOM ロジックは checkpoint `88faffa` と文字単位同一**（`result/provenance/um_code_diff_audit.md`）。
- **時間値は最新 block 性能値として非採用**（旧セッションの driver/CMake/thermal/co-tenancy は独立再検証していない）。
- **feasibility（SUCCESS/OOM 傾向）のみ限定採用**。文字単位同一のメモリサイジング + 同一割当により、
  pure が先に OOM / UM がより大きなバッチまで到達 / chunked が最大、という傾向を再利用（境界の 88faffa 再実測は未実施）。
- 正確性: um_experiment_*.log の各 run で Max BC = `39343001000.11`（独立参照 PathMerge と一致）。
"""

for sid,(title,covers,jobs,queue,cmds,note) in S.items():
    dest = os.path.join(CS,sid)
    oldtree = (note=="OLDTREE" or sid.startswith("oldtree"))
    from_map = f"code_snapshots/_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv"
    lines = [f"# BUILD_ENV — {sid}", "",
             f"**用途**: {title}", "",
             f"- SourceSnapshotID: `{sid}`",
             f"- 元 commit（監査用対応表 `../_legacy_audit/LEGACY_COMMIT_TO_SNAPSHOT.tsv`）: 参照",
             f"- 実験対象: {covers}",
             f"- PBS job ID: `{jobs}`",
             f"- queue: {queue}", "",
             "## 再現コマンド（GPU 計算ノード, group gj17）","","```bash"]
    lines += cmds
    lines += ["```",""]
    if not oldtree:
        lines.append(COMMON_TBP_BUILD); lines.append(COMMON_ENV)
    else:
        lines.append(OLDTREE_BUILD); lines.append(OLDTREE_ENV)
    if note and note!="OLDTREE":
        lines.append("## 備考\n\n"+note+"\n")
    with open(os.path.join(dest,"BUILD_ENV.md"),"w") as f:
        f.write("\n".join(lines).rstrip()+"\n")
    print(f"wrote {sid}/BUILD_ENV.md")
