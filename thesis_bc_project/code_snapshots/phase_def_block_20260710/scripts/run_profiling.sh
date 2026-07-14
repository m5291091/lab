#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=2:00:00
#PBS -N bc_profiling
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
#  プロファイリングジョブ (残タスク: nsys 2本 + bandwidth)
#
#  1. bandwidth_benchmark          … HBM3/LPDDR5X 帯域計測
#  2. nsys: ablation H1W1A0 vs H1W1A1 … async 2-stream init (A) の
#                                       タイムライン重なりを比較
#  3. nsys: gpu_opt (UM)           … UM prefetch / page-migration の
#                                       compute との重なり
#
#  環境変数:
#    GRAPH        プロファイル対象グラフ (default: 56438_300801)
#    UM_GRAPH     UM prefetch 用グラフ (default: 325557_3216152)
#    SKIP_BUILD   1 でビルドをスキップ (default: 1)
# ============================================================
set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}"; SCRIPT_DIR="${PBS_O_WORKDIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
    SCRIPT_DIR="${SCRIPT_DIR:-$(pwd)}"
fi
PROJECT_DIR="${SCRIPT_DIR}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_miyabi}"
DATA_DIR="${PROJECT_DIR}/data"
GRAPH="${GRAPH:-56438_300801}"
UM_GRAPH="${UM_GRAPH:-325557_3216152}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_SUFFIX="${PBS_JOBID:-$$}"; JOB_SUFFIX="${JOB_SUFFIX%%.*}"
RESULT_DIR="${RESULT_DIR:-${BUILD_DIR}/result_profiling_${TIMESTAMP}_${JOB_SUFFIX}}"
mkdir -p "${RESULT_DIR}"

resolve() { case "$1" in /*) echo "$1";; *) echo "${DATA_DIR}/$1";; esac; }
GRAPH_PATH="$(resolve "${GRAPH}")"
UM_GRAPH_PATH="$(resolve "${UM_GRAPH}")"

NSYS="$(command -v nsys || true)"
echo "=== BC Profiling Job ==="
echo "Project : ${PROJECT_DIR}"
echo "Graph   : ${GRAPH_PATH}"
echo "UM graph: ${UM_GRAPH_PATH}"
echo "Result  : ${RESULT_DIR}"
echo "nsys    : ${NSYS:-<not found>}"

# ---------- 1. bandwidth_benchmark ----------
echo "=== [1/4] bandwidth_benchmark ==="
if [ -x "${BUILD_DIR}/bandwidth_benchmark" ]; then
    "${BUILD_DIR}/bandwidth_benchmark" 2>&1 | tee "${RESULT_DIR}/bandwidth.log"
else
    echo "[WARN] bandwidth_benchmark not found" | tee "${RESULT_DIR}/bandwidth.log"
fi

run_nsys() {
    local out="$1"; local dur="$2"; shift 2
    local dur_arg=()
    [ -n "${dur}" ] && dur_arg=(--duration="${dur}")
    echo "=== nsys profile -> ${out} (duration=${dur:-full}) ==="
    echo "    cmd: $*"
    "${NSYS}" profile -o "${out}" --force-overwrite true \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --cuda-um-cpu-page-faults=true \
        --cuda-um-gpu-page-faults=true \
        "${dur_arg[@]}" \
        --stats=true \
        "$@" > "${out}.console.log" 2>&1
    echo "    exit=$?"
    # テキスト統計を書き出す (rep が生成された場合)
    if [ -f "${out}.nsys-rep" ]; then
        "${NSYS}" stats --force-export=true \
            --report gpukernsum --report gpumemtimesum --report cudaapisum \
            "${out}.nsys-rep" > "${out}.stats.txt" 2>&1 || true
    fi
}

if [ -n "${NSYS}" ]; then
    # ---------- 2. ablation H1W1A0 vs H1W1A1 ----------
    echo "=== [2/4] nsys ablation H1W1A0 (async init OFF) ==="
    run_nsys "${RESULT_DIR}/ablation_H1W1A0" "" \
        "${BUILD_DIR}/run_ablation" "${GRAPH_PATH}" H1W1A0

    echo "=== [3/4] nsys ablation H1W1A1 (async init ON) ==="
    run_nsys "${RESULT_DIR}/ablation_H1W1A1" "" \
        "${BUILD_DIR}/run_ablation" "${GRAPH_PATH}" H1W1A1

    # ---------- 4. UM prefetch overlap (gpu_opt) ----------
    echo "=== [4/4] nsys gpu_opt UM prefetch (${UM_GRAPH}) ==="
    run_nsys "${RESULT_DIR}/um_prefetch_gpu_opt" "25" \
        "${BUILD_DIR}/run_benchmark" gpu_opt "${UM_GRAPH_PATH}"
else
    echo "[ERROR] nsys not found; skipped profiling captures"
fi

echo "=== Profiling Complete ==="
ls -la "${RESULT_DIR}"
exit 0
