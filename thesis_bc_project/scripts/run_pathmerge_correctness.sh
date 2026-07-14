#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=4:00:00
#PBS -N bc_pm_correct
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
#  PathMerge tuned 正確性確認ジョブ
#
#  同一グラフに対し 2 つのバッチサイズ (既定 64 と 2048) で PathMerge BC を
#  --dump-bc 出力し、両 BC ベクトルの数値一致を後段 (compare_bc_vectors.py) で
#  検証するための dump を生成する。
#
#  出力 (RESULT_DIR, build_miyabi 配下 = gitignored):
#    bc_b<N>.txt         BC dump (# ヘッダ + node_idx<TAB>bc_value)
#    bc_b<N>.stderr.log  runner stderr (Max BC / Elapse time / clamp 警告)
#
#  巨大な dump 自体は git に追加しない。比較サマリは後段スクリプトが
#  result/correctness/pathmerge_tuned/ に書き出す。
#
#  環境変数:
#    GRAPH        対象グラフ (default: snap/email-EuAll)
#    BATCHES      空白区切りバッチ (default: "64 2048")
#    TIMEOUT_SEC  1 バッチあたり最大実行時間 (default: 7200)
#    SKIP_BUILD   1 でビルドをスキップ (事前クリーンビルド + SKIP_BUILD=1 推奨)
#    DRY_RUN      1 でコマンド表示のみ
#    JOBS         ビルド並列数 (default: 8)
#    CHECKPOINT_SHA  記録用 (checkpoint commit SHA)
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
RUNNER="${BUILD_DIR}/run_pathmerge_sweep"
DATA_DIR="${PROJECT_DIR}/data"

GRAPH="${GRAPH:-snap/email-EuAll}"
BATCHES="${BATCHES:-64 2048}"
read -r -a BATCH_ARR <<< "${BATCHES}"
TIMEOUT_SEC="${TIMEOUT_SEC:-7200}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"
JOBS="${JOBS:-8}"
CHECKPOINT_SHA="${CHECKPOINT_SHA:-$(git -C "${PROJECT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)}"
ANY_FAIL=0

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_SUFFIX="${PBS_JOBID:-$$}"; JOB_SUFFIX="${JOB_SUFFIX%%.*}"
RESULT_DIR="${RESULT_DIR:-${BUILD_DIR}/result_pathmerge_correctness_${TIMESTAMP}_${JOB_SUFFIX}}"
mkdir -p "${RESULT_DIR}"
LOG_FILE="${RESULT_DIR}/correctness.log"

CMAKE_BIN="${CMAKE_BIN:-}"
if [ -z "${CMAKE_BIN}" ]; then
    for c in "${HOME}/.local/bin/cmake" cmake3 cmake; do
        if command -v "$c" >/dev/null 2>&1; then CMAKE_BIN="$c"; break; fi
    done
fi

case "${GRAPH}" in
    /*) graph_path="${GRAPH}" ;;
    *)  graph_path="${DATA_DIR}/${GRAPH}" ;;
esac
graph_name="$(basename "${GRAPH}")"

echo "=== PathMerge Correctness (dump-bc) ==="
echo "Project    : ${PROJECT_DIR}"
echo "Runner     : ${RUNNER}"
echo "Graph      : ${graph_path}"
echo "Batches    : ${BATCH_ARR[*]}"
echo "Result     : ${RESULT_DIR}"
echo "Checkpoint : ${CHECKPOINT_SHA}"

if [ "${SKIP_BUILD}" != "1" ]; then
    echo "[Build] configuring + building run_pathmerge_sweep (cugraph 不要)"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  ${CMAKE_BIN} -S ${PROJECT_DIR} -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release"
        echo "  ${CMAKE_BIN} --build ${BUILD_DIR} --target run_pathmerge_sweep -j${JOBS}"
    else
        if ! "${CMAKE_BIN}" -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER="${CC_FOR_CUGRAPH:-gcc}" -DCMAKE_CXX_COMPILER="${CXX_FOR_CUGRAPH:-g++}"; then
            echo "[ERROR] CMake configure failed; aborting (fail-fast)" >&2; exit 1
        fi
        if ! "${CMAKE_BIN}" --build "${BUILD_DIR}" --target run_pathmerge_sweep -j"${JOBS}"; then
            echo "[ERROR] Build failed; aborting (fail-fast)" >&2; exit 1
        fi
    fi
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -x "${RUNNER}" ]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2; exit 1
fi

: > "${LOG_FILE}"
echo "checkpoint_sha=${CHECKPOINT_SHA}" >> "${LOG_FILE}"

for b in "${BATCH_ARR[@]}"; do
    dump_file="${RESULT_DIR}/bc_b${b}.txt"
    err_file="${RESULT_DIR}/bc_b${b}.stderr.log"
    echo "[RUN] ${RUNNER} ${graph_path} ${b} --dump-bc  -> ${dump_file}"
    if [ "${DRY_RUN}" = "1" ]; then continue; fi
    rc=0
    timeout "${TIMEOUT_SEC}" "${RUNNER}" "${graph_path}" "${b}" --dump-bc \
        > "${dump_file}" 2> "${err_file}" || rc=$?
    cat "${err_file}" >> "${LOG_FILE}"
    if [ ${rc} -eq 124 ]; then
        echo "  -> TIMEOUT (>${TIMEOUT_SEC}s)"; ANY_FAIL=1; continue
    elif [ ${rc} -ne 0 ]; then
        echo "  -> FAILED (exit=${rc}, see log)"; ANY_FAIL=1; continue
    fi
    nlines="$(grep -vc '^#' "${dump_file}" 2>/dev/null || echo 0)"
    maxbc="$(grep 'Maximum Betweenness Centrality' "${err_file}" | tail -1 || true)"
    echo "  -> dump lines=${nlines}; ${maxbc}"
done

echo "=== Correctness dump complete ==="
ls -la "${RESULT_DIR}"
echo ""
echo "次段の数値比較:"
echo "  python3 ${SCRIPT_DIR}/scripts/compare_bc_vectors.py \\"
first="${BATCH_ARR[0]}"; second="${BATCH_ARR[1]:-}"
echo "    ${RESULT_DIR}/bc_b${first}.txt ${RESULT_DIR}/bc_b${second}.txt"

if [ "${ANY_FAIL}" -ne 0 ]; then
    echo "[ERROR] 1 件以上の FAIL/TIMEOUT が発生しました。STOP & REPORT。" >&2
    exit 2
fi
exit 0
