#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=6:00:00
#PBS -N bc_pm_sweep
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
#  PathMerge バッチサイズ探索ジョブ
#
#  各グラフに対し run_pathmerge_sweep を実行し、複数バッチサイズの
#  実行時間・GTEPS を TSV に集約する。現行実装は 64 上限が無いため、
#  メモリの許す限り大きなバッチも探索できる。
#
#  環境変数:
#    GRAPHS_STR    対象グラフ (space 区切り, data/ 相対 or 絶対)
#    BATCH_LIST    カンマ区切りのバッチサイズ (既定 1,2,4,8,16,32,64,128,256)
#    TRIALS        試行回数 (default: 1)
#    TIMEOUT_SEC   1 バッチ 1 試行あたりの最大実行時間 (default: 21600 = 6h)
#    SKIP_BUILD    1 でビルドをスキップ
#    DRY_RUN       1 でコマンド表示のみ
#    JOBS          ビルド並列数 (default: 8)
#
#  信頼性:
#    - ビルドは fail-fast (configure/build の終了コードが非 0 ならジョブ終了。
#      古い runner で計測を継続しない)。
#    - 各バッチは 1 回ずつ runner を呼び (per-batch)、timeout で保護する。
#    - FAIL / TIMEOUT は TSV にマーカー行を記録し、1 件でも発生したらジョブ終了コードを
#      非 0 (2) にする (STOP & REPORT / 自動異常検査のため)。
# ============================================================

set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}"
    SCRIPT_DIR="${PBS_O_WORKDIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
    SCRIPT_DIR="${SCRIPT_DIR:-$(pwd)}"
fi
PROJECT_DIR="${SCRIPT_DIR}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_miyabi}"
RUNNER="${BUILD_DIR}/run_pathmerge_sweep"
DATA_DIR="${PROJECT_DIR}/data"

GRAPHS_STR="${GRAPHS_STR:-benchmark_7000_41459 benchmark_11023_62184 56438_300801}"
read -r -a GRAPHS <<< "${GRAPHS_STR}"
BATCH_LIST="${BATCH_LIST:-1,2,4,8,16,32,64,128,256}"
# スペース区切りでも受理する (qsub -v はカンマ不可のため。既にカンマ区切りなら不変)
BATCH_LIST="${BATCH_LIST//[[:space:]]/,}"
TRIALS="${TRIALS:-1}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"
JOBS="${JOBS:-8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-21600}"
ANY_FAIL=0

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_SUFFIX="${PBS_JOBID:-$$}"; JOB_SUFFIX="${JOB_SUFFIX%%.*}"
RESULT_DIR="${RESULT_DIR:-${BUILD_DIR}/result_pathmerge_sweep_${TIMESTAMP}_${JOB_SUFFIX}}"
mkdir -p "${RESULT_DIR}"
TSV_FILE="${RESULT_DIR}/pathmerge_sweep_results.tsv"
LOG_FILE="${RESULT_DIR}/pathmerge_sweep.log"

# --- CMake 検出 (対象ターゲットのみビルド; cugraph_bc_mini 不要) ---
CMAKE_BIN="${CMAKE_BIN:-}"
if [ -z "${CMAKE_BIN}" ]; then
    for c in "${HOME}/.local/bin/cmake" cmake3 cmake; do
        if command -v "$c" >/dev/null 2>&1; then CMAKE_BIN="$c"; break; fi
    done
fi

echo "=== PathMerge Batch-Size Sweep ==="
echo "Project : ${PROJECT_DIR}"
echo "Runner  : ${RUNNER}"
echo "Graphs  : ${GRAPHS[*]}"
echo "Batches : ${BATCH_LIST}   Trials: ${TRIALS}"
echo "Result  : ${RESULT_DIR}"

if [ "${SKIP_BUILD}" != "1" ]; then
    echo "[Build] configuring + building run_pathmerge_sweep (cugraph 不要)"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  ${CMAKE_BIN} -S ${PROJECT_DIR} -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CXX_FOR_CUGRAPH:-g++}"
        echo "  ${CMAKE_BIN} --build ${BUILD_DIR} --target run_pathmerge_sweep -j${JOBS}"
    else
        if ! "${CMAKE_BIN}" -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER="${CC_FOR_CUGRAPH:-gcc}" -DCMAKE_CXX_COMPILER="${CXX_FOR_CUGRAPH:-g++}"; then
            echo "[ERROR] CMake configure failed; aborting (fail-fast: 古い runner で継続しない)" >&2
            exit 1
        fi
        if ! "${CMAKE_BIN}" --build "${BUILD_DIR}" --target run_pathmerge_sweep -j"${JOBS}"; then
            echo "[ERROR] Build failed; aborting (fail-fast: 古い runner で継続しない)" >&2
            exit 1
        fi
    fi
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -x "${RUNNER}" ]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    exit 1
fi

echo -e "Config\tGraph\tTrial\tTime_sec\tGTEPS" > "${TSV_FILE}"
: > "${LOG_FILE}"

# BATCH_LIST を配列化 (per-batch 実行で timeout / FAIL マーカーを個別管理する)
IFS=',' read -r -a BATCH_ARR <<< "${BATCH_LIST}"

for g in "${GRAPHS[@]}"; do
    case "${g}" in
        /*) graph_path="${g}" ;;
        *)  graph_path="${DATA_DIR}/${g}" ;;
    esac
    graph_name="$(basename "${g}")"

    if [ "${DRY_RUN}" != "1" ] && [ ! -f "${graph_path}" ]; then
        echo "  [SKIP] ${graph_name} (グラフなし: ${graph_path})"
        continue
    fi

    for b in "${BATCH_ARR[@]}"; do
        for trial in $(seq 1 "${TRIALS}"); do
            echo "[RUN] ${RUNNER} ${graph_path} ${b}  (trial ${trial}/${TRIALS})"
            if [ "${DRY_RUN}" = "1" ]; then
                continue
            fi
            tmp_out="${RESULT_DIR}/.tmp_out"
            { echo "=== graph=${graph_name} batch=${b} trial=${trial} ==="; } >> "${LOG_FILE}"
            rc=0
            timeout "${TIMEOUT_SEC}" "${RUNNER}" "${graph_path}" "${b}" \
                > "${tmp_out}" 2>> "${LOG_FILE}" || rc=$?
            if [ ${rc} -eq 124 ]; then
                echo "  -> TIMEOUT (>${TIMEOUT_SEC}s)"
                echo -e "PathMerge_b${b}\t${graph_name}\t${trial}\tTIMEOUT\t0" >> "${TSV_FILE}"
                ANY_FAIL=1; rm -f "${tmp_out}"; continue
            elif [ ${rc} -ne 0 ]; then
                echo "  -> FAILED (exit=${rc}, see log)"
                echo -e "PathMerge_b${b}\t${graph_name}\t${trial}\tFAIL\t0" >> "${TSV_FILE}"
                ANY_FAIL=1; rm -f "${tmp_out}"; continue
            fi
            # run_pathmerge_sweep の各 TSV 行: PathMerge_b<N><TAB>Graph<TAB>Time<TAB>GTEPS
            awk -v tr="${trial}" -F'\t' 'NF>=4 {print $1"\t"$2"\t"tr"\t"$3"\t"$4}' "${tmp_out}" >> "${TSV_FILE}"
            time_val="$(awk -F'\t' 'NF>=4 {print $3; exit}' "${tmp_out}")"
            echo "  -> ${time_val} sec"
            rm -f "${tmp_out}"
        done
    done
done

echo "=== PathMerge Sweep Complete ==="
[ "${DRY_RUN}" != "1" ] && cat "${TSV_FILE}"
if [ "${ANY_FAIL}" -ne 0 ]; then
    echo "[ERROR] 1 件以上の FAIL/TIMEOUT が発生しました。STOP & REPORT (原因と job ID を確認)。" >&2
    exit 2
fi
exit 0
