#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=6:00:00
#PBS -N bc_ablation
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
#  アブレーション実験ジョブ (提案手法 3 工夫の寄与測定)
#
#  各グラフに対し run_ablation を mode=all で実行し、8 構成
#  (H{0,1}×W{0,1}×A{0,1}) の実行時間・GTEPS を TSV に集約する。
#
#  環境変数:
#    GRAPHS_STR   対象グラフ (space 区切り, data/ 相対 or 絶対)
#    MODE         run_ablation のモード (default: all)
#    TRIALS       試行回数 (default: 1)
#    SKIP_BUILD   1 でビルドをスキップ
#    DRY_RUN      1 でコマンド表示のみ
#    JOBS         ビルド並列数 (default: 8)
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
RUNNER="${BUILD_DIR}/run_ablation"
DATA_DIR="${PROJECT_DIR}/data"

GRAPHS_STR="${GRAPHS_STR:-benchmark_7000_41459 benchmark_11023_62184 56438_300801 325557_3216152}"
read -r -a GRAPHS <<< "${GRAPHS_STR}"
MODE="${MODE:-all}"
TRIALS="${TRIALS:-5}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"
JOBS="${JOBS:-8}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_SUFFIX="${PBS_JOBID:-$$}"; JOB_SUFFIX="${JOB_SUFFIX%%.*}"
RESULT_DIR="${RESULT_DIR:-${BUILD_DIR}/result_ablation_${TIMESTAMP}_${JOB_SUFFIX}}"
mkdir -p "${RESULT_DIR}"
TSV_FILE="${RESULT_DIR}/ablation_results.tsv"
LOG_FILE="${RESULT_DIR}/ablation.log"

# --- CMake 検出 (対象ターゲットのみビルド; cugraph_bc_mini 不要) ---
CMAKE_BIN="${CMAKE_BIN:-}"
if [ -z "${CMAKE_BIN}" ]; then
    for c in "${HOME}/.local/bin/cmake" cmake3 cmake; do
        if command -v "$c" >/dev/null 2>&1; then CMAKE_BIN="$c"; break; fi
    done
fi

echo "=== BC Ablation Experiment ==="
echo "Project : ${PROJECT_DIR}"
echo "Runner  : ${RUNNER}"
echo "Graphs  : ${GRAPHS[*]}"
echo "Mode    : ${MODE}   Trials: ${TRIALS}"
echo "Result  : ${RESULT_DIR}"

if [ "${SKIP_BUILD}" != "1" ]; then
    echo "[Build] configuring + building run_ablation (cugraph 不要)"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  ${CMAKE_BIN} -S ${PROJECT_DIR} -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=${CXX_FOR_CUGRAPH:-g++}"
        echo "  ${CMAKE_BIN} --build ${BUILD_DIR} --target run_ablation -j${JOBS}"
    else
        "${CMAKE_BIN}" -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_C_COMPILER="${CC_FOR_CUGRAPH:-gcc}" -DCMAKE_CXX_COMPILER="${CXX_FOR_CUGRAPH:-g++}"
        "${CMAKE_BIN}" --build "${BUILD_DIR}" --target run_ablation -j"${JOBS}"
    fi
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -x "${RUNNER}" ]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    exit 1
fi

echo -e "Config\tGraph\tTrial\tTime_sec\tGTEPS" > "${TSV_FILE}"
: > "${LOG_FILE}"

for g in "${GRAPHS[@]}"; do
    # 絶対パスでなければ data/ 相対とみなす
    case "${g}" in
        /*) graph_path="${g}" ;;
        *)  graph_path="${DATA_DIR}/${g}" ;;
    esac

    for trial in $(seq 1 "${TRIALS}"); do
        echo "[RUN] ${RUNNER} ${graph_path} ${MODE}  (trial ${trial})"
        if [ "${DRY_RUN}" = "1" ]; then
            continue
        fi
        tmp_out="${RESULT_DIR}/.tmp_out"
        {
            echo "=== graph=${g} trial=${trial} ==="
        } >> "${LOG_FILE}"
        "${RUNNER}" "${graph_path}" "${MODE}" > "${tmp_out}" 2>> "${LOG_FILE}" || {
            echo "  -> FAILED (see log)"; continue;
        }
        # run_ablation の各 TSV 行: Config<TAB>Graph<TAB>Time<TAB>GTEPS
        awk -v tr="${trial}" -F'\t' 'NF>=4 {print $1"\t"$2"\t"tr"\t"$3"\t"$4}' "${tmp_out}" >> "${TSV_FILE}"
        rm -f "${tmp_out}"
    done
done

echo "=== Ablation Complete ==="
if [ "${DRY_RUN}" != "1" ]; then
    cat "${TSV_FILE}"
    SUMMARIZER="${SCRIPT_DIR}/scripts/summarize_ablation.py"
    if [ -f "${SUMMARIZER}" ] && command -v python3 >/dev/null 2>&1; then
        echo ""
        echo "=== 自動サマリ生成 (寄与表 + 交互作用 + フェーズ帰属) ==="
        python3 "${SUMMARIZER}" "${TSV_FILE}" "${RESULT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    else
        echo "手動サマリ生成: python3 scripts/summarize_ablation.py ${TSV_FILE} ${RESULT_DIR}"
    fi
fi
exit 0
