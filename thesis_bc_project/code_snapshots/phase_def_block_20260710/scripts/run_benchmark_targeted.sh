#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=12:00:00
#PBS -N bc_targeted
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
#  汎用ターゲット計測ジョブ (任意グラフ × 任意実装のベンチマーク)
#
#  指定したグラフ・実装を run_benchmark で計測し、summarize_benchmark.py が
#  読める 7 列 TSV (results.tsv) を出力する。BFS カーネル選択「常時 block」化
#  後の legacy 影響分 (email-EuAll, roadNet-PA/TX/CA) の再計測などに使う。
#
#  出力 (RESULT_DIR):
#    results.tsv       Implementation<TAB>Graph<TAB>Nodes<TAB>Edges<TAB>Trial<TAB>Time_sec<TAB>GTEPS
#    max_bc.tsv        Implementation<TAB>Graph<TAB>MaxBC_Index<TAB>MaxBC_Value
#    phase_timing.log  runner の stderr (Running:/Phase/Maximum 行を含む)
#    benchmark.log     全ログ
#
#  環境変数:
#    GRAPHS_STR   対象グラフ (space 区切り, data/ 相対 or 絶対)。必須。
#    IMPLS_STR    対象実装 (space 区切り, 既定 "gpu_opt gpu_opt_pure gpu_opt_pure_chunked")
#    TRIALS       試行回数 (default: 3)
#    TIMEOUT_SEC  1 回あたりの最大実行時間 (default: 21600 = 6h)
#    SKIP_BUILD   1 でビルドをスキップ (同時投入時は事前ビルド + SKIP_BUILD=1)
#    DRY_RUN      1 でコマンド表示のみ (ログインノードで疎通確認可)
#
#  注意: qsub -v は変数間をカンマ区切り、値内のリスト (GRAPHS_STR の複数グラフ) は
#        スペース区切り。例:
#          qsub -v 'GRAPHS_STR=snap/email-EuAll,TRIALS=5,SKIP_BUILD=1' scripts/run_benchmark_targeted.sh
#          qsub -v 'GRAPHS_STR=snap/roadNet-CA,TRIALS=3,TIMEOUT_SEC=21600,SKIP_BUILD=1' scripts/run_benchmark_targeted.sh
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
RUNNER="${BUILD_DIR}/run_benchmark"
DATA_DIR="${PROJECT_DIR}/data"

GRAPHS_STR="${GRAPHS_STR:-}"
read -r -a GRAPHS <<< "${GRAPHS_STR}"
IMPLS_STR="${IMPLS_STR:-gpu_opt gpu_opt_pure gpu_opt_pure_chunked}"
read -r -a IMPLS <<< "${IMPLS_STR}"
TRIALS="${TRIALS:-3}"
TIMEOUT_SEC="${TIMEOUT_SEC:-21600}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [ "${#GRAPHS[@]}" -eq 0 ]; then
    echo "[ERROR] GRAPHS_STR が空です。対象グラフを指定してください。" >&2
    echo "        例: qsub -v 'GRAPHS_STR=snap/email-EuAll,TRIALS=5,SKIP_BUILD=1' scripts/run_benchmark_targeted.sh" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_SUFFIX="${PBS_JOBID:-$$}"; JOB_SUFFIX="${JOB_SUFFIX%%.*}"
RESULT_DIR="${RESULT_DIR:-${BUILD_DIR}/result_benchmark_${TIMESTAMP}_${JOB_SUFFIX}}"
mkdir -p "${RESULT_DIR}"
TSV_FILE="${RESULT_DIR}/results.tsv"
MAXBC_FILE="${RESULT_DIR}/max_bc.tsv"
PHASE_LOG="${RESULT_DIR}/phase_timing.log"
LOG_FILE="${RESULT_DIR}/benchmark.log"

echo "=== Targeted Benchmark ==="
echo "Project : ${PROJECT_DIR}"
echo "Runner  : ${RUNNER}"
echo "Graphs  : ${GRAPHS[*]}"
echo "Impls   : ${IMPLS[*]}   Trials: ${TRIALS}   Timeout: ${TIMEOUT_SEC}s"
echo "Result  : ${RESULT_DIR}"

# --- ビルド (run_benchmark は cuGraph mini に依存 → 統合ビルドスクリプト) ---
if [ "${SKIP_BUILD}" != "1" ]; then
    echo "[Build] build_miyabi_interactive.sh (Stage1+2)"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  bash ${SCRIPT_DIR}/scripts/build_miyabi_interactive.sh"
    else
        bash "${SCRIPT_DIR}/scripts/build_miyabi_interactive.sh" 2>&1 | tee -a "${LOG_FILE}"
        build_rc=${PIPESTATUS[0]}
        if [ "${build_rc}" -ne 0 ]; then
            echo "[ERROR] Build failed (exit=${build_rc}); aborting (fail-fast: 古い runner で継続しない)" >&2
            exit 1
        fi
    fi
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -x "${RUNNER}" ]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    echo "        先に build_miyabi_interactive.sh でビルドしてください。" >&2
    exit 1
fi

echo -e "Implementation\tGraph\tNodes\tEdges\tTrial\tTime_sec\tGTEPS" > "${TSV_FILE}"
echo -e "Implementation\tGraph\tMaxBC_Index\tMaxBC_Value" > "${MAXBC_FILE}"
: > "${PHASE_LOG}"

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

    # ヘッダ 1 行目から nodes edges を取得
    nodes=""; edges=""
    if [ -f "${graph_path}" ]; then
        first_line="$(head -1 "${graph_path}" 2>/dev/null)"
        nodes="$(echo "${first_line}" | awk '{print $1}')"
        edges="$(echo "${first_line}" | awk '{print $2}')"
    fi

    for impl in "${IMPLS[@]}"; do
        for trial in $(seq 1 "${TRIALS}"); do
            echo "[RUN] ${RUNNER} ${impl} ${graph_path}  (trial ${trial}/${TRIALS})"
            if [ "${DRY_RUN}" = "1" ]; then
                continue
            fi
            tmp_out="${RESULT_DIR}/.tmp_out"
            tmp_err="${RESULT_DIR}/.tmp_err"
            { echo "=== graph=${graph_name} impl=${impl} trial=${trial} ==="; } >> "${LOG_FILE}"
            rc=0
            timeout "${TIMEOUT_SEC}" "${RUNNER}" "${impl}" "${graph_path}" \
                > "${tmp_out}" 2> "${tmp_err}" || rc=$?

            # stderr → ログ + フェーズ計測ログ
            cat "${tmp_err}" >> "${LOG_FILE}"
            cat "${tmp_err}" >> "${PHASE_LOG}"

            if [ ${rc} -eq 124 ]; then
                echo "  -> TIMEOUT (>${TIMEOUT_SEC}s)"
                echo -e "${impl}\t${graph_name}\t${nodes}\t${edges}\t${trial}\tTIMEOUT\t0" >> "${TSV_FILE}"
                rm -f "${tmp_out}" "${tmp_err}"
                continue
            elif [ ${rc} -ne 0 ]; then
                echo "  -> FAILED (exit=${rc}, see log)"
                echo -e "${impl}\t${graph_name}\t${nodes}\t${edges}\t${trial}\tFAIL\t0" >> "${TSV_FILE}"
                rm -f "${tmp_out}" "${tmp_err}"
                continue
            fi

            # run_benchmark stdout: Impl<TAB>Graph<TAB>Time<TAB>GTEPS
            impl_name="$(awk -F'\t' 'NF>=4 {print $1; exit}' "${tmp_out}")"
            time_val="$(awk -F'\t' 'NF>=4 {print $3; exit}' "${tmp_out}")"
            gteps_val="$(awk -F'\t' 'NF>=4 {print $4; exit}' "${tmp_out}")"
            [ -z "${impl_name}" ] && impl_name="${impl}"
            echo -e "${impl_name}\t${graph_name}\t${nodes}\t${edges}\t${trial}\t${time_val}\t${gteps_val}" >> "${TSV_FILE}"
            echo "  -> ${time_val} sec (GTEPS=${gteps_val})"

            # Max BC を回収 (正確性サニティ)
            max_bc_line="$(grep 'Maximum Betweenness Centrality' "${tmp_err}" 2>/dev/null | tail -1 || true)"
            if [ -n "${max_bc_line}" ]; then
                max_idx="$(echo "${max_bc_line}" | sed 's/.*index : \([0-9]*\).*/\1/')"
                max_val="$(echo "${max_bc_line}" | sed 's/.*==> \(.*\)/\1/')"
                echo -e "${impl_name}\t${graph_name}\t${max_idx}\t${max_val}" >> "${MAXBC_FILE}"
            fi
            rm -f "${tmp_out}" "${tmp_err}"
        done
    done
done

echo "=== Targeted Benchmark Complete ==="
if [ "${DRY_RUN}" != "1" ]; then
    cat "${TSV_FILE}"
    SUMMARIZER="${SCRIPT_DIR}/scripts/summarize_benchmark.py"
    if [ -f "${SUMMARIZER}" ] && command -v python3 >/dev/null 2>&1; then
        echo ""
        echo "=== 自動サマリ生成 ==="
        python3 "${SUMMARIZER}" "${TSV_FILE}" "${RESULT_DIR}" 2>&1 | tee -a "${LOG_FILE}" || \
            echo "  [WARN] summarize 失敗 (実験結果は保持)"
    else
        echo "手動サマリ生成: python3 scripts/summarize_benchmark.py ${TSV_FILE} ${RESULT_DIR}"
    fi
fi
exit 0
