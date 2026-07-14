#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=6:00:00
#PBS -N bc_kernel_sel
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
#  BFS カーネル選択 2×2 強制計測ジョブ
#
#  提案手法 gpu_opt (host_um.cu) は avg_deg<5 で BFS カーネルを
#  自動選択する (shared-frontier / block-per-source)。本ジョブは
#  環境変数 BC_FORCE_BFS_KERNEL で両カーネルを強制し、閾値をまたぐ
#  複数グラフで「正しい選択 vs 誤った選択」の速度差を計測する。
#  これにより「選択機構の寄与」を定量化できる (アブレーションの枠外)。
#
#  出力 TSV: kernel_selection_results.tsv
#    Kernel<TAB>Graph<TAB>Trial<TAB>Time_sec<TAB>GTEPS
#
#  環境変数:
#    GRAPHS_STR         対象グラフ (space 区切り, data/ 相対 or 絶対)
#                       既定は閾値 avg_deg=5 をまたぐ組 (低次数1 + 高次数2)
#    KERNELS            強制するカーネル (space 区切り, 既定 "shared block")
#    TRIALS             試行回数 (default: 3)
#    BC_BATCH_OVERRIDE  バッチサイズ固定 (default: 512, 公平性のため両カーネルで統一)
#    TIMEOUT_SEC        1 回あたりの最大実行時間 (default: 18000)
#    SKIP_BUILD         1 でビルドをスキップ
#    DRY_RUN            1 でコマンド表示のみ
#
#  注意: run_benchmark は cuGraph mini ライブラリに依存するため、初回は
#        build_miyabi_interactive.sh で Stage1+2 をビルドしておくこと。
#        道路網など大規模・低次数グラフは実行が長いので TRIALS を絞るか
#        別ジョブ (walltime 延長) にすること。
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

# 閾値 avg_deg=5 をまたぐ既定グラフ (email-EuAll≈2.75 が shared 側,
# benchmark_85830≈5.62 が閾値近傍の中間帯,
# benchmark_7000≈11.85 / 56438≈10.66 が block 側)
GRAPHS_STR="${GRAPHS_STR:-snap/email-EuAll benchmark_85830.data benchmark_7000_41459 56438_300801}"
read -r -a GRAPHS <<< "${GRAPHS_STR}"
KERNELS_STR="${KERNELS:-shared block}"
read -r -a KERNELS_ARR <<< "${KERNELS_STR}"
TRIALS="${TRIALS:-3}"
BC_BATCH_OVERRIDE="${BC_BATCH_OVERRIDE:-512}"
TIMEOUT_SEC="${TIMEOUT_SEC:-18000}"
SKIP_BUILD="${SKIP_BUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_SUFFIX="${PBS_JOBID:-$$}"; JOB_SUFFIX="${JOB_SUFFIX%%.*}"
RESULT_DIR="${RESULT_DIR:-${BUILD_DIR}/result_kernel_selection_${TIMESTAMP}_${JOB_SUFFIX}}"
mkdir -p "${RESULT_DIR}"
TSV_FILE="${RESULT_DIR}/kernel_selection_results.tsv"
MAXBC_FILE="${RESULT_DIR}/kernel_selection_max_bc.tsv"
LOG_FILE="${RESULT_DIR}/kernel_selection.log"

echo "=== BFS Kernel Selection 2x2 Experiment ==="
echo "Project : ${PROJECT_DIR}"
echo "Runner  : ${RUNNER}  (gpu_opt)"
echo "Graphs  : ${GRAPHS[*]}"
echo "Kernels : ${KERNELS_ARR[*]}   Trials: ${TRIALS}   Batch: ${BC_BATCH_OVERRIDE}"
echo "Result  : ${RESULT_DIR}"

# --- ビルド (run_benchmark は cuGraph mini に依存 → 統合ビルドスクリプトを使用) ---
if [ "${SKIP_BUILD}" != "1" ]; then
    echo "[Build] build_miyabi_interactive.sh (Stage1+2)"
    if [ "${DRY_RUN}" = "1" ]; then
        echo "  bash ${SCRIPT_DIR}/scripts/build_miyabi_interactive.sh"
    else
        bash "${SCRIPT_DIR}/scripts/build_miyabi_interactive.sh" 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -x "${RUNNER}" ]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    echo "        先に build_miyabi_interactive.sh でビルドしてください。" >&2
    exit 1
fi

echo -e "Kernel\tGraph\tTrial\tTime_sec\tGTEPS" > "${TSV_FILE}"
echo -e "Kernel\tGraph\tMaxBC_Index\tMaxBC_Value" > "${MAXBC_FILE}"

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

    for kernel in "${KERNELS_ARR[@]}"; do
        for trial in $(seq 1 "${TRIALS}"); do
            echo "[RUN] BC_FORCE_BFS_KERNEL=${kernel} BC_BATCH_OVERRIDE=${BC_BATCH_OVERRIDE} ${RUNNER} gpu_opt ${graph_path}  (trial ${trial})"
            if [ "${DRY_RUN}" = "1" ]; then
                continue
            fi
            tmp_out="${RESULT_DIR}/.tmp_out"
            tmp_err="${RESULT_DIR}/.tmp_err"
            { echo "=== graph=${graph_name} kernel=${kernel} trial=${trial} ==="; } >> "${LOG_FILE}"
            rc=0
            BC_FORCE_BFS_KERNEL="${kernel}" BC_BATCH_OVERRIDE="${BC_BATCH_OVERRIDE}" \
                timeout "${TIMEOUT_SEC}" "${RUNNER}" gpu_opt "${graph_path}" \
                > "${tmp_out}" 2> "${tmp_err}" || rc=$?
            cat "${tmp_err}" >> "${LOG_FILE}"

            if [ ${rc} -ne 0 ]; then
                echo "  -> FAILED (exit=${rc}, see log)"
                echo -e "${kernel}\t${graph_name}\t${trial}\tFAIL\t0" >> "${TSV_FILE}"
                rm -f "${tmp_out}" "${tmp_err}"
                continue
            fi

            # run_benchmark stdout: Impl<TAB>Graph<TAB>Time<TAB>GTEPS
            time_val="$(awk -F'\t' 'NF>=4 {print $3; exit}' "${tmp_out}")"
            gteps_val="$(awk -F'\t' 'NF>=4 {print $4; exit}' "${tmp_out}")"
            echo -e "${kernel}\t${graph_name}\t${trial}\t${time_val}\t${gteps_val}" >> "${TSV_FILE}"
            echo "  -> ${time_val} sec (GTEPS=${gteps_val})"

            # 正確性サニティ: shared/block で max BC が一致することを後で確認
            max_bc_line="$(grep 'Maximum Betweenness Centrality' "${tmp_err}" 2>/dev/null || true)"
            if [ -n "${max_bc_line}" ]; then
                max_idx="$(echo "${max_bc_line}" | sed 's/.*index : \([0-9]*\).*/\1/')"
                max_val="$(echo "${max_bc_line}" | sed 's/.*==> \(.*\)/\1/')"
                echo -e "${kernel}\t${graph_name}\t${max_idx}\t${max_val}" >> "${MAXBC_FILE}"
            fi
            rm -f "${tmp_out}" "${tmp_err}"
        done
    done
done

echo "=== Kernel Selection Complete ==="
if [ "${DRY_RUN}" != "1" ]; then
    cat "${TSV_FILE}"
    SUMMARIZER="${SCRIPT_DIR}/scripts/summarize_kernel_selection.py"
    if [ -f "${SUMMARIZER}" ] && command -v python3 >/dev/null 2>&1; then
        echo ""
        echo "=== 自動サマリ生成 ==="
        python3 "${SUMMARIZER}" "${TSV_FILE}" "${RESULT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
    else
        echo "手動サマリ生成: python3 scripts/summarize_kernel_selection.py ${TSV_FILE} ${RESULT_DIR}"
    fi
fi
exit 0
