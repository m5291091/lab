#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=24:00:00
#PBS -N um_oversubscribe
#PBS -W group_list=gj17
#PBS -j oe

set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}"
    SCRIPT_DIR="${PBS_O_WORKDIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
    SCRIPT_DIR="${SCRIPT_DIR:-$(pwd)}"
fi
LAB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_miyabi}"
RUNNER="${BUILD_DIR}/brandes_runner"

GRAPH="${LAB_DIR}/data/325557_3216152"
GRAPH_NAME="$(basename "${GRAPH}")"

RESULT_DIR="${BUILD_DIR}/result_um_oversubscribe"
mkdir -p "${RESULT_DIR}"
TSV_FILE="${RESULT_DIR}/oversubscribe_results.tsv"

echo -e "Implementation\tBatchSize\tTrial\tTime_sec\tGTEPS\tStatus" > "${TSV_FILE}"

# バッチサイズのスイープ範囲
# 8192 付近で HBM3 (96GB) 上限に到達する想定
BATCH_SIZES=(512 1024 2048 4096 8192 10240 12288 16384)
METHODS=("gpu_opt_pure" "gpu_opt_pure_chunked" "gpu_opt")

echo "=== UM Oversubscription Experiment ==="
echo "Graph: ${GRAPH_NAME}"
echo "Runner: ${RUNNER}"

for method in "${METHODS[@]}"; do
    for batch in "${BATCH_SIZES[@]}"; do
        for trial in $(seq 1 5); do
            echo "[RUN] Method: ${method}, BC_BATCH_OVERRIDE: ${batch}, Trial: ${trial}"
            
            export BC_BATCH_OVERRIDE="${batch}"
            
            tmp_stdout="${RESULT_DIR}/.tmp_stdout_${method}_${batch}_${trial}"
            tmp_stderr="${RESULT_DIR}/.tmp_stderr_${method}_${batch}_${trial}"
            
            rc=0
            "${RUNNER}" "${method}" "${GRAPH}" > "${tmp_stdout}" 2> "${tmp_stderr}" || rc=$?

            # summarize_oversubscribe.py が parse できる形式でヘッダ行をログに追加
            echo "=== ${method} batch=${batch} trial=${trial} rc=${rc} ===" \
                >> "${RESULT_DIR}/um_experiment.log"

            if [ ${rc} -ne 0 ]; then
                echo -e "${method}\t${batch}\t${trial}\t0\t0\tOOM_OR_FAIL" >> "${TSV_FILE}"
                echo "  -> FAILED (rc=${rc})"
            else
                time_val="$(cat "${tmp_stdout}" | awk -F'\t' '{print $3}')"
                gteps_val="$(cat "${tmp_stdout}" | awk -F'\t' '{print $4}')"
                echo -e "${method}\t${batch}\t${trial}\t${time_val}\t${gteps_val}\tSUCCESS" >> "${TSV_FILE}"
                echo "  -> SUCCESS: ${time_val} sec, GTEPS=${gteps_val}"
            fi
            
            # ログ保存
            cat "${tmp_stderr}" >> "${RESULT_DIR}/um_experiment.log"
            rm -f "${tmp_stdout}" "${tmp_stderr}"
        done
    done
done

echo "=== Experiment Complete ==="
cat "${TSV_FILE}"

