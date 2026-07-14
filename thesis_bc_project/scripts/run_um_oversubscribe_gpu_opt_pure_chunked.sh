#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=24:00:00
#PBS -N um_ovr_purechk
#PBS -W group_list=gj17
#PBS -j oe

set -euo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}"
    SCRIPT_DIR="${PBS_O_WORKDIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"
    SCRIPT_DIR="${SCRIPT_DIR:-$(pwd)}"
fi
PROJECT_DIR="${SCRIPT_DIR}"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_miyabi}"
RUNNER="${RUNNER:-${BUILD_DIR}/run_benchmark}"

GRAPH="${GRAPH:-${PROJECT_DIR}/data/325557_3216152}"
GRAPH_NAME="$(basename "${GRAPH}")"

METHOD="gpu_opt_pure_chunked"
RESULT_DIR="${RESULT_DIR:-${BUILD_DIR}/result_um_oversubscribe_gpu_opt_pure_chunked}"
mkdir -p "${RESULT_DIR}"
TSV_FILE="${TSV_FILE:-${RESULT_DIR}/oversubscribe_results.tsv}"
LOG_FILE="${LOG_FILE:-${RESULT_DIR}/um_gpu_opt_pure_chunked_experiment.log}"

BATCH_SIZES_STR="${BATCH_SIZES_STR:-512 1024 2048 4096 8192 10240 12288 16384}"
read -r -a BATCH_SIZES <<< "${BATCH_SIZES_STR}"
TRIALS="${TRIALS:-5}"

if [ ! -x "${RUNNER}" ]; then
    echo "[ERROR] Runner not found or not executable: ${RUNNER}" >&2
    exit 1
fi
if [ ! -f "${GRAPH}" ]; then
    echo "[ERROR] Graph not found: ${GRAPH}" >&2
    exit 1
fi

echo -e "Implementation\tBatchSize\tTrial\tTime_sec\tGTEPS\tStatus" > "${TSV_FILE}"
: > "${LOG_FILE}"

echo "=== UM Oversubscription Experiment (gpu_opt_pure_chunked only) ==="
echo "Graph: ${GRAPH_NAME}"
echo "Runner: ${RUNNER}"
echo "ResultDir: ${RESULT_DIR}"
echo "Trials: ${TRIALS}"
echo "BatchSizes: ${BATCH_SIZES[*]}"
echo "Method: ${METHOD}"

for batch in "${BATCH_SIZES[@]}"; do
    for trial in $(seq 1 "${TRIALS}"); do
        echo "[RUN] Method: ${METHOD}, BC_BATCH_OVERRIDE: ${batch}, Trial: ${trial}"

        export BC_BATCH_OVERRIDE="${batch}"

        tmp_stdout="${RESULT_DIR}/.tmp_stdout_${METHOD}_${batch}_${trial}"
        tmp_stderr="${RESULT_DIR}/.tmp_stderr_${METHOD}_${batch}_${trial}"

        rc=0
        "${RUNNER}" "${METHOD}" "${GRAPH}" > "${tmp_stdout}" 2> "${tmp_stderr}" || rc=$?

        {
            echo "=== ${METHOD} batch=${batch} trial=${trial} rc=${rc} ==="
            cat "${tmp_stderr}"
            echo
        } >> "${LOG_FILE}"

        if [ "${rc}" -ne 0 ]; then
            echo -e "${METHOD}\t${batch}\t${trial}\t0\t0\tOOM_OR_FAIL" >> "${TSV_FILE}"
            echo "  -> FAILED (rc=${rc})"
        else
            time_val="$(awk -F'\t' 'NR==1 {print $3}' "${tmp_stdout}")"
            gteps_val="$(awk -F'\t' 'NR==1 {print $4}' "${tmp_stdout}")"

            if [ -z "${time_val}" ] || [ -z "${gteps_val}" ]; then
                echo -e "${METHOD}\t${batch}\t${trial}\t0\t0\tOOM_OR_FAIL" >> "${TSV_FILE}"
                echo "  -> FAILED (could not parse stdout summary)"
            else
                echo -e "${METHOD}\t${batch}\t${trial}\t${time_val}\t${gteps_val}\tSUCCESS" >> "${TSV_FILE}"
                echo "  -> SUCCESS: ${time_val} sec, GTEPS=${gteps_val}"
            fi
        fi

        rm -f "${tmp_stdout}" "${tmp_stderr}"
    done
done

unset BC_BATCH_OVERRIDE

echo "=== Experiment Complete ==="
cat "${TSV_FILE}"
