#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=6:00:00
#PBS -N bc_corr_abl
#PBS -W group_list=gj17
#PBS -j oe

# Series C: corrected 325557, exact H{0,1} x W{0,1} x A{0,1}, n=5.
# The runner performs one global-to-the-eight-config-set untimed H1W1A1
# warmup per invocation.  Those warmups are excluded from the formal 40 rows.

set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}" || exit 2
    PROJECT_DIR="${PBS_O_WORKDIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${PROJECT_DIR}" || exit 2
fi

source "${PROJECT_DIR}/scripts/build_dir_guard.sh"

# root と cugraph_bc_mini は別の CMake binary directory を使う (Gate W7.3B1.1)。
# run_ablation は cuGraph 非依存だが、build_miyabi/ の cache 汚染を避けるため
# Series A/B と同じ job 固有の root build directory を使う。
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${PBS_JOBID:-not_pbs}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/build_miyabi}"
BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build_corrected_325557/${TIMESTAMP}_${JOB_ID}}"
CUGRAPH_BC_MINI_SRC_DIR="${PROJECT_DIR}/cugraph_bc_mini"
CUGRAPH_BC_MINI_BUILD_DIR="${CUGRAPH_BC_MINI_BUILD_DIR:-${CUGRAPH_BC_MINI_SRC_DIR}/build}"
RUNNER="${BUILD_DIR}/run_ablation"
GRAPH_VALIDATOR="${PROJECT_DIR}/tools/validate_graph_csr.py"
RESULT_VALIDATOR="${PROJECT_DIR}/scripts/validate_ablation_results.py"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
JOBS="${JOBS:-8}"
TRIALS="${TRIALS:-5}"

GRAPH_REL="${GRAPH:-data/325557_3216152_corrected_v1}"
EXPECTED_GRAPH_SHA="${EXPECTED_GRAPH_SHA:-8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22}"
EXPECTED_N="${EXPECTED_N:-325557}"
EXPECTED_M="${EXPECTED_M:-3216152}"
LEGACY_GRAPH_SHA="a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584"
EXISTING_GRAPHS="benchmark_7000_41459 benchmark_11023_62184 56438_300801"
EXPECTED_CONFIGS="Ablation_H0_W0_A0 Ablation_H0_W0_A1 Ablation_H0_W1_A0 Ablation_H0_W1_A1 Ablation_H1_W0_A0 Ablation_H1_W0_A1 Ablation_H1_W1_A0 Ablation_H1_W1_A1"

if [ "${DRY_RUN}" = "1" ]; then
    printf '%s\n' \
        "DRY RUN: no build, runner, GPU access, qsub, or result update" \
        "Project    : ${PROJECT_DIR}" \
        "Root build : ${BUILD_DIR}" \
        "Mini build : ${CUGRAPH_BC_MINI_BUILD_DIR}" \
        "Runner     : ${RUNNER}" \
        "Graph      : ${GRAPH_REL} (n=${EXPECTED_N}, m=${EXPECTED_M}, sha=${EXPECTED_GRAPH_SHA})" \
        "Trials     : ${TRIALS}" \
        "Configs    : ${EXPECTED_CONFIGS}" \
        "Formal rows: 8 * ${TRIALS} = $((8 * TRIALS)); exact set/trials, finite positive Time/GTEPS, RunnerExit=0" \
        "Warmup     : one global untimed H1W1A1 before each runner invocation/config set; excluded from formal rows" \
        "Warmup evidence: job 2354994 script+raw log+TSV show one marker before each 8-row graph/trial set" \
        "Output     : fresh result_corrected_325557_ablation_<timestamp>_<PBS_JOBID> under ${RESULT_ROOT}; collision is fatal" \
        "CMake      : configure and build checked separately; configure failure never continues with an old binary" \
        "Build dirs : root and mini are distinct and job-specific; collision or foreign CMake cache aborts before configure" \
        "Not rerun  : ${EXISTING_GRAPHS}" \
        "Future submission command (display only; DO NOT run in DRY_RUN):" \
        "cd /work/gj17/j17000/m5291091/lab/thesis_bc_project" \
        "qsub -v EXPECTED_SHA=<POST_COMMIT_SHA>,EXPECTED_GRAPH_SHA=8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22,TRIALS=5 \\" \
        "  scripts/run_corrected_325557_ablation.sh" \
        "<POST_COMMIT_SHA> is determined only after the checkpoint commit; it is intentionally not replaced by the current uncommitted HEAD."
    exit 0
fi

abort() {
    echo "ABORTED: $*" >&2
    [ -n "${RUN_LOG:-}" ] && printf 'ABORTED: %s\n' "$*" >> "${RUN_LOG}" 2>/dev/null
    [ -n "${MANIFEST:-}" ] && printf 'final_status=ABORTED_NOT_FORMAL_RESULT\nfinal_reason=%s\n' "$*" >> "${MANIFEST}" 2>/dev/null
    exit 2
}

fail_series_c() {
    echo "FAILED: $*" >&2
    [ -n "${RUN_LOG:-}" ] && printf 'FAILED: %s\n' "$*" >> "${RUN_LOG}" 2>/dev/null
    [ -n "${MANIFEST:-}" ] && printf 'final_status=INCOMPLETE_NOT_FORMAL_RESULT\nfinal_reason=%s\n' "$*" >> "${MANIFEST}" 2>/dev/null
    exit 3
}

: "${EXPECTED_SHA:?EXPECTED_SHA must be set (post-commit checkpoint SHA)}"
[ "${TRIALS}" -eq 5 ] 2>/dev/null || abort "TRIALS must be exactly 5 for Series C (actual=${TRIALS})"
ACTUAL_SHA="$(git rev-parse HEAD)" || abort "cannot resolve HEAD"
[ "${ACTUAL_SHA}" = "${EXPECTED_SHA}" ] || abort "checkpoint mismatch (HEAD=${ACTUAL_SHA} != EXPECTED_SHA=${EXPECTED_SHA})"
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

case "${GRAPH_REL}" in
    /*) GRAPH_PATH="${GRAPH_REL}" ;;
    *) GRAPH_PATH="${PROJECT_DIR}/${GRAPH_REL}" ;;
esac
[ -r "${GRAPH_PATH}" ] || abort "graph not readable: ${GRAPH_PATH}"
read -r N M < "${GRAPH_PATH}" || abort "cannot read graph header"
GRAPH_SHA="$(sha256_file "${GRAPH_PATH}")" || abort "cannot hash graph"
[ "${N}" = "${EXPECTED_N}" ] || abort "graph n mismatch (${N} != ${EXPECTED_N})"
[ "${M}" = "${EXPECTED_M}" ] || abort "graph m mismatch (${M} != ${EXPECTED_M})"
[ "${GRAPH_SHA}" != "${LEGACY_GRAPH_SHA}" ] || abort "legacy malformed graph selected"
[ "${GRAPH_SHA}" = "${EXPECTED_GRAPH_SHA}" ] || abort "graph sha256 mismatch (${GRAPH_SHA} != ${EXPECTED_GRAPH_SHA})"

RESULT_DIR="${RESULT_DIR:-${RESULT_ROOT}/result_corrected_325557_ablation_${TIMESTAMP}_${JOB_ID}}"
[ ! -e "${RESULT_DIR}" ] || abort "output collision: ${RESULT_DIR} already exists"
mkdir -p "$(dirname "${RESULT_DIR}")" || abort "cannot create output parent"
mkdir "${RESULT_DIR}" || abort "cannot create fresh result directory: ${RESULT_DIR}"

MANIFEST="${RESULT_DIR}/MANIFEST.txt"
RAW_TSV="${RESULT_DIR}/ablation_results.partial.tsv"
FORMAL_TSV="${RESULT_DIR}/ablation_results.tsv"
STATS_TSV="${RESULT_DIR}/ablation_per_config_stats.tsv"
RUN_LOG="${RESULT_DIR}/run.log"
STDERR_LOG="${RESULT_DIR}/ablation.stderr.log"
: > "${RUN_LOG}" || abort "cannot write run.log"
: > "${STDERR_LOG}" || abort "cannot write ablation stderr log"
log() { printf '%s\n' "$*" | tee -a "${RUN_LOG}"; }

[ -f "${GRAPH_VALIDATOR}" ] || abort "graph validator missing: ${GRAPH_VALIDATOR}"
[ -f "${RESULT_VALIDATOR}" ] || abort "result validator missing: ${RESULT_VALIDATOR}"
if ! python3 "${GRAPH_VALIDATOR}" "${GRAPH_PATH}" --json "${RESULT_DIR}/graph_validation.json" >> "${RUN_LOG}" 2>&1; then
    abort "graph validation failure: ${GRAPH_REL}"
fi

GRAPH_SIZE="$(stat -c '%s' "${GRAPH_PATH}")"
{
    printf 'checkpoint_sha=%s\n' "${ACTUAL_SHA}"
    printf 'pbs_job_id=%s\n' "${JOB_ID}"
    printf 'series=C_ablation\n'
    printf 'graph=%s\ngraph_sha256=%s\ngraph_size_bytes=%s\n' "${GRAPH_REL}" "${GRAPH_SHA}" "${GRAPH_SIZE}"
    printf 'n_nodes=%s\nn_edges=%s\n' "${N}" "${M}"
    printf 'configurations=%s\n' "${EXPECTED_CONFIGS}"
    printf 'trials=%s\nformal_row_count=%s\n' "${TRIALS}" "$((8 * TRIALS))"
    printf 'warmup_classification=global_to_eight_config_set_once_per_runner_invocation\n'
    printf 'warmup_count_expected=%s\n' "${TRIALS}"
    printf 'warmup_in_formal_rows=no\n'
    printf 'warmup_primary_evidence=code_snapshots/phase_def_block_20260710/scripts/run_ablation.sh:95-109;raw_data/ablation/synthetic/job_2354994_20260710/ablation.log;raw_data/ablation/synthetic/job_2354994_20260710/ablation_results.tsv;experiments/run_ablation.cu:111-120\n'
    printf 'existing_graphs_not_rerun=%s\n' "${EXISTING_GRAPHS}"
    printf 'aggregation=median,mean,sample_sd,min,max\n'
} > "${MANIFEST}" || abort "cannot write manifest"

bcguard_assert_separate \
    "${PROJECT_DIR}" "${BUILD_DIR}" \
    "${CUGRAPH_BC_MINI_SRC_DIR}" "${CUGRAPH_BC_MINI_BUILD_DIR}" \
    || abort "build directory collision or foreign CMake cache"

BUILD_STATUS=skipped
if [ "${SKIP_BUILD}" != "1" ]; then
    CMAKE_BIN="${CMAKE_BIN:-}"
    if [ -z "${CMAKE_BIN}" ]; then
        for candidate in "${HOME}/.local/bin/cmake" cmake3 cmake; do
            if command -v "${candidate}" >/dev/null 2>&1; then
                CMAKE_BIN="${candidate}"
                break
            fi
        done
    fi
    [ -n "${CMAKE_BIN}" ] || abort "CMake executable not found"
    log "[CMAKE configure] ${CMAKE_BIN} -S ${PROJECT_DIR} -B ${BUILD_DIR}"
    if ! "${CMAKE_BIN}" -S "${PROJECT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="${CC_FOR_CUGRAPH:-gcc}" -DCMAKE_CXX_COMPILER="${CXX_FOR_CUGRAPH:-g++}" \
        2>&1 | tee -a "${RUN_LOG}"; then
        abort "CMake configure failed; refusing to continue with any existing binary"
    fi
    log "[CMAKE build] target=run_ablation"
    if ! "${CMAKE_BIN}" --build "${BUILD_DIR}" --target run_ablation -j"${JOBS}" \
        2>&1 | tee -a "${RUN_LOG}"; then
        abort "CMake build failed; refusing to continue with any existing binary"
    fi
    BUILD_STATUS=built
    bcguard_write_provenance "${BUILD_DIR}" "${ACTUAL_SHA}"
fi
[ -x "${RUNNER}" ] || abort "runner not found/executable: ${RUNNER}"
bcguard_assert_provenance "${BUILD_DIR}" "${ACTUAL_SHA}" \
    || abort "runner is not verifiably built from checkpoint ${ACTUAL_SHA}"
BINARY_SHA="$(sha256_file "${RUNNER}")" || abort "cannot hash runner binary"
{
    printf 'root_build_dir=%s\n' "${BUILD_DIR}"
    printf 'mini_build_dir=%s\n' "${CUGRAPH_BC_MINI_BUILD_DIR}"
    printf 'runner_path=%s\n' "${RUNNER}"
    printf 'runner_sha256=%s\n' "${BINARY_SHA}"
    printf 'cmake_source_dir=%s\n' "${PROJECT_DIR}"
    printf 'cmake_cache_home_directory=%s\n' "$(bcguard_cache_home "${BUILD_DIR}")"
    printf 'mini_cmake_cache_home_directory=%s\n' "$(bcguard_cache_home "${CUGRAPH_BC_MINI_BUILD_DIR}")"
    printf 'build_status=%s\n' "${BUILD_STATUS}"
} >> "${MANIFEST}" || abort "cannot record build manifest"

assert_integrity() {
    local head_now graph_now
    head_now="$(git rev-parse HEAD)" || abort "cannot resolve HEAD during run"
    [ "${head_now}" = "${EXPECTED_SHA}" ] || abort "checkpoint changed during run (${head_now})"
    graph_now="$(sha256_file "${GRAPH_PATH}")" || abort "cannot hash graph during run"
    [ "${graph_now}" = "${EXPECTED_GRAPH_SHA}" ] || abort "graph SHA changed during run (${graph_now})"
}

printf '%s\n' $'checkpoint_sha\tpbs_job_id\tConfig\tGraph\tTrial\tTime_sec\tGTEPS\tRunnerExit\tStatus' > "${RAW_TSV}"

for trial in $(seq 1 "${TRIALS}"); do
    assert_integrity
    trial_out="${RESULT_DIR}/trial_${trial}.stdout.tsv"
    [ ! -e "${trial_out}" ] || abort "trial output collision: ${trial_out}"
    log "[RUN trial ${trial}/${TRIALS}] ${RUNNER} ${GRAPH_REL} all (one untimed global H1W1A1 warmup)"
    runner_rc=0
    BC_ABLATION_WARMUP=1 "${RUNNER}" "${GRAPH_PATH}" all > "${trial_out}" 2>> "${STDERR_LOG}" || runner_rc=$?
    row_status=SUCCESS
    [ "${runner_rc}" -eq 0 ] || row_status=RUNNER_FAIL
    awk -F '\t' -v trial="${trial}" -v sha="${ACTUAL_SHA}" -v job="${JOB_ID}" \
        -v rc="${runner_rc}" -v status="${row_status}" \
        'NF==4 {print sha"\t"job"\t"$1"\t"$2"\t"trial"\t"$3"\t"$4"\t"rc"\t"status}' \
        "${trial_out}" >> "${RAW_TSV}"

    partial_json="${RESULT_DIR}/completeness_after_trial_${trial}.json"
    if ! python3 "${RESULT_VALIDATOR}" "${RAW_TSV}" --expected-trials "${trial}" \
        --expected-graph "$(basename "${GRAPH_PATH}")" --stderr-log "${STDERR_LOG}" \
        --json "${partial_json}" >> "${RUN_LOG}" 2>&1; then
        fail_series_c "trial ${trial} incomplete/invalid or runner failure (exit=${runner_rc}); partial TSV retained at ${RAW_TSV}"
    fi
done

warmup_count="$(grep -c '^=== Warmup (untimed, H1W1A1) ===$' "${STDERR_LOG}" || true)"
[ "${warmup_count}" -eq "${TRIALS}" ] || fail_series_c "warmup marker count expected=${TRIALS} actual=${warmup_count}"
printf 'warmup_count_recorded=%s\n' "${warmup_count}" >> "${MANIFEST}"

FINAL_JSON="${RESULT_DIR}/ablation_completeness.json"
if ! python3 "${RESULT_VALIDATOR}" "${RAW_TSV}" --expected-trials "${TRIALS}" \
    --expected-graph "$(basename "${GRAPH_PATH}")" --stderr-log "${STDERR_LOG}" \
    --json "${FINAL_JSON}" >> "${RUN_LOG}" 2>&1; then
    fail_series_c "final 40-row completeness validation failed; partial TSV retained at ${RAW_TSV}"
fi

# Only a validated 40-row set receives the formal filename.
cp --no-clobber "${RAW_TSV}" "${FORMAL_TSV}" || abort "cannot create formal TSV without overwrite"

if ! python3 - "${FORMAL_TSV}" "${STATS_TSV}" "${GRAPH_SHA}" <<'PY' 2>&1 | tee -a "${RUN_LOG}"; then
import csv
import statistics
import sys

raw, out, graph_sha = sys.argv[1:]
rows = list(csv.DictReader(open(raw, encoding="utf-8"), delimiter="\t"))
by_config = {}
for row in rows:
    by_config.setdefault((row["Config"], row["Graph"]), []).append(float(row["Time_sec"]))
columns = ["checkpoint_sha", "pbs_job_id", "Config", "Graph", "graph_sha256", "n",
           "median_sec", "mean_sec", "sample_sd_sec", "min_sec", "max_sec"]
with open(out, "w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(columns)
    for (config, graph), values in sorted(by_config.items()):
        writer.writerow([
            rows[0]["checkpoint_sha"], rows[0]["pbs_job_id"], config, graph, graph_sha,
            len(values), f"{statistics.median(values):.6f}", f"{statistics.mean(values):.6f}",
            f"{statistics.stdev(values):.6f}", f"{min(values):.6f}", f"{max(values):.6f}",
        ])
print(f"per-config stats: {len(by_config)} rows -> {out}")
PY
    abort "statistics generation failed after complete result validation"
fi

printf 'formal_tsv=%s\ncompleteness_json=%s\nfinal_status=SUCCESS_COMPLETE_40\n' \
    "${FORMAL_TSV}" "${FINAL_JSON}" >> "${MANIFEST}"
log "=== Series C complete: exact 8 configs x 5 trials = 40 formal rows ==="
log "manifest=${MANIFEST}"
log "runner_sha256=${BINARY_SHA}; checkpoint_sha=${ACTUAL_SHA}"
exit 0
