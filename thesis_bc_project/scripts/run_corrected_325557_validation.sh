#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=24:00:00
#PBS -N bc_corr325557
#PBS -W group_list=gj17
#PBS -j oe

# Corrected 325557 limited validation (Gate W7.3A.1).
# Series A is correctness acquisition and stops at the first failed config.
# Series B is feasibility confirmation and records each expected/unexpected
# outcome before continuing.  Neither series treats a failed run as 0 seconds.

set -uo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}" || exit 2
    PROJECT_DIR="${PBS_O_WORKDIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${PROJECT_DIR}" || exit 2
fi

BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build_miyabi}"
RUNNER="${BUILD_DIR}/run_benchmark"
COMPARE="${PROJECT_DIR}/scripts/compare_bc_vectors.py"
VECTOR_VALIDATOR="${PROJECT_DIR}/scripts/validate_bc_vector.py"
MANIFEST_PARSER="${PROJECT_DIR}/scripts/parse_corrected_325557_manifest.py"
GRAPH_VALIDATOR="${PROJECT_DIR}/tools/validate_graph_csr.py"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
JOBS="${JOBS:-8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-5400}"
ABS_TOL="${ABS_TOL:-1e-3}"
REL_TOL="${REL_TOL:-1e-6}"
SERIES="${SERIES:-AB}"

GRAPH_REL="${GRAPH:-data/325557_3216152_corrected_v1}"
EXPECTED_GRAPH_SHA="${EXPECTED_GRAPH_SHA:-8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22}"
EXPECTED_N="${EXPECTED_N:-325557}"
EXPECTED_M="${EXPECTED_M:-3216152}"
LEGACY_GRAPH_REL="data/325557_3216152"
LEGACY_GRAPH_SHA="a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584"

A_NAMES=(gpu_opt_b1024 gpu_opt_pure_chunked_b1024 gpu_opt_pure_b1024 gpu_opt_b9792 gpu_opt_pure_chunked_b16384 pathmerge_b4096)
A_IMPLS=(gpu_opt gpu_opt_pure_chunked gpu_opt_pure gpu_opt gpu_opt_pure_chunked pathmerge_bc)
A_LABELS=(GPU_Opt GPU_Opt_Pure_Chunked GPU_Opt_Pure GPU_Opt GPU_Opt_Pure_Chunked PathMerge)
A_BATCHENV=(BC_BATCH_OVERRIDE BC_BATCH_OVERRIDE BC_BATCH_OVERRIDE BC_BATCH_OVERRIDE BC_BATCH_OVERRIDE PATHMERGE_BC_BATCH_SIZE)
A_BATCH=(1024 1024 1024 9792 16384 4096)
A_PATHTYPE=(um chunked pure um chunked pathmerge)
A_MODE=(in_capacity non_chunk pure oversubscribed_stress chunked_stress comparator)

CMP_CLASS=(
    same_impl_diff_batch same_impl_diff_batch
    same_batch_diff_path same_batch_diff_path same_batch_diff_path
    pathmerge_cross pathmerge_cross pathmerge_cross pathmerge_cross pathmerge_cross
)
CMP_A=(
    gpu_opt_b9792 gpu_opt_pure_chunked_b16384
    gpu_opt_b1024 gpu_opt_b1024 gpu_opt_pure_b1024
    pathmerge_b4096 pathmerge_b4096 pathmerge_b4096 pathmerge_b4096 pathmerge_b4096
)
CMP_B=(
    gpu_opt_b1024 gpu_opt_pure_chunked_b1024
    gpu_opt_pure_b1024 gpu_opt_pure_chunked_b1024 gpu_opt_pure_chunked_b1024
    gpu_opt_b1024 gpu_opt_b9792 gpu_opt_pure_b1024 gpu_opt_pure_chunked_b1024 gpu_opt_pure_chunked_b16384
)

B_NAMES=(pure_b4096 pure_b8192 um_b10240 um_b12288 chunked_b16384)
B_IMPLS=(gpu_opt_pure gpu_opt_pure gpu_opt gpu_opt gpu_opt_pure_chunked)
B_LABELS=(GPU_Opt_Pure GPU_Opt_Pure GPU_Opt GPU_Opt GPU_Opt_Pure_Chunked)
B_BATCH=(4096 8192 10240 12288 16384)
B_PATHTYPE=(pure pure um um chunked)
B_EXPECT=(success cuda_oom success failure success)
B_EXPECT_TEXT=(pure_success_boundary pure_cuda_oom um_success_boundary um_failure_status_not_assumed_oom chunked_tested_limit_success)

if [ "${DRY_RUN}" = "1" ]; then
    printf '%s\n' \
        "DRY RUN: no build, runner, GPU access, qsub, result update, or BC dump" \
        "Project    : ${PROJECT_DIR}" \
        "Runner     : ${RUNNER}" \
        "Graph      : ${GRAPH_REL} (n=${EXPECTED_N}, m=${EXPECTED_M}, sha=${EXPECTED_GRAPH_SHA})" \
        "Series     : ${SERIES}" \
        "Output     : fresh result_corrected_325557_<timestamp>_<PBS_JOBID>; collision is fatal"
    echo "Series A (first failure is recorded, then the job exits nonzero):"
    for i in "${!A_NAMES[@]}"; do
        printf '  %d. %-30s impl=%-22s batch=%-6s path=%s\n' \
            "$((i + 1))" "${A_NAMES[$i]}" "${A_IMPLS[$i]}" "${A_BATCH[$i]}" "${A_PATHTYPE[$i]}"
    done
    printf '  7. validate all six vectors (length=%s, exact index set, finite values), then run %d comparisons\n' \
        "${EXPECTED_N}" "${#CMP_A[@]}"
    echo "  Comparison failure preserves every acquired vector and exits nonzero after the comparison matrix is complete."
    echo "Series B (record outcome and continue through all five configurations):"
    for i in "${!B_NAMES[@]}"; do
        printf '  %d. %-18s impl=%-22s batch=%-6s expectation=%s\n' \
            "$((i + 1))" "${B_NAMES[$i]}" "${B_IMPLS[$i]}" "${B_BATCH[$i]}" "${B_EXPECT_TEXT[$i]}"
    done
    printf '%s\n' \
        "Series A failures: runner nonzero, OOM, TIMEOUT, CUDA error, SIGKILL, FAIL marker, missing/incomplete vector" \
        "Structural aborts: checkpoint mismatch, graph SHA/n/m mismatch, graph validation failure, build failure, output collision" \
        "Series B: expected failure and unexpected failure are distinct; failed RuntimeSec=not_recorded (never 0)" \
        "Comparison: abs_diff <= ${ABS_TOL} + ${REL_TOL} * max(|a|,|b|) (unchanged)" \
        "Future submission command (display only; DO NOT run in DRY_RUN):" \
        "cd /work/gj17/j17000/m5291091/lab/thesis_bc_project" \
        "qsub -v EXPECTED_SHA=<POST_COMMIT_SHA>,EXPECTED_GRAPH_SHA=8373244f209a3ee489fe72a7b237a5639d142e3a10ac451a2c81b09194eeaa22,SERIES=AB \\" \
        "  scripts/run_corrected_325557_validation.sh" \
        "<POST_COMMIT_SHA> is determined only after the checkpoint commit; it is intentionally not replaced by the current uncommitted HEAD."
    exit 0
fi

abort() {
    echo "ABORTED: $*" >&2
    [ -n "${RUN_LOG:-}" ] && printf 'ABORTED: %s\n' "$*" >> "${RUN_LOG}" 2>/dev/null
    exit 2
}

fail_run() {
    echo "FAILED: $*" >&2
    [ -n "${RUN_LOG:-}" ] && printf 'FAILED: %s\n' "$*" >> "${RUN_LOG}" 2>/dev/null
    [ -n "${MANIFEST:-}" ] && printf 'final_status=FAILED\nfinal_reason=%s\n' "$*" >> "${MANIFEST}" 2>/dev/null
    exit 3
}

: "${EXPECTED_SHA:?EXPECTED_SHA must be set (post-commit checkpoint SHA)}"
ACTUAL_SHA="$(git rev-parse HEAD)" || abort "cannot resolve HEAD"
[ "${ACTUAL_SHA}" = "${EXPECTED_SHA}" ] || abort "checkpoint mismatch (HEAD=${ACTUAL_SHA} != EXPECTED_SHA=${EXPECTED_SHA})"
case "${SERIES}" in A|B|AB) ;; *) abort "invalid SERIES=${SERIES} (expected A, B, or AB)" ;; esac

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
[ "${GRAPH_SHA}" != "${LEGACY_GRAPH_SHA}" ] || abort "legacy malformed graph selected (${LEGACY_GRAPH_REL})"
[ "${GRAPH_SHA}" = "${EXPECTED_GRAPH_SHA}" ] || abort "graph sha256 mismatch (${GRAPH_SHA} != ${EXPECTED_GRAPH_SHA})"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${PBS_JOBID:-not_pbs}"
RESULT_DIR="${RESULT_DIR:-${BUILD_DIR}/result_corrected_325557_${TIMESTAMP}_${JOB_ID}}"
[ ! -e "${RESULT_DIR}" ] || abort "output collision: ${RESULT_DIR} already exists"
mkdir -p "$(dirname "${RESULT_DIR}")" || abort "cannot create output parent"
mkdir "${RESULT_DIR}" || abort "cannot create fresh result directory: ${RESULT_DIR}"

MANIFEST="${RESULT_DIR}/MANIFEST.txt"
RUN_LOG="${RESULT_DIR}/run.log"
IMPL_MANIFEST="${RESULT_DIR}/implementation_manifest.tsv"
VECTOR_INVENTORY="${RESULT_DIR}/vector_inventory.tsv"
CMP_MATRIX="${RESULT_DIR}/comparison_matrix.tsv"
FEASIBILITY="${RESULT_DIR}/feasibility_results.tsv"
: > "${RUN_LOG}" || abort "cannot write run.log"
log() { printf '%s\n' "$*" | tee -a "${RUN_LOG}"; }

for required in "${GRAPH_VALIDATOR}" "${VECTOR_VALIDATOR}" "${COMPARE}" "${MANIFEST_PARSER}"; do
    [ -f "${required}" ] || abort "required script missing: ${required}"
done

log "[Validate graph] ${GRAPH_REL}"
if ! python3 "${GRAPH_VALIDATOR}" "${GRAPH_PATH}" --json "${RESULT_DIR}/graph_validation.json" >> "${RUN_LOG}" 2>&1; then
    abort "graph validation failure: ${GRAPH_REL}"
fi

GRAPH_SIZE="$(stat -c '%s' "${GRAPH_PATH}")"
MAX_DEPTH_ESTIMATE="$(awk -v n="${N}" -v m="${M}" 'BEGIN { d=2.0*m/n; print (d<5.0)?4096:((d<20.0)?256:64) }')"
{
    printf 'checkpoint_sha=%s\n' "${ACTUAL_SHA}"
    printf 'pbs_job_id=%s\n' "${JOB_ID}"
    printf 'series=%s\n' "${SERIES}"
    printf 'graph=%s\n' "${GRAPH_REL}"
    printf 'graph_sha256=%s\n' "${GRAPH_SHA}"
    printf 'graph_size_bytes=%s\n' "${GRAPH_SIZE}"
    printf 'n_nodes=%s\n' "${N}"
    printf 'n_edges=%s\n' "${M}"
    printf 'max_depth_estimate=%s\n' "${MAX_DEPTH_ESTIMATE}"
    printf 'vector_expected_length=%s\n' "${EXPECTED_N}"
    printf 'abs_tol=%s\nrel_tol=%s\n' "${ABS_TOL}" "${REL_TOL}"
    printf 'allocation_values=recorded_log_values_or_code_derived_estimated_values_only\n'
    printf 'unknown_policy=not_recorded;inapplicable_policy=not_applicable\n'
    printf 'pathmerge_formula_policy=PathMerge_specific_only;GPU_Opt_formula_not_applied\n'
} > "${MANIFEST}" || abort "cannot write manifest"

printf '%s\n' $'Implementation\tRequestedBatch\tEffectiveBatch\tRequestedNS\tEffectiveNS\tSubBatch\tNumSubs\tHBMCapacityBytes\tFreeHBMBeforeBytes\tPerSourceStateBytes\tCodeDerivedAllocationBytes\tAllocationFormula\tMemoryMode\tPrefetchMode\tExitCode\tStatus\tFailureReason\tValueSource' > "${IMPL_MANIFEST}"
printf '%s\n' $'Config\tImplementation\tVectorPath\tSHA256\tExpectedLength\tStatus\tValidationJSON' > "${VECTOR_INVENTORY}"
printf '%s\n' $'ComparisonClass\tLabelA\tLabelB\tComparisonExit\tMismatchedElements\tMaxAbsError\tMaxRelError\tSHA256A\tSHA256B\tStatus\tSummaryJSON' > "${CMP_MATRIX}"
printf '%s\n' $'Config\tImplementation\tRequestedBatch\tExpectation\tObserved\tOutcomeClass\tRuntimeSec\tRunnerExit\tFailureReason' > "${FEASIBILITY}"

if [ "${SKIP_BUILD}" != "1" ]; then
    log "[BUILD] scripts/build_miyabi_interactive.sh"
    if ! JOBS="${JOBS}" BUILD_DIR="${BUILD_DIR}" bash "${PROJECT_DIR}/scripts/build_miyabi_interactive.sh" 2>&1 | tee -a "${RUN_LOG}"; then
        abort "build failed"
    fi
fi
[ -x "${RUNNER}" ] || abort "runner not found/executable: ${RUNNER}"

assert_integrity() {
    local head_now graph_now
    head_now="$(git rev-parse HEAD)" || abort "cannot resolve HEAD during run"
    [ "${head_now}" = "${EXPECTED_SHA}" ] || abort "checkpoint changed during run (${head_now})"
    graph_now="$(sha256_file "${GRAPH_PATH}")" || abort "cannot hash graph during run"
    [ "${graph_now}" = "${EXPECTED_GRAPH_SHA}" ] || abort "graph SHA changed during run (${graph_now})"
}

classify_observed() {
    local rc="$1" stderr_file="$2" output_file="$3"
    if grep -Eiq 'out of memory|cudaErrorMemoryAllocation|\bOOM\b' "${stderr_file}" "${output_file}" 2>/dev/null; then
        OBSERVED=OOM
        OBSERVED_REASON="oom_marker;runner_exit=${rc}"
    elif [ "${rc}" -eq 124 ]; then
        OBSERVED=TIMEOUT
        OBSERVED_REASON="timeout_exit_124"
    elif [ "${rc}" -eq 137 ]; then
        OBSERVED=SIGKILL
        OBSERVED_REASON="sigkill_or_timeout_kill_exit_137"
    elif grep -Eiq 'CUDA[ _-]*error|cudaError' "${stderr_file}" "${output_file}" 2>/dev/null; then
        OBSERVED=CUDA_ERROR
        OBSERVED_REASON="cuda_error_marker;runner_exit=${rc}"
    elif grep -Eiq '\bFAIL(ED)?\b|\bFATAL\b' "${stderr_file}" "${output_file}" 2>/dev/null; then
        OBSERVED=FAIL_MARKER
        OBSERVED_REASON="fail_marker;runner_exit=${rc}"
    elif [ "${rc}" -ne 0 ]; then
        OBSERVED=RUNNER_NONZERO
        OBSERVED_REASON="runner_exit=${rc}"
    else
        OBSERVED=SUCCESS
        OBSERVED_REASON=not_applicable
    fi
}

execute_config() {
    local name="$1" impl="$2" batch_env="$3" batch="$4" dump="$5"
    local args=("${impl}" "${GRAPH_PATH}")
    CFG_STDERR="${RESULT_DIR}/${name}.stderr.log"
    if [ "${dump}" = "yes" ]; then
        CFG_OUTPUT="${RESULT_DIR}/${name}.bc.tsv"
        args+=(--dump-bc)
    else
        CFG_OUTPUT="${RESULT_DIR}/${name}.stdout.tsv"
    fi
    [ ! -e "${CFG_STDERR}" ] && [ ! -e "${CFG_OUTPUT}" ] || abort "per-config output collision: ${name}"
    assert_integrity
    CFG_RC=0
    env "${batch_env}=${batch}" timeout -k 30 "${TIMEOUT_SEC}" \
        "${RUNNER}" "${args[@]}" > "${CFG_OUTPUT}" 2> "${CFG_STDERR}" || CFG_RC=$?
    classify_observed "${CFG_RC}" "${CFG_STDERR}" "${CFG_OUTPUT}"
    RUNTIME_SEC=not_recorded
    if [ "${dump}" != "yes" ] && [ "${OBSERVED}" = "SUCCESS" ]; then
        RUNTIME_SEC="$(awk -F '\t' 'NF==4 {print $3; exit}' "${CFG_OUTPUT}")"
        [ -n "${RUNTIME_SEC}" ] || RUNTIME_SEC=not_recorded
    fi
}

record_implementation() {
    local implementation="$1" batch="$2" status="$3" reason="$4"
    if ! python3 "${MANIFEST_PARSER}" \
        --implementation "${implementation}" --requested-batch "${batch}" \
        --nodes "${N}" --max-depth "${MAX_DEPTH_ESTIMATE}" --log "${CFG_STDERR}" \
        --exit-code "${CFG_RC}" --status "${status}" --failure-reason "${reason}" \
        --no-header >> "${IMPL_MANIFEST}"; then
        abort "implementation manifest parser failed for ${implementation}"
    fi
}

declare -A VECTOR_OK

if [ "${SERIES}" = "A" ] || [ "${SERIES}" = "AB" ]; then
    log "=== Series A: correctness vectors; stop at first acquisition failure ==="
    for i in "${!A_NAMES[@]}"; do
        name="${A_NAMES[$i]}"
        log "[A $((i + 1))/6] ${name}: ${A_IMPLS[$i]} batch=${A_BATCH[$i]}"
        execute_config "${name}" "${A_IMPLS[$i]}" "${A_BATCHENV[$i]}" "${A_BATCH[$i]}" yes
        if [ "${OBSERVED}" != "SUCCESS" ]; then
            record_implementation "${A_LABELS[$i]}" "${A_BATCH[$i]}" "SERIES_A_FAILED_${OBSERVED}" "${OBSERVED_REASON}"
            fail_run "Series A ${name}: ${OBSERVED_REASON}"
        fi

        vector_json="${RESULT_DIR}/${name}.vector_validation.json"
        vector_sha=not_recorded
        [ -f "${CFG_OUTPUT}" ] && vector_sha="$(sha256_file "${CFG_OUTPUT}")"
        if [ ! -s "${CFG_OUTPUT}" ]; then
            record_implementation "${A_LABELS[$i]}" "${A_BATCH[$i]}" SERIES_A_MISSING_VECTOR missing_or_empty_vector
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${A_LABELS[$i]}" "${CFG_OUTPUT}" "${vector_sha}" "${EXPECTED_N}" FAIL not_created >> "${VECTOR_INVENTORY}"
            fail_run "Series A ${name}: missing vector"
        fi
        if ! python3 "${VECTOR_VALIDATOR}" "${CFG_OUTPUT}" --expected-length "${EXPECTED_N}" --json "${vector_json}" >> "${RUN_LOG}" 2>&1; then
            record_implementation "${A_LABELS[$i]}" "${A_BATCH[$i]}" SERIES_A_INCOMPLETE_VECTOR "vector_validation_failed:${vector_json}"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${A_LABELS[$i]}" "${CFG_OUTPUT}" "${vector_sha}" "${EXPECTED_N}" FAIL "${vector_json}" >> "${VECTOR_INVENTORY}"
            fail_run "Series A ${name}: incomplete vector (see ${vector_json})"
        fi
        record_implementation "${A_LABELS[$i]}" "${A_BATCH[$i]}" SUCCESS not_applicable
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${A_LABELS[$i]}" "${CFG_OUTPUT}" "${vector_sha}" "${EXPECTED_N}" PASS "${vector_json}" >> "${VECTOR_INVENTORY}"
        VECTOR_OK["${name}"]=yes
    done

    log "=== Series A: all vectors complete; running full comparison matrix ==="
    comparison_failed=0
    for i in "${!CMP_A[@]}"; do
        label_a="${CMP_A[$i]}"; label_b="${CMP_B[$i]}"; class="${CMP_CLASS[$i]}"
        file_a="${RESULT_DIR}/${label_a}.bc.tsv"; file_b="${RESULT_DIR}/${label_b}.bc.tsv"
        [ "${VECTOR_OK[$label_a]:-no}" = yes ] && [ "${VECTOR_OK[$label_b]:-no}" = yes ] \
            || fail_run "internal error: comparison requested before complete vectors"
        summary_md="${RESULT_DIR}/${label_a}__vs__${label_b}.md"
        summary_json="${RESULT_DIR}/${label_a}__vs__${label_b}.json"
        compare_rc=0
        python3 "${COMPARE}" "${file_a}" "${file_b}" \
            --label-a "${label_a}" --label-b "${label_b}" \
            --expected-length "${EXPECTED_N}" --rel-tol "${REL_TOL}" --abs-tol "${ABS_TOL}" \
            --out "${summary_md}" --json "${summary_json}" >> "${RUN_LOG}" 2>&1 || compare_rc=$?
        read -r mismatch max_abs max_rel < <(python3 - "${summary_json}" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("mismatched_elements", "not_recorded"),
          d.get("max_abs_error", "not_recorded"),
          d.get("max_rel_error", "not_recorded"))
except Exception:
    print("not_recorded not_recorded not_recorded")
PY
)
        compare_status=PASS
        if [ "${compare_rc}" -ne 0 ]; then
            compare_status=FAIL
            comparison_failed=1
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${class}" "${label_a}" "${label_b}" "${compare_rc}" "${mismatch}" "${max_abs}" "${max_rel}" \
            "$(sha256_file "${file_a}")" "$(sha256_file "${file_b}")" "${compare_status}" "${summary_json}" >> "${CMP_MATRIX}"
        log "[COMPARE ${class}] ${label_a} vs ${label_b}: ${compare_status}"
    done
    [ "${comparison_failed}" -eq 0 ] || fail_run "Series A vector comparison failure; acquired vectors retained"
fi

if [ "${SERIES}" = "B" ] || [ "${SERIES}" = "AB" ]; then
    log "=== Series B: feasibility; record every outcome and continue ==="
    series_b_unexpected=0
    for i in "${!B_NAMES[@]}"; do
        name="${B_NAMES[$i]}"
        log "[B $((i + 1))/5] ${name}: ${B_IMPLS[$i]} batch=${B_BATCH[$i]} expectation=${B_EXPECT_TEXT[$i]}"
        execute_config "${name}" "${B_IMPLS[$i]}" BC_BATCH_OVERRIDE "${B_BATCH[$i]}" no
        outcome_class=not_recorded
        case "${B_EXPECT[$i]}:${OBSERVED}" in
            success:SUCCESS) outcome_class=EXPECTED_SUCCESS ;;
            success:*) outcome_class=UNEXPECTED_FAILURE; series_b_unexpected=1 ;;
            cuda_oom:OOM) outcome_class=EXPECTED_CUDA_OOM ;;
            cuda_oom:SUCCESS) outcome_class=UNEXPECTED_SUCCESS; series_b_unexpected=1 ;;
            cuda_oom:*) outcome_class=UNEXPECTED_FAILURE_NOT_CUDA_OOM; series_b_unexpected=1 ;;
            failure:SUCCESS) outcome_class=UNEXPECTED_SUCCESS; series_b_unexpected=1 ;;
            failure:*) outcome_class=EXPECTED_FAILURE_STATUS ;;
        esac
        record_implementation "${B_LABELS[$i]}" "${B_BATCH[$i]}" "${outcome_class}" "${OBSERVED_REASON}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${name}" "${B_LABELS[$i]}" "${B_BATCH[$i]}" "${B_EXPECT_TEXT[$i]}" "${OBSERVED}" \
            "${outcome_class}" "${RUNTIME_SEC}" "${CFG_RC}" "${OBSERVED_REASON}" >> "${FEASIBILITY}"
        log "  observed=${OBSERVED}; outcome=${outcome_class}; runtime=${RUNTIME_SEC}"
    done
    [ "${series_b_unexpected}" -eq 0 ] || fail_run "Series B completed all configs but one or more outcomes were unexpected"
fi

printf 'final_status=SUCCESS\n' >> "${MANIFEST}"
log "=== Complete ==="
log "manifest=${MANIFEST}"
log "implementation_manifest=${IMPL_MANIFEST}"
log "Series A/B timings are correctness/feasibility observations, not performance results."
exit 0
