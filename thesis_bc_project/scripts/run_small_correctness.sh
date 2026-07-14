#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=2:00:00
#PBS -N bc_small_correct
#PBS -W group_list=gj17
#PBS -j oe

set -euo pipefail

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}"
    PROJECT_DIR="${PBS_O_WORKDIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    cd "${PROJECT_DIR}"
fi

BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build_miyabi}"
RUNNER="${BUILD_DIR}/run_benchmark"
COMPARE="${PROJECT_DIR}/scripts/compare_bc_vectors.py"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
JOBS="${JOBS:-8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
ABS_TOL="${ABS_TOL:-1e-3}"
REL_TOL="${REL_TOL:-1e-6}"
REQUESTED_BATCH="${BC_BATCH_OVERRIDE:-auto_default}"
GRAPHS=(
    "data/benchmark_7000_41459"
    "data/benchmark_11023_62184"
    "data/chain_200"
)

if [ "${DRY_RUN}" = "1" ]; then
    printf '%s\n' \
        "DRY RUN: no build, runner, GPU access, qsub, result update, or BC dump" \
        "Project: ${PROJECT_DIR}" \
        "Runner: ${RUNNER}" \
        "Planned output: ${BUILD_DIR}/result_small_correctness_<timestamp>_<PBS_JOBID>/" \
        "Runs per graph (n=1, no warmup):" \
        "  ${RUNNER} sequential <graph> --dump-bc" \
        "  ${RUNNER} gpu_opt <graph> --dump-bc" \
        "Comparison: abs_diff <= ${ABS_TOL} + ${REL_TOL} * max(|reference|,|candidate|)" \
        "Required comparison result: zero mixed-tolerance mismatches, complete indices, finite values" \
        "Requested GPU batch: ${REQUESTED_BATCH}"
    printf 'Graphs:\n'
    printf '  %s\n' "${GRAPHS[@]}"
    exit 0
fi

: "${EXPECTED_SHA:?EXPECTED_SHA must be set}"
ACTUAL_SHA="$(git rev-parse HEAD)"

test "$ACTUAL_SHA" = "$EXPECTED_SHA" || {
    echo "ERROR: checkpoint mismatch" >&2
    exit 2
}

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${PBS_JOBID:-not_pbs}"
RESULT_DIR="${BUILD_DIR}/result_small_correctness_${TIMESTAMP}_${JOB_ID}"
MANIFEST="${RESULT_DIR}/MANIFEST.txt"
SUMMARY="${RESULT_DIR}/correctness_summary.tsv"
RUN_LOG="${RESULT_DIR}/run.log"
mkdir -p "${RESULT_DIR}"
: > "${RUN_LOG}"

log() {
    printf '%s\n' "$*" | tee -a "${RUN_LOG}"
}

sha256_file() {
    sha256sum "$1" | awk '{print $1}'
}

md_value() {
    local key="$1"
    local file="$2"
    awk -F '|' -v wanted="${key}" '
        {
            k=$2
            v=$3
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            if (k == wanted) { print v; exit }
        }
    ' "${file}"
}

validate_dump() {
    local dump_file="$1"
    local expected_n="$2"
    awk -F '\t' -v expected="${expected_n}" '
        BEGIN {
            numeric="^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$"
        }
        /^#/ { headers++; next }
        NF == 0 { next }
        {
            lower=tolower($2)
            special=(lower == "nan" || lower == "+nan" || lower == "-nan" ||
                     lower == "inf" || lower == "+inf" || lower == "-inf" ||
                     lower == "infinity" || lower == "+infinity" || lower == "-infinity")
            if (NF != 2 || $1 !~ /^[0-9]+$/ || $1 < 0 || $1 >= expected ||
                seen[$1]++ || ($2 !~ numeric && !special)) {
                bad=1
            }
            count++
        }
        END {
            if (headers != 1 || count != expected) bad=1
            for (i=0; i<expected; i++) if (!(i in seen)) bad=1
            exit bad ? 1 : 0
        }
    ' "${dump_file}"
}

log_has_failure_marker() {
    grep -Eiq '(^|[^[:alnum:]_])(FAIL(ED|URE)?|OOM|TIMEOUT|NaN|[+-]?Inf(inity)?)([^[:alnum:]_]|$)' "$1"
}

record_graph() {
    local status="$1"
    local reason="$2"
    local row=(
        "${ACTUAL_SHA}" "${JOB_ID}" "${graph_rel}" "${graph_sha}" "${n}" "${m}"
        "${seq_rc}" "${gpu_rc}" "${comparison_rc}" "${seq_sha}" "${gpu_sha}"
        "${REQUESTED_BATCH}" "${effective_batch}" "${sub_batch}" "${num_subs}" "${ns_eff}"
        "${ABS_TOL}" "${REL_TOL}" "${len_a}" "${len_b}" "${missing_a}" "${missing_b}"
        "${mismatches}" "${max_abs}" "${max_abs_idx}" "${max_abs_a}" "${max_abs_b}"
        "${max_rel}" "${max_rel_idx}" "${max_rel_a}" "${max_rel_b}"
        "${max_bc_a_idx}" "${max_bc_a_value}" "${max_bc_b_idx}" "${max_bc_b_value}"
        "${status}" "${reason}"
    )
    (IFS=$'\t'; printf '%s\n' "${row[*]}") >> "${SUMMARY}"

    {
        printf '\n[graph %s]\n' "${graph_name}"
        printf 'path=%s\nsha256=%s\nn=%s\nm=%s\n' "${graph_rel}" "${graph_sha}" "${n}" "${m}"
        printf 'sequential_exit=%s\ngpu_opt_exit=%s\ncomparison_exit=%s\n' "${seq_rc}" "${gpu_rc}" "${comparison_rc}"
        printf 'sequential_vector_sha256=%s\ngpu_opt_vector_sha256=%s\n' "${seq_sha}" "${gpu_sha}"
        printf 'requested_batch=%s\neffective_batch=%s\nSUB_BATCH=%s\nnum_subs=%s\nNS_eff=%s\n' \
            "${REQUESTED_BATCH}" "${effective_batch}" "${sub_batch}" "${num_subs}" "${ns_eff}"
        printf 'vector_length_reference=%s\nvector_length_candidate=%s\n' "${len_a}" "${len_b}"
        printf 'missing_reference_only=%s\nmissing_candidate_only=%s\nmismatched_elements=%s\n' \
            "${missing_a}" "${missing_b}" "${mismatches}"
        printf 'max_abs_error=%s\nmax_abs_index=%s\nmax_abs_reference=%s\nmax_abs_candidate=%s\n' \
            "${max_abs}" "${max_abs_idx}" "${max_abs_a}" "${max_abs_b}"
        printf 'max_rel_error=%s\nmax_rel_index=%s\nmax_rel_reference=%s\nmax_rel_candidate=%s\n' \
            "${max_rel}" "${max_rel_idx}" "${max_rel_a}" "${max_rel_b}"
        printf 'max_bc_reference_index=%s\nmax_bc_reference_value=%s\n' "${max_bc_a_idx}" "${max_bc_a_value}"
        printf 'max_bc_candidate_index=%s\nmax_bc_candidate_value=%s\n' "${max_bc_b_idx}" "${max_bc_b_value}"
        printf 'status=%s\nreason=%s\n' "${status}" "${reason}"
    } >> "${MANIFEST}"
}

printf '%s\n' \
    "checkpoint_sha=${ACTUAL_SHA}" \
    "pbs_job_id=${JOB_ID}" \
    "project_dir=${PROJECT_DIR}" \
    "result_dir=${RESULT_DIR}" \
    "runner=${RUNNER}" \
    "runs_per_configuration=1" \
    "warmup=none" \
    "timing_usage=correctness_only_not_performance" \
    "abs_tol=${ABS_TOL}" \
    "rel_tol=${REL_TOL}" \
    "criterion=abs_diff <= abs_tol + rel_tol * max(abs(reference),abs(candidate))" \
    "requested_batch_source=BC_BATCH_OVERRIDE_or_auto_default" > "${MANIFEST}"

printf '%s\n' \
    $'checkpoint_sha\tpbs_job_id\tgraph_path\tgraph_sha256\tn\tm\tsequential_exit\tgpu_opt_exit\tcomparison_exit\tsequential_vector_sha256\tgpu_opt_vector_sha256\trequested_batch\teffective_batch\tSUB_BATCH\tnum_subs\tNS_eff\tabs_tol\trel_tol\treference_vector_length\tcandidate_vector_length\tmissing_reference_only\tmissing_candidate_only\tmismatched_elements\tmax_abs_error\tmax_abs_index\tmax_abs_reference\tmax_abs_candidate\tmax_rel_error\tmax_rel_index\tmax_rel_reference\tmax_rel_candidate\tmax_bc_reference_index\tmax_bc_reference_value\tmax_bc_candidate_index\tmax_bc_candidate_value\tstatus\treason' \
    > "${SUMMARY}"

log "checkpoint=${ACTUAL_SHA} job=${JOB_ID} result=${RESULT_DIR}"

if [ "${SKIP_BUILD}" != "1" ]; then
    log "[BUILD] scripts/build_miyabi_interactive.sh (output is not a performance result)"
    JOBS="${JOBS}" BUILD_DIR="${BUILD_DIR}" \
        bash "${PROJECT_DIR}/scripts/build_miyabi_interactive.sh" 2>&1 | tee -a "${RUN_LOG}"
fi

if [ ! -x "${RUNNER}" ]; then
    log "ERROR: runner not found or not executable: ${RUNNER}"
    exit 2
fi

run_graph() {
    local graph_rel="$1"
    local graph_path="${PROJECT_DIR}/${graph_rel}"
    local graph_name
    graph_name="$(basename "${graph_rel}")"
    local graph_dir="${RESULT_DIR}/${graph_name}"
    local n m graph_sha
    read -r n m < "${graph_path}"
    graph_sha="$(sha256_file "${graph_path}")"
    mkdir -p "${graph_dir}"

    local seq_dump="${graph_dir}/sequential.bc.tsv"
    local gpu_dump="${graph_dir}/gpu_opt.bc.tsv"
    local seq_stderr="${graph_dir}/sequential.stderr.log"
    local gpu_stderr="${graph_dir}/gpu_opt.stderr.log"
    local comparison="${graph_dir}/comparison.md"

    local seq_rc=not_recorded gpu_rc=not_recorded comparison_rc=not_recorded
    local seq_sha=not_recorded gpu_sha=not_recorded
    local effective_batch=not_recorded sub_batch=not_recorded num_subs=not_recorded ns_eff=not_recorded
    local len_a=not_recorded len_b=not_recorded missing_a=not_recorded missing_b=not_recorded
    local mismatches=not_recorded max_abs=not_recorded max_abs_idx=not_recorded
    local max_abs_a=not_recorded max_abs_b=not_recorded max_rel=not_recorded max_rel_idx=not_recorded
    local max_rel_a=not_recorded max_rel_b=not_recorded
    local max_bc_a_idx=not_recorded max_bc_a_value=not_recorded
    local max_bc_b_idx=not_recorded max_bc_b_value=not_recorded

    log "[RUN] sequential ${graph_rel} (n=1, no warmup)"
    seq_rc=0
    timeout "${TIMEOUT_SEC}" "${RUNNER}" sequential "${graph_path}" --dump-bc \
        > "${seq_dump}" 2> "${seq_stderr}" || seq_rc=$?
    [ -f "${seq_dump}" ] && seq_sha="$(sha256_file "${seq_dump}")"
    log "[EXIT] sequential ${graph_name}: ${seq_rc}"
    if [ "${seq_rc}" -ne 0 ]; then
        record_graph FAIL "sequential_runner_exit_${seq_rc}"
        return "${seq_rc}"
    fi

    log "[RUN] gpu_opt ${graph_rel} (n=1, no warmup)"
    gpu_rc=0
    timeout "${TIMEOUT_SEC}" "${RUNNER}" gpu_opt "${graph_path}" --dump-bc \
        > "${gpu_dump}" 2> "${gpu_stderr}" || gpu_rc=$?
    [ -f "${gpu_dump}" ] && gpu_sha="$(sha256_file "${gpu_dump}")"
    log "[EXIT] gpu_opt ${graph_name}: ${gpu_rc}"
    if [ "${gpu_rc}" -ne 0 ]; then
        record_graph FAIL "gpu_opt_runner_exit_${gpu_rc}"
        return "${gpu_rc}"
    fi

    local mem_line
    mem_line="$(grep '\[Mem\].*BATCH=.*SUB_BATCH=.*num_subs=.*NS_eff=' "${gpu_stderr}" | tail -n 1 || true)"
    if [ -n "${mem_line}" ]; then
        effective_batch="$(printf '%s\n' "${mem_line}" | sed -n 's/.*dynamic(UM).*BATCH=\([0-9][0-9]*\), SUB_BATCH=.*/\1/p')"
        sub_batch="$(printf '%s\n' "${mem_line}" | sed -n 's/.*SUB_BATCH=\([0-9][0-9]*\),.*/\1/p')"
        num_subs="$(printf '%s\n' "${mem_line}" | sed -n 's/.*num_subs=\([0-9][0-9]*\),.*/\1/p')"
        ns_eff="$(printf '%s\n' "${mem_line}" | sed -n 's/.*NS_eff=\([0-9][0-9]*\).*/\1/p')"
        effective_batch="${effective_batch:-not_recorded}"
        sub_batch="${sub_batch:-not_recorded}"
        num_subs="${num_subs:-not_recorded}"
        ns_eff="${ns_eff:-not_recorded}"
    fi

    if log_has_failure_marker "${seq_stderr}" || log_has_failure_marker "${gpu_stderr}"; then
        record_graph FAIL "failure_marker_in_stderr"
        return 4
    fi
    if ! validate_dump "${seq_dump}" "${n}" || ! validate_dump "${gpu_dump}" "${n}"; then
        record_graph FAIL "invalid_or_incomplete_vector"
        return 4
    fi

    comparison_rc=0
    python3 "${COMPARE}" "${seq_dump}" "${gpu_dump}" \
        --label-a Sequential --label-b GPU_Opt \
        --abs-tol "${ABS_TOL}" --rel-tol "${REL_TOL}" --out "${comparison}" \
        --extra "checkpoint_sha=${ACTUAL_SHA}" "pbs_job_id=${JOB_ID}" \
        "graph_path=${graph_path}" "graph_sha256=${graph_sha}" "n=${n}" "m=${m}" \
        "requested_batch=${REQUESTED_BATCH}" \
        "effective_batch=${effective_batch}" "SUB_BATCH=${sub_batch}" \
        "num_subs=${num_subs}" "NS_eff=${ns_eff}" \
        >> "${RUN_LOG}" 2>&1 || comparison_rc=$?

    len_a="$(md_value 'ベクトル長 A' "${comparison}")"
    len_b="$(md_value 'ベクトル長 B' "${comparison}")"
    missing_a="$(md_value '欠損 index 数 (A のみ)' "${comparison}")"
    missing_b="$(md_value '欠損 index 数 (B のみ)' "${comparison}")"
    mismatches="$(sed -n 's/.*不一致要素数 | \([0-9][0-9]*\) |$/\1/p' "${comparison}" | tail -n 1)"

    local value_pair max_bc_pair
    value_pair="$(md_value '最大絶対誤差' "${comparison}")"
    max_abs="${value_pair%% *}"
    max_abs_idx="$(printf '%s\n' "${value_pair}" | sed -n 's/.*index \([^)]*\)).*/\1/p')"
    value_pair="$(md_value '最大絶対誤差 index の値' "${comparison}")"
    max_abs_a="$(printf '%s\n' "${value_pair}" | sed -n 's/^A=\([^,]*\), B=.*/\1/p')"
    max_abs_b="$(printf '%s\n' "${value_pair}" | sed -n 's/^A=[^,]*, B=\(.*\)$/\1/p')"

    value_pair="$(md_value '最大相対誤差' "${comparison}")"
    max_rel="${value_pair%% *}"
    max_rel_idx="$(printf '%s\n' "${value_pair}" | sed -n 's/.*index \([^)]*\)).*/\1/p')"
    value_pair="$(md_value '最大相対誤差 index の値' "${comparison}")"
    max_rel_a="$(printf '%s\n' "${value_pair}" | sed -n 's/^A=\([^,]*\), B=.*/\1/p')"
    max_rel_b="$(printf '%s\n' "${value_pair}" | sed -n 's/^A=[^,]*, B=\(.*\)$/\1/p')"

    max_bc_pair="$(md_value 'Max BC A' "${comparison}")"
    max_bc_a_idx="$(printf '%s\n' "${max_bc_pair}" | sed -n 's/^index \([^,]*\), value .*/\1/p')"
    max_bc_a_value="$(printf '%s\n' "${max_bc_pair}" | sed -n 's/^index [^,]*, value \(.*\)$/\1/p')"
    max_bc_pair="$(md_value 'Max BC B' "${comparison}")"
    max_bc_b_idx="$(printf '%s\n' "${max_bc_pair}" | sed -n 's/^index \([^,]*\), value .*/\1/p')"
    max_bc_b_value="$(printf '%s\n' "${max_bc_pair}" | sed -n 's/^index [^,]*, value \(.*\)$/\1/p')"

    if [ "${comparison_rc}" -ne 0 ] || [ "${mismatches}" != "0" ] || \
       [ "${missing_a}" != "0" ] || [ "${missing_b}" != "0" ] || \
       [ "${len_a}" != "${n}" ] || [ "${len_b}" != "${n}" ]; then
        record_graph FAIL "full_vector_comparison_failed"
        return 3
    fi

    record_graph PASS "mixed_tolerance_mismatches_0"
    log "[PASS] ${graph_name}: all ${n} elements within mixed tolerance"
}

for graph in "${GRAPHS[@]}"; do
    run_graph "${graph}"
done

log "PASS: all three graphs completed; timings are not performance results"
