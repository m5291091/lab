#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=6:00:00
#PBS -N bc_mem_correct
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
#  run_memory_correctness.sh — メモリ経路 正確性 比較行列 (Stage 4A / Gate G2.0)
#
#  1 グラフ (325557_3216152) だけを用い、UM / Pure / Chunked のメモリ経路と
#  バッチ差、および外部比較対象 PathMerge の BC を、全構成を 1 度ずつ実行してから
#  最後にまとめて比較する「比較行列 (comparison matrix)」方式。
#  実行途中の数値不一致で後続構成を止めない (構成単位で記録して継続)。
#
#  重要: PathMerge は唯一の ground truth ではなく external comparator
#  (cross-implementation comparator) として扱う。数値不一致を許容値緩和で PASS にしない。
#
#  実行順 (各 1 回のみ; 途中の数値不一致では止めない):
#    1. GPU_Opt b9792              (gpu_opt, BC_BATCH_OVERRIDE=9792)  UM oversubscription
#    2. GPU_Opt b1024              (gpu_opt, BC_BATCH_OVERRIDE=1024)   同一実装 in-capacity 対照
#    3. GPU_Opt_Pure_Chunked b16384(gpu_opt_pure_chunked, =16384)      num_subs>1 chunking
#    4. GPU_Opt_Pure_Chunked b1024 (gpu_opt_pure_chunked, =1024)       非 chunk 対照
#    5. GPU_Opt_Pure b1024         (gpu_opt_pure, =1024)               Pure/cudaMalloc
#    6. PathMerge b4096            (pathmerge_bc, PATHMERGE_BC_BATCH_SIZE=4096) external comparator
#
#  構造的失敗 (checkpoint/build/graph SHA・n/m/runner・compare 不在/mkdir/manifest) は
#  ジョブ全体を即停止 (ABORTED)。構成単位の失敗 (runner非0/OOM/TIMEOUT/vector欠損/NaN/Inf/
#  経路証拠不足/batch不一致) は該当構成を FAIL 記録し次の独立構成へ進む。自動再試行しない。
#
#  出力 (RESULT_DIR, build_miyabi 配下 = gitignored):
#    MANIFEST.txt / execution_summary.tsv / comparison_matrix.tsv / run.log
#    <config>.bc.tsv / <config>.stderr.log
#    <a>__vs__<b>.md   (比較ごと)
#
#  最終判定は 2 系統に分離: core_memory_path_status (CORE 5件) と
#  pathmerge_cross_impl_status (PathMerge 診断 5件) から overall_status を決める。
#  overall: PASS / CORE_PASS_CROSS_IMPL_DIFFERENCE / CORE_FAIL / INCOMPLETE / ABORTED。
#  CORE_PASS_CROSS_IMPL_DIFFERENCE は「完全な正確性証明」ではない。
#
#  環境変数:
#    EXPECTED_SHA         checkpoint SHA (実行時必須; HEAD と不一致なら ABORTED exit 2)
#    EXPECTED_GRAPH_SHA   グラフ SHA256 (既定=325557 の既知値; 空で検査省略)
#    EXPECTED_N/EXPECTED_M グラフ n/m (既定 325557/3216152; 空で検査省略)
#    DRY_RUN 1 で計画表示のみ / SKIP_BUILD 1 / JOBS(8) / TIMEOUT_SEC(5400)
#    ABS_TOL(1e-3) / REL_TOL(1e-6) / GRAPH(data/325557_3216152)
# ============================================================

set -uo pipefail   # set -e は使わない: 構成単位失敗で止めないため個別に exit を制御

if [ -n "${PBS_O_WORKDIR:-}" ]; then
    cd "${PBS_O_WORKDIR}"; PROJECT_DIR="${PBS_O_WORKDIR}"
else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${PROJECT_DIR}"
fi

BUILD_DIR="${BUILD_DIR:-${PROJECT_DIR}/build_miyabi}"
RUNNER="${BUILD_DIR}/run_benchmark"
COMPARE="${PROJECT_DIR}/scripts/compare_bc_vectors.py"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
JOBS="${JOBS:-8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-5400}"
ABS_TOL="${ABS_TOL:-1e-3}"
REL_TOL="${REL_TOL:-1e-6}"
GRAPH_REL="${GRAPH:-data/325557_3216152}"
EXPECTED_GRAPH_SHA="${EXPECTED_GRAPH_SHA:-a095b2e7564e6c620bd0f5437917e0b28f4fecab289adf77633e850aa07da584}"
EXPECTED_N="${EXPECTED_N:-325557}"
EXPECTED_M="${EXPECTED_M:-3216152}"

# --- Miyabi-G メモリ制約 (Gate G1.1) ------------------------------------
# 1 ノードの利用可能ホストメモリは 100 GiB (= 約 107.374 GB 10進)。大メモリ queue では解決しない。
readonly MIYABI_HOST_MEM_GIB=100
readonly MIYABI_HOST_MEM_GB=107.374        # 100 * 2^30 / 1e9
readonly UM_NS=2                           # host_um.cu の NS 定数 (表示 dynamic(UM) は NS=2 前提)

# --- 検証構成 (各 1 回のみ実行) -----------------------------------------
CONFIG_NAMES=(gpu_opt_b9792 gpu_opt_b1024 gpu_opt_pure_chunked_b16384 gpu_opt_pure_chunked_b1024 gpu_opt_pure_b1024 pathmerge_b4096)
CONFIG_IMPLS=(gpu_opt gpu_opt gpu_opt_pure_chunked gpu_opt_pure_chunked gpu_opt_pure pathmerge_bc)
CONFIG_BATCHENV=(BC_BATCH_OVERRIDE BC_BATCH_OVERRIDE BC_BATCH_OVERRIDE BC_BATCH_OVERRIDE BC_BATCH_OVERRIDE PATHMERGE_BC_BATCH_SIZE)
CONFIG_BATCH=(9792 1024 16384 1024 1024 4096)
CONFIG_PATHTYPE=(um um chunked chunked pure pathmerge)
CONFIG_MODE=(oversubscribed in_capacity chunked non_chunk pure comparator)

# --- 比較行列 (class, A, B) ---------------------------------------------
CMP_CLASS=(
    same_impl_diff_batch same_impl_diff_batch
    same_batch_diff_path same_batch_diff_path same_batch_diff_path
    pathmerge_cross pathmerge_cross pathmerge_cross pathmerge_cross pathmerge_cross)
CMP_A=(
    gpu_opt_b9792 gpu_opt_pure_chunked_b16384
    gpu_opt_b1024 gpu_opt_b1024 gpu_opt_pure_b1024
    pathmerge_b4096 pathmerge_b4096 pathmerge_b4096 pathmerge_b4096 pathmerge_b4096)
CMP_B=(
    gpu_opt_b1024 gpu_opt_pure_chunked_b1024
    gpu_opt_pure_b1024 gpu_opt_pure_chunked_b1024 gpu_opt_pure_chunked_b1024
    gpu_opt_b1024 gpu_opt_b9792 gpu_opt_pure_b1024 gpu_opt_pure_chunked_b1024 gpu_opt_pure_chunked_b16384)

# ============================================================
#  DRY_RUN: 計画表示のみ。build / runner / GPU / qsub / 比較 / result 更新なし。
# ============================================================
if [ "${DRY_RUN}" = "1" ]; then
    printf '%s\n' \
        "DRY RUN: no build, runner, GPU access, qsub, result update, or BC dump" \
        "Project    : ${PROJECT_DIR}" \
        "Runner     : ${RUNNER}" \
        "Compare    : ${COMPARE}" \
        "Graph      : ${GRAPH_REL} (expect n=${EXPECTED_N} m=${EXPECTED_M} sha=${EXPECTED_GRAPH_SHA:0:12}...)" \
        "Planned out: ${BUILD_DIR}/result_memory_correctness_<timestamp>_<PBS_JOBID>/" \
        "Design     : comparison matrix; run all 6 configs once (record+continue on per-config fail)," \
        "             then compare all pairs. PathMerge = external comparator (NOT ground truth)." \
        "Configs (n=1 each, no warmup; timings NOT performance results):"
    for i in "${!CONFIG_NAMES[@]}"; do
        printf '  %d. %s: %s=%s %s (path=%s mode=%s)\n' \
            "$((i+1))" "${CONFIG_NAMES[$i]}" "${CONFIG_BATCHENV[$i]}" "${CONFIG_BATCH[$i]}" \
            "${CONFIG_IMPLS[$i]}" "${CONFIG_PATHTYPE[$i]}" "${CONFIG_MODE[$i]}"
    done
    printf 'Comparison matrix (%d comparisons; CORE_MEMORY_PATH=5 required, PATHMERGE_CROSS_IMPL_DIAGNOSTIC=5 diagnostic):\n' "${#CMP_A[@]}"
    for i in "${!CMP_A[@]}"; do
        printf '  [%s] %s  vs  %s\n' "${CMP_CLASS[$i]}" "${CMP_A[$i]}" "${CMP_B[$i]}"
    done
    printf '%s\n' \
        "Comparison : abs_diff <= ${ABS_TOL} + ${REL_TOL} * max(|a|,|b|); Max BC 一致だけでは PASS にしない" \
        "Structural ABORT: checkpoint/build/graph sha/n/m/runner/compare/mkdir/manifest" \
        "Per-config FAIL+continue: runner!=0/OOM(137)/TIMEOUT/missing vector/NaN,Inf/path evidence/batch" \
        "Status (2-tier): core_memory_path_status(CORE 5) / pathmerge_cross_impl_status(CROSS 5) => overall_status" \
        "  overall: PASS / CORE_PASS_CROSS_IMPL_DIFFERENCE / CORE_FAIL / INCOMPLETE / ABORTED" \
        "Timeout    : ${TIMEOUT_SEC} s/config"
    exit 0
fi

# ============================================================
#  構造的失敗ヘルパ: 即 ABORTED (exit 2)
# ============================================================
abort() { echo "ABORTED: $*" >&2; [ -n "${RUN_LOG:-}" ] && echo "ABORTED: $*" >> "${RUN_LOG}" 2>/dev/null; exit 2; }

# --- checkpoint 検証 (構造的失敗) ---
: "${EXPECTED_SHA:?EXPECTED_SHA must be set (checkpoint SHA)}"
ACTUAL_SHA="$(git rev-parse HEAD)"
test "${ACTUAL_SHA}" = "${EXPECTED_SHA}" || abort "checkpoint mismatch (HEAD=${ACTUAL_SHA} != EXPECTED_SHA=${EXPECTED_SHA})"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${PBS_JOBID:-not_pbs}"
RESULT_DIR="${BUILD_DIR}/result_memory_correctness_${TIMESTAMP}_${JOB_ID}"
MANIFEST="${RESULT_DIR}/MANIFEST.txt"
EXEC_SUMMARY="${RESULT_DIR}/execution_summary.tsv"
CMP_MATRIX="${RESULT_DIR}/comparison_matrix.tsv"
FINAL_STATUS_FILE="${RESULT_DIR}/FINAL_STATUS.txt"
RUN_LOG="${RESULT_DIR}/run.log"
GRAPH_PATH="${PROJECT_DIR}/${GRAPH_REL}"
mkdir -p "${RESULT_DIR}" || abort "mkdir RESULT_DIR failed: ${RESULT_DIR}"
: > "${RUN_LOG}" || abort "cannot write run.log"

log() { printf '%s\n' "$*" | tee -a "${RUN_LOG}"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

md_value() {
    local key="$1" file="$2"
    awk -F '|' -v wanted="${key}" '
        { k=$2; v=$3;
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", k);
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", v);
          if (k == wanted) { print v; exit } }' "${file}"
}

validate_dump() {
    local dump_file="$1" expected_n="$2"
    awk -F '\t' -v expected="${expected_n}" '
        BEGIN { numeric="^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$" }
        /^#/ { headers++; next }
        NF == 0 { next }
        {
            lower=tolower($2)
            special=(lower=="nan"||lower=="+nan"||lower=="-nan"||
                     lower=="inf"||lower=="+inf"||lower=="-inf"||
                     lower=="infinity"||lower=="+infinity"||lower=="-infinity")
            if (NF != 2 || $1 !~ /^[0-9]+$/ || $1 < 0 || $1 >= expected ||
                seen[$1]++ || ($2 !~ numeric && !special)) { bad=1 }
            count++
        }
        END {
            if (headers != 1 || count != expected) bad=1
            for (i=0; i<expected; i++) if (!(i in seen)) bad=1
            exit bad ? 1 : 0
        }' "${dump_file}"
}

dump_has_nonfinite() {
    awk -F '\t' '
        !/^#/ && NF == 2 { v=tolower($2); if (v ~ /nan|inf/) { found=1; exit } }
        END { exit found ? 0 : 1 }' "$1"
}

log_has_failure_marker() {
    grep -Ev 'exceeds safe limit|exceeds HBM3 budget|may cause cudaMalloc OOM|clamping to' "$1" \
        | grep -Eiq '(^|[^[:alnum:]_])(FAIL(ED|URE)?|OOM|TIMEOUT|NaN|[+-]?Inf(inity)?)([^[:alnum:]_]|$)|CUDA error at|out of memory'
}

# --- グラフ情報 + 構造的検査 --------------------------------------------
[ -r "${GRAPH_PATH}" ] || abort "graph not readable: ${GRAPH_PATH}"
read -r N M < "${GRAPH_PATH}"
GRAPH_SHA="$(sha256_file "${GRAPH_PATH}")"
GRAPH_NAME="$(basename "${GRAPH_REL}")"
[ -n "${EXPECTED_N}" ] && [ "${N}" != "${EXPECTED_N}" ] && abort "graph n mismatch (${N} != ${EXPECTED_N})"
[ -n "${EXPECTED_M}" ] && [ "${M}" != "${EXPECTED_M}" ] && abort "graph m mismatch (${M} != ${EXPECTED_M})"
[ -n "${EXPECTED_GRAPH_SHA}" ] && [ "${GRAPH_SHA}" != "${EXPECTED_GRAPH_SHA}" ] && abort "graph sha256 mismatch (${GRAPH_SHA} != ${EXPECTED_GRAPH_SHA})"

# --- MANIFEST / summary ヘッダ ------------------------------------------
{
  printf '%s\n' \
    "checkpoint_sha=${ACTUAL_SHA}" "pbs_job_id=${JOB_ID}" "project_dir=${PROJECT_DIR}" \
    "result_dir=${RESULT_DIR}" "runner=${RUNNER}" "graph=${GRAPH_REL}" \
    "graph_sha256=${GRAPH_SHA}" "n_nodes=${N}" "n_edges=${M}" \
    "runs_per_configuration=1" "warmup=none" "timing_usage=correctness_only_not_performance" \
    "design=comparison_matrix" "external_comparator=pathmerge_b4096" \
    "miyabi_host_mem_gib=${MIYABI_HOST_MEM_GIB}" "miyabi_host_mem_gb=${MIYABI_HOST_MEM_GB}" "um_ns_constant=${UM_NS}" \
    "um_mem_model=managed_alloc_estimate_gb = displayed_dynamic_um_gb * NS_eff / NS (log-derived estimate, NOT measured RSS/CPU residency)" \
    "um_route_evidence=oversubscription_route_evidence combines (managed_alloc_estimate_gb>free_before_gb, HBM3 streaming, NS_eff=1, num_subs=2, SUB_BATCH<batch, Prefetch cum>0); NOT a direct migration-byte measurement" \
    "abs_tol=${ABS_TOL}" "rel_tol=${REL_TOL}" \
    "criterion=abs_diff <= abs_tol + rel_tol * max(abs(a),abs(b))" ; } > "${MANIFEST}" || abort "cannot write MANIFEST"

printf '%s\n' \
    $'checkpoint_sha\tpbs_job_id\tconfig\timpl\tpath_type\tmode\trequested_batch\teffective_batch\trunner_exit\tvector_sha256\tvector_valid\tSUB_BATCH\tnum_subs\tNS_eff\toversubscribed\tuses_um\tNS\tdynamic_um_displayed_gb\tfree_before_gb\tmanaged_alloc_estimate_gb\tmanaged_alloc_headroom_estimate_gb\tmanaged_alloc_minus_free_before_estimate_gb\tprefetch_cum_s\toversubscription_route_evidence\tstatus\treason' \
    > "${EXEC_SUMMARY}"
printf '%s\n' \
    $'checkpoint_sha\tpbs_job_id\tComparisonClass\tRequiredForCoreMemoryPath\tcomparison_subclass\tlabel_a\tlabel_b\ta_valid\tb_valid\tcomparison_exit\tvector_length_a\tvector_length_b\tmissing_a\tmissing_b\tmismatched_elements\tmax_abs_error\tmax_abs_index\tmax_abs_a\tmax_abs_b\tmax_rel_error\tmax_rel_index\tmax_rel_a\tmax_rel_b\tmax_bc_a_index\tmax_bc_a_value\tmax_bc_b_index\tmax_bc_b_value\tsha256_a\tsha256_b\tabs_tol\trel_tol\tstatus' \
    > "${CMP_MATRIX}"

log "checkpoint=${ACTUAL_SHA} job=${JOB_ID} result=${RESULT_DIR}"
log "graph=${GRAPH_REL} n=${N} m=${M} sha256=${GRAPH_SHA}"

# --- ビルド (構造的失敗なら ABORTED) ------------------------------------
if [ "${SKIP_BUILD}" != "1" ]; then
    log "[BUILD] scripts/build_miyabi_interactive.sh (run_benchmark は cugraph_bc_mini に依存)"
    if ! JOBS="${JOBS}" BUILD_DIR="${BUILD_DIR}" bash "${PROJECT_DIR}/scripts/build_miyabi_interactive.sh" 2>&1 | tee -a "${RUN_LOG}"; then
        abort "build failed"
    fi
fi
[ -x "${RUNNER}" ]  || abort "runner not found/executable: ${RUNNER}"
[ -f "${COMPARE}" ] || abort "compare script not found: ${COMPARE}"

# --- 有効ベクトル追跡 (比較行列で参照) ----------------------------------
declare -A VALID   # name -> yes/no
N_CONFIG_OK=0; N_CONFIG_FAIL=0

# --- per-config 状態変数 (record_exec が参照) ---------------------------
reset_fields() {
    runner_exit=not_recorded vector_sha=not_recorded vector_valid=no
    effective_batch=not_recorded sub_batch=not_recorded num_subs=not_recorded ns_eff=not_recorded
    oversubscribed=not_applicable uses_um=not_recorded
    ns=not_applicable dyn_um_gb=not_applicable free_before_gb=not_applicable
    managed_alloc_estimate_gb=not_applicable managed_alloc_headroom_estimate_gb=not_applicable
    managed_alloc_minus_free_before_estimate_gb=not_applicable prefetch_cum_s=not_applicable
    oversubscription_route_evidence=not_applicable
    path_status=FAIL path_reason=unset
}

record_exec() {
    local status="$1" reason="$2"
    local row=(
        "${ACTUAL_SHA}" "${JOB_ID}" "${cfg_name}" "${cfg_impl}" "${cfg_path}" "${cfg_mode}"
        "${requested_batch}" "${effective_batch}" "${runner_exit}" "${vector_sha}" "${vector_valid}"
        "${sub_batch}" "${num_subs}" "${ns_eff}" "${oversubscribed}" "${uses_um}"
        "${ns}" "${dyn_um_gb}" "${free_before_gb}" "${managed_alloc_estimate_gb}"
        "${managed_alloc_headroom_estimate_gb}" "${managed_alloc_minus_free_before_estimate_gb}"
        "${prefetch_cum_s}" "${oversubscription_route_evidence}" "${status}" "${reason}" )
    (IFS=$'\t'; printf '%s\n' "${row[*]}") >> "${EXEC_SUMMARY}"
    {
        printf '\n[exec %s | impl=%s | path=%s | mode=%s]\n' "${cfg_name}" "${cfg_impl}" "${cfg_path}" "${cfg_mode}"
        printf 'requested_batch=%s effective_batch=%s runner_exit=%s vector_valid=%s\n' \
            "${requested_batch}" "${effective_batch}" "${runner_exit}" "${vector_valid}"
        printf 'SUB_BATCH=%s num_subs=%s NS_eff=%s oversubscribed=%s uses_um=%s\n' \
            "${sub_batch}" "${num_subs}" "${ns_eff}" "${oversubscribed}" "${uses_um}"
        printf 'um_mem_model: NS=%s NS_eff=%s dynamic_um_displayed_gb=%s managed_alloc_estimate_gb=%s (log-derived estimate, NOT measured RSS)\n' \
            "${ns}" "${ns_eff}" "${dyn_um_gb}" "${managed_alloc_estimate_gb}"
        printf 'um_mem: free_before_gb=%s managed_alloc_minus_free_before_estimate_gb=%s managed_alloc_headroom_estimate_gb=%s prefetch_cum_s=%s\n' \
            "${free_before_gb}" "${managed_alloc_minus_free_before_estimate_gb}" "${managed_alloc_headroom_estimate_gb}" "${prefetch_cum_s}"
        printf 'oversubscription_route_evidence=%s\nvector_sha256=%s\nstatus=%s\nreason=%s\n' \
            "${oversubscription_route_evidence}" "${vector_sha}" "${status}" "${reason}"
    } >> "${MANIFEST}"
}

extract_path_evidence() {
    local stderr_file="$1" path_type="$2" line
    case "${path_type}" in
        um)
            line="$(grep 'dynamic(UM).*BATCH=.*SUB_BATCH=.*num_subs=.*NS_eff=' "${stderr_file}" | tail -n1 || true)"
            if [ -n "${line}" ]; then
                effective_batch="$(printf '%s\n' "${line}" | sed -n 's/.*BATCH=\([0-9]*\), SUB_BATCH=.*/\1/p')"
                sub_batch="$(printf '%s\n' "${line}" | sed -n 's/.*SUB_BATCH=\([0-9]*\),.*/\1/p')"
                num_subs="$(printf '%s\n' "${line}" | sed -n 's/.*num_subs=\([0-9]*\),.*/\1/p')"
                ns_eff="$(printf '%s\n' "${line}" | sed -n 's/.*NS_eff=\([0-9]*\).*/\1/p')"
            fi
            uses_um=true
            if grep -q '\[Mode\] HBM3 streaming' "${stderr_file}"; then oversubscribed=true; else oversubscribed=false; fi
            ns="${UM_NS}"
            dyn_um_gb="$(printf '%s\n' "${line}" | sed -n 's/.*dynamic(UM)=\([0-9.]*\) GB.*/\1/p')"
            free_before_gb="$(grep 'GPU HBM3: total=.*free_before=' "${stderr_file}" | tail -n1 | sed -n 's/.*free_before=\([0-9.]*\) GB.*/\1/p')"
            prefetch_cum_s="$(grep 'Prefetch cum=' "${stderr_file}" | tail -n1 | sed -n 's/.*Prefetch cum=\([0-9.]*\) s.*/\1/p')"
            if [ -n "${dyn_um_gb}" ] && [ -n "${ns_eff}" ] && [ "${ns_eff}" != "not_recorded" ]; then
                managed_alloc_estimate_gb="$(awk -v d="${dyn_um_gb}" -v ne="${ns_eff}" -v ns="${UM_NS}" 'BEGIN{ if (ns+0>0) printf "%.2f", d*ne/ns }')"
                managed_alloc_headroom_estimate_gb="$(awk -v h="${MIYABI_HOST_MEM_GB}" -v a="${managed_alloc_estimate_gb}" 'BEGIN{ if (a!="") printf "%.2f", h-a }')"
            fi
            if [ -n "${managed_alloc_estimate_gb}" ] && [ "${managed_alloc_estimate_gb}" != "not_applicable" ] && [ -n "${free_before_gb}" ]; then
                managed_alloc_minus_free_before_estimate_gb="$(awk -v a="${managed_alloc_estimate_gb}" -v f="${free_before_gb}" 'BEGIN{ printf "%.2f", a-f }')"
            fi
            oversubscription_route_evidence=NOT_PROVEN
            dyn_um_gb="${dyn_um_gb:-not_recorded}"; free_before_gb="${free_before_gb:-not_recorded}"
            prefetch_cum_s="${prefetch_cum_s:-not_recorded}"
            managed_alloc_estimate_gb="${managed_alloc_estimate_gb:-not_recorded}"
            managed_alloc_headroom_estimate_gb="${managed_alloc_headroom_estimate_gb:-not_recorded}"
            managed_alloc_minus_free_before_estimate_gb="${managed_alloc_minus_free_before_estimate_gb:-not_recorded}"
            ;;
        chunked)
            line="$(grep 'dynamic(SUB_BATCH alloc).*BATCH=.*SUB_BATCH=.*num_subs=.*NS_eff=' "${stderr_file}" | tail -n1 || true)"
            if [ -n "${line}" ]; then
                effective_batch="$(printf '%s\n' "${line}" | sed -n 's/.*BATCH=\([0-9]*\), SUB_BATCH=.*/\1/p')"
                sub_batch="$(printf '%s\n' "${line}" | sed -n 's/.*SUB_BATCH=\([0-9]*\),.*/\1/p')"
                num_subs="$(printf '%s\n' "${line}" | sed -n 's/.*num_subs=\([0-9]*\),.*/\1/p')"
                ns_eff="$(printf '%s\n' "${line}" | sed -n 's/.*NS_eff=\([0-9]*\).*/\1/p')"
            fi
            uses_um=false
            if grep -q '\[Mode\] Manual chunking' "${stderr_file}"; then oversubscribed=true; else oversubscribed=false; fi
            ;;
        pure)
            line="$(grep 'dynamic(GPU)=.*batch_per_stream=' "${stderr_file}" | tail -n1 || true)"
            [ -n "${line}" ] && effective_batch="$(printf '%s\n' "${line}" | sed -n 's/.*batch_per_stream=\([0-9]*\).*/\1/p')"
            sub_batch=not_applicable num_subs=not_applicable ns_eff=not_applicable oversubscribed=not_applicable
            if grep -q 'dynamic(UM)' "${stderr_file}"; then uses_um=true; else uses_um=false; fi
            ;;
        pathmerge)
            line="$(grep '\[PathMerge\].*batch_size=.*num_sources=.*num_batches=' "${stderr_file}" | tail -n1 || true)"
            if [ -n "${line}" ]; then
                effective_batch="$(printf '%s\n' "${line}" | sed -n 's/.*batch_size=\([0-9]*\),.*/\1/p')"
                num_subs="$(printf '%s\n' "${line}" | sed -n 's/.*num_batches=\([0-9]*\).*/\1/p')"
            fi
            sub_batch=not_applicable ns_eff=not_applicable uses_um=managed_csr oversubscribed=not_applicable
            ;;
    esac
    effective_batch="${effective_batch:-not_recorded}"; sub_batch="${sub_batch:-not_recorded}"
    num_subs="${num_subs:-not_recorded}"; ns_eff="${ns_eff:-not_recorded}"
}

# --- 経路証拠判定 (path_type, mode 依存; 単に成功しただけでは合格にしない) ---
path_evidence_ok() {
    local path_type="$1" mode="$2"
    path_status=FAIL
    [ "${effective_batch}" = "${requested_batch}" ] || { path_reason="effective_batch_${effective_batch}_ne_requested_${requested_batch}"; return 1; }
    case "${path_type}" in
        um)
            [ "${uses_um}" = "true" ] || { path_reason="um_path_not_detected"; return 1; }
            case "${mode}" in
                oversubscribed)
                    [ "${oversubscribed}" = "true" ] || { path_reason="hbm3_streaming_mode_absent"; return 1; }
                    [ "${ns_eff}" = "1" ] || { path_reason="ns_eff_${ns_eff}_ne_1"; return 1; }
                    case "${num_subs}" in ''|*[!0-9]*) path_reason="num_subs_not_recorded"; return 1;; esac
                    [ "${num_subs}" = "2" ] || { path_reason="num_subs_${num_subs}_ne_2"; return 1; }
                    case "${sub_batch}" in ''|*[!0-9]*) path_reason="sub_batch_not_recorded"; return 1;; esac
                    [ "${sub_batch}" -lt "${requested_batch}" ] || { path_reason="sub_batch_${sub_batch}_not_lt_${requested_batch}"; return 1; }
                    awk -v p="${prefetch_cum_s}" 'BEGIN{ exit (p+0 > 0) ? 0 : 1 }' || { path_reason="prefetch_cum_not_gt_0_${prefetch_cum_s}"; return 1; }
                    awk -v a="${managed_alloc_estimate_gb}" -v f="${free_before_gb}" 'BEGIN{ exit (a+0 > f+0) ? 0 : 1 }' || {
                        path_status=ROUTE_NOT_PROVEN; oversubscription_route_evidence=NOT_PROVEN
                        path_reason="managed_alloc_estimate_${managed_alloc_estimate_gb}GB_le_free_before_${free_before_gb}GB_no_real_oversubscription"; return 1; }
                    oversubscription_route_evidence=PASS ;;
                in_capacity)
                    [ "${oversubscribed}" = "false" ] || { path_reason="in_capacity_unexpectedly_oversubscribed"; return 1; }
                    [ "${num_subs}" = "1" ] || { path_reason="in_capacity_num_subs_${num_subs}_ne_1"; return 1; }
                    oversubscription_route_evidence=in_capacity_control ;;
            esac ;;
        chunked)
            [ "${uses_um}" = "false" ] || { path_reason="chunked_unexpectedly_uses_um"; return 1; }
            case "${num_subs}" in ''|*[!0-9]*) path_reason="num_subs_not_recorded"; return 1;; esac
            case "${mode}" in
                chunked)
                    [ "${oversubscribed}" = "true" ] || { path_reason="manual_chunking_mode_absent"; return 1; }
                    [ "${num_subs}" -gt 1 ] || { path_reason="num_subs_not_gt_1_chunked_path_not_exercised"; return 1; } ;;
                non_chunk)
                    [ "${num_subs}" = "1" ] || { path_reason="non_chunk_num_subs_${num_subs}_ne_1"; return 1; } ;;
            esac ;;
        pure)
            [ "${uses_um}" = "false" ] || { path_reason="pure_path_unexpectedly_uses_um"; return 1; } ;;
        pathmerge)
            : ;;  # effective==requested (clamp) は上で確認済
    esac
    path_status=PASS; path_reason="path_evidence_ok"; return 0
}

# ============================================================
#  1 構成を実行 (record して常に return 0; 失敗でも次構成へ)
# ============================================================
run_config() {
    local idx="$1"
    cfg_name="${CONFIG_NAMES[$idx]}"; cfg_impl="${CONFIG_IMPLS[$idx]}"
    local cfg_batchenv="${CONFIG_BATCHENV[$idx]}"
    cfg_path="${CONFIG_PATHTYPE[$idx]}"; cfg_mode="${CONFIG_MODE[$idx]}"
    reset_fields
    requested_batch="${CONFIG_BATCH[$idx]}"
    VALID["${cfg_name}"]=no
    cfg_stderr="${RESULT_DIR}/${cfg_name}.stderr.log"
    local cfg_dump="${RESULT_DIR}/${cfg_name}.bc.tsv"

    log "[RUN] ${cfg_name}: ${cfg_batchenv}=${requested_batch} ${cfg_impl} (path=${cfg_path} mode=${cfg_mode}) (n=1)"
    runner_exit=0
    env "${cfg_batchenv}=${requested_batch}" timeout "${TIMEOUT_SEC}" "${RUNNER}" "${cfg_impl}" "${GRAPH_PATH}" --dump-bc \
        > "${cfg_dump}" 2> "${cfg_stderr}" || runner_exit=$?
    [ -f "${cfg_dump}" ] && vector_sha="$(sha256_file "${cfg_dump}")"
    log "[EXIT] ${cfg_name}: ${runner_exit}"
    extract_path_evidence "${cfg_stderr}" "${cfg_path}"

    if [ "${runner_exit}" = "124" ]; then N_CONFIG_FAIL=$((N_CONFIG_FAIL+1)); record_exec FAIL "runner_timeout_${TIMEOUT_SEC}s"; return 0; fi
    if [ "${runner_exit}" -ne 0 ]; then N_CONFIG_FAIL=$((N_CONFIG_FAIL+1)); record_exec FAIL "runner_exit_${runner_exit}"; return 0; fi
    if log_has_failure_marker "${cfg_stderr}"; then N_CONFIG_FAIL=$((N_CONFIG_FAIL+1)); record_exec FAIL "failure_marker_in_stderr"; return 0; fi
    if ! validate_dump "${cfg_dump}" "${N}"; then N_CONFIG_FAIL=$((N_CONFIG_FAIL+1)); record_exec FAIL "invalid_or_incomplete_vector"; return 0; fi
    if dump_has_nonfinite "${cfg_dump}"; then N_CONFIG_FAIL=$((N_CONFIG_FAIL+1)); record_exec FAIL "nonfinite_value_in_vector"; return 0; fi
    if ! path_evidence_ok "${cfg_path}" "${cfg_mode}"; then N_CONFIG_FAIL=$((N_CONFIG_FAIL+1)); record_exec "${path_status}" "path_${path_reason}"; return 0; fi

    # ここまで来たら vector は構造的に有効 (比較行列で使用可能)
    vector_valid=yes; VALID["${cfg_name}"]=yes; N_CONFIG_OK=$((N_CONFIG_OK+1))
    record_exec PASS "runner_ok_and_path_evidence_ok"
    log "[OK  ] ${cfg_name}: valid vector, ${path_reason}"
    return 0
}

# ============================================================
#  比較行列: 全構成実行後に、有効ベクトル同士を比較 (常に return 0)
# ============================================================
N_CMP_PASS=0; N_CMP_FAIL=0; N_CMP_SKIP=0
# メモリ経路必須 (CORE) と PathMerge 診断 (CROSS) を分離集計
N_CORE_PASS=0; N_CORE_FAIL=0; N_CORE_SKIP=0
N_CROSS_PASS=0; N_CROSS_FAIL=0; N_CROSS_SKIP=0
run_comparison() {
    local cls="$1" a="$2" b="$3"
    # 比較分類: pathmerge_cross のみ診断 (CORE 非必須)、他は CORE メモリ経路必須
    local grp reqcore
    if [ "${cls}" = "pathmerge_cross" ]; then grp=PATHMERGE_CROSS_IMPL_DIAGNOSTIC; reqcore=no
    else grp=CORE_MEMORY_PATH; reqcore=yes; fi
    local a_dump="${RESULT_DIR}/${a}.bc.tsv" b_dump="${RESULT_DIR}/${b}.bc.tsv"
    local a_valid="${VALID[$a]:-no}" b_valid="${VALID[$b]:-no}"
    local cmp_exit=not_applicable len_a=not_applicable len_b=not_applicable
    local miss_a=not_applicable miss_b=not_applicable mism=not_applicable
    local mabs=not_applicable mabs_i=not_applicable mabs_a=not_applicable mabs_b=not_applicable
    local mrel=not_applicable mrel_i=not_applicable mrel_a=not_applicable mrel_b=not_applicable
    local mbca_i=not_applicable mbca_v=not_applicable mbcb_i=not_applicable mbcb_v=not_applicable
    local sha_a=not_applicable sha_b=not_applicable status

    if [ "${a_valid}" != "yes" ] || [ "${b_valid}" != "yes" ]; then
        status=SKIPPED; N_CMP_SKIP=$((N_CMP_SKIP+1))
        log "[CMP ] ${cls}: ${a} vs ${b} -> SKIPPED (a_valid=${a_valid} b_valid=${b_valid})"
    else
        sha_a="$(sha256_file "${a_dump}")"; sha_b="$(sha256_file "${b_dump}")"
        local md="${RESULT_DIR}/${a}__vs__${b}.md"
        cmp_exit=0
        python3 "${COMPARE}" "${a_dump}" "${b_dump}" --label-a "${a}" --label-b "${b}" \
            --abs-tol "${ABS_TOL}" --rel-tol "${REL_TOL}" --out "${md}" \
            --extra "checkpoint_sha=${ACTUAL_SHA}" "pbs_job_id=${JOB_ID}" "comparison_class=${cls}" \
            "graph_sha256=${GRAPH_SHA}" "n=${N}" "m=${M}" >> "${RUN_LOG}" 2>&1 || cmp_exit=$?
        len_a="$(md_value 'ベクトル長 A' "${md}")"; len_b="$(md_value 'ベクトル長 B' "${md}")"
        miss_a="$(md_value '欠損 index 数 (A のみ)' "${md}")"; miss_b="$(md_value '欠損 index 数 (B のみ)' "${md}")"
        mism="$(sed -n 's/.*不一致要素数 | \([0-9][0-9]*\) |$/\1/p' "${md}" | tail -n1)"
        local vp
        vp="$(md_value '最大絶対誤差' "${md}")"; mabs="${vp%% *}"; mabs_i="$(printf '%s\n' "${vp}" | sed -n 's/.*index \([^)]*\)).*/\1/p')"
        vp="$(md_value '最大絶対誤差 index の値' "${md}")"; mabs_a="$(printf '%s\n' "${vp}" | sed -n 's/^A=\([^,]*\), B=.*/\1/p')"; mabs_b="$(printf '%s\n' "${vp}" | sed -n 's/^A=[^,]*, B=\(.*\)$/\1/p')"
        vp="$(md_value '最大相対誤差' "${md}")"; mrel="${vp%% *}"; mrel_i="$(printf '%s\n' "${vp}" | sed -n 's/.*index \([^)]*\)).*/\1/p')"
        vp="$(md_value '最大相対誤差 index の値' "${md}")"; mrel_a="$(printf '%s\n' "${vp}" | sed -n 's/^A=\([^,]*\), B=.*/\1/p')"; mrel_b="$(printf '%s\n' "${vp}" | sed -n 's/^A=[^,]*, B=\(.*\)$/\1/p')"
        vp="$(md_value 'Max BC A' "${md}")"; mbca_i="$(printf '%s\n' "${vp}" | sed -n 's/^index \([^,]*\), value .*/\1/p')"; mbca_v="$(printf '%s\n' "${vp}" | sed -n 's/^index [^,]*, value \(.*\)$/\1/p')"
        vp="$(md_value 'Max BC B' "${md}")"; mbcb_i="$(printf '%s\n' "${vp}" | sed -n 's/^index \([^,]*\), value .*/\1/p')"; mbcb_v="$(printf '%s\n' "${vp}" | sed -n 's/^index [^,]*, value \(.*\)$/\1/p')"
        if [ "${cmp_exit}" = "0" ] && [ "${mism}" = "0" ] && [ "${miss_a}" = "0" ] && [ "${miss_b}" = "0" ] && [ "${len_a}" = "${N}" ] && [ "${len_b}" = "${N}" ]; then
            status=PASS; N_CMP_PASS=$((N_CMP_PASS+1))
        else
            status=FAIL; N_CMP_FAIL=$((N_CMP_FAIL+1))
        fi
        log "[CMP ] ${cls}: ${a} vs ${b} -> ${status} (mismatch=${mism} max_abs=${mabs} max_rel=${mrel})"
    fi
    # CORE / CROSS 分離集計
    if [ "${reqcore}" = "yes" ]; then
        case "${status}" in PASS) N_CORE_PASS=$((N_CORE_PASS+1));; FAIL) N_CORE_FAIL=$((N_CORE_FAIL+1));; SKIPPED) N_CORE_SKIP=$((N_CORE_SKIP+1));; esac
    else
        case "${status}" in PASS) N_CROSS_PASS=$((N_CROSS_PASS+1));; FAIL) N_CROSS_FAIL=$((N_CROSS_FAIL+1));; SKIPPED) N_CROSS_SKIP=$((N_CROSS_SKIP+1));; esac
    fi
    local row=(
        "${ACTUAL_SHA}" "${JOB_ID}" "${grp}" "${reqcore}" "${cls}" "${a}" "${b}" "${a_valid}" "${b_valid}" "${cmp_exit}"
        "${len_a}" "${len_b}" "${miss_a}" "${miss_b}" "${mism}"
        "${mabs}" "${mabs_i}" "${mabs_a}" "${mabs_b}" "${mrel}" "${mrel_i}" "${mrel_a}" "${mrel_b}"
        "${mbca_i}" "${mbca_v}" "${mbcb_i}" "${mbcb_v}" "${sha_a}" "${sha_b}" "${ABS_TOL}" "${REL_TOL}" "${status}" )
    (IFS=$'\t'; printf '%s\n' "${row[*]}") >> "${CMP_MATRIX}"
    return 0
}

# ============================================================
#  実行: 全構成を1回ずつ → 比較行列 → 最終 status
# ============================================================
for idx in "${!CONFIG_NAMES[@]}"; do run_config "${idx}"; done
log "[RUN DONE] configs ok=${N_CONFIG_OK} fail=${N_CONFIG_FAIL}"

log "[MATRIX] comparing valid vectors..."
for i in "${!CMP_A[@]}"; do run_comparison "${CMP_CLASS[$i]}" "${CMP_A[$i]}" "${CMP_B[$i]}"; done

# --- 判定を 2 系統へ分離 -----------------------------------------------
# core_memory_path_status: CORE 5件 (メモリ経路必須) の集計
if   [ "${N_CORE_FAIL}" -gt 0 ]; then core_memory_path_status=FAIL
elif [ "${N_CORE_SKIP}" -gt 0 ]; then core_memory_path_status=INCOMPLETE
elif [ "${N_CORE_PASS}" -eq 5 ]; then core_memory_path_status=PASS
else core_memory_path_status=INCOMPLETE; fi
# pathmerge_cross_impl_status: PathMerge 診断 5件 (external comparator)
if   [ "${N_CROSS_FAIL}" -gt 0 ]; then pathmerge_cross_impl_status=DIFFERENCE_OBSERVED
elif [ "${N_CROSS_SKIP}" -gt 0 ]; then pathmerge_cross_impl_status=INCOMPLETE
elif [ "${N_CROSS_PASS}" -eq 5 ]; then pathmerge_cross_impl_status=PASS
else pathmerge_cross_impl_status=INCOMPLETE; fi
# overall_status
if   [ "${core_memory_path_status}" = "FAIL" ];       then overall_status=CORE_FAIL; FINAL_CODE=1
elif [ "${core_memory_path_status}" = "INCOMPLETE" ]; then overall_status=INCOMPLETE; FINAL_CODE=1
else # core=PASS
    case "${pathmerge_cross_impl_status}" in
        PASS)                overall_status=PASS; FINAL_CODE=0 ;;
        DIFFERENCE_OBSERVED) overall_status=CORE_PASS_CROSS_IMPL_DIFFERENCE; FINAL_CODE=1 ;;
        *)                   overall_status=INCOMPLETE; FINAL_CODE=1 ;;  # cross INCOMPLETE
    esac
fi

{
    printf 'core_memory_path_status=%s\n' "${core_memory_path_status}"
    printf 'pathmerge_cross_impl_status=%s\n' "${pathmerge_cross_impl_status}"
    printf 'overall_status=%s\n' "${overall_status}"
    printf 'exit_code=%s\n' "${FINAL_CODE}"
    printf 'configs_ok=%s configs_fail=%s\n' "${N_CONFIG_OK}" "${N_CONFIG_FAIL}"
    printf 'core_pass=%s core_fail=%s core_skip=%s (of 5 CORE_MEMORY_PATH)\n' "${N_CORE_PASS}" "${N_CORE_FAIL}" "${N_CORE_SKIP}"
    printf 'cross_pass=%s cross_fail=%s cross_skip=%s (of 5 PATHMERGE_CROSS_IMPL_DIAGNOSTIC)\n' "${N_CROSS_PASS}" "${N_CROSS_FAIL}" "${N_CROSS_SKIP}"
    printf 'note=PathMerge is external comparator (NOT ground truth). CORE_PASS_CROSS_IMPL_DIFFERENCE is NOT a complete correctness proof; it separates memory-path-specific results from the unresolved PathMerge difference.\n'
} > "${FINAL_STATUS_FILE}"

{
    printf '\n[final]\nconfigs_ok=%s configs_fail=%s\n' "${N_CONFIG_OK}" "${N_CONFIG_FAIL}"
    printf 'CORE_MEMORY_PATH: pass=%s fail=%s skip=%s -> core_memory_path_status=%s\n' \
        "${N_CORE_PASS}" "${N_CORE_FAIL}" "${N_CORE_SKIP}" "${core_memory_path_status}"
    printf 'PATHMERGE_CROSS_IMPL_DIAGNOSTIC: pass=%s fail=%s skip=%s -> pathmerge_cross_impl_status=%s\n' \
        "${N_CROSS_PASS}" "${N_CROSS_FAIL}" "${N_CROSS_SKIP}" "${pathmerge_cross_impl_status}"
    printf 'overall_status=%s (exit=%s)\n' "${overall_status}" "${FINAL_CODE}"
} >> "${MANIFEST}"
log "core_memory_path_status=${core_memory_path_status} | pathmerge_cross_impl_status=${pathmerge_cross_impl_status} | overall_status=${overall_status}"
log "NOTE: PathMerge=external comparator; acquired results retained; failures recorded, not deleted."
exit "${FINAL_CODE}"
