#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=2:00:00
#PBS -N bc_mem_diag
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
#  run_memory_correctness_diagnostic.sh — T-RESET / T-NSEFF 最小診断 (Gate G2.4)
#
#  325557 で観測した stress 条件 8 頂点差の原因を、GPU_Opt (host_um) だけを用いて
#  1 因子ずつ切り分ける。性能測定ではなく正確性原因の診断 (時間値は性能表に使わない)。
#
#  診断スイッチ (host_um.cu, 既定 OFF; 未指定で checkpoint と同一動作):
#    BC_DIAG_FORCE_FULL_RESET=1 : visited-only reset を使わず常に full memset
#    BC_DIAG_FORCE_NS_EFF_ONE=1 : in-capacity でも NS_eff=1 (stream/occupancy 診断)
#
#  構成 (各 1 回, batch=1024, GPU_Opt のみ):
#    CONTROL : diag なし          (expected NS_eff=2, SUB_BATCH=1024, num_subs=1)
#    T-RESET : force_full_reset=1 (expected NS_eff=2, SUB_BATCH=1024, num_subs=1)
#    T-NSEFF : force_ns_eff_one=1 (expected NS_eff=1, SUB_BATCH=1024, num_subs=1)
#  両診断変数を同時に有効化しない。各構成 1 回のみ・自動再試行しない。
#
#  比較 (正式許容 abs_tol=1e-3, rel_tol=1e-6; 緩めて PASS にしない):
#    内部: CONTROL vs T-RESET / CONTROL vs T-NSEFF / T-RESET vs T-NSEFF
#    外部(job 2368587, SHA256 検証後 read-only): CONTROL/T-RESET/T-NSEFF vs 旧b1024,
#          CONTROL/T-RESET/T-NSEFF vs 旧b9792, T-NSEFF vs 旧Chunked b16384
#    外部 vector 欠損/SHA不一致は当該比較を SKIPPED (値を推測しない)。
#
#  非干渉確認: CONTROL vs 旧b1024 が混合許容不一致 0 でなければ
#    DIAGNOSTIC_INSTRUMENTATION_CHANGED_BASELINE として停止し reset/nseff 判定を行わない。
#
#  出力 (RESULT_DIR, build_miyabi 配下 = gitignored):
#    MANIFEST.txt execution_summary.tsv comparison_matrix.tsv affected_vertices.tsv
#    DIAGNOSIS.md run.log FINAL_STATUS.txt  <config>/vector.bc.tsv <config>/stderr.log
#
#  環境変数: EXPECTED_SHA(必須) / DRY_RUN / SKIP_BUILD / JOBS(8) / TIMEOUT_SEC(5400)
#            ABS_TOL(1e-3) / REL_TOL(1e-6) / GRAPH(data/325557_3216152) / OLD_JOB_DIR
# ============================================================

set -uo pipefail

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

# --- 外部 vector (job 2368587, read-only, SHA256 検証) -------------------
OLD_JOB_DIR="${OLD_JOB_DIR:-}"
if [ -z "${OLD_JOB_DIR}" ]; then
    OLD_JOB_DIR="$(ls -d "${BUILD_DIR}"/result_memory_correctness_*_2368587.opbs 2>/dev/null | head -1 || true)"
fi
declare -A EXT_FILE EXT_SHA
EXT_FILE[old_b1024]="${OLD_JOB_DIR}/gpu_opt_b1024.bc.tsv"
EXT_SHA[old_b1024]="4a40a553a388ba2cb29d4ea366db979983fa398c55bb8a694882f260efd431cb"
EXT_FILE[old_b9792]="${OLD_JOB_DIR}/gpu_opt_b9792.bc.tsv"
EXT_SHA[old_b9792]="be8f52d32ac03cd495a08c5c6cd138fcdcec916a16830e62e7bc8d3c968d25c5"
EXT_FILE[old_chunk_b16384]="${OLD_JOB_DIR}/gpu_opt_pure_chunked_b16384.bc.tsv"
EXT_SHA[old_chunk_b16384]="618ffdc4108f0c24a148bc1aa2a18b83d14b48319cd18438f89196b40930a86d"

AFFECTED_IDX="7954 95156 143358 165886 226184 228350 289284 325556"

# --- 構成 (各 1 回) -----------------------------------------------------
CONF_NAMES=(CONTROL T-RESET T-NSEFF)
CONF_ENV=(""  "BC_DIAG_FORCE_FULL_RESET=1"  "BC_DIAG_FORCE_NS_EFF_ONE=1")
CONF_EXP_FULLRESET=(false true false)
CONF_EXP_NSEFFONE=(false false true)
CONF_EXP_NSEFF=(2 2 1)
# 期待実行経路カウンタ (n=325557, batch=1024, num_subs=1, outer_batches=318; 静的計算値)。
CONF_EXP_FULLMEMSET=(3 318 2)
CONF_EXP_VISITED=(315 0 316)

# --- 比較行列 (A B) : 内部3 + 外部7 = 10 -------------------------------
CMP_A=(CONTROL CONTROL T-RESET   CONTROL T-RESET T-NSEFF CONTROL T-RESET T-NSEFF T-NSEFF)
CMP_B=(T-RESET T-NSEFF T-NSEFF   old_b1024 old_b1024 old_b1024 old_b9792 old_b9792 old_b9792 old_chunk_b16384)

# ============================================================
#  DRY_RUN
# ============================================================
if [ "${DRY_RUN}" = "1" ]; then
    printf '%s\n' \
        "DRY RUN: no build, runner, GPU, qsub, result update, or BC dump" \
        "Project : ${PROJECT_DIR}" \
        "Runner  : ${RUNNER}" \
        "Graph   : ${GRAPH_REL} (expect n=${EXPECTED_N} m=${EXPECTED_M} sha=${EXPECTED_GRAPH_SHA:0:12}...)" \
        "OldJob  : ${OLD_JOB_DIR:-<not found>}" \
        "Planned : ${BUILD_DIR}/result_memory_diagnostic_<timestamp>_<PBS_JOBID>/" \
        "Purpose : correctness diagnosis (T-RESET / T-NSEFF), 1 factor each; timings NOT performance" \
        "Configs (GPU_Opt only, batch=1024, n=1 each):"
    for i in "${!CONF_NAMES[@]}"; do
        printf '  %s: env[%s] expect force_full_reset=%s force_ns_eff_one=%s NS_eff=%s SUB_BATCH=1024 num_subs=1\n' \
            "${CONF_NAMES[$i]}" "${CONF_ENV[$i]:-<none>}" \
            "${CONF_EXP_FULLRESET[$i]}" "${CONF_EXP_NSEFFONE[$i]}" "${CONF_EXP_NSEFF[$i]}"
    done
    printf 'Comparisons (%d):\n' "${#CMP_A[@]}"
    for i in "${!CMP_A[@]}"; do printf '  %s vs %s\n' "${CMP_A[$i]}" "${CMP_B[$i]}"; done
    printf '%s\n' \
        "Affected indices: ${AFFECTED_IDX}" \
        "Non-interference: CONTROL vs old_b1024 must be mixed-tol mismatch=0 (else DIAGNOSTIC_INSTRUMENTATION_CHANGED_BASELINE)" \
        "Tolerance: abs_diff <= ${ABS_TOL} + ${REL_TOL}*max(|a|,|b|) (unchanged; not relaxed for PASS)" \
        "Judgments: reset {RESET_NOT_DISTINGUISHED|RESET_PATH_OR_SCHEDULING_ASSOCIATED|RESET_INCONCLUSIVE};" \
        "           nseff {NS_EFF_NOT_DISTINGUISHED|NS_EFF_OR_OCCUPANCY_ASSOCIATED|NS_EFF_INCONCLUSIVE}" \
        "Structural ABORT(exit2): checkpoint/build/graph/CONTROL fail/diag-not-reflected/runner|compare missing/CONTROL-vs-oldb1024 structural" \
        "Per-config record+continue: T-RESET / T-NSEFF runner!=0, missing vector, NaN/Inf, comparison mismatch" \
        "Timeout : ${TIMEOUT_SEC} s/config"
    exit 0
fi

abort() { echo "ABORTED: $*" >&2; [ -n "${RUN_LOG:-}" ] && echo "ABORTED: $*" >> "${RUN_LOG}" 2>/dev/null; exit 2; }

# --- checkpoint 検証 ---
: "${EXPECTED_SHA:?EXPECTED_SHA must be set (checkpoint SHA)}"
ACTUAL_SHA="$(git rev-parse HEAD)"
test "${ACTUAL_SHA}" = "${EXPECTED_SHA}" || abort "checkpoint mismatch (HEAD=${ACTUAL_SHA} != EXPECTED_SHA=${EXPECTED_SHA})"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${PBS_JOBID:-not_pbs}"
RESULT_DIR="${BUILD_DIR}/result_memory_diagnostic_${TIMESTAMP}_${JOB_ID}"
MANIFEST="${RESULT_DIR}/MANIFEST.txt"
EXEC_SUMMARY="${RESULT_DIR}/execution_summary.tsv"
CMP_MATRIX="${RESULT_DIR}/comparison_matrix.tsv"
AFFECTED_TSV="${RESULT_DIR}/affected_vertices.tsv"
DIAGNOSIS="${RESULT_DIR}/DIAGNOSIS.md"
FINAL_STATUS_FILE="${RESULT_DIR}/FINAL_STATUS.txt"
RUN_LOG="${RESULT_DIR}/run.log"
GRAPH_PATH="${PROJECT_DIR}/${GRAPH_REL}"
mkdir -p "${RESULT_DIR}" || abort "mkdir RESULT_DIR failed"
: > "${RUN_LOG}" || abort "cannot write run.log"

log() { printf '%s\n' "$*" | tee -a "${RUN_LOG}"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
md_value() {
    local key="$1" file="$2"
    awk -F '|' -v w="${key}" '{k=$2;v=$3;gsub(/^[[:space:]]+|[[:space:]]+$/,"",k);gsub(/^[[:space:]]+|[[:space:]]+$/,"",v);if(k==w){print v;exit}}' "${file}"
}
validate_dump() {
    local f="$1" e="$2"
    awk -F '\t' -v expected="${e}" '
        BEGIN{numeric="^[+-]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][+-]?[0-9]+)?$"}
        /^#/{h++;next} NF==0{next}
        {lo=tolower($2);sp=(lo=="nan"||lo=="+nan"||lo=="-nan"||lo=="inf"||lo=="+inf"||lo=="-inf"||lo=="infinity"||lo=="+infinity"||lo=="-infinity");
         if(NF!=2||$1!~/^[0-9]+$/||$1<0||$1>=expected||seen[$1]++||($2!~numeric&&!sp))bad=1;count++}
        END{if(h!=1||count!=expected)bad=1;for(i=0;i<expected;i++)if(!(i in seen))bad=1;exit bad?1:0}' "${f}"
}
dump_has_nonfinite() { awk -F '\t' '!/^#/&&NF==2{v=tolower($2);if(v~/nan|inf/){f=1;exit}}END{exit f?0:1}' "$1"; }
log_has_failure_marker() {
    grep -Ev 'exceeds safe limit|exceeds HBM3 budget|may cause cudaMalloc OOM|clamping to' "$1" \
      | grep -Eiq '(^|[^[:alnum:]_])(FAIL(ED|URE)?|OOM|TIMEOUT|NaN|[+-]?Inf(inity)?)([^[:alnum:]_]|$)|CUDA error at|out of memory'
}

# --- graph 構造的検査 ---
[ -r "${GRAPH_PATH}" ] || abort "graph not readable: ${GRAPH_PATH}"
read -r N M < "${GRAPH_PATH}"
GRAPH_SHA="$(sha256_file "${GRAPH_PATH}")"
[ "${N}" = "${EXPECTED_N}" ] || abort "graph n mismatch (${N} != ${EXPECTED_N})"
[ "${M}" = "${EXPECTED_M}" ] || abort "graph m mismatch (${M} != ${EXPECTED_M})"
[ "${GRAPH_SHA}" = "${EXPECTED_GRAPH_SHA}" ] || abort "graph sha256 mismatch (${GRAPH_SHA} != ${EXPECTED_GRAPH_SHA})"

{
  printf '%s\n' "checkpoint_sha=${ACTUAL_SHA}" "pbs_job_id=${JOB_ID}" "project_dir=${PROJECT_DIR}" \
    "result_dir=${RESULT_DIR}" "runner=${RUNNER}" "graph=${GRAPH_REL}" "graph_sha256=${GRAPH_SHA}" \
    "n_nodes=${N}" "n_edges=${M}" "old_job_dir=${OLD_JOB_DIR:-not_found}" \
    "purpose=diagnosis_T-RESET_T-NSEFF_one_factor_each_not_performance" \
    "diag_env=BC_DIAG_FORCE_FULL_RESET,BC_DIAG_FORCE_NS_EFF_ONE (default OFF = checkpoint behavior)" \
    "route_counters=[Diag] full_memset_calls / visited_reset_calls (host-side branch counts; kernels/memset unchanged)" \
    "expected_counters=CONTROL 3/315, T-RESET 318/0, T-NSEFF 2/316 (full_memset/visited; static for n=325557 b1024)" \
    "route_not_confirmed=config excluded from causal judgment (DIAGNOSTIC_ROUTE_NOT_CONFIRMED)" \
    "abs_tol=${ABS_TOL}" "rel_tol=${REL_TOL}" \
    "criterion=abs_diff <= abs_tol + rel_tol*max(abs(a),abs(b))" \
    "affected_indices=${AFFECTED_IDX}"; } > "${MANIFEST}" || abort "cannot write MANIFEST"

printf '%s\n' \
  $'config\tenv\texp_force_full_reset\texp_force_ns_eff_one\texp_NS_eff\trunner_exit\tvector_valid\teffective_batch\tNS_eff\tSUB_BATCH\tnum_subs\tdiag_force_full_reset\tdiag_force_ns_eff_one\tdiag_mode\tfull_memset_calls\tvisited_reset_calls\texp_full_memset\texp_visited\troute_confirmed\tvector_sha256\tstatus\treason' \
  > "${EXEC_SUMMARY}"
printf '%s\n' \
  $'label_a\tlabel_b\ta_valid\tb_valid\tcomparison_exit\tvector_length_a\tvector_length_b\tmissing_a\tmissing_b\tmismatched_elements\tmax_abs_error\tmax_abs_index\tmax_abs_a\tmax_abs_b\tmax_rel_error\tmax_rel_index\tsha256_a\tsha256_b\tabs_tol\trel_tol\tstatus' \
  > "${CMP_MATRIX}"

log "checkpoint=${ACTUAL_SHA} job=${JOB_ID} result=${RESULT_DIR}"
log "graph=${GRAPH_REL} n=${N} m=${M} sha256=${GRAPH_SHA}"
log "old_job_dir=${OLD_JOB_DIR:-not_found}"

# --- build (構造的失敗なら ABORT) ---
if [ "${SKIP_BUILD}" != "1" ]; then
    log "[BUILD] scripts/build_miyabi_interactive.sh"
    if ! JOBS="${JOBS}" BUILD_DIR="${BUILD_DIR}" bash "${PROJECT_DIR}/scripts/build_miyabi_interactive.sh" 2>&1 | tee -a "${RUN_LOG}"; then
        abort "build failed"
    fi
fi
[ -x "${RUNNER}" ]  || abort "runner not found/executable: ${RUNNER}"
[ -f "${COMPARE}" ] || abort "compare script not found: ${COMPARE}"

# --- vector 追跡 ---
declare -A VEC_FILE VEC_OK ROUTE_CONF
resolve_external() {
    local name f want got
    for name in old_b1024 old_b9792 old_chunk_b16384; do
        f="${EXT_FILE[$name]}"; want="${EXT_SHA[$name]}"
        VEC_FILE[$name]="${f}"; VEC_OK[$name]=no
        if [ -n "${OLD_JOB_DIR}" ] && [ -f "${f}" ]; then
            got="$(sha256_file "${f}")"
            if [ "${got}" = "${want}" ]; then VEC_OK[$name]=yes; log "[EXT ] ${name}: sha256 OK (${got:0:12}...)"
            else log "[EXT ] ${name}: sha256 MISMATCH (got ${got:0:12}..., want ${want:0:12}...) -> SKIPPED"; fi
        else
            log "[EXT ] ${name}: file not found (${f}) -> SKIPPED"
        fi
    done
}
resolve_external

# --- 1 構成実行 (CONTROL は critical, T-* は per-config) -----------------
run_config() {
    local idx="$1"
    local name="${CONF_NAMES[$idx]}" cenv="${CONF_ENV[$idx]}"
    local exp_fr="${CONF_EXP_FULLRESET[$idx]}" exp_no="${CONF_EXP_NSEFFONE[$idx]}" exp_nseff="${CONF_EXP_NSEFF[$idx]}"
    local cdir="${RESULT_DIR}/${name}"; mkdir -p "${cdir}"
    local dump="${cdir}/vector.bc.tsv" err="${cdir}/stderr.log"
    VEC_FILE[$name]="${dump}"; VEC_OK[$name]=no
    local critical=no; [ "${name}" = "CONTROL" ] && critical=yes

    log "[RUN] ${name}: env[${cenv:-none}] BC_BATCH_OVERRIDE=1024 gpu_opt (n=1)"
    local rc=0
    # shellcheck disable=SC2086
    env BC_BATCH_OVERRIDE=1024 ${cenv} timeout "${TIMEOUT_SEC}" "${RUNNER}" gpu_opt "${GRAPH_PATH}" --dump-bc \
        > "${dump}" 2> "${err}" || rc=$?
    local vsha=not_recorded; [ -f "${dump}" ] && vsha="$(sha256_file "${dump}")"
    log "[EXIT] ${name}: ${rc}"

    # 診断ログ抽出
    local d_fr d_no d_dm eff_batch nseff sub_batch num_subs mem_line
    d_fr="$(grep -m1 '\[Diag\] force_full_reset=' "${err}" | sed -n 's/.*force_full_reset=\([a-z]*\).*/\1/p')"
    d_no="$(grep -m1 '\[Diag\] force_ns_eff_one=' "${err}" | sed -n 's/.*force_ns_eff_one=\([a-z]*\).*/\1/p')"
    d_dm="$(grep -m1 '\[Diag\] diagnostic_mode=' "${err}" | sed -n 's/.*diagnostic_mode=\([a-z]*\).*/\1/p')"
    mem_line="$(grep 'dynamic(UM).*BATCH=.*SUB_BATCH=.*num_subs=.*NS_eff=' "${err}" | tail -n1 || true)"
    eff_batch="$(printf '%s\n' "${mem_line}" | sed -n 's/.*BATCH=\([0-9]*\), SUB_BATCH=.*/\1/p')"
    sub_batch="$(printf '%s\n' "${mem_line}" | sed -n 's/.*SUB_BATCH=\([0-9]*\),.*/\1/p')"
    num_subs="$(printf '%s\n' "${mem_line}" | sed -n 's/.*num_subs=\([0-9]*\),.*/\1/p')"
    nseff="$(printf '%s\n' "${mem_line}" | sed -n 's/.*NS_eff=\([0-9]*\).*/\1/p')"
    d_fr="${d_fr:-missing}"; d_no="${d_no:-missing}"; d_dm="${d_dm:-missing}"
    eff_batch="${eff_batch:-not_recorded}"; sub_batch="${sub_batch:-not_recorded}"
    num_subs="${num_subs:-not_recorded}"; nseff="${nseff:-not_recorded}"

    # 実行経路カウンタ (host 側分岐計数) と route_confirmed
    local fm vr exp_fm="${CONF_EXP_FULLMEMSET[$idx]}" exp_vr="${CONF_EXP_VISITED[$idx]}" route_confirmed
    fm="$(grep -m1 '\[Diag\] full_memset_calls=' "${err}" | sed -n 's/.*full_memset_calls=\([0-9]*\).*/\1/p')"
    vr="$(grep -m1 '\[Diag\] visited_reset_calls=' "${err}" | sed -n 's/.*visited_reset_calls=\([0-9]*\).*/\1/p')"
    fm="${fm:-not_recorded}"; vr="${vr:-not_recorded}"
    if [ "${fm}" = "${exp_fm}" ] && [ "${vr}" = "${exp_vr}" ]; then route_confirmed=yes; else route_confirmed=no; fi
    ROUTE_CONF[$name]="${route_confirmed}"

    local status=PASS reason=ok
    local fail_fatal=no
    if [ "${rc}" = "124" ]; then status=FAIL; reason="runner_timeout_${TIMEOUT_SEC}s"
    elif [ "${rc}" -ne 0 ]; then status=FAIL; reason="runner_exit_${rc}"
    elif log_has_failure_marker "${err}"; then status=FAIL; reason="failure_marker_in_stderr"
    elif ! validate_dump "${dump}" "${N}"; then status=FAIL; reason="invalid_or_incomplete_vector"
    elif dump_has_nonfinite "${dump}"; then status=FAIL; reason="nonfinite_value_in_vector"
    elif [ "${d_fr}" != "${exp_fr}" ] || [ "${d_no}" != "${exp_no}" ]; then
        status=FAIL; reason="diag_flags_not_reflected(force_full_reset=${d_fr}!=${exp_fr} or force_ns_eff_one=${d_no}!=${exp_no})"; fail_fatal=yes
    elif [ "${nseff}" != "${exp_nseff}" ]; then
        status=FAIL; reason="NS_eff_${nseff}_ne_expected_${exp_nseff}"; fail_fatal=yes
    elif [ "${eff_batch}" != "1024" ] || [ "${sub_batch}" != "1024" ] || [ "${num_subs}" != "1" ]; then
        status=FAIL; reason="batch_structure_unexpected(eff=${eff_batch},sub=${sub_batch},num=${num_subs})"; fail_fatal=yes
    else
        status=PASS; VEC_OK[$name]=yes
    fi

    (IFS=$'\t'; printf '%s\n' "${name}	${cenv:-none}	${exp_fr}	${exp_no}	${exp_nseff}	${rc}	${VEC_OK[$name]}	${eff_batch}	${nseff}	${sub_batch}	${num_subs}	${d_fr}	${d_no}	${d_dm}	${fm}	${vr}	${exp_fm}	${exp_vr}	${route_confirmed}	${vsha}	${status}	${reason}") >> "${EXEC_SUMMARY}"
    log "[${status}] ${name}: ${reason} (NS_eff=${nseff}, diag fr=${d_fr} no=${d_no}, full_memset=${fm}/${exp_fm} visited=${vr}/${exp_vr} route_confirmed=${route_confirmed})"

    # fail-fast: CONTROL 失敗 or 診断反映不良 は構造的 ABORT
    if [ "${status}" != "PASS" ]; then
        if [ "${critical}" = "yes" ]; then abort "CONTROL failed: ${reason}"; fi
        if [ "${fail_fatal}" = "yes" ]; then abort "${name} diagnostic instrumentation invalid: ${reason}"; fi
    fi
    return 0
}

for idx in "${!CONF_NAMES[@]}"; do run_config "${idx}"; done

# --- 比較 1 件を実行して comparison_matrix へ記録 -----------------------
compare_pair() {
    local a="$1" b="$2"
    local af="${VEC_FILE[$a]:-}" bf="${VEC_FILE[$b]:-}"
    local aok="${VEC_OK[$a]:-no}" bok="${VEC_OK[$b]:-no}"
    local cexit=na la=na lb=na ma=na mb=na mm=na mabs=na mabsi=na mabsa=na mabsb=na mrel=na mreli=na
    local sa=na sb=na status
    if [ "${aok}" != "yes" ] || [ "${bok}" != "yes" ]; then
        status=SKIPPED; log "[CMP ] ${a} vs ${b} -> SKIPPED (a=${aok} b=${bok})"
    else
        sa="$(sha256_file "${af}")"; sb="$(sha256_file "${bf}")"
        local md="${RESULT_DIR}/${a}__vs__${b}.md"; cexit=0
        python3 "${COMPARE}" "${af}" "${bf}" --label-a "${a}" --label-b "${b}" \
            --abs-tol "${ABS_TOL}" --rel-tol "${REL_TOL}" --out "${md}" \
            --extra "checkpoint_sha=${ACTUAL_SHA}" "pbs_job_id=${JOB_ID}" >> "${RUN_LOG}" 2>&1 || cexit=$?
        la="$(md_value 'ベクトル長 A' "${md}")"; lb="$(md_value 'ベクトル長 B' "${md}")"
        ma="$(md_value '欠損 index 数 (A のみ)' "${md}")"; mb="$(md_value '欠損 index 数 (B のみ)' "${md}")"
        mm="$(sed -n 's/.*不一致要素数 | \([0-9][0-9]*\) |$/\1/p' "${md}" | tail -n1)"
        local vp
        vp="$(md_value '最大絶対誤差' "${md}")"; mabs="${vp%% *}"; mabsi="$(printf '%s\n' "${vp}"|sed -n 's/.*index \([^)]*\)).*/\1/p')"
        vp="$(md_value '最大絶対誤差 index の値' "${md}")"; mabsa="$(printf '%s\n' "${vp}"|sed -n 's/^A=\([^,]*\), B=.*/\1/p')"; mabsb="$(printf '%s\n' "${vp}"|sed -n 's/^A=[^,]*, B=\(.*\)$/\1/p')"
        vp="$(md_value '最大相対誤差' "${md}")"; mrel="${vp%% *}"; mreli="$(printf '%s\n' "${vp}"|sed -n 's/.*index \([^)]*\)).*/\1/p')"
        if [ "${cexit}" = "0" ] && [ "${mm}" = "0" ] && [ "${ma}" = "0" ] && [ "${mb}" = "0" ] && [ "${la}" = "${N}" ] && [ "${lb}" = "${N}" ]; then
            status=PASS; else status=DIFF; fi
        log "[CMP ] ${a} vs ${b} -> ${status} (mismatch=${mm} max_abs=${mabs} max_rel=${mrel})"
    fi
    (IFS=$'\t'; printf '%s\n' "${a}	${b}	${aok}	${bok}	${cexit}	${la}	${lb}	${ma}	${mb}	${mm}	${mabs}	${mabsi}	${mabsa}	${mabsb}	${mrel}	${mreli}	${sa}	${sb}	${ABS_TOL}	${REL_TOL}	${status}") >> "${CMP_MATRIX}"
    # 主要 mismatch を変数へ (判定用)
    eval "MM_${a//-/_}__${b//-/_}='${mm}'"
    return 0
}

# --- 非干渉確認: CONTROL vs old_b1024 -----------------------------------
NONINTERF="not_verified"
if [ "${VEC_OK[old_b1024]:-no}" = "yes" ]; then
    compare_pair CONTROL old_b1024
    _ni="$(awk -F'\t' 'NR>1 && $1=="CONTROL" && $2=="old_b1024"{print $10}' "${CMP_MATRIX}" | tail -n1)"
    if [ "${_ni}" = "0" ]; then NONINTERF="verified_mismatch0"
    else
        NONINTERF="CHANGED_BASELINE(mismatch=${_ni})"
        log "[NONINTERF] CONTROL vs old_b1024 mismatch=${_ni} (!=0)"
        { printf 'overall_status=DIAGNOSTIC_INSTRUMENTATION_CHANGED_BASELINE\n';
          printf 'non_interference=%s\n' "${NONINTERF}";
          printf 'note=CONTROL differs from checkpoint baseline (old gpu_opt b1024) beyond mixed tolerance; reset/nseff judgments withheld.\n'; } > "${FINAL_STATUS_FILE}"
        { printf '# DIAGNOSIS (withheld)\n\noverall_status=DIAGNOSTIC_INSTRUMENTATION_CHANGED_BASELINE\n';
          printf 'CONTROL vs old_b1024 mismatch=%s (!=0). 診断計装が baseline を変えた可能性。reset/nseff 判定は行わない。\n' "${_ni}"; } > "${DIAGNOSIS}"
        log "overall_status=DIAGNOSTIC_INSTRUMENTATION_CHANGED_BASELINE"
        exit 1
    fi
else
    log "[NONINTERF] old_b1024 unavailable -> non-interference not_verified (external SKIPPED)"
fi

# --- 残り比較 -----------------------------------------------------------
for i in "${!CMP_A[@]}"; do
    a="${CMP_A[$i]}"; b="${CMP_B[$i]}"
    [ "${a}" = "CONTROL" ] && [ "${b}" = "old_b1024" ] && continue   # 既に実施
    compare_pair "${a}" "${b}"
done

# --- affected 8 頂点 + 判定 (python; raw 値無補正) -----------------------
DIAG_JSON="${RESULT_DIR}/.judgment.env"
python3 - "$N" "$GRAPH_PATH" "$AFFECTED_TSV" "$DIAG_JSON" "$ABS_TOL" "$REL_TOL" \
  "${VEC_OK[CONTROL]:-no}:${VEC_FILE[CONTROL]:-}" \
  "${VEC_OK[T-RESET]:-no}:${VEC_FILE[T-RESET]:-}" \
  "${VEC_OK[T-NSEFF]:-no}:${VEC_FILE[T-NSEFF]:-}" \
  "${VEC_OK[old_b1024]:-no}:${VEC_FILE[old_b1024]:-}" \
  "${VEC_OK[old_b9792]:-no}:${VEC_FILE[old_b9792]:-}" \
  "${VEC_OK[old_chunk_b16384]:-no}:${VEC_FILE[old_chunk_b16384]:-}" <<'PYEOF'
import sys, math
N=int(sys.argv[1]); GRAPH=sys.argv[2]; OUT=sys.argv[3]; JENV=sys.argv[4]
ABS_TOL=float(sys.argv[5]); REL_TOL=float(sys.argv[6])
names=["CONTROL","T-RESET","T-NSEFF","old_b1024","old_b9792","old_chunk_b16384"]
specs=sys.argv[7:7+6]
AFFECTED=[7954,95156,143358,165886,226184,228350,289284,325556]
def load(p):
    v={}
    with open(p) as f:
        for line in f:
            if not line.strip() or line.startswith('#'): continue
            a,b=line.split('\t')[:2]; v[int(a)]=float(b)
    return v
V={}; OKV={}
for nm,sp in zip(names,specs):
    ok,path=sp.split(':',1); OKV[nm]=(ok=='yes' and path)
    V[nm]=load(path) if OKV[nm] else None
with open(GRAPH) as f:
    n,m=map(int,f.readline().split()); ptr=list(map(int,f.readline().split()))
deg=[ptr[i+1]-ptr[i] for i in range(n)]
def within(a,b):
    if a is None or b is None: return None
    return abs(a-b)<=ABS_TOL+REL_TOL*max(abs(a),abs(b))
# affected_vertices.tsv
with open(OUT,'w') as o:
    o.write("index\tdegree\t"+"\t".join(names)+"\tdiff_TRESET_minus_CONTROL\tdiff_TNSEFF_minus_CONTROL\tdiff_CONTROL_minus_old_b9792\n")
    for i in AFFECTED:
        vals={nm:(V[nm].get(i) if V[nm] is not None else None) for nm in names}
        def r(x): return repr(x) if x is not None else "NA"
        d1=(vals["T-RESET"]-vals["CONTROL"]) if vals["T-RESET"] is not None and vals["CONTROL"] is not None else None
        d2=(vals["T-NSEFF"]-vals["CONTROL"]) if vals["T-NSEFF"] is not None and vals["CONTROL"] is not None else None
        d3=(vals["CONTROL"]-vals["old_b9792"]) if vals["CONTROL"] is not None and vals["old_b9792"] is not None else None
        o.write(f"{i}\t{deg[i]}\t"+"\t".join(r(vals[nm]) for nm in names)+f"\t{r(d1)}\t{r(d2)}\t{r(d3)}\n")
# full-vector mixed-tol mismatch
def mism(a,b):
    if V[a] is None or V[b] is None: return None
    c=0
    for i in set(V[a])&set(V[b]):
        x=V[a][i]; y=V[b][i]
        if not(math.isfinite(x) and math.isfinite(y)): continue
        if abs(x-y)>ABS_TOL+REL_TOL*max(abs(x),abs(y)): c+=1
    return c
def moved_to_stress(cand, stress):
    # affected 頂点で cand が stress 側と混合許容内一致する数 (CONTROL は不一致のはず)
    if V[cand] is None or V[stress] is None or V["CONTROL"] is None: return (None,None,None)
    match_stress=0; match_control=0; diff_from_control=0
    for i in AFFECTED:
        c=V["CONTROL"].get(i); k=V[cand].get(i); s=V[stress].get(i)
        if k is None: continue
        if s is not None and within(k,s): match_stress+=1
        if within(k,c): match_control+=1
        else: diff_from_control+=1
    return (match_stress,match_control,diff_from_control)
mm_ctrl_reset=mism("CONTROL","T-RESET")
mm_ctrl_nseff=mism("CONTROL","T-NSEFF")
# reset 判定
def judge(cand, stress_list):
    mm=mism("CONTROL",cand)
    if mm is None: return ("INCONCLUSIVE", "control_or_cand_unavailable", mm)
    if mm==0: return ("NOT_DISTINGUISHED", "control_vs_cand_mismatch0_within_mixed_tol", mm)
    # 変化あり: affected 頂点で stress 側へ寄ったか
    best=None
    for st in stress_list:
        ms,mc,dc=moved_to_stress(cand, st)
        if ms is None: continue
        if best is None or ms>best[1]: best=(st,ms,mc,dc)
    if best is None: return ("INCONCLUSIVE", f"changed(mismatch={mm})_but_no_stress_ref", mm)
    st,ms,mc,dc=best
    if ms>0 and ms>=dc: return ("IMPLICATED", f"changed(mismatch={mm}); at {ms}/{len(AFFECTED)} affected verts cand matches stress({st})", mm)
    return ("INCONCLUSIVE", f"changed(mismatch={mm}) but affected verts not clearly stress-side (match_stress={ms})", mm)
rj=judge("T-RESET", ["old_b9792","old_chunk_b16384"])
nj=judge("T-NSEFF", ["old_b9792","old_chunk_b16384"])
# 判定名は非因果表現 (ASSOCIATED): T-RESET で値が変化しても reset 内容単独ではなく full
# memset に伴う実行タイミング・GPU スケジューリング・atomicAdd 順序の変化を含むため単独原因
# と断定しない。T-NSEFF も stream 数・occupancy・atomic 順序を含む「関連」として扱う。
reset_status = {"NOT_DISTINGUISHED":"RESET_NOT_DISTINGUISHED","IMPLICATED":"RESET_PATH_OR_SCHEDULING_ASSOCIATED","INCONCLUSIVE":"RESET_INCONCLUSIVE"}[rj[0]]
nseff_status = {"NOT_DISTINGUISHED":"NS_EFF_NOT_DISTINGUISHED","IMPLICATED":"NS_EFF_OR_OCCUPANCY_ASSOCIATED","INCONCLUSIVE":"NS_EFF_INCONCLUSIVE"}[nj[0]]
with open(JENV,'w') as j:
    j.write(f"RESET_STATUS='{reset_status}'\n")
    j.write(f"NSEFF_STATUS='{nseff_status}'\n")
    j.write(f"MM_CONTROL_TRESET='{mm_ctrl_reset}'\n")
    j.write(f"MM_CONTROL_TNSEFF='{mm_ctrl_nseff}'\n")
    j.write(f"RESET_REASON='{rj[1]}'\n")
    j.write(f"NSEFF_REASON='{nj[1]}'\n")
print(f"reset={reset_status} ({rj[1]})")
print(f"nseff={nseff_status} ({nj[1]})")
PYEOF

# --- 判定を読み込み DIAGNOSIS / FINAL_STATUS へ --------------------------
RESET_STATUS=NS_reset_unknown; NSEFF_STATUS=NS_eff_unknown
MM_CONTROL_TRESET=na; MM_CONTROL_TNSEFF=na; RESET_REASON=na; NSEFF_REASON=na
if [ -f "${DIAG_JSON}" ]; then . "${DIAG_JSON}"; fi

{
  printf '# DIAGNOSIS — T-RESET / T-NSEFF (checkpoint %s)\n\n' "${ACTUAL_SHA}"
  printf -- '- non_interference (CONTROL vs old_b1024): %s\n' "${NONINTERF}"
  printf -- '- CONTROL vs T-RESET mixed-tol mismatch: %s\n' "${MM_CONTROL_TRESET}"
  printf -- '- CONTROL vs T-NSEFF mixed-tol mismatch: %s\n' "${MM_CONTROL_TNSEFF}"
  printf '\n## reset judgment: **%s**\n%s\n' "${RESET_STATUS}" "${RESET_REASON}"
  printf '\n## NS_eff judgment: **%s**\n%s\n' "${NSEFF_STATUS}" "${NSEFF_REASON}"
  printf '\n注: mismatch=0 は「事前設定した混合許容内で一致」であり bitwise/SHA256 一致ではない。\n'
  printf 'PathMerge は本診断に含めない。許容値は不変で FAIL を PASS に書き換えない。\n'
  printf '「近づく」は affected 8 頂点の値/誤差距離/mismatch 集合で定量判断 (affected_vertices.tsv 参照)。\n'
  printf '判定名は非因果 (ASSOCIATED): RESET_PATH_OR_SCHEDULING_ASSOCIATED は full memset に伴う実行\n'
  printf 'タイミング・GPU スケジューリング・atomicAdd 順序を、NS_EFF_OR_OCCUPANCY_ASSOCIATED は stream\n'
  printf '数・occupancy・atomic 順序を含む「関連」であり、reset 内容/stream 数の単独原因断定ではない。\n'
} > "${DIAGNOSIS}"

{
  printf 'overall_status=DIAGNOSTIC_COMPLETE\n'
  printf 'non_interference=%s\n' "${NONINTERF}"
  printf 'reset_status=%s\n' "${RESET_STATUS}"
  printf 'nseff_status=%s\n' "${NSEFF_STATUS}"
  printf 'mm_control_treset=%s\n' "${MM_CONTROL_TRESET}"
  printf 'mm_control_tnseff=%s\n' "${MM_CONTROL_TNSEFF}"
} > "${FINAL_STATUS_FILE}"

{ printf '\n[final]\nnon_interference=%s\nreset_status=%s\nnseff_status=%s\n' \
    "${NONINTERF}" "${RESET_STATUS}" "${NSEFF_STATUS}"; } >> "${MANIFEST}"
log "reset_status=${RESET_STATUS} | nseff_status=${NSEFF_STATUS} | non_interference=${NONINTERF}"
log "NOTE: diagnosis only; timings not performance; acquired results retained."
exit 0
