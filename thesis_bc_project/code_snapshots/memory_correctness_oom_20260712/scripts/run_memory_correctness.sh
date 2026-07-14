#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=6:00:00
#PBS -N bc_mem_correct
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
#  run_memory_correctness.sh — メモリ経路 full-vector 正確性検証 (Stage 4A)
#
#  1 グラフ (325557_3216152) だけを用い、PathMerge を独立参照として
#  UM / Pure / Chunked 固有経路の全 BC ベクトル正確性を検証する。
#  全面バッチ掃引や性能再測定は行わない。時間値は性能結果として使わない。
#
#  参照 (reference):
#    PathMerge          b4096   (pathmerge_bc, PATHMERGE_BC_BATCH_SIZE=4096)
#  候補 (candidate):
#    GPU_Opt (UM)       b10240  (gpu_opt,              BC_BATCH_OVERRIDE=10240)
#    GPU_Opt_Pure       b1024   (gpu_opt_pure,         BC_BATCH_OVERRIDE=1024)
#    GPU_Opt_Pure_Chunked b16384(gpu_opt_pure_chunked, BC_BATCH_OVERRIDE=16384)
#
#  各構成 n=1、warmup なし。stdout=BC ベクトル (--dump-bc)、stderr=phase/mem ログ。
#  runner 非 0 / FAIL / OOM / TIMEOUT / NaN / Inf / 欠損 / 不一致 が起きたら
#  直ちに停止し、次構成へ進まない (fail-fast)。
#
#  出力 (RESULT_DIR, build_miyabi 配下 = gitignored):
#    MANIFEST.txt / correctness_summary.tsv / run.log
#    pathmerge_b4096.bc.tsv / pathmerge_b4096.stderr.log
#    gpu_opt_b10240.bc.tsv / gpu_opt_b10240.stderr.log / gpu_opt_b10240_vs_reference.md
#    gpu_opt_pure_b1024.bc.tsv / .stderr.log / _vs_reference.md
#    gpu_opt_pure_chunked_b16384.bc.tsv / .stderr.log / _vs_reference.md
#
#  巨大な BC ベクトル (.bc.tsv) 自体は Git へ追加しない (サマリのみ curate)。
#
#  環境変数:
#    EXPECTED_SHA  checkpoint SHA (実行時は必須; HEAD と一致しなければ exit 2)
#    DRY_RUN       1 で計画表示のみ (build/runner/GPU/qsub/比較を一切行わない)
#    SKIP_BUILD    1 でビルドをスキップ (事前クリーンビルド + SKIP_BUILD=1 推奨)
#    JOBS          ビルド並列数 (default 8)
#    TIMEOUT_SEC   1 構成あたり最大実行時間 (default 5400)
#    ABS_TOL       絶対許容 (default 1e-3)
#    REL_TOL       相対許容 (default 1e-6)
#    GRAPH         対象グラフ相対パス (default data/325557_3216152)
# ============================================================

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
TIMEOUT_SEC="${TIMEOUT_SEC:-5400}"
ABS_TOL="${ABS_TOL:-1e-3}"
REL_TOL="${REL_TOL:-1e-6}"
GRAPH_REL="${GRAPH:-data/325557_3216152}"

# --- 検証構成 (index 0 = 参照, 1..3 = 候補) -----------------------------
CONFIG_NAMES=(pathmerge_b4096 gpu_opt_b10240 gpu_opt_pure_b1024 gpu_opt_pure_chunked_b16384)
CONFIG_IMPLS=(pathmerge_bc gpu_opt gpu_opt_pure gpu_opt_pure_chunked)
CONFIG_BATCHENV=(PATHMERGE_BC_BATCH_SIZE BC_BATCH_OVERRIDE BC_BATCH_OVERRIDE BC_BATCH_OVERRIDE)
CONFIG_BATCH=(4096 10240 1024 16384)
CONFIG_PATHTYPE=(pathmerge um pure chunked)
CONFIG_ROLE=(reference candidate candidate candidate)

# ============================================================
#  DRY_RUN: 計画表示のみ。build / runner / GPU / qsub / 比較 / result 更新なし。
# ============================================================
if [ "${DRY_RUN}" = "1" ]; then
    printf '%s\n' \
        "DRY RUN: no build, runner, GPU access, qsub, result update, or BC dump" \
        "Project    : ${PROJECT_DIR}" \
        "Runner     : ${RUNNER}" \
        "Compare    : ${COMPARE}" \
        "Graph      : ${GRAPH_REL}" \
        "Planned out: ${BUILD_DIR}/result_memory_correctness_<timestamp>_<PBS_JOBID>/" \
        "Runs (n=1 each, no warmup; timings are NOT performance results):"
    for i in "${!CONFIG_NAMES[@]}"; do
        printf '  [%s] %s: %s=%s %s %s --dump-bc  (path=%s)\n' \
            "${CONFIG_ROLE[$i]}" "${CONFIG_NAMES[$i]}" \
            "${CONFIG_BATCHENV[$i]}" "${CONFIG_BATCH[$i]}" \
            "${CONFIG_IMPLS[$i]}" "${GRAPH_REL}" "${CONFIG_PATHTYPE[$i]}"
    done
    printf '%s\n' \
        "Reference vector = pathmerge_b4096.bc.tsv (all candidates compared to it)" \
        "Comparison : abs_diff <= ${ABS_TOL} + ${REL_TOL} * max(|reference|,|candidate|)" \
        "PASS       : zero mixed-tolerance mismatches, complete indices, finite values," \
        "             plus path evidence (um: oversubscribed HBM3 streaming;" \
        "             chunked: num_subs>1 manual chunking; pure: cudaMalloc, no UM, no OOM)" \
        "Timeout    : ${TIMEOUT_SEC} s/config"
    exit 0
fi

# ============================================================
#  checkpoint 検証 (実行時は EXPECTED_SHA 必須)
# ============================================================
: "${EXPECTED_SHA:?EXPECTED_SHA must be set (checkpoint SHA)}"
ACTUAL_SHA="$(git rev-parse HEAD)"
test "${ACTUAL_SHA}" = "${EXPECTED_SHA}" || {
    echo "ERROR: checkpoint mismatch (HEAD=${ACTUAL_SHA} != EXPECTED_SHA=${EXPECTED_SHA})" >&2
    exit 2
}

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_ID="${PBS_JOBID:-not_pbs}"
RESULT_DIR="${BUILD_DIR}/result_memory_correctness_${TIMESTAMP}_${JOB_ID}"
MANIFEST="${RESULT_DIR}/MANIFEST.txt"
SUMMARY="${RESULT_DIR}/correctness_summary.tsv"
RUN_LOG="${RESULT_DIR}/run.log"
GRAPH_PATH="${PROJECT_DIR}/${GRAPH_REL}"
mkdir -p "${RESULT_DIR}"
: > "${RUN_LOG}"

log() { printf '%s\n' "$*" | tee -a "${RUN_LOG}"; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }

md_value() {
    # comparison.md の表 (| キー | 値 |) から値を取り出す
    local key="$1" file="$2"
    awk -F '|' -v wanted="${key}" '
        { k=$2; v=$3;
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", k);
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", v);
          if (k == wanted) { print v; exit } }' "${file}"
}

validate_dump() {
    # --dump-bc の全 index (0..n-1) が過不足なく、値が有限/特殊値であることを検証
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
    # --dump-bc の値カラム (2列目) に非有限値 (nan/inf) があれば 0 を返す。
    # 有限値は %.15e 形式なので nan/inf 部分文字列を含まない。参照・候補の両方に適用。
    awk -F '\t' '
        !/^#/ && NF == 2 { v=tolower($2); if (v ~ /nan|inf/) { found=1; exit } }
        END { exit found ? 0 : 1 }' "$1"
}

log_has_failure_marker() {
    # 良性警告 (Pure の safe-limit 警告, PathMerge の clamp 警告) を除外してから
    # 実障害マーカー (FAIL/OOM/TIMEOUT/NaN/Inf, および CUDA error / out of memory) を探す
    grep -Ev 'exceeds safe limit|exceeds HBM3 budget|may cause cudaMalloc OOM|clamping to' "$1" \
        | grep -Eiq '(^|[^[:alnum:]_])(FAIL(ED|URE)?|OOM|TIMEOUT|NaN|[+-]?Inf(inity)?)([^[:alnum:]_]|$)|CUDA error at|out of memory'
}

# --- グラフ情報 ---------------------------------------------------------
read -r N M < "${GRAPH_PATH}"
GRAPH_SHA="$(sha256_file "${GRAPH_PATH}")"
GRAPH_NAME="$(basename "${GRAPH_REL}")"

# --- MANIFEST / SUMMARY プリアンブル ------------------------------------
printf '%s\n' \
    "checkpoint_sha=${ACTUAL_SHA}" \
    "pbs_job_id=${JOB_ID}" \
    "project_dir=${PROJECT_DIR}" \
    "result_dir=${RESULT_DIR}" \
    "runner=${RUNNER}" \
    "graph=${GRAPH_REL}" \
    "graph_sha256=${GRAPH_SHA}" \
    "n_nodes=${N}" \
    "n_edges=${M}" \
    "runs_per_configuration=1" \
    "warmup=none" \
    "timing_usage=correctness_only_not_performance" \
    "reference=pathmerge_b4096" \
    "abs_tol=${ABS_TOL}" \
    "rel_tol=${REL_TOL}" \
    "criterion=abs_diff <= abs_tol + rel_tol * max(abs(reference),abs(candidate))" > "${MANIFEST}"

printf '%s\n' \
    $'checkpoint_sha\tpbs_job_id\tconfig\timpl\trole\tpath_type\tgraph\tgraph_sha256\tn\tm\trunner_exit\tvector_sha256\trequested_batch\teffective_batch\tSUB_BATCH\tnum_subs\tNS_eff\toversubscribed\tuses_um\tcomparison_exit\tabs_tol\trel_tol\treference_vector_length\tcandidate_vector_length\tmissing_reference_only\tmissing_candidate_only\tmismatched_elements\tmax_abs_error\tmax_abs_index\tmax_abs_reference\tmax_abs_candidate\tmax_rel_error\tmax_rel_index\tmax_rel_reference\tmax_rel_candidate\tmax_bc_reference_index\tmax_bc_reference_value\tmax_bc_candidate_index\tmax_bc_candidate_value\tstatus\treason' \
    > "${SUMMARY}"

log "checkpoint=${ACTUAL_SHA} job=${JOB_ID} result=${RESULT_DIR}"
log "graph=${GRAPH_REL} n=${N} m=${M} sha256=${GRAPH_SHA}"

# --- ビルド (成果物であり性能結果ではない) ------------------------------
if [ "${SKIP_BUILD}" != "1" ]; then
    log "[BUILD] scripts/build_miyabi_interactive.sh (run_benchmark は cugraph_bc_mini に依存)"
    JOBS="${JOBS}" BUILD_DIR="${BUILD_DIR}" \
        bash "${PROJECT_DIR}/scripts/build_miyabi_interactive.sh" 2>&1 | tee -a "${RUN_LOG}"
fi
if [ ! -x "${RUNNER}" ]; then
    log "ERROR: runner not found or not executable: ${RUNNER}"
    exit 2
fi

# --- per-config の状態変数 (グローバル、record_config が参照) ------------
reset_fields() {
    runner_exit=not_recorded vector_sha=not_recorded
    requested_batch=not_recorded effective_batch=not_recorded
    sub_batch=not_recorded num_subs=not_recorded ns_eff=not_recorded
    oversubscribed=not_recorded uses_um=not_recorded
    comparison_exit=not_applicable
    len_a=not_applicable len_b=not_applicable
    missing_a=not_applicable missing_b=not_applicable mismatches=not_applicable
    max_abs=not_applicable max_abs_idx=not_applicable max_abs_a=not_applicable max_abs_b=not_applicable
    max_rel=not_applicable max_rel_idx=not_applicable max_rel_a=not_applicable max_rel_b=not_applicable
    max_bc_a_idx=not_applicable max_bc_a_value=not_applicable
    max_bc_b_idx=not_applicable max_bc_b_value=not_applicable
}

record_config() {
    local status="$1" reason="$2"
    local row=(
        "${ACTUAL_SHA}" "${JOB_ID}" "${cfg_name}" "${cfg_impl}" "${cfg_role}" "${cfg_path}"
        "${GRAPH_NAME}" "${GRAPH_SHA}" "${N}" "${M}"
        "${runner_exit}" "${vector_sha}" "${requested_batch}" "${effective_batch}"
        "${sub_batch}" "${num_subs}" "${ns_eff}" "${oversubscribed}" "${uses_um}"
        "${comparison_exit}" "${ABS_TOL}" "${REL_TOL}" "${len_a}" "${len_b}"
        "${missing_a}" "${missing_b}" "${mismatches}"
        "${max_abs}" "${max_abs_idx}" "${max_abs_a}" "${max_abs_b}"
        "${max_rel}" "${max_rel_idx}" "${max_rel_a}" "${max_rel_b}"
        "${max_bc_a_idx}" "${max_bc_a_value}" "${max_bc_b_idx}" "${max_bc_b_value}"
        "${status}" "${reason}"
    )
    (IFS=$'\t'; printf '%s\n' "${row[*]}") >> "${SUMMARY}"
    {
        printf '\n[config %s | role=%s | path=%s]\n' "${cfg_name}" "${cfg_role}" "${cfg_path}"
        printf 'impl=%s\nrequested_batch=%s\neffective_batch=%s\n' "${cfg_impl}" "${requested_batch}" "${effective_batch}"
        printf 'SUB_BATCH=%s\nnum_subs=%s\nNS_eff=%s\noversubscribed=%s\nuses_um=%s\n' \
            "${sub_batch}" "${num_subs}" "${ns_eff}" "${oversubscribed}" "${uses_um}"
        printf 'runner_exit=%s\nvector_sha256=%s\n' "${runner_exit}" "${vector_sha}"
        printf 'comparison_exit=%s\nreference_vector_length=%s\ncandidate_vector_length=%s\n' \
            "${comparison_exit}" "${len_a}" "${len_b}"
        printf 'missing_reference_only=%s\nmissing_candidate_only=%s\nmismatched_elements=%s\n' \
            "${missing_a}" "${missing_b}" "${mismatches}"
        printf 'max_abs_error=%s (index %s; ref=%s cand=%s)\n' "${max_abs}" "${max_abs_idx}" "${max_abs_a}" "${max_abs_b}"
        printf 'max_rel_error=%s (index %s; ref=%s cand=%s)\n' "${max_rel}" "${max_rel_idx}" "${max_rel_a}" "${max_rel_b}"
        printf 'max_bc_reference=index %s value %s\nmax_bc_candidate=index %s value %s\n' \
            "${max_bc_a_idx}" "${max_bc_a_value}" "${max_bc_b_idx}" "${max_bc_b_value}"
        printf 'status=%s\nreason=%s\n' "${status}" "${reason}"
    } >> "${MANIFEST}"
}

# --- メモリ経路の証拠を stderr から抽出 --------------------------------
extract_path_evidence() {
    local stderr_file="$1" path_type="$2" line
    case "${path_type}" in
        um)
            line="$(grep 'dynamic(UM).*BATCH=.*SUB_BATCH=.*num_subs=.*NS_eff=' "${stderr_file}" | tail -n1 || true)"
            if [ -n "${line}" ]; then
                # 貪欲マッチが末尾 SUB_BATCH= を拾わないよう ", SUB_BATCH=" を右アンカーにする
                effective_batch="$(printf '%s\n' "${line}" | sed -n 's/.*BATCH=\([0-9]*\), SUB_BATCH=.*/\1/p')"
                sub_batch="$(printf '%s\n' "${line}" | sed -n 's/.*SUB_BATCH=\([0-9]*\),.*/\1/p')"
                num_subs="$(printf '%s\n' "${line}" | sed -n 's/.*num_subs=\([0-9]*\),.*/\1/p')"
                ns_eff="$(printf '%s\n' "${line}" | sed -n 's/.*NS_eff=\([0-9]*\).*/\1/p')"
            fi
            uses_um=true
            if grep -q '\[Mode\] HBM3 streaming' "${stderr_file}"; then oversubscribed=true; else oversubscribed=false; fi
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
            if [ -n "${line}" ]; then
                effective_batch="$(printf '%s\n' "${line}" | sed -n 's/.*batch_per_stream=\([0-9]*\).*/\1/p')"
            fi
            sub_batch=not_applicable num_subs=not_applicable ns_eff=not_applicable
            oversubscribed=not_applicable
            if grep -q 'dynamic(UM)' "${stderr_file}"; then uses_um=true; else uses_um=false; fi
            ;;
        pathmerge)
            line="$(grep '\[PathMerge\].*batch_size=.*num_sources=.*num_batches=' "${stderr_file}" | tail -n1 || true)"
            if [ -n "${line}" ]; then
                effective_batch="$(printf '%s\n' "${line}" | sed -n 's/.*batch_size=\([0-9]*\),.*/\1/p')"
                num_subs="$(printf '%s\n' "${line}" | sed -n 's/.*num_batches=\([0-9]*\).*/\1/p')"
            fi
            sub_batch=not_applicable ns_eff=not_applicable
            uses_um=managed_csr
            oversubscribed=not_applicable
            ;;
    esac
    effective_batch="${effective_batch:-not_recorded}"
    sub_batch="${sub_batch:-not_recorded}"
    num_subs="${num_subs:-not_recorded}"
    ns_eff="${ns_eff:-not_recorded}"
}

# --- 経路 PASS 条件 (単に成功しただけでは PASS にしない) ----------------
path_evidence_ok() {
    local path_type="$1"
    case "${path_type}" in
        um)
            [ "${effective_batch}" = "${requested_batch}" ] || { path_reason="effective_batch_${effective_batch}_ne_requested_${requested_batch}"; return 1; }
            [ "${uses_um}" = "true" ] || { path_reason="um_path_not_detected"; return 1; }
            [ "${oversubscribed}" = "true" ] || { path_reason="um_oversubscription_path_not_exercised"; return 1; }
            case "${num_subs}" in ''|*[!0-9]*) path_reason="num_subs_not_recorded"; return 1;; esac
            [ "${num_subs}" -ge 1 ] || { path_reason="num_subs_lt_1"; return 1; }
            grep -q 'Prefetch cum=' "${cfg_stderr}" || { path_reason="prefetch_migration_line_absent"; return 1; }
            ;;
        chunked)
            [ "${effective_batch}" = "${requested_batch}" ] || { path_reason="effective_batch_${effective_batch}_ne_requested_${requested_batch}"; return 1; }
            [ "${oversubscribed}" = "true" ] || { path_reason="manual_chunking_mode_absent"; return 1; }
            case "${num_subs}" in ''|*[!0-9]*) path_reason="num_subs_not_recorded"; return 1;; esac
            [ "${num_subs}" -gt 1 ] || { path_reason="num_subs_not_gt_1_chunked_path_not_exercised"; return 1; }
            ;;
        pure)
            [ "${effective_batch}" = "${requested_batch}" ] || { path_reason="effective_batch_${effective_batch}_ne_requested_${requested_batch}"; return 1; }
            [ "${uses_um}" = "false" ] || { path_reason="pure_path_unexpectedly_uses_um"; return 1; }
            ;;
        pathmerge)
            [ "${effective_batch}" = "${requested_batch}" ] || { path_reason="reference_batch_clamped_${effective_batch}_ne_${requested_batch}"; return 1; }
            ;;
    esac
    path_reason="path_evidence_ok"
    return 0
}

# ============================================================
#  1 構成を実行し、経路検証 + (候補なら) 参照との全ベクトル比較を行う。
#  失敗時は record してから非 0 を返す (set -e により呼び出し側で停止)。
# ============================================================
REFERENCE_DUMP=""

run_config() {
    local idx="$1"
    cfg_name="${CONFIG_NAMES[$idx]}"
    cfg_impl="${CONFIG_IMPLS[$idx]}"
    local cfg_batchenv="${CONFIG_BATCHENV[$idx]}"
    cfg_role="${CONFIG_ROLE[$idx]}"
    cfg_path="${CONFIG_PATHTYPE[$idx]}"
    reset_fields
    requested_batch="${CONFIG_BATCH[$idx]}"

    cfg_stderr="${RESULT_DIR}/${cfg_name}.stderr.log"
    local cfg_dump="${RESULT_DIR}/${cfg_name}.bc.tsv"

    log "[RUN] ${cfg_name}: ${cfg_batchenv}=${requested_batch} ${cfg_impl} ${GRAPH_REL} --dump-bc (n=1, no warmup)"
    runner_exit=0
    env "${cfg_batchenv}=${requested_batch}" \
        timeout "${TIMEOUT_SEC}" "${RUNNER}" "${cfg_impl}" "${GRAPH_PATH}" --dump-bc \
        > "${cfg_dump}" 2> "${cfg_stderr}" || runner_exit=$?
    [ -f "${cfg_dump}" ] && vector_sha="$(sha256_file "${cfg_dump}")"
    log "[EXIT] ${cfg_name}: ${runner_exit}"

    extract_path_evidence "${cfg_stderr}" "${cfg_path}"

    if [ "${runner_exit}" = "124" ]; then
        record_config FAIL "runner_timeout_${TIMEOUT_SEC}s"; return 4
    fi
    if [ "${runner_exit}" -ne 0 ]; then
        record_config FAIL "runner_exit_${runner_exit}"; return 4
    fi
    if log_has_failure_marker "${cfg_stderr}"; then
        record_config FAIL "failure_marker_in_stderr"; return 4
    fi
    if ! validate_dump "${cfg_dump}" "${N}"; then
        record_config FAIL "invalid_or_incomplete_vector"; return 4
    fi
    if dump_has_nonfinite "${cfg_dump}"; then
        record_config FAIL "nonfinite_value_in_vector"; return 4
    fi

    local path_reason
    if ! path_evidence_ok "${cfg_path}"; then
        record_config FAIL "path_${path_reason}"; return 4
    fi

    if [ "${cfg_role}" = "reference" ]; then
        REFERENCE_DUMP="${cfg_dump}"
        len_a="${N}"; len_b=not_applicable
        record_config PASS "reference_valid_${path_reason}"
        log "[REF ] ${cfg_name}: length=${N}, batch=${effective_batch}, num_batches=${num_subs}"
        return 0
    fi

    # --- 候補: 参照 (PathMerge b4096) との全ベクトル比較 -----------------
    if [ -z "${REFERENCE_DUMP}" ] || [ ! -f "${REFERENCE_DUMP}" ]; then
        record_config FAIL "reference_dump_missing"; return 5
    fi
    local vs_md="${RESULT_DIR}/${cfg_name}_vs_reference.md"
    comparison_exit=0
    python3 "${COMPARE}" "${REFERENCE_DUMP}" "${cfg_dump}" \
        --label-a PathMerge_b4096 --label-b "${cfg_name}" \
        --abs-tol "${ABS_TOL}" --rel-tol "${REL_TOL}" --out "${vs_md}" \
        --extra "checkpoint_sha=${ACTUAL_SHA}" "pbs_job_id=${JOB_ID}" \
        "graph_path=${GRAPH_PATH}" "graph_sha256=${GRAPH_SHA}" "n=${N}" "m=${M}" \
        "reference_impl=pathmerge_bc" "reference_batch=4096" \
        "candidate_impl=${cfg_impl}" "path_type=${cfg_path}" \
        "requested_batch=${requested_batch}" "effective_batch=${effective_batch}" \
        "SUB_BATCH=${sub_batch}" "num_subs=${num_subs}" "NS_eff=${ns_eff}" \
        "oversubscribed=${oversubscribed}" "uses_um=${uses_um}" \
        >> "${RUN_LOG}" 2>&1 || comparison_exit=$?

    len_a="$(md_value 'ベクトル長 A' "${vs_md}")"
    len_b="$(md_value 'ベクトル長 B' "${vs_md}")"
    missing_a="$(md_value '欠損 index 数 (A のみ)' "${vs_md}")"
    missing_b="$(md_value '欠損 index 数 (B のみ)' "${vs_md}")"
    mismatches="$(sed -n 's/.*不一致要素数 | \([0-9][0-9]*\) |$/\1/p' "${vs_md}" | tail -n1)"

    local vp
    vp="$(md_value '最大絶対誤差' "${vs_md}")"
    max_abs="${vp%% *}"
    max_abs_idx="$(printf '%s\n' "${vp}" | sed -n 's/.*index \([^)]*\)).*/\1/p')"
    vp="$(md_value '最大絶対誤差 index の値' "${vs_md}")"
    max_abs_a="$(printf '%s\n' "${vp}" | sed -n 's/^A=\([^,]*\), B=.*/\1/p')"
    max_abs_b="$(printf '%s\n' "${vp}" | sed -n 's/^A=[^,]*, B=\(.*\)$/\1/p')"
    vp="$(md_value '最大相対誤差' "${vs_md}")"
    max_rel="${vp%% *}"
    max_rel_idx="$(printf '%s\n' "${vp}" | sed -n 's/.*index \([^)]*\)).*/\1/p')"
    vp="$(md_value '最大相対誤差 index の値' "${vs_md}")"
    max_rel_a="$(printf '%s\n' "${vp}" | sed -n 's/^A=\([^,]*\), B=.*/\1/p')"
    max_rel_b="$(printf '%s\n' "${vp}" | sed -n 's/^A=[^,]*, B=\(.*\)$/\1/p')"
    vp="$(md_value 'Max BC A' "${vs_md}")"
    max_bc_a_idx="$(printf '%s\n' "${vp}" | sed -n 's/^index \([^,]*\), value .*/\1/p')"
    max_bc_a_value="$(printf '%s\n' "${vp}" | sed -n 's/^index [^,]*, value \(.*\)$/\1/p')"
    vp="$(md_value 'Max BC B' "${vs_md}")"
    max_bc_b_idx="$(printf '%s\n' "${vp}" | sed -n 's/^index \([^,]*\), value .*/\1/p')"
    max_bc_b_value="$(printf '%s\n' "${vp}" | sed -n 's/^index [^,]*, value \(.*\)$/\1/p')"

    if [ "${comparison_exit}" -ne 0 ] || [ "${mismatches}" != "0" ] || \
       [ "${missing_a}" != "0" ] || [ "${missing_b}" != "0" ] || \
       [ "${len_a}" != "${N}" ] || [ "${len_b}" != "${N}" ]; then
        record_config FAIL "full_vector_comparison_failed"; return 3
    fi
    record_config PASS "mixed_tolerance_mismatches_0"
    log "[PASS] ${cfg_name}: all ${N} elements within mixed tolerance vs PathMerge b4096"
    return 0
}

# ============================================================
#  実行: 参照 → 候補 (fail-fast; 失敗したら次構成へ進まない)
# ============================================================
for idx in "${!CONFIG_NAMES[@]}"; do
    run_config "${idx}"
done

log "PASS: reference + 3 candidates全て full-vector 一致 (timings are not performance results)"
