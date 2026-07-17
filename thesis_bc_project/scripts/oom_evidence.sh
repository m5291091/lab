#!/bin/bash

# 強い OOM 証拠の判定 (Gate W7.3B2.2)。
#
# job 2404249 は Series A の gpu_opt_pure_b1024 を OOM 扱いで停止させたが、
# 一致したのは src/proposed/host_pure.cu:115 が出力する助言的警告
#   "  > [Warn] BC_BATCH_OVERRIDE=1024 exceeds safe limit 512; may cause cudaMalloc OOM"
# の 1 行だけであった。runner exit=0、BC vector は 325557 要素で完全であり、
# 実際の確保は dynamic(GPU)=21.34 GB / free=101.4 GB で成功していた。
# 直接原因は `grep -Ei '\bOOM\b'` が「OOM という語の言及」に一致したことである。
#
# 本ファイルは source して使う。OOM は「確保が実際に失敗した証拠」に限定し、
# 語の言及・予告・閾値・設定名では成立させない。判定関数は診断を出さず、
# 状態を変数へ書いて 0/1 を返すのみで、exit は呼び出し側の規約に委ねる。

# 強い証拠クラス。列挙順が、1 行が複数クラスに一致した場合の優先順になる。
#   cuda_oom           : CUDA runtime/driver が確保失敗を報告した
#   host_alloc_failure : C++ の確保例外が送出された (std/rmm/thrust の bad_alloc)
#   kernel_oom_kill    : Linux OOM killer がプロセスを停止した
BCOOM_CLASSES=(cuda_oom host_alloc_failure kernel_oom_kill)

# クラスごとの ERE。grep は行単位で評価するため `.*` は行内のみに掛かる。
# いずれも「OOM」単独の語では成立しない (これが job 2404249 の誤判定要因)。
bcoom_pattern() {
    case "$1" in
        cuda_oom)
            # "CUDA error at f.cu:142: out of memory" (include/core/common.hpp)
            # "CUDA Error at f.cu:38 - out of memory"  (include/baseline/baseline_utils.h)
            printf '%s' 'CUDA[ _-]*error.*out of memory|cudaErrorMemoryAllocation|CUDA_ERROR_OUT_OF_MEMORY'
            ;;
        host_alloc_failure)
            printf '%s' '\bbad_alloc\b'
            ;;
        kernel_oom_kill)
            printf '%s' 'out of memory:[[:space:]]*killed process|killed process.*out of memory|\boom[-_]kill'
            ;;
        *)
            return 1
            ;;
    esac
}

# 全クラスの和集合 ERE。証拠行の特定に使う。
bcoom_any_pattern() {
    local class joined=""
    for class in "${BCOOM_CLASSES[@]}"; do
        [ -n "${joined}" ] && joined="${joined}|"
        joined="${joined}$(bcoom_pattern "${class}")"
    done
    printf '%s' "${joined}"
}

# 1 行を分類する。強い証拠でなければ none を出力して 1 を返す。
bcoom_classify_line() {
    local line="$1" class
    for class in "${BCOOM_CLASSES[@]}"; do
        if printf '%s\n' "${line}" | grep -qEi -- "$(bcoom_pattern "${class}")"; then
            printf '%s' "${class}"
            return 0
        fi
    done
    printf 'none'
    return 1
}

# 指定ファイル群を走査し、最初の強い証拠行を記録する。
# 設定する変数:
#   BCOOM_EVIDENCE_CLASS : 上記クラス、証拠なしなら none
#   BCOOM_MATCHED_FILE   : 一致ファイル、なければ not_applicable
#   BCOOM_LINE_NUMBER    : 一致行番号、なければ not_applicable
#   BCOOM_EXACT_LINE     : 完全な一致行 (無加工)、なければ not_applicable
# 証拠があれば 0、なければ 1 を返す。
bcoom_scan() {
    BCOOM_EVIDENCE_CLASS=none
    BCOOM_MATCHED_FILE=not_applicable
    BCOOM_LINE_NUMBER=not_applicable
    BCOOM_EXACT_LINE=not_applicable

    local any file hit
    any="$(bcoom_any_pattern)"
    for file in "$@"; do
        [ -f "${file}" ] || continue
        hit="$(grep -n -m1 -E -i -- "${any}" "${file}" 2>/dev/null)" || continue
        BCOOM_MATCHED_FILE="${file}"
        BCOOM_LINE_NUMBER="${hit%%:*}"
        BCOOM_EXACT_LINE="${hit#*:}"
        BCOOM_EVIDENCE_CLASS="$(bcoom_classify_line "${BCOOM_EXACT_LINE}")"
        return 0
    done
    return 1
}

# 判定順序: runner exit code → 強い OOM 証拠 → vector 存在 → vector 完全性 → status。
# vector_state は complete / invalid / missing / not_applicable (vector を採取しない構成)。
# 警告文だけでは決して失敗にしない。
bcoom_decide_status() {
    local rc="$1" evidence_class="$2" vector_state="$3"

    if [ "${rc}" -ne 0 ]; then
        if [ "${evidence_class}" != none ]; then
            printf 'OOM_CONFIRMED'
        else
            printf 'RUNTIME_FAILED'
        fi
        return 0
    fi

    # exit0 でも強い証拠があれば、runner が確保失敗を握り潰したことになる。
    if [ "${evidence_class}" != none ]; then
        printf 'RUNNER_SWALLOWED_OOM'
        return 0
    fi

    case "${vector_state}" in
        complete|not_applicable) printf 'SUCCESS' ;;
        *)                       printf 'VECTOR_INVALID' ;;
    esac
}

# runner レベル status の理由文字列。timeout/SIGKILL は RUNTIME_FAILED の内訳として残す。
bcoom_reason() {
    local rc="$1" status="$2" detail=""
    case "${status}" in
        OOM_CONFIRMED|RUNNER_SWALLOWED_OOM)
            printf 'oom_evidence=%s;matched=%s:%s;runner_exit=%s' \
                "${BCOOM_EVIDENCE_CLASS}" "${BCOOM_MATCHED_FILE}" "${BCOOM_LINE_NUMBER}" "${rc}"
            ;;
        RUNTIME_FAILED)
            [ "${rc}" -eq 124 ] && detail=';timeout_exit_124'
            [ "${rc}" -eq 137 ] && detail=';sigkill_exit_137'
            printf 'oom_evidence=none;runner_exit=%s%s' "${rc}" "${detail}"
            ;;
        *)
            printf 'not_applicable'
            ;;
    esac
}

# TSV 1 セルへ格納するための正規化 (tab/CR/LF のみ空白へ置換; 他は無加工)。
bcoom_tsv_safe() {
    printf '%s' "$1" | tr '\t\r\n' '   '
}
