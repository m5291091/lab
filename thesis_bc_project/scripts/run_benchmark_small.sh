#!/bin/bash -l
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=2:00:00
#PBS -N bc_bench_small
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
# Miyabi-G 全手法ベンチマーク
#
# 一回のジョブで全手法 × 対象グラフを計測し、結果を自動集約する。
#
# 手法 → 対象グラフ:
#   sequential   : benchmark_7000_41459, benchmark_11023_62184, random
#   omp          : benchmark_7000_41459, benchmark_11023_62184, random, 56438_300801
#   cugraph_bc   : benchmark_7000_41459, benchmark_11023_62184, random, 56438_300801
#   gpu_opt      : 上記 + 325557_3216152 + snap 全9種
#   gpu_opt_pure : 上記 + 325557_3216152 + snap 全9種
#   pathmerge_bc : 上記 + 325557_3216152 + snap 全9種
#
# 使用方法:
#   cd thesis_bc_project
#   qsub scripts/run_benchmark_full.sh
#
# 環境変数:
#   DRY_RUN=1    コマンド表示のみ (実行しない)
#   SKIP_BUILD=1 ビルドをスキップ
#   NUM_TRIALS=N 試行回数 (デフォルト: 1, 短い実行は3推奨)
#   TIMEOUT_SEC  1回あたりの最大実行時間 (デフォルト: 18000 = 5時間)
# ============================================================

set -uo pipefail

# --- ディレクトリ設定 ---
# PBS はジョブスクリプトをスプール領域にコピーして実行するため、
# BASH_SOURCE[0] はオリジナルのパスではない。PBS_O_WORKDIR を優先する。
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
SNAP_DIR="${DATA_DIR}/snap"
DRY_RUN="${DRY_RUN:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"
NUM_TRIALS_OVERRIDE="${NUM_TRIALS:-}"
TIMEOUT_SEC="${TIMEOUT_SEC:-18000}"

# --- 結果ディレクトリ ---
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
JOB_SUFFIX="${PBS_JOBID:-$$}"; JOB_SUFFIX="${JOB_SUFFIX%%.*}"
RESULT_DIR="${BUILD_DIR}/result_benchmark_${TIMESTAMP}_${JOB_SUFFIX}"
mkdir -p "${RESULT_DIR}"

TSV_FILE="${RESULT_DIR}/results.tsv"
LOG_FILE="${RESULT_DIR}/benchmark.log"
PHASE_LOG="${RESULT_DIR}/phase_timing.log"

# --- ヘッダ出力 ---
log() {
    echo "$@" | tee -a "${LOG_FILE}"
}

log "╔══════════════════════════════════════════════════════════════╗"
log "║  Miyabi-G 全手法ベンチマーク                                ║"
log "╠══════════════════════════════════════════════════════════════╣"
log "║  開始時刻 : $(date '+%Y-%m-%d %H:%M:%S')"
log "║  Runner   : ${RUNNER}"
log "║  Data     : ${DATA_DIR}"
log "║  結果     : ${RESULT_DIR}"
log "║  DryRun   : ${DRY_RUN}"
log "║  Trials   : (Dynamic per graph/method)"
log "║  Timeout  : ${TIMEOUT_SEC}s"
log "╚══════════════════════════════════════════════════════════════╝"
log ""

# ============================================================
# Step 0: ビルド (SKIP_BUILD=1 でスキップ可能)
# ============================================================
if [ "${SKIP_BUILD}" != "1" ]; then
    log "=== [Step 0] ビルド ==="
    bash "${SCRIPT_DIR}/scripts/build_miyabi_interactive.sh" 2>&1 | tee -a "${LOG_FILE}"
    log ""
fi

if [ ! -x "${RUNNER}" ] && [ "${DRY_RUN}" != "1" ]; then
    log "ERROR: runner が見つかりません: ${RUNNER}"
    exit 2
fi

# ============================================================
# グラフ定義
# ============================================================

# 基本グラフ (小規模)
GRAPHS_SEQ=(
    "${DATA_DIR}/benchmark_7000_41459"
    "${DATA_DIR}/benchmark_11023_62184"
    "${DATA_DIR}/random"
)
GRAPHS_ALL=(
    "${GRAPHS_SEQ[@]}"
    "${DATA_DIR}/56438_300801"
)
# ============================================================
# 手法 → グラフ マッピング
# ============================================================

# 手法の実行順序 (高速な手法を先に実行)
METHODS=(sequential omp cugraph_bc gpu_opt gpu_opt_pure gpu_opt_pure_chunked pathmerge_bc)

get_graphs_for_method() {
    local method="$1"
    case "${method}" in
        sequential) echo "${GRAPHS_SEQ[@]}" ;;
        *)          echo "${GRAPHS_ALL[@]}" ;;
    esac
}

get_num_trials() {
    local method="$1"
    if [ -n "${NUM_TRIALS_OVERRIDE}" ]; then
        echo "${NUM_TRIALS_OVERRIDE}"
        return
    fi
    echo 10 # 全手法10試行 (小規模グラフは高速なため)
}


# ============================================================
# 実行ヘルパー
# ============================================================

TOTAL_RUNS=0
PASS_RUNS=0
FAIL_RUNS=0
SKIP_RUNS=0
START_EPOCH=$(date +%s)

# TSV ヘッダ (graph_meta 追加)
echo -e "Implementation\tGraph\tNodes\tEdges\tTrial\tTime_sec\tGTEPS" > "${TSV_FILE}"

# 正確性検証用 (max BC 値記録)
MAXBC_FILE="${RESULT_DIR}/max_bc.tsv"
echo -e "Implementation\tGraph\tMaxBC_Index\tMaxBC_Value" > "${MAXBC_FILE}"

# グラフメタデータキャッシュ
declare -A GRAPH_META_NODES
declare -A GRAPH_META_EDGES

get_graph_meta() {
    local graph_path="$1"
    if [ -z "${GRAPH_META_NODES[$graph_path]+x}" ]; then
        local first_line
        first_line="$(head -1 "${graph_path}" 2>/dev/null)"
        GRAPH_META_NODES[$graph_path]="$(echo "${first_line}" | awk '{print $1}')"
        GRAPH_META_EDGES[$graph_path]="$(echo "${first_line}" | awk '{print $2}')"
    fi
}

run_one() {
    local method="$1"
    local graph_path="$2"
    local graph_name
    graph_name="$(basename "${graph_path}")"

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    if [ ! -f "${graph_path}" ]; then
        log "  [SKIP] ${method} × ${graph_name} (グラフなし)"
        SKIP_RUNS=$((SKIP_RUNS + 1))
        return 0
    fi

    get_graph_meta "${graph_path}"
    local nodes="${GRAPH_META_NODES[$graph_path]}"
    local edges="${GRAPH_META_EDGES[$graph_path]}"

    local elapsed_wall
    elapsed_wall=$(( $(date +%s) - START_EPOCH ))
    local hh=$(( elapsed_wall / 3600 ))
    local mm=$(( (elapsed_wall % 3600) / 60 ))

    local trial
    local num_trials
    num_trials=$(get_num_trials "${method}")
    for trial in $(seq 1 "${num_trials}"); do
        local trial_label=""
        if [ "${num_trials}" -gt 1 ]; then
            trial_label=" [trial ${trial}/${num_trials}]"
        fi

        log "  [RUN] ${method} × ${graph_name}${trial_label}  (経過: ${hh}h${mm}m)"

        if [ "${DRY_RUN}" = "1" ]; then
            log "    >>> ${RUNNER} ${method} ${graph_path}"
            continue
        fi

        local tmp_stdout="${RESULT_DIR}/.tmp_stdout"
        local tmp_stderr="${RESULT_DIR}/.tmp_stderr"
        local rc=0

        timeout "${TIMEOUT_SEC}" "${RUNNER}" "${method}" "${graph_path}" \
            > "${tmp_stdout}" \
            2> "${tmp_stderr}" \
            || rc=$?

        # stderr → ログ + フェーズ計測ログ
        cat "${tmp_stderr}" >> "${LOG_FILE}"
        cat "${tmp_stderr}" >> "${PHASE_LOG}"

        if [ ${rc} -eq 124 ]; then
            log "  [TIMEOUT] ${method} × ${graph_name} (>${TIMEOUT_SEC}s)"
            echo -e "${method}\t${graph_name}\t${nodes}\t${edges}\t${trial}\tTIMEOUT\t0" >> "${TSV_FILE}"
            FAIL_RUNS=$((FAIL_RUNS + 1))
        elif [ ${rc} -ne 0 ]; then
            log "  [FAIL] ${method} × ${graph_name} (exit=${rc})"
            echo -e "${method}\t${graph_name}\t${nodes}\t${edges}\t${trial}\tFAIL\t0" >> "${TSV_FILE}"
            FAIL_RUNS=$((FAIL_RUNS + 1))
        else
            # stdout → TSV (元: Impl\tGraph\tTime\tGTEPS → 拡張形式に変換)
            local orig_line
            orig_line="$(cat "${tmp_stdout}")"
            local time_val gteps_val
            time_val="$(echo "${orig_line}" | awk -F'\t' '{print $3}')"
            gteps_val="$(echo "${orig_line}" | awk -F'\t' '{print $4}')"
            local impl_name
            impl_name="$(echo "${orig_line}" | awk -F'\t' '{print $1}')"
            echo -e "${impl_name}\t${graph_name}\t${nodes}\t${edges}\t${trial}\t${time_val}\t${gteps_val}" >> "${TSV_FILE}"
            log "    → ${time_val} sec (GTEPS=${gteps_val})"
            PASS_RUNS=$((PASS_RUNS + 1))

            # 正確性検証: max BC 値を抽出
            local max_bc_line
            max_bc_line="$(grep 'Maximum Betweenness Centrality' "${tmp_stderr}" 2>/dev/null || true)"
            if [ -n "${max_bc_line}" ]; then
                local max_idx max_val
                max_idx="$(echo "${max_bc_line}" | sed 's/.*index : \([0-9]*\).*/\1/')"
                max_val="$(echo "${max_bc_line}" | sed 's/.*==> \(.*\)/\1/')"
                echo -e "${impl_name}\t${graph_name}\t${max_idx}\t${max_val}" >> "${MAXBC_FILE}"
            fi
        fi

        rm -f "${tmp_stdout}" "${tmp_stderr}"
    done
}

# ============================================================
# メイン実行ループ
# ============================================================

for method in "${METHODS[@]}"; do
    log ""
    log "=========================================="
    log "  手法: ${method}"
    log "=========================================="

    # get_graphs_for_method の結果を配列に格納
    graphs_str="$(get_graphs_for_method "${method}")"
    read -ra graphs <<< "${graphs_str}"

    for graph in "${graphs[@]}"; do
        run_one "${method}" "${graph}"
    done
done

# ============================================================
# サマリ出力
# ============================================================

END_EPOCH=$(date +%s)
TOTAL_WALL=$(( END_EPOCH - START_EPOCH ))
TOTAL_H=$(( TOTAL_WALL / 3600 ))
TOTAL_M=$(( (TOTAL_WALL % 3600) / 60 ))
TOTAL_S=$(( TOTAL_WALL % 60 ))

log ""
log "╔══════════════════════════════════════════════════════════════╗"
log "║  ベンチマーク完了                                           ║"
log "╠══════════════════════════════════════════════════════════════╣"
log "║  終了時刻  : $(date '+%Y-%m-%d %H:%M:%S')"
log "║  総実行時間: ${TOTAL_H}h ${TOTAL_M}m ${TOTAL_S}s"
log "║  実行数    : ${TOTAL_RUNS} (成功=${PASS_RUNS}, 失敗=${FAIL_RUNS}, スキップ=${SKIP_RUNS})"
log "║  結果 TSV  : ${TSV_FILE}"
log "║  正確性    : ${MAXBC_FILE}"
log "║  ログ      : ${LOG_FILE}"
log "║  フェーズ  : ${PHASE_LOG}"
log "╚══════════════════════════════════════════════════════════════╝"

# --- 正確性検証: 同一グラフでの max BC 比較 ---
log ""
log "=== 正確性検証 (max BC 値の一致) ==="
if [ -f "${MAXBC_FILE}" ]; then
    # グラフごとに max BC 値を比較
    awk -F'\t' 'NR>1 {
        graph=$2; impl=$1; val=$4;
        if (!(graph in first_val)) {
            first_val[graph] = val;
            first_impl[graph] = impl;
        }
        vals[graph] = vals[graph] impl "=" val "  ";
    }
    END {
        for (g in vals) {
            print "  " g ": " vals[g];
        }
    }' "${MAXBC_FILE}" | tee -a "${LOG_FILE}"
fi

# --- 結果テーブル表示 ---
log ""
log "=== 結果一覧 (TSV) ==="
cat "${TSV_FILE}" | tee -a "${LOG_FILE}"

# --- 自動サマリ生成 ---
SUMMARIZER="${SCRIPT_DIR}/scripts/summarize_benchmark.py"
if [ -f "${SUMMARIZER}" ] && command -v python3 &>/dev/null; then
    log ""
    log "=== 自動サマリ生成 ==="
    python3 "${SUMMARIZER}" "${TSV_FILE}" "${RESULT_DIR}" 2>&1 | tee -a "${LOG_FILE}"
else
    log ""
    log "手動サマリ生成:"
    log "  python3 scripts/summarize_benchmark.py ${TSV_FILE} ${RESULT_DIR}"
fi
