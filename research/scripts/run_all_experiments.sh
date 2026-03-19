#!/bin/bash
set -euo pipefail

# run_all_experiments.sh
# 全実験を一括実行するスクリプト
#
# 実行手順:
#   1. メモリ帯域計測                                               (~30 min)
#   2. 全5実装 × 全グラフ の実行時間計測                            (~24 h)
#   3. 閾値感度実験 (--topo-threshold 0.001, 0.01, 0.1, 0.35)       (~1 h)
#   4. Nsight プロファイル (代表グラフ 3つ)                         (~2 h)
#   5. 全結果を data/ 以下に TSV 形式で保存
#   6. analyze_all.py を呼び出して図・表を生成
#
# 使い方:
#   bash scripts/run_all_experiments.sh [build_dir]
#   デフォルトの build_dir は build_miyabi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESEARCH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${RESEARCH_DIR}/../data"
SNAP_DIR="${DATA_DIR}/snap"

BUILD_DIR="${1:-${RESEARCH_DIR}/build_miyabi}"
RUNNER="${BUILD_DIR}/brandes_runner"
BANDWIDTH="${BUILD_DIR}/bandwidth_benchmark"

# ログとデータの保存先
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${RESEARCH_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_all_${TIMESTAMP}.log"

# 全出力をログファイルにも記録
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== run_all_experiments.sh start: ${TIMESTAMP} ==="
echo "build_dir : ${BUILD_DIR}"
echo "log_file  : ${LOG_FILE}"
echo ""

# ============================================================
# ヘルパー
# ============================================================
step_header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================"
}

run_or_warn() {
    # 失敗しても続行し、エラーメッセージを表示する
    "$@" || {
        echo "  [WARN] command failed (exit $?): $*"
        echo "  継続します..."
    }
}

check_file() {
    local f="$1"
    if [ ! -f "$f" ]; then
        echo "  スキップ (ファイルなし): $f"
        return 1
    fi
    return 0
}

# ============================================================
# Step 0: ビルド
# ============================================================
step_header "Step 0: ビルド"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [ ! -f "CMakeCache.txt" ]; then
    cmake "${RESEARCH_DIR}" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -10
fi
make -j8 2>&1 | tail -15

if [ ! -x "${RUNNER}" ]; then
    echo "[ERROR] brandes_runner が見つかりません: ${RUNNER}"
    exit 1
fi
echo "ビルド完了: ${RUNNER}"

cd "${RESEARCH_DIR}"

# ============================================================
# Step 1: メモリ帯域計測
# ============================================================
step_header "Step 1: メモリ帯域計測 (~30 min)"

mkdir -p data/bandwidth
BW_TSV="data/bandwidth/bandwidth_${TIMESTAMP}.tsv"

if [ -x "${BANDWIDTH}" ]; then
    echo "128 MB, 512 MB, 1 GB, 4 GB の各サイズで計測..."
    for SIZE_MB in 128 512 1024 4096; do
        run_or_warn "${BANDWIDTH}" "${SIZE_MB}" >> "${BW_TSV}" 2>&1
    done
    echo "保存先: ${BW_TSV}"
else
    echo "  [WARN] bandwidth_benchmark が見つかりません。スキップ。"
fi

# ============================================================
# Step 2: 全実装 × 全グラフ の実行時間計測
# ============================================================
step_header "Step 2: 全実装 × 全グラフ (~24 h)"

mkdir -p data/timing
TIMING_TSV="data/timing/timing_${TIMESTAMP}.tsv"
TIMING_LOG="data/timing/timing_${TIMESTAMP}.phase.log"
echo "Implementation	Graph	Time_sec	GTEPS" > "${TIMING_TSV}"
> "${TIMING_LOG}"

IMPLS_ALL=(sequential omp gpu gpu_stream gpu_managed gpu_readmostly gpu_opt)
IMPLS_GPU=(gpu gpu_stream gpu_managed gpu_readmostly gpu_opt)
IMPLS_FAST=(gpu gpu_stream gpu_readmostly gpu_opt)

SMALL_GRAPHS=(
    "${DATA_DIR}/benchmark_7000_41459"
    "${DATA_DIR}/benchmark_11023_62184"
)
MEDIUM_GRAPHS=(
    "${DATA_DIR}/benchmark_85830.data"
    "${DATA_DIR}/56438_300801"
    "${SNAP_DIR}/email-EuAll"
    "${SNAP_DIR}/amazon0302"
    "${SNAP_DIR}/web-Stanford"
    "${SNAP_DIR}/web-NotreDame"
    "${SNAP_DIR}/amazon0505"
)
LARGE_GRAPHS=(
    "${DATA_DIR}/325557_3216152"
    "${SNAP_DIR}/web-Google"
    "${SNAP_DIR}/roadNet-PA"
    "${SNAP_DIR}/roadNet-TX"
    "${SNAP_DIR}/roadNet-CA"
)

run_impl() {
    local impl="$1"
    local graph="$2"
    check_file "$graph" || return 0
    echo "  [${impl}] $(basename $graph)"
    run_or_warn "${RUNNER}" "${impl}" "${graph}" \
        >> "${TIMING_TSV}" \
        2>> "${TIMING_LOG}"
}

echo "--- small: 全実装 ---"
for G in "${SMALL_GRAPHS[@]}"; do
    check_file "$G" || continue
    echo "--- $(basename $G) ---"
    for IMPL in "${IMPLS_ALL[@]}"; do
        run_impl "${IMPL}" "${G}"
    done
done

echo ""
echo "--- medium: omp + GPU 5実装 ---"
for G in "${MEDIUM_GRAPHS[@]}"; do
    check_file "$G" || continue
    echo "--- $(basename $G) ---"
    for IMPL in omp "${IMPLS_GPU[@]}"; do
        run_impl "${IMPL}" "${G}"
    done
done

echo ""
echo "--- large: GPU 4実装 ---"
for G in "${LARGE_GRAPHS[@]}"; do
    check_file "$G" || continue
    echo "--- $(basename $G) ---"
    for IMPL in "${IMPLS_FAST[@]}"; do
        run_impl "${IMPL}" "${G}"
    done
done

echo "保存先: ${TIMING_TSV}"

# ============================================================
# Step 3: 閾値感度実験
# ============================================================
step_header "Step 3: 閾値感度実験 (~1 h)"

# 対象: web-Google (トポロジ 44 MB 程度)
THRESH_GRAPHS=(
    "${SNAP_DIR}/web-Google"
    "${SNAP_DIR}/web-NotreDame"
    "${DATA_DIR}/56438_300801"
)
THRESHOLDS=(0.001 0.01 0.1 0.35)

mkdir -p data/threshold
THRESH_TSV="data/threshold/threshold_${TIMESTAMP}.tsv"
THRESH_LOG="data/threshold/threshold_${TIMESTAMP}.phase.log"
echo "Implementation	Graph	Threshold	Time_sec	GTEPS" > "${THRESH_TSV}"
> "${THRESH_LOG}"

for G in "${THRESH_GRAPHS[@]}"; do
    check_file "$G" || continue
    echo "--- $(basename $G) ---"
    for T in "${THRESHOLDS[@]}"; do
        echo "  [gpu_readmostly --topo-threshold ${T}]"
        run_or_warn bash -c \
            "${RUNNER} gpu_readmostly ${G} --topo-threshold ${T} 2>>${THRESH_LOG} | \
             awk -v OFS='\t' -v t=${T} '{print \$1, \$2, t, \$3, \$4}' >> ${THRESH_TSV}"
        echo "  [gpu_opt --topo-threshold ${T}]"
        run_or_warn bash -c \
            "${RUNNER} gpu_opt ${G} --topo-threshold ${T} 2>>${THRESH_LOG} | \
             awk -v OFS='\t' -v t=${T} '{print \$1, \$2, t, \$3, \$4}' >> ${THRESH_TSV}"
    done
done

echo "保存先: ${THRESH_TSV}"

# ============================================================
# Step 4: Nsight プロファイル (代表グラフ 3つ)
# ============================================================
step_header "Step 4: Nsight プロファイル (~2 h)"

PROFILE_GRAPHS=(
    "${DATA_DIR}/56438_300801"
    "${SNAP_DIR}/web-NotreDame"
    "${SNAP_DIR}/roadNet-CA"
)
PROFILE_IMPLS=(gpu gpu_stream gpu_opt)

mkdir -p data/profile

if command -v nsys &>/dev/null; then
    for G in "${PROFILE_GRAPHS[@]}"; do
        check_file "$G" || continue
        GNAME="$(basename $G)"
        for IMPL in "${PROFILE_IMPLS[@]}"; do
            OUT="data/profile/${IMPL}_${GNAME}_${TIMESTAMP}"
            echo "  nsys: ${IMPL} on ${GNAME}"
            run_or_warn nsys profile \
                --output="${OUT}" \
                --trace=cuda,nvtx,osrt \
                --force-overwrite=true \
                "${RUNNER}" "${IMPL}" "${G}" \
                > /dev/null 2>> "${LOG_FILE}"
        done
    done
    echo "プロファイルデータ: data/profile/"
else
    echo "  [WARN] nsys が見つかりません。Nsight プロファイルをスキップ。"
fi

# ============================================================
# Step 5: analyze_all.py で図・表を生成
# ============================================================
step_header "Step 5: 結果の分析・可視化"

ANALYZE="${RESEARCH_DIR}/analysis/analyze_all.py"
if [ -f "${ANALYZE}" ]; then
    echo "analyze_all.py を実行..."
    run_or_warn python3 "${ANALYZE}" \
        --timing  "${TIMING_TSV}" \
        --threshold "${THRESH_TSV}" \
        --outdir "${RESEARCH_DIR}/analysis"
    echo "図・表: ${RESEARCH_DIR}/analysis/figures/, ${RESEARCH_DIR}/analysis/tables/"
else
    echo "  [WARN] analyze_all.py が見つかりません。スキップ。"
fi

# ============================================================
# 完了
# ============================================================
echo ""
echo "=== 全実験完了: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "ログ  : ${LOG_FILE}"
echo "帯域  : data/bandwidth/"
echo "実行時間: data/timing/"
echo "閾値感度: data/threshold/"
echo "プロファイル: data/profile/"
