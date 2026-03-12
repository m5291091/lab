#!/bin/bash
#PBS -N bc_ablation
#PBS -q regular-g
#PBS -l select=1:ncpus=72
#PBS -l walltime=02:00:00
#PBS -W group_list=gj17
#PBS -j oe

# ============================================================
#  アブレーションスタディ実験スクリプト
#
#  目的: 修士論文の「新手法2つ」の寄与を定量化
#   手法1: ReadMostly + Prefetch (GPU_Managed → GPU_ReadMostly)
#   手法2: 2-stream 非同期パイプライン (GPU_ReadMostly → GPU_Opt)
#
#  実行対象実装:
#   GPU          : ベースライン (cudaMalloc, HBM3 直接配置)
#   GPU_Managed  : Unified Memory (ReadMostly なし, LPDDR5X 配置)
#   GPU_ReadMostly: [手法1] ReadMostly + 適応型 Prefetch のみ
#   GPU_Opt      : [手法1 + 手法2] ReadMostly + 2-stream 非同期
#
#  グラフ: 7K / 11K / 56K / 325K ノード × 4 実装
#
#  使用方法:
#    qsub scripts/run_ablation.sh     # バッチジョブ
#    bash scripts/run_ablation.sh     # インタラクティブ (GPU ノード上)
# ============================================================

set -e

# ---- パス設定 ----
# PBS バッチジョブでは PBS_O_WORKDIR (qsub を実行したディレクトリ) を優先。
# インタラクティブ実行では $0 を基準に解決する。
SCRIPT_DIR="${PBS_O_WORKDIR:-$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)}"
BUILD_DIR="${SCRIPT_DIR}/build_miyabi"
RUNNER="${BUILD_DIR}/brandes_runner"
DATA_DIR="${SCRIPT_DIR}/../data"
RESULT_DIR="${BUILD_DIR}/result_ablation"
SUMMARY="${RESULT_DIR}/ablation_summary.tsv"

mkdir -p "${RESULT_DIR}"

echo "========================================"
echo "  アブレーションスタディ実験"
echo "  手法1: GPU_Managed → GPU_ReadMostly (ReadMostly + Prefetch)"
echo "  手法2: GPU_ReadMostly → GPU_Opt (2-stream 非同期)"
echo "  結果: ${SUMMARY}"
echo "========================================"

# TSV ヘッダー (analyze_all.py の load_tsv_skip_duplicate_headers が
# 先頭列 "Implementation" でヘッダー行を検出する)
printf "Implementation\tGraph\tTime_sec\tGTEPS\n" > "${SUMMARY}"

# ---- アブレーション実行関数 ----
run_impl() {
    local impl="$1"
    local graph="$2"
    local gname
    gname="$(basename "${graph}")"
    echo ""
    echo ">>> ${impl} on ${gname}"
    if ! "${RUNNER}" "${impl}" "${graph}" >> "${SUMMARY}" 2>> "${RESULT_DIR}/ablation_phase_timing.log"; then
        echo "    [WARNING] ${impl} failed on ${gname}" >&2
    fi
}

# ---- 評価グラフ (4グラフ) ----
GRAPHS=(
    "benchmark_7000_41459"
    "benchmark_11023_62184"
    "56438_300801"
    "325557_3216152"
)

IMPLS=("gpu" "gpu_managed" "gpu_readmostly" "gpu_opt")

for GNAME in "${GRAPHS[@]}"; do
    GPATH="${DATA_DIR}/${GNAME}"
    if [ ! -f "${GPATH}" ]; then
        echo "[SKIP] グラフファイルが見つかりません: ${GPATH}" >&2
        continue
    fi
    echo ""
    echo "========================================"
    echo "  グラフ: ${GNAME}"
    echo "========================================"
    for impl in "${IMPLS[@]}"; do
        run_impl "${impl}" "${GPATH}"
    done
done

echo ""
echo "========================================"
echo "  アブレーション完了"
echo "  結果: ${SUMMARY}"
echo "========================================"

# ---- 簡易サマリ表示 ----
echo ""
echo "--- GTEPS 一覧 ---"
awk -F'\t' 'NR>1 {printf "%-16s  %-30s  %s GTEPS\n", $1, $2, $4}' "${SUMMARY}"
