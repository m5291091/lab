#!/bin/bash
# BC 正確性検証スクリプト
# sequential を参照実装として全実装の BC 値を比較する
#
# 使用方法 (インタラクティブジョブ内):
#   ./scripts/verify_correctness.sh [graph_file]
#
# 引数省略時は benchmark_7000_41459 を使用

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_miyabi"
VERIFY_DIR="${BUILD_DIR}/result_verify"
RUNNER="${BUILD_DIR}/brandes_runner"

GRAPH="${1:-${SCRIPT_DIR}/../data/benchmark_7000_41459}"
GNAME="$(basename "$GRAPH")"

mkdir -p "${VERIFY_DIR}"

echo "========================================"
echo "  BC 正確性検証"
echo "  グラフ: ${GNAME}"
echo "========================================"

# ---- 参照 BC の生成 (sequential) ----
echo ""
echo ">>> [1/6] Sequential を参照として実行 (--dump-bc)"
REF="${VERIFY_DIR}/ref_sequential_${GNAME}.txt"
"${RUNNER}" sequential "${GRAPH}" --dump-bc > "${REF}" 2>/dev/null
echo "  -> 保存: ${REF}  ($(wc -l < "${REF}") 行)"

# ---- 各実装との比較 ----
PASS=0
FAIL=0

compare_impl() {
    local impl="$1"
    local label="$2"
    echo ""
    echo ">>> 比較: ${label}"
    local out="${VERIFY_DIR}/bc_${impl}_${GNAME}.txt"
    "${RUNNER}" "${impl}" "${GRAPH}" --dump-bc > "${out}" 2>/dev/null
    if python3 "${SCRIPT_DIR}/scripts/compare_bc.py" "${REF}" "${out}" --rel-tol 1e-6; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
}

compare_impl "omp"             "OpenMP"
compare_impl "gpu"             "GPU (cudaMalloc)"
compare_impl "gpu_managed"     "GPU_Managed (Unified Memory v1)"
compare_impl "gpu_readmostly"  "GPU_ReadMostly (ReadMostly + Prefetch, 手法1)"
compare_impl "gpu_opt"         "GPU_Opt (Unified Memory v2 + 4opt)"

echo ""
echo "========================================"
echo "  検証結果サマリ"
echo "  PASS: ${PASS}  FAIL: ${FAIL}"
echo "========================================"

if [ "${FAIL}" -gt 0 ]; then
    echo "ERROR: ${FAIL} 実装で不一致が検出されました" >&2
    exit 1
fi

echo "全実装の BC 値が sequential と一致しました (rel_tol=1e-6)"