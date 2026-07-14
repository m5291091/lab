#!/bin/bash

# Miyabi-G のインタラクティブジョブ向け 統合ビルドスクリプト。
#
# 1 回の実行で brandes_runner の全機能をビルドする:
#   Stage 1: cugraph_bc_mini (BC ライブラリ, ソース変更時リビルド)
#   Stage 2: brandes_runner 本体
#
# 環境変数:
#   JOBS                   並列ジョブ数 (default: 8)
#   CLEAN_CACHE            1 で CMake キャッシュ再生成 (default: 0)
#   SKIP_CUGRAPH_MINI      1 で Stage 1 をスキップ (default: 0)
#   AUTO_INSTALL_CMAKE     1 で pip による CMake 自動インストール (default: 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_miyabi}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-8}"
CLEAN_CACHE="${CLEAN_CACHE:-0}"

SKIP_CUGRAPH_MINI="${SKIP_CUGRAPH_MINI:-0}"

echo "╔══════════════════════════════════════════╗"
echo "║  Miyabi-G 統合ビルド (全 BC 実装)        ║"
echo "╠══════════════════════════════════════════╣"
echo "║  LAB    : ${LAB_DIR}"
echo "║  BUILD  : ${BUILD_DIR}"
echo "║  JOBS   : ${JOBS}"
echo "╚══════════════════════════════════════════╝"
echo ""

# ============================================================
# Stage 1: cugraph_bc_mini
# ============================================================
CUGRAPH_BC_MINI_BUILD_DIR="${SCRIPT_DIR}/cugraph_bc_mini/build"

build_cugraph_mini() {
    if [ "${SKIP_CUGRAPH_MINI}" = "1" ]; then
        echo "[Stage 1] SKIP (SKIP_CUGRAPH_MINI=1)"
        return
    fi
    local lib="${CUGRAPH_BC_MINI_BUILD_DIR}/libcugraph_bc_mini.a"
    local src="${LAB_DIR}/cugraph/cpp/src/centrality/betweenness_centrality_impl.cuh"
    # ソースがライブラリより新しい場合はリビルド
    if [ -f "${lib}" ] && [ ! "${src}" -nt "${lib}" ]; then
        echo "[Stage 1] cugraph_bc_mini: OK (up-to-date)"
        return
    fi
    if [ -f "${lib}" ] && [ "${src}" -nt "${lib}" ]; then
        echo "[Stage 1] cugraph_bc_mini: SOURCE CHANGED → rebuilding..."
    else
        echo "[Stage 1] Building cugraph_bc_mini..."
    fi
    AUTO_INSTALL_CMAKE="${AUTO_INSTALL_CMAKE:-0}" \
        bash "${SCRIPT_DIR}/scripts/build_cugraph_bc_mini.sh"
    echo "[Stage 1] cugraph_bc_mini: DONE"
}

# ============================================================
# Stage 2: brandes_runner 本体
# ============================================================
build_brandes_runner() {
    echo "[Stage 2] Building brandes_runner..."
    mkdir -p "${BUILD_DIR}"

    if [ "${CLEAN_CACHE}" = "1" ]; then
        echo "[Stage 2] Removing CMake cache"
        rm -f "${BUILD_DIR}/CMakeCache.txt"
    fi

    CMAKE_ARGS=()
    CMAKE_ARGS+=(-DCMAKE_BUILD_TYPE="${BUILD_TYPE}")
    CMAKE_ARGS+=(-DCMAKE_C_COMPILER="${CC_FOR_CUGRAPH:-gcc}")
    CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="${CXX_FOR_CUGRAPH:-g++}")
    CMAKE_ARGS+=(-DUSE_CUGRAPH_BC_MINI=ON)
    CMAKE_ARGS+=(-DCUGRAPH_BC_MINI_BUILD_DIR="${CUGRAPH_BC_MINI_BUILD_DIR}")

    cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"
    cmake --build "${BUILD_DIR}" -j"${JOBS}"
    echo "[Stage 2] brandes_runner: DONE (${BUILD_DIR}/brandes_runner)"
}

# ============================================================
# メイン実行
# ============================================================
build_cugraph_mini
build_brandes_runner

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Build Complete!                         ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Runner : ${BUILD_DIR}/brandes_runner"
echo "╠══════════════════════════════════════════╣"
echo "║  Next: qsub scripts/run_benchmark_full.sh  (PBS) または bash で実行"
echo "╚══════════════════════════════════════════╝"
