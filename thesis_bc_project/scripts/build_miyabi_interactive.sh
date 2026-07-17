#!/bin/bash

# Miyabi-G のインタラクティブジョブ向け 統合ビルドスクリプト。
#
# 1 回の実行で run_benchmark の全機能をビルドする:
#   Stage 1: cugraph_bc_mini (BC ライブラリ, ソース変更時リビルド)
#   Stage 2: run_benchmark 本体
#
# 環境変数:
#   JOBS                       並列ジョブ数 (default: 8)
#   CLEAN_CACHE                1 で CMake キャッシュ再生成 (default: 0)
#   SKIP_CUGRAPH_MINI          1 で Stage 1 をスキップ (default: 0)
#   AUTO_INSTALL_CMAKE         1 で pip による CMake 自動インストール (default: 0)
#   BUILD_DIR                  root project の binary directory (default: build_miyabi)
#   CUGRAPH_BC_MINI_BUILD_DIR  mini の binary directory (default: cugraph_bc_mini/build)
#
# BUILD_DIR と CUGRAPH_BC_MINI_BUILD_DIR は必ず別 directory であること。
# 呼び出し側が env prefix (BUILD_DIR=... bash ...) で BUILD_DIR を渡すと、
# 本 script 内で再代入しても export 属性が残り child へ継承される。そのため
# Stage 1 へは mini 専用の値を明示的に渡す (job 2403658 の直接原因)。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${SCRIPT_DIR}/scripts/build_dir_guard.sh"
PROJECT_DIR="${SCRIPT_DIR}"
CUGRAPH_ROOT="${SCRIPT_DIR}/third_party/cugraph"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_miyabi}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-8}"
CLEAN_CACHE="${CLEAN_CACHE:-0}"

SKIP_CUGRAPH_MINI="${SKIP_CUGRAPH_MINI:-0}"

# ============================================================
# Stage 1: cugraph_bc_mini
# ============================================================
CUGRAPH_BC_MINI_SRC_DIR="${SCRIPT_DIR}/cugraph_bc_mini"
CUGRAPH_BC_MINI_BUILD_DIR="${CUGRAPH_BC_MINI_BUILD_DIR:-${CUGRAPH_BC_MINI_SRC_DIR}/build}"

echo "╔══════════════════════════════════════════╗"
echo "║  Miyabi-G 統合ビルド (全 BC 実装)        ║"
echo "╠══════════════════════════════════════════╣"
echo "║  PROJECT   : ${PROJECT_DIR}"
echo "║  BUILD     : ${BUILD_DIR}"
echo "║  MINI BUILD: ${CUGRAPH_BC_MINI_BUILD_DIR}"
echo "║  JOBS      : ${JOBS}"
echo "╚══════════════════════════════════════════╝"
echo ""

# configure 前に binary directory の分離と cache の由来を確認する。
bcguard_assert_separate \
    "${PROJECT_DIR}" "${BUILD_DIR}" \
    "${CUGRAPH_BC_MINI_SRC_DIR}" "${CUGRAPH_BC_MINI_BUILD_DIR}" \
    || exit 1

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY RUN: no configure, build, or GPU access"
    echo "  root build directory: ${BUILD_DIR}"
    echo "  mini build directory: ${CUGRAPH_BC_MINI_BUILD_DIR}"
    exit 0
fi

build_cugraph_mini() {
    if [ "${SKIP_CUGRAPH_MINI}" = "1" ]; then
        echo "[Stage 1] SKIP (SKIP_CUGRAPH_MINI=1)"
        return
    fi
    local lib="${CUGRAPH_BC_MINI_BUILD_DIR}/libcugraph_bc_mini.a"
    local src="${CUGRAPH_ROOT}/cpp/src/centrality/betweenness_centrality_impl.cuh"
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
    # root の BUILD_DIR が env 経由で流入しても mini 側が使わないよう明示的に上書きする。
    BUILD_DIR="${CUGRAPH_BC_MINI_BUILD_DIR}" \
        CUGRAPH_BC_MINI_BUILD_DIR="${CUGRAPH_BC_MINI_BUILD_DIR}" \
        AUTO_INSTALL_CMAKE="${AUTO_INSTALL_CMAKE:-0}" \
        bash "${SCRIPT_DIR}/scripts/build_cugraph_bc_mini.sh"
    if [ ! -f "${lib}" ]; then
        echo "[Stage 1] FAILED: expected library not produced: ${lib}" >&2
        exit 1
    fi
    echo "[Stage 1] cugraph_bc_mini: DONE"
}

# ============================================================
# Stage 2: run_benchmark 本体
# ============================================================
build_run_benchmark() {
    echo "[Stage 2] Building run_benchmark..."
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

    if ! cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" "${CMAKE_ARGS[@]}"; then
        echo "[Stage 2] FAILED: configure failed; refusing to build or reuse any existing binary" >&2
        bcguard_report_context "${PROJECT_DIR}" "${BUILD_DIR}" \
            "${CUGRAPH_BC_MINI_SRC_DIR}" "${CUGRAPH_BC_MINI_BUILD_DIR}"
        exit 1
    fi
    if ! cmake --build "${BUILD_DIR}" -j"${JOBS}"; then
        echo "[Stage 2] FAILED: build failed; refusing to reuse any existing binary" >&2
        exit 1
    fi
    echo "[Stage 2] run_benchmark: DONE (${BUILD_DIR}/run_benchmark)"
}

# ============================================================
# メイン実行
# ============================================================
build_cugraph_mini
build_run_benchmark

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Build Complete!                         ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Runners:"
echo "║    ${BUILD_DIR}/run_benchmark        (全実装ベンチ/正確性検証)"
echo "║    ${BUILD_DIR}/run_ablation         (アブレーション実験)"
echo "║    ${BUILD_DIR}/run_pathmerge_sweep  (PathMerge バッチ探索)"
echo "║    ${BUILD_DIR}/bandwidth_benchmark  (帯域計測)"
echo "╠══════════════════════════════════════════╣"
echo "║  Next: qsub scripts/run_benchmark_small.sh (PBS) または直接実行"
echo "╚══════════════════════════════════════════╝"
