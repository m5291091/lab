#!/bin/bash

# cugraph_bc_mini ビルドスクリプト (Miyabi-G 向け)
#
# libcugraph 全体の代わりに、BC 計算に必要な最小限のソースのみコンパイルする。
# ビルド時間: 30〜60 分 → 数分 (推定)
#
# 使い方:
#   bash scripts/build_cugraph_bc_mini.sh
#
# 環境変数:
#   CUGRAPH_BC_MINI_BUILD_DIR  mini 専用ビルドディレクトリ (BUILD_DIR より優先)
#   BUILD_DIR          ビルドディレクトリ (default: cugraph_bc_mini/build)
#                      ※ root project の BUILD_DIR が env 経由で流入すると
#                        binary directory が衝突するため、呼び出し側は
#                        CUGRAPH_BC_MINI_BUILD_DIR を明示すること。
#   JOBS               並列ジョブ数 (default: 8)
#   CLEAN_CACHE        1 にすると build dir を再作成 (default: 0)
#   CMAKE_GENERATOR    CMake generator (default: Ninja)
#   BUILD_TYPE         Release / Debug (default: Release)
#   AUTO_INSTALL_CMAKE 1 で pip による CMake 自動インストール (default: 0)
#   CMAKE_BIN          使用する cmake バイナリのパス (auto-detect)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${SCRIPT_DIR}/scripts/build_dir_guard.sh"
MINI_SRC_DIR="${SCRIPT_DIR}/cugraph_bc_mini"
BUILD_DIR="${CUGRAPH_BC_MINI_BUILD_DIR:-${BUILD_DIR:-${MINI_SRC_DIR}/build}}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-8}"
CLEAN_CACHE="${CLEAN_CACHE:-0}"
REQUIRED_CMAKE_VERSION="${REQUIRED_CMAKE_VERSION:-3.30.4}"
CMAKE_BIN="${CMAKE_BIN:-}"
AUTO_INSTALL_CMAKE="${AUTO_INSTALL_CMAKE:-0}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"

# --- CMake バージョン検出 (build_cugraph.sh と同一ロジック) ---
version_ge() {
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

cmake_version_of() {
    "$1" --version 2>/dev/null | awk 'NR==1 {print $3}'
}

if [ -n "${CMAKE_BIN}" ]; then
    if [ ! -x "${CMAKE_BIN}" ]; then
        echo "Error: CMAKE_BIN is not executable: ${CMAKE_BIN}"
        exit 1
    fi
    CMAKE_VERSION="$(cmake_version_of "${CMAKE_BIN}")"
    if ! version_ge "${CMAKE_VERSION}" "${REQUIRED_CMAKE_VERSION}"; then
        echo "Error: CMAKE_BIN (${CMAKE_BIN}) is version ${CMAKE_VERSION}, but requires >= ${REQUIRED_CMAKE_VERSION}"
        exit 1
    fi
else
    CMAKE_CANDIDATES=()
    [ -x "${HOME}/.local/bin/cmake" ] && CMAKE_CANDIDATES+=("${HOME}/.local/bin/cmake")
    if command -v cmake3 >/dev/null 2>&1; then
        CMAKE_CANDIDATES+=("$(command -v cmake3)")
    fi
    if command -v cmake >/dev/null 2>&1; then
        CMAKE_CANDIDATES+=("$(command -v cmake)")
    fi

    for candidate in "${CMAKE_CANDIDATES[@]}"; do
        candidate_ver="$(cmake_version_of "${candidate}")"
        if version_ge "${candidate_ver}" "${REQUIRED_CMAKE_VERSION}"; then
            CMAKE_BIN="${candidate}"
            CMAKE_VERSION="${candidate_ver}"
            break
        fi
    done

    if [ -z "${CMAKE_BIN}" ] && [ "${AUTO_INSTALL_CMAKE}" = "1" ]; then
        echo "[build_cugraph_bc_mini] No suitable CMake found. Installing via pip..."
        python3 -m pip install --user "cmake>=${REQUIRED_CMAKE_VERSION},<3.31"
        if [ -x "${HOME}/.local/bin/cmake" ]; then
            candidate_ver="$(cmake_version_of "${HOME}/.local/bin/cmake")"
            if version_ge "${candidate_ver}" "${REQUIRED_CMAKE_VERSION}"; then
                CMAKE_BIN="${HOME}/.local/bin/cmake"
                CMAKE_VERSION="${candidate_ver}"
            fi
        fi
    fi

    if [ -z "${CMAKE_BIN}" ]; then
        found_path="$(command -v cmake 2>/dev/null || true)"
        found_ver="unknown"
        if [ -n "${found_path}" ]; then
            found_ver="$(cmake_version_of "${found_path}")"
        fi
        echo "Error: Requires CMake >= ${REQUIRED_CMAKE_VERSION}, but found ${found_ver} (${found_path:-not found})."
        echo "Fix: AUTO_INSTALL_CMAKE=1 bash scripts/build_cugraph_bc_mini.sh"
        exit 1
    fi
fi

if [ -z "${CMAKE_VERSION:-}" ]; then
    CMAKE_VERSION="$(cmake_version_of "${CMAKE_BIN}")"
fi

# --- CPM ソースキャッシュ (既存ダウンロード済み deps を再利用) ---
# 既にフル cugraph ビルド済みの場合、CPM キャッシュを共有して再ダウンロードを回避
if [ -z "${CPM_SOURCE_CACHE:-}" ]; then
    EXISTING_BUILD="${SCRIPT_DIR}/third_party/cugraph/cpp/build"
    if [ -d "${EXISTING_BUILD}/_deps" ]; then
        export CPM_SOURCE_CACHE="${EXISTING_BUILD}/_deps"
        echo "[build_cugraph_bc_mini] Reusing CPM deps from full cugraph build: ${CPM_SOURCE_CACHE}"
    fi
fi

echo "========================================"
echo "  cugraph_bc_mini Build (BC-only)"
echo "  SRC  : ${MINI_SRC_DIR}"
echo "  BUILD: ${BUILD_DIR}"
echo "  TYPE : ${BUILD_TYPE}"
echo "  JOBS : ${JOBS}"
echo "  CMAKE: ${CMAKE_BIN} (${CMAKE_VERSION})"
echo "  GEN  : ${CMAKE_GENERATOR}"
echo "========================================"

if [ "${CLEAN_CACHE}" = "1" ]; then
    echo "[build_cugraph_bc_mini] Removing build directory"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

# root project の binary directory を誤って受け取っていないか確認する。
MINI_CACHE_HOME="$(bcguard_cache_home "${BUILD_DIR}")"
if [ -n "${MINI_CACHE_HOME}" ] && \
   [ "$(bcguard_canon "${MINI_CACHE_HOME}")" != "$(bcguard_canon "${MINI_SRC_DIR}")" ]; then
    echo "FOREIGN CMAKE CACHE: cugraph_bc_mini binary directory belongs to another source tree" >&2
    printf '  mini source=%s\n  mini binary=%s\n  cache CMAKE_HOME_DIRECTORY=%s\n' \
        "${MINI_SRC_DIR}" "${BUILD_DIR}" "${MINI_CACHE_HOME}" >&2
    printf '  checkpoint_sha=%s\n  pbs_job_id=%s\n' \
        "${EXPECTED_SHA:-$(git -C "${SCRIPT_DIR}" rev-parse HEAD 2>/dev/null || echo not_recorded)}" \
        "${PBS_JOBID:-not_pbs}" >&2
    echo "Fix: pass CUGRAPH_BC_MINI_BUILD_DIR, or use CLEAN_CACHE=1 to regenerate." >&2
    exit 1
fi

# NVHPC は GCC 由来の -Wno-error フラグをサポートしないため GCC を使用
CC_FOR_BUILD="${CC_FOR_CUGRAPH:-gcc}"
CXX_FOR_BUILD="${CXX_FOR_CUGRAPH:-g++}"

"${CMAKE_BIN}" -G "${CMAKE_GENERATOR}" \
    -S "${MINI_SRC_DIR}" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="${CC_FOR_BUILD}" \
    -DCMAKE_CXX_COMPILER="${CXX_FOR_BUILD}" \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DCUGRAPH_ROOT="${SCRIPT_DIR}/third_party/cugraph"

"${CMAKE_BIN}" --build "${BUILD_DIR}" -j"${JOBS}"

echo ""
echo "[build_cugraph_bc_mini] Done."
echo "  Build dir: ${BUILD_DIR}"
echo "  Next: bash scripts/build_miyabi_interactive.sh"
