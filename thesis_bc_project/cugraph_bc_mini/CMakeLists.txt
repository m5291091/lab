# =============================================================================
# cugraph_bc_mini — BC 専用の最小 cugraph ビルド
#
# libcugraph 全体 (150+ .cu, cuVS, RAFT compiled) をビルドする代わりに、
# Betweenness Centrality に必要なソースのみコンパイルして静的ライブラリを生成する。
# ビルド時間: 30〜60 分 → 数分 (推定)
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

# --- rapids-cmake ブートストラップ ---
# cugraph ツリーの rapids_config.cmake を使ってバージョンと rapids-cmake を取得
set(CUGRAPH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cugraph" CACHE PATH
    "Path to vendored cugraph source root (contains VERSION, RAPIDS_BRANCH, cmake/)")
if(NOT EXISTS "${CUGRAPH_ROOT}/VERSION")
    message(FATAL_ERROR "CUGRAPH_ROOT does not point to a valid cugraph tree: ${CUGRAPH_ROOT}")
endif()

# rapids "main" 追従だと rmm/raft/cuco が破壊的に変わって cugraph public headers と
# 不整合になる (例: 2026-04 時点で rmm main から owning_wrapper.hpp が消失)。
# rapids-cmake 自体と downstream パッケージ (rmm/raft/cuco) を v26.04.00 に固定する
# (cugraph 26.06.00 ヘッダと互換)。BEFORE include(rapids_config.cmake): こうしないと
# rapids_config.cmake が RAPIDS_BRANCH を読んで "main" にしてしまう。
set(rapids-cmake-version "26.04" CACHE STRING "rapids-cmake version pin" FORCE)
set(rapids-cmake-branch "release/26.04" CACHE STRING "rapids-cmake branch pin" FORCE)
set(rapids-cmake-checkout-tag "v26.04.00" CACHE STRING "rapids-cmake checkout-tag pin" FORCE)

include("${CUGRAPH_ROOT}/cmake/rapids_config.cmake")
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(CUGRAPH_BC_MINI)

project(CUGRAPH_BC_MINI VERSION "${RAPIDS_VERSION}" LANGUAGES C CXX CUDA)

# Miyabi-G (GH200) 向け: sm_90 のみに限定
# rapids_cuda_init_architectures は sm_90a (arch-specific) を含むが、
# sm_90a の atom.global.gpu.* 命令が GH200 で cudaErrorInvalidAddressSpace を
# 引き起こす場合があるため、sm_90 (generic) を使用する。
# 環境変数 CUGRAPH_BC_MINI_CUDA_ARCHITECTURES で上書き可能。
if(DEFINED ENV{CUGRAPH_BC_MINI_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUGRAPH_BC_MINI_CUDA_ARCHITECTURES}")
else()
    set(CMAKE_CUDA_ARCHITECTURES "90")
endif()

set(CUGRAPH_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CUGRAPH_VERSION_MINOR ${PROJECT_VERSION_MINOR})

# ビルドタイプ
rapids_cmake_build_type(Release)

# CUDA runtime — brandes_runner が Shared を使うので合わせる
rapids_cuda_init_runtime(USE_STATIC OFF)
rapids_find_package(CUDAToolkit REQUIRED
    BUILD_EXPORT_SET    cugraph-bc-mini-exports
    INSTALL_EXPORT_SET  cugraph-bc-mini-exports)

# --- 依存関係の取得 (header-only, cuVS なし) ---
rapids_cpm_init()

# CCCL (header-only)
include(${rapids-cmake-dir}/cpm/cccl.cmake)
rapids_cpm_cccl(BUILD_EXPORT_SET cugraph-bc-mini-exports)

# RMM (header-only)
include(${rapids-cmake-dir}/cpm/rmm.cmake)
rapids_cpm_rmm(BUILD_EXPORT_SET cugraph-bc-mini-exports)

# cuCollections (header-only)
include(${rapids-cmake-dir}/cpm/cuco.cmake)
rapids_cpm_cuco(BUILD_EXPORT_SET cugraph-bc-mini-exports)

# RAFT — ヘッダオンリーモード (コンパイル済みライブラリは不要)
set(CUGRAPH_MIN_VERSION_raft "${CUGRAPH_VERSION_MAJOR}.${CUGRAPH_VERSION_MINOR}.00")
rapids_cpm_find(raft ${CUGRAPH_MIN_VERSION_raft}
    GLOBAL_TARGETS      raft::raft
    BUILD_EXPORT_SET    cugraph-bc-mini-exports
    INSTALL_EXPORT_SET  cugraph-bc-mini-exports
    CPM_ARGS
        EXCLUDE_FROM_ALL TRUE
        GIT_REPOSITORY https://github.com/rapidsai/raft.git
        GIT_TAG        ${rapids-cmake-checkout-tag}
        SOURCE_SUBDIR  cpp
        OPTIONS
            "RAFT_COMPILE_LIBRARY OFF"
            "BUILD_TESTS OFF"
            "BUILD_PRIMS_BENCH OFF"
            "BUILD_CAGRA_HNSWLIB OFF"
)

# cuVS は取得しない (BC に不要)

# --- cugraph ソースパス ---
set(CUGRAPH_CPP "${CUGRAPH_ROOT}/cpp")

# --- コンパイルフラグ ---
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUGRAPH_BC_MINI_CUDA_FLAGS "")
# CMake の CMAKE_CUDA_FLAGS_RELEASE はデフォルトで空またはプラットフォーム依存のため
# Release ビルドでも -O3 が保証されない。明示的に指定する。
list(APPEND CUGRAPH_BC_MINI_CUDA_FLAGS -O3 -DNDEBUG)
list(APPEND CUGRAPH_BC_MINI_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUGRAPH_BC_MINI_CUDA_FLAGS -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS -Xptxas=--disable-warnings)
list(APPEND CUGRAPH_BC_MINI_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable,-Wno-error=deprecated-declarations)

set(CUGRAPH_BC_MINI_CXX_FLAGS "")
# C++ ホストコードも明示的に -O3 を指定
list(APPEND CUGRAPH_BC_MINI_CXX_FLAGS -O3 -DNDEBUG)
if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUGRAPH_BC_MINI_CXX_FLAGS -Wno-error=deprecated-declarations -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS)
endif()

# --- BC 専用ソースファイル (SG, v32_e32 のみ) ---
set(CUGRAPH_BC_MINI_SOURCES
    # BC アルゴリズム
    ${CUGRAPH_CPP}/src/centrality/betweenness_centrality_sg_v32_e32.cu

    # グラフ構造
    ${CUGRAPH_CPP}/src/structure/create_graph_from_edgelist_sg_v32_e32.cu
    ${CUGRAPH_CPP}/src/structure/graph_sg_v32_e32.cu
    ${CUGRAPH_CPP}/src/structure/graph_view_sg_v32_e32.cu
    ${CUGRAPH_CPP}/src/structure/renumber_edgelist_sg_v32_e32.cu
    ${CUGRAPH_CPP}/src/structure/renumber_utils_sg_v32_e32.cu
    ${CUGRAPH_CPP}/src/structure/symmetrize_edgelist_sg_v32_e32.cu

    # ユーティリティ
    ${CUGRAPH_CPP}/src/detail/utility_wrappers_32.cu
    ${CUGRAPH_CPP}/src/detail/utility_wrappers_64.cu
    ${CUGRAPH_CPP}/src/utilities/shuffle_properties_mg.cu
    ${CUGRAPH_CPP}/src/utilities/validation_checks.cu
    ${CUGRAPH_CPP}/src/utilities/invert_flags.cu
    ${CUGRAPH_CPP}/src/detail/permute_range_v32.cu
)

# --- 静的ライブラリ ---
add_library(cugraph_bc_mini STATIC ${CUGRAPH_BC_MINI_SOURCES})
add_library(cugraph_bc_mini::cugraph_bc_mini ALIAS cugraph_bc_mini)

set_target_properties(cugraph_bc_mini PROPERTIES
    CXX_STANDARD            20
    CXX_STANDARD_REQUIRED   ON
    CUDA_STANDARD           20
    CUDA_STANDARD_REQUIRED  ON
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cugraph_bc_mini
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_BC_MINI_CXX_FLAGS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_BC_MINI_CUDA_FLAGS}>"
)

target_compile_definitions(cugraph_bc_mini PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)

target_include_directories(cugraph_bc_mini
    PRIVATE
        "${CUGRAPH_CPP}/../thirdparty"
        "${CUGRAPH_CPP}/src"
    PUBLIC
        "$<BUILD_INTERFACE:${CUGRAPH_CPP}/include>"
)

target_link_libraries(cugraph_bc_mini
    PUBLIC
        rmm::rmm
        raft::raft
        $<BUILD_LOCAL_INTERFACE:CUDA::toolkit>
    PRIVATE
        cuco::cuco
)

# --- version_config.hpp の生成 ---
rapids_cmake_write_version_file("${CMAKE_CURRENT_BINARY_DIR}/include/cugraph/version_config.hpp")

target_include_directories(cugraph_bc_mini
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)

# --- CMake config の生成 (find_package 用) ---
include(CMakePackageConfigHelpers)

set(CUGRAPH_BC_MINI_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}")

file(WRITE "${CUGRAPH_BC_MINI_CONFIG_DIR}/cugraph_bc_mini-config.cmake" [[
include(CMakeFindDependencyMacro)
if(NOT TARGET cugraph_bc_mini::cugraph_bc_mini)
    include("${CMAKE_CURRENT_LIST_DIR}/cugraph_bc_mini-targets.cmake")
endif()
]])

export(TARGETS cugraph_bc_mini
    NAMESPACE cugraph_bc_mini::
    FILE "${CUGRAPH_BC_MINI_CONFIG_DIR}/cugraph_bc_mini-targets.cmake"
)

message(STATUS "cugraph_bc_mini: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "  Sources: ${CUGRAPH_BC_MINI_SOURCES}")
message(STATUS "  cugraph headers: ${CUGRAPH_CPP}/include")
