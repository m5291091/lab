cmake_minimum_required(VERSION 3.26)
project(ParallelBC LANGUAGES CUDA CXX C)

# HPC 環境では共有 CUDA ランタイムを使用 (Miyabi-G では LD_LIBRARY_PATH が自動設定される)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# --- cugraph バックエンド選択 ---
# USE_CUGRAPH_BC_MINI=ON (デフォルト): BC 専用ミニビルドを使用 (高速ビルド)
# USE_CUGRAPH_BC_MINI=OFF           : libcugraph フルビルドを使用
option(USE_CUGRAPH_BC_MINI "Use minimal BC-only cugraph build instead of full libcugraph" ON)

if(USE_CUGRAPH_BC_MINI)
    set(CUGRAPH_BC_MINI_BUILD_DIR "${PROJECT_SOURCE_DIR}/cugraph_bc_mini/build" CACHE PATH
        "Path to pre-built cugraph_bc_mini build directory")

    # CPM ソースキャッシュから依存ライブラリのヘッダーパスを自動検出
    set(_CPM_CACHE "${CUGRAPH_BC_MINI_BUILD_DIR}/_deps")

    # 各依存のソースディレクトリを取得
    # CPM_SOURCE_CACHE 経由 (<cache>/<pkg>/<hash>/) と、内蔵 _deps レイアウト
    # (<build>/_deps/<pkg>-src/) の両方をサポートする。
    macro(_find_cpm_src_dir varname pkg)
        set(${varname} "")
        # まず _deps/<pkg>-src を確認
        if(IS_DIRECTORY "${_CPM_CACHE}/${pkg}-src")
            set(${varname} "${_CPM_CACHE}/${pkg}-src")
        else()
            file(GLOB _candidates "${_CPM_CACHE}/${pkg}/????")
            foreach(_c ${_candidates})
                if(IS_DIRECTORY "${_c}")
                    set(${varname} "${_c}")
                    break()
                endif()
            endforeach()
        endif()
    endmacro()
    _find_cpm_src_dir(_CCCL_SRC    cccl)
    _find_cpm_src_dir(_RMM_SRC     rmm)
    _find_cpm_src_dir(_RAFT_SRC    raft)
    _find_cpm_src_dir(_CUCO_SRC    cuco)
    _find_cpm_src_dir(_SPDLOG_SRC  spdlog)
    _find_cpm_src_dir(_NVTX3_SRC   nvtx3)
    _find_cpm_src_dir(_RLOGGER_SRC rapids_logger)

    # IMPORTED static library として mini ビルドの成果物を参照
    add_library(cugraph_bc_mini STATIC IMPORTED)
    set_target_properties(cugraph_bc_mini PROPERTIES
        IMPORTED_LOCATION "${CUGRAPH_BC_MINI_BUILD_DIR}/libcugraph_bc_mini.a"
    )

    # ヘッダーパスを集約
    set(_MINI_INCLUDES
        "${PROJECT_SOURCE_DIR}/../../cugraph/cpp/include"      # cugraph API
        "${CUGRAPH_BC_MINI_BUILD_DIR}/include"                 # version_config.hpp
        "${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/raft-build/include"  # RAFT generated
        "${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/rmm-build/include"   # RMM generated
        "${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/cuco-build/include"  # cuco generated
    )
    # CPM キャッシュのソースヘッダー
    if(_RAFT_SRC)
        list(APPEND _MINI_INCLUDES "${_RAFT_SRC}/cpp/include")
    endif()
    if(_RMM_SRC)
        list(APPEND _MINI_INCLUDES "${_RMM_SRC}/cpp/include")
    endif()
    if(_CUCO_SRC)
        list(APPEND _MINI_INCLUDES "${_CUCO_SRC}/include")
    endif()
    if(_CCCL_SRC)
        list(APPEND _MINI_INCLUDES "${_CCCL_SRC}/thrust" "${_CCCL_SRC}/cub" "${_CCCL_SRC}/libcudacxx/include")
    endif()
    if(_SPDLOG_SRC)
        list(APPEND _MINI_INCLUDES "${_SPDLOG_SRC}/include")
    endif()
    if(_NVTX3_SRC)
        list(APPEND _MINI_INCLUDES "${_NVTX3_SRC}/c/include")
    endif()
    if(_RLOGGER_SRC)
        list(APPEND _MINI_INCLUDES "${_RLOGGER_SRC}/include")
    endif()

    set_target_properties(cugraph_bc_mini PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_MINI_INCLUDES}"
    )

    # リンク依存: RMM / rapids_logger (共有ライブラリ) + spdlog (静的)
    set_target_properties(cugraph_bc_mini PROPERTIES
        INTERFACE_LINK_LIBRARIES
            "${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/rmm-build/librmm.so;${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/rapids_logger-build/librapids_logger.so;${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/spdlog-build/libspdlog.a"
    )

    set(CUGRAPH_LINK_TARGET cugraph_bc_mini)
    message(STATUS "Using cugraph_bc_mini from: ${CUGRAPH_BC_MINI_BUILD_DIR}")
else()
    # libcugraph ビルドツリーへのパス (事前に build_cugraph.sh でビルド済みであること)
    set(CUGRAPH_BUILD_DIR "${PROJECT_SOURCE_DIR}/../../cugraph/cpp/build" CACHE PATH
        "Path to pre-built libcugraph build directory")
    list(APPEND CMAKE_PREFIX_PATH "${CUGRAPH_BUILD_DIR}")
    find_package(cugraph REQUIRED)
    set(CUGRAPH_LINK_TARGET cugraph::cugraph)
    message(STATUS "Using full libcugraph from: ${cugraph_DIR}")
endif()

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# コンパイルオプション
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wall,-fopenmp")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3 -Wall -fopenmp")

# ヘッダー検索パス
include_directories(${PROJECT_SOURCE_DIR})

# NVTX v3 (CUDA 12.x 以降はヘッダーオンリー、リンク不要)
find_path(NVTX3_INCLUDE_DIR
    NAMES nvtx3/nvToolsExt.h
    PATHS
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        /work/opt/local/aarch64/cores/nvidia/25.9/Linux_aarch64/25.9/cuda/include
    NO_DEFAULT_PATH
)
if(NVTX3_INCLUDE_DIR)
    message(STATUS "NVTX3 found: ${NVTX3_INCLUDE_DIR}")
    include_directories(${NVTX3_INCLUDE_DIR})
else()
    message(WARNING "NVTX3 not found: nvtxRange* calls may fail to compile")
endif()

# --- path-merging-bc (Galloit BC) ソース ---
set(PATHMERGE_BC_DIR "${PROJECT_SOURCE_DIR}/../../path-merging-bc")
set(PATHMERGE_BC_SOURCES
    ${PATHMERGE_BC_DIR}/src/galliot.cu
    ${PATHMERGE_BC_DIR}/src/kernels.cu
)

# --- メイン実行ファイル ---
add_executable(brandes_runner
    main.cpp
    Graph.cpp
    brandes_sequential.cpp
    brandes_omp.cpp
    brandes_gpu.cu
    brandes_gpu_opt.cu
    brandes_gpu_opt_pure.cu
    brandes_gpu_opt_pure_chunked.cu
    brandes_cugraph.cu
    brandes_pathmerge.cu
    ${PATHMERGE_BC_SOURCES}
)
set_target_properties(brandes_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

# path-merging-bc ヘッダー検索パス
target_include_directories(brandes_runner PRIVATE ${PATHMERGE_BC_DIR}/include)

target_link_libraries(brandes_runner PRIVATE
    OpenMP::OpenMP_CXX
    CUDA::cudart
    ${CUGRAPH_LINK_TARGET}
)

# --- 帯域計測ツール (NVLink-C2C / HBM3 実効帯域) ---
add_executable(bandwidth_benchmark
    tools/bandwidth_benchmark.cu
)
set_target_properties(bandwidth_benchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_link_libraries(bandwidth_benchmark PRIVATE CUDA::cudart)


# 1. テスト対象のグラフファイルリストを取得
set(GRAPH_DATA_DIR ${PROJECT_SOURCE_DIR}/../../data)
file(GLOB GRAPH_FILES
    "${GRAPH_DATA_DIR}/benchmark_7000_41459"
    "${GRAPH_DATA_DIR}/benchmark_11023_62184"
    "${GRAPH_DATA_DIR}/random"
)

# 2. 結果を保存するディレクトリとファイルを定義
set(RESULT_DIR ${CMAKE_CURRENT_BINARY_DIR}/result)
set(RESULT_FILE ${RESULT_DIR}/summary.tsv)

# 3. 'make run' ターゲットを定義
add_custom_target(run
    # resultディレクトリを作成
    COMMAND ${CMAKE_COMMAND} -E make_directory ${RESULT_DIR}
    # 結果ファイルにヘッダーを書き込む
    COMMAND ${CMAKE_COMMAND} -E echo "Implementation\tGraph\tTime_sec" > ${RESULT_FILE}
    COMMENT "Running all implementations on all benchmark graphs..."
    DEPENDS brandes_runner
)

# 4. 各グラフファイルに対する実行コマンドを 'run' ターゲットに追加
foreach(GRAPH_FILE ${GRAPH_FILES})
    add_custom_command(
        TARGET run
        POST_BUILD
        COMMAND $<TARGET_FILE:brandes_runner> all ${GRAPH_FILE} >> ${RESULT_FILE}
        COMMAND_EXPAND_LISTS
    )
endforeach()
