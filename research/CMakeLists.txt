cmake_minimum_required(VERSION 3.20)
project(ParallelBC LANGUAGES CUDA CXX)

# HPC 環境では共有 CUDA ランタイムを使用 (Miyabi-G では LD_LIBRARY_PATH が自動設定される)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)  # CUDA::cudart ターゲットを提供

# アーキテクチャと言語標準の設定
set(CMAKE_CUDA_ARCHITECTURES 90)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# コンパイルオプション
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Wall --extended-lambda -Xcompiler=-fopenmp")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3 -Wall -fopenmp")

# ヘッダー検索パス
include_directories(${PROJECT_SOURCE_DIR})

# NVTX v3 (CUDA 12.x 以降はヘッダーオンリー、リンク不要)
# CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES に nvtx3/ が含まれていない場合の手動パス
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

# --- メイン実行ファイル (フェーズ1 + フェーズ2) ---
add_executable(brandes_runner
    main.cpp
    Graph.cpp
    GraphManaged.cpp
    brandes_sequential.cpp
    brandes_omp.cpp
    brandes_gpu.cu
    brandes_gpu_managed.cu
    brandes_gpu_readmostly.cu
    brandes_gpu_opt.cu
)

# OpenMP をリンク (NVTX v3 はヘッダーオンリーのためリンク不要)
target_link_libraries(brandes_runner PRIVATE OpenMP::OpenMP_CXX)

# --- 帯域計測ツール (NVLink-C2C / HBM3 実効帯域) ---
add_executable(bandwidth_benchmark
    tools/bandwidth_benchmark.cu
)
target_link_libraries(bandwidth_benchmark PRIVATE)



# 1. テスト対象のグラフファイルリストを取得
set(GRAPH_DATA_DIR ${PROJECT_SOURCE_DIR}/../data)
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
