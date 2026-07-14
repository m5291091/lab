cmake_minimum_required(VERSION 3.18)
project(thesis_bc_project LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Set to 90 for GH200 if native is not sufficient, or keep native
set(CMAKE_CUDA_ARCHITECTURES 90)

# HPC 環境では共有 CUDA ランタイムを使用
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Enable OpenMP
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_compile_options(-O3 -g -Wall)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-O3>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>")
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>")

include_directories(include include/core include/proposed include/baseline)

# --- cugraph_bc_mini 統合 ---
set(CUGRAPH_BC_MINI_BUILD_DIR "${PROJECT_SOURCE_DIR}/cugraph_bc_mini/build" CACHE PATH
    "Path to pre-built cugraph_bc_mini build directory")

# CPM ソースキャッシュから依存ライブラリのヘッダーパスを自動検出
set(_CPM_CACHE "${CUGRAPH_BC_MINI_BUILD_DIR}/_deps")

macro(_find_cpm_src_dir varname pkg)
    set(${varname} "")
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
    "${PROJECT_SOURCE_DIR}/third_party/cugraph/cpp/include"   # cugraph API (vendored)
    "${CUGRAPH_BC_MINI_BUILD_DIR}/include"                 # version_config.hpp
    "${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/raft-build/include"  # RAFT generated
    "${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/rmm-build/include"   # RMM generated
    "${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/cuco-build/include"  # cuco generated
)
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
set_target_properties(cugraph_bc_mini PROPERTIES
    INTERFACE_LINK_LIBRARIES
        "${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/rmm-build/librmm.so;${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/rapids_logger-build/librapids_logger.so;${CUGRAPH_BC_MINI_BUILD_DIR}/_deps/spdlog-build/libspdlog.a"
)
# --- 終わり ---

# Core library
add_library(core src/core/graph.cpp src/core/runner.cpp)
target_link_libraries(core OpenMP::OpenMP_CXX)

# Proposed methods library
add_library(proposed 
    src/proposed/host_um.cu
    src/proposed/host_pure.cu
    src/proposed/host_chunked.cu
    src/proposed/host_ablation.cu
)
target_link_libraries(proposed core)

# Baseline library (cugraph 非依存 — pathmerge/ablation を独立ビルド可能にする)
add_library(baseline
    src/baseline/pathmerge.cu
    src/baseline/galliot.cu
    src/baseline/galliot_kernels.cu
    src/baseline/sequential.cpp
    src/baseline/omp.cpp
    src/baseline/gpu_unopt.cu
)
target_link_libraries(baseline core)

# cuGraph ベースライン (cugraph_bc_mini 依存を分離)
add_library(baseline_cugraph src/baseline/cugraph_bc.cu)
target_link_libraries(baseline_cugraph core cugraph_bc_mini)

# Executables
add_executable(run_ablation experiments/run_ablation.cu)
target_link_libraries(run_ablation proposed core)

add_executable(run_pathmerge_sweep experiments/run_pathmerge_sweep.cu)
target_link_libraries(run_pathmerge_sweep baseline core)

add_executable(run_benchmark experiments/run_benchmark.cu)
target_link_libraries(run_benchmark baseline baseline_cugraph proposed core cugraph_bc_mini)

add_executable(bandwidth_benchmark tools/bandwidth_benchmark.cu)
target_link_libraries(bandwidth_benchmark PRIVATE CUDA::cudart)
