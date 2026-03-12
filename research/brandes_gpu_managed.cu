#include "common.h"
#include "brandes.h"
#include "GraphManaged.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace cg = cooperative_groups;
using namespace std;

// ============================================================
//  GH200 ユニファイドメモリ最適化版 Brandes GPU 実装
//
//  設計方針 (BC_Miyabi_report.pdf セクション 3 に基づく):
//
//  [静的トポロジデータ] R (row_offsets), C (col_indices)
//    → cudaMallocManaged + SetPreferredLocation(CPU) + SetAccessedBy(GPU)
//    → NVLink-C2C (900 GB/s) 経由でキャッシュラインフェッチ
//    → GPU VRAM (96 GB) を超えるグラフも扱える
//
//  [動的ステートデータ] d, sigma, delta, Q_curr, Q_next, S, S_ends
//    → cudaMallocManaged + PrefetchAsync(GPU)
//    → HBM3 (4.02 TB/s) 上で高速なアトミック操作
// ============================================================

// --- find_shortest_paths / accumulate_dependencies は元の brandes_gpu.cu と同一 ---

__device__ bool isUndirected_managed = true;

__device__ void find_shortest_paths_managed(
        int *R, int *C, int *d_d, int *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, int batch_idx, int n_nodes, int &Q_curr_len,
        int &Q_next_len, int &S_len, int &S_ends_len, int &depth) {

    int tid = threadIdx.x;
    int bsize = blockDim.x;
    int v, w;

    while (true) {
        int threshold = min(max(n_nodes / 20, 32), 1024);

        if (Q_curr_len <= threshold) {
            for (int i = tid; i < Q_curr_len; i += bsize) {
                v = d_Q_curr[batch_idx * n_nodes + i];
                for (int j = R[v]; j < R[v+1]; j++) {
                    w = C[j];
                    if (atomicCAS(&d_d[batch_idx * n_nodes + w], -1, depth + 1) == -1) {
                        int pos = atomicAdd(&Q_next_len, 1);
                        d_Q_next[batch_idx * n_nodes + pos] = w;
                    }
                    if (d_d[batch_idx * n_nodes + w] == depth + 1) {
                        atomicAdd(&d_sigma[batch_idx * n_nodes + w], d_sigma[batch_idx * n_nodes + v]);
                    }
                }
            }
        } else {
            for (int i = tid; i < n_nodes; i += bsize) {
                w = i;
                if (d_d[batch_idx * n_nodes + w] == -1) {
                    int sum_sigma = 0;
                    for (int j = R[w]; j < R[w+1]; j++) {
                        v = C[j];
                        if (d_d[batch_idx * n_nodes + v] == depth)
                            sum_sigma += d_sigma[batch_idx * n_nodes + v];
                    }
                    if (sum_sigma > 0) {
                        if (atomicCAS(&d_d[batch_idx * n_nodes + w], -1, depth + 1) == -1) {
                            atomicAdd(&d_sigma[batch_idx * n_nodes + w], sum_sigma);
                            int pos = atomicAdd(&Q_next_len, 1);
                            d_Q_next[batch_idx * n_nodes + pos] = w;
                        } else if (d_d[batch_idx * n_nodes + w] == depth + 1) {
                            atomicAdd(&d_sigma[batch_idx * n_nodes + w], sum_sigma);
                        }
                    }
                }
            }
        }

        __syncthreads();

        if (Q_next_len == 0) {
            if (tid == 0) depth = d_d[batch_idx * n_nodes + d_S[batch_idx * n_nodes + S_len-1]];
            break;
        }

        int curr_Q_next_len = Q_next_len;
        for (int i = tid; i < curr_Q_next_len; i += bsize) {
            d_Q_curr[batch_idx * n_nodes + i] = d_Q_next[batch_idx * n_nodes + i];
            d_S[batch_idx * n_nodes + S_len + i] = d_Q_next[batch_idx * n_nodes + i];
        }

        __syncthreads();

        if (tid == 0) {
            d_S_ends[batch_idx * (n_nodes+1) + S_ends_len] = S_len + curr_Q_next_len;
            S_ends_len++;
            Q_curr_len = curr_Q_next_len;
            S_len += curr_Q_next_len;
            Q_next_len = 0;
            depth++;
        }

        __syncthreads();
    }
}

__device__ void accumulate_dependencies_managed(
        int *R, int *C, int *d_d, int *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, int batch_idx, int n_nodes, int &depth) {

    auto block = cg::this_thread_block();
    auto warp  = cg::tiled_partition<32>(block);
    int tid_in_block     = block.thread_rank();
    int warp_id          = tid_in_block / warp.size();
    int num_warps        = block.size() / warp.size();

    while (depth > 0) {
        int start = d_S_ends[batch_idx * (n_nodes+1) + depth];
        int end   = d_S_ends[batch_idx * (n_nodes+1) + depth + 1];
        int nodes_in_level = end - start;

        for (int i = warp_id; i < nodes_in_level; i += num_warps) {
            int w = d_S[batch_idx * n_nodes + start + i];
            double sigma_w    = d_sigma[batch_idx * n_nodes + w];
            double local_sum  = 0.0;

            for (int j = R[w] + warp.thread_rank(); j < R[w+1]; j += warp.size()) {
                int v = C[j];
                if (d_d[batch_idx * n_nodes + v] == d_d[batch_idx * n_nodes + w] + 1) {
                    local_sum += (sigma_w / (double)d_sigma[batch_idx * n_nodes + v])
                               * (1.0 + d_delta[batch_idx * n_nodes + v]);
                }
            }

            for (int offset = warp.size() / 2; offset > 0; offset /= 2)
                local_sum += warp.shfl_down(local_sum, offset);

            if (warp.thread_rank() == 0)
                d_delta[batch_idx * n_nodes + w] = local_sum;
        }

        block.sync();
        if (tid_in_block == 0) depth--;
        block.sync();
    }
}

// BFS フェーズカーネル (Unified Memory 版)
__global__ void brandes_bfs_kernel_managed(
        int *R, int *C, int n_nodes,
        int *d_d, int *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, double *d_delta, int *d_depth, int s_start) {
    int batch_idx = blockIdx.x;
    int s   = s_start + batch_idx;
    int tid = threadIdx.x;

    __shared__ int Q_curr_len, Q_next_len, S_len, S_ends_len, depth;

    if (tid == 0) {
        for (int v = 0; v < n_nodes; v++) {
            d_d    [batch_idx * n_nodes + v] = (v == s) ? 0  : -1;
            d_sigma[batch_idx * n_nodes + v] = (v == s) ? 1  :  0;
            d_delta[batch_idx * n_nodes + v] = 0.0;
        }
        d_Q_curr[batch_idx * n_nodes] = s;
        Q_curr_len = 1; Q_next_len = 0;
        d_S[batch_idx * n_nodes] = s;
        S_len = 1;
        d_S_ends[batch_idx * (n_nodes+1)]     = 0;
        d_S_ends[batch_idx * (n_nodes+1) + 1] = 1;
        S_ends_len = 2;
        depth = 0;
    }
    __syncthreads();

    find_shortest_paths_managed(R, C, d_d, d_sigma, d_Q_curr, d_Q_next,
                                d_S, d_S_ends, batch_idx, n_nodes,
                                Q_curr_len, Q_next_len, S_len, S_ends_len, depth);
    __syncthreads();

    if (tid == 0) d_depth[batch_idx] = depth;
}

// バックワードフェーズカーネル (Unified Memory 版)
__global__ void brandes_back_kernel_managed(
        int *R, int *C, double *CB, int n_nodes,
        int *d_d, int *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, const int *d_depth, int s_start) {
    int batch_idx = blockIdx.x;
    int s   = s_start + batch_idx;
    int tid = threadIdx.x;

    __shared__ int depth;
    if (tid == 0) depth = d_depth[batch_idx];
    __syncthreads();

    accumulate_dependencies_managed(R, C, d_d, d_sigma, d_delta,
                                    d_S, d_S_ends, batch_idx, n_nodes, depth);
    __syncthreads();

    for (int v = tid; v < n_nodes; v += blockDim.x) {
        if (v != s) {
            double contrib = isUndirected_managed
                           ? d_delta[batch_idx * n_nodes + v] / 2.0
                           : d_delta[batch_idx * n_nodes + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

__global__ void brandes_kernel_managed(
        int *R, int *C, double *CB, int n_nodes,
        int *d_d, int *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, double *d_delta, int s_start) {

    int batch_idx = blockIdx.x;
    int s   = s_start + batch_idx;
    int tid = threadIdx.x;

    __shared__ int Q_curr_len, Q_next_len, S_len, S_ends_len, depth;

    if (tid == 0) {
        for (int v = 0; v < n_nodes; v++) {
            d_d    [batch_idx * n_nodes + v] = (v == s) ? 0  : -1;
            d_sigma[batch_idx * n_nodes + v] = (v == s) ? 1  : 0;
            d_delta[batch_idx * n_nodes + v] = 0.0;
        }
        d_Q_curr[batch_idx * n_nodes] = s;
        Q_curr_len = 1;
        Q_next_len = 0;
        d_S[batch_idx * n_nodes] = s;
        S_len = 1;
        d_S_ends[batch_idx * (n_nodes+1)] = 0;
        d_S_ends[batch_idx * (n_nodes+1) + 1] = 1;
        S_ends_len = 2;
        depth = 0;
    }
    __syncthreads();

    find_shortest_paths_managed(R, C, d_d, d_sigma, d_Q_curr, d_Q_next,
                                d_S, d_S_ends, batch_idx, n_nodes,
                                Q_curr_len, Q_next_len, S_len, S_ends_len, depth);
    __syncthreads();

    accumulate_dependencies_managed(R, C, d_d, d_sigma, d_delta,
                                    d_S, d_S_ends, batch_idx, n_nodes, depth);
    __syncthreads();

    for (int v = tid; v < n_nodes; v += blockDim.x) {
        if (v != s) {
            double contrib = isUndirected_managed
                           ? d_delta[batch_idx * n_nodes + v] / 2.0
                           : d_delta[batch_idx * n_nodes + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

// ホスト側エントリポイント
// GraphManaged を使って入力グラフを受け取る
// (main.cpp では Graph を渡すが、ここでは内部で GraphManaged を使用する)
vector<double> brandes_gpu_managed_impl(
        int *R, int *C, double *CB_managed,
        int n_nodes, int edge_size, int gpu_id) {

    cudaDeviceProp prop;
    CUDA_ERR_CHK(cudaGetDeviceProperties(&prop, gpu_id));

    int threads_per_block = std::min(prop.maxThreadsPerBlock, n_nodes);
    threads_per_block = (threads_per_block / 32) * 32;
    threads_per_block = std::max(threads_per_block, 32);

    // 動的ステートデータ: GPU HBM3 に事前転送 (PrefetchAsync)
    int *d_d, *d_sigma, *d_Q_curr, *d_Q_next, *d_S, *d_S_ends;
    double *d_delta;

    size_t free_mem, total_mem;
    CUDA_ERR_CHK(cudaMemGetInfo(&free_mem, &total_mem));

    const size_t per_batch_mem =
        (size_t)n_nodes * (4 * sizeof(int) + sizeof(double)) +
        (size_t)n_nodes * sizeof(int) +
        (size_t)(n_nodes + 1) * sizeof(int);

    const size_t safety_margin = (size_t)(free_mem * 0.15);
    int BATCH_SIZE = (int)((free_mem - safety_margin) / per_batch_mem);
    BATCH_SIZE = std::max(1, std::min(BATCH_SIZE, 1024));

    // 使用予定メモリを報告 (96 GB HBM3 + NVLink-C2C 活用の証拠)
    size_t topology_bytes = ((size_t)(n_nodes + 1) + (size_t)edge_size) * sizeof(int);
    size_t dynamic_bytes  = (size_t)BATCH_SIZE * per_batch_mem;
    fprintf(stderr, "  > [Mem] GPU HBM3: total=%.1f GB, free_before=%.1f GB\n",
            total_mem / 1e9, free_mem / 1e9);
    fprintf(stderr, "  > [Mem] topology(CPU LPDDR5X)=%.2f GB, dynamic(HBM3)=%.2f GB, batch=%d\n",
            topology_bytes / 1e9, dynamic_bytes / 1e9, BATCH_SIZE);

    // 動的データを cudaMallocManaged + PrefetchAsync(GPU) で確保
    CUDA_ERR_CHK(cudaMallocManaged(&d_d,      (size_t)BATCH_SIZE * n_nodes * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&d_sigma,  (size_t)BATCH_SIZE * n_nodes * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&d_Q_curr, (size_t)BATCH_SIZE * n_nodes * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&d_Q_next, (size_t)BATCH_SIZE * n_nodes * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&d_S,      (size_t)BATCH_SIZE * n_nodes * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&d_S_ends, (size_t)BATCH_SIZE * (n_nodes+1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&d_delta,  (size_t)BATCH_SIZE * n_nodes * sizeof(double)));

    // フェーズ別時間計測用配列
    int *d_depth;
    CUDA_ERR_CHK(cudaMallocManaged(&d_depth, BATCH_SIZE * sizeof(int)));

    // 動的データを GPU HBM3 に事前転送
    nvtxRangePushA("Prefetch_dynamic_to_GPU");
    CUDA_ERR_CHK(cudaMemPrefetchAsync(d_d,      (size_t)BATCH_SIZE * n_nodes * sizeof(int),      gpu_id, 0));
    CUDA_ERR_CHK(cudaMemPrefetchAsync(d_sigma,  (size_t)BATCH_SIZE * n_nodes * sizeof(int),      gpu_id, 0));
    CUDA_ERR_CHK(cudaMemPrefetchAsync(d_Q_curr, (size_t)BATCH_SIZE * n_nodes * sizeof(int),      gpu_id, 0));
    CUDA_ERR_CHK(cudaMemPrefetchAsync(d_Q_next, (size_t)BATCH_SIZE * n_nodes * sizeof(int),      gpu_id, 0));
    CUDA_ERR_CHK(cudaMemPrefetchAsync(d_S,      (size_t)BATCH_SIZE * n_nodes * sizeof(int),      gpu_id, 0));
    CUDA_ERR_CHK(cudaMemPrefetchAsync(d_S_ends, (size_t)BATCH_SIZE * (n_nodes+1) * sizeof(int),  gpu_id, 0));
    CUDA_ERR_CHK(cudaMemPrefetchAsync(d_delta,  (size_t)BATCH_SIZE * n_nodes * sizeof(double),   gpu_id, 0));
    CUDA_ERR_CHK(cudaMemPrefetchAsync(d_depth,  BATCH_SIZE * sizeof(int),                        gpu_id, 0));

    // CB も GPU HBM3 に事前転送
    CUDA_ERR_CHK(cudaMemPrefetchAsync(CB_managed, n_nodes * sizeof(double), gpu_id, 0));
    CUDA_ERR_CHK(cudaDeviceSynchronize());
    nvtxRangePop(); // Prefetch_dynamic_to_GPU

    // フェーズ別時間計測用 CUDA イベント
    cudaEvent_t ev_bfs_s, ev_bfs_e, ev_back_e;
    CUDA_ERR_CHK(cudaEventCreate(&ev_bfs_s));
    CUDA_ERR_CHK(cudaEventCreate(&ev_bfs_e));
    CUDA_ERR_CHK(cudaEventCreate(&ev_back_e));
    float total_bfs_ms = 0.0f, total_back_ms = 0.0f;

    for (int s_start = 0; s_start < n_nodes; s_start += BATCH_SIZE) {
        int curr_batch = std::min(BATCH_SIZE, n_nodes - s_start);
        if (curr_batch <= 0) continue;

        CUDA_ERR_CHK(cudaEventRecord(ev_bfs_s));
        nvtxRangePushA("BFS_kernel_managed");
        brandes_bfs_kernel_managed<<<curr_batch, threads_per_block>>>(
                R, C, n_nodes,
                d_d, d_sigma, d_Q_curr, d_Q_next,
                d_S, d_S_ends, d_delta, d_depth, s_start);
        CUDA_ERR_CHK(cudaEventRecord(ev_bfs_e));
        nvtxRangePop(); // BFS_kernel_managed

        nvtxRangePushA("Backward_kernel_managed");
        brandes_back_kernel_managed<<<curr_batch, threads_per_block>>>(
                R, C, CB_managed, n_nodes,
                d_d, d_sigma, d_delta,
                d_S, d_S_ends, d_depth, s_start);
        CUDA_ERR_CHK(cudaEventRecord(ev_back_e));
        nvtxRangePop(); // Backward_kernel_managed

        CUDA_ERR_CHK(cudaEventSynchronize(ev_back_e));
        float b_ms = 0.0f, bk_ms = 0.0f;
        CUDA_ERR_CHK(cudaEventElapsedTime(&b_ms,  ev_bfs_s, ev_bfs_e));
        CUDA_ERR_CHK(cudaEventElapsedTime(&bk_ms, ev_bfs_e, ev_back_e));
        total_bfs_ms  += b_ms;
        total_back_ms += bk_ms;

        CUDA_ERR_CHK(cudaPeekAtLastError());
    }
    CUDA_ERR_CHK(cudaDeviceSynchronize());

    CUDA_ERR_CHK(cudaEventDestroy(ev_bfs_s));
    CUDA_ERR_CHK(cudaEventDestroy(ev_bfs_e));
    CUDA_ERR_CHK(cudaEventDestroy(ev_back_e));

    fprintf(stderr, "  > [GPU Phase] BFS: %.4f sec, Backward: %.4f sec\n",
            total_bfs_ms / 1000.0f, total_back_ms / 1000.0f);

    vector<double> result(CB_managed, CB_managed + n_nodes);

    CUDA_ERR_CHK(cudaFree(d_d));
    CUDA_ERR_CHK(cudaFree(d_sigma));
    CUDA_ERR_CHK(cudaFree(d_Q_curr));
    CUDA_ERR_CHK(cudaFree(d_Q_next));
    CUDA_ERR_CHK(cudaFree(d_S));
    CUDA_ERR_CHK(cudaFree(d_S_ends));
    CUDA_ERR_CHK(cudaFree(d_delta));
    CUDA_ERR_CHK(cudaFree(d_depth));

    return result;
}

// Graph (通常) を受け取るラッパー: main.cpp の共通インターフェースに合わせる
vector<double> brandes_gpu_managed(Graph &G) {
    int *R       = G.getAdjacencyListPointers();
    int *C       = G.getAdjacencyList();
    int n_nodes  = G.getNodeCount();
    int edge_size = 2 * G.getEdgeCount();

    int num_gpus;
    CUDA_ERR_CHK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        std::cerr << "No GPU found" << std::endl;
        exit(EXIT_FAILURE);
    }

    // CSR データを cudaMallocManaged に再アロケートし、ヒントを設定
    int *R_m, *C_m;
    CUDA_ERR_CHK(cudaMallocManaged(&R_m, (n_nodes + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&C_m, edge_size     * sizeof(int)));
    memcpy(R_m, R, (n_nodes + 1) * sizeof(int));
    memcpy(C_m, C, edge_size     * sizeof(int));

    // 静的データ: CPU LPDDR5X に固定
    CUDA_ERR_CHK(cudaMemAdvise(R_m, (n_nodes + 1) * sizeof(int),
                               cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_ERR_CHK(cudaMemAdvise(R_m, (n_nodes + 1) * sizeof(int),
                               cudaMemAdviseSetAccessedBy, 0 /* gpu_id */));
    CUDA_ERR_CHK(cudaMemAdvise(C_m, edge_size * sizeof(int),
                               cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_ERR_CHK(cudaMemAdvise(C_m, edge_size * sizeof(int),
                               cudaMemAdviseSetAccessedBy, 0 /* gpu_id */));

    // 結果バッファ (GPU HBM3 に配置)
    double *CB_managed;
    CUDA_ERR_CHK(cudaMallocManaged(&CB_managed, n_nodes * sizeof(double)));
    memset(CB_managed, 0, n_nodes * sizeof(double));

    vector<double> result = brandes_gpu_managed_impl(R_m, C_m, CB_managed, n_nodes, edge_size, 0);

    CUDA_ERR_CHK(cudaFree(R_m));
    CUDA_ERR_CHK(cudaFree(C_m));
    CUDA_ERR_CHK(cudaFree(CB_managed));

    return result;
}
