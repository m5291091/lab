#ifndef BRANDES_KERNELS_CUH
#define BRANDES_KERNELS_CUH

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "common.hpp"   // K_SOURCES_PER_BLOCK, THREADS_PER_SOURCE

namespace cg = cooperative_groups;

__device__ inline size_t batch_node_offset(int batch_idx, int n_nodes)
{
    return (size_t)batch_idx * (size_t)n_nodes;
}

__device__ inline size_t batch_level_offset(int batch_idx, int max_depth_estimate)
{
    return (size_t)batch_idx * (size_t)(max_depth_estimate + 1);
}

// ============================================================
//  デバイス関数: BFS 前向き探索 (トップダウン/ボトムアップ ハイブリッド)
// ============================================================
template <bool USE_HYBRID_BFS>
__device__ void find_shortest_paths_opt(
        int *R, int *C, int *d_d, double *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, int batch_idx, int n_nodes,
        int max_depth_estimate, int *d_overflow,
        int &Q_curr_len, int &Q_next_len, int &S_len, int &S_ends_len, int &depth)
{
    int tid   = threadIdx.x;
    int bsize = blockDim.x;
    int v, w;
    const size_t base = batch_node_offset(batch_idx, n_nodes);
    const size_t ends_base = batch_level_offset(batch_idx, max_depth_estimate);

    __shared__ int direction;        // 0 = TOP_DOWN, 1 = BOTTOM_UP
    __shared__ int m_f, m_u;
    __shared__ int m_f_next;
    const int alpha = 14, beta = 24; // Beamer 推奨値

    // 初期状態の設定
    if (tid == 0) {
        direction = 0;
        int s = d_S[base];
        m_f = R[s + 1] - R[s];
        m_u = R[n_nodes] - m_f;
    }
    __syncthreads();

    while (true) {
        // 各レベル開始時に方向選択
        if (tid == 0) {
            if constexpr (USE_HYBRID_BFS) {
                if (direction == 0 && m_f > m_u / alpha)        direction = 1;
                else if (direction == 1 && Q_curr_len < n_nodes / beta) direction = 0;
            } else {
                direction = 0;
            }
            m_f_next = 0;
        }
        __syncthreads();

        if (direction == 0) {
            for (int i = tid; i < Q_curr_len; i += bsize) {
                v = d_Q_curr[base + i];
                for (int j = R[v]; j < R[v + 1]; j++) {
                    w = C[j];
                    if (atomicCAS(&d_d[base + w], -1, depth + 1) == -1) {
                        int pos = atomicAdd(&Q_next_len, 1);
                        d_Q_next[base + pos] = w;
                    }
                    if (d_d[base + w] == depth + 1) {
                        atomicAdd(&d_sigma[base + w],
                                  d_sigma[base + v]);
                    }
                }
            }
        } else {
            for (int i = tid; i < n_nodes; i += bsize) {
                w = i;
                if (d_d[base + w] == -1) {
                    double sum_sigma = 0.0;
                    for (int j = R[w]; j < R[w + 1]; j++) {
                        v = C[j];
                        if (d_d[base + v] == depth)
                            sum_sigma += d_sigma[base + v];
                    }
                    if (sum_sigma > 0.0) {
                        if (atomicCAS(&d_d[base + w], -1, depth + 1) == -1) {
                            atomicAdd(&d_sigma[base + w], sum_sigma);
                            int pos = atomicAdd(&Q_next_len, 1);
                            d_Q_next[base + pos] = w;
                        } else if (d_d[base + w] == depth + 1) {
                            atomicAdd(&d_sigma[base + w], sum_sigma);
                        }
                    }
                }
            }
        }

        __syncthreads();

        if (Q_next_len == 0) {
            if (tid == 0)
                depth = d_d[base + d_S[base + S_len - 1]];
            break;
        }

        int curr_Q_next_len = Q_next_len;

        // m_f の計算: warp 内 shfl_down で集約
        int local_m_f = 0;
        for (int i = tid; i < curr_Q_next_len; i += bsize) {
            int u = d_Q_next[base + i];
            local_m_f += R[u + 1] - R[u];
        }
        for (int offset = 16; offset > 0; offset /= 2) {
            local_m_f += __shfl_down_sync(0xffffffff, local_m_f, offset);
        }
        if (tid % 32 == 0) {
            atomicAdd(&m_f_next, local_m_f);
        }
        __syncthreads();

        for (int i = tid; i < curr_Q_next_len; i += bsize) {
            d_Q_curr[base + i]    = d_Q_next[base + i];
            d_S[base + S_len + i] = d_Q_next[base + i];
        }
        __syncthreads();

        if (tid == 0) {
            if (S_ends_len >= max_depth_estimate + 1) {
                atomicExch(d_overflow, 1);
                Q_next_len = 0;
            } else {
                d_S_ends[ends_base + S_ends_len] = S_len + curr_Q_next_len;
                S_ends_len++;
            }
            Q_curr_len = curr_Q_next_len;
            S_len     += curr_Q_next_len;
            Q_next_len = 0;
            depth++;

            m_f = m_f_next;
            m_u -= m_f; // 既訪頂点のエッジ数を累積で減算
        }
        __syncthreads();
    }
}

// ============================================================
//  デバイス関数: BC 後向き依存集計
//  Warp レベルシャッフル還元により各頂点の依存値を効率的に集計。
// ============================================================
__device__ inline void accumulate_dependencies_opt(
        int *R, int *C, int *d_d, double *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, int batch_idx, int n_nodes, int max_depth_estimate, int &depth)
{
    auto block     = cg::this_thread_block();
    auto warp      = cg::tiled_partition<32>(block);
    int  tid_block = block.thread_rank();
    int  warp_id   = tid_block / warp.size();
    int  num_warps = block.size() / warp.size();
    const size_t base = batch_node_offset(batch_idx, n_nodes);
    const size_t ends_base = batch_level_offset(batch_idx, max_depth_estimate);

    while (depth > 0) {
        int start          = d_S_ends[ends_base + depth];
        int end            = d_S_ends[ends_base + depth + 1];
        int nodes_in_level = end - start;

        for (int i = warp_id; i < nodes_in_level; i += num_warps) {
            int    w         = d_S[base + start + i];
            double sigma_w   = d_sigma[base + w];
            double local_sum = 0.0;

            for (int j = R[w] + warp.thread_rank(); j < R[w + 1]; j += warp.size()) {
                int v = C[j];
                if (d_d[base + v] == d_d[base + w] + 1) {
                    local_sum += (sigma_w / d_sigma[base + v])
                                 * (1.0 + d_delta[base + v]);
                }
            }

            // Warp シャッフル還元
            for (int offset = warp.size() / 2; offset > 0; offset /= 2)
                local_sum += warp.shfl_down(local_sum, offset);

            if (warp.thread_rank() == 0)
                d_delta[base + w] = local_sum;
        }

        block.sync();
        if (tid_block == 0) depth--;
        block.sync();
    }
}

// thread-per-vertex バックワード (低密度グラフ向け: 平均次数 < 8)
__device__ inline void accumulate_dependencies_tpv_opt(
        int *R, int *C, int *d_d, double *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, int batch_idx, int n_nodes, int max_depth_estimate, int &depth)
{
    int tid_block = threadIdx.x;
    int bsize     = blockDim.x;
    const size_t base = batch_node_offset(batch_idx, n_nodes);
    const size_t ends_base = batch_level_offset(batch_idx, max_depth_estimate);

    while (depth > 0) {
        int start          = d_S_ends[ends_base + depth];
        int end            = d_S_ends[ends_base + depth + 1];
        int nodes_in_level = end - start;

        for (int i = tid_block; i < nodes_in_level; i += bsize) {
            int    w       = d_S[base + start + i];
            double sigma_w = d_sigma[base + w];
            double sum     = 0.0;
            int    dw      = d_d[base + w];
            for (int j = R[w]; j < R[w + 1]; j++) {
                int v = C[j];
                if (d_d[base + v] == dw + 1) {
                    sum += (sigma_w / d_sigma[base + v])
                         * (1.0 + d_delta[base + v]);
                }
            }
            d_delta[base + w] = sum;
        }
        __syncthreads();
        if (tid_block == 0) depth--;
        __syncthreads();
    }
}

// 到達済み頂点のみリセット
inline __global__ void reset_visited_batch_kernel(
        int *d_d, double *d_sigma, double *d_delta,
        const int *d_S, int n_nodes,
        const int *d_depth, const int *d_S_ends, int max_depth_estimate)
{
    int batch_idx = blockIdx.x;
    const size_t base = batch_node_offset(batch_idx, n_nodes);
    const size_t ends_base = batch_level_offset(batch_idx, max_depth_estimate);
    __shared__ int reachable_count;
    if (threadIdx.x == 0) {
        int depth = d_depth[batch_idx];
        reachable_count = d_S_ends[ends_base + depth + 1];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < reachable_count; i += blockDim.x) {
        int v = d_S[base + i];
        d_d   [base + v] = -1;
        d_sigma[base + v] = 0.0;
        d_delta[base + v] = 0.0;
    }
}

// BFS フェーズカーネル
template <bool USE_HYBRID_BFS>
__global__ void brandes_bfs_kernel_opt(
        int *R, int *C, int n_nodes,
        int max_depth_estimate, int *d_overflow,
        int *d_d, double *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, int *d_depth, int s_start)
{
    int batch_idx = blockIdx.x;
    int s         = s_start + batch_idx;
    int tid       = threadIdx.x;
    const size_t base = batch_node_offset(batch_idx, n_nodes);
    const size_t ends_base = batch_level_offset(batch_idx, max_depth_estimate);

    __shared__ int Q_curr_len, Q_next_len, S_len, S_ends_len, depth;

    if (tid == 0) {
        d_d   [base + s] = 0;
        d_sigma[base + s] = 1.0;
        d_Q_curr[base]   = s;
        Q_curr_len  = 1;
        Q_next_len  = 0;
        d_S[base]        = s;
        S_len       = 1;
        d_S_ends[ends_base]     = 0;
        d_S_ends[ends_base + 1] = 1;
        S_ends_len  = 2;
        depth       = 0;
    }
    __syncthreads();

    find_shortest_paths_opt<USE_HYBRID_BFS>(R, C, d_d, d_sigma, d_Q_curr, d_Q_next,
                            d_S, d_S_ends, batch_idx, n_nodes,
                            max_depth_estimate, d_overflow,
                            Q_curr_len, Q_next_len, S_len, S_ends_len, depth);
    __syncthreads();

    if (tid == 0) d_depth[batch_idx] = depth;
}

// バックワードフェーズカーネル
template <bool IS_UNDIRECTED>
__global__ void brandes_back_kernel_opt(
        int *R, int *C, double *CB, int n_nodes, int max_depth_estimate,
        int *d_d, double *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, const int *d_depth, int s_start)
{
    int batch_idx = blockIdx.x;
    int s         = s_start + batch_idx;
    int tid       = threadIdx.x;
    const size_t base = batch_node_offset(batch_idx, n_nodes);

    __shared__ int depth;
    if (tid == 0) depth = d_depth[batch_idx];
    __syncthreads();

    accumulate_dependencies_opt(R, C, d_d, d_sigma, d_delta,
                                d_S, d_S_ends, batch_idx, n_nodes, max_depth_estimate, depth);
    __syncthreads();

    for (int v = tid; v < n_nodes; v += blockDim.x) {
        if (v != s) {
            double contrib = IS_UNDIRECTED
                           ? d_delta[base + v] / 2.0
                           : d_delta[base + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

template <bool IS_UNDIRECTED>
__global__ void brandes_back_kernel_tpv_opt(
        int *R, int *C, double *CB, int n_nodes, int max_depth_estimate,
        int *d_d, double *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, const int *d_depth, int s_start)
{
    int batch_idx = blockIdx.x;
    int s         = s_start + batch_idx;
    int tid       = threadIdx.x;
    const size_t base = batch_node_offset(batch_idx, n_nodes);

    __shared__ int depth;
    if (tid == 0) depth = d_depth[batch_idx];
    __syncthreads();

    accumulate_dependencies_tpv_opt(R, C, d_d, d_sigma, d_delta,
                                    d_S, d_S_ends, batch_idx, n_nodes, max_depth_estimate, depth);
    __syncthreads();

    for (int v = tid; v < n_nodes; v += blockDim.x) {
        if (v != s) {
            double contrib = IS_UNDIRECTED
                           ? d_delta[base + v] / 2.0
                           : d_delta[base + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

template <bool USE_HYBRID_BFS>
__global__ void brandes_bfs_kernel_shared_frontier(
        int *R, int *C, int n_nodes,
        int max_depth_estimate, int *d_overflow,
        int *d_d, double *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, int *d_depth, int s_start, int curr_batch)
{
    int tid   = threadIdx.x;
    int k_idx = tid / THREADS_PER_SOURCE;
    int lane  = tid % THREADS_PER_SOURCE;

    int batch_idx = blockIdx.x * K_SOURCES_PER_BLOCK + k_idx;
    bool active   = (batch_idx < curr_batch);

    int s = active ? (s_start + batch_idx) : -1;
    size_t b_offset    = (size_t)batch_idx * n_nodes;
    size_t ends_offset = (size_t)batch_idx * (max_depth_estimate + 1);

    __shared__ int Q_curr_len[K_SOURCES_PER_BLOCK];
    __shared__ int Q_next_len[K_SOURCES_PER_BLOCK];
    __shared__ int S_len[K_SOURCES_PER_BLOCK];
    __shared__ int S_ends_len[K_SOURCES_PER_BLOCK];
    __shared__ int depth[K_SOURCES_PER_BLOCK];
    __shared__ int local_done[K_SOURCES_PER_BLOCK];

    if (lane == 0) {
        if (active) {
            d_d[b_offset + s]     = 0;
            d_sigma[b_offset + s] = 1.0;
            d_Q_curr[b_offset]    = s;
            Q_curr_len[k_idx] = 1;
            Q_next_len[k_idx] = 0;
            d_S[b_offset]         = s;
            S_len[k_idx]      = 1;
            d_S_ends[ends_offset]     = 0;
            d_S_ends[ends_offset + 1] = 1;
            S_ends_len[k_idx] = 2;
            depth[k_idx]      = 0;
            local_done[k_idx] = 0;
        } else {
            local_done[k_idx] = 1;
        }
    }
    __syncthreads();

    while (true) {
        __shared__ int all_done;
        if (tid == 0) all_done = 1;
        __syncthreads();

        if (lane == 0 && local_done[k_idx] == 0)
            all_done = 0;
        __syncthreads();

        if (all_done) break;

        // 1. Top-Down 展開
        if (local_done[k_idx] == 0) {
            for (int i = lane; i < Q_curr_len[k_idx]; i += THREADS_PER_SOURCE) {
                int v = d_Q_curr[b_offset + i];
                for (int j = R[v]; j < R[v + 1]; j++) {
                    int w = C[j];
                    if (atomicCAS(&d_d[b_offset + w], -1, depth[k_idx] + 1) == -1) {
                        int pos = atomicAdd(&Q_next_len[k_idx], 1);
                        d_Q_next[b_offset + pos] = w;
                    }
                    if (d_d[b_offset + w] == depth[k_idx] + 1)
                        atomicAdd(&d_sigma[b_offset + w], d_sigma[b_offset + v]);
                }
            }
        }
        __syncthreads();

        // 2. Q_next の処理と配列コピー
        if (local_done[k_idx] == 0) {
            if (Q_next_len[k_idx] == 0) {
                if (lane == 0) {
                    depth[k_idx]      = d_d[b_offset + d_S[b_offset + S_len[k_idx] - 1]];
                    local_done[k_idx] = 1;
                }
            } else {
                int curr_Q_next_len = Q_next_len[k_idx];
                for (int i = lane; i < curr_Q_next_len; i += THREADS_PER_SOURCE) {
                    d_Q_curr[b_offset + i]           = d_Q_next[b_offset + i];
                    d_S[b_offset + S_len[k_idx] + i] = d_Q_next[b_offset + i];
                }
            }
        }
        __syncthreads();

        // 3. S_ends とメタデータの更新
        if (local_done[k_idx] == 0) {
            if (lane == 0) {
                int curr_Q_next_len = Q_next_len[k_idx];
                if (S_ends_len[k_idx] >= max_depth_estimate + 1) {
                    atomicExch(d_overflow, 1);
                    curr_Q_next_len = 0;
                } else {
                    d_S_ends[ends_offset + S_ends_len[k_idx]] =
                        S_len[k_idx] + curr_Q_next_len;
                    S_ends_len[k_idx]++;
                }
                Q_curr_len[k_idx] = curr_Q_next_len;
                S_len[k_idx]     += curr_Q_next_len;
                Q_next_len[k_idx] = 0;
                depth[k_idx]++;
            }
        }
        __syncthreads();
    }

    if (active && lane == 0)
        d_depth[batch_idx] = depth[k_idx];
}


template <bool IS_UNDIRECTED>
__global__ void brandes_kernel_opt(
        int *R, int *C, double *CB, int n_nodes,
        int max_depth_estimate, int *d_overflow,
        int *d_d, double *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, double *d_delta, int s_start)
{
    int batch_idx = blockIdx.x;
    int s         = s_start + batch_idx;
    int tid       = threadIdx.x;
    const size_t base = batch_node_offset(batch_idx, n_nodes);
    const size_t ends_base = batch_level_offset(batch_idx, max_depth_estimate);

    __shared__ int Q_curr_len, Q_next_len, S_len, S_ends_len, depth;

    // ホスト側 memsetAsync で配列は初期化済み。ソース頂点のみ上書き (O(1))。
    if (tid == 0) {
        d_d   [base + s] = 0;
        d_sigma[base + s] = 1.0;
        d_Q_curr[base]   = s;
        Q_curr_len  = 1;
        Q_next_len  = 0;
        d_S[base]        = s;
        S_len       = 1;
        d_S_ends[ends_base]     = 0;
        d_S_ends[ends_base + 1] = 1;
        S_ends_len  = 2;
        depth       = 0;
    }
    __syncthreads();

    find_shortest_paths_opt<true>(R, C, d_d, d_sigma, d_Q_curr, d_Q_next,
                                d_S, d_S_ends, batch_idx, n_nodes,
                                max_depth_estimate, d_overflow,
                                Q_curr_len, Q_next_len, S_len, S_ends_len, depth);
    __syncthreads();

    accumulate_dependencies_opt(R, C, d_d, d_sigma, d_delta,
                                d_S, d_S_ends, batch_idx, n_nodes, max_depth_estimate, depth);
    __syncthreads();

    // BC 値の集計 (無向グラフは 1/2)
    for (int v = tid; v < n_nodes; v += blockDim.x) {
        if (v != s) {
            double contrib = IS_UNDIRECTED
                           ? d_delta[base + v] / 2.0
                           : d_delta[base + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

template <bool IS_UNDIRECTED>
__global__ void single_source_dep_back_kernel_opt(
        int *R, int *C, int n_nodes, int max_depth_estimate,
        int *d_d, double *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, const int *d_depth)
{
    int tid = threadIdx.x;
    __shared__ int depth;
    if (tid == 0) depth = d_depth[0];
    __syncthreads();

    accumulate_dependencies_opt(R, C, d_d, d_sigma, d_delta,
                                     d_S, d_S_ends, 0, n_nodes, max_depth_estimate, depth);
}

#endif // BRANDES_KERNELS_CUH

