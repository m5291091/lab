#include "common.h"
#include "brandes.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

// [最適化3.5] shared-frontier カーネル定数 (_pure_chunked.cu)
#define K_SOURCES_PER_BLOCK 16
#define THREADS_PER_SOURCE  16  // 1ブロック = 256 スレッド (16*16)

namespace cg = cooperative_groups;
using namespace std;

namespace {
inline int choose_tpb_for_graph(const cudaDeviceProp& prop, int n_nodes, int edge_size,
                                bool log_choice)
{
    int estimated_avg_frontier = max(32, (int)std::sqrt((double)n_nodes));
    int tpb_candidate = ((estimated_avg_frontier + 31) / 32) * 32;
    tpb_candidate = min(tpb_candidate, prop.maxThreadsPerBlock);

    double avg_deg = (n_nodes > 0) ? ((double)edge_size / (double)n_nodes) : 0.0;
    int tpb;
    if (avg_deg < 5.0) {
        tpb = 128;
    } else if (avg_deg < 20.0) {
        tpb = 256;
    } else {
        tpb = 512;
    }

    tpb = min(tpb, prop.maxThreadsPerBlock);
    tpb = min(tpb, tpb_candidate);
    tpb = max(tpb, 32);
    tpb = (tpb / 32) * 32;

    if (log_choice) {
#ifdef DEBUG_BRANDES
        fprintf(stderr, "  > [tpb] avg_deg=%.2f, chosen tpb=%d\n", avg_deg, tpb);
#endif
    }
    return tpb;
}
} // namespace

// ============================================================
//  brandes_gpu_opt_pure_chunked.cu
//  純 GPU メモリ版 BC 最適化実装 (Unified Memory 不使用)
//
//  brandes_gpu_opt.cu からの変更点:
//    - cudaMalloc で GPU メモリを直接確保
//    - cudaMemAdvise / cudaMemPrefetchAsync を全て除去
//    - CSR トポロジ (R, C) は cudaMemcpy(HostToDevice) で転送
//    - 結果バッファ (CB) は cudaMemcpy(DeviceToHost) で回収
//    - UM ページフォルト/マイグレーションに依存しない明示的メモリ管理
//
//  保持された最適化:
//    [最適化3] ホスト側 cudaMemsetAsync によるカーネル外初期化
//    [最適化4] 2ストリーム ダブルバッファリング
//    - トップダウン/ボトムアップ ハイブリッド BFS
//    - Warp レベルシャッフル還元
// ============================================================

static __device__ bool isUndirected_opt_pure_chunked = true;

// ============================================================
//  デバイス関数: BFS 前向き探索 (トップダウン/ボトムアップ ハイブリッド)
// ============================================================
__device__ void find_shortest_paths_opt_pure_chunked(
        int *R, int *C, int *d_d, double *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, int batch_idx, int n_nodes,
        int max_depth_estimate, int *d_overflow,
        int &Q_curr_len, int &Q_next_len, int &S_len, int &S_ends_len, int &depth)
{
    int tid   = threadIdx.x;
    int bsize = blockDim.x;
    int v, w;

    __shared__ int direction;        // 0 = TOP_DOWN, 1 = BOTTOM_UP
    __shared__ int m_f, m_u;
    __shared__ int m_f_next;
    const int alpha = 14, beta = 24; // Beamer 推奨値

    // 初期状態の設定
    if (tid == 0) {
        direction = 0;
        int s = d_S[batch_idx * n_nodes];
        m_f = R[s + 1] - R[s];
        m_u = R[n_nodes] - m_f;
    }
    __syncthreads();

    while (true) {
        // 各レベル開始時に方向選択
        if (tid == 0) {
            if (direction == 0 && m_f > m_u / alpha)        direction = 1;
            else if (direction == 1 && Q_curr_len < n_nodes / beta) direction = 0;
            m_f_next = 0;
        }
        __syncthreads();

        if (direction == 0) {
            for (int i = tid; i < Q_curr_len; i += bsize) {
                v = d_Q_curr[batch_idx * n_nodes + i];
                for (int j = R[v]; j < R[v + 1]; j++) {
                    w = C[j];
                    if (atomicCAS(&d_d[batch_idx * n_nodes + w], -1, depth + 1) == -1) {
                        int pos = atomicAdd(&Q_next_len, 1);
                        d_Q_next[batch_idx * n_nodes + pos] = w;
                    }
                    if (d_d[batch_idx * n_nodes + w] == depth + 1) {
                        atomicAdd(&d_sigma[batch_idx * n_nodes + w],
                                  d_sigma[batch_idx * n_nodes + v]);
                    }
                }
            }
        } else {
            for (int i = tid; i < n_nodes; i += bsize) {
                w = i;
                if (d_d[batch_idx * n_nodes + w] == -1) {
                    double sum_sigma = 0.0;
                    for (int j = R[w]; j < R[w + 1]; j++) {
                        v = C[j];
                        if (d_d[batch_idx * n_nodes + v] == depth)
                            sum_sigma += d_sigma[batch_idx * n_nodes + v];
                    }
                    if (sum_sigma > 0.0) {
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
            if (tid == 0)
                depth = d_d[batch_idx * n_nodes + d_S[batch_idx * n_nodes + S_len - 1]];
            break;
        }

        int curr_Q_next_len = Q_next_len;

        // m_f の計算: warp 内 shfl_down で集約
        int local_m_f = 0;
        for (int i = tid; i < curr_Q_next_len; i += bsize) {
            int u = d_Q_next[batch_idx * n_nodes + i];
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
            d_Q_curr[batch_idx * n_nodes + i]        = d_Q_next[batch_idx * n_nodes + i];
            d_S[batch_idx * n_nodes + S_len + i]     = d_Q_next[batch_idx * n_nodes + i];
        }
        __syncthreads();

        if (tid == 0) {
            if (S_ends_len >= max_depth_estimate + 1) {
                atomicExch(d_overflow, 1);
                Q_next_len = 0;
            } else {
                d_S_ends[batch_idx * (max_depth_estimate + 1) + S_ends_len] = S_len + curr_Q_next_len;
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
__device__ void accumulate_dependencies_opt_pure_chunked(
        int *R, int *C, int *d_d, double *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, int batch_idx, int n_nodes, int max_depth_estimate, int &depth)
{
    auto block     = cg::this_thread_block();
    auto warp      = cg::tiled_partition<32>(block);
    int  tid_block = block.thread_rank();
    int  warp_id   = tid_block / warp.size();
    int  num_warps = block.size() / warp.size();

    while (depth > 0) {
        int start          = d_S_ends[batch_idx * (max_depth_estimate + 1) + depth];
        int end            = d_S_ends[batch_idx * (max_depth_estimate + 1) + depth + 1];
        int nodes_in_level = end - start;

        for (int i = warp_id; i < nodes_in_level; i += num_warps) {
            int    w         = d_S[batch_idx * n_nodes + start + i];
            double sigma_w   = d_sigma[batch_idx * n_nodes + w];
            double local_sum = 0.0;

            for (int j = R[w] + warp.thread_rank(); j < R[w + 1]; j += warp.size()) {
                int v = C[j];
                if (d_d[batch_idx * n_nodes + v] == d_d[batch_idx * n_nodes + w] + 1) {
                    local_sum += (sigma_w / d_sigma[batch_idx * n_nodes + v])
                                 * (1.0 + d_delta[batch_idx * n_nodes + v]);
                }
            }

            // Warp シャッフル還元
            for (int offset = warp.size() / 2; offset > 0; offset /= 2)
                local_sum += warp.shfl_down(local_sum, offset);

            if (warp.thread_rank() == 0)
                d_delta[batch_idx * n_nodes + w] = local_sum;
        }

        block.sync();
        if (tid_block == 0) depth--;
        block.sync();
    }
}

// thread-per-vertex バックワード (低密度グラフ向け: 平均次数 < 8)
__device__ void accumulate_dependencies_tpv_opt_pure_chunked(
        int *R, int *C, int *d_d, double *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, int batch_idx, int n_nodes, int max_depth_estimate, int &depth)
{
    int tid_block = threadIdx.x;
    int bsize     = blockDim.x;

    while (depth > 0) {
        int start          = d_S_ends[batch_idx * (max_depth_estimate + 1) + depth];
        int end            = d_S_ends[batch_idx * (max_depth_estimate + 1) + depth + 1];
        int nodes_in_level = end - start;

        for (int i = tid_block; i < nodes_in_level; i += bsize) {
            int    w       = d_S[batch_idx * n_nodes + start + i];
            double sigma_w = d_sigma[batch_idx * n_nodes + w];
            double sum     = 0.0;
            int    dw      = d_d[batch_idx * n_nodes + w];
            for (int j = R[w]; j < R[w + 1]; j++) {
                int v = C[j];
                if (d_d[batch_idx * n_nodes + v] == dw + 1) {
                    sum += (sigma_w / d_sigma[batch_idx * n_nodes + v])
                         * (1.0 + d_delta[batch_idx * n_nodes + v]);
                }
            }
            d_delta[batch_idx * n_nodes + w] = sum;
        }
        __syncthreads();
        if (tid_block == 0) depth--;
        __syncthreads();
    }
}

// ============================================================
//  GPU カーネル
//
//  [最適化3] ホスト側 cudaMemsetAsync により以下が設定済み:
//    d_d    : 全要素 -1  (0xFFFFFFFF)
//    d_sigma: 全要素  0
//    d_delta: 全要素  0.0
//  カーネル内ではソース頂点 s の 1 点セットアップ (O(1)) のみ実施。
// ============================================================

// [最適化6a] 到達済み頂点のみリセット (全 N 頂点の memset を回避)
// blockIdx.x = batch_idx, threadIdx.x = スレッドID
__global__ void reset_visited_batch_kernel_pure_chunked(
        int *d_d, double *d_sigma, double *d_delta,
        const int *d_S, int n_nodes,
        const int *d_depth, const int *d_S_ends, int max_depth_estimate)
{
    int batch_idx = blockIdx.x;
    __shared__ int reachable_count;
    if (threadIdx.x == 0) {
        int depth = d_depth[batch_idx];
        reachable_count = d_S_ends[batch_idx * (max_depth_estimate + 1) + depth + 1];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < reachable_count; i += blockDim.x) {
        int v = d_S[batch_idx * n_nodes + i];
        d_d   [batch_idx * n_nodes + v] = -1;
        d_sigma[batch_idx * n_nodes + v] = 0.0;
        d_delta[batch_idx * n_nodes + v] = 0.0;
    }
}

// BFS フェーズカーネル: O(1) 初期化 (memsetAsync 済み) + BFS + 深さ保存
__global__ void brandes_bfs_kernel_opt_pure_chunked(
        int *R, int *C, int n_nodes,
        int max_depth_estimate, int *d_overflow,
        int *d_d, double *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, int *d_depth, int s_start)
{
    int batch_idx = blockIdx.x;
    int s         = s_start + batch_idx;
    int tid       = threadIdx.x;

    __shared__ int Q_curr_len, Q_next_len, S_len, S_ends_len, depth;

    // ホスト側 memsetAsync で配列は初期化済み。ソース頂点のみ上書き (O(1))。
    if (tid == 0) {
        d_d   [batch_idx * n_nodes + s] = 0;
        d_sigma[batch_idx * n_nodes + s] = 1.0;
        d_Q_curr[batch_idx * n_nodes]   = s;
        Q_curr_len  = 1;
        Q_next_len  = 0;
        d_S[batch_idx * n_nodes]        = s;
        S_len       = 1;
        d_S_ends[batch_idx * (max_depth_estimate + 1)]     = 0;
        d_S_ends[batch_idx * (max_depth_estimate + 1) + 1] = 1;
        S_ends_len  = 2;
        depth       = 0;
    }
    __syncthreads();

    find_shortest_paths_opt_pure_chunked(R, C, d_d, d_sigma, d_Q_curr, d_Q_next,
                                 d_S, d_S_ends, batch_idx, n_nodes,
                                 max_depth_estimate, d_overflow,
                                 Q_curr_len, Q_next_len, S_len, S_ends_len, depth);
    __syncthreads();

    if (tid == 0) d_depth[batch_idx] = depth;
}

// [最適化3.5] shared-frontier BFS カーネル (road network 等 avg_deg < 5 向け, pure版)
__global__ void brandes_bfs_kernel_shared_frontier_pure_chunked(
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

// バックワードフェーズカーネル
__global__ void brandes_back_kernel_opt_pure_chunked(
        int *R, int *C, double *CB, int n_nodes, int max_depth_estimate,
        int *d_d, double *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, const int *d_depth, int s_start)
{
    int batch_idx = blockIdx.x;
    int s         = s_start + batch_idx;
    int tid       = threadIdx.x;

    __shared__ int depth;
    if (tid == 0) depth = d_depth[batch_idx];
    __syncthreads();

    accumulate_dependencies_opt_pure_chunked(R, C, d_d, d_sigma, d_delta,
                                     d_S, d_S_ends, batch_idx, n_nodes, max_depth_estimate, depth);
    __syncthreads();

    for (int v = tid; v < n_nodes; v += blockDim.x) {
        if (v != s) {
            double contrib = isUndirected_opt_pure_chunked
                           ? d_delta[batch_idx * n_nodes + v] / 2.0
                           : d_delta[batch_idx * n_nodes + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

// バックワードフェーズカーネル (thread-per-vertex 版, avg_deg < 8 向け)
__global__ void brandes_back_kernel_tpv_opt_pure_chunked(
        int *R, int *C, double *CB, int n_nodes, int max_depth_estimate,
        int *d_d, double *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, const int *d_depth, int s_start)
{
    int batch_idx = blockIdx.x;
    int s         = s_start + batch_idx;
    int tid       = threadIdx.x;

    __shared__ int depth;
    if (tid == 0) depth = d_depth[batch_idx];
    __syncthreads();

    accumulate_dependencies_tpv_opt_pure_chunked(R, C, d_d, d_sigma, d_delta,
                                         d_S, d_S_ends, batch_idx, n_nodes, max_depth_estimate, depth);
    __syncthreads();

    for (int v = tid; v < n_nodes; v += blockDim.x) {
        if (v != s) {
            double contrib = isUndirected_opt_pure_chunked
                           ? d_delta[batch_idx * n_nodes + v] / 2.0
                           : d_delta[batch_idx * n_nodes + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

// 単一 source 依存値計算用バックワードカーネル
__global__ void single_source_dep_back_kernel_opt_pure_chunked(
        int *R, int *C, int n_nodes, int max_depth_estimate,
        int *d_d, double *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, const int *d_depth)
{
    int tid = threadIdx.x;
    __shared__ int depth;
    if (tid == 0) depth = d_depth[0];
    __syncthreads();

    accumulate_dependencies_opt_pure_chunked(R, C, d_d, d_sigma, d_delta,
                                     d_S, d_S_ends, 0, n_nodes, max_depth_estimate, depth);
}

// ============================================================
//  内部実装: ダブルバッファリング + 2ストリーム処理 (純 GPU メモリ版)
// ============================================================
static vector<double> brandes_gpu_opt_pure_chunked_impl(
        int *d_R, int *d_C, double *d_CB,
        int n_nodes, int edge_size, int gpu_id, double h2d_time = 0.0)
{
    cudaDeviceProp prop;
    CUDA_ERR_CHK(cudaGetDeviceProperties(&prop, gpu_id));

    int tpb = choose_tpb_for_graph(prop, n_nodes, edge_size, true);
    double avg_deg = (n_nodes > 0) ? ((double)edge_size / n_nodes) : 0.0;

    int max_depth_estimate;
    if (avg_deg < 5.0)        max_depth_estimate = 4096;
    else if (avg_deg < 20.0)  max_depth_estimate = 256;
    else                      max_depth_estimate = 64;

    // [最適化4] NS=2 ストリームでダブルバッファリング
    const int NS = 2;

    // バッチサイズ計算: NS バッファ分の動的メモリを確保できるよう調整
    size_t free_mem, total_mem;
    CUDA_ERR_CHK(cudaMemGetInfo(&free_mem, &total_mem));

    // d_d, d_Q_curr, d_Q_next  (int) ; d_sigma, d_delta (double)
    const size_t per_batch_mem =
        (size_t)n_nodes * (3 * sizeof(int) + 2 * sizeof(double))  // d_d, d_Q_curr, d_Q_next, d_sigma, d_delta
        + (size_t)n_nodes * sizeof(int)                            // d_S
        + (size_t)(max_depth_estimate + 1) * sizeof(int)           // d_S_ends
        + sizeof(int);                                             // d_depth

    const size_t safety          = (size_t)(free_mem * 0.15);
    const size_t available       = (free_mem > safety) ? (free_mem - safety) : 0;
    int          BATCH_PER_STREAM = (int)(available / ((size_t)NS * per_batch_mem));
    BATCH_PER_STREAM = max(1, min(BATCH_PER_STREAM, 512));

    // BC_BATCH_OVERRIDE 環境変数でバッチサイズを上書き (感度分析用)
    // pure_chunked 版: 上限なしで上書き可。実バッファは SUB_BATCH 単位なので OOM なし。
    if (const char *env = getenv("BC_BATCH_OVERRIDE")) {
        int override_val = atoi(env);
        if (override_val > 0)
            BATCH_PER_STREAM = max(1, override_val);
    }

    // ---- SUB_BATCH の動的算出 (gpu_opt.cu と同等のロジック) -------------
    // pure 版は cudaMalloc を SUB_BATCH × N でしか確保しないため、
    // BATCH_PER_STREAM がいくら大きくても HBM3 上限を超えない。
    // SUB_BATCH は free_mem の 80% から topology + CB を控除した予算に基づき決定。
    size_t topology_bytes = ((size_t)(n_nodes + 1) + (size_t)edge_size) * sizeof(int);
    size_t dynamic_bytes  = (size_t)NS * BATCH_PER_STREAM * per_batch_mem;
    const bool oversubscribed = (dynamic_bytes > free_mem * 0.90);
    const int  NS_eff = oversubscribed ? 1 : NS;
    const size_t cb_bytes = (size_t)n_nodes * sizeof(double);
    size_t hbm3_budget = (size_t)(free_mem * 0.80);
    const size_t resident_bytes = topology_bytes + cb_bytes;
    hbm3_budget = (hbm3_budget > resident_bytes) ? (hbm3_budget - resident_bytes) : (size_t)(free_mem * 0.50);
    int sub_batch_max = BATCH_PER_STREAM;
    if (oversubscribed) {
        const size_t denom = (size_t)NS_eff * per_batch_mem;
        sub_batch_max = (denom > 0) ? (int)min((size_t)BATCH_PER_STREAM, hbm3_budget / denom) : BATCH_PER_STREAM;
    }
    const int safe_sub_batch = (n_nodes > 0) ? max(1, std::numeric_limits<int>::max() / n_nodes) : 1;
    sub_batch_max = min(sub_batch_max, safe_sub_batch);
    int SUB_BATCH = min(BATCH_PER_STREAM, sub_batch_max);
    SUB_BATCH = max(1, SUB_BATCH);
    if (BATCH_PER_STREAM >= 32)
        SUB_BATCH = max(SUB_BATCH, 32);
    SUB_BATCH = min(SUB_BATCH, BATCH_PER_STREAM);
    int num_subs = (BATCH_PER_STREAM + SUB_BATCH - 1) / SUB_BATCH;

    // 実バッファサイズ (SUB_BATCH 単位)
    const size_t buf_int_sub = (size_t)SUB_BATCH * n_nodes * sizeof(int);
    const size_t buf_dbl_sub = (size_t)SUB_BATCH * n_nodes * sizeof(double);
    const size_t buf_se_sub  = (size_t)SUB_BATCH * (max_depth_estimate + 1) * sizeof(int);
    const size_t buf_dpth_sub = (size_t)SUB_BATCH * sizeof(int);
    const size_t actual_dynamic_bytes = (size_t)NS_eff *
        (3 * buf_int_sub + buf_dbl_sub + buf_int_sub + buf_se_sub + buf_dbl_sub + buf_dpth_sub);

    fprintf(stderr, "  > [Mem] GPU: total=%.1f GB, free_before=%.1f GB\n",
            total_mem / 1e9, free_mem / 1e9);
    fprintf(stderr, "  > [Mem] topology(GPU)=%.2f GB, dynamic(SUB_BATCH alloc)=%.2f GB, "
                    "BATCH=%d, SUB_BATCH=%d, num_subs=%d, NS_eff=%d\n",
            topology_bytes / 1e9, actual_dynamic_bytes / 1e9,
            BATCH_PER_STREAM, SUB_BATCH, num_subs, NS_eff);
    if (oversubscribed)
        fprintf(stderr, "  > [Mode] Manual chunking: %d sources/launch (HBM3 budget=%.1f GB), "
                "%d sub-launches per BATCH\n",
                SUB_BATCH, hbm3_budget / 1e9, num_subs);

    // NS 組の動的ステートバッファを cudaMalloc で確保 (UM 不使用, SUB_BATCH 単位)
    struct DynBuf {
        int    *d_d, *d_Q_curr, *d_Q_next, *d_S, *d_S_ends;
        double *d_sigma, *d_delta;
        int    *d_depth;
        cudaEvent_t ev_bfs_s, ev_bfs_e, ev_back_e;
        float bfs_ms, back_ms;
        int prev_sub_batch_n;
        int prev_sub_batch_off;
    };
    DynBuf bufs[NS];

    for (int i = 0; i < NS_eff; i++) {
        CUDA_ERR_CHK(cudaMalloc(&bufs[i].d_d,      buf_int_sub));
        CUDA_ERR_CHK(cudaMalloc(&bufs[i].d_sigma,  buf_dbl_sub));
        CUDA_ERR_CHK(cudaMalloc(&bufs[i].d_Q_curr, buf_int_sub));
        CUDA_ERR_CHK(cudaMalloc(&bufs[i].d_Q_next, buf_int_sub));
        CUDA_ERR_CHK(cudaMalloc(&bufs[i].d_S,      buf_int_sub));
        CUDA_ERR_CHK(cudaMalloc(&bufs[i].d_S_ends, buf_se_sub));
        CUDA_ERR_CHK(cudaMalloc(&bufs[i].d_delta,  buf_dbl_sub));
        CUDA_ERR_CHK(cudaMalloc(&bufs[i].d_depth,  buf_dpth_sub));

        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_bfs_s));
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_bfs_e));
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_back_e));
        bufs[i].bfs_ms = 0.0f;
        bufs[i].back_ms = 0.0f;
        bufs[i].prev_sub_batch_n   = 0;
        bufs[i].prev_sub_batch_off = 0;
    }

    // ストリーム作成
    cudaStream_t streams[NS] = {};
    for (int i = 0; i < NS_eff; i++)
        CUDA_ERR_CHK(cudaStreamCreate(&streams[i]));

    int *d_overflow;
    CUDA_ERR_CHK(cudaMallocManaged(&d_overflow, sizeof(int)));
    *d_overflow = 0;

    // ============================================================
    //  メインループ: BATCH_PER_STREAM を SUB_BATCH 単位で手動 chunking
    //  バッファは SUB_BATCH × N で確保済み。各サブバッチは
    //  バッファ先頭 (batch_idx=0..sub_n-1) を使い、s_eff = s_start + sub_off
    //  でソースインデックスをシフトする。
    // ============================================================
    bool buf_used[NS] = {};
    for (int s_start = 0; s_start < n_nodes; s_start += BATCH_PER_STREAM) {
        int          si         = (s_start / BATCH_PER_STREAM) % NS_eff;
        cudaStream_t st         = streams[si];
        int          curr_batch = min(BATCH_PER_STREAM, n_nodes - s_start);
        if (curr_batch <= 0) continue;

        // 同じバッファを再利用する前に前回バッチのフェーズ時間を回収
        if (buf_used[si]) {
            CUDA_ERR_CHK(cudaEventSynchronize(bufs[si].ev_back_e));
            float b_ms = 0.0f, bk_ms = 0.0f;
            CUDA_ERR_CHK(cudaEventElapsedTime(&b_ms,  bufs[si].ev_bfs_s, bufs[si].ev_bfs_e));
            CUDA_ERR_CHK(cudaEventElapsedTime(&bk_ms, bufs[si].ev_bfs_e, bufs[si].ev_back_e));
            bufs[si].bfs_ms  += b_ms;
            bufs[si].back_ms += bk_ms;
        }

        bool first_event = true;
        for (int sub_off = 0; sub_off < curr_batch; sub_off += SUB_BATCH) {
            int sub_n = min(SUB_BATCH, curr_batch - sub_off);

            // [最適化6a] 同じ範囲を再利用する場合は到達済み頂点のみリセット
            nvtxRangePushA("Memset_or_reset_subbatch_pure_chunked");
            if (bufs[si].prev_sub_batch_n   == sub_n
             && bufs[si].prev_sub_batch_off == sub_off
             && bufs[si].prev_sub_batch_n   >  0) {
                reset_visited_batch_kernel_pure_chunked<<<sub_n, tpb, 0, st>>>(
                    bufs[si].d_d, bufs[si].d_sigma, bufs[si].d_delta,
                    bufs[si].d_S, n_nodes,
                    bufs[si].d_depth, bufs[si].d_S_ends, max_depth_estimate);
            } else {
                CUDA_ERR_CHK(cudaMemsetAsync(bufs[si].d_d,
                    0xFF, (size_t)sub_n * n_nodes * sizeof(int), st));
                CUDA_ERR_CHK(cudaMemsetAsync(bufs[si].d_sigma,
                    0,    (size_t)sub_n * n_nodes * sizeof(double), st));
                CUDA_ERR_CHK(cudaMemsetAsync(bufs[si].d_delta,
                    0,    (size_t)sub_n * n_nodes * sizeof(double), st));
            }
            nvtxRangePop(); // Memset_or_reset_subbatch_pure_chunked

            if (first_event) {
                CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_bfs_s, st));
                first_event = false;
            }

            int s_eff = s_start + sub_off;

            // BFS フェーズ (avg_deg < 5: shared-frontier, それ以外: 1ブロック=1ソース)
            if (avg_deg < 5.0) {
                nvtxRangePushA("BFS_kernel_shared_frontier_pure_chunked");
                const int K          = K_SOURCES_PER_BLOCK;
                const int tpb_shared = K * THREADS_PER_SOURCE; // 256
                int grid_shared      = (sub_n + K - 1) / K;
                brandes_bfs_kernel_shared_frontier_pure_chunked<<<grid_shared, tpb_shared, 0, st>>>(
                    d_R, d_C, n_nodes, max_depth_estimate, d_overflow,
                    bufs[si].d_d,      bufs[si].d_sigma,
                    bufs[si].d_Q_curr, bufs[si].d_Q_next,
                    bufs[si].d_S,      bufs[si].d_S_ends,
                    bufs[si].d_depth,  s_eff, sub_n);
                nvtxRangePop(); // BFS_kernel_shared_frontier_pure_chunked
            } else {
                nvtxRangePushA("BFS_kernel_opt_pure_chunked");
                brandes_bfs_kernel_opt_pure_chunked<<<sub_n, tpb, 0, st>>>(
                    d_R, d_C, n_nodes, max_depth_estimate, d_overflow,
                    bufs[si].d_d,      bufs[si].d_sigma,
                    bufs[si].d_Q_curr, bufs[si].d_Q_next,
                    bufs[si].d_S,      bufs[si].d_S_ends,
                    bufs[si].d_depth,  s_eff);
                nvtxRangePop(); // BFS_kernel_opt_pure_chunked
            }
            CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_bfs_e, st));

            // バックワードフェーズ (avg_deg < 8: thread-per-vertex, それ以外: warp-per-vertex)
            nvtxRangePushA("Backward_kernel_opt_pure_chunked");
            if (avg_deg < 8.0) {
                brandes_back_kernel_tpv_opt_pure_chunked<<<sub_n, tpb, 0, st>>>(
                    d_R, d_C, d_CB, n_nodes, max_depth_estimate,
                    bufs[si].d_d,     bufs[si].d_sigma,
                    bufs[si].d_delta,
                    bufs[si].d_S,     bufs[si].d_S_ends,
                    bufs[si].d_depth, s_eff);
            } else {
                brandes_back_kernel_opt_pure_chunked<<<sub_n, tpb, 0, st>>>(
                    d_R, d_C, d_CB, n_nodes, max_depth_estimate,
                    bufs[si].d_d,     bufs[si].d_sigma,
                    bufs[si].d_delta,
                    bufs[si].d_S,     bufs[si].d_S_ends,
                    bufs[si].d_depth, s_eff);
            }
            CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_back_e, st));
            nvtxRangePop(); // Backward_kernel_opt_pure_chunked

            bufs[si].prev_sub_batch_n   = sub_n;
            bufs[si].prev_sub_batch_off = sub_off;
        }

        buf_used[si] = true;
        CUDA_ERR_CHK(cudaPeekAtLastError());
    }

    // 全ストリームの完了を待機し、最後のバッチのフェーズ時間を回収
    float wall_bfs_ms = 0.0f, wall_back_ms = 0.0f;
    float total_bfs_ms = 0.0f, total_back_ms = 0.0f;
    for (int i = 0; i < NS_eff; i++) {
        if (!buf_used[i]) continue;
        CUDA_ERR_CHK(cudaStreamSynchronize(streams[i]));

        float b_ms = 0.0f, bk_ms = 0.0f;
        CUDA_ERR_CHK(cudaEventElapsedTime(&b_ms,  bufs[i].ev_bfs_s, bufs[i].ev_bfs_e));
        CUDA_ERR_CHK(cudaEventElapsedTime(&bk_ms, bufs[i].ev_bfs_e, bufs[i].ev_back_e));
        bufs[i].bfs_ms  += b_ms;
        bufs[i].back_ms += bk_ms;

        wall_bfs_ms   = max(wall_bfs_ms,  bufs[i].bfs_ms);
        wall_back_ms  = max(wall_back_ms, bufs[i].back_ms);
        total_bfs_ms  += bufs[i].bfs_ms;
        total_back_ms += bufs[i].back_ms;
    }

    if (*d_overflow) {
        fprintf(stderr, "\n[ERROR] GPU Memory Bounds Exceeded: BFS depth exceeded max_depth_estimate (%d).\n", max_depth_estimate);
        fprintf(stderr, "Fallback: Please process this graph with sequential CPU BC or increase max_depth_estimate.\n\n");
        exit(1);
    }

    double d2h_start = omp_get_wtime();

    // [最適化6b] ピンドメモリ + cudaMemcpyAsync で GPU → ホスト転送 (DMA 並走)
    double *h_CB_pinned = nullptr;
    CUDA_ERR_CHK(cudaMallocHost(&h_CB_pinned, (size_t)n_nodes * sizeof(double)));
    cudaStream_t copy_stream;
    CUDA_ERR_CHK(cudaStreamCreate(&copy_stream));
    CUDA_ERR_CHK(cudaMemcpyAsync(h_CB_pinned, d_CB,
                                 (size_t)n_nodes * sizeof(double),
                                 cudaMemcpyDeviceToHost, copy_stream));
    CUDA_ERR_CHK(cudaStreamSynchronize(copy_stream));
    CUDA_ERR_CHK(cudaStreamDestroy(copy_stream));
    vector<double> result(h_CB_pinned, h_CB_pinned + n_nodes);
    CUDA_ERR_CHK(cudaFreeHost(h_CB_pinned));

    double d2h_end = omp_get_wtime();
    double d2h_time = d2h_end - d2h_start;

    fprintf(stderr,
        "  > [GPU Phase] BFS wall=%.4f s (cum=%.4f s), Backward wall=%.4f s (cum=%.4f s), "
        "SUB_BATCH=%d, num_subs=%d\n",
        wall_bfs_ms / 1000.0f, total_bfs_ms / 1000.0f,
        wall_back_ms / 1000.0f, total_back_ms / 1000.0f,
        SUB_BATCH, num_subs);
#ifdef DEBUG_BRANDES
    fprintf(stderr,
        "  > [GPU Phase] H2D wall=%.4f s, BFS wall=%.4f s (cum=%.4f s), Backward wall=%.4f s (cum=%.4f s), D2H wall=%.4f s\n",
        h2d_time,
        wall_bfs_ms / 1000.0f, total_bfs_ms / 1000.0f,
        wall_back_ms / 1000.0f, total_back_ms / 1000.0f,
        d2h_time);
#endif

    // クリーンアップ
    for (int i = 0; i < NS_eff; i++) {
        CUDA_ERR_CHK(cudaFree(bufs[i].d_d));
        CUDA_ERR_CHK(cudaFree(bufs[i].d_sigma));
        CUDA_ERR_CHK(cudaFree(bufs[i].d_Q_curr));
        CUDA_ERR_CHK(cudaFree(bufs[i].d_Q_next));
        CUDA_ERR_CHK(cudaFree(bufs[i].d_S));
        CUDA_ERR_CHK(cudaFree(bufs[i].d_S_ends));
        CUDA_ERR_CHK(cudaFree(bufs[i].d_delta));
        CUDA_ERR_CHK(cudaFree(bufs[i].d_depth));
        CUDA_ERR_CHK(cudaEventDestroy(bufs[i].ev_bfs_s));
        CUDA_ERR_CHK(cudaEventDestroy(bufs[i].ev_bfs_e));
        CUDA_ERR_CHK(cudaEventDestroy(bufs[i].ev_back_e));
        CUDA_ERR_CHK(cudaStreamDestroy(streams[i]));
    }
    CUDA_ERR_CHK(cudaFree(d_overflow));

    return result;
}

static void ensure_gpu_device_0()
{
    int num_gpus;
    CUDA_ERR_CHK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        cerr << "No GPU found" << endl;
        exit(EXIT_FAILURE);
    }
    CUDA_ERR_CHK(cudaSetDevice(0));
}

// ============================================================
//  公開エントリポイント: CSR 直受け版
// ============================================================
vector<double> brandes_gpu_opt_pure_chunked_csr(const int* R, const int* C,
                                        int n_nodes, int edge_size)
{
    if (n_nodes <= 0) return {};
    ensure_gpu_device_0();

    double h2d_start = omp_get_wtime();

    // CSR トポロジデータを cudaMalloc で GPU に確保し、明示的に転送
    int *d_R, *d_C;
    CUDA_ERR_CHK(cudaMalloc(&d_R, (size_t)(n_nodes + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc(&d_C, (size_t)edge_size     * sizeof(int)));
    CUDA_ERR_CHK(cudaMemcpy(d_R, R, (size_t)(n_nodes + 1) * sizeof(int),
                             cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_C, C, (size_t)edge_size     * sizeof(int),
                             cudaMemcpyHostToDevice));

    // 結果バッファ: GPU 上に確保しゼロクリア
    double *d_CB;
    CUDA_ERR_CHK(cudaMalloc(&d_CB, (size_t)n_nodes * sizeof(double)));
    CUDA_ERR_CHK(cudaMemset(d_CB, 0, (size_t)n_nodes * sizeof(double)));

    double h2d_end = omp_get_wtime();
    double h2d_time = h2d_end - h2d_start;

    vector<double> result = brandes_gpu_opt_pure_chunked_impl(d_R, d_C, d_CB,
                                                       n_nodes, edge_size, 0, h2d_time);

    CUDA_ERR_CHK(cudaFree(d_R));
    CUDA_ERR_CHK(cudaFree(d_C));
    CUDA_ERR_CHK(cudaFree(d_CB));

    return result;
}

// ============================================================
//  公開エントリポイント: Graph 版（既存API）
// ============================================================
vector<double> brandes_gpu_opt_pure_chunked(Graph &G)
{
    int *R        = G.getAdjacencyListPointers();
    int *C        = G.getAdjacencyList();
    int  n_nodes  = G.getNodeCount();
    int  edge_size = 2 * G.getEdgeCount();
    return brandes_gpu_opt_pure_chunked_csr(R, C, n_nodes, edge_size);
}

// ============================================================
//  公開エントリポイント: 単一 source dependency (CSR 直受け)
// ============================================================
GpuSingleSourceDepResult single_source_dependency_gpu_opt_pure_chunked_csr(
    const int* R, const int* C, int n_nodes, int edge_size, int source)
{
    GpuSingleSourceDepResult out;
    if (n_nodes <= 0 || source < 0 || source >= n_nodes) {
        return out;
    }

    ensure_gpu_device_0();

    int *d_R = nullptr, *d_C = nullptr;
    int *d_d = nullptr, *d_Q_curr = nullptr, *d_Q_next = nullptr;
    int *d_S = nullptr, *d_S_ends = nullptr, *d_depth = nullptr;
    double *d_sigma = nullptr;
    double *d_delta = nullptr;

    CUDA_ERR_CHK(cudaMalloc(&d_R, (size_t)(n_nodes + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc(&d_C, (size_t)edge_size * sizeof(int)));
    CUDA_ERR_CHK(cudaMemcpy(d_R, R, (size_t)(n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(d_C, C, (size_t)edge_size * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_ERR_CHK(cudaMalloc(&d_d, (size_t)n_nodes * sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc(&d_sigma, (size_t)n_nodes * sizeof(double)));
    CUDA_ERR_CHK(cudaMalloc(&d_Q_curr, (size_t)n_nodes * sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc(&d_Q_next, (size_t)n_nodes * sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc(&d_S, (size_t)n_nodes * sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc(&d_S_ends, (size_t)(n_nodes + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc(&d_depth, sizeof(int)));
    CUDA_ERR_CHK(cudaMalloc(&d_delta, (size_t)n_nodes * sizeof(double)));

    CUDA_ERR_CHK(cudaMemset(d_d, 0xFF, (size_t)n_nodes * sizeof(int)));
    CUDA_ERR_CHK(cudaMemset(d_sigma, 0, (size_t)n_nodes * sizeof(double)));
    CUDA_ERR_CHK(cudaMemset(d_delta, 0, (size_t)n_nodes * sizeof(double)));

    cudaDeviceProp prop;
    CUDA_ERR_CHK(cudaGetDeviceProperties(&prop, 0));
    int tpb = choose_tpb_for_graph(prop, n_nodes, edge_size, false);

    // d_S_ends は (n_nodes + 1) で確保済み; max_depth_estimate = n_nodes が安全
    const int max_depth_estimate_ss = n_nodes;
    int *d_overflow_ss = nullptr;
    CUDA_ERR_CHK(cudaMallocManaged(&d_overflow_ss, sizeof(int)));
    *d_overflow_ss = 0;

    brandes_bfs_kernel_opt_pure_chunked<<<1, tpb>>>(
        d_R, d_C, n_nodes,
        max_depth_estimate_ss, d_overflow_ss,
        d_d, d_sigma, d_Q_curr, d_Q_next,
        d_S, d_S_ends, d_depth, source);
    CUDA_ERR_CHK(cudaPeekAtLastError());

    single_source_dep_back_kernel_opt_pure_chunked<<<1, tpb>>>(
        d_R, d_C, n_nodes, max_depth_estimate_ss,
        d_d, d_sigma, d_delta,
        d_S, d_S_ends, d_depth);
    CUDA_ERR_CHK(cudaPeekAtLastError());
    CUDA_ERR_CHK(cudaDeviceSynchronize());
    CUDA_ERR_CHK(cudaFree(d_overflow_ss));

    out.delta.resize(n_nodes, 0.0);
    CUDA_ERR_CHK(cudaMemcpy(out.delta.data(), d_delta,
                            (size_t)n_nodes * sizeof(double), cudaMemcpyDeviceToHost));

    int depth = 0;
    CUDA_ERR_CHK(cudaMemcpy(&depth, d_depth, sizeof(int), cudaMemcpyDeviceToHost));
    if (depth >= 0 && depth + 1 <= n_nodes) {
        CUDA_ERR_CHK(cudaMemcpy(&out.reachableCount, d_S_ends + (depth + 1),
                                sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        out.reachableCount = 0;
    }

    CUDA_ERR_CHK(cudaFree(d_R));
    CUDA_ERR_CHK(cudaFree(d_C));
    CUDA_ERR_CHK(cudaFree(d_d));
    CUDA_ERR_CHK(cudaFree(d_sigma));
    CUDA_ERR_CHK(cudaFree(d_Q_curr));
    CUDA_ERR_CHK(cudaFree(d_Q_next));
    CUDA_ERR_CHK(cudaFree(d_S));
    CUDA_ERR_CHK(cudaFree(d_S_ends));
    CUDA_ERR_CHK(cudaFree(d_depth));
    CUDA_ERR_CHK(cudaFree(d_delta));

    return out;
}
