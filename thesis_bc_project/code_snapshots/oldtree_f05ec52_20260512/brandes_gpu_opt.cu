#include "common.h"
#include "brandes.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace cg = cooperative_groups;
using namespace std;

namespace {
inline cudaMemLocation make_device_location(int device_id)
{
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = device_id;
    return loc;
}

inline cudaMemLocation make_host_location()
{
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeHost;
    loc.id   = 0;
    return loc;
}

inline cudaError_t cuda_mem_prefetch_async_compat(const void *ptr, size_t bytes,
                                                  int device_id, cudaStream_t stream = 0)
{
#if CUDART_VERSION >= 13000
    return cudaMemPrefetchAsync(ptr, bytes, make_device_location(device_id), 0u, stream);
#else
    return cudaMemPrefetchAsync(ptr, bytes, device_id, stream);
#endif
}

inline cudaError_t cuda_mem_prefetch_host_async_compat(const void *ptr, size_t bytes,
                                                       cudaStream_t stream = 0)
{
#if CUDART_VERSION >= 13000
    return cudaMemPrefetchAsync(ptr, bytes, make_host_location(), 0u, stream);
#else
    return cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream);
#endif
}

inline cudaError_t cuda_mem_advise_device_compat(const void *ptr, size_t bytes,
                                                 cudaMemoryAdvise advice, int device_id)
{
#if CUDART_VERSION >= 13000
    return cudaMemAdvise(ptr, bytes, advice, make_device_location(device_id));
#else
    return cudaMemAdvise(ptr, bytes, advice, device_id);
#endif
}

inline cudaError_t cuda_mem_advise_host_compat(const void *ptr, size_t bytes,
                                                cudaMemoryAdvise advice)
{
#if CUDART_VERSION >= 13000
    return cudaMemAdvise(ptr, bytes, advice, make_host_location());
#else
    return cudaMemAdvise(ptr, bytes, advice, cudaCpuDeviceId);
#endif
}

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
//  brandes_gpu_opt.cu
//  GH200 Grace Hopper Superchip 特化 BC 最適化実装
//
//  改善点:
//
//  [最適化1] cudaMemAdviseSetReadMostly
//    - 静的 CSR トポロジデータ (R, C) を読み取り専用としてマーク
//    - GH200 の HBM3 L2 キャッシュが重複コピーを保持できるようになる
//    - バッチ内の複数ソース頂点が同一隣接リストを参照する際のキャッシュヒット率が向上
//    - 参考: BC_Miyabi_report.pdf §3.2「静的・動的データの分離に基づくメモリヒント戦略」
//
//  [最適化2] グラフサイズ適応型メモリ配置
//    - グラフが HBM3 総容量の 35% 以内に収まる場合: R/C を HBM3 に直接配置
//    - それ以上の場合: CPU LPDDR5X に配置し NVLink-C2C (900 GB/s) 経由でアクセス
//    - 小〜中規模グラフでは brandes_gpu.cu と同等以上の性能を達成
//
//  [最適化3] ホスト側 cudaMemsetAsync によるカーネル外初期化
//    - v1 の最大ボトルネック: tid==0 が n_nodes を逐次初期化 (O(n) per source)
//      → SM の全スレッドが 1 スレッドの完了を待つため、実効スループットが 1/1024 程度
//    - 改善: ホスト側で cudaMemsetAsync をストリームキューに投入
//      → GPU Copy Engine (DMA) が SM とは独立して実行
//    - カーネル内ではソース頂点 s の 1 点セットアップのみ (O(1))
//
//  [最適化4] 2ストリーム ダブルバッファリング
//    - 2 本の CUDA ストリームが交互にバッチを処理
//    - Stream0 のカーネル実行中に Stream1 の cudaMemsetAsync が Copy Engine で並走
//    - GH200 の Copy Engine は SM とは独立した実行ユニット → 真の非同期オーバーラップ
//    - オーバーラップパターン:
//        Stream0: [memset buf0] → [kernel s=0..B] → [memset buf0] → [kernel s=2B..3B]
//        Stream1:               → [memset buf1] → [kernel s=B..2B] → ...
//      kernel0 実行中に memset buf1 が並走 → kernel1 開始時には初期化完了済み
//
//  参考文献:
//    - BC_Miyabi_report.pdf §3 GH200 メモリ階層最適化
//    - Miyabi.pdf §5 GH200 アーキテクチャ仕様
//    - Beamer et al. (2012) "Direction-Optimizing Breadth-First Search" (top-down/bottom-up)
// ============================================================

static __device__ bool isUndirected_opt = true;

__host__ __device__ inline size_t batch_node_offset(int batch_idx, int n_nodes)
{
    return (size_t)batch_idx * (size_t)n_nodes;
}

__host__ __device__ inline size_t batch_level_offset(int batch_idx, int max_depth_estimate)
{
    return (size_t)batch_idx * (size_t)(max_depth_estimate + 1);
}

// ============================================================
//  デバイス関数: BFS 前向き探索 (トップダウン/ボトムアップ ハイブリッド)
//  SetReadMostly + SetAccessedBy により HBM3 L2 がキャッシュラインを保持する。
// ============================================================
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
            if (direction == 0 && m_f > m_u / alpha)        direction = 1;
            else if (direction == 1 && Q_curr_len < n_nodes / beta) direction = 0;
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
__device__ void accumulate_dependencies_opt(
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
__device__ void accumulate_dependencies_tpv_opt(
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
__global__ void reset_visited_batch_kernel(
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

// BFS フェーズカーネル (最適化版): O(1) 初期化 (memsetAsync 済み) + BFS + 深さ保存
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

    find_shortest_paths_opt(R, C, d_d, d_sigma, d_Q_curr, d_Q_next,
                            d_S, d_S_ends, batch_idx, n_nodes,
                            max_depth_estimate, d_overflow,
                            Q_curr_len, Q_next_len, S_len, S_ends_len, depth);
    __syncthreads();

    if (tid == 0) d_depth[batch_idx] = depth;
}

// [最適化3.5] shared-frontier BFS カーネル (road network 等 avg_deg < 5 向け)
// 1ブロック = K_SOURCES_PER_BLOCK ソースを協調処理し、小さいフロンティアでも SM を占有
#define K_SOURCES_PER_BLOCK 16
#define THREADS_PER_SOURCE  16  // tpb = 256 = 16 * 16

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

// バックワードフェーズカーネル (最適化版)
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
            double contrib = isUndirected_opt
                           ? d_delta[base + v] / 2.0
                           : d_delta[base + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

// バックワードフェーズカーネル (thread-per-vertex 版, avg_deg < 8 向け)
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
            double contrib = isUndirected_opt
                           ? d_delta[base + v] / 2.0
                           : d_delta[base + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

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

    find_shortest_paths_opt(R, C, d_d, d_sigma, d_Q_curr, d_Q_next,
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
            double contrib = isUndirected_opt
                           ? d_delta[base + v] / 2.0
                           : d_delta[base + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

struct DynBuf {
    int    *d_d, *d_Q_curr, *d_Q_next, *d_S, *d_S_ends;
    double *d_sigma, *d_delta;
    int    *d_depth;
    cudaEvent_t ev_bfs_s, ev_bfs_e, ev_back_e;
    cudaEvent_t ev_pref_s, ev_pref_e;
    float bfs_ms, back_ms;
    float pref_ms;            // この buf のサブバッチ累積 prefetch 時間 (raw, 隠蔽前)
    int prev_batch;
    int prev_sub_batch_n;     // 直前サブバッチの sub_n (reset_visited gating 用)
    int prev_sub_batch_off;   // 直前サブバッチの sub_off (reset_visited gating 用)
};

static inline void prefetch_subbatch(const DynBuf &b, int sub_off, int sub_n,
                                     int n_nodes, int max_depth_estimate,
                                     int gpu_id, cudaStream_t st)
{
    const size_t off_int  = (size_t)sub_off * n_nodes;
    const size_t size_int = (size_t)sub_n   * n_nodes * sizeof(int);
    const size_t size_dbl = (size_t)sub_n   * n_nodes * sizeof(double);
    const size_t off_se   = (size_t)sub_off * (max_depth_estimate + 1);
    const size_t size_se  = (size_t)sub_n   * (max_depth_estimate + 1) * sizeof(int);

    CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(b.d_d      + off_int, size_int, gpu_id, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(b.d_sigma  + off_int, size_dbl, gpu_id, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(b.d_Q_curr + off_int, size_int, gpu_id, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(b.d_Q_next + off_int, size_int, gpu_id, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(b.d_S      + off_int, size_int, gpu_id, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(b.d_S_ends + off_se,  size_se,  gpu_id, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(b.d_delta  + off_int, size_dbl, gpu_id, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(b.d_depth  + sub_off, (size_t)sub_n * sizeof(int),
                                                gpu_id, st));
}

static inline void memset_subbatch(const DynBuf &b, int sub_off, int sub_n,
                                   int n_nodes, cudaStream_t st)
{
    const size_t off_int  = (size_t)sub_off * n_nodes;
    const size_t size_int = (size_t)sub_n   * n_nodes * sizeof(int);
    const size_t size_dbl = (size_t)sub_n   * n_nodes * sizeof(double);

    CUDA_ERR_CHK(cudaMemsetAsync(b.d_d     + off_int, 0xFF, size_int, st));
    CUDA_ERR_CHK(cudaMemsetAsync(b.d_sigma + off_int, 0,    size_dbl, st));
    CUDA_ERR_CHK(cudaMemsetAsync(b.d_delta + off_int, 0,    size_dbl, st));
}

static inline void evict_subbatch_to_host(const DynBuf &b, int sub_off, int sub_n,
                                          int n_nodes, int max_depth_estimate,
                                          cudaStream_t st)
{
    const size_t off_int  = (size_t)sub_off * n_nodes;
    const size_t size_int = (size_t)sub_n   * n_nodes * sizeof(int);
    const size_t size_dbl = (size_t)sub_n   * n_nodes * sizeof(double);
    const size_t off_se   = (size_t)sub_off * (max_depth_estimate + 1);
    const size_t size_se  = (size_t)sub_n   * (max_depth_estimate + 1) * sizeof(int);

    CUDA_ERR_CHK(cuda_mem_prefetch_host_async_compat(b.d_d      + off_int, size_int, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_host_async_compat(b.d_sigma  + off_int, size_dbl, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_host_async_compat(b.d_Q_curr + off_int, size_int, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_host_async_compat(b.d_Q_next + off_int, size_int, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_host_async_compat(b.d_S      + off_int, size_int, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_host_async_compat(b.d_S_ends + off_se,  size_se,  st));
    CUDA_ERR_CHK(cuda_mem_prefetch_host_async_compat(b.d_delta  + off_int, size_dbl, st));
    CUDA_ERR_CHK(cuda_mem_prefetch_host_async_compat(b.d_depth  + sub_off,
                                                     (size_t)sub_n * sizeof(int), st));
}

// ============================================================
//  内部実装: ダブルバッファリング + 2ストリーム処理
// ============================================================
static vector<double> brandes_gpu_opt_impl(
        int *R_m, int *C_m, double *CB_managed,
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
    // UM版: 上限なしで上書き可。oversubscribed 判定により適切な配置アドバイスが自動で付く。
    if (const char *env = getenv("BC_BATCH_OVERRIDE")) {
        int override_val = atoi(env);
        if (override_val > 0)
            BATCH_PER_STREAM = max(1, override_val);
    }

    // 使用予定メモリを報告 (96 GB HBM3 + NVLink-C2C 活用の証拠)
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

    fprintf(stderr, "  > [Mem] GPU HBM3: total=%.1f GB, free_before=%.1f GB\n",
            total_mem / 1e9, free_mem / 1e9);
    fprintf(stderr, "  > [Mem] topology(CPU/HBM3)=%.2f GB, dynamic(UM)=%.2f GB, BATCH=%d, SUB_BATCH=%d, num_subs=%d, NS_eff=%d\n",
            topology_bytes / 1e9, dynamic_bytes / 1e9, BATCH_PER_STREAM, SUB_BATCH, num_subs, NS_eff);
    if (oversubscribed)
        fprintf(stderr, "  > [Mode] HBM3 streaming: %d sources/launch (HBM3 budget=%.1f GB), "
                "%d sub-launches per BATCH\n",
                SUB_BATCH, hbm3_budget / 1e9, num_subs);

    // NS 組の動的ステートバッファを確保し HBM3 に配置
    DynBuf bufs[NS];
    const size_t buf_int_full  = (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int);
    const size_t buf_dbl_full  = (size_t)BATCH_PER_STREAM * n_nodes * sizeof(double);
    const size_t buf_se_full   = (size_t)BATCH_PER_STREAM * (max_depth_estimate + 1) * sizeof(int);
    const size_t buf_dpth_full = (size_t)BATCH_PER_STREAM * sizeof(int);

    for (int i = 0; i < NS_eff; i++) {
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_d,      buf_int_full));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_sigma,  buf_dbl_full));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_Q_curr, buf_int_full));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_Q_next, buf_int_full));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_S,      buf_int_full));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_S_ends, buf_se_full));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_delta,  buf_dbl_full));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_depth,  buf_dpth_full));

        if (!oversubscribed) {
            // HBM3 に収まる: 事前に GPU へプリフェッチ
            CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(bufs[i].d_d,      buf_int_full,  gpu_id, 0));
            CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(bufs[i].d_sigma,  buf_dbl_full,  gpu_id, 0));
            CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(bufs[i].d_Q_curr, buf_int_full,  gpu_id, 0));
            CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(bufs[i].d_Q_next, buf_int_full,  gpu_id, 0));
            CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(bufs[i].d_S,      buf_int_full,  gpu_id, 0));
            CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(bufs[i].d_S_ends, buf_se_full,   gpu_id, 0));
            CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(bufs[i].d_delta,  buf_dbl_full,  gpu_id, 0));
            CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(bufs[i].d_depth,  buf_dpth_full, gpu_id, 0));
        }

        // フェーズ計測イベントの初期化
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_bfs_s));
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_bfs_e));
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_back_e));
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_pref_s));
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_pref_e));
        bufs[i].bfs_ms = 0.0f;
        bufs[i].back_ms = 0.0f;
        bufs[i].pref_ms = 0.0f;
        bufs[i].prev_batch = 0;
        bufs[i].prev_sub_batch_n   = 0;
        bufs[i].prev_sub_batch_off = 0;
    }

    // CB も HBM3 に事前転送
    nvtxRangePushA("Prefetch_dynamic_to_GPU_opt");
    CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(CB_managed,
        (size_t)n_nodes * sizeof(double), gpu_id, 0));

    CUDA_ERR_CHK(cudaDeviceSynchronize());
    nvtxRangePop(); // Prefetch_dynamic_to_GPU_opt

    // ストリーム作成
    cudaStream_t streams[NS] = {};
    for (int i = 0; i < NS_eff; i++)
        CUDA_ERR_CHK(cudaStreamCreate(&streams[i]));
    cudaStream_t pref_st = 0;
    cudaEvent_t pref_done = 0;
    if (oversubscribed) {
        CUDA_ERR_CHK(cudaStreamCreate(&pref_st));
        CUDA_ERR_CHK(cudaEventCreate(&pref_done));
    }

    int *d_overflow;
    CUDA_ERR_CHK(cudaMallocManaged(&d_overflow, sizeof(int)));
    *d_overflow = 0;

    // ============================================================
    //  メインループ: 2ストリーム交互処理 (BFS + バックワードを分割計測)
    //
    //  Stream0: [memset buf0] → [bfs_kernel s=0..B] → [back_kernel s=0..B] → ...
    //  Stream1:               → [memset buf1] → [bfs_kernel s=B..2B] → [back_kernel] → ...
    //
    //  memset と bfs_kernel のオーバーラップは2ストリーム間で維持される。
    //  バッファ si の再利用前に前回のイベント時間を回収して累積する。
    // ============================================================
    bool buf_used[NS] = {};
    for (int s_start = 0; s_start < n_nodes; s_start += BATCH_PER_STREAM) {
        int          si         = (s_start / BATCH_PER_STREAM) % NS_eff;
        cudaStream_t st         = streams[si];
        int          curr_batch = min(BATCH_PER_STREAM, n_nodes - s_start);
        if (curr_batch <= 0) continue;

        // 同じバッファを再利用する前に前回バッチのフェーズ時間を回収
        // (このストリームだけ同期するため、もう一方のストリームとのオーバーラップは維持)
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

            if (oversubscribed) {
                if (sub_off == 0) {
                    // 初回サブバッチの prefetch を計測 (kernel と直列、stall)
                    CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_pref_s, pref_st));
                    prefetch_subbatch(bufs[si], sub_off, sub_n,
                                      n_nodes, max_depth_estimate, gpu_id, pref_st);
                    CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_pref_e, pref_st));
                    CUDA_ERR_CHK(cudaEventSynchronize(bufs[si].ev_pref_e));
                    float p_ms = 0.0f;
                    CUDA_ERR_CHK(cudaEventElapsedTime(&p_ms,
                        bufs[si].ev_pref_s, bufs[si].ev_pref_e));
                    bufs[si].pref_ms += p_ms;
                }
                CUDA_ERR_CHK(cudaEventRecord(pref_done, pref_st));
                CUDA_ERR_CHK(cudaStreamWaitEvent(st, pref_done, 0));
            }

            // [最適化6a] 同じバッファ範囲を再利用する場合は到達済み頂点のみリセット
            nvtxRangePushA("Memset_or_reset_subbatch_opt");
            if (bufs[si].prev_sub_batch_n   == sub_n
             && bufs[si].prev_sub_batch_off == sub_off
             && bufs[si].prev_sub_batch_n   >  0) {
                reset_visited_batch_kernel<<<sub_n, tpb, 0, st>>>(
                    bufs[si].d_d     + (size_t)sub_off * n_nodes,
                    bufs[si].d_sigma + (size_t)sub_off * n_nodes,
                    bufs[si].d_delta + (size_t)sub_off * n_nodes,
                    bufs[si].d_S     + (size_t)sub_off * n_nodes,
                    n_nodes,
                    bufs[si].d_depth  + sub_off,
                    bufs[si].d_S_ends + (size_t)sub_off * (max_depth_estimate + 1),
                    max_depth_estimate);
            } else {
                memset_subbatch(bufs[si], sub_off, sub_n, n_nodes, st);
            }
            nvtxRangePop(); // Memset_or_reset_subbatch_opt

            if (first_event) {
                CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_bfs_s, st));
                first_event = false;
            }

            const size_t off_int = (size_t)sub_off * n_nodes;
            const size_t off_se  = (size_t)sub_off * (max_depth_estimate + 1);
            int s_eff = s_start + sub_off;

            // BFS フェーズ (avg_deg < 5: shared-frontier, それ以外: 1ブロック=1ソース)
            if (avg_deg < 5.0) {
                nvtxRangePushA("BFS_kernel_shared_frontier");
                const int K          = K_SOURCES_PER_BLOCK;
                const int tpb_shared = K * THREADS_PER_SOURCE; // 256
                int grid_shared      = (sub_n + K - 1) / K;
                brandes_bfs_kernel_shared_frontier<<<grid_shared, tpb_shared, 0, st>>>(
                    R_m, C_m, n_nodes, max_depth_estimate, d_overflow,
                    bufs[si].d_d      + off_int, bufs[si].d_sigma  + off_int,
                    bufs[si].d_Q_curr + off_int, bufs[si].d_Q_next + off_int,
                    bufs[si].d_S      + off_int, bufs[si].d_S_ends + off_se,
                    bufs[si].d_depth  + sub_off, s_eff, sub_n);
                nvtxRangePop(); // BFS_kernel_shared_frontier
            } else {
                nvtxRangePushA("BFS_kernel_opt");
                brandes_bfs_kernel_opt<<<sub_n, tpb, 0, st>>>(
                    R_m, C_m, n_nodes, max_depth_estimate, d_overflow,
                    bufs[si].d_d      + off_int, bufs[si].d_sigma  + off_int,
                    bufs[si].d_Q_curr + off_int, bufs[si].d_Q_next + off_int,
                    bufs[si].d_S      + off_int, bufs[si].d_S_ends + off_se,
                    bufs[si].d_depth  + sub_off, s_eff);
                nvtxRangePop(); // BFS_kernel_opt
            }
            CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_bfs_e, st));

            if (oversubscribed) {
                int next_off = sub_off + SUB_BATCH;
                if (next_off < curr_batch) {
                    int next_n = min(SUB_BATCH, curr_batch - next_off);
                    // 並行 prefetch (kernel と重なる; 累積 raw 時間を測定し隠蔽率の議論に利用)
                    CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_pref_s, pref_st));
                    prefetch_subbatch(bufs[si], next_off, next_n,
                                      n_nodes, max_depth_estimate, gpu_id, pref_st);
                    CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_pref_e, pref_st));
                    CUDA_ERR_CHK(cudaEventSynchronize(bufs[si].ev_pref_e));
                    float p_ms = 0.0f;
                    CUDA_ERR_CHK(cudaEventElapsedTime(&p_ms,
                        bufs[si].ev_pref_s, bufs[si].ev_pref_e));
                    bufs[si].pref_ms += p_ms;
                }
            }

            // バックワードフェーズ (avg_deg < 8: thread-per-vertex, それ以外: warp-per-vertex)
            nvtxRangePushA("Backward_kernel_opt");
            if (avg_deg < 8.0) {
                brandes_back_kernel_tpv_opt<<<sub_n, tpb, 0, st>>>(
                    R_m, C_m, CB_managed, n_nodes, max_depth_estimate,
                    bufs[si].d_d     + off_int, bufs[si].d_sigma  + off_int,
                    bufs[si].d_delta + off_int,
                    bufs[si].d_S     + off_int, bufs[si].d_S_ends + off_se,
                    bufs[si].d_depth + sub_off, s_eff);
            } else {
                brandes_back_kernel_opt<<<sub_n, tpb, 0, st>>>(
                    R_m, C_m, CB_managed, n_nodes, max_depth_estimate,
                    bufs[si].d_d     + off_int, bufs[si].d_sigma  + off_int,
                    bufs[si].d_delta + off_int,
                    bufs[si].d_S     + off_int, bufs[si].d_S_ends + off_se,
                    bufs[si].d_depth + sub_off, s_eff);
            }
            nvtxRangePop(); // Backward_kernel_opt

            // [P1-C] 最後のサブバッチでは evict 不要 (読み返さないので無駄)
            if (oversubscribed && (sub_off + SUB_BATCH < curr_batch)) {
                evict_subbatch_to_host(bufs[si], sub_off, sub_n,
                                       n_nodes, max_depth_estimate, st);
            }
            CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_back_e, st));

            // [P0-A] reset_visited gating 用に直前サブバッチ範囲を記録
            bufs[si].prev_sub_batch_n   = sub_n;
            bufs[si].prev_sub_batch_off = sub_off;
        }

        buf_used[si] = true;
        bufs[si].prev_batch = curr_batch;
        CUDA_ERR_CHK(cudaPeekAtLastError());
    }

    // 全ストリームの完了を待機し、最後のバッチのフェーズ時間を回収
    float wall_bfs_ms = 0.0f, wall_back_ms = 0.0f;
    float total_bfs_ms = 0.0f, total_back_ms = 0.0f;
    float wall_pref_ms = 0.0f, total_pref_ms = 0.0f;
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
        wall_pref_ms   = max(wall_pref_ms, bufs[i].pref_ms);
        total_pref_ms += bufs[i].pref_ms;
    }

    if (*d_overflow) {
        fprintf(stderr, "\n[ERROR] GPU Memory Bounds Exceeded: BFS depth exceeded max_depth_estimate (%d).\n", max_depth_estimate);
        fprintf(stderr, "Fallback: Please process this graph with sequential CPU BC or increase max_depth_estimate.\n\n");
        exit(1);
    }

    double d2h_start = omp_get_wtime();
    vector<double> result(CB_managed, CB_managed + n_nodes);
    double d2h_end = omp_get_wtime();
    double d2h_time = d2h_end - d2h_start;

    fprintf(stderr,
        "  > [GPU Phase] BFS wall=%.4f s (cum=%.4f s), Backward wall=%.4f s (cum=%.4f s), "
        "Prefetch cum=%.4f s, SUB_BATCH=%d, num_subs=%d\n",
        wall_bfs_ms / 1000.0f, total_bfs_ms / 1000.0f,
        wall_back_ms / 1000.0f, total_back_ms / 1000.0f,
        total_pref_ms / 1000.0f, SUB_BATCH, num_subs);
#ifdef DEBUG_BRANDES
    fprintf(stderr,
        "  > [GPU Phase] H2D (UM lazy) wall=%.4f s, BFS wall=%.4f s (cum=%.4f s), Backward wall=%.4f s (cum=%.4f s), D2H (UM fault) wall=%.4f s\n",
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
        CUDA_ERR_CHK(cudaEventDestroy(bufs[i].ev_pref_s));
        CUDA_ERR_CHK(cudaEventDestroy(bufs[i].ev_pref_e));
        CUDA_ERR_CHK(cudaStreamDestroy(streams[i]));
    }
    if (oversubscribed) {
        CUDA_ERR_CHK(cudaEventDestroy(pref_done));
        CUDA_ERR_CHK(cudaStreamDestroy(pref_st));
    }
    CUDA_ERR_CHK(cudaFree(d_overflow));

    return result;
}

// ============================================================
//  公開エントリポイント (brandes.h の共通インターフェース)
// ============================================================
vector<double> brandes_gpu_opt(Graph &G)
{
    int *R        = G.getAdjacencyListPointers();
    int *C        = G.getAdjacencyList();
    int  n_nodes  = G.getNodeCount();
    int  edge_size = 2 * G.getEdgeCount();

    int num_gpus;
    CUDA_ERR_CHK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        cerr << "No GPU found" << endl;
        exit(EXIT_FAILURE);
    }
    CUDA_ERR_CHK(cudaSetDevice(0));

    double h2d_start = omp_get_wtime();

    // CSR トポロジデータを cudaMallocManaged に再確保
    int *R_m, *C_m;
    CUDA_ERR_CHK(cudaMallocManaged(&R_m, (size_t)(n_nodes + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&C_m, (size_t)edge_size     * sizeof(int)));
    memcpy(R_m, R, (size_t)(n_nodes + 1) * sizeof(int));
    memcpy(C_m, C, (size_t)edge_size     * sizeof(int));

    // [最適化1] SetReadMostly: HBM3 L2 への複製を許可
    //   バッチ内の複数ソースが同一隣接リストを参照 → 初回フェッチ以降は L2 から供給
    CUDA_ERR_CHK(cuda_mem_advise_device_compat(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                               cudaMemAdviseSetReadMostly, 0));
    CUDA_ERR_CHK(cuda_mem_advise_device_compat(C_m, (size_t)edge_size     * sizeof(int),
                               cudaMemAdviseSetReadMostly, 0));

    // [最適化2] グラフサイズ適応型メモリ配置
    //   小グラフ (topo < HBM3 総容量の 35%): HBM3 に直接配置 → NVLink-C2C レイテンシなし
    //   大グラフ                           : CPU LPDDR5X に固定 + NVLink-C2C 経由アクセス
    cudaDeviceProp prop;
    CUDA_ERR_CHK(cudaGetDeviceProperties(&prop, 0));
    const size_t topo_bytes = (size_t)(n_nodes + 1) * sizeof(int)
                            + (size_t)edge_size      * sizeof(int);
    const bool   topo_on_gpu = (topo_bytes < (size_t)(prop.totalGlobalMem * 0.35));

    if (topo_on_gpu) {
        // 小グラフ: SetAccessedBy + PrefetchAsync で HBM3 に直接配置
        CUDA_ERR_CHK(cuda_mem_advise_device_compat(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
        CUDA_ERR_CHK(cuda_mem_advise_device_compat(C_m, (size_t)edge_size     * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
        CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(R_m, (size_t)(n_nodes + 1) * sizeof(int), 0, 0));
        CUDA_ERR_CHK(cuda_mem_prefetch_async_compat(C_m, (size_t)edge_size     * sizeof(int), 0, 0));
    } else {
        // 大グラフ: CPU LPDDR5X に固定 → NVLink-C2C (900 GB/s) + SetReadMostly で L2 キャッシュ
        CUDA_ERR_CHK(cuda_mem_advise_host_compat(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                                   cudaMemAdviseSetPreferredLocation));
        CUDA_ERR_CHK(cuda_mem_advise_device_compat(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
        CUDA_ERR_CHK(cuda_mem_advise_host_compat(C_m, (size_t)edge_size     * sizeof(int),
                                   cudaMemAdviseSetPreferredLocation));
        CUDA_ERR_CHK(cuda_mem_advise_device_compat(C_m, (size_t)edge_size     * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
    }

    // 結果バッファ: HBM3 に配置 (アトミック操作が高速な HBM3 上に強制配置)
    double *CB_managed;
    CUDA_ERR_CHK(cudaMallocManaged(&CB_managed, (size_t)n_nodes * sizeof(double)));
    CUDA_ERR_CHK(cuda_mem_advise_device_compat(CB_managed, (size_t)n_nodes * sizeof(double),
                               cudaMemAdviseSetPreferredLocation, 0));
    memset(CB_managed, 0, (size_t)n_nodes * sizeof(double));

    double h2d_end = omp_get_wtime();
    double h2d_time = h2d_end - h2d_start;

    vector<double> result = brandes_gpu_opt_impl(R_m, C_m, CB_managed,
                                                  n_nodes, edge_size, 0, h2d_time);

    CUDA_ERR_CHK(cudaFree(R_m));
    CUDA_ERR_CHK(cudaFree(C_m));
    CUDA_ERR_CHK(cudaFree(CB_managed));

    return result;
}
