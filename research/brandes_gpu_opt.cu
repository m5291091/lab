#include "common.h"
#include "brandes.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

namespace cg = cooperative_groups;
using namespace std;

// ============================================================
//  brandes_gpu_opt.cu
//  GH200 Grace Hopper Superchip 特化 BC 最適化実装
//
//  brandes_gpu_managed.cu (v1) からの改善点:
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

// ============================================================
//  デバイス関数: BFS 前向き探索 (トップダウン/ボトムアップ ハイブリッド)
//  ロジックは brandes_gpu_managed.cu と同一。NVLink-C2C 経由の R/C アクセスに対し
//  SetReadMostly + SetAccessedBy により HBM3 L2 がキャッシュラインを保持する。
// ============================================================
__device__ void find_shortest_paths_opt(
        int *R, int *C, int *d_d, int *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, int batch_idx, int n_nodes, int &Q_curr_len,
        int &Q_next_len, int &S_len, int &S_ends_len, int &depth)
{
    int tid   = threadIdx.x;
    int bsize = blockDim.x;
    int v, w;

    while (true) {
        // フロンティアサイズに基づく探索方向の切り替え
        // 小フロンティア: トップダウン (辺ごとに処理)
        // 大フロンティア: ボトムアップ (未訪問頂点ごとに処理) → キャッシュ効率向上
        int threshold = min(max(n_nodes / 20, 32), 1024);

        if (Q_curr_len <= threshold) {
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
                    int sum_sigma = 0;
                    for (int j = R[w]; j < R[w + 1]; j++) {
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
            if (tid == 0)
                depth = d_d[batch_idx * n_nodes + d_S[batch_idx * n_nodes + S_len - 1]];
            break;
        }

        int curr_Q_next_len = Q_next_len;
        for (int i = tid; i < curr_Q_next_len; i += bsize) {
            d_Q_curr[batch_idx * n_nodes + i]        = d_Q_next[batch_idx * n_nodes + i];
            d_S[batch_idx * n_nodes + S_len + i]     = d_Q_next[batch_idx * n_nodes + i];
        }
        __syncthreads();

        if (tid == 0) {
            d_S_ends[batch_idx * (n_nodes + 1) + S_ends_len] = S_len + curr_Q_next_len;
            S_ends_len++;
            Q_curr_len = curr_Q_next_len;
            S_len     += curr_Q_next_len;
            Q_next_len = 0;
            depth++;
        }
        __syncthreads();
    }
}

// ============================================================
//  デバイス関数: BC 後向き依存集計
//  Warp レベルシャッフル還元により各頂点の依存値を効率的に集計。
// ============================================================
__device__ void accumulate_dependencies_opt(
        int *R, int *C, int *d_d, int *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, int batch_idx, int n_nodes, int &depth)
{
    auto block     = cg::this_thread_block();
    auto warp      = cg::tiled_partition<32>(block);
    int  tid_block = block.thread_rank();
    int  warp_id   = tid_block / warp.size();
    int  num_warps = block.size() / warp.size();

    while (depth > 0) {
        int start          = d_S_ends[batch_idx * (n_nodes + 1) + depth];
        int end            = d_S_ends[batch_idx * (n_nodes + 1) + depth + 1];
        int nodes_in_level = end - start;

        for (int i = warp_id; i < nodes_in_level; i += num_warps) {
            int    w         = d_S[batch_idx * n_nodes + start + i];
            double sigma_w   = (double)d_sigma[batch_idx * n_nodes + w];
            double local_sum = 0.0;

            for (int j = R[w] + warp.thread_rank(); j < R[w + 1]; j += warp.size()) {
                int v = C[j];
                if (d_d[batch_idx * n_nodes + v] == d_d[batch_idx * n_nodes + w] + 1) {
                    local_sum += (sigma_w / (double)d_sigma[batch_idx * n_nodes + v])
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

// ============================================================
//  GPU カーネル
//
//  [最適化3] ホスト側 cudaMemsetAsync により以下が設定済み:
//    d_d    : 全要素 -1  (0xFFFFFFFF)
//    d_sigma: 全要素  0
//    d_delta: 全要素  0.0
//  カーネル内ではソース頂点 s の 1 点セットアップ (O(1)) のみ実施。
// ============================================================
// BFS フェーズカーネル (最適化版): O(1) 初期化 (memsetAsync 済み) + BFS + 深さ保存
__global__ void brandes_bfs_kernel_opt(
        int *R, int *C, int n_nodes,
        int *d_d, int *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, int *d_depth, int s_start)
{
    int batch_idx = blockIdx.x;
    int s         = s_start + batch_idx;
    int tid       = threadIdx.x;

    __shared__ int Q_curr_len, Q_next_len, S_len, S_ends_len, depth;

    // ホスト側 memsetAsync で配列は初期化済み。ソース頂点のみ上書き (O(1))。
    if (tid == 0) {
        d_d   [batch_idx * n_nodes + s] = 0;
        d_sigma[batch_idx * n_nodes + s] = 1;
        d_Q_curr[batch_idx * n_nodes]   = s;
        Q_curr_len  = 1;
        Q_next_len  = 0;
        d_S[batch_idx * n_nodes]        = s;
        S_len       = 1;
        d_S_ends[batch_idx * (n_nodes + 1)]     = 0;
        d_S_ends[batch_idx * (n_nodes + 1) + 1] = 1;
        S_ends_len  = 2;
        depth       = 0;
    }
    __syncthreads();

    find_shortest_paths_opt(R, C, d_d, d_sigma, d_Q_curr, d_Q_next,
                            d_S, d_S_ends, batch_idx, n_nodes,
                            Q_curr_len, Q_next_len, S_len, S_ends_len, depth);
    __syncthreads();

    if (tid == 0) d_depth[batch_idx] = depth;
}

// バックワードフェーズカーネル (最適化版)
__global__ void brandes_back_kernel_opt(
        int *R, int *C, double *CB, int n_nodes,
        int *d_d, int *d_sigma, double *d_delta,
        int *d_S, int *d_S_ends, const int *d_depth, int s_start)
{
    int batch_idx = blockIdx.x;
    int s         = s_start + batch_idx;
    int tid       = threadIdx.x;

    __shared__ int depth;
    if (tid == 0) depth = d_depth[batch_idx];
    __syncthreads();

    accumulate_dependencies_opt(R, C, d_d, d_sigma, d_delta,
                                d_S, d_S_ends, batch_idx, n_nodes, depth);
    __syncthreads();

    for (int v = tid; v < n_nodes; v += blockDim.x) {
        if (v != s) {
            double contrib = isUndirected_opt
                           ? d_delta[batch_idx * n_nodes + v] / 2.0
                           : d_delta[batch_idx * n_nodes + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

__global__ void brandes_kernel_opt(
        int *R, int *C, double *CB, int n_nodes,
        int *d_d, int *d_sigma, int *d_Q_curr, int *d_Q_next,
        int *d_S, int *d_S_ends, double *d_delta, int s_start)
{
    int batch_idx = blockIdx.x;
    int s         = s_start + batch_idx;
    int tid       = threadIdx.x;

    __shared__ int Q_curr_len, Q_next_len, S_len, S_ends_len, depth;

    // ホスト側 memsetAsync で配列は初期化済み。ソース頂点のみ上書き (O(1))。
    if (tid == 0) {
        d_d   [batch_idx * n_nodes + s] = 0;
        d_sigma[batch_idx * n_nodes + s] = 1;
        d_Q_curr[batch_idx * n_nodes]   = s;
        Q_curr_len  = 1;
        Q_next_len  = 0;
        d_S[batch_idx * n_nodes]        = s;
        S_len       = 1;
        d_S_ends[batch_idx * (n_nodes + 1)]     = 0;
        d_S_ends[batch_idx * (n_nodes + 1) + 1] = 1;
        S_ends_len  = 2;
        depth       = 0;
    }
    __syncthreads();

    find_shortest_paths_opt(R, C, d_d, d_sigma, d_Q_curr, d_Q_next,
                            d_S, d_S_ends, batch_idx, n_nodes,
                            Q_curr_len, Q_next_len, S_len, S_ends_len, depth);
    __syncthreads();

    accumulate_dependencies_opt(R, C, d_d, d_sigma, d_delta,
                                d_S, d_S_ends, batch_idx, n_nodes, depth);
    __syncthreads();

    // BC 値の集計 (無向グラフは 1/2)
    for (int v = tid; v < n_nodes; v += blockDim.x) {
        if (v != s) {
            double contrib = isUndirected_opt
                           ? d_delta[batch_idx * n_nodes + v] / 2.0
                           : d_delta[batch_idx * n_nodes + v];
            atomicAdd(&CB[v], contrib);
        }
    }
}

// ============================================================
//  内部実装: ダブルバッファリング + 2ストリーム処理
// ============================================================
static vector<double> brandes_gpu_opt_impl(
        int *R_m, int *C_m, double *CB_managed,
        int n_nodes, int edge_size, int gpu_id)
{
    cudaDeviceProp prop;
    CUDA_ERR_CHK(cudaGetDeviceProperties(&prop, gpu_id));

    int tpb = min(prop.maxThreadsPerBlock, n_nodes);
    tpb = (tpb / 32) * 32;
    tpb = max(tpb, 32);

    // [最適化4] NS=2 ストリームでダブルバッファリング
    const int NS = 2;

    // バッチサイズ計算: NS バッファ分の動的メモリを確保できるよう調整
    size_t free_mem, total_mem;
    CUDA_ERR_CHK(cudaMemGetInfo(&free_mem, &total_mem));

    // d_d, d_sigma, d_Q_curr, d_Q_next, d_S, d_S_ends, d_delta
    const size_t per_batch_mem =
        (size_t)n_nodes * (4 * sizeof(int) + sizeof(double))  // d_d, d_sigma, d_Q_curr, d_Q_next, d_delta
        + (size_t)n_nodes * sizeof(int)                        // d_S
        + (size_t)(n_nodes + 1) * sizeof(int);                 // d_S_ends

    const size_t safety          = (size_t)(free_mem * 0.15);
    const size_t available       = (free_mem > safety) ? (free_mem - safety) : 0;
    int          BATCH_PER_STREAM = (int)(available / ((size_t)NS * per_batch_mem));
    BATCH_PER_STREAM = max(1, min(BATCH_PER_STREAM, 512));

    // BC_BATCH_OVERRIDE 環境変数でバッチサイズを上書き (感度分析用)
    if (const char *env = getenv("BC_BATCH_OVERRIDE")) {
        int override_val = atoi(env);
        if (override_val > 0)
            BATCH_PER_STREAM = max(1, min(override_val, BATCH_PER_STREAM));
    }

    // 使用予定メモリを報告 (96 GB HBM3 + NVLink-C2C 活用の証拠)
    size_t topology_bytes = ((size_t)(n_nodes + 1) + (size_t)edge_size) * sizeof(int);
    size_t dynamic_bytes  = (size_t)NS * BATCH_PER_STREAM * per_batch_mem;
    fprintf(stderr, "  > [Mem] GPU HBM3: total=%.1f GB, free_before=%.1f GB\n",
            total_mem / 1e9, free_mem / 1e9);
    fprintf(stderr, "  > [Mem] topology(CPU/HBM3)=%.2f GB, dynamic(HBM3)=%.2f GB, batch_per_stream=%d\n",
            topology_bytes / 1e9, dynamic_bytes / 1e9, BATCH_PER_STREAM);

    // NS 組の動的ステートバッファを確保し HBM3 に配置
    struct DynBuf {
        int    *d_d, *d_sigma, *d_Q_curr, *d_Q_next, *d_S, *d_S_ends;
        double *d_delta;
        int    *d_depth;
        // フェーズ別計測イベント (BFS 開始/終了, バックワード終了)
        cudaEvent_t ev_bfs_s, ev_bfs_e, ev_back_e;
        float bfs_ms, back_ms;
    };
    DynBuf bufs[NS];

    for (int i = 0; i < NS; i++) {
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_d,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int)));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_sigma,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int)));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_Q_curr,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int)));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_Q_next,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int)));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_S,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int)));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_S_ends,
            (size_t)BATCH_PER_STREAM * (n_nodes + 1) * sizeof(int)));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_delta,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(double)));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_depth,
            (size_t)BATCH_PER_STREAM * sizeof(int)));

        // HBM3 に事前転送 (デフォルトストリームで非同期投入)
        CUDA_ERR_CHK(cudaMemPrefetchAsync(bufs[i].d_d,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int), gpu_id, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(bufs[i].d_sigma,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int), gpu_id, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(bufs[i].d_Q_curr,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int), gpu_id, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(bufs[i].d_Q_next,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int), gpu_id, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(bufs[i].d_S,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int), gpu_id, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(bufs[i].d_S_ends,
            (size_t)BATCH_PER_STREAM * (n_nodes + 1) * sizeof(int), gpu_id, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(bufs[i].d_delta,
            (size_t)BATCH_PER_STREAM * n_nodes * sizeof(double), gpu_id, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(bufs[i].d_depth,
            (size_t)BATCH_PER_STREAM * sizeof(int), gpu_id, 0));

        // フェーズ計測イベントの初期化
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_bfs_s));
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_bfs_e));
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_back_e));
        bufs[i].bfs_ms = 0.0f;
        bufs[i].back_ms = 0.0f;
    }

    // CB も HBM3 に事前転送
    nvtxRangePushA("Prefetch_dynamic_to_GPU_opt");
    CUDA_ERR_CHK(cudaMemPrefetchAsync(CB_managed,
        (size_t)n_nodes * sizeof(double), gpu_id, 0));

    CUDA_ERR_CHK(cudaDeviceSynchronize());
    nvtxRangePop(); // Prefetch_dynamic_to_GPU_opt

    // ストリーム作成
    cudaStream_t streams[NS];
    for (int i = 0; i < NS; i++)
        CUDA_ERR_CHK(cudaStreamCreate(&streams[i]));

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
        int          si         = (s_start / BATCH_PER_STREAM) % NS;
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

        // [最適化3] GPU Copy Engine で非同期 memset
        nvtxRangePushA("Memset_async_opt");
        CUDA_ERR_CHK(cudaMemsetAsync(bufs[si].d_d,
            0xFF, (size_t)curr_batch * n_nodes * sizeof(int), st));
        CUDA_ERR_CHK(cudaMemsetAsync(bufs[si].d_sigma,
            0,    (size_t)curr_batch * n_nodes * sizeof(int), st));
        CUDA_ERR_CHK(cudaMemsetAsync(bufs[si].d_delta,
            0,    (size_t)curr_batch * n_nodes * sizeof(double), st));
        nvtxRangePop(); // Memset_async_opt

        // BFS フェーズ
        CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_bfs_s, st));
        nvtxRangePushA("BFS_kernel_opt");
        brandes_bfs_kernel_opt<<<curr_batch, tpb, 0, st>>>(
            R_m, C_m, n_nodes,
            bufs[si].d_d,      bufs[si].d_sigma,
            bufs[si].d_Q_curr, bufs[si].d_Q_next,
            bufs[si].d_S,      bufs[si].d_S_ends,
            bufs[si].d_depth,  s_start);
        CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_bfs_e, st));
        nvtxRangePop(); // BFS_kernel_opt

        // バックワードフェーズ
        nvtxRangePushA("Backward_kernel_opt");
        brandes_back_kernel_opt<<<curr_batch, tpb, 0, st>>>(
            R_m, C_m, CB_managed, n_nodes,
            bufs[si].d_d,     bufs[si].d_sigma,
            bufs[si].d_delta,
            bufs[si].d_S,     bufs[si].d_S_ends,
            bufs[si].d_depth, s_start);
        CUDA_ERR_CHK(cudaEventRecord(bufs[si].ev_back_e, st));
        nvtxRangePop(); // Backward_kernel_opt

        buf_used[si] = true;
        CUDA_ERR_CHK(cudaPeekAtLastError());
    }

    // 全ストリームの完了を待機し、最後のバッチのフェーズ時間を回収
    float total_bfs_ms = 0.0f, total_back_ms = 0.0f;
    for (int i = 0; i < NS; i++) {
        if (!buf_used[i]) continue;
        CUDA_ERR_CHK(cudaStreamSynchronize(streams[i]));
        float b_ms = 0.0f, bk_ms = 0.0f;
        CUDA_ERR_CHK(cudaEventElapsedTime(&b_ms,  bufs[i].ev_bfs_s, bufs[i].ev_bfs_e));
        CUDA_ERR_CHK(cudaEventElapsedTime(&bk_ms, bufs[i].ev_bfs_e, bufs[i].ev_back_e));
        total_bfs_ms  += bufs[i].bfs_ms  + b_ms;
        total_back_ms += bufs[i].back_ms + bk_ms;
    }

    fprintf(stderr, "  > [GPU Phase] BFS: %.4f sec, Backward: %.4f sec\n",
            total_bfs_ms / 1000.0f, total_back_ms / 1000.0f);

    vector<double> result(CB_managed, CB_managed + n_nodes);

    // クリーンアップ
    for (int i = 0; i < NS; i++) {
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

    // CSR トポロジデータを cudaMallocManaged に再確保
    int *R_m, *C_m;
    CUDA_ERR_CHK(cudaMallocManaged(&R_m, (size_t)(n_nodes + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&C_m, (size_t)edge_size     * sizeof(int)));
    memcpy(R_m, R, (size_t)(n_nodes + 1) * sizeof(int));
    memcpy(C_m, C, (size_t)edge_size     * sizeof(int));

    // [最適化1] SetReadMostly: HBM3 L2 への複製を許可
    //   バッチ内の複数ソースが同一隣接リストを参照 → 初回フェッチ以降は L2 から供給
    CUDA_ERR_CHK(cudaMemAdvise(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                               cudaMemAdviseSetReadMostly, 0));
    CUDA_ERR_CHK(cudaMemAdvise(C_m, (size_t)edge_size     * sizeof(int),
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
        CUDA_ERR_CHK(cudaMemAdvise(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
        CUDA_ERR_CHK(cudaMemAdvise(C_m, (size_t)edge_size     * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(R_m, (size_t)(n_nodes + 1) * sizeof(int), 0, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(C_m, (size_t)edge_size     * sizeof(int), 0, 0));
    } else {
        // 大グラフ: CPU LPDDR5X に固定 → NVLink-C2C (900 GB/s) + SetReadMostly で L2 キャッシュ
        CUDA_ERR_CHK(cudaMemAdvise(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                                   cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_ERR_CHK(cudaMemAdvise(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
        CUDA_ERR_CHK(cudaMemAdvise(C_m, (size_t)edge_size     * sizeof(int),
                                   cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_ERR_CHK(cudaMemAdvise(C_m, (size_t)edge_size     * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
    }

    // 結果バッファ: HBM3 に配置 (アトミック操作が高速な HBM3 上に強制配置)
    double *CB_managed;
    CUDA_ERR_CHK(cudaMallocManaged(&CB_managed, (size_t)n_nodes * sizeof(double)));
    CUDA_ERR_CHK(cudaMemAdvise(CB_managed, (size_t)n_nodes * sizeof(double),
                               cudaMemAdviseSetPreferredLocation, 0));
    memset(CB_managed, 0, (size_t)n_nodes * sizeof(double));

    vector<double> result = brandes_gpu_opt_impl(R_m, C_m, CB_managed,
                                                  n_nodes, edge_size, 0);

    CUDA_ERR_CHK(cudaFree(R_m));
    CUDA_ERR_CHK(cudaFree(C_m));
    CUDA_ERR_CHK(cudaFree(CB_managed));

    return result;
}
