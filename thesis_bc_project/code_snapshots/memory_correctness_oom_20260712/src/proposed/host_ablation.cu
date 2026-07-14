// ============================================================
//  host_ablation.cu — アブレーション実験用ホスト制御
//
//  提案手法の 3 つの工夫を C++ テンプレート引数でコンパイル時に ON/OFF する:
//    HYBRID     : ハイブリッド (方向最適化) BFS  → brandes_bfs_kernel_opt<HYBRID>
//    WARP_COOP  : Warp 協調依存蓄積            → warp 版 vs thread-per-vertex 版
//    ASYNC_INIT : 2 ストリーム非同期初期化      → NS = 2 (ダブルバッファ) vs 1
//
//  カーネル内では 3 フラグを一切分岐しない (ブランチダイバージェンス回避)。
//  ランタイムの構成は brandes_gpu_opt_ablation() が 8 通りの実体へ振り分ける。
// ============================================================

#include "common.hpp"
#include "graph.hpp"
#include "brandes_kernels.cuh"
#include "ablation_config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

using namespace std;

namespace {

constexpr bool IS_UNDIRECTED = true;

// --- CUDA 12/13 互換: prefetch / advise ---
inline cudaError_t prefetch_to_gpu(const void* p, size_t bytes, int dev, cudaStream_t st = 0)
{
#if CUDART_VERSION >= 13000
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = dev;
    return cudaMemPrefetchAsync(p, bytes, loc, 0u, st);
#else
    return cudaMemPrefetchAsync(p, bytes, dev, st);
#endif
}

inline cudaError_t advise_read_mostly(const void* p, size_t bytes, int dev)
{
#if CUDART_VERSION >= 13000
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = dev;
    return cudaMemAdvise(p, bytes, cudaMemAdviseSetReadMostly, loc);
#else
    return cudaMemAdvise(p, bytes, cudaMemAdviseSetReadMostly, dev);
#endif
}

int choose_tpb(const cudaDeviceProp& prop, double avg_deg)
{
    int tpb;
    if (avg_deg < 5.0)        tpb = 128;
    else if (avg_deg < 20.0)  tpb = 256;
    else                      tpb = 512;
    tpb = min(tpb, prop.maxThreadsPerBlock);
    tpb = max(tpb, 32);
    tpb = (tpb / 32) * 32;
    return tpb;
}

int choose_max_depth(double avg_deg)
{
    if (avg_deg < 5.0)       return 4096;
    else if (avg_deg < 20.0) return 256;
    else                     return 64;
}

struct AblBuf {
    int    *d_d = nullptr, *d_Q_curr = nullptr, *d_Q_next = nullptr;
    int    *d_S = nullptr, *d_S_ends = nullptr, *d_depth = nullptr;
    double *d_sigma = nullptr, *d_delta = nullptr;
    cudaEvent_t ev_bfs_s, ev_bfs_e, ev_back_e;
    float bfs_ms = 0.0f, back_ms = 0.0f;
    bool  used = false;
};

// ============================================================
//  テンプレート化された処理本体
// ============================================================
template <bool HYBRID, bool WARP_COOP, bool ASYNC_INIT>
vector<double> ablation_impl(int* R_m, int* C_m, double* CB_managed,
                             int n_nodes, int edge_size, int gpu_id)
{
    cudaDeviceProp prop;
    CUDA_ERR_CHK(cudaGetDeviceProperties(&prop, gpu_id));

    const double avg_deg = (n_nodes > 0) ? ((double)edge_size / n_nodes) : 0.0;
    const int tpb                = choose_tpb(prop, avg_deg);
    const int max_depth_estimate = choose_max_depth(avg_deg);

    // ③ 2 ストリーム非同期初期化: ON なら NS=2 (ダブルバッファ), OFF なら NS=1
    constexpr int NS = ASYNC_INIT ? 2 : 1;

    size_t free_mem = 0, total_mem = 0;
    CUDA_ERR_CHK(cudaMemGetInfo(&free_mem, &total_mem));
    const size_t per_batch_mem =
          (size_t)n_nodes * (3 * sizeof(int) + 2 * sizeof(double)) // d_d,d_Q_curr,d_Q_next,d_sigma,d_delta
        + (size_t)n_nodes * sizeof(int)                            // d_S
        + (size_t)(max_depth_estimate + 1) * sizeof(int)           // d_S_ends
        + sizeof(int);                                             // d_depth
    const size_t safety    = (size_t)(free_mem * 0.15);
    const size_t available = (free_mem > safety) ? (free_mem - safety) : 0;
    int BATCH_PER_STREAM = (int)(available / ((size_t)NS * per_batch_mem));
    BATCH_PER_STREAM = max(1, min(BATCH_PER_STREAM, 512));
    if (const char* env = getenv("BC_BATCH_OVERRIDE")) {
        int v = atoi(env);
        if (v > 0) BATCH_PER_STREAM = max(1, v);
    }
    const int safe_batch = (n_nodes > 0)
        ? max(1, std::numeric_limits<int>::max() / n_nodes) : 1;
    BATCH_PER_STREAM = min(BATCH_PER_STREAM, safe_batch);

    fprintf(stderr,
        "  > [Ablation H%d W%d A%d] NS=%d, tpb=%d, BATCH=%d, max_depth=%d, avg_deg=%.2f\n",
        (int)HYBRID, (int)WARP_COOP, (int)ASYNC_INIT, NS, tpb, BATCH_PER_STREAM,
        max_depth_estimate, avg_deg);

    const size_t buf_int = (size_t)BATCH_PER_STREAM * n_nodes * sizeof(int);
    const size_t buf_dbl = (size_t)BATCH_PER_STREAM * n_nodes * sizeof(double);
    const size_t buf_se  = (size_t)BATCH_PER_STREAM * (max_depth_estimate + 1) * sizeof(int);
    const size_t buf_dp  = (size_t)BATCH_PER_STREAM * sizeof(int);

    AblBuf bufs[NS];
    cudaStream_t streams[NS];
    for (int i = 0; i < NS; i++) {
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_d,      buf_int));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_sigma,  buf_dbl));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_Q_curr, buf_int));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_Q_next, buf_int));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_S,      buf_int));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_S_ends, buf_se));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_delta,  buf_dbl));
        CUDA_ERR_CHK(cudaMallocManaged(&bufs[i].d_depth,  buf_dp));
        prefetch_to_gpu(bufs[i].d_d,      buf_int, gpu_id);
        prefetch_to_gpu(bufs[i].d_sigma,  buf_dbl, gpu_id);
        prefetch_to_gpu(bufs[i].d_Q_curr, buf_int, gpu_id);
        prefetch_to_gpu(bufs[i].d_Q_next, buf_int, gpu_id);
        prefetch_to_gpu(bufs[i].d_S,      buf_int, gpu_id);
        prefetch_to_gpu(bufs[i].d_S_ends, buf_se,  gpu_id);
        prefetch_to_gpu(bufs[i].d_delta,  buf_dbl, gpu_id);
        prefetch_to_gpu(bufs[i].d_depth,  buf_dp,  gpu_id);
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_bfs_s));
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_bfs_e));
        CUDA_ERR_CHK(cudaEventCreate(&bufs[i].ev_back_e));
        CUDA_ERR_CHK(cudaStreamCreate(&streams[i]));
    }

    prefetch_to_gpu(CB_managed, (size_t)n_nodes * sizeof(double), gpu_id);
    CUDA_ERR_CHK(cudaDeviceSynchronize());

    int* d_overflow = nullptr;
    CUDA_ERR_CHK(cudaMallocManaged(&d_overflow, sizeof(int)));
    *d_overflow = 0;

    // メインループ: NS 本のストリームを round-robin
    for (int s_start = 0; s_start < n_nodes; s_start += BATCH_PER_STREAM) {
        const int si         = (s_start / BATCH_PER_STREAM) % NS;
        cudaStream_t st      = streams[si];
        const int curr_batch = min(BATCH_PER_STREAM, n_nodes - s_start);
        if (curr_batch <= 0) continue;
        AblBuf& b = bufs[si];

        // 同一バッファ再利用前に前回バッチのフェーズ時間を回収 (該当ストリームのみ同期)
        if (b.used) {
            CUDA_ERR_CHK(cudaEventSynchronize(b.ev_back_e));
            float a = 0.0f, c = 0.0f;
            CUDA_ERR_CHK(cudaEventElapsedTime(&a, b.ev_bfs_s, b.ev_bfs_e));
            CUDA_ERR_CHK(cudaEventElapsedTime(&c, b.ev_bfs_e, b.ev_back_e));
            b.bfs_ms += a;
            b.back_ms += c;
        }

        // ホスト側 cudaMemsetAsync による初期化 (d=-1, sigma=0, delta=0)。
        // NS=2 のとき、この memset は他ストリームのカーネルと Copy Engine 上で並走する。
        const size_t cb_int = (size_t)curr_batch * n_nodes * sizeof(int);
        const size_t cb_dbl = (size_t)curr_batch * n_nodes * sizeof(double);
        CUDA_ERR_CHK(cudaMemsetAsync(b.d_d,     0xFF, cb_int, st));
        CUDA_ERR_CHK(cudaMemsetAsync(b.d_sigma, 0,    cb_dbl, st));
        CUDA_ERR_CHK(cudaMemsetAsync(b.d_delta, 0,    cb_dbl, st));

        CUDA_ERR_CHK(cudaEventRecord(b.ev_bfs_s, st));

        // ① 前向き BFS (HYBRID で方向最適化を切替)
        brandes_bfs_kernel_opt<HYBRID><<<curr_batch, tpb, 0, st>>>(
            R_m, C_m, n_nodes, max_depth_estimate, d_overflow,
            b.d_d, b.d_sigma, b.d_Q_curr, b.d_Q_next,
            b.d_S, b.d_S_ends, b.d_depth, s_start);
        CUDA_ERR_CHK(cudaEventRecord(b.ev_bfs_e, st));

        // ② 後向き依存蓄積 (WARP_COOP でコンパイル時に実装を選択)
        if constexpr (WARP_COOP) {
            brandes_back_kernel_opt<IS_UNDIRECTED><<<curr_batch, tpb, 0, st>>>(
                R_m, C_m, CB_managed, n_nodes, max_depth_estimate,
                b.d_d, b.d_sigma, b.d_delta, b.d_S, b.d_S_ends, b.d_depth, s_start);
        } else {
            brandes_back_kernel_tpv_opt<IS_UNDIRECTED><<<curr_batch, tpb, 0, st>>>(
                R_m, C_m, CB_managed, n_nodes, max_depth_estimate,
                b.d_d, b.d_sigma, b.d_delta, b.d_S, b.d_S_ends, b.d_depth, s_start);
        }
        CUDA_ERR_CHK(cudaEventRecord(b.ev_back_e, st));

        b.used = true;
        CUDA_ERR_CHK(cudaPeekAtLastError());
    }

    // 全ストリーム完了待ちと最終バッチのフェーズ時間回収
    float total_bfs = 0.0f, total_back = 0.0f;
    for (int i = 0; i < NS; i++) {
        if (!bufs[i].used) continue;
        CUDA_ERR_CHK(cudaStreamSynchronize(streams[i]));
        float a = 0.0f, c = 0.0f;
        CUDA_ERR_CHK(cudaEventElapsedTime(&a, bufs[i].ev_bfs_s, bufs[i].ev_bfs_e));
        CUDA_ERR_CHK(cudaEventElapsedTime(&c, bufs[i].ev_bfs_e, bufs[i].ev_back_e));
        bufs[i].bfs_ms += a;
        bufs[i].back_ms += c;
        total_bfs  += bufs[i].bfs_ms;
        total_back += bufs[i].back_ms;
    }

    if (*d_overflow) {
        fprintf(stderr, "\n[ERROR] BFS depth exceeded max_depth_estimate (%d).\n", max_depth_estimate);
        exit(1);
    }

    vector<double> result(CB_managed, CB_managed + n_nodes);

    fprintf(stderr, "  > [Ablation Phase] BFS cum=%.4f s, Backward cum=%.4f s\n",
            total_bfs / 1000.0f, total_back / 1000.0f);

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
    CUDA_ERR_CHK(cudaFree(d_overflow));

    return result;
}

// ============================================================
//  ランタイム構成 → コンパイル時テンプレート実体への振り分け (3 段カスケード)
//  カーネル内で分岐しないため、各実体はブランチのない専用コードになる。
// ============================================================
template <bool HYBRID, bool WARP_COOP>
vector<double> dispatch_async(bool async_init,
                              int* R_m, int* C_m, double* CB, int n, int e, int gpu)
{
    return async_init ? ablation_impl<HYBRID, WARP_COOP, true >(R_m, C_m, CB, n, e, gpu)
                      : ablation_impl<HYBRID, WARP_COOP, false>(R_m, C_m, CB, n, e, gpu);
}

template <bool HYBRID>
vector<double> dispatch_warp(bool warp_coop, bool async_init,
                             int* R_m, int* C_m, double* CB, int n, int e, int gpu)
{
    return warp_coop ? dispatch_async<HYBRID, true >(async_init, R_m, C_m, CB, n, e, gpu)
                     : dispatch_async<HYBRID, false>(async_init, R_m, C_m, CB, n, e, gpu);
}

vector<double> dispatch_hybrid(bool hybrid, bool warp_coop, bool async_init,
                               int* R_m, int* C_m, double* CB, int n, int e, int gpu)
{
    return hybrid ? dispatch_warp<true >(warp_coop, async_init, R_m, C_m, CB, n, e, gpu)
                  : dispatch_warp<false>(warp_coop, async_init, R_m, C_m, CB, n, e, gpu);
}

} // namespace

// ============================================================
//  公開エントリポイント
// ============================================================
std::string ablation_label(const AblationConfig& cfg)
{
    char buf[16];
    snprintf(buf, sizeof(buf), "H%d_W%d_A%d",
             cfg.hybrid_bfs ? 1 : 0, cfg.warp_coop ? 1 : 0, cfg.async_init ? 1 : 0);
    return std::string(buf);
}

std::string ablation_describe(const AblationConfig& cfg)
{
    std::string s;
    s += "hybrid="; s += (cfg.hybrid_bfs ? "ON " : "OFF");
    s += " warp=";  s += (cfg.warp_coop  ? "ON " : "OFF");
    s += " async="; s += (cfg.async_init ? "ON " : "OFF");
    return s;
}

std::vector<double> brandes_gpu_opt_ablation(Graph& G, const AblationConfig& cfg)
{
    int* R         = G.getAdjacencyListPointers();
    int* C         = G.getAdjacencyList();
    int  n_nodes   = G.getNodeCount();
    int  edge_size = 2 * G.getEdgeCount();

    int num_gpus = 0;
    CUDA_ERR_CHK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        cerr << "No GPU found" << endl;
        exit(EXIT_FAILURE);
    }
    CUDA_ERR_CHK(cudaSetDevice(0));

    // CSR トポロジ + 結果バッファを Managed に確保 (全構成で共通の前処理)
    int*    R_m = nullptr;
    int*    C_m = nullptr;
    double* CB_managed = nullptr;
    CUDA_ERR_CHK(cudaMallocManaged(&R_m, (size_t)(n_nodes + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&C_m, (size_t)edge_size     * sizeof(int)));
    memcpy(R_m, R, (size_t)(n_nodes + 1) * sizeof(int));
    memcpy(C_m, C, (size_t)edge_size     * sizeof(int));
    advise_read_mostly(R_m, (size_t)(n_nodes + 1) * sizeof(int), 0);
    advise_read_mostly(C_m, (size_t)edge_size     * sizeof(int), 0);
    prefetch_to_gpu(R_m, (size_t)(n_nodes + 1) * sizeof(int), 0);
    prefetch_to_gpu(C_m, (size_t)edge_size     * sizeof(int), 0);

    CUDA_ERR_CHK(cudaMallocManaged(&CB_managed, (size_t)n_nodes * sizeof(double)));
    memset(CB_managed, 0, (size_t)n_nodes * sizeof(double));

    std::vector<double> result = dispatch_hybrid(
        cfg.hybrid_bfs, cfg.warp_coop, cfg.async_init,
        R_m, C_m, CB_managed, n_nodes, edge_size, 0);

    CUDA_ERR_CHK(cudaFree(R_m));
    CUDA_ERR_CHK(cudaFree(C_m));
    CUDA_ERR_CHK(cudaFree(CB_managed));

    return result;
}
