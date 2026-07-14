#include "common.hpp"


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


#include "common.hpp"
#include "graph.hpp"
#include "brandes_kernels.cuh"
constexpr bool USE_HYBRID_BFS = true;
constexpr bool IS_UNDIRECTED = true;

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

    // BC_FORCE_BFS_KERNEL 環境変数で BFS カーネル選択を強制上書き (カーネル選択機構の寄与測定用)
    //   shared → shared-frontier 版を強制 / block → 1ブロック=1ソース版を強制 / auto(既定) → 常時 block
    //   max_depth_estimate は avg_deg にキーされたまま (グラフ依存で正しい値) なので、カーネルのみ強制すれば安全。
    int force_bfs_kernel = -1; // -1=auto, 0=block, 1=shared
    if (const char *env = getenv("BC_FORCE_BFS_KERNEL")) {
        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return (char)((c >= 'A' && c <= 'Z') ? c + 32 : c); });
        if      (v == "shared") force_bfs_kernel = 1;
        else if (v == "block")  force_bfs_kernel = 0;
        else if (v == "auto")   force_bfs_kernel = -1;
        else fprintf(stderr,
                     "  > [BFS kernel] WARNING: unknown BC_FORCE_BFS_KERNEL='%s' "
                     "(expected shared|block|auto); falling back to auto\n", env);
    }
    // BFS カーネル選択の既定を常時 block に変更。
    //   2×2 実測 (BC_FORCE_BFS_KERNEL 強制, BATCH=512, 中央値) で全 6 グラフとも block が
    //   shared に勝利 (email-EuAll 6.2× 〜 roadNet-PA 1.52×)。ハブなし道路網でも block 優位のため
    //   avg_deg 等での自動選択は不要。詳細は docs/kernel_selection_decision.md 参照。
    //   BC_FORCE_BFS_KERNEL=shared|block の強制切替は再現実験用に温存する。
    const bool heuristic_shared = false;
    const bool use_shared_bfs   = (force_bfs_kernel < 0) ? heuristic_shared
                                                         : (force_bfs_kernel == 1);
    fprintf(stderr, "  > [BFS kernel] %s=%s (avg_deg=%.2f, heuristic=%s)\n",
            (force_bfs_kernel < 0) ? "auto" : "forced",
            use_shared_bfs ? "shared" : "block",
            avg_deg, heuristic_shared ? "shared" : "block");

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

            // BFS フェーズ (use_shared_bfs: shared-frontier, それ以外: 1ブロック=1ソース)
            // 既定は常時 block。BC_FORCE_BFS_KERNEL=shared|block による強制切替のみ可能
            // (avg_deg での自動選択は行わない)。
            if (use_shared_bfs) {
                nvtxRangePushA("BFS_kernel_shared_frontier");
                const int K          = K_SOURCES_PER_BLOCK;
                const int tpb_shared = K * THREADS_PER_SOURCE; // 256
                int grid_shared      = (sub_n + K - 1) / K;
                brandes_bfs_kernel_shared_frontier<USE_HYBRID_BFS><<<grid_shared, tpb_shared, 0, st>>>(
                    R_m, C_m, n_nodes, max_depth_estimate, d_overflow,
                    bufs[si].d_d      + off_int, bufs[si].d_sigma  + off_int,
                    bufs[si].d_Q_curr + off_int, bufs[si].d_Q_next + off_int,
                    bufs[si].d_S      + off_int, bufs[si].d_S_ends + off_se,
                    bufs[si].d_depth  + sub_off, s_eff, sub_n);
                nvtxRangePop(); // BFS_kernel_shared_frontier
            } else {
                nvtxRangePushA("BFS_kernel_opt");
                brandes_bfs_kernel_opt<USE_HYBRID_BFS><<<sub_n, tpb, 0, st>>>(
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
                brandes_back_kernel_tpv_opt<IS_UNDIRECTED><<<sub_n, tpb, 0, st>>>(
                    R_m, C_m, CB_managed, n_nodes, max_depth_estimate,
                    bufs[si].d_d     + off_int, bufs[si].d_sigma  + off_int,
                    bufs[si].d_delta + off_int,
                    bufs[si].d_S     + off_int, bufs[si].d_S_ends + off_se,
                    bufs[si].d_depth + sub_off, s_eff);
            } else {
                brandes_back_kernel_opt<IS_UNDIRECTED><<<sub_n, tpb, 0, st>>>(
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
