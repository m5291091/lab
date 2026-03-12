// ============================================================
//  bandwidth_benchmark.cu
//  NVLink-C2C および HBM3 の実効帯域計測ツール
//
//  GH200 Grace Hopper Superchip のメモリ帯域を計測:
//    - HBM3:           GPU → GPU コピー帯域 (ピーク 4.02 TB/s)
//    - NVLink-C2C:     CPU LPDDR5X → GPU コピー帯域 (ピーク 900 GB/s)
//    - LPDDR5X:        CPU メモリ帯域 (参考, ピーク 512 GB/s)
//
//  出力フォーマット (TSV):
//    Transfer_Type  Size_GB  Time_ms  Bandwidth_GBs  Theoretical_GBs  Ratio_pct
//
//  使用方法:
//    ./bandwidth_benchmark [size_MB]
//    size_MB: 計測バッファサイズ (デフォルト: 1024 MB = 1 GB)
//
//  参考: BC_Miyabi_report.pdf §3 GH200 メモリ階層最適化
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CUDA_ERR_CHK(call) do {                                        \
    cudaError_t _err = (call);                                         \
    if (_err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_err));         \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while(0)

// 計測定数
static const double HBM3_PEAK_GBS      = 4020.0;  // GH200 HBM3 ピーク帯域 (GB/s)
static const double NVLINK_C2C_PEAK_GBS = 900.0;  // NVLink-C2C ピーク帯域 (双方向合計)
static const double LPDDR5X_PEAK_GBS   = 512.0;   // Grace CPU LPDDR5X ピーク帯域

// ワームアップ: 初回計測バイアスを除去
static void warmup(void *dst, const void *src, size_t size, cudaMemcpyKind kind, int reps=3) {
    for (int i = 0; i < reps; i++) {
        CUDA_ERR_CHK(cudaMemcpy(dst, src, size, kind));
    }
    CUDA_ERR_CHK(cudaDeviceSynchronize());
}

// 帯域計測: CUDA イベントで GPU タイマーを使用
static double measure_bandwidth_GBs(void *dst, const void *src, size_t size,
                                     cudaMemcpyKind kind, int reps=5) {
    cudaEvent_t start, stop;
    CUDA_ERR_CHK(cudaEventCreate(&start));
    CUDA_ERR_CHK(cudaEventCreate(&stop));

    warmup(dst, src, size, kind);

    CUDA_ERR_CHK(cudaEventRecord(start));
    for (int i = 0; i < reps; i++) {
        CUDA_ERR_CHK(cudaMemcpy(dst, src, size, kind));
    }
    CUDA_ERR_CHK(cudaEventRecord(stop));
    CUDA_ERR_CHK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_ERR_CHK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_ERR_CHK(cudaEventDestroy(start));
    CUDA_ERR_CHK(cudaEventDestroy(stop));

    double elapsed_s = elapsed_ms / 1000.0 / reps;
    return (size / 1e9) / elapsed_s;  // GB/s
}

// Unified Memory を使った C2C 帯域計測 (cudaMallocManaged + Prefetch)
static double measure_c2c_prefetch_bandwidth(size_t size, int gpu_id, int reps=5) {
    void *ptr;
    CUDA_ERR_CHK(cudaMallocManaged(&ptr, size));

    // CPU LPDDR5X に配置 → GPU がアクセスする際に NVLink-C2C を使用
    CUDA_ERR_CHK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
    CUDA_ERR_CHK(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, gpu_id));

    // CPU 側で初期化
    memset(ptr, 0xAB, size);
    CUDA_ERR_CHK(cudaDeviceSynchronize());

    // ワームアップ
    for (int i = 0; i < 3; i++) {
        CUDA_ERR_CHK(cudaMemPrefetchAsync(ptr, size, gpu_id, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, 0));
        CUDA_ERR_CHK(cudaDeviceSynchronize());
    }

    // 計測: CPU → GPU 方向 (NVLink-C2C 経由)
    cudaEvent_t start, stop;
    CUDA_ERR_CHK(cudaEventCreate(&start));
    CUDA_ERR_CHK(cudaEventCreate(&stop));

    // 先に CPU 側に戻す
    CUDA_ERR_CHK(cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, 0));
    CUDA_ERR_CHK(cudaDeviceSynchronize());

    CUDA_ERR_CHK(cudaEventRecord(start));
    for (int i = 0; i < reps; i++) {
        CUDA_ERR_CHK(cudaMemPrefetchAsync(ptr, size, gpu_id, 0));
        CUDA_ERR_CHK(cudaDeviceSynchronize());
        CUDA_ERR_CHK(cudaMemPrefetchAsync(ptr, size, cudaCpuDeviceId, 0));
        CUDA_ERR_CHK(cudaDeviceSynchronize());
    }
    CUDA_ERR_CHK(cudaEventRecord(stop));
    CUDA_ERR_CHK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_ERR_CHK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_ERR_CHK(cudaEventDestroy(start));
    CUDA_ERR_CHK(cudaEventDestroy(stop));

    CUDA_ERR_CHK(cudaFree(ptr));

    // 往復 (CPU→GPU + GPU→CPU) なので 2× で割る
    double elapsed_s = elapsed_ms / 1000.0 / reps / 2.0;
    return (size / 1e9) / elapsed_s;  // GB/s (一方向)
}

int main(int argc, char *argv[]) {
    size_t size_mb = (argc >= 2) ? (size_t)atol(argv[1]) : 1024;
    size_t buf_size = size_mb * 1024 * 1024;

    int gpu_id = 0;
    CUDA_ERR_CHK(cudaSetDevice(gpu_id));

    cudaDeviceProp prop;
    CUDA_ERR_CHK(cudaGetDeviceProperties(&prop, gpu_id));

    fprintf(stderr, "=== GH200 メモリ帯域ベンチマーク ===\n");
    fprintf(stderr, "  GPU: %s\n", prop.name);
    fprintf(stderr, "  HBM3 総容量: %.0f GB\n", prop.totalGlobalMem / 1e9);
    fprintf(stderr, "  計測バッファ: %zu MB\n\n", size_mb);

    printf("Transfer_Type\tSize_GB\tTime_ms\tBandwidth_GBs\tTheoretical_GBs\tRatio_pct\n");

    // ---- 1. HBM3 帯域: GPU → GPU コピー (Device to Device) ----
    {
        void *d_src, *d_dst;
        CUDA_ERR_CHK(cudaMalloc(&d_src, buf_size));
        CUDA_ERR_CHK(cudaMalloc(&d_dst, buf_size));
        CUDA_ERR_CHK(cudaMemset(d_src, 0x1, buf_size));

        double bw = measure_bandwidth_GBs(d_dst, d_src, buf_size, cudaMemcpyDeviceToDevice);
        double ratio = bw / HBM3_PEAK_GBS * 100.0;
        // elapsed_ms を逆算 (参考表示)
        double elapsed_ms = (buf_size / 1e9) / bw * 1000.0;
        printf("HBM3_DtoD\t%.3f\t%.2f\t%.1f\t%.0f\t%.1f\n",
               buf_size / 1e9, elapsed_ms, bw, HBM3_PEAK_GBS, ratio);
        fprintf(stderr, "  HBM3 (DtoD):        %.1f GB/s  (%.0f GB/s 理論比: %.1f%%)\n",
                bw, HBM3_PEAK_GBS, ratio);

        CUDA_ERR_CHK(cudaFree(d_src));
        CUDA_ERR_CHK(cudaFree(d_dst));
    }

    // ---- 2. PCIe/NVLink Host→Device 帯域 (ピン留めメモリ) ----
    {
        void *h_src, *d_dst;
        CUDA_ERR_CHK(cudaHostAlloc(&h_src, buf_size, cudaHostAllocDefault));
        CUDA_ERR_CHK(cudaMalloc(&d_dst, buf_size));
        memset(h_src, 0x2, buf_size);

        double bw = measure_bandwidth_GBs(d_dst, h_src, buf_size, cudaMemcpyHostToDevice);
        double ratio = bw / NVLINK_C2C_PEAK_GBS * 100.0;
        double elapsed_ms = (buf_size / 1e9) / bw * 1000.0;
        printf("Pinned_HtoD\t%.3f\t%.2f\t%.1f\t%.0f\t%.1f\n",
               buf_size / 1e9, elapsed_ms, bw, NVLINK_C2C_PEAK_GBS, ratio);
        fprintf(stderr, "  Pinned HtoD:        %.1f GB/s  (%.0f GB/s 理論比: %.1f%%)\n",
                bw, NVLINK_C2C_PEAK_GBS, ratio);

        CUDA_ERR_CHK(cudaFreeHost(h_src));
        CUDA_ERR_CHK(cudaFree(d_dst));
    }

    // ---- 3. PCIe/NVLink Device→Host 帯域 (ピン留めメモリ) ----
    {
        void *h_dst, *d_src;
        CUDA_ERR_CHK(cudaHostAlloc(&h_dst, buf_size, cudaHostAllocDefault));
        CUDA_ERR_CHK(cudaMalloc(&d_src, buf_size));
        CUDA_ERR_CHK(cudaMemset(d_src, 0x3, buf_size));

        double bw = measure_bandwidth_GBs(h_dst, d_src, buf_size, cudaMemcpyDeviceToHost);
        double ratio = bw / NVLINK_C2C_PEAK_GBS * 100.0;
        double elapsed_ms = (buf_size / 1e9) / bw * 1000.0;
        printf("Pinned_DtoH\t%.3f\t%.2f\t%.1f\t%.0f\t%.1f\n",
               buf_size / 1e9, elapsed_ms, bw, NVLINK_C2C_PEAK_GBS, ratio);
        fprintf(stderr, "  Pinned DtoH:        %.1f GB/s  (%.0f GB/s 理論比: %.1f%%)\n",
                bw, NVLINK_C2C_PEAK_GBS, ratio);

        CUDA_ERR_CHK(cudaFreeHost(h_dst));
        CUDA_ERR_CHK(cudaFree(d_src));
    }

    // ---- 4. NVLink-C2C 帯域: Unified Memory PrefetchAsync ----
    //        これが論文の核心 — BC トポロジデータのアクセスパターンを模擬
    {
        double bw = measure_c2c_prefetch_bandwidth(buf_size, gpu_id);
        double ratio = bw / NVLINK_C2C_PEAK_GBS * 100.0;
        double elapsed_ms = (buf_size / 1e9) / bw * 1000.0;
        printf("NVLink_C2C_Prefetch\t%.3f\t%.2f\t%.1f\t%.0f\t%.1f\n",
               buf_size / 1e9, elapsed_ms, bw, NVLINK_C2C_PEAK_GBS, ratio);
        fprintf(stderr, "  NVLink-C2C Prefetch: %.1f GB/s  (%.0f GB/s 理論比: %.1f%%)\n",
                bw, NVLINK_C2C_PEAK_GBS, ratio);
    }

    fprintf(stderr, "\nTSV 出力は stdout に出力されました。\n");
    fprintf(stderr, "論文への記載: NVLink-C2C Prefetch の帯域が Pinned HtoD と比較して\n");
    fprintf(stderr, "どれほど近いかが GH200 特性活用の直接証拠になります。\n");

    return 0;
}
