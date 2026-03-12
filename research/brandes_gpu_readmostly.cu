#include "common.h"
#include "brandes.h"
#include "GraphManaged.h"

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <vector>
#include <cstring>

// ============================================================
//  GPU_ReadMostly — 手法1 独立評価用 (アブレーション Step 2/4)
//
//  GH200 NVLink-C2C 対応 Unified Memory 最適化のみ。
//  2-stream 非同期パイプライン (手法2) は含まない。
//
//  GPU_Managed (Step 1) との差分:
//    [追加] cudaMemAdviseSetReadMostly — HBM3 L2 への複製を許可
//           バッチ内の複数ソースが同一隣接リストを参照するため
//           初回フェッチ以降は NVLink-C2C を介さず L2 から供給される
//    [追加] グラフサイズ適応型配置
//           小グラフ (topo < HBM3 の 35%): PrefetchAsync で HBM3 に直接配置
//           大グラフ: CPU LPDDR5X に固定 + SetReadMostly で L2 キャッシュ活用
//
//  GPU_Opt (Step 4) との差分:
//    [欠如] 2-stream BFS/Backward 重複実行 (手法2)
//    [欠如] cudaMemsetAsync によるメモリ初期化の非同期化 (手法2)
//
//  本ファイルは brandes_gpu_managed_impl() を再利用し、
//  トポロジデータの設定部分のみを変更する。
// ============================================================

// brandes_gpu_managed.cu で定義されるシングルストリーム実行関数
extern std::vector<double> brandes_gpu_managed_impl(
        int *R, int *C, double *CB_managed,
        int n_nodes, int edge_size, int gpu_id);

std::vector<double> brandes_gpu_readmostly(Graph &graph) {
    int *R        = graph.getAdjacencyListPointers();
    int *C        = graph.getAdjacencyList();
    int n_nodes   = graph.getNodeCount();
    int edge_size = 2 * graph.getEdgeCount();

    int num_gpus = 0;
    CUDA_ERR_CHK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus == 0) {
        std::cerr << "No GPU found" << std::endl;
        exit(EXIT_FAILURE);
    }

    // CSR トポロジデータを cudaMallocManaged に再確保
    int *R_m, *C_m;
    CUDA_ERR_CHK(cudaMallocManaged(&R_m, (size_t)(n_nodes + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&C_m, (size_t)edge_size     * sizeof(int)));
    memcpy(R_m, R, (size_t)(n_nodes + 1) * sizeof(int));
    memcpy(C_m, C, (size_t)edge_size     * sizeof(int));

    // ── 手法1 ① SetReadMostly ──────────────────────────────────────────────
    //   HBM3 L2 への複製を許可 (GPU_Managed にはこれがない)
    CUDA_ERR_CHK(cudaMemAdvise(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                               cudaMemAdviseSetReadMostly, 0));
    CUDA_ERR_CHK(cudaMemAdvise(C_m, (size_t)edge_size     * sizeof(int),
                               cudaMemAdviseSetReadMostly, 0));

    // ── 手法1 ② グラフサイズ適応型配置 ──────────────────────────────────────
    //   小グラフ: HBM3 直接配置 → NVLink-C2C レイテンシなし
    //   大グラフ: CPU LPDDR5X + SetReadMostly → NVLink-C2C 経由で L2 キャッシュ
    cudaDeviceProp prop;
    CUDA_ERR_CHK(cudaGetDeviceProperties(&prop, 0));
    const size_t topo_bytes = (size_t)(n_nodes + 1) * sizeof(int)
                            + (size_t)edge_size      * sizeof(int);
    const bool topo_on_gpu  = (topo_bytes < (size_t)(prop.totalGlobalMem * 0.35));

    if (topo_on_gpu) {
        // 小グラフ: HBM3 に直接配置
        CUDA_ERR_CHK(cudaMemAdvise(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
        CUDA_ERR_CHK(cudaMemAdvise(C_m, (size_t)edge_size     * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(R_m, (size_t)(n_nodes + 1) * sizeof(int), 0, 0));
        CUDA_ERR_CHK(cudaMemPrefetchAsync(C_m, (size_t)edge_size     * sizeof(int), 0, 0));
        fprintf(stderr, "  > [ReadMostly] topology placement: HBM3 (%.2f GB < 35%% of %.0f GB)\n",
                topo_bytes / 1e9, prop.totalGlobalMem / 1e9);
    } else {
        // 大グラフ: CPU LPDDR5X + SetReadMostly
        CUDA_ERR_CHK(cudaMemAdvise(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                                   cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_ERR_CHK(cudaMemAdvise(R_m, (size_t)(n_nodes + 1) * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
        CUDA_ERR_CHK(cudaMemAdvise(C_m, (size_t)edge_size     * sizeof(int),
                                   cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
        CUDA_ERR_CHK(cudaMemAdvise(C_m, (size_t)edge_size     * sizeof(int),
                                   cudaMemAdviseSetAccessedBy, 0));
        fprintf(stderr, "  > [ReadMostly] topology placement: CPU LPDDR5X + ReadMostly L2 cache (%.2f GB)\n",
                topo_bytes / 1e9);
    }

    // 結果バッファ (GPU HBM3 に配置)
    double *CB_managed;
    CUDA_ERR_CHK(cudaMallocManaged(&CB_managed, (size_t)n_nodes * sizeof(double)));
    memset(CB_managed, 0, (size_t)n_nodes * sizeof(double));

    // シングルストリーム実行 (GPU_Managed と同一カーネル・同一ループ)
    // 2-stream 非同期化は GPU_Opt (手法2) で追加される
    std::vector<double> result = brandes_gpu_managed_impl(
            R_m, C_m, CB_managed, n_nodes, edge_size, 0);

    CUDA_ERR_CHK(cudaFree(R_m));
    CUDA_ERR_CHK(cudaFree(C_m));
    CUDA_ERR_CHK(cudaFree(CB_managed));

    return result;
}
