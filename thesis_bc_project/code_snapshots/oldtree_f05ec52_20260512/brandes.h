#ifndef BRANDES_H
#define BRANDES_H

#include <vector>
#include "Graph.h"

struct GpuSingleSourceDepResult {
    std::vector<double> delta;
    int reachableCount = 0;
};

// Function prototypes for different Brandes algorithm implementations
std::vector<double> brandes_sequential(Graph &graph);
std::vector<double> brandes_omp(Graph &graph);
std::vector<double> brandes_gpu(Graph &graph);
// GH200 高速化版: ReadMostly + 適応型配置 + cudaMemsetAsync + 2ストリーム (フェーズ2 改良)
std::vector<double> brandes_gpu_opt(Graph &graph);
// 純 GPU メモリ版 (UM 不使用): cudaMalloc + cudaMemcpy + cudaMemsetAsync + 2ストリーム
std::vector<double> brandes_gpu_opt_pure(Graph &graph);
// 純 GPU メモリ版 + 手動 chunking: SUB_BATCH × N で確保し outer ループで反復
std::vector<double> brandes_gpu_opt_pure_chunked(Graph &graph);
// libcugraph C++/CUDA コア直接呼び出し版（Exact BC, 重み無し無向グラフ）
std::vector<double> brandes_cugraph_bc(Graph &graph);
// Galloit (path-merging) バッチ並列 BFS BC (gobardhanm/path-merging-bc)
std::vector<double> brandes_pathmerge_bc(Graph &graph);
std::vector<double> brandes_gpu_opt_pure_csr(const int* R, const int* C, int n_nodes, int edge_size);
GpuSingleSourceDepResult single_source_dependency_gpu_opt_pure_csr(
    const int* R, const int* C, int n_nodes, int edge_size, int source);

#endif // BRANDES_H
