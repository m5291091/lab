#include "GraphManaged.h"
#include <iostream>
#include <cstdlib>

#define CUDA_ERR_CHK(err) managed_cuda_check(err, __FILE__, __LINE__)
static inline void managed_cuda_check(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

GraphManaged::GraphManaged()
    : nodeCount(0), edgeCount(0),
      adjacencyList(nullptr), adjacencyListPointers(nullptr) {}

GraphManaged::~GraphManaged() {
    if (adjacencyListPointers) cudaFree(adjacencyListPointers);
    if (adjacencyList)         cudaFree(adjacencyList);
}

int GraphManaged::getNodeCount() const { return nodeCount; }
int GraphManaged::getEdgeCount() const { return edgeCount; }
int* GraphManaged::getAdjacencyList() const        { return adjacencyList; }
int* GraphManaged::getAdjacencyListPointers() const { return adjacencyListPointers; }

void GraphManaged::readGraph() {
    std::cin >> nodeCount >> edgeCount;

    // --- cudaMallocManaged でアロケート ---
    // 静的トポロジデータ: GPU VRAM ではなく CPU LPDDR5X 側に常駐させる。
    // NVLink-C2C (900 GB/s) 経由でキャッシュラインフェッチを行うため、
    // PCIe ボトルネックなしに 96 GB を超えるグラフを扱える。
    CUDA_ERR_CHK(cudaMallocManaged(&adjacencyListPointers, (nodeCount + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&adjacencyList, 2 * edgeCount * sizeof(int)));

    for (int i = 0; i <= nodeCount; ++i)
        std::cin >> adjacencyListPointers[i];
    for (int i = 0; i < 2 * edgeCount; ++i)
        std::cin >> adjacencyList[i];

    // GPU がアクセスすることを宣言 (マイグレーションのヒント)
    int gpu_id = 0;
    cudaGetDevice(&gpu_id);

    // 静的データ: CPU メモリに固定し、GPU からの読み出しはキャッシュ経由
    CUDA_ERR_CHK(cudaMemAdvise(adjacencyListPointers,
                               (nodeCount + 1) * sizeof(int),
                               cudaMemAdviseSetPreferredLocation,
                               cudaCpuDeviceId));
    CUDA_ERR_CHK(cudaMemAdvise(adjacencyListPointers,
                               (nodeCount + 1) * sizeof(int),
                               cudaMemAdviseSetAccessedBy,
                               gpu_id));

    CUDA_ERR_CHK(cudaMemAdvise(adjacencyList,
                               2 * edgeCount * sizeof(int),
                               cudaMemAdviseSetPreferredLocation,
                               cudaCpuDeviceId));
    CUDA_ERR_CHK(cudaMemAdvise(adjacencyList,
                               2 * edgeCount * sizeof(int),
                               cudaMemAdviseSetAccessedBy,
                               gpu_id));
}
