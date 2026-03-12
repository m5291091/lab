#pragma once
#include <iostream>
#include <cuda_runtime.h>

// cudaMallocManaged を用いた Graph クラス
// GH200 のユニファイドメモリ特性を活用するために、
// グラフの CSR 表現を cudaMallocManaged でアロケートする。
// 静的なトポロジデータ (R, C) は CPU (LPDDR5X) 側に配置し、
// NVLink-C2C (900 GB/s) 経由でGPUからゼロコピーアクセスする。
class GraphManaged {
public:
    GraphManaged();
    ~GraphManaged();

    int getNodeCount() const;
    int getEdgeCount() const;

    void readGraph();

    int* getAdjacencyList() const;
    int* getAdjacencyListPointers() const;

private:
    int nodeCount, edgeCount;
    int *adjacencyList, *adjacencyListPointers;
};
