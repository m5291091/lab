#include "common.h"
#include "brandes.h"

// path-merging-bc headers
#include "graph.h"
#include "galliot.h"
#include "utils.h"

#include <cstring>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

// ============================================================
//  path-merging-bc の utils.cpp / graph.cpp のスタブ実装
//  (nvcc でコンパイルするため .cu 内に配置)
// ============================================================

// --- CSRGraph ---
CSRGraph::~CSRGraph() { free(); }

void CSRGraph::allocate(int vertices, int edges) {
    numVertices = vertices;
    numEdges = edges;
    CUDA_CHECK(cudaMallocManaged(&rowPtr, (size_t)(numVertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&colIdx, (size_t)numEdges * sizeof(int)));
    std::memset(rowPtr, 0, (numVertices + 1) * sizeof(int));
    std::memset(colIdx, 0, numEdges * sizeof(int));
}

void CSRGraph::free() {
    if (rowPtr) { cudaFree(rowPtr); rowPtr = nullptr; }
    if (colIdx) { cudaFree(colIdx); colIdx = nullptr; }
    numVertices = 0; numEdges = 0;
}

// --- Timer ---
Timer::Timer() : started(false) {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
}
Timer::~Timer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
void Timer::startTimer() { CUDA_CHECK(cudaEventRecord(start, 0)); started = true; }
float Timer::stopTimer() {
    if (!started) return 0.0f;
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    started = false; return ms;
}

// --- I/O ユーティリティ (galliot.cu の compute() がリンク参照) ---
void saveBCToFile(const std::string&, const std::vector<double>&, bool) {}
void printProgress(int, int, const std::string&) {}
void printVerbose(const std::string&, bool) {}
void printDeviceInfo() {}

// loadGraphFromTSV / convertToCSR / printGraphStats — ラッパーでは不使用だがリンク用
CSRGraph* loadGraphFromTSV(const std::string&, bool) { return nullptr; }
void convertToCSR(const std::vector<Edge>&, int, CSRGraph*) {}
void printGraphStats(const CSRGraph*) {}

// ============================================================
//  brandes_pathmerge_bc — メインラッパー
// ============================================================

std::vector<double> brandes_pathmerge_bc(Graph &graph) {
    int n_nodes  = graph.getNodeCount();
    const int* h_R = graph.getAdjacencyListPointers();
    const int* h_C = graph.getAdjacencyList();

    // 有向辺数は rowPtr[n_nodes] から取得 (getEdgeCount() は無向辺数のため半分)
    int directed_edges = h_R[n_nodes];

    // --- CSRGraph を作成 (cudaMallocManaged, .cu 内から直接呼び出し) ---
    CSRGraph* csr = new CSRGraph();
    csr->numVertices = n_nodes;
    csr->numEdges    = directed_edges;
    CUDA_ERR_CHK(cudaMallocManaged(&csr->rowPtr, (size_t)(n_nodes + 1) * sizeof(int)));
    CUDA_ERR_CHK(cudaMallocManaged(&csr->colIdx, (size_t)directed_edges * sizeof(int)));
    std::memcpy(csr->rowPtr, h_R, (size_t)(n_nodes + 1) * sizeof(int));
    std::memcpy(csr->colIdx, h_C, (size_t)directed_edges * sizeof(int));

    // バッチサイズ: 環境変数でオーバーライド可能 (デフォルト 64, 最大 64)
    int batchSize = 64;
    const char* env_batch = std::getenv("PATHMERGE_BC_BATCH_SIZE");
    if (env_batch) {
        int val = std::atoi(env_batch);
        if (val > 0 && val <= 64) batchSize = val;
    }

    // メモリ使用量を表示
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    fprintf(stderr, "  > [PathMerge] free_mem=%.1f GB, batch_size=%d, num_sources=%d\n",
            free_mem / 1e9, batchSize, n_nodes);
    int numBatches = (n_nodes + batchSize - 1) / batchSize;
    fprintf(stderr, "  > [PathMerge] num_batches=%d\n", numBatches);

    // --- GalloitBC を構築 (verbose=false: 進捗バー抑制) ---
    GalloitBC bcComputer(csr, batchSize, false);

    // --- フェーズ別計測付きバッチループ ---
    cudaEvent_t ev_bfs_start, ev_bfs_end, ev_back_end;
    CUDA_ERR_CHK(cudaEventCreate(&ev_bfs_start));
    CUDA_ERR_CHK(cudaEventCreate(&ev_bfs_end));
    CUDA_ERR_CHK(cudaEventCreate(&ev_back_end));

    float total_bfs_ms = 0.0f, total_back_ms = 0.0f;

    for (int b = 0; b < numBatches; b++) {
        int start = b * batchSize;
        int end   = std::min(start + batchSize, n_nodes);

        std::vector<int> sources;
        sources.reserve(end - start);
        for (int i = start; i < end; i++) {
            sources.push_back(i);
        }

        bcComputer.initializeBatch(sources);

        // BFS (forward) フェーズ計測
        CUDA_ERR_CHK(cudaEventRecord(ev_bfs_start, 0));
        bcComputer.forwardPhase(sources);
        CUDA_ERR_CHK(cudaEventRecord(ev_bfs_end, 0));

        // Backward (dependency) フェーズ計測
        bcComputer.backwardPhase(sources);
        CUDA_ERR_CHK(cudaEventRecord(ev_back_end, 0));
        CUDA_ERR_CHK(cudaEventSynchronize(ev_back_end));

        float bfs_ms = 0.0f, back_ms = 0.0f;
        CUDA_ERR_CHK(cudaEventElapsedTime(&bfs_ms, ev_bfs_start, ev_bfs_end));
        CUDA_ERR_CHK(cudaEventElapsedTime(&back_ms, ev_bfs_end, ev_back_end));
        total_bfs_ms  += bfs_ms;
        total_back_ms += back_ms;
    }

    fprintf(stderr, "  > [GPU Phase] BFS: %.4f sec, Backward: %.4f sec\n",
            total_bfs_ms / 1000.0f, total_back_ms / 1000.0f);

    // --- 結果取得: 無向グラフ補正 (/2) ---
    // Brandes on undirected graph double-counts each path (both edge directions).
    const double* d_bc = bcComputer.getBC();
    std::vector<double> result(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        result[i] = d_bc[i] / 2.0;
    }

    // --- クリーンアップ ---
    CUDA_ERR_CHK(cudaEventDestroy(ev_bfs_start));
    CUDA_ERR_CHK(cudaEventDestroy(ev_bfs_end));
    CUDA_ERR_CHK(cudaEventDestroy(ev_back_end));
    // csr は GalloitBC のデストラクタでは解放されないので手動で解放
    delete csr;

    return result;
}
